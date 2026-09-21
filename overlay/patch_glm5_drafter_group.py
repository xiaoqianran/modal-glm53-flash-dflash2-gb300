#!/usr/bin/env python3
"""Patch current vLLM GLM-5-Next KV layout for the DFlash2 SWA drafter.

The original GB300 recipe patched many small string anchors in
v1/core/kv_cache_utils.py. The current vllm/vllm-openai:glm53-flash has
evolved (notably the KVCacheTensor layout API), so this version replaces
only the five GLM5-specific functions by AST source spans.

The generic vLLM KV path is left untouched.

Two drafter layouts are supported:
- exact-fit: resize each drafter block so one drafter page equals one MLA page;
  drafter layer i aliases MLA tensor i at the same byte offset.
- standalone: keep the original drafter page size and append compact per-layer
  regions to the GLM5 pool.

Both layouts keep the drafter as a separate KV group so scheduler/admission
accounts for its sliding-window block demand.
"""

from __future__ import annotations

import argparse
import ast
import sys
from textwrap import dedent

DEFAULT_KV_FILE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
)
MARKER = "DFLASH2-DRAFTER-GROUP-AST-V2"


REPLACEMENTS: dict[str, str] = {
    "_get_kv_cache_groups_glm5_next": dedent(
        r"""
        def _get_kv_cache_groups_glm5_next(
            vllm_config: VllmConfig,
            kv_cache_spec: dict[str, KVCacheSpec],
        ) -> list[KVCacheGroupSpec] | None:
            # Build GLM-5.3-Flash groups with Mamba/MLA/tail and DFlash2.
            # DFLASH2-DRAFTER-GROUP-AST-V2
            mamba_specs = {
                name: spec
                for name, spec in kv_cache_spec.items()
                if isinstance(spec, MambaSpec)
            }
            tail_specs = {
                name: spec
                for name, spec in kv_cache_spec.items()
                if isinstance(spec, KpoolTailSpec)
            }
            draft_specs = {
                name: spec
                for name, spec in kv_cache_spec.items()
                if type(spec) is SlidingWindowSpec
            }
            attn_specs = {
                name: spec
                for name, spec in kv_cache_spec.items()
                if not isinstance(spec, (MambaSpec, KpoolTailSpec))
                and type(spec) is not SlidingWindowSpec
            }
            if not mamba_specs or not all(
                type(spec) is MLAAttentionSpec for spec in attn_specs.values()
            ):
                return None

            mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
            idx_pages = {
                spec.page_size_bytes
                for spec in mla_specs.values()
                if spec.tokens_per_state > 1
            }
            if not idx_pages:
                return None

            assert all(spec.page_size_padded is None for spec in mla_specs.values())
            assert len(idx_pages) == 1
            mla_names = [
                name for name, spec in mla_specs.items() if spec.tokens_per_state == 1
            ]
            mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
            assert len(mla_pages) == 1
            mla_page = mla_pages.pop()
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)
            assert uniform_spec is not None

            tail_group: KVCacheGroupSpec | None = None
            if tail_specs:
                idx_page = next(iter(idx_pages))
                padded_tail_specs: dict[str, KVCacheSpec] = {
                    name: replace(spec, page_size_padded=idx_page)
                    for name, spec in tail_specs.items()
                }
                tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
                assert tail_uniform is not None
                tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

            any_mamba = next(iter(mamba_specs.values()))
            assert all(spec == any_mamba for spec in mamba_specs.values())
            if any_mamba.real_page_size_bytes > mla_page:
                raise ValueError(
                    f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
                    f"does not fit the MLA page ({mla_page} bytes); increase tensor "
                    "parallelism or use a wider KV cache dtype"
                )
            padded_specs: dict[str, KVCacheSpec] = {
                name: replace(any_mamba, page_size_padded=mla_page)
                for name in mamba_specs
            }
            num_groups = _pp_balanced_mamba_group_count(
                vllm_config, list(mamba_specs), mla_names
            )
            if num_groups is None:
                raise ValueError(
                    "a pipeline stage has mamba layers but no MLA layer to share "
                    "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
                )
            mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
            for index, name in enumerate(mamba_specs):
                mamba_grouped_names[index % num_groups].append(name)

            draft_group: KVCacheGroupSpec | None = None
            if draft_specs:
                any_draft = next(iter(draft_specs.values()))
                assert all(spec == any_draft for spec in draft_specs.values()), (
                    "DFlash2 SlidingWindowSpec layers must share one spec"
                )
                assert any_draft.page_size_bytes % any_draft.block_size == 0
                draft_bytes_per_token = (
                    any_draft.page_size_bytes // any_draft.block_size
                )
                mla_block = mla_specs[mla_names[0]].block_size
                fit_block = (
                    mla_page // draft_bytes_per_token
                    if mla_page % draft_bytes_per_token == 0
                    else 0
                )
                exact_fit = bool(
                    fit_block
                    and fit_block % 64 == 0
                    and (fit_block % mla_block == 0 or mla_block % fit_block == 0)
                    and len(draft_specs) <= len(mla_names)
                )
                if exact_fit:
                    new_draft_specs: dict[str, KVCacheSpec] = {
                        name: replace(spec, block_size=fit_block)
                        for name, spec in draft_specs.items()
                    }
                else:
                    new_draft_specs = dict(draft_specs)

                assert all(
                    spec.page_size_padded is None for spec in new_draft_specs.values()
                )
                draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
                assert draft_uniform is not None
                draft_group = KVCacheGroupSpec(
                    list(new_draft_specs),
                    draft_uniform,
                )

            return (
                [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
                + ([tail_group] if tail_group is not None else [])
                + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
                + ([draft_group] if draft_group is not None else [])
            )
        """
    ).strip()
    + "\n",
    "_glm5_next_tensor_layout": dedent(
        r"""
        def _glm5_next_tensor_layout(
            kv_cache_groups: list[KVCacheGroupSpec],
        ) -> (
            tuple[
                KVCacheGroupSpec,
                list[KVCacheGroupSpec],
                list[str],
                list[str],
                int,
                int,
                list[str],
                int,
                KVCacheGroupSpec | None,
            ]
            | None
        ):
            # Recognize GLM-5.3-Flash groups, including optional DFlash2.
            uniform_groups = [
                group
                for group in kv_cache_groups
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            ]
            mamba_groups = [
                group
                for group in kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ]

            attn_group: KVCacheGroupSpec | None = None
            tail_group: KVCacheGroupSpec | None = None
            draft_group: KVCacheGroupSpec | None = None
            for group in uniform_groups:
                inner = cast(
                    UniformTypeKVCacheSpecs, group.kv_cache_spec
                ).kv_cache_specs
                if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
                    attn_group = group
                elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
                    tail_group = group
                elif inner and all(
                    type(spec) is SlidingWindowSpec for spec in inner.values()
                ):
                    draft_group = group

            if attn_group is None or not mamba_groups:
                return None
            if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
                return None

            attn_uniform = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
            mla_inner = cast(
                dict[str, MLAAttentionSpec], attn_uniform.kv_cache_specs
            )
            if not all(
                type(spec) is MLAAttentionSpec and spec.page_size_padded is None
                for spec in mla_inner.values()
            ):
                return None

            mla_names = [
                name
                for name in attn_group.layer_names
                if mla_inner[name].tokens_per_state == 1
            ]
            idx_names = [
                name
                for name in attn_group.layer_names
                if mla_inner[name].tokens_per_state > 1
            ]
            mla_pages = {mla_inner[name].page_size_bytes for name in mla_names}
            idx_pages = {mla_inner[name].page_size_bytes for name in idx_names}
            if len(mla_pages) != 1 or len(idx_pages) != 1:
                return None
            mla_page = mla_pages.pop()
            idx_page = idx_pages.pop()

            if any(
                group.kv_cache_spec.page_size_bytes != mla_page
                for group in mamba_groups
            ):
                return None

            if draft_group is not None:
                draft_inner = cast(
                    UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                ).kv_cache_specs
                draft_pages = {
                    spec.page_size_bytes for spec in draft_inner.values()
                }
                if len(draft_pages) != 1:
                    return None
                if any(
                    spec.page_size_padded is not None
                    for spec in draft_inner.values()
                ):
                    return None
                draft_page = next(iter(draft_pages))
                if (
                    draft_page == mla_page
                    and len(draft_group.layer_names) > len(mla_names)
                ):
                    return None

            tail_names: list[str] = []
            tail_page = 0
            if tail_group is not None:
                tail_names = list(tail_group.layer_names)
                tail_inner = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                tail_pages = {
                    cast(KpoolTailSpec, spec).unpadded_page_size_bytes
                    for spec in tail_inner.values()
                }
                if len(tail_pages) != 1 or len(tail_names) != len(idx_names):
                    return None
                tail_page = tail_pages.pop()
                if tail_page > idx_page:
                    return None

            return (
                attn_group,
                mamba_groups,
                mla_names,
                idx_names,
                mla_page,
                idx_page,
                tail_names,
                tail_page,
                draft_group,
            )
        """
    ).strip()
    + "\n",
    "_get_kv_cache_bytes_per_block": dedent(
        r"""
        def _get_kv_cache_bytes_per_block(
            kv_cache_groups: list[KVCacheGroupSpec],
        ) -> int:
            # Return bytes consumed by one block in the shared KV pool.
            if (
                glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)
            ) is not None:
                (
                    _,
                    _,
                    mla_names,
                    idx_names,
                    mla_page,
                    idx_page,
                    _,
                    _,
                    draft_group,
                ) = glm5_layout
                bytes_per_block = (
                    len(mla_names) * mla_page + len(idx_names) * idx_page
                )
                if draft_group is not None:
                    draft_uniform = cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    )
                    draft_page = next(
                        iter(draft_uniform.kv_cache_specs.values())
                    ).page_size_bytes
                    if draft_page != mla_page:
                        bytes_per_block += (
                            len(draft_group.layer_names) * draft_page
                        )
                return bytes_per_block

            bytes_per_block = max(
                sum(
                    _get_per_layer_spec(group, layer_name).page_size_bytes
                    for layer_name in group.layer_names
                )
                for group in kv_cache_groups
            )
            assert bytes_per_block > 0
            return bytes_per_block
        """
    ).strip()
    + "\n",
    "get_kv_cache_config_from_groups": dedent(
        r"""
        def get_kv_cache_config_from_groups(
            vllm_config: VllmConfig,
            kv_cache_groups: list[KVCacheGroupSpec],
            available_memory: int,
        ) -> KVCacheConfig:
            # Generate the KV cache configuration from grouped layer specs.
            if len(kv_cache_groups) == 0:
                return KVCacheConfig(
                    num_blocks=1,
                    kv_cache_tensors=[],
                    kv_cache_groups=kv_cache_groups,
                    prefix_cache_retention_interval=(
                        vllm_config.cache_config.prefix_cache_retention_interval
                    ),
                )

            if (
                glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)
            ) is not None:
                (
                    attn_group,
                    mamba_groups,
                    mla_names,
                    idx_names,
                    mla_page,
                    idx_page,
                    tail_names,
                    _,
                    draft_group,
                ) = glm5_layout

                draft_names: list[str] = []
                draft_specs: dict[str, KVCacheSpec] = {}
                draft_page = 0
                draft_shared = False
                if draft_group is not None:
                    draft_names = list(draft_group.layer_names)
                    draft_specs = dict(
                        cast(
                            UniformTypeKVCacheSpecs,
                            draft_group.kv_cache_spec,
                        ).kv_cache_specs
                    )
                    draft_page = next(
                        iter(draft_specs.values())
                    ).page_size_bytes
                    draft_shared = draft_page == mla_page

                bytes_per_block = (
                    len(mla_names) * mla_page + len(idx_names) * idx_page
                )
                if draft_names and not draft_shared:
                    bytes_per_block += len(draft_names) * draft_page

                num_blocks = may_override_num_blocks(
                    vllm_config, available_memory // bytes_per_block
                )
                size = bytes_per_block * num_blocks
                attn_specs = cast(
                    UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
                ).kv_cache_specs

                kv_cache_tensors: list[KVCacheTensor] = []

                def add_tensor(
                    layer_name: str,
                    spec: KVCacheSpec,
                    offset: int,
                ) -> None:
                    kv_cache_tensors.append(
                        KVCacheTensor(
                            size=size,
                            layers=[layer_name],
                            layer_stride=spec.page_size_bytes * num_blocks,
                            block_stride=spec.page_size_bytes,
                            offset=offset,
                        )
                    )

                for index, mla_name in enumerate(mla_names):
                    offset = index * mla_page * num_blocks
                    add_tensor(mla_name, attn_specs[mla_name], offset)
                    for group in mamba_groups:
                        if index < len(group.layer_names):
                            add_tensor(
                                group.layer_names[index],
                                group.kv_cache_spec,
                                offset,
                            )
                    if draft_shared and index < len(draft_names):
                        draft_name = draft_names[index]
                        add_tensor(
                            draft_name,
                            draft_specs[draft_name],
                            offset,
                        )

                idx_base = len(mla_names) * mla_page * num_blocks
                for index, idx_name in enumerate(idx_names):
                    offset = idx_base + index * idx_page * num_blocks
                    add_tensor(idx_name, attn_specs[idx_name], offset)
                    if tail_names:
                        tail_name = tail_names[index]
                        tail_group = next(
                            group
                            for group in kv_cache_groups
                            if tail_name in group.layer_names
                        )
                        tail_specs = cast(
                            UniformTypeKVCacheSpecs,
                            tail_group.kv_cache_spec,
                        ).kv_cache_specs
                        add_tensor(
                            tail_name,
                            tail_specs[tail_name],
                            offset,
                        )

                if draft_names and not draft_shared:
                    draft_base = (
                        idx_base + len(idx_names) * idx_page * num_blocks
                    )
                    for index, draft_name in enumerate(draft_names):
                        offset = (
                            draft_base + index * draft_page * num_blocks
                        )
                        add_tensor(
                            draft_name,
                            draft_specs[draft_name],
                            offset,
                        )

                return KVCacheConfig(
                    num_blocks=num_blocks,
                    kv_cache_tensors=kv_cache_tensors,
                    kv_cache_groups=kv_cache_groups,
                    prefix_cache_retention_interval=(
                        vllm_config.cache_config.prefix_cache_retention_interval
                    ),
                )

            layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
            validate_kv_cache_layout(layout, kv_cache_groups)
            bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups)
            interleaved_block_stride = (
                bytes_per_block if layout.is_block_outermost else None
            )

            num_blocks = available_memory // bytes_per_block
            num_blocks = may_override_num_blocks(vllm_config, num_blocks)
            size = bytes_per_block * num_blocks

            kv_cache_tensors = []
            for group in kv_cache_groups:
                group_spec = group.kv_cache_spec
                layers_by_spec: defaultdict[
                    KVCacheSpec, list[str]
                ] = defaultdict(list)
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    for layer_name, spec in group_spec.kv_cache_specs.items():
                        layers_by_spec[spec].append(layer_name)
                elif group.layer_names:
                    layers_by_spec[group_spec].extend(group.layer_names)

                byte_offset = 0
                for spec, layer_names in layers_by_spec.items():
                    (
                        layer_stride,
                        block_stride,
                        _,
                        _,
                        _,
                    ) = compute_layout_strides(
                        spec,
                        num_blocks,
                        len(layer_names),
                        layout,
                        fixed_strides=(
                            None,
                            interleaved_block_stride,
                            None,
                            None,
                            None,
                        ),
                    )
                    offset = (
                        byte_offset
                        * max(layer_stride, spec.page_size_bytes)
                        // spec.page_size_bytes
                    )
                    kv_cache_tensors.append(
                        KVCacheTensor(
                            size=size,
                            layers=layer_names,
                            layer_stride=layer_stride,
                            block_stride=block_stride,
                            offset=offset,
                        )
                    )
                    byte_offset += (
                        len(layer_names) * spec.page_size_bytes
                    )

            return KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
                prefix_cache_retention_interval=(
                    vllm_config.cache_config.prefix_cache_retention_interval
                ),
            )
        """
    ).strip()
    + "\n",
    "_max_memory_usage_bytes_from_groups": dedent(
        r"""
        def _max_memory_usage_bytes_from_groups(
            vllm_config: VllmConfig,
            kv_cache_groups: list[KVCacheGroupSpec],
        ) -> int:
            # Calculate maximum memory usage in bytes from KV cache groups.
            if not kv_cache_groups:
                return 0

            if (
                glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)
            ) is not None:
                (
                    attn_group,
                    mamba_groups,
                    mla_names,
                    idx_names,
                    mla_page,
                    idx_page,
                    tail_names,
                    _,
                    draft_group,
                ) = glm5_layout
                uniform_spec = cast(
                    UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
                )
                total_blocks = uniform_spec.max_memory_usage_pages(
                    vllm_config
                )
                total_blocks += sum(
                    cdiv(
                        group.kv_cache_spec.max_memory_usage_bytes(
                            vllm_config
                        ),
                        group.kv_cache_spec.page_size_bytes,
                    )
                    for group in mamba_groups
                )
                if tail_names:
                    total_blocks += 1

                bytes_per_block = (
                    len(mla_names) * mla_page + len(idx_names) * idx_page
                )
                if draft_group is not None:
                    draft_uniform = cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    )
                    total_blocks += draft_uniform.max_memory_usage_pages(
                        vllm_config
                    )
                    draft_page = next(
                        iter(draft_uniform.kv_cache_specs.values())
                    ).page_size_bytes
                    if draft_page != mla_page:
                        bytes_per_block += (
                            len(draft_group.layer_names) * draft_page
                        )

                return total_blocks * bytes_per_block

            bytes_per_block = _pool_bytes_per_block(kv_cache_groups)
            total_blocks = 0
            for group in kv_cache_groups:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    total_blocks += spec.max_memory_usage_pages(vllm_config)
                else:
                    total_blocks += cdiv(
                        spec.max_memory_usage_bytes(vllm_config),
                        spec.page_size_bytes,
                    )

            return bytes_per_block * total_blocks
        """
    ).strip()
    + "\n",
}


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def patch_file(path: str, dry_run: bool = False) -> int:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    if MARKER in source:
        print(
            f"[patch_glm5_drafter_group] {path}: already patched "
            f"({MARKER}); no-op."
        )
        return 0

    tree = ast.parse(source, filename=path)
    functions = _top_level_functions(tree)
    missing = sorted(set(REPLACEMENTS) - set(functions))
    assert not missing, f"Required GLM5 KV functions missing: {missing}"

    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str, str]] = []
    for name, replacement in REPLACEMENTS.items():
        node = functions[name]
        assert node.end_lineno is not None
        edits.append((node.lineno - 1, node.end_lineno, replacement, name))

    for start, end, replacement, name in sorted(edits, reverse=True):
        print(
            f"[patch_glm5_drafter_group] replace {name}: "
            f"lines {start + 1}-{end}"
        )
        lines[start:end] = [replacement + "\n"]

    patched = "".join(lines)
    ast.parse(patched, filename=path)

    if dry_run:
        print(
            f"[patch_glm5_drafter_group] DRY RUN: "
            f"{len(edits)} AST function replacements valid."
        )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(
            f"[patch_glm5_drafter_group] wrote {len(edits)} "
            "AST function replacements; ast.parse OK."
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kv-file", default=DEFAULT_KV_FILE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return patch_file(args.kv_file, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
