"""CPU geometry validation for GLM-5.3-Flash + DFlash2 KV sharing.

This harness targets the current vLLM KVCacheTensor API, where each layer gets
an explicit tensor view described by layers, offset, layer_stride and
block_stride.

It validates:
- exact-fit DFlash2 pages alias MLA regions at identical byte offsets;
- standalone DFlash2 pages get independent compact regions;
- every model/drafter KV layer is represented exactly once;
- per-block accounting matches emitted layout;
- kernel-block splitting stays within each contiguous manager page.
"""

from types import SimpleNamespace

import torch
from vllm.v1.core import kv_cache_utils as K
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

MLA_BLOCK = 8576
MLA_PAGE = 4_390_912  # 8576 tokens * 512 bytes/token
AVAIL = 64_000_000_000

vllm_config = SimpleNamespace(
    parallel_config=SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
    ),
    model_config=SimpleNamespace(max_model_len=1_000_000),
    cache_config=SimpleNamespace(
        num_gpu_blocks_override=None,
        mamba_cache_mode="none",
        prefix_cache_retention_interval=None,
    ),
    max_in_flight_tokens=8192,
    speculative_config=None,
)


def build_spec(draft_kv_heads: int) -> dict:
    spec: dict = {}
    for i in range(34):
        spec[f"model.layers.{i}.kda"] = MambaSpec(
            block_size=16,
            shapes=((128, 128),),
            dtypes=(torch.float32,),
        )
    for i in range(11):
        spec[f"model.layers.{i}.mla"] = MLAAttentionSpec(
            block_size=MLA_BLOCK,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
        )
        spec[f"model.layers.{i}.indexer"] = MLAAttentionSpec(
            block_size=MLA_BLOCK,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.uint8,
            tokens_per_state=4,
        )
        spec[f"model.layers.{i}.kpool_tail"] = KpoolTailSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=4,
        )
    for i in range(5):
        spec[f"drafter.layers.{i}.attn"] = SlidingWindowSpec(
            block_size=2048,
            num_kv_heads=draft_kv_heads,
            head_size=128,
            dtype=torch.uint8,
            sliding_window=2048,
        )
    return spec


def tensor_for_layer(cfg, layer_name: str):
    matches = [t for t in cfg.kv_cache_tensors if layer_name in t.layers]
    assert len(matches) == 1, (layer_name, len(matches))
    return matches[0]


def assert_all_layers_covered(cfg, spec: dict) -> None:
    seen: list[str] = []
    for tensor in cfg.kv_cache_tensors:
        seen.extend(tensor.layers)
    assert len(seen) == len(set(seen)), "duplicate layer tensor views"
    assert set(seen) == set(spec), set(spec) - set(seen)


def assert_tensor_bounds(tensor, page_size: int, num_blocks: int) -> None:
    end = tensor.offset + (num_blocks - 1) * tensor.block_stride + page_size
    assert end <= tensor.size, (
        f"tensor view out of bounds: end={end}, size={tensor.size}, "
        f"offset={tensor.offset}, block_stride={tensor.block_stride}"
    )


def run_geometry(spec: dict, label: str):
    groups = K._get_kv_cache_groups_glm5_next(vllm_config, spec)
    assert groups is not None, f"[{label}] glm5_next path rejected the model"

    layout = K._glm5_next_tensor_layout(groups)
    assert layout is not None, f"[{label}] layout detection failed"
    (
        _,
        _,
        _,
        _,
        mla_page,
        _,
        _,
        _,
        draft_group,
    ) = layout
    assert mla_page == MLA_PAGE
    assert draft_group is groups[-1]

    cfg = K.get_kv_cache_config_from_groups(vllm_config, groups, AVAIL)
    bytes_per_block = K._pool_bytes_per_block(groups)
    assert cfg.num_blocks == AVAIL // bytes_per_block

    assert_all_layers_covered(cfg, spec)
    max_memory = K._max_memory_usage_bytes_from_groups(vllm_config, groups)

    print(
        f"[{label}] groups={len(groups)} layers={len(spec)} "
        f"tensor_views={len(cfg.kv_cache_tensors)} "
        f"num_blocks={cfg.num_blocks} bytes/block={bytes_per_block} "
        f"max_request_bytes={max_memory}"
    )
    return groups, layout, cfg, bytes_per_block, max_memory


# Geometry A: 4 heads * 2(K/V) * 128 * fp8 = 1024 bytes/token.
# MLA page / 1024 = 4288 tokens, and 4288 divides MLA block 8576.
spec_a = build_spec(draft_kv_heads=4)
groups_a, layout_a, cfg_a, pb_a, _ = run_geometry(spec_a, "A/exact-fit")

draft_group_a = layout_a[8]
draft_specs_a = draft_group_a.kv_cache_spec.kv_cache_specs
draft_names_a = list(draft_group_a.layer_names)
mla_names_a = layout_a[2]
d0 = next(iter(draft_specs_a.values()))

assert type(d0) is SlidingWindowSpec
assert d0.block_size == 4288, d0.block_size
assert d0.page_size_padded is None
assert d0.page_size_bytes == d0.real_page_size_bytes == MLA_PAGE

base_spec = {
    name: spec
    for name, spec in spec_a.items()
    if not name.startswith("drafter.")
}
base_groups = K._get_kv_cache_groups_glm5_next(vllm_config, base_spec)
assert base_groups is not None
assert K._glm5_next_tensor_layout(base_groups)[8] is None
assert pb_a == K._pool_bytes_per_block(base_groups)

for index, draft_name in enumerate(draft_names_a):
    mla_name = mla_names_a[index]
    draft_tensor = tensor_for_layer(cfg_a, draft_name)
    mla_tensor = tensor_for_layer(cfg_a, mla_name)
    assert draft_tensor.offset == mla_tensor.offset
    assert draft_tensor.block_stride == MLA_PAGE
    assert mla_tensor.block_stride == MLA_PAGE
    assert draft_tensor.layer_stride == mla_tensor.layer_stride
    assert_tensor_bounds(draft_tensor, MLA_PAGE, cfg_a.num_blocks)

# A backend may split the 4288-token manager block into 64-token kernels.
kernel_block = 64
assert d0.block_size % kernel_block == 0
ratio = d0.block_size // kernel_block
kernel_page = 2 * kernel_block * d0.num_kv_heads * d0.head_size
assert ratio * kernel_page == d0.page_size_bytes
print(
    f"[A] exact fit OK: block 2048->{d0.block_size}, "
    f"page={d0.page_size_bytes}, kernel split {d0.block_size}->{kernel_block} "
    f"(x{ratio}) stays inside each MLA page"
)


# Geometry B: 3 heads => 768 bytes/token. MLA page is not divisible by 768,
# so exact-fit is impossible and the drafter remains standalone.
spec_b = build_spec(draft_kv_heads=3)
groups_b, layout_b, cfg_b, pb_b, _ = run_geometry(spec_b, "B/standalone")

draft_group_b = layout_b[8]
draft_specs_b = draft_group_b.kv_cache_spec.kv_cache_specs
draft_names_b = list(draft_group_b.layer_names)
db = next(iter(draft_specs_b.values()))
assert db.block_size == 2048
assert db.page_size_padded is None

draft_page_b = db.page_size_bytes
idx_page_b = layout_b[5]
expected_pb_b = 11 * MLA_PAGE + 11 * idx_page_b + 5 * draft_page_b
assert pb_b == expected_pb_b, (pb_b, expected_pb_b)

target_offsets = {
    tensor_for_layer(cfg_b, name).offset
    for name in layout_b[2]
}
for draft_name in draft_names_b:
    tensor = tensor_for_layer(cfg_b, draft_name)
    assert tensor.offset not in target_offsets
    assert tensor.block_stride == draft_page_b
    assert_tensor_bounds(tensor, draft_page_b, cfg_b.num_blocks)

kernel_block_b = 64
assert db.block_size % kernel_block_b == 0
ratio_b = db.block_size // kernel_block_b
kernel_page_b = 2 * kernel_block_b * db.num_kv_heads * db.head_size
assert ratio_b * kernel_page_b == draft_page_b
print(
    f"[B] standalone OK: page={draft_page_b}, "
    f"kernel split {db.block_size}->{kernel_block_b} (x{ratio_b})"
)

print("ALL SIMULATION CHECKS PASSED")
