# GLM-5.3-Flash + DFlash2 on DGX Station GB300 (SM100)

First-known (to me!) **DFlash2 speculative decoding for GLM-5.3-Flash on a GB300 / SM100** (NVIDIA DGX Station class).

This is an **SM100 adaptation** of [tonyd2wild's GB10/SM121 DFlash2 overlay](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark). The vLLM fork point is identical (`0.1.dev20051+g487ecf187`); we re-derived the GLM-5-Next KV-cache geometry for GB300 and validated it at build time before the first boot.

## Results (DGX Station GB300, single GPU, TP1)

| metric | value |
|---|---|
| model | `local-inference-lab/GLM-5.3-Flash-NVFP4` (calibrated, 169B) |
| drafter | `incoai/GLM-5.3-Flash-DFlash2` (BF16 block-diffusion, 7-token blocks) |
| decode (no-think, spec ON) | ~305–470 tok/s |
| decode (no-think, no-spec) | ~137 tok/s |
| speedup | ~2.2–3.4× |
| draft acceptance | ~54–67% (rises warm) |
| spec-decode KV | exact-fit, **zero extra KV cost** |

Per-position acceptance shows the healthy decaying curve (207→96 over the 7-token block), confirming the drafter + target + rejection sampler are wired correctly on SM100.

## Why this repo exists

The published DFlash2 recipes for GLM-5.3-Flash target DGX Spark / GB10 (SM121). The **local-inference-lab MXFP8 drafter** paired with that quant requires the **B12X** kernel stack, which is **SM120/SM121-only and cannot run on GB300 (SM100)**. So on a DGX Station you use the **BF16 incoai drafter** with an SM100 port of the overlay — that is what this repo ships.

## Quickstart

### Pull the prebuilt image (GHCR)

A prebuilt `vllm-glm53-dflash2` image is published to GHCR (SM100/GB300 build, 2026-09-01):

```bash
docker pull ghcr.io/ebfio/glm53-flash-dflash2-gb300:latest
# pin: ghcr.io/ebfio/glm53-flash-dflash2-gb300:sm100-gb300-20260901
```

### Or build it yourself

Dockerfile builds a ~25 s overlay on the public `vllm/vllm-openai:glm53-flash` image (same fork point as the overlay). No source build, no B12X.

```bash
docker build -t vllm-glm53-dflash2:latest .
```

### Serve

```bash
docker run --rm --gpus '"device=<GB300-uuid>"' \
  --ipc host --init \
  -v /models:/models:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_KV_CACHE_LAYOUT=HND \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
  vllm-glm53-dflash2:latest \
  --model /models/huggingface-cache/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots/<snapshot> \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --speculative-config '{"method":"dflash","model":"/models/huggingface-cache/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/<snapshot>","num_speculative_tokens":7}' \
  --reasoning-parser deepseek_r1 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --served-model-name glm-5.3-flash
```

`num_speculative_tokens` **must be 7** (drafter block size 8 minus the target's own token).

## Validation

- `overlay/sim_glm5_drafter_hades.py` validates the patched GLM-5-Next KV layout at GB300 geometry and **fails the build** if the exact-fit drafter slot-sharing ever breaks:
  - exact-fit: drafter block 2048→4288, real page == mla_page (4,390,912 B), no padding
  - per-block KV cost unchanged by the drafter
  - kernel-split reshape 4288→64 page-aligned, fits exactly
- Runs inside the Docker build (`RUN python3 /tmp/sim_glm5_drafter_hades.py`).

## Files

```
Dockerfile                  overlay onto vllm/vllm-openai:glm53-flash, build-time geometry check
overlay/
  qwen3_dflash2.py          DFlash2 drafter model (port from tonyd2wild overlay)
  dflash2/speculator.py     DFlash2 speculator
  patch_registry_and_select.py
  patch_glm_aux_capture.py  GLM aux-capture (hc_post contraction for the 5 tap layers)
  patch_kv_page_lcm2.py     no-op stub (kept for build-chain compat)
  patch_glm5_drafter_group.py  GLM-5-Next KV layout with drafter slot-sharing (the hard part)
  sim_glm5_drafter_hades.py GB300-geometry KV validation
  sim_glm5_drafter.py       upstream GB10 sim (unmodified, for reference)
```

## Credits

- [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) — the GB10 overlay this is ported from.
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — the BF16 block-diffusion drafter.
- [local-inference-lab/GLM-5.3-Flash-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4) — the target quant.
- [vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816) — upstream DFlash2 support.

## License / usage notes

- The **drafter** (`incoai/GLM-5.3-Flash-DFlash2`) is **CC BY-NC-ND 4.0** (non-commercial). Respect its terms.
- DFlash2 drafts **text only**; multimodal (image/video) requests pass through unspeculated.
- Port validated on one GB300 (SM100) at TP1; other SM100/SM120 configs are untested.

## Modal B300

This fork also contains a Modal-native B300 deployment path with CPU-only model downloads. See [MODAL.md](MODAL.md).
