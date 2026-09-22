# Modal B300 deployment

This fork now includes modal_pull.py / modal_serve.py for a Modal-native deployment.

## Design

Model weights are never downloaded on a GPU container.

- pull_target, pull_drafter, and pull_models are CPU-only Modal Functions.
- They persist weights in the glm53-flash-dflash2-models Modal Volume.
- Each completed model gets a _modal_manifest.json containing the resolved Hugging Face revision.
- The B300 server mounts the model Volume read-only.
- If either model is missing, the B300 container fails immediately instead of spending GPU time downloading weights.
- The serving image is pinned to `vllm/vllm-openai@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4` (vLLM `0.1.dev20051+g487ecf187`, FlashInfer 0.6.17), then applies the matching SM100/B300 overlay and geometry validation.
- Compile/JIT artifacts use the runtime-specific `glm53-flash-compile-cache-good-v1` Volume; do not reuse caches from newer incompatible vLLM/FlashInfer builds.

## Setup

    uv sync
    uv run modal profile current

## API authentication

The public Modal endpoint disables Modal proxy auth and lets vLLM enforce Bearer authentication. The API key is injected from the Modal Secret `glm53-api-key` via `GLM53_API_KEY`; never commit the value to this repository.

Create or rotate it with:

    uv run modal secret create glm53-api-key GLM53_API_KEY=<your-fixed-key> --force

## CPU-only pulls

Inspect:

    uv run modal run modal_pull.py --action inspect

Pull DFlash2 drafter:

    uv run modal run modal_pull.py --action pull-drafter

Pull NVFP4 target:

    uv run modal run modal_pull.py --action pull-target

Pull both:

    uv run modal run modal_pull.py --action pull-all

No pull Function requests a GPU.

Defaults:

- target: local-inference-lab/GLM-5.3-Flash-NVFP4
- drafter: incoai/GLM-5.3-Flash-DFlash2

## B300 serving

After both CPU pulls complete:

    uv run modal deploy modal_serve.py

Serving configuration:

- 1x Modal B300
- TP1
- FP8 KV
- DFlash2 with 7 speculative tokens
- 1,000,000 max model length by default
- glm47 tool parser
- glm47 reasoning parser
- model Volume mounted read-only
- Modal proxy authentication disabled; vLLM Bearer API-key authentication enabled
- maximum one B300 container (`max_containers=1`)
- scale to zero after 15 minutes idle

## License

Repository code and model artifacts have separate licenses. The upstream README currently states that incoai/GLM-5.3-Flash-DFlash2 uses CC BY-NC-ND 4.0. Re-check the model licenses before commercial use.

