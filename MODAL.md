# Modal B300 deployment

This fork now includes modal_pull.py / modal_serve.py for a Modal-native deployment.

## Design

Model weights are never downloaded on a GPU container.

- pull_target, pull_drafter, and pull_models are CPU-only Modal Functions.
- They persist weights in the glm53-flash-dflash2-models Modal Volume.
- Each completed model gets a _modal_manifest.json containing the resolved Hugging Face revision.
- The B300 server mounts the model Volume read-only.
- If either model is missing, the B300 container fails immediately instead of spending GPU time downloading weights.
- The serving image reproduces this repository Dockerfile: vllm/vllm-openai:glm53-flash plus overlay patches and the SM100/B300 geometry validation.

## Setup

    uv sync
    uv run modal profile current

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
- deepseek_r1 reasoning parser
- model Volume mounted read-only
- Modal proxy authentication enabled

## License

Repository code and model artifacts have separate licenses. The upstream README currently states that incoai/GLM-5.3-Flash-DFlash2 uses CC BY-NC-ND 4.0. Re-check the model licenses before commercial use.

