from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

APP_NAME = "glm53-flash-dflash2-b300"
MODEL_VOLUME_NAME = "glm53-flash-dflash2-models"
COMPILE_CACHE_VOLUME_NAME = "glm53-flash-compile-cache-v1"
COMPILE_CACHE_TAG = "glm53-flash-dflash2-vllm0281rc1-fi0618-sm103a-b300-cache-v1"

MODEL_MOUNT = Path("/models")
TARGET_DIR = MODEL_MOUNT / "target"
DRAFTER_DIR = MODEL_MOUNT / "drafter"
COMPILE_CACHE_DIR = Path("/compile-cache")
OVERLAY_DIR = Path("/opt/glm53-overlay")

# vllm/vllm-openai:glm53-flash already ships vLLM under the distro Python.
# Modal adds its own Python at /usr/local/bin for lifecycle code, so all vLLM
# patch scripts must explicitly use the original interpreter.
VLLM_PYTHON = "/usr/bin/python3"
VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SERVER_PORT = 8000

# DFlash2 is strongest for interactive C1-C8 workloads in the upstream GB300
# recipe. Keep a single B300 replica until we benchmark Modal-specific behavior.
MAX_MODEL_LEN = os.environ.get("GLM53_MAX_MODEL_LEN", "1000000")
MAX_NUM_SEQS = os.environ.get("GLM53_MAX_NUM_SEQS", "8")
TARGET_CONCURRENCY = int(os.environ.get("GLM53_TARGET_CONCURRENCY", "8"))
ENABLE_MULTIMODAL = os.environ.get("GLM53_ENABLE_MULTIMODAL", "0") == "1"
ENABLE_WARMUP = os.environ.get("GLM53_WARMUP", "1") == "1"

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
compile_cache = modal.Volume.from_name(COMPILE_CACHE_VOLUME_NAME, create_if_missing=True)

# The specialized vLLM image cannot accept additional RUN layers in Modal's
# current builder. Instead, keep it immutable and mount this repo's small
# overlay at runtime. Image construction and validation remain CPU-only.
serve_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:glm53-flash",
        add_python="3.12",
    )
    .entrypoint([])
    .env(
        {
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "VLLM_LOG_STATS_INTERVAL": "1",
            "VLLM_CACHE_ROOT": "/compile-cache/vllm",
            "TILELANG_CACHE_DIR": "/compile-cache/tilelang",
            "TRITON_CACHE_DIR": "/compile-cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/compile-cache/torchinductor",
            "CUDA_CACHE_PATH": "/compile-cache/nv",
            "FLASHINFER_WORKSPACE_BASE": "/compile-cache/flashinfer-workspace",
            "OMP_NUM_THREADS": "4",
            # Serving must never download weights on an expensive B300.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    # copy=False makes this a runtime mount; it must be the final Image step.
    .add_local_dir(Path(__file__).parent / "overlay", OVERLAY_DIR, copy=False)
)


def _run_vllm_python(*args: str) -> None:
    cmd = [VLLM_PYTHON, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _prepare_runtime() -> None:
    """Apply and validate the repo's DFlash2 overlay before vLLM starts."""
    qwen2_dst = VLLM_ROOT / "model_executor/models/qwen3_dflash2.py"
    if not qwen2_dst.exists():
        shutil.copy2(OVERLAY_DIR / "qwen3_dflash2.py", qwen2_dst)

    dflash_dst = VLLM_ROOT / "v1/worker/gpu/spec_decode/dflash2"
    if not (dflash_dst / "speculator.py").exists():
        dflash_dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OVERLAY_DIR / "dflash2", dflash_dst, dirs_exist_ok=True)

    for patch in (
        "patch_registry_and_select.py",
        "patch_glm_aux_capture.py",
        "patch_kv_page_lcm2.py",
        "patch_glm5_drafter_group.py",
    ):
        _run_vllm_python(str(OVERLAY_DIR / patch))

    _run_vllm_python(
        "-c",
        (
            "from vllm.model_executor.models.registry import ModelRegistry; "
            'assert "DFlash2DraftModel" in ModelRegistry.get_supported_archs(); '
            'print("DFlash2 overlay registry check OK")'
        ),
    )
    _run_vllm_python(str(OVERLAY_DIR / "sim_glm5_drafter_hades.py"))


def _read_manifest(destination: Path) -> dict | None:
    marker = destination / "_modal_manifest.json"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _require_cached_model(path: Path, label: str) -> dict:
    manifest = _read_manifest(path)
    if not path.exists() or manifest is None:
        raise RuntimeError(
            f"{label} is not prepared at {path}. "
            "Run modal_pull.py first. The B300 server never downloads model weights."
        )
    if not (path / "config.json").exists():
        raise RuntimeError(f"{label} at {path} is incomplete: config.json is missing.")
    return manifest


def _require_compile_cache() -> dict:
    release_marker = COMPILE_CACHE_DIR / ".glm53-release-manifest.json"
    precompile_marker = COMPILE_CACHE_DIR / ".glm53-precompile.json"
    marker = release_marker if release_marker.exists() else precompile_marker
    if not marker.exists():
        raise RuntimeError(
            "GLM53 compile cache is not prepared. Run modal_cache.py on CPU before "
            "starting a B300, or run the modal-build precompile job."
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    tag = payload.get("tag")
    if tag != COMPILE_CACHE_TAG:
        raise RuntimeError(
            f"compile cache tag mismatch: {tag!r} != {COMPILE_CACHE_TAG!r}"
        )
    return payload


def _build_vllm_command() -> list[str]:
    speculative = json.dumps(
        {
            "method": "dflash",
            "model": str(DRAFTER_DIR),
            "num_speculative_tokens": 7,
        },
        separators=(",", ":"),
    )

    cmd = [
        "vllm",
        "serve",
        str(TARGET_DIR),
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--tensor-parallel-size",
        "1",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        MAX_MODEL_LEN,
        "--max-num-seqs",
        MAX_NUM_SEQS,
        "--speculative-config",
        speculative,
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm47",
        "--enable-auto-tool-choice",
        "--served-model-name",
        "glm-5.3-flash",
        "--generation-config",
        "vllm",
        "--uvicorn-log-level",
        "info",
    ]

    if not ENABLE_MULTIMODAL:
        cmd += [
            "--limit-mm-per-prompt",
            json.dumps({"image": 0, "video": 0}, separators=(",", ":")),
        ]

    return cmd


def _wait_until_ready(process: subprocess.Popen, timeout_s: int = 40 * 60) -> None:
    url = f"http://127.0.0.1:{SERVER_PORT}/health"
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM exited before becoming ready (exit={code}).")

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print("vLLM is ready.", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            pass

        time.sleep(2)

    raise TimeoutError(f"Timed out waiting for vLLM readiness at {url}.")


def _warmup_once() -> None:
    payload = json.dumps(
        {
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 16,
            "temperature": 0,
            "reasoning_effort": "low",
        }
    ).encode()

    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10 * 60) as response:
        body = response.read().decode("utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(f"Warmup failed HTTP {response.status}: {body[:1000]}")
        print("Warmup completed.", flush=True)


@app.function(
    image=serve_image,
    cpu=4,
    memory=8192,
    timeout=15 * 60,
)
def validate_runtime() -> dict:
    """CPU-only overlay validation; this function never allocates a GPU."""
    _prepare_runtime()

    root = VLLM_ROOT
    tool_init = (root / "tool_parsers/__init__.py").read_text(errors="ignore")
    reasoning_init = (root / "reasoning/__init__.py").read_text(errors="ignore")

    checks = {
        "dflash2_overlay": True,
        "glm47_tool_parser": '"glm47"' in tool_init,
        "glm47_reasoning_parser": '"glm47"' in reasoning_init,
        "cpu_only_validation": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Runtime validation failed: {checks}")
    return checks


@app.server(
    image=serve_image,
    gpu="B300",
    cpu=16,
    memory=(32768, 65536),
    volumes={
        str(MODEL_MOUNT): models.with_mount_options(read_only=True),
        str(COMPILE_CACHE_DIR): compile_cache,
    },
    port=SERVER_PORT,
    startup_timeout=45 * 60,
    min_containers=0,
    max_containers=1,
    target_concurrency=TARGET_CONCURRENCY,
    scaledown_window=30 * 60,
    exit_grace_period=5 * 60,
)
class Server:
    """Authenticated single-B300 OpenAI-compatible vLLM Server."""

    @modal.enter()
    def start(self):
        target_manifest = _require_cached_model(TARGET_DIR, "target model")
        drafter_manifest = _require_cached_model(DRAFTER_DIR, "DFlash2 drafter")
        cache_manifest = _require_compile_cache()
        print(
            "Target revision:",
            target_manifest.get("revision"),
            "Drafter revision:",
            drafter_manifest.get("revision"),
            "Compile cache:",
            cache_manifest.get("tag"),
            flush=True,
        )

        _prepare_runtime()

        cmd = _build_vllm_command()
        print("Starting:", " ".join(cmd), flush=True)
        self.process = subprocess.Popen(cmd, env=os.environ.copy())
        _wait_until_ready(self.process)

        if ENABLE_WARMUP:
            _warmup_once()

        # Persist any cache misses/new JIT artifacts produced by this exact
        # production configuration so the next container can reuse them.
        compile_cache.commit()
        print("Compile cache committed.", COMPILE_CACHE_TAG, flush=True)

    @modal.exit()
    def stop(self):
        # Persist late cache misses/JIT products generated after startup warmup.
        # Exit hooks are best-effort, so never mask shutdown with a commit error.
        try:
            compile_cache.commit()
            print("Compile cache committed on exit.", COMPILE_CACHE_TAG, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Best-effort compile cache commit failed on exit: {exc}", flush=True)

        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@app.local_entrypoint()
def validate():
    print(json.dumps(validate_runtime.remote(), indent=2, sort_keys=True))

