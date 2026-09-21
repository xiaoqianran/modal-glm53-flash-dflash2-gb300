from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "glm53-flash-dflash2-pull"
MODEL_VOLUME_NAME = "glm53-flash-dflash2-models"

TARGET_REPO = os.environ.get(
    "GLM53_TARGET_REPO", "local-inference-lab/GLM-5.3-Flash-NVFP4"
)
TARGET_REVISION = os.environ.get(
    "GLM53_TARGET_REVISION", "175ae8ce3b5af842b0d0140dbeb43e9cfc557c49"
)
DRAFTER_REPO = os.environ.get(
    "GLM53_DRAFTER_REPO", "incoai/GLM-5.3-Flash-DFlash2"
)
DRAFTER_REVISION = os.environ.get(
    "GLM53_DRAFTER_REVISION", "bf582e4eacc1810f76656d1811693ff6c6737d2a"
)

MODEL_MOUNT = Path("/models")
TARGET_DIR = MODEL_MOUNT / "target"
DRAFTER_DIR = MODEL_MOUNT / "drafter"

app = modal.App(APP_NAME)
models = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# Deliberately CPU-only. No function in this file requests a GPU.
pull_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]>=0.35,<1")
    .env(
        {
            "HF_HOME": "/models/.hf",
            "HF_HUB_CACHE": "/models/.hf/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)


def _manifest_path(destination: Path) -> Path:
    return destination / "_modal_manifest.json"


def _read_manifest(destination: Path) -> dict | None:
    marker = _manifest_path(destination)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _pull_repo(repo_id: str, revision: str | None, destination: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    destination.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(repo_id, revision=revision)
    resolved_revision = info.sha

    current = _read_manifest(destination)
    if (
        current
        and current.get("repo_id") == repo_id
        and current.get("revision") == resolved_revision
    ):
        return {
            "repo_id": repo_id,
            "revision": resolved_revision,
            "path": str(destination),
            "cached": True,
            "bytes": _dir_size_bytes(destination),
        }

    snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        local_dir=str(destination),
        max_workers=16,
    )

    manifest = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "revision": resolved_revision,
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    _manifest_path(destination).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Make the completed snapshot visible to future containers immediately.
    models.commit()

    manifest["bytes"] = _dir_size_bytes(destination)
    manifest["cached"] = False
    return manifest


_CPU_PULL_OPTIONS = {
    "image": pull_image,
    "volumes": {str(MODEL_MOUNT): models},
    "cpu": 16,
    "memory": 32768,
    "timeout": 24 * 60 * 60,
    "retries": 2,
}


@app.function(**_CPU_PULL_OPTIONS)
def pull_target() -> dict:
    return _pull_repo(TARGET_REPO, TARGET_REVISION, TARGET_DIR)


@app.function(**_CPU_PULL_OPTIONS)
def pull_drafter() -> dict:
    return _pull_repo(DRAFTER_REPO, DRAFTER_REVISION, DRAFTER_DIR)


@app.function(**_CPU_PULL_OPTIONS)
def pull_models() -> dict:
    return {
        "target": _pull_repo(TARGET_REPO, TARGET_REVISION, TARGET_DIR),
        "drafter": _pull_repo(DRAFTER_REPO, DRAFTER_REVISION, DRAFTER_DIR),
    }


@app.function(
    image=pull_image,
    volumes={str(MODEL_MOUNT): models},
    cpu=1,
    memory=1024,
    timeout=300,
)
def inspect_cache() -> dict:
    models.reload()
    return {
        "target": {
            "exists": TARGET_DIR.exists(),
            "manifest": _read_manifest(TARGET_DIR),
            "bytes": _dir_size_bytes(TARGET_DIR),
        },
        "drafter": {
            "exists": DRAFTER_DIR.exists(),
            "manifest": _read_manifest(DRAFTER_DIR),
            "bytes": _dir_size_bytes(DRAFTER_DIR),
        },
    }


@app.local_entrypoint()
def main(action: str = "inspect"):
    actions = {
        "inspect": inspect_cache,
        "pull-target": pull_target,
        "pull-drafter": pull_drafter,
        "pull-all": pull_models,
    }
    if action not in actions:
        raise ValueError(f"Unknown action {action!r}; choose from {sorted(actions)}")
    print(json.dumps(actions[action].remote(), indent=2, sort_keys=True))
