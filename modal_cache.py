from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import modal

APP_NAME = "glm53-flash-cache-restore"
CACHE_VOLUME_NAME = "glm53-flash-compile-cache-good-v1"
CACHE_TAG = "glm53-flash-dflash2-g487ecf187-fi0617-sm103a-b300-cache-v1"
RELEASE_REPO = "xiaoqianran/modal-build"
CACHE_MOUNT = Path("/compile-cache")

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "modal-glm53-cache-restore/1"},
    )
    with (
        urllib.request.urlopen(request, timeout=15 * 60) as response,
        destination.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"unsafe archive member outside cache root: {member.name}"
                ) from exc
        bundle.extractall(destination)


def _release_url(filename: str) -> str:
    return (
        f"https://github.com/{RELEASE_REPO}/releases/download/"
        f"{CACHE_TAG}/{filename}"
    )


@app.function(
    image=image,
    volumes={str(CACHE_MOUNT): cache},
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    retries=2,
)
def restore(force: bool = False) -> dict:
    """Restore the pinned public GitHub Release entirely on CPU."""
    marker = CACHE_MOUNT / ".glm53-release-manifest.json"
    if marker.exists() and not force:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("tag") == CACHE_TAG:
            return {"cached": True, "manifest": payload}

    with tempfile.TemporaryDirectory(prefix="glm53-cache-") as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / f"{CACHE_TAG}.manifest.json"
        archive_path = tmp_dir / f"{CACHE_TAG}.cache.tar.gz"
        sha_path = tmp_dir / f"{CACHE_TAG}.cache.tar.gz.sha256"

        for remote_name, local_path in (
            (manifest_path.name, manifest_path),
            (archive_path.name, archive_path),
            (sha_path.name, sha_path),
        ):
            _download(_release_url(remote_name), local_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tag") != CACHE_TAG:
            raise RuntimeError(
                f"release tag mismatch: {manifest.get('tag')!r} != {CACHE_TAG!r}"
            )
        if manifest.get("contains_model_weights") is not False:
            raise RuntimeError("refusing cache artifact that may contain model weights")

        expected_sha = sha_path.read_text(encoding="utf-8").split()[0].lower()
        actual_sha = _sha256(archive_path)
        if expected_sha != actual_sha:
            raise RuntimeError(
                f"cache SHA256 mismatch: expected={expected_sha} actual={actual_sha}"
            )
        if manifest.get("archive_sha256") != actual_sha:
            raise RuntimeError("manifest archive_sha256 does not match downloaded cache")

        _safe_extract(archive_path, CACHE_MOUNT)

        marker.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cache.commit()

        return {
            "cached": False,
            "tag": CACHE_TAG,
            "archive_bytes": archive_path.stat().st_size,
            "sha256": actual_sha,
            "file_count": manifest.get("file_count"),
            "uncompressed_bytes": manifest.get("uncompressed_bytes"),
        }



@app.function(
    image=image,
    volumes={str(CACHE_MOUNT): cache},
    cpu=1,
    memory=1024,
    timeout=300,
)
def seed(force: bool = False) -> dict:
    """Create a clean bootstrap marker for a brand-new runtime-specific cache."""
    marker = CACHE_MOUNT / ".glm53-precompile.json"
    if marker.exists() and not force:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("tag") == CACHE_TAG:
            return {"cached": True, "manifest": payload}
    payload = {
        "tag": CACHE_TAG,
        "contains_model_weights": False,
        "source": "fresh-bootstrap",
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cache.commit()
    return {"cached": False, "manifest": payload}

@app.function(
    image=image,
    volumes={str(CACHE_MOUNT): cache},
    cpu=1,
    memory=1024,
    timeout=300,
)
def inspect() -> dict:
    marker = CACHE_MOUNT / ".glm53-release-manifest.json"
    precompile = CACHE_MOUNT / ".glm53-precompile.json"
    roots = {}
    if CACHE_MOUNT.exists():
        for item in CACHE_MOUNT.iterdir():
            roots[item.name] = "dir" if item.is_dir() else "file"
    return {
        "tag": CACHE_TAG,
        "release_marker": (
            json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        ),
        "precompile_marker": (
            json.loads(precompile.read_text(encoding="utf-8"))
            if precompile.exists()
            else None
        ),
        "roots": roots,
    }


@app.local_entrypoint()
def main(action: str = "inspect", force: bool = False):
    if action == "restore":
        print(json.dumps(restore.remote(force), indent=2, sort_keys=True))
    elif action == "seed":
        print(json.dumps(seed.remote(force), indent=2, sort_keys=True))
    elif action == "inspect":
        print(json.dumps(inspect.remote(), indent=2, sort_keys=True))
    else:
        raise ValueError("action must be 'restore', 'seed', or 'inspect'")

