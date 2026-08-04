"""Lazy, checksummed model-weight download to ``~/.stable/models``.

Deep-learning plugins need checkpoints that are far too large to ship in a pip
package (and, for CoTracker, are non-commercially licensed and must never be
bundled). This module is the one place that fetches them, under four rules:

#. **Lazy.** Nothing is downloaded at import, registration or
   :meth:`~stable.trackers.base.TrackerPlugin.is_available` time — only when a
   plugin actually runs :meth:`~stable.trackers.base.TrackerPlugin.prepare`.
#. **Checksummed.** A download is written to a ``.part`` file, hashed, and only
   then moved into place, so an interrupted or corrupted transfer can never be
   mistaken for a good checkpoint. Verification results are cached in a sidecar
   file keyed on size and mtime, so re-preparing a plugin does not re-hash
   hundreds of megabytes.
#. **Fail closed.** A :class:`ModelSpec` whose ``sha256`` is unknown refuses to
   download unless the caller (or ``STABLE_ALLOW_UNPINNED_WEIGHTS=1``) opts in.
   The digest of what was actually fetched is then logged so it can be pinned.
#. **Non-commercial weights are opt-in.** Downloading a spec marked
   ``non_commercial`` requires an explicit acknowledgement (argument or
   ``STABLE_ACCEPT_NONCOMMERCIAL=1``). Weights already on disk are always usable
   — the gate is on acquiring them, not on running them.

Environment variables
---------------------
``STABLE_MODEL_DIR``
    Overrides the ``~/.stable/models`` root (useful in CI and in Docker).
``STABLE_ALLOW_UNPINNED_WEIGHTS``
    ``1`` to permit downloading specs with no pinned digest.
``STABLE_ACCEPT_NONCOMMERCIAL``
    ``1`` to accept non-commercial model licences such as CoTracker's CC-BY-NC.
``STABLE_MODEL_SHA256_<KEY>``
    Pins (or re-pins) one spec's digest without editing code, where ``<KEY>`` is
    the spec key upper-cased with non-alphanumerics replaced by ``_``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..core.errors import WeightsError
from .base import ProgressCallback, noop_progress

__all__ = [
    "ModelSpec",
    "ENV_MODEL_DIR",
    "ENV_ALLOW_UNPINNED",
    "ENV_ACCEPT_NONCOMMERCIAL",
    "model_root",
    "local_path",
    "expected_sha256",
    "sha256_file",
    "file_digest",
    "is_downloaded",
    "ensure_weights",
    "clear_weights",
    "register_model",
    "registered_models",
    "describe_models",
    "env_flags",
]

logger = logging.getLogger(__name__)

ENV_MODEL_DIR = "STABLE_MODEL_DIR"
"""Environment variable overriding the model cache root."""

ENV_ALLOW_UNPINNED = "STABLE_ALLOW_UNPINNED_WEIGHTS"
"""Environment variable allowing downloads of specs with no pinned digest."""

ENV_ACCEPT_NONCOMMERCIAL = "STABLE_ACCEPT_NONCOMMERCIAL"
"""Environment variable accepting non-commercial model licences."""

_DEFAULT_ROOT = Path("~/.stable/models")
_CHUNK = 1 << 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _truthy(value: str | None) -> bool:
    """Return whether an environment-variable string means "yes"."""
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable checkpoint: where it lives, what it must hash to, its licence.

    Args:
        key: Stable identifier, also the cache sub-directory name (e.g.
            ``"cotracker3_offline"``).
        url: HTTP(S) — or, for tests and air-gapped mirrors, ``file://`` — URL of
            the checkpoint.
        filename: Name to store it under; defaults to the URL's basename.
        sha256: Expected lower-case hex digest, or ``None`` when the digest has
            not been pinned yet (downloads then require an explicit opt-in).
        size_bytes: Expected size, used only to show a progress percentage
            when the server sends no ``Content-Length``.
        license: SPDX-ish licence of the **weights** (not of our code).
        non_commercial: True when the licence forbids commercial use; such
            weights are never bundled and require an explicit opt-in to fetch.
        notes: Free-text provenance shown to the user (paper, release page).

    Raises:
        WeightsError: If ``key`` or ``url`` is empty, or ``sha256`` is not a
            64-character hex digest.
    """

    key: str
    url: str
    filename: str = ""
    sha256: str | None = None
    size_bytes: int | None = None
    license: str = "MIT"
    non_commercial: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalise and validate the spec (dataclass hook)."""
        if not self.key.strip():
            raise WeightsError("ModelSpec.key must be a non-empty identifier")
        if not self.url.strip():
            raise WeightsError(f"ModelSpec {self.key!r} has an empty url")
        if not self.filename:
            basename = os.path.basename(urlparse(self.url).path) or f"{self.key}.pth"
            object.__setattr__(self, "filename", basename)
        if self.sha256 is not None:
            digest = self.sha256.strip().lower()
            if not _HEX64.match(digest):
                raise WeightsError(
                    f"ModelSpec {self.key!r} sha256 must be 64 hex characters, got {self.sha256!r}"
                )
            object.__setattr__(self, "sha256", digest)

    @property
    def pinned(self) -> bool:
        """Whether a digest is known for this checkpoint (in the spec or the env)."""
        return expected_sha256(self) is not None

    def describe(self) -> dict[str, Any]:
        """Return a UI-ready dict describing this checkpoint and its cache state.

        Returns:
            Keys ``key``, ``filename``, ``url``, ``license``, ``non_commercial``,
            ``pinned``, ``path`` and ``downloaded``.
        """
        return {
            "key": self.key,
            "filename": self.filename,
            "url": self.url,
            "license": self.license,
            "non_commercial": self.non_commercial,
            "pinned": self.pinned,
            "path": str(local_path(self)),
            "downloaded": is_downloaded(self),
        }


_MODELS: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    """Add ``spec`` to the process-wide model table.

    Plugins call this at import time so a model manager (and ``stable models``
    in a later phase) can enumerate every checkpoint the installed plugins may
    want, without importing torch or touching the network.

    Args:
        spec: The checkpoint to register. Re-registering an identical spec is a
            no-op.

    Returns:
        ``spec``, unchanged.

    Raises:
        WeightsError: If a *different* spec already claims that key.
    """
    existing = _MODELS.get(spec.key)
    if existing is not None and existing != spec:
        raise WeightsError(f"model key {spec.key!r} is already registered by another spec")
    _MODELS[spec.key] = spec
    return spec


def registered_models() -> list[ModelSpec]:
    """Return every registered :class:`ModelSpec`, sorted by key."""
    return [_MODELS[key] for key in sorted(_MODELS)]


def describe_models() -> list[dict[str, Any]]:
    """Return :meth:`ModelSpec.describe` for every registered checkpoint."""
    return [spec.describe() for spec in registered_models()]


def env_flags() -> dict[str, bool]:
    """Report whether the process-wide download opt-in environment variables are set.

    The model manager UI (:mod:`stable.server.routes.models`) surfaces this so
    "why is this download blocked" has a real answer instead of the user
    discovering the *fail closed* rule (see the module docstring) from a failed
    request. It does not replace the per-call ``accept_noncommercial`` /
    ``allow_unpinned`` arguments to :func:`ensure_weights` — either the env var
    or an explicit opt-in satisfies the gate.

    Returns:
        ``{"allow_unpinned": ..., "accept_noncommercial": ...}``.
    """
    return {
        "allow_unpinned": _truthy(os.environ.get(ENV_ALLOW_UNPINNED)),
        "accept_noncommercial": _truthy(os.environ.get(ENV_ACCEPT_NONCOMMERCIAL)),
    }


def model_root() -> Path:
    """Return the directory checkpoints are cached in.

    Returns:
        ``$STABLE_MODEL_DIR`` when set, else ``~/.stable/models``. The directory
        is not created here — :func:`ensure_weights` creates it on demand.
    """
    override = os.environ.get(ENV_MODEL_DIR)
    root = Path(override).expanduser() if override else _DEFAULT_ROOT.expanduser()
    return root


def local_path(spec: ModelSpec) -> Path:
    """Return where ``spec`` is (or would be) cached on disk."""
    return model_root() / spec.key / spec.filename


def expected_sha256(spec: ModelSpec) -> str | None:
    """Return the digest to verify ``spec`` against, honouring the env override.

    Args:
        spec: The checkpoint.

    Returns:
        A lower-case hex digest, or ``None`` if the checkpoint is unpinned.

    Raises:
        WeightsError: If the environment override is not a valid digest.
    """
    env_key = "STABLE_MODEL_SHA256_" + re.sub(r"[^0-9A-Za-z]", "_", spec.key).upper()
    override = os.environ.get(env_key)
    if override:
        digest = override.strip().lower()
        if not _HEX64.match(digest):
            raise WeightsError(f"{env_key} must be 64 hex characters, got {override!r}")
        return digest
    return spec.sha256


def sha256_file(path: str | os.PathLike[str], *, progress: ProgressCallback = noop_progress) -> str:
    """Return the SHA-256 of a file, streaming it in 1 MiB blocks.

    Args:
        path: File to hash.
        progress: Reporter called with the fraction hashed so far.

    Returns:
        The lower-case hex digest.
    """
    path = Path(path)
    total = path.stat().st_size or 1
    digest = hashlib.sha256()
    done = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
            done += len(block)
            progress(min(done / total, 1.0), "verifying checksum")
    return digest.hexdigest()


def _stamp_path(path: Path) -> Path:
    """Return the sidecar path caching a verified digest for ``path``."""
    return path.with_name(path.name + ".sha256.json")


def _read_stamp(path: Path) -> dict[str, Any] | None:
    """Return the cached verification record for ``path``, if it is still valid."""
    stamp = _stamp_path(path)
    try:
        record = json.loads(stamp.read_text(encoding="utf-8"))
        info = path.stat()
    except (OSError, ValueError):
        return None
    if record.get("size") != info.st_size or record.get("mtime_ns") != info.st_mtime_ns:
        return None
    digest = record.get("sha256")
    return record if isinstance(digest, str) else None


def _write_stamp(path: Path, digest: str) -> None:
    """Record that ``path`` hashed to ``digest`` at its current size and mtime."""
    info = path.stat()
    record = {"sha256": digest, "size": info.st_size, "mtime_ns": info.st_mtime_ns}
    try:
        _stamp_path(path).write_text(json.dumps(record), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - read-only cache dir only
        logger.debug("could not write checksum stamp for %s: %s", path, exc)


def file_digest(path: Path, *, progress: ProgressCallback = noop_progress) -> str:
    """Return ``path``'s digest, using the sidecar cache when it is still valid.

    Args:
        path: File to hash.
        progress: Reporter forwarded to :func:`sha256_file` on a cache miss.

    Returns:
        The lower-case hex digest.
    """
    cached = _read_stamp(path)
    if cached is not None:
        return str(cached["sha256"])
    digest = sha256_file(path, progress=progress)
    _write_stamp(path, digest)
    return digest


def is_downloaded(spec: ModelSpec) -> bool:
    """Whether ``spec`` is already cached and matches its pinned digest.

    An unpinned spec counts as downloaded as soon as the file exists — there is
    nothing to check it against.

    Args:
        spec: The checkpoint.

    Returns:
        True if a later :func:`ensure_weights` call would not need the network.
    """
    path = local_path(spec)
    if not path.is_file():
        return False
    digest = expected_sha256(spec)
    if digest is None:
        return True
    try:
        return file_digest(path) == digest
    except OSError:  # pragma: no cover - unreadable file
        return False


def _download(
    url: str,
    destination: Path,
    *,
    progress: ProgressCallback,
    timeout: float,
    expected_size: int | None,
) -> None:
    """Stream ``url`` into ``destination`` via a temporary file in the same dir.

    Raises:
        WeightsError: If the transfer fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            request = urllib.request.Request(url, headers={"User-Agent": "stable-tracker"})
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                header = response.headers.get("Content-Length") if response.headers else None
                total = int(header) if header and header.isdigit() else (expected_size or 0)
                done = 0
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    fraction = min(done / total, 1.0) if total else 0.0
                    progress(fraction, f"downloading {destination.name}")
        shutil.move(str(temp_path), str(destination))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        raise WeightsError(f"could not download {url}: {exc}") from exc
    finally:
        if temp_path.exists():  # pragma: no cover - only on a failed move
            temp_path.unlink(missing_ok=True)


def ensure_weights(
    spec: ModelSpec,
    *,
    progress: ProgressCallback = noop_progress,
    allow_download: bool = True,
    accept_noncommercial: bool | None = None,
    allow_unpinned: bool | None = None,
    timeout: float = 60.0,
) -> Path:
    """Return the local path to ``spec``'s checkpoint, downloading it if needed.

    This is the single entry point plugins call from ``prepare()``. It is safe
    to call repeatedly: a cached, verified file short-circuits everything.

    Args:
        spec: The checkpoint to materialise.
        progress: Reporter for the download and checksum passes.
        allow_download: When False, a missing checkpoint raises instead of
            hitting the network (offline mode / "is it cached?" checks).
        accept_noncommercial: Explicitly accept a non-commercial licence.
            ``None`` falls back to ``STABLE_ACCEPT_NONCOMMERCIAL``.
        allow_unpinned: Explicitly allow downloading a spec with no pinned
            digest. ``None`` falls back to ``STABLE_ALLOW_UNPINNED_WEIGHTS``.
        timeout: Socket timeout in seconds.

    Returns:
        Path to the verified checkpoint.

    Raises:
        WeightsError: If the file is missing and downloads are disabled, the
            licence or pinning opt-in is missing, the transfer fails, or the
            downloaded bytes do not match the pinned digest (in which case the
            bad file is removed).
    """
    path = local_path(spec)
    digest = expected_sha256(spec)

    if path.is_file():
        if digest is None:
            return path
        actual = file_digest(path, progress=progress)
        if actual == digest:
            return path
        raise WeightsError(
            f"cached weights for {spec.key!r} at {path} are corrupt "
            f"(sha256 {actual}, expected {digest}); delete the file and retry"
        )

    if not allow_download:
        raise WeightsError(
            f"weights for {spec.key!r} are not downloaded (expected at {path}) "
            "and downloading is disabled"
        )

    accepted = (
        _truthy(os.environ.get(ENV_ACCEPT_NONCOMMERCIAL))
        if accept_noncommercial is None
        else accept_noncommercial
    )
    if spec.non_commercial and not accepted:
        raise WeightsError(
            f"{spec.key!r} weights are licensed {spec.license} (non-commercial) and are never "
            f"bundled. Re-run with accept_noncommercial=True or set "
            f"{ENV_ACCEPT_NONCOMMERCIAL}=1 to download them. Source: {spec.url}"
        )

    unpinned_ok = (
        _truthy(os.environ.get(ENV_ALLOW_UNPINNED)) if allow_unpinned is None else allow_unpinned
    )
    if digest is None and not unpinned_ok:
        raise WeightsError(
            f"no sha256 is pinned for {spec.key!r}, so the download cannot be verified. "
            f"Pin one via STABLE_MODEL_SHA256_{re.sub(r'[^0-9A-Za-z]', '_', spec.key).upper()}, "
            f"or set {ENV_ALLOW_UNPINNED}=1 to download it unverified (the digest of what you "
            "get is then logged so it can be pinned)."
        )

    logger.info("downloading %s weights from %s", spec.key, spec.url)
    _download(spec.url, path, progress=progress, timeout=timeout, expected_size=spec.size_bytes)

    actual = sha256_file(path, progress=progress)
    if digest is not None and actual != digest:
        path.unlink(missing_ok=True)
        raise WeightsError(
            f"checksum mismatch for {spec.key!r}: downloaded sha256 {actual}, expected {digest}. "
            "The file has been deleted; check the URL or re-pin the digest."
        )
    _write_stamp(path, actual)
    if digest is None:
        logger.warning(
            "downloaded %s unverified; pin this digest to make future downloads checked: %s",
            spec.key,
            actual,
        )
    progress(1.0, f"{spec.key} ready")
    return path


def clear_weights(spec: ModelSpec) -> bool:
    """Delete ``spec``'s cached checkpoint and its checksum stamp.

    Args:
        spec: The checkpoint to remove.

    Returns:
        True if a file was deleted.
    """
    path = local_path(spec)
    stamp = _stamp_path(path)
    existed = path.is_file()
    path.unlink(missing_ok=True)
    stamp.unlink(missing_ok=True)
    return existed
