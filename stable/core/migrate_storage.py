"""One-time relocation of a ``~/.stable`` store into the current data root.

:mod:`stable.core.paths` changed where projects and checkpoints live: next to
the installation rather than under the user's home directory. Anyone upgrading
already has both stores on the old drive — often tens of gigabytes of proxies
and renders — and "your projects are gone" is not an acceptable upgrade note.
So the app moves them on its next launch.

What this is careful about, because the data is large and irreplaceable:

#. **Per entry, not per tree.** Each project directory is moved on its own, so
   an interrupted run resumes on the next launch instead of restarting, and a
   name that already exists at the destination is left alone rather than
   overwritten.
#. **Space is checked first.** ``C:`` to ``D:`` is a copy-then-delete, not a
   rename, so a destination with no room would otherwise half-fill the disk
   before failing. Too little space and the store is reported and skipped.
#. **A failed copy leaves no rubble.** If a copy dies partway, the half-written
   destination is removed, leaving the source untouched and the next launch
   free to retry.
#. **It never blocks startup.** Every failure is reported and skipped; the app
   comes up regardless, reading whatever is in the new root.

Entry points call :func:`migrate_legacy_storage` once, before serving.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .paths import legacy_root

__all__ = ["StoreMove", "legacy_stores", "planned_moves", "migrate_legacy_storage"]

HEADROOM_FRACTION = 0.05
MIN_HEADROOM_BYTES = 64 << 20
"""Free space to leave spare at the destination, over and above what is moved.

A drive filled to the last byte by a migration is a broken machine, not a
successful upgrade — and the very next thing the app does is write proxies. The
margin scales with the payload (5%, never below 64 MB) rather than being a flat
figure: a fixed gigabyte would refuse to relocate a handful of small projects
onto a nearly-full disk that has ample room for them, which is exactly the
situation a user hits on the way *out* of a full drive.
"""

_PREFIX = "stable:"


@dataclass(frozen=True)
class StoreMove:
    """One directory to relocate, and where it is going.

    Args:
        store: The store's name (``"projects"``, ``"models"``), used in messages.
        source: The directory under the legacy root, known to exist.
        destination: The directory it should end up as, which may not exist yet.
    """

    store: str
    source: Path
    destination: Path

    @property
    def entries(self) -> list[Path]:
        """Return the individual children to move, sorted for stable output."""
        return sorted(self.source.iterdir())


def legacy_stores() -> Mapping[str, Path]:
    """Return each store's current root, keyed by its legacy sub-directory name.

    The two roots are owned by modules that sit *above* :mod:`stable.core` in
    the dependency order (the server owns the workspace, the tracker package
    owns the weight cache), so they are imported inside this function rather
    than at module scope. Both are tiny configuration modules — importing them
    drags in neither FastAPI nor torch.

    Returns:
        ``{"projects": <workspace root>, "models": <model cache root>}``,
        honouring ``$STABLE_WORKSPACE_DIR`` and ``$STABLE_MODEL_DIR``.
    """
    from ..server.config import workspace_root  # noqa: PLC0415 - see the docstring
    from ..trackers.weights import model_root  # noqa: PLC0415 - see the docstring

    return {"projects": workspace_root(), "models": model_root()}


def planned_moves(destinations: Mapping[str, Path] | None = None) -> list[StoreMove]:
    """Return the stores that still need moving out of the legacy root.

    Args:
        destinations: Where each legacy sub-directory should end up; defaults
            to :func:`legacy_stores`.

    Returns:
        A :class:`StoreMove` per store that exists under ``~/.stable``, holds
        something, and is not already the destination. Empty when there is
        nothing to do — which is every launch after the first.
    """
    legacy = legacy_root()
    if not legacy.is_dir():
        return []
    moves: list[StoreMove] = []
    for store, destination in (destinations or legacy_stores()).items():
        source = legacy / store
        if not source.is_dir() or _same_path(source, destination):
            continue
        if not any(source.iterdir()):
            continue
        moves.append(StoreMove(store=store, source=source, destination=destination))
    return moves


def migrate_legacy_storage(
    destinations: Mapping[str, Path] | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> list[StoreMove]:
    """Move everything still under ``~/.stable`` into the current data root.

    Safe to call on every launch: once the move has happened it is a directory
    listing and nothing more. Never raises — a store that cannot be moved is
    reported and left where it is, because failing to relocate old data is not
    a reason to refuse to start.

    Args:
        destinations: Where each legacy sub-directory should end up; defaults
            to :func:`legacy_stores`.
        log: Where progress goes; defaults to printing to stdout, which is the
            terminal the user started ``stable serve`` in.

    Returns:
        The moves that were attempted (empty when there was nothing to do).
    """
    emit = log if log is not None else _print
    try:
        moves = planned_moves(destinations)
    except OSError as exc:  # pragma: no cover - an unreadable home directory
        emit(f"{_PREFIX} could not inspect {legacy_root()}: {exc}")
        return []
    if not moves:
        return []

    emit(f"{_PREFIX} moving your data out of {legacy_root()} (this happens once)")
    for move in moves:
        try:
            _run_move(move, emit)
        except OSError as exc:  # pragma: no cover - reported, never fatal
            emit(f"{_PREFIX}   {move.store}: failed, left in place ({exc})")
    _prune_empty(legacy_root(), emit)
    return moves


# -- internals -------------------------------------------------------------


def _print(message: str) -> None:
    """Print a progress line, flushed (startup output is often redirected)."""
    print(message, flush=True)


def _same_path(left: Path, right: Path) -> bool:
    """Return whether two paths name the same location, existing or not."""
    try:
        return left.resolve() == right.resolve()
    except OSError:  # pragma: no cover - unresolvable on a broken mount
        return False


def _entry_size(path: Path) -> int:
    """Return the total size of ``path``, summed over every file beneath it."""
    if path.is_file() and not path.is_symlink():
        return path.stat().st_size
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:  # pragma: no cover - vanished mid-walk
            continue
    return total


def _human(size: float) -> str:
    """Format a byte count the way a progress line wants it (``"8.4 GB"``)."""
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _free_space(path: Path) -> int | None:
    """Return free bytes at ``path``'s nearest existing ancestor, or ``None``."""
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            try:
                return shutil.disk_usage(candidate).free
            except OSError:  # pragma: no cover - a disappearing mount
                return None
    return None  # pragma: no cover - some ancestor always exists


def _run_move(move: StoreMove, emit: Callable[[str], None]) -> None:
    """Move one store's entries, skipping names the destination already has.

    Args:
        move: The store to relocate.
        emit: Progress sink.

    Raises:
        OSError: If the source cannot be listed or the destination created.
    """
    present = move.entries
    entries = [entry for entry in present if not (move.destination / entry.name).exists()]
    if len(present) != len(entries):
        already = len(present) - len(entries)
        emit(f"{_PREFIX}   {move.store}: {already} already in the new location, left alone")
    if not entries:
        _prune_empty(move.source, emit)
        return

    needed = sum(_entry_size(entry) for entry in entries)
    required = needed + max(int(needed * HEADROOM_FRACTION), MIN_HEADROOM_BYTES)
    free = _free_space(move.destination)
    if free is not None and required > free:
        emit(
            f"{_PREFIX}   {move.store}: needs {_human(needed)} but only {_human(free)} is "
            f"free at {move.destination} — skipped, still readable from {move.source}"
        )
        return

    emit(f"{_PREFIX}   {move.store}: {len(entries)} items, {_human(needed)} -> {move.destination}")
    move.destination.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries, start=1):
        target = move.destination / entry.name
        try:
            shutil.move(str(entry), str(target))
        except (OSError, shutil.Error) as exc:
            _discard(target)
            emit(f"{_PREFIX}     {entry.name}: failed, left in place ({exc})")
            continue
        emit(f"{_PREFIX}     {entry.name}  ({index}/{len(entries)})")
    _prune_empty(move.source, emit)


def _discard(path: Path) -> None:
    """Remove a half-written destination so the next launch can retry cleanly."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - locked by another process
        pass


def _prune_empty(path: Path, emit: Callable[[str], None]) -> None:
    """Remove ``path`` if it is now empty, and say so once it is gone."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            emit(f"{_PREFIX}   removed the empty {path}")
    except OSError:  # pragma: no cover - in use, or not ours to remove
        pass
