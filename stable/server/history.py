"""Undo/redo over the corrections layer and the anchors it hangs off.

``PLAN.md`` makes the promise that pays off here: *raw tracker output is
immutable, user edits are a correction layer*. So the entire undoable state of
an editing session is each point's corrections — and, since todo 6 of the
seventh round, its anchors, because deleting one now takes the corrections
rooted on it with it and half of that coming back would be worse than none of
it. Both together are a few hundred bytes of JSON, not arrays of frames. Undo is
therefore a snapshot stack rather than a command pattern with an inverse per
operation, which means:

* "undo restores the exact prior state" is true by construction, not by every
  operation implementing its own inverse correctly;
* an operation that touches several points at once (delete a range on all of
  them, undo an auto-repair) undoes as one step;
* the stack survives a server restart, because it is small enough to write to
  disk next to the project file.

Nothing else in the project is versioned here. Tracking, solving and smoothing
are recomputable from the project file, and putting them on the undo stack would
turn "undo my nudge" into "throw away twenty minutes of GPU time".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.corrections import correction_from_dict
from ..core.errors import ProjectError
from ..core.project import Project
from ..core.types import Query

__all__ = ["Snapshot", "EditHistory", "snapshot_corrections", "restore_corrections"]

Snapshot = dict[str, dict[str, list[dict[str, Any]]]]
"""Every point's anchors and corrections, keyed by point id — one undoable state.

Each entry is ``{"queries": [...], "corrections": [...]}``. Stacks written
before anchors were versioned hold a bare correction list per point instead;
:func:`restore_corrections` reads both, so an in-progress session survives the
upgrade with its undo depth intact.
"""

HISTORY_FILENAME = "history.json"
"""File the stack is persisted to, inside the project directory."""

DEFAULT_LIMIT = 200
"""How many undo steps are kept before the oldest is dropped."""


def snapshot_corrections(project: Project) -> Snapshot:
    """Return a deep, JSON-safe copy of every point's anchors and corrections."""
    return {
        point.id: {
            "queries": [q.to_dict() for q in point.queries],
            "corrections": [c.to_dict() for c in point.corrections],
        }
        for point in project.points
    }


def _entry(state: Any) -> tuple[list[Any] | None, list[Any]]:
    """Split one point's snapshot entry into ``(queries, corrections)``.

    ``queries`` is ``None`` for a stack written before anchors were versioned:
    "this snapshot says nothing about anchors", which is not the same as "this
    snapshot says the point had none".
    """
    if isinstance(state, dict):
        queries = state.get("queries")
        return (list(queries) if isinstance(queries, list) else None), list(
            state.get("corrections", [])
        )
    return None, list(state or [])


def restore_corrections(project: Project, snapshot: Snapshot) -> None:
    """Set every point's anchors and corrections from ``snapshot``, in place.

    Points the snapshot does not mention have their corrections cleared: a point
    that exists now but did not when the snapshot was taken must come back with
    no edits, or undo would leave edits the user never made. Their **anchors**
    are left alone, because undoing a point into an anchorless one would leave
    it untrackable, and creating a point is not an undoable operation in the
    first place.

    Raises:
        ProjectError: If the snapshot holds a correction or anchor this build
            cannot read.
    """
    for point in project.points:
        queries, corrections = _entry(snapshot.get(point.id))
        try:
            point.corrections = [correction_from_dict(dict(entry)) for entry in corrections]
            if queries is not None:
                point.queries = [Query.from_dict(dict(entry)) for entry in queries]
        except Exception as exc:  # noqa: BLE001 - re-raised as a project-level error
            raise ProjectError(f"cannot restore edits for point {point.id!r}: {exc}") from exc


class EditHistory:
    """A persisted undo/redo stack of :data:`Snapshot` states for one project.

    Args:
        path: The ``history.json`` file backing this stack.
        limit: Maximum undo depth; the oldest entry is dropped past it.
    """

    def __init__(self, path: Path, *, limit: int = DEFAULT_LIMIT) -> None:
        """Load the stack from ``path`` if it exists, otherwise start empty."""
        self.path = path
        self.limit = int(limit)
        self._undo: list[tuple[str, Snapshot]] = []
        self._redo: list[tuple[str, Snapshot]] = []
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        """Read the stack from disk, tolerating a missing or corrupt file."""
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._undo = [(str(e["label"]), dict(e["state"])) for e in data.get("undo", [])]
            self._redo = [(str(e["label"]), dict(e["state"])) for e in data.get("redo", [])]
        except (OSError, ValueError, KeyError, TypeError):  # pragma: no cover - corrupt history
            # A broken history file must never stop a user opening their project;
            # the worst case is losing undo depth, and the project itself is intact.
            self._undo, self._redo = [], []

    def save(self) -> None:
        """Write the stack to disk. Call after the project write succeeds."""
        payload = {
            "version": 1,
            "undo": [{"label": label, "state": state} for label, state in self._undo],
            "redo": [{"label": label, "state": state} for label, state in self._redo],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    # -- stack -------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        """Whether there is a state to go back to."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether an undone state can be re-applied."""
        return bool(self._redo)

    @property
    def undo_label(self) -> str | None:
        """Human label of the edit :meth:`undo` would reverse."""
        return self._undo[-1][0] if self._undo else None

    @property
    def redo_label(self) -> str | None:
        """Human label of the edit :meth:`redo` would re-apply."""
        return self._redo[-1][0] if self._redo else None

    def record(self, project: Project, label: str) -> None:
        """Push ``project``'s *current* anchors and corrections as an undo point.

        Call this immediately before mutating, from inside the same
        read-modify-write the mutation happens in, so the snapshot is exactly
        the state the edit is about to replace.

        Args:
            project: The project as loaded, before the edit.
            label: Short description of the edit being made ("nudge p1 @ 42").
        """
        self._undo.append((label, snapshot_corrections(project)))
        if len(self._undo) > self.limit:
            del self._undo[0 : len(self._undo) - self.limit]
        self._redo.clear()

    def undo(self, project: Project) -> str | None:
        """Restore the previous anchor/correction state onto ``project``.

        Args:
            project: The project to mutate in place.

        Returns:
            The label of the edit that was reversed, or ``None`` if there was
            nothing to undo (in which case ``project`` is untouched).
        """
        if not self._undo:
            return None
        label, state = self._undo.pop()
        self._redo.append((label, snapshot_corrections(project)))
        restore_corrections(project, state)
        return label

    def redo(self, project: Project) -> str | None:
        """Re-apply the most recently undone state onto ``project``.

        Args:
            project: The project to mutate in place.

        Returns:
            The label of the edit that was re-applied, or ``None`` if there was
            nothing to redo (in which case ``project`` is untouched).
        """
        if not self._redo:
            return None
        label, state = self._redo.pop()
        self._undo.append((label, snapshot_corrections(project)))
        restore_corrections(project, state)
        return label

    def clear(self) -> None:
        """Drop both stacks (used when a project's points are rebuilt wholesale)."""
        self._undo.clear()
        self._redo.clear()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe status the UI's undo/redo buttons render from."""
        return {
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "undo_label": self.undo_label,
            "redo_label": self.redo_label,
            "undo_depth": len(self._undo),
            "redo_depth": len(self._redo),
        }
