"""Windowed inference planning and overlap stitching for long clips.

Transformer point trackers see a fixed-length window of frames at once. A 30 s
1080p clip is hundreds of frames and will not fit on a consumer GPU, so a run is
cut into overlapping chunks: each chunk is tracked separately, re-anchored on the
position the previous chunk predicted for its first frame, and the results are
crossfaded across the overlap so no seam appears at a chunk boundary.

Everything here is pure integer/array logic in **run-local index space** — index
``0`` is the run's anchor frame and index ``i`` is ``i`` frames further along the
run's direction. Mapping local indices back to absolute frame numbers is the
plugin's job (a backward run maps ``i`` to ``anchor - i``). That keeps this
module tracker-agnostic, dependency-free and exhaustively testable without a GPU,
and means CoTracker3 and TAPIR share one implementation of the tricky part.

Why crossfade rather than cut
-----------------------------
Chunk ``k+1`` starts from chunk ``k``'s prediction, so the two agree exactly at
the seam and drift apart slowly. Hard-cutting at the seam is therefore
*continuous* but not *smooth* — the velocity jumps, which a stabilization solve
turns into a visible tick. :class:`ChunkStitcher` instead ramps each chunk's
weight in and out across its overlaps and blends by ``ramp x confidence``, so a
chunk that lost the point hands over to the one that did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ContractError

__all__ = ["Chunk", "plan_chunks", "chunk_weights", "ChunkStitcher"]

_EPS = 1e-6


@dataclass(frozen=True)
class Chunk:
    """One inference window, in run-local index space.

    Args:
        start: First local index in the window (inclusive).
        end: One past the last local index (exclusive).
        blend_in: How many leading frames overlap the previous chunk; ``0`` for
            the first chunk.
        blend_out: How many trailing frames overlap the next chunk; ``0`` for
            the last chunk.

    Raises:
        ContractError: If the window is empty or the blend regions do not fit
            inside it.
    """

    start: int
    end: int
    blend_in: int = 0
    blend_out: int = 0

    def __post_init__(self) -> None:
        """Validate the window (dataclass hook)."""
        if self.end <= self.start:
            raise ContractError(f"Chunk must be non-empty, got [{self.start}, {self.end})")
        if self.start < 0:
            raise ContractError(f"Chunk.start must be >= 0, got {self.start}")
        if self.blend_in < 0 or self.blend_out < 0:
            raise ContractError("Chunk blend lengths must be >= 0")
        if self.blend_in > self.n_frames or self.blend_out > self.n_frames:
            raise ContractError(
                f"blend_in={self.blend_in}/blend_out={self.blend_out} exceed the "
                f"{self.n_frames}-frame window"
            )

    @property
    def n_frames(self) -> int:
        """Number of frames in this window."""
        return self.end - self.start

    def indices(self) -> np.ndarray:
        """Return the local indices this window covers, as an ``int`` array."""
        return np.arange(self.start, self.end, dtype=np.int64)


def plan_chunks(length: int, chunk_size: int, overlap: int) -> list[Chunk]:
    """Cut a run of ``length`` frames into overlapping inference windows.

    Args:
        length: Number of frames the run covers, including its anchor frame at
            local index ``0``. ``0`` yields no chunks.
        chunk_size: Maximum frames per window. Must be at least 2 — a
            single-frame window could never predict anything.
        overlap: Frames each window shares with the previous one. Clamped to
            ``chunk_size - 1`` so progress is always made; ``0`` means hard cuts.

    Returns:
        Windows in run order, covering ``[0, length)`` exactly once (modulo the
        overlaps), each carrying the ``blend_in``/``blend_out`` lengths
        :func:`chunk_weights` needs.

    Raises:
        ContractError: If ``length`` is negative, ``chunk_size`` is below 2, or
            ``overlap`` is negative.
    """
    length = int(length)
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    if length < 0:
        raise ContractError(f"length must be >= 0, got {length}")
    if chunk_size < 2:
        raise ContractError(f"chunk_size must be >= 2, got {chunk_size}")
    if overlap < 0:
        raise ContractError(f"overlap must be >= 0, got {overlap}")
    if length == 0:
        return []
    overlap = min(overlap, chunk_size - 1)

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + chunk_size, length)
        spans.append((start, end))
        if end >= length:
            break
        start = end - overlap

    chunks: list[Chunk] = []
    for index, (span_start, span_end) in enumerate(spans):
        blend_in = 0 if index == 0 else spans[index - 1][1] - span_start
        blend_out = 0 if index == len(spans) - 1 else span_end - spans[index + 1][0]
        # With a large overlap the two blend regions can meet in the middle of a
        # short window; chunk_weights takes the minimum of the two ramps, so that
        # degrades to a tent rather than needing a special case here.
        chunks.append(Chunk(start=span_start, end=span_end, blend_in=blend_in, blend_out=blend_out))
    return chunks


def chunk_weights(chunk: Chunk) -> np.ndarray:
    """Return the ``[n_frames]`` crossfade ramp for one window.

    The ramp rises linearly from ``1 / (blend_in + 1)`` to ``1`` across the
    leading overlap, stays at ``1`` in the exclusive middle, and mirrors on the
    trailing overlap. It is strictly positive everywhere, so a window always has
    *some* say over every frame it covers, and complementary ramps of two
    adjacent windows sum smoothly across their shared region.

    Args:
        chunk: The window to weight.

    Returns:
        ``float64`` weights in ``(0, 1]``.
    """
    n = chunk.n_frames
    positions = np.arange(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    if chunk.blend_in > 0:
        rise = (positions + 1.0) / (chunk.blend_in + 1.0)
        weights = np.minimum(weights, np.where(positions < chunk.blend_in, rise, 1.0))
    if chunk.blend_out > 0:
        from_end = (n - 1) - positions
        fall = (from_end + 1.0) / (chunk.blend_out + 1.0)
        weights = np.minimum(weights, np.where(from_end < chunk.blend_out, fall, 1.0))
    return weights


class ChunkStitcher:
    """Accumulates per-chunk predictions into one seamless run-length track.

    Chunks may be added in any order; the result depends only on their
    ``blend_in``/``blend_out`` metadata, never on insertion order.

    Args:
        length: Number of frames in the run (run-local length).
        n_points: Number of points tracked in parallel.

    Raises:
        ContractError: If either dimension is negative.
    """

    def __init__(self, length: int, n_points: int) -> None:
        """Allocate the accumulators for a run of ``length`` frames."""
        length, n_points = int(length), int(n_points)
        if length < 0 or n_points < 0:
            raise ContractError("ChunkStitcher needs non-negative dimensions")
        self.length = length
        self.n_points = n_points
        self._weight = np.zeros((length, n_points), dtype=np.float64)
        self._position = np.zeros((length, n_points, 2), dtype=np.float64)
        self._confidence = np.zeros((length, n_points), dtype=np.float64)
        self._visible = np.zeros((length, n_points), dtype=np.float64)

    def add(
        self,
        chunk: Chunk,
        points: np.ndarray,
        confidence: np.ndarray,
        visible: np.ndarray,
    ) -> None:
        """Fold one window's prediction into the accumulators.

        Args:
            chunk: The window these arrays cover.
            points: ``[chunk.n_frames, n_points, 2]`` positions. Non-finite
                samples are ignored rather than poisoning the average.
            confidence: ``[chunk.n_frames, n_points]`` in ``[0, 1]``.
            visible: ``[chunk.n_frames, n_points]`` boolean-ish visibility.

        Raises:
            ContractError: If an array's shape does not match the window, or the
                window runs past the end of the run.
        """
        if chunk.end > self.length:
            raise ContractError(f"chunk [{chunk.start}, {chunk.end}) exceeds length {self.length}")
        expected = (chunk.n_frames, self.n_points)
        points = np.asarray(points, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        visible = np.asarray(visible)
        if points.shape != (*expected, 2):
            raise ContractError(f"points must have shape {(*expected, 2)}, got {points.shape}")
        if confidence.shape != expected:
            raise ContractError(f"confidence must have shape {expected}, got {confidence.shape}")
        if visible.shape != expected:
            raise ContractError(f"visible must have shape {expected}, got {visible.shape}")

        ramp = chunk_weights(chunk)[:, None]
        finite = np.isfinite(points).all(axis=2)
        scores = np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)
        weight = np.where(finite, ramp * (scores + _EPS), 0.0)

        span = slice(chunk.start, chunk.end)
        self._weight[span] += weight
        self._position[span] += np.where(finite[..., None], points, 0.0) * weight[..., None]
        self._confidence[span] += scores * weight
        self._visible[span] += visible.astype(np.float64) * weight

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the stitched ``(points, confidence, visible)`` arrays.

        Returns:
            ``points`` ``[length, n_points, 2]`` float32 (``NaN`` where no chunk
            contributed), ``confidence`` ``[length, n_points]`` float32 in
            ``[0, 1]``, and ``visible`` ``[length, n_points]`` bool — the
            weighted visibility vote, True at or above one half.
        """
        covered = self._weight > 0.0
        safe = np.where(covered, self._weight, 1.0)
        points = np.where(covered[..., None], self._position / safe[..., None], np.nan).astype(
            np.float32
        )
        confidence = np.where(covered, self._confidence / safe, 0.0).astype(np.float32)
        visible = np.where(covered, self._visible / safe, 0.0) >= 0.5
        return points, np.clip(confidence, 0.0, 1.0), visible
