"""Shared fixtures and helpers for the Phase 0, 1, 2 and 3 contract tests."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from stable.core import paths
from stable.core.geometry import SimilarityTransform
from stable.core.project import MIGRATIONS
from stable.core.types import Query, TrackResult
from stable.core.video import SyntheticVideoSource
from stable.trackers.base import TrackerPlugin
from stable.trackers.deep import BackendPrediction
from stable.trackers.merge import RunSpec, TrackRun


@pytest.fixture(autouse=True)
def isolated_legacy_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Hide the real ``~/.stable`` from every test, no exceptions.

    :func:`stable.core.migrate_storage.migrate_legacy_storage` *moves* whatever
    it finds under the legacy root — gigabytes of somebody's actual projects —
    and it runs from ``run_serve``, which the CLI tests call for real with a
    faked uvicorn. One test exercising the startup path must never be able to
    relocate the machine's data, so the legacy root is pointed at a path that
    does not exist and the migration becomes a no-op everywhere. Tests that
    want to exercise it (``tests/test_paths.py``) override this with a
    populated temp directory of their own.
    """
    legacy = tmp_path_factory.getbasetemp() / "no-legacy-home" / ".stable"
    monkeypatch.setattr(paths, "LEGACY_HOME", legacy)
    return legacy


class PlainVideoSource:
    """A minimal VideoSource with no ground truth (exercises the drift path)."""

    def __init__(self, frame_count: int = 20, width: int = 32, height: int = 24) -> None:
        self._frame_count = frame_count
        self._width = width
        self._height = height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return 25.0

    def get_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self._frame_count:
            raise IndexError(index)
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def iter_frames(self, start: int = 0, stop: int | None = None):
        stop = self._frame_count if stop is None else stop
        for i in range(start, stop):
            yield i, self.get_frame(i)


class FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint, loadable or exploding."""

    def __init__(self, name: str, target: object, *, error: Exception | None = None) -> None:
        self.name = name
        self.group = "stable.trackers"
        self._target = target
        self._error = error

    def load(self) -> object:
        if self._error is not None:
            raise self._error
        return self._target


@pytest.fixture
def video() -> SyntheticVideoSource:
    """A 24-frame synthetic clip with a known marker path."""
    return SyntheticVideoSource(frame_count=24, width=64, height=48)


@pytest.fixture
def plain_video() -> PlainVideoSource:
    """A 20-frame source with no ground-truth positions."""
    return PlainVideoSource(frame_count=20)


@pytest.fixture
def clean_migrations() -> Iterator[None]:
    """Restore the global migration table after a test registers one."""
    saved = dict(MIGRATIONS)
    try:
        yield
    finally:
        MIGRATIONS.clear()
        MIGRATIONS.update(saved)


def execute_run(tracker: TrackerPlugin, video_source: object, spec: RunSpec) -> TrackRun:
    """Execute one planned run with a plugin and wrap it as a TrackRun."""
    result = tracker.track(video_source, [spec.anchor], spec.frame_range, spec.direction)
    return TrackRun.from_spec(spec, result.point(0))


def anchor_at(video_source: SyntheticVideoSource, frame: int) -> Query:
    """Return a Query sitting exactly on the synthetic clip's marker."""
    x, y = video_source.true_position(frame)
    return Query(frame=frame, x=x, y=y)


# -- Phase 1 helpers: classic-CV tracker and real-video-file fixtures --------


class OccludedVideoSource(SyntheticVideoSource):
    """A :class:`SyntheticVideoSource` whose marker vanishes over a frame range.

    Used to prove that a real tracker's forward-backward (or NCC) confidence
    check reacts to genuine occlusion, rather than just decaying with distance
    from the anchor like :class:`~stable.trackers.dummy.DummyTracker` does.
    """

    def __init__(self, occlude_range: tuple[int, int], *args: object, **kwargs: object) -> None:
        """Create a synthetic clip whose marker is hidden during ``occlude_range``."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.occlude_range = occlude_range

    def get_frame(self, index: int) -> np.ndarray:
        """Return the frame, with the marker erased if ``index`` is occluded."""
        frame = super().get_frame(index)
        lo, hi = self.occlude_range
        if lo <= index < hi:
            ys, xs = np.mgrid[0 : self.height, 0 : self.width]
            base = ((xs + ys) % 256).astype(np.uint8)
            frame = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=0)], axis=-1)
        return frame


class GrowingMarkerSource(SyntheticVideoSource):
    """A stationary marker whose size grows over time.

    Its appearance drifts frame to frame without ever moving, which isolates
    whether a template tracker's adaptive update actually helps: a frozen
    template's NCC score against the (now much bigger) patch decays, while an
    adaptively updated template keeps re-matching the current appearance.
    """

    def true_position(self, frame: int) -> tuple[float, float]:
        """Return the fixed centre position (the marker never moves)."""
        return (self.width / 2.0, self.height / 2.0)

    def get_frame(self, index: int) -> np.ndarray:
        """Return the frame with a square marker whose half-size grows with ``index``."""
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        base = ((xs + ys) % 256).astype(np.uint8)
        frame = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=0)], axis=-1)
        cx, cy = int(self.width // 2), int(self.height // 2)
        half = 2 + index // 3
        x0, x1 = max(cx - half, 0), min(cx + half + 1, self.width)
        y0, y1 = max(cy - half, 0), min(cy + half + 1, self.height)
        frame[y0:y1, x0:x1] = 255
        return frame


@pytest.fixture
def occluded_video() -> OccludedVideoSource:
    """A 30-frame synthetic clip whose marker is hidden across frames [12, 18)."""
    return OccludedVideoSource(
        (12, 18), frame_count=30, width=96, height=72, radius=12.0, period=30.0
    )


@pytest.fixture
def growing_marker_video() -> GrowingMarkerSource:
    """A 40-frame synthetic clip with a stationary, steadily growing marker."""
    return GrowingMarkerSource(frame_count=40, width=96, height=72)


def encode_synthetic_clip(
    source: SyntheticVideoSource,
    path: str,
    *,
    with_audio: bool = True,
    crf: int = 15,
) -> str:
    """Encode a :class:`SyntheticVideoSource` to a real H.264 file via raw PyAV calls.

    Deliberately bypasses :mod:`stable.video.encode` (the code under test
    elsewhere) so fixtures that need "a real video file on disk" do not depend
    on the export utility they are also used to validate.

    Args:
        source: The synthetic clip to encode.
        path: Destination file path (``.mp4``).
        with_audio: Whether to add a silent AAC audio track, for tests that
            need to prove audio passthrough.
        crf: Constant rate factor for the encoder.

    Returns:
        ``path``, unchanged, for convenient chaining.
    """
    import av

    output = av.open(path, mode="w")
    try:
        # Every stream must be declared before the first packet is muxed (the
        # container header is written on that first mux call) — so both
        # streams are added up front, even though audio is only encoded after.
        video_stream = output.add_stream("libx264", rate=int(round(source.fps)) or 24)
        video_stream.width = source.width
        video_stream.height = source.height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": str(crf)}
        audio_stream = output.add_stream("aac", rate=48000) if with_audio else None

        for index, frame in source.iter_frames():
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = index
            for packet in video_stream.encode(video_frame):
                output.mux(packet)
        for packet in video_stream.encode(None):
            output.mux(packet)

        if audio_stream is not None:
            seconds = max(1, int(source.frame_count / (source.fps or 24.0)) + 1)
            samples = np.zeros((1, 48000 * seconds), dtype=np.int16)
            audio_frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            audio_frame.sample_rate = 48000
            audio_frame.pts = 0
            for packet in audio_stream.encode(audio_frame):
                output.mux(packet)
            for packet in audio_stream.encode(None):
                output.mux(packet)
    finally:
        output.close()
    return path


@pytest.fixture(scope="session")
def tiny_clip_factory(tmp_path_factory: pytest.TempPathFactory):
    """Return a factory that encodes a small deterministic real video file.

    Session-scoped so multiple tests can each mint their own clip cheaply; the
    factory itself does no caching, every call encodes a fresh file.
    """
    counter = {"n": 0}

    def _make(
        *,
        frame_count: int = 30,
        width: int = 64,
        height: int = 48,
        fps: float = 24.0,
        radius: float = 10.0,
        period: float = 30.0,
        with_audio: bool = True,
    ) -> tuple[str, SyntheticVideoSource]:
        source = SyntheticVideoSource(
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            radius=radius,
            period=period,
        )
        counter["n"] += 1
        directory = tmp_path_factory.mktemp("clip")
        path = str(directory / f"clip_{counter['n']}.mp4")
        encode_synthetic_clip(source, path, with_audio=with_audio)
        return path, source

    return _make


@pytest.fixture
def tiny_clip(tiny_clip_factory) -> tuple[str, SyntheticVideoSource]:
    """A small (64x48, 30 frame) real H.264+AAC clip with a known marker path.

    Returns:
        ``(path, source)`` — the file path, and the
        :class:`~stable.core.video.SyntheticVideoSource` used to generate it
        (so tests can compare decoded tracking results against
        ``source.true_position(frame)`` ground truth).
    """
    return tiny_clip_factory()


# -- Phase 2 helpers: fake model backends and a drifting tracker --------------


def find_marker(frame: np.ndarray) -> tuple[float, float] | None:
    """Return the centroid of the synthetic clip's bright marker, if it is there.

    The gradient background can reach 255 in a single channel, but never in all
    three at once (the channels are rolled by 7 and 13), so requiring every
    channel to be saturated isolates the white marker square exactly.

    Args:
        frame: ``[H, W, 3]`` uint8 RGB.

    Returns:
        ``(x, y)`` of the marker centroid, or ``None`` when it is not visible.
    """
    mask = frame.min(axis=2) >= 250
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean()))


class MarkerBackend:
    """A :class:`~stable.trackers.deep.TrackerBackend` that "detects" the marker.

    Stands in for CoTracker3 / TAPIR on machines with no GPU, no torch and no
    checkpoints, while still exercising everything the plugin actually owns:
    window loading, resolution scaling, re-anchoring across chunk boundaries,
    stitching, confidence normalisation and the timeline assembly. It reads real
    pixels — so its answers are only as good as the frames the plugin handed it,
    which is exactly what makes it a useful fake.

    Args:
        oom_on: Call indices (1-based) that should raise a CUDA-style
            out-of-memory error while the backend is not on the CPU.
        fail_on: Call indices that should raise a plain ``RuntimeError``.
        lost_visibility: Visibility reported for frames with no marker.
    """

    def __init__(
        self,
        *,
        oom_on: tuple[int, ...] = (),
        fail_on: tuple[int, ...] = (),
        lost_visibility: float = 0.02,
    ) -> None:
        """Create a fake backend on the CPU."""
        self.device = "cpu"
        self.oom_on = set(oom_on)
        self.fail_on = set(fail_on)
        self.lost_visibility = float(lost_visibility)
        self.calls = 0
        self.windows: list[tuple[int, int, int]] = []
        self.queries: list[np.ndarray] = []
        self.released = False

    def infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Locate the marker in every frame and offset it onto each query."""
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("simulated backend explosion")
        if self.calls in self.oom_on and self.device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.windows.append((frames.shape[0], frames.shape[1], frames.shape[2]))
        self.queries.append(np.array(queries, copy=True))

        centres = [find_marker(frame) for frame in frames]
        length, n_points = frames.shape[0], queries.shape[0]
        points = np.zeros((length, n_points, 2), dtype=np.float32)
        visibility = np.zeros((length, n_points), dtype=np.float32)
        confidence = np.ones((length, n_points), dtype=np.float32)
        base = centres[0]
        for point in range(n_points):
            query = np.asarray(queries[point][1:3], dtype=np.float64)
            offset = query - np.asarray(base) if base is not None else np.zeros(2)
            last = query
            for index, centre in enumerate(centres):
                if centre is not None:
                    last = np.asarray(centre) + offset
                    visibility[index, point] = 1.0
                else:
                    visibility[index, point] = self.lost_visibility
                    confidence[index, point] = self.lost_visibility
                points[index, point] = last
        return BackendPrediction(points=points, visibility=visibility, confidence=confidence)

    def to_device(self, device: str) -> None:
        """Record a device move (there is nothing to move)."""
        self.device = device

    def release(self) -> None:
        """Record that the plugin released this backend."""
        self.released = True


class EchoBackend:
    """A backend that reports every query as motionless, at full confidence.

    Useful for asserting coordinate bookkeeping (resolution scaling, run
    direction, re-anchoring) without any detection logic in the way.
    """

    def __init__(self) -> None:
        """Create a fake backend on the CPU."""
        self.device = "cpu"
        self.frame_sizes: list[tuple[int, int]] = []
        self.queries: list[np.ndarray] = []

    def infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Hold every query point exactly where it was handed in."""
        self.frame_sizes.append((frames.shape[1], frames.shape[2]))
        self.queries.append(np.array(queries, copy=True))
        length = frames.shape[0]
        points = np.repeat(queries[None, :, 1:3].astype(np.float32), length, axis=0)
        return BackendPrediction(
            points=points,
            visibility=np.ones((length, queries.shape[0]), dtype=np.float32),
            confidence=np.ones((length, queries.shape[0]), dtype=np.float32),
        )

    def to_device(self, device: str) -> None:
        """Record a device move."""
        self.device = device

    def release(self) -> None:
        """Nothing to release."""


class DriftingTracker(TrackerPlugin):
    """A tracker that follows ground truth but slides along its run direction.

    Real trackers accumulate error away from their anchor, in a direction set by
    the way they are travelling — which is precisely what multi-anchor merging
    exists to cancel. This models that directly: the error at frame ``f`` is
    ``drift`` times the number of *drifting* frames between the anchor and
    ``f``, signed by the run direction.

    Args:
        drift: Per-drifting-frame ``(dx, dy)`` error.
        drift_range: Half-open frame range in which drift accumulates; ``None``
            means the whole clip. Restricting it puts the trouble in one span so
            a test can show the merge staying clean elsewhere.
        confidence: Confidence reported for every tracked frame.
    """

    name = "drifting"
    display_name = "Drifting (testing)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        drift: tuple[float, float] = (0.6, 0.0),
        drift_range: tuple[int, int] | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Create a drifting tracker; see the class docstring."""
        super().__init__()
        self.drift = (float(drift[0]), float(drift[1]))
        self.drift_range = drift_range
        self.confidence = float(confidence)

    def _drifting_frames(self, anchor: int, frame: int) -> int:
        """Return the signed count of drifting frames between anchor and frame."""
        if frame == anchor:
            return 0
        step = 1 if frame > anchor else -1
        walked = range(anchor + 1, frame + 1) if step > 0 else range(frame, anchor)
        if self.drift_range is None:
            count = len(walked)
        else:
            low, high = self.drift_range
            count = sum(1 for k in walked if low <= k < high)
        return step * count

    def track(self, video, queries, frame_range, direction="bidir", progress=None):
        """Follow ``video.true_position`` with an accumulating directional error."""
        n_frames = int(video.frame_count)
        start, end = int(frame_range[0]), int(frame_range[1])
        points = np.full((len(queries), n_frames, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(queries), n_frames), dtype=np.float32)
        visible = np.zeros((len(queries), n_frames), dtype=bool)
        for index, query in enumerate(queries):
            if direction == "fwd":
                lo, hi = max(start, query.frame), end
            elif direction == "bwd":
                lo, hi = start, min(end, query.frame + 1)
            else:
                lo, hi = start, end
            anchor_truth = np.asarray(video.true_position(query.frame), dtype=np.float64)
            offset = np.asarray(query.position, dtype=np.float64) - anchor_truth
            for frame in range(lo, hi):
                slide = self._drifting_frames(query.frame, frame)
                truth = np.asarray(video.true_position(frame), dtype=np.float64)
                points[index, frame] = truth + offset + slide * np.asarray(self.drift)
                confidence[index, frame] = self.confidence
                visible[index, frame] = True
        return TrackResult(points=points, confidence=confidence, visible=visible)


# -- Phase 3 helpers: clips warped by a known X/Y/R/S motion ------------------


def handheld_transforms(
    n_frames: int,
    *,
    pivot: tuple[float, float],
    fps: float = 25.0,
    amplitude: float = 6.0,
    rotation: float = 0.02,
    scale: float = 0.01,
    drift_hz: float = 0.35,
    jitter_hz: float = 6.0,
    jitter: float = 0.3,
    seed: int = 5,
) -> list[SimilarityTransform]:
    """Return a plausible handheld camera path as one similarity per frame.

    Handheld footage is two things superimposed, and a stabiliser has to tell
    them apart: a slow **drift** (the operator re-framing, which the shot wants
    to keep) and a fast **judder** (breathing, footsteps, which it wants gone).
    Both are generated here at explicit frequencies, so a test can set a filter
    cutoff between them and assert that exactly the right half was removed.

    Args:
        n_frames: Clip length.
        pivot: Point the rotation and scale happen about, usually the frame centre.
        fps: Frame rate the frequencies are expressed against.
        amplitude: Peak drift translation in pixels.
        rotation: Peak drift rotation in radians.
        scale: Peak fractional drift in scale.
        drift_hz: Frequency of the slow, deliberate movement.
        jitter_hz: Frequency of the judder.
        jitter: Judder size as a fraction of the drift amplitudes.
        seed: RNG seed for the broadband noise mixed into the judder.

    Returns:
        One :class:`~stable.core.geometry.SimilarityTransform` per frame, with
        the first frame exactly the identity so it can serve as a reference.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n_frames, dtype=np.float64) / float(fps)
    noise = rng.normal(scale=0.25 * jitter * amplitude, size=(n_frames, 2))

    def wave(frequency: float, phase: float = 0.0) -> np.ndarray:
        return np.sin(2.0 * math.pi * frequency * time + phase)

    tx = amplitude * wave(drift_hz) + jitter * amplitude * wave(jitter_hz, 0.9) + noise[:, 0]
    ty = (
        0.7 * amplitude * wave(0.8 * drift_hz, 1.4)
        + jitter * amplitude * wave(1.3 * jitter_hz)
        + noise[:, 1]
    )
    angle = rotation * wave(0.6 * drift_hz) + jitter * rotation * wave(0.9 * jitter_hz, 0.3)
    zoom = 1.0 + scale * wave(0.5 * drift_hz, 0.6) + jitter * scale * wave(1.1 * jitter_hz)

    transforms = [
        SimilarityTransform.about_pivot(
            pivot,
            rotation=float(angle[frame]),
            scale=float(zoom[frame]),
            translation=(float(tx[frame]), float(ty[frame])),
        )
        for frame in range(n_frames)
    ]
    # Make frame 0 exactly the identity so "reference frame 0" is unambiguous.
    inverse = transforms[0].inverse()
    return [t.compose(inverse) for t in transforms]


class WarpedClipSource:
    """A textured reference frame warped by a *known* similarity per frame.

    This is the ground truth the Phase 3 acceptance test is written against: the
    motion in the clip is not estimated from anything, it is imposed, so a solve
    can be compared against the exact numbers that produced the pixels.

    Args:
        transforms: One similarity per frame, mapping reference-frame
            coordinates to that frame's coordinates.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Nominal frame rate.
        seed: RNG seed for the texture.
        n_blobs: How many bright squares the texture holds; their centres are
            the trackable features returned by :meth:`feature_points`.
        flat: Render every frame as pure white instead of the texture — for
            coverage tests, where any dark pixel in the output is a border gap
            and nothing else.
    """

    def __init__(
        self,
        transforms: Sequence[SimilarityTransform],
        *,
        width: int = 320,
        height: int = 240,
        fps: float = 25.0,
        seed: int = 7,
        n_blobs: int = 40,
        flat: bool = False,
    ) -> None:
        """Build the reference texture; see the class docstring."""
        self._transforms = list(transforms)
        self._width, self._height, self._fps = int(width), int(height), float(fps)
        self.flat = bool(flat)
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 50, (self._height, self._width, 3), dtype=np.uint8)
        margin = 24
        centres: list[tuple[float, float]] = []
        for _ in range(n_blobs):
            cx = int(rng.integers(margin, self._width - margin))
            cy = int(rng.integers(margin, self._height - margin))
            half = int(rng.integers(2, 5))
            image[cy - half : cy + half + 1, cx - half : cx + half + 1] = rng.integers(
                170, 256, 3, dtype=np.uint8
            )
            centres.append((float(cx), float(cy)))
        self.reference_image = image
        self._blobs = centres
        self._cache: dict[int, np.ndarray] = {}

    # -- VideoSource protocol ---------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total number of frames ``T`` in the clip."""
        return len(self._transforms)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def fps(self) -> float:
        """Frame rate in frames per second."""
        return self._fps

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame ``index``: the reference texture under ``transform_at(index)``."""
        if not 0 <= index < self.frame_count:
            raise IndexError(f"frame {index} out of range for T={self.frame_count}")
        if self.flat:
            return np.full((self._height, self._width, 3), 255, dtype=np.uint8)
        cached = self._cache.get(index)
        if cached is None:
            import cv2

            cached = cv2.warpAffine(
                self.reference_image,
                self._transforms[index].to_matrix(),
                (self._width, self._height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            self._cache[index] = cached
        return cached.copy()

    def iter_frames(self, start: int = 0, stop: int | None = None):
        """Yield ``(index, frame)`` pairs over ``[start, stop)`` in order."""
        stop = self.frame_count if stop is None else min(stop, self.frame_count)
        for index in range(max(start, 0), stop):
            yield index, self.get_frame(index)

    # -- ground truth ------------------------------------------------------

    def transform_at(self, frame: int) -> SimilarityTransform:
        """Return the known similarity that produced ``frame`` from the reference."""
        return self._transforms[frame]

    def relative_transform(self, frame: int, reference: int = 0) -> SimilarityTransform:
        """Return the true motion of ``frame`` measured against ``reference``.

        This is what a solve with ``reference_frame=reference`` must recover.
        """
        return self._transforms[frame].compose(self._transforms[reference].inverse())

    def feature_points(self, count: int | None = None) -> list[tuple[float, float]]:
        """Return trackable feature positions in the reference frame.

        Args:
            count: How many to return; ``None`` returns all of them.

        Returns:
            The centres of the texture's bright squares, which is where a corner
            tracker will lock on.
        """
        return list(self._blobs if count is None else self._blobs[:count])

    def true_tracks(
        self, points: Sequence[tuple[float, float]], *, confidence: float = 1.0
    ) -> TrackResult:
        """Return the exact tracks of ``points`` through the clip.

        No tracker involved: the positions are computed from the same transforms
        that generated the pixels, which isolates the solver from any tracking
        error when a test wants to measure the solver alone.
        """
        positions = np.asarray(points, dtype=np.float64)
        n_points, n_frames = positions.shape[0], self.frame_count
        out = np.empty((n_points, n_frames, 2), dtype=np.float64)
        for frame in range(n_frames):
            out[:, frame] = self._transforms[frame].apply(positions)
        return TrackResult(
            points=out,
            confidence=np.full((n_points, n_frames), confidence, dtype=np.float32),
            visible=np.ones((n_points, n_frames), dtype=bool),
        )


@pytest.fixture
def warped_clip() -> WarpedClipSource:
    """A 40-frame 320x240 textured clip moved by a known handheld camera path."""
    return WarpedClipSource(handheld_transforms(40, pivot=(160.0, 120.0)))


@pytest.fixture
def flat_clip() -> WarpedClipSource:
    """A 30-frame all-white clip: any dark output pixel is a border gap."""
    return WarpedClipSource(
        handheld_transforms(30, pivot=(160.0, 120.0), amplitude=10.0), flat=True
    )
