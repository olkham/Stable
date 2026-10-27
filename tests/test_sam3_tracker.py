"""SAM 3's own logic: the point rides the mask, and occlusion is a gap (todo 27).

What is and is not verified here
--------------------------------
`tests/test_tracker_suite.py` covers everything the *plugin* owns — windowing,
re-anchoring, stitching, CPU fallback, confidence normalisation — with an
injected backend, exactly as it does for CoTracker3 and TAPIR.

This module covers :class:`~stable.trackers.sam3.Sam3Backend`, the part that
talks to ``transformers``, by driving it with a stub that mimics the documented
``Sam3TrackerVideoModel`` / ``Sam3TrackerVideoProcessor`` API: a session, point
prompts, ``propagate_in_video_iterator`` yielding one output per frame with
``object_ids`` / ``pred_masks`` / ``object_score_logits`` / ``frame_idx``. That
pins the arithmetic and the failure handling that are Stable's own.

What is **not** covered anywhere is a real inference run: ``facebook/sam3`` is a
gated repo, and accepting its licence is the operator's decision, not a test's.
The class and method names this backend calls were checked against
``transformers`` 5.14 rather than only against the docs.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.trackers.deep import BackendPrediction
from stable.trackers.sam3 import DEFAULT_CHECKPOINT, Sam3Backend, Sam3Tracker, _centroids


class StubSession:
    """Stands in for ``Sam3TrackerVideoInferenceSession``."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.video_height = frames[0].shape[0]
        self.video_width = frames[0].shape[1]
        self.prompts: list[dict] = []


class StubOutput:
    """One frame of ``Sam3TrackerVideoSegmentationOutput``."""

    def __init__(self, frame_idx: int, masks: np.ndarray, scores: np.ndarray, ids: list[int]):
        self.frame_idx = frame_idx
        self.pred_masks = _Tensorish(masks)
        self.object_score_logits = _Tensorish(scores)
        self.object_ids = ids


class _Tensorish:
    """The smallest thing that behaves like the tensors the backend touches."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._array

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)

    @property
    def ndim(self):
        return self._array.ndim


class StubProcessor:
    """Mimics the documented video-processor surface the backend calls."""

    def __init__(self) -> None:
        self.sessions: list[StubSession] = []

    def init_video_session(self, *, video, inference_device="cpu", **kwargs):
        session = StubSession(list(video))
        self.sessions.append(session)
        return session

    def add_inputs_to_inference_session(
        self, *, inference_session, frame_idx, obj_ids, input_points=None, input_labels=None, **kw
    ):
        inference_session.prompts.append(
            {"frame_idx": frame_idx, "obj_ids": obj_ids, "points": input_points}
        )

    def post_process_masks(self, masks, original_sizes, **kwargs):
        # The real one interpolates masks[i] (4D) to original_sizes[i] per image.
        assert isinstance(masks, list) and len(masks) == len(original_sizes)
        out = []
        for batch, size in zip(masks, original_sizes, strict=True):
            array = np.asarray(batch)
            assert array.ndim == 4, "bilinear mode needs 4D input"
            assert tuple(size) == array.shape[-2:]
            out.append(array)
        return out


class StubModel:
    """A model whose object is a square moving by a known amount per frame."""

    def __init__(self, motion=(3.0, 2.0), occluded: tuple[int, int] = (), size: int = 10) -> None:
        self.motion = motion
        self.occluded = occluded
        self.size = size
        self.moved_to: str | None = None

    def propagate_in_video_iterator(self, *, inference_session, start_frame_idx=0, **kwargs):
        height, width = inference_session.video_height, inference_session.video_width
        n_objects = len(inference_session.prompts)
        for index, _frame in enumerate(inference_session.frames):
            masks = np.zeros((n_objects, 1, height, width), dtype=bool)
            scores = np.zeros((n_objects,), dtype=np.float32)
            hidden = bool(self.occluded) and self.occluded[0] <= index < self.occluded[1]
            for obj in range(n_objects):
                prompt = inference_session.prompts[obj]["points"][0][0][0]
                cx = prompt[0] + self.motion[0] * index
                cy = prompt[1] + self.motion[1] * index
                if hidden:
                    scores[obj] = -8.0  # the presence head collapsing
                    continue
                x0 = int(round(cx - self.size / 2))
                y0 = int(round(cy - self.size / 2))
                masks[obj, 0, y0 : y0 + self.size, x0 : x0 + self.size] = True
                scores[obj] = 6.0
            yield StubOutput(index, masks, scores, list(range(n_objects)))

    def to(self, device: str):
        self.moved_to = device
        return self


def backend(**model_kwargs) -> tuple[Sam3Backend, StubModel]:
    model = StubModel(**model_kwargs)
    return Sam3Backend(model, StubProcessor(), "cpu"), model


FRAMES = np.zeros((6, 120, 160, 3), dtype=np.uint8)
QUERY = np.array([[0.0, 60.0, 40.0]], dtype=np.float32)


# -- the arithmetic ------------------------------------------------------------


def test_the_point_rides_the_mask_rather_than_snapping_to_its_centre():
    """The operator picked a corner, not the middle of the object.

    The stub's mask is a square centred on the prompt and moving 3px/2px per
    frame; the reported track must move with it while *starting* exactly where
    the operator put it.
    """
    sam, _model = backend()
    prediction = sam.infer(FRAMES, QUERY)

    assert prediction.points.shape == (6, 1, 2)
    assert prediction.points[0, 0] == pytest.approx([60.0, 40.0], abs=1e-4)
    for frame in range(6):
        assert prediction.points[frame, 0] == pytest.approx(
            [60.0 + 3.0 * frame, 40.0 + 2.0 * frame], abs=0.75
        )


def test_an_offset_point_keeps_its_offset_from_the_object():
    """A point placed off-centre stays off-centre — the whole reason for the ride."""
    sam, _model = backend()
    off_centre = np.array([[0.0, 64.0, 44.0]], dtype=np.float32)
    prediction = sam.infer(FRAMES, off_centre)
    assert prediction.points[0, 0] == pytest.approx([64.0, 44.0], abs=1e-4)
    assert prediction.points[3, 0] == pytest.approx([64.0 + 9.0, 44.0 + 6.0], abs=0.75)


def test_an_occluded_stretch_is_a_gap_and_the_track_returns_after_it():
    """The todo in one test: hidden means unknown, and reappearance is handled."""
    sam, _model = backend(occluded=(2, 4))
    prediction = sam.infer(FRAMES, QUERY)

    assert not np.isfinite(prediction.points[2:4, 0, 0]).any(), "occluded frames must be gaps"
    assert np.isfinite(prediction.points[4:, 0, 0]).all(), "it must come back"
    # ...and back in the right place: the mask's own memory, not a drifted guess.
    assert prediction.points[5, 0] == pytest.approx([60.0 + 15.0, 40.0 + 10.0], abs=0.75)


def test_the_presence_head_is_what_reports_the_occlusion():
    sam, _model = backend(occluded=(2, 4))
    prediction = sam.infer(FRAMES, QUERY)
    assert prediction.visibility[0, 0] > 0.9
    assert prediction.visibility[2, 0] < 0.1
    assert prediction.visibility[5, 0] > 0.9


def test_every_prompted_object_gets_its_own_track():
    sam, _model = backend()
    queries = np.array([[0.0, 40.0, 30.0], [0.0, 90.0, 70.0]], dtype=np.float32)
    prediction = sam.infer(FRAMES, queries)
    assert prediction.points.shape == (6, 2, 2)
    assert prediction.points[0, 0] == pytest.approx([40.0, 30.0], abs=1e-4)
    assert prediction.points[0, 1] == pytest.approx([90.0, 70.0], abs=1e-4)


def test_each_query_is_prompted_as_its_own_object_on_the_first_frame():
    sam, _model = backend()
    queries = np.array([[0.0, 40.0, 30.0], [0.0, 90.0, 70.0]], dtype=np.float32)
    sam.infer(FRAMES, queries)
    session = sam._processor.sessions[0]  # noqa: SLF001 - asserting the call we make
    assert [p["obj_ids"] for p in session.prompts] == [0, 1]
    assert all(p["frame_idx"] == 0 for p in session.prompts)


# -- the confidence contract ---------------------------------------------------


def test_confidence_normalisation_maps_presence_onto_the_shared_scale():
    plugin = Sam3Tracker(backend=None)
    prediction = BackendPrediction(
        points=np.zeros((3, 1, 2), dtype=np.float32),
        visibility=np.array([[0.99], [0.02], [0.6]], dtype=np.float32),
        confidence=np.ones((3, 1), dtype=np.float32),
    )
    confidence, visible = plugin.normalize_confidence(prediction)
    assert ((confidence >= 0.0) & (confidence <= 1.0)).all()
    assert confidence[0, 0] > confidence[2, 0] > confidence[1, 0]
    assert visible[0, 0] and not visible[1, 0]


# -- helpers and reporting -----------------------------------------------------


def test_centroids_of_an_empty_mask_are_not_a_position():
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:4, 2:6] = True
    centres = _centroids(masks)
    assert centres[0] == pytest.approx([3.5, 2.5])
    assert not np.isfinite(centres[1]).any()


def test_the_plugin_owns_no_model_spec_and_says_where_its_weights_live():
    assert Sam3Tracker.model_spec is None
    assert DEFAULT_CHECKPOINT == "facebook/sam3"
    # `weights_downloaded` must answer without raising even with no hub cache.
    assert isinstance(Sam3Tracker.weights_downloaded(), bool)


def test_it_never_requires_a_gpu():
    """PLAN.md's contract: a deep tracker prefers CUDA and must run without it."""
    assert Sam3Tracker.requires_gpu is False


def test_an_unavailable_environment_explains_itself():
    reason = Sam3Tracker.unavailable_reason()
    assert reason is None or "sam3" in reason
