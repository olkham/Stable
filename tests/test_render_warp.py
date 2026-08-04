"""The per-frame warp and the three border policies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stable.core.errors import RenderError
from stable.core.geometry import SimilarityTransform
from stable.render import (
    INTERPOLATIONS,
    OutputGeometry,
    coverage_scale,
    cuda_available,
    plan_output,
    rescale_transform,
    warp_frame,
)

pytest.importorskip("cv2")

WIDTH, HEIGHT = 200, 100
IDENTITY = SimilarityTransform.identity()


def texture(width: int = WIDTH, height: int = HEIGHT, seed: int = 0) -> np.ndarray:
    """Return a frame whose every pixel is non-black, so gaps are visible."""
    rng = np.random.default_rng(seed)
    return rng.integers(40, 256, (height, width, 3), dtype=np.uint8)


# -- coverage ----------------------------------------------------------------


def test_a_still_shot_needs_no_crop_at_all():
    """Nothing moved, so the whole frame is covered in every frame."""
    assert coverage_scale([IDENTITY] * 5, WIDTH, HEIGHT) == 1.0


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (SimilarityTransform(tx=10.0), 1.0 - 2 * 10.0 / WIDTH),
        (SimilarityTransform(tx=-10.0), 1.0 - 2 * 10.0 / WIDTH),
        (SimilarityTransform(ty=5.0), 1.0 - 2 * 5.0 / HEIGHT),
        (SimilarityTransform(tx=10.0, ty=5.0), 1.0 - 2 * 10.0 / WIDTH),
    ],
)
def test_a_pure_shift_costs_twice_the_shift(transform, expected):
    """A centred crop loses the shift at both ends, so the closed form is exact."""
    assert coverage_scale([transform], WIDTH, HEIGHT) == pytest.approx(expected)


def test_scaling_up_never_needs_a_crop():
    """A frame pushed larger than the output covers all of it."""
    zoomed = SimilarityTransform.about_pivot((WIDTH / 2, HEIGHT / 2), scale=1.3)
    assert coverage_scale([zoomed], WIDTH, HEIGHT) == 1.0


def test_scaling_down_costs_exactly_the_scale():
    """Shrinking the frame by 20% leaves 80% of it covered."""
    shrunk = SimilarityTransform.about_pivot((WIDTH / 2, HEIGHT / 2), scale=0.8)
    assert coverage_scale([shrunk], WIDTH, HEIGHT) == pytest.approx(0.8)


def test_rotation_costs_what_the_geometry_says_it_should():
    """A rotated rectangle admits a smaller aligned rectangle, by a known amount."""
    angle = math.radians(5.0)
    rotated = SimilarityTransform.about_pivot((WIDTH / 2, HEIGHT / 2), rotation=angle)
    # Largest centred W:H rectangle inside a W x H rectangle rotated by `angle`.
    expected = min(
        HEIGHT / (WIDTH * math.sin(abs(angle)) + HEIGHT * math.cos(abs(angle))),
        WIDTH / (WIDTH * math.cos(abs(angle)) + HEIGHT * math.sin(abs(angle))),
    )
    assert coverage_scale([rotated], WIDTH, HEIGHT) == pytest.approx(expected, rel=1e-9)


def test_the_worst_frame_decides_the_crop():
    """Coverage is the minimum over the clip, not an average of it."""
    transforms = [IDENTITY, SimilarityTransform(tx=4.0), SimilarityTransform(tx=20.0)]
    assert coverage_scale(transforms, WIDTH, HEIGHT) == pytest.approx(
        coverage_scale([SimilarityTransform(tx=20.0)], WIDTH, HEIGHT)
    )


def test_a_frame_pushed_entirely_off_screen_gives_zero_coverage():
    """There is no common area at all, and the number says so."""
    assert coverage_scale([SimilarityTransform(tx=WIDTH * 2.0)], WIDTH, HEIGHT) == 0.0


def test_an_empty_clip_is_fully_covered():
    """Nothing to warp means nothing to crop."""
    assert coverage_scale([], WIDTH, HEIGHT) == 1.0


def test_a_bad_frame_size_is_rejected():
    """A zero-sized frame is a caller error, not something to guess about."""
    with pytest.raises(RenderError, match="frame size"):
        coverage_scale([IDENTITY], 0, HEIGHT)


# -- planning ----------------------------------------------------------------


def test_crop_outputs_the_common_area_at_its_own_size():
    """Crop gives up resolution, not sharpness: no scaling happens."""
    transforms = [SimilarityTransform(tx=10.0)]
    geometry = plan_output(transforms, WIDTH, HEIGHT, "crop")

    assert geometry.zoom == 1.0
    assert geometry.coverage == pytest.approx(0.9)
    assert (geometry.width, geometry.height) == (180, 90)
    assert geometry.fit.scale == 1.0
    assert geometry.fit.translation == (-10.0, -5.0)


def test_crop_dimensions_are_even():
    """Most encoders refuse an odd frame dimension outright."""
    geometry = plan_output([SimilarityTransform(tx=7.3)], 201, 101, "crop")
    assert geometry.width % 2 == 0
    assert geometry.height % 2 == 0


def test_auto_zoom_keeps_the_delivery_size_and_scales_to_fit():
    """Same resolution out as in, with the minimum zoom that covers every frame."""
    geometry = plan_output([SimilarityTransform(tx=10.0)], WIDTH, HEIGHT, "auto_zoom")
    assert (geometry.width, geometry.height) == (WIDTH, HEIGHT)
    assert geometry.zoom == pytest.approx(1.0 / 0.9)
    assert geometry.fit.scale == pytest.approx(1.0 / 0.9)
    # The zoom is about the frame centre, so the framing does not drift.
    centre = np.asarray([[WIDTH / 2.0, HEIGHT / 2.0]])
    assert np.allclose(geometry.fit.apply(centre), centre)


def test_auto_zoom_of_a_still_shot_is_no_zoom_at_all():
    """Nothing was uncovered, so nothing is thrown away."""
    geometry = plan_output([IDENTITY] * 4, WIDTH, HEIGHT, "auto_zoom")
    assert geometry.zoom == 1.0
    assert geometry.fit.is_identity(atol=1e-12)


def test_max_zoom_caps_the_zoom_and_says_it_did():
    """The user keeps their framing; the plan records that gaps may remain."""
    transforms = [SimilarityTransform(tx=30.0)]
    uncapped = plan_output(transforms, WIDTH, HEIGHT, "auto_zoom")
    capped = plan_output(transforms, WIDTH, HEIGHT, "auto_zoom", max_zoom=1.1)

    assert uncapped.zoom > 1.1
    assert uncapped.clamped is False
    assert capped.zoom == 1.1
    assert capped.clamped is True


def test_a_max_zoom_below_one_is_rejected():
    """Zooming *out* would only add gaps."""
    with pytest.raises(RenderError, match="max_zoom"):
        plan_output([SimilarityTransform(tx=30.0)], WIDTH, HEIGHT, "auto_zoom", max_zoom=0.5)


def test_transparent_keeps_the_whole_frame_and_measures_nothing():
    """Nothing is hidden, so there is nothing to crop to."""
    geometry = plan_output([SimilarityTransform(tx=50.0)], WIDTH, HEIGHT, "transparent")
    assert (geometry.width, geometry.height) == (WIDTH, HEIGHT)
    assert geometry.fit.is_identity()
    assert geometry.coverage == 1.0


def test_no_common_area_is_a_clear_error_with_a_way_out():
    """Crop and auto-zoom cannot do anything, and the message says what can."""
    transforms = [IDENTITY, SimilarityTransform(tx=WIDTH * 3.0)]
    with pytest.raises(RenderError, match="transparent"):
        plan_output(transforms, WIDTH, HEIGHT, "crop")
    with pytest.raises(RenderError, match="share no common area"):
        plan_output(transforms, WIDTH, HEIGHT, "auto_zoom")
    # ...and 'transparent' really is a way out.
    assert plan_output(transforms, WIDTH, HEIGHT, "transparent").width == WIDTH


def test_an_unknown_border_policy_lists_the_real_ones():
    """Typos name their alternatives."""
    with pytest.raises(RenderError, match="auto_zoom"):
        plan_output([IDENTITY], WIDTH, HEIGHT, "letterbox")


def test_geometry_serialises_for_a_ui():
    """The render panel needs the zoom and output size as plain data."""
    data = plan_output([SimilarityTransform(tx=10.0)], WIDTH, HEIGHT, "auto_zoom").to_dict()
    assert data["border"] == "auto_zoom"
    assert data["width"] == WIDTH
    assert data["coverage"] == pytest.approx(0.9)
    assert set(data["fit"]) == {"x", "y", "r", "s"}
    assert data["clamped"] is False


def test_geometry_reports_its_size_tuple():
    """A convenience the renderer and the encoder both use."""
    assert OutputGeometry(320, 240, IDENTITY, 1.0, 1.0, "crop").size == (320, 240)


# -- warping -----------------------------------------------------------------


def test_an_identity_warp_returns_the_frame_unchanged():
    """The cheapest possible correctness check on the affine convention."""
    frame = texture()
    assert np.array_equal(warp_frame(frame, IDENTITY), frame)


def test_the_transform_maps_source_pixels_forward():
    """A tx of +10 moves the picture right by 10 — not left, which is the classic bug."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[50, 20] = 255
    warped = warp_frame(frame, SimilarityTransform(tx=10.0, ty=4.0))
    assert warped[54, 30].tolist() == [255, 255, 255]
    assert warped[50, 20].tolist() == [0, 0, 0]


def test_uncovered_area_is_filled_with_the_background():
    """A shifted frame leaves a gap, and the gap is the colour we asked for."""
    warped = warp_frame(
        texture(), SimilarityTransform(tx=12.0), border="crop", background=(255, 0, 255)
    )
    assert warped[:, :10].reshape(-1, 3).tolist() == [[255, 0, 255]] * (HEIGHT * 10)
    assert not np.array_equal(warped[:, 100], np.full((HEIGHT, 3), (255, 0, 255)))


def test_transparent_returns_rgba_with_alpha_only_where_pixels_exist():
    """The alpha channel is the gap mask, which is what compositing needs."""
    warped = warp_frame(texture(), SimilarityTransform(tx=12.0), border="transparent")
    assert warped.shape == (HEIGHT, WIDTH, 4)
    assert (warped[:, :12, 3] == 0).all()
    assert (warped[:, 12:, 3] == 255).all()


def test_the_output_size_can_differ_from_the_source():
    """Crop renders straight into a smaller canvas — one resample, not two."""
    warped = warp_frame(texture(), SimilarityTransform(tx=-10.0), size=(180, 90))
    assert warped.shape == (90, 180, 3)


def test_the_crop_fit_lands_the_common_area_in_the_output():
    """Planning and warping agree: the cropped output has no gap in it."""
    transform = SimilarityTransform(tx=10.0, ty=5.0)
    geometry = plan_output([transform], WIDTH, HEIGHT, "crop")
    white = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    warped = warp_frame(
        white, geometry.fit.compose(transform), size=geometry.size, border="crop"
    )
    assert warped.min() > 0


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_every_interpolation_kernel_works(interpolation):
    """All four resampling modes are wired to a real OpenCV flag."""
    warped = warp_frame(
        texture(), SimilarityTransform(tx=0.5), interpolation=interpolation
    )
    assert warped.shape == (HEIGHT, WIDTH, 3)


def test_an_unknown_interpolation_kernel_is_rejected():
    """A typo must not silently fall back to nearest-neighbour."""
    with pytest.raises(RenderError, match="unknown interpolation"):
        warp_frame(texture(), IDENTITY, interpolation="bicubic-ish")


def test_a_non_rgb_frame_is_rejected():
    """The RGB (H, W, 3) convention is fixed project-wide."""
    with pytest.raises(RenderError, match="RGB"):
        warp_frame(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), IDENTITY)


def test_a_float_frame_is_coerced_rather_than_refused():
    """Callers holding float frames get a sensible uint8 result."""
    warped = warp_frame(texture().astype(np.float32), IDENTITY)
    assert warped.dtype == np.uint8


def test_the_gpu_path_falls_back_silently():
    """``use_gpu`` is an optimisation; without CUDA it must still produce the frame."""
    frame = texture()
    assert np.array_equal(warp_frame(frame, IDENTITY, use_gpu=True), frame)
    assert isinstance(cuda_available(), bool)


# -- proxy resolutions -------------------------------------------------------


def test_rescale_transform_scales_only_the_translation():
    """Rotation and the similarity's own scale are dimensionless; pixels are not."""
    warp = SimilarityTransform(tx=12.0, ty=-8.0, rotation=0.3, scale=1.2)
    half = rescale_transform(warp, 0.5)
    assert half.translation == (6.0, -4.0)
    assert half.rotation == warp.rotation
    assert half.scale == warp.scale


def test_rescaling_matches_warping_a_downscaled_frame():
    """A proxy warped by the rescaled transform lands where the full-res one did."""
    import cv2

    full = texture(240, 120, seed=2)
    warp = SimilarityTransform.about_pivot((120.0, 60.0), rotation=0.05, translation=(9.0, -6.0))

    warped_then_shrunk = cv2.resize(
        warp_frame(full, warp), (120, 60), interpolation=cv2.INTER_AREA
    )
    shrunk_then_warped = warp_frame(
        cv2.resize(full, (120, 60), interpolation=cv2.INTER_AREA),
        rescale_transform(warp, 0.5),
        size=(120, 60),
    )
    # Resampling order changes the pixels slightly; the geometry must agree.
    difference = np.abs(
        warped_then_shrunk.astype(np.int16) - shrunk_then_warped.astype(np.int16)
    )
    assert difference.mean() < 12.0


def test_a_non_positive_resolution_scale_is_rejected():
    """A zero or negative proxy scale is a caller error."""
    with pytest.raises(RenderError, match="resolution scale"):
        rescale_transform(IDENTITY, 0.0)
