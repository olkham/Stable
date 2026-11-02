"""A stand-in for SUTrack's backend, for tests with no torch and no weights.

Kept out of ``conftest.py`` only so that adding it touches no shared test file;
it plays the part :class:`conftest.SiameseMarkerBackend` plays for Ocean and
SiamDW, in SUTrack's own map layout.
"""

from __future__ import annotations

import numpy as np
from conftest import find_marker

from stable.trackers.siamese import EXEMPLAR_SIZE
from stable.trackers.sutrack import PATCH, SEARCH_SIDE, SUTrackMaps, network_scale


class SUTrackMarkerBackend:
    """A :class:`~stable.trackers.siamese.SiameseBackend` that answers like SUTrack.

    It finds the synthetic clip's white marker in the real search crops the
    plugin cut and draws SUTrack's maps around it: 14x14 cells
    :func:`~stable.trackers.sutrack.network_scale` apart, a centre score of
    ``0.95`` on the cells next to the marker and ``0.02`` elsewhere (and
    everywhere when it is not visible), and on every cell the box of the target
    centred on the marker. That box is the point's square — half the exemplar's
    side at the anchor's scale — or, with ``measure_box``, the white
    rectangle's own extent, as a box regression would report it.

    Args:
        bias: ``(dx, dy)`` crop pixels added to where the target is reported, a
            stand-in for a network boxing something beside the patch.
        measure_box: Report the white region's own box instead of a fixed one.
        oom_on: Match-call indices (1-based) that raise a CUDA-style
            out-of-memory error while the backend is not on the CPU.
        fail_on: Match-call indices that raise a plain ``RuntimeError``.
    """

    def __init__(
        self,
        *,
        bias: tuple[float, float] = (0.0, 0.0),
        measure_box: bool = False,
        oom_on: tuple[int, ...] = (),
        fail_on: tuple[int, ...] = (),
    ) -> None:
        self.bias = (float(bias[0]), float(bias[1]))
        self.measure_box = measure_box
        self.device = "cpu"
        self.oom_on = set(oom_on)
        self.fail_on = set(fail_on)
        self.calls = 0
        self.embeds = 0
        self.searches: list[tuple[int, ...]] = []
        self.template_sizes: list[np.ndarray] = []
        self.released = False

    def use_template_boxes(self, sizes: np.ndarray) -> None:
        """Record the template boxes the plugin announced."""
        self.template_sizes.append(np.array(sizes, dtype=np.float64))

    def embed(self, exemplars: np.ndarray) -> dict[str, object]:
        """Return a stand-in template, stamped with the device it was built on."""
        self.embeds += 1
        return {"device": self.device, "count": int(exemplars.shape[0])}

    def match(self, kernels: dict[str, object], searches: np.ndarray) -> SUTrackMaps:
        """Find the marker in every search crop and draw SUTrack's maps around it."""
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("simulated backend explosion")
        if self.calls in self.oom_on and self.device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if kernels["device"] != self.device:
            raise RuntimeError("templates live on another device; rebuild them after a move")
        self.searches.append(tuple(searches.shape))

        count, scales, side = searches.shape[:3]
        cells = SEARCH_SIDE // PATCH
        stride = PATCH / network_scale(side)
        grid = (np.arange(cells) - (cells - 1) / 2.0) * stride
        grid_x, grid_y = np.meshgrid(grid, grid)
        half = EXEMPLAR_SIZE / 4.0
        score = np.full((count, scales, cells, cells), 0.02)
        boxes = np.full((count, scales, 4, cells, cells), half)
        centre = (side - 1) / 2.0
        for i in range(count):
            for k in range(scales):
                crop = searches[i, k]
                found = find_marker(crop)
                if found is None:
                    continue
                mx, my = found[0] - centre + self.bias[0], found[1] - centre + self.bias[1]
                left = top = right = bottom = half
                if self.measure_box:
                    ys, xs = np.nonzero(crop.min(axis=2) >= 250)
                    left, right = mx - (xs.min() - 0.5 - centre), (xs.max() + 0.5 - centre) - mx
                    top, bottom = my - (ys.min() - 0.5 - centre), (ys.max() + 0.5 - centre) - my
                near = (np.abs(grid_x - mx) <= stride) & (np.abs(grid_y - my) <= stride)
                score[i, k] = np.where(near, 0.95, 0.02)
                boxes[i, k] = [
                    grid_x - (mx - left),
                    grid_y - (my - top),
                    (mx + right) - grid_x,
                    (my + bottom) - grid_y,
                ]
        return SUTrackMaps(score=score, boxes=boxes, stride=stride)

    def to_device(self, device: str) -> None:
        """Record a device move (there is nothing to move)."""
        self.device = device

    def release(self) -> None:
        """Record that the plugin released this backend."""
        self.released = True
