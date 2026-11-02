"""A fake HiT backend for the tests: corner maps drawn around the synthetic marker.

Kept out of ``conftest.py`` because only the HiT tests and the shared suite use
it. Like ``conftest.SiameseMarkerBackend`` it reads the real crop pixels the
plugin cut, so everything the plugin owns — crop geometry, resampling to the
network's input, the soft-argmax read-out, the anchor's own reading and the
point riding on it, the scale bookkeeping, holding through a loss and the
confidence calibration — is genuinely exercised, and only the network is stubbed.
"""

from __future__ import annotations

import math

import numpy as np

CELLS = 16
"""Side of HiT's corner and appearance maps."""

NETWORK = 256
"""Side of HiT's search input, in network pixels."""


def corner_logits(x: float, y: float, cells: int = CELLS, size: int = NETWORK) -> np.ndarray:
    """Return logits whose soft-argmax over HiT's corner grid is exactly ``(x, y)``.

    The probability is split between the two grid lines either side of each
    coordinate (a tent), so the expectation lands between cells as precisely as
    HiT's own soft-argmax can.
    """
    stride = size / cells

    def tent(value: float) -> np.ndarray:
        position = min(max(value / stride, 0.0), cells - 1.0)
        low = min(int(math.floor(position)), cells - 2)
        weight = np.zeros(cells)
        weight[low] = 1.0 - (position - low)
        weight[low + 1] = position - low
        return weight

    prob = np.outer(tent(y), tent(x))
    return np.log(prob + 1e-12).astype(np.float32)


class HiTMarkerBackend:
    """A :class:`~stable.trackers.siamese.SiameseBackend` that answers like HiT.

    For every search crop it finds the marker (the saturated pixels) and draws
    HiT's two corner maps at the marker's own bounding box — the object, as HiT
    would box it — plus an appearance map high around it. Where the marker is
    not visible the corner maps are flat and the appearance map low, as HiT's
    are for an absent target. Positions are converted to the 256 px network
    input exactly as resampling the crop would place them.

    Args:
        bias: ``(dx, dy)`` network pixels added to the box — a stand-in for HiT
            boxing an object beside the patch, which the point must ride on.
        oom_on: Match-call indices (1-based) that raise a CUDA-style
            out-of-memory error while the backend is not on the CPU.
        fail_on: Match-call indices that raise a plain ``RuntimeError``.
    """

    def __init__(
        self,
        *,
        bias: tuple[float, float] = (0.0, 0.0),
        oom_on: tuple[int, ...] = (),
        fail_on: tuple[int, ...] = (),
    ) -> None:
        self.bias = (float(bias[0]), float(bias[1]))
        self.device = "cpu"
        self.oom_on = set(oom_on)
        self.fail_on = set(fail_on)
        self.calls = 0
        self.embeds = 0
        self.searches: list[tuple[int, ...]] = []
        self.exemplars: list[tuple[int, ...]] = []
        self.released = False

    def embed(self, exemplars: np.ndarray) -> dict[str, object]:
        """Return a stand-in template, stamped with the device it was built on."""
        self.embeds += 1
        self.exemplars.append(tuple(exemplars.shape))
        return {"device": self.device, "count": int(exemplars.shape[0])}

    def match(self, kernels: dict[str, object], searches: np.ndarray):
        """Box the marker in every search crop, in HiT's map layout."""
        from stable.trackers.hit import HiTMaps

        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("simulated backend explosion")
        if self.calls in self.oom_on and self.device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if kernels["device"] != self.device:
            raise RuntimeError("templates live on another device; rebuild them after a move")
        self.searches.append(tuple(searches.shape))

        count, scales, side = searches.shape[:3]
        scale = NETWORK / side
        middle = (NETWORK - 1) / 2.0
        top_left = np.zeros((count, scales, CELLS, CELLS), dtype=np.float32)
        bottom_right = np.zeros_like(top_left)
        similarity = np.full_like(top_left, 0.05)
        grid = np.arange(CELLS, dtype=np.float64)
        for i in range(count):
            for k in range(scales):
                ys, xs = np.nonzero(searches[i, k].min(axis=2) >= 250)
                if not len(xs):
                    continue

                def network(value: float) -> float:
                    return (value - (side - 1) / 2.0) * scale + middle

                x0 = network(xs.min() - 0.5) + self.bias[0]
                x1 = network(xs.max() + 0.5) + self.bias[0]
                y0 = network(ys.min() - 0.5) + self.bias[1]
                y1 = network(ys.max() + 0.5) + self.bias[1]
                top_left[i, k] = corner_logits(x0, y0)
                bottom_right[i, k] = corner_logits(x1, y1)
                cell_x = ((x0 + x1) / 2.0 + 0.5) / (NETWORK / CELLS)
                cell_y = ((y0 + y1) / 2.0 + 0.5) / (NETWORK / CELLS)
                near = np.hypot(grid[None, :] - cell_x, grid[:, None] - cell_y) <= 2.0
                similarity[i, k] = np.where(near, 0.9, 0.05)
        return HiTMaps(
            score=top_left,
            bottom_right=bottom_right,
            similarity=similarity,
            crop_size=int(side),
            network_size=NETWORK,
        )

    def to_device(self, device: str) -> None:
        """Record a device move (there is nothing to move)."""
        self.device = device

    def release(self) -> None:
        """Record that the plugin released this backend."""
        self.released = True
