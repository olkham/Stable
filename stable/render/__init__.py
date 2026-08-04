"""Stabilized render: per-frame inverse similarity warp and export.

The stabilisation warp is ``smoothed(t) . raw(t)^-1`` applied as a single affine
per frame, with border policies crop / auto_zoom / transparent (see
:data:`stable.core.project.BORDER_POLICIES`). The crop rectangle and the
auto-zoom factor are computed exactly, from every frame's warp at once, so
"minimal zoom with no gaps" is a guarantee.

Typical use::

    from stable.render import StabilizationPlan, render_stabilized

    plan = StabilizationPlan.build(motion, source_size=(video.width, video.height),
                                   border="auto_zoom")
    plan.geometry.zoom                      # what the shot cost in crop
    render_stabilized(video, plan, "out.mp4", audio_source="in.mp4")

Layout:

``warp``       the per-frame affine, the border policies and the zoom solve
``stabilize``  planning a whole clip and streaming it to a file or PNG sequence
"""

from __future__ import annotations

from .stabilize import (
    StabilizationPlan,
    iter_stabilized_frames,
    render_stabilized,
    stabilize,
    stabilized_frame,
    write_frame_sequence,
)
from .warp import (
    INTERPOLATIONS,
    OutputGeometry,
    coverage_scale,
    cuda_available,
    plan_output,
    rescale_transform,
    warp_frame,
)

__all__ = [
    # warping
    "INTERPOLATIONS",
    "OutputGeometry",
    "coverage_scale",
    "plan_output",
    "warp_frame",
    "rescale_transform",
    "cuda_available",
    # rendering
    "StabilizationPlan",
    "iter_stabilized_frames",
    "render_stabilized",
    "write_frame_sequence",
    "stabilize",
    "stabilized_frame",
]
