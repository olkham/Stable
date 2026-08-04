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
    plan.is_no_op                           # True => nothing asked it to move
    render_stabilized(video, plan, "out.mp4", audio_source="in.mp4")

The operator's three choices — what to hold still, how much to crop, what goes
in the exposed edge — are :class:`~stable.render.options.RenderOptions`::

    plan = StabilizationPlan.build(
        motion, source_size=(w, h), border="crop",
        target_positions=path_of_the_point_to_pin,   # [T, 2] in source pixels
        crop=CropSpec(scale=0.9, centre="target"),
        fill=BorderFill(mode="blur", feather=8.0),
    )

Layout:

``warp``       the per-frame affine, the border policies and the zoom solve
``fill``       what goes in the uncovered edge (solid / stretch / mirror / blur)
``options``    the serialisable target / crop / fill choices
``stabilize``  planning a whole clip and streaming it to a file or PNG sequence
"""

from __future__ import annotations

from .fill import FILL_MODES, BorderFill, composite_fill
from .options import (
    CROP_CENTRES,
    TARGET_MODES,
    CropSpec,
    RenderOptions,
    StabilizationTarget,
    average_positions,
    hold_gaps,
)
from .stabilize import (
    IDENTITY_TOLERANCE_PX,
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
    # border fill
    "FILL_MODES",
    "BorderFill",
    "composite_fill",
    # operator options
    "TARGET_MODES",
    "CROP_CENTRES",
    "StabilizationTarget",
    "CropSpec",
    "RenderOptions",
    "average_positions",
    "hold_gaps",
    # rendering
    "IDENTITY_TOLERANCE_PX",
    "StabilizationPlan",
    "iter_stabilized_frames",
    "render_stabilized",
    "write_frame_sequence",
    "stabilize",
    "stabilized_frame",
]
