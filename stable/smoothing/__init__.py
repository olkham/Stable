"""Smoothing filters over the solved X, Y, R, S curves.

Moving average, Gaussian, zero-phase Butterworth, Savitzky-Golay, smoothing
spline and tripod mode — selectable per channel, each with a strength, all
implemented in pure NumPy so the stabiliser has no scientific-stack dependency.

Typical use::

    from stable.smoothing import SmoothingSettings, smooth_motion

    settings = SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0})
    settings.set_enabled("s", False)          # leave zoom alone
    settings.set_strength(0.8, channels=["r"])

    motion = smooth_motion(solve, settings, fps=video.fps)
    motion.stabilization_transforms()          # what the renderer warps by

Layout:

``dsp``       filter design and the numerical kernels, in pure NumPy
``filters``   the filter catalogue, its parameter schema and the strength blend
``edit``      curve-editor operations: sub-range smoothing and keyframe pinning
``channels``  per-channel enable/strength and the project-file round trip
``motion``    smoothing a solve and turning it into a stabilisation warp
"""

from __future__ import annotations

from .channels import (
    DIRECTIVES,
    PIN_DIRECTIVE,
    RANGE_DIRECTIVE,
    ChannelSmoothing,
    SmoothingSettings,
    smooth_curve,
)
from .edit import DEFAULT_PIN_RADIUS, DEFAULT_RANGE_BLEND, blend_subrange, pin_curve
from .filters import (
    FILTERS,
    STRENGTH,
    FilterContext,
    FilterInfo,
    FilterParam,
    apply_filter,
    describe_filters,
    fill_gaps,
    filter_names,
    get_filter,
)
from .motion import MIN_SCALE, SmoothedMotion, smooth_motion

__all__ = [
    # catalogue
    "FILTERS",
    "STRENGTH",
    "FilterContext",
    "FilterInfo",
    "FilterParam",
    "apply_filter",
    "describe_filters",
    "fill_gaps",
    "filter_names",
    "get_filter",
    # curve-editor operations
    "blend_subrange",
    "pin_curve",
    "DEFAULT_PIN_RADIUS",
    "DEFAULT_RANGE_BLEND",
    # configuration
    "ChannelSmoothing",
    "SmoothingSettings",
    "smooth_curve",
    "RANGE_DIRECTIVE",
    "PIN_DIRECTIVE",
    "DIRECTIVES",
    # applying it to a solve
    "SmoothedMotion",
    "smooth_motion",
    "MIN_SCALE",
]
