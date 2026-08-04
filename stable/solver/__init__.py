"""Motion solver: point tracks to a per-frame similarity transform.

Confidence-weighted Umeyama fit with RANSAC outlier rejection when there are
three or more points, degrading to translation-only with one point and to
interpolation across frames with none. Output is the four ``X, Y, R, S`` curves
plus a per-frame residual and status — everything the curve editor, the
smoothing library and the stabilisation renderer are written against.

Typical use::

    from stable.solver import SolveConfig, solve_motion

    solve = solve_motion(track_result, config=SolveConfig(reference_frame=0),
                         frame_size=(video.width, video.height))
    solve.curve("x")            # [T] pixels the shot moved horizontally
    solve.transform_at(12)      # the origin-based similarity for frame 12
    solve.flagged               # [T] bool: frames worth the user's attention

Layout:

``similarity``  the closed-form weighted Umeyama fit and its DOF constraints
``ransac``      robust fitting from minimal (two-point) samples
``motion``      the per-clip drive loop, curve assembly and quality record
"""

from __future__ import annotations

from .motion import (
    IDENTITY_CURVES,
    FrameSolve,
    MotionSolve,
    SolveConfig,
    SolveStatus,
    solve_frame,
    solve_motion,
)
from .ransac import RansacConfig, RansacResult, solve_similarity_ransac
from .similarity import SolveConstraints, solve_similarity, transform_residuals, weighted_rms

__all__ = [
    # per-frame geometry
    "SolveConstraints",
    "solve_similarity",
    "transform_residuals",
    "weighted_rms",
    # robust fitting
    "RansacConfig",
    "RansacResult",
    "solve_similarity_ransac",
    # per-clip solve
    "SolveStatus",
    "SolveConfig",
    "FrameSolve",
    "MotionSolve",
    "solve_motion",
    "solve_frame",
    "IDENTITY_CURVES",
]
