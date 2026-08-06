"""One producer of "a frame, and the geometry that gets drawn over it".

Three callers want the same two things and must never disagree about either:
:func:`stable.server.routes.projects.get_frame` and
:func:`stable.server.routes.solve.get_stabilized_frame` want the pixels,
:func:`stable.server.routes.export.export_preview` wants the per-frame geometry,
and the frame WebSocket (:mod:`stable.server.routes.ws`) wants **both, in one
message** — which is the point of it, since pixels and overlay arriving
separately can always be drawn a frame apart.

So the knowledge lives here, once, and those routes are thin wrappers:

* :func:`source_frame_jpeg` — the proxy JPEG, read off disk as bytes. It is
  already a JPEG, so nothing on this path decodes or re-encodes anything.
* :func:`output_frame_jpeg` — the same frame warped by the render's own
  ``output_transform``: stabilisation, crop and border fill, re-expressed for
  the proxy's resolution. This one costs a decode, a warp and an encode, and is
  the reason the stream coalesces requests rather than serving every one.
* :func:`frame_geometry` — the output rectangle in source pixels, the
  stabilisation warp, and where each mapped target lands. No pixels at all.

The project's motion caches are loaded here too (:func:`load_solve`,
:func:`load_smoothed`, :func:`load_options`, :func:`preview_plan`), because a
frame cannot be produced without them and :mod:`stable.server.routes.solve`
would otherwise have to import this module while this module imported it.
Those four are re-exported from ``routes.solve``, which is where they used to
live and where the rest of the server still reaches for them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import StableError
from ..core.project import Project
from ..render.options import RenderOptions
from ..render.stabilize import StabilizationPlan
from ..render.warp import rescale_transform, warp_frame
from ..smoothing.motion import SmoothedMotion
from ..solver.motion import MotionSolve
from . import motion as motion_glue
from .storage import Workspace

__all__ = [
    "VARIANTS",
    "FrameUnavailable",
    "solve_cache_path",
    "smoothing_cache_path",
    "load_options",
    "load_solve",
    "load_smoothed",
    "has_motion",
    "preview_plan",
    "plan_revision",
    "source_frame_jpeg",
    "output_frame_jpeg",
    "frame_geometry",
]

VARIANTS = ("source", "output")
"""The two pictures of one frame: the proxy as shot, and what the export writes."""


class FrameUnavailable(StableError):
    """A frame cannot be produced, carrying the HTTP status that says why.

    The REST routes turn this into an ``HTTPException`` and the frame socket
    turns it into an ``error`` message; both need the same distinction between
    "there is no such frame" (404) and "this project has not been solved and
    smoothed yet" (409), which is a state the UI shows rather than a failure.
    """

    def __init__(self, detail: str, status: int = 404) -> None:
        """Record why the frame is unavailable, and the status that expresses it."""
        super().__init__(detail)
        self.status = status


# -- the motion caches ----------------------------------------------------------


def solve_cache_path(workspace: Workspace, project_id: str) -> Path:
    """Return where a project's cached raw solve lives."""
    return workspace.project_dir(project_id) / motion_glue.SOLVE_CACHE


def smoothing_cache_path(workspace: Workspace, project_id: str) -> Path:
    """Return where a project's cached smoothed curves live."""
    return workspace.project_dir(project_id) / motion_glue.SMOOTHING_CACHE


def _options_path(workspace: Workspace, project_id: str) -> Path:
    return motion_glue.options_path(workspace.project_dir(project_id))


def load_options(workspace: Workspace, project_id: str) -> RenderOptions:
    """Return a project's stabilisation options (target / crop / fill).

    Shared by the preview, the frame stream and the export, so all three are
    planned from exactly the same choices.
    """
    return motion_glue.load_render_options(workspace.project_dir(project_id))


def load_solve(workspace: Workspace, project_id: str) -> MotionSolve | None:
    """Return a project's cached raw solve, or ``None`` if it has not been solved."""
    payload = motion_glue.read_json(solve_cache_path(workspace, project_id))
    return None if payload is None else motion_glue.solve_from_cache(payload)


def load_smoothed(
    workspace: Workspace, project_id: str, project: Project | None = None
) -> SmoothedMotion | None:
    """Return a project's smoothed motion, or ``None`` if it has not been solved.

    The smoothed *curves* are cached, but the object handed back is rebuilt on
    top of the cached solve so callers get the real
    :class:`~stable.smoothing.motion.SmoothedMotion` — the thing that knows how
    to produce a stabilisation warp — rather than four bare lists.
    """
    solve = load_solve(workspace, project_id)
    if solve is None:
        return None
    if not smoothing_cache_path(workspace, project_id).is_file():
        return None
    return motion_glue.smooth_solve(project or workspace.load_project(project_id), solve)


def has_motion(workspace: Workspace, project_id: str) -> bool:
    """Return whether :func:`load_smoothed` would answer with something.

    Two ``stat`` calls, where :func:`load_smoothed` is two JSON reads and a
    re-smooth. The frame stream asks this once per requested frame and only
    pays for the real load when the memoised plan misses.
    """
    return solve_cache_path(workspace, project_id).is_file() and (
        smoothing_cache_path(workspace, project_id).is_file()
    )


def _plan_key(workspace: Workspace, project_id: str, project: Project) -> tuple[Any, ...]:
    """Everything a built plan depends on, as a comparable tuple.

    The mtimes cover the solve, the smoothing and the stabilisation options; the
    border policy and clip size come off the project file, which the caches do
    not carry.
    """
    options_file = _options_path(workspace, project_id)
    return (
        solve_cache_path(workspace, project_id).stat().st_mtime_ns,
        smoothing_cache_path(workspace, project_id).stat().st_mtime_ns,
        options_file.stat().st_mtime_ns if options_file.is_file() else 0,
        project.render.border,
        project.video.w,
        project.video.h,
    )


def preview_plan(
    workspace: Workspace,
    project_id: str,
    project: Project,
    motion: SmoothedMotion | None = None,
) -> StabilizationPlan:
    """Return the render plan for the preview, memoised on the caches' mtimes.

    The same call the export uses, so "what you see is what you get" survives
    the stabilisation target, the crop scale and the border fill as well as the
    border policy.

    Args:
        workspace: The workspace holding the project.
        project_id: Which project to plan for.
        project: That project, already loaded.
        motion: The smoothed motion, when the caller already has it. It is only
            *needed* on a cache miss, so the frame stream leaves it out and
            avoids re-reading two JSON caches for every frame it serves.

    Raises:
        FrameUnavailable: 409 if the project has not been solved and smoothed.
    """
    key = _plan_key(workspace, project_id, project)

    def build() -> StabilizationPlan:
        smoothed = motion if motion is not None else load_smoothed(workspace, project_id, project)
        if smoothed is None:
            raise FrameUnavailable(
                "solve and smooth this project before previewing the export", status=409
            )
        return motion_glue.build_plan(
            project, smoothed, options=load_options(workspace, project_id)
        )

    return motion_glue.cached_plan(project_id, key, build)


def plan_revision(workspace: Workspace, project_id: str, project: Project) -> str:
    """A short id for the inputs a rendered output frame depends on.

    Output frames cannot be cached by URL — the pixels change whenever the
    smoothing or the render options do — so the stream stamps every one with
    this and the client throws away everything under an older stamp. That is
    the whole invalidation story: no explicit purge, no stale warp on screen.
    """
    key = repr(_plan_key(workspace, project_id, project)).encode()
    digest = hashlib.blake2b(key, digest_size=8)
    return digest.hexdigest()


# -- the pixels -----------------------------------------------------------------


def _proxy_revision(workspace: Workspace, project_id: str, tier: str) -> str:
    """A short id for a proxy tier, so regenerated proxies invalidate the client's cache."""
    manifest = workspace.proxies_dir(project_id) / tier / "proxy.json"
    stamp = manifest.stat().st_mtime_ns if manifest.is_file() else 0
    return f"{tier}-{stamp:x}"


def source_frame_jpeg(
    workspace: Workspace, project_id: str, index: int, tier: str = "half"
) -> tuple[bytes, str]:
    """Return one proxy frame's JPEG bytes and the tier's revision id.

    Disk I/O only — the proxy store holds JPEGs, so this neither decodes nor
    re-encodes anything, which is what keeps timeline scrubbing cheap.

    Raises:
        FrameUnavailable: 404 if the tier has not been generated or the index is
            outside the clip.
    """
    try:
        store = workspace.get_proxy_store(project_id, tier)
        path = Path(store.frame_path(index))
    except StableError as exc:
        raise FrameUnavailable(str(exc)) from exc
    except IndexError as exc:
        raise FrameUnavailable(str(exc)) from exc
    return path.read_bytes(), _proxy_revision(workspace, project_id, tier)


def _scaled_fill(fill: Any, scale: float) -> Any:
    """Re-express a border fill for a proxy's resolution.

    ``blur`` and ``feather`` are pixel distances, so they scale with the frame;
    the mode, colour and tint are dimensionless and pass through untouched.
    """
    if fill is None or scale == 1.0:
        return fill
    from dataclasses import replace  # noqa: PLC0415 - one caller, keeps the import local

    return replace(fill, blur=fill.blur * scale, feather=fill.feather * scale)


def output_frame_jpeg(
    workspace: Workspace,
    project_id: str,
    index: int,
    tier: str = "half",
    project: Project | None = None,
) -> tuple[bytes, str]:
    """Return what the export would write for one frame, as JPEG bytes.

    The warp is the render's own ``output_transform`` — stabilisation (including
    the stabilisation target's lock) plus the project's border policy and crop —
    re-expressed for the proxy's resolution by
    :func:`stable.render.warp.rescale_transform`, so what the preview shows is
    what the export will produce, border fill and all.

    Raises:
        FrameUnavailable: 409 if the project has not been solved and smoothed,
            404 if the frame index or proxy tier is invalid.
    """
    proj = project if project is not None else workspace.load_project(project_id)
    if not has_motion(workspace, project_id):
        raise FrameUnavailable(
            "run solve and smoothing before requesting a stabilized preview", status=409
        )
    try:
        store = workspace.get_proxy_store(project_id, tier)
        frame = store.get_frame(index)
    except StableError as exc:
        raise FrameUnavailable(str(exc)) from exc
    except IndexError as exc:
        raise FrameUnavailable(str(exc)) from exc
    try:
        plan = preview_plan(workspace, project_id, proj)
    except StableError as exc:
        if isinstance(exc, FrameUnavailable):
            raise
        raise FrameUnavailable(str(exc), status=400) from exc
    if not 0 <= index < len(plan.transforms):
        raise FrameUnavailable(f"frame {index} outside the solved range")

    scale = float(store.manifest.scale)
    warped = warp_frame(
        frame,
        rescale_transform(plan.output_transform(index), scale),
        size=(
            max(int(round(plan.geometry.width * scale)), 1),
            max(int(round(plan.geometry.height * scale)), 1),
        ),
        border=plan.border,
        interpolation=plan.interpolation,
        # The fill's blur and feather are in pixels, so a proxy needs them
        # scaled or the preview shows a softer edge than the export will.
        fill=_scaled_fill(plan.fill, scale),
    )

    import cv2  # noqa: PLC0415 - lazy, matches the rest of the video stack

    # JPEG carries no alpha, so the transparent policy's uncovered area is shown
    # as the black it was composited against rather than failing the request.
    rgb = warped[:, :, :3] if warped.shape[2] == 4 else warped
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - encoder failure only
        raise FrameUnavailable("could not encode stabilized frame", status=500)
    return buf.tobytes(), plan_revision(workspace, project_id, proj)


# -- the geometry drawn over the pixels ------------------------------------------


def _similarity(transform: Any) -> dict[str, float]:
    """Return a similarity as the four numbers the browser applies it with."""
    return {
        "tx": float(transform.tx),
        "ty": float(transform.ty),
        "rotation": float(transform.rotation),
        "scale": float(transform.scale),
    }


def frame_geometry(
    project: Project, plan: StabilizationPlan, options: RenderOptions, index: int
) -> dict[str, Any]:
    """Describe one frame's export geometry, without touching a pixel.

    Returns:
        ``rect``, the output rectangle mapped back into **source** pixels so the
        viewer can draw it over the frame; ``warp``, the stabilisation warp
        *before* the crop/zoom fit, which is the space mapped targets are stated
        in; ``output_warp``, the whole transform including the fit, which is the
        map from source pixels into **output** pixels; and ``targets``, one row
        per mapped point saying where it lands and by how much it misses.

    A mapped target's destination is stated before the fit because the fit is
    only knowable once every frame's transform exists. The viewer draws and
    drags in source pixels, so it needs that warp itself; four numbers describe
    it exactly, and sending them beats the viewer inferring it from the output
    rectangle (which carries the fit, and so would place every crosshair
    slightly off whenever the render crops).

    ``output_warp`` is the other direction, and is what the side-by-side output
    canvas draws in (todo 4, sixth round): a crosshair shown *on the output* is
    at ``output_warp(source position)``, and one dragged there comes back the
    same way. ``rect`` is this transform's inverse applied to the output
    corners, so the two cannot disagree — which is precisely why it is sent
    rather than recovered from those corners by a second implementation of the
    same arithmetic.
    """
    total = len(plan.transforms)
    idx = max(0, min(int(index), total - 1)) if total else 0
    geometry = plan.geometry
    output_transform = plan.output_transform(idx)
    corners = [
        [0.0, 0.0],
        [float(geometry.width), 0.0],
        [float(geometry.width), float(geometry.height)],
        [0.0, float(geometry.height)],
    ]
    # Where the output rectangle sits *on the source frame*: the inverse of the
    # transform that put the source there. This is what makes the preview an
    # answer to "what will be kept?" rather than a picture of the same frame.
    inverse = output_transform.inverse()
    rect = [[float(x), float(y)] for x, y in inverse.apply(np.asarray(corners))]
    warp = plan.transforms[idx] if plan.transforms else output_transform
    return {
        "frame": idx,
        "frames": total,
        "source": {"width": plan.source_size[0], "height": plan.source_size[1]},
        "output": {"width": geometry.width, "height": geometry.height},
        "rect": rect,
        "warp": _similarity(warp),
        "output_warp": _similarity(output_transform),
        # Two or more pairs is what turns a translation into a mapping that can
        # rotate and scale; the screen says which the operator has asked for.
        "uses_rotation_and_scale": bool(options.target.uses_rotation_and_scale),
        "targets": motion_glue.target_residuals(project, plan, options.target, idx),
    }
