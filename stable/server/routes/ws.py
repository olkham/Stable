"""WebSocket streaming: background-job progress, and video frames.

Two different problems, both better served by a socket than by polling or by a
request per item.

Job progress (:func:`project_progress`, :func:`models_progress`)
    ``/ws/projects/{project_id}`` for project-scoped jobs
    (track/solve/render/export/proxy) and ``/ws/models`` for model-checkpoint
    downloads, which are process-wide rather than tied to any project — see
    :data:`stable.server.jobs.MODELS_JOB_SCOPE` and
    :mod:`stable.server.routes.models`. On connect each sends a snapshot of
    every job currently tracked for its scope, then polls
    :class:`~stable.server.jobs.JobManager` a few times a second and pushes any
    job whose event payload changed. Polling (rather than a thread-to-asyncio
    pub/sub bridge) is a deliberate simplification — see the module docstring in
    :mod:`stable.server.jobs` for why that tradeoff is fine for a local,
    single-user tool.

Frames (:func:`frame_stream`)
    ``/ws/projects/{project_id}/frames``. See that function for the protocol.
    Three things a socket gives here that ``GET /frames/{n}`` cannot:

    1. **Superseded work is dropped.** Over HTTP the server cannot know that
       the request for frame 244 is worthless because the user has scrubbed to
       259; it warps all of them. Here only the newest request survives, so a
       fast scrub costs a handful of warps instead of one per frame. On the
       ``output`` variant — a decode, a warp and a JPEG encode each — that is
       the difference between usable and not.
    2. **One queue, ordered by the server.** Browsers cap concurrent
       connections per origin, so over HTTP the frame being *looked at* can
       queue behind prefetches. Here wants are served first and prefetch only
       when nothing is wanted.
    3. **Pixels and geometry in one message**, which is the real reason. The
       viewer draws the export's output rectangle, its thirds and its target
       crosshairs *over* the frame; two HTTP responses can land in either order,
       so the overlay could be a frame ahead of the picture it describes. One
       message cannot be.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..frames import (
    VARIANTS,
    FrameUnavailable,
    frame_geometry,
    geometry_context,
    output_frame_jpeg,
    source_frame_jpeg,
)
from ..jobs import MODELS_JOB_SCOPE, JobManager, get_job_manager
from ..storage import Workspace, get_workspace

__all__ = ["router"]

router = APIRouter()

_POLL_INTERVAL = 0.15

_PREFETCH_LIMIT = 32
"""How many prefetch requests are remembered. Beyond this the oldest is dropped.

A prefetch is a guess about where the playhead is going. Holding an unbounded
backlog of guesses would mean serving frames the user left behind minutes ago.
"""


async def _stream_jobs(websocket: WebSocket, scope: str) -> None:
    """Push job-status events for one :class:`JobManager` scope until disconnect.

    Args:
        websocket: The (not-yet-accepted) connection.
        scope: A project id, or :data:`MODELS_JOB_SCOPE` for model downloads.
    """
    await websocket.accept()
    jobs: JobManager = get_job_manager()
    last: dict[str, dict] = {}
    try:
        while True:
            for job in jobs.jobs_for_project(scope):
                event = job.to_event()
                if last.get(job.id) != event:
                    last[job.id] = event
                    await websocket.send_json({"type": "job", **event})
            await asyncio.sleep(_POLL_INTERVAL)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


@router.websocket("/ws/projects/{project_id}")
async def project_progress(websocket: WebSocket, project_id: str) -> None:
    """Stream job status updates for one project until the client disconnects."""
    await _stream_jobs(websocket, project_id)


@router.websocket("/ws/models")
async def models_progress(websocket: WebSocket) -> None:
    """Stream model-checkpoint download job status until the client disconnects."""
    await _stream_jobs(websocket, MODELS_JOB_SCOPE)


# -- the frame stream ------------------------------------------------------------


@dataclass
class _FrameRequest:
    """One frame the client asked for, and in which pictures of it."""

    frame: int
    variants: tuple[str, ...]
    tier: str
    seq: int
    # Only a *want* carries geometry: a prefetch is pixels for later, and the
    # geometry that goes with them is only true for the frame being drawn now.
    geometry: bool


@dataclass
class _StreamState:
    """What one connection has outstanding: at most one want, plus guesses.

    The want is a single slot on purpose. A newer one replaces the older
    unanswered one, which is the coalescing the whole design rests on — the
    client asks for the frame under the playhead, and by the time the server
    gets to it the playhead may have moved, in which case the older frame is
    worthless and serving it would only delay the one that is not.
    """

    want: _FrameRequest | None = None
    prefetch: OrderedDict[tuple[str, int, str], _FrameRequest] = field(default_factory=OrderedDict)
    signal: asyncio.Event = field(default_factory=asyncio.Event)

    def submit(self, message: dict[str, Any]) -> None:
        """Record one client message, dropping anything it supersedes."""
        kind = message.get("type")
        tier = str(message.get("tier") or "half")
        variants = tuple(v for v in message.get("variants") or ("source",) if v in VARIANTS)
        if not variants:
            return
        if kind == "want":
            frame = message.get("frame")
            if not isinstance(frame, int):
                return
            self.want = _FrameRequest(
                frame=frame,
                variants=variants,
                tier=tier,
                seq=int(message.get("seq") or 0),
                geometry=bool(message.get("geometry", True)),
            )
        elif kind == "prefetch":
            frames = message.get("frames")
            if not isinstance(frames, list):
                return
            for frame in frames:
                if not isinstance(frame, int):
                    continue
                for variant in variants:
                    key = (variant, frame, tier)
                    self.prefetch.pop(key, None)
                    self.prefetch[key] = _FrameRequest(
                        frame=frame, variants=(variant,), tier=tier, seq=0, geometry=False
                    )
            while len(self.prefetch) > _PREFETCH_LIMIT:
                self.prefetch.popitem(last=False)
        else:
            return
        self.signal.set()

    def take(self) -> _FrameRequest | None:
        """Return the next thing to serve: the want if there is one, else a guess."""
        if self.want is not None:
            request, self.want = self.want, None
            return request
        if self.prefetch:
            _key, request = self.prefetch.popitem(last=False)
            return request
        return None


def _frame_payload(
    workspace: Workspace, project_id: str, frame: int, variant: str, tier: str
) -> tuple[bytes, str]:
    """Produce one picture of one frame. Blocking: always called in a thread."""
    if variant == "source":
        return source_frame_jpeg(workspace, project_id, frame, tier)
    # The project as the cached context already holds it, rather than read and
    # parsed again for every output frame. Without a context the render path
    # loads it itself and reports whatever is wrong in its own words.
    try:
        context = geometry_context(workspace, project_id)
    except Exception:  # noqa: BLE001 - the output path states the failure
        context = None
    project = context.project if context is not None else None
    return output_frame_jpeg(workspace, project_id, frame, tier, project=project)


def _geometry_payload(
    workspace: Workspace, project_id: str, frame: int
) -> tuple[dict[str, Any] | None, str]:
    """The export geometry for one frame, with the stamp it is valid under.

    ``None`` rather than an error: a project that has not been solved yet has no
    geometry, which is a state the viewer draws (no rectangle, no crosshairs)
    rather than a failure it reports.

    The stamp is the *plan's*, not the proxy tier's, because geometry moves when
    the smoothing or the render options move and the proxy frames do not. A
    client caching geometry by frame keys it on this.

    Everything but the frame's own rows comes from the cached
    :func:`~stable.server.frames.geometry_context`: this runs for every frame
    the export tab draws, and rebuilding the plan's inputs here — the solve,
    the smoothing, every target's held path — cost O(clip length) per frame.
    """
    try:
        context = geometry_context(workspace, project_id)
    except Exception:  # noqa: BLE001 - an unplannable project still gets its pixels
        return None, ""
    if context is None:
        return None, ""
    geometry = frame_geometry(
        context.project,
        context.plan,
        context.options,
        frame,
        context.motion,
        targets=context.targets,
    )
    return geometry, context.revision


def _pack(header: dict[str, Any], payload: bytes) -> bytes:
    """Frame one message: a 4-byte big-endian header length, the header, the JPEG.

    Binary rather than JSON-with-base64 because a half-res JPEG is ~50 kB and
    base64 would add a third to every one of them, at playback rate.
    """
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw + payload


async def _serve_frames(websocket: WebSocket, project_id: str, state: _StreamState) -> None:
    """Serve outstanding requests, newest first, one at a time, until cancelled.

    One at a time is deliberate: the work is a blocking decode/warp/encode, and
    running several would put the frame the user is waiting for behind threads
    doing work that is already obsolete.
    """
    workspace = get_workspace()
    while True:
        await state.signal.wait()
        state.signal.clear()
        while (request := state.take()) is not None:
            # Computed once per request and repeated on every reply to it —
            # including the failures — so a client waiting for "this frame,
            # whole" always gets an answer about the geometry too, whichever
            # picture of the frame turns out to be unavailable. A prefetch
            # carries no geometry *key at all*: it is pixels for later, and a
            # client must not be able to mistake it for the frame it is drawing.
            geometry: dict[str, Any] | None = None
            geometry_rev = ""
            if request.geometry:
                geometry, geometry_rev = await asyncio.to_thread(
                    _geometry_payload, workspace, project_id, request.frame
                )
            extra: dict[str, Any] = (
                {"geometry": geometry, "geometry_rev": geometry_rev} if request.geometry else {}
            )
            for variant in request.variants:
                try:
                    payload, revision = await asyncio.to_thread(
                        _frame_payload, workspace, project_id, request.frame, variant, request.tier
                    )
                except FrameUnavailable as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "seq": request.seq,
                            "frame": request.frame,
                            "variant": variant,
                            "status": exc.status,
                            "detail": str(exc),
                            **extra,
                        }
                    )
                    continue
                await websocket.send_bytes(
                    _pack(
                        {
                            "type": "frame",
                            "seq": request.seq,
                            "frame": request.frame,
                            "variant": variant,
                            "tier": request.tier,
                            "rev": revision,
                            **extra,
                        },
                        payload,
                    )
                )


@router.websocket("/ws/projects/{project_id}/frames")
async def frame_stream(websocket: WebSocket, project_id: str) -> None:
    """Stream proxy frames — and the export geometry drawn over them — on demand.

    Protocol, client to server, as JSON text::

        {"type": "want",     "seq": 41, "frame": 247,
         "variants": ["source", "output"], "tier": "half", "geometry": true}
        {"type": "prefetch", "frames": [248, 249], "variants": ["source"]}

    ``want`` is "this is the frame I am showing"; only the newest survives.
    ``prefetch`` is "I will probably want these next", served only while nothing
    is wanted, and capped at :data:`_PREFETCH_LIMIT` outstanding guesses.

    Server to client, as **binary**::

        [uint32 big-endian header length][header, UTF-8 JSON][JPEG bytes]

    The header carries ``frame``, ``variant``, ``seq``, ``rev`` — a short id for
    the inputs this picture depends on, so the client can throw away everything
    rendered under older settings — and, for a want, ``geometry``: the export's
    output rectangle, stabilisation warp and target residuals for **this** frame
    (:func:`stable.server.frames.frame_geometry`), or ``null`` before the
    project is solved.

    A frame that cannot be produced comes back as JSON text
    (``{"type": "error", …}``) with the status that would have been the HTTP
    one, so "not solved yet" reads as a state rather than as a hang.
    """
    await websocket.accept()
    state = _StreamState()
    worker = asyncio.create_task(_serve_frames(websocket, project_id, state))
    try:
        while True:
            text = await websocket.receive_text()
            if worker.done():  # the socket is gone, or serving raised
                break
            try:
                message = json.loads(text)
            except ValueError:
                continue  # malformed frames are ignored, not fatal
            if isinstance(message, dict):
                state.submit(message)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker
        with contextlib.suppress(Exception):
            await websocket.close()
