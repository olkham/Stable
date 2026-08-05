"""Measure the video pipeline's throughput, with each optimisation on and off.

Todo 6 ("it barely uses 25% of the CPU and 0% of the GPU") is a performance
claim, so it needs numbers rather than an assurance. This script produces them
for the three things :mod:`stable.video` actually changed, by running each one
against its own disabled setting on the same clip in the same process:

``proxy``
    Proxy generation with one encoding thread (the original loop) against the
    default pool. Decode is the same in both; what varies is whether resizing
    and JPEG-encoding overlap it.

``decode``
    The multi-run access pattern a real track has — several passes over the same
    span from different anchors — through a private-cache
    :class:`~stable.video.source.PyAVVideoSource` against a
    :class:`~stable.video.pool.PooledVideoSource` with a shared cache and
    read-ahead.

``concurrent``
    Several tracking runs at once, which is what the job pool now allows. One
    shared :class:`~stable.video.source.PyAVVideoSource` serialises them all on
    its decode lock *and* drags a single cursor back and forth between them;
    a pool gives each thread its own decoder over the shared cache. This is the
    case pooling exists for.

``scrub``
    Scattered single-frame reads, the UI's pattern. Included because it is the
    case read-ahead must *not* make worse: a viewer jumping around the timeline
    should not be paying for prefetches it never uses.

Usage::

    python scripts/benchmark_video.py                 # synthesise a 1080p clip
    python scripts/benchmark_video.py --video CLIP    # measure a real one
    python scripts/benchmark_video.py --frames 480 --only decode
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable.core.video import SyntheticVideoSource  # noqa: E402
from stable.video.pool import PooledVideoSource  # noqa: E402
from stable.video.proxy import default_proxy_workers, generate_proxies  # noqa: E402
from stable.video.source import PyAVVideoSource  # noqa: E402


def synthesise(path: Path, *, frames: int, width: int, height: int) -> None:
    """Encode a throwaway H.264 clip to measure against."""
    import av

    source = SyntheticVideoSource(frame_count=frames, width=width, height=height)
    output = av.open(str(path), mode="w")
    try:
        stream = output.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        for index, frame in source.iter_frames():
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = index
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    finally:
        output.close()


@contextmanager
def scratch_dir() -> Iterator[Path]:
    """A temporary directory removed on the way out."""
    directory = Path(tempfile.mkdtemp(prefix="stable-bench-"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def timed(fn: Callable[[], int], *, repeats: int) -> tuple[float, int]:
    """Run ``fn`` ``repeats`` times, returning the best wall time and its frame count.

    The *best* run, not the mean: this is measuring how fast the code can go, and
    the slow runs are measuring whatever else the machine was doing.
    """
    times: list[float] = []
    processed = 0
    for _ in range(repeats):
        start = time.perf_counter()
        processed = fn()
        times.append(time.perf_counter() - start)
    return min(times), processed


def report(label: str, before: tuple[float, int], after: tuple[float, int]) -> None:
    """Print one before/after comparison as a rate and a speed-up."""
    before_time, frames = before
    after_time, _ = after
    before_rate = frames / before_time if before_time else 0.0
    after_rate = frames / after_time if after_time else 0.0
    speedup = before_time / after_time if after_time else float("nan")
    print(f"\n{label}")
    print(f"  before : {before_time:7.2f} s  ({before_rate:8.1f} frames/s)")
    print(f"  after  : {after_time:7.2f} s  ({after_rate:8.1f} frames/s)")
    print(f"  change : {speedup:.2f}x")


# -- the three measurements ---------------------------------------------------


def bench_proxy(video_path: Path, repeats: int) -> None:
    """Proxy generation: one encoding thread against the default pool."""

    def run(workers: int) -> Callable[[], int]:
        def once() -> int:
            with scratch_dir() as out, PyAVVideoSource(video_path) as source:
                generate_proxies(source, out, scale=0.5, workers=workers)
                return source.frame_count

        return once

    before = timed(run(1), repeats=repeats)
    after = timed(run(default_proxy_workers()), repeats=repeats)
    report(f"proxy generation (1 encode thread -> {default_proxy_workers()})", before, after)


def _multi_run_pattern(open_source: Callable[[], object], spans: list[tuple[int, int]]) -> int:
    """Read several overlapping spans, the way a multi-anchor track does."""
    source = open_source()
    read = 0
    try:
        for start, stop in spans:
            step = 1 if stop >= start else -1
            for index in range(start, stop, step):
                source.get_frame_view(index)  # type: ignore[attr-defined]
                read += 1
    finally:
        source.close()  # type: ignore[attr-defined]
    return read


def bench_decode(video_path: Path, frames: int, repeats: int) -> None:
    """The access pattern of a three-anchor track: six overlapping runs."""
    last = frames - 1
    mid = frames // 2
    spans = [
        (0, last),  # anchor 0 forwards
        (mid, 0),  # anchor mid backwards
        (mid, last),  # anchor mid forwards
        (last, mid),  # anchor last backwards
        (0, mid),  # anchor 0 forwards again
        (last, 0),  # a full backward sweep
    ]

    def private_cache() -> object:
        # One decoder and a private 64-frame cache -- `PyAVVideoSource`'s own
        # defaults, which is exactly what the server handed every job before
        # pooling existed. Deliberately not a smaller cache: that would make the
        # "before" number worse than the code ever actually was.
        return PyAVVideoSource(video_path)

    def pooled() -> object:
        return PooledVideoSource(video_path)

    before = timed(lambda: _multi_run_pattern(private_cache, spans), repeats=repeats)
    after = timed(lambda: _multi_run_pattern(pooled, spans), repeats=repeats)
    report("multi-run decode (private cache -> pooled + read-ahead)", before, after)


def bench_concurrent(video_path: Path, frames: int, repeats: int, workers: int) -> None:
    """Several runs at once: one shared decoder against a pool of them."""
    from concurrent.futures import ThreadPoolExecutor

    # Each worker sweeps its own quarter of the clip, then the whole thing --
    # overlapping enough that a shared cache pays off, spread out enough that a
    # single cursor has to seek between them.
    def spans_for(worker: int) -> list[tuple[int, int]]:
        quarter = max(1, frames // workers)
        start = min(worker * quarter, frames - 1)
        return [(start, min(start + quarter, frames - 1)), (0, frames - 1)]

    def read(source: Any, spans: list[tuple[int, int]]) -> int:
        count = 0
        for start, stop in spans:
            step = 1 if stop >= start else -1
            for index in range(start, stop, step):
                source.get_frame_view(index)
                count += 1
        return count

    def once(shared: bool) -> Callable[[], int]:
        def run() -> int:
            source = PyAVVideoSource(video_path) if shared else PooledVideoSource(video_path)
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(read, source, spans_for(worker)) for worker in range(workers)
                    ]
                    return sum(future.result() for future in futures)
            finally:
                source.close()

        return run

    before = timed(once(shared=True), repeats=repeats)
    after = timed(once(shared=False), repeats=repeats)
    report(f"{workers} concurrent runs (one shared decoder -> pool)", before, after)


def bench_scrub(video_path: Path, frames: int, repeats: int) -> None:
    """Scattered reads: the case read-ahead must not slow down."""
    # Deterministic and deliberately non-sequential, so read-ahead never helps.
    targets = [(index * 37) % frames for index in range(min(frames, 120))]

    def once(open_source: Callable[[], object]) -> Callable[[], int]:
        def run() -> int:
            source = open_source()
            try:
                for index in targets:
                    source.get_frame_view(index)  # type: ignore[attr-defined]
                return len(targets)
            finally:
                source.close()  # type: ignore[attr-defined]

        return run

    before = timed(once(lambda: PyAVVideoSource(video_path)), repeats=repeats)
    after = timed(once(lambda: PooledVideoSource(video_path)), repeats=repeats)
    report("scattered scrub (single source -> pooled)", before, after)


def main() -> int:
    """Parse arguments, prepare a clip, and run the requested benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, help="measure against this clip instead")
    parser.add_argument("--frames", type=int, default=240, help="synthetic clip length")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=2, help="runs per measurement (best wins)")
    parser.add_argument(
        "--only",
        choices=("proxy", "decode", "concurrent", "scrub"),
        action="append",
        help="run only these benchmarks (repeatable)",
    )
    parser.add_argument("--workers", type=int, default=4, help="threads for the concurrent case")
    args = parser.parse_args()
    wanted = set(args.only or ("proxy", "decode", "concurrent", "scrub"))

    with scratch_dir() as workdir:
        if args.video:
            video_path = args.video
            if not video_path.is_file():
                print(f"error: no such file: {video_path}", file=sys.stderr)
                return 1
        else:
            video_path = workdir / "bench.mp4"
            print(f"encoding {args.frames} frames at {args.width}x{args.height}...")
            synthesise(video_path, frames=args.frames, width=args.width, height=args.height)

        with PyAVVideoSource(video_path) as probe:
            frames, width, height = probe.frame_count, probe.width, probe.height
        print(f"clip: {video_path.name}  {width}x{height}  {frames} frames")
        print(f"cpus: {os.cpu_count()}  repeats: {args.repeats}")

        if "proxy" in wanted:
            bench_proxy(video_path, args.repeats)
        if "decode" in wanted:
            bench_decode(video_path, frames, args.repeats)
        if "concurrent" in wanted:
            bench_concurrent(video_path, frames, args.repeats, max(2, args.workers))
        if "scrub" in wanted:
            bench_scrub(video_path, frames, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
