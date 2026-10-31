"""Where the uint16 pipeline runs out: the server's cost per frame, the wire's
throughput, and (from the browser, see the notes at the end) the draw.

    uv run python tools/u16_bench.py                       # server-side, synthetic frames
    uv run python tools/u16_bench.py --url http://localhost:8080/api/sessions/<sid>/snapshot/<stem>.u16

Server side: a 2048² uint16 frame as a TIFF on disk, timed through the same
functions the routes use: read, block mean at 1/2/4, bytes. Wire: the given
.u16 URL fetched repeatedly at scale 1, 2, 4 with 1, 4 and 8 clients at
once; frames per second and MB per second.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def server_side(size: int, reps: int) -> None:
    import numpy as np
    import tifffile

    from gently.core.u16 import u16_payload

    rng = np.random.default_rng(0)
    frame = (rng.normal(2000, 300, (size, size)).clip(0, 65535)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "frame.tif"
        tifffile.imwrite(str(path), frame)
        print(f"server side, {size}×{size} uint16 ({frame.nbytes / 1e6:.1f} MB), median of {reps}:")
        t = [timed(lambda: tifffile.imread(str(path))) for _ in range(reps)]
        print(f"  read TIFF            {statistics.median(t):7.1f} ms")
        for k in (1, 2, 4, 8):
            t = [timed(lambda k=k: u16_payload(frame, k)) for _ in range(reps)]
            body, _ = u16_payload(frame, k)
            mb = len(body) / 1e6
            print(f"  payload scale {k}      {statistics.median(t):7.1f} ms  → {mb:5.2f} MB")
        from gently.core.imaging import normalize_to_uint8

        t = [timed(lambda: normalize_to_uint8(frame)) for _ in range(reps)]
        print(f"  (old) percentile→u8  {statistics.median(t):7.1f} ms, before the PNG encode")


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def wire(url: str, seconds: float) -> None:
    def one(u: str) -> int:
        with urllib.request.urlopen(u, timeout=30) as r:
            return len(r.read())

    print(f"wire, {url}, {seconds:.0f} s per cell:")
    print("  scale  clients   frames/s   MB/s   ms/frame(median)")
    for scale in (1, 2, 4):
        u = f"{url}{'&' if '?' in url else '?'}scale={scale}"
        size = one(u)
        for clients in (1, 4, 8):
            times: list[float] = []
            stop = time.perf_counter() + seconds
            n = 0

            def worker(u: str = u, stop: float = stop, times: list[float] = times) -> None:
                nonlocal n
                while time.perf_counter() < stop:
                    t0 = time.perf_counter()
                    one(u)
                    times.append((time.perf_counter() - t0) * 1000)
                    n += 1

            with ThreadPoolExecutor(max_workers=clients) as ex:
                for _ in range(clients):
                    ex.submit(worker)
            fps = n / seconds
            mbps = fps * size / 1e6
            med = statistics.median(times)
            print(f"  {scale:<6} {clients:<8} {fps:8.1f}  {mbps:6.1f}   {med:7.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", type=int, default=2048, help="frame side for the server-side part")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--url", help="a .u16 URL on a running server, for the wire part")
    ap.add_argument("--seconds", type=float, default=3.0, help="per wire cell")
    args = ap.parse_args()
    server_side(args.size, args.reps)
    if args.url:
        wire(args.url, args.seconds)
    print(
        "\nbrowser side: open a frame in the viewer and read view._lastLoadMs (fetch+decode)\n"
        "and view._lastDrawMs (LUT+draw) from the console; the Playwright run in the PR did that."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
