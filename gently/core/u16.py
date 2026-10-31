"""Pixels as they are: a frame as little-endian uint16 bytes for the browser.

Every image gently showed until now left the server as 8-bit PNG or JPEG,
stretched by a percentile the browser never learned, so there was nothing
exact to read under a cursor and no contrast to move without a round trip.
This is the other way: the array itself, optionally averaged down by an
integer factor, with its shape and range in headers. The browser draws it
with its own lookup table and reads values from the bytes it holds.

The format is deliberately plain. Little-endian uint16, row-major, no
container; a 2048² frame is 8 MB, a 1024² one 2 MB, which the loopback and
a local browser carry at many frames a second (tools/u16_bench.py measures
it). A different dtype is clipped into uint16's range, which loses nothing
for the cameras here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MEDIA_TYPE = "application/octet-stream"
U16_MAX = 65535


def block_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Average k×k blocks; the trailing rows and columns that do not fill a
    block are dropped. k of 1 or less is the array itself. An unsigned
    source comes back as rounded whole numbers (uint32, for as_u16 to clip);
    anything else as float32."""
    if k <= 1 or arr.ndim != 2:
        return arr
    h, w = arr.shape
    hh, ww = (h // k) * k, (w // k) * k
    if hh == 0 or ww == 0:
        return arr
    # k² strided adds into one accumulator: 7 ms for a 2048² frame at k=2,
    # where reshape-and-mean in float32 took 32 (tools/u16_bench.py).
    whole = arr.dtype.kind == "u"
    acc: np.ndarray = np.zeros((hh // k, ww // k), np.uint32 if whole else np.float32)
    for i in range(k):
        for j in range(k):
            acc += arr[i:hh:k, j:ww:k]
    n = k * k
    return (acc + n // 2) // n if whole else acc / n


def as_u16(arr: np.ndarray) -> np.ndarray:
    """The array as little-endian uint16, clipped into range, rounded if it
    was float (a float source's block mean), contiguous so tobytes() is one copy."""
    if arr.dtype == np.dtype("<u2") and arr.flags.c_contiguous:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return np.ascontiguousarray(np.clip(arr, 0, U16_MAX).astype("<u2"))


def first_plane(arr: np.ndarray) -> np.ndarray:
    """A 2D frame out of whatever a TIFF held: a stack or a colour plane
    gives its first 2D frame."""
    if arr.ndim > 2:
        arr = arr.reshape(-1, *arr.shape[-2:])[0]
    return arr


def u16_payload(
    arr: np.ndarray,
    scale: int = 1,
    extra: dict[str, Any] | None = None,
    max_dim: int | None = None,
) -> tuple[bytes, dict[str, str]]:
    """The bytes and the headers for one frame. ``scale`` averages blocks
    of that size down first; ``max_dim`` asks instead for the smallest
    integer scale that brings the longer side to at most that many pixels
    (a thumbnail says the size it has, not a factor); the larger of the two
    wins. The headers carry both the sent shape and the source shape, so a
    cursor position can be mapped back."""
    src = first_plane(np.asarray(arr))
    scale = max(1, int(scale))
    if max_dim and max_dim > 0 and src.ndim == 2 and src.size:
        scale = max(scale, -(-max(src.shape) // int(max_dim)))
    out = as_u16(block_mean(src, scale))
    headers = {
        "X-Image-Width": str(out.shape[1]),
        "X-Image-Height": str(out.shape[0]),
        "X-Image-Dtype": "uint16",
        "X-Image-Scale": str(scale),
        "X-Image-Source-Width": str(src.shape[1]),
        "X-Image-Source-Height": str(src.shape[0]),
        "X-Image-Min": str(int(out.min())) if out.size else "0",
        "X-Image-Max": str(int(out.max())) if out.size else "0",
    }
    for k, v in (extra or {}).items():
        if v is not None:
            headers[f"X-Image-{k}"] = str(v)
    return out.tobytes(), headers
