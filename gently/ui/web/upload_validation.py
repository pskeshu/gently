"""Validation helpers for HTTP array upload routes."""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Iterable

import numpy as np
from fastapi import HTTPException


def decode_array_payload(
    encoded: str,
    shape: Iterable[int],
    dtype_name: str,
    *,
    max_nbytes: int,
    label: str,
) -> np.ndarray:
    """Decode a base64 array after validating shape, dtype, and byte count.

    Guards the raw ``np.frombuffer(b64decode(...)).reshape(shape)`` path against
    attacker-controlled input: bounds the dimension count, forbids object
    dtypes, caps the decoded size *before* allocating, and requires the decoded
    byte length to match shape x dtype exactly.
    """
    if not isinstance(encoded, str) or not encoded:
        raise HTTPException(status_code=400, detail=f"{label} payload must be base64 text")
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Iterable):
        raise HTTPException(status_code=400, detail=f"{label} shape must be a list of dimensions")

    try:
        shape_tuple = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail=f"{label} shape must contain integers"
        ) from None
    if not shape_tuple or len(shape_tuple) > 4 or any(dim <= 0 for dim in shape_tuple):
        raise HTTPException(
            status_code=400, detail=f"{label} shape must have 1-4 positive dimensions"
        )

    try:
        dtype = np.dtype(dtype_name)
    except TypeError:
        raise HTTPException(status_code=400, detail=f"{label} dtype is not supported") from None
    if dtype.hasobject:
        raise HTTPException(status_code=400, detail=f"{label} dtype may not contain Python objects")

    expected_nbytes = math.prod(shape_tuple) * dtype.itemsize
    if expected_nbytes > max_nbytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} payload is too large ({expected_nbytes} bytes > {max_nbytes} bytes)",
        )

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400, detail=f"{label} payload is not valid base64"
        ) from None
    if len(raw) != expected_nbytes:
        raise HTTPException(
            status_code=400,
            detail=f"{label} byte length {len(raw)} does not match shape/dtype {expected_nbytes}",
        )

    return np.frombuffer(raw, dtype=dtype).reshape(shape_tuple)
