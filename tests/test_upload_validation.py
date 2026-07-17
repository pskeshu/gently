"""Regression tests for the HTTP array-upload validator (see PR #24).

Guards the raw ``np.frombuffer(b64decode(...)).reshape(shape)`` path on the
image/volume upload routes against attacker-controlled shape/dtype/payload.
"""

import base64

import numpy as np
import pytest
from fastapi import HTTPException

from gently.ui.web.upload_validation import decode_array_payload

MAX = 1 << 20  # 1 MiB


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode()


def test_valid_payload_decodes():
    arr = np.arange(4, dtype=np.uint8).reshape(2, 2)
    out = decode_array_payload(_b64(arr), [2, 2], "uint8", max_nbytes=MAX, label="image")
    assert out.shape == (2, 2)
    assert out.dtype == np.uint8
    assert out.tolist() == arr.tolist()


def test_oversized_shape_rejected_before_alloc():
    # shape * itemsize exceeds the cap -> 413, no allocation attempted
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("AAAA", [100_000, 100_000], "uint8", max_nbytes=MAX, label="image")
    assert ei.value.status_code == 413


def test_object_dtype_rejected():
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("AAAA", [2, 2], "O", max_nbytes=MAX, label="image")
    assert ei.value.status_code == 400


def test_too_many_dimensions_rejected():
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("AAAA", [1, 1, 1, 1, 1], "uint8", max_nbytes=MAX, label="image")
    assert ei.value.status_code == 400


def test_nonpositive_dimension_rejected():
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("AAAA", [2, 0], "uint8", max_nbytes=MAX, label="image")
    assert ei.value.status_code == 400


def test_byte_length_mismatch_rejected():
    arr = np.zeros((2, 2), dtype=np.uint8)  # 4 bytes
    with pytest.raises(HTTPException) as ei:
        decode_array_payload(_b64(arr), [3, 3], "uint8", max_nbytes=MAX, label="image")
    assert ei.value.status_code == 400


def test_invalid_base64_rejected():
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("!!!not-base64!!!", [2, 2], "uint8", max_nbytes=MAX, label="image")
    assert ei.value.status_code == 400


def test_non_iterable_shape_rejected():
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("AAAA", 4, "uint8", max_nbytes=MAX, label="image")  # type: ignore[arg-type]
    assert ei.value.status_code == 400


def test_string_shape_rejected():
    # a str is iterable but is not a valid shape
    with pytest.raises(HTTPException) as ei:
        decode_array_payload("AAAA", "22", "uint8", max_nbytes=MAX, label="image")  # type: ignore[arg-type]
    assert ei.value.status_code == 400
