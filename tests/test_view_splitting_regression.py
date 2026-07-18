"""
Regression tests for the aspect-ratio view-splitting bug (XY-rendered-halfway).

Several code paths used to guess "dual-view format" from the frame aspect ratio
(``width > height * 2``, or ``* 1.5``) and keep only the left half. Native SPIM
frames on this rig are 2048x512 (4:1), so the guess fired on *every* real frame
and silently discarded the right half of the image.

View selection is now driven by array rank alone: an explicit 4D
(Views, Z, Y, X) volume selects View A via ``[0]``; a 3D volume is by
construction a single view and is kept whole.

Tests cover:
- load_volume keeps full width for a 4:1 single-view TIFF
- load_volume still selects View A for a genuine 4D volume
- generate_jpeg_projection (the already-correct reference path)
- VisualizationServer._array_to_image_data preserves right-half signal
- TimelapseOrchestrator._volume_to_b64 returns an un-truncated view_a
- EmbryoDataset._load_projection_from_volume preserves right-half signal

Strategy: build a 2048x512 volume whose ONLY signal is a bright block in the
RIGHT half (x=1600..1700). If a path truncates to the left half, what remains
is a uniform background, so the rendered projection collapses to near-zero
variance. Surviving signal => the marker was preserved.
"""

import base64
import io

import numpy as np
import pytest

BACKGROUND = 100
MARKER = 60000
MARKER_X0, MARKER_X1 = 1600, 1700
MARKER_Y0, MARKER_Y1 = 200, 300
WIDTH, HEIGHT, DEPTH = 2048, 512, 8


def make_right_half_marked_volume() -> np.ndarray:
    """(Z, Y, X) = (8, 512, 2048) uint16 volume, bright block only in right half."""
    vol = np.full((DEPTH, HEIGHT, WIDTH), BACKGROUND, dtype=np.uint16)
    vol[:, MARKER_Y0:MARKER_Y1, MARKER_X0:MARKER_X1] = MARKER
    return vol


def decode_to_array(b64: str) -> np.ndarray:
    """Decode a base64 image string (data-URI prefix tolerated) to a numpy array."""
    from PIL import Image

    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def assert_signal_survived(arr: np.ndarray, path_name: str) -> None:
    """A rendered projection must retain contrast from the right-half marker."""
    assert arr.size > 0, f"{path_name}: produced an empty image"
    spread = float(arr.max()) - float(arr.min())
    assert spread > 10, (
        f"{path_name}: rendered projection is nearly uniform "
        f"(min={arr.min()}, max={arr.max()}). The right-half marker was "
        f"discarded — aspect-ratio view splitting has regressed."
    )


@pytest.fixture
def stub_tifffile(monkeypatch):
    """Install a stub TIFF reader returning a caller-chosen array.

    tifffile is not a hard dependency of the test environment, and the real
    library is irrelevant to what is under test here (the post-read view
    handling). The fixture yields a setter for the array `imread` returns.
    """
    import sys
    import types

    holder: dict[str, np.ndarray] = {}

    stub = types.ModuleType("tifffile")
    stub.imread = lambda _path: holder["array"]  # type: ignore[attr-defined]
    stub.imwrite = lambda *_a, **_k: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tifffile", stub)

    import gently.core.imaging as imaging_mod

    monkeypatch.setattr(imaging_mod, "_tifffile", stub, raising=False)

    def set_array(arr: np.ndarray) -> None:
        holder["array"] = arr

    return set_array


class TestLoadVolume:
    def test_full_width_preserved_for_4to1_single_view(self, tmp_path, stub_tifffile):
        """A 4:1 3D volume must load at full width, not halved."""
        stub_tifffile(make_right_half_marked_volume())

        from gently.core.imaging import load_volume

        vol = load_volume(tmp_path / "vol.tif")

        assert vol.shape[-1] == WIDTH, (
            f"load_volume truncated width to {vol.shape[-1]} (expected {WIDTH})"
        )
        assert vol[:, MARKER_Y0:MARKER_Y1, MARKER_X0:MARKER_X1].max() == MARKER, (
            "right-half marker missing after load_volume"
        )

    def test_view_a_selected_for_genuine_4d_volume(self, tmp_path, stub_tifffile):
        """An explicit (Views, Z, Y, X) volume still resolves to View A."""
        view_a = make_right_half_marked_volume()
        view_b = np.full_like(view_a, 7)
        stub_tifffile(np.stack([view_a, view_b]))

        from gently.core.imaging import load_volume

        vol = load_volume(tmp_path / "dual.tif")

        assert vol.shape == (DEPTH, HEIGHT, WIDTH), f"unexpected shape {vol.shape}"
        assert vol.max() == MARKER, "View A was not the selected view"


class TestGenerateJpegProjection:
    """The already-correct reference path — guards against re-introduction."""

    def test_right_half_marker_survives(self, tmp_path):
        pytest.importorskip("PIL")
        from gently.core.imaging import generate_jpeg_projection

        out = tmp_path / "proj.jpg"
        result = generate_jpeg_projection(make_right_half_marked_volume(), out)

        assert result is not None, "generate_jpeg_projection returned None"
        from PIL import Image

        assert_signal_survived(np.asarray(Image.open(result)), "generate_jpeg_projection")


@pytest.fixture
def crop_bounds_spy(monkeypatch):
    """Record the array handed to compute_crop_bounds.

    Rendering-level assertions are unreliable for the three-view paths: the
    layout composes separator/padding regions that create contrast even when
    the image content is uniform, so a truncated frame can still yield a
    high-contrast PNG. Observing the array at the crop step is direct.
    """
    import gently.core.imaging as imaging_mod

    seen: list[np.ndarray] = []
    real = imaging_mod.compute_crop_bounds

    def spy(volume, *args, **kwargs):
        seen.append(volume)
        return real(volume, *args, **kwargs)

    monkeypatch.setattr(imaging_mod, "compute_crop_bounds", spy)
    return seen


class TestVisualizationServerArrayToImageData:
    def test_full_width_reaches_crop_step(self, crop_bounds_spy):
        pytest.importorskip("PIL")
        from gently.ui.web.server import VisualizationServer

        # _array_to_image_data does not touch `self`; call it unbound.
        image_data = VisualizationServer._array_to_image_data(
            None, make_right_half_marked_volume(), "test-uid", "volume_projection"
        )

        assert image_data.base64_png, "_array_to_image_data produced no PNG"
        assert crop_bounds_spy, "compute_crop_bounds was never called"
        assert crop_bounds_spy[0].shape[-1] == WIDTH, (
            f"_array_to_image_data truncated the volume to width "
            f"{crop_bounds_spy[0].shape[-1]} (expected {WIDTH})"
        )
        assert crop_bounds_spy[0][:, MARKER_Y0:MARKER_Y1, MARKER_X0:MARKER_X1].max() == MARKER, (
            "right-half marker missing before crop"
        )


class TestTimelapseVolumeToB64:
    def test_view_a_not_truncated(self):
        pytest.importorskip("PIL")
        from gently.app.orchestration.timelapse import TimelapseOrchestrator

        # _volume_to_b64 does not touch `self`; call it unbound.
        view_a, image_b64 = TimelapseOrchestrator._volume_to_b64(
            None, make_right_half_marked_volume()
        )

        assert view_a is not None and image_b64 is not None
        assert view_a.shape[-1] == WIDTH, (
            f"_volume_to_b64 returned a truncated view_a (width {view_a.shape[-1]}); "
            "this value feeds the perceiver"
        )
        assert_signal_survived(decode_to_array(image_b64), "TimelapseOrchestrator._volume_to_b64")


class TestEmbryoDatasetProjection:
    def test_right_half_marker_survives(self, tmp_path, stub_tifffile):
        pytest.importorskip("PIL")
        from gently.dataset.embryo_dataset import EmbryoDataset

        stub_tifffile(make_right_half_marked_volume())

        # _load_projection_from_volume does not touch `self`; call it unbound.
        b64 = EmbryoDataset._load_projection_from_volume(None, str(tmp_path / "vol.tif"))

        assert_signal_survived(decode_to_array(b64), "EmbryoDataset._load_projection_from_volume")
