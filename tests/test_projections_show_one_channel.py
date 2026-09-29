"""A projection shows the left channel, not the whole frame.

"one more thing i am noticing in the Embryos tab images is that the images
are full frames, and not crops of the left channel... at least in the
projections"

The SPIM camera's chip carries two channels side by side. A frame read out
at full width (2048 x 512) holds both, and a projection of all of it was
mostly empty field: the embryo small in the left half, a dim copy of it in
the right.

This has been decided by the frame's shape before, twice, and was wrong both
ways. It is decided by the rig's settings here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gently.core import imaging
from gently.core.imaging import generate_jpeg_projection, shown_channel

SRC = Path(imaging.__file__).read_text(encoding="utf-8")

# The function itself, held before any test replaces the module's name for it.
SHOWN = imaging.shown_channel

Z, H, W = 6, 512, 2048


def frame(left: int = 0, right: int = 0, width: int = W) -> np.ndarray:
    """A volume with a block of ``left`` in its left half and ``right`` in its right."""
    vol = np.full((Z, H, width), 100, dtype=np.uint16)
    if left:
        vol[:, 200:300, width // 4 - 50 : width // 4 + 50] = left
    if right:
        vol[:, 200:300, 3 * width // 4 - 50 : 3 * width // 4 + 50] = right
    return vol


class TestWhichPart:
    def test_the_left_channel_of_a_full_readout(self):
        out = shown_channel(frame(left=4000, right=900), "left", 2048)
        assert out.shape == (Z, H, 1024)
        assert out.max() == 4000

    def test_the_right_channel_when_that_is_asked_for(self):
        out = shown_channel(frame(left=4000, right=900), "right", 2048)
        assert out.shape == (Z, H, 1024)
        assert out.max() == 900

    def test_the_whole_frame_when_that_is_asked_for(self):
        vol = frame(left=4000, right=900)
        assert shown_channel(vol, "both", 2048) is vol

    def test_a_frame_that_is_one_channel_already_is_never_divided(self):
        # 1024 x 512: twice as wide as tall, with the embryo in its middle.
        # Taking "the left half" of this cut the embryo in two.
        vol = np.full((Z, H, 1024), 100, dtype=np.uint16)
        vol[:, 200:300, 462:562] = 4000
        out = shown_channel(vol, "left", 2048)
        assert out is vol
        assert (out[:, 250, 462:562] == 4000).all()

    @pytest.mark.parametrize(
        "shape", [(Z, 512, 1024), (Z, 256, 1024), (Z, 100, 2000), (Z, 512, 512)]
    )
    def test_the_shape_of_a_frame_decides_nothing(self, shape):
        vol = np.zeros(shape, dtype=np.uint16)
        assert shown_channel(vol, "left", 2048) is vol

    def test_a_rig_with_another_chip_says_its_own_width(self):
        vol = np.zeros((Z, 512, 1024), dtype=np.uint16)
        assert shown_channel(vol, "left", 1024).shape == (Z, 512, 512)

    def test_a_single_frame_is_divided_as_a_volume_is(self):
        assert shown_channel(np.zeros((512, 2048), dtype=np.uint16), "left", 2048).shape == (
            512,
            1024,
        )

    def test_nothing_is_copied_or_changed(self):
        vol = frame(left=4000, right=900)
        before = vol.copy()
        out = shown_channel(vol, "left", 2048)
        assert np.shares_memory(out, vol)
        assert (vol == before).all()

    @pytest.mark.parametrize("view", ["", "centre", "LEFTISH", "a"])
    def test_a_setting_that_says_nothing_shows_the_frame_whole(self, view):
        vol = frame(left=4000)
        assert shown_channel(vol, view, 2048) is vol

    def test_case_and_space_are_forgiven(self):
        assert shown_channel(frame(), " Left ", 2048).shape[-1] == 1024

    def test_a_width_of_nothing_divides_nothing(self):
        vol = frame()
        assert shown_channel(vol, "left", 0) is vol


class TestTheSetting:
    def test_the_left_channel_is_what_is_shown(self):
        from gently.settings import UISettings

        assert UISettings().projection_view == "left"
        assert UISettings().spim_full_width == 2048

    def test_it_is_read_when_a_projection_is_drawn(self):
        fn = SRC[SRC.index("def shown_channel(") :][:2300]
        assert "settings.ui.projection_view" in fn and "settings.ui.spim_full_width" in fn

    def test_it_is_a_setting_of_the_rig(self):
        from gently.ui.web.settings_registry import SETTINGS

        by_key = {s.key: s for s in SETTINGS}
        view = by_key["microscope.projectionView"]
        assert view.store == "env:GENTLY_PROJECTION_VIEW" and view.source == "ui.projection_view"
        assert [c[0] for c in view.choices] == ["left", "right", "both"]
        assert by_key["microscope.spimFullWidth"].store == "env:GENTLY_SPIM_FULL_WIDTH"

    def test_no_frame_is_divided_by_its_shape(self):
        fn = SRC[SRC.index("def shown_channel(") : SRC.index("def generate_jpeg_projection(")]
        code = "\n".join(line for line in fn.splitlines() if not line.strip().startswith("#"))
        code = code[code.index('"""', code.index('"""') + 3) :]
        assert "shape[-2]" not in code and "height" not in code, (
            "the frame's height is not evidence"
        )


class TestTheProjection:
    def _draw(self, tmp_path, vol, view, monkeypatch):
        from PIL import Image

        monkeypatch.setattr(imaging, "shown_channel", lambda v: SHOWN(v, view, 2048))
        out = generate_jpeg_projection(vol, tmp_path / f"{view}.jpg")
        assert out is not None
        return np.asarray(Image.open(out))

    @staticmethod
    def _bright(img) -> float:
        """How much of the picture is bright. The lines between the three
        views are white and a pixel wide, so an empty field is not zero."""
        return float((img >= 128).mean())

    def test_it_is_drawn_from_one_channel(self, tmp_path, monkeypatch):
        vol = frame(left=4000, right=900)
        left = self._draw(tmp_path, vol, "left", monkeypatch)
        both = self._draw(tmp_path, vol, "both", monkeypatch)
        # The same height on screen, half the width of field: the embryo is
        # twice as large in the picture.
        assert left.shape[1] / left.shape[0] < both.shape[1] / both.shape[0]

    def test_the_embryo_is_in_it(self, tmp_path, monkeypatch):
        img = self._draw(tmp_path, frame(left=4000), "left", monkeypatch)
        assert self._bright(img) > 0.01

    def test_the_other_channel_is_not(self, tmp_path, monkeypatch):
        # Only the right half holds anything. Drawn from the left channel it
        # is the picture of an empty field, pixel for pixel.
        img = self._draw(tmp_path, frame(right=4000), "left", monkeypatch)
        empty = self._draw(tmp_path, frame(), "left", monkeypatch)
        assert img.shape == empty.shape and (img == empty).all()
        with_it = self._draw(tmp_path, frame(right=4000), "both", monkeypatch)
        assert self._bright(with_it) > self._bright(img), "the whole frame should show the block"

    def test_the_volume_handed_in_is_left_whole(self, tmp_path, monkeypatch):
        vol = frame(left=4000, right=900)
        before = vol.copy()
        self._draw(tmp_path, vol, "left", monkeypatch)
        assert vol.shape == (Z, H, W) and (vol == before).all()

    def test_the_store_keeps_the_whole_frame(self, tmp_path):
        from gently.core.file_store import FileStore

        store = FileStore(root=tmp_path)
        store.create_session("s1")
        store.register_embryo("s1", "embryo_1", position_coarse={"x": 1.0, "y": 2.0}, role="test")
        vol = frame(left=4000, right=900)
        store.put_volume("s1", "embryo_1", 0, vol)
        import tifffile

        kept = tifffile.imread(str(store.get_volume_path("s1", "embryo_1", 0)))
        assert kept.shape == (Z, H, W), "the volume on disk must be the whole frame"
        assert kept[:, 250, 3 * W // 4].max() == 900, "the right channel must still be in it"
        assert store.get_projection_path("s1", "embryo_1", 0) is not None
