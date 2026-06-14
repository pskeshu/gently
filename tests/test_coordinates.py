"""
Tests for coordinates — pixel/stage coordinate transformations.

Tests cover:
- um_per_pixel calculation
- pixel_to_stage_position basic conversion
- stage_to_pixel_position basic conversion
- Round-trip pixel→stage→pixel consistency
- pixel_displacement_to_stage_movement
- Zero displacement edge case
- Center pixel maps to current stage position
- Negative displacement handling
"""

import pytest

from gently.core.coordinates import (
    DEFAULT_OBJECTIVE_MAG,
    DEFAULT_PIXEL_SIZE_UM,
    get_um_per_pixel,
    pixel_displacement_to_stage_movement,
    pixel_to_stage_position,
    stage_to_pixel_position,
)


class TestUmPerPixel:
    def test_um_per_pixel_defaults(self):
        result = get_um_per_pixel()
        assert result == DEFAULT_PIXEL_SIZE_UM / DEFAULT_OBJECTIVE_MAG
        assert result == pytest.approx(0.65)

    def test_um_per_pixel_custom(self):
        result = get_um_per_pixel(pixel_size_um=13.0, objective_mag=20.0)
        assert result == pytest.approx(0.65)


class TestPixelToStage:
    def test_pixel_to_stage_basic(self):
        # Embryo 100px right of center → stage X increases
        sx, sy = pixel_to_stage_position(
            pixel_x=612,
            pixel_y=512,
            image_center_x=512,
            image_center_y=512,
            stage_x=1000.0,
            stage_y=2000.0,
            um_per_pixel=1.0,
        )
        assert sx == pytest.approx(1100.0)
        # Y is inverted
        assert sy == pytest.approx(2000.0)

    def test_center_pixel_maps_to_stage_origin(self):
        # Embryo at image center → stage position unchanged
        sx, sy = pixel_to_stage_position(
            pixel_x=512,
            pixel_y=512,
            image_center_x=512,
            image_center_y=512,
            stage_x=1000.0,
            stage_y=2000.0,
            um_per_pixel=1.0,
        )
        assert sx == pytest.approx(1000.0)
        assert sy == pytest.approx(2000.0)


class TestStageToPixel:
    def test_stage_to_pixel_basic(self):
        # Embryo 100um right of current stage → pixel right of center
        px, py = stage_to_pixel_position(
            stage_x=1100.0,
            stage_y=2000.0,
            current_stage_x=1000.0,
            current_stage_y=2000.0,
            image_center_x=512,
            image_center_y=512,
            um_per_pixel=1.0,
        )
        assert px == pytest.approx(612.0)
        assert py == pytest.approx(512.0)


class TestRoundTrip:
    def test_round_trip_pixel_stage(self):
        """pixel→stage→pixel should recover the original pixel coordinates."""
        um_pp = 1.625
        cx, cy = 512.0, 512.0
        stg_x, stg_y = 5000.0, 3000.0
        orig_px, orig_py = 300.0, 700.0

        # Forward: pixel → stage
        embryo_sx, embryo_sy = pixel_to_stage_position(
            orig_px,
            orig_py,
            cx,
            cy,
            stg_x,
            stg_y,
            um_per_pixel=um_pp,
        )

        # Reverse: stage → pixel (same capture position)
        recovered_px, recovered_py = stage_to_pixel_position(
            embryo_sx,
            embryo_sy,
            stg_x,
            stg_y,
            cx,
            cy,
            um_per_pixel=um_pp,
        )

        assert recovered_px == pytest.approx(orig_px)
        assert recovered_py == pytest.approx(orig_py)


class TestPixelDisplacement:
    def test_pixel_displacement_to_stage(self):
        # Move embryo 100px to the right visually → stage should move -X
        dx, dy = pixel_displacement_to_stage_movement(100.0, 0.0, um_per_pixel=1.0)
        assert dx == pytest.approx(-100.0)  # X is inverted for movement
        assert dy == pytest.approx(0.0)

    def test_zero_displacement(self):
        dx, dy = pixel_displacement_to_stage_movement(0.0, 0.0, um_per_pixel=1.0)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_negative_displacement(self):
        dx, dy = pixel_displacement_to_stage_movement(-50.0, 30.0, um_per_pixel=2.0)
        assert dx == pytest.approx(100.0)  # inverted: -(-50)*2
        assert dy == pytest.approx(60.0)  # not inverted: 30*2
