"""
Tests for analysis.core — focus scoring and curve fitting.

Tests cover:
- Volath focus score returns positive value
- Gradient focus score higher for sharp edges
- FFT bandpass focus score
- Gaussian fit finds known peak
- Parabolic fit finds known peak
- AdaptiveSweepState detects peak via decline
- create_focus_montage output shape
- analyze_focus_stack end-to-end
"""

import numpy as np
import pytest

from gently.analysis.core import (
    AdaptiveSweepState,
    FitFunction,
    FocusAlgorithm,
    FocusAnalysisConfig,
    analyze_focus_stack,
    calculate_focus_score,
    create_focus_montage,
    fit_focus_curve,
)


def _make_focused_image(size=128):
    """Create a synthetic image with sharp edges (high focus)."""
    img = np.zeros((size, size), dtype=np.float64)
    # Sharp square in center
    q = size // 4
    img[q : 3 * q, q : 3 * q] = 200.0
    return img


def _make_blurry_image(size=128):
    """Create a synthetic blurry image (low focus)."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(_make_focused_image(size), sigma=10.0)


def _make_gaussian_focus_stack(n_positions=20, peak_pos=50.0):
    """Create synthetic focus stack with Gaussian focus profile."""
    positions = np.linspace(30.0, 70.0, n_positions)
    # Generate images whose variance follows a Gaussian around peak_pos
    images = []
    for pos in positions:
        # Scale brightness by Gaussian envelope so focus score peaks at peak_pos
        scale = np.exp(-((pos - peak_pos) ** 2) / (2 * 5.0**2))
        img = _make_focused_image() * scale + np.random.normal(0, 1, (128, 128))
        img = np.clip(img, 0, 255)
        images.append(img)
    return positions.tolist(), images


# =========================================================================
# Focus scoring
# =========================================================================


class TestFocusScoring:
    def test_volath_focus_score_positive(self):
        img = _make_focused_image()
        score = calculate_focus_score(img, FocusAlgorithm.VOLATH.value)
        assert score > 0

    def test_gradient_focus_higher_for_edges(self):
        focused = _make_focused_image()
        blurry = _make_blurry_image()

        score_focused = calculate_focus_score(focused, FocusAlgorithm.GRADIENT.value)
        score_blurry = calculate_focus_score(blurry, FocusAlgorithm.GRADIENT.value)

        assert score_focused > score_blurry

    def test_fft_bandpass_score(self):
        img = _make_focused_image()
        score = calculate_focus_score(img, FocusAlgorithm.FFT_BANDPASS.value)
        assert score > 0

    def test_variance_score(self):
        img = _make_focused_image()
        score = calculate_focus_score(img, FocusAlgorithm.VARIANCE.value)
        assert score > 0

    def test_3d_image_handled(self):
        """3D (multichannel) images should be converted to 2D automatically."""
        img_3d = np.random.rand(128, 128, 3) * 255
        score = calculate_focus_score(img_3d, FocusAlgorithm.VOLATH.value)
        assert isinstance(score, float)


# =========================================================================
# Curve fitting
# =========================================================================


class TestCurveFitting:
    def test_gaussian_fit_finds_peak(self):
        # Create perfect Gaussian data
        positions = np.linspace(0, 100, 20)
        true_peak = 50.0
        scores = 100 * np.exp(-((positions - true_peak) ** 2) / (2 * 10**2)) + 5

        fitted_pos, fitted_scores, params, r_sq = fit_focus_curve(
            positions, scores, FitFunction.GAUSSIAN.value
        )

        # Peak parameter (mu) should be close to 50
        assert params[1] == pytest.approx(true_peak, abs=1.0)
        assert r_sq > 0.95

    def test_parabolic_fit_finds_peak(self):
        # Downward parabola peaking at x=50
        positions = np.linspace(0, 100, 20)
        true_peak = 50.0
        scores = -0.5 * (positions - true_peak) ** 2 + 200

        fitted_pos, fitted_scores, params, r_sq = fit_focus_curve(
            positions, scores, FitFunction.PARABOLIC.value
        )

        # Parabolic vertex: -b / (2a)
        a, b, c = params
        vertex = -b / (2 * a)
        assert vertex == pytest.approx(true_peak, abs=1.0)
        assert r_sq > 0.99

    def test_fit_needs_minimum_points(self):
        positions = np.array([1.0, 2.0])
        scores = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="at least 3"):
            fit_focus_curve(positions, scores)


# =========================================================================
# Adaptive sweep
# =========================================================================


class TestAdaptiveSweep:
    def test_adaptive_sweep_detects_peak(self):
        state = AdaptiveSweepState()

        # Simulate a sweep that rises then falls
        positions = list(range(0, 20))
        scores = [float(10 + 5 * p - 0.3 * p**2) for p in positions]
        # Scores peak around p=8, then decline

        stopped = False
        stop_reason = None
        for pos, score in zip(positions, scores, strict=False):
            result = state.add_point(float(pos), max(score, 0.1))
            if result["should_stop"]:
                stopped = True
                stop_reason = result["reason"]
                break

        assert stopped is True
        assert stop_reason in ("peak_passed", "high_confidence_fit")


# =========================================================================
# Montage
# =========================================================================


class TestFocusMontage:
    def test_focus_montage_shape(self):
        imgs = [np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(3)]
        montage = create_focus_montage(imgs, gap=4)

        # Height = 64 + 30 (label space), Width = 64*3 + 4*2 = 200
        assert montage.shape[0] == 64 + 30
        assert montage.shape[1] == 64 * 3 + 4 * 2

    def test_focus_montage_single_image(self):
        imgs = [np.random.randint(0, 255, (64, 64), dtype=np.uint8)]
        montage = create_focus_montage(imgs)
        assert montage.shape[1] == 64  # no gaps for single image


# =========================================================================
# End-to-end: analyze_focus_stack
# =========================================================================


class TestAnalyzeFocusStack:
    def test_analyze_focus_stack_basic(self):
        positions, images = _make_gaussian_focus_stack(n_positions=15, peak_pos=50.0)
        config = FocusAnalysisConfig(
            algorithm=FocusAlgorithm.VARIANCE.value,
            fit_function=FitFunction.GAUSSIAN.value,
            minimum_r_squared=0.5,
        )

        result = analyze_focus_stack(positions, images, config)

        assert result.success is True
        assert result.best_position == pytest.approx(50.0, abs=5.0)
        assert result.all_scores is not None
        assert len(result.all_scores) == 15

    def test_analyze_focus_stack_length_mismatch(self):
        config = FocusAnalysisConfig()
        result = analyze_focus_stack([1, 2, 3], [np.zeros((10, 10))], config)
        assert result.success is False
        assert "mismatch" in result.error_message.lower()

    def test_analyze_focus_stack_too_few(self):
        config = FocusAnalysisConfig()
        result = analyze_focus_stack([1, 2], [np.zeros((10, 10)), np.zeros((10, 10))], config)
        assert result.success is False
