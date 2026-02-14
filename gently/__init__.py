"""
Gently DiSPIM
============

Device-agnostic Bluesky plans and Ophyd devices for DiSPIM microscopy.
Built around atomic microscope actions that compose into complex experimental workflows.

Key Components:
    - devices: Proper Ophyd devices for DiSPIM hardware
    - plans: Device-agnostic Bluesky plans for autofocus, calibration, and acquisition
    - analysis: Focus scoring and curve fitting utilities  
    - coordinates: Coordinate transformations and reference mapping

Complete DiSPIM functionality from autofocus and calibration 
to embryo detection and multi-embryo acquisition workflows.
"""

# Core device classes - proper Ophyd devices for Bluesky
from .devices import (
    DiSPIMZstage,
    DiSPIMCamera,
    DiSPIMDualCamera,
    DiSPIMXYStage,
    DiSPIMFDrive,
    DiSPIMPiezo,
    DiSPIMScanner,
    DiSPIMLED,
    DiSPIMLaserControl,
    DiSPIMVolumeScanner,
    DiSPIMBottomCamera,
    DiSPIMLightSheetSnap
)

# Plan functions - device-agnostic Bluesky plans
from .plans import (
    compute_fft_bandpass_score,
    detect_embryo_roi,
    select_best_camera_view,
    focus_sweep_plan,
    calibrate_piezo_galvo_plan,
    mark_embryo_interactive_plan,
    acquire_single_volume_plan,
    timelapse_volume_plan,
    multi_embryo_calibration_workflow
)

# Calibration plans - embryo-based piezo-galvo calibration
try:
    from .calibration_plans import (
        verify_embryo_centered,
        detect_embryo_edge,
        calibrate_focus_at_position,
        calibrate_embryo_piezo_galvo,
        EMBRYO_CENTERING_PROMPT,
        EMBRYO_EDGE_PROMPT
    )
    _CALIBRATION_PLANS_AVAILABLE = True
except ImportError:
    _CALIBRATION_PLANS_AVAILABLE = False

# Configuration utilities - hardware profiles and calibration data
try:
    from .config import (
        HardwareProfile,
        CameraMode,
        CameraConfig,
        ScannerPattern,
        ScannerMode,
        GalvoAxisConfig,
        ScannerConfig,
        PiezoGalvoCalibration,
        calculate_spim_timing,
        get_calibration_camera_config,
        get_hardware_spim_camera_config,
        get_standard_scanner_config
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

# Analysis utilities - device-agnostic focus analysis
try:
    from .analysis.core import (
        FocusAnalysisConfig,
        FocusResult,
        FocusAlgorithm,
        FitFunction,
        calculate_focus_score,
        analyze_focus_stack,
        fit_focus_curve
    )
    _ANALYSIS_AVAILABLE = True
except ImportError:
    _ANALYSIS_AVAILABLE = False

# Coordinate utilities - transformations for pixel/stage conversions
try:
    from .coordinates import (
        pixel_to_stage_position,
        stage_to_pixel_position,
        pixel_displacement_to_stage_movement,
        get_um_per_pixel,
        DEFAULT_PIXEL_SIZE_UM,
        DEFAULT_OBJECTIVE_MAG
    )
    _COORDINATES_AVAILABLE = True
except ImportError:
    _COORDINATES_AVAILABLE = False

# Visualization utilities - optional napari integration
try:
    from .visualization import (
        setup_napari_callback,
        create_napari_viewer,
        enable_focus_sweep_visualization,
        enable_embryo_detection_visualization,
        enable_full_visualization,
        NapariCallback,
        NAPARI_AVAILABLE
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    # Napari not available - visualization features disabled
    _VISUALIZATION_AVAILABLE = False
    NAPARI_AVAILABLE = False

# Main entry point
from .gently import Gently, create_gently

# Core infrastructure
from .core import (
    TiledStore,
    DatabrokerStore,
    EventBus,
    EventType,
    get_event_bus,
    get_data_store,
)

__version__ = "0.6.0"
__all__ = [
    # Main entry point
    "Gently",
    "create_gently",

    # Core infrastructure
    "TiledStore",
    "DatabrokerStore",
    "EventBus",
    "EventType",
    "get_event_bus",
    "get_data_store",
    # Device classes
    "DiSPIMPiezo",
    "DiSPIMGalvo",
    "DiSPIMCamera",
    "DiSPIMXYStage",
    "DiSPIMLaserControl",
    "DiSPIMLightSheet",
    "DiSPIMSystem",
    "DiSPIMVolumeScanner",
    "DiSPIMVolumeAcquisition",
    "create_dispim_system",

    # Plan functions
    "focus_sweep",
    "move_and_acquire",
    "synchronized_move",
    "dispim_piezo_autofocus",
    "dispim_galvo_autofocus",
    "dual_sided_autofocus",
    "dispim_two_point_calibration",
    "dispim_full_calibration",
    "find_embryos_with_bottom_camera",
    "acquire_embryo_lightsheet",
    "full_dispim_workflow",
    "quick_autofocus",
    "quick_calibration",
    "AutofocusConfig",
    "CalibrationConfig",

    # Volume acquisition plans
    "acquire_spim_volume",
    "multi_position_volume",
    "volume_timelapse",
    "multi_position_volume_timelapse",
    "acquire_embryo_volume",

    # Configuration classes and utilities
    "HardwareProfile",
    "CameraMode",
    "CameraConfig",
    "ScannerPattern",
    "ScannerMode",
    "GalvoAxisConfig",
    "ScannerConfig",
    "PiezoGalvoCalibration",
    "calculate_spim_timing",
    "get_calibration_camera_config",
    "get_hardware_spim_camera_config",
    "get_standard_scanner_config",

    # Analysis functions
    "calculate_focus_score",
    "fit_focus_curve",
    "find_curve_maximum", 
    "validate_autofocus_result",
    "analyze_focus_stack",
    "FocusAnalysisConfig",
    "FocusResult",
    "FocusAlgorithm", 
    "FitFunction",
    
    # Coordinate functions
    "pixel_to_stage_position",
    "stage_to_pixel_position",
    "pixel_displacement_to_stage_movement",
    "get_um_per_pixel",
    "DEFAULT_PIXEL_SIZE_UM",
    "DEFAULT_OBJECTIVE_MAG"
]

# Add visualization functions if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        # Visualization functions
        "setup_napari_callback",
        "create_napari_viewer",
        "enable_focus_sweep_visualization",
        "enable_embryo_detection_visualization", 
        "enable_full_visualization",
        "NapariCallback",
        "NAPARI_AVAILABLE"
    ])