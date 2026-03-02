"""
Gently DiSPIM
============

Device-agnostic Bluesky plans and Ophyd devices for diSPIM imaging.
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

# Visualization utilities - plots and embryo marking
try:
    from .visualization import (
        EmbryoMarker,
        mark_embryos_napari,
        get_visualization_server,
        generate_focus_curve_plot,
        generate_calibration_summary_plot,
        generate_edge_detection_plot,
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

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

__version__ = "0.9.0"
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
    "DiSPIMZstage",
    "DiSPIMCamera",
    "DiSPIMDualCamera",
    "DiSPIMXYStage",
    "DiSPIMFDrive",
    "DiSPIMPiezo",
    "DiSPIMScanner",
    "DiSPIMLED",
    "DiSPIMLaserControl",
    "DiSPIMVolumeScanner",
    "DiSPIMBottomCamera",
    "DiSPIMLightSheetSnap",

    # Plan functions
    "compute_fft_bandpass_score",
    "detect_embryo_roi",
    "select_best_camera_view",
    "focus_sweep_plan",
    "calibrate_piezo_galvo_plan",
    "mark_embryo_interactive_plan",
    "acquire_single_volume_plan",
    "timelapse_volume_plan",
    "multi_embryo_calibration_workflow",

    # Calibration plans (conditionally imported)
    "verify_embryo_centered",
    "detect_embryo_edge",
    "calibrate_focus_at_position",
    "calibrate_embryo_piezo_galvo",
    "EMBRYO_CENTERING_PROMPT",
    "EMBRYO_EDGE_PROMPT",

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
    "FocusAnalysisConfig",
    "FocusResult",
    "FocusAlgorithm",
    "FitFunction",
    "calculate_focus_score",
    "analyze_focus_stack",
    "fit_focus_curve",

    # Coordinate functions
    "pixel_to_stage_position",
    "stage_to_pixel_position",
    "pixel_displacement_to_stage_movement",
    "get_um_per_pixel",
    "DEFAULT_PIXEL_SIZE_UM",
    "DEFAULT_OBJECTIVE_MAG",
]

# Add visualization functions if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        # Visualization functions
        "EmbryoMarker",
        "mark_embryos_napari",
        "get_visualization_server",
        "generate_focus_curve_plot",
        "generate_calibration_summary_plot",
        "generate_edge_detection_plot",
    ])