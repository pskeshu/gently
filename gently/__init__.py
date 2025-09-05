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
    DiSPIMPiezo,
    DiSPIMGalvo, 
    DiSPIMCamera,
    DiSPIMXYStage,
    DiSPIMLaserControl,
    DiSPIMLightSheet,
    DiSPIMSystem,
    create_dispim_system
)

# Plan functions - device-agnostic Bluesky plans
from .plans import (
    # Atomic plans
    focus_sweep,
    move_and_acquire,
    synchronized_move,
    
    # Autofocus plans
    dispim_piezo_autofocus,
    dispim_galvo_autofocus, 
    dual_sided_autofocus,
    
    # Calibration plans
    dispim_two_point_calibration,
    dispim_full_calibration,
    
    # Embryo workflow plans
    find_embryos_with_bottom_camera,
    acquire_embryo_lightsheet,
    full_dispim_workflow,
    
    # Convenience functions
    quick_autofocus,
    quick_calibration,
    
    # Configuration classes
    AutofocusConfig,
    CalibrationConfig
)

# Analysis utilities - device-agnostic focus analysis
from .analysis import (
    calculate_focus_score,
    fit_focus_curve,
    find_curve_maximum,
    validate_autofocus_result,
    analyze_focus_stack,
    FocusAnalysisConfig,
    FocusResult,
    FocusAlgorithm,
    FitFunction
)

# Coordinate utilities - transformations and reference mapping
from .coordinates import (
    piezo_to_galvo,
    galvo_to_piezo,
    calculate_piezo_galvo_calibration,
    transform_coordinates_2d,
    create_affine_transform_2d,
    create_reference_map,
    add_calibration_point,
    add_embryo_position_stage,
    stage_to_lightsheet_coordinates,
    find_nearest_embryos,
    save_reference_map,
    load_reference_map,
    validate_reference_map,
    CalibrationPoint,
    ReferenceMap
)

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

__version__ = "0.2.0"
__all__ = [
    # Device classes
    "DiSPIMPiezo",
    "DiSPIMGalvo", 
    "DiSPIMCamera",
    "DiSPIMXYStage", 
    "DiSPIMLaserControl",
    "DiSPIMLightSheet",
    "DiSPIMSystem",
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
    "piezo_to_galvo",
    "galvo_to_piezo",
    "calculate_piezo_galvo_calibration",
    "transform_coordinates_2d",
    "create_affine_transform_2d",
    "create_reference_map",
    "add_calibration_point",
    "add_embryo_position_stage", 
    "stage_to_lightsheet_coordinates",
    "find_nearest_embryos",
    "save_reference_map",
    "load_reference_map",
    "validate_reference_map",
    "CalibrationPoint",
    "ReferenceMap"
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