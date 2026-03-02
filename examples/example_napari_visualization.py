#!/usr/bin/env python
"""
DiSPIM Napari Visualization Examples
===================================

Demonstrates real-time image visualization for DiSPIM experiments using napari.
Shows different visualization patterns for various experiment types.

This example shows:
1. Basic napari setup with Bluesky RunEngine
2. Focus sweep visualization (3D image stacks)
3. Embryo detection visualization (2D image sequences)
4. Dual-sided DiSPIM visualization (multi-channel)
5. Custom visualization configurations
6. Integration with complete DiSPIM workflows

Requirements:
    pip install napari[all]
    # or with specific backend: pip install napari[pyqt5]
"""

import logging
import numpy as np
from bluesky import RunEngine
from bluesky.callbacks import LiveTable

# Import gently components
from gently import (
    # Device classes
    create_dispim_system,
    DiSPIMSystem,
    
    # Plan functions
    focus_sweep,
    dispim_piezo_autofocus,
    find_embryos_with_bottom_camera,
    full_dispim_workflow,
    
    # Configuration classes
    AutofocusConfig,
    CalibrationConfig,
    
    # Analysis utilities
    FocusAlgorithm,
    FitFunction
)

# Import visualization utilities
from gently.visualization import (
    EmbryoMarker,
    mark_embryos_napari,
    generate_focus_curve_plot,
    generate_calibration_summary_plot,
    generate_edge_detection_plot,
)

# Napari availability check
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False


def check_napari_installation():
    """Check if napari is available and provide installation instructions"""
    if not NAPARI_AVAILABLE:
        print("❌ Napari not available!")
        print("\nTo enable image visualization, install napari:")
        print("  pip install napari[all]")
        print("\nOr with specific backend:")
        print("  pip install napari[pyqt5]")
        print("  # or napari[pyside2]")
        print("\nAfter installation, restart and run this example again.")
        return False
    
    print("✅ Napari is available - image visualization enabled!")
    return True


def setup_demo_system():
    """Setup demo DiSPIM system for visualization examples"""
    print("Setting up demo DiSPIM system...")
    
    # Create RunEngine
    RE = RunEngine({})
    
    # For demonstration, we'll use mock system
    # In practice: system = create_dispim_system("/path/to/micromanager", "config.cfg")
    print("  [Note: Using mock system for demonstration]")
    system = None  # Would be actual DiSPIMSystem
    light_sheet = None  # Would be system.side_a
    
    return RE, system, light_sheet


def demonstrate_basic_napari_setup(RE):
    """Demonstrate basic napari visualization setup"""
    print("\n" + "="*60)
    print("1. BASIC NAPARI SETUP - Real-time Image Visualization")
    print("="*60)
    
    print("\nSetting up napari for DiSPIM visualization:")
    
    # Create napari callback with default settings
    napari_callback = setup_napari_callback()
    
    if not napari_callback.enabled:
        print("  ❌ Napari callback disabled (napari not available)")
        return None
    
    # Subscribe to RunEngine
    RE.subscribe(napari_callback)
    
    print("  ✅ Napari callback created and subscribed")
    print(f"  ✅ Viewer title: {napari_callback.viewer.title}")
    print(f"  ✅ Focus sweeps: {napari_callback.show_focus_sweeps}")
    print(f"  ✅ Embryo detection: {napari_callback.show_embryo_detection}")
    print(f"  ✅ Dual channel: {napari_callback.dual_channel_mode}")
    
    print("\nBasic usage pattern:")
    print("  RE = RunEngine({})")
    print("  napari_callback = setup_napari_callback()")
    print("  RE.subscribe(napari_callback)")
    print("  # Now any plan with images will display in napari!")
    
    return napari_callback


def demonstrate_focus_sweep_visualization(RE, light_sheet, napari_callback):
    """Demonstrate focus sweep visualization"""
    print("\n" + "="*60)
    print("2. FOCUS SWEEP VISUALIZATION - 3D Image Stacks")
    print("="*60)
    
    if not napari_callback or not napari_callback.enabled:
        print("  ⚠ Skipping - napari not available")
        return
    
    print("\nFocus sweep creates 3D image stacks visualized in real-time:")
    
    # Configure autofocus
    config = AutofocusConfig(
        num_positions=15,  # Fewer positions for faster demo
        step_size_um=1.0,
        algorithm=FocusAlgorithm.VOLATH.value,
        fit_function=FitFunction.GAUSSIAN.value
    )
    
    print(f"\nAutofocus configuration:")
    print(f"  Positions: {config.num_positions}")
    print(f"  Step size: {config.step_size_um} μm")
    print(f"  Total range: ±{config.num_positions * config.step_size_um / 2} μm")
    
    if light_sheet is not None:
        print(f"\nExecuting autofocus with napari visualization:")
        print(f"  RE(dispim_piezo_autofocus(light_sheet, config))")
        
        # This would display images in napari as they're acquired
        # RE(dispim_piezo_autofocus(light_sheet, config))
        
        print(f"\nNapari display:")
        print(f"  ✅ Images stream to napari as they're acquired")
        print(f"  ✅ 3D stack builds up in real-time")
        print(f"  ✅ Can scrub through Z positions")
        print(f"  ✅ Focus curve visible as image stack")
        
    else:
        print(f"\n[Would execute: RE(dispim_piezo_autofocus(light_sheet, config))]")
        print(f"\nExpected napari behavior:")
        print(f"  - New layer: 'Focus Sweep (Side A)'")
        print(f"  - Green colormap for side A data")
        print(f"  - 3D stack: shape (15, height, width)")
        print(f"  - Real-time updates as images acquired")
    
    print(f"\nVisualization features:")
    print(f"  - Real-time focus quality assessment")
    print(f"  - Immediate feedback on scan progress")
    print(f"  - Visual validation of focus curve")


def demonstrate_embryo_detection_visualization(RE, system, napari_callback):
    """Demonstrate embryo detection visualization"""
    print("\n" + "="*60)
    print("3. EMBRYO DETECTION VISUALIZATION - 2D Image Sequences")
    print("="*60)
    
    if not napari_callback or not napari_callback.enabled:
        print("  ⚠ Skipping - napari not available")
        return
    
    print("\nEmbryo detection creates sequences of 2D images from XY scanning:")
    
    # Configure embryo detection
    detection_config = {
        'scan_area': {
            'x_start': -500, 'x_stop': 500,  # Smaller area for demo
            'y_start': -500, 'y_stop': 500,
            'step_size': 100  # μm between positions
        },
        'detection': {
            'min_size_pixels': 50,
            'max_size_pixels': 500,
            'brightness_threshold': 0.3
        }
    }
    
    print(f"\nDetection configuration:")
    print(f"  Scan area: {detection_config['scan_area']['x_start']} to {detection_config['scan_area']['x_stop']} μm")
    print(f"  Step size: {detection_config['scan_area']['step_size']} μm")
    print(f"  Grid size: 11x11 = 121 positions")
    
    if system is not None:
        print(f"\nExecuting embryo detection with napari visualization:")
        print(f"  RE(find_embryos_with_bottom_camera(system, detection_config))")
        
        # This would display images in napari as XY scan progresses
        # RE(find_embryos_with_bottom_camera(system, detection_config))
        
        print(f"\nNapari display:")
        print(f"  ✅ Each XY position shows in napari immediately")
        print(f"  ✅ Can see scan progress across sample")
        print(f"  ✅ Potential embryos highlighted as found")
        print(f"  ✅ Final mosaic view of scanned area")
        
    else:
        print(f"\n[Would execute: RE(find_embryos_with_bottom_camera(system, detection_config))]")
        print(f"\nExpected napari behavior:")
        print(f"  - New layer: 'Embryo Detection (Side A)'")
        print(f"  - Updates with each XY position")
        print(f"  - 121 total images in sequence")
        print(f"  - Detected embryos marked/highlighted")
    
    print(f"\nVisualization benefits:")
    print(f"  - Real-time quality control of scan")
    print(f"  - Immediate feedback on embryo locations")
    print(f"  - Visual verification of detection algorithm")


def demonstrate_dual_channel_visualization(RE, system, napari_callback):
    """Demonstrate dual-sided DiSPIM visualization"""
    print("\n" + "="*60)
    print("4. DUAL-CHANNEL VISUALIZATION - Multi-Camera Display")
    print("="*60)
    
    if not napari_callback or not napari_callback.enabled:
        print("  ⚠ Skipping - napari not available")
        return
    
    print("\nDual-sided DiSPIM generates images from two cameras simultaneously:")
    
    if system is not None:
        print(f"\nSimulating dual-sided acquisition:")
        print(f"  # Both sides acquire simultaneously")
        print(f"  side_a_image = system.side_a.camera.read()")
        print(f"  side_b_image = system.side_b.camera.read()")
        
        print(f"\nNapari display:")
        print(f"  ✅ Side A: Green channel")
        print(f"  ✅ Side B: Magenta channel") 
        print(f"  ✅ Additive blending for overlay")
        print(f"  ✅ Separate layers for independent control")
        print(f"  ✅ Synchronized updates")
        
    else:
        print(f"\n[Would show both camera feeds simultaneously]")
    
    print(f"\nColor scheme:")
    print(f"  - Side A (illumination from left): Green")
    print(f"  - Side B (illumination from right): Magenta")
    print(f"  - Overlaid: Shows complementary information")
    
    print(f"\nVisualization advantages:")
    print(f"  - Compare image quality from both sides")
    print(f"  - See complementary sample information")
    print(f"  - Identify optimal viewing angle")
    print(f"  - Real-time feedback for dual-sided experiments")


def demonstrate_custom_visualization_configs(RE):
    """Demonstrate custom visualization configurations"""
    print("\n" + "="*60)
    print("5. CUSTOM CONFIGURATIONS - Tailored Visualization")
    print("="*60)
    
    if not NAPARI_AVAILABLE:
        print("  ⚠ Skipping - napari not available")
        return
    
    print("\nCustom configurations for different experiment needs:")
    
    # Configuration 1: Focus-only visualization
    print(f"\n1. Focus-Only Configuration:")
    print(f"   config = {{'show_focus_sweeps': True, 'show_embryo_detection': False}}")
    print(f"   napari_callback = setup_napari_callback(config)")
    
    focus_config = {
        'show_focus_sweeps': True,
        'show_embryo_detection': False,
        'show_single_images': False,
        'update_interval': 0.05  # Faster updates
    }
    
    print(f"   - Only shows focus sweep experiments")
    print(f"   - Faster update rate (0.05s)")
    print(f"   - Optimized for autofocus development")
    
    # Configuration 2: High-throughput visualization
    print(f"\n2. High-Throughput Configuration:")
    print(f"   config = {{'show_single_images': False, 'update_interval': 1.0}}")
    
    throughput_config = {
        'show_focus_sweeps': True,
        'show_embryo_detection': True,
        'show_single_images': False,  # Skip individual images
        'update_interval': 1.0  # Slower updates for performance
    }
    
    print(f"   - Skip individual images to reduce overhead")
    print(f"   - Slower update rate (1.0s) for performance")
    print(f"   - Better for automated, high-throughput experiments")
    
    # Configuration 3: Development/debugging
    print(f"\n3. Development/Debugging Configuration:")
    print(f"   config = {{'show_single_images': True, 'update_interval': 0.01}}")
    
    debug_config = {
        'show_focus_sweeps': True,
        'show_embryo_detection': True,
        'show_single_images': True,
        'update_interval': 0.01  # Very fast updates
    }
    
    print(f"   - Show every image for detailed inspection")
    print(f"   - Very fast updates (0.01s)")
    print(f"   - Maximum detail for troubleshooting")
    
    print(f"\nUsage pattern:")
    print(f"  # Choose configuration for your needs")
    print(f"  config = focus_config  # or throughput_config, debug_config")
    print(f"  napari_callback = setup_napari_callback(config)")
    print(f"  RE.subscribe(napari_callback)")


def demonstrate_convenience_functions(RE):
    """Demonstrate convenience functions for common patterns"""
    print("\n" + "="*60)
    print("6. CONVENIENCE FUNCTIONS - Common Usage Patterns")
    print("="*60)
    
    if not NAPARI_AVAILABLE:
        print("  ⚠ Skipping - napari not available")
        return
    
    print("\nConvenience functions for common visualization needs:")
    
    print(f"\n1. Focus Sweep Only:")
    print(f"   from gently.visualization import enable_focus_sweep_visualization")
    print(f"   enable_focus_sweep_visualization(RE)")
    print(f"   # Optimized for autofocus experiments")
    
    print(f"\n2. Embryo Detection Only:")
    print(f"   from gently.visualization import enable_embryo_detection_visualization")
    print(f"   enable_embryo_detection_visualization(RE)")
    print(f"   # Optimized for sample detection")
    
    print(f"\n3. Full Visualization:")
    print(f"   from gently.visualization import enable_full_visualization")
    print(f"   enable_full_visualization(RE)")
    print(f"   # Shows everything - good for general use")
    
    print(f"\n4. Custom Viewer:")
    print(f"   from gently.visualization import create_napari_viewer")
    print(f"   viewer = create_napari_viewer('My DiSPIM Experiment')")
    print(f"   callback = setup_napari_callback(viewer=viewer)")
    print(f"   # Use your own configured viewer")
    
    print(f"\nBenefits:")
    print(f"  - One-line setup for common patterns")
    print(f"  - Pre-configured for specific experiment types")
    print(f"  - Easy to integrate into existing workflows")


def demonstrate_complete_workflow_visualization(RE, system):
    """Demonstrate visualization with complete DiSPIM workflow"""
    print("\n" + "="*60)
    print("7. COMPLETE WORKFLOW VISUALIZATION - Full Experiment")
    print("="*60)
    
    if not NAPARI_AVAILABLE:
        print("  ⚠ Skipping - napari not available")
        return
    
    print("\nVisualization during complete multi-embryo workflow:")
    
    # Setup full visualization
    print(f"\nSetting up comprehensive visualization:")
    print(f"  napari_callback = enable_full_visualization(RE)")
    print(f"  # Will show all stages of the workflow")
    
    # Complete workflow configuration
    workflow_config = {
        'system_setup': {
            'center_devices': True,
            'run_calibration': True
        },
        'calibration': {
            'point1_um': 25.0,
            'point2_um': 75.0,
            'autofocus_each_point': True
        },
        'embryo_detection': {
            'x_start': -1000, 'x_stop': 1000,
            'y_start': -1000, 'y_stop': 1000,
            'step_size': 200
        },
        'acquisition': {
            'z_stack': {'range_um': 50, 'step_size_um': 1.0},
            'dual_sided': True,
            'time_points': 3
        }
    }
    
    print(f"\nWorkflow stages with visualization:")
    
    if system is not None:
        print(f"\n  RE(full_dispim_workflow(system, workflow_config))")
        print(f"\n  Expected napari display sequence:")
        
    print(f"  1. Calibration stage:")
    print(f"     - Focus sweeps at calibration points")
    print(f"     - Real-time focus quality assessment")
    
    print(f"  2. Embryo detection stage:")
    print(f"     - XY scan images streaming in")
    print(f"     - Detected embryo positions highlighted")
    
    print(f"  3. Multi-embryo acquisition:")
    print(f"     - Focus sweeps for each embryo")
    print(f"     - Z-stack acquisitions (dual-channel)")
    print(f"     - Time series progression")
    
    print(f"\nVisualization benefits for complete workflow:")
    print(f"  ✅ Monitor entire experiment progress")
    print(f"  ✅ Quality control at each stage")
    print(f"  ✅ Early detection of issues")
    print(f"  ✅ Real-time data assessment")
    print(f"  ✅ Immediate feedback on results")


def main():
    """Main napari visualization demonstration"""
    print("DiSPIM Napari Visualization Examples")
    print("=" * 60)
    print()
    print("This example demonstrates real-time image visualization")
    print("for DiSPIM experiments using napari and Bluesky callbacks.")
    print()
    
    # Check napari installation
    if not check_napari_installation():
        return
    
    # Setup demo system
    RE, system, light_sheet = setup_demo_system()
    
    # Run demonstrations
    napari_callback = demonstrate_basic_napari_setup(RE)
    demonstrate_focus_sweep_visualization(RE, light_sheet, napari_callback)
    demonstrate_embryo_detection_visualization(RE, system, napari_callback)
    demonstrate_dual_channel_visualization(RE, system, napari_callback)
    demonstrate_custom_visualization_configs(RE)
    demonstrate_convenience_functions(RE)
    demonstrate_complete_workflow_visualization(RE, system)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Napari Visualization Integration")
    print("="*60)
    
    print(f"\n✅ Napari integration complete:")
    print(f"  - Real-time image streaming from Bluesky plans")
    print(f"  - Automatic 3D stack visualization for focus sweeps")
    print(f"  - 2D image sequences for embryo detection")
    print(f"  - Dual-channel support for two-sided DiSPIM")
    print(f"  - Configurable visualization options")
    
    print(f"\n✅ Key benefits:")
    print(f"  - Immediate visual feedback during experiments")
    print(f"  - Quality control and error detection")
    print(f"  - Interactive data exploration")
    print(f"  - Non-intrusive - works with existing plans")
    print(f"  - Optional - graceful fallback if napari not available")
    
    print(f"\n✅ Usage patterns:")
    print(f"  - Basic: setup_napari_callback() → RE.subscribe()")
    print(f"  - Custom: setup_napari_callback(config) for specific needs")
    print(f"  - Convenience: enable_focus_sweep_visualization(RE)")
    print(f"  - Integration: Works with all existing DiSPIM plans")
    
    print(f"\nNext steps:")
    print(f"  1. Install napari: pip install napari[all]")
    print(f"  2. Add visualization to your DiSPIM experiments")
    print(f"  3. Customize configurations for your needs")
    print(f"  4. Enjoy real-time image feedback!")
    
    if napari_callback and napari_callback.enabled:
        print(f"\nNapari viewer is open - explore the interface!")
        print(f"  - Layer controls for each image type")
        print(f"  - Color/brightness adjustments")
        print(f"  - 3D visualization controls")
        print(f"  - Screenshot and movie export options")
    
    print(f"\nGently DiSPIM + Napari: Real-time microscopy visualization! 🔬✨")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    main()