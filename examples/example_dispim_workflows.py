#!/usr/bin/env python
"""
Gently DiSPIM Workflow Examples
==============================

Demonstrates the complete DiSPIM functionality using device-agnostic Bluesky plans.
Shows the progression from atomic plans to complex workflows for real DiSPIM experiments.

This example shows:
1. Device-agnostic atomic plans (focus_sweep)
2. Autofocus functionality for precise positioning
3. Two-point calibration for coordinate mapping  
4. Embryo detection with bottom camera
5. Complete multi-embryo acquisition workflows

All plans work with proper Ophyd devices and standard Bluesky plan stubs.
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
    focus_sweep,                    # Atomic plan
    dispim_piezo_autofocus,        # Autofocus functionality
    dispim_two_point_calibration,  # Calibration plan
    full_dispim_workflow,          # Complete workflow
    
    # Configuration
    AutofocusConfig,
    CalibrationConfig,
    
    # Analysis utilities
    FocusAlgorithm,
    FitFunction
)

# Import optional napari visualization
try:
    from gently import (
        setup_napari_callback,
        enable_focus_sweep_visualization,
        enable_full_visualization,
        NAPARI_AVAILABLE
    )
except ImportError:
    # Napari not available
    NAPARI_AVAILABLE = False


def setup_dispim_session():
    """
    Setup DiSPIM session with devices and RunEngine
    
    In practice, this would use actual MM installation paths:
    system = create_dispim_system("/path/to/micromanager", "/path/to/config.cfg")
    """
    print("Setting up DiSPIM session...")
    
    # Create RunEngine
    RE = RunEngine({})
    
    # For demonstration, create a mock system
    # In practice: system = create_dispim_system(mm_dir, config_file)
    print("  [Note: Using mock system for demonstration]")
    print("  [In practice: system = create_dispim_system(mm_dir, config_file)]")
    system = None  # Would be actual DiSPIMSystem
    
    # Setup live callbacks
    live_table = LiveTable(['piezo_a_position', 'galvo_a_position', 'camera_a_image'])
    RE.subscribe(live_table)
    
    return RE, system


def setup_napari_visualization(RE):
    """Setup optional napari visualization for real-time image display"""
    print("\n" + "="*60)
    print("OPTIONAL: NAPARI VISUALIZATION SETUP")
    print("="*60)
    
    if not NAPARI_AVAILABLE:
        print("\n⚠ Napari not available - skipping visualization setup")
        print("  To enable real-time image visualization:")
        print("    pip install napari[all]")
        print("  Then restart and run this example again.")
        return None
    
    print("\n✅ Napari available - setting up real-time visualization")
    
    # Setup napari callback for all DiSPIM experiments
    napari_callback = enable_full_visualization(RE)
    
    if napari_callback.enabled:
        print("  ✅ Napari callback enabled and subscribed to RunEngine")
        print("  ✅ Real-time image display will show:")
        print("    - Focus sweep image stacks (3D)")
        print("    - Embryo detection scan images")
        print("    - Dual-channel DiSPIM data (green/magenta)")
        print("    - Individual camera acquisitions")
        
        print(f"\n  Napari viewer: '{napari_callback.viewer.title}'")
        print(f"  - Focus sweeps: {napari_callback.show_focus_sweeps}")
        print(f"  - Embryo detection: {napari_callback.show_embryo_detection}")
        print(f"  - Dual channels: {napari_callback.dual_channel_mode}")
        
        print("\n  💡 The napari window is now open - you can:")
        print("    - Adjust layer visibility and colors")
        print("    - Explore 3D image stacks")
        print("    - Take screenshots and export movies")
        print("    - View images in real-time during experiments")
        
    else:
        print("  ❌ Napari callback failed to initialize")
        return None
    
    return napari_callback


def demonstrate_atomic_plans(RE, light_sheet):
    """Demonstrate device-agnostic atomic plans"""
    print("\n" + "="*60)
    print("1. ATOMIC PLANS - Device-Agnostic Building Blocks")
    print("="*60)
    
    print("\nThe foundation: focus_sweep(positioner, positions, detector)")
    print("This atomic plan works with ANY positioner and detector:")
    
    # Define scan positions
    positions = np.linspace(-10, 10, 11)  # 11 positions from -10 to +10 μm
    
    print(f"\n  Positions to scan: {positions}")
    print(f"  Number of positions: {len(positions)}")
    
    if light_sheet is not None:
        # Example 1: Piezo focus sweep
        print("\n  Example 1: Piezo focus sweep")
        print("    focus_sweep(light_sheet.piezo, positions, light_sheet.camera)")
        # RE(focus_sweep(light_sheet.piezo, positions, light_sheet.camera))
    
        # Example 2: Galvo focus sweep  
        print("\n  Example 2: Galvo focus sweep")
        print("    focus_sweep(light_sheet.galvo, positions, light_sheet.camera)")
        # RE(focus_sweep(light_sheet.galvo, positions, light_sheet.camera))
    
        # Example 3: XY stage sweep (device-agnostic)
        print("\n  Example 3: XY stage sweep (device-agnostic!)")
        print("    focus_sweep(xy_stage.x, positions, light_sheet.camera)")
        # RE(focus_sweep(xy_stage.x, positions, light_sheet.camera))
    else:
        print("    [Would execute: RE(focus_sweep(device, positions, detector))]")
    
    print("\nKey insight: Same atomic plan, different devices!")
    print("This is the power of device-agnostic Bluesky plans.")


def demonstrate_autofocus_functionality(RE, light_sheet):
    """Demonstrate autofocus functionality for precise positioning"""
    print("\n" + "="*60)
    print("2. AUTOFOCUS FUNCTIONALITY - Precise Device Positioning")
    print("="*60)
    
    print("\nAutofocus builds on the atomic focus_sweep plan:")
    print("dispim_piezo_autofocus() = focus_sweep() + analysis + validation")
    
    # Create autofocus configuration
    config = AutofocusConfig(
        num_positions=21,
        step_size_um=0.5,
        algorithm=FocusAlgorithm.VOLATH.value,
        fit_function=FitFunction.GAUSSIAN.value,
        minimum_r_squared=0.75,
        center_at_current=True
    )
    
    print(f"\nAutofocus Configuration:")
    print(f"  Positions: {config.num_positions}")
    print(f"  Step size: {config.step_size_um} μm")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Fit function: {config.fit_function}")
    print(f"  Min R²: {config.minimum_r_squared}")
    
    if light_sheet is not None:
        print(f"\nExecuting autofocus:")
        print(f"  RE(dispim_piezo_autofocus(light_sheet, config))")
        # RE(dispim_piezo_autofocus(light_sheet, config))
        
        if NAPARI_AVAILABLE:
            print(f"\n  📺 With napari visualization:")
            print(f"    - Images stream to napari in real-time")
            print(f"    - 3D focus stack builds as positions are scanned")
            print(f"    - Can see focus quality at each position")
            print(f"    - Visual feedback on autofocus progress")
    else:
        print(f"\n[Would execute: RE(dispim_piezo_autofocus(light_sheet, config))]")
    
    print(f"\nAutofocus workflow:")
    print(f"  1. bps.stage(light_sheet)      # Save current state")
    print(f"  2. focus_sweep(piezo, positions, camera)  # Atomic plan")
    print(f"  3. analyze_focus_stack(positions, images) # Find best position")
    print(f"  4. bps.mv(piezo, best_position) # Move to focus")
    print(f"  5. bps.unstage(light_sheet)     # Restore if failed")
    
    print(f"\nThis enables precise, automated positioning for experiments!")


def demonstrate_calibration_workflow(RE, light_sheet):
    """Demonstrate calibration workflows for coordinate mapping"""
    print("\n" + "="*60)
    print("3. CALIBRATION WORKFLOW - Coordinate System Mapping")
    print("="*60)
    
    print("\nTwo-point calibration uses autofocus at each calibration point:")
    print("dispim_two_point_calibration() = move + autofocus + move + autofocus + fit")
    
    # Create calibration configuration
    autofocus_config = AutofocusConfig(
        num_positions=11,  # Faster for calibration
        step_size_um=1.0,
        algorithm=FocusAlgorithm.VOLATH.value
    )
    
    cal_config = CalibrationConfig(
        point1_um=25.0,
        point2_um=75.0,
        autofocus_each_point=True,
        autofocus_config=autofocus_config
    )
    
    print(f"\nCalibration Configuration:")
    print(f"  Point 1: {cal_config.point1_um} μm")
    print(f"  Point 2: {cal_config.point2_um} μm")  
    print(f"  Autofocus at each point: {cal_config.autofocus_each_point}")
    
    if light_sheet is not None:
        print(f"\nExecuting calibration:")
        print(f"  RE(dispim_two_point_calibration(light_sheet, cal_config))")
        # RE(dispim_two_point_calibration(light_sheet, cal_config))
    else:
        print(f"\n[Would execute: RE(dispim_two_point_calibration(light_sheet, cal_config))]")
    
    print(f"\nCalibration workflow:")
    print(f"  1. bps.mv(piezo, point1)        # Move to first point")
    print(f"  2. dispim_galvo_autofocus()     # Focus galvo (uses atomic plans)")
    print(f"  3. bps.trigger_and_read([...])  # Record positions")
    print(f"  4. bps.mv(piezo, point2)        # Move to second point")
    print(f"  5. dispim_galvo_autofocus()     # Focus galvo again")  
    print(f"  6. bps.trigger_and_read([...])  # Record positions")
    print(f"  7. calculate_linear_fit()       # Determine calibration")
    
    print(f"\nCalibration enables coordinate transformations between devices!")


def demonstrate_embryo_detection_workflow(RE, dispim_system):
    """Demonstrate embryo detection with bottom camera"""
    print("\n" + "="*60)
    print("4. EMBRYO DETECTION WORKFLOW - Automated Sample Finding")
    print("="*60)
    
    print("\nEmbryo detection uses bottom camera for automated sample finding:")
    print("find_embryos_with_bottom_camera() = XY scan + image analysis + position recording")
    
    # Create detection configuration
    detection_config = {
        'scan_area': {
            'x_start': -1000, 'x_stop': 1000,  # μm
            'y_start': -1000, 'y_stop': 1000,  # μm
            'step_size': 200  # μm between positions
        },
        'detection': {
            'min_size_pixels': 50,
            'max_size_pixels': 500,
            'brightness_threshold': 0.3,
            'circularity_threshold': 0.7
        },
        'safety': {
            'z_position_um': 0.0,  # Safe Z position during XY scan
            'max_scan_time_minutes': 10
        }
    }
    
    print(f"\nDetection Configuration:")
    print(f"  Scan area: {detection_config['scan_area']['x_start']} to {detection_config['scan_area']['x_stop']} μm (X)")
    print(f"             {detection_config['scan_area']['y_start']} to {detection_config['scan_area']['y_stop']} μm (Y)")
    print(f"  Step size: {detection_config['scan_area']['step_size']} μm")
    print(f"  Detection thresholds: size {detection_config['detection']['min_size_pixels']}-{detection_config['detection']['max_size_pixels']} pixels")
    
    if dispim_system is not None:
        print(f"\nExecuting embryo detection:")
        print(f"  RE(find_embryos_with_bottom_camera(dispim_system, detection_config))")
        # RE(find_embryos_with_bottom_camera(dispim_system, detection_config))
        
        if NAPARI_AVAILABLE:
            print(f"\n  📺 With napari visualization:")
            print(f"    - Each XY position shows in napari as it's acquired")
            print(f"    - See scan progress across the sample area")
            print(f"    - Detected embryos can be highlighted in real-time")
            print(f"    - Build up a mosaic view of the scanned region")
    else:
        print(f"\n[Would execute: RE(find_embryos_with_bottom_camera(dispim_system, detection_config))]")
    
    print(f"\nDetection workflow:")
    print(f"  1. bps.mv(xy_stage.z, safe_z_position)    # Move to safe Z")
    print(f"  2. XY grid scan with bottom camera        # Scan entire area")
    print(f"  3. Analyze images for embryo features     # Find circular objects")
    print(f"  4. Record embryo positions in stage coords # Store locations")
    print(f"  5. Convert to light sheet coordinates     # Transform coords")
    
    print(f"\nAutomated detection finds all samples for batch processing!")


def demonstrate_complete_workflow(RE, dispim_system):
    """Demonstrate complete multi-embryo acquisition workflow"""
    print("\n" + "="*60)
    print("5. COMPLETE WORKFLOW - Multi-Embryo Light Sheet Acquisition")
    print("="*60)
    
    print("\nThe complete workflow combines all components:")
    print("full_dispim_workflow() = calibration + embryo_detection + acquisition")
    
    # Create complete workflow configuration
    workflow_config = {
        'system_setup': {
            'center_devices': True,
            'run_calibration': True,
            'validate_hardware': True
        },
        'calibration': {
            'point1_um': 25.0,
            'point2_um': 75.0,
            'autofocus_each_point': True,
            'autofocus_config': {
                'num_positions': 11,
                'step_size_um': 1.0,
                'algorithm': 'volath'
            }
        },
        'embryo_detection': {
            'x_start': -1000, 'x_stop': 1000,
            'y_start': -1000, 'y_stop': 1000, 
            'step_size': 200,
            'detection_thresholds': {
                'min_size': 50, 'max_size': 500,
                'brightness': 0.3, 'circularity': 0.7
            }
        },
        'acquisition': {
            'autofocus_config': {
                'num_positions': 21, 
                'step_size_um': 0.5,
                'algorithm': 'volath'
            },
            'z_stack': {
                'range_um': 50,  # ±25 μm around focus
                'step_size_um': 1.0
            },
            'dual_sided': True,
            'time_points': 1
        }
    }
    
    print(f"\nWorkflow Configuration:")
    print(f"  Calibration: {workflow_config['calibration']['point1_um']} to {workflow_config['calibration']['point2_um']} μm")
    print(f"  Detection area: {workflow_config['embryo_detection']['x_start']} to {workflow_config['embryo_detection']['x_stop']} μm")
    print(f"  Z-stack range: ±{workflow_config['acquisition']['z_stack']['range_um']//2} μm")
    print(f"  Dual-sided: {workflow_config['acquisition']['dual_sided']}")
    
    if dispim_system is not None:
        print(f"\nExecuting complete workflow:")
        print(f"  RE(full_dispim_workflow(dispim_system, workflow_config))")
        # RE(full_dispim_workflow(dispim_system, workflow_config))
    else:
        print(f"\n[Would execute: RE(full_dispim_workflow(dispim_system, workflow_config))]")
    
    print(f"\nComplete workflow stages:")
    print(f"  1. System initialization and hardware validation")
    print(f"  2. Two-point calibration (with autofocus)")
    print(f"  3. Embryo detection with bottom camera")
    print(f"  4. For each detected embryo:")
    print(f"     a. Move to embryo position")
    print(f"     b. Autofocus both sides")
    print(f"     c. Acquire dual-sided Z-stack")
    print(f"     d. Save data with metadata")
    
    print(f"\nAutomated, high-throughput DiSPIM experiments!")


def demonstrate_extensibility(RE):
    """Demonstrate how the atomic approach enables easy extension"""
    print("\n" + "="*60)
    print("6. EXTENSIBILITY - Easy Addition of New Capabilities")
    print("="*60)
    
    print("\nBecause plans are device-agnostic, new capabilities are easy:")
    
    print(f"\nNew hardware? Same plans work:")
    print(f"  focus_sweep(new_positioner, positions, new_detector)")
    print(f"  dispim_piezo_autofocus(new_light_sheet, config)")
    
    print(f"\nNew algorithms? Just swap the analysis:")
    print(f"  config.algorithm = 'new_algorithm'")
    print(f"  Same dispim_piezo_autofocus() plan!")
    
    print(f"\nNew workflows? Compose existing plans:")
    print(f"  def adaptive_autofocus_with_ai(light_sheet, ai_callback):")
    print(f"    yield from dispim_piezo_autofocus(light_sheet, config)")
    print(f"    decision = ai_callback(result)")
    print(f"    if decision.refine:")
    print(f"      yield from dispim_galvo_autofocus(light_sheet, refined_config)")
    
    print(f"\nIntegration with other systems:")
    print(f"  def combined_microscopy_workflow(dispim, confocal, shared_stage):")
    print(f"    yield from focus_sweep(shared_stage.z, positions, dispim.camera)")
    print(f"    yield from focus_sweep(shared_stage.z, positions, confocal.camera)")
    print(f"    # Same atomic plan, different systems!")
    
    print(f"\nThe atomic approach scales naturally!")


def main():
    """Main demonstration of DiSPIM workflow functionality"""
    print("Gently DiSPIM Workflow Examples")
    print("=" * 60)
    print()
    print("This example demonstrates complete DiSPIM functionality using")
    print("device-agnostic Bluesky plans for real microscopy experiments.")
    print()
    print("Key concepts:")
    print("  - Atomic plans work with ANY compatible devices")
    print("  - Autofocus enables precise positioning") 
    print("  - Complex workflows compose atomic plans")
    print("  - Easy extensibility through device agnosticism")
    
    # Setup session
    RE, system = setup_dispim_session()
    light_sheet = getattr(system, 'side_a', None) if system else None
    
    # Setup optional napari visualization
    napari_callback = setup_napari_visualization(RE)
    
    # Run demonstrations
    demonstrate_atomic_plans(RE, light_sheet)
    demonstrate_autofocus_functionality(RE, light_sheet)
    demonstrate_calibration_workflow(RE, light_sheet)
    demonstrate_embryo_detection_workflow(RE, system)
    demonstrate_complete_workflow(RE, system)
    demonstrate_extensibility(RE)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Complete DiSPIM Functionality")
    print("="*60)
    
    print(f"\n✓ Created proper Ophyd devices (DiSPIMPiezo, DiSPIMCamera, etc.)")
    print(f"✓ Built device-agnostic atomic plans (focus_sweep, move_and_acquire)")
    print(f"✓ Implemented autofocus for precise positioning")
    print(f"✓ Added calibration workflows for coordinate mapping")
    print(f"✓ Created embryo detection for automated sample finding")
    print(f"✓ Integrated complete multi-embryo acquisition workflows")
    if NAPARI_AVAILABLE:
        print(f"✓ Enabled real-time image visualization with napari")
    
    print(f"\nKey benefits:")
    print(f"  1. Device-agnostic - plans work with any compatible hardware")
    print(f"  2. Composable - atomic plans build into complex workflows")
    print(f"  3. Extensible - easy to add new capabilities")
    print(f"  4. Reliable - proper Bluesky integration with error handling")
    
    print(f"\nNext steps:")
    print(f"  1. Test with real DiSPIM hardware using safety protocols")
    if not NAPARI_AVAILABLE:
        print(f"  2. Install napari for real-time visualization: pip install napari[all]")
        print(f"  3. Add image analysis for embryo detection")
        print(f"  4. Integrate with VLM for intelligent workflows")  
        print(f"  5. Extend to other microscopy systems")
    else:
        print(f"  2. Add image analysis for embryo detection")
        print(f"  3. Integrate with VLM for intelligent workflows")
        print(f"  4. Extend to other microscopy systems")
    
    print(f"\nThe transformation is complete:")
    print(f"  635-line Java monolith → Composable Bluesky atomic plans")
    print(f"  Device-specific code → Device-agnostic interfaces")
    print(f"  Rigid workflows → Flexible, extensible compositions")
    
    print(f"\nGently DiSPIM: Where atomic plans meet experimental flexibility! 🔬")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    main()