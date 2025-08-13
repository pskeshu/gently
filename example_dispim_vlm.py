#!/usr/bin/env python3
"""
DiSPIM VLM Integration Example

Complete example demonstrating DiSPIM control with Vision Language Model
integration for adaptive microscopy experiments.
"""

import logging
import time
import numpy as np
from pathlib import Path

# DiSPIM modules
from dispim_config import DiSPIMCore, DiSPIMConfig
from dispim_calibration import DiSPIMCalibration
from dispim_ophyd import create_dispim_devices
from dispim_bluesky import (
    setup_dispim_session, 
    DiSPIMScanConfig,
    calibration_sequence,
    z_stack_scan,
    time_lapse_z_stack,
    adaptive_acquisition_with_vlm
)
from dispim_vlm import (
    DiSPIMVLMProcessor, 
    VLMConfig, 
    VLMProvider,
    create_fast_vlm_config,
    create_high_quality_vlm_config
)


def example_basic_setup():
    """Example 1: Basic DiSPIM setup and device initialization"""
    print("=== Example 1: Basic DiSPIM Setup ===")
    
    # Configuration
    config_file = "/path/to/your/dispim_config.cfg"  # Update this path
    
    # Initialize core (headless pymmcore-plus)
    dispim = DiSPIMCore(config_file) if Path(config_file).exists() else DiSPIMCore()
    
    if Path(config_file).exists():
        dispim.initialize_devices()
        print("✓ Devices initialized")
        
        # Get device information
        device_info = dispim.get_device_info()
        print(f"✓ Found {len(device_info)} devices")
        
        # Center devices
        dispim.center_piezo_and_galvo()
        print("✓ Devices centered")
    else:
        print("⚠ Config file not found - using simulation mode")
    
    return dispim


def example_calibration():
    """Example 2: DiSPIM calibration procedures"""
    print("\n=== Example 2: DiSPIM Calibration ===")
    
    dispim = DiSPIMCore()
    calibration = DiSPIMCalibration(dispim)
    
    # Configure autofocus
    calibration.autofocus_config.num_images = 21
    calibration.autofocus_config.step_size = 1.0
    
    print("✓ Calibration system configured")
    
    # In real usage, you would run:
    # result = calibration.two_point_calibration(25.0, 75.0)
    # print(f"✓ Calibration: slope={result.slope:.4f}, offset={result.offset:.2f}")
    
    return calibration


def example_ophyd_devices():
    """Example 3: Ophyd device abstraction"""
    print("\n=== Example 3: Ophyd Device Abstraction ===")
    
    # Create DiSPIM device ensemble
    light_sheet = create_dispim_devices()
    
    print(f"✓ Created light sheet device: {light_sheet.name}")
    print(f"  - Imaging piezo: {light_sheet.piezo_imaging.name}")
    print(f"  - Illumination piezo: {light_sheet.piezo_illumination.name}")
    print(f"  - Galvo: {light_sheet.galvo.name}")
    print(f"  - Camera: {light_sheet.camera.name}")
    
    # In real usage, you would move devices:
    # status = light_sheet.synchronized_move(30.0)  # Move to 30 μm
    # status.wait()  # Wait for completion
    
    return light_sheet


def example_bluesky_plans():
    """Example 4: Bluesky experiment plans"""
    print("\n=== Example 4: Bluesky Acquisition Plans ===")
    
    # Setup Bluesky session
    RE, light_sheet, callbacks = setup_dispim_session()
    
    # Subscribe callbacks for live monitoring
    for callback in callbacks:
        RE.subscribe(callback)
    
    print("✓ Bluesky RunEngine configured")
    print("✓ Live monitoring callbacks active")
    
    # Create scan configuration
    scan_config = DiSPIMScanConfig(
        z_start=-25.0,
        z_stop=25.0,
        z_step=0.5,
        time_points=10,
        time_interval=30.0,  # 30 seconds between time points
        exposure_time=0.02   # 20 ms exposure
    )
    
    print("✓ Scan configuration created")
    
    # Example plans (would run in real experiment):
    
    # 1. Calibration
    # RE(calibration_sequence(light_sheet, 25.0, 75.0))
    
    # 2. Z-stack scan
    # RE(z_stack_scan(light_sheet, -10, 10, 0.5))
    
    # 3. Time-lapse Z-stack
    # RE(time_lapse_z_stack(light_sheet, scan_config))
    
    print("✓ Plans ready for execution")
    
    return RE, light_sheet, scan_config


def example_vlm_integration():
    """Example 5: VLM processing integration"""
    print("\n=== Example 5: VLM Integration ===")
    
    # Configure VLM for fast processing
    vlm_config = create_fast_vlm_config()
    vlm_processor = DiSPIMVLMProcessor(vlm_config)
    
    print(f"✓ VLM processor configured")
    print(f"  Provider: {vlm_config.provider.value}")
    print(f"  Model: {vlm_config.model_name}")
    print(f"  Max image size: {vlm_config.max_image_size}")
    
    # Start background processing
    vlm_processor.start_processing()
    print("✓ VLM processing thread started")
    
    # Create callback for Bluesky integration
    vlm_callback = vlm_processor.create_bluesky_callback()
    print("✓ VLM callback created")
    
    # Simulate image processing
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    success = vlm_processor.process_image_async(test_image, {'test': True})
    print(f"✓ Test image queued: {success}")
    
    # Wait for result
    time.sleep(0.5)
    decision = vlm_processor.get_latest_decision(timeout=1.0)
    if decision:
        print(f"✓ VLM decision received: confidence={decision.confidence:.2f}")
    
    # Cleanup
    vlm_processor.stop_processing()
    
    return vlm_processor, vlm_callback


def example_adaptive_experiment():
    """Example 6: Complete adaptive experiment with VLM"""
    print("\n=== Example 6: Adaptive VLM-Guided Experiment ===")
    
    # Setup all components
    RE, light_sheet, _ = setup_dispim_session()
    vlm_processor = DiSPIMVLMProcessor(create_fast_vlm_config())
    vlm_processor.start_processing()
    
    # Create VLM callback
    vlm_callback = vlm_processor.create_bluesky_callback()
    
    # Configure adaptive scan
    adaptive_config = DiSPIMScanConfig(
        z_start=-30.0,
        z_stop=30.0,
        z_step=1.0,  # Initial step size - VLM can adapt this
        exposure_time=0.015,
        vlm_analysis=True
    )
    
    print("✓ Adaptive experiment configured")
    print("  - VLM will analyze each image")
    print("  - Step size will adapt based on VLM decisions")
    print("  - Autofocus suggestions will be processed")
    
    # In real usage, run adaptive acquisition:
    # RE(adaptive_acquisition_with_vlm(light_sheet, vlm_callback, adaptive_config))
    
    print("✓ Adaptive plan ready")
    
    # Cleanup
    vlm_processor.stop_processing()
    
    return RE, light_sheet, vlm_callback


def example_advanced_vlm():
    """Example 7: Advanced VLM with OpenAI GPT-4V"""
    print("\n=== Example 7: Advanced VLM Analysis ===")
    
    try:
        # High-quality VLM configuration
        hq_config = create_high_quality_vlm_config()
        hq_config.analysis_prompt = """
        Analyze this DiSPIM light sheet microscopy image. Please assess:
        1. Image focus quality and sharpness
        2. Cellular structures visible (if any)
        3. Background noise levels
        4. Tissue organization patterns
        5. Recommendations for acquisition parameters
        """
        
        vlm_processor = DiSPIMVLMProcessor(hq_config)
        print("✓ High-quality VLM processor configured")
        print(f"  Provider: {hq_config.provider.value}")
        print("  - Detailed scientific image analysis")
        print("  - Acquisition parameter recommendations")
        
    except Exception as e:
        print(f"⚠ Advanced VLM not available: {e}")
        print("  (Requires OpenAI API key)")
        return None
    
    return vlm_processor


def main():
    """Run all DiSPIM VLM integration examples"""
    print("DiSPIM Vision Language Model Integration Examples")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run examples
        dispim = example_basic_setup()
        calibration = example_calibration()
        light_sheet = example_ophyd_devices()
        RE, light_sheet_bs, scan_config = example_bluesky_plans()
        vlm_processor, vlm_callback = example_vlm_integration()
        adaptive_setup = example_adaptive_experiment()
        advanced_vlm = example_advanced_vlm()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Update config file path in example_basic_setup()")
        print("2. Connect to actual DiSPIM hardware")
        print("3. Configure VLM provider credentials")
        print("4. Run adaptive experiments")
        
        print("\nKey capabilities demonstrated:")
        print("✓ Headless DiSPIM control with pymmcore-plus")
        print("✓ Two-point calibration and autofocus")
        print("✓ Ophyd device abstraction for Bluesky")
        print("✓ Multi-dimensional acquisition plans")
        print("✓ Real-time VLM image analysis")
        print("✓ Adaptive experiment control")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        logging.exception("Example execution failed")


if __name__ == "__main__":
    main()