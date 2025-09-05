#!/usr/bin/env python
"""
Safe DiSPIM Hardware Connection Test
===================================

Phase 1 of hardware deployment: READ-ONLY connection test
This script safely connects to real DiSPIM hardware and reads all device states
without making any movements or changes.

SAFETY: This test is READ-ONLY - no hardware movements will be made.

Usage:
    Edit the mm_dir and config_file variables in main(), then run:
    python test_hardware_connection.py

What this test does:
1. Connects to real MM system with provided paths  
2. Creates all DiSPIM device objects
3. Reads current positions/states from all devices
4. Logs all values to establish baseline
5. Verifies describe() methods work with real hardware
6. Saves current state snapshot for later write-back test

What this test does NOT do:
- Make any device movements
- Change any settings
- Trigger any acquisitions
- Modify any hardware state
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Import gently components
from gently.devices import (
    DiSPIMPiezo,
    DiSPIMGalvo, 
    DiSPIMCamera,
    DiSPIMXYStage,
    DiSPIMLaserControl,
    DiSPIMLightSheet,
    DiSPIMSystem
)

# Import MM core
import pymmcore


def setup_logging():
    """Setup logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"hardware_connection_test_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def initialize_micromanager(mm_dir: str, config_file: str):
    """Initialize Micro-Manager core with real hardware"""
    print(f"Initializing Micro-Manager...")
    print(f"  MM Directory: {mm_dir}")
    print(f"  Config File: {config_file}")
    
    # Validate paths exist
    if not Path(mm_dir).exists():
        raise FileNotFoundError(f"Micro-Manager directory not found: {mm_dir}")
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Create MM core
    try:
        core = pymmcore.CMMCore()
        
        # Set MM path and load configuration
        core.setDeviceAdapterSearchPaths([mm_dir])
        core.loadSystemConfiguration(config_file)
        
        print(f"✓ Micro-Manager initialized successfully")
        print(f"  Loaded devices: {core.getLoadedDevices()}")
        
        return core
        
    except Exception as e:
        print(f"✗ Failed to initialize Micro-Manager: {e}")
        raise


def create_dispim_devices(core):
    """Create all DiSPIM device objects with real hardware"""
    print("\nCreating DiSPIM device objects...")
    
    devices = {}
    
    try:
        # Create individual devices with real MM device names
        # NOTE: These device names should match your actual MM configuration
        devices['piezo_a'] = DiSPIMPiezo("PiezoStage:P:34", core, name="piezo_A")
        devices['piezo_b'] = DiSPIMPiezo("PiezoStage:P:35", core, name="piezo_B") 
        print("✓ Piezo devices created")
        
        devices['galvo_a'] = DiSPIMGalvo("GalvoStage:G:34", core, name="galvo_A")
        devices['galvo_b'] = DiSPIMGalvo("GalvoStage:G:35", core, name="galvo_B")
        print("✓ Galvo devices created")
        
        devices['camera_a'] = DiSPIMCamera("CameraSide:A", core, name="camera_A")
        devices['camera_b'] = DiSPIMCamera("CameraSide:B", core, name="camera_B")
        print("✓ Camera devices created")
        
        devices['xy_stage'] = DiSPIMXYStage("XYStage", core, name="xy_stage")
        print("✓ XY stage device created")
        
        devices['laser'] = DiSPIMLaserControl(core, name="laser_control")
        print("✓ Laser control device created")
        
        # Create composite devices
        device_mapping = {
            'piezo_a': 'PiezoStage:P:34',
            'galvo_a': 'GalvoStage:G:34', 
            'camera_a': 'CameraSide:A'
        }
        
        devices['light_sheet_a'] = DiSPIMLightSheet('A', core, device_mapping, name="light_sheet_A")
        print("✓ Light sheet A device created")
        
        devices['dispim_system'] = DiSPIMSystem(core, name="dispim_system")
        print("✓ DiSPIM system device created")
        
        return devices
        
    except Exception as e:
        print(f"✗ Failed to create devices: {e}")
        print("NOTE: Device names may need to be adjusted to match your MM configuration")
        raise


def read_all_device_states(devices):
    """READ-ONLY: Read current state from all devices"""
    print("\n" + "="*60)
    print("READING ALL DEVICE STATES (READ-ONLY)")
    print("="*60)
    
    all_states = {}
    timestamp = time.time()
    
    for device_name, device in devices.items():
        print(f"\nReading {device_name}...")
        
        try:
            # Read current state
            state = device.read()
            description = device.describe()
            
            all_states[device_name] = {
                'state': state,
                'description': description,
                'timestamp': timestamp,
                'device_type': type(device).__name__
            }
            
            print(f"  ✓ State: {len(state)} signals")
            
            # Log key values for visibility
            for signal_name, signal_data in state.items():
                if isinstance(signal_data, dict) and 'value' in signal_data:
                    print(f"    {signal_name}: {signal_data['value']}")
                else:
                    print(f"    {signal_name}: {signal_data}")
            
        except Exception as e:
            print(f"  ✗ Failed to read {device_name}: {e}")
            all_states[device_name] = {'error': str(e), 'timestamp': timestamp}
    
    return all_states


def validate_hardware_connection(devices):
    """Validate that we can communicate with all hardware"""
    print("\n" + "="*60) 
    print("VALIDATING HARDWARE COMMUNICATION")
    print("="*60)
    
    validation_results = {}
    
    for device_name, device in devices.items():
        print(f"\nValidating {device_name}...")
        
        try:
            # Test basic communication
            state = device.read()
            desc = device.describe()
            
            # Check if we got real data
            has_data = bool(state)
            has_description = bool(desc)
            
            validation_results[device_name] = {
                'communication': True,
                'has_data': has_data,
                'has_description': has_description,
                'signal_count': len(state),
                'description_count': len(desc)
            }
            
            if has_data:
                print(f"  ✓ Communication successful ({len(state)} signals)")
            else:
                print(f"  ⚠ Communication successful but no data returned")
                
        except Exception as e:
            print(f"  ✗ Communication failed: {e}")
            validation_results[device_name] = {
                'communication': False,
                'error': str(e)
            }
    
    return validation_results


def save_baseline_state(all_states, validation_results):
    """Save current hardware state for future reference"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dispim_baseline_state_{timestamp}.json"
    
    baseline_data = {
        'timestamp': timestamp,
        'test_type': 'hardware_connection_baseline',
        'device_states': all_states,
        'validation_results': validation_results,
        'notes': 'READ-ONLY baseline state capture before any hardware movements'
    }
    
    with open(filename, 'w') as f:
        json.dump(baseline_data, f, indent=2, default=str)
    
    print(f"\n✓ Baseline state saved to: {filename}")
    return filename


def print_summary(validation_results, baseline_file):
    """Print test summary"""
    print("\n" + "="*60)
    print("HARDWARE CONNECTION TEST SUMMARY") 
    print("="*60)
    
    total_devices = len(validation_results)
    successful = sum(1 for r in validation_results.values() if r.get('communication', False))
    
    print(f"\nDevice Connection Results:")
    print(f"  Total devices tested: {total_devices}")
    print(f"  Successful connections: {successful}")
    print(f"  Failed connections: {total_devices - successful}")
    
    if successful > 0:
        print(f"\n✅ SUCCESS: Connected to {successful}/{total_devices} devices")
        print(f"✅ Baseline state captured: {baseline_file}")
        print(f"✅ Ready for Phase 2: Read-Write-Back Test")
    else:
        print(f"\n❌ FAILURE: Could not connect to any devices")
        print(f"❌ Check MM installation and device names")
    
    print(f"\nFailed devices:")
    for device_name, result in validation_results.items():
        if not result.get('communication', False):
            print(f"  - {device_name}: {result.get('error', 'Unknown error')}")
    
    print(f"\nNext steps:")
    print(f"  1. If failures occurred, check device names in code vs MM config")
    print(f"  2. If successful, proceed to: python test_read_write_back.py")
    print(f"  3. Consult colleague about safe operating limits")


def main():
    """Main hardware connection test"""
    print("Safe DiSPIM Hardware Connection Test")
    print("="*60)
    print("Phase 1: READ-ONLY connection and state reading")
    print("SAFETY: No hardware movements will be made")
    print()
    
    # Configure your MicroManager paths here
    mm_dir = "/opt/micromanager"  # Path to your MicroManager installation
    config_file = "config.cfg"    # Path to your MM configuration file
    
    print(f"Using MicroManager directory: {mm_dir}")
    print(f"Using configuration file: {config_file}")
    print("(Edit these paths in the script as needed)")
    print()
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting hardware connection test")
    
    try:
        # Initialize MM
        core = initialize_micromanager(mm_dir, config_file)
        
        # Create devices
        devices = create_dispim_devices(core)
        
        # Read all states (READ-ONLY)
        all_states = read_all_device_states(devices)
        
        # Validate communication
        validation_results = validate_hardware_connection(devices)
        
        # Save baseline
        baseline_file = save_baseline_state(all_states, validation_results)
        
        # Print summary
        print_summary(validation_results, baseline_file)
        
        logging.info("Hardware connection test completed successfully")
        
    except Exception as e:
        logging.error(f"Hardware connection test failed: {e}")
        print(f"\n❌ CRITICAL FAILURE: {e}")
        print(f"❌ Check log file: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()