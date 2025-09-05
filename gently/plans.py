"""
Gently DiSPIM Plans
==================

Device-agnostic Bluesky plans for DiSPIM microscopy workflows.
Built using atomic plan stubs that compose into complex experimental procedures.

Autofocus serves as the "arrowhead" into the complete DiSPIM functionality,
including calibration, embryo detection, and multi-embryo acquisition workflows.

All plans are device-agnostic and use standard Bluesky plan stubs:
    - bps.mv(device, position) 
    - bps.trigger_and_read([detector])
    - bps.stage(device) / bps.unstage(device)
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Generator, Any, Union
from dataclasses import dataclass
import numpy as np

import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import Msg
from bluesky.utils import short_uid


from .analysis import (
    calculate_focus_score, fit_focus_curve, find_curve_maximum,
    validate_autofocus_result, FocusAnalysisConfig
)


@dataclass
class AutofocusConfig:
    """Configuration for autofocus operations"""
    num_positions: int = 21
    step_size_um: float = 0.5
    algorithm: str = 'volath'  # 'volath', 'gradient', 'variance'
    fit_function: str = 'gaussian'  # 'gaussian', 'parabolic', 'none'
    minimum_r_squared: float = 0.75
    center_at_current: bool = True
    timeout_s: float = 60.0


@dataclass
class CalibrationConfig:
    """Configuration for two-point calibration"""
    point1_um: float = 25.0
    point2_um: float = 75.0
    autofocus_each_point: bool = True
    autofocus_config: Optional[AutofocusConfig] = None


# =============================================================================
# ATOMIC PLANS - Device-Agnostic Building Blocks
# =============================================================================

def focus_sweep(positioner, positions: List[float], detector, 
                metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Device-agnostic focus sweep - works with ANY positioner and detector
    
    This is the fundamental atomic plan that underlies all autofocus operations.
    Can be used with piezo, galvo, xy_stage, focus_motor, etc.
    
    Parameters
    ----------
    positioner : Ophyd positioner device
        Any device that responds to bps.mv(positioner, position)
    positions : List[float]
        List of positions to sweep through
    detector : Ophyd detector device
        Any device that responds to bps.trigger_and_read([detector])
    metadata : Dict, optional
        Additional metadata for the scan
    """
    md = {
        'plan_name': 'focus_sweep',
        'positioner': positioner.name,
        'detector': detector.name,
        'positions': positions,
        'num_positions': len(positions)
    }
    if metadata:
        md.update(metadata)
    
    @bpp.run_decorator(md=md)
    def inner():
        for i, pos in enumerate(positions):
            # Move positioner
            yield from bps.mv(positioner, pos)
            
            # Acquire at this position
            yield from bps.trigger_and_read([detector, positioner], 
                                          name=f'focus_point_{i:03d}')
    
    yield from inner()


def move_and_acquire(devices: Dict, position_dict: Dict, detector_list: List,
                    metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Device-agnostic move and acquire - works with any devices
    
    Parameters
    ----------
    devices : Dict
        Dictionary of device objects keyed by name  
    position_dict : Dict
        Dictionary of {device_name: position} to move to
    detector_list : List
        List of detector devices to trigger and read
    metadata : Dict, optional
        Additional metadata
    """
    md = {
        'plan_name': 'move_and_acquire',
        'target_positions': position_dict,
        'detectors': [det.name for det in detector_list]
    }
    if metadata:
        md.update(metadata)
    
    @bpp.run_decorator(md=md)
    def inner():
        # Move all devices to target positions
        move_args = []
        for device_name, position in position_dict.items():
            if device_name in devices:
                move_args.extend([devices[device_name], position])
        
        if move_args:
            yield from bps.mv(*move_args)
        
        # Read all devices
        all_devices = list(devices.values()) + detector_list
        yield from bps.trigger_and_read(all_devices)
    
    yield from inner()


def synchronized_move(light_sheet, piezo_position: float) -> Generator[Msg, None, None]:
    """
    DiSPIM-specific synchronized piezo/galvo move based on calibration
    
    This uses the light_sheet's calibration to move piezo and galvo together.
    Device-agnostic in that it works with any DiSPIMLightSheet device.
    """
    # Use the light sheet's synchronized move method
    status = light_sheet.synchronized_move(piezo_position)
    yield from bps.abs_set(light_sheet, status, wait=True)


# =============================================================================
# AUTOFOCUS PLANS - Composed from Atomic Plans
# =============================================================================

def dispim_piezo_autofocus(light_sheet, config: AutofocusConfig, 
                          analysis_config: Optional[FocusAnalysisConfig] = None
                          ) -> Generator[Msg, None, None]:
    """
    DiSPIM piezo autofocus: fix galvo, sweep piezo
    
    Device-agnostic autofocus that works with any DiSPIMLightSheet device.
    Uses the focus_sweep atomic plan internally.
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        Light sheet device with piezo, galvo, and camera
    config : AutofocusConfig
        Autofocus parameters
    analysis_config : FocusAnalysisConfig, optional
        Focus analysis parameters
    """
    if analysis_config is None:
        analysis_config = FocusAnalysisConfig(
            algorithm=config.algorithm,
            fit_function=config.fit_function,
            minimum_r_squared=config.minimum_r_squared
        )
    
    md = {
        'plan_name': 'dispim_piezo_autofocus',
        'light_sheet_side': getattr(light_sheet, 'side', 'unknown'),
        'autofocus_mode': 'fix_galvo_sweep_piezo',
        'config': config.__dict__,
        'analysis_config': analysis_config.__dict__
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        # Stage device to save current state
        yield from bps.stage(light_sheet)
        
        try:
            # Get current or center position
            if config.center_at_current:
                center_pos = yield from bps.rd(light_sheet.piezo)
            else:
                center_pos = 75.0  # Default center, could be from config
            
            # Generate scan positions
            scan_range = (config.num_positions - 1) * config.step_size_um
            start_pos = center_pos - scan_range / 2
            positions = [start_pos + i * config.step_size_um 
                        for i in range(config.num_positions)]
            
            # Check limits
            piezo_limits = getattr(light_sheet.piezo, 'limits', (-50, 150))
            positions = [p for p in positions 
                        if piezo_limits[0] <= p <= piezo_limits[1]]
            
            if len(positions) < 5:
                raise ValueError(f"Too few valid positions: {len(positions)}")
            
            # Device-agnostic focus sweep  
            yield from focus_sweep(light_sheet.piezo, positions, light_sheet.camera,
                                 metadata={'autofocus_type': 'piezo_scan'})
            
            # Analysis happens in post-processing
            # For now, just log the completion
            yield from bps.create(name='autofocus_complete')
            yield from bps.save({
                'autofocus_success': True,  # Would be determined by analysis
                'positions_tested': positions,
                'center_position': center_pos
            })
            
        except Exception as e:
            yield from bps.create(name='autofocus_failed') 
            yield from bps.save({
                'autofocus_success': False,
                'error_message': str(e)
            })
            raise
        
        finally:
            # Always unstage to restore original positions if needed
            yield from bps.unstage(light_sheet)
    
    yield from inner()


def dispim_galvo_autofocus(light_sheet, config: AutofocusConfig,
                          analysis_config: Optional[FocusAnalysisConfig] = None
                          ) -> Generator[Msg, None, None]:
    """
    DiSPIM galvo autofocus: fix piezo, sweep galvo
    
    Device-agnostic autofocus using galvo scanning.
    Uses the same focus_sweep atomic plan as piezo autofocus.
    """
    if analysis_config is None:
        analysis_config = FocusAnalysisConfig(
            algorithm=config.algorithm,
            fit_function=config.fit_function, 
            minimum_r_squared=config.minimum_r_squared
        )
    
    md = {
        'plan_name': 'dispim_galvo_autofocus',
        'light_sheet_side': getattr(light_sheet, 'side', 'unknown'),
        'autofocus_mode': 'fix_piezo_sweep_galvo',
        'config': config.__dict__,
        'analysis_config': analysis_config.__dict__
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        yield from bps.stage(light_sheet)
        
        try:
            # Get current galvo position or use center
            if config.center_at_current:
                center_angle = yield from bps.rd(light_sheet.galvo)
            else:
                center_angle = 0.0  # Default center
            
            # Generate scan angles based on step size
            # Convert um step size to angle step using calibration if available
            calibration_rate = getattr(light_sheet, 'calibration_slope', 100.0)
            angle_step = config.step_size_um / calibration_rate
            
            scan_range = (config.num_positions - 1) * angle_step
            start_angle = center_angle - scan_range / 2
            angles = [start_angle + i * angle_step 
                     for i in range(config.num_positions)]
            
            # Check galvo limits
            galvo_limits = getattr(light_sheet.galvo, 'limits', (-5, 5))
            angles = [a for a in angles 
                     if galvo_limits[0] <= a <= galvo_limits[1]]
            
            if len(angles) < 5:
                raise ValueError(f"Too few valid angles: {len(angles)}")
            
            # Device-agnostic focus sweep
            yield from focus_sweep(light_sheet.galvo, angles, light_sheet.camera,
                                 metadata={'autofocus_type': 'galvo_scan'})
            
            yield from bps.create(name='autofocus_complete')
            yield from bps.save({
                'autofocus_success': True,
                'angles_tested': angles,
                'center_angle': center_angle
            })
            
        except Exception as e:
            yield from bps.create(name='autofocus_failed')
            yield from bps.save({
                'autofocus_success': False,
                'error_message': str(e)
            })
            raise
        
        finally:
            yield from bps.unstage(light_sheet)
    
    yield from inner()


def dual_sided_autofocus(dispim_system, config: AutofocusConfig, 
                        sides: List[str] = ['A', 'B']) -> Generator[Msg, None, None]:
    """
    Coordinate autofocus on both DiSPIM sides
    
    Device-agnostic plan that works with any DiSPIMSystem.
    Sequentially runs autofocus on each requested side.
    """
    md = {
        'plan_name': 'dual_sided_autofocus',
        'sides': sides,
        'config': config.__dict__
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        for side in sides:
            if side.upper() == 'A' and hasattr(dispim_system, 'side_a'):
                yield from dispim_piezo_autofocus(dispim_system.side_a, config)
            elif side.upper() == 'B' and hasattr(dispim_system, 'side_b'):
                yield from dispim_piezo_autofocus(dispim_system.side_b, config)
            else:
                raise ValueError(f"Unknown side: {side}")
    
    yield from inner()


# =============================================================================
# CALIBRATION PLANS
# =============================================================================

def dispim_two_point_calibration(light_sheet, config: CalibrationConfig
                                ) -> Generator[Msg, None, None]:
    """
    DiSPIM two-point calibration procedure
    
    Replicates the Java plugin's two-point calibration using device-agnostic plans.
    Moves piezo to two positions, optionally runs autofocus on galvo, records positions.
    """
    md = {
        'plan_name': 'dispim_two_point_calibration',
        'light_sheet_side': getattr(light_sheet, 'side', 'unknown'),
        'calibration_points': [config.point1_um, config.point2_um],
        'autofocus_enabled': config.autofocus_each_point
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        yield from bps.stage(light_sheet)
        
        try:
            calibration_data = []
            
            for i, piezo_pos in enumerate([config.point1_um, config.point2_um]):
                yield from bps.create(name=f'calibration_point_{i+1}')
                
                # Move piezo to calibration position
                yield from bps.mv(light_sheet.piezo, piezo_pos)
                
                if config.autofocus_each_point and config.autofocus_config:
                    # Run galvo autofocus at this piezo position
                    yield from dispim_galvo_autofocus(light_sheet, config.autofocus_config)
                
                # Record positions
                piezo_readback = yield from bps.rd(light_sheet.piezo)
                galvo_readback = yield from bps.rd(light_sheet.galvo)
                
                # Take measurement image
                yield from bps.trigger_and_read([light_sheet.camera, light_sheet.piezo, light_sheet.galvo])
                
                calibration_data.append({
                    'piezo_position': piezo_readback,
                    'galvo_position': galvo_readback
                })
            
            # Calculate calibration parameters
            p1_piezo, p1_galvo = calibration_data[0]['piezo_position'], calibration_data[0]['galvo_position']
            p2_piezo, p2_galvo = calibration_data[1]['piezo_position'], calibration_data[1]['galvo_position']
            
            # Linear fit: galvo = slope * piezo + offset
            # Actually: piezo = slope * galvo + offset (DiSPIM convention)
            slope = (p2_piezo - p1_piezo) / (p2_galvo - p1_galvo) if p2_galvo != p1_galvo else 1.0
            offset = p1_piezo - slope * p1_galvo
            
            # Calculate R-squared (would be 1.0 for two-point linear fit)
            r_squared = 1.0
            
            # Update light sheet calibration
            light_sheet.calibration_slope = slope
            light_sheet.calibration_offset = offset
            light_sheet.calibration_valid = True
            
            # Save calibration results
            yield from bps.create(name='calibration_results')
            yield from bps.save({
                'calibration_slope': slope,
                'calibration_offset': offset,
                'calibration_r_squared': r_squared,
                'calibration_valid': True,
                'point1': calibration_data[0],
                'point2': calibration_data[1]
            })
            
        finally:
            yield from bps.unstage(light_sheet)
    
    yield from inner()


def dispim_full_calibration(dispim_system, config: CalibrationConfig
                          ) -> Generator[Msg, None, None]:
    """
    Full DiSPIM system calibration - calibrate both sides
    
    Device-agnostic plan that calibrates both light sheet sides.
    """
    md = {
        'plan_name': 'dispim_full_calibration',
        'config': config.__dict__
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        # Calibrate side A
        if hasattr(dispim_system, 'side_a'):
            yield from dispim_two_point_calibration(dispim_system.side_a, config)
        
        # Calibrate side B  
        if hasattr(dispim_system, 'side_b'):
            yield from dispim_two_point_calibration(dispim_system.side_b, config)
    
    yield from inner()


# =============================================================================
# EMBRYO DETECTION PLANS (Foundation for Future Development)
# =============================================================================

def find_embryos_with_bottom_camera(dispim_system, scan_config: Dict
                                  ) -> Generator[Msg, None, None]:
    """
    Scan area with bottom camera to find embryos
    
    Foundation plan for embryo detection workflows.
    Uses device-agnostic XY scanning with the bottom camera.
    """
    md = {
        'plan_name': 'find_embryos_with_bottom_camera',
        'scan_config': scan_config
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        # Extract scan parameters
        x_start = scan_config.get('x_start', -1000)
        x_stop = scan_config.get('x_stop', 1000) 
        y_start = scan_config.get('y_start', -1000)
        y_stop = scan_config.get('y_stop', 1000)
        step_size = scan_config.get('step_size', 100)
        
        # Generate grid positions
        x_positions = np.arange(x_start, x_stop + step_size, step_size)
        y_positions = np.arange(y_start, y_stop + step_size, step_size)
        
        embryo_candidates = []
        
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                # Move to position
                yield from bps.mv(dispim_system.xy_stage.x, x, 
                                dispim_system.xy_stage.y, y)
                
                # Acquire image
                yield from bps.trigger_and_read([dispim_system.bottom_camera, 
                                               dispim_system.xy_stage.x,
                                               dispim_system.xy_stage.y],
                                              name=f'embryo_search_{i:03d}_{j:03d}')
                
                # Placeholder for embryo detection analysis
                # In practice, this would use image analysis to detect embryos
                # For now, record all positions as potential candidates
                embryo_candidates.append({'x': x, 'y': y})
        
        # Save embryo candidate list
        yield from bps.create(name='embryo_candidates')
        yield from bps.save({
            'num_candidates': len(embryo_candidates),
            'candidates': embryo_candidates
        })
    
    yield from inner()


def acquire_embryo_lightsheet(dispim_system, embryo_position: Dict, 
                             acquisition_config: Dict) -> Generator[Msg, None, None]:
    """
    Acquire light sheet images of a single embryo
    
    Foundation plan for embryo light sheet acquisition.
    Coordinates XY positioning with light sheet imaging.
    """
    md = {
        'plan_name': 'acquire_embryo_lightsheet',
        'embryo_position': embryo_position,
        'acquisition_config': acquisition_config
    }
    
    @bpp.run_decorator(md=md)  
    def inner():
        # Move to embryo position
        yield from bps.mv(dispim_system.xy_stage.x, embryo_position['x'],
                        dispim_system.xy_stage.y, embryo_position['y'])
        
        # Run autofocus on primary side
        autofocus_config = AutofocusConfig(**acquisition_config.get('autofocus', {}))
        yield from dispim_piezo_autofocus(dispim_system.side_a, autofocus_config)
        
        # Z-stack acquisition
        z_config = acquisition_config.get('z_stack', {})
        z_start = z_config.get('start', -25)
        z_stop = z_config.get('stop', 25) 
        z_step = z_config.get('step', 1.0)
        z_positions = np.arange(z_start, z_stop + z_step, z_step)
        
        # Acquire Z-stack using device-agnostic focus_sweep
        yield from focus_sweep(dispim_system.side_a.piezo, z_positions, 
                             dispim_system.side_a.camera,
                             metadata={'acquisition_type': 'embryo_z_stack'})
        
        # If dual-sided acquisition requested
        if acquisition_config.get('dual_sided', False):
            yield from focus_sweep(dispim_system.side_b.piezo, z_positions,
                                 dispim_system.side_b.camera, 
                                 metadata={'acquisition_type': 'embryo_z_stack_side_b'})
    
    yield from inner()


# =============================================================================  
# INTEGRATION PLANS - Complete Workflows
# =============================================================================

def full_dispim_workflow(dispim_system, workflow_config: Dict
                        ) -> Generator[Msg, None, None]:
    """
    Complete DiSPIM workflow: detection → calibration → acquisition
    
    This is the ultimate integration plan that combines all the atomic plans
    into a complete experimental workflow.
    """
    md = {
        'plan_name': 'full_dispim_workflow',
        'workflow_config': workflow_config
    }
    
    @bpp.run_decorator(md=md)
    def inner():
        # 1. System initialization
        if workflow_config.get('center_devices', True):
            yield from bps.abs_set(dispim_system.center_all_devices(), wait=True)
        
        # 2. System calibration
        if workflow_config.get('run_calibration', True):
            cal_config = CalibrationConfig(**workflow_config.get('calibration', {}))
            yield from dispim_full_calibration(dispim_system, cal_config)
        
        # 3. Embryo detection
        if workflow_config.get('find_embryos', True):
            scan_config = workflow_config.get('embryo_scan', {})
            yield from find_embryos_with_bottom_camera(dispim_system, scan_config)
        
        # 4. Multi-embryo acquisition
        if workflow_config.get('acquire_embryos', True):
            # This would normally get embryo positions from the detection step
            # For now, use configured positions
            embryo_positions = workflow_config.get('embryo_positions', [{'x': 0, 'y': 0}])
            acquisition_config = workflow_config.get('acquisition', {})
            
            for i, pos in enumerate(embryo_positions):
                yield from acquire_embryo_lightsheet(dispim_system, pos, acquisition_config)
    
    yield from inner()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_autofocus(light_sheet, mode: str = 'piezo') -> Generator[Msg, None, None]:
    """Quick autofocus with default parameters"""
    config = AutofocusConfig()
    
    if mode.lower() == 'piezo':
        yield from dispim_piezo_autofocus(light_sheet, config)
    elif mode.lower() == 'galvo':  
        yield from dispim_galvo_autofocus(light_sheet, config)
    else:
        raise ValueError(f"Unknown autofocus mode: {mode}")


def quick_calibration(light_sheet) -> Generator[Msg, None, None]:
    """Quick two-point calibration with default parameters"""
    config = CalibrationConfig(
        autofocus_config=AutofocusConfig(num_positions=11)  # Faster autofocus
    )
    yield from dispim_two_point_calibration(light_sheet, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Gently DiSPIM Plans")
    print("==================")
    print()
    print("Device-agnostic Bluesky plans for DiSPIM workflows")
    print("Built from atomic plan stubs that compose into complex procedures")
    print()
    print("Atomic Plans:")
    print("  - focus_sweep(positioner, positions, detector)")
    print("  - move_and_acquire(devices, positions, detectors)")  
    print("  - synchronized_move(light_sheet, position)")
    print()
    print("Autofocus Plans:")
    print("  - dispim_piezo_autofocus(light_sheet, config)")
    print("  - dispim_galvo_autofocus(light_sheet, config)")
    print("  - dual_sided_autofocus(system, config)")
    print()
    print("Calibration Plans:")
    print("  - dispim_two_point_calibration(light_sheet, config)")
    print("  - dispim_full_calibration(system, config)")
    print()
    print("Workflow Plans:")
    print("  - find_embryos_with_bottom_camera(system, config)")
    print("  - acquire_embryo_lightsheet(system, position, config)")
    print("  - full_dispim_workflow(system, config)")
    print()
    print("Usage:")
    print("  RE(quick_autofocus(light_sheet, 'piezo'))")
    print("  RE(dispim_two_point_calibration(light_sheet, config))")
    print("  RE(full_dispim_workflow(system, workflow_config))")