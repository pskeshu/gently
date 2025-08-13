"""
DiSPIM Bluesky Plans

Bluesky experiment plans for DiSPIM light sheet microscopy including:
- Multi-dimensional acquisition plans
- Calibration and setup procedures
- Time-lapse and Z-stack acquisition
- Real-time processing integration
"""

import time
import logging
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import numpy as np

try:
    import bluesky.plans as bp
    import bluesky.plan_stubs as bps
    from bluesky import RunEngine
    from bluesky.utils import short_uid
    from bluesky.callbacks import LiveTable, LivePlot
    from databroker import Broker
except ImportError:
    raise ImportError("bluesky is required. Install with: pip install bluesky")

from dispim_ophyd import DiSPIMLightSheet, create_dispim_devices


@dataclass
class DiSPIMScanConfig:
    """Configuration for DiSPIM scanning experiments"""
    # Z-stack parameters
    z_start: float = -25.0  # μm
    z_stop: float = 25.0    # μm
    z_step: float = 0.5     # μm
    
    # Time-lapse parameters
    time_points: int = 1
    time_interval: float = 60.0  # seconds
    
    # Multi-view parameters
    dual_sided: bool = True
    view_angles: List[float] = None  # degrees
    
    # Acquisition parameters
    exposure_time: float = 0.01  # seconds
    channels: List[str] = None   # laser channels
    
    # Processing parameters
    live_processing: bool = False
    vlm_analysis: bool = False


def calibration_sequence(light_sheet: DiSPIMLightSheet,
                        point1: float = 25.0, 
                        point2: float = 75.0) -> Generator[Any, Any, None]:
    """
    Bluesky plan for DiSPIM two-point calibration
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        DiSPIM device ensemble
    point1, point2 : float
        Calibration positions in μm
    """
    
    def calibration_plan():
        # Record start of calibration
        yield from bps.open_run(md={'plan_name': 'dispim_calibration',
                                   'purpose': 'light_sheet_alignment',
                                   'calibration_points': [point1, point2]})
        
        try:
            # Center devices first
            yield from bps.mv(light_sheet.piezo_imaging, 
                             light_sheet.core.config.piezo_center)
            yield from bps.mv(light_sheet.galvo, 
                             light_sheet.core.config.galvo_center)
            
            # Calibration point 1
            yield from bps.mv(light_sheet.piezo_imaging, point1)
            
            # Run autofocus on galvo
            optimal_galvo1 = light_sheet.run_autofocus('galvo')
            yield from bps.mv(light_sheet.galvo, optimal_galvo1)
            
            # Take measurement image
            yield from bps.trigger_and_read([light_sheet.camera], 
                                          name='calibration_point_1')
            
            # Calibration point 2
            yield from bps.mv(light_sheet.piezo_imaging, point2)
            
            # Run autofocus on galvo
            optimal_galvo2 = light_sheet.run_autofocus('galvo')
            yield from bps.mv(light_sheet.galvo, optimal_galvo2)
            
            # Take measurement image
            yield from bps.trigger_and_read([light_sheet.camera], 
                                          name='calibration_point_2')
            
            # Calculate and apply calibration
            result = light_sheet.calibrate_two_point(point1, point2, auto_focus=False)
            
            # Store calibration results in metadata
            yield from bps.create(name='primary')  # Create event document
            yield from bps.save({'calibration_slope': result.slope,
                               'calibration_offset': result.offset,
                               'calibration_r_squared': result.r_squared,
                               'calibration_valid': result.is_valid})
            
        finally:
            yield from bps.close_run()
    
    return calibration_plan()


def z_stack_scan(light_sheet: DiSPIMLightSheet,
                 z_start: float = -25.0,
                 z_stop: float = 25.0, 
                 z_step: float = 0.5,
                 per_step: Optional[Generator] = None) -> Generator[Any, Any, None]:
    """
    Bluesky plan for DiSPIM Z-stack acquisition
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        DiSPIM device ensemble
    z_start, z_stop, z_step : float
        Z-scanning parameters in μm
    per_step : Generator, optional
        Additional actions to perform at each Z position
    """
    
    def z_scan_plan():
        num_points = int(abs(z_stop - z_start) / z_step) + 1
        z_positions = np.linspace(z_start, z_stop, num_points)
        
        md = {
            'plan_name': 'dispim_z_stack',
            'z_start': z_start,
            'z_stop': z_stop, 
            'z_step': z_step,
            'num_z_points': len(z_positions)
        }
        
        yield from bp.scan([light_sheet.camera], 
                          light_sheet.piezo_imaging,
                          z_start, z_stop, num_points,
                          per_step=per_step,
                          md=md)
    
    return z_scan_plan()


def dual_view_acquisition(light_sheet: DiSPIMLightSheet,
                         z_range: float = 50.0,
                         z_step: float = 0.5) -> Generator[Any, Any, None]:
    """
    Bluesky plan for dual-view DiSPIM acquisition
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        DiSPIM device ensemble
    z_range : float
        Total Z range in μm
    z_step : float
        Z step size in μm
    """
    
    def dual_view_plan():
        z_center = light_sheet.core.config.piezo_center
        z_start = z_center - z_range/2
        z_stop = z_center + z_range/2
        
        md = {
            'plan_name': 'dispim_dual_view',
            'acquisition_type': 'dual_sided_light_sheet',
            'z_range': z_range,
            'z_step': z_step
        }
        
        yield from bps.open_run(md=md)
        
        try:
            # View A acquisition
            yield from bps.create(name='view_A_start')
            
            # Z-stack for view A
            for z_pos in np.arange(z_start, z_stop + z_step, z_step):
                yield from light_sheet.synchronized_move(z_pos)
                yield from bps.trigger_and_read([light_sheet.camera],
                                              name=f'view_A_z_{z_pos:.1f}')
            
            # Rotate sample or switch illumination for View B
            # (Implementation depends on specific hardware setup)
            yield from bps.create(name='view_B_start')
            
            # Z-stack for view B
            for z_pos in np.arange(z_start, z_stop + z_step, z_step):
                yield from light_sheet.synchronized_move(z_pos)
                yield from bps.trigger_and_read([light_sheet.camera],
                                              name=f'view_B_z_{z_pos:.1f}')
                
        finally:
            yield from bps.close_run()
    
    return dual_view_plan()


def time_lapse_z_stack(light_sheet: DiSPIMLightSheet,
                       config: DiSPIMScanConfig) -> Generator[Any, Any, None]:
    """
    Bluesky plan for time-lapse Z-stack acquisition
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        DiSPIM device ensemble
    config : DiSPIMScanConfig
        Scan configuration parameters
    """
    
    def time_lapse_plan():
        md = {
            'plan_name': 'dispim_time_lapse_z_stack',
            'time_points': config.time_points,
            'time_interval': config.time_interval,
            'z_start': config.z_start,
            'z_stop': config.z_stop,
            'z_step': config.z_step,
            'exposure_time': config.exposure_time
        }
        
        yield from bps.open_run(md=md)
        
        try:
            for t in range(config.time_points):
                # Create time point marker
                yield from bps.create(name=f'time_point_{t}')
                
                # Set exposure time
                light_sheet.camera.exposure_time = config.exposure_time
                
                # Z-stack acquisition
                z_positions = np.arange(config.z_start, 
                                       config.z_stop + config.z_step, 
                                       config.z_step)
                
                for i, z_pos in enumerate(z_positions):
                    # Synchronized move
                    yield from light_sheet.synchronized_move(z_pos)
                    
                    # Acquire image with metadata
                    yield from bps.trigger_and_read(
                        [light_sheet.camera],
                        name=f't_{t:03d}_z_{i:03d}',
                        md={'time_point': t, 
                           'z_index': i, 
                           'z_position': z_pos}
                    )
                
                # Wait for next time point (except last)
                if t < config.time_points - 1:
                    yield from bps.sleep(config.time_interval)
                    
        finally:
            yield from bps.close_run()
    
    return time_lapse_plan()


def multi_position_scan(light_sheet: DiSPIMLightSheet,
                       positions: List[tuple],
                       config: DiSPIMScanConfig) -> Generator[Any, Any, None]:
    """
    Bluesky plan for multi-position DiSPIM acquisition
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        DiSPIM device ensemble
    positions : List[tuple]
        List of (x, y) positions in μm
    config : DiSPIMScanConfig
        Scan configuration parameters
    """
    
    def multi_pos_plan():
        md = {
            'plan_name': 'dispim_multi_position',
            'num_positions': len(positions),
            'positions': positions,
            'z_range': config.z_stop - config.z_start,
            'z_step': config.z_step
        }
        
        yield from bps.open_run(md=md)
        
        try:
            for pos_idx, (x, y) in enumerate(positions):
                # Create position marker
                yield from bps.create(name=f'position_{pos_idx}')
                
                # Move to XY position (assuming XY stage exists)
                # yield from bps.mv(xy_stage.x, x, xy_stage.y, y)
                
                # Z-stack at this position
                z_positions = np.arange(config.z_start, 
                                       config.z_stop + config.z_step, 
                                       config.z_step)
                
                for z_idx, z_pos in enumerate(z_positions):
                    yield from light_sheet.synchronized_move(z_pos)
                    
                    yield from bps.trigger_and_read(
                        [light_sheet.camera],
                        name=f'pos_{pos_idx:02d}_z_{z_idx:03d}',
                        md={'position_index': pos_idx,
                           'x_position': x,
                           'y_position': y,
                           'z_position': z_pos}
                    )
                    
        finally:
            yield from bps.close_run()
    
    return multi_pos_plan()


def adaptive_acquisition_with_vlm(light_sheet: DiSPIMLightSheet,
                                 vlm_callback: callable,
                                 config: DiSPIMScanConfig) -> Generator[Any, Any, None]:
    """
    Bluesky plan for VLM-guided adaptive DiSPIM acquisition
    
    Parameters
    ----------
    light_sheet : DiSPIMLightSheet
        DiSPIM device ensemble
    vlm_callback : callable
        Function that processes images and returns acquisition decisions
    config : DiSPIMScanConfig
        Scan configuration parameters
    """
    
    def adaptive_plan():
        md = {
            'plan_name': 'dispim_adaptive_vlm',
            'adaptive_acquisition': True,
            'vlm_guided': True
        }
        
        yield from bps.open_run(md=md)
        
        try:
            z_pos = config.z_start
            z_index = 0
            continue_scan = True
            
            while continue_scan and z_pos <= config.z_stop:
                # Move to position
                yield from light_sheet.synchronized_move(z_pos)
                
                # Acquire image
                yield from bps.trigger_and_read(
                    [light_sheet.camera],
                    name=f'adaptive_z_{z_index:03d}',
                    md={'z_position': z_pos, 'adaptive_step': z_index}
                )
                
                # Get image for VLM analysis
                image_data = light_sheet.camera._last_image
                
                if image_data is not None and vlm_callback:
                    # VLM decision making
                    decision = vlm_callback(image_data, z_pos, z_index)
                    
                    # Process VLM decision
                    if decision.get('stop_scan', False):
                        continue_scan = False
                        break
                    
                    # Adaptive step size
                    step_size = decision.get('next_step', config.z_step)
                    z_pos += step_size
                    
                    # Log VLM decision
                    yield from bps.create(name='vlm_decision')
                    yield from bps.save({
                        'vlm_decision': decision,
                        'next_z_step': step_size,
                        'continue_scan': continue_scan
                    })
                else:
                    # Default step if no VLM
                    z_pos += config.z_step
                
                z_index += 1
                
        finally:
            yield from bps.close_run()
    
    return adaptive_plan()


def setup_dispim_session(config_file: str = None) -> tuple:
    """
    Setup complete DiSPIM session with RunEngine and devices
    
    Parameters
    ----------
    config_file : str, optional
        Path to Micro-Manager configuration file
        
    Returns
    -------
    tuple
        (RunEngine, DiSPIMLightSheet, callbacks)
    """
    
    # Create devices
    light_sheet = create_dispim_devices(config_file)
    
    # Create run engine
    RE = RunEngine({})
    
    # Setup data broker (optional)
    try:
        db = Broker.named('temp')
        RE.subscribe(db.insert)
    except:
        db = None
    
    # Setup live callbacks
    callbacks = []
    
    # Live table for monitoring
    live_table = LiveTable([
        light_sheet.piezo_imaging.name + '_user_readback',
        light_sheet.galvo.name + '_user_readback'
    ])
    callbacks.append(live_table)
    
    # Live plot for Z positions
    live_plot = LivePlot(
        light_sheet.piezo_imaging.name + '_user_readback',
        x='seq_num',
        ax=None
    )
    callbacks.append(live_plot)
    
    return RE, light_sheet, callbacks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Setup session
    RE, light_sheet, callbacks = setup_dispim_session()
    
    # Subscribe callbacks
    for callback in callbacks:
        RE.subscribe(callback)
    
    # Example scan configuration
    scan_config = DiSPIMScanConfig(
        z_start=-20.0,
        z_stop=20.0,
        z_step=1.0,
        exposure_time=0.02
    )
    
    print("DiSPIM Bluesky plans ready")
    print("Available plans: calibration_sequence, z_stack_scan, dual_view_acquisition")
    print("Time-lapse and adaptive VLM plans also available")
    
    # Example: Run calibration
    # RE(calibration_sequence(light_sheet))
    
    # Example: Run Z-stack
    # RE(z_stack_scan(light_sheet, -10, 10, 0.5))