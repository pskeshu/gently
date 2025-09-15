"""
Gently DiSPIM Napari Integration
===============================

Real-time image visualization for DiSPIM experiments using napari and Bluesky callbacks.
Provides optional, non-intrusive visualization that works with any image-generating plan.

Key Features:
- Real-time streaming of images from Bluesky plans
- Handles 2D images, 3D stacks, and multi-channel data
- Works with focus sweeps, embryo detection, Z-stacks
- Completely optional - graceful fallback if napari not available
- Standard Bluesky callback pattern

Usage:
    from gently.visualization import setup_napari_callback
    
    RE = RunEngine({})
    napari_callback = setup_napari_callback()
    RE.subscribe(napari_callback)
    
    # Now any plan with images will display in napari
    RE(dispim_piezo_autofocus(light_sheet, config))
"""

import time
import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
import numpy as np

# Handle optional napari dependency
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    napari = None

# Bluesky imports
from bluesky.callbacks.core import CallbackBase

logger = logging.getLogger(__name__)


class ImageStackManager:
    """Manages accumulation of 2D images into 3D/4D stacks for napari display"""
    
    def __init__(self, name: str):
        self.name = name
        self.images = []
        self.positions = []
        self.timestamps = []
        self.metadata = {}
        self.current_stack = None
        
    def add_image(self, image: np.ndarray, position: float = None, 
                  timestamp: float = None, metadata: Dict = None):
        """Add a 2D image to the stack"""
        self.images.append(image.copy())
        self.positions.append(position)
        self.timestamps.append(timestamp or time.time())
        
        if metadata:
            for key, value in metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = []
                self.metadata[key].append(value)
    
    def get_stack(self) -> Tuple[np.ndarray, Dict]:
        """Get current image stack as 3D array with metadata"""
        if not self.images:
            return None, {}
            
        # Stack images along new axis (z-dimension for focus sweeps)
        stack = np.stack(self.images, axis=0)
        
        # Prepare metadata for napari
        napari_metadata = {
            'name': self.name,
            'positions': self.positions,
            'timestamps': self.timestamps,
            'shape': stack.shape,
            'dtype': stack.dtype
        }
        napari_metadata.update(self.metadata)
        
        return stack, napari_metadata
    
    def clear(self):
        """Clear accumulated data for new stack"""
        self.images.clear()
        self.positions.clear()
        self.timestamps.clear()
        self.metadata.clear()
        self.current_stack = None


class NapariCallback(CallbackBase):
    """
    Bluesky callback for real-time napari visualization of DiSPIM images
    
    Subscribes to Bluesky document stream and displays images in napari viewer.
    Handles different data types and experiment patterns automatically.
    """
    
    def __init__(self, 
                 show_focus_sweeps: bool = True,
                 show_embryo_detection: bool = True,
                 show_single_images: bool = True,
                 dual_channel_mode: bool = True,
                 update_interval: float = 0.1,
                 viewer: 'napari.Viewer' = None,
                 **kwargs):
        """
        Initialize napari callback
        
        Parameters
        ----------
        show_focus_sweeps : bool
            Display focus sweep image stacks
        show_embryo_detection : bool  
            Display embryo detection scan images
        show_single_images : bool
            Display individual camera acquisitions
        dual_channel_mode : bool
            Handle dual-sided DiSPIM with separate channels
        update_interval : float
            Minimum time between napari updates (seconds)
        viewer : napari.Viewer, optional
            Existing napari viewer to use
        """
        super().__init__()
        
        if not NAPARI_AVAILABLE:
            warnings.warn("Napari not available. Image visualization disabled. "
                         "Install with: pip install napari[all]", UserWarning)
            self.enabled = False
            return
        
        self.enabled = True
        self.show_focus_sweeps = show_focus_sweeps
        self.show_embryo_detection = show_embryo_detection
        self.show_single_images = show_single_images
        self.dual_channel_mode = dual_channel_mode
        self.update_interval = update_interval
        
        # Create or use existing viewer - defer to main thread
        self.viewer = viewer
        self.viewer_created = False
        if self.viewer is None:
            self._create_viewer_safe()
        
        # Track experiment state
        self.current_plan = None
        self.plan_metadata = {}
        self.last_update_time = 0
        
        # Manage image stacks for different experiment types
        self.stack_managers = {}
        self.active_stacks = set()
        
        # Track channels for dual-sided DiSPIM
        self.channels = {'side_a': {}, 'side_b': {}}
        
        logger.info("NapariCallback initialized - real-time visualization enabled")
    
    def _create_viewer_safe(self):
        """Create napari viewer safely"""
        try:
            import threading
            if threading.current_thread() is threading.main_thread():
                self.viewer = napari.Viewer(title="DiSPIM Live View")
                self.viewer_created = True
            else:
                # Defer viewer creation to when actually needed
                self.viewer_created = False
        except Exception as e:
            logger.warning(f"Could not create napari viewer: {e}")
            self.enabled = False
    
    def start(self, doc):
        """Called at start of Bluesky run"""
        if not self.enabled:
            return
            
        self.current_plan = doc.get('plan_name', 'unknown_plan')
        self.plan_metadata = doc.get('plan_args', {})
        
        # Determine experiment type and setup appropriate visualization
        self._setup_for_plan_type()
        
        logger.info(f"Started visualization for plan: {self.current_plan}")
    
    def event(self, doc):
        """Called for each Bluesky event (data point)"""
        if not self.enabled:
            return
            
        # Throttle updates to avoid overwhelming napari
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        # Process image data from event
        self._process_event_data(doc)
        
        self.last_update_time = current_time
    
    def stop(self, doc):
        """Called at end of Bluesky run"""
        if not self.enabled:
            return
            
        # Finalize any remaining image stacks
        self._finalize_stacks()
        
        logger.info(f"Completed visualization for plan: {self.current_plan}")
        
        # Clear state for next run
        self.current_plan = None
        self.plan_metadata = {}
        self.active_stacks.clear()
    
    def _setup_for_plan_type(self):
        """Setup visualization based on detected plan type"""
        plan_name = self.current_plan.lower()
        
        if 'focus' in plan_name and self.show_focus_sweeps:
            # Focus sweep - prepare for 3D stack visualization
            self.stack_managers['focus_sweep'] = ImageStackManager('Focus Sweep')
            self.active_stacks.add('focus_sweep')
            logger.debug("Setup focus sweep visualization")
            
        elif 'embryo' in plan_name and self.show_embryo_detection:
            # Embryo detection - prepare for XY scan visualization
            self.stack_managers['embryo_scan'] = ImageStackManager('Embryo Detection')
            self.active_stacks.add('embryo_scan')
            logger.debug("Setup embryo detection visualization")
            
        elif self.show_single_images:
            # General image acquisition
            logger.debug("Setup single image visualization")
    
    def _process_event_data(self, doc):
        """Process image data from Bluesky event document"""
        data = doc.get('data', {})
        timestamps = doc.get('timestamps', {})
        
        # Look for image data in the event
        for signal_name, signal_data in data.items():
            if self._is_image_signal(signal_name, signal_data):
                self._handle_image_data(signal_name, signal_data, 
                                      timestamps.get(signal_name), doc)
    
    def _is_image_signal(self, signal_name: str, signal_data: Any) -> bool:
        """Check if signal contains image data"""
        # Look for camera image signals
        if isinstance(signal_data, np.ndarray) and signal_data.ndim >= 2:
            # Check if signal name indicates camera/image data
            signal_lower = signal_name.lower()
            if any(keyword in signal_lower for keyword in ['camera', 'image', 'detector']):
                return True
        return False
    
    def _handle_image_data(self, signal_name: str, image_data: np.ndarray, 
                          timestamp: float, event_doc: Dict):
        """Handle image data based on current experiment context"""
        
        # Extract metadata
        position_data = self._extract_position_data(event_doc)
        metadata = {
            'signal_name': signal_name,
            'timestamp': timestamp,
            'positions': position_data
        }
        
        # Determine channel (side A or B)
        channel = self._determine_channel(signal_name)
        
        if self.active_stacks:
            # Add to appropriate stack
            for stack_name in self.active_stacks:
                stack_manager = self.stack_managers[stack_name]
                position = position_data.get('z_position', position_data.get('focus_position'))
                stack_manager.add_image(image_data, position, timestamp, metadata)
                
                # Update napari display
                self._update_stack_display(stack_name, stack_manager, channel)
        
        elif self.show_single_images:
            # Display single image immediately
            self._display_single_image(signal_name, image_data, metadata, channel)
    
    def _extract_position_data(self, event_doc: Dict) -> Dict[str, float]:
        """Extract position information from event document"""
        data = event_doc.get('data', {})
        positions = {}
        
        # Look for position signals
        for signal_name, value in data.items():
            if 'position' in signal_name or 'readback' in signal_name:
                try:
                    positions[signal_name] = float(value)
                except (ValueError, TypeError):
                    pass
        
        return positions
    
    def _determine_channel(self, signal_name: str) -> str:
        """Determine which DiSPIM channel (side) this image is from"""
        signal_lower = signal_name.lower()
        
        if '_a' in signal_lower or 'side_a' in signal_lower:
            return 'side_a'
        elif '_b' in signal_lower or 'side_b' in signal_lower:
            return 'side_b'
        else:
            return 'side_a'  # Default to side A
    
    def _update_stack_display(self, stack_name: str, stack_manager: ImageStackManager, 
                             channel: str):
        """Update napari display with current image stack"""
        stack, metadata = stack_manager.get_stack()
        if stack is None:
            return
        
        # Ensure viewer exists
        if not self.viewer:
            try:
                self.viewer = napari.Viewer(title="DiSPIM Live View")
                self.viewer_created = True
            except Exception as e:
                logger.warning(f"Cannot create napari viewer: {e}")
                return
        
        try:
            layer_name = f"{metadata['name']} ({channel.title()})"
            
            # Choose colors for dual-sided DiSPIM
            colormap = 'green' if channel == 'side_a' else 'magenta'
            
            # Update or create napari layer
            if layer_name in self.viewer.layers:
                # Update existing layer
                self.viewer.layers[layer_name].data = stack
            else:
                # Create new layer
                self.viewer.add_image(
                    stack,
                    name=layer_name,
                    colormap=colormap,
                    blending='additive' if self.dual_channel_mode else 'translucent',
                    metadata=metadata
                )
            
            # Update display
            self.viewer.reset_view()
        except Exception as e:
            logger.warning(f"Failed to update napari: {e}")
            # Continue without disabling
    
    def _display_single_image(self, signal_name: str, image_data: np.ndarray, 
                             metadata: Dict, channel: str):
        """Display a single 2D image in napari"""
        # Ensure viewer exists
        if not self.viewer:
            try:
                self.viewer = napari.Viewer(title="DiSPIM Live View")
                self.viewer_created = True
            except Exception as e:
                logger.warning(f"Cannot create napari viewer: {e}")
                return
        
        try:
            layer_name = f"{signal_name} ({channel.title()})"
            colormap = 'green' if channel == 'side_a' else 'magenta'
            
            # Update or create napari layer
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = image_data
            else:
                self.viewer.add_image(
                    image_data,
                    name=layer_name,
                    colormap=colormap,
                    blending='additive' if self.dual_channel_mode else 'translucent',
                    metadata=metadata
                )
        except Exception as e:
            logger.warning(f"Failed to display image in napari: {e}")
    
    def _finalize_stacks(self):
        """Finalize any remaining image stacks at end of run"""
        for stack_name, stack_manager in self.stack_managers.items():
            if stack_manager.images:
                # Final update of stack display
                for channel in ['side_a', 'side_b']:
                    self._update_stack_display(stack_name, stack_manager, channel)
        
        # Clear stack managers for next run
        self.stack_managers.clear()


def setup_napari_callback(config: Optional[Dict] = None, 
                         viewer: 'napari.Viewer' = None) -> NapariCallback:
    """
    Convenience function to setup napari callback with sensible defaults
    
    Parameters
    ----------
    config : Dict, optional
        Configuration options for NapariCallback
    viewer : napari.Viewer, optional
        Existing napari viewer to use
        
    Returns
    -------
    NapariCallback
        Configured callback ready for RunEngine.subscribe()
        
    Examples
    --------
    Basic usage:
    >>> RE = RunEngine({})
    >>> napari_callback = setup_napari_callback()
    >>> RE.subscribe(napari_callback)
    
    Custom configuration:
    >>> config = {'show_focus_sweeps': True, 'update_interval': 0.5}
    >>> napari_callback = setup_napari_callback(config)
    >>> RE.subscribe(napari_callback)
    """
    if not NAPARI_AVAILABLE:
        logger.warning("Napari not available - returning disabled callback")
        callback = NapariCallback()  # Will be disabled automatically
        return callback
    
    # Default configuration
    default_config = {
        'show_focus_sweeps': True,
        'show_embryo_detection': True,
        'show_single_images': True,
        'dual_channel_mode': True,
        'update_interval': 0.1
    }
    
    # Merge user config
    if config:
        default_config.update(config)
    
    callback = NapariCallback(viewer=viewer, **default_config)
    
    if callback.enabled:
        logger.info("Napari visualization enabled - images will display in real-time")
    
    return callback


def create_napari_viewer(title: str = "DiSPIM Live View") -> 'napari.Viewer':
    """
    Create a new napari viewer optimized for DiSPIM visualization
    
    Parameters
    ----------
    title : str
        Window title for the viewer
        
    Returns
    -------
    napari.Viewer
        Configured napari viewer
    """
    if not NAPARI_AVAILABLE:
        raise ImportError("Napari not available. Install with: pip install napari[all]")
    
    viewer = napari.Viewer(title=title)
    
    # Configure viewer for microscopy data
    viewer.theme = 'dark'
    
    return viewer


# Convenience functions for common usage patterns

def enable_focus_sweep_visualization(RE, viewer: 'napari.Viewer' = None):
    """Enable napari visualization for focus sweep experiments"""
    config = {
        'show_focus_sweeps': True,
        'show_embryo_detection': False,
        'show_single_images': False,
        'dual_channel_mode': True
    }
    
    callback = setup_napari_callback(config, viewer)
    RE.subscribe(callback)
    return callback


def enable_embryo_detection_visualization(RE, viewer: 'napari.Viewer' = None):
    """Enable napari visualization for embryo detection experiments"""  
    config = {
        'show_focus_sweeps': False,
        'show_embryo_detection': True,
        'show_single_images': False,
        'dual_channel_mode': False  # Usually single camera for detection
    }
    
    callback = setup_napari_callback(config, viewer)
    RE.subscribe(callback)
    return callback


def enable_full_visualization(RE, viewer: 'napari.Viewer' = None):
    """Enable napari visualization for all DiSPIM experiments"""
    config = {
        'show_focus_sweeps': True,
        'show_embryo_detection': True, 
        'show_single_images': True,
        'dual_channel_mode': True
    }
    
    callback = setup_napari_callback(config, viewer)
    RE.subscribe(callback)
    return callback