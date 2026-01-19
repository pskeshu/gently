"""
Visualization Manager for Microscopy Copilot

Handles visualization in a thread-safe way by:
1. Always saving images/figures to disk
2. Optionally launching viewers in subprocesses
3. Providing async-compatible API for the copilot

This avoids Qt/main-thread issues when RunEngine runs in a background thread.
"""

import asyncio
import json
import multiprocessing
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
import numpy as np


@dataclass
class VisualizationRequest:
    """A request to visualize data"""
    viz_type: str  # 'image', 'volume', 'plot', 'embryo_detection'
    data: Any
    metadata: Dict = field(default_factory=dict)
    title: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    saved_path: Optional[Path] = None


class VisualizationManager:
    """
    Thread-safe visualization manager for microscopy copilot

    All visualization requests are queued and can be:
    1. Saved to disk (always)
    2. Displayed in napari/matplotlib subprocess (optional)
    3. Served via web (future)

    Example
    -------
    >>> viz = VisualizationManager(output_dir=Path("./viz_output"))
    >>>
    >>> # Queue a visualization (thread-safe)
    >>> viz.show_image(image, title="Latest acquisition")
    >>>
    >>> # Launch napari to view recent images
    >>> await viz.open_napari_viewer()
    """

    def __init__(
        self,
        output_dir: Path = Path("./visualization_output"),
        auto_save: bool = True,
        max_history: int = 50,
    ):
        """
        Parameters
        ----------
        output_dir : Path
            Directory to save visualization outputs
        auto_save : bool
            Automatically save all visualizations to disk
        max_history : int
            Maximum number of recent visualizations to track
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save
        self.max_history = max_history

        # Thread-safe queue for visualization requests
        self._queue: Queue[VisualizationRequest] = Queue()
        self._history: List[VisualizationRequest] = []
        self._lock = threading.Lock()

        # Active viewer processes
        self._napari_process: Optional[subprocess.Popen] = None

        # Callbacks for when visualizations are ready
        self._callbacks: List[Callable[[VisualizationRequest], None]] = []

    def show_image(
        self,
        image: np.ndarray,
        title: str = "Image",
        metadata: Optional[Dict] = None,
        colormap: str = "gray",
    ) -> VisualizationRequest:
        """
        Queue an image for visualization

        Parameters
        ----------
        image : np.ndarray
            2D or 3D image array
        title : str
            Display title
        metadata : dict, optional
            Additional metadata
        colormap : str
            Colormap name for display

        Returns
        -------
        VisualizationRequest
            The queued request
        """
        request = VisualizationRequest(
            viz_type='image',
            data=image.copy(),  # Copy to avoid mutation
            title=title,
            metadata={
                'colormap': colormap,
                'shape': image.shape,
                'dtype': str(image.dtype),
                **(metadata or {})
            }
        )

        self._enqueue(request)
        return request

    def show_volume(
        self,
        volume: np.ndarray,
        title: str = "Volume",
        metadata: Optional[Dict] = None,
    ) -> VisualizationRequest:
        """
        Queue a 3D volume for visualization

        Parameters
        ----------
        volume : np.ndarray
            3D volume array (Z, Y, X)
        title : str
            Display title
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        VisualizationRequest
            The queued request
        """
        request = VisualizationRequest(
            viz_type='volume',
            data=volume.copy(),
            title=title,
            metadata={
                'shape': volume.shape,
                'dtype': str(volume.dtype),
                **(metadata or {})
            }
        )

        self._enqueue(request)
        return request

    def show_embryo_detection(
        self,
        image: np.ndarray,
        embryos: List[Dict],
        title: str = "Embryo Detection",
        metadata: Optional[Dict] = None,
    ) -> VisualizationRequest:
        """
        Queue embryo detection results for visualization

        Parameters
        ----------
        image : np.ndarray
            Background image
        embryos : list of dict
            Detected embryo positions and metadata
        title : str
            Display title
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        VisualizationRequest
            The queued request
        """
        request = VisualizationRequest(
            viz_type='embryo_detection',
            data={
                'image': image.copy(),
                'embryos': embryos,
            },
            title=title,
            metadata=metadata or {}
        )

        self._enqueue(request)
        return request

    def show_max_projection(
        self,
        volume: np.ndarray,
        title: str = "Max Projection",
        axis: int = 0,
        metadata: Optional[Dict] = None,
    ) -> VisualizationRequest:
        """
        Create and queue max projection of a volume

        Parameters
        ----------
        volume : np.ndarray
            3D volume
        title : str
            Display title
        axis : int
            Axis along which to project (default 0 = Z)
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        VisualizationRequest
            The queued request
        """
        max_proj = np.max(volume, axis=axis)

        return self.show_image(
            max_proj,
            title=title,
            metadata={
                'projection_axis': axis,
                'original_shape': volume.shape,
                **(metadata or {})
            }
        )

    def show_three_view_projection(
        self,
        volume: np.ndarray,
        title: str = "Three-View Projection",
        metadata: Optional[Dict] = None,
    ) -> VisualizationRequest:
        """
        Create and queue three-view orthogonal projection of a volume.

        Generates XY (top), YZ (side), and XZ (front) views combined
        into a single image for comprehensive 3D morphology assessment.

        Parameters
        ----------
        volume : np.ndarray
            3D volume (Z, Y, X)
        title : str
            Display title
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        VisualizationRequest
            The queued request
        """
        from gently.agent.perception.projection import (
            projection_three_view,
            compute_crop_bounds,
            apply_crop_bounds,
        )

        # Handle dual-view format
        if volume.ndim == 3:
            z_depth, height, width = volume.shape
            if width > height * 2:
                volume = volume[:, :, :width // 2]

        # Auto-crop to embryo region
        bounds = compute_crop_bounds(volume)
        volume = apply_crop_bounds(volume, bounds)

        # Generate three-view projection
        three_view_img, description = projection_three_view(volume)

        return self.show_image(
            three_view_img,
            title=title,
            metadata={
                'projection_type': 'three_view',
                'description': description,
                'original_shape': volume.shape,
                **(metadata or {})
            }
        )

    def _enqueue(self, request: VisualizationRequest):
        """Add request to queue and process"""
        with self._lock:
            # Save to disk if auto_save enabled
            if self.auto_save:
                request.saved_path = self._save_to_disk(request)

            # Add to history
            self._history.append(request)
            if len(self._history) > self.max_history:
                self._history.pop(0)

            # Add to queue for any waiting consumers
            self._queue.put(request)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(request)
                except Exception as e:
                    print(f"Visualization callback error: {e}")

    def _save_to_disk(self, request: VisualizationRequest) -> Path:
        """Save visualization to disk"""
        timestamp = request.timestamp.strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.title)

        if request.viz_type in ('image', 'volume'):
            # Save as TIFF
            filename = f"{timestamp}_{safe_title}.tif"
            filepath = self.output_dir / filename

            try:
                import tifffile
                tifffile.imwrite(filepath, request.data)
            except ImportError:
                # Fallback to numpy
                filepath = filepath.with_suffix('.npy')
                np.save(filepath, request.data)

        elif request.viz_type == 'embryo_detection':
            # Save image and metadata separately
            img_filename = f"{timestamp}_{safe_title}_image.tif"
            meta_filename = f"{timestamp}_{safe_title}_embryos.json"

            img_path = self.output_dir / img_filename
            meta_path = self.output_dir / meta_filename

            try:
                import tifffile
                tifffile.imwrite(img_path, request.data['image'])
            except ImportError:
                img_path = img_path.with_suffix('.npy')
                np.save(img_path, request.data['image'])

            # Save embryo data as JSON
            with open(meta_path, 'w') as f:
                json.dump(request.data['embryos'], f, indent=2)

            filepath = img_path

        else:
            # Generic: save as numpy
            filename = f"{timestamp}_{safe_title}.npy"
            filepath = self.output_dir / filename
            np.save(filepath, request.data)

        return filepath

    def get_recent(self, n: int = 5) -> List[VisualizationRequest]:
        """Get n most recent visualizations"""
        with self._lock:
            return self._history[-n:]

    def get_latest(self) -> Optional[VisualizationRequest]:
        """Get the most recent visualization"""
        with self._lock:
            return self._history[-1] if self._history else None

    async def open_napari_viewer(
        self,
        requests: Optional[List[VisualizationRequest]] = None,
        show_recent: int = 5,
    ) -> bool:
        """
        Open napari in a subprocess to view visualizations

        Parameters
        ----------
        requests : list, optional
            Specific requests to show. If None, shows recent.
        show_recent : int
            Number of recent visualizations to show if requests is None

        Returns
        -------
        bool
            True if napari was launched successfully
        """
        if requests is None:
            requests = self.get_recent(show_recent)

        if not requests:
            print("No visualizations to show")
            return False

        # Collect file paths
        paths = [str(r.saved_path) for r in requests if r.saved_path and r.saved_path.exists()]

        if not paths:
            print("No saved files to display")
            return False

        # Launch napari in subprocess
        script = f'''
import napari
import tifffile
import numpy as np
import sys

paths = {paths}
viewer = napari.Viewer()

for path in paths:
    try:
        if path.endswith('.tif'):
            data = tifffile.imread(path)
        else:
            data = np.load(path)
        name = path.split('/')[-1].split('\\\\')[-1]
        viewer.add_image(data, name=name)
    except Exception as e:
        print(f"Could not load {{path}}: {{e}}")

napari.run()
'''

        try:
            # Run in subprocess to avoid Qt thread issues
            self._napari_process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            print(f"Napari viewer launched (PID: {self._napari_process.pid})")
            return True

        except Exception as e:
            print(f"Could not launch napari: {e}")
            return False

    async def open_napari_for_embryo_detection(
        self,
        image: np.ndarray,
        embryos: List[Dict],
        block: bool = False,
    ) -> bool:
        """
        Open napari specifically for embryo detection review

        Parameters
        ----------
        image : np.ndarray
            Background image
        embryos : list of dict
            Detected embryos with positions
        block : bool
            If True, wait for napari to close

        Returns
        -------
        bool
            True if launched successfully
        """
        # Save data to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "image.npy"
            embryo_path = Path(tmpdir) / "embryos.json"

            np.save(img_path, image)
            with open(embryo_path, 'w') as f:
                json.dump(embryos, f)

            script = f'''
import napari
import numpy as np
import json

image = np.load(r"{img_path}")
with open(r"{embryo_path}") as f:
    embryos = json.load(f)

viewer = napari.Viewer(title="Embryo Detection - Review Required")
viewer.add_image(image, name='Bottom Camera', colormap='gray')

# Add embryo markers
if embryos:
    centers = []
    texts = []
    for emb in embryos:
        centers.append([emb['pixel_y'], emb['pixel_x']])
        texts.append(str(emb['embryo_id']))

    import numpy as np
    centers = np.array(centers)

    viewer.add_points(
        centers,
        size=50,
        face_color='transparent',
        edge_color='lime',
        edge_width=3,
        name='Detected Embryos'
    )

    # Add labels
    props = {{'text': texts}}
    viewer.add_points(
        centers,
        size=1,
        properties=props,
        text={{'string': 'text', 'size': 14, 'color': 'yellow'}},
        name='Labels',
        visible=True
    )

napari.run()
'''

            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, '-c', script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                if block:
                    await process.wait()

                return True

            except Exception as e:
                print(f"Could not launch napari: {e}")
                return False

    def close_napari(self):
        """Close any running napari viewer"""
        if self._napari_process and self._napari_process.returncode is None:
            self._napari_process.terminate()
            self._napari_process = None

    def add_callback(self, callback: Callable[[VisualizationRequest], None]):
        """Add callback to be notified of new visualizations"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[VisualizationRequest], None]):
        """Remove a callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def list_saved_files(self) -> List[Path]:
        """List all saved visualization files"""
        return sorted(self.output_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    def get_output_summary(self) -> str:
        """Get summary of saved visualizations"""
        files = self.list_saved_files()

        if not files:
            return f"No visualizations saved yet.\nOutput directory: {self.output_dir}"

        summary = f"Visualization output directory: {self.output_dir}\n"
        summary += f"Total files: {len(files)}\n\n"
        summary += "Recent files:\n"

        for f in files[:10]:
            size_kb = f.stat().st_size / 1024
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S")
            summary += f"  {mtime} - {f.name} ({size_kb:.1f} KB)\n"

        if len(files) > 10:
            summary += f"  ... and {len(files) - 10} more files\n"

        return summary
