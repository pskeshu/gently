#!/usr/bin/env python3
"""
Napari-Based Interactive Embryo Marking
=======================================

Provides interactive embryo position marking using napari viewer.
Non-blocking interface suitable for integration with Bluesky plans.

Key Features:
- Display bottom camera overview image
- Interactive point annotation for embryo positions
- Thread-safe communication with RunEngine
- Automatic embryo numbering
- Visual feedback during marking
"""

import numpy as np
import threading
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    print("Warning: napari not installed. Install with: pip install napari[all]")


class EmbryoMarker:
    """
    Interactive embryo marker using napari viewer.

    Allows user to mark embryo positions by clicking on bottom camera image.
    Thread-safe for use with Bluesky RunEngine.

    Parameters
    ----------
    image : np.ndarray
        Bottom camera overview image (2D grayscale or RGB)
    initial_stage_position : tuple of float
        Initial XY stage position in micrometers (x, y)
    pixel_size_um : float, optional
        Effective pixel size in micrometers/pixel (default: 0.65)
    viewer : napari.Viewer, optional
        Existing napari viewer to use. If None, creates new viewer.

    Attributes
    ----------
    marked_embryos : list of dict
        List of marked embryo dictionaries with:
        - embryo_number : int
        - embryo_id : str
        - pixel_position : tuple (x, y)
        - initial_stage_position : tuple (x, y)
        - marking_timestamp : str
    marking_complete : bool
        True when user has finished marking
    """

    def __init__(
        self,
        image: np.ndarray,
        initial_stage_position: Tuple[float, float],
        pixel_size_um: float = 0.65,
        viewer: Optional['napari.Viewer'] = None
    ):
        """Initialize embryo marker."""
        if not NAPARI_AVAILABLE:
            raise ImportError(
                "napari is required for interactive marking. "
                "Install with: pip install napari[all]"
            )

        self.image = image
        self.initial_stage_position = initial_stage_position
        self.pixel_size_um = pixel_size_um

        # Embryo tracking
        self.marked_embryos: List[Dict] = []
        self.marking_complete = False
        self._lock = threading.Lock()
        self._num_points_processed = 0  # Track how many points we've already processed

        # Create or use existing viewer
        if viewer is None:
            self.viewer = napari.Viewer(title="Embryo Marker - Click to mark embryos")
            self._owns_viewer = True
        else:
            self.viewer = viewer
            self._owns_viewer = False

        # Add image layer
        self.image_layer = self.viewer.add_image(
            image,
            name='Bottom Camera Overview',
            colormap='gray',
            contrast_limits=[np.percentile(image, 1), np.percentile(image, 99)]
        )

        # Add points layer for marking
        # Note: napari API varies by version - use minimal parameters for compatibility
        self.points_layer = self.viewer.add_points(
            ndim=2,
            name='Marked Embryos',
            size=30,
            face_color='cyan',
            symbol='cross'
        )

        # Try to set border properties if API supports it
        try:
            self.points_layer.border_color = 'white'
            self.points_layer.border_width = 3
        except AttributeError:
            # Older napari version - border properties not available
            pass

        # Connect callback for point addition
        self.points_layer.events.data.connect(self._on_point_added)

        # Add instructions
        self._show_instructions()

    def _show_instructions(self):
        """Display marking instructions in console."""
        print("\n" + "="*70)
        print("EMBRYO MARKING - INTERACTIVE MODE")
        print("="*70)
        print("\nINSTRUCTIONS:")
        print("  1. Click on each embryo center to mark its position")
        print("  2. Embryos will be numbered automatically (1, 2, 3, ...)")
        print("  3. You can zoom and pan as needed")
        print("  4. When done marking, close the napari window")
        print("\nTIP: Mark embryos in order of priority (most important first)")
        print("="*70 + "\n")

    def _on_point_added(self, event):
        """
        Callback when user adds a point.

        This is called automatically by napari when a point is added to the layer.
        Note: This event can fire multiple times for the same data state,
        so we track how many points we've already processed.
        """
        with self._lock:
            # Get current points
            points = self.points_layer.data

            if len(points) == 0:
                return

            # Check if we have new points to process
            num_current_points = len(points)
            if num_current_points <= self._num_points_processed:
                # No new points, callback fired but data unchanged
                return

            # Process all new points (usually just 1, but could be multiple if pasted)
            for i in range(self._num_points_processed, num_current_points):
                new_point = points[i]
                pixel_y, pixel_x = new_point  # napari uses (row, col) = (y, x)

                # Create embryo entry
                embryo_number = len(self.marked_embryos) + 1
                embryo_id = f"embryo_{embryo_number:03d}"

                embryo_entry = {
                    'embryo_number': embryo_number,
                    'embryo_id': embryo_id,
                    'pixel_position': (float(pixel_x), float(pixel_y)),
                    'initial_stage_position': self.initial_stage_position,
                    'marking_timestamp': datetime.now().isoformat()
                }

                self.marked_embryos.append(embryo_entry)

                print(f"✓ Marked {embryo_id} at pixel position ({pixel_x:.1f}, {pixel_y:.1f})")

            # Update count of processed points
            self._num_points_processed = num_current_points

    def wait_for_completion(self):
        """
        Block until user closes the napari window (marking complete).

        This method blocks the calling thread until the user finishes marking
        and closes the napari viewer.
        """
        print("\nWaiting for marking to complete (close napari window when done)...")

        # Wait for viewer to close
        # Note: napari.run() only works if viewer was created by us
        if self._owns_viewer:
            # Blocking call - waits until viewer closes
            napari.run()
        else:
            # Viewer owned by someone else, just wait for window to close
            while self.viewer.window._qt_window.isVisible():
                napari.qt.get_app().processEvents()
                threading.Event().wait(0.1)

        with self._lock:
            self.marking_complete = True

        print(f"\n✓ Marking complete! Total embryos marked: {len(self.marked_embryos)}")

    def get_marked_embryos(self) -> List[Dict]:
        """
        Get list of marked embryo dictionaries.

        Returns
        -------
        list of dict
            Marked embryos with positions and metadata

        Examples
        --------
        >>> marker = EmbryoMarker(image, (1000.0, 2000.0))
        >>> # User marks embryos interactively...
        >>> marker.wait_for_completion()
        >>> embryos = marker.get_marked_embryos()
        >>> for embryo in embryos:
        ...     print(f"{embryo['embryo_id']}: {embryo['pixel_position']}")
        """
        with self._lock:
            return list(self.marked_embryos)  # Return copy

    def save_marked_image(self, output_path: Path):
        """
        Save marked image with embryo positions to disk.

        Parameters
        ----------
        output_path : Path
            Output path for PNG file
        """
        from PIL import Image, ImageDraw, ImageFont

        output_path = Path(output_path)

        # Convert image to PIL
        if self.image.dtype != np.uint8:
            # Normalize to 0-255
            img_normalized = ((self.image - self.image.min()) /
                             (self.image.max() - self.image.min()) * 255).astype(np.uint8)
        else:
            img_normalized = self.image

        pil_image = Image.fromarray(img_normalized)

        # Convert to RGB for colored annotations
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        draw = ImageDraw.Draw(pil_image)

        # Draw each marked embryo
        for embryo in self.marked_embryos:
            pixel_x, pixel_y = embryo['pixel_position']
            embryo_num = embryo['embryo_number']

            # Draw cross marker
            marker_size = 20
            draw.line(
                [(pixel_x - marker_size, pixel_y), (pixel_x + marker_size, pixel_y)],
                fill=(0, 255, 255),
                width=3
            )
            draw.line(
                [(pixel_x, pixel_y - marker_size), (pixel_x, pixel_y + marker_size)],
                fill=(0, 255, 255),
                width=3
            )

            # Draw circle
            circle_radius = 40
            draw.ellipse(
                [pixel_x - circle_radius, pixel_y - circle_radius,
                 pixel_x + circle_radius, pixel_y + circle_radius],
                outline=(0, 255, 255),
                width=2
            )

            # Draw embryo number
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            text = str(embryo_num)
            # Place text below marker
            text_pos = (pixel_x - 10, pixel_y + circle_radius + 5)
            draw.text(text_pos, text, fill=(0, 255, 255), font=font)

        # Save
        pil_image.save(output_path)
        print(f"  Saved marked image: {output_path}")


def mark_embryos_napari(
    image: np.ndarray,
    initial_stage_position: Tuple[float, float],
    pixel_size_um: float = 0.65,
    save_image_path: Optional[Path] = None
) -> List[Dict]:
    """
    Convenience function for interactive embryo marking with napari.

    Opens napari viewer, allows user to mark embryos, waits for completion,
    and returns marked positions.

    Parameters
    ----------
    image : np.ndarray
        Bottom camera overview image
    initial_stage_position : tuple of float
        Initial XY stage position in micrometers (x, y)
    pixel_size_um : float, optional
        Effective pixel size in micrometers/pixel (default: 0.65)
    save_image_path : Path, optional
        If provided, saves marked image to this path

    Returns
    -------
    list of dict
        Marked embryos with positions and metadata

    Examples
    --------
    >>> import numpy as np
    >>> from gently.visualization import mark_embryos_napari
    >>>
    >>> # Capture bottom camera image
    >>> image = np.random.randint(0, 255, (2048, 2048), dtype=np.uint8)
    >>>
    >>> # Mark embryos interactively
    >>> embryos = mark_embryos_napari(
    ...     image,
    ...     initial_stage_position=(1000.0, 2000.0),
    ...     save_image_path=Path("marked_embryos.png")
    ... )
    >>>
    >>> # Use marked positions
    >>> for embryo in embryos:
    ...     print(f"Moving to {embryo['embryo_id']}...")
    ...     # Move stage, run calibration, etc.
    """
    if not NAPARI_AVAILABLE:
        raise ImportError(
            "napari is required for interactive marking. "
            "Install with: pip install napari[all]"
        )

    # Create marker
    marker = EmbryoMarker(
        image=image,
        initial_stage_position=initial_stage_position,
        pixel_size_um=pixel_size_um
    )

    # Wait for user to finish marking
    marker.wait_for_completion()

    # Get marked embryos
    embryos = marker.get_marked_embryos()

    # Save marked image if requested
    if save_image_path is not None:
        marker.save_marked_image(save_image_path)

    return embryos


# ============================================================================
# NON-BLOCKING MARKER (FOR BLUESKY INTEGRATION)
# ============================================================================

class NonBlockingEmbryoMarker:
    """
    Non-blocking embryo marker for integration with Bluesky RunEngine.

    Unlike EmbryoMarker which blocks on wait_for_completion(), this version
    allows the RunEngine to check marking status without blocking.

    Use with Bluesky's Msg('pause') and resume mechanisms.
    """

    def __init__(
        self,
        image: np.ndarray,
        initial_stage_position: Tuple[float, float],
        pixel_size_um: float = 0.65
    ):
        """Initialize non-blocking marker."""
        self.marker = EmbryoMarker(
            image=image,
            initial_stage_position=initial_stage_position,
            pixel_size_um=pixel_size_um
        )

        # Start napari in separate thread
        self._marker_thread = threading.Thread(target=self._run_marker)
        self._marker_thread.daemon = True
        self._marker_thread.start()

    def _run_marker(self):
        """Run marker in separate thread."""
        self.marker.wait_for_completion()

    def is_complete(self) -> bool:
        """Check if marking is complete (non-blocking)."""
        return self.marker.marking_complete

    def get_marked_embryos(self) -> List[Dict]:
        """Get marked embryos (can be called while marking)."""
        return self.marker.get_marked_embryos()

    def wait_with_timeout(self, timeout_seconds: Optional[float] = None) -> bool:
        """
        Wait for marking to complete with optional timeout.

        Parameters
        ----------
        timeout_seconds : float, optional
            Timeout in seconds. If None, waits indefinitely.

        Returns
        -------
        bool
            True if completed, False if timeout
        """
        self._marker_thread.join(timeout=timeout_seconds)
        return not self._marker_thread.is_alive()
