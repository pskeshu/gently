"""
SAM Detection Server for Gently DiSPIM

Lightweight rpyc server that handles SAM + Claude Vision embryo detection.
This runs separately from the Queue Server because SAM detection is not
a Bluesky plan and doesn't need RunEngine.

Usage:
    python backend/sam_server.py

The server listens on port 18862 by default.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rpyc
from rpyc.utils.server import ThreadedServer

# Ensure gently package is importable
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import centralized coordinate conversion functions
from gently.coordinates import (
    pixel_to_stage_position,
    get_um_per_pixel,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_OBJECTIVE_MAG,
)


class SAMService(rpyc.Service):
    """
    RPC Service for SAM embryo detection.

    This service handles all SAM + Claude Vision operations:
    - Embryo detection from images
    - Claude Vision verification
    - Confidence scoring

    The detector is lazy-loaded on first use and shared across all clients.
    """

    # Class-level shared detector (initialized once)
    _detector = None
    _sam_checkpoint = "sam_vit_b_01ec64.pth"
    _sam_model_type = "vit_b"
    _device = "cuda"  # Use GPU by default

    def on_connect(self, conn):
        print("SAM Service: Client connected")

    def on_disconnect(self, conn):
        print("SAM Service: Client disconnected")

    @classmethod
    def _get_detector(cls):
        """Lazy-load the SAM detector"""
        if cls._detector is None:
            print(f"  Initializing SAM detector on {cls._device}...")
            from gently.agent.sam_detection import SAMEmbryoDetector

            cls._detector = SAMEmbryoDetector(
                sam_checkpoint=cls._sam_checkpoint,
                sam_model_type=cls._sam_model_type,
                device=cls._device
            )
            print("  SAM detector ready")
        return cls._detector

    def exposed_detect_embryos(
        self,
        image: np.ndarray,
        stage_position: tuple,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        use_claude_review: bool = True,
        min_confidence: float = 0.7,
        save_visualizations: bool = False,
        output_dir: Optional[str] = None,
        brightness_percentile: float = 99.0,
        min_area: int = 5000,
        max_area: int = 150000,
    ) -> dict:
        """
        Detect embryos using brightness detection + SAM refinement.

        Parameters
        ----------
        image : np.ndarray
            Input image from bottom camera
        stage_position : tuple
            Current (x, y) stage position in micrometers
        pixel_size_um : float
            Camera pixel size in micrometers
        objective_mag : float
            Objective magnification
        use_claude_review : bool
            Whether to use Claude Vision for verification
        min_confidence : float
            Minimum confidence threshold (0-1)
        save_visualizations : bool
            Whether to save debug visualizations
        output_dir : str, optional
            Directory for saving visualizations
        brightness_percentile : float
            Percentile threshold for brightness-based detection.
            99.0 = fewer, confident detections. 98.0 = more. Default: 99.0
        min_area : int
            Minimum embryo area in pixels. Default: 5000
        max_area : int
            Maximum embryo area in pixels. Default: 150000

        Returns
        -------
        dict
            Detection results with:
            - embryos: list of detected embryo dicts
            - initial_detections: count before Claude review
            - final_detections: count after Claude review
            - verification: Claude verification summary
        """
        print(f"\n[SAM Server] Detecting embryos...")
        print(f"  Image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        print(f"  Stage position: {stage_position}")
        print(f"  Use Claude review: {use_claude_review}")

        # Get detector (lazy load)
        detector = self._get_detector()

        # Convert image to numpy if needed (rpyc netref)
        # Must explicitly set dtype as rpyc netrefs lose dtype info
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)
        elif hasattr(image, 'dtype') and not isinstance(image.dtype, np.dtype):
            # Handle rpyc netref dtype
            image = np.array(image, dtype=np.uint16)

        # Convert stage position to tuple
        if not isinstance(stage_position, tuple):
            stage_position = tuple(stage_position)

        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("./detection_results")

        # Run async detection in a new event loop
        # (SAM detector is async because of Claude Vision API)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                detector.detect_embryos(
                    image=image,
                    stage_position=stage_position,
                    pixel_size_um=pixel_size_um,
                    objective_mag=objective_mag,
                    use_claude_review=use_claude_review,
                    save_visualizations=save_visualizations,
                    output_dir=output_path,
                    brightness_percentile=brightness_percentile,
                    min_area=min_area,
                    max_area=max_area
                )
            )

            # Ensure results are serializable (convert numpy types)
            embryos = results.get('embryos', [])
            for embryo in embryos:
                for key, value in embryo.items():
                    if isinstance(value, np.floating):
                        embryo[key] = float(value)
                    elif isinstance(value, np.integer):
                        embryo[key] = int(value)

            print(f"  Detected {len(embryos)} embryos")

            return {
                'embryos': embryos,
                'initial_detections': results.get('initial_detections', len(embryos)),
                'final_detections': len(embryos),
                'verification': results.get('verification', {}),
                'success': True
            }

        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }
        finally:
            loop.close()

    def exposed_get_status(self) -> dict:
        """Get SAM server status"""
        return {
            'detector_loaded': SAMService._detector is not None,
            'sam_model': SAMService._sam_model_type,
            'device': SAMService._device,
        }

    def exposed_manual_mark_embryos(
        self,
        image: np.ndarray,
        stage_position: tuple,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        existing_embryos: list = None,
        title: str = "Click on embryo centers (close window when done)"
    ) -> dict:
        """
        Manual embryo marking via napari.

        Opens a napari viewer where user can click to add points.
        Close the viewer window when done.

        Parameters
        ----------
        image : np.ndarray
            Input image from bottom camera
        stage_position : tuple
            Current (x, y) stage position in micrometers
        pixel_size_um : float
            Camera pixel size in micrometers
        objective_mag : float
            Objective magnification
        existing_embryos : list, optional
            List of existing embryos to display. Each should have
            'embryo_id', 'pixel_x', 'pixel_y' keys.
        title : str
            Window title

        Returns
        -------
        dict
            Detection results with 'embryos' list
        """
        import napari

        print(f"\n[SAM Server] Manual embryo marking...")
        print(f"  Image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        print(f"  Stage position: {stage_position}")
        if existing_embryos:
            print(f"  Existing embryos: {len(existing_embryos)}")

        # Convert image to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)
        elif hasattr(image, 'dtype') and not isinstance(image.dtype, np.dtype):
            image = np.array(image, dtype=np.uint16)

        # Convert stage position
        if not isinstance(stage_position, tuple):
            stage_position = tuple(stage_position)

        # Calculate scale
        um_per_pixel = pixel_size_um / objective_mag

        # Create napari viewer
        viewer = napari.Viewer(title=title)

        # Add image layer
        viewer.add_image(image, name='Bottom Camera', colormap='gray')

        # Add existing embryos as green points
        if existing_embryos:
            existing_points = []
            existing_labels = []
            for emb in existing_embryos:
                px = emb.get('pixel_x')
                py = emb.get('pixel_y')
                eid = emb.get('embryo_id', '?')
                if px is not None and py is not None:
                    existing_points.append([py, px])  # napari uses [row, col] = [y, x]
                    existing_labels.append(eid)

            if existing_points:
                viewer.add_points(
                    existing_points,
                    name='Existing Embryos',
                    face_color='green',
                    border_color='white',
                    size=30,
                    symbol='cross'
                )

        # Add points layer for new embryos (user will add points here)
        new_points_layer = viewer.add_points(
            name='New Embryos (click to add)',
            face_color='red',
            border_color='white',
            size=30,
            symbol='cross'
        )
        new_points_layer.mode = 'add'  # Start in add mode

        print("\n  [INTERACTIVE] Click on embryo centers in the napari window.")
        print("  Use the 'New Embryos' layer to add points.")
        print("  Close the window when done marking.\n")

        # Run napari (blocking)
        napari.run()

        # Get the points that were added
        clicked_points = new_points_layer.data  # Array of [y, x] coordinates

        # Convert clicked points to embryo positions
        embryos = []
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        for i, point in enumerate(clicked_points):
            py, px = point[0], point[1]  # napari stores as [y, x]

            # Convert pixel to stage position using centralized function
            embryo_x, embryo_y = pixel_to_stage_position(
                pixel_x=px,
                pixel_y=py,
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                stage_x=stage_position[0],
                stage_y=stage_position[1],
                um_per_pixel=um_per_pixel
            )

            embryos.append({
                'embryo_id': f'manual_{i + 1}',
                'pixel_x': float(px),
                'pixel_y': float(py),
                'stage_x_um': float(embryo_x),
                'stage_y_um': float(embryo_y),
                'confidence': 1.0,
                'source': 'manual'
            })
            print(f"    Marked embryo {i + 1} at pixel ({px:.0f}, {py:.0f})")

        print(f"  Marked {len(embryos)} embryos manually")

        return {
            'embryos': embryos,
            'initial_detections': len(embryos),
            'final_detections': len(embryos),
            'method': 'manual',
            'success': True
        }

    def exposed_view_image(
        self,
        image: np.ndarray,
        title: str = "Image View",
        cmap: str = "gray",
        save_path: Optional[str] = None,
        show: bool = True,
        embryo_annotations: list = None
    ) -> dict:
        """
        View an image in a napari window with optional embryo annotations.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        title : str
            Window title
        cmap : str
            Colormap (napari colormap name)
        save_path : str, optional
            Path to save the image
        show : bool
            Whether to show in napari window (blocking)
        embryo_annotations : list, optional
            List of embryo dicts with 'embryo_id', 'pixel_x', 'pixel_y', 'label'

        Returns
        -------
        dict
            Success status and save path if saved
        """
        import napari

        # Convert image to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)

        result = {'success': True}

        # Save if path provided (before showing to ensure it's saved even if viewer is closed early)
        if save_path:
            import tifffile
            tifffile.imwrite(save_path, image)
            result['saved_to'] = save_path
            print(f"  Saved image to: {save_path}")

        # Show if requested
        if show:
            viewer = napari.Viewer(title=title)

            # Add image layer
            viewer.add_image(image, name='Image', colormap=cmap)

            # Draw embryo annotations if provided
            if embryo_annotations:
                in_view_points = []
                out_of_view_points = []
                in_view_labels = []
                out_of_view_labels = []

                for emb in embryo_annotations:
                    px = emb.get('pixel_x')
                    py = emb.get('pixel_y')
                    label = emb.get('label', emb.get('embryo_id', '?'))

                    if px is not None and py is not None:
                        in_view = 0 <= px < image.shape[1] and 0 <= py < image.shape[0]
                        if in_view:
                            in_view_points.append([py, px])
                            in_view_labels.append(label)
                        else:
                            out_of_view_points.append([py, px])
                            out_of_view_labels.append(label)

                # Add in-view embryos (green)
                if in_view_points:
                    viewer.add_points(
                        in_view_points,
                        name='Embryos (in view)',
                        face_color='lime',
                        border_color='white',
                        size=30,
                        symbol='cross'
                    )

                # Add out-of-view embryos (orange)
                if out_of_view_points:
                    viewer.add_points(
                        out_of_view_points,
                        name='Embryos (out of view)',
                        face_color='orange',
                        border_color='white',
                        size=30,
                        symbol='cross'
                    )

            napari.run()

        return result

    def exposed_view_embryos(
        self,
        image: np.ndarray,
        embryos: list,
        title: str = "Detected Embryos",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> dict:
        """
        View embryos with markers overlaid on image using napari.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        embryos : list
            List of embryo dicts with pixel_x, pixel_y, and optionally bbox_pixel
        title : str
            Window title
        save_path : str, optional
            Path to save the image.
        show : bool
            Whether to display in napari window (blocking)

        Returns
        -------
        dict
            Success status and save path if saved
        """
        import napari

        # Convert image to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)

        # Convert embryos from rpyc netref if needed
        if not isinstance(embryos, list):
            embryos = list(embryos)
        embryos = [dict(e) for e in embryos]

        result = {'success': True, 'num_embryos': len(embryos)}

        # Save if path provided
        if save_path:
            from pathlib import Path
            from PIL import Image as PILImage

            # Normalize image to 8-bit for saving
            if image.dtype != np.uint8:
                img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                img_normalized = image

            pil_img = PILImage.fromarray(img_normalized)

            # Use appropriate format based on extension
            ext = Path(save_path).suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                pil_img.save(save_path, 'JPEG', quality=70, optimize=True)
            elif ext == '.png':
                pil_img.save(save_path, 'PNG', optimize=True)
            else:
                # Fallback to tifffile for other formats
                import tifffile
                tifffile.imwrite(save_path, image)

            result['saved_to'] = save_path
            print(f"  Saved image to: {save_path}")

        # Show if requested
        if show:
            viewer = napari.Viewer(title=title)

            # Add image layer
            viewer.add_image(image, name='Image', colormap='gray')

            # Collect embryo points and colors
            points = []
            colors = []
            labels = []

            # Define distinct colors for embryos
            color_palette = [
                'red', 'blue', 'green', 'yellow', 'magenta',
                'cyan', 'orange', 'purple', 'pink', 'lime'
            ]

            for i, embryo in enumerate(embryos):
                px = embryo.get('pixel_x', embryo.get('center_x', 0))
                py = embryo.get('pixel_y', embryo.get('center_y', 0))

                points.append([py, px])  # napari uses [row, col] = [y, x]
                colors.append(color_palette[i % len(color_palette)])

                embryo_id = embryo.get('embryo_id', embryo.get('id', i))
                confidence = embryo.get('confidence', embryo.get('stability_score', 0))
                label = f"#{embryo_id}"
                if confidence:
                    label += f" ({confidence:.2f})"
                labels.append(label)

            # Add points layer for embryos with text labels
            if points:
                # Build properties dict for text display
                properties = {'label': labels}

                viewer.add_points(
                    points,
                    name=f'Embryos ({len(embryos)})',
                    face_color=colors,
                    border_color='white',
                    size=40,
                    symbol='disc',
                    properties=properties,
                    text={
                        'string': '{label}',
                        'size': 14,
                        'color': 'white',
                        'anchor': 'upper_left',
                    }
                )

                print(f"  Showing {len(embryos)} embryos:")
                for i, label in enumerate(labels):
                    print(f"    {label} at ({points[i][1]:.0f}, {points[i][0]:.0f})")

            napari.run()

        return result


    def exposed_edit_embryos(
        self,
        image: np.ndarray,
        embryos: list,
        stage_position: tuple,
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        title: str = "Edit Embryos - Add/Delete/Move points, then close window"
    ) -> dict:
        """
        Interactive embryo editor using napari.

        Allows adding, deleting, and moving embryo positions.
        - Click to add new embryos
        - Select + Delete key to remove
        - Drag to move existing embryos

        Parameters
        ----------
        image : np.ndarray
            Input image from bottom camera
        embryos : list
            List of existing embryo dicts with pixel_x, pixel_y, embryo_id
        stage_position : tuple
            Current (x, y) stage position in micrometers
        pixel_size_um : float
            Camera pixel size in micrometers
        objective_mag : float
            Objective magnification
        title : str
            Window title

        Returns
        -------
        dict
            Updated embryo list with:
            - embryos: list of all embryos (original + new, minus deleted)
            - added: count of new embryos
            - removed: count of removed embryos
            - moved: count of moved embryos
        """
        import napari

        print(f"\n[SAM Server] Interactive embryo editing...")
        print(f"  Image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        print(f"  Existing embryos: {len(embryos)}")

        # Convert image to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)
        elif hasattr(image, 'dtype') and not isinstance(image.dtype, np.dtype):
            image = np.array(image, dtype=np.uint16)

        # Convert embryos from rpyc netref if needed
        if not isinstance(embryos, list):
            embryos = list(embryos)
        embryos = [dict(e) for e in embryos]

        # Convert stage position
        if not isinstance(stage_position, tuple):
            stage_position = tuple(stage_position)

        um_per_pixel = pixel_size_um / objective_mag
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        # Build initial points array and track original positions
        original_points = []
        original_ids = []
        for emb in embryos:
            px = emb.get('pixel_x')
            py = emb.get('pixel_y')
            if px is not None and py is not None:
                original_points.append([py, px])  # napari [row, col]
                original_ids.append(emb.get('embryo_id', f'embryo_{len(original_ids)+1}'))

        # Create napari viewer
        viewer = napari.Viewer(title=title)

        # Add image layer
        viewer.add_image(image, name='Bottom Camera', colormap='gray')

        # Add editable points layer
        points_layer = viewer.add_points(
            original_points if original_points else None,
            name='Embryos (editable)',
            face_color='lime',
            border_color='white',
            size=35,
            symbol='cross'
        )
        points_layer.mode = 'select'  # Start in select mode for editing

        # Store original state for comparison
        original_points_set = set(tuple(p) for p in original_points)

        print("\n  [INTERACTIVE] Edit embryos in napari:")
        print("    - Press '2' or click 'Add points' to add new embryos")
        print("    - Press '3' or click 'Select points' to select existing")
        print("    - Press Delete/Backspace to remove selected points")
        print("    - Drag points to move them")
        print("    - Close window when done\n")

        # Run napari (blocking)
        napari.run()

        # Get final points
        final_points = points_layer.data

        # Analyze changes
        final_points_set = set(tuple(p) for p in final_points)

        # Points that were in original but not in final = removed
        removed_points = original_points_set - final_points_set

        # Points that are in final but not in original = added (or moved)
        new_points = final_points_set - original_points_set

        # Build final embryo list
        final_embryos = []
        next_id = len(embryos) + 1

        for i, point in enumerate(final_points):
            py, px = point[0], point[1]

            # Check if this point matches an original (not moved)
            point_tuple = tuple(point)
            original_idx = None
            for j, orig_point in enumerate(original_points):
                if tuple(orig_point) == point_tuple:
                    original_idx = j
                    break

            if original_idx is not None:
                # Existing embryo, keep original data
                emb = embryos[original_idx].copy()
            else:
                # New or moved embryo - calculate stage coordinates using centralized function
                stage_x_um, stage_y_um = pixel_to_stage_position(
                    pixel_x=px,
                    pixel_y=py,
                    image_center_x=image_center_x,
                    image_center_y=image_center_y,
                    stage_x=stage_position[0],
                    stage_y=stage_position[1],
                    um_per_pixel=um_per_pixel
                )

                emb = {
                    'embryo_id': f'embryo_{next_id}',
                    'pixel_x': float(px),
                    'pixel_y': float(py),
                    'stage_x_um': float(stage_x_um),
                    'stage_y_um': float(stage_y_um),
                    'confidence': 1.0,
                    'source': 'manual_edit'
                }
                next_id += 1

            # Always update pixel coordinates from napari
            emb['pixel_x'] = float(px)
            emb['pixel_y'] = float(py)

            # Recalculate stage coordinates using centralized function
            stage_x_um, stage_y_um = pixel_to_stage_position(
                pixel_x=px,
                pixel_y=py,
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                stage_x=stage_position[0],
                stage_y=stage_position[1],
                um_per_pixel=um_per_pixel
            )
            emb['stage_x_um'] = float(stage_x_um)
            emb['stage_y_um'] = float(stage_y_um)

            final_embryos.append(emb)

        # Summary
        added = len(new_points)
        removed = len(removed_points)
        moved = len([p for p in final_points if tuple(p) not in original_points_set]) - added

        print(f"  Edit complete:")
        print(f"    Original: {len(embryos)} embryos")
        print(f"    Final: {len(final_embryos)} embryos")
        print(f"    Added: {added}, Removed: {removed}")

        return {
            'embryos': final_embryos,
            'original_count': len(embryos),
            'final_count': len(final_embryos),
            'added': added,
            'removed': removed,
            'success': True
        }


def start_sam_server(
    port: int = 18862,
    hostname: str = 'localhost',
    sam_checkpoint: str = "sam_vit_b_01ec64.pth",
    sam_model_type: str = "vit_b",
    device: str = "cuda"
):
    """
    Start the SAM detection server.

    Parameters
    ----------
    port : int
        Port to listen on (default: 18862)
    hostname : str
        Hostname to bind to (default: localhost)
    sam_checkpoint : str
        Path to SAM model checkpoint
    sam_model_type : str
        SAM model type (vit_b, vit_l, vit_h)
    device : str
        Device to run SAM on (cuda or cpu)
    """
    print("=" * 60)
    print("SAM DETECTION SERVER")
    print("=" * 60)

    # Configure SAM model
    SAMService._sam_checkpoint = sam_checkpoint
    SAMService._sam_model_type = sam_model_type
    SAMService._device = device

    print(f"\nConfiguration:")
    print(f"  SAM checkpoint: {sam_checkpoint}")
    print(f"  SAM model type: {sam_model_type}")
    print(f"  Device: {device}")

    # Configure rpyc
    config = {
        'allow_all_attrs': True,
        'allow_pickle': True,
        'sync_request_timeout': 300,  # 5 minute timeout for detection
    }

    server = ThreadedServer(
        SAMService,
        hostname=hostname,
        port=port,
        protocol_config=config
    )

    print(f"\n" + "=" * 60)
    print(f"SAM Server ready at {hostname}:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\nShutting down SAM Server...")
    finally:
        server.close()
        print("SAM Server stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM Detection Server")
    parser.add_argument("--port", type=int, default=18862, help="Port to listen on")
    parser.add_argument("--hostname", default="localhost", help="Hostname to bind to")
    parser.add_argument("--sam-checkpoint", default="sam_vit_b_01ec64.pth",
                        help="Path to SAM checkpoint")
    parser.add_argument("--sam-model-type", default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run SAM on (default: cuda)")

    args = parser.parse_args()

    start_sam_server(
        port=args.port,
        hostname=args.hostname,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device
    )
