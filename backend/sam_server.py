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
        pixel_size_um: float = 6.5,
        objective_mag: float = 4.0,
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
                'final_detections': results.get('final_detections', len(embryos)),
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
        pixel_size_um: float = 6.5,
        objective_mag: float = 4.0,
        title: str = "Click on embryo centers (close window when done)"
    ) -> dict:
        """
        Manual embryo marking via matplotlib.

        Opens a matplotlib window where user can click on embryo centers.
        Right-click or close window to finish.

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
        title : str
            Window title

        Returns
        -------
        dict
            Detection results with 'embryos' list
        """
        import matplotlib
        matplotlib.use('TkAgg')  # Use interactive backend
        import matplotlib.pyplot as plt

        print(f"\n[SAM Server] Manual embryo marking...")
        print(f"  Image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
        print(f"  Stage position: {stage_position}")

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

        # Store clicked points
        clicked_points = []

        def onclick(event):
            if event.button == 1 and event.inaxes:  # Left click
                clicked_points.append((event.xdata, event.ydata))
                # Draw marker
                ax.plot(event.xdata, event.ydata, 'r+', markersize=20, markeredgewidth=3)
                ax.annotate(f'{len(clicked_points)}', (event.xdata + 30, event.ydata - 30),
                           color='red', fontsize=12, fontweight='bold')
                fig.canvas.draw()
                print(f"    Marked embryo {len(clicked_points)} at pixel ({event.xdata:.0f}, {event.ydata:.0f})")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Left-click to mark embryos. Close window when done.')

        # Connect click event
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        print("\n  [INTERACTIVE] Click on embryo centers in the window.")
        print("  Close the window when done marking.\n")

        # Show window (blocking)
        plt.show()

        # Disconnect event
        fig.canvas.mpl_disconnect(cid)

        # Convert clicked points to embryo positions
        embryos = []
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        for i, (px, py) in enumerate(clicked_points):
            # Calculate offset from image center in pixels
            dx_pixels = px - image_center_x
            dy_pixels = py - image_center_y

            # Convert to stage coordinates - match multi_embryo_calibration.py formula
            # dx_pixels = (embryo - center), target = current + dx_pixels * pixel_size
            # This gives the TARGET stage position to center this embryo
            dx_stage = dx_pixels * um_per_pixel
            dy_stage = dy_pixels * um_per_pixel

            # Calculate absolute stage position for this embryo
            embryo_x = stage_position[0] + dx_stage
            embryo_y = stage_position[1] + dy_stage

            embryos.append({
                'id': i + 1,
                'pixel_x': float(px),
                'pixel_y': float(py),
                'stage_x': float(embryo_x),
                'stage_y': float(embryo_y),
                'confidence': 1.0,  # Manual marking = 100% confidence
                'source': 'manual'
            })

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
        show: bool = True
    ) -> dict:
        """
        View an image in a matplotlib window.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        title : str
            Window title
        cmap : str
            Colormap
        save_path : str, optional
            Path to save the image
        show : bool
            Whether to show in matplotlib window (blocking)

        Returns
        -------
        dict
            Success status and save path if saved
        """
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        # Convert image to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)

        result = {'success': True}

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            result['saved_to'] = save_path
            print(f"  Saved image to: {save_path}")

        # Show if requested
        if show:
            plt.show()

        plt.close(fig)  # Clean up figure to prevent duplicates

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
        View embryos with bounding boxes overlaid on image.

        Parameters
        ----------
        image : np.ndarray
            Image to display
        embryos : list
            List of embryo dicts with pixel_x, pixel_y, and optionally bbox_pixel
        title : str
            Window title
        save_path : str, optional
            Path to save the image. If provided, saves instead of/in addition to showing.
        show : bool
            Whether to display in matplotlib window (blocking)

        Returns
        -------
        dict
            Success status and save path if saved
        """
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Convert image to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint16)

        # Convert embryos from rpyc netref if needed
        if not isinstance(embryos, list):
            embryos = list(embryos)
        embryos = [dict(e) for e in embryos]

        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(image, cmap='gray')

        # Define colors for embryos
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, embryo in enumerate(embryos):
            color = colors[i % 10]

            # Get pixel position
            px = embryo.get('pixel_x', embryo.get('center_x', 0))
            py = embryo.get('pixel_y', embryo.get('center_y', 0))

            # Draw center marker
            ax.plot(px, py, 'o', color=color, markersize=15,
                   markeredgecolor='white', markeredgewidth=2)

            # Draw bounding box if available
            bbox = embryo.get('bbox_pixel', embryo.get('bbox'))
            if bbox:
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)

            # Add label
            embryo_id = embryo.get('embryo_id', embryo.get('id', i))
            confidence = embryo.get('confidence', embryo.get('stability_score', 0))
            label = f"#{embryo_id}"
            if confidence:
                label += f" ({confidence:.2f})"

            ax.annotate(
                label, (px + 30, py - 30),
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f"Detected {len(embryos)} embryos")
        plt.tight_layout()

        result = {'success': True, 'num_embryos': len(embryos)}

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            result['saved_to'] = save_path
            print(f"  Saved visualization to: {save_path}")

        # Show if requested
        if show:
            plt.show()

        plt.close(fig)
        return result


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
