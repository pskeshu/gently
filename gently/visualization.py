"""
Supports the visual aspects of the gently library
"""

import numpy as np
import matplotlib.pyplot as plt
import napari


def setup_napari_live_view(title: str = "DiSPIM Live View"):
    """
    Simple napari setup for live imaging - returns viewer and image layer

    This matches the pattern in test_embryo_focus.py where you update image_layer.data directly.

    Parameters
    ----------
    title : str
        Window title for napari viewer

    Returns
    -------
    tuple
        (viewer, image_layer) - viewer for manual control, image_layer for direct data updates
    """
    # Create viewer with live image layer
    viewer = napari.Viewer(title=title)
    dummy_image = np.zeros((2048, 2048), dtype=np.uint16)
    image_layer = viewer.add_image(dummy_image, name='Live Image', colormap='gray')

    # Enable continuous autocontrast for optimal viewing
    image_layer.contrast_limits_range = (0, 65535)

    return viewer, image_layer


def create_napari_callback(image_layer, camera_name: str = 'bottom_camera'):
    """
    Create a callback function for Bluesky RunEngine that updates napari image layer

    Parameters
    ----------
    image_layer : napari.layers.Image
        Napari image layer to update
    camera_name : str
        Name of camera device in RunEngine documents

    Returns
    -------
    callable
        Callback function for RE.subscribe()

    TODO: move this function to a callbacks module. The rationale is that this has mostly document
    processing code, which is shared by all callback layer objects.
    """
    def napari_live_update(name, doc):
        """Update napari with live images during acquisition"""
        if name == 'event':
            data = doc.get('data', {})
            if camera_name in data:
                image = data[camera_name]
                image_layer.data = image
                # Force contrast adjustment for each new image
                image_layer.reset_contrast_limits()

    return napari_live_update


def setup_napari_camera_feed(title: str = "DiSPIM Live View", camera_name: str = 'bottom_camera'):
    """
    Single function to setup napari viewer and return camera feed callback

    Combines setup_napari_live_view() + create_napari_callback() into one call.

    Parameters
    ----------
    title : str
        Window title for napari viewer
    camera_name : str
        Name of camera device in RunEngine documents

    Returns
    -------
    tuple
        (viewer, camera_feed_callback) - viewer for manual control, callback for RE.subscribe()
    """
    viewer, image_layer = setup_napari_live_view(title)
    camera_feed_callback = create_napari_callback(image_layer, camera_name)
    return viewer, camera_feed_callback


def create_simple_focus_plotter(title: str = "Focus Analysis"):
    """
    Simple matplotlib plotter for focus curves - no complex threading

    Parameters
    ----------
    title : str
        Plot window title

    Returns
    -------
    callable
        Update function that can be used as callback: update(scan_type, position, score, image, roi)
    """
    # Create figure with focus curve and current image
    fig, (ax_curve, ax_image) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Setup focus curve plot
    ax_curve.set_xlabel('Position (μm)')
    ax_curve.set_ylabel('Focus Score')
    ax_curve.set_title('Focus Score vs Position')
    ax_curve.grid(True, alpha=0.3)

    # Setup image display
    ax_image.set_title('Current Image')
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    # Data storage
    coarse_positions, coarse_scores = [], []
    fine_positions, fine_scores = [], []

    # Turn on interactive mode for live updates
    plt.ion()
    plt.show(block=False)

    def update_plot(scan_type: str, position: float, score: float, image: np.ndarray, roi=None):
        """Update the plot with new focus data"""
        try:
            # Store data points
            if scan_type == 'coarse':
                coarse_positions.append(position)
                coarse_scores.append(score)
            elif scan_type == 'fine':
                fine_positions.append(position)
                fine_scores.append(score)

            # Clear and redraw focus curve
            ax_curve.clear()
            ax_curve.set_xlabel('Position (μm)')
            ax_curve.set_ylabel('Focus Score')
            ax_curve.set_title('Focus Score vs Position')
            ax_curve.grid(True, alpha=0.3)

            # Plot data
            if coarse_positions:
                ax_curve.plot(coarse_positions, coarse_scores, 'bo-',
                             label='Coarse Scan', markersize=6, linewidth=2, alpha=0.7)
            if fine_positions:
                ax_curve.plot(fine_positions, fine_scores, 'ro-',
                             label='Fine Scan', markersize=6, linewidth=2, alpha=0.7)

            if coarse_positions or fine_positions:
                ax_curve.legend()

            # Update current image
            if image is not None:
                ax_image.clear()
                ax_image.imshow(image, cmap='gray', aspect='auto')
                ax_image.set_title('Current Image')
                ax_image.set_xticks([])
                ax_image.set_yticks([])

                # Overlay ROI if provided
                if roi is not None:
                    x, y, w, h = roi
                    rect = plt.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor='red', facecolor='none', alpha=0.8)
                    ax_image.add_patch(rect)
                    ax_image.text(x, y-5, 'Embryo ROI', color='red', fontsize=8)

            # Refresh display
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)  # Small pause to allow GUI update

        except Exception as e:
            print(f"Error updating focus plot: {e}")

    # Add helper methods to the function
    update_plot.save_plot = lambda filename: fig.savefig(filename, dpi=150, bbox_inches='tight')
    update_plot.close = lambda: plt.close(fig)
    update_plot.clear_data = lambda: (coarse_positions.clear(), coarse_scores.clear(),
                                     fine_positions.clear(), fine_scores.clear())

    return update_plot


def create_live_focus_plotter(title: str = "DiSPIM Live Focus Analysis"):
    """
    Convenience function to create a live focus plotter

    Maintains backward compatibility with existing code while using the simple implementation.

    Parameters
    ----------
    title : str
        Plot window title

    Returns
    -------
    callable
        Focus plotter function
    """
    return create_simple_focus_plotter(title)


def add_focus_analysis_markers(plotter, coarse_best: float = None, fine_best: float = None):
    """
    Add vertical lines to mark best focus positions

    Parameters
    ----------
    plotter : callable
        Plotter function returned by create_simple_focus_plotter()
    coarse_best : float, optional
        Best coarse focus position
    fine_best : float, optional
        Best fine focus position
    """
    try:
        # Get the current figure
        fig = plt.gcf()
        ax_curve = fig.axes[0]  # First axis is the curve plot

        if coarse_best is not None:
            ax_curve.axvline(coarse_best, color='blue', linestyle='--', alpha=0.7,
                           label=f'Coarse Best: {coarse_best:.1f}μm')

        if fine_best is not None:
            ax_curve.axvline(fine_best, color='red', linestyle='--', alpha=0.7,
                           label=f'Fine Best: {fine_best:.1f}μm')

        ax_curve.legend()
        plt.draw()

    except Exception as e:
        print(f"Error adding analysis markers: {e}")

