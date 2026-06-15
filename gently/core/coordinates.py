"""
Coordinate System Conversions for DiSPIM Microscope

Hardware behavior (confirmed empirically 2024-11-30):
- Stage +X → embryo moves LEFT in camera
- Stage +Y → embryo moves DOWN in camera

Position calculation (pixel → stage position):
- X: NOT inverted (stage_x + dx_pixels * um_per_pixel)
- Y: IS inverted (stage_y - dy_pixels * um_per_pixel)

This was determined by measuring centering error experimentally.

This module is the SINGLE SOURCE OF TRUTH for coordinate conversions.
All other files should import from here.
"""

# Default optical parameters for bottom detection camera
DEFAULT_PIXEL_SIZE_UM = 6.5
DEFAULT_OBJECTIVE_MAG = 10.0  # 10x objective on bottom camera

# SPIM light-sheet path (used for volume acquisition)
SPIM_OBJECTIVE_MAG = 40.0  # 40x SPIM objectives
SPIM_PIXEL_SIZE_UM = DEFAULT_PIXEL_SIZE_UM / SPIM_OBJECTIVE_MAG  # 0.1625 µm/pixel
SPIM_Z_STEP_UM = 1.0  # Z-step between slices


def get_um_per_pixel(
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    objective_mag: float = DEFAULT_OBJECTIVE_MAG,
) -> float:
    """
    Calculate microns per pixel for the optical system.

    Parameters
    ----------
    pixel_size_um : float
        Physical pixel size in micrometers (default: 6.5 for bottom camera)
    objective_mag : float
        Objective magnification (default: 10.0)

    Returns
    -------
    float
        Effective pixel size in micrometers (um_per_pixel)
    """
    return pixel_size_um / objective_mag


def pixel_to_stage_position(
    pixel_x: float,
    pixel_y: float,
    image_center_x: float,
    image_center_y: float,
    stage_x: float,
    stage_y: float,
    um_per_pixel: float | None = None,
) -> tuple[float, float]:
    """
    Convert pixel coordinates to stage position (for embryo POSITION calculation).

    Given an embryo at (pixel_x, pixel_y) in an image captured at stage position
    (stage_x, stage_y), returns the stage position that would CENTER this embryo.

    NO X-axis inversion - image coords map directly to stage coords.

    Parameters
    ----------
    pixel_x, pixel_y : float
        Pixel coordinates of the embryo in the image
    image_center_x, image_center_y : float
        Center of the image in pixels
    stage_x, stage_y : float
        Stage position when image was captured (in micrometers)
    um_per_pixel : float, optional
        Microns per pixel. If None, uses default optical parameters.

    Returns
    -------
    Tuple[float, float]
        (embryo_stage_x, embryo_stage_y) - the stage position that would center this embryo

    Example
    -------
    >>> # Embryo is 100 pixels to the right of center
    >>> # Stage is at X=1000
    >>> stage_x, stage_y = pixel_to_stage_position(
    ...     pixel_x=612, pixel_y=512,  # embryo position
    ...     image_center_x=512, image_center_y=512,  # image center
    ...     stage_x=1000, stage_y=2000,  # current stage position
    ...     um_per_pixel=1.625
    ... )
    >>> # stage_x = 1000 + 100 * 1.625 = 1162.5
    """
    if um_per_pixel is None:
        um_per_pixel = get_um_per_pixel()

    dx_pixels = pixel_x - image_center_x
    dy_pixels = pixel_y - image_center_y

    # X: NOT inverted (stage coords match image coords for X)
    # Y: IS inverted (stage +Y moves embryo down, but image +Y is also down,
    # need to invert for centering)
    embryo_stage_x = stage_x + dx_pixels * um_per_pixel
    embryo_stage_y = stage_y - dy_pixels * um_per_pixel

    return embryo_stage_x, embryo_stage_y


def stage_to_pixel_position(
    stage_x: float,
    stage_y: float,
    current_stage_x: float,
    current_stage_y: float,
    image_center_x: float,
    image_center_y: float,
    um_per_pixel: float | None = None,
) -> tuple[float, float]:
    """
    Convert stage position to pixel coordinates (for DISPLAY/visualization).

    Given an embryo at stage position (stage_x, stage_y), returns where it would
    appear in an image captured at (current_stage_x, current_stage_y).

    This is the inverse of pixel_to_stage_position.

    Parameters
    ----------
    stage_x, stage_y : float
        Stage position of the embryo (in micrometers)
    current_stage_x, current_stage_y : float
        Current stage position when displaying (in micrometers)
    image_center_x, image_center_y : float
        Center of the image in pixels
    um_per_pixel : float, optional
        Microns per pixel. If None, uses default optical parameters.

    Returns
    -------
    Tuple[float, float]
        (pixel_x, pixel_y) - where the embryo would appear in the current image
    """
    if um_per_pixel is None:
        um_per_pixel = get_um_per_pixel()

    dx_stage = stage_x - current_stage_x
    dy_stage = stage_y - current_stage_y

    # X: NOT inverted, Y: IS inverted (inverse of pixel_to_stage_position)
    pixel_x = image_center_x + dx_stage / um_per_pixel
    pixel_y = image_center_y - dy_stage / um_per_pixel

    return pixel_x, pixel_y


def pixel_displacement_to_stage_movement(
    pixel_displacement_x: float, pixel_displacement_y: float, um_per_pixel: float | None = None
) -> tuple[float, float]:
    """
    Convert pixel displacement to stage MOVEMENT (for centering an embryo).

    Given that we want to move an embryo by (pixel_displacement_x, pixel_displacement_y)
    pixels visually, returns how much the stage should move.

    X-axis IS inverted here because stage +X causes visual LEFT movement.

    Parameters
    ----------
    pixel_displacement_x : float
        Desired visual displacement in X (positive = move right in image)
    pixel_displacement_y : float
        Desired visual displacement in Y (positive = move down in image)
    um_per_pixel : float, optional
        Microns per pixel. If None, uses default optical parameters.

    Returns
    -------
    Tuple[float, float]
        (dx_stage, dy_stage) - how much to move the stage in micrometers

    Example
    -------
    >>> # Embryo is to the RIGHT of center, we want to move it LEFT (to center)
    >>> # pixel_displacement = center - embryo = negative (move left)
    >>> dx, dy = pixel_displacement_to_stage_movement(-100, 0, um_per_pixel=1.625)
    >>> # dx = -(-100) * 1.625 = +162.5 (stage moves +X, which moves embryo left)
    """
    if um_per_pixel is None:
        um_per_pixel = get_um_per_pixel()

    # X inverted for MOVEMENT (stage +X → visual LEFT)
    dx_stage = -pixel_displacement_x * um_per_pixel
    dy_stage = pixel_displacement_y * um_per_pixel

    return dx_stage, dy_stage
