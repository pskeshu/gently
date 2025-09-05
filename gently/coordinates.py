"""
Gently DiSPIM Coordinates
========================

Coordinate transformation and reference mapping utilities for DiSPIM microscopy.
Handles conversions between different coordinate systems and maintains reference 
maps for embryo detection and light sheet acquisition workflows.

Device-agnostic coordinate transformations that work with any positioning devices.
Foundation for bottom camera → light sheet coordinate mapping and multi-embryo workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
import json
import time


class CoordinateSystem(NamedTuple):
    """Define a coordinate system with origin and axes"""
    name: str
    origin: Tuple[float, float, float]  # (x, y, z) origin
    x_axis: Tuple[float, float, float]  # Unit vector for X axis  
    y_axis: Tuple[float, float, float]  # Unit vector for Y axis
    z_axis: Tuple[float, float, float]  # Unit vector for Z axis


@dataclass 
class CalibrationPoint:
    """Single calibration point with positions in different coordinate systems"""
    id: str
    timestamp: float
    # Piezo/galvo coordinates
    piezo_position: float
    galvo_position: float
    # XY stage coordinates  
    x_position: float
    y_position: float
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceMap:
    """Reference coordinate map for DiSPIM system"""
    name: str
    created: float
    last_updated: float
    
    # Calibration data
    calibration_points: List[CalibrationPoint] = field(default_factory=list)
    
    # Transformation parameters
    piezo_galvo_slope: float = 1.0
    piezo_galvo_offset: float = 0.0
    piezo_galvo_r_squared: float = 0.0
    
    # XY stage to light sheet transformations (if calibrated)
    stage_to_lightsheet_transform: Optional[np.ndarray] = None  # 4x4 transformation matrix
    
    # Embryo positions (in different coordinate systems)
    embryo_positions_stage: List[Tuple[float, float]] = field(default_factory=list)
    embryo_positions_lightsheet: List[Dict[str, float]] = field(default_factory=list)  # side A/B coords
    
    # Quality metrics
    calibration_quality: float = 0.0
    valid: bool = False


# =============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# =============================================================================

def piezo_to_galvo(piezo_position: float, slope: float, offset: float) -> float:
    """
    Convert piezo position to galvo position using linear calibration
    
    Based on DiSPIM calibration: galvo = (piezo - offset) / slope
    
    Parameters
    ----------
    piezo_position : float
        Piezo position in micrometers
    slope : float  
        Calibration slope (um/degree)
    offset : float
        Calibration offset in micrometers
        
    Returns
    -------
    float
        Corresponding galvo position in degrees
    """
    if slope == 0:
        logging.getLogger(__name__).warning("Calibration slope is zero, returning 0")
        return 0.0
    
    return (piezo_position - offset) / slope


def galvo_to_piezo(galvo_position: float, slope: float, offset: float) -> float:
    """
    Convert galvo position to piezo position using linear calibration
    
    Based on DiSPIM calibration: piezo = slope * galvo + offset
    
    Parameters
    ----------
    galvo_position : float
        Galvo position in degrees
    slope : float
        Calibration slope (um/degree)
    offset : float
        Calibration offset in micrometers
        
    Returns
    -------
    float
        Corresponding piezo position in micrometers
    """
    return slope * galvo_position + offset


def calculate_piezo_galvo_calibration(calibration_points: List[CalibrationPoint]
                                    ) -> Tuple[float, float, float]:
    """
    Calculate linear calibration between piezo and galvo positions
    
    Fits linear relationship: piezo = slope * galvo + offset
    
    Parameters
    ----------
    calibration_points : List[CalibrationPoint]
        List of calibration points with piezo and galvo positions
        
    Returns
    -------
    Tuple[float, float, float]
        (slope, offset, r_squared)
    """
    if len(calibration_points) < 2:
        raise ValueError("Need at least 2 calibration points")
    
    # Extract piezo and galvo positions
    piezo_positions = np.array([cp.piezo_position for cp in calibration_points])
    galvo_positions = np.array([cp.galvo_position for cp in calibration_points])
    
    # Linear fit: piezo = slope * galvo + offset
    coeffs = np.polyfit(galvo_positions, piezo_positions, 1)
    slope, offset = coeffs
    
    # Calculate R-squared
    piezo_pred = slope * galvo_positions + offset
    ss_res = np.sum((piezo_positions - piezo_pred)**2)
    ss_tot = np.sum((piezo_positions - np.mean(piezo_positions))**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    logging.getLogger(__name__).info(
        f"Piezo-galvo calibration: slope={slope:.4f} um/deg, "
        f"offset={offset:.2f} um, R²={r_squared:.4f}"
    )
    
    return slope, offset, r_squared


def transform_coordinates_2d(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Apply 2D coordinate transformation using homogeneous coordinates
    
    Parameters
    ----------
    points : np.ndarray
        Array of points to transform, shape (N, 2)
    transform_matrix : np.ndarray  
        3x3 transformation matrix for 2D homogeneous coordinates
        
    Returns
    -------
    np.ndarray
        Transformed points, shape (N, 2)
    """
    if points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")
    
    if transform_matrix.shape != (3, 3):
        raise ValueError("Transform matrix must be 3x3")
    
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_homo = points_homo @ transform_matrix.T
    
    # Convert back to Cartesian coordinates
    transformed = transformed_homo[:, :2] / transformed_homo[:, 2:3]
    
    return transformed


def create_affine_transform_2d(scale_x: float = 1.0, scale_y: float = 1.0,
                              rotation_deg: float = 0.0, 
                              translation_x: float = 0.0, translation_y: float = 0.0) -> np.ndarray:
    """
    Create 2D affine transformation matrix
    
    Parameters
    ----------
    scale_x, scale_y : float
        Scaling factors for X and Y axes
    rotation_deg : float
        Rotation angle in degrees
    translation_x, translation_y : float
        Translation in X and Y
        
    Returns
    -------
    np.ndarray
        3x3 transformation matrix
    """
    # Convert rotation to radians
    rotation_rad = np.radians(rotation_deg)
    cos_r = np.cos(rotation_rad)
    sin_r = np.sin(rotation_rad)
    
    # Create transformation matrix
    transform = np.array([
        [scale_x * cos_r, -scale_y * sin_r, translation_x],
        [scale_x * sin_r,  scale_y * cos_r, translation_y],
        [0,                0,               1]
    ])
    
    return transform


# =============================================================================
# REFERENCE MAP MANAGEMENT
# =============================================================================

def create_reference_map(name: str) -> ReferenceMap:
    """Create a new reference map"""
    now = time.time()
    return ReferenceMap(
        name=name,
        created=now,
        last_updated=now
    )


def add_calibration_point(ref_map: ReferenceMap, point: CalibrationPoint) -> None:
    """Add calibration point to reference map"""
    ref_map.calibration_points.append(point)
    ref_map.last_updated = time.time()
    
    # Recalculate calibration if we have enough points
    if len(ref_map.calibration_points) >= 2:
        try:
            slope, offset, r_squared = calculate_piezo_galvo_calibration(ref_map.calibration_points)
            ref_map.piezo_galvo_slope = slope
            ref_map.piezo_galvo_offset = offset
            ref_map.piezo_galvo_r_squared = r_squared
            ref_map.calibration_quality = r_squared
            ref_map.valid = r_squared > 0.8  # Quality threshold
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to update calibration: {e}")


def add_embryo_position_stage(ref_map: ReferenceMap, x: float, y: float) -> None:
    """Add embryo position in stage coordinates"""
    ref_map.embryo_positions_stage.append((x, y))
    ref_map.last_updated = time.time()


def add_embryo_position_lightsheet(ref_map: ReferenceMap, side_a_coords: Dict[str, float],
                                  side_b_coords: Optional[Dict[str, float]] = None) -> None:
    """Add embryo position in light sheet coordinates"""
    lightsheet_pos = {'side_a': side_a_coords}
    if side_b_coords:
        lightsheet_pos['side_b'] = side_b_coords
    
    ref_map.embryo_positions_lightsheet.append(lightsheet_pos)
    ref_map.last_updated = time.time()


def stage_to_lightsheet_coordinates(stage_x: float, stage_y: float, 
                                   ref_map: ReferenceMap, side: str = 'A'
                                  ) -> Optional[Dict[str, float]]:
    """
    Convert stage coordinates to light sheet coordinates
    
    This is a placeholder for future development when the coordinate
    transformation between stage and light sheet is calibrated.
    """
    if not ref_map.valid:
        logging.getLogger(__name__).warning("Reference map not valid for coordinate transformation")
        return None
    
    if ref_map.stage_to_lightsheet_transform is None:
        logging.getLogger(__name__).warning("Stage to light sheet transformation not calibrated")
        return None
    
    # Apply transformation (placeholder - would use actual calibrated transform)
    # For now, assume simple 1:1 mapping
    lightsheet_coords = {
        'piezo': 75.0,  # Default center position
        'galvo': 0.0,   # Default center position
        'x_offset': stage_x,
        'y_offset': stage_y
    }
    
    return lightsheet_coords


def find_nearest_embryos(query_position: Tuple[float, float], ref_map: ReferenceMap,
                        max_distance: float = 1000.0, max_results: int = 5
                       ) -> List[Tuple[int, float, Tuple[float, float]]]:
    """
    Find embryos near a query position
    
    Parameters
    ----------
    query_position : Tuple[float, float]
        Query position (x, y) in stage coordinates
    ref_map : ReferenceMap
        Reference map containing embryo positions
    max_distance : float
        Maximum distance to search (micrometers)
    max_results : int
        Maximum number of results to return
        
    Returns
    -------
    List[Tuple[int, float, Tuple[float, float]]]
        List of (index, distance, position) for nearest embryos
    """
    if not ref_map.embryo_positions_stage:
        return []
    
    # Calculate distances
    query_array = np.array([query_position])
    embryo_array = np.array(ref_map.embryo_positions_stage)
    
    distances = cdist(query_array, embryo_array)[0]
    
    # Filter by maximum distance
    valid_indices = distances <= max_distance
    valid_distances = distances[valid_indices]
    valid_positions = embryo_array[valid_indices]
    valid_idx_original = np.where(valid_indices)[0]
    
    # Sort by distance and limit results
    sorted_indices = np.argsort(valid_distances)[:max_results]
    
    results = []
    for i in sorted_indices:
        original_idx = valid_idx_original[i]
        distance = valid_distances[i]
        position = tuple(valid_positions[i])
        results.append((original_idx, distance, position))
    
    return results


# =============================================================================
# REFERENCE MAP PERSISTENCE
# =============================================================================

def save_reference_map(ref_map: ReferenceMap, filepath: str) -> None:
    """Save reference map to JSON file"""
    # Convert to serializable format
    data = {
        'name': ref_map.name,
        'created': ref_map.created,
        'last_updated': ref_map.last_updated,
        'calibration_points': [],
        'piezo_galvo_slope': ref_map.piezo_galvo_slope,
        'piezo_galvo_offset': ref_map.piezo_galvo_offset,
        'piezo_galvo_r_squared': ref_map.piezo_galvo_r_squared,
        'embryo_positions_stage': ref_map.embryo_positions_stage,
        'embryo_positions_lightsheet': ref_map.embryo_positions_lightsheet,
        'calibration_quality': ref_map.calibration_quality,
        'valid': ref_map.valid
    }
    
    # Convert calibration points
    for cp in ref_map.calibration_points:
        point_data = {
            'id': cp.id,
            'timestamp': cp.timestamp,
            'piezo_position': cp.piezo_position,
            'galvo_position': cp.galvo_position,
            'x_position': cp.x_position,
            'y_position': cp.y_position,
            'metadata': cp.metadata
        }
        data['calibration_points'].append(point_data)
    
    # Handle transformation matrix
    if ref_map.stage_to_lightsheet_transform is not None:
        data['stage_to_lightsheet_transform'] = ref_map.stage_to_lightsheet_transform.tolist()
    
    # Save to file
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logging.getLogger(__name__).info(f"Reference map saved to {filepath}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save reference map: {e}")
        raise


def load_reference_map(filepath: str) -> ReferenceMap:
    """Load reference map from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create reference map
        ref_map = ReferenceMap(
            name=data['name'],
            created=data['created'],
            last_updated=data['last_updated'],
            piezo_galvo_slope=data['piezo_galvo_slope'],
            piezo_galvo_offset=data['piezo_galvo_offset'],
            piezo_galvo_r_squared=data['piezo_galvo_r_squared'],
            embryo_positions_stage=data['embryo_positions_stage'],
            embryo_positions_lightsheet=data['embryo_positions_lightsheet'],
            calibration_quality=data['calibration_quality'],
            valid=data['valid']
        )
        
        # Load calibration points
        for point_data in data['calibration_points']:
            cp = CalibrationPoint(
                id=point_data['id'],
                timestamp=point_data['timestamp'],
                piezo_position=point_data['piezo_position'],
                galvo_position=point_data['galvo_position'],
                x_position=point_data['x_position'],
                y_position=point_data['y_position'],
                metadata=point_data['metadata']
            )
            ref_map.calibration_points.append(cp)
        
        # Load transformation matrix if present
        if 'stage_to_lightsheet_transform' in data:
            ref_map.stage_to_lightsheet_transform = np.array(data['stage_to_lightsheet_transform'])
        
        logging.getLogger(__name__).info(f"Reference map loaded from {filepath}")
        return ref_map
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load reference map: {e}")
        raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_reference_map(ref_map: ReferenceMap) -> Tuple[bool, List[str]]:
    """
    Validate reference map quality and completeness
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check calibration quality
    if ref_map.piezo_galvo_r_squared < 0.8:
        issues.append(f"Low calibration R² ({ref_map.piezo_galvo_r_squared:.3f})")
    
    # Check number of calibration points
    if len(ref_map.calibration_points) < 2:
        issues.append("Insufficient calibration points (need at least 2)")
    
    # Check for reasonable calibration parameters
    if abs(ref_map.piezo_galvo_slope) < 1e-6:
        issues.append("Calibration slope too small")
    
    if abs(ref_map.piezo_galvo_slope) > 1000:
        issues.append("Calibration slope too large")
    
    # Check age of calibration
    age_hours = (time.time() - ref_map.last_updated) / 3600
    if age_hours > 24:
        issues.append(f"Calibration is {age_hours:.1f} hours old")
    
    return len(issues) == 0, issues


def print_reference_map_summary(ref_map: ReferenceMap) -> None:
    """Print summary of reference map contents"""
    print(f"Reference Map: {ref_map.name}")
    print(f"Created: {time.ctime(ref_map.created)}")
    print(f"Last Updated: {time.ctime(ref_map.last_updated)}")
    print(f"Valid: {ref_map.valid}")
    print()
    
    print(f"Calibration:")
    print(f"  Points: {len(ref_map.calibration_points)}")
    print(f"  Slope: {ref_map.piezo_galvo_slope:.4f} μm/°")
    print(f"  Offset: {ref_map.piezo_galvo_offset:.2f} μm")
    print(f"  R²: {ref_map.piezo_galvo_r_squared:.4f}")
    print()
    
    print(f"Embryos:")
    print(f"  Stage positions: {len(ref_map.embryo_positions_stage)}")
    print(f"  Light sheet positions: {len(ref_map.embryo_positions_lightsheet)}")
    
    # Validation
    is_valid, issues = validate_reference_map(ref_map)
    if not is_valid:
        print(f"\nIssues:")
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Gently DiSPIM Coordinates")
    print("========================")
    print()
    print("Coordinate transformation and reference mapping utilities")
    print("Device-agnostic transformations for DiSPIM workflows")
    print()
    
    # Test coordinate transformations
    print("Testing piezo-galvo calibration...")
    
    # Simulate calibration points
    cal_points = [
        CalibrationPoint("p1", time.time(), 25.0, -1.5, 0.0, 0.0),
        CalibrationPoint("p2", time.time(), 75.0, 1.5, 0.0, 0.0)
    ]
    
    slope, offset, r_squared = calculate_piezo_galvo_calibration(cal_points)
    print(f"Calibration: slope={slope:.2f} μm/°, offset={offset:.2f} μm, R²={r_squared:.4f}")
    
    # Test coordinate conversion
    test_piezo = 50.0
    test_galvo = piezo_to_galvo(test_piezo, slope, offset)
    back_piezo = galvo_to_piezo(test_galvo, slope, offset)
    print(f"Conversion: {test_piezo:.1f} μm → {test_galvo:.2f}° → {back_piezo:.1f} μm")
    
    # Test reference map
    print("\nTesting reference map...")
    ref_map = create_reference_map("test_map")
    
    for cp in cal_points:
        add_calibration_point(ref_map, cp)
    
    # Add some embryo positions
    embryo_positions = [(100, 200), (-50, 150), (0, 0)]
    for x, y in embryo_positions:
        add_embryo_position_stage(ref_map, x, y)
    
    print_reference_map_summary(ref_map)
    
    # Test nearest neighbor search
    query_pos = (90, 210)
    nearest = find_nearest_embryos(query_pos, ref_map, max_distance=200)
    print(f"\nNearest embryos to {query_pos}:")
    for idx, distance, pos in nearest:
        print(f"  Embryo {idx}: {pos} (distance: {distance:.1f} μm)")
    
    print("\nFunctions available:")
    print("  piezo_to_galvo() / galvo_to_piezo()")
    print("  calculate_piezo_galvo_calibration()")
    print("  create_reference_map() / add_calibration_point()")
    print("  stage_to_lightsheet_coordinates()")
    print("  find_nearest_embryos()")
    print("  save_reference_map() / load_reference_map()")