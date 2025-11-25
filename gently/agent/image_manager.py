"""
Image management for copilot analysis
Handles storage, retrieval, and compression for Claude Vision API

Now integrated with UID-based DataStore for:
- Provenance tracking (parent-child lineage)
- Cross-service data sharing via UIDs
- Unified storage through Databroker
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, TYPE_CHECKING
import tifffile
import base64
import io
import logging
from PIL import Image

from .state import ImageRecord, EmbryoState

if TYPE_CHECKING:
    from ..core.data_store import DataStore, DataReference

logger = logging.getLogger(__name__)


def extract_view_a_and_max_project(volume: np.ndarray) -> np.ndarray:
    """
    Extract View A and create max projection

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)

    Returns
    -------
    np.ndarray
        2D max projection (Y, X)
    """
    # For diSPIM, View A is typically the full volume
    # Create max projection along Z axis
    max_proj = np.max(volume, axis=0)
    return max_proj


def compress_image_for_api(image: np.ndarray, max_dimension: int = 800,
                           quality: int = 85) -> tuple[str, float]:
    """
    Compress image for Claude Vision API

    Parameters
    ----------
    image : np.ndarray
        2D image (Y, X) or (Y, X, C)
    max_dimension : int
        Maximum width or height in pixels
    quality : int
        JPEG quality (1-100)

    Returns
    -------
    b64_string : str
        Base64-encoded JPEG
    size_kb : float
        Size in kilobytes
    """
    # Normalize to 0-255 uint8
    if image.dtype != np.uint8:
        # Handle different bit depths
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            # Normalize to range
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Resize if too large
    width, height = pil_image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    # Compress to JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
    jpeg_bytes = buffer.getvalue()

    # Base64 encode
    b64_string = base64.b64encode(jpeg_bytes).decode('utf-8')
    size_kb = len(jpeg_bytes) / 1024

    return b64_string, size_kb


class ImageManager:
    """
    Manages image storage and retrieval for copilot analysis

    Now integrates with UID-based DataStore for:
    - All volumes/images stored with UIDs
    - Lineage tracking (volume -> projection)
    - Cross-service data sharing
    """

    def __init__(
        self,
        storage_path: Path,
        history_length: int = 10,
        data_store: Optional['DataStore'] = None,
    ):
        """
        Parameters
        ----------
        storage_path : Path
            Directory to store volume TIFFs
        history_length : int
            Number of recent images to keep per embryo for temporal context
        data_store : DataStore, optional
            UID-based data store for unified storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_length = history_length
        self._data_store = data_store

    @property
    def data_store(self) -> Optional['DataStore']:
        """Get data store, lazily initializing if needed"""
        if self._data_store is None:
            try:
                from ..core.data_store import get_data_store
                self._data_store = get_data_store()
            except ImportError:
                logger.debug("DataStore not available, using file-only storage")
        return self._data_store

    def set_data_store(self, data_store: 'DataStore'):
        """Set the data store instance"""
        self._data_store = data_store

    def store_volume(self, embryo_state: EmbryoState, timepoint: int,
                     volume: np.ndarray) -> ImageRecord:
        """
        Store volume and generate max projection for analysis

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo being imaged
        timepoint : int
            Timepoint number
        volume : np.ndarray
            3D volume (Z, Y, X)

        Returns
        -------
        ImageRecord
            Record of stored image (includes UIDs if DataStore available)
        """
        embryo_id = embryo_state.id

        # Save full volume to disk
        volume_filename = f"{embryo_id}_t{timepoint:04d}.tif"
        volume_path = self.storage_path / volume_filename
        tifffile.imwrite(str(volume_path), volume, compression='zlib')

        # Generate max projection for Claude Vision
        max_proj = extract_view_a_and_max_project(volume)
        b64_image, size_kb = compress_image_for_api(max_proj)

        # Store in DataStore with lineage tracking
        volume_uid = None
        projection_uid = None

        if self.data_store is not None:
            try:
                # Store volume with metadata
                volume_ref = self.data_store.store(
                    data=volume,
                    data_type="volume",
                    metadata={
                        'embryo_id': embryo_id,
                        'timepoint': timepoint,
                        'shape': list(volume.shape),
                        'dtype': str(volume.dtype),
                        'volume_path': str(volume_path),
                    }
                )
                volume_uid = volume_ref.uid
                logger.debug(f"Stored volume with UID: {volume_uid[:8]}")

                # Store max projection with parent lineage
                projection_ref = self.data_store.store(
                    data=max_proj,
                    data_type="image",
                    metadata={
                        'embryo_id': embryo_id,
                        'timepoint': timepoint,
                        'projection_type': 'max_z',
                        'shape': list(max_proj.shape),
                        'b64_size_kb': size_kb,
                    },
                    parent_uid=volume_uid,  # Lineage: projection derives from volume
                )
                projection_uid = projection_ref.uid
                logger.debug(f"Stored projection with UID: {projection_uid[:8]} (parent: {volume_uid[:8]})")

            except Exception as e:
                logger.warning(f"DataStore storage failed, continuing with file-only: {e}")

        # Create record with UIDs
        record = ImageRecord(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            volume_path=str(volume_path),
            max_projection_b64=b64_image,
            size_kb=size_kb,
            volume_uid=volume_uid,
            projection_uid=projection_uid,
        )

        # Add to embryo's recent images (sliding window)
        embryo_state.recent_images.append(record)

        # Keep only last N
        if len(embryo_state.recent_images) > self.history_length:
            embryo_state.recent_images.pop(0)

        # Update embryo state
        embryo_state.last_imaged = datetime.now()
        embryo_state.timepoints_acquired = timepoint + 1

        return record

    def get_recent_context(self, embryo_state: EmbryoState,
                           num_images: int = 5) -> List[Dict]:
        """
        Get recent images for Claude Vision temporal context

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo to get images for
        num_images : int
            Number of recent images to include

        Returns
        -------
        List[Dict]
            List of image content blocks for Claude API
        """
        recent = embryo_state.recent_images[-num_images:]

        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img.max_projection_b64
                }
            }
            for img in recent
        ]

    def get_latest_image(self, embryo_state: EmbryoState) -> Optional[ImageRecord]:
        """Get most recent image for embryo"""
        if embryo_state.recent_images:
            return embryo_state.recent_images[-1]
        return None

    def load_volume(self, image_record: ImageRecord) -> np.ndarray:
        """Load full volume from disk"""
        return tifffile.imread(image_record.volume_path)

    # ===== UID-based retrieval methods =====

    def get_volume_by_uid(self, uid: str) -> Optional[np.ndarray]:
        """
        Retrieve volume by its UID from DataStore

        Parameters
        ----------
        uid : str
            Volume UID

        Returns
        -------
        np.ndarray or None
            The volume data, or None if not found
        """
        if self.data_store is None:
            logger.warning("DataStore not available for UID lookup")
            return None

        try:
            return self.data_store.retrieve(uid)
        except KeyError:
            logger.warning(f"Volume not found for UID: {uid}")
            return None

    def get_image_by_uid(self, uid: str) -> Optional[np.ndarray]:
        """
        Retrieve image (e.g., projection) by its UID from DataStore

        Parameters
        ----------
        uid : str
            Image UID

        Returns
        -------
        np.ndarray or None
            The image data, or None if not found
        """
        if self.data_store is None:
            logger.warning("DataStore not available for UID lookup")
            return None

        try:
            return self.data_store.retrieve(uid)
        except KeyError:
            logger.warning(f"Image not found for UID: {uid}")
            return None

    def get_lineage(self, uid: str) -> List['DataReference']:
        """
        Get the lineage (parent chain) for a data item

        Parameters
        ----------
        uid : str
            UID of data item

        Returns
        -------
        list of DataReference
            Parent chain from oldest to newest
        """
        if self.data_store is None:
            return []

        try:
            return self.data_store.get_lineage(uid)
        except Exception as e:
            logger.warning(f"Failed to get lineage for {uid}: {e}")
            return []

    def get_children(self, uid: str) -> List['DataReference']:
        """
        Get all data derived from a given UID

        Parameters
        ----------
        uid : str
            Parent UID

        Returns
        -------
        list of DataReference
            Child references
        """
        if self.data_store is None:
            return []

        try:
            return self.data_store.get_children(uid)
        except Exception as e:
            logger.warning(f"Failed to get children for {uid}: {e}")
            return []

    def query_by_embryo(self, embryo_id: str, data_type: Optional[str] = None) -> List['DataReference']:
        """
        Query all data for a specific embryo

        Parameters
        ----------
        embryo_id : str
            Embryo ID to query
        data_type : str, optional
            Filter by data type (volume, image, analysis)

        Returns
        -------
        list of DataReference
            Matching references
        """
        if self.data_store is None:
            return []

        try:
            return self.data_store.query(data_type=data_type, embryo_id=embryo_id)
        except Exception as e:
            logger.warning(f"Failed to query data for {embryo_id}: {e}")
            return []

    def store_analysis_result(
        self,
        result: Dict,
        analysis_type: str,
        embryo_id: str,
        parent_uid: Optional[str] = None,
        timepoint: Optional[int] = None,
    ) -> Optional[str]:
        """
        Store an analysis result with lineage tracking

        Parameters
        ----------
        result : dict
            Analysis result to store
        analysis_type : str
            Type of analysis (e.g., 'hatching_detection', 'morphology')
        embryo_id : str
            Embryo this analysis is for
        parent_uid : str, optional
            UID of the source data (e.g., the image analyzed)
        timepoint : int, optional
            Timepoint of the analysis

        Returns
        -------
        str or None
            UID of stored analysis, or None if storage failed
        """
        if self.data_store is None:
            logger.debug("DataStore not available, analysis result not stored with UID")
            return None

        try:
            ref = self.data_store.store(
                data=result,
                data_type="analysis",
                metadata={
                    'embryo_id': embryo_id,
                    'analysis_type': analysis_type,
                    'timepoint': timepoint,
                },
                parent_uid=parent_uid,
            )
            logger.debug(f"Stored {analysis_type} analysis with UID: {ref.uid[:8]}")
            return ref.uid
        except Exception as e:
            logger.warning(f"Failed to store analysis result: {e}")
            return None
