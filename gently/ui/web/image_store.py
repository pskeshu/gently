"""
Image Store for the Visualization Server
==========================================

Organized storage for images by type and embryo.
"""

from .models import (
    ANALYSIS_TYPES,
    CALIBRATION_TYPES,
    VOLUME_TYPES,
    EmbryoImageCache,
    ImageData,
    Volume3DData,
)


class ImageStore:
    """Organized storage for images by type and embryo (unlimited)"""

    def __init__(self):
        self._embryo_caches: dict[str, EmbryoImageCache] = {}
        self._global_images: list[ImageData] = []  # Images without embryo_id
        self._calibration_images: list[ImageData] = []  # Global calibration
        self._volume_images: list[ImageData] = []  # Global volumes
        self._volumes_3d: dict[str, Volume3DData] = {}  # 3D volumes by UID

    def _get_embryo_cache(self, embryo_id: str) -> EmbryoImageCache:
        if embryo_id not in self._embryo_caches:
            self._embryo_caches[embryo_id] = EmbryoImageCache(embryo_id=embryo_id)
        return self._embryo_caches[embryo_id]

    def add_image(self, image: ImageData):
        """Add image to appropriate storage based on type and embryo"""
        embryo_id = image.metadata.get("embryo_id")
        data_type = image.data_type

        if data_type in CALIBRATION_TYPES or data_type in ANALYSIS_TYPES:
            # Both calibration and analysis go to calibration tab
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.calibration.append(image)
            else:
                self._calibration_images.append(image)

        elif data_type in VOLUME_TYPES:
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.volumes.append(image)
            else:
                self._volume_images.append(image)
        else:
            # General snapshot/other
            if embryo_id:
                cache = self._get_embryo_cache(embryo_id)
                cache.snapshots.append(image)
            else:
                self._global_images.append(image)

    def get_all_calibration(self, embryo_id: str | None = None) -> list[ImageData]:
        """Get calibration images, optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            return cache.calibration if cache else []
        # Return all calibration images
        all_cal = list(self._calibration_images)
        for cache in self._embryo_caches.values():
            all_cal.extend(cache.calibration)
        return sorted(all_cal, key=lambda x: x.timestamp)

    def get_all_volumes(self, embryo_id: str | None = None) -> list[ImageData]:
        """Get volume images, optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            return cache.volumes if cache else []
        all_vol = list(self._volume_images)
        for cache in self._embryo_caches.values():
            all_vol.extend(cache.volumes)
        return sorted(all_vol, key=lambda x: x.timestamp)

    def get_all_snapshots(self, embryo_id: str | None = None) -> list[ImageData]:
        """Get snapshot images (including volume projections), optionally filtered by embryo"""
        if embryo_id:
            cache = self._embryo_caches.get(embryo_id)
            if not cache:
                return []
            # Include both snapshots and volumes for the embryo
            return sorted(cache.snapshots + cache.volumes, key=lambda x: x.timestamp)
        # Include all snapshots and volumes
        all_snap = list(self._global_images) + list(self._volume_images)
        for cache in self._embryo_caches.values():
            all_snap.extend(cache.snapshots)
            all_snap.extend(cache.volumes)
        return sorted(all_snap, key=lambda x: x.timestamp)

    def get_embryo_ids(self) -> list[str]:
        """Get list of all embryo IDs with images"""
        return list(self._embryo_caches.keys())

    def get_image_by_uid(self, uid: str) -> ImageData | None:
        """Find image by UID across all storage"""
        for img in self._global_images:
            if img.uid == uid:
                return img
        for img in self._calibration_images:
            if img.uid == uid:
                return img
        for img in self._volume_images:
            if img.uid == uid:
                return img
        for cache in self._embryo_caches.values():
            for img in cache.volumes + cache.calibration + cache.snapshots:
                if img.uid == uid:
                    return img
        return None

    def add_volume_3d(self, volume_data: Volume3DData):
        """Add a 3D volume with segmentation"""
        self._volumes_3d[volume_data.uid] = volume_data
        # Keep only last 10 3D volumes to manage memory
        if len(self._volumes_3d) > 10:
            oldest_uid = next(iter(self._volumes_3d))
            del self._volumes_3d[oldest_uid]

    def get_volume_3d(self, uid: str) -> Volume3DData | None:
        """Get a 3D volume by UID"""
        return self._volumes_3d.get(uid)

    def get_all_volumes_3d(self) -> list[dict]:
        """Get info for all 3D volumes (without heavy data)"""
        return [v.to_info_dict() for v in self._volumes_3d.values()]

    def get_sequence(
        self,
        embryo_id: str,
        start: int = 0,
        end: int | None = None,
        data_type: str | None = None,
    ) -> list[ImageData]:
        """Get ordered sequence of images for an embryo within a timepoint range.

        Args:
            embryo_id: The embryo to get images for
            start: Starting timepoint (inclusive)
            end: Ending timepoint (inclusive), None for all
            data_type: Filter by data type (e.g., 'volume_projection')

        Returns:
            List of ImageData sorted by timepoint
        """
        cache = self._embryo_caches.get(embryo_id)
        if not cache:
            return []

        # Get all images for this embryo (volumes + snapshots)
        all_images = list(cache.volumes) + list(cache.snapshots)

        # Filter by data type if specified
        if data_type:
            all_images = [img for img in all_images if img.data_type == data_type]

        # Filter by timepoint range
        def get_timepoint(img: ImageData) -> int | None:
            tp = img.metadata.get("timepoint")
            if tp is not None:
                return int(tp)
            return None

        filtered = []
        for img in all_images:
            tp = get_timepoint(img)
            if tp is None:
                continue
            if tp < start:
                continue
            if end is not None and tp > end:
                continue
            filtered.append(img)

        # Sort by timepoint
        filtered.sort(key=lambda x: get_timepoint(x) or 0)
        return filtered

    def get_stats(self) -> dict:
        """Get storage statistics"""
        total_cal = len(self._calibration_images)
        total_vol = len(self._volume_images)
        total_snap = len(self._global_images)

        for cache in self._embryo_caches.values():
            total_cal += len(cache.calibration)
            total_vol += len(cache.volumes)
            total_snap += len(cache.snapshots)

        return {
            "embryo_count": len(self._embryo_caches),
            "calibration_count": total_cal,
            "volume_count": total_vol,
            "snapshot_count": total_snap,
            "volumes_3d_count": len(self._volumes_3d),
            "embryo_ids": list(self._embryo_caches.keys()),
        }
