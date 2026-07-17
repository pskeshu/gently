"""Volume routes - projections, raw volumes, 3D volume data."""

import base64
import io
import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..volume_helpers import image_to_base64_png, load_volume_from_disk

logger = logging.getLogger(__name__)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/projections/{embryo_id}/{timepoint}")
    async def get_projections(embryo_id: str, timepoint: int, method: str = "all"):
        """
        Generate projections from volume file on disk.

        Args:
            embryo_id: Embryo identifier
            timepoint: Timepoint number (1-indexed)
            method: Projection method - 'all', 'three_view', 'dual_view',
                'depth_colored', 'multi_slice'

        Returns:
            List of projections with method name, description, and base64 PNG data
        """
        from gently.core.imaging import projection_three_view

        # Look up volume path (timelapse tracker + FileStore fallback)
        volume_path = server._resolve_volume_path(embryo_id, timepoint)
        if not volume_path:
            raise HTTPException(
                status_code=404,
                detail=f"No volume for {embryo_id} at timepoint {timepoint}",
            )

        # Load volume from disk
        try:
            vol = load_volume_from_disk(volume_path)

            # Normalize to 0-1 float
            vol = vol.astype(np.float32)
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to load volume: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load volume: {e}") from e

        PROJECTION_METHODS: dict[str, Any] = {
            "three_view": projection_three_view,
        }

        # Try to import additional projection methods from explorer
        try:
            from gently.dataset.explorer_server import (
                projection_depth_colored,
                projection_dual_view,
                projection_multi_slice,
                projection_spin_3d,
            )

            PROJECTION_METHODS.update(
                {
                    "dual_view": projection_dual_view,
                    "depth_colored": projection_depth_colored,
                    "multi_slice": projection_multi_slice,
                    "spin_3d": projection_spin_3d,
                }
            )
        except ImportError:
            pass  # Explorer projections not available

        projections = []

        if method == "all":
            for method_name, method_func in PROJECTION_METHODS.items():
                try:
                    proj_img, desc = method_func(vol)
                    projections.append(
                        {
                            "method": method_name,
                            "description": desc,
                            "data": image_to_base64_png(proj_img),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Projection {method_name} failed: {e}")
        else:
            if method not in PROJECTION_METHODS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown method: {method}. Available: {list(PROJECTION_METHODS.keys())}"
                    ),
                )
            proj_img, desc = PROJECTION_METHODS[method](vol)
            projections.append(
                {
                    "method": method,
                    "description": desc,
                    "data": image_to_base64_png(proj_img),
                }
            )

        return {
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "volume_shape": list(vol.shape),
            "projections": projections,
        }

    @router.get("/api/volume-raw/{embryo_id}/{timepoint}")
    async def get_volume_raw(embryo_id: str, timepoint: int):
        """
        Get raw volume data for 3D viewer.

        Returns the volume as base64-encoded uint8 bytes with shape info.
        """
        from scipy import ndimage

        # Look up volume path (timelapse tracker + FileStore fallback)
        volume_path = server._resolve_volume_path(embryo_id, timepoint)
        if not volume_path:
            raise HTTPException(
                status_code=404,
                detail=f"No volume for {embryo_id} at timepoint {timepoint}",
            )

        try:
            vol = load_volume_from_disk(volume_path)

            # Normalize using percentile-based contrast stretching (like projection_explorer)
            vol = vol.astype(np.float32)
            p1, p99 = np.percentile(vol, [1, 99])
            vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

            # Apply Gaussian blur along Z axis to reduce banding at side views
            vol = ndimage.gaussian_filter1d(vol, sigma=1.0, axis=0)

            vol_uint8 = (vol * 255).astype(np.uint8)

            # Encode as base64
            vol_bytes = vol_uint8.tobytes()
            vol_b64 = base64.b64encode(vol_bytes).decode("utf-8")

            # Physical voxel size for isometric 3D rendering.
            # Matches the default in gently.core.imaging.projection_three_view:
            # (dz, dy, dx) in microns. 1.0 um Z step, 0.1625 um XY (6.5 um
            # camera pixel / 40x SPIM objective). If per-volume metadata
            # becomes available later, prefer that over this default.
            voxel_size_um = [1.0, 0.1625, 0.1625]

            return {
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "shape": list(vol_uint8.shape),
                "voxel_size_um": voxel_size_um,
                "data": vol_b64,
            }

        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to load volume: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load volume: {e}") from e

    @router.get("/api/volumes3d")
    async def list_volumes_3d():
        """Get list of 3D volumes (without heavy data)"""
        return {
            "volumes_3d": server.store.get_all_volumes_3d(),
            "count": len(server.store._volumes_3d),
        }

    @router.get("/api/volumes3d/{uid}")
    async def get_volume_3d_info(uid: str):
        """Get 3D volume info by UID"""
        vol = server.store.get_volume_3d(uid)
        if vol:
            return vol.to_info_dict()
        raise HTTPException(status_code=404, detail=f"3D Volume {uid} not found")

    @router.get("/api/volumes3d/{uid}/slice/{z}")
    async def get_volume_3d_slice(uid: str, z: int):
        """Get a specific Z-slice as PNG with segmentation overlay"""
        vol = server.store.get_volume_3d(uid)
        if not vol:
            raise HTTPException(status_code=404, detail=f"3D Volume {uid} not found")

        rgb = vol.get_slice_overlay(z)

        if PIL_AVAILABLE:
            img = Image.fromarray(rgb)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return Response(content=buffer.getvalue(), media_type="image/png")

        raise HTTPException(status_code=500, detail="PIL not available")

    @router.get("/api/volume-data/{uid}")
    async def get_volume_data_for_3d_viewer(uid: str):
        """Get raw volume data as base64 for 3D viewer (projection viewer)"""
        # First check if it's a 3D segmented volume
        vol = server.store.get_volume_3d(uid)
        if vol:
            # Return the raw volume data (normalized to uint8)
            volume = vol.volume
            if volume.dtype != np.uint8:
                vmin, vmax = volume.min(), volume.max()
                if vmax > vmin:
                    volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    volume = np.zeros(volume.shape, dtype=np.uint8)
            return {
                "shape": list(volume.shape),
                "data": base64.b64encode(volume.tobytes()).decode("utf-8"),
                "uid": uid,
            }

        # Check if it's a regular image with stored volume data
        image = server.store.get_image_by_uid(uid)
        if image and image.shape and len(image.shape) == 3:
            raise HTTPException(
                status_code=404,
                detail=f"Volume data for {uid} not available - only segmented volumes supported",
            )

        raise HTTPException(status_code=404, detail=f"Volume {uid} not found")

    @router.post("/api/volumes3d")
    async def push_volume_3d_http(request: Request):
        """Push a 3D volume with segmentation via HTTP (for CV subagent)"""
        try:
            data = await request.json()

            # Decode the volume and masks from base64
            volume_b64 = data.get("volume_b64")
            masks_b64 = data.get("masks_b64")
            uid = data.get("uid")
            shape = data.get("shape")
            dtype_vol = data.get("dtype_vol", "uint16")
            dtype_mask = data.get("dtype_mask", "uint16")
            metadata = data.get("metadata", {})

            if not all([volume_b64, masks_b64, uid, shape]):
                raise HTTPException(status_code=400, detail="Missing required fields")

            # Decode arrays
            volume = np.frombuffer(base64.b64decode(volume_b64), dtype=np.dtype(dtype_vol)).reshape(
                shape
            )

            masks = np.frombuffer(base64.b64decode(masks_b64), dtype=np.dtype(dtype_mask)).reshape(
                shape
            )

            # Push using the existing method
            await server.push_volume_3d(volume, masks, uid, metadata)

            return {"status": "ok", "uid": uid, "shape": shape}

        except Exception as e:
            logger.error(f"Failed to push 3D volume via HTTP: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
