"""Image routes - retrieve and push images."""

import base64
import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

from gently.ui.web.auth import require_control

from ..upload_validation import decode_array_payload
from ..volume_helpers import parse_volume_uid

logger = logging.getLogger(__name__)

MAX_IMAGE_UPLOAD_BYTES = 64 * 1024 * 1024


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/images/{uid}")
    async def get_image(uid: str):
        """Get image by UID"""
        image = server.store.get_image_by_uid(uid)
        if image:
            return image.to_dict()
        # Fallback to persistent DataStore
        if server.data_store:
            try:
                data = server.data_store.retrieve(uid)
                if data is not None:
                    return {"uid": uid, "data": "loaded_from_store"}
            except Exception:
                pass
        raise HTTPException(status_code=404, detail=f"Image {uid} not found")

    @router.get("/api/images/{uid}/png")
    async def get_image_png(uid: str):
        """Get image as PNG (cached - images are immutable)"""
        # Cache headers - images are immutable so cache aggressively
        cache_headers = {
            "Cache-Control": "public, max-age=86400, immutable",  # 24 hours
            "ETag": f'"{uid}"',
        }

        image = server.store.get_image_by_uid(uid)

        # Fallback: If UID follows volume_EMBRYOID_tNNNN pattern, try looking up real UID
        parsed = parse_volume_uid(uid)
        if not image and parsed:
            embryo_id, timepoint = parsed
            # Look up real projection UID from timelapse tracker
            if embryo_id in server.timelapse_tracker.projection_uids:
                real_uid = server.timelapse_tracker.projection_uids[embryo_id].get(timepoint)
                if real_uid:
                    image = server.store.get_image_by_uid(real_uid)

        if image and image.base64_png:
            png_bytes = base64.b64decode(image.base64_png)
            return Response(content=png_bytes, media_type="image/png", headers=cache_headers)

        # Fallback to persistent DataStore
        if server.data_store:
            try:
                # Try with original UID first
                data = server.data_store.retrieve(uid)
                # If not found and this is a fallback pattern, try with real UID
                if data is None and parsed:
                    embryo_id, timepoint = parsed
                    if embryo_id in server.timelapse_tracker.projection_uids:
                        real_uid = server.timelapse_tracker.projection_uids[embryo_id].get(
                            timepoint
                        )
                        if real_uid:
                            data = server.data_store.retrieve(real_uid)
                if data is not None:
                    from io import BytesIO

                    from PIL import Image

                    from gently.core.imaging import (
                        apply_crop_bounds,
                        compute_crop_bounds,
                        projection_three_view,
                    )

                    # Handle numpy array
                    if isinstance(data, np.ndarray):
                        # Handle 4D volumes (Views, Z, Y, X) - take View A
                        if data.ndim == 4:
                            data = data[0]
                        # Handle 3D volumes - generate three-view projection
                        if data.ndim == 3:
                            # View A already selected by the 4D branch above;
                            # never split a 3D volume by aspect ratio.
                            # Auto-crop and project
                            bounds = compute_crop_bounds(data)
                            data = apply_crop_bounds(data, bounds)
                            data, _ = projection_three_view(data)
                        # Normalize to uint8 if needed
                        if data.dtype != np.uint8:
                            data = (
                                (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
                            ).astype(np.uint8)
                        img = Image.fromarray(data)
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        return Response(
                            content=buf.getvalue(),
                            media_type="image/png",
                            headers=cache_headers,
                        )
            except Exception as e:
                logger.warning(f"Failed to load image {uid} from DataStore: {e}")

        # Fallback to FileStore JPEG projections (persistent on-disk).
        # Unlike the in-memory base64 images, an on-disk projection CAN change
        # (e.g. regenerated after a projection-format fix), so we must NOT mark
        # it immutable with a content-independent (uid) ETag — that pins the
        # browser to the stale image. Use a content-aware ETag (mtime+size)
        # and a short max-age so a regeneration is picked up.
        if server.gently_store and parsed:
            embryo_id, timepoint = parsed
            proj_path = server._resolve_projection_path(embryo_id, timepoint)
            if proj_path:
                st = proj_path.stat()
                return FileResponse(
                    str(proj_path),
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "ETag": f'"{uid}-{int(st.st_mtime)}-{st.st_size}"',
                    },
                )

        raise HTTPException(status_code=404, detail=f"Image {uid} not found")

    @router.post("/api/images", dependencies=[Depends(require_control)])
    async def push_image_http(request: Request):
        """Push a 2D image via HTTP (for CV subagent visualizations)"""
        try:
            data = await request.json()

            # Decode the image from base64
            image_b64 = data.get("image_b64")
            uid = data.get("uid")
            shape = data.get("shape")
            dtype = data.get("dtype", "uint8")
            data_type = data.get("data_type", "cv_visualization")
            metadata = data.get("metadata", {})

            if not all([image_b64, uid, shape]):
                raise HTTPException(status_code=400, detail="Missing required fields")

            # Decode array (validates shape/dtype and caps size before allocating)
            array = decode_array_payload(
                image_b64,
                shape,
                dtype,
                max_nbytes=MAX_IMAGE_UPLOAD_BYTES,
                label="image",
            )

            # Push using the existing method
            await server.push_image(array, uid, data_type, metadata)

            return {"status": "ok", "uid": uid, "data_type": data_type}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to push image via HTTP: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
