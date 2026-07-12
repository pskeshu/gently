"""
Analysis Step Implementations

Provides concrete implementations of AnalysisStep for:
- VLM (Claude Vision) analysis
- SAM segmentation
- Classical computer vision operations
"""

import asyncio
import logging
import os
from typing import Any, cast

import numpy as np

from ..settings import settings
from .pipeline import AnalysisResult, AnalysisStep, StepType

logger = logging.getLogger(__name__)


# =============================================================================
# Image Processing Steps
# =============================================================================


class MaxProjectionStep(AnalysisStep):
    """
    Create max intensity projection along an axis

    Reduces 3D volume to 2D image by taking maximum along specified axis.
    """

    def __init__(self, axis: int = 0, name: str = "max_projection"):
        super().__init__(name=name, step_type=StepType.PROJECTION)
        self.axis = axis

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Execute max projection"""
        if not isinstance(input_data, np.ndarray):
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Input must be a numpy array",
            )

        # Handle 2D input (already projected)
        if input_data.ndim == 2:
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                data=input_data,
                success=True,
                metadata={"already_2d": True},
            )

        # Perform projection
        projection = np.max(input_data, axis=self.axis)

        return AnalysisResult(
            step_name=self.name,
            step_type=self.step_type,
            data=projection,
            success=True,
            metadata={
                "input_shape": list(input_data.shape),
                "output_shape": list(projection.shape),
                "axis": self.axis,
            },
        )


class ThresholdStep(AnalysisStep):
    """
    Apply thresholding to create binary mask

    Methods:
    - "otsu": Otsu's automatic threshold
    - "percentile": Use nth percentile as threshold
    - "percentile_bright": Threshold to keep bright regions
    - "manual": Use specified value
    """

    def __init__(
        self,
        method: str = "otsu",
        value: float | None = None,
        name: str = "threshold",
    ):
        super().__init__(name=name, step_type=StepType.THRESHOLD)
        self.method = method
        self.value = value

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Apply threshold"""
        if not isinstance(input_data, np.ndarray):
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Input must be a numpy array",
            )

        image = input_data.astype(np.float64)
        threshold_value = 0.0

        if self.method == "otsu":
            # Otsu's method
            try:
                from skimage.filters import threshold_otsu

                threshold_value = float(threshold_otsu(image))
            except ImportError:
                # Fallback to simple percentile
                threshold_value = float(np.percentile(image, 50))

        elif self.method == "percentile":
            threshold_value = float(np.percentile(image, self.value or 50))

        elif self.method == "percentile_bright":
            # Keep values above percentile (bright regions)
            threshold_value = float(np.percentile(image, self.value or 95))

        elif self.method == "manual":
            if self.value is None:
                return AnalysisResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    error="Manual threshold requires a value",
                )
            threshold_value = self.value

        # Apply threshold
        if self.method == "percentile_bright":
            binary = image >= threshold_value
        else:
            binary = image > threshold_value

        return AnalysisResult(
            step_name=self.name,
            step_type=self.step_type,
            data=binary.astype(np.uint8) * 255,
            success=True,
            metadata={
                "method": self.method,
                "threshold_value": float(threshold_value),
                "pixels_above": int(np.sum(binary)),
            },
        )


class MorphologyStep(AnalysisStep):
    """
    Apply morphological operations

    Operations:
    - "erode": Shrink white regions
    - "dilate": Expand white regions
    - "open": Erode then dilate (removes small bright spots)
    - "close": Dilate then erode (fills small holes)
    """

    def __init__(
        self,
        operation: str = "open",
        kernel_size: int = 3,
        iterations: int = 1,
        name: str = "morphology",
    ):
        super().__init__(name=name, step_type=StepType.MORPHOLOGY)
        self.operation = operation
        self.kernel_size = kernel_size
        self.iterations = iterations

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Apply morphological operation"""
        if not isinstance(input_data, np.ndarray):
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Input must be a numpy array",
            )

        try:
            import cv2

            kernel: np.ndarray = np.ones((self.kernel_size, self.kernel_size), np.uint8)

            if self.operation == "erode":
                result = cv2.erode(input_data, kernel, iterations=self.iterations)
            elif self.operation == "dilate":
                result = cv2.dilate(input_data, kernel, iterations=self.iterations)
            elif self.operation == "open":
                result = cv2.morphologyEx(input_data, cv2.MORPH_OPEN, kernel)
            elif self.operation == "close":
                result = cv2.morphologyEx(input_data, cv2.MORPH_CLOSE, kernel)
            else:
                return AnalysisResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    error=f"Unknown operation: {self.operation}",
                )

        except ImportError:
            # Fallback without OpenCV using scipy
            try:
                from scipy import ndimage

                kernel = np.ones((self.kernel_size, self.kernel_size))

                if self.operation in ("erode", "open"):
                    result = (
                        ndimage.binary_erosion(
                            input_data > 0, structure=kernel, iterations=self.iterations
                        ).astype(np.uint8)
                        * 255
                    )
                    if self.operation == "open":
                        result = (
                            ndimage.binary_dilation(
                                result > 0, structure=kernel, iterations=self.iterations
                            ).astype(np.uint8)
                            * 255
                        )

                elif self.operation in ("dilate", "close"):
                    result = (
                        ndimage.binary_dilation(
                            input_data > 0, structure=kernel, iterations=self.iterations
                        ).astype(np.uint8)
                        * 255
                    )
                    if self.operation == "close":
                        result = (
                            ndimage.binary_erosion(
                                result > 0, structure=kernel, iterations=self.iterations
                            ).astype(np.uint8)
                            * 255
                        )

            except ImportError:
                return AnalysisResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    error="Neither cv2 nor scipy available for morphology",
                )

        return AnalysisResult(
            step_name=self.name,
            step_type=self.step_type,
            data=result,
            success=True,
            metadata={
                "operation": self.operation,
                "kernel_size": self.kernel_size,
                "iterations": self.iterations,
            },
        )


class BlobDetectionStep(AnalysisStep):
    """
    Detect blob-like objects in image

    Uses Laplacian of Gaussian (LoG) blob detection.
    """

    def __init__(
        self,
        min_sigma: float = 10,
        max_sigma: float = 50,
        num_sigma: int = 10,
        threshold: float = 0.1,
        name: str = "blob_detection",
    ):
        super().__init__(name=name, step_type=StepType.DETECTION)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Detect blobs"""
        if not isinstance(input_data, np.ndarray):
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Input must be a numpy array",
            )

        try:
            from skimage.feature import blob_log

            # Normalize image
            image = input_data.astype(np.float64)
            if image.max() > 1:
                image = image / image.max()

            # Detect blobs
            blobs = blob_log(
                image,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                threshold=self.threshold,
            )

            # Convert to list of dicts
            detections = []
            for blob in blobs:
                y, x, sigma = blob
                detections.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "radius": float(sigma * np.sqrt(2)),
                    }
                )

            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                data={
                    "detections": detections,
                    "count": len(detections),
                    "image": input_data,  # Keep original for next step
                },
                success=True,
                metadata={
                    "num_blobs": len(detections),
                    "min_sigma": self.min_sigma,
                    "max_sigma": self.max_sigma,
                },
            )

        except ImportError:
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="scikit-image not available for blob detection",
            )


# =============================================================================
# VLM (Vision Language Model) Step
# =============================================================================


class VLMStep(AnalysisStep):
    """
    Analyze image using Claude Vision API

    Sends image to Claude with a prompt and returns the analysis.
    """

    def __init__(
        self,
        prompt: str,
        model: str = settings.models.perception,
        max_tokens: int = 1024,
        name: str = "vlm_analysis",
        api_key: str | None = None,
    ):
        super().__init__(name=name, step_type=StepType.VLM)
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 JPEG."""
        from gently.core.imaging import image_to_base64, normalize_to_uint8

        img = normalize_to_uint8(image, method="minmax")
        return image_to_base64(img, format="JPEG", quality=85)

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Analyze with Claude Vision"""
        context = context or {}

        # Get image from input (handle dict from previous step)
        if isinstance(input_data, dict):
            image = input_data.get("image", input_data.get("data"))
        else:
            image = input_data

        if not isinstance(image, np.ndarray):
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Input must be a numpy array or dict with 'image' key",
            )

        try:
            import anthropic

            # Get API key
            api_key = self.api_key or context.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return AnalysisResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    error="No Anthropic API key available",
                )

            # Encode image
            b64_image = self._encode_image(image)

            # Build message
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_image,
                    },
                },
                {
                    "type": "text",
                    "text": self.prompt,
                },
            ]

            # Call Claude
            client = anthropic.Anthropic(api_key=api_key)
            response = await asyncio.to_thread(
                lambda: client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": cast(Any, content)}],
                )
            )

            result_text = cast(Any, response.content[0]).text

            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                data={
                    "analysis": result_text,
                    "image": image,  # Pass through for next step
                },
                success=True,
                metadata={
                    "model": self.model,
                    "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error=str(e),
            )


# =============================================================================
# SAM (Segment Anything Model) Step
# =============================================================================


class SAMStep(AnalysisStep):
    """
    Segment image using SAM (Segment Anything Model)

    Can use:
    - Automatic mask generation
    - Point prompts
    - Text prompts (via CLIP if available)
    """

    def __init__(
        self,
        prompt: str | None = None,
        points: list[tuple[int, int]] | None = None,
        model_path: str | None = None,
        min_area: int = 1000,
        name: str = "sam_segmentation",
    ):
        super().__init__(name=name, step_type=StepType.SAM)
        self.prompt = prompt
        self.points = points
        self.model_path = model_path or "sam_vit_b_01ec64.pth"
        self.min_area = min_area
        self._sam = None
        self._predictor: Any = None

    def _load_sam(self):
        """Lazy load SAM model"""
        if self._sam is not None:
            return

        try:
            import torch
            from segment_anything import (
                SamAutomaticMaskGenerator,
                SamPredictor,
                sam_model_registry,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try to load model
            if os.path.exists(self.model_path):
                self._sam = sam_model_registry["vit_b"](checkpoint=self.model_path)
                self._sam.to(device)
                self._predictor = SamPredictor(self._sam)
                self._mask_generator = SamAutomaticMaskGenerator(self._sam)
                logger.info(f"Loaded SAM model on {device}")
            else:
                logger.warning(f"SAM model not found at {self.model_path}")
                self._sam = None

        except ImportError:
            logger.warning("segment_anything not installed")
            self._sam = None

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Run SAM segmentation"""
        # Get image from input
        if isinstance(input_data, dict):
            image = input_data.get("image", input_data.get("data"))
        else:
            image = input_data

        if not isinstance(image, np.ndarray):
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Input must be a numpy array",
            )

        # Load SAM if needed
        self._load_sam()

        if self._sam is None:
            # Fallback to simple thresholding
            logger.warning("SAM not available, using fallback segmentation")
            return await self._fallback_segment(image)

        try:
            # Convert to RGB if grayscale
            if image.ndim == 2:
                image_rgb = np.stack([image, image, image], axis=-1)
            else:
                image_rgb = image

            # Normalize to uint8
            if image_rgb.dtype != np.uint8:
                if image_rgb.max() <= 1.0:
                    image_rgb = (image_rgb * 255).astype(np.uint8)
                else:
                    image_rgb = (
                        (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min()) * 255
                    ).astype(np.uint8)

            # Run in thread (SAM is CPU/GPU intensive)
            masks = await asyncio.to_thread(self._run_sam, image_rgb)

            # Filter by area
            filtered_masks = [m for m in masks if m["area"] >= self.min_area]

            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                data={
                    "masks": filtered_masks,
                    "count": len(filtered_masks),
                    "image": image,
                },
                success=True,
                metadata={
                    "total_masks": len(masks),
                    "filtered_masks": len(filtered_masks),
                    "min_area": self.min_area,
                },
            )

        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error=str(e),
            )

    def _run_sam(self, image_rgb: np.ndarray) -> list[dict]:
        """Run SAM (blocking, called in thread)"""
        if self.points:
            # Point-prompted segmentation
            self._predictor.set_image(image_rgb)
            points = np.array(self.points)
            labels = np.ones(len(points))

            masks, scores, _ = self._predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            return [
                {
                    "segmentation": masks[i],
                    "area": int(np.sum(masks[i])),
                    "score": float(scores[i]),
                }
                for i in range(len(masks))
            ]
        else:
            # Automatic mask generation
            return self._mask_generator.generate(image_rgb)

    async def _fallback_segment(self, image: np.ndarray) -> AnalysisResult:
        """Simple fallback segmentation when SAM is not available"""
        # Simple Otsu threshold
        try:
            from skimage.filters import threshold_otsu
            from skimage.measure import label, regionprops

            thresh = threshold_otsu(image)
            binary = image > thresh

            # Label connected components
            labeled = label(binary)
            regions = regionprops(labeled)

            masks = []
            for region in regions:
                if region.area >= self.min_area:
                    mask = labeled == region.label
                    masks.append(
                        {
                            "segmentation": mask,
                            "area": region.area,
                            "centroid": region.centroid,
                            "bbox": region.bbox,
                        }
                    )

            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                data={
                    "masks": masks,
                    "count": len(masks),
                    "image": image,
                },
                success=True,
                metadata={
                    "method": "fallback_threshold",
                    "num_masks": len(masks),
                },
            )

        except ImportError:
            return AnalysisResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                error="Neither SAM nor scikit-image available",
            )
