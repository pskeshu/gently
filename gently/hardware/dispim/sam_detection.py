"""
SAM + Claude Vision Embryo Detection Module

Extracts detection logic from test_sam_claude_hybrid_detection.py into reusable module.
Returns embryo positions (pixel + stage coordinates) for calibration workflow.
"""

import base64
import json
import logging
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import anthropic
import cv2
import numpy as np
from PIL import Image

from gently.core.coordinates import (
    DEFAULT_OBJECTIVE_MAG,
    DEFAULT_PIXEL_SIZE_UM,
    get_um_per_pixel,
    pixel_to_stage_position,
)
from gently.settings import settings

logger = logging.getLogger(__name__)


class SAMEmbryoDetector:
    """
    Embryo detector using SAM + Claude Vision hybrid approach

    Features:
    - Initial segmentation with SAM
    - 2-round Claude Vision review for false positives/negatives
    - Napari visualization (optional)
    - Returns embryo positions as simple list of coordinates
    """

    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_b_01ec64.pth",
        sam_model_type: str = "vit_b",
        device: str = "cpu",
        anthropic_api_key: str | None = None,
    ):
        """
        Initialize SAM detector

        Parameters
        ----------
        sam_checkpoint : str
            Path to SAM model checkpoint
        sam_model_type : str
            SAM model type (vit_b, vit_l, vit_h)
        device : str
            Device for SAM (cpu or cuda)
        anthropic_api_key : str, optional
            Anthropic API key for Claude Vision. If None, uses env var.
        """
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.device = device

        # Claude API
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.claude_client = anthropic.Anthropic(api_key=api_key) if api_key else None

        # Detection parameters
        self.max_embryos = 20
        self.min_area = 2000
        self.max_area = 15000
        self.min_circularity = 0.4
        self.min_separation_pixels = 100

        # SAM models (lazy loaded)
        self._mask_generator: Any = None
        self._predictor: Any = None

    def _load_sam(self):
        """Lazy load SAM models"""
        if self._mask_generator is not None:
            return

        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

        if not Path(self.sam_checkpoint).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")

        logger.info("Loading SAM model: %s on %s", self.sam_model_type, self.device)
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.80,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            box_nms_thresh=0.7,
        )

        self._predictor = SamPredictor(sam)
        logger.info("SAM model loaded")

    def preprocess_image(
        self,
        image: np.ndarray,
        bg_kernel_size: int = 150,
        use_clahe: bool = True,
        clahe_clip_limit: float = 3.0,
        clahe_tile_size: int = 16,
        gaussian_sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Preprocess image for better SAM detection.

        Key insight: Embryos appear as BRIGHT objects against darker background.
        This preprocessing enhances contrast and removes background variations
        to make embryo boundaries clearly visible.

        Parameters
        ----------
        image : np.ndarray
            Input image (16-bit or 8-bit grayscale)
        bg_kernel_size : int
            Kernel size for background subtraction via morphological opening.
            Should be larger than largest embryo. Default: 150
        use_clahe : bool
            Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
            Default: True
        clahe_clip_limit : float
            CLAHE clip limit. Higher = more contrast. Default: 3.0
        clahe_tile_size : int
            CLAHE tile grid size. Smaller = more local enhancement. Default: 16
        gaussian_sigma : float
            Gaussian blur sigma for noise reduction. Default: 2.0

        Returns
        -------
        np.ndarray
            Preprocessed 8-bit image with enhanced contrast
        """
        logger.debug("Preprocessing image (shape: %s, dtype: %s)...", image.shape, image.dtype)
        logger.debug("Input range: %s - %s", image.min(), image.max())

        # Step 1: Percentile normalization (handles low dynamic range)
        # This stretches the narrow range (e.g., 84-354) to full 0-255
        logger.debug("Percentile normalization (2-98%%)...")
        p2, p98 = np.percentile(image, (2, 98))
        img_norm = np.clip((image.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255).astype(
            np.uint8
        )
        logger.debug("Normalized to 0-255")

        # Step 2: Background subtraction with large morphological opening
        # Removes large-scale illumination variations
        if bg_kernel_size > 0:
            logger.debug("Background subtraction (kernel=%d)...", bg_kernel_size)
            kernel_bg = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size)
            )
            background = cv2.morphologyEx(img_norm, cv2.MORPH_OPEN, kernel_bg)
            img_no_bg = cv2.subtract(img_norm, background)
            img_no_bg = cv2.normalize(img_no_bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            logger.debug("Background subtracted")
        else:
            img_no_bg = img_norm

        # Step 3: CLAHE for local contrast enhancement
        # Makes embryo boundaries much more visible
        if use_clahe:
            logger.debug("CLAHE (clip=%.1f, tile=%d)...", clahe_clip_limit, clahe_tile_size)
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size)
            )
            img_enhanced = clahe.apply(img_no_bg)
            logger.debug("CLAHE applied")
        else:
            img_enhanced = img_no_bg

        # Step 4: Light Gaussian smoothing to reduce noise
        if gaussian_sigma > 0:
            logger.debug("Gaussian blur (sigma=%.1f)...", gaussian_sigma)
            img_smooth = cv2.GaussianBlur(img_enhanced, (5, 5), gaussian_sigma)
            logger.debug("Smoothing applied")
        else:
            img_smooth = img_enhanced

        logger.debug(
            "Preprocessing complete (output range: %s - %s)", img_smooth.min(), img_smooth.max()
        )
        return img_smooth

    def find_embryo_candidates(
        self,
        image: np.ndarray,
        brightness_percentile: float = 99.0,
        min_area: int = 5000,
        max_area: int = 150000,
        clahe_clip: float = 3.0,
        clahe_tile: int = 16,
    ) -> tuple[list[dict], np.ndarray]:
        """
        Find embryo candidates using brightness-based detection.

        Embryos appear as BRIGHT objects against darker background.
        This method finds them by thresholding the brightest pixels.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale image (16-bit or 8-bit)
        brightness_percentile : float
            Percentile threshold for detecting bright embryos.
            99.0 = fewer, confident detections. 98.0 = more detections.
        min_area : int
            Minimum embryo area in pixels (filters small noise)
        max_area : int
            Maximum embryo area in pixels (filters large artifacts)
        clahe_clip : float
            CLAHE clip limit for contrast enhancement
        clahe_tile : int
            CLAHE tile grid size

        Returns
        -------
        candidates : List[Dict]
            List of candidate embryos with keys:
            - bbox: (x, y, w, h) bounding box
            - centroid: (cx, cy) center point
            - area: area in pixels
        enhanced_image : np.ndarray
            Contrast-enhanced 8-bit image for SAM
        """
        logger.info(
            "Finding embryo candidates (brightness percentile=%.1f)...", brightness_percentile
        )
        logger.debug("Input range: %s - %s", image.min(), image.max())

        # Step 1: Percentile normalization (handles low dynamic range)
        p2, p98 = np.percentile(image, (2, 98))
        img_norm = np.clip((image.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255).astype(
            np.uint8
        )
        logger.debug("Normalized to 0-255")

        # Step 2: CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        img_enhanced = clahe.apply(img_norm)
        logger.debug("CLAHE applied (clip=%.1f, tile=%d)", clahe_clip, clahe_tile)

        # Step 3: Light smoothing
        img_smooth = cv2.GaussianBlur(img_enhanced, (5, 5), 2)

        # Step 4: Threshold brightest pixels (embryos are BRIGHT)
        threshold_value = np.percentile(img_smooth, brightness_percentile)
        _, mask = cv2.threshold(img_smooth, threshold_value, 255, cv2.THRESH_BINARY)
        logger.debug("Threshold at %.1f (percentile %.1f)", threshold_value, brightness_percentile)

        # Step 5: Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Step 6: Dilate to capture full embryo extent
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel_dilate, iterations=2)
        logger.debug("Morphological cleanup complete")

        # Step 7: Find connected components and filter by area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        candidates = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                candidates.append({"bbox": (x, y, w, h), "centroid": (cx, cy), "area": area})

        logger.info("Found %d embryo candidates", len(candidates))
        return candidates, img_smooth

    def refine_with_sam(
        self, image: np.ndarray, candidates: list[dict], padding: int = 20
    ) -> list[dict]:
        """
        Refine embryo candidates using SAM with bounding box prompts.

        Parameters
        ----------
        image : np.ndarray
            Enhanced 8-bit image
        candidates : List[Dict]
            Candidate embryos from find_embryo_candidates
        padding : int
            Padding to add around bounding boxes

        Returns
        -------
        embryos : List[Dict]
            Refined embryos with SAM masks and updated properties
        """
        if not candidates:
            return []

        # Load SAM if needed
        self._load_sam()

        # Convert to RGB for SAM (it expects 3-channel)
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Set image for SAM predictor
        self._predictor.set_image(image_rgb)

        embryos = []
        h, w = image.shape[:2]

        for i, candidate in enumerate(candidates):
            x, y, bw, bh = candidate["bbox"]

            # Add padding and clip to image bounds
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)

            # SAM box format: [x1, y1, x2, y2]
            input_box = np.array([x1, y1, x2, y2])

            # Get SAM prediction with box prompt
            masks, scores, _ = self._predictor.predict(
                point_coords=None, point_labels=None, box=input_box, multimask_output=True
            )

            # Take best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]

            # Calculate properties from SAM mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Get largest contour
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)

                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = candidate["centroid"]

                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

                # Get bounding box from contour
                bx, by, bw, bh = cv2.boundingRect(contour)

                embryos.append(
                    {
                        "embryo_id": f"embryo_{i + 1}",
                        "uid": str(
                            uuid.uuid4()
                        ),  # Global unique identifier for cross-session tracking
                        "pixel_x": float(cx),
                        "pixel_y": float(cy),
                        "bbox": (bx, by, bw, bh),  # Used by visualization functions
                        "area_pixels": int(area),
                        "circularity": float(circularity),
                        "confidence": float(score),
                        "mask": mask,
                    }
                )

        logger.info("SAM refined %d embryos", len(embryos))
        return embryos

    async def detect_embryos(
        self,
        image: np.ndarray,
        stage_position: tuple[float, float],
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        use_claude_review: bool = True,
        save_visualizations: bool = True,
        output_dir: Path | None = None,
        brightness_percentile: float = 99.0,
        min_area: int = 5000,
        max_area: int = 150000,
    ) -> dict:
        """
        Detect embryos using brightness-based detection + SAM refinement.

        This hybrid approach:
        1. Uses brightness thresholding to find candidate embryo regions
        2. Uses SAM with bounding box prompts to get precise segmentation
        3. Optionally uses Claude Vision for verification

        Parameters
        ----------
        image : np.ndarray
            Bottom camera image (grayscale or RGB)
        stage_position : tuple
            Current XY stage position (x, y) in micrometers
        pixel_size_um : float
            Camera pixel size in micrometers (default: 6.5 for PCO)
        objective_mag : float
            Objective magnification (default: 10x for bottom camera)
        use_claude_review : bool
            Whether to use Claude Vision for review (default: True)
        save_visualizations : bool
            Whether to save annotated images (default: True)
        output_dir : Path, optional
            Where to save visualizations. If None, uses './detection_results'
        brightness_percentile : float
            Percentile threshold for brightness detection.
            99.0 = fewer, confident detections. 98.0 = more detections.
        min_area : int
            Minimum embryo area in pixels. Default: 5000
        max_area : int
            Maximum embryo area in pixels. Default: 150000

        Returns
        -------
        dict
            Detection results with keys:
            - embryos: List[Dict] - Embryo positions and metadata
            - initial_detections: int
            - final_detections: int
            - verification: Dict - Claude's verification results
            - images: Dict - Paths to saved images

        Each embryo dict contains:
            - embryo_id: int
            - pixel_x, pixel_y: float - Center in pixels
            - stage_x_um, stage_y_um: float - Stage coordinates
            - bbox_pixel: (x, y, w, h)
            - area_pixels: int
            - circularity: float
            - confidence: float
        """
        # Setup output directory
        if output_dir is None:
            output_dir = Path("./detection_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("BRIGHTNESS + SAM EMBRYO DETECTION")
        logger.info("=" * 70)

        # Step 1: Find candidates using brightness detection
        logger.info("[1/4] Finding embryo candidates (brightness-based)...")
        candidates, image_enhanced = self.find_embryo_candidates(
            image, brightness_percentile=brightness_percentile, min_area=min_area, max_area=max_area
        )

        if len(candidates) == 0:
            logger.warning("No embryo candidates found!")
            return {
                "embryos": [],
                "initial_detections": 0,
                "final_detections": 0,
                "verification": {"verified": False},
                "images": {},
            }

        # Step 2: Refine with SAM
        logger.info("[2/4] Refining with SAM...")
        embryos_sam = self.refine_with_sam(image_enhanced, candidates)
        logger.info("SAM refined %d embryos", len(embryos_sam))

        # Use enhanced image for visualization
        image_8bit = image_enhanced

        if len(embryos_sam) == 0:
            logger.warning("No embryos detected by SAM!")
            return {
                "embryos": [],
                "initial_detections": 0,
                "final_detections": 0,
                "verification": {"verified": False},
                "images": {},
            }

        # Save initial detection
        if save_visualizations:
            initial_viz = self._create_annotated_image(image_8bit, embryos_sam)
            cv2.imwrite(str(output_dir / "detection_initial.png"), initial_viz)

        # Claude review (if enabled)
        embryos_final = embryos_sam
        verification: dict[str, Any] = {"verified": True, "skipped": not use_claude_review}
        changes: dict[str, Any] = {"round1": {"removed": [], "added": []}}

        if use_claude_review and self.claude_client:
            logger.info("[2/4] Claude Vision review (Round 1)...")
            annotated = self._create_annotated_image(image_8bit, embryos_sam)
            review_r1 = await self._review_with_claude(image_8bit, annotated, embryos_sam)

            logger.info("[3/4] Applying corrections...")
            embryos_r1, changes["round1"] = self._apply_corrections(
                embryos_sam, review_r1, image, self._predictor
            )

            if save_visualizations:
                r1_viz = self._create_annotated_image(image_8bit, embryos_r1)
                cv2.imwrite(str(output_dir / "detection_round1.png"), r1_viz)

            # Round 2: Verification
            logger.info("[4/4] Claude verification (Round 2)...")
            r1_viz = self._create_annotated_image(image_8bit, embryos_r1)
            verification = await self._verify_with_claude(
                image_8bit, r1_viz, embryos_r1, changes["round1"]
            )

            # Apply round 2 corrections if needed
            has_r2_changes = (
                len(verification.get("additional_false_positives", [])) > 0
                or len(verification.get("additional_false_negatives", [])) > 0
            )

            if has_r2_changes:
                logger.info("Applying Round 2 corrections...")
                review_r2: dict[str, Any] = {
                    "false_positives": verification.get("additional_false_positives", []),
                    "false_negatives": verification.get("additional_false_negatives", []),
                }
                embryos_final, changes["round2"] = self._apply_corrections(
                    embryos_r1, review_r2, image, self._predictor
                )
            else:
                embryos_final = embryos_r1
                logger.info("No additional corrections needed")

        # Convert to stage coordinates
        logger.info("Converting to stage coordinates...")
        embryo_positions = self._pixel_to_stage_coordinates(
            embryos_final,
            stage_position,
            pixel_size_um,
            objective_mag,
            image_shape=image.shape[:2],  # (height, width)
        )

        # Save final visualization
        if save_visualizations:
            final_viz = self._create_annotated_image(image_8bit, embryos_final)
            cv2.imwrite(str(output_dir / "detection_final.png"), final_viz)

        # Package results
        results: dict[str, Any] = {
            "embryos": embryo_positions,
            "initial_detections": len(embryos_sam),
            "final_detections": len(embryos_final),
            "verification": verification,
            "changes": changes,
            "images": {
                "initial": str(output_dir / "detection_initial.png"),
                "final": str(output_dir / "detection_final.png"),
            },
        }

        if use_claude_review and save_visualizations:
            results["images"]["round1"] = str(output_dir / "detection_round1.png")

        logger.info("=" * 70)
        logger.info("DETECTION COMPLETE: %d embryos", len(embryo_positions))
        logger.info("=" * 70)

        return results

    @staticmethod
    def _to_rgb8(image: np.ndarray) -> np.ndarray:
        """Convert image to 8-bit RGB for SAM."""
        if image.dtype == np.uint16:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def _detect_with_sam(self, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """Run SAM automatic segmentation (extracted from test script)"""
        image_rgb = self._to_rgb8(image)

        # Generate masks
        masks = self._mask_generator.generate(image_rgb)

        # Filter candidates
        embryo_candidates = []
        for mask_data in masks:
            area = mask_data["area"]

            if not (self.min_area <= area <= self.max_area):
                continue

            bbox = mask_data["bbox"]
            mask = mask_data["segmentation"]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            perimeter = cv2.arcLength(contours[0], True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter**2)

            if circularity < self.min_circularity:
                continue

            embryo_candidates.append(
                {
                    "mask": mask,
                    "bbox": bbox,
                    "area": area,
                    "circularity": circularity,
                    "stability_score": mask_data["stability_score"],
                    "predicted_iou": mask_data["predicted_iou"],
                }
            )

        # Sort by quality and apply spatial separation
        embryo_candidates.sort(key=lambda x: x["area"] * x["stability_score"], reverse=True)

        selected_embryos: list[Any] = []
        for candidate in embryo_candidates:
            if len(selected_embryos) >= self.max_embryos:
                break

            bbox = candidate["bbox"]
            candidate_center_x = bbox[0] + bbox[2] / 2
            candidate_center_y = bbox[1] + bbox[3] / 2

            too_close = False
            for selected in selected_embryos:
                sel_bbox = selected["bbox"]
                sel_center_x = sel_bbox[0] + sel_bbox[2] / 2
                sel_center_y = sel_bbox[1] + sel_bbox[3] / 2

                distance = np.sqrt(
                    (candidate_center_x - sel_center_x) ** 2
                    + (candidate_center_y - sel_center_y) ** 2
                )

                if distance < self.min_separation_pixels:
                    too_close = True
                    break

            if not too_close:
                selected_embryos.append(candidate)

        return selected_embryos, image_rgb

    def _create_annotated_image(self, image: np.ndarray, embryos: list[dict]) -> np.ndarray:
        """Create annotated image with numbered boxes"""
        viz = image.copy()
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 128, 0),
            (128, 0, 128),
        ]

        for i, embryo in enumerate(embryos):
            bbox = embryo["bbox"]
            x, y, w, h = bbox
            color = colors[i % len(colors)]

            cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)

            label = f"{i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label, font, 0.8, 2)
            cv2.rectangle(viz, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            cv2.putText(viz, label, (x + 5, y - 5), font, 0.8, (255, 255, 255), 2)

            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(viz, (center_x, center_y), 5, color, -1)

        return viz

    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode image for Claude API"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(image)

        # Resize if too large
        if pil_image.width > 1500 or pil_image.height > 1500:
            scale = 1400 / max(pil_image.width, pil_image.height)
            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        # Compress
        quality = 92
        max_bytes = int(4.8 * 1024 * 1024)

        while quality > 30:
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=quality, optimize=True)
            if buffered.tell() <= max_bytes:
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            quality -= 5

        # Last resort
        scale = 1000 / max(pil_image.width, pil_image.height)
        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def _review_with_claude(
        self, image: np.ndarray, annotated: np.ndarray, embryos: list[dict]
    ) -> dict:
        """Round 1: Claude reviews detections (from test script)"""
        if not self.claude_client:
            return {"false_positives": [], "false_negatives": []}

        image_base64 = self._encode_image_base64(annotated)

        prompt = f"""\
You are a microscopy expert analyzing embryo detections from a bottom camera view.

CURRENT DETECTIONS: {len(embryos)} embryos labeled 0-{len(embryos) - 1} with colored bounding boxes.

EMBRYO CHARACTERISTICS:
- Small, BRIGHT white/light gray oval or rice grain shapes
- Typical size: 40-120 pixels in diameter
- Stand out clearly against dark gray background
- Have defined boundaries and smooth edges

YOUR TASK:
1. Scan the ENTIRE image systematically
2. Verify each numbered detection matches embryo characteristics
3. Look for any bright oval objects WITHOUT boxes

FALSE POSITIVES (remove): Edge artifacts, irregular shapes, dark objects, debris
FALSE NEGATIVES (add): ANY bright oval WITHOUT a box, especially in image center

Respond in JSON:
{{
  "false_positives": [detection numbers to remove],
  "false_negatives": [
    {{"x": pixel_x, "y": pixel_y, "description": "..."}}
  ],
  "analysis": "systematic check",
  "summary": "..."
}}"""

        try:
            message = self.claude_client.messages.create(
                model=settings.models.perception,
                max_tokens=8000,
                output_config={
                    "effort": "high"
                },  # was thinking budget_tokens (Opus 4.8 rejects it)
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            response_text = next((b.text for b in message.content if b.type == "text"), "")

            # Parse JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)

        except Exception as e:
            logger.warning("Claude review failed: %s", e)
            return {"false_positives": [], "false_negatives": []}

    async def _verify_with_claude(
        self, image: np.ndarray, annotated: np.ndarray, embryos: list[dict], previous_changes: dict
    ) -> dict:
        """Round 2: Claude verifies corrections (from test script)"""
        if not self.claude_client:
            return {"verified": True, "skipped": True}

        image_base64 = self._encode_image_base64(annotated)

        removed = previous_changes.get("removed", [])
        added = previous_changes.get("added", [])

        prompt = f"""VERIFICATION ROUND - You previously reviewed this image.

PREVIOUS CHANGES:
- Removed: {removed if removed else "none"}
- Added: {added if added else "none"}

CURRENT: {len(embryos)} detections (numbered 0-{len(embryos) - 1})

TASK: Verify corrections and catch any remaining issues.
Only report CLEAR remaining problems.

Respond in JSON:
{{
  "additional_false_positives": [],
  "additional_false_negatives": [],
  "verified": true/false,
  "verification_summary": "..."
}}"""

        try:
            message = self.claude_client.messages.create(
                model=settings.models.perception,
                max_tokens=6000,
                output_config={
                    "effort": "high"
                },  # was thinking budget_tokens (Opus 4.8 rejects it)
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            response_text = next((b.text for b in message.content if b.type == "text"), "")

            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)

        except Exception as e:
            logger.warning("Verification failed: %s", e)
            return {"verified": False}

    def _apply_corrections(
        self, embryos: list[dict], review: dict, image: np.ndarray, predictor
    ) -> tuple[list[dict], dict]:
        """Apply Claude's corrections (from test script)"""
        corrected = []
        changes: dict[str, Any] = {"removed": [], "added": []}

        # Remove false positives
        false_positives = set(review.get("false_positives", []))
        if false_positives:
            changes["removed"] = list(false_positives)

        for i, embryo in enumerate(embryos):
            if i not in false_positives:
                corrected.append(embryo)

        # Add false negatives
        false_negatives = review.get("false_negatives", [])
        if false_negatives:
            for fn in false_negatives:
                point = (fn["x"], fn["y"])
                new_embryo = self._segment_with_sam(image, predictor, point)

                if new_embryo and (
                    self.min_area <= new_embryo["area"] <= self.max_area
                    and new_embryo["circularity"] >= self.min_circularity
                ):
                    corrected.append(new_embryo)
                    changes["added"].append(point)

        return corrected, changes

    def _segment_with_sam(self, image: np.ndarray, predictor, point: tuple) -> dict | None:
        """Use SAM predictor to segment region (from test script)"""
        image_rgb = self._to_rgb8(image)
        predictor.set_image(image_rgb)

        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=True
        )

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        rows, cols = np.where(mask)
        if len(rows) == 0:
            return None

        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        area = mask.sum()
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        else:
            circularity = 0

        return {
            "mask": mask,
            "bbox": bbox,
            "area": int(area),
            "circularity": float(circularity),
            "stability_score": float(scores[best_idx]),
            "predicted_iou": float(scores[best_idx]),
        }

    def _pixel_to_stage_coordinates(
        self,
        embryos: list[dict],
        stage_pos: tuple[float, float],
        pixel_size_um: float,
        objective_mag: float,
        image_shape: tuple[int, int] = (2048, 2048),
    ) -> list[dict]:
        """
        Convert pixel coordinates to stage coordinates.

        Uses centralized coordinate transformation from gently/coordinates.py.
        Returns the stage position that would CENTER each embryo.
        """
        effective_pixel_um = get_um_per_pixel(pixel_size_um, objective_mag)
        stage_x, stage_y = stage_pos

        # Image center (for offset calculation)
        image_center_x = image_shape[1] / 2  # width
        image_center_y = image_shape[0] / 2  # height

        embryo_positions = []
        for i, embryo in enumerate(embryos):
            bbox = embryo["bbox"]
            x, y, w, h = bbox

            center_x_px = x + w / 2
            center_y_px = y + h / 2

            # Convert to stage coordinates using centralized function
            # This returns the stage position that would CENTER this embryo
            embryo_stage_x, embryo_stage_y = pixel_to_stage_position(
                pixel_x=center_x_px,
                pixel_y=center_y_px,
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                stage_x=stage_x,
                stage_y=stage_y,
                um_per_pixel=effective_pixel_um,
            )

            embryo_positions.append(
                {
                    "embryo_id": f"embryo_{i + 1}",
                    "pixel_x": float(center_x_px),
                    "pixel_y": float(center_y_px),
                    "stage_x_um": float(embryo_stage_x),
                    "stage_y_um": float(embryo_stage_y),
                    "bbox_pixel": tuple(bbox),
                    "area_pixels": embryo.get("area_pixels", embryo.get("area", 0)),
                    "circularity": embryo.get("circularity", 0),
                    "confidence": embryo.get("confidence", embryo.get("stability_score", 0)),
                }
            )

        return embryo_positions

    def show_in_napari(self, image: np.ndarray, embryos: list[dict], block: bool = False):
        """Deprecated: napari display was retired in Phase 1.

        SAM detection results are now reviewed via the web map view —
        :func:`gently.ui.web.embryo_marker.mark_embryos_web` accepts the
        detections as ``initial_markers``. This stub is kept so older
        callers don't import-error; it logs a warning and returns None.
        """
        logger.warning(
            "show_in_napari is deprecated; use the web map view "
            "(mark_embryos_web) to review SAM detections."
        )
        return None
