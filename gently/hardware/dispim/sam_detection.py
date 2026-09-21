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
from typing import Any, cast

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
            img_no_bg = cv2.normalize(  # type: ignore[call-overload]  # cv2 stub disallows None dst, valid at runtime
                img_no_bg, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
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

    # Reference image size the blob-detector scales were validated at: the
    # 410 px bottom-camera display frame (5x downsample of the 2048 px raw
    # capture). Larger inputs are area-downsampled to this size for candidate
    # finding and the results mapped back, so every resolution sees the same
    # validated parameters (and the raw frame stays fast).
    _REF_MAXDIM = 410.0

    def find_embryo_candidates(
        self,
        image: np.ndarray,
        brightness_percentile: float = 99.0,
        min_area: int | None = None,
        max_area: int | None = None,
        clahe_clip: float = 3.0,
        clahe_tile: int = 16,
        mad_k: float = 6.0,
        min_relative_peak: float = 0.6,
        max_candidates: int | None = None,
    ) -> tuple[list[dict], np.ndarray]:
        """
        Find embryo candidates by flat-fielding + scale-matched blob detection.

        Embryos are compact BRIGHT ovals on a noisy background that carries a
        strong low-frequency illumination gradient (dark corners) and, often, a
        diffuse bright glow. The previous approach — a global brightness
        percentile + dilation — could not separate a compact embryo from those
        large bright regions, so it fired on the gradient and the glow (≈1 false
        positive per real embryo) while its 2048²-tuned ``min_area`` silently
        rejected genuine embryos on a downsampled frame.

        This method instead:

        1. Median-filters to kill salt-and-pepper noise.
        2. Applies a white top-hat with a kernel larger than an embryo, which
           removes anything bigger than an embryo — the gradient and the diffuse
           glow — while keeping compact bright structures.
        3. Smooths at the embryo scale (matched filter) to pool signal and
           suppress single-pixel spikes.
        4. Takes local maxima (with a minimum separation) above a noise floor,
           ``median + mad_k · 1.4826 · MAD`` of that response.
        5. Keeps only peaks at least ``min_relative_peak`` × the strongest one.
           Embryos are the brightest compact objects in the field; debris and
           specks are real structure, often far above the noise floor, but
           clearly dimmer than the embryos. On raw frames a noise-relative
           threshold alone let that debris through.

        Inputs larger than ``_REF_MAXDIM`` are area-downsampled to it for these
        steps and the candidates mapped back to input pixels, so the same call
        works on the 410 px display frame and the raw 2048 px capture.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale image (16-bit or 8-bit).
        brightness_percentile : float
            Deprecated / ignored. Kept for call-site compatibility.
        min_area, max_area : int, optional
            Hard bounds (in pixels, at the input resolution) on a candidate's
            above-threshold blob area. ``None`` (default) means no lower bound
            and an upper bound of ~12x the nominal embryo area at this
            resolution. Pass explicit values only to override — a fixed
            ``min_area`` tuned for one resolution is what used to reject
            genuine embryos on another.
        clahe_clip, clahe_tile : float, int
            CLAHE settings for the 8-bit image handed to SAM (contrast only;
            does not affect which candidates are found).
        mad_k : float
            Noise floor in robust standard deviations above the background;
            peaks below it are never candidates. Guards against returning pure
            noise, while ``min_relative_peak`` does the main discrimination.
        min_relative_peak : float
            Keep peaks whose background-subtracted response is at least this
            fraction of the strongest peak's. Lower = more recall for dim
            embryos, more debris. Note this normalises by the brightest peak, so
            a compact artifact brighter than every embryo suppresses them; pass
            0.0 with ``max_candidates`` when a later stage can reject junk.
        max_candidates : int, optional
            Keep at most this many candidates, strongest first. Bounds the work
            handed to a later filter without thresholding on relative strength.

        Returns
        -------
        candidates : List[Dict]
            Candidates with keys ``bbox`` (x, y, w, h), ``centroid`` (cx, cy),
            ``area`` (pixels) and ``relative_strength`` (this peak's response as
            a fraction of the strongest peak's).
        enhanced_image : np.ndarray
            Contrast-enhanced 8-bit image for SAM.
        """
        from skimage.feature import peak_local_max

        logger.info("Finding embryo candidates (flat-field + blob, mad_k=%.1f)...", mad_k)
        logger.debug("Input range: %s - %s", image.min(), image.max())

        h, w = image.shape[:2]

        # 8-bit normalization (percentile stretch handles low dynamic range).
        p2, p98 = np.percentile(image, (2, 98))
        denom = float(p98 - p2) or 1.0
        img_norm = np.clip((image.astype(np.float32) - p2) / denom * 255, 0, 255).astype(np.uint8)

        # CLAHE image is only for SAM's benefit (better contrast to segment on),
        # so it stays at full resolution.
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        img_enhanced = clahe.apply(img_norm)

        # Candidate finding runs at (at most) the reference resolution: the
        # scales below were validated there, and a top-hat with an
        # embryo-scaled ellipse kernel costs ~13 s on a 2048² frame versus
        # <0.1 s at 410 px. Area-averaged downsampling also lifts embryo SNR.
        factor = max(1.0, max(h, w) / self._REF_MAXDIM)
        if factor > 1.0:
            work_size = (max(1, round(w / factor)), max(1, round(h / factor)))
            work = cv2.resize(img_norm, work_size, interpolation=cv2.INTER_AREA)
        else:
            work = img_norm
        wh, ww = work.shape[:2]
        sx, sy = w / ww, h / wh  # working frame -> input frame

        # Scales at the working resolution (== the reference values unless the
        # input is smaller than the reference frame).
        s = max(wh, ww) / self._REF_MAXDIM
        r_emb = max(2.0, 7.7 * s)  # embryo radius ≈ 7.7 px at the reference frame
        r_bg = max(3, int(round(25 * s)))  # top-hat kernel: > embryo, < gradient
        embryo_sigma = max(1.0, 4.0 * s)  # matched-filter smoothing
        min_sep = max(2, int(round(12 * s)))  # min separation between embryos
        border = max(1, int(round(6 * s)))  # ignore peaks this close to the edge

        denoised = cv2.medianBlur(work, 3).astype(np.float32)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_bg + 1, 2 * r_bg + 1))
        tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kern)
        resp = cv2.GaussianBlur(tophat, (0, 0), sigmaX=embryo_sigma)

        med = float(np.median(resp))
        mad = float(np.median(np.abs(resp - med))) + 1e-6
        thr = med + mad_k * 1.4826 * mad

        peaks = peak_local_max(resp, min_distance=min_sep, threshold_abs=thr, exclude_border=border)
        # Strength of each peak as a fraction of the strongest one. Kept on every
        # candidate so a later stage (e.g. the Claude filter) can run this step
        # permissively and still fall back to a conservative cut if it fails.
        rel_strength = np.ones(len(peaks), dtype=float)
        if len(peaks):
            strength = resp[peaks[:, 0], peaks[:, 1]] - med
            rel_strength = strength / (strength.max() or 1.0)
            keep = rel_strength >= min_relative_peak
            peaks, rel_strength = peaks[keep], rel_strength[keep]

        # Area bounds, in INPUT-resolution pixels. Explicit values override;
        # by default there is only an upper bound. The measured area is the
        # above-threshold footprint, which shrinks as mad_k rises — an automatic
        # lower bound would couple to the threshold and drop faint embryos
        # that the matched filter has already accepted.
        px_area = sx * sy
        lo = float(min_area) if min_area is not None else 0.0
        hi = float(max_area) if max_area is not None else 12.0 * np.pi * r_emb**2 * px_area

        # Bounding boxes from the thresholded blob mask; peaks sharing one blob
        # (touching embryos) each get a local box so SAM refines them separately.
        mask = (resp > thr).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        peaks_per_label: dict[int, int] = {}
        for py, px in peaks:
            lbl = int(labels[py, px])
            peaks_per_label[lbl] = peaks_per_label.get(lbl, 0) + 1

        candidates = []
        for (py, px), rel in zip(peaks, rel_strength, strict=True):
            lbl = int(labels[py, px])
            if lbl != 0 and peaks_per_label[lbl] == 1:
                ys, xs = np.where(labels == lbl)
                bx0, bx1 = float(xs.min()), float(xs.max() + 1)
                by0, by1 = float(ys.min()), float(ys.max() + 1)
                area_work = float(len(xs))
            else:
                # No blob (edge) or a shared blob: use a local embryo-sized box.
                half = 1.5 * r_emb
                bx0, bx1 = px + 0.5 - half, px + 0.5 + half
                by0, by1 = py + 0.5 - half, py + 0.5 + half
                area_work = float(np.pi * r_emb**2)

            area = area_work * px_area
            if not (lo <= area <= hi):
                continue
            # Map working-frame box/peak back to input pixels.
            x0 = max(0, int(np.floor(bx0 * sx)))
            y0 = max(0, int(np.floor(by0 * sy)))
            x1 = min(w, int(np.ceil(bx1 * sx)))
            y1 = min(h, int(np.ceil(by1 * sy)))
            candidates.append(
                {
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "centroid": ((px + 0.5) * sx - 0.5, (py + 0.5) * sy - 0.5),
                    "area": area,
                    "relative_strength": float(rel),
                }
            )

        if max_candidates is not None and len(candidates) > max_candidates:
            candidates.sort(key=lambda c: -c["relative_strength"])
            candidates = candidates[:max_candidates]

        logger.info("Found %d embryo candidates", len(candidates))
        return candidates, img_enhanced

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

    @staticmethod
    def embryos_from_candidates(candidates: list[dict]) -> list[dict]:
        """Blob candidates in the shape SAM refinement would have returned.

        SAM contributes an outline, not a position: it is given the boxes the
        blob finder proposed and segments inside them. So when it is switched
        off — no GPU, a checkpoint that will not load, or an operator who wants
        the fast path — the candidates themselves are already a usable answer,
        and every consumer downstream (stage conversion, the marking canvas,
        Register) only reads the centre, the box and a confidence.

        ``confidence`` carries the blob's relative strength, which is what the
        candidate finder ranks on; ``circularity`` is SAM's to measure, so it
        is reported as 0.0 rather than guessed.
        """
        out: list[dict] = []
        for i, cand in enumerate(candidates):
            cx, cy = cand["centroid"]
            bx, by, bw, bh = cand["bbox"]
            out.append(
                {
                    "embryo_id": f"embryo_{i + 1}",
                    "uid": str(uuid.uuid4()),
                    "pixel_x": float(cx),
                    "pixel_y": float(cy),
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "area_pixels": int(cand.get("area", bw * bh)),
                    "circularity": 0.0,
                    "confidence": float(cand.get("relative_strength", 0.0)),
                    "mask": None,
                }
            )
        return out

    async def detect_embryos(
        self,
        image: np.ndarray,
        stage_position: tuple[float, float],
        pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
        objective_mag: float = DEFAULT_OBJECTIVE_MAG,
        use_claude_review: bool = True,
        use_sam: bool = True,
        save_visualizations: bool = True,
        output_dir: Path | None = None,
        brightness_percentile: float = 99.0,
        min_area: int | None = None,
        max_area: int | None = None,
        min_relative_peak: float | None = None,
    ) -> dict:
        """
        Detect embryos using blob-based candidate finding + SAM refinement.

        This hybrid approach:
        1. Finds candidate embryos with flat-fielding + blob detection
           (:meth:`find_embryo_candidates`) — this step sets recall, since SAM
           only refines the boxes it is given
        2. Optionally asks Claude to classify each candidate crop, removing
           the ones that are not embryos (it never adds — recall stays with
           step 1). Falls back to a conservative cut if the call fails.
        3. Uses SAM with bounding box prompts to get precise segmentation

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
        use_sam : bool
            Whether to refine the candidate boxes with SAM (default: True).
            False returns the blob candidates themselves — no GPU, no
            checkpoint, no outline. Recall is unchanged either way: SAM only
            refines boxes step 1 proposed.
        save_visualizations : bool
            Whether to save annotated images (default: True)
        output_dir : Path, optional
            Where to save visualizations. If None, uses './detection_results'
        brightness_percentile : float
            Deprecated / ignored (see ``min_relative_peak``).
        min_area, max_area : int, optional
            Optional hard blob-area bounds in input pixels; ``None`` (default)
            auto-scales from the image resolution.
        min_relative_peak : float, optional
            Keep candidates at least this fraction as strong as the strongest
            one. ``None`` (default) picks it from whether the Claude filter is
            available: permissive when it is, conservative when it is not.

        Returns
        -------
        dict
            Detection results with keys:
            - embryos: List[Dict] - Embryo positions and metadata
            - initial_detections: int
            - final_detections: int
            - review: Dict - Claude candidate-review outcome
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
        logger.info("BLOB + SAM EMBRYO DETECTION")
        logger.info("=" * 70)

        # Step 1: Find candidates (flat-field + blob detection).
        # With the Claude filter available, propose permissively and let the
        # filter remove the junk; without it, cut conservatively here.
        review_enabled = bool(use_claude_review and self.claude_client)
        if min_relative_peak is None:
            min_relative_peak = 0.0 if review_enabled else self._NO_REVIEW_RELATIVE_PEAK
        max_candidates = self._REVIEW_MAX_CANDIDATES if review_enabled else None
        logger.info(
            "[1/3] Finding embryo candidates (min_relative_peak=%.2f, max=%s)...",
            min_relative_peak,
            max_candidates,
        )
        candidates, image_enhanced = self.find_embryo_candidates(
            image,
            brightness_percentile=brightness_percentile,
            min_area=min_area,
            max_area=max_area,
            min_relative_peak=min_relative_peak,
            max_candidates=max_candidates,
        )

        # Step 2: Claude classifies each candidate (removes only; never adds).
        review: dict[str, Any] = {"reviewed": False, "skipped": not use_claude_review}
        if use_claude_review and candidates:
            logger.info("[2/3] Claude reviewing %d candidates...", len(candidates))
            h_img, w_img = image.shape[:2]
            half = max(32, int(round(3.0 * self._embryo_radius_px((h_img, w_img)))))
            candidates, review = await self._classify_candidates_with_claude(
                image_enhanced, candidates, half
            )

        if len(candidates) == 0:
            logger.warning("No embryo candidates found!")
            return {
                "embryos": [],
                "initial_detections": 0,
                "final_detections": 0,
                "review": {"reviewed": False},
                "images": {},
            }

        # Step 3: Refine with SAM — optional, because it contributes outlines
        # rather than positions. Skipping it keeps every candidate.
        if use_sam:
            logger.info("[3/3] Refining with SAM...")
            embryos_sam = self.refine_with_sam(image_enhanced, candidates)
            logger.info("SAM refined %d embryos", len(embryos_sam))
        else:
            logger.info("[3/3] SAM skipped — using %d blob candidates", len(candidates))
            embryos_sam = self.embryos_from_candidates(candidates)

        # Use enhanced image for visualization
        image_8bit = image_enhanced

        if len(embryos_sam) == 0:
            logger.warning("No embryos left after the %s step!", "SAM" if use_sam else "candidate")
            return {
                "embryos": [],
                "initial_detections": 0,
                "final_detections": 0,
                "review": {"reviewed": False},
                "images": {},
            }

        # Save initial detection
        if save_visualizations:
            initial_viz = self._create_annotated_image(image_8bit, embryos_sam)
            cv2.imwrite(str(output_dir / "detection_initial.png"), initial_viz)

        embryos_final = embryos_sam

        # Convert to stage coordinates
        logger.info("Converting to stage coordinates...")
        embryo_positions = self._pixel_to_stage_coordinates(
            embryos_final,
            stage_position,
            pixel_size_um,
            objective_mag,
            image_shape=cast("tuple[int, int]", image.shape[:2]),  # (height, width)
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
            "review": review,
            "images": {
                "initial": str(output_dir / "detection_initial.png"),
                "final": str(output_dir / "detection_final.png"),
            },
        }

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

    def _embryo_radius_px(self, shape: tuple[int, int]) -> float:
        """Nominal embryo radius in input pixels for an image of this size."""
        return max(2.0, 7.7 * max(shape) / self._REF_MAXDIM)

    # Candidate finding runs permissively when the Claude filter is available
    # (the filter removes the junk), and conservatively when it is not.
    # With the Claude filter available, do NOT threshold on relative strength
    # at all: just hand it the strongest N peaks above the noise floor. A
    # relative-strength cut normalises by the BRIGHTEST peak, so one compact
    # artifact brighter than the embryos (a dust glint, a bubble catching the
    # light) collapses every real embryo's score and silently drops them --
    # measured at 3 of 4 real frames losing most or all of their embryos.
    # Proposals are cheap (~11 per field) and the filter removes the junk.
    _REVIEW_MAX_CANDIDATES = 16
    _NO_REVIEW_RELATIVE_PEAK = 0.6

    # Contact-sheet geometry for the candidate classifier.
    _TILE_PX = 224
    _TILE_COLS = 5

    def _candidate_contact_sheet(
        self, image8: np.ndarray, candidates: list[dict], half: int
    ) -> np.ndarray:
        """Tile a crop of each candidate into one numbered contact sheet.

        Crops, not the whole frame: an embryo is ~77 px in a 2048 px capture,
        which is marginal once the full frame is downscaled for the API. Each
        tile is independently contrast-stretched so a faint embryo is visible,
        and carries a centre reticle marking which object is being judged.
        """
        tile, cols = self._TILE_PX, self._TILE_COLS
        rows = (len(candidates) + cols - 1) // cols
        canvas = np.zeros((rows * tile, cols * tile, 3), np.uint8)
        h, w = image8.shape[:2]

        for i, cand in enumerate(candidates):
            cx, cy = cand["centroid"]
            x, y = int(cx), int(cy)
            x0, y0 = max(0, x - half), max(0, y - half)
            x1, y1 = min(w, x + half), min(h, y + half)
            crop = image8[y0:y1, x0:x1]
            pad = np.zeros((2 * half, 2 * half), np.uint8)
            pad[: crop.shape[0], : crop.shape[1]] = crop

            lo, hi = np.percentile(pad, (1, 99))
            pad = np.clip((pad.astype(np.float32) - lo) / ((hi - lo) or 1.0) * 255, 0, 255).astype(
                np.uint8
            )
            t = cv2.cvtColor(cv2.resize(pad, (tile, tile)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(t, (0, 0), (tile - 1, tile - 1), (60, 60, 60), 1)
            cv2.rectangle(t, (0, 0), (34, 20), (0, 0, 0), -1)
            cv2.putText(t, str(i), (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.drawMarker(t, (tile // 2, tile // 2), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

            r, c = divmod(i, cols)
            canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = t
        return canvas

    _CLASSIFY_PROMPT = (
        "These are candidate regions cropped from a single bottom-camera microscope "
        "image of a glass slide holding C. elegans embryos. Each tile is centred on "
        "one candidate and numbered.\n\n"
        "A C. elegans embryo is a compact, self-contained OVAL / rice-grain shape, "
        "brighter than its surroundings, with a defined edge all the way around and "
        "some internal texture. It sits in the MIDDLE of the tile - that is where "
        "the candidate was found.\n\n"
        "Common false alarms: a bright RIDGE, arc or streak that runs off the side "
        "of the tile (the out-of-focus edge of a bubble or the meniscus); a broad "
        "smooth glow with no defined boundary; a speck of debris far smaller than "
        "an embryo; flat noise with nothing structured at the centre.\n\n"
        "For EVERY numbered tile, decide whether the object at the CENTRE of that "
        "tile is an embryo.\n\n"
        "Return ONLY JSON, no prose:\n"
        '{"verdicts": [{"i": <tile number>, "embryo": true|false, "confidence": 0.0-1.0}]}'
    )

    async def _classify_candidates_with_claude(
        self, image8: np.ndarray, candidates: list[dict], half: int
    ) -> tuple[list[dict], dict]:
        """Ask Claude which candidates are really embryos.

        The detector has already answered *where*; this only answers *what*,
        which is the part a vision model is reliable at. It can only remove
        candidates, never add one - recall stays the detector's job.

        On any failure (no API key, network, unparseable reply) this falls back
        to a conservative cut on ``relative_strength``, i.e. what the detector
        would have returned on its own. Detection must not depend on a network
        call succeeding.
        """
        fallback = [
            c
            for c in candidates
            if c.get("relative_strength", 1.0) >= self._NO_REVIEW_RELATIVE_PEAK
        ]
        if not self.claude_client or not candidates:
            return fallback, {"reviewed": False, "reason": "no Claude client"}

        try:
            sheet = self._candidate_contact_sheet(image8, candidates, half)
            b64 = self._encode_image_base64(sheet)
            message = self.claude_client.messages.create(
                model=settings.models.perception,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": self._CLASSIFY_PROMPT},
                        ],
                    }
                ],
            )
            text = "".join(b.text for b in message.content if b.type == "text")
            payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
            verdicts = payload["verdicts"]
        except Exception as e:
            logger.warning(
                "Claude candidate review failed (%s); falling back to relative_strength >= %.2f",
                e,
                self._NO_REVIEW_RELATIVE_PEAK,
            )
            return fallback, {"reviewed": False, "reason": str(e)}

        kept, conf = [], {}
        for v in verdicts:
            try:
                i = int(v["i"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(candidates) and v.get("embryo"):
                kept.append(candidates[i])
                conf[i] = v.get("confidence")

        logger.info("Claude kept %d of %d candidates", len(kept), len(candidates))
        return kept, {
            "reviewed": True,
            "proposed": len(candidates),
            "kept": len(kept),
            "removed": len(candidates) - len(kept),
            "confidence": conf,
        }

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
