"""
SAM + Claude Vision Embryo Detection Module

Extracts detection logic from test_sam_claude_hybrid_detection.py into reusable module.
Returns embryo positions (pixel + stage coordinates) for calibration workflow.
"""

import time
import json
import numpy as np
from pathlib import Path
import cv2
import base64
from io import BytesIO
from PIL import Image
import anthropic
from typing import Dict, List, Tuple, Optional
import os


class SAMEmbryoDetector:
    """
    Embryo detector using SAM + Claude Vision hybrid approach

    Features:
    - Initial segmentation with SAM
    - 2-round Claude Vision review for false positives/negatives
    - Napari visualization (optional)
    - Returns embryo positions as simple list of coordinates
    """

    def __init__(self,
                 sam_checkpoint: str = "sam_vit_b_01ec64.pth",
                 sam_model_type: str = "vit_b",
                 device: str = "cpu",
                 anthropic_api_key: Optional[str] = None):
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
        self._mask_generator = None
        self._predictor = None

    def _load_sam(self):
        """Lazy load SAM models"""
        if self._mask_generator is not None:
            return

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        if not Path(self.sam_checkpoint).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")

        print(f"Loading SAM model: {self.sam_model_type} on {self.device}")
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
        print("✓ SAM model loaded")

    async def detect_embryos(self,
                            image: np.ndarray,
                            stage_position: Tuple[float, float],
                            pixel_size_um: float = 6.5,
                            objective_mag: float = 4.0,
                            use_claude_review: bool = True,
                            save_visualizations: bool = True,
                            output_dir: Optional[Path] = None) -> Dict:
        """
        Detect embryos in bottom camera image

        Parameters
        ----------
        image : np.ndarray
            Bottom camera image (grayscale or RGB)
        stage_position : tuple
            Current XY stage position (x, y) in micrometers
        pixel_size_um : float
            Camera pixel size in micrometers (default: 6.5 for PCO)
        objective_mag : float
            Objective magnification (default: 4x for bottom camera)
        use_claude_review : bool
            Whether to use Claude Vision for review (default: True)
        save_visualizations : bool
            Whether to save annotated images (default: True)
        output_dir : Path, optional
            Where to save visualizations. If None, uses './detection_results'

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
        # Load SAM if needed
        self._load_sam()

        # Setup output directory
        if output_dir is None:
            output_dir = Path("./detection_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("SAM + CLAUDE EMBRYO DETECTION")
        print("="*70)

        # Initial SAM detection
        print("\n[1/4] Running SAM segmentation...")
        embryos_sam, image_8bit = self._detect_with_sam(image)
        print(f"  ✓ SAM detected {len(embryos_sam)} candidates")

        if len(embryos_sam) == 0:
            print("  ⚠ No embryos detected by SAM!")
            return {
                'embryos': [],
                'initial_detections': 0,
                'final_detections': 0,
                'verification': {'verified': False},
                'images': {}
            }

        # Save initial detection
        if save_visualizations:
            initial_viz = self._create_annotated_image(image_8bit, embryos_sam)
            cv2.imwrite(str(output_dir / "detection_initial.png"), initial_viz)

        # Claude review (if enabled)
        embryos_final = embryos_sam
        verification = {'verified': True, 'skipped': not use_claude_review}
        changes = {'round1': {'removed': [], 'added': []}}

        if use_claude_review and self.claude_client:
            print("\n[2/4] Claude Vision review (Round 1)...")
            annotated = self._create_annotated_image(image_8bit, embryos_sam)
            review_r1 = await self._review_with_claude(image_8bit, annotated, embryos_sam)

            print("\n[3/4] Applying corrections...")
            embryos_r1, changes['round1'] = self._apply_corrections(
                embryos_sam, review_r1, image, self._predictor
            )

            if save_visualizations:
                r1_viz = self._create_annotated_image(image_8bit, embryos_r1)
                cv2.imwrite(str(output_dir / "detection_round1.png"), r1_viz)

            # Round 2: Verification
            print("\n[4/4] Claude verification (Round 2)...")
            r1_viz = self._create_annotated_image(image_8bit, embryos_r1)
            verification = await self._verify_with_claude(
                image_8bit, r1_viz, embryos_r1, changes['round1']
            )

            # Apply round 2 corrections if needed
            has_r2_changes = (
                len(verification.get('additional_false_positives', [])) > 0 or
                len(verification.get('additional_false_negatives', [])) > 0
            )

            if has_r2_changes:
                print("  Applying Round 2 corrections...")
                review_r2 = {
                    'false_positives': verification.get('additional_false_positives', []),
                    'false_negatives': verification.get('additional_false_negatives', [])
                }
                embryos_final, changes['round2'] = self._apply_corrections(
                    embryos_r1, review_r2, image, self._predictor
                )
            else:
                embryos_final = embryos_r1
                print("  ✓ No additional corrections needed")

        # Convert to stage coordinates
        print("\nConverting to stage coordinates...")
        embryo_positions = self._pixel_to_stage_coordinates(
            embryos_final,
            stage_position,
            pixel_size_um,
            objective_mag
        )

        # Save final visualization
        if save_visualizations:
            final_viz = self._create_annotated_image(image_8bit, embryos_final)
            cv2.imwrite(str(output_dir / "detection_final.png"), final_viz)

        # Package results
        results = {
            'embryos': embryo_positions,
            'initial_detections': len(embryos_sam),
            'final_detections': len(embryos_final),
            'verification': verification,
            'changes': changes,
            'images': {
                'initial': str(output_dir / "detection_initial.png"),
                'final': str(output_dir / "detection_final.png")
            }
        }

        if use_claude_review and save_visualizations:
            results['images']['round1'] = str(output_dir / "detection_round1.png")

        print("\n" + "="*70)
        print(f"DETECTION COMPLETE: {len(embryo_positions)} embryos")
        print("="*70)

        return results

    def _detect_with_sam(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Run SAM automatic segmentation (extracted from test script)"""
        # Convert to 8-bit
        if image.dtype == np.uint16:
            image_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_8bit = image.astype(np.uint8)

        # Convert grayscale to RGB for SAM
        if len(image_8bit.shape) == 2:
            image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_8bit

        # Generate masks
        masks = self._mask_generator.generate(image_rgb)

        # Filter candidates
        embryo_candidates = []
        for mask_data in masks:
            area = mask_data['area']

            if not (self.min_area <= area <= self.max_area):
                continue

            bbox = mask_data['bbox']
            mask = mask_data['segmentation']
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            perimeter = cv2.arcLength(contours[0], True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < self.min_circularity:
                continue

            embryo_candidates.append({
                'mask': mask,
                'bbox': bbox,
                'area': area,
                'circularity': circularity,
                'stability_score': mask_data['stability_score'],
                'predicted_iou': mask_data['predicted_iou']
            })

        # Sort by quality and apply spatial separation
        embryo_candidates.sort(key=lambda x: (x['area'] * x['stability_score']), reverse=True)

        selected_embryos = []
        for candidate in embryo_candidates:
            if len(selected_embryos) >= self.max_embryos:
                break

            bbox = candidate['bbox']
            candidate_center_x = bbox[0] + bbox[2] / 2
            candidate_center_y = bbox[1] + bbox[3] / 2

            too_close = False
            for selected in selected_embryos:
                sel_bbox = selected['bbox']
                sel_center_x = sel_bbox[0] + sel_bbox[2] / 2
                sel_center_y = sel_bbox[1] + sel_bbox[3] / 2

                distance = np.sqrt((candidate_center_x - sel_center_x)**2 +
                                 (candidate_center_y - sel_center_y)**2)

                if distance < self.min_separation_pixels:
                    too_close = True
                    break

            if not too_close:
                selected_embryos.append(candidate)

        return selected_embryos, image_8bit

    def _create_annotated_image(self, image: np.ndarray, embryos: List[Dict]) -> np.ndarray:
        """Create annotated image with numbered boxes"""
        viz = image.copy()
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)]

        for i, embryo in enumerate(embryos):
            bbox = embryo['bbox']
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
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            quality -= 5

        # Last resort
        scale = 1000 / max(pil_image.width, pil_image.height)
        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def _review_with_claude(self, image: np.ndarray, annotated: np.ndarray, embryos: List[Dict]) -> Dict:
        """Round 1: Claude reviews detections (from test script)"""
        if not self.claude_client:
            return {'false_positives': [], 'false_negatives': []}

        image_base64 = self._encode_image_base64(annotated)

        prompt = f"""You are a microscopy expert analyzing embryo detections from a bottom camera view.

CURRENT DETECTIONS: {len(embryos)} embryos labeled 0-{len(embryos)-1} with colored bounding boxes.

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
                model="claude-sonnet-4-5-20250929",
                max_tokens=8000,
                thinking={"type": "enabled", "budget_tokens": 5000},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
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
            print(f"  ⚠ Claude review failed: {e}")
            return {'false_positives': [], 'false_negatives': []}

    async def _verify_with_claude(self, image: np.ndarray, annotated: np.ndarray,
                                  embryos: List[Dict], previous_changes: Dict) -> Dict:
        """Round 2: Claude verifies corrections (from test script)"""
        if not self.claude_client:
            return {'verified': True, 'skipped': True}

        image_base64 = self._encode_image_base64(annotated)

        removed = previous_changes.get('removed', [])
        added = previous_changes.get('added', [])

        prompt = f"""VERIFICATION ROUND - You previously reviewed this image.

PREVIOUS CHANGES:
- Removed: {removed if removed else "none"}
- Added: {added if added else "none"}

CURRENT: {len(embryos)} detections (numbered 0-{len(embryos)-1})

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
                model="claude-sonnet-4-5-20250929",
                max_tokens=6000,
                thinking={"type": "enabled", "budget_tokens": 4000},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
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
            print(f"  ⚠ Verification failed: {e}")
            return {'verified': False}

    def _apply_corrections(self, embryos: List[Dict], review: Dict,
                          image: np.ndarray, predictor) -> Tuple[List[Dict], Dict]:
        """Apply Claude's corrections (from test script)"""
        corrected = []
        changes = {'removed': [], 'added': []}

        # Remove false positives
        false_positives = set(review.get('false_positives', []))
        if false_positives:
            changes['removed'] = list(false_positives)

        for i, embryo in enumerate(embryos):
            if i not in false_positives:
                corrected.append(embryo)

        # Add false negatives
        false_negatives = review.get('false_negatives', [])
        if false_negatives:
            for fn in false_negatives:
                point = (fn['x'], fn['y'])
                new_embryo = self._segment_with_sam(image, predictor, point)

                if new_embryo and (self.min_area <= new_embryo['area'] <= self.max_area and
                                   new_embryo['circularity'] >= self.min_circularity):
                    corrected.append(new_embryo)
                    changes['added'].append(point)

        return corrected, changes

    def _segment_with_sam(self, image: np.ndarray, predictor, point: Tuple) -> Optional[Dict]:
        """Use SAM predictor to segment region (from test script)"""
        # Convert image
        if image.dtype == np.uint16:
            image_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_8bit = image.astype(np.uint8)

        if len(image_8bit.shape) == 2:
            image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_8bit

        predictor.set_image(image_rgb)

        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
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
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        else:
            circularity = 0

        return {
            'mask': mask,
            'bbox': bbox,
            'area': int(area),
            'circularity': float(circularity),
            'stability_score': float(scores[best_idx]),
            'predicted_iou': float(scores[best_idx])
        }

    def _pixel_to_stage_coordinates(self, embryos: List[Dict], stage_pos: Tuple[float, float],
                                    pixel_size_um: float, objective_mag: float) -> List[Dict]:
        """Convert pixel coordinates to stage coordinates"""
        effective_pixel_um = pixel_size_um / objective_mag
        stage_x, stage_y = stage_pos

        embryo_positions = []
        for i, embryo in enumerate(embryos):
            bbox = embryo['bbox']
            x, y, w, h = bbox

            center_x_px = x + w / 2
            center_y_px = y + h / 2

            # Stage coordinates
            embryo_stage_x = stage_x + (center_x_px * effective_pixel_um)
            embryo_stage_y = stage_y + (center_y_px * effective_pixel_um)

            embryo_positions.append({
                'embryo_id': i,
                'pixel_x': float(center_x_px),
                'pixel_y': float(center_y_px),
                'stage_x_um': float(embryo_stage_x),
                'stage_y_um': float(embryo_stage_y),
                'bbox_pixel': tuple(bbox),
                'area_pixels': embryo['area'],
                'circularity': embryo['circularity'],
                'confidence': embryo['stability_score']
            })

        return embryo_positions

    def show_in_napari(self, image: np.ndarray, embryos: List[Dict], block: bool = False):
        """
        Display detection results in napari viewer

        Parameters
        ----------
        image : np.ndarray
            Original image
        embryos : list
            Detected embryos (from detect_embryos)
        block : bool
            Whether to block until viewer closed (default: False for non-blocking)
        """
        try:
            import napari
        except ImportError:
            print("  ⚠ napari not installed. Install with: pip install napari[all]")
            return None

        viewer = napari.Viewer(title="Embryo Detection - Review Required")
        viewer.add_image(image, name='Bottom Camera', colormap='gray')

        # Add bounding boxes
        shapes = []
        properties = {'detection_id': [], 'confidence': []}

        for embryo in embryos:
            bbox = embryo['bbox_pixel']
            x, y, w, h = bbox
            rect = np.array([[y, x], [y, x+w], [y+h, x+w], [y+h, x]])
            shapes.append(rect)
            properties['detection_id'].append(embryo['embryo_id'])
            properties['confidence'].append(embryo['confidence'])

        viewer.add_shapes(
            shapes,
            shape_type='rectangle',
            edge_color='lime',
            edge_width=2,
            face_color='transparent',
            name='Detections',
            properties=properties,
            text='detection_id'
        )

        print("\n👁️  Napari viewer opened - review green boxes")
        print("   Close viewer or return to CLI to continue\n")

        if block:
            napari.run()
        else:
            viewer.show(block=False)

        return viewer
