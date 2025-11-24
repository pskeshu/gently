#!/usr/bin/env python3
"""
Hybrid SAM + Claude Vision Embryo Detection
============================================

Combines Segment Anything Model with Claude Vision API:
1. SAM generates initial segmentation masks
2. Claude Vision reviews detections and identifies:
   - False positives (artifacts to remove)
   - False negatives (missed embryos to detect)
3. Script corrects based on Claude's feedback

Usage:
    python test_sam_claude_hybrid_detection.py
"""

import time
import json
import numpy as np
from pathlib import Path
from client import get_mmc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import base64
from io import BytesIO
from PIL import Image
import anthropic
import os

# Import SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Device configuration
core = get_mmc()
CAMERA_NAME = "Bottom PCO"
CAMERA_EXPOSURE_MS = 50.0

# SAM configuration
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# Detection parameters
MAX_EMBRYOS = 20
MIN_EMBRYO_AREA = 2000
MAX_EMBRYO_AREA = 15000
MIN_CIRCULARITY = 0.4
MIN_SEPARATION_PIXELS = 100

# Claude API configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY not set. Claude review will be skipped.")


def load_sam_model(checkpoint_path: str = SAM_CHECKPOINT,
                   model_type: str = SAM_MODEL_TYPE,
                   device: str = "cpu"):
    """Load SAM model for segmentation."""
    print(f"\nLoading SAM model: {model_type}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    # Mask generator for automatic segmentation
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.80,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        box_nms_thresh=0.7,
    )

    # Predictor for targeted segmentation (for false negatives)
    predictor = SamPredictor(sam)

    print("  ✓ SAM model loaded successfully")
    return mask_generator, predictor


def configure_bottom_camera():
    """Configure bottom camera for snapshot."""
    print(f"\nConfiguring camera: {CAMERA_NAME}")
    core.setCameraDevice(CAMERA_NAME)
    core.setExposure(CAMERA_NAME, CAMERA_EXPOSURE_MS)

    try:
        core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
    except:
        pass

    time.sleep(0.1)
    print(f"  ✓ Camera configured (exposure: {CAMERA_EXPOSURE_MS} ms)")


def capture_bottom_camera_image():
    """Capture a single image from the bottom camera."""
    import rpyc

    print("\nCapturing image from bottom camera...")
    core.snapImage()
    img = core.getImage()

    try:
        img = rpyc.classic.obtain(img)
    except (ImportError, AttributeError):
        pass

    print(f"  ✓ Captured image: {img.shape}, dtype: {img.dtype}")
    return img


def get_stage_position():
    """Get current XY stage position."""
    try:
        xy_stage = core.getXYStageDevice()
        x = core.getXPosition(xy_stage)
        y = core.getYPosition(xy_stage)
        return (x, y)
    except Exception as e:
        print(f"  Warning: Could not get stage position: {e}")
        return (0.0, 0.0)


def detect_embryos_with_sam(image: np.ndarray,
                             mask_generator: SamAutomaticMaskGenerator,
                             max_embryos: int = MAX_EMBRYOS,
                             min_area: int = MIN_EMBRYO_AREA,
                             max_area: int = MAX_EMBRYO_AREA,
                             min_circularity: float = MIN_CIRCULARITY,
                             min_separation_pixels: int = MIN_SEPARATION_PIXELS):
    """Detect embryos using SAM automatic mask generation."""
    print("\nRunning SAM segmentation...")

    # Convert to 8-bit if needed
    if image.dtype == np.uint16:
        print(f"  Converting uint16 image to uint8...")
        image_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        image_8bit = image.astype(np.uint8)

    # Convert grayscale to RGB for SAM
    if len(image_8bit.shape) == 2:
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_8bit

    print(f"  Image prepared for SAM: shape={image_rgb.shape}, dtype={image_rgb.dtype}")

    # Generate masks
    masks = mask_generator.generate(image_rgb)
    print(f"  ✓ SAM generated {len(masks)} total masks")

    if len(masks) > 0:
        areas = [m['area'] for m in masks]
        print(f"  ℹ Mask areas: min={min(areas)}, max={max(areas)}, median={np.median(areas):.0f}")

    # Filter masks
    embryo_candidates = []
    filtered_by_size = 0
    filtered_by_circularity = 0

    for mask_data in masks:
        area = mask_data['area']

        if not (min_area <= area <= max_area):
            filtered_by_size += 1
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

        if circularity < min_circularity:
            filtered_by_circularity += 1
            continue

        embryo_candidates.append({
            'mask': mask,
            'bbox': bbox,
            'area': area,
            'circularity': circularity,
            'stability_score': mask_data['stability_score'],
            'predicted_iou': mask_data['predicted_iou']
        })

    print(f"  ✓ Found {len(embryo_candidates)} embryo candidates after filtering")
    print(f"    (filtered: {filtered_by_size} by size, {filtered_by_circularity} by circularity)")

    # Sort by quality
    embryo_candidates.sort(key=lambda x: (x['area'] * x['stability_score']), reverse=True)

    # Apply spatial separation filtering
    print(f"  Applying spatial separation filter (min {min_separation_pixels}px)...")
    selected_embryos = []

    for candidate in embryo_candidates:
        if len(selected_embryos) >= max_embryos:
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

            if distance < min_separation_pixels:
                too_close = True
                break

        if not too_close:
            selected_embryos.append(candidate)

    print(f"  ✓ Selected {len(selected_embryos)} well-separated embryos")

    filtered_out = len(embryo_candidates) - len(selected_embryos)
    if filtered_out > 0:
        print(f"  ℹ Filtered out {filtered_out} detections (too close or exceeded limit)")

    return selected_embryos, image_8bit


def create_annotated_image(image: np.ndarray, embryos: list) -> np.ndarray:
    """Create image with numbered embryo annotations for Claude review."""
    viz = image.copy()
    if len(viz.shape) == 2:
        viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)]

    for i, embryo in enumerate(embryos):
        bbox = embryo['bbox']
        x, y, w, h = bbox
        color = colors[i % len(colors)]

        # Draw bounding box
        cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)

        # Draw label with number
        label = f"{i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(viz, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(viz, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)

        # Draw center point
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        cv2.circle(viz, (center_x, center_y), 5, color, -1)

    return viz


def encode_image_to_base64(image: np.ndarray, max_size_mb: float = 4.8) -> str:
    """Encode image to base64 for Claude API with size limit."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(image)

    # Try resizing first to maintain quality
    # 2048x2048 is often too large, resize to ~1400x1400
    if pil_image.width > 1500 or pil_image.height > 1500:
        scale = 1400 / max(pil_image.width, pil_image.height)
        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"  ℹ Resized image to: {new_size[0]}x{new_size[1]}")

    # Start with high quality
    quality = 92
    max_bytes = int(max_size_mb * 1024 * 1024)

    while quality > 30:
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=quality, optimize=True)
        size_bytes = buffered.tell()

        if size_bytes <= max_bytes:
            print(f"  ✓ Image encoded: {size_bytes / 1024 / 1024:.2f} MB (quality={quality})")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        quality -= 5

    # Last resort - more aggressive resize
    print(f"  ⚠ Still too large, reducing to 1000x1000...")
    scale = 1000 / max(pil_image.width, pil_image.height)
    new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
    size_bytes = buffered.tell()
    print(f"  ✓ Final image: {size_bytes / 1024 / 1024:.2f} MB at {new_size[0]}x{new_size[1]}")

    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def verify_corrections_with_claude(image: np.ndarray,
                                   annotated_image: np.ndarray,
                                   embryos: list,
                                   previous_changes: dict) -> dict:
    """Second round: Claude verifies the corrections look good."""
    if not ANTHROPIC_API_KEY:
        print("\n⚠ Skipping verification (API key not set)")
        return {'additional_false_positives': [], 'additional_false_negatives': [], 'verified': True}

    print("\n[Round 2 - Verification] Checking corrected detections...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    image_base64 = encode_image_to_base64(annotated_image)

    removed = previous_changes.get('removed', [])
    added = previous_changes.get('added', [])

    prompt = f"""VERIFICATION ROUND - You previously reviewed this image and made corrections.

PREVIOUS CHANGES:
- Removed detections: {removed if removed else "none"}
- Added embryos at: {added if added else "none"}

CURRENT STATE: {len(embryos)} detected embryos (numbered 0-{len(embryos)-1})

YOUR TASK: Verify the corrections were correct and catch any remaining issues.

1. Check that removed false positives are truly gone
2. Check that added embryos were segmented correctly
3. Look for any NEW issues you might have missed in round 1

IMPORTANT: Be thorough but only report CLEAR remaining issues. If corrections look good, return empty lists.

Respond in JSON:
{{
  "additional_false_positives": [detection numbers that are STILL wrong],
  "additional_false_negatives": [
    {{"x": pixel_x, "y": pixel_y, "description": "still missed embryo"}}
  ],
  "verified": true/false,
  "verification_summary": "Assessment of corrections and any remaining issues"
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=6000,
            thinking={
                "type": "enabled",
                "budget_tokens": 4000
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Extract response
        thinking_text = None
        response_text = None

        for block in message.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                response_text = block.text

        if thinking_text:
            print(f"\n[Claude's Verification Thinking]:")
            print(thinking_text[:400] + "..." if len(thinking_text) > 400 else thinking_text)

        print(f"\n[Claude Verification]:\n{response_text}")

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        verification = json.loads(json_str)
        return verification

    except Exception as e:
        print(f"\n⚠ Verification failed: {e}")
        return {'additional_false_positives': [], 'additional_false_negatives': [], 'verified': False}


def review_with_claude(image: np.ndarray, annotated_image: np.ndarray, embryos: list) -> dict:
    """Send detection to Claude for review."""
    if not ANTHROPIC_API_KEY:
        print("\n⚠ Skipping Claude review (API key not set)")
        return {'false_positives': [], 'false_negatives': []}

    print("\n[Claude Review] Analyzing detections...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Encode annotated image
    image_base64 = encode_image_to_base64(annotated_image)

    # Create prompt
    prompt = f"""You are a microscopy expert analyzing embryo detections from a bottom camera view (2048x2048 pixels).

CURRENT DETECTIONS: {len(embryos)} embryos labeled 0-{len(embryos)-1} with colored bounding boxes.

EMBRYO CHARACTERISTICS:
- Small, BRIGHT white/light gray oval or rice grain shapes
- Typical size: 40-120 pixels in diameter
- Stand out clearly against the dark gray background
- Have defined boundaries and smooth edges
- Located in the field of view, not at edges

YOUR TASK:
1. Scan the ENTIRE image systematically (top-to-bottom, left-to-right)
2. Identify ALL bright oval objects, including those WITHOUT boxes
3. For each numbered detection, verify it matches embryo characteristics

FALSE POSITIVES (remove these):
- Edge artifacts or corner detections
- Very elongated or irregular shapes (circularity < 0.4)
- Dark objects or low-contrast regions
- Image defects or debris

FALSE NEGATIVES (add these):
- ANY bright oval object WITHOUT a colored box around it
- Pay special attention to the CENTER of the image
- Look for isolated bright spots that look like other detected embryos

IMPORTANT: The image center (around pixel 1024, 1024) often has missed detections. Check carefully!

Respond in JSON format:
{{
  "false_positives": [list of detection numbers to remove, e.g., [0, 3]],
  "false_negatives": [
    {{"x": 1024, "y": 512, "description": "bright oval embryo in upper center"}}
  ],
  "analysis": "systematic check: found X embryos total, Y with boxes, Z missed",
  "summary": "Added X missed embryos, removed Y false positives"
}}

Count ALL bright oval objects you see, then compare to the number of boxes."""

    try:
        # Use Sonnet 4.5 with extended thinking for careful visual analysis
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            thinking={
                "type": "enabled",
                "budget_tokens": 5000
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Extract thinking and response
        thinking_text = None
        response_text = None

        for block in message.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                response_text = block.text

        if thinking_text:
            print(f"\n[Claude's Thinking Process]:")
            print(thinking_text[:500] + "..." if len(thinking_text) > 500 else thinking_text)

        print(f"\n[Claude Response]:\n{response_text}")

        # Parse JSON response
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        review = json.loads(json_str)
        return review

    except Exception as e:
        print(f"\n⚠ Claude review failed: {e}")
        return {'false_positives': [], 'false_negatives': []}


def segment_region_with_sam(image: np.ndarray,
                            predictor: SamPredictor,
                            point: tuple,
                            crop_size: int = 200) -> dict:
    """Use SAM predictor to segment a specific region around a point."""
    print(f"  Segmenting region around ({point[0]}, {point[1]})...")

    # Set image for predictor
    if image.dtype == np.uint16:
        image_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        image_8bit = image.astype(np.uint8)

    if len(image_8bit.shape) == 2:
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_8bit

    predictor.set_image(image_rgb)

    # Use point prompt
    point_coords = np.array([[point[0], point[1]]])
    point_labels = np.array([1])  # 1 = foreground point

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    # Take best mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    # Calculate bbox
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Calculate area and circularity
    area = mask.sum()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        perimeter = cv2.arcLength(contours[0], True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    else:
        circularity = 0

    print(f"    ✓ Found mask: area={area}, circularity={circularity:.3f}, score={scores[best_idx]:.3f}")

    return {
        'mask': mask,
        'bbox': bbox,
        'area': int(area),
        'circularity': float(circularity),
        'stability_score': float(scores[best_idx]),
        'predicted_iou': float(scores[best_idx])
    }


def apply_claude_corrections(embryos: list,
                             review: dict,
                             image: np.ndarray,
                             predictor: SamPredictor) -> tuple:
    """Apply Claude's corrections to embryo list. Returns (corrected_embryos, changes_made)."""
    corrected_embryos = []
    changes_made = {'removed': [], 'added': []}

    # Remove false positives
    false_positives = set(review.get('false_positives', []))
    if false_positives:
        print(f"\n[Corrections] Removing {len(false_positives)} false positives: {false_positives}")
        changes_made['removed'] = list(false_positives)

    for i, embryo in enumerate(embryos):
        if i not in false_positives:
            corrected_embryos.append(embryo)

    # Add false negatives
    false_negatives = review.get('false_negatives', [])
    if false_negatives:
        print(f"\n[Corrections] Adding {len(false_negatives)} missed embryos...")

        for fn in false_negatives:
            point = (fn['x'], fn['y'])
            new_embryo = segment_region_with_sam(image, predictor, point)

            if new_embryo:
                # Check if it meets criteria
                if (MIN_EMBRYO_AREA <= new_embryo['area'] <= MAX_EMBRYO_AREA and
                    new_embryo['circularity'] >= MIN_CIRCULARITY):
                    corrected_embryos.append(new_embryo)
                    changes_made['added'].append(point)
                    print(f"    ✓ Added embryo at {point}")
                else:
                    print(f"    ✗ Rejected: area={new_embryo['area']}, circ={new_embryo['circularity']:.3f}")

    print(f"\n  Round complete: {len(embryos)} → {len(corrected_embryos)} embryos")
    return corrected_embryos, changes_made


def pixel_to_stage_coordinates(embryos, stage_pos, pixel_size_um=6.5, objective_mag=4.0):
    """Convert embryo pixel coordinates to stage coordinates."""
    effective_pixel_um = pixel_size_um / objective_mag
    stage_x, stage_y = stage_pos

    embryo_positions = []
    for i, embryo in enumerate(embryos):
        bbox = embryo['bbox']
        x, y, w, h = bbox

        center_x_px = x + w / 2
        center_y_px = y + h / 2

        embryo_stage_x = stage_x + (center_x_px * effective_pixel_um)
        embryo_stage_y = stage_y + (center_y_px * effective_pixel_um)

        embryo_positions.append({
            'embryo_id': i,
            'stage_x_um': embryo_stage_x,
            'stage_y_um': embryo_stage_y,
            'bbox_pixel': bbox,
            'center_pixel': (center_x_px, center_y_px),
            'area_pixels': embryo['area'],
            'circularity': embryo['circularity'],
            'stability_score': embryo['stability_score']
        })

    return embryo_positions


def main():
    """Main workflow for hybrid SAM + Claude detection."""
    print("="*70)
    print("HYBRID SAM + CLAUDE EMBRYO DETECTION")
    print("="*70)

    try:
        # Load SAM model
        print("\n[1/8] Loading SAM model...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        mask_generator, predictor = load_sam_model(
            checkpoint_path=SAM_CHECKPOINT,
            model_type=SAM_MODEL_TYPE,
            device=device
        )

        # Configure camera
        print("\n[2/8] Configuring bottom camera...")
        configure_bottom_camera()

        # Get stage position
        print("\n[3/8] Reading stage position...")
        stage_pos = get_stage_position()
        print(f"  Current stage: X={stage_pos[0]:.2f} µm, Y={stage_pos[1]:.2f} µm")

        # Capture image
        print("\n[4/8] Capturing image...")
        image = capture_bottom_camera_image()

        # Initial SAM detection
        print("\n[5/8] Initial SAM detection...")
        embryos, image_8bit = detect_embryos_with_sam(
            image,
            mask_generator,
            max_embryos=MAX_EMBRYOS,
            min_area=MIN_EMBRYO_AREA,
            max_area=MAX_EMBRYO_AREA,
            min_circularity=MIN_CIRCULARITY,
            min_separation_pixels=MIN_SEPARATION_PIXELS
        )

        if len(embryos) == 0:
            print("\n⚠ No embryos detected by SAM!")
            return

        print(f"\n  SAM detected {len(embryos)} embryos")

        # Create annotated image
        print("\n[6/8] Creating annotated image for Claude review...")
        annotated_image = create_annotated_image(image_8bit, embryos)
        cv2.imwrite('sam_claude_initial_detection.png', annotated_image)
        print(f"  ✓ Saved initial detection: sam_claude_initial_detection.png")

        # === ROUND 1: Initial Review ===
        print("\n[7/10] ROUND 1: Claude initial review...")
        review_round1 = review_with_claude(image_8bit, annotated_image, embryos)

        if review_round1.get('summary'):
            print(f"\n  Round 1 summary: {review_round1['summary']}")

        # Apply round 1 corrections
        print("\n[8/10] Applying Round 1 corrections...")
        embryos_round1, changes_round1 = apply_claude_corrections(embryos, review_round1, image, predictor)

        # Save round 1 result
        round1_viz = create_annotated_image(image_8bit, embryos_round1)
        cv2.imwrite('sam_claude_round1_corrected.png', round1_viz)
        print(f"  ✓ Saved Round 1 corrections: sam_claude_round1_corrected.png")

        # === ROUND 2: Verification ===
        print("\n[9/10] ROUND 2: Claude verification...")
        verification = verify_corrections_with_claude(image_8bit, round1_viz, embryos_round1, changes_round1)

        if verification.get('verification_summary'):
            print(f"\n  Verification: {verification['verification_summary']}")

        # Apply round 2 corrections if needed
        print("\n[10/10] Final adjustments...")
        has_additional_changes = (
            len(verification.get('additional_false_positives', [])) > 0 or
            len(verification.get('additional_false_negatives', [])) > 0
        )

        if has_additional_changes:
            print("  Applying Round 2 corrections...")
            review_round2 = {
                'false_positives': verification.get('additional_false_positives', []),
                'false_negatives': verification.get('additional_false_negatives', [])
            }
            final_embryos, changes_round2 = apply_claude_corrections(embryos_round1, review_round2, image, predictor)
        else:
            print("  ✓ No additional corrections needed - Round 1 was good!")
            final_embryos = embryos_round1

        # Convert to stage coordinates
        embryo_positions = pixel_to_stage_coordinates(final_embryos, stage_pos)

        # Display results
        print("\n" + "="*70)
        print("FINAL RESULTS (After Claude Review)")
        print("="*70)

        for embryo in embryo_positions:
            print(f"\nEmbryo {embryo['embryo_id']}:")
            print(f"  Stage: ({embryo['stage_x_um']:.2f}, {embryo['stage_y_um']:.2f}) µm")
            print(f"  Area: {embryo['area_pixels']} pixels")
            print(f"  Circularity: {embryo['circularity']:.3f}")

        # Save results
        output_file = Path("sam_claude_hybrid_results.json")
        results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'Hybrid SAM + Claude Vision (2-Round Iterative)',
            'initial_detections': len(embryos),
            'round1_detections': len(embryos_round1),
            'final_detections': len(final_embryos),
            'claude_round1': review_round1,
            'claude_verification': verification,
            'changes_round1': changes_round1,
            'embryos': [{k: v for k, v in e.items() if k != 'mask'} for e in embryo_positions]
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to: {output_file}")

        # Create final visualization
        final_viz = create_annotated_image(image_8bit, final_embryos)
        cv2.imwrite('sam_claude_final_detection.png', final_viz)
        print(f"✓ Saved final detection: sam_claude_final_detection.png")

        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
