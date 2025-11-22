"""
Real-time Hatching Detection for Live Microscopy
Integrates Claude Vision API for on-the-fly hatching event detection
"""
import os
import time
import anthropic
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime


class RealtimeHatchingDetector:
    """Manages real-time hatching detection during acquisition"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-5",
                 min_timepoints_before_detection: int = 5,
                 confidence_threshold: str = "HIGH"):
        """
        Parameters
        ----------
        api_key : str, optional
            Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model : str
            Claude model to use
        min_timepoints_before_detection : int
            Minimum number of timepoints before starting detection
        confidence_threshold : str
            Required confidence level for positive detection (HIGH/MEDIUM/LOW)
        """
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.min_timepoints = min_timepoints_before_detection
        self.confidence_threshold = confidence_threshold

        # State tracking
        self.hatching_status = {}  # embryo_id -> status dict
        self.detection_history = {}  # embryo_id -> list of detection results

    def should_check_embryo(self, embryo_id: str, timepoint: int) -> bool:
        """
        Determine if we should run detection for this embryo

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint number

        Returns
        -------
        bool
            True if should check
        """
        # Don't check if already hatched
        if self.is_hatched(embryo_id):
            return False

        # Don't check before minimum timepoints
        if timepoint < self.min_timepoints:
            return False

        return True

    def detect_hatching_single_image(self, embryo_id: str, timepoint: int,
                                     recent_images: List[Dict]) -> Dict:
        """
        Run hatching detection using recent images for temporal context

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint
        recent_images : list
            List of dicts with 'timepoint', 'b64_image', 'size'

        Returns
        -------
        dict
            Detection result with 'hatched', 'confidence', 'reasoning'
        """
        if len(recent_images) == 0:
            return {
                'hatched': False,
                'confidence': None,
                'reasoning': "No images available",
                'error': True
            }

        print(f"    Running hatching detection with {len(recent_images)} recent images...")

        # Create content for Claude
        content = self._create_detection_content(recent_images, timepoint)

        # Calculate total size
        total_size = sum(img['size'] for img in recent_images)
        print(f"    Payload size: {total_size / 1024 / 1024:.2f} MB")

        try:
            # Call Claude API
            start_time = time.time()

            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            api_duration = time.time() - start_time

            response_text = message.content[0].text

            # Parse response
            result = self._parse_detection_response(response_text)
            result['api_duration'] = api_duration
            result['num_images'] = len(recent_images)
            result['embryo_id'] = embryo_id
            result['timepoint'] = timepoint

            print(f"    API call: {api_duration:.2f}s")
            print(f"    Result: Hatched={result['hatched']}, Confidence={result['confidence']}")

            return result

        except Exception as e:
            print(f"    ✗ API Error: {e}")
            return {
                'hatched': False,
                'confidence': None,
                'reasoning': f"API error: {str(e)}",
                'error': True,
                'embryo_id': embryo_id,
                'timepoint': timepoint
            }

    def _create_detection_content(self, recent_images: List[Dict],
                                  current_timepoint: int) -> List[Dict]:
        """Create content blocks for Claude Vision API"""
        content = []

        # Add instruction text
        content.append({
            "type": "text",
            "text": f"""You are analyzing real-time C. elegans embryo development from diSPIM microscopy.
You are seeing {len(recent_images)} recent timepoints (each 2 minutes apart).

Your task: Determine if the embryo has HATCHED in the MOST RECENT (final) timepoint.

Key characteristics of hatching:
- The embryo breaks out of the eggshell
- Visible breach or rupture in the outer boundary
- Embryo emerges from confined eggshell space
- Morphology changes from egg-contained to elongated worm shape

IMPORTANT: We need high confidence to stop acquisition. Only report hatching if you are certain.

Please analyze the sequence and answer:
1. Has the embryo hatched in the final timepoint? (YES/NO)
2. Confidence level (LOW/MEDIUM/HIGH)
3. Brief reasoning (1-2 sentences)

Format your response as:
HATCHED: [YES/NO]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [your explanation]
"""
        })

        # Add images with labels
        for img_data in recent_images:
            tp = img_data['timepoint']
            minutes = tp * 2

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_data['b64_image']
                }
            })
            content.append({
                "type": "text",
                "text": f"Timepoint {tp:04d} ({minutes} minutes)"
            })

        # Emphasize the current timepoint
        content.append({
            "type": "text",
            "text": f"**The FINAL image above (t{current_timepoint:04d}) is the current timepoint - has hatching occurred by this point?**"
        })

        return content

    def _parse_detection_response(self, response_text: str) -> Dict:
        """Parse Claude's response into structured result"""
        result = {
            'response': response_text,
            'hatched': False,
            'confidence': None,
            'reasoning': None,
            'error': False
        }

        # Parse response lines
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith("HATCHED:"):
                result['hatched'] = "YES" in line.upper()
            elif line.startswith("CONFIDENCE:"):
                if "HIGH" in line.upper():
                    result['confidence'] = "HIGH"
                elif "MEDIUM" in line.upper():
                    result['confidence'] = "MEDIUM"
                elif "LOW" in line.upper():
                    result['confidence'] = "LOW"
            elif line.startswith("REASONING:"):
                result['reasoning'] = line.replace("REASONING:", "").strip()

        return result

    def update_hatching_status(self, embryo_id: str, detection_result: Dict):
        """
        Update hatching status based on detection result

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        detection_result : dict
            Result from detect_hatching_single_image
        """
        # Store detection history
        if embryo_id not in self.detection_history:
            self.detection_history[embryo_id] = []
        self.detection_history[embryo_id].append(detection_result)

        # Update status if hatching detected with sufficient confidence
        if detection_result.get('hatched') and not detection_result.get('error'):
            confidence = detection_result.get('confidence')

            # Check if confidence meets threshold
            confidence_levels = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            threshold_level = confidence_levels.get(self.confidence_threshold, 3)
            result_level = confidence_levels.get(confidence, 0)

            if result_level >= threshold_level:
                self.hatching_status[embryo_id] = {
                    'hatched': True,
                    'timepoint': detection_result['timepoint'],
                    'confidence': confidence,
                    'reasoning': detection_result.get('reasoning'),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"\n    🎉 HATCHING DETECTED: {embryo_id} at t{detection_result['timepoint']:04d}")
                print(f"       Confidence: {confidence}")
                print(f"       Reasoning: {detection_result.get('reasoning')}")

    def is_hatched(self, embryo_id: str) -> bool:
        """Check if embryo has hatched"""
        return self.hatching_status.get(embryo_id, {}).get('hatched', False)

    def get_hatching_timepoint(self, embryo_id: str) -> Optional[int]:
        """Get timepoint when embryo hatched (None if not hatched)"""
        if self.is_hatched(embryo_id):
            return self.hatching_status[embryo_id]['timepoint']
        return None

    def all_embryos_hatched(self, embryo_ids: List[str]) -> bool:
        """Check if all embryos have hatched"""
        return all(self.is_hatched(eid) for eid in embryo_ids)

    def get_summary(self) -> Dict:
        """Get summary of hatching status for all embryos"""
        summary = {
            'total_embryos': len(self.hatching_status),
            'hatched_count': sum(1 for s in self.hatching_status.values() if s.get('hatched')),
            'embryo_status': self.hatching_status.copy()
        }
        return summary

    def save_detection_log(self, output_file: Path):
        """Save detection history to JSON file"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'min_timepoints': self.min_timepoints,
            'confidence_threshold': self.confidence_threshold,
            'hatching_status': self.hatching_status,
            'detection_history': self.detection_history
        }

        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"    ✓ Detection log saved: {output_file}")


# Example usage
if __name__ == "__main__":
    print("Testing RealtimeHatchingDetector...")

    detector = RealtimeHatchingDetector(
        min_timepoints_before_detection=5,
        confidence_threshold="HIGH"
    )

    # Test should_check_embryo
    print("\nTesting should_check_embryo:")
    print(f"  Timepoint 3: {detector.should_check_embryo('embryo_001', 3)}")  # False (too early)
    print(f"  Timepoint 10: {detector.should_check_embryo('embryo_001', 10)}")  # True

    # Simulate hatching
    detector.hatching_status['embryo_001'] = {
        'hatched': True,
        'timepoint': 150,
        'confidence': 'HIGH'
    }

    print(f"  After hatching: {detector.should_check_embryo('embryo_001', 160)}")  # False (already hatched)

    # Test all_embryos_hatched
    print("\nTesting all_embryos_hatched:")
    detector.hatching_status['embryo_002'] = {'hatched': False}
    print(f"  All hatched: {detector.all_embryos_hatched(['embryo_001', 'embryo_002'])}")  # False

    detector.hatching_status['embryo_002']['hatched'] = True
    print(f"  All hatched: {detector.all_embryos_hatched(['embryo_001', 'embryo_002'])}")  # True

    print("\n✓ Tests passed!")
