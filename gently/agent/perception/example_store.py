"""
Few-shot example storage for stage classification.

Manages reference images organized by developmental stage and anomaly type.
Examples are loaded from disk and base64-encoded for inclusion in VLM prompts.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExampleStore:
    """
    Manages few-shot example images for stage classification and anomaly detection.

    Directory structure:
    examples/
        stages/
            early/
                example_001.jpg
            comma/
                example_001.jpg
            pretzel/
                example_001.jpg
            3fold/
                example_001.jpg
            hatching/
                example_001.jpg
            hatched/
                example_001.jpg
        anomalies/
            dead_embryo/
                example_001.jpg
            blank_technical/
                example_001.jpg
            blank_biological/
                example_001.jpg
        metadata.json
    """

    # Supported developmental stages
    STAGES = ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]

    # Supported anomaly types
    ANOMALY_TYPES = ["dead_embryo", "blank_technical", "blank_biological"]

    def __init__(self, examples_path: Path):
        """
        Parameters
        ----------
        examples_path : Path
            Root directory containing example images
        """
        self.examples_path = Path(examples_path)
        self.stages_path = self.examples_path / "stages"
        self.anomalies_path = self.examples_path / "anomalies"

        # Cache loaded examples to avoid repeated disk I/O
        self._stage_cache: Dict[str, List[str]] = {}  # stage -> list of b64 images
        self._anomaly_cache: Dict[str, List[str]] = {}

        # Load metadata if exists
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load optional metadata.json with annotations"""
        metadata_path = self.examples_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save metadata.json"""
        metadata_path = self.examples_path / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def get_stage_examples(
        self,
        stage: str,
        max_examples: int = 3,
    ) -> List[str]:
        """
        Get example images for a developmental stage.

        Parameters
        ----------
        stage : str
            Stage name (early, comma, pretzel, 3fold, hatching, hatched)
        max_examples : int
            Maximum number of examples to return

        Returns
        -------
        List[str]
            Base64-encoded example images (JPEG format)
        """
        if stage not in self._stage_cache:
            self._load_stage_examples(stage)

        return self._stage_cache.get(stage, [])[:max_examples]

    def _load_stage_examples(self, stage: str) -> None:
        """Load and cache examples for a stage"""
        stage_dir = self.stages_path / stage

        if not stage_dir.exists():
            logger.debug(f"No examples directory for stage '{stage}'")
            self._stage_cache[stage] = []
            return

        examples = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]

        for pattern in image_patterns:
            for img_path in sorted(stage_dir.glob(pattern)):
                b64 = self._load_and_encode_image(img_path)
                if b64:
                    examples.append(b64)
                    logger.debug(f"Loaded example: {img_path.name}")

        self._stage_cache[stage] = examples
        logger.info(f"Loaded {len(examples)} examples for stage '{stage}'")

    def get_anomaly_examples(
        self,
        anomaly_type: str,
        max_examples: int = 2,
    ) -> List[str]:
        """
        Get example images for an anomaly type.

        Parameters
        ----------
        anomaly_type : str
            Anomaly type (dead_embryo, blank_technical, blank_biological)
        max_examples : int
            Maximum examples to return

        Returns
        -------
        List[str]
            Base64-encoded example images
        """
        if anomaly_type not in self._anomaly_cache:
            self._load_anomaly_examples(anomaly_type)

        return self._anomaly_cache.get(anomaly_type, [])[:max_examples]

    def _load_anomaly_examples(self, anomaly_type: str) -> None:
        """Load and cache anomaly examples"""
        anomaly_dir = self.anomalies_path / anomaly_type

        if not anomaly_dir.exists():
            logger.debug(f"No examples directory for anomaly '{anomaly_type}'")
            self._anomaly_cache[anomaly_type] = []
            return

        examples = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]

        for pattern in image_patterns:
            for img_path in sorted(anomaly_dir.glob(pattern)):
                b64 = self._load_and_encode_image(img_path)
                if b64:
                    examples.append(b64)

        self._anomaly_cache[anomaly_type] = examples
        logger.info(f"Loaded {len(examples)} examples for anomaly '{anomaly_type}'")

    def _load_and_encode_image(self, path: Path) -> Optional[str]:
        """
        Load image and encode as base64 JPEG.

        Images are resized to max 800px and converted to JPEG for
        efficient transmission to the VLM API.
        """
        try:
            from PIL import Image

            img = Image.open(path)

            # Handle different image modes
            if img.mode in ("RGBA", "LA", "P"):
                # Convert transparent images to RGB with white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if too large (preserving aspect ratio)
            max_dim = 800
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)

            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            logger.warning(f"Failed to load example image {path}: {e}")
            return None

    def list_available_stages(self) -> List[str]:
        """List all stages with at least one example"""
        available = []
        for stage in self.STAGES:
            examples = self.get_stage_examples(stage, max_examples=1)
            if examples:
                available.append(stage)
        return available

    def list_available_anomalies(self) -> List[str]:
        """List all anomaly types with at least one example"""
        available = []
        for anomaly_type in self.ANOMALY_TYPES:
            examples = self.get_anomaly_examples(anomaly_type, max_examples=1)
            if examples:
                available.append(anomaly_type)
        return available

    def get_example_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get counts of examples per category.

        Returns
        -------
        Dict with structure:
            {
                "stages": {"early": 3, "comma": 2, ...},
                "anomalies": {"dead_embryo": 1, ...}
            }
        """
        counts = {
            "stages": {},
            "anomalies": {},
        }

        for stage in self.STAGES:
            examples = self.get_stage_examples(stage, max_examples=100)
            counts["stages"][stage] = len(examples)

        for anomaly_type in self.ANOMALY_TYPES:
            examples = self.get_anomaly_examples(anomaly_type, max_examples=100)
            counts["anomalies"][anomaly_type] = len(examples)

        return counts

    def add_example(
        self,
        image_b64: str,
        category: str,  # "stages" or "anomalies"
        subcategory: str,  # stage name or anomaly type
        annotation: Optional[str] = None,
    ) -> str:
        """
        Add a new example image.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded image (JPEG or PNG)
        category : str
            "stages" or "anomalies"
        subcategory : str
            Stage name (early, comma, etc.) or anomaly type
        annotation : str, optional
            Optional annotation for metadata

        Returns
        -------
        str
            Path where image was saved
        """
        # Determine directory
        if category == "stages":
            if subcategory not in self.STAGES:
                raise ValueError(f"Unknown stage: {subcategory}. Valid: {self.STAGES}")
            target_dir = self.stages_path / subcategory
        elif category == "anomalies":
            if subcategory not in self.ANOMALY_TYPES:
                raise ValueError(f"Unknown anomaly type: {subcategory}. Valid: {self.ANOMALY_TYPES}")
            target_dir = self.anomalies_path / subcategory
        else:
            raise ValueError(f"Unknown category: {category}. Use 'stages' or 'anomalies'")

        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        existing = list(target_dir.glob("example_*.jpg"))
        next_num = len(existing) + 1
        filename = f"example_{next_num:03d}.jpg"
        filepath = target_dir / filename

        # Decode and save
        try:
            img_bytes = base64.b64decode(image_b64)
            with open(filepath, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            raise ValueError(f"Failed to save image: {e}")

        # Clear cache for this category
        if category == "stages":
            self._stage_cache.pop(subcategory, None)
        else:
            self._anomaly_cache.pop(subcategory, None)

        # Add to metadata if annotation provided
        if annotation:
            if subcategory not in self.metadata:
                self.metadata[subcategory] = {}
            self.metadata[subcategory][filename] = annotation
            self._save_metadata()

        logger.info(f"Added example: {filepath}")
        return str(filepath)

    def remove_example(
        self,
        category: str,
        subcategory: str,
        filename: str,
    ) -> bool:
        """
        Remove an example image.

        Parameters
        ----------
        category : str
            "stages" or "anomalies"
        subcategory : str
            Stage name or anomaly type
        filename : str
            Name of the file to remove

        Returns
        -------
        bool
            True if removed, False if not found
        """
        if category == "stages":
            target_dir = self.stages_path / subcategory
        else:
            target_dir = self.anomalies_path / subcategory

        filepath = target_dir / filename

        if filepath.exists():
            filepath.unlink()

            # Clear cache
            if category == "stages":
                self._stage_cache.pop(subcategory, None)
            else:
                self._anomaly_cache.pop(subcategory, None)

            # Remove from metadata
            if subcategory in self.metadata and filename in self.metadata[subcategory]:
                del self.metadata[subcategory][filename]
                self._save_metadata()

            logger.info(f"Removed example: {filepath}")
            return True

        return False

    def clear_cache(self) -> None:
        """Clear all cached examples (force reload from disk)"""
        self._stage_cache.clear()
        self._anomaly_cache.clear()
        logger.info("Cleared example cache")
