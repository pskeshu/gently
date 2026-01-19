"""
Few-shot example storage for stage classification.

Manages reference images organized by developmental stage and anomaly type.
Examples are loaded from disk and base64-encoded for inclusion in VLM prompts.

Also supports loading 3D volumes for reference examples when available.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stages import STAGES

logger = logging.getLogger(__name__)


class ExampleStore:
    """
    Manages few-shot example images for stage classification and anomaly detection.

    Directory structure:
    examples/
        stages/
            early/
            bean/
            comma/
            1.5fold/
            2fold/
            3fold/
            hatching/
            hatched/
        anomalies/
            dead_embryo/
            blank_technical/
            blank_biological/
        metadata.json

    Stage definitions are imported from stages.py (single source of truth).
    """

    # Supported developmental stages (imported from stages.py)
    STAGES = STAGES  # Re-export for backwards compatibility

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
        self._stage_metadata_cache: Dict[str, Dict] = {}  # stage -> metadata dict
        self._volume_cache: Dict[str, Optional[np.ndarray]] = {}  # stage -> volume

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

        # Prefer three_view.jpg if available (new three-view format)
        three_view_path = stage_dir / "three_view.jpg"
        if three_view_path.exists():
            b64 = self._load_and_encode_image(three_view_path)
            if b64:
                examples.append(b64)
                logger.debug(f"Loaded three_view example for {stage}")

        # Fallback to legacy patterns if no three_view
        if not examples:
            image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]

            for pattern in image_patterns:
                for img_path in sorted(stage_dir.glob(pattern)):
                    # Skip three_view files (already handled above)
                    if img_path.name.startswith("three_view"):
                        continue
                    b64 = self._load_and_encode_image(img_path)
                    if b64:
                        examples.append(b64)
                        logger.debug(f"Loaded example: {img_path.name}")

        self._stage_cache[stage] = examples
        logger.info(f"Loaded {len(examples)} examples for stage '{stage}'")

    def _load_stage_metadata(self, stage: str) -> Dict:
        """Load metadata.json from a stage folder"""
        if stage in self._stage_metadata_cache:
            return self._stage_metadata_cache[stage]

        stage_dir = self.stages_path / stage
        metadata_path = stage_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self._stage_metadata_cache[stage] = metadata
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to load stage metadata for {stage}: {e}")

        self._stage_metadata_cache[stage] = {}
        return {}

    def get_stage_examples_with_descriptions(
        self,
        stage: str,
        max_examples: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Get example images with their descriptions for a developmental stage.

        Parameters
        ----------
        stage : str
            Stage name (early, comma, pretzel, etc.)
        max_examples : int
            Maximum number of examples to return

        Returns
        -------
        List[Dict[str, str]]
            List of dicts with 'image' (base64) and 'description' keys
        """
        # Load examples
        if stage not in self._stage_cache:
            self._load_stage_examples(stage)

        examples_b64 = self._stage_cache.get(stage, [])[:max_examples]

        # Load metadata
        metadata = self._load_stage_metadata(stage)
        example_descriptions = metadata.get("examples", {})

        # Build result with descriptions
        result = []
        stage_dir = self.stages_path / stage

        # Get sorted filenames to match with b64 images
        # Match the same patterns as _load_stage_examples
        image_files = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
        for pattern in image_patterns:
            image_files.extend(sorted(stage_dir.glob(pattern)))
        image_files = sorted(image_files)[:max_examples]

        for i, (b64, img_path) in enumerate(zip(examples_b64, image_files)):
            description = example_descriptions.get(img_path.name, "")
            result.append({
                "image": b64,
                "description": description,
                "filename": img_path.name,
            })

        return result

    def get_stage_description(self, stage: str) -> str:
        """Get the overall description for a stage from metadata"""
        metadata = self._load_stage_metadata(stage)
        return metadata.get("description", "")

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
        self._volume_cache.clear()
        logger.info("Cleared example cache")

    # ==========================================================================
    # Volume support for 3D viewing of reference examples
    # ==========================================================================

    def get_stage_volume(self, stage: str, timepoint: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get the 3D volume for a stage reference example.

        Parameters
        ----------
        stage : str
            Stage name (early, comma, pretzel, etc.)
        timepoint : int, optional
            Specific timepoint to load. If None, loads first available.

        Returns
        -------
        np.ndarray or None
            3D volume array (Z, Y, X) if available, None otherwise
        """
        cache_key = f"{stage}_{timepoint}" if timepoint else stage

        if cache_key not in self._volume_cache:
            self._load_stage_volume(stage, timepoint)

        return self._volume_cache.get(cache_key)

    def _load_stage_volume(self, stage: str, timepoint: Optional[int] = None) -> None:
        """Load and cache volume for a stage"""
        stage_dir = self.stages_path / stage
        volumes_dir = stage_dir / "volumes"
        cache_key = f"{stage}_{timepoint}" if timepoint else stage

        # Try new format: volumes/T{timepoint}.npz
        if volumes_dir.exists():
            if timepoint is not None:
                volume_path = volumes_dir / f"T{timepoint:03d}.npz"
            else:
                # Get first available volume
                npz_files = sorted(volumes_dir.glob("T*.npz"))
                if npz_files:
                    volume_path = npz_files[0]
                else:
                    logger.debug(f"No volume files in {volumes_dir}")
                    self._volume_cache[cache_key] = None
                    return

            if volume_path.exists():
                try:
                    data = np.load(volume_path)
                    volume = data["volume"]
                    self._volume_cache[cache_key] = volume
                    logger.info(f"Loaded volume for stage '{stage}' from {volume_path.name}: shape={volume.shape}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load volume {volume_path}: {e}")

        # Fallback: try old format volume.npz
        old_volume_path = stage_dir / "volume.npz"
        if old_volume_path.exists():
            try:
                data = np.load(old_volume_path)
                volume = data["volume"]
                self._volume_cache[cache_key] = volume
                logger.info(f"Loaded volume for stage '{stage}' (legacy): shape={volume.shape}")
                return
            except Exception as e:
                logger.warning(f"Failed to load legacy volume for stage '{stage}': {e}")

        self._volume_cache[cache_key] = None

    def has_volume(self, stage: str) -> bool:
        """Check if a stage has a 3D volume available."""
        stage_dir = self.stages_path / stage
        volumes_dir = stage_dir / "volumes"

        # Check new format
        if volumes_dir.exists() and list(volumes_dir.glob("T*.npz")):
            return True
        # Check old format
        return (stage_dir / "volume.npz").exists()

    def list_stages_with_volumes(self) -> List[str]:
        """List all stages that have 3D volumes available."""
        return [stage for stage in self.STAGES if self.has_volume(stage)]

    def get_stage_volume_metadata(self, stage: str) -> Optional[Dict]:
        """
        Get metadata for a stage's volume.

        Returns
        -------
        Dict or None
            Metadata dict with stage info, timepoint, shape, etc.
        """
        stage_dir = self.stages_path / stage
        metadata_path = stage_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load volume metadata for {stage}: {e}")

        return None
