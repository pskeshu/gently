"""
Architecture Registry — Known model architectures with metadata.

The agent reasons over this registry to pick the best architecture
for a given task, dataset size, and hardware constraints.
"""

from typing import Any

from .models import ModelArchitectureType

# Architecture registry: metadata per architecture
ARCHITECTURE_REGISTRY: dict[str, dict[str, Any]] = {
    ModelArchitectureType.RESNET_18.value: {
        "name": "ResNet-18",
        "family": "resnet",
        "param_count_m": 11.7,
        "min_vram_gb": 2.0,
        "recommended_vram_gb": 4.0,
        "min_dataset_size": 100,
        "recommended_dataset_size": 500,
        "training_speed": "fast",
        "suitability": (
            "Good baseline for microscopy. Fast to train, works well with "
            "transfer learning even on small datasets. Grayscale-friendly."
        ),
        "strengths": ["fast training", "low memory", "good baseline"],
        "weaknesses": ["lower capacity for complex patterns"],
    },
    ModelArchitectureType.RESNET_50.value: {
        "name": "ResNet-50",
        "family": "resnet",
        "param_count_m": 25.6,
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 8.0,
        "min_dataset_size": 300,
        "recommended_dataset_size": 1000,
        "training_speed": "medium",
        "suitability": (
            "Strong general architecture. More capacity than ResNet-18 "
            "for distinguishing subtle morphological differences."
        ),
        "strengths": ["well-studied", "good capacity", "reliable"],
        "weaknesses": ["more data needed", "slower than ResNet-18"],
    },
    ModelArchitectureType.EFFICIENTNET_B0.value: {
        "name": "EfficientNet-B0",
        "family": "efficientnet",
        "param_count_m": 5.3,
        "min_vram_gb": 2.0,
        "recommended_vram_gb": 4.0,
        "min_dataset_size": 100,
        "recommended_dataset_size": 500,
        "training_speed": "fast",
        "suitability": (
            "Compact and efficient. Best accuracy/compute tradeoff for "
            "small to medium datasets. Excellent for grayscale microscopy."
        ),
        "strengths": ["compact", "efficient", "good accuracy/size ratio"],
        "weaknesses": ["less capacity than larger models"],
    },
    ModelArchitectureType.EFFICIENTNET_B2.value: {
        "name": "EfficientNet-B2",
        "family": "efficientnet",
        "param_count_m": 9.2,
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 8.0,
        "min_dataset_size": 200,
        "recommended_dataset_size": 800,
        "training_speed": "medium",
        "suitability": (
            "Sweet spot for microscopy classification. Good capacity for "
            "embryo stage differences, fits comfortably on A5000 24GB. "
            "Recommended default for gently."
        ),
        "strengths": ["good capacity", "efficient", "recommended for embryo work"],
        "weaknesses": ["slightly slower than B0"],
    },
    ModelArchitectureType.EFFICIENTNET_B4.value: {
        "name": "EfficientNet-B4",
        "family": "efficientnet",
        "param_count_m": 19.3,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 16.0,
        "min_dataset_size": 500,
        "recommended_dataset_size": 2000,
        "training_speed": "slow",
        "suitability": (
            "High-capacity model for large datasets. Use when you have "
            "abundant annotated data and want maximum accuracy."
        ),
        "strengths": ["high capacity", "strong accuracy on large datasets"],
        "weaknesses": ["needs more data", "slow training", "high VRAM"],
    },
    ModelArchitectureType.MOBILENET_V3.value: {
        "name": "MobileNetV3-Large",
        "family": "mobilenet",
        "param_count_m": 5.4,
        "min_vram_gb": 2.0,
        "recommended_vram_gb": 4.0,
        "min_dataset_size": 100,
        "recommended_dataset_size": 500,
        "training_speed": "fast",
        "suitability": (
            "Lightweight model optimized for inference speed. Good for "
            "real-time classification during acquisition."
        ),
        "strengths": ["fast inference", "low memory", "deployment-friendly"],
        "weaknesses": ["lower accuracy than EfficientNet"],
    },
    ModelArchitectureType.CONVNEXT_TINY.value: {
        "name": "ConvNeXt-Tiny",
        "family": "convnext",
        "param_count_m": 28.6,
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 8.0,
        "min_dataset_size": 300,
        "recommended_dataset_size": 1000,
        "training_speed": "medium",
        "suitability": (
            "Modern ConvNet with transformer-like design. Strong on "
            "microscopy tasks, especially texture recognition."
        ),
        "strengths": ["modern design", "strong features", "good for textures"],
        "weaknesses": ["more parameters", "needs decent dataset"],
    },
    ModelArchitectureType.CONVNEXT_SMALL.value: {
        "name": "ConvNeXt-Small",
        "family": "convnext",
        "param_count_m": 50.2,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 16.0,
        "min_dataset_size": 500,
        "recommended_dataset_size": 2000,
        "training_speed": "slow",
        "suitability": (
            "Larger ConvNeXt variant. High capacity for complex tasks. "
            "Use with large, well-annotated datasets."
        ),
        "strengths": ["highest capacity", "strongest features"],
        "weaknesses": ["slow", "high VRAM", "needs large dataset"],
    },
}


def get_suitable_architectures(
    dataset_size: int,
    vram_gb: float,
    image_type: str = "microscopy",
) -> list[dict[str, Any]]:
    """Filter architectures suitable for given constraints.

    Parameters
    ----------
    dataset_size : int
        Number of annotated samples available.
    vram_gb : float
        Available GPU VRAM in GB.
    image_type : str
        Type of images (currently only "microscopy" supported).

    Returns
    -------
    list of dict
        Suitable architectures with reasoning hints, sorted by recommendation.
    """
    suitable = []
    for arch_id, meta in ARCHITECTURE_REGISTRY.items():
        if vram_gb < meta["min_vram_gb"]:
            continue
        if dataset_size < meta["min_dataset_size"]:
            continue

        # Score: prefer architectures where dataset_size >= recommended
        score = 0
        if dataset_size >= meta["recommended_dataset_size"]:
            score += 2
        if vram_gb >= meta["recommended_vram_gb"]:
            score += 1
        if meta["training_speed"] == "fast":
            score += 1

        result = dict(meta)
        result["architecture_id"] = arch_id
        result["fit_score"] = score
        result["reason"] = _build_reason(arch_id, meta, dataset_size, vram_gb)
        suitable.append(result)

    # Sort by fit_score descending
    suitable.sort(key=lambda x: -x["fit_score"])
    return suitable


def _build_reason(arch_id: str, meta: dict, dataset_size: int, vram_gb: float) -> str:
    """Build a human-readable reason for recommending this architecture."""
    parts = []
    if dataset_size >= meta["recommended_dataset_size"]:
        parts.append("dataset large enough for full potential")
    elif dataset_size >= meta["min_dataset_size"]:
        parts.append("dataset sufficient but below recommended size")

    if vram_gb >= meta["recommended_vram_gb"]:
        parts.append("VRAM comfortable")
    else:
        parts.append("VRAM tight — may need smaller batch size")

    return "; ".join(parts) if parts else "meets minimum requirements"
