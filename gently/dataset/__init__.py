"""
Dataset management module for embryo imaging data.

Provides:
- SQLite database for indexing volumes, images, and perception results
- Data aggregator for scanning existing data directories
- EmbryoDataset for streaming images per embryo
- Web explorer for browsing and annotating datasets

Example usage:
    from gently.dataset import EmbryoDataset

    dataset = EmbryoDataset()

    # Iterate through embryos
    for embryo in dataset.iter_embryos(session_id="59799c78"):
        print(f"Processing {embryo.embryo_id}")

        # Stream images
        for img in embryo.iter_images():
            result = my_perception(img.image_b64)
            dataset.store_prediction(
                run_id=run_id,
                embryo_id=embryo.embryo_id,
                timepoint=img.timepoint,
                predicted_stage=result.stage,
            )
"""

from .schema import (
    init_database,
    get_connection,
    migrate_to_v2,
    migrate_to_v3,
    DATABASE_VERSION,
)
from .aggregator import DatasetAggregator
from .embryo_dataset import EmbryoDataset, EmbryoInfo, ImageData

__all__ = [
    "init_database",
    "get_connection",
    "migrate_to_v2",
    "migrate_to_v3",
    "DatasetAggregator",
    "EmbryoDataset",
    "EmbryoInfo",
    "ImageData",
    "DATABASE_VERSION",
]
