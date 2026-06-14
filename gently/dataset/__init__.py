"""
Dataset management module for embryo imaging data.

.. deprecated::
    This module is deprecated and will be removed in a future version.
    Use :class:`~gently.core.file_store.FileStore` for data access instead.

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

import warnings

warnings.warn(
    "gently.dataset is deprecated and will be removed in a future version. "
    "Use FileStore for data access.",
    DeprecationWarning,
    stacklevel=2,
)

from .aggregator import DatasetAggregator  # noqa: E402
from .embryo_dataset import DatasetEmbryoEntry, EmbryoDataset, ImageData  # noqa: E402
from .schema import (  # noqa: E402
    DATABASE_VERSION,
    get_connection,
    init_database,
    migrate_to_v2,
    migrate_to_v3,
)

__all__ = [
    "init_database",
    "get_connection",
    "migrate_to_v2",
    "migrate_to_v3",
    "DatasetAggregator",
    "EmbryoDataset",
    "DatasetEmbryoEntry",
    "ImageData",
    "DATABASE_VERSION",
]
