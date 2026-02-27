"""
UID-based data storage through Databroker

Provides a unified interface for storing and retrieving data with:
- Unique identifiers (UIDs) for all data
- Parent-child lineage tracking
- Type-based querying
- Lazy data retrieval
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import json
import logging
import uuid
import numpy as np

from ..settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DataReference:
    """
    Universal reference to data in the data store

    All data is referenced by UID, never passed directly.
    This enables:
    - Provenance tracking
    - Lazy loading
    - Cross-service data sharing
    """
    uid: str
    data_type: str  # "image", "volume", "analysis", "session", "detection", "calibration"
    parent_uid: Optional[str] = None  # For lineage tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"{self.data_type}:{self.uid[:8]}"

    def __repr__(self) -> str:
        return f"DataReference(uid='{self.uid}', type='{self.data_type}')"

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'uid': self.uid,
            'data_type': self.data_type,
            'parent_uid': self.parent_uid,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DataReference':
        """Deserialize from dictionary"""
        return cls(
            uid=data['uid'],
            data_type=data['data_type'],
            parent_uid=data.get('parent_uid'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
        )


class DataStore(ABC):
    """
    Abstract interface for data storage

    All implementations must support:
    - Store: Save data and return a reference
    - Retrieve: Get data by reference
    - Query: Find data by criteria
    - Lineage: Track parent-child relationships
    """

    @abstractmethod
    def store(
        self,
        data: Any,
        data_type: str,
        metadata: Optional[Dict] = None,
        parent_uid: Optional[str] = None,
    ) -> DataReference:
        """
        Store data and return a reference

        Parameters
        ----------
        data : any
            Data to store (numpy array, dict, etc.)
        data_type : str
            Type of data (image, volume, analysis, etc.)
        metadata : dict, optional
            Additional metadata to store
        parent_uid : str, optional
            UID of parent data (for lineage)

        Returns
        -------
        DataReference
            Reference to the stored data
        """
        pass

    @abstractmethod
    def retrieve(self, ref: Union[DataReference, str]) -> Any:
        """
        Retrieve data by reference or UID

        Parameters
        ----------
        ref : DataReference or str
            Reference or UID to retrieve

        Returns
        -------
        any
            The stored data
        """
        pass

    @abstractmethod
    def get_reference(self, uid: str) -> Optional[DataReference]:
        """
        Get reference by UID

        Parameters
        ----------
        uid : str
            UID to look up

        Returns
        -------
        DataReference or None
            The reference, or None if not found
        """
        pass

    @abstractmethod
    def query(
        self,
        data_type: Optional[str] = None,
        parent_uid: Optional[str] = None,
        **metadata_filters
    ) -> List[DataReference]:
        """
        Query for data matching criteria

        Parameters
        ----------
        data_type : str, optional
            Filter by data type
        parent_uid : str, optional
            Filter by parent UID
        **metadata_filters
            Additional metadata filters

        Returns
        -------
        list of DataReference
            Matching references
        """
        pass

    @abstractmethod
    def get_lineage(self, ref: Union[DataReference, str]) -> List[DataReference]:
        """
        Get parent chain for provenance

        Parameters
        ----------
        ref : DataReference or str
            Starting reference

        Returns
        -------
        list of DataReference
            Parent chain (oldest first)
        """
        pass

    @abstractmethod
    def get_children(self, ref: Union[DataReference, str]) -> List[DataReference]:
        """
        Get child data derived from this reference

        Parameters
        ----------
        ref : DataReference or str
            Parent reference

        Returns
        -------
        list of DataReference
            Child references
        """
        pass


class DatabrokerStore(DataStore):
    """
    Databroker implementation of DataStore

    Uses Bluesky's Databroker for persistence with:
    - Msgpack storage for efficient numpy array handling
    - Full metadata in run.start document
    - Lineage tracking via parent_uid
    """

    def __init__(
        self,
        catalog_name: str = "gently",
        temp_fallback: bool = True,
    ):
        """
        Parameters
        ----------
        catalog_name : str
            Name of the Databroker catalog
        temp_fallback : bool
            If True, fall back to temp catalog if named catalog fails
        """
        self.catalog_name = catalog_name
        self._db = None
        self._temp_fallback = temp_fallback

        # In-memory index for fast lookups
        self._index: Dict[str, DataReference] = {}

        # Initialize databroker connection
        self._init_databroker()

    def _init_databroker(self):
        """Initialize Databroker connection"""
        try:
            from databroker import Broker
            try:
                self._db = Broker.named(self.catalog_name)
                logger.info(f"Connected to Databroker catalog: {self.catalog_name}")
            except Exception as e:
                if self._temp_fallback:
                    logger.warning(f"Failed to connect to '{self.catalog_name}', using temp catalog: {e}")
                    self._db = Broker.named('temp')
                else:
                    raise
        except ImportError:
            logger.warning("Databroker not available, using in-memory storage only")
            self._db = None

    def store(
        self,
        data: Any,
        data_type: str,
        metadata: Optional[Dict] = None,
        parent_uid: Optional[str] = None,
    ) -> DataReference:
        """Store data in memory with optional Databroker backup"""
        metadata = metadata or {}
        uid = str(uuid.uuid4())

        # Create reference
        ref = DataReference(
            uid=uid,
            data_type=data_type,
            parent_uid=parent_uid,
            metadata=metadata.copy(),
            timestamp=datetime.now(),
        )

        # Always store in memory for fast access
        ref.metadata['_data'] = data
        self._index[uid] = ref

        # Optionally store in Databroker for persistence
        # (Currently disabled due to API compatibility issues)
        # if self._db is not None:
        #     try:
        #         self._store_in_databroker(uid, data, data_type, metadata, parent_uid)
        #     except Exception as e:
        #         logger.debug(f"Databroker storage skipped: {e}")

        logger.debug(f"Stored {data_type} with UID: {uid[:8]}")
        return ref

    def _store_in_databroker(
        self,
        uid: str,
        data: Any,
        data_type: str,
        metadata: Dict,
        parent_uid: Optional[str],
    ):
        """Store data in Databroker using RunEngine"""
        import bluesky.plan_stubs as bps
        from bluesky import RunEngine
        from ophyd.sim import NullStatus
        import time as time_module

        # Create a synthetic device to hold our data
        class DataDevice:
            def __init__(self, data, name):
                self.name = name
                self._data = data
                self.parent = None

            def read(self):
                return {self.name: {"value": self._serialize(self._data), "timestamp": time_module.time()}}

            def describe(self):
                shape = self._data.shape if hasattr(self._data, 'shape') else []
                # Databroker requires specific dtype values
                if hasattr(self._data, 'dtype'):
                    dtype = 'array'  # numpy arrays
                elif isinstance(self._data, (dict, list)):
                    dtype = 'string'  # JSON-serialized
                elif isinstance(self._data, bool):
                    dtype = 'boolean'
                elif isinstance(self._data, int):
                    dtype = 'integer'
                elif isinstance(self._data, float):
                    dtype = 'number'
                else:
                    dtype = 'string'
                return {self.name: {"source": "gently", "dtype": dtype, "shape": list(shape)}}

            def trigger(self):
                return NullStatus()

            def _serialize(self, data):
                """Serialize data for storage"""
                if isinstance(data, np.ndarray):
                    return data
                elif isinstance(data, dict):
                    return json.dumps(data)
                else:
                    return data

        device = DataDevice(data, data_type)

        # Create and run mini-plan
        def store_plan():
            yield from bps.open_run(md={
                'gently_uid': uid,
                'gently_data_type': data_type,
                'gently_parent_uid': parent_uid,
                'gently_metadata': metadata,
                'gently_timestamp': datetime.now().isoformat(),
            })
            yield from bps.trigger_and_read([device])
            yield from bps.close_run()

        # Use existing RunEngine or create temporary one
        RE = RunEngine({})
        RE.subscribe(self._db.insert)
        RE(store_plan())

    def retrieve(self, ref: Union[DataReference, str]) -> Any:
        """Retrieve data from in-memory storage"""
        uid = ref.uid if isinstance(ref, DataReference) else ref

        # Check in-memory index
        if uid in self._index:
            indexed_ref = self._index[uid]
            if '_data' in indexed_ref.metadata:
                return indexed_ref.metadata['_data']

        raise KeyError(f"Data not found for UID: {uid}")

    def _retrieve_from_databroker(self, uid: str) -> Any:
        """Retrieve data from Databroker"""
        # Search for run with this gently_uid
        runs = list(self._db(gently_uid=uid))
        if not runs:
            raise KeyError(f"No run found for UID: {uid}")

        run = self._db[runs[0]]
        data_type = run.metadata['start'].get('gently_data_type', 'unknown')

        # Get data from primary stream
        try:
            # Try v2 API first
            data = run.primary.read()[data_type]
            if hasattr(data, 'values'):
                data = data.values
            if hasattr(data, '__getitem__') and len(data) > 0:
                data = data[0]
        except:
            # Fall back to table API
            table = run.table()
            data = table[data_type].iloc[0]

        # Deserialize if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                pass

        return data

    def get_reference(self, uid: str) -> Optional[DataReference]:
        """Get reference by UID"""
        if uid in self._index:
            return self._index[uid]

        # Try to reconstruct from Databroker
        if self._db is not None:
            try:
                runs = list(self._db(gently_uid=uid))
                if runs:
                    run = self._db[runs[0]]
                    start = run.metadata['start']
                    ref = DataReference(
                        uid=start.get('gently_uid', uid),
                        data_type=start.get('gently_data_type', 'unknown'),
                        parent_uid=start.get('gently_parent_uid'),
                        metadata=start.get('gently_metadata', {}),
                        timestamp=datetime.fromisoformat(start.get('gently_timestamp', datetime.now().isoformat())),
                    )
                    self._index[uid] = ref
                    return ref
            except Exception as e:
                logger.debug(f"Failed to get reference from Databroker: {e}")

        return None

    def query(
        self,
        data_type: Optional[str] = None,
        parent_uid: Optional[str] = None,
        **metadata_filters
    ) -> List[DataReference]:
        """Query for data matching criteria"""
        results = []

        # Query in-memory index
        for uid, ref in self._index.items():
            if data_type and ref.data_type != data_type:
                continue
            if parent_uid and ref.parent_uid != parent_uid:
                continue

            # Check metadata filters
            match = True
            for key, value in metadata_filters.items():
                if ref.metadata.get(key) != value:
                    match = False
                    break

            if match:
                results.append(ref)

        # Query Databroker for additional results
        if self._db is not None:
            try:
                db_query = {}
                if data_type:
                    db_query['gently_data_type'] = data_type
                if parent_uid:
                    db_query['gently_parent_uid'] = parent_uid

                for run_uid in self._db(**db_query):
                    if run_uid not in [r.uid for r in results]:
                        ref = self.get_reference(run_uid)
                        if ref:
                            results.append(ref)
            except Exception as e:
                logger.debug(f"Databroker query failed: {e}")

        # Sort by timestamp (newest first)
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results

    def get_lineage(self, ref: Union[DataReference, str]) -> List[DataReference]:
        """Get parent chain for provenance"""
        uid = ref.uid if isinstance(ref, DataReference) else ref
        lineage = []

        current_uid = uid
        while current_uid:
            current_ref = self.get_reference(current_uid)
            if not current_ref:
                break
            lineage.append(current_ref)
            current_uid = current_ref.parent_uid

        # Reverse to get oldest first
        lineage.reverse()
        return lineage

    def get_children(self, ref: Union[DataReference, str]) -> List[DataReference]:
        """Get child data derived from this reference"""
        uid = ref.uid if isinstance(ref, DataReference) else ref
        return self.query(parent_uid=uid)

    def list_recent(self, limit: int = 10, data_type: Optional[str] = None) -> List[DataReference]:
        """
        List most recent data references

        Parameters
        ----------
        limit : int
            Maximum number to return
        data_type : str, optional
            Filter by data type

        Returns
        -------
        list of DataReference
            Recent references
        """
        results = self.query(data_type=data_type)
        return results[:limit]

    def delete(self, ref: Union[DataReference, str]) -> bool:
        """
        Delete data by reference

        Note: Only removes from index, not from Databroker
        (Databroker is append-only by design)

        Parameters
        ----------
        ref : DataReference or str
            Reference to delete

        Returns
        -------
        bool
            True if deleted
        """
        uid = ref.uid if isinstance(ref, DataReference) else ref
        if uid in self._index:
            del self._index[uid]
            return True
        return False

    def clear_index(self):
        """Clear in-memory index"""
        self._index.clear()

    @property
    def stats(self) -> Dict:
        """Get storage statistics"""
        type_counts = {}
        for ref in self._index.values():
            type_counts[ref.data_type] = type_counts.get(ref.data_type, 0) + 1

        return {
            'total_entries': len(self._index),
            'by_type': type_counts,
            'has_databroker': self._db is not None,
        }


# =============================================================================
# Tiled-based Persistent Store
# =============================================================================

class TiledStore(DataStore):
    """
    Tiled-based persistent data store

    Uses Tiled for efficient storage of microscopy data:
    - Numpy arrays stored as zarr/array
    - Metadata stored in Tiled's catalog
    - Full lineage tracking
    - Persistent storage at specified path

    Parameters
    ----------
    storage_path : str or Path
        Base path for data storage (e.g., "D:/Gently")
    catalog_name : str
        Name of the Tiled catalog
    """

    def __init__(
        self,
        storage_path: str = str(settings.storage.base_path),
        catalog_name: str = "gently",
    ):
        from pathlib import Path
        self.storage_path = Path(storage_path)
        self.catalog_name = catalog_name

        # Create directories
        self.data_dir = self.storage_path / "data"
        self.index_dir = self.storage_path / "index"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookups
        self._index: Dict[str, DataReference] = {}

        # Tiled client (lazy loaded)
        self._client = None
        self._tiled_available = False

        # Initialize
        self._init_storage()
        self._load_index()

        logger.info(f"TiledStore initialized at {self.storage_path}")

    def _init_storage(self):
        """Initialize Tiled storage"""
        try:
            from tiled.client import from_uri
            from tiled.server.app import build_app
            import tiled.config

            # Check if tiled server is running, or start embedded
            try:
                self._client = from_uri(f"http://localhost:8000/api/v1/node/{self.catalog_name}")
                self._tiled_available = True
                logger.info("Connected to existing Tiled server")
            except Exception:
                # Try to use local file-based catalog
                catalog_path = self.storage_path / "catalog"
                catalog_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Tiled server not available, using file-based storage at {self.storage_path}")
                self._tiled_available = False

        except ImportError:
            logger.info("Tiled not installed, using file-based storage")
            self._tiled_available = False

    def _load_index(self):
        """Load index from disk"""
        index_file = self.index_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                for uid, ref_dict in data.items():
                    self._index[uid] = DataReference.from_dict(ref_dict)
                logger.info(f"Loaded {len(self._index)} entries from index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")

    def _save_index(self):
        """Save index to disk"""
        index_file = self.index_dir / "index.json"
        try:
            data = {uid: ref.to_dict() for uid, ref in self._index.items()}
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")

    def store(
        self,
        data: Any,
        data_type: str,
        metadata: Optional[Dict] = None,
        parent_uid: Optional[str] = None,
    ) -> DataReference:
        """Store data to disk with Tiled/file backend"""
        metadata = metadata or {}
        uid = str(uuid.uuid4())
        timestamp = datetime.now()

        # Create reference
        ref = DataReference(
            uid=uid,
            data_type=data_type,
            parent_uid=parent_uid,
            metadata=metadata.copy(),
            timestamp=timestamp,
        )

        # Determine storage format and save
        if isinstance(data, np.ndarray):
            self._store_array(uid, data, data_type, metadata)
        elif isinstance(data, dict):
            self._store_json(uid, data, data_type)
        else:
            # Fallback: pickle
            self._store_pickle(uid, data, data_type)

        # Add to index
        self._index[uid] = ref
        self._save_index()

        logger.debug(f"Stored {data_type} with UID: {uid[:8]}")
        return ref

    def _store_array(self, uid: str, data: np.ndarray, data_type: str, metadata: Dict):
        """Store numpy array as TIFF (ImageJ compatible)"""
        # Organize by data type and date
        date_str = datetime.now().strftime("%Y%m%d")
        type_dir = self.data_dir / data_type / date_str
        type_dir.mkdir(parents=True, exist_ok=True)

        # Save as TIFF (standard for microscopy, ImageJ compatible)
        try:
            import tifffile
            array_path = type_dir / f"{uid}.tif"
            # Use compression for efficiency
            tifffile.imwrite(str(array_path), data, compression='zlib')
            logger.debug(f"Saved array as TIFF: {array_path}")
        except ImportError:
            # Fall back to numpy format if tifffile not available
            array_path = type_dir / f"{uid}.npy"
            np.save(str(array_path), data)
            logger.debug(f"Saved array as npy (tifffile not available): {array_path}")

        # Save metadata alongside
        meta_path = type_dir / f"{uid}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'uid': uid,
                'data_type': data_type,
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)

    def _store_json(self, uid: str, data: Dict, data_type: str):
        """Store JSON data"""
        date_str = datetime.now().strftime("%Y%m%d")
        type_dir = self.data_dir / data_type / date_str
        type_dir.mkdir(parents=True, exist_ok=True)

        file_path = type_dir / f"{uid}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _store_pickle(self, uid: str, data: Any, data_type: str):
        """Store arbitrary data as pickle"""
        import pickle
        date_str = datetime.now().strftime("%Y%m%d")
        type_dir = self.data_dir / data_type / date_str
        type_dir.mkdir(parents=True, exist_ok=True)

        file_path = type_dir / f"{uid}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def retrieve(self, ref: Union[DataReference, str]) -> Any:
        """Retrieve data from disk"""
        uid = ref.uid if isinstance(ref, DataReference) else ref

        # Search for file
        data_file = self._find_data_file(uid)
        if data_file is None:
            raise KeyError(f"Data not found for UID: {uid}")

        # Load based on extension
        suffix = data_file.suffix
        if suffix in ('.tif', '.tiff'):
            import tifffile
            return tifffile.imread(str(data_file))
        elif suffix == '.zarr':
            try:
                import zarr
                return zarr.load(str(data_file))
            except Exception:
                raise ValueError(f"Cannot load zarr file (version incompatibility): {data_file}")
        elif suffix == '.npy':
            return np.load(str(data_file))
        elif suffix == '.json':
            with open(data_file, 'r') as f:
                return json.load(f)
        elif suffix == '.pkl':
            import pickle
            with open(data_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown file format: {suffix}")

    def _find_data_file(self, uid: str):
        """Find data file for UID"""
        from pathlib import Path

        # Search all data directories
        for type_dir in self.data_dir.iterdir():
            if not type_dir.is_dir():
                continue
            for date_dir in type_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                # Check for any file with this UID (prefer TIFF)
                for ext in ['.tif', '.tiff', '.zarr', '.npy', '.json', '.pkl']:
                    file_path = date_dir / f"{uid}{ext}"
                    if file_path.exists():
                        return file_path
                    # Check if zarr directory
                    if ext == '.zarr' and file_path.is_dir():
                        return file_path

        return None

    def get_reference(self, uid: str) -> Optional[DataReference]:
        """Get reference by UID"""
        return self._index.get(uid)

    def query(
        self,
        data_type: Optional[str] = None,
        parent_uid: Optional[str] = None,
        **metadata_filters
    ) -> List[DataReference]:
        """Query for data matching criteria"""
        results = []

        for uid, ref in self._index.items():
            if data_type and ref.data_type != data_type:
                continue
            if parent_uid and ref.parent_uid != parent_uid:
                continue

            # Check metadata filters
            match = True
            for key, value in metadata_filters.items():
                if ref.metadata.get(key) != value:
                    match = False
                    break

            if match:
                results.append(ref)

        # Sort by timestamp (newest first)
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results

    def get_lineage(self, ref: Union[DataReference, str]) -> List[DataReference]:
        """Get parent chain for provenance"""
        uid = ref.uid if isinstance(ref, DataReference) else ref
        lineage = []

        current_uid = uid
        while current_uid:
            current_ref = self.get_reference(current_uid)
            if not current_ref:
                break
            lineage.append(current_ref)
            current_uid = current_ref.parent_uid

        lineage.reverse()
        return lineage

    def get_children(self, ref: Union[DataReference, str]) -> List[DataReference]:
        """Get child data derived from this reference"""
        uid = ref.uid if isinstance(ref, DataReference) else ref
        return self.query(parent_uid=uid)

    def list_recent(self, limit: int = 10, data_type: Optional[str] = None) -> List[DataReference]:
        """List most recent data references"""
        results = self.query(data_type=data_type)
        return results[:limit]

    @property
    def stats(self) -> Dict:
        """Get storage statistics"""
        type_counts = {}
        for ref in self._index.values():
            type_counts[ref.data_type] = type_counts.get(ref.data_type, 0) + 1

        # Calculate disk usage
        total_size = 0
        try:
            for path in self.data_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        except:
            pass

        return {
            'total_entries': len(self._index),
            'by_type': type_counts,
            'storage_path': str(self.storage_path),
            'disk_usage_mb': total_size / (1024 * 1024),
            'tiled_available': self._tiled_available,
        }


# =============================================================================
# Global Store Management
# =============================================================================

_global_store: Optional[DataStore] = None


def get_data_store(
    storage_path: str = str(settings.storage.base_path),
    catalog_name: str = "gently",
    use_tiled: bool = True,
) -> DataStore:
    """
    Get or create the global data store

    Parameters
    ----------
    storage_path : str
        Path for persistent storage
    catalog_name : str
        Catalog name
    use_tiled : bool
        If True, use TiledStore for persistence

    Returns
    -------
    DataStore
        The global data store instance
    """
    global _global_store
    if _global_store is None:
        if use_tiled:
            _global_store = TiledStore(
                storage_path=storage_path,
                catalog_name=catalog_name,
            )
        else:
            _global_store = DatabrokerStore(catalog_name=catalog_name)
    return _global_store


def set_data_store(store: DataStore):
    """Set the global data store"""
    global _global_store
    _global_store = store
