"""
Tests for core data store: DataReference and DatabrokerStore (in-memory mode).
"""

import pytest
import numpy as np
from datetime import datetime

from gently.core.data_store import DataReference, DatabrokerStore


# ===========================================================================
# DataReference
# ===========================================================================

class TestDataReference:
    """DataReference creation and serialization."""

    def test_creation(self):
        ref = DataReference(uid="abc123", data_type="image")
        assert ref.uid == "abc123"
        assert ref.data_type == "image"
        assert ref.parent_uid is None

    def test_with_parent(self):
        ref = DataReference(uid="child", data_type="analysis", parent_uid="parent")
        assert ref.parent_uid == "parent"

    def test_str_repr(self):
        ref = DataReference(uid="abcdef1234567890", data_type="volume")
        assert "volume" in str(ref)
        assert "abcdef12" in str(ref)

    def test_to_dict(self):
        ref = DataReference(uid="abc", data_type="image", metadata={"embryo": "e1"})
        d = ref.to_dict()
        assert d['uid'] == 'abc'
        assert d['data_type'] == 'image'
        assert d['metadata']['embryo'] == 'e1'
        assert 'timestamp' in d

    def test_roundtrip(self):
        original = DataReference(
            uid="abc", data_type="volume", parent_uid="parent",
            metadata={"slices": 50}
        )
        restored = DataReference.from_dict(original.to_dict())
        assert restored.uid == original.uid
        assert restored.data_type == original.data_type
        assert restored.parent_uid == original.parent_uid
        assert restored.metadata == original.metadata


# ===========================================================================
# DatabrokerStore (in-memory, no Databroker dependency)
# ===========================================================================

class TestDatabrokerStore:
    """DatabrokerStore stores and retrieves data in memory."""

    @pytest.fixture
    def store(self, monkeypatch):
        """Create store that won't try to connect to real Databroker."""
        import sys
        # Ensure databroker import fails so store uses in-memory only
        monkeypatch.setitem(sys.modules, 'databroker', None)
        return DatabrokerStore(catalog_name="test")

    def test_store_and_retrieve_dict(self, store):
        ref = store.store({"key": "value"}, "analysis")
        assert ref.data_type == "analysis"
        data = store.retrieve(ref)
        assert data == {"key": "value"}

    def test_store_and_retrieve_array(self, store):
        arr = np.zeros((10, 10), dtype=np.uint16)
        ref = store.store(arr, "image")
        retrieved = store.retrieve(ref)
        assert np.array_equal(retrieved, arr)

    def test_retrieve_by_uid_string(self, store):
        ref = store.store(42, "scalar")
        data = store.retrieve(ref.uid)
        assert data == 42

    def test_retrieve_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.retrieve("nonexistent_uid")

    def test_parent_uid_tracking(self, store):
        parent = store.store({"type": "volume"}, "volume")
        child = store.store({"type": "projection"}, "analysis", parent_uid=parent.uid)
        assert child.parent_uid == parent.uid

    def test_query_by_type(self, store):
        store.store({"a": 1}, "image")
        store.store({"b": 2}, "image")
        store.store({"c": 3}, "analysis")
        results = store.query(data_type="image")
        assert len(results) == 2

    def test_query_by_parent(self, store):
        parent = store.store({}, "volume")
        store.store({}, "analysis", parent_uid=parent.uid)
        store.store({}, "analysis", parent_uid=parent.uid)
        store.store({}, "analysis", parent_uid="other")
        children = store.query(parent_uid=parent.uid)
        assert len(children) == 2

    def test_query_with_metadata_filter(self, store):
        store.store({}, "image", metadata={"embryo": "e1"})
        store.store({}, "image", metadata={"embryo": "e2"})
        results = store.query(data_type="image", embryo="e1")
        assert len(results) == 1

    def test_get_reference(self, store):
        ref = store.store({}, "image")
        found = store.get_reference(ref.uid)
        assert found.uid == ref.uid

    def test_get_reference_missing(self, store):
        assert store.get_reference("nonexistent") is None

    def test_lineage(self, store):
        grandparent = store.store({}, "volume")
        parent = store.store({}, "projection", parent_uid=grandparent.uid)
        child = store.store({}, "analysis", parent_uid=parent.uid)
        lineage = store.get_lineage(child)
        assert len(lineage) == 3
        assert lineage[0].uid == grandparent.uid
        assert lineage[2].uid == child.uid

    def test_get_children(self, store):
        parent = store.store({}, "volume")
        store.store({}, "analysis", parent_uid=parent.uid)
        children = store.get_children(parent)
        assert len(children) == 1

    def test_delete(self, store):
        ref = store.store({}, "image")
        assert store.delete(ref) is True
        assert store.get_reference(ref.uid) is None
        assert store.delete(ref) is False

    def test_list_recent(self, store):
        for i in range(5):
            store.store({"i": i}, "image")
        recent = store.list_recent(limit=3)
        assert len(recent) == 3

    def test_clear_index(self, store):
        store.store({}, "image")
        store.store({}, "volume")
        store.clear_index()
        assert store.stats['total_entries'] == 0

    def test_stats(self, store):
        store.store({}, "image")
        store.store({}, "volume")
        stats = store.stats
        assert stats['total_entries'] == 2
        assert stats['by_type']['image'] == 1
        assert stats['by_type']['volume'] == 1
