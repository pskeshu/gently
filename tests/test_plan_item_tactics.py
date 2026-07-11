"""
Task 9 — ImagingSpec tactical outline.

Verifies that `tactics` round-trips through the FileContextStore YAML path
and through the SQLite-backed ContextStore path.
"""

import dataclasses

from gently.harness.memory.model import ImagingSpec

TACTICS_OUTLINE = [
    {
        "kind": "standing_timelapse",
        "name": "baseline timelapse",
        "scope": "all embryos",
    },
    {
        "kind": "reactive_monitor",
        "name": "comma-stage speedup",
        "target": "comma",
        "structure": {"interval_s": 60},
    },
]


# ---------------------------------------------------------------------------
# 1. Model-level: field exists and defaults to empty list
# ---------------------------------------------------------------------------


class TestImagingSpecField:
    def test_tactics_field_exists(self):
        spec = ImagingSpec()
        assert hasattr(spec, "tactics")

    def test_tactics_default_empty_list(self):
        spec = ImagingSpec()
        assert spec.tactics == []

    def test_tactics_accepts_outline(self):
        spec = ImagingSpec(strain="N2", tactics=TACTICS_OUTLINE)
        assert spec.tactics == TACTICS_OUTLINE

    def test_tactics_field_in_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(ImagingSpec)}
        assert "tactics" in field_names


# ---------------------------------------------------------------------------
# 2. FileContextStore (YAML) round-trip
# ---------------------------------------------------------------------------


class TestFileStoreTactics:
    def test_tactics_survive_create_and_get(self, file_context_store):
        """Tactics stored in spec persist through create_plan_item → get_plan_item."""
        cid = file_context_store.create_campaign(description="Tactics test")
        iid = file_context_store.create_plan_item(
            campaign_id=cid,
            type="imaging",
            title="Tactic-bearing item",
            spec={
                "strain": "OH904",
                "interval_s": 180,
                "tactics": TACTICS_OUTLINE,
            },
        )
        item = file_context_store.get_plan_item(iid)
        assert item is not None
        assert item.imaging_spec is not None
        assert item.imaging_spec.tactics == TACTICS_OUTLINE

    def test_tactics_survive_update(self, file_context_store):
        """Tactics set via update_plan_item also persist."""
        cid = file_context_store.create_campaign(description="Update tactics")
        iid = file_context_store.create_plan_item(
            campaign_id=cid,
            type="imaging",
            title="No tactics yet",
            spec={"strain": "N2"},
        )
        # Initially empty
        item = file_context_store.get_plan_item(iid)
        assert item.imaging_spec.tactics == []

        # Add tactics via update
        new_spec = {"strain": "N2", "tactics": TACTICS_OUTLINE[:1]}
        file_context_store.update_plan_item(iid, spec=new_spec)
        item = file_context_store.get_plan_item(iid)
        assert item.imaging_spec.tactics == TACTICS_OUTLINE[:1]

    def test_existing_items_without_tactics_default_empty(self, file_context_store):
        """Items stored without a tactics key default to [] on read-back."""
        cid = file_context_store.create_campaign(description="Legacy item")
        iid = file_context_store.create_plan_item(
            campaign_id=cid,
            type="imaging",
            title="Legacy (no tactics key)",
            spec={"strain": "N2", "num_slices": 80},
        )
        # Manually strip tactics from the persisted YAML to simulate a pre-tactics record
        loc = file_context_store._find_plan_item_location(iid)
        campaign_id_found, items, idx = loc
        raw = items[idx]
        if isinstance(raw.get("spec"), dict):
            raw["spec"].pop("tactics", None)
        file_context_store._write_plan_items(campaign_id_found, items)

        item = file_context_store.get_plan_item(iid)
        assert item.imaging_spec.tactics == []


# ---------------------------------------------------------------------------
# 3. SQLite ContextStore round-trip
# ---------------------------------------------------------------------------


class TestContextStoreTactics:
    def test_tactics_survive_sqlite_round_trip(self, context_store):
        """Tactics stored in spec dict survive through the SQLite ContextStore."""
        cid = context_store.create_campaign(description="SQLite tactics test")
        iid = context_store.create_plan_item(
            campaign_id=cid,
            type="imaging",
            title="SQLite tactic item",
            spec={
                "strain": "OH904",
                "tactics": TACTICS_OUTLINE,
            },
        )
        item = context_store.get_plan_item(iid)
        assert item is not None
        assert item.imaging_spec is not None
        assert item.imaging_spec.tactics == TACTICS_OUTLINE
