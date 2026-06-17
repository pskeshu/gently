"""Connection D: inline edits to plan items / imaging specs via PATCH.

The inspector PATCHes changed fields; the store fires PLAN_UPDATED so the
Plans UI refreshes live. Spec edits merge into the existing spec.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(context_store):
    from gently.ui.web.routes.campaigns import create_router

    app = FastAPI()

    class _Server:
        pass

    server = _Server()
    server.context_store = context_store
    app.include_router(create_router(server))
    return app


def _seed_imaging(cs):
    cid = cs.create_campaign(description="c", target="goal")
    iid = cs.create_plan_item(
        campaign_id=cid,
        type="imaging",
        title="WT baseline",
        spec={"strain": "N2", "laser_wavelength_nm": 488},
    )
    return cid, iid


class TestPatchItem:
    def test_edit_title_and_status(self, file_context_store):
        cid, iid = _seed_imaging(file_context_store)
        client = TestClient(_make_app(file_context_store))
        r = client.patch(
            f"/api/campaigns/{cid}/items/{iid}",
            json={"title": "WT baseline (rev)", "status": "in_progress"},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True
        item = file_context_store.get_plan_item(iid)
        assert item.title == "WT baseline (rev)"
        assert item.status.value == "in_progress"

    def test_fill_laser_power_merges_spec(self, file_context_store):
        cid, iid = _seed_imaging(file_context_store)
        client = TestClient(_make_app(file_context_store))
        r = client.patch(
            f"/api/campaigns/{cid}/items/{iid}",
            json={"spec": {"laser_power_pct": 8}},
        )
        assert r.status_code == 200
        item = file_context_store.get_plan_item(iid)
        # the filled field is set...
        assert item.imaging_spec.laser_power_pct == 8
        # ...and the pre-existing spec fields survive the merge
        assert item.imaging_spec.strain == "N2"
        assert item.imaging_spec.laser_wavelength_nm == 488

    def test_empty_string_clears_field(self, file_context_store):
        cid, iid = _seed_imaging(file_context_store)
        client = TestClient(_make_app(file_context_store))
        r = client.patch(f"/api/campaigns/{cid}/items/{iid}", json={"spec": {"strain": ""}})
        assert r.status_code == 200
        item = file_context_store.get_plan_item(iid)
        assert item.imaging_spec.strain is None

    def test_patch_fires_plan_updated(self, file_context_store):
        from gently.core.event_bus import EventType, on

        cid, iid = _seed_imaging(file_context_store)
        client = TestClient(_make_app(file_context_store))
        seen = []
        unsub = on(EventType.PLAN_UPDATED, lambda e: seen.append(e))
        try:
            client.patch(f"/api/campaigns/{cid}/items/{iid}", json={"title": "x"})
        finally:
            unsub()
        assert any((e.data or {}).get("campaign_id") == cid for e in seen)

    def test_no_fields_is_400(self, file_context_store):
        cid, iid = _seed_imaging(file_context_store)
        client = TestClient(_make_app(file_context_store))
        r = client.patch(f"/api/campaigns/{cid}/items/{iid}", json={})
        assert r.status_code == 400

    def test_missing_item_is_404(self, file_context_store):
        cid, _ = _seed_imaging(file_context_store)
        client = TestClient(_make_app(file_context_store))
        r = client.patch(f"/api/campaigns/{cid}/items/nope", json={"title": "x"})
        assert r.status_code == 404
