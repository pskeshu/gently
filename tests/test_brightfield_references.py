"""Dark and flat-field references for brightfield frames.

"to get good quality images - we need to store - when we do any brightfield
imaging, a dark image - and an image for flat field correction … we do not
have read back from the room light device - so we need to cycle it … these
images have to be structured and stored, so they can be imported with the
data files"
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app import brightfield as bf
from gently.core.export import export_session
from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import brightfield as bf_routes

pytest.importorskip("tifffile")

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
ORCH = (ROOT / "gently" / "app" / "orchestration" / "timelapse.py").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
PANEL = (WEB / "static" / "js" / "panels" / "brightfield-refs.js").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")


def _client(dark_level=100, flat_level=3000, shape=(20, 30)):
    """A microscope that answers every call and remembers the order."""
    calls: list[tuple] = []
    lit = {"room": "unknown", "led": "Closed"}

    async def set_room_light(state):
        calls.append(("room", state))
        lit["room"] = state
        return {"success": True}

    async def set_led(state="Closed"):
        calls.append(("led", state))
        lit["led"] = state
        return {"success": True}

    async def set_led_intensity(pct):
        calls.append(("led_pct", pct))
        return {"success": True}

    async def capture_bottom_image(use_led=False, exposure_ms=None):
        calls.append(("capture", exposure_ms))
        level = flat_level if (lit["room"] == "on" or lit["led"] == "Open") else dark_level
        rng = np.random.default_rng(len(calls))
        img = (level + rng.integers(-5, 6, size=shape)).astype(np.uint16)
        return {"image": img}

    client = SimpleNamespace(
        set_room_light=set_room_light,
        set_led=set_led,
        set_led_intensity=set_led_intensity,
        capture_bottom_image=capture_bottom_image,
        calls=calls,
    )
    return client


class TestTakingADark:
    def test_the_room_light_is_cycled_and_the_led_closed_before_the_frame(self):
        c = _client()
        with patch.object(bf.asyncio, "sleep", new=AsyncMock()) as sleep:
            got = asyncio.run(bf.take_dark(c, 20.0, settle_s=1.5))
        assert c.calls == [("led", "Closed"), ("room", "on"), ("room", "off"), ("capture", 20.0)]
        assert sleep.await_count == 2 and all(call.args == (1.5,) for call in sleep.await_args_list)
        assert got["stats"]["mean"] == pytest.approx(100, abs=3)
        assert (
            "room light commanded on" in got["steps"] and "room light commanded off" in got["steps"]
        )

    def test_a_placeholder_frame_is_an_error_not_a_reference(self):
        c = _client()

        async def nothing(use_led=False, exposure_ms=None):
            return {"image": np.zeros((100, 100), dtype=np.uint16)}

        c.capture_bottom_image = nothing
        with patch.object(bf.asyncio, "sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="no frame"):
                asyncio.run(bf.take_dark(c, 20.0))


class TestTakingAFlat:
    def test_frames_are_averaged_under_the_led_which_is_closed_after(self):
        c = _client()
        spec = bf.ReferenceSpec(light="led", led_intensity_pct=1, exposure_ms=20.0)
        with patch.object(bf.asyncio, "sleep", new=AsyncMock()):
            got = asyncio.run(bf.take_flat(c, spec, frames=4))
        assert c.calls[:2] == [("led_pct", 1), ("led", "Open")]
        assert c.calls.count(("capture", 20.0)) == 4
        assert c.calls[-1] == ("led", "Closed")
        assert got["frames"] == 4 and got["image"].dtype == np.uint16
        assert got["stats"]["mean"] == pytest.approx(3000, abs=3)

    def test_under_the_room_light_it_goes_off_again_even_if_a_frame_fails(self):
        c = _client()
        n = {"k": 0}

        async def flaky(use_led=False, exposure_ms=None):
            n["k"] += 1
            if n["k"] == 2:
                return {"image": None}
            return {"image": np.full((4, 4), 500, dtype=np.uint16)}

        c.capture_bottom_image = flaky
        with patch.object(bf.asyncio, "sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="no frame"):
                asyncio.run(
                    bf.take_flat(c, bf.ReferenceSpec(light="room", exposure_ms=5.0), frames=3)
                )
        assert c.calls[-1] == ("room", "off")


@pytest.fixture
def store(tmp_path):
    s = FileStore(root=tmp_path / "data")
    s.create_session("s1")
    return s


class TestFiling:
    def test_a_record_holds_both_images_and_says_what_they_measured(self, store):
        spec = bf.ReferenceSpec(light="led", led_intensity_pct=1, exposure_ms=20.0)
        folder = bf.open_record(store, "s1", spec)
        dark = np.full((8, 8), 100, dtype=np.uint16)
        flat = np.full((8, 8), 3000, dtype=np.uint16)
        bf.file_image(folder, "dark", dark, spec, {"stats": bf.stats(dark), "steps": ["x"]})
        doc = bf.file_image(
            folder, "flat", flat, spec, {"stats": bf.stats(flat), "steps": ["y"], "frames": 5}
        )
        assert (folder / "dark_20ms.tif").is_file() and (
            folder / "flat_led-1pct_20ms.tif"
        ).is_file()
        assert doc["checks"] == {
            "dark_to_flat_mean_ratio": pytest.approx(0.033, abs=0.001),
            "dark_is_dark": True,
            "flat_saturated_fraction": 0.0,
            "flat_unsaturated": True,
            "complete": True,
        }
        on_disk = yaml.safe_load((folder / "brightfield.yaml").read_text(encoding="utf-8"))
        assert on_disk["flat"]["frames_averaged"] == 5 and on_disk["spec"]["led_intensity_pct"] == 1

    def test_a_dark_as_bright_as_the_flat_is_called_out(self, store):
        spec = bf.ReferenceSpec(light="room", exposure_ms=20.0)
        folder = bf.open_record(store, "s1", spec)
        bright = np.full((8, 8), 2900, dtype=np.uint16)
        flat = np.full((8, 8), 3000, dtype=np.uint16)
        bf.file_image(folder, "dark", bright, spec, {"stats": bf.stats(bright)})
        doc = bf.file_image(folder, "flat", flat, spec, {"stats": bf.stats(flat), "frames": 5})
        assert doc["checks"]["dark_is_dark"] is False

    def test_matching_wants_the_same_light_and_exposure_and_both_images(self, store):
        spec = bf.ReferenceSpec(light="led", led_intensity_pct=1, exposure_ms=20.0)
        folder = bf.open_record(store, "s1", spec)
        img = np.full((4, 4), 7, dtype=np.uint16)
        bf.file_image(folder, "dark", img, spec, {"stats": bf.stats(img)})
        records = bf.list_records(store, "s1")
        assert bf.matching(records, spec) is None  # no flat yet
        bf.file_image(folder, "flat", img * 10, spec, {"stats": bf.stats(img * 10), "frames": 1})
        records = bf.list_records(store, "s1")
        assert bf.matching(records, spec)["record"] == folder.name
        assert (
            bf.matching(
                records, bf.ReferenceSpec(light="led", led_intensity_pct=2, exposure_ms=20.0)
            )
            is None
        )
        assert bf.matching(records, bf.ReferenceSpec(light="room", exposure_ms=20.0)) is None
        frame = bf.for_frame(bf.matching(records, spec))
        assert frame == {
            "record": folder.name,
            "dark": f"calibration/brightfield/{folder.name}/dark_20ms.tif",
            "flat": f"calibration/brightfield/{folder.name}/flat_led-1pct_20ms.tif",
        }


# ── the routes ──────────────────────────────────────────────────────────────


def _app(store, client, running=False):
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.store = store
    agent.session_id = "s1"
    agent.client = client
    agent.timelapse_orchestrator = (
        SimpleNamespace(_status=SimpleNamespace(value="running")) if running else None
    )
    app = FastAPI()
    app.include_router(bf_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


class TestTheRoutes:
    def test_dark_then_flat_file_one_record_that_then_matches(self, store):
        c = _app(store, _client())
        with patch.object(bf.asyncio, "sleep", new=AsyncMock()):
            d = c.post(
                "/api/brightfield/references/dark",
                json={"light": "led", "led_intensity_pct": 1, "exposure_ms": 20},
            )
            assert d.status_code == 200, d.text
            dark = d.json()
            f = c.post(
                "/api/brightfield/references/flat",
                json={
                    "light": "led",
                    "led_intensity_pct": 1,
                    "exposure_ms": 20,
                    "frames": 3,
                    "record": dark["record"],
                },
            )
        assert f.status_code == 200, f.text
        flat = f.json()
        assert flat["record"] == dark["record"] and flat["frames"] == 3
        assert flat["checks"]["complete"] is True and flat["checks"]["dark_is_dark"] is True
        assert dark["thumbnail"] and flat["thumbnail"]
        got = c.get(
            "/api/brightfield/references",
            params={"light": "led", "led_intensity_pct": 1, "exposure_ms": 20},
        ).json()
        assert got["match"]["record"] == dark["record"]
        assert (
            c.get(
                "/api/brightfield/references", params={"light": "room", "exposure_ms": 20}
            ).json()["match"]
            is None
        )

    def test_a_live_run_refuses(self, store):
        c = _app(store, _client(), running=True)
        r = c.post("/api/brightfield/references/dark", json={"light": "room", "exposure_ms": 20})
        assert r.status_code == 409

    def test_bad_input_is_400(self, store):
        c = _app(store, _client())
        assert c.post("/api/brightfield/references/dark", json={"light": "sun"}).status_code == 400
        assert (
            c.post(
                "/api/brightfield/references/dark", json={"light": "led", "led_intensity_pct": 0}
            ).status_code
            == 400
        )


# ── the frames name their references; the export carries them ───────────────


class TestTheFramesAndTheExport:
    def test_the_orchestrator_names_the_references_on_every_frame_and_warns_when_none(self):
        assert '"references": getattr(self, "_dic_references", None),' in ORCH
        assert "self._dic_references = self._find_brightfield_references()" in ORCH
        assert "No dark/flat references for the overview" in ORCH
        assert "EventType.WARNING_ISSUED" in ORCH

    def test_the_export_copies_the_references_and_names_them_per_frame(self, store):
        spec = bf.ReferenceSpec(light="led", led_intensity_pct=1, exposure_ms=20.0)
        folder = bf.open_record(store, "s1", spec)
        img = np.full((4, 4), 100, dtype=np.uint16)
        bf.file_image(folder, "dark", img, spec, {"stats": bf.stats(img)})
        bf.file_image(folder, "flat", img * 20, spec, {"stats": bf.stats(img * 20), "frames": 5})
        refs = bf.for_frame(bf.matching(bf.list_records(store, "s1"), spec))
        store.put_snapshot(
            "s1",
            "dic",
            np.zeros((6, 9), dtype=np.uint16),
            metadata={
                "channel": "dic",
                "frame": 1,
                "captured_at": "2026-10-06T21:00:00",
                "references": refs,
            },
        )
        out = export_session(store, "s1")
        assert (out / "dic" / "references" / folder.name / "brightfield.yaml").is_file()
        assert (out / "dic" / "references" / folder.name / "dark_20ms.tif").is_file()
        import csv

        with open(out / "dic" / "dic.csv", newline="", encoding="utf-8") as fh:
            (row,) = list(csv.DictReader(fh))
        assert row["dark"] == f"references/{folder.name}/dark_20ms.tif"
        assert row["flat"] == f"references/{folder.name}/flat_led-1pct_20ms.tif"
        text = (out / "README.txt").read_text(encoding="utf-8")
        assert "corrected = (frame - dark) / (flat - dark) * mean(flat - dark)" in text


class TestThePane:
    def test_the_led_defaults_to_one_percent_and_the_panel_is_under_the_dic_fields(self):
        assert 'id="op-plan-dic-led" type="number" min="1" max="100" step="1" value="1"' in INDEX
        assert 'id="op-bfref-host"' in INDEX and "panels/brightfield-refs.js" in INDEX
        assert "BrightfieldRefs.mount('op-bfref-host')" in OPERATE
        for needle in (
            "'op-plan-dic-light'",
            "'op-plan-dic-led'",
            "'op-plan-dic-exposure'",
            "/api/brightfield/references/",
            "no embryo",
        ):
            assert needle in PANEL, needle


class TestReferencesTakenAfterTheRun:
    def test_frames_taken_before_the_references_existed_still_find_them_in_the_export(self, store):
        # The run's frames first, with no references to name …
        store.put_snapshot(
            "s1",
            "dic",
            np.zeros((6, 9), dtype=np.uint16),
            metadata={
                "channel": "dic",
                "frame": 1,
                "captured_at": "2026-10-06T21:00:00",
                "light": "led",
                "led_intensity_pct": 1,
                "exposure_ms": 20.0,
                "references": None,
            },
        )
        # … then the dark and flat, after the run, at the same light and exposure.
        spec = bf.ReferenceSpec(light="led", led_intensity_pct=1, exposure_ms=20.0)
        folder = bf.open_record(store, "s1", spec)
        img = np.full((4, 4), 100, dtype=np.uint16)
        bf.file_image(folder, "dark", img, spec, {"stats": bf.stats(img)})
        bf.file_image(folder, "flat", img * 20, spec, {"stats": bf.stats(img * 20), "frames": 5})
        # A set for a different exposure must not be picked.
        other = bf.ReferenceSpec(light="led", led_intensity_pct=1, exposure_ms=50.0)
        f2 = bf.open_record(store, "s1", other)
        bf.file_image(f2, "dark", img, other, {"stats": bf.stats(img)})
        bf.file_image(f2, "flat", img * 20, other, {"stats": bf.stats(img * 20), "frames": 5})

        out = export_session(store, "s1")
        import csv

        with open(out / "dic" / "dic.csv", newline="", encoding="utf-8") as fh:
            (row,) = list(csv.DictReader(fh))
        assert row["dark"] == f"references/{folder.name}/dark_20ms.tif"
        assert row["flat"] == f"references/{folder.name}/flat_led-1pct_20ms.tif"
