import json
from types import SimpleNamespace

from gently.app.tools.figure_tools import draw_figure, pool_rows
from gently.core.file_store import FileStore, _write_yaml
from gently.harness.memory.file_store import FileContextStore


def _project(tmp_path):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("Twitch to hatch", shorthand="t2h")
    for sid in ("s1", "s2"):
        store.create_session(sid, name="run", project_id=proj)
        sd = store._session_dir(sid)
        for n in (1, 2):
            meta = {"frame": n, "field": 0, "captured_at": f"2026-10-27T0{n}:00:00"}
            store.put_snapshot(sid, "bf", [[0]], metadata=meta, stem=f"f{n}")
        row = {"field": 0, "x": 0.1, "y": 0.1, "twitch_frame": 1, "hatch_frame": 2}
        _write_yaml(sd / "annotations" / "twitch_hatch.yaml", [row])
    return store, cs, proj


async def test_agent_pools_rows_with_unique_session_labels(tmp_path):
    store, cs, proj = _project(tmp_path)
    ctx = {"agent": SimpleNamespace(store=store, context_store=cs)}
    pooled = json.loads(await pool_rows(project="t2h", context=ctx))
    assert pooled["label"] == "t2h"
    assert [r["minutes"] for r in pooled["rows"]] == [60.0, 60.0]
    assert len({r["session_name"] for r in pooled["rows"]}) == 2  # same name, told apart
    assert (await pool_rows(tool="nothing", project="t2h", context=ctx)).startswith("Error")


async def test_agent_draws_default_and_its_own_code(tmp_path):
    store, cs, proj = _project(tmp_path)
    ctx = {"agent": SimpleNamespace(store=store, context_store=cs)}
    out = json.loads(
        await draw_figure(project="t2h", title="Twitch to hatch, control", context=ctx)
    )
    assert out["rows"] == 2 and out["reference"] == {"Sulston": 370.0}
    png = cs.campaign_dir(proj) / "figures" / "twitch-to-hatch-control.png"
    assert png.is_file() and png.read_bytes()[:4] == b"\x89PNG"
    assert out["url"] == f"/api/campaigns/{proj}/figures/twitch-to-hatch-control.png"

    code = "ax.hist([r['minutes'] for r in rows]); ax.axvline(ref['Sulston'])"
    out = json.loads(await draw_figure(project="t2h", title="hist", code=code, context=ctx))
    assert (cs.campaign_dir(proj) / "figures" / "hist.png").is_file()

    bad = await draw_figure(project="t2h", title="bad", code="ax.plot(undefined_name)", context=ctx)
    assert bad.startswith("Error: the plotting code failed") and "NameError" in bad
    assert (await draw_figure(project="nope", title="x", context=ctx)).startswith("Error: project")
