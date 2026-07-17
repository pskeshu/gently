# ruff: noqa: E501
"""US-43 — Multi-instance / mesh. As a user running several gently rigs, I want to
see and switch between peer instances on the mesh network from the UI."""

from _harness import count_text, dom_count, goto, skip_landing

META = {
    "id": "US-43",
    "title": "Multi-instance / mesh",
    "cluster": "14 Config, session & mesh",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    nav = await count_text(page, r"\bmesh\b|instances|\bpeers?\b|other rigs?|switch (rig|instance)")
    surfaces = await dom_count(
        page, '[data-tab="mesh"], [data-tab="instances"], [data-view="mesh"], [data-view="peers"]'
    )
    await rec.shot("workspace-no-mesh")
    if nav or surfaces:
        rec.partial(
            f"an unexpected mesh/instances affordance appeared (nav={nav}, surfaces={surfaces}) — verify it actually lists/switches peer instances"
        )
    else:
        rec.gap(
            "no multi-instance / mesh UI: mesh exists only server-side (peer-discovery events, a read-only 'mesh' block in Settings→Effective config, Advanced→'Mesh network' thresholds) — no surface to view, switch, or pair peer gently instances"
        )
