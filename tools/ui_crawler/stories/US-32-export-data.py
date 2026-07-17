# ruff: noqa: E501
"""US-32 — Export data. As a researcher, I want to export/download my acquired data and results (volumes, predictions, traces), so I can analyse them offline."""

from _harness import count_text, goto, skip_landing, tab

META = {
    "id": "US-32",
    "title": "Export data",
    "cluster": "10 Perception & results",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    rx = r"export data|download data|download all|export results|export dataset|export csv|download csv|\bdownload\b|\bexport\b"
    await tab(page, "embryos")
    emb = await count_text(page, rx)
    await tab(page, "plans")
    pln = await count_text(page, rx)
    await tab(page, "gallery")
    gal = await count_text(page, rx)
    await rec.shot("gallery-no-export")
    if emb + pln + gal:
        rec.ok(
            f"a data export/download control is present (embryos={emb}, plans={pln}, gallery={gal})"
        )
    else:
        rec.gap(
            "no export/download-data control across embryos, plans, or gallery — the only exports are the plan-markdown button (landing overlay) and a dashboard-prefs JSON dump (settings); acquired volumes/predictions/traces/results cannot be exported from the UI"
        )
