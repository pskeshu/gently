# ruff: noqa: E501
"""US-12 — Lower SPIM → focus → acquire. As an operator, I bring the objectives down onto the sample, tune the sheet, then acquire a volume."""

from _harness import dom_count, goto, skip_landing, tab, view

META = {
    "id": "US-12",
    "title": "Lower SPIM head, focus, acquire",
    "cluster": "6 Operate (acquire)",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")

    # Height and sheet tuning live on the SPIM head surface.
    on_spim = await view(page, "spim")
    await rec.shot("operate-spim-head")
    fdrive = await dom_count(page, "#op-fd-nudge [data-nudge]")  # position-banded F-drive ladder
    spim_view = await dom_count(page, "#op-spim-toggle")
    led = await dom_count(page, "#op-led")
    galvo = await dom_count(page, "[data-gv]")
    calibrate = await dom_count(page, "#op-calibrate")

    # Acquisition is its own surface, not the end of a sequence.
    on_acq = await view(page, "acquire")
    await rec.shot("operate-acquisition")
    single = await dom_count(page, '[data-mode="single"]')
    slices = await dom_count(page, "#op-vol-slices")
    start = await dom_count(page, "#op-run-start")

    if not (on_spim and fdrive and spim_view and on_acq and single and start):
        rec.gap("SPIM height/sheet controls or the single-volume acquisition control are missing")
        return
    rec.blocked(
        f"needs device: F-drive ladder ({fdrive} steps), light-sheet view + LED ({led}), sheet alignment ({galvo}), calibrate ({calibrate}) "
        f"and single-volume acquire (slices={slices}) are all present and live; motion and acquisition 502 with the device layer offline"
    )
