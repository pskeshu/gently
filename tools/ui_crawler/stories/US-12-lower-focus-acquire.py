# ruff: noqa: E501
"""US-12 — Lower SPIM → focus → acquire. As an operator, I lower the SPIM head, focus the light-sheet objective, then acquire the volume for a centered embryo."""
from _harness import dom_count, goto, skip_landing, tab, view

META = {"id": "US-12", "title": "Lower SPIM → focus → acquire",
        "cluster": "6 Operate (acquire)", "mode": "rig", "needs_account": False}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    rail = await dom_count(page, "#op-rail")                     # operate surface loaded
    lower = await dom_count(page, "#op-fd-nudge [data-fd]")      # b2 Lower: SPIM head F-drive nudges
    tofocus = await dom_count(page, "#op-tofocus")               # b2 → Focus SPIM
    infocus = await dom_count(page, "#op-infocus")               # b3 mark-in-focus
    acquire = await dom_count(page, "#op-acquire")               # b4 Acquire volume
    # reveal the Acquire step group so the audit shot shows the culminating control
    await page.evaluate("() => { const r=document.getElementById('op-rail'); if (r) r.dataset.active='b4'; }")
    await rec.shot("acquire-step")
    if rail and lower and tofocus and infocus and acquire:
        rec.blocked("needs rig: Lower (F-drive nudges) + Focus SPIM (→) + mark-in-focus + Acquire volume controls are all wired, but driving lower→focus→acquire needs the SPIM device and a centered embryo")
    else:
        rec.gap(f"operate acquire-step controls missing (rail={rail}, lower={lower}, tofocus={tofocus}, infocus={infocus}, acquire={acquire})")
