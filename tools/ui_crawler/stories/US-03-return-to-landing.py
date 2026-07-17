# ruff: noqa: E501
"""US-03 — Return to the landing after dismissing it. As a user who dismissed the landing, I want to get back to it, so I can re-choose how to start."""

from _harness import exists, goto, skip_landing

META = {
    "id": "US-03",
    "title": "Return to the landing after dismissing it",
    "cluster": "1 Onboarding",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)  # dismiss → workspace
    dismissed = not await exists(page, "#v2-landing")
    brand = await exists(page, "a.header-logo")  # header "Gently Microscopy" → "/"
    if not brand:
        await rec.shot("no-return")
        rec.gap("no header brand / labelled control to return to the landing")
        return
    await page.click("a.header-logo")  # the only non-chat path back
    await page.wait_for_load_state("domcontentloaded")
    await page.wait_for_timeout(900)  # landing.js re-init + greet
    reshown = await exists(page, "#v2-landing")
    await rec.shot("landing-reshown")
    if reshown:
        rec.partial(
            f"header brand 'Gently Microscopy' → / re-shows the landing — functional but incidental (a logo click, not a labelled 'back to start'; dismissed-before={dismissed})"
        )
    else:
        rec.gap("header brand present but navigating to / did not re-show the landing overlay")
