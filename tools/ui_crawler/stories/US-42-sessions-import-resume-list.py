# ruff: noqa: E501
"""US-42 — Sessions: list / resume / import. As a user I want to browse past
sessions and resume one in the live agent (and, ideally, import a session from a
file or peer)."""

from _harness import count_text, exists, goto, skip_landing, tab

META = {
    "id": "US-42",
    "title": "Sessions: list, resume, import",
    "cluster": "14 Config, session & mesh",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "sessions")  # ReviewApp.init() → GET /api/sessions
    await page.wait_for_timeout(900)  # let the session list fetch settle
    listed = await exists(page, "#session-list") or await exists(page, "#session-sidebar")
    resume = await count_text(page, r"resume")  # per non-active session with content
    imp = await count_text(
        page, r"import"
    )  # session import (from file/peer) — expected absent here
    await rec.shot("sessions-list")
    if listed and resume:
        rec.ok(
            f"Sessions tab lists sessions with a per-session Resume ({resume}); no session import-from-file/peer affordance (import={imp})"
        )
    elif listed:
        rec.partial(
            f"session list surface present but no Resume to exercise headless (no session data; resume renders per non-active session); no import-from-file/peer affordance (import={imp})"
        )
    else:
        rec.gap("no session-list surface on the Sessions tab")
