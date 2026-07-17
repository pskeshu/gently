# ruff: noqa: E501
"""US-02 — Skip to workspace / resume a prior session. As a returning user, I skip the landing and resume a saved session, so I continue where I left off."""

from _harness import dom_count, exists, goto, tab

META = {
    "id": "US-02",
    "title": "Skip to workspace and resume a prior session",
    "cluster": "1 Onboarding",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    had_landing = await exists(page, "#v2-landing")
    if had_landing:
        await page.click("#v2-landing-skip")  # the story's skip affordance
        await page.wait_for_timeout(700)
    dismissed = not await exists(page, "#v2-landing")  # landing gone → workspace reached
    on_sessions = await tab(page, "sessions")  # ReviewApp.init() → GET /api/sessions
    await page.wait_for_timeout(1200)  # let the session list fetch+render
    listbox = await exists(page, "#session-list")
    items = await dom_count(page, ".session-item")
    resume = await dom_count(
        page, ".session-resume-btn"
    )  # "Resume in agent" per non-active session
    await rec.shot("sessions-list")
    if had_landing and dismissed and on_sessions and resume:
        rec.ok(
            f"skip → workspace; sessions tab lists {items} session(s) with {resume} Resume control(s)"
        )
    elif dismissed and on_sessions and listbox:
        rec.partial(
            f"skip→workspace + sessions list present, but no resumable prior session in dev data (items={items}, resume={resume})"
        )
    else:
        rec.gap(
            f"skip/resume path broken (landing={had_landing}, dismissed={dismissed}, sessions_tab={on_sessions}, list={listbox})"
        )
