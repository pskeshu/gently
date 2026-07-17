# ruff: noqa: E501
"""US-44 — Auth: log in / gain control / view-only. As a remote user against an
account-enabled server, I want to watch in view-only and sign in to take control."""

from _harness import count_text, exists, goto

META = {
    "id": "US-44",
    "title": "Auth: log in / gain control / view-only",
    "cluster": "14 Config, session & mesh",
    "mode": "headless",
    "needs_account": True,
}


async def flow(page, url, rec):
    await goto(page, url, "/login")  # account server serves the sign-in surface
    me = await page.evaluate(
        "async () => { try { return await (await fetch('/api/auth/me')).json(); } catch (e) { return {}; } }"
    )
    form = await exists(page, "#login-form") and await exists(page, "#password")
    viewonly = await count_text(page, r"continue without signing in|view.?only")
    await rec.shot("auth-login-view-only")
    accounts = bool(me.get("accounts"))
    loggedout = accounts and not me.get("authenticated")
    if not accounts:
        rec.gap(
            "auth/me accounts=false: no account store, so localhost is always control — no view-only/login gating to exercise (needs the account server)"
        )
    elif form and viewonly and loggedout:
        rec.partial(
            "account server: logged-out → view-only confirmed; /login offers sign-in (gain control) + 'continue view-only'. But the workspace shows no discoverable sign-in affordance — control is only surfaced reactively via the 403 control-toast (control-auth.js) or the direct /login URL"
        )
    elif form:
        rec.partial(
            f"/login sign-in present but the view-only/continue option is missing (viewonly={viewonly}, loggedout={loggedout})"
        )
    else:
        rec.gap("no sign-in surface rendered at /login on the account server")
