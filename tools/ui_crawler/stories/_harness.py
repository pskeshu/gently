# ruff: noqa: E501
"""Shared helpers + verdict object for per-story UI-audit flows.

Each story lives in its own tools/ui_crawler/stories/US-XX-*.py, defines a META
dict and an `async def flow(page, url, rec)`, and uses these helpers to drive the
story's intended path and record a verdict via `rec`. The runner (run_stories.py)
discovers the files, runs each flow in its own Playwright trace, and collects the
verdicts into a status report.
"""

from __future__ import annotations

import asyncio


class Rec:
    """A story verdict + agent-readable artifacts.

    A flow calls exactly one of ok/partial/gap/blocked, and may call
    `await rec.shot(name)` at key moments to save a full-page PNG. The runner
    always captures a final screenshot + the final visible text, so every story
    produces images AND text an agent (not just show-trace) can read.
    """

    def __init__(self):
        self.status = None  # works | partial | gap | blocked
        self.observed = ""
        self.console = []  # console errors seen during the flow
        self.shots = []  # PNG filenames captured for this story
        self._page = None  # set by the runner
        self._dir = None  # shots dir (Path), set by the runner
        self._id = None  # story id, set by the runner

    def ok(self, msg):
        self.status, self.observed = "works", msg

    def partial(self, msg):
        self.status, self.observed = "partial", msg

    def gap(self, msg):
        """The story cannot be accomplished — a missing/undiscoverable affordance."""
        self.status, self.observed = "gap", msg

    def blocked(self, msg):
        """Can't be judged headless — needs a device, a live agent turn, etc."""
        self.status, self.observed = "blocked", msg

    async def shot(self, name):
        """Save a full-page PNG at a named step (agent-readable via the Read tool)."""
        if self._page is None or self._dir is None:
            return
        fn = f"{self._id}-{name}.png"
        try:
            await self._page.screenshot(path=str(self._dir / fn), full_page=True)
            self.shots.append(fn)
        except Exception:
            pass


async def goto(page, url, path="/"):
    await page.goto(url.rstrip("/") + path, wait_until="domcontentloaded", timeout=12000)
    await asyncio.sleep(0.7)


async def skip_landing(page):
    """Dismiss the landing overlay and WAIT until it's actually gone.

    The dismiss has a CSS transition (adds `.dismissed`, fades opacity, then
    display:none), so a fixed short sleep can race it and leave the overlay
    reading as 'still visible'. Poll until it's dismissed/hidden."""
    try:
        await page.evaluate(
            "() => { const s=document.getElementById('v2-landing-skip'); if (s) s.click(); }"
        )
        for _ in range(25):
            gone = await page.evaluate(
                "() => { const l=document.getElementById('v2-landing'); if (!l) return true;"
                " const cs=getComputedStyle(l);"
                " return l.classList.contains('dismissed') || cs.display==='none' || parseFloat(cs.opacity||'1')===0; }"
            )
            if gone:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)
    except Exception:
        pass


async def tab(page, name):
    """Switch to a top-level tab (data-tab). Returns True if it activated.

    A legacy hidden navbar (_navbar.html) carries duplicate data-tab elements
    that precede the visible v2 nav, so target the :visible match, not the first.
    """
    try:
        await page.locator(f'[data-tab="{name}"]:visible').first.click(timeout=6000)
        await asyncio.sleep(0.6)
        return True
    except Exception:
        return False


async def view(page, name):
    """Switch a sub-view (data-view, e.g. devices operate/manual). True if clicked.

    Some data-view names (e.g. board) appear in more than one view-switcher, so
    target the :visible match on the active tab, not merely the first in the DOM.
    """
    try:
        await page.locator(f'[data-view="{name}"]:visible').first.click(timeout=6000)
        await asyncio.sleep(0.6)
        return True
    except Exception:
        return False


async def click_text(page, regex):
    """Click the first visible control whose text matches regex. True if clicked."""
    try:
        await page.click(f"text=/{regex}/i", timeout=4000)
        await asyncio.sleep(0.6)
        return True
    except Exception:
        return False


async def count_text(page, regex):
    """Count visible clickable controls whose text matches regex (case-insensitive)."""
    return await page.evaluate(
        """(q) => {
          const vis = el => { const cs=getComputedStyle(el); if (cs.display==='none'||cs.visibility==='hidden') return false;
            const r=el.getBoundingClientRect(); return r.width>1 && r.height>1; };
          const re = new RegExp(q, 'i');
          return [...document.querySelectorAll('button, a, [role=button], .btn')].filter(e => vis(e) && re.test(e.textContent||'')).length;
        }""",
        regex,
    )


async def exists(page, selector):
    """True if a selector matches a VISIBLE element (has a rendered box)."""
    return await page.evaluate(
        """(sel) => { const el=document.querySelector(sel); if (!el) return false;
          const cs=getComputedStyle(el); if (cs.display==='none'||cs.visibility==='hidden'||parseFloat(cs.opacity||'1')===0) return false;
          const r=el.getBoundingClientRect(); return r.width>1 && r.height>1; }""",
        selector,
    )


async def present(page, selector):
    """True if a selector matches an element in the DOM that isn't display:none.

    Use for form controls (radios/checkboxes) that are custom-styled to ~0 size —
    `exists`'s box check would wrongly reject them."""
    return await page.evaluate(
        """(sel) => { const el=document.querySelector(sel); if (!el) return false;
          return getComputedStyle(el).display !== 'none'; }""",
        selector,
    )


async def dom_count(page, selector):
    """Count elements matching a selector in the DOM, ignoring visibility.

    Use for custom-styled inputs (e.g. `display:none` radios whose visible part is
    a styled label) where presence, not a rendered box, is what matters."""
    return await page.evaluate("(s) => document.querySelectorAll(s).length", selector)
