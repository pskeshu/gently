/**
 * Reveal — show a thing Gently keeps, in the file manager or in Fiji.
 *
 *     Reveal.button({ what: 'embryo', embryo_id: 'embryo_2' })
 *     Reveal.button({ what: 'timepoint', embryo_id: 'embryo_2', timepoint: 14 }, 'fiji')
 *     Reveal.run({ what: 'session', session_id: '6f090787' })
 *
 * WHY
 *
 * Everything Gently keeps is a file a person can browse, and the way to one
 * was to know the layout and walk to it. There was a folder button in the
 * header and another in the Calibration pane, each with its own route and
 * its own function. This is the one of them.
 *
 * WHAT IT SENDS
 *
 * What the thing is, never where it is: `{what, session_id, embryo_id,
 * timepoint, ...}`. The server finds it through the store. No path goes up.
 *
 * WHERE THE WINDOW OPENS
 *
 * On the computer Gently runs on. Under the desktop app that is this screen.
 * From a browser on another computer a window would open on the microscope,
 * so the server sends the path back instead and it is copied here.
 */
const Reveal = (() => {
    'use strict';

    let _about = null;
    let _asking = null;

    const FOLDER = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"'
        + ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        + '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>';

    function say(msg, level) {
        if (typeof showGentlyToast !== 'function') return;
        showGentlyToast(msg, null, null, level === 'error' ? 9000 : 5000, level || 'success');
    }

    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    /**
     * What this browser can ask for. Said on <html>, so a button drawn later
     * is already right: Fiji's is shown by a stylesheet rule, not by a pass
     * over the buttons that existed when the answer came.
     */
    function about() {
        if (_about) return Promise.resolve(_about);
        if (_asking) return _asking;
        _asking = fetch('/api/reveal/about')
            .then(r => (r.ok ? r.json() : null))
            .catch(() => null)
            .then(d => {
                _about = d || { local: false, fiji: { found: false }, file_manager: 'the file manager' };
                _asking = null;
                const root = document.documentElement;
                root.dataset.revealLocal = _about.local ? '1' : '0';
                root.dataset.revealFiji = _about.local && _about.fiji && _about.fiji.found ? '1' : '0';
                return _about;
            });
        return _asking;
    }

    async function copy(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (e) {
            return false;
        }
    }

    async function post(what, action) {
        const res = await fetch('/api/reveal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...what, action }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            let detail = data.detail != null ? data.detail : data.error;
            if (Array.isArray(detail)) detail = detail.map(x => (x && x.msg) || '').join('; ');
            throw new Error(detail || String(res.status));
        }
        return data;
    }

    /** Is it on disk? The answer, or null. Says nothing on the page. */
    async function probe(what) {
        if (!what) return null;
        try {
            return await post(what, 'path');
        } catch (e) {
            return null;
        }
    }

    /** Show it, or open it in Fiji. Says what happened. */
    async function run(what, action) {
        const act = action || 'show';
        let data;
        try {
            data = await post(what, act);
        } catch (e) {
            say(act === 'fiji' ? `Not opened in Fiji: ${e.message}` : `Not opened: ${e.message}`, 'error');
            return null;
        }
        if (data.opened) {
            const name = String(data.path).split(/[\\/]/).pop();
            say(act === 'fiji' ? `Opening ${name} in Fiji` : `Opened ${data.path}`);
        } else if (data.reason === 'remote') {
            const copied = await copy(data.path);
            say(copied
                ? `Path copied. It is on the microscope computer: ${data.path}`
                : `It is on the microscope computer: ${data.path}`);
        }
        return data;
    }

    /**
     * What a viewer's image is, from what the viewer knows about it. An image
     * that is only in memory (a live frame, a plot that was never kept) is
     * nothing on disk, and is null.
     */
    function describe(img) {
        if (!img) return null;
        if (img.reveal) return img.reveal;
        const url = String(img.url || '');
        let m = url.match(/\/api\/calibration\/records\/([^/]+)\/([^/]+)\/image\/(\d+)\.png/);
        if (m) {
            return {
                what: 'calibration_image',
                embryo_id: decodeURIComponent(m[1]),
                run: decodeURIComponent(m[2]),
                n: Number(m[3]),
            };
        }
        m = url.match(/\/api\/dic\/frames\/([^/?]+)\.png/);
        if (m) return { what: 'dic', stem: decodeURIComponent(m[1]) };
        const md = img.metadata || {};
        const tp = md.timepoint;
        if (md.embryo_id && tp !== undefined && tp !== null && Number.isFinite(Number(tp))) {
            const what = { what: 'timepoint', embryo_id: md.embryo_id, timepoint: Number(tp) };
            if (md.session_id) what.session_id = md.session_id;
            return what;
        }
        return null;
    }

    /**
     * A button. `action` is 'show' (the default) or 'fiji'.
     * opts: { label, title, cls }.
     */
    function button(what, action, opts) {
        const o = opts || {};
        const act = action || 'show';
        const fiji = act === 'fiji';
        const label = o.label != null ? o.label : (fiji ? 'Fiji' : '');
        const title = o.title || (fiji ? 'Open in Fiji' : 'Show in the file manager');
        return `<button type="button" class="reveal-btn${fiji ? ' reveal-fiji' : ''}${o.cls ? ' ' + esc(o.cls) : ''}"`
            + ` data-reveal="${esc(JSON.stringify(what))}" data-reveal-action="${act}"`
            + ` title="${esc(title)}" aria-label="${esc(title)}">`
            + `${fiji ? '' : FOLDER}${label ? `<span>${esc(label)}</span>` : ''}</button>`;
    }

    /** Both buttons for an image, once it is known to be on disk. */
    async function fill(host, what, opts) {
        if (!host) return;
        const key = what ? JSON.stringify(what) : '';
        host.dataset.revealFor = key;
        host.innerHTML = '';
        if (!what) return;
        const found = await probe(what);
        // The viewer moved on while the server was answering.
        if (host.dataset.revealFor !== key || !found) return;
        const o = opts || {};
        host.innerHTML = button(what, 'show', { label: o.showLabel != null ? o.showLabel : 'Show file', title: found.path })
            + (found.kind === 'file' ? button(what, 'fiji', { label: 'Open in Fiji' }) : '');
    }

    // In the capture phase: a button inside a row that opens on click must
    // not also open the row.
    document.addEventListener('click', e => {
        const b = e.target && e.target.closest ? e.target.closest('[data-reveal]') : null;
        if (!b) return;
        e.preventDefault();
        e.stopPropagation();
        let what = null;
        try { what = JSON.parse(b.dataset.reveal); } catch (err) { what = null; }
        if (what) run(what, b.dataset.revealAction || 'show');
    }, true);

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', about);
    else about();

    return { about, run, probe, describe, button, fill };
})();
