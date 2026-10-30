// ══════════════════════════════════════════════════════════
//  Shared utilities – loaded before all other app scripts
// ══════════════════════════════════════════════════════════

// Tab and view name constants
const TABS = { HOME: 'home', EMBRYOS: 'embryos', CALIBRATION: 'calibration', EVENTS: 'events', PLANS: 'plans', SESSIONS: 'sessions', DEVICES: 'devices', EXPERIMENT: 'experiment', NOTEBOOK: 'notebook', GALLERY: 'gallery', SETTINGS: 'settings' };

/**
 * Extract the XY firmware fence (the addressable stage box) from a device-state
 * properties map. The ASI adapter exposes LowerLimX/UpperLimX/LowerLimY/
 * UpperLimY in mm; we convert to µm. Single source of truth for both the 2D
 * devices map and the 3D optical-space view.
 *
 * @param {Object} propsByDevice - payload.properties from DEVICE_STATE_UPDATE
 * @returns {{x:[number,number], y:[number,number]}|null}
 */
function extractFirmwareBox(propsByDevice) {
    if (!propsByDevice) return null;
    for (const name of Object.keys(propsByDevice)) {
        const p = propsByDevice[name] || {};
        const xMinMm = parseFloat(p['LowerLimX(mm)']);
        const xMaxMm = parseFloat(p['UpperLimX(mm)']);
        const yMinMm = parseFloat(p['LowerLimY(mm)']);
        const yMaxMm = parseFloat(p['UpperLimY(mm)']);
        if (isFinite(xMinMm) && isFinite(xMaxMm) &&
            isFinite(yMinMm) && isFinite(yMaxMm)) {
            return {
                x: [xMinMm * 1000, xMaxMm * 1000],   // mm → µm
                y: [yMinMm * 1000, yMaxMm * 1000],
            };
        }
    }
    return null;
}

/**
 * Build a coordinate mapper from device microns to Three.js scene units.
 * Centers each axis on its range midpoint and divides by the LARGEST span so
 * the whole scene fits a ~[-0.5, 0.5] cube while keeping axes proportional
 * (anisotropic Z vs XY preserved). Returns helpers used by all scene objects
 * so a single scale governs geometry and camera distance.
 *
 * @param {{xRange:[number,number], yRange:[number,number], zRange:[number,number]}} ranges (µm)
 */
function makeSceneScaler(ranges) {
    const xc = (ranges.xRange[0] + ranges.xRange[1]) / 2;
    const yc = (ranges.yRange[0] + ranges.yRange[1]) / 2;
    const zc = (ranges.zRange[0] + ranges.zRange[1]) / 2;
    const xs = Math.abs(ranges.xRange[1] - ranges.xRange[0]);
    const ys = Math.abs(ranges.yRange[1] - ranges.yRange[0]);
    const zs = Math.abs(ranges.zRange[1] - ranges.zRange[0]);
    const maxExtent = Math.max(xs, ys, zs, 1e-6);
    return {
        maxExtent,
        center: { x: xc, y: yc, z: zc },
        // Map an absolute µm position on one axis into scene space.
        toScene(um, axis) {
            const c = axis === 'x' ? xc : axis === 'y' ? yc : zc;
            return (um - c) / maxExtent;
        },
        // Map a µm length (span) into scene units (no centering).
        scaleLen(um) {
            return um / maxExtent;
        },
    };
}

/**
 * HTML-escape a string (safe for insertion into innerHTML).
 * Uses the browser's built-in text node escaping.
 */
const _ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, c => _ESC[c]);
}

/**
 * Format an ISO date string for display.
 * @param {string} isoStr
 * @returns {string} e.g. "Mar 14, 2:30 PM"
 */
const _DAY = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const _TIME = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit' });
function formatDate(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        return _DAY.format(d) + ', ' + _TIME.format(d);
    } catch { return isoStr; }
}

/**
 * Initialize a view-switcher: click delegation + keyboard shortcuts.
 *
 * @param {string}   containerId  - ID of the `.view-switcher` container element.
 * @param {Function} callback     - Called with the view name string when a view is selected.
 * @param {Object}   [opts]
 * @param {string[]} [opts.views] - Ordered view names for keyboard shortcuts (keys 1-N).
 * @param {Function} [opts.guard] - Return true to allow shortcuts (e.g., check active tab).
 */
function initViewSwitcher(containerId, callback, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.addEventListener('click', e => {
        const btn = e.target.closest('[data-view]');
        if (btn) callback(btn.dataset.view);
    });
    const views = opts?.views;
    const guard = opts?.guard;
    if (views && views.length) {
        document.addEventListener('keydown', e => {
            if (guard && !guard()) return;
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const idx = parseInt(e.key) - 1;
            if (idx >= 0 && idx < views.length) callback(views[idx]);
        });
    }
}

/**
 * Update view-switcher button active states.
 * @param {string} containerId - ID of the `.view-switcher` container.
 * @param {string} activeView  - The data-view value to mark active.
 */
function updateViewButtons(containerId, activeView) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    container.querySelector(`[data-view="${activeView}"]`)?.classList.add('active');
}

/**
 * Toggle a dropdown element's visibility and register a one-shot
 * outside-click handler to close it.
 *
 * @param {HTMLElement} dropdownEl  - The element to show/hide (via the `hidden` class).
 * @param {Event}       event      - The triggering click event (will be stopPropagated).
 */
const toggleDropdown = (() => {
    // Track the current close handler so double-opens don't leak listeners.
    let _activeClose = null;

    return function toggleDropdown(dropdownEl, event) {
        event.stopPropagation();
        if (!dropdownEl) return;

        // If there is already a close handler for a *different* dropdown, fire it first.
        if (_activeClose && _activeClose._dropdown !== dropdownEl) {
            _activeClose._dropdown.classList.add('hidden');
            document.removeEventListener('click', _activeClose);
            _activeClose = null;
        }

        dropdownEl.classList.toggle('hidden');

        if (!dropdownEl.classList.contains('hidden')) {
            const close = (e) => {
                if (!dropdownEl.contains(e.target)) {
                    dropdownEl.classList.add('hidden');
                    document.removeEventListener('click', close);
                    _activeClose = null;
                }
            };
            close._dropdown = dropdownEl;
            _activeClose = close;
            setTimeout(() => document.addEventListener('click', close), 0);
        } else {
            // Toggled closed manually – clean up any pending listener.
            if (_activeClose && _activeClose._dropdown === dropdownEl) {
                document.removeEventListener('click', _activeClose);
                _activeClose = null;
            }
        }
    };
})();

/**
 * Who — a person as the pages show them: the display name in the person's
 * own colour, the same colour the presence indicator uses. Ids come from
 * records (`by`, `operators`, `asked_by`); names and colours from
 * /api/auth/users, loaded once. An id nobody knows shows as itself, grey.
 */
const Who = {
    people: {},
    ready: null,
    load() {
        if (!this.ready) {
            this.ready = fetch('/api/auth/users').then(r => r.json()).then(d => {
                (d.users || []).forEach(u => { this.people[u.id] = u; });
            }).catch(() => {});
        }
        return this.ready;
    },
    name(id) { const u = this.people[id]; return u ? u.name : (id || ''); },
    color(id) { const u = this.people[id]; return (u && u.color) || '#8b949e'; },
    initial(id) { const n = this.name(id); return (n || '?').trim().charAt(0).toUpperCase(); },
    // The person as a line's author: their initial in their colour, then their name.
    // This is the head of a line; a mention inside the words is Mentions.html.
    head(id) {
        if (id === 'agent') return `<span class="rd-avatar is-agent" aria-hidden="true">A</span><b class="rd-who is-agent">agent</b>`;
        if (!id) return `<span class="rd-avatar is-nobody" aria-hidden="true">·</span><b class="rd-who is-nobody">someone</b>`;
        const c = escapeHtml(this.color(id));
        return `<span class="rd-avatar" style="--who:${c}" aria-hidden="true">${escapeHtml(this.initial(id))}</span><b class="rd-who" style="--who:${c}" title="${escapeHtml(id)}">${escapeHtml(this.name(id))}</b>`;
    },
    chip(id, opts = {}) {
        if (!id) return opts.empty || '';
        const name = id === 'agent' ? 'agent' : this.name(id);
        const when = opts.when ? ` <span class="who-when">${escapeHtml(formatDate(opts.when))}</span>` : '';
        return `<span class="who-chip" style="--who:${escapeHtml(this.color(id))}" title="${escapeHtml(id)}"><i></i>${escapeHtml(name)}</span>${when}`;
    },
};

/**
 * Mentions — "@ryan" in a line is an ask. Typing @ in a box offers the rig's
 * people; a saved line draws each person it asks as their chip. The server
 * decides which @words are people (the note's `mentions`); the box only
 * helps type them.
 */
const Mentions = {
    RE: /(^|[^\w@])@([a-z0-9][a-z0-9-]*)/gi,
    // The line as html: the people it asks as chips, everything else escaped.
    html(text, mentions) {
        const asked = new Set(mentions || []);
        const s = text || '';
        let out = '', last = 0, m;
        const re = new RegExp(this.RE.source, 'gi');
        while ((m = re.exec(s)) !== null) {
            const id = m[2].replace(/-+$/, '');
            if (!asked.has(id)) continue;
            const at = m.index + m[1].length;
            out += escapeHtml(s.slice(last, at))
                + `<b class="rd-mention" style="--who:${escapeHtml(Who.color(id))}" title="${escapeHtml(id)}">@${escapeHtml(Who.name(id))}</b>`;
            last = at + 1 + id.length;
        }
        return out + escapeHtml(s.slice(last));
    },
    // Offer people while the caret is in an @word; Enter, Tab or a click completes it.
    attach(input) {
        if (!input || input._mentions) return;
        input._mentions = true;
        let pop = null, hits = [], sel = 0;
        const close = () => { if (pop) pop.remove(); pop = null; hits = []; };
        const word = () => {
            const m = /(^|[^\w@])@([a-z0-9-]*)$/i.exec(input.value.slice(0, input.selectionStart));
            return m ? { start: input.selectionStart - m[2].length - 1, q: m[2].toLowerCase() } : null;
        };
        const pick = i => {
            const w = word(), u = hits[i];
            if (!w || !u) return;
            const v = input.value, c = input.selectionStart;
            input.value = v.slice(0, w.start) + '@' + u.id + ' ' + v.slice(c);
            const at = w.start + u.id.length + 2;
            input.setSelectionRange(at, at);
            close();
        };
        const draw = () => {
            const w = word();
            if (!w) { close(); return; }
            hits = Object.values(Who.people).filter(u => u.id !== 'agent'
                && (u.id.toLowerCase().startsWith(w.q) || (u.name || '').toLowerCase().startsWith(w.q))).slice(0, 6);
            if (!hits.length) { close(); return; }
            sel = Math.min(sel, hits.length - 1);
            if (!pop) { pop = document.createElement('div'); pop.className = 'mention-pop'; input.insertAdjacentElement('afterend', pop); }
            pop.innerHTML = hits.map((u, i) => `<button type="button" class="mention-opt${i === sel ? ' on' : ''}" data-i="${i}">${Who.chip(u.id)}</button>`).join('');
            pop.querySelectorAll('.mention-opt').forEach(b => { b.onmousedown = e => { e.preventDefault(); pick(Number(b.dataset.i)); }; });
        };
        input.addEventListener('input', () => { sel = 0; draw(); });
        // Before the box's own Enter: while the list is up, Enter completes the name.
        input.addEventListener('keydown', e => {
            if (!pop) return;
            if (e.key === 'ArrowDown') { sel = (sel + 1) % hits.length; draw(); e.preventDefault(); }
            else if (e.key === 'ArrowUp') { sel = (sel + hits.length - 1) % hits.length; draw(); e.preventDefault(); }
            else if (e.key === 'Enter' || e.key === 'Tab') { pick(sel); e.preventDefault(); e.stopImmediatePropagation(); }
            else if (e.key === 'Escape') { close(); e.stopImmediatePropagation(); }
        }, true);
        input.addEventListener('blur', () => setTimeout(close, 150));
    },
};

/**
 * One stream line, the same on a project's Overview and on a session: when,
 * who, a tool's title as a chip or a person's headline as the lead, the
 * words with the people they ask as chips, and the session it cites.
 */
function streamLineHtml(n, toolTitles, opts = {}) {
    const isTool = !!(toolTitles && n.title && toolTitles.has(n.title));
    const lead = n.title && !isTool ? `<b>${escapeHtml(n.title)}</b>: ` : '';
    const who = n.author === 'agent' ? Who.head('agent') : Who.head(n.by);
    const cite = opts.cite === false || !(n.sessions || []).length ? ''
        : `<span class="pj-cite" title="The session this was written on">${n.sessions.map(x => escapeHtml(String(x).slice(0, 8))).join(' ')}</span>`;
    return `<div class="rd-line${isTool ? ' is-tool' : ''}${n.author === 'agent' ? ' is-agent' : ''}">
        <div class="rd-head">${who}${isTool ? `<span class="pj-chip">${escapeHtml(n.title)}</span>` : ''}${n.kind && n.kind !== 'observation' ? `<span class="pj-chip">${escapeHtml(n.kind)}</span>` : ''}<span class="rd-when">${escapeHtml(formatDate(n.created_at))}</span>${cite}</div>
        <div class="rd-text">${lead}${Mentions.html(n.body, n.mentions)}</div>
    </div>`;
}

