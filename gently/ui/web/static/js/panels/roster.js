/**
 * Roster — the embryo list, rendered once and mounted wherever it is needed.
 *
 *     RosterPanel.mount('op-erail-list', { actions: ['remove'] });
 *     RosterPanel.mount('op-roster',     { actions: ['role', 'centre', 'remove'],
 *                                          emptyAction: 'bottom' });
 *
 * WHY
 *
 * There were three renderings of one roster: `renderEmbryoRail` and
 * `renderRoster` in operate.js, plus a count badge in embryos.js. The two in
 * Operate were ~80% the same code, each keeping its own count element — which
 * is the root of #129, counts disagreeing across surfaces.
 *
 * Worse than the duplication, the action sets differed **arbitrarily**. On
 * Bottom cam you could delete an embryo but not centre on it; on Acquisition
 * you could centre and assign a role but not delete. Nothing about either pane
 * justified the split — it is just where each button happened to be added.
 * Actions are now declared per mount, so the difference is a decision someone
 * made rather than an accident of history.
 *
 * WHAT IT DOES NOT OWN
 *
 * The roster itself. `_embryos`, the selection and every verb stay in
 * operate.js, next to the frame geometry and the endpoints they depend on.
 * This renders `SharedState.embryos` and calls `OperateManager.roster.*` — the
 * same split as the Marking panel, and the reason both can mount in the
 * Atrium's EMBRYOS window where there is no canvas.
 */
const RosterPanel = (() => {
    'use strict';

    const mounts = new Map();   // hostId -> opts

    // label, then a cell per action. One row style at both widths — the narrow
    // rail simply declares fewer actions than the Acquisition pane.
    const ACTIONS = {
        remove: {
            cls: 'rp-del',
            text: '×',
            title: 'Remove this embryo (false positive)',
            verb: 'remove',
        },
        centre: {
            cls: 'rp-centre',
            text: 'Centre',
            title: 'Centre the stage on this embryo',
            verb: 'centre',
        },
        // Label and styling depend on the row, so this one is built by render().
        role: { cls: 'rp-role', verb: 'toggleRole' },
    };

    function mount(hostId, opts) {
        mounts.set(hostId, {
            actions: (opts && opts.actions) || [],
            emptyAction: opts && opts.emptyAction,
        });
        if (mounts.size === 1) {
            SharedState.on('embryos', render);
            SharedState.on('selectedEmbryoId', render);
        }
        render();
    }

    function unmount(hostId) { mounts.delete(hostId); }

    const verbs = () =>
        (typeof OperateManager !== 'undefined' && OperateManager.roster) || null;

    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, c =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    /** Fine position beats coarse, same order operate.js resolves in. */
    function xyOf(emb) {
        const f = emb && emb.position_fine;
        if (f && Number.isFinite(f.x) && Number.isFinite(f.y)) return f;
        const c = emb && emb.position_coarse;
        if (c && Number.isFinite(c.x) && Number.isFinite(c.y)) return c;
        return null;
    }

    function labelOf(emb) {
        const m = emb && emb.id && String(emb.id).match(/(\d+)/);
        return m ? m[1] : '?';
    }

    function render() {
        const embryos = SharedState.get('embryos') || [];
        const selected = SharedState.get('selectedEmbryoId');

        mounts.forEach((opts, hostId) => {
            const host = document.getElementById(hostId);
            if (!host) return;
            host.innerHTML = embryos.length
                ? embryos.map(e => row(e, selected, opts)).join('')
                : empty(opts);
            wire(host);
        });
    }

    function empty(opts) {
        // The actionable empty state used to live only on Acquisition, so the
        // pane you were already on described the fix without offering it.
        // Every mount can now offer the way forward.
        const cta = opts.emptyAction === 'bottom'
            ? '<button class="rp-cta" type="button" data-goto="bottom">Go to Bottom cam</button>'
            : '';
        return `<div class="rp-empty">No embryos yet — detect on the bottom camera,
                then register.${cta}</div>`;
    }

    function row(emb, selected, opts) {
        const xy = xyOf(emb);
        const role = (emb.role && emb.role !== 'unassigned') ? emb.role : 'test';
        const isRef = role === 'calibration';

        const buttons = opts.actions.map(name => {
            const a = ACTIONS[name];
            if (!a) return '';
            if (name === 'role') {
                return `<button class="rp-btn rp-role${isRef ? ' is-reference' : ''}" type="button"
                         title="${isRef ? 'Reference embryo — click to make it a subject'
                                        : 'Subject embryo — click to make it a reference'}"
                         data-verb="toggleRole" data-id="${esc(emb.id)}"
                         >${isRef ? 'ref' : 'subj'}</button>`;
            }
            return `<button class="rp-btn ${a.cls}" type="button" title="${a.title}"
                     data-verb="${a.verb}" data-id="${esc(emb.id)}">${a.text}</button>`;
        }).join('');

        return `<div class="rp-row${emb.id === selected ? ' is-sel' : ''}" tabindex="0"
                 data-embryo="${esc(emb.id)}">
                  <span class="rp-main">
                    <span class="rp-label">Embryo ${esc(labelOf(emb))}</span>
                    <span class="rp-xy">${xy ? `${xy.x.toFixed(0)}, ${xy.y.toFixed(0)}` : '—'}</span>
                  </span>
                  <span class="rp-acts">${buttons}</span>
                </div>`;
    }

    function wire(host) {
        host.onclick = e => {
            const v = verbs();
            // An action button first: it sits inside the row, and a click on it
            // must not also select. This was a live bug on the old rail.
            const act = e.target.closest('[data-verb]');
            if (act) {
                e.stopPropagation();
                const fn = v && v[act.dataset.verb];
                if (typeof fn === 'function') fn(act.dataset.id);
                return;
            }
            const goto = e.target.closest('[data-goto]');
            if (goto) { if (v && v.goTo) v.goTo(goto.dataset.goto); return; }
            const row = e.target.closest('[data-embryo]');
            if (row && v && v.select) v.select(row.dataset.embryo);
        };
        host.onkeydown = e => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const row = e.target.closest('[data-embryo]');
            if (!row) return;
            e.preventDefault();
            const v = verbs();
            if (v && v.select) v.select(row.dataset.embryo);
        };
    }

    return { mount, unmount, render, ACTIONS };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = RosterPanel;
