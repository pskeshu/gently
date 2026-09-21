/**
 * Marking — turning a camera frame into a registered embryo roster.
 *
 * Fourth panel under docs/architecture/PANELS.md.
 *
 *     MarkingPanel.mount('op-marking-host');
 *
 * WHY
 *
 * #113, from the 2026-08-07 walkthrough. Three defects, all one cause — two
 * different counts were being shown as one number:
 *
 *   MARKED     pending marks on the frame, not yet sent anywhere
 *   REGISTERED embryos the server holds
 *
 * `renderMarkCount` greyed Register and Clear whenever `_markers.length === 0`.
 * So after registering, the panel read "MARKED 0" with both buttons dead, while
 * the caption said "Registered 4 embryos" — and there was no visible way to
 * clear or re-register the roster that existed. The operator is told about a
 * roster they cannot act on.
 *
 * Detection also ran behind a bare "DETECTING…" spinner with no count, no
 * elapsed time and no way to tell a slow SAM load from a hung one. It counts
 * out loud now, the same as Calibrate.
 *
 * WHAT IT DOES NOT OWN
 *
 * The marking itself. `_markers`, the canvas and the frame geometry stay in
 * operate.js, because marks are placed in stage coordinates derived from the
 * live frame and that arithmetic belongs with the pixels. This panel renders
 * `SharedState.marking` and calls `OperateManager.marking.*` — the readout and
 * the verbs, which is the part that needs to exist away from the canvas (the
 * Atrium's EMBRYOS window has no canvas at all).
 */
const MarkingPanel = (() => {
    'use strict';

    const hosts = new Set();
    let ticker = null;

    // Detection is a three-stage pipeline, and which stages run is the
    // operator's call:
    //
    //   blobs   flat-field + scale-matched blob finder. Sets RECALL — every
    //           later stage only removes or refines what this proposes.
    //   claude  classifies each candidate crop and drops the ones that are not
    //           embryos. Removes only, never adds. Needs an API key.
    //   sam     segments inside the boxes it is handed, for an outline and an
    //           area. Needs the checkpoint on the device layer.
    //
    // Sensitivity is the blob finder's min_relative_peak: how strong a
    // candidate must be next to the strongest one on the frame. Permissive
    // proposes more and leans on the filter; strict cuts at the source.
    const SETTINGS_KEY = 'gently.detect.settings';
    const SENSITIVITY = [
        { id: 'permissive', label: 'Permissive', peak: 0, hint: 'Propose more, let the filter cut' },
        { id: 'balanced', label: 'Balanced', peak: 0.35, hint: 'Middle ground' },
        { id: 'strict', label: 'Strict', peak: 0.6, hint: 'Only strong blobs' },
    ];
    const DEFAULTS = { claude: true, sam: true, sensitivity: 'permissive', fresh: false };
    let settings = Object.assign({}, DEFAULTS);
    try {
        const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
        if (saved && typeof saved === 'object') settings = Object.assign({}, DEFAULTS, saved);
    } catch (e) { /* defaults */ }

    function saveSettings() {
        try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch (e) { /* not fatal */ }
    }

    function sensitivity() {
        return SENSITIVITY.find(x => x.id === settings.sensitivity) || SENSITIVITY[0];
    }

    // What the Detect verb should run. operate.js owns the request; this panel
    // owns the choice.
    function detectOptions() {
        return {
            use_claude_review: !!settings.claude,
            use_sam: !!settings.sam,
            min_relative_peak: sensitivity().peak,
            fresh: !!settings.fresh,
        };
    }

    const state = () => SharedState.get('marking') || {};
    const verbs = () =>
        (typeof OperateManager !== 'undefined' && OperateManager.marking) || null;

    function mount(hostId) {
        hosts.add(hostId);
        if (hosts.size === 1) SharedState.on('marking', render);
        render();
    }

    function unmount(hostId) {
        hosts.delete(hostId);
        if (!hosts.size) stopTicker();
    }

    // Only runs while something is running — a 1 Hz timer for a static readout
    // is a wakeup a second for nothing.
    function startTicker() {
        if (ticker) return;
        ticker = setInterval(render, 1000);
    }
    function stopTicker() {
        if (!ticker) return;
        clearInterval(ticker);
        ticker = null;
    }

    function elapsed(s) {
        if (!s.detecting || !s.startedAt) return null;
        return Math.round((Date.now() - s.startedAt) / 1000);
    }

    function render() {
        const s = state();
        if (s.detecting) startTicker(); else stopTicker();
        const secs = elapsed(s);
        const marked = s.marked || 0;
        const registered = s.registered || 0;

        hosts.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            const sens = sensitivity();
            el.innerHTML = `
              <div class="lp">
                <div class="lp-head">
                  <span class="lp-title">Detect</span>
                  ${s.detecting
                    ? `<span class="mk-busy">detecting… ${secs}s</span>`
                    : ''}
                </div>

                <div class="mk-pipe" role="group" aria-label="Detection pipeline">
                  <span class="mk-stage is-fixed" title="Flat-field + blob candidate finder. Sets recall: the later stages only remove or refine what this proposes.">Blobs</span>
                  <span class="mk-arrow" aria-hidden="true">›</span>
                  <button type="button" class="mk-stage" data-set="claude" aria-pressed="${settings.claude}"
                          title="Claude classifies each candidate crop and drops the ones that are not embryos. Removes only; needs an API key.">Claude filter</button>
                  <span class="mk-arrow" aria-hidden="true">›</span>
                  <button type="button" class="mk-stage" data-set="sam" aria-pressed="${settings.sam}"
                          title="SAM segments inside each candidate box for an outline and an area. Needs the checkpoint on the device layer.">SAM outline</button>
                </div>

                <div class="mk-opts">
                  <label class="mk-opt">
                    <span class="mk-opt-cap">Sensitivity</span>
                    <select class="mk-select" data-set="sensitivity" title="${escape(sens.hint)}">
                      ${SENSITIVITY.map(o => `<option value="${o.id}"${o.id === settings.sensitivity ? ' selected' : ''}>${o.label}</option>`).join('')}
                    </select>
                  </label>
                  <label class="mk-opt mk-opt-check" title="Capture a new frame instead of detecting on the one already on screen.">
                    <input type="checkbox" data-set="fresh"${settings.fresh ? ' checked' : ''}>
                    <span class="mk-opt-cap">New capture</span>
                  </label>
                </div>

                <div class="mk-counts">
                  <div class="mk-count">
                    <b class="mk-n">${marked}</b>
                    <span class="mk-cap">marked</span>
                  </div>
                  <div class="mk-count">
                    <b class="mk-n">${registered}</b>
                    <span class="mk-cap">registered</span>
                  </div>
                </div>

                <div class="mk-acts">
                  <button class="lp-btn mk-detect" data-act="detect" ${s.detecting ? 'disabled' : ''}
                    >${s.detecting ? 'Detecting…' : 'Detect'}</button>
                  <button class="lp-btn" data-act="register" ${marked ? '' : 'disabled'}
                          title="${marked ? '' : 'Nothing marked to register'}"
                    >Register${marked ? ` ${marked}` : ''}</button>
                  <button class="lp-btn" data-act="clear" ${marked ? '' : 'disabled'}
                          title="${marked ? 'Discard pending marks' : 'No pending marks'}"
                    >Clear</button>
                </div>

                ${session(s)}
                ${s.note ? `<p class="mk-note">${escape(s.note)}</p>` : ''}
              </div>`;
            wire(el);
        });
    }

    /**
     * The agent-initiated session, when there is one.
     *
     * Present only while the agent is waiting — this is the one part of the
     * panel that is a transient condition rather than a standing control, so it
     * seats and retires itself (PANELS.md rule 6).
     *
     * The per-marker role list exists because the contract needs it:
     * `marking_done` carries a role per marker and the waiting agent reads
     * them, so the operator has to be able to say which of these is a
     * reference before answering. Registered embryos get their roles in the
     * Acquisition roster; these are not registered yet.
     */
    function session(s) {
        if (!s.session) return '';
        const rows = (s.session.pending || []).map(m => {
            const ref = m.role === 'calibration';
            return `<div class="mk-prow">
                      <span class="mk-pnum">${m.index + 1}</span>
                      <span class="mk-psrc">${escape(m.source)}</span>
                      <button class="lp-btn mk-prole${ref ? ' is-reference' : ''}" type="button"
                              title="${ref ? 'Reference — click to make it a subject'
                                           : 'Subject — click to make it a reference'}"
                              data-act="cycleRole" data-index="${m.index}"
                        >${ref ? 'ref' : 'subj'}</button>
                    </div>`;
        }).join('');

        return `<div class="mk-session">
                  <div class="mk-shead">The agent is waiting</div>
                  ${rows || '<p class="mk-note">Nothing marked yet — click each embryo on the image.</p>'}
                  <div class="mk-acts">
                    <button class="lp-btn" data-act="redetect">Re-detect</button>
                    <button class="lp-btn lp-btn-primary" data-act="done">Done</button>
                  </div>
                </div>`;
    }

    function escape(t) {
        return String(t).replace(/[&<>"]/g, c =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function wire(el) {
        el.querySelectorAll('[data-act]').forEach(b => {
            b.onclick = () => {
                const v = verbs();
                if (!v) return;
                const fn = v[b.dataset.act];
                // Detect carries the pipeline the operator chose; the other
                // verbs take `data-index`, present only on the role toggles.
                if (typeof fn !== 'function') return;
                if (b.dataset.act === 'detect') fn(detectOptions());
                else fn(b.dataset.index);
            };
        });

        // The two optional stages toggle in place. Re-render rather than
        // mutate, so every mounted copy of the panel agrees.
        el.querySelectorAll('.mk-stage[data-set]').forEach(b => {
            b.onclick = () => {
                const key = b.dataset.set;
                settings[key] = !settings[key];
                saveSettings();
                render();
            };
        });

        el.querySelectorAll('select[data-set], input[data-set]').forEach(input => {
            input.onchange = () => {
                settings[input.dataset.set] =
                    input.type === 'checkbox' ? !!input.checked : input.value;
                saveSettings();
                render();
            };
        });
    }

    return { mount, unmount, render, _elapsed: elapsed, _options: detectOptions };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = MarkingPanel;
