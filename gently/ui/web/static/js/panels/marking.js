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
            el.innerHTML = `
              <div class="lp">
                <div class="lp-head">
                  <span class="lp-title">Marking</span>
                  ${s.detecting
                    ? `<span class="mk-busy">detecting… ${secs}s</span>`
                    : ''}
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
                  <button class="lp-btn" data-act="detect" ${s.detecting ? 'disabled' : ''}
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
                // `data-index` is only present on the per-marker role toggles.
                if (typeof fn === 'function') fn(b.dataset.index);
            };
        });
    }

    return { mount, unmount, render, _elapsed: elapsed };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = MarkingPanel;
