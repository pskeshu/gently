/**
 * Light — the standard control panel for illumination.
 *
 * LED, beam, routed lines and per-line power. Exposure is a camera
 * property, not a light one, and lives in panels/camera.js — the bottom
 * camera needs it without needing any of this.
 *
 * The first panel built to docs/architecture/PANELS.md. Mount it anywhere that
 * needs light control; every mount shows the same state because the state
 * lives in SharedState, not in the panel.
 *
 *     LightPanel.mount('op-light-host');
 *
 * WHY THIS EXISTS
 *
 * `LASER: ON` used to mean "an HTTP request returned 200". Ryan watched the
 * physical microscope on 2026-08-07 and reported the beam was not firing while
 * the UI said it was (#106). Three independent facts were being collapsed into
 * that one word:
 *
 *   Laser       — the Laser config group ("ALL OFF", "488 only", ...). On this
 *                 rig this IS what an operator means by the laser being on or
 *                 off, so it keeps that name. PLogic gating; it emits nothing
 *                 by itself.
 *   BeamEnabled — the Micro-Manager property on the scanner card, named as
 *                 Micro-Manager names it, because that is where an operator
 *                 has seen it before. Left "No" after every volume
 *                 acquisition with nothing to set it back, which is how a
 *                 correctly configured laser still emits nothing.
 *   power       — per-line setpoint. The calibrate path never touched it.
 *
 * Deliberately not collapsed into one "laser" switch, however much tidier that
 * would read: collapsing them is what produced #106. "Arm" was worse still —
 * it invented a word this instrument does not use.
 *
 * Any of the three can be wrong on its own, so the panel shows all three and
 * computes "emitting" from them rather than believing a flag.
 *
 * Everything rendered here is read back from hardware. A value that was sent is
 * not a value that is true; a value not yet read is an em dash, never a
 * plausible default.
 */
const LightPanel = (() => {
    'use strict';

    const hosts = new Set();

    // Bounds and preset names come from the server (PANELS.md rule 4). 488 is
    // limited to 2-6%, so a control hardcoded 0-100 would offer settings the
    // device layer refuses.
    let limits = null;
    let configs = [];

    // RIG-NOTE: the device layer already logs "Property read slow: 2.4s" every
    // 15s on this scope, so polling here is deliberately gentle and only runs
    // while a panel is on screen. Raise it if the readout feels stale, but
    // watch the device-layer log before you do.
    const POLL_MS = 10000;
    let timer = null;

    const state = () => SharedState.get('light') || {};

    /* ── reading ─────────────────────────────────────────────────────────── */

    async function readAll() {
        const next = { ...state() };
        const get = async (url, key, pick) => {
            try {
                const r = await fetch(url);
                if (!r.ok) { next[key] = null; return; }
                const d = await r.json();
                next[key] = pick(d);
            } catch (_) { next[key] = null; }
        };

        await Promise.all([
            get('/api/devices/beam', 'beam', d => (d && d.beam) || null),
            get('/api/devices/led/status', 'led', d => (d && d.current_state) || null),
        ]);

        // Power is per wavelength, and only the routed lines are interesting.
        next.power = {};
        for (const wl of wavelengthsOf(next.config)) {
            try {
                const r = await fetch(`/api/devices/laser/power?wavelength=${wl}`);
                const d = r.ok ? await r.json() : null;
                next.power[wl] = d && d.pct != null ? Number(d.pct) : null;
            } catch (_) { next.power[wl] = null; }
        }

        next.readAt = Date.now();
        SharedState.set('light', next);
    }

    /** Wavelengths named by a config, e.g. "488 and 561" → [488, 561]. */
    function wavelengthsOf(config) {
        const known = limits ? Object.keys(limits).map(Number) : [405, 488, 561, 637];
        if (!config) return known;
        const found = String(config).match(/\d{3}/g);
        if (!found) return [];                       // e.g. "ALL OFF" — nothing routed
        return found.map(Number).filter(w => known.includes(w));
    }

    /**
     * Is light actually coming out? Derived, never commanded (PANELS.md rule 5).
     * Anything unknown makes this unknown — an unread beam is not a dark one.
     */
    function emitting(s) {
        const sides = s.beam ? Object.values(s.beam) : null;
        if (!sides || !sides.length) return null;
        if (!sides.some(v => v === true)) {
            // A side that could not be read might be armed. Only every side
            // definitively false is a safe "not emitting".
            return sides.some(v => v == null) ? null : false;
        }
        const lines = wavelengthsOf(s.config);
        if (!lines.length) return false;
        const powers = lines.map(w => (s.power || {})[w]);
        if (powers.some(p => p == null)) return null;
        return powers.some(p => p > 0);
    }

    /* ── writing ─────────────────────────────────────────────────────────── */

    async function send(url, body) {
        const r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const text = await r.text().catch(() => '');
        let d = {}; try { d = text ? JSON.parse(text) : {}; } catch (_) { /* not JSON */ }
        if (!r.ok) {
            const msg = d.detail || d.error || `${r.status}`;
            if (typeof showGentlyToast === 'function') {
                showGentlyToast(String(msg), null, null, 10000, 'error');
            }
            throw new Error(msg);
        }
        return d;
    }

    // Always re-read after a write. The response is not evidence: that is the
    // assumption that produced #106.
    async function act(fn) {
        try { await fn(); } catch (_) { /* toasted in send() */ }
        await readAll();
    }

    /* ── rendering ───────────────────────────────────────────────────────── */

    const dash = v => (v == null ? '—' : v);

    function render() {
        const s = state();
        const em = emitting(s);
        hosts.forEach(host => {
            const el = document.getElementById(host);
            if (!el) return;
            el.innerHTML = markup(s, em);
            wire(el);
        });
    }

    /**
     * The lines this config actually routes, for DISPLAY.
     *
     * Deliberately not `wavelengthsOf`, which answers "which lines could be
     * involved" and returns all of them for an unknown config — correct for
     * `emitting()`, because an unread config with unread power must come out
     * unknown rather than safe. But rendering four disabled sliders reading an
     * em dash is maximum clutter for zero information, so the panel asks a
     * narrower question: which lines do we KNOW are routed.
     */
    /**
     * The preset list, always including whatever is actually set.
     *
     * The list comes from `/api/devices/laser/configs`, which 503s with the
     * device layer down — so the select would read an em dash while the detail
     * below it showed routed lines and live power sliders. The read-back config
     * is a fact whether or not the catalogue of presets is available, and the
     * two halves of the panel must not contradict each other.
     */
    function configOptions(current) {
        const opts = configs.slice();
        if (current && !opts.includes(current)) opts.unshift(current);
        if (!opts.length) return '<option>—</option>';
        return opts.map(c =>
            `<option value="${c}" ${c === current ? 'selected' : ''}>${c}</option>`).join('');
    }

    function routedLines(s) {
        return s.config ? wavelengthsOf(s.config) : [];
    }

    function markup(s, em) {
        const armed = s.beam ? Object.values(s.beam).some(v => v === true) : null;
        const age = s.readAt ? `${Math.round((Date.now() - s.readAt) / 1000)}s ago` : 'never';
        const lines = routedLines(s);

        return `
          <div class="lp">
            <div class="lp-head">
              <span class="lp-title">Light</span>
              <span class="lp-age" title="Values are read from the hardware, not remembered">read ${age}</span>
            </div>

            <div class="lp-row">
              <span class="lp-label">LED</span>
              <button class="lp-btn" data-led="${s.led === 'Open' ? 'Closed' : 'Open'}"
                      aria-pressed="${s.led === 'Open'}">${s.led === 'Open' ? 'On' : 'Off'}</button>
              <span class="lp-val">${dash(s.led)}</span>
            </div>

            <div class="lp-row">
              <span class="lp-label" title="The Laser config group — on this rig this is the laser ON/OFF control">Laser</span>
              <select class="lp-select" data-config aria-label="Laser config">
                ${configOptions(s.config)}
              </select>
            </div>
            ${idleBeamNote(s, armed, lines)}

            ${lines.length ? laserDetail(s, armed, lines) : ''}

            ${em === true
                ? `<div class="lp-emit" role="status">EMITTING · ${s.config || 'lines unknown'}</div>`
                : em === null
                    ? '<div class="lp-emit lp-emit-unknown" role="status">Emission state unknown</div>'
                    : ''}
          </div>`;
    }

    /**
     * The laser's own settings, revealed once a line is routed.
     *
     * Nested for SCOPE, not for dependency. Beam and power only matter once
     * something is routed, which is why they are hidden until then — but they
     * are not caused by the config, and #106 is exactly what happens when
     * someone assumes they are. So the contradiction gets a line of its own
     * rather than being softened by the indent.
     */
    function laserDetail(s, armed, lines) {
        const contradicts = lines.length && armed === false;
        const powerRows = lines.map(wl => {
            const lim = (limits && limits[wl]) || { min: 0, max: 100 };
            const val = (s.power || {})[wl];
            return `
              <div class="lp-row">
                <span class="lp-label">${wl}</span>
                <input class="lp-range" type="range" data-power="${wl}"
                       min="${lim.min}" max="${lim.max}" step="0.1"
                       value="${val == null ? lim.min : val}"
                       ${val == null ? 'disabled' : ''}
                       aria-label="${wl} nm power percent">
                <span class="lp-val">${val == null ? '—' : Number(val).toFixed(1)} %</span>
                <span class="lp-lim">${lim.min}–${lim.max}</span>
              </div>`;
        }).join('');

        return `
          <div class="lp-sub">
            <div class="lp-row">
              <span class="lp-label" title="BeamEnabled on the scanner card — the Micro-Manager property name">BeamEnabled</span>
              <button class="lp-btn ${armed ? 'is-armed' : ''}" data-beam="${armed ? 'off' : 'on'}"
                      aria-pressed="${armed === true}">${armed ? 'Set No' : 'Set Yes'}</button>
              <span class="lp-val">${armed == null ? '—' : armed ? 'Yes' : 'No'}</span>
              ${sideDetail(s.beam)}
            </div>
            ${contradicts
                ? `<p class="lp-warn">Lines are routed but the beam is off — this
                   configuration will not emit. Every volume acquisition leaves
                   BeamEnabled at No.</p>`
                : ''}
            ${powerRows}
          </div>`;
    }

    /**
     * An armed beam with nothing routed is safe but surprising, and it is the
     * state the rig is left in. Say it on the Laser row so hiding the detail
     * never hides the fact.
     */
    function idleBeamNote(s, armed, lines) {
        if (lines.length || armed !== true) return '';
        return '<p class="lp-note">Beam is armed, but no lines are routed — nothing emits.</p>';
    }

    /** Only worth showing when the two sides disagree, which is a real state. */
    function sideDetail(beam) {
        if (!beam) return '';
        const vals = Object.values(beam);
        if (vals.length < 2 || vals.every(v => v === vals[0])) return '';
        const txt = Object.entries(beam)
            .map(([k, v]) => `${k.toUpperCase()} ${v === null ? '?' : v ? 'on' : 'off'}`).join(' · ');
        return `<span class="lp-lim">${txt}</span>`;
    }

    function wire(el) {
        const led = el.querySelector('[data-led]');
        if (led) led.onclick = () => act(() =>
            send('/api/devices/led/set', { state: led.dataset.led }));

        const beam = el.querySelector('[data-beam]');
        if (beam) beam.onclick = () => act(() =>
            send('/api/devices/beam', { enabled: beam.dataset.beam === 'on' }));

        const cfg = el.querySelector('[data-config]');
        if (cfg) cfg.onchange = () => act(() =>
            send('/api/devices/laser/config', { config: cfg.value })
                .then(() => SharedState.set('light', { ...state(), config: cfg.value })));

        el.querySelectorAll('[data-power]').forEach(r => {
            // On release, not on drag: every input event would be a hardware write.
            r.onchange = () => act(() => send('/api/devices/laser/power',
                { wavelength: Number(r.dataset.power), pct: Number(r.value) }));
        });

    }

    /* ── lifecycle ───────────────────────────────────────────────────────── */

    async function loadStatics() {
        if (limits && configs.length) return;
        try {
            const r = await fetch('/api/devices/laser/limits');
            if (r.ok) limits = (await r.json()).limits || null;
        } catch (_) { /* the panel still renders, with default bounds */ }
        try {
            const r = await fetch('/api/devices/laser/configs');
            if (r.ok) {
                const d = await r.json();
                configs = d.configs || d.available_configs || [];
            }
        } catch (_) { /* select shows an em dash */ }
    }

    async function mount(hostId) {
        hosts.add(hostId);
        if (hosts.size === 1) {
            SharedState.on('light', render);
            timer = setInterval(() => { if (visible()) readAll(); }, POLL_MS);
        }
        await loadStatics();
        render();
        await readAll();
    }

    function visible() {
        for (const h of hosts) {
            const el = document.getElementById(h);
            if (el && el.offsetParent !== null) return true;
        }
        return false;
    }

    function unmount(hostId) {
        hosts.delete(hostId);
        if (!hosts.size && timer) { clearInterval(timer); timer = null; }
    }

    return { mount, unmount, refresh: readAll, emitting, wavelengthsOf, _state: state };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = LightPanel;
