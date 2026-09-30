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
 *     LightPanel.mount('op-led-host', { only: 'led' });
 *
 * `only: 'led'` draws the LED's state and brightness and nothing of the laser —
 * for the bottom camera, which is lit by the LED and has no use for lines or
 * a beam. Same panel, same state: it is a narrower window, not a second
 * control.
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
 * MODE IS THE ROOT
 *
 * Above those three sits one question: LED or laser. They are not settings that
 * happen to be adjacent — they are the two ways this instrument illuminates a
 * sample, and the workflow alternates between them (LED to find embryos, laser
 * to calibrate and acquire). So the panel opens with a mode, and each mode's
 * own controls are disclosed beneath it.
 *
 * Nothing in the hardware enforces the choice: they are two independent
 * Micro-Manager config groups and both can be open at once. `mode()` therefore
 * reports `both` rather than the panel making it unrepresentable — see #106,
 * which is that exact state going unnoticed.
 *
 * Everything rendered here is read back from hardware. A value that was sent is
 * not a value that is true; a value not yet read is an em dash, never a
 * plausible default.
 */
const LightPanel = (() => {
    'use strict';

    const hosts = new Map();   // hostId -> { only }

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

    // Which branch is disclosed. UI scope, not device state: the operator can
    // open the laser branch on a dark rig in order to configure it. The mode
    // READOUT is always derived from hardware — this only decides what is on
    // screen, and the derived mode wins whenever it is unambiguous.
    let branch = null;

    const state = () => SharedState.get('light') || {};

    /* ── reading ─────────────────────────────────────────────────────────── */

    /** State and brightness come back in the one status read. */
    const ledOf = d => ({
        led: (d && d.current_state) || null,
        ledPct: d && d.intensity_pct != null ? Number(d.intensity_pct) : null,
        ledLim: (d && d.intensity_limits_pct) || null,
    });

    /**
     * Just the LED, for when that is all anyone is looking at.
     *
     * The bottom camera's card shows no beam, config or power, and reading them
     * every poll to show none of them is traffic this device layer cannot
     * spare. Stamped `ledReadAt`, not `readAt`: the full panel's age must not
     * be refreshed by a read that skipped everything else it displays.
     */
    async function readLed() {
        let d = null;
        try {
            const r = await fetch('/api/devices/led/status');
            if (r.ok) d = await r.json();
        } catch (_) { /* unread is an em dash */ }
        // Merged into the state as it is NOW: a full read may have landed while
        // this one was in flight, and its beam and config must survive.
        SharedState.set('light', { ...state(), ...ledOf(d), ledReadAt: Date.now() });
    }

    /** Read what the asking surface shows, and no more. */
    const read = scope => (scope === 'led' ? readLed() : readAll());

    async function readAll() {
        const next = { ...state() };
        const get = async (url, key, pick) => {
            try {
                const r = await fetch(url);
                if (!r.ok) { next[key] = null; return null; }
                const d = await r.json();
                next[key] = pick(d);
                return d;
            } catch (_) { next[key] = null; return null; }
        };

        await Promise.all([
            get('/api/devices/beam', 'beam', d => (d && d.beam) || null),
            // One read answers both questions: the state and the brightness
            // come back together, and a second request would double the
            // traffic on a device layer that is already slow to read.
            get('/api/devices/led/status', 'led', d => (d && d.current_state) || null)
                .then(d => Object.assign(next, ledOf(d))),
            // The config is read, not remembered. It used to be whatever this
            // panel last wrote, which made the whole laser branch an echo.
            // `read()` yields the string "unknown" when the group cannot be
            // queried, and that must stay an em dash rather than become a
            // preset name.
            get('/api/devices/laser/configs', 'config',
                d => (d && d.current && d.current !== 'unknown') ? d.current : null),
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

        next.readAt = next.ledReadAt = Date.now();
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

    /**
     * Which illumination mode the rig is actually in.
     *
     * LED and Laser are two independent Micro-Manager config groups, so
     * "either LED or laser" is a POLICY, not a hardware interlock — nothing
     * stops both being open, and #106 is precisely that: `calibrateSelected`
     * never closed the LED, so every vision frame was LED brightfield with a
     * 50 ms laser gate on top, and the detector was hunting nuclei in a
     * DIC-like image.
     *
     * So `both` is a state this returns, not a state the panel makes
     * unreachable. A mode selector that could not express the fault would hide
     * the only bug it was built for.
     */
    function mode(s) {
        const ledOn = s.led == null ? null : s.led === 'Open';
        const routed = s.config == null ? null : routedLines(s).length > 0;
        if (ledOn === null || routed === null) return null;
        if (ledOn && routed) return 'both';
        if (ledOn) return 'led';
        if (routed) return 'laser';
        return 'off';
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
    async function act(fn, scope) {
        try { await fn(); } catch (_) { /* toasted in send() */ }
        await read(scope);
    }

    /**
     * Switch modes. Closing the OTHER source is the point — that is what makes
     * this a mode selector rather than two switches side by side.
     *
     * Entering laser mode deliberately does NOT route a line. Picking a
     * wavelength is an emission decision and stays an explicit act; all this
     * does is close the LED and open the branch. `ALL OFF` gates outputs 5-8 at
     * the PLogic, which is the documented brightfield-safe state (spec §2.7),
     * so it is what "off" and "led" both assert.
     *
     * BeamEnabled is left alone throughout. Setting it No here would recreate
     * the #106 state that nothing sets back.
     */
    async function enter(to) {
        const s = state();
        const routed = routedLines(s).length > 0;
        if (to === 'led' || to === 'off') {
            if (routed) await send('/api/devices/laser/config', { config: 'ALL OFF' });
        }
        await send('/api/devices/led/set', { state: to === 'led' ? 'Open' : 'Closed' });
    }

    /* ── rendering ───────────────────────────────────────────────────────── */

    const dash = v => (v == null ? '—' : v);

    function render() {
        const s = state();
        const em = emitting(s);
        hosts.forEach((opts, host) => {
            const el = document.getElementById(host);
            if (!el) return;
            el.innerHTML = opts.only === 'led' ? ledCard(s) : markup(s, em);
            wire(el, opts.only === 'led' ? 'led' : undefined);
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

    /** The three modes, plus the fault, as a segmented control. */
    function modeRow(m, show) {
        const label = { led: 'LED', laser: 'Laser', off: 'Off' };
        const btns = ['led', 'laser', 'off'].map(k => {
            const on = m === k || (m === 'both' && k !== 'off');
            const open = show === k;
            return `<button class="lp-mbtn ${on ? 'is-on' : ''} ${open ? 'is-open' : ''}"
                            data-mode="${k}" aria-pressed="${on}">${label[k]}</button>`;
        }).join('');
        // No text readout: the buttons ARE the readout, filled when the device
        // says that source is open. A word beside them had nowhere to go in a
        // 246 px panel and got clipped, and the two states words would have
        // added — both open, and nothing read — each say so on their own line.
        return `
          <div class="lp-row lp-mode">
            <span class="lp-label">Mode</span>
            <span class="lp-seg">${btns}</span>
          </div>
          ${m == null ? '<p class="lp-note">Nothing read from the light path yet.</p>' : ''}`;
    }

    const ageOf = t => (t ? `${Math.round((Date.now() - t) / 1000)}s ago` : 'never');

    /**
     * The LED on its own, for a surface the laser has no part in.
     *
     * No mode selector: this card cannot see the laser, so it must not offer
     * to choose between the two. It reports the LED and sets its brightness.
     */
    function ledCard(s) {
        return `
          <div class="lp">
            <div class="lp-head">
              <span class="lp-title">LED</span>
              <span class="lp-age" title="Values are read from the hardware, not remembered">read ${ageOf(s.ledReadAt)}</span>
            </div>
            ${ledRows(s)}
          </div>`;
    }

    function markup(s, em) {
        const armed = s.beam ? Object.values(s.beam).some(v => v === true) : null;
        const age = ageOf(s.readAt);
        const lines = routedLines(s);
        const m = mode(s);
        // The fault outranks the cursor: if both sources are open, the laser
        // branch is shown whatever the operator last clicked, because that is
        // where the contradiction can be resolved.
        const show = m === 'both' ? 'laser' : (branch || m);

        return `
          <div class="lp">
            <div class="lp-head">
              <span class="lp-title">Light</span>
              <span class="lp-age" title="Values are read from the hardware, not remembered">read ${age}</span>
            </div>

            ${modeRow(m, show)}
            ${m === 'both' ? bothWarn(s) : ''}
            ${idleBeamNote(s, armed, lines)}

            ${show === 'led' ? ledDetail(s) : ''}
            ${show === 'laser' ? laserBranch(s, armed, lines) : ''}

            ${em === true
                ? `<div class="lp-emit" role="status">EMITTING · ${s.config || 'lines unknown'}</div>`
                : em === null
                    ? '<div class="lp-emit lp-emit-unknown" role="status">Emission state unknown</div>'
                    : ''}
          </div>`;
    }

    /**
     * LED mode: the shutter's state, and its brightness.
     *
     * The brightness is the ASI Tiger adapter's own `LED Intensity(%)`
     * property. The adapter only sends it to the controller while the LED is
     * open, so on a closed LED the slider sets what the next open will use —
     * which the note says, because a slider that moves with no change in the
     * image otherwise reads as broken.
     */
    function ledDetail(s) {
        return `
          <div class="lp-sub">
            ${ledRows(s)}
            <p class="lp-note">Brightfield. Good for finding embryos; nuclei are
               not visible, so calibration and acquisition need the laser.</p>
          </div>`;
    }

    /** State and brightness — the rows every LED mount shares. */
    function ledRows(s) {
        const lim = s.ledLim || { min: 1, max: 100 };
        const val = s.ledPct;
        return `
            <div class="lp-row">
              <span class="lp-label">State</span>
              <span class="lp-val">${dash(s.led)}</span>
            </div>
            <div class="lp-row">
              <span class="lp-label" title="LED Intensity(%) on the ASI Tiger LED — the Micro-Manager property name">Intensity</span>
              <input class="lp-range" type="range" data-led-pct
                     min="${lim.min}" max="${lim.max}" step="1"
                     value="${val == null ? lim.min : val}"
                     ${val == null ? 'disabled' : ''}
                     title="${lim.min}–${lim.max} %"
                     aria-label="LED intensity percent">
              <span class="lp-val">${val == null ? '—' : val} %</span>
            </div>
            ${s.led === 'Closed' && val != null
                ? '<p class="lp-note">The LED is closed — this is the brightness it will open at.</p>'
                : ''}`;
    }

    /** The config select, then the laser's own settings once a line is routed. */
    function laserBranch(s, armed, lines) {
        return `
          <div class="lp-sub">
            <div class="lp-row">
              <span class="lp-label" title="The Laser config group — which lines the PLogic routes to the SPIM trigger">Config</span>
              <select class="lp-select" data-config aria-label="Laser config">
                ${configOptions(s.config)}
              </select>
            </div>
            ${lines.length ? laserDetail(s, armed, lines) : ''}
          </div>`;
    }

    /**
     * Both sources open. Safe, and the reason vision-guided calibration was
     * looking for nuclei in a DIC-like image for weeks (#106).
     */
    function bothWarn(s) {
        return `<p class="lp-warn">The LED is open and the laser is routing
                ${routedLines(s).join(', ')} nm — fluorescence sits on top of
                brightfield. Pick a mode.</p>`;
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
     * state the rig is left in.
     *
     * At the PANEL ROOT, not inside the laser branch — `mode()` calls this
     * state `off`, so the laser branch is closed and a note living inside it
     * would be invisible in the one state it is about. Hiding the detail must
     * never hide the fact, or the disclosure would have made the panel less
     * honest than the flat version it replaced.
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

    function wire(el, scope) {
        const beam = el.querySelector('[data-beam]');
        if (beam) beam.onclick = () => act(() =>
            send('/api/devices/beam', { enabled: beam.dataset.beam === 'on' }));

        const cfg = el.querySelector('[data-config]');
        if (cfg) cfg.onchange = () => act(() =>
            send('/api/devices/laser/config', { config: cfg.value }));

        el.querySelectorAll('[data-mode]').forEach(b => {
            b.onclick = () => { branch = b.dataset.mode; act(() => enter(b.dataset.mode)); };
        });

        el.querySelectorAll('[data-power]').forEach(r => {
            // On release, not on drag: every input event would be a hardware write.
            r.onchange = () => act(() => send('/api/devices/laser/power',
                { wavelength: Number(r.dataset.power), pct: Number(r.value) }));
        });

        const ledPct = el.querySelector('[data-led-pct]');
        // On release, for the same reason as the power sliders.
        if (ledPct) ledPct.onchange = () => act(() =>
            send('/api/devices/led/intensity', { pct: Number(ledPct.value) }), scope);

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

    async function mount(hostId, opts) {
        const only = opts && opts.only === 'led' ? 'led' : null;
        hosts.set(hostId, { only });
        if (hosts.size === 1) {
            SharedState.on('light', render);
            timer = setInterval(() => {
                const scope = visible();
                if (scope) read(scope);
            }, POLL_MS);
        }
        // Limits and presets are the laser's; an LED card has no use for them.
        if (!only) await loadStatics();
        render();
        await read(only);
    }

    /**
     * What is on screen: 'all' if a full panel is, 'led' if only LED cards
     * are, null if nothing is. The poll reads no more than that.
     */
    function visible() {
        let scope = null;
        for (const [h, opts] of hosts) {
            const el = document.getElementById(h);
            if (!el || el.offsetParent === null) continue;
            if (!opts.only) return 'all';
            scope = 'led';
        }
        return scope;
    }

    function unmount(hostId) {
        hosts.delete(hostId);
        if (!hosts.size && timer) { clearInterval(timer); timer = null; }
    }

    return { mount, unmount, refresh: readAll, emitting, mode, wavelengthsOf, _state: state };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = LightPanel;
