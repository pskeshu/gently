/**
 * Whether the controller is fencing the XY stage — in one place.
 *
 * WHAT THIS IS NOT
 *
 * Not a Gently setting. `set_firmware_limits` writes `LowerLimX(mm)` and
 * friends into the ASI Tiger, and the Tiger enforces them against EVERY
 * motion source: the joystick, MMCore, scripts, and anyone driving this stage
 * from Micro-Manager. A region we wrote on Monday still stops their stage
 * short on Friday, with nothing on their screen to explain it. This store
 * backs the switch that gives them the stage back.
 *
 * "OFF" MEANS FULL TRAVEL
 *
 * The controller always holds some box, so there is no "no limits" — off is
 * the stage's whole physical range. The operator's region is kept while it is
 * off, so switching back on restores it without walking the corners again.
 *
 * ENFORCED IS READ, NOT REMEMBERED
 *
 * The device layer decides `enforced` by comparing what the controller
 * actually holds against full travel — so a write that did not take reads
 * back as what it is. A flag in a config file would say what we meant.
 */
const XYLimitsState = (() => {
    'use strict';

    const READ_URL = '/api/devices/stage/envelope';
    const WRITE_URL = '/api/devices/stage/envelope/enforced';

    // How long to wait before asking again after an answer we could not get.
    const RETRY_MS = 4000;

    let _state = {
        enforced: null, region: null, box: null, travel: null, reason: '', busy: false,
    };
    const _subs = new Set();
    let _retry = null;

    /** Is the rig up, as far as the boot poll knows? */
    const rigReady = () => typeof window !== 'undefined' && window.gentlyDeviceReady === true;

    const snapshot = () => Object.assign({}, _state);

    function notify() {
        const s = snapshot();
        _subs.forEach(fn => {
            try { fn(s); } catch (e) { console.debug('xy-limits subscriber failed', e); }
        });
    }

    /**
     * Say which of the two it is.
     *
     * "Microscope not connected" printed under a live X/Y readout is the card
     * contradicting itself — the operator can see the stage moving. When the
     * rig is up, an unreadable fence is our problem, and saying so is the
     * difference between a fault and a wait.
     */
    const offlineReason = () =>
        rigReady() ? 'Limits unavailable — asking again…' : 'Microscope not connected';

    function unknown(reason) {
        _state = {
            enforced: null, region: null, box: null, travel: null,
            reason: reason || '', busy: false,
        };
        notify();
    }

    /**
     * Ask again, because an unanswered question is not an answer.
     *
     * The read can fail for a second at boot — the device layer is up but the
     * stage is not connected yet — and there is exactly one DEVICE_LAYER_STATE
     * event per state change to trigger a retry. Miss it and the card keeps
     * the first answer forever.
     */
    function scheduleRetry() {
        if (_retry !== null || _state.enforced !== null) return;
        _retry = setTimeout(() => { _retry = null; read(); }, RETRY_MS);
    }

    function adopt(d) {
        if (_retry !== null) { clearTimeout(_retry); _retry = null; }
        _state = {
            enforced: !!d.enforced,
            region: d.region || null,
            box: {
                x_min: d.x_min, x_max: d.x_max,
                y_min: d.y_min, y_max: d.y_max,
            },
            travel: d.full_travel || null,
            reason: '',
            busy: false,
        };
        notify();
    }

    /** Ask the controller what it is holding. Never throws. */
    async function read() {
        try {
            const r = await fetch(READ_URL);
            const d = await r.json().catch(() => ({}));
            if (r.ok && d && d.success !== false && d.enforced !== undefined) adopt(d);
            else if (r.status === 403) unknown('Sign in to see the limits');
            else { unknown(offlineReason()); scheduleRetry(); }
        } catch (e) {
            unknown(offlineReason());
            scheduleRetry();
        }
        return snapshot();
    }

    /**
     * Turn the controller's limits on or off, then render what came back.
     *
     * Turning them ON with no saved region is refused by the device layer
     * (409) rather than inventing a box — a fence nobody walked is not a
     * safety feature.
     */
    async function write(enforced) {
        if (_state.busy) return snapshot();
        _state.busy = true;
        notify();
        try {
            const r = await fetch(WRITE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enforced: !!enforced }),
            });
            const d = await r.json().catch(() => ({}));
            if (r.ok && d && d.success !== false) {
                adopt(d);
            } else {
                _state.busy = false;
                _state.reason = r.status === 403
                    ? 'Sign in to change the limits'
                    : (d.detail || d.error || `Failed (${r.status})`);
                notify();
                await read();   // never leave the stored answer at a guess
            }
        } catch (e) {
            _state.busy = false;
            _state.reason = `Failed: ${e.message}`;
            notify();
            await read();
        }
        return snapshot();
    }

    function subscribe(fn) {
        _subs.add(fn);
        try { fn(snapshot()); } catch (e) { /* a bad subscriber is not our problem */ }
        return () => _subs.delete(fn);
    }

    // A device layer that comes or goes changes the answer, and a stale
    // "enforced" would misdescribe a stage nobody is fencing.
    if (typeof ClientEventBus !== 'undefined') {
        ClientEventBus.on('DEVICE_LAYER_STATE', d => {
            if (d && d.ready) read();
            else unknown('Microscope not connected');
        });
    }

    // That event fires once per state change, and this file is loaded AFTER
    // boot-banner.js, which is what emits it. A rig that was already up when
    // the page loaded can therefore announce itself before this store exists —
    // and then never again, because 'ready' does not change twice. The flag
    // boot-banner leaves behind is the same news, still readable afterwards.
    if (rigReady()) read();

    const BOUNDS = ['x_min', 'x_max', 'y_min', 'y_max'];
    const same = (a, b) =>
        !!a && !!b && BOUNDS.every(k => Math.abs((a[k] ?? 0) - (b[k] ?? 0)) < 1);

    /**
     * The box the operator is working inside — what the map should call
     * "optimal", and what the region editor edits.
     *
     * The map used to draw this from the controller's own limits, read off
     * `LowerLimX(mm)` and friends. That was the same box back when writing a
     * region always wrote the firmware. It is not any more: the firmware fence
     * is opt-in now, so with it off those properties report the stage's whole
     * travel — and the map cheerfully shaded the entire sheet "OPTIMAL".
     *
     * So the working region is the answer when there is one. A rig that
     * predates the region record has no such entry, and there the controller's
     * box is still real evidence someone fenced this stage — but only if it is
     * narrower than the travel, because full travel is not a region, it is the
     * absence of one.
     */
    function workingBox() {
        const s = _state;
        if (s.region) return { box: s.region, source: 'region' };
        if (s.box && s.travel && !same(s.box, s.travel)) {
            return { box: s.box, source: 'controller' };
        }
        return { box: null, source: 'none' };
    }

    return { read, write, subscribe, snapshot, workingBox };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = XYLimitsState;
