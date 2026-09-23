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

    let _state = { enforced: null, region: null, box: null, reason: '', busy: false };
    const _subs = new Set();

    const snapshot = () => Object.assign({}, _state);

    function notify() {
        const s = snapshot();
        _subs.forEach(fn => {
            try { fn(s); } catch (e) { console.debug('xy-limits subscriber failed', e); }
        });
    }

    function unknown(reason) {
        _state = { enforced: null, region: null, box: null, reason: reason || '', busy: false };
        notify();
    }

    function adopt(d) {
        _state = {
            enforced: !!d.enforced,
            region: d.region || null,
            box: {
                x_min: d.x_min, x_max: d.x_max,
                y_min: d.y_min, y_max: d.y_max,
            },
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
            else unknown(r.status === 403 ? 'Sign in to see the limits' : 'Microscope not connected');
        } catch (e) {
            unknown('Microscope not connected');
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

    return { read, write, subscribe, snapshot };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = XYLimitsState;
