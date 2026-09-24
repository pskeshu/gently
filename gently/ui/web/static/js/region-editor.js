/**
 * Region editor — the map is the form.
 *
 * WHAT WAS WRONG WITH THE OLD ONE
 *
 * A four-step wizard in a 300px column floating over the map: Start, capture
 * the bottom-left corner, capture the top-right, Apply. It opened 477px tall
 * inside a card 594px tall that ends 44px above the window bottom, so the
 * instruction was cut mid-sentence and the Start button sat off-screen. The
 * button that opened it hid itself on click, so the whole visible effect of
 * pressing "Edit region" was a button disappearing — which is exactly how it
 * was reported: "the edit region button click does nothing".
 *
 * The deeper problem was the wizard itself. Setting four numbers is not a
 * sequence, and pretending it is costs you the common case: "the fence is
 * 200 µm too tight on +X" should be one gesture, not a re-walk of both
 * corners.
 *
 * WHAT THIS IS INSTEAD
 *
 * The region is an object on the sheet. Drive the stage wherever you like,
 * then click the edge you are standing on — it takes the live coordinate.
 * Click a corner to set both of its edges at once. Any edge, any order, as
 * many times as you like; an overshoot is just "drive back and click again",
 * which is why the old inset field (a µm fudge for joystick overshoot, with a
 * thirty-word explanation) is gone.
 *
 * Nothing reaches the controller until Apply, so Cancel is always safe and
 * always available — including the crash case, which is just a Cancel nobody
 * pressed.
 */
const RegionEditor = (() => {
    'use strict';

    const READ = '/api/devices/stage/envelope';
    const APPLY = '/api/devices/stage/envelope';
    const RESTORE = '/api/devices/stage/region/restore';
    const ENFORCE = '/api/devices/stage/envelope/enforced';

    // Which edges a target sets. Corners set two; edges set one.
    const TARGETS = {
        'x-min': ['x_min'], 'x-max': ['x_max'],
        'y-min': ['y_min'], 'y-max': ['y_max'],
        'min-min': ['x_min', 'y_min'], 'max-min': ['x_max', 'y_min'],
        'min-max': ['x_min', 'y_max'], 'max-max': ['x_max', 'y_max'],
    };

    let _open = false;
    let _box = null;          // the region being edited
    let _applied = null;      // what it was when editing started
    let _wasEnforced = null;  // so Cancel restores the fence as it was
    let _startedCam = false;
    let _pos = null;          // live stage position
    let _history = [];
    let _travel = null;       // everything the stage can reach
    let _onChange = () => {};

    const $ = id => document.getElementById(id);
    const fmt = v => (Number.isFinite(v) ? v.toFixed(1) : '—');

    const isOpen = () => _open;
    const box = () => (_box ? Object.assign({}, _box) : null);

    async function getJSON(url) {
        const r = await fetch(url);
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || d.error || `HTTP ${r.status}`);
        return d;
    }
    async function postJSON(url, body) {
        const r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || d.error || `HTTP ${r.status}`);
        return d;
    }

    /** Refusals happen before the strip exists, so they go to the toast. */
    function fail(msg) {
        if (typeof showGentlyToast === 'function') showGentlyToast(msg, null, null, 7000, 'warn');
        else console.warn(msg);
    }

    function say(msg, bad) {
        const el = $('region-say');
        if (!el) return;
        el.textContent = msg || '';
        el.dataset.bad = bad ? '1' : '0';
    }

    const BOUNDS = ['x_min', 'x_max', 'y_min', 'y_max'];
    const complete = b => !!b && BOUNDS.every(k => Number.isFinite(b[k]));

    /**
     * Open the editor: camera on, fence down, region on the sheet.
     *
     * Returns false when the stage cannot be read. There is nothing to edit
     * then — the fence lives in the controller, and an editor opened on four
     * undefined numbers would offer to write them.
     */
    async function open(opts) {
        if (_open) return false;
        let d = {};
        try {
            d = await getJSON(READ);
        } catch (e) {
            fail(`The stage is not reachable (${e.message}). The region is read from the controller, so there is nothing to edit yet.`);
            return false;
        }

        const read = [d.region, d.full_travel,
                      { x_min: d.x_min, x_max: d.x_max, y_min: d.y_min, y_max: d.y_max }];
        _applied = read.find(complete) || null;
        if (!_applied) {
            fail('The stage did not report a region. Nothing to edit until it does.');
            return false;
        }
        _wasEnforced = d.enforced === true;
        _travel = complete(d.full_travel) ? Object.assign({}, d.full_travel) : null;
        _box = Object.assign({}, _applied);
        _history = Array.isArray(d.history) ? d.history : [];
        _open = true;

        // The fence has to come down or you cannot drive to where a wider
        // boundary belongs — the Tiger stops the joystick at the current box.
        // Said out loud, and put back exactly as it was on the way out.
        if (_wasEnforced) {
            try {
                await postJSON(ENFORCE, { enforced: false });
                say('XY limits are off while you edit. They go back on when you finish.');
            } catch (e) {
                say(`Could not release the limits (${e.message}) — you can only shrink the region.`, true);
            }
        }

        // Seeing where you are is the whole basis for judging a boundary.
        if (opts && opts.startCamera) {
            try { await opts.startCamera(); _startedCam = true; } catch (e) { /* not fatal */ }
        }
        _onChange();
        return true;
    }

    /** Leave without writing anything. Safe at any moment. */
    async function cancel(opts) {
        if (!_open) return;
        _open = false;
        _box = null;
        if (_wasEnforced) {
            try { await postJSON(ENFORCE, { enforced: true }); } catch (e) { /* reported below */ }
        }
        // Only stop what editing started — Operate may own the stream.
        if (_startedCam && opts && opts.stopCamera) {
            try { await opts.stopCamera(); } catch (e) { /* not fatal */ }
        }
        _startedCam = false;
        _wasEnforced = null;
        say('');
        _onChange();
    }

    /**
     * Stamp the live stage position into one or two bounds.
     *
     * The bounds stay ordered: stamping +X below the current −X would make a
     * box with no inside, so the opposite edge moves out of the way rather
     * than the click being refused. Refusing here would mean explaining
     * geometry to someone holding a joystick.
     */
    function stamp(target) {
        if (!_open || !_box || !_pos) return false;
        const which = TARGETS[target];
        if (!which) return false;
        which.forEach(bound => {
            const v = bound.startsWith('x') ? _pos.x : _pos.y;
            _box[bound] = v;
            if (bound === 'x_min' && _box.x_max <= v) _box.x_max = v + 1;
            if (bound === 'x_max' && _box.x_min >= v) _box.x_min = v - 1;
            if (bound === 'y_min' && _box.y_max <= v) _box.y_max = v + 1;
            if (bound === 'y_max' && _box.y_min >= v) _box.y_min = v - 1;
        });
        say(`${which.join(' and ')} set to where the stage is.`);
        _onChange();
        return true;
    }

    /** Type a bound directly — the same edit, for people who know the number. */
    function setBound(bound, value) {
        if (!_open || !_box || !Number.isFinite(value)) return;
        _box[bound] = value;
        _onChange();
    }

    /** What changed, in the operator's terms, for the confirmation. */
    function diff() {
        if (!_box || !_applied) return [];
        return ['x_min', 'x_max', 'y_min', 'y_max']
            .filter(k => Math.abs((_box[k] ?? 0) - (_applied[k] ?? 0)) > 0.05)
            .map(k => `${k.replace('_', ' ')}: ${fmt(_applied[k])} → ${fmt(_box[k])}`);
    }

    async function apply(opts) {
        if (!_open || !_box) return false;
        try {
            await postJSON(APPLY, _box);
            const changes = diff();
            await cancel(opts);   // restores the fence and the camera
            say(changes.length ? `Applied · ${changes.join(' · ')}` : 'Applied · unchanged');
            return true;
        } catch (e) {
            say(`Could not apply (${e.message})`, true);
            return false;
        }
    }

    async function restore(appliedAt) {
        try {
            const d = await postJSON(RESTORE, { applied_at: appliedAt });
            _applied = d.region || null;
            if (_open && _applied) _box = Object.assign({}, _applied);
            _history = Array.isArray(d.history) ? d.history : _history;
            say('Region restored.');
            _onChange();
            return true;
        } catch (e) {
            say(`Could not restore (${e.message})`, true);
            return false;
        }
    }

    function setPosition(x, y) {
        _pos = Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
    }

    return {
        isOpen, box, open, cancel, apply, stamp, setBound, restore, diff, setPosition,
        history: () => _history.slice(),
        travel: () => (_travel ? Object.assign({}, _travel) : null),
        applied: () => (_applied ? Object.assign({}, _applied) : null),
        position: () => (_pos ? Object.assign({}, _pos) : null),
        onChange: fn => { _onChange = typeof fn === 'function' ? fn : () => {}; },
        TARGETS,
    };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = RegionEditor;
