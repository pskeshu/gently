/**
 * Region editor — the stage is the pointer.
 *
 * WHAT THIS IS
 *
 * Two corners, walked. Drive to the bottom-left of your working area and
 * press Capture; drive to the top-right and press Capture again. While you
 * drive to the second corner the box follows the stage, so the region is
 * drawn by the very thing it is about to bound.
 *
 * WHY THIS, AGAIN
 *
 * This is what the first editor did, and replacing the model was a mistake.
 * What was broken was the packaging: it opened 477px tall inside a card that
 * ends above the window, so its own Start button sat off-screen, and the
 * button that opened it could not receive a click at all — it lived inside a
 * `pointer-events: none` overlay, so every press went through it to the map.
 * Nothing about "drive there and capture" was wrong; it never ran.
 *
 * The version in between made the map the form: click the edge you are
 * standing on. Fewer steps, but it asks you to aim at a dashed line with one
 * hand on a joystick, and — the part that actually sank it — nothing on
 * screen said what it wanted. Reported as "not clear, what it is asking".
 * A walk says what to do next; the stage says where.
 *
 * ORDER DOES NOT MATTER
 *
 * The prompts name bottom-left then top-right because a sequence needs an
 * order, but the box is built from min and max, so two opposite corners in
 * any order give the same region. Getting it "wrong" costs nothing.
 *
 * NOTHING REACHES THE CONTROLLER UNTIL APPLY
 *
 * So Cancel is always safe and always available, including the crash case,
 * which is just a Cancel nobody pressed.
 */
const RegionEditor = (() => {
    'use strict';

    const READ = '/api/devices/stage/envelope';
    const APPLY = '/api/devices/stage/envelope';
    const RESTORE = '/api/devices/stage/region/restore';
    const ENFORCE = '/api/devices/stage/envelope/enforced';

    const BOUNDS = ['x_min', 'x_max', 'y_min', 'y_max'];
    // Two corners this close together are a double-press, not a region.
    const MIN_SPAN_UM = 10;

    // Where the walk is. 'a' and 'b' are the two corners; 'review' is the box.
    const STEPS = ['a', 'b', 'review'];

    let _open = false;
    let _step = 'a';
    let _a = null;            // first captured corner {x, y}
    let _b = null;            // second captured corner
    let _edited = null;       // bounds typed or snapped after the walk
    let _applied = null;      // the region as it stands on the controller
    let _travel = null;       // everything the stage can reach
    let _wasEnforced = null;  // so Cancel puts the fence back as it was
    let _startedCam = false;
    let _pos = null;          // live stage position
    let _history = [];
    let _say = '';
    let _bad = false;
    let _note = '';           // a standing aside (limits are off); never displaces the prompt
    let _onChange = () => {};

    const fmt = v => (Number.isFinite(v) ? v.toFixed(1) : '—');
    const complete = b => !!b && BOUNDS.every(k => Number.isFinite(b[k]));
    const isOpen = () => _open;
    const step = () => _step;
    const isBad = () => _bad;

    /** The box as it stands: from the walk, then from any later edits. */
    function box() {
        if (_edited) return Object.assign({}, _edited);
        const second = _step === 'b' ? _pos : _b;      // rubber-band while driving
        if (!_a || !second) return null;
        return {
            x_min: Math.min(_a.x, second.x), x_max: Math.max(_a.x, second.x),
            y_min: Math.min(_a.y, second.y), y_max: Math.max(_a.y, second.y),
        };
    }

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
        _say = msg || '';
        _bad = !!bad;
    }

    /** Something that stuck: the walk closed and the strip with it, so it goes to the toast. */
    function tell(msg) {
        if (typeof showGentlyToast === 'function') showGentlyToast(msg, null, null, 5000, 'success');
        else console.info(msg);
    }

    /**
     * What to do right now, in one sentence.
     *
     * The old wizard's failure was not only that its buttons were off-screen,
     * and the one after it did not fail for want of features: neither ever
     * said what the operator was meant to do with the joystick in their hand.
     */
    function prompt() {
        if (_say) return _say;
        if (_step === 'a') return 'Drive to the bottom-left of your working area, then press Capture.';
        if (_step === 'b') return 'Now drive to the top-right. The box follows the stage as you go.';
        return 'This is your working region. Apply it, or nudge an edge below.';
    }

    /**
     * Open the walk: camera on, fence down, first corner asked for.
     *
     * Returns false when the stage cannot be read — there is nothing to edit
     * then, and an editor opened on four undefined numbers would offer to
     * write them.
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
        _travel = complete(d.full_travel) ? Object.assign({}, d.full_travel) : null;
        _wasEnforced = d.enforced === true;
        _history = Array.isArray(d.history) ? d.history : [];
        _step = 'a';
        _a = _b = _edited = null;
        say('');
        _note = '';
        _open = true;

        // The fence has to come down or you cannot drive to where a wider
        // boundary belongs — the Tiger stops the joystick at the current box.
        // Said out loud, and put back exactly as it was on the way out.
        if (_wasEnforced) {
            try {
                await postJSON(ENFORCE, { enforced: false });
                _note = 'XY limits are off while you edit. They go back on when you finish.';
            } catch (e) {
                say(`Could not release the limits (${e.message}) — you can only shrink the region.`, true);
            }
        }

        // Seeing where you are is the whole basis for judging a corner.
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
        _step = 'a';
        _a = _b = _edited = null;
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
        _note = '';
        _onChange();
    }

    /**
     * Take the corner the stage is standing on.
     *
     * Refuses a second corner on top of the first: two presses without moving
     * would make a region with no inside, and the honest reading of that is a
     * double-press, not an instruction.
     */
    function capture() {
        if (!_open || !_pos) return false;
        if (_step === 'a') {
            _a = { x: _pos.x, y: _pos.y };
            _step = 'b';
            say('');
            _onChange();
            return true;
        }
        if (_step === 'b') {
            if (Math.abs(_pos.x - _a.x) < MIN_SPAN_UM || Math.abs(_pos.y - _a.y) < MIN_SPAN_UM) {
                say('That is the corner you already took. Drive to the opposite one.', true);
                _onChange();
                return false;
            }
            _b = { x: _pos.x, y: _pos.y };
            _step = 'review';
            say('');
            _onChange();
            return true;
        }
        return false;
    }

    /** Undo the last step of the walk. */
    function back() {
        if (!_open) return false;
        if (_step === 'review') { _b = null; _edited = null; _step = 'b'; }
        else if (_step === 'b') { _a = null; _step = 'a'; }
        else return false;
        say('');
        _onChange();
        return true;
    }

    /** Start the walk again from the first corner. */
    function redo() {
        if (!_open) return false;
        _a = _b = _edited = null;
        _step = 'a';
        say('');
        _onChange();
        return true;
    }

    /** Type a bound directly — for people who know the number. */
    function setBound(bound, value) {
        if (!_open || _step !== 'review' || !Number.isFinite(value)) return false;
        const current = box();
        if (!current) return false;
        _edited = Object.assign({}, current, { [bound]: value });
        say('');
        _onChange();
        return true;
    }

    /**
     * Move one edge to where the stage is now.
     *
     * The common case after a walk is one edge being slightly wrong, and
     * re-walking both corners to fix it is a poor trade. Same gesture as a
     * capture, aimed at a single bound.
     */
    function useStage(bound) {
        if (!_pos) return false;
        return setBound(bound, bound.startsWith('x') ? _pos.x : _pos.y);
    }

    /** What changed, in the operator's terms, for the confirmation. */
    function diff() {
        const b = box();
        if (!b || !_applied) return [];
        return BOUNDS
            .filter(k => Math.abs((b[k] ?? 0) - (_applied[k] ?? 0)) > 0.05)
            .map(k => `${k.replace('_', ' ')}: ${fmt(_applied[k])} → ${fmt(b[k])}`);
    }

    async function apply(opts) {
        const b = box();
        if (!_open || !b) return false;
        try {
            await postJSON(APPLY, b);
            const changes = diff();
            await cancel(opts);   // restores the fence and the camera
            tell(changes.length ? `Region applied · ${changes.join(' · ')}` : 'Region applied · unchanged');
            return true;
        } catch (e) {
            say(`Could not apply (${e.message})`, true);
            _onChange();
            return false;
        }
    }

    async function restore(appliedAt) {
        try {
            const d = await postJSON(RESTORE, { applied_at: appliedAt });
            _applied = d.region || null;
            _history = Array.isArray(d.history) ? d.history : _history;
            say('Region restored.');
            _onChange();
            return true;
        } catch (e) {
            say(`Could not restore (${e.message})`, true);
            _onChange();
            return false;
        }
    }

    /**
     * Where the stage is now.
     *
     * This drives more than a readout: at the second corner the proposed box
     * IS the stage position, and whether Capture can be pressed at all
     * depends on having one. So a new position is a change, and says so,
     * rather than waiting for whatever else happens to redraw next.
     */
    function setPosition(x, y) {
        const next = Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
        const moved = !!next !== !!_pos
            || (next && _pos && (Math.abs(next.x - _pos.x) > 0.01
                                 || Math.abs(next.y - _pos.y) > 0.01));
        _pos = next;
        if (moved && _open && _bad && _step === 'b') say('');
        if (moved && _open) _onChange();
    }

    return {
        isOpen, step, box, open, cancel, apply, capture, back, redo,
        setBound, useStage, restore, diff, setPosition, prompt, isBad,
        note: () => _note,
        corners: () => ({
            a: _a ? Object.assign({}, _a) : null,
            b: _b ? Object.assign({}, _b) : null,
        }),
        history: () => _history.slice(),
        applied: () => (_applied ? Object.assign({}, _applied) : null),
        travel: () => (_travel ? Object.assign({}, _travel) : null),
        position: () => (_pos ? Object.assign({}, _pos) : null),
        onChange: fn => { _onChange = typeof fn === 'function' ? fn : () => {}; },
        STEPS,
        MIN_SPAN_UM,
    };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = RegionEditor;
