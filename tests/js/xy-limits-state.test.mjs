/**
 * The XY-limits store, and the answer it is allowed to keep.
 *
 *   node --test tests/js/xy-limits-state.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * Reported from the rig: the card showed a live X/Y readout —  -1636.0,
 * -1081.1, the stage plainly connected and moving — with "Microscope not
 * connected" printed underneath it. The route was fine; a GET returned 200
 * with `enforced: false`. What was wrong was that the store had asked once,
 * early, been told no, and had no way to ever ask again.
 *
 * It re-read on DEVICE_LAYER_STATE, which fires ONCE per state change and is
 * emitted by boot-banner.js — loaded BEFORE this file. A rig that was already
 * up when the page loaded announces itself into an empty room.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const PATH = '../../gently/ui/web/static/js/xy-limits-state.js';

const GOOD = {
    success: true, enforced: false,
    x_min: -2252.1, x_max: 983.0, y_min: -1677.0, y_max: 586.6,
    region: null,
};

/** A fresh store each time — it holds state, and so does require's cache. */
function load({ ready = false } = {}) {
    globalThis.window = { gentlyDeviceReady: ready };
    delete require.cache[require.resolve(PATH)];
    return require(PATH);
}

/**
 * Load with the rig marked down, then bring it up.
 *
 * Loading with it already up makes the store ask on its own — which is the
 * point of the fix, and tested below. These tests want to drive `read()`
 * themselves, so they arrive a moment earlier.
 */
function loadThenRigUp() {
    const S = load({ ready: false });
    globalThis.window.gentlyDeviceReady = true;
    return S;
}

/** Queue of replies; each fetch takes the next one. */
function replies(...queue) {
    const calls = [];
    globalThis.fetch = async url => {
        calls.push(url);
        const next = queue.length > 1 ? queue.shift() : queue[0];
        if (next === 'boom') throw new Error('network');
        return { ok: next.ok !== false, status: next.status || 200, json: async () => next.body || {} };
    };
    return calls;
}

/** Run scheduled retries now instead of in four seconds. */
function immediateTimers() {
    const pending = [];
    globalThis.setTimeout = fn => { pending.push(fn); return pending.length; };
    globalThis.clearTimeout = () => {};
    // The scheduled callback kicks off an async read without returning its
    // promise, so draining the queue is not enough — let the microtasks run.
    return async () => {
        while (pending.length) {
            pending.shift()();
            await new Promise(r => setImmediate(r));
        }
    };
}

test('a stage it cannot read, while the rig is up, is not "not connected"', async () => {
    // The readout beside it is live. Saying the microscope is absent makes the
    // card contradict itself, and sends the operator to check a cable.
    immediateTimers();
    replies({ ok: false, status: 502, body: { detail: 'nope' } });
    const S = loadThenRigUp();
    const s = await S.read();
    assert.equal(s.enforced, null);
    assert.match(s.reason, /asking again/i);
    assert.doesNotMatch(s.reason, /not connected/i);
});

test('with the rig actually down, it still says so', async () => {
    immediateTimers();
    replies('boom');
    const S = load({ ready: false });
    assert.equal((await S.read()).reason, 'Microscope not connected');
});

test('a read that failed is asked again, and the later answer sticks', async () => {
    const run = immediateTimers();
    replies({ ok: false, status: 502 }, { body: GOOD });
    const S = loadThenRigUp();
    await S.read();
    assert.equal(S.snapshot().enforced, null);
    await run();
    assert.equal(S.snapshot().enforced, false, 'the retry adopted the real answer');
    assert.equal(S.snapshot().reason, '', 'and stopped explaining itself');
});

test('an answer stops the retries', async () => {
    const run = immediateTimers();
    const calls = replies({ body: GOOD });
    const S = loadThenRigUp();
    await S.read();
    await run();
    assert.equal(calls.length, 1, 'no retry was scheduled after a good read');
});

test('a rig that was already up when the page loaded is still heard', async () => {
    // boot-banner.js emits DEVICE_LAYER_STATE once per state change and is
    // loaded first, so this store can miss the only 'ready' it will ever get.
    // The flag boot-banner leaves behind is the same news, still readable.
    immediateTimers();
    const calls = replies({ body: GOOD });
    load({ ready: true });
    await new Promise(r => setImmediate(r));
    assert.equal(calls.length, 1, 'it asked on its own at load');
});

test('a rig that is still coming up is not asked at load', async () => {
    immediateTimers();
    const calls = replies({ body: GOOD });
    load({ ready: false });
    await new Promise(r => setImmediate(r));
    assert.equal(calls.length, 0, 'DEVICE_LAYER_STATE will say when');
});

test('403 is a sign-in problem, not a hardware one, and is not retried', async () => {
    const run = immediateTimers();
    const calls = replies({ ok: false, status: 403 });
    const S = loadThenRigUp();
    assert.match((await S.read()).reason, /sign in/i);
    await run();
    assert.equal(calls.length, 1, 'asking again will not sign anyone in');
});
