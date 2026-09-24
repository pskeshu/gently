/**
 * The region editor's state, without a browser.
 *
 *   node --test tests/js/region-editor.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * The editor's job is to hold a proposed fence while the operator drives the
 * stage around, and to leave the rig exactly as it found it if they change
 * their mind. Both of those are decisions made in this module, before anything
 * is drawn, so this is where they can be checked.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// The module reaches for the strip's message line when it has something to
// say. Nothing here renders, so a stub that swallows it is enough.
globalThis.document = { getElementById: () => null };

const require = createRequire(import.meta.url);
const R = require('../../gently/ui/web/static/js/region-editor.js');

const TRAVEL = { x_min: -2252.1, x_max: 983.0, y_min: -1677.0, y_max: 586.6 };
const REGION = { x_min: -900, x_max: 400, y_min: -800, y_max: 100 };

/** Stand in for the device layer. Records what was written to it. */
function stubStage({ enforced = false, region = REGION, fail = false } = {}) {
    const posts = [];
    globalThis.fetch = async (url, init) => {
        if (fail) return { ok: false, status: 502, json: async () => ({ detail: 'not connected' }) };
        if (init && init.method === 'POST') posts.push({ url, body: JSON.parse(init.body) });
        return {
            ok: true,
            status: 200,
            json: async () => ({ ...region, enforced, region, full_travel: TRAVEL, history: [] }),
        };
    };
    return posts;
}

test('a stage it cannot read is not a region it can edit', async () => {
    stubStage({ fail: true });
    assert.equal(await R.open({}), false);
    assert.equal(R.isOpen(), false);
    assert.equal(R.box(), null);
});

test('opening takes the applied region, and the travel it may grow into', async () => {
    stubStage();
    assert.equal(await R.open({}), true);
    assert.deepEqual(R.box(), REGION);
    assert.deepEqual(R.travel(), TRAVEL);
    await R.cancel({});
});

test('the firmware fence comes down to edit, and goes back up on cancel', async () => {
    // It has to: the Tiger stops the joystick at the current box, so you
    // could never drive to where a WIDER boundary belongs.
    const posts = stubStage({ enforced: true });
    await R.open({});
    assert.deepEqual(posts.map(p => p.body.enforced), [false]);
    await R.cancel({});
    assert.deepEqual(posts.map(p => p.body.enforced), [false, true]);
});

test('a fence that was already off is left off', async () => {
    const posts = stubStage({ enforced: false });
    await R.open({});
    await R.cancel({});
    assert.deepEqual(posts, []);
});

test('stamping an edge writes the live stage position into that bound', async () => {
    stubStage();
    await R.open({});
    R.setPosition(-250.5, 190.0);
    assert.equal(R.stamp('x-max'), true);
    assert.equal(R.box().x_max, -250.5);
    assert.equal(R.box().x_min, REGION.x_min, 'the other bounds are untouched');
    assert.deepEqual(R.diff(), ['x max: 400.0 → -250.5']);
    await R.cancel({});
});

test('a corner sets both of its edges at once', async () => {
    stubStage();
    await R.open({});
    R.setPosition(-1000, -900);
    R.stamp('min-min');
    assert.deepEqual(
        [R.box().x_min, R.box().y_min], [-1000, -900],
        'the corner you are standing on is two bounds, not one',
    );
    await R.cancel({});
});

test('stamping past the opposite edge pushes it out rather than refusing', async () => {
    // Refusing would mean explaining geometry to someone holding a joystick.
    stubStage();
    await R.open({});
    R.setPosition(-950, 0);          // below the current x_min of -900
    R.stamp('x-max');
    assert.equal(R.box().x_max, -950);
    assert.ok(R.box().x_min < R.box().x_max, 'the box still has an inside');
    await R.cancel({});
});

test('with no position there is nothing to stamp', async () => {
    stubStage();
    await R.open({});
    R.setPosition(NaN, NaN);
    assert.equal(R.stamp('x-max'), false);
    assert.deepEqual(R.box(), REGION);
    await R.cancel({});
});

test('every target names bounds that exist', () => {
    const known = new Set(['x_min', 'x_max', 'y_min', 'y_max']);
    for (const [target, bounds] of Object.entries(R.TARGETS)) {
        assert.ok(bounds.length === 1 || bounds.length === 2, target);
        bounds.forEach(b => assert.ok(known.has(b), `${target} -> ${b}`));
    }
    assert.equal(Object.keys(R.TARGETS).length, 8, 'four edges and four corners');
});

test('nothing reaches the controller until Apply', async () => {
    const posts = stubStage();
    await R.open({});
    R.setPosition(-250.5, 190.0);
    R.stamp('x-max');
    R.setBound('y_min', -700);
    assert.deepEqual(posts, [], 'driving and stamping write nothing');
    assert.equal(await R.apply({}), true);
    assert.equal(posts.length, 1);
    assert.equal(posts[0].body.x_max, -250.5);
    assert.equal(posts[0].body.y_min, -700);
    assert.equal(R.isOpen(), false, 'applying closes the editor');
});

test('cancel stops only a camera the editor started', async () => {
    stubStage();
    let stopped = 0;
    await R.open({});                        // no startCamera passed
    await R.cancel({ stopCamera: () => { stopped++; } });
    assert.equal(stopped, 0, 'Operate may own the stream');

    await R.open({ startCamera: async () => {} });
    await R.cancel({ stopCamera: async () => { stopped++; } });
    assert.equal(stopped, 1);
});
