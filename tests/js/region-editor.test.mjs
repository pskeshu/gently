/**
 * The region walk, without a browser.
 *
 *   node --test tests/js/region-editor.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * Two corners, driven to and captured. The editor's job is to hold a proposed
 * fence while the operator drives the stage around, to say at every moment
 * what it is asking for, and to leave the rig exactly as it found it if they
 * change their mind. All three are decided here, before anything is drawn.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// The module reaches for the strip when it has something to say. Nothing here
// renders, so a stub that swallows it is enough.
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

/** Drive somewhere and take the corner. */
function captureAt(x, y) {
    R.setPosition(x, y);
    return R.capture();
}

test('a stage it cannot read is not a region it can edit', async () => {
    stubStage({ fail: true });
    assert.equal(await R.open({}), false);
    assert.equal(R.isOpen(), false);
});

test('it opens by asking for the first corner, in words', async () => {
    stubStage();
    assert.equal(await R.open({}), true);
    assert.equal(R.step(), 'a');
    assert.match(R.prompt(), /bottom-left/i);
    assert.match(R.prompt(), /capture/i, 'it names the button it wants pressed');
    assert.equal(R.box(), null, 'nothing is proposed before the first corner');
    await R.cancel({});
});

test('the box follows the stage while you drive to the second corner', async () => {
    // This is the whole of "point to the bottom left, then the top right":
    // between the two presses the region is wherever the joystick is.
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    assert.equal(R.step(), 'b');

    R.setPosition(-400, -200);
    assert.deepEqual(R.box(), { x_min: -800, x_max: -400, y_min: -600, y_max: -200 });

    R.setPosition(0, 100);
    assert.deepEqual(R.box(), { x_min: -800, x_max: 0, y_min: -600, y_max: 100 },
        'it grew with the stage, without a second press');
    await R.cancel({});
});

test('two opposite corners in any order give the same region', async () => {
    // The prompts name an order because a sequence needs one. The box does not.
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    captureAt(-200, 100);
    const walked = R.box();
    await R.cancel({});

    await R.open({});
    captureAt(-200, 100);
    captureAt(-800, -600);
    assert.deepEqual(R.box(), walked, 'top-right first is not a mistake');
    await R.cancel({});
});

test('pressing Capture twice in the same place is a double-press, not a region', async () => {
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    assert.equal(captureAt(-800, -600), false);
    assert.equal(R.step(), 'b', 'still asking for the opposite corner');
    assert.equal(R.isBad(), true);
    assert.match(R.prompt(), /already took/i);
    await R.cancel({});
});

test('Back undoes one step of the walk, and Redo restarts it', async () => {
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    captureAt(-200, 100);
    assert.equal(R.step(), 'review');

    R.back();
    assert.equal(R.step(), 'b', 'the second corner is given back');
    R.back();
    assert.equal(R.step(), 'a');
    assert.equal(R.box(), null);
    assert.equal(R.back(), false, 'there is nothing before the first corner');

    captureAt(-800, -600);
    captureAt(-200, 100);
    R.redo();
    assert.equal(R.step(), 'a');
    await R.cancel({});
});

test('after the walk, one edge can be moved without walking again', async () => {
    // The common case is a single edge being slightly wrong, and re-walking
    // both corners to fix it is a poor trade.
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    captureAt(-200, 100);

    R.setPosition(-250.5, 190);
    assert.equal(R.useStage('x_max'), true);
    assert.equal(R.box().x_max, -250.5);
    assert.equal(R.box().x_min, -800, 'the other edges stay where they were walked');

    R.setBound('y_min', -700);
    assert.equal(R.box().y_min, -700);
    await R.cancel({});
});

test('edges cannot be nudged mid-walk, only once there is a box', async () => {
    stubStage();
    await R.open({});
    R.setPosition(-250, 0);
    assert.equal(R.useStage('x_max'), false, 'there is no box to correct yet');
    await R.cancel({});
});

test('the firmware fence comes down to walk, and goes back up on cancel', async () => {
    // It has to: the Tiger stops the joystick at the current box, so you could
    // never drive to where a WIDER boundary belongs.
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

test('nothing reaches the controller until Apply', async () => {
    const posts = stubStage();
    await R.open({});
    captureAt(-800, -600);
    captureAt(-200, 100);
    assert.deepEqual(posts, [], 'driving and capturing write nothing');

    assert.equal(await R.apply({}), true);
    assert.equal(posts.length, 1);
    assert.deepEqual(posts[0].body, { x_min: -800, x_max: -200, y_min: -600, y_max: 100 });
    assert.equal(R.isOpen(), false, 'applying closes the walk');
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

test('every step says what it is asking for', async () => {
    // The complaint that produced this rewrite was not a missing feature:
    // "it is not clear, what it is asking".
    stubStage();
    await R.open({});
    const seen = [];
    seen.push(R.prompt());
    captureAt(-800, -600);
    seen.push(R.prompt());
    captureAt(-200, 100);
    seen.push(R.prompt());
    assert.equal(seen.length, R.STEPS.length);
    seen.forEach(line => {
        assert.ok(line && line.length > 20, `a step says nothing useful: ${line}`);
        assert.match(line, /[.!]$/, 'prompts are sentences');
    });
    await R.cancel({});
});

test('taking the fence down does not take the instruction with it', async () => {
    // The limits notice used to be written into the prompt slot, so step 1
    // showed "XY limits are off" and never "drive to the bottom-left".
    stubStage({ enforced: true });
    await R.open({});
    assert.match(R.prompt(), /bottom-left/i);
    assert.match(R.prompt(), /capture/i);
    assert.match(R.note(), /limits are off/i, 'the aside is still said, beside the prompt');
    await R.cancel({});
    assert.equal(R.note(), '', 'the aside leaves with the walk');
});

test('a refusal at the second corner clears once the stage moves', async () => {
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    assert.equal(captureAt(-800, -600), false);
    assert.equal(R.isBad(), true);
    R.setPosition(-500, -300);
    assert.equal(R.isBad(), false, 'driving away answers "drive to the opposite one"');
    assert.match(R.prompt(), /top-right/i);
    await R.cancel({});
});

test('Apply says what it did, somewhere that is still on screen', async () => {
    // apply() closes the walk, and the strip with it, so a message written to
    // the strip afterwards was never seen.
    const toasts = [];
    globalThis.showGentlyToast = msg => toasts.push(msg);
    stubStage();
    await R.open({});
    captureAt(-800, -600);
    captureAt(-200, 100);
    assert.equal(await R.apply({}), true);
    assert.equal(toasts.length, 1);
    assert.match(toasts[0], /applied/i);
    assert.match(toasts[0], /-800/, 'it names the numbers that changed');
    delete globalThis.showGentlyToast;
});
