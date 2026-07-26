/**
 * Unit tests for the Operate surface's pure logic.
 *
 *   node --test tests/js/
 *
 * Zero dependencies — Node's built-in runner. There is no root package.json, so
 * operate-math.js resolves as CJS and is pulled in with createRequire.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const M = require('../../gently/ui/web/static/js/operate-math.js');

const FRAME = { w: 400, h: 300, downsample: 1 };
const AT = [1000, 2000];

test('frameToStage: the centre pixel is exactly where the frame was captured', () => {
    assert.deepEqual(M.frameToStage(200, 150, FRAME, AT), [1000, 2000]);
});

test('frameToStage: +X in the image is +X on the stage', () => {
    const [x, y] = M.frameToStage(300, 150, FRAME, AT);
    assert.equal(x, 1000 + 100 * M.BASE_UM_PER_PX);
    assert.equal(y, 2000);
});

test('frameToStage: +Y in the image is -Y on the stage (the sign flip)', () => {
    // Image Y grows downward, stage Y grows upward. Getting this wrong mirrors
    // every embryo about the frame centre.
    const [, below] = M.frameToStage(200, 250, FRAME, AT);
    const [, above] = M.frameToStage(200, 50, FRAME, AT);
    assert.equal(below, 2000 - 100 * M.BASE_UM_PER_PX);
    assert.equal(above, 2000 + 100 * M.BASE_UM_PER_PX);
    assert.ok(above > below);
});

test('frameToStage: downsample scales the offset, not the origin', () => {
    const ds4 = { w: 400, h: 300, downsample: 4 };
    const [x1] = M.frameToStage(300, 150, FRAME, AT);
    const [x4] = M.frameToStage(300, 150, ds4, AT);
    assert.equal(x4 - AT[0], (x1 - AT[0]) * 4);
});

test('frameToStage: refuses to guess without a real capture position', () => {
    // Defaulting to [0,0] here is the bug this guard exists to prevent.
    assert.equal(M.frameToStage(10, 10, FRAME, null), null);
    assert.equal(M.frameToStage(10, 10, FRAME, [0]), null);
    assert.equal(M.frameToStage(10, 10, FRAME, [NaN, 5]), null);
    assert.equal(M.frameToStage(10, 10, null, AT), null);
});

test('stageToFrame: round-trips frameToStage for arbitrary pixels', () => {
    for (const [fx, fy] of [[0, 0], [200, 150], [399, 299], [37, 288]]) {
        const s = M.frameToStage(fx, fy, FRAME, AT);
        const [bx, by] = M.stageToFrame(s[0], s[1], FRAME, AT);
        assert.ok(Math.abs(bx - fx) < 1e-9 && Math.abs(by - fy) < 1e-9,
            `(${fx},${fy}) round-tripped to (${bx},${by})`);
    }
});

test('stageToFrame: a marker re-projects as the stage moves under it', () => {
    // Marker pinned to the sample at stage centre of the first frame.
    const marker = M.frameToStage(200, 150, FRAME, AT);
    // Stage then moves +X by 100 px worth of travel; the marker must slide the
    // opposite way in the frame, i.e. stay on the same bit of sample.
    const moved = [AT[0] + 100 * M.BASE_UM_PER_PX, AT[1]];
    const [fx, fy] = M.stageToFrame(marker[0], marker[1], FRAME, moved);
    assert.ok(Math.abs(fx - 100) < 1e-9, `expected x≈100, got ${fx}`);
    assert.equal(fy, 150);
});

test('stageToFrame: refuses to guess on the same conditions as frameToStage', () => {
    assert.equal(M.stageToFrame(0, 0, FRAME, null), null);
    assert.equal(M.stageToFrame(0, 0, null, AT), null);
    assert.equal(M.stageToFrame(0, 0, FRAME, [NaN, 1]), null);
});

test('fdBand: each height offers steps proportionate to it', () => {
    assert.deepEqual(M.fdBand(25000).steps, [5000, 1000]);
    assert.deepEqual(M.fdBand(5000).steps, [1000, 500]);
    assert.deepEqual(M.fdBand(1500).steps, [500, 100]);
    assert.deepEqual(M.fdBand(500).steps, [100, 50]);
    assert.deepEqual(M.fdBand(60).steps, [50, 10, 5]);
});

test('fdBand: boundaries are strict, so a boundary value takes the finer band', () => {
    assert.deepEqual(M.fdBand(10000).steps, [1000, 500]);
    assert.deepEqual(M.fdBand(2000).steps, [500, 100]);
    assert.deepEqual(M.fdBand(1000).steps, [100, 50]);
    assert.deepEqual(M.fdBand(200).steps, [50, 10, 5]);
});

test('fdBand: unknown position falls to the finest band, not the coarsest', () => {
    assert.deepEqual(M.fdBand(null).steps, [50, 10, 5]);
    assert.deepEqual(M.fdBand(undefined).steps, [50, 10, 5]);
    assert.deepEqual(M.fdBand(NaN).steps, [50, 10, 5]);
});

test('fdBand: steps get finer, never coarser, as the sample closes in', () => {
    let prev = Infinity;
    for (const pos of [25000, 10000, 5000, 2000, 1000, 500, 200, 60, 31]) {
        const coarsest = Math.max(...M.fdBand(pos).steps);
        assert.ok(coarsest <= prev, `at ${pos} µm the offered step grew to ${coarsest}`);
        prev = coarsest;
    }
});

// Banding is an affordance; this is the safety gate. They are separate on
// purpose — at 31 µm the band still offers ±50, and it is stepAllowed (plus the
// server fence) that stops the down-half of it.
test('stepAllowed: a down-step may never exceed the remaining travel', () => {
    assert.equal(M.stepAllowed(-50, 1), false);
    assert.equal(M.stepAllowed(-50, 49.9), false);
    assert.equal(M.stepAllowed(-50, 50), true);
    assert.equal(M.stepAllowed(-50, 200), true);
    assert.equal(M.stepAllowed(-5, 0), false);
});

test('stepAllowed: up-steps are always offered; the server fences the ceiling', () => {
    assert.equal(M.stepAllowed(100, 0), true);
    assert.equal(M.stepAllowed(5000, 1), true);
    assert.equal(M.stepAllowed(0, 0), true);
});

test('stepAllowed: unknown travel allows the step rather than faking knowledge', () => {
    // The server-side F_DRIVE_MIN_UM fence is the real backstop here.
    assert.equal(M.stepAllowed(-50, null), true);
    assert.equal(M.stepAllowed(-50, undefined), true);
    assert.equal(M.stepAllowed(-50, NaN), true);
    assert.equal(M.stepAllowed(NaN, 500), false);
});

test('stepAllowed: every band step is gated correctly at its own band floor', () => {
    // Walk the real bands and confirm nothing slips past the gate near the floor.
    for (const pos of [25000, 5000, 1500, 500, 60]) {
        for (const s of M.fdBand(pos).steps) {
            assert.equal(M.stepAllowed(-s, s - 1), false);
            assert.equal(M.stepAllowed(-s, s), true);
        }
    }
});

test('gaugeFraction: linear maps min/mid/max to 0/0.5/1 and clamps outside', () => {
    assert.equal(M.gaugeFraction(50, 50, 250), 0);
    assert.equal(M.gaugeFraction(150, 50, 250), 0.5);
    assert.equal(M.gaugeFraction(250, 50, 250), 1);
    assert.equal(M.gaugeFraction(10, 50, 250), 0);
    assert.equal(M.gaugeFraction(900, 50, 250), 1);
});

test('gaugeFraction: returns null rather than drawing a marker it cannot place', () => {
    assert.equal(M.gaugeFraction(null, 50, 250), null);
    assert.equal(M.gaugeFraction(100, null, 250), null);
    assert.equal(M.gaugeFraction(100, 50, null), null);
    assert.equal(M.gaugeFraction(100, 250, 250), null);  // zero span
    assert.equal(M.gaugeFraction(100, 250, 50), null);   // inverted
});

test('gaugeFraction: log still anchors both ends of the F-drive axis', () => {
    assert.equal(M.gaugeFraction(30, 30, 25000, 'log'), 0);
    assert.equal(M.gaugeFraction(25000, 30, 25000, 'log'), 1);
});

test('gaugeFraction: log makes the approach region visible where linear cannot', () => {
    // The whole approach-and-crash region is the last 200 µm. Linearly that is
    // under 1% of the track — sub-pixel, i.e. the gauge is blind exactly where
    // the operator needs it.
    const linear = M.gaugeFraction(200, 30, 25000);
    const log = M.gaugeFraction(200, 30, 25000, 'log');
    assert.ok(linear < 0.01, `linear put 200 µm at ${linear}`);
    assert.ok(log > 0.3, `log put 200 µm at ${log}`);
    assert.ok(M.gaugeFraction(100, 30, 25000, 'log') < log);
});

test('isEngaged: with no telemetry the latch alone decides (fail safe)', () => {
    assert.equal(M.isEngaged(true, null), true);
    assert.equal(M.isEngaged(false, null), false);
});

test('isEngaged: measured close locks regardless of the latch', () => {
    assert.equal(M.isEngaged(false, 500), true);
    assert.equal(M.isEngaged(true, 500), true);
    assert.equal(M.isEngaged(false, 0), true);
});

test('isEngaged: measured clear clears a stale latch', () => {
    // Someone raised the head at the controller box. The old rule could only ever
    // ADD lock from telemetry, so XY stayed locked forever with no UI escape.
    assert.equal(M.isEngaged(true, 5000), false);
    assert.equal(M.isEngaged(false, 5000), false);
});

test('isEngaged: inside the hysteresis band the latch breaks the tie', () => {
    assert.equal(M.isEngaged(true, 1500), true);
    assert.equal(M.isEngaged(false, 1500), false);
});

test('isEngaged: non-finite telemetry is treated as no telemetry, not as clear', () => {
    assert.equal(M.isEngaged(true, NaN), true);
    assert.equal(M.isEngaged(true, undefined), true);
});

test('isEngaged: the lock is monotone in proximity for a given latch', () => {
    // Closer must never be safer.
    for (const latch of [true, false]) {
        let prev = M.isEngaged(latch, 20000);
        for (const floor of [5000, 2500, 2000, 1500, 1000, 500, 0]) {
            const now = M.isEngaged(latch, floor);
            assert.ok(!(prev === true && now === false),
                `latch=${latch}: lock released while closing in at ${floor} µm`);
            prev = now;
        }
    }
});
