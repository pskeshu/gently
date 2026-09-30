/**
 * Laser power, in the plan.
 *
 *   node --test tests/js/plan-laser-power.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * The SPIM volume channel asked which lines to route and never how hard. A
 * power is per line, and what it may be is the device layer's to say — 488 is
 * held to 2-6 % on this rig — so this module is handed the limits and writes
 * none down. What it owns is that the sentence, the request and the saved
 * plan all carry the same powers, for the same lines.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../gently/ui/web/static/js/acquisition-plan.js');

const SUBJECTS = [{ id: 'embryo_1', label: '1' }, { id: 'embryo_2', label: '2' }];
const IDS = SUBJECTS.map(s => s.id);
// As /api/devices/laser/limits gives them.
const LIMITS = {
    405: { min: 0, max: 100 }, 488: { min: 2, max: 6 },
    561: { min: 0, max: 100 }, 637: { min: 0, max: 100 },
};

const plan = extra => P.fromForm({ interval: 5, intervalUnit: 'min', ...extra });

test('the lines are the ones the preset names', () => {
    assert.deepEqual(P.linesOf('488 only'), [488]);
    assert.deepEqual(P.linesOf('488 and 561'), [488, 561]);
    assert.deepEqual(P.linesOf('561 and 488'), [488, 561]);
    assert.deepEqual(P.linesOf('ALL OFF'), []);
    assert.deepEqual(P.linesOf('488 and 999'), [488]);
});

test('with no preset chosen no line is ruled out', () => {
    for (const none of ['', null, undefined]) assert.deepEqual(P.linesOf(none), [405, 488, 561, 637]);
    assert.deepEqual(P.LASER_LINES, [405, 488, 561, 637]);
});

test('an untouched form sets no power', () => {
    const p = plan();
    assert.deepEqual(p.spim.laserPowers, {});
    assert.ok(!('laser_powers' in P.toPayload(p, IDS)));
    assert.doesNotMatch(P.describe(p, SUBJECTS), /%/);
});

test('a power is said, and sent, per line', () => {
    const p = plan({ laserConfig: '488 and 561', laserPowers: { 488: '4', 561: '12.5' } });
    assert.deepEqual(p.spim.laserPowers, { 488: 4, 561: 12.5 });
    assert.deepEqual(P.toPayload(p, IDS).laser_powers, { 488: 4, 561: 12.5 });
    assert.match(P.describe(p, SUBJECTS),
        /SPIM volumes \(50 slices · 10 ms · 488 and 561 · 488 at 4 % · 561 at 12\.5 %\) of 2 embryos/);
});

test('an empty line keeps the power it has: it is not sent', () => {
    for (const empty of ['', null, undefined]) {
        const p = plan({ laserConfig: '488 and 561', laserPowers: { 488: 4, 561: empty } });
        assert.deepEqual(P.toPayload(p, IDS).laser_powers, { 488: 4 });
    }
});

test('a power for a line the preset does not route is dropped', () => {
    // Typed under "488 and 561", then the preset changed to "488 only".
    const p = plan({ laserConfig: '488 only', laserPowers: { 488: 4, 561: 12.5 } });
    assert.deepEqual(p.spim.laserPowers, { 488: 4 });
    assert.doesNotMatch(P.describe(p, SUBJECTS), /561/);
});

test('the limits are the ones it is handed', () => {
    const at = pct => P.validate(plan({ laserConfig: '488 only', laserPowers: { 488: pct } }), IDS, LIMITS);
    assert.deepEqual(at(2), []);
    assert.deepEqual(at(6), []);
    assert.deepEqual(at(4.5), []);
    for (const bad of [1.9, 6.1, 50, 0, -1]) {
        assert.deepEqual(at(bad), ['488 nm power must be from 2 to 6 %.'], String(bad));
    }
    // Another rig, another limit: nothing here knows the number.
    const loose = { 488: { min: 0, max: 20 } };
    assert.deepEqual(
        P.validate(plan({ laserConfig: '488 only', laserPowers: { 488: 15 } }), IDS, loose), []);
});

test('each line is held to its own limit', () => {
    const p = plan({ laserConfig: '488 and 561', laserPowers: { 488: 50, 561: 50 } });
    assert.deepEqual(P.validate(p, IDS, LIMITS), ['488 nm power must be from 2 to 6 %.']);
});

test('what is not a number is refused, not sent as nothing', () => {
    const p = plan({ laserConfig: '488 only', laserPowers: { 488: 'lots' } });
    assert.deepEqual(P.validate(p, IDS, LIMITS), ['488 nm power must be from 2 to 6 %.']);
});

test('without the limits a power is still a percentage', () => {
    const at = pct => P.validate(plan({ laserConfig: '561 only', laserPowers: { 561: pct } }), IDS);
    assert.deepEqual(at(50), []);
    assert.deepEqual(at(101), ['561 nm power must be from 0 to 100 %.']);
    assert.deepEqual(at(-1), ['561 nm power must be from 0 to 100 %.']);
});

test('a brightfield run sets no power, whatever is in the form', () => {
    const p = plan({ volumes: false, dicLight: 'led', laserConfig: '488 only', laserPowers: { 488: 50 } });
    assert.deepEqual(p.spim.laserPowers, {});
    assert.ok(!('laser_powers' in P.toPayload(p, IDS)));
    // ... and a number left in a hidden field cannot make the plan invalid.
    assert.deepEqual(P.validate(p, [], LIMITS), []);
    assert.equal(P.toStructure(p).laser_powers, null);
});

test('the powers survive being saved and read back', () => {
    const p = plan({ laserConfig: '488 and 561', laserPowers: { 488: 4, 561: 12.5 } });
    const st = P.toStructure(p);
    assert.deepEqual(st.laser_powers, { 488: 4, 561: 12.5 });
    const back = P.fromStructure(st);
    assert.deepEqual(back.spim, p.spim);
    assert.equal(P.describe(back, SUBJECTS), P.describe(p, SUBJECTS));
});

test('read back from the server, where the lines are strings', () => {
    // JSON has no integer keys: acquisition.yaml and the tactic give "488".
    const back = P.fromStructure({ cadence_s: 300, laser_config: '488 only', laser_powers: { '488': 4 } });
    assert.deepEqual(back.spim.laserPowers, { 488: 4 });
});

test('a plan saved before there were powers has none', () => {
    const back = P.fromStructure({ cadence_s: 300, num_slices: 40, laser_config: '488 only' });
    assert.deepEqual(back.spim.laserPowers, {});
    assert.equal(P.toStructure(plan()).laser_powers, null);
});
