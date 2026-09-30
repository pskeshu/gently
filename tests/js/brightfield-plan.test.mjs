/**
 * A brightfield run, as a plan.
 *
 *   node --test tests/js/brightfield-plan.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * The SPIM volume is a channel, and a plan can switch it off. What is left is
 * the bottom camera's frame of the field on the cadence, and the plan has to
 * ask for less than a volume run does and send less: no embryos required, no
 * slices, no exposure, no laser preset, no ending read from a stage. Each of
 * those is a thing that, sent anyway, would do something on the microscope.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../gently/ui/web/static/js/acquisition-plan.js');

const SUBJECTS = [{ id: 'embryo_1', label: '1' }, { id: 'embryo_2', label: '2' }];
const IDS = SUBJECTS.map(s => s.id);

const brightfield = extra => P.fromForm({
    interval: 5, intervalUnit: 'min', volumes: false, dicLight: 'led', ...extra,
});

test('a plan takes volumes unless it says it does not', () => {
    assert.equal(P.fromForm({}).spim.enabled, true);
    assert.equal(P.fromForm({ volumes: true }).spim.enabled, true);
    assert.equal(P.fromForm({ volumes: undefined }).spim.enabled, true);
    assert.equal(P.fromForm({ volumes: false }).spim.enabled, false);
});

test('with no volumes the overview is the run, and a round is a frame', () => {
    const plan = P.fromForm({ volumes: false, dic: false, dicEveryRounds: 4 });
    assert.equal(plan.dic.enabled, true);
    assert.equal(plan.dic.everyRounds, 1);
    assert.equal(P.toPayload(brightfield(), IDS).dic.every_seconds, 300);
});

test('it is said as what it is', () => {
    const plan = brightfield({ dicLedPct: 40, dicExposureMs: 20, stopKind: 'timepoints', stopValue: 12 });
    assert.equal(P.describe(plan, SUBJECTS),
        'Every 5 min: one brightfield frame of the field on the bottom camera (20 ms), ' +
        'from the centroid, under the LED at 40 % · after 12 frames. No SPIM volumes.');
});

test('with no embryo placed there is no centroid to claim', () => {
    assert.match(P.describe(brightfield(), []), /from where the stage is/);
    assert.match(P.describe(brightfield(), SUBJECTS), /from the centroid/);
    const pinned = brightfield({ dicPosition: 'here', dicPin: { x: -120.4, y: 88 } });
    assert.match(P.describe(pinned, []), /from -120, 88/);
});

test('it needs no embryos', () => {
    assert.deepEqual(P.validate(brightfield(), []), []);
    assert.deepEqual(P.validate(P.fromForm({}), []), ['No embryos to image — mark some first.']);
});

test('it cannot end on a stage, and says why', () => {
    for (const kind of ['hatching', 'comma', 'all_test_hatched']) {
        const problems = P.validate(brightfield({ stopKind: kind }), IDS);
        assert.equal(problems.length, 1, kind);
        assert.match(problems[0], /takes no volumes to read a stage from/);
    }
    for (const kind of P.BRIGHTFIELD_STOPS) {
        assert.deepEqual(P.validate(brightfield({ stopKind: kind, stopValue: 3 }), IDS), [], kind);
    }
    assert.deepEqual(P.BRIGHTFIELD_STOPS, ['manual', 'timepoints', 'duration']);
});

test('a count of a brightfield run is a count of frames', () => {
    assert.equal(P.stopWords('timepoints', 1, 'frame'), 'after 1 frame');
    assert.equal(P.stopWords('timepoints', 12, 'frame'), 'after 12 frames');
    assert.equal(P.stopWords('timepoints', 12), 'after 12 timepoints');
    assert.deepEqual(P.validate(brightfield({ stopKind: 'timepoints', stopValue: '' }), []),
        ['Stop after how many frames?']);
});

test('nothing of the volume goes on the wire', () => {
    const body = P.toPayload(brightfield({
        slices: 80, exposureMs: 25, laserConfig: '488 and 561',
        monitoringMode: 'expression_monitoring',
        overrides: [{ embryoId: 'embryo_1', kind: 'hatching' }],
    }), IDS);
    assert.equal(body.volumes, false);
    for (const key of ['num_slices', 'exposure_ms', 'laser_config', 'monitoring_mode', 'stop_conditions']) {
        assert.ok(!(key in body), `${key} was sent`);
    }
    // The embryos still are: their centroid is where the frame is taken from.
    assert.deepEqual(body.embryo_ids, IDS);
});

test('a volume run sends what it always sent, and no `volumes` key', () => {
    const body = P.toPayload(P.fromForm({ interval: 5, intervalUnit: 'min', laserConfig: '488 only' }), IDS);
    assert.ok(!('volumes' in body));
    assert.equal(body.num_slices, 50);
    assert.equal(body.exposure_ms, 10);
    assert.equal(body.laser_config, '488 only');
    assert.equal(body.monitoring_mode, 'idle');
});

test('per-embryo endings are dropped from a plan that images no embryo', () => {
    const plan = brightfield({ overrides: [{ embryoId: 'embryo_1', kind: 'hatching' }] });
    assert.deepEqual(plan.overrides, []);
    // ... so a roster that has changed cannot make the plan invalid.
    assert.deepEqual(P.validate(plan, []), []);
});

/* ── the LED's brightness ─────────────────────────────────────────────── */

test('the brightness is sent under the LED, as a whole percent', () => {
    assert.equal(P.toPayload(brightfield({ dicLedPct: '40' }), IDS).dic.led_intensity_pct, 40);
    assert.equal(P.toPayload(brightfield({ dicLedPct: 1 }), IDS).dic.led_intensity_pct, 1);
    assert.equal(P.toPayload(brightfield({ dicLedPct: 100 }), IDS).dic.led_intensity_pct, 100);
});

test('an empty brightness leaves the LED as it is', () => {
    for (const empty of ['', null, undefined]) {
        const plan = brightfield({ dicLedPct: empty });
        assert.equal(plan.dic.ledPct, null);
        assert.ok(!('led_intensity_pct' in P.toPayload(plan, IDS).dic));
        assert.match(P.describe(plan, SUBJECTS), /under the LED ·/);
    }
});

test('a brightness means nothing under another light, and is not sent', () => {
    for (const light of ['room', 'none']) {
        const plan = brightfield({ dicLight: light, dicLedPct: 40 });
        assert.equal(plan.dic.ledPct, null);
        assert.ok(!('led_intensity_pct' in P.toPayload(plan, IDS).dic));
        assert.doesNotMatch(P.describe(plan, SUBJECTS), /%/);
    }
});

test('what is not a brightness is refused, not sent', () => {
    for (const bad of [0, 101, -5, 'bright']) {
        const problems = P.validate(brightfield({ dicLedPct: bad }), IDS);
        assert.deepEqual(problems, ['LED brightness must be a whole percent from 1 to 100.'], String(bad));
    }
    assert.deepEqual(P.LED_PCT, { min: 1, max: 100 });
});

test('a volume run can set the overview\'s brightness too', () => {
    const plan = P.fromForm({ dic: true, dicLight: 'led', dicLedPct: 30 });
    assert.equal(P.toPayload(plan, IDS).dic.led_intensity_pct, 30);
    assert.match(P.describe(plan, SUBJECTS), /one DIC overview per round from the centroid, under the LED at 30 %/);
});

/* ── saved, and read back ─────────────────────────────────────────────── */

test('a brightfield plan survives being saved and read back', () => {
    const plan = brightfield({ dicLedPct: 40, dicExposureMs: 20, stopKind: 'duration', stopValue: 6 });
    const back = P.fromStructure(P.toStructure(plan));
    assert.equal(back.spim.enabled, false);
    assert.deepEqual(back.dic, plan.dic);
    assert.deepEqual(back.stop, plan.stop);
    assert.equal(P.describe(back, SUBJECTS), P.describe(plan, SUBJECTS));
});

test('a saved brightfield plan carries no laser preset to restore', () => {
    const st = P.toStructure(brightfield({ laserConfig: '488 and 561', slices: 80, exposureMs: 25 }));
    assert.equal(st.volumes, false);
    assert.equal(st.laser_config, null);
    assert.equal(st.num_slices, null);
    assert.equal(st.exposure_ms, null);
});

test('read back, the volume fields hold their defaults and not the floor', () => {
    // `Number(null)` is 0, and the floor of 0 is 1: a plan saved without a
    // slice count came back asking for 1 slice at 1 ms. Unticking brightfield
    // on a restored plan would have started that run.
    const back = P.fromStructure(P.toStructure(brightfield()));
    assert.equal(back.spim.slices, 50);
    assert.equal(back.spim.exposureMs, 10);
    const old = P.fromStructure({ cadence_s: 300, num_slices: null, exposure_ms: null });
    assert.equal(old.spim.slices, 50);
    assert.equal(old.spim.exposureMs, 10);
});

test('a plan saved before there was a choice is a volume plan', () => {
    const back = P.fromStructure({ cadence_s: 300, num_slices: 40, exposure_ms: 8 });
    assert.equal(back.spim.enabled, true);
    assert.equal(back.dic.ledPct, null);
    assert.equal(P.toStructure(P.fromForm({})).volumes, true);
});
