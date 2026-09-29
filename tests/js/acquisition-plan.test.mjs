/**
 * The acquisition plan, said once.
 *
 *   node --test tests/js/acquisition-plan.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * The pane reads a form into a plan; this module says it and turns it into
 * the start request. Both directions are what a biologist checks before
 * pressing Start, so both are what is tested: does the sentence say what the
 * form holds, and does the request carry exactly the sentence.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../gently/ui/web/static/js/acquisition-plan.js');

const SUBJECTS = [{ id: 'embryo_1', label: '1' }, { id: 'embryo_2', label: '2' },
                  { id: 'embryo_3', label: '3' }, { id: 'embryo_4', label: '4' }];
const IDS = SUBJECTS.map(s => s.id);

test('an untouched form is still a complete, sayable plan', () => {
    const plan = P.fromForm({});
    assert.equal(plan.intervalSeconds, 120);
    assert.equal(plan.spim.slices, 50);
    assert.equal(plan.dic.enabled, false);
    assert.equal(plan.stop.kind, 'manual');
    assert.deepEqual(P.validate(plan, IDS), []);
    assert.match(P.describe(plan, SUBJECTS), /^Every 2 min: SPIM volumes \(50 slices · 10 ms\) of 4 embryos · until stopped\.$/);
});

test('minutes are the unit the biologist thinks in; seconds are what is sent', () => {
    const plan = P.fromForm({ interval: 5, intervalUnit: 'min' });
    assert.equal(plan.intervalSeconds, 300);
    assert.equal(P.toPayload(plan, IDS).interval_seconds, 300);
    assert.match(P.describe(plan, SUBJECTS), /^Every 5 min/);
    assert.match(P.describe(P.fromForm({ interval: 90, intervalUnit: 's' }), SUBJECTS), /^Every 90 s/);
    assert.match(P.describe(P.fromForm({ interval: 2, intervalUnit: 'min' }), SUBJECTS), /^Every 2 min/);
});

test('the DIC channel rides on its own clock, in rounds', () => {
    const plan = P.fromForm({ interval: 5, intervalUnit: 'min', dic: true, dicEveryRounds: 3, dicExposureMs: 8 });
    const body = P.toPayload(plan, IDS);
    assert.deepEqual(body.dic, { enabled: true, every_seconds: 900, position: null, exposure_ms: 8, light: 'room' });
    assert.match(P.describe(plan, SUBJECTS), /\+ one DIC overview every 3 rounds from the centroid/);
    assert.match(P.describe(P.fromForm({ dic: true }), SUBJECTS), /one DIC overview per round from the centroid/);
});

test('a plan without DIC sends no dic at all', () => {
    // The orchestrator treats an absent dic as off, and the existing exact-kwargs
    // route tests pin that a plain run calls start() exactly as before.
    assert.equal('dic' in P.toPayload(P.fromForm({}), IDS), false);
});

test('"taken from here" is the position captured when it was chosen', () => {
    const plan = P.fromForm({ dic: true, dicPosition: 'here', dicPin: { x: -512.4, y: -388.9 } });
    assert.deepEqual(P.toPayload(plan, IDS).dic.position, { x: -512.4, y: -388.9 });
    assert.match(P.describe(plan, SUBJECTS), /from -512, -389/);
    // and without a captured position it is not a plan yet
    const none = P.fromForm({ dic: true, dicPosition: 'here', dicPin: null });
    assert.match(P.validate(none, IDS).join(' '), /no stage position/i);
});

test('how it ends, for the run and for one embryo', () => {
    const plan = P.fromForm({
        stopKind: 'duration', stopValue: 12,
        overrides: [{ embryoId: 'embryo_2', kind: 'hatching' },
                    { embryoId: 'embryo_3', kind: 'timepoints', value: 3 },
                    { embryoId: 'embryo_4', kind: 'default' }],
    });
    const body = P.toPayload(plan, IDS);
    assert.equal(body.stop_condition, 'duration:12h');
    assert.deepEqual(body.stop_conditions, { embryo_2: 'hatching', embryo_3: 'timepoints:3' },
        '"as the run" is no override, and the run keeps its own');
    assert.match(P.describe(plan, SUBJECTS), /· after 12 h; 2 at hatching, 3 after 3 timepoints\.$/);
});

test('the orchestrator hears the combined spec, never the bare word', () => {
    assert.equal(P.stopSpec('timepoints', 12), 'timepoints:12');
    assert.equal(P.stopSpec('timepoints', '7.6'), 'timepoints:8');
    assert.equal(P.stopSpec('duration', 6), 'duration:6h');
    assert.equal(P.stopSpec('manual'), 'manual');
    assert.equal(P.stopSpec('nonsense'), 'manual');
});

test('a stop that needs a number refuses to start without one', () => {
    assert.match(P.validate(P.fromForm({ stopKind: 'timepoints', stopValue: '' }), IDS)[0], /how many timepoints/);
    assert.match(P.validate(P.fromForm({ stopKind: 'duration', stopValue: 0 }), IDS)[0], /how many hours/);
    assert.deepEqual(P.validate(P.fromForm({ stopKind: 'timepoints', stopValue: 12 }), IDS), []);
});

test('no embryos is the first thing it says', () => {
    assert.match(P.validate(P.fromForm({}), [])[0], /No embryos/);
    assert.match(P.describe(P.fromForm({}), []), /of no embryos/);
});

test('an override for an embryo not in the run is a problem, not a silent drop', () => {
    const plan = P.fromForm({ overrides: [{ embryoId: 'embryo_9', kind: 'hatching' }] });
    assert.match(P.validate(plan, IDS).join(' '), /embryo_9 is not in this run/);
});

test('the SPIM channel carries its settings, and the preset only when chosen', () => {
    const plan = P.fromForm({ slices: 80, exposureMs: 12, laserConfig: '488 and 561' });
    const body = P.toPayload(plan, IDS);
    assert.equal(body.num_slices, 80);
    assert.equal(body.exposure_ms, 12);
    assert.equal(body.laser_config, '488 and 561');
    assert.match(P.describe(plan, SUBJECTS), /\(80 slices · 12 ms · 488 and 561\)/);
    assert.equal('laser_config' in P.toPayload(P.fromForm({}), IDS), false);
});

test('one embryo is named, several are counted', () => {
    assert.match(P.describe(P.fromForm({}), [SUBJECTS[1]]), /of embryo 2 ·/);
    assert.match(P.describe(P.fromForm({}), SUBJECTS), /of 4 embryos ·/);
});

// ── a plan ⇄ a saved tactic's structure ──────────────────────────────────

test('a plan survives being saved and reloaded', () => {
    const plan = P.fromForm({
        interval: 5, intervalUnit: 'min', slices: 80, exposureMs: 12, laserConfig: '488 and 561',
        dic: true, dicEveryRounds: 2, dicPosition: 'here', dicPin: { x: -500, y: -400 }, dicExposureMs: 8,
        stopKind: 'duration', stopValue: 12,
        overrides: [{ embryoId: 'embryo_2', kind: 'hatching' }, { embryoId: 'embryo_3', kind: 'timepoints', value: 3 }],
        monitoringMode: 'expression_monitoring',
    });
    const st = P.toStructure(plan);
    assert.equal(st.cadence_s, 300);
    assert.equal(st.stop_condition, 'duration:12h');
    assert.deepEqual(st.dic, { enabled: true, every_seconds: 600, position: { x: -500, y: -400 }, exposure_ms: 8, light: 'room' });
    assert.deepEqual(st.stop_conditions, { embryo_2: 'hatching', embryo_3: 'timepoints:3' });

    const back = P.fromStructure(st);
    assert.deepEqual(back, plan, 'what was saved is what comes back');
    assert.equal(P.describe(back, SUBJECTS), P.describe(plan, SUBJECTS), 'and says the same sentence');
});

test('a structure the start route seeded, before any of this, still reads as a plan', () => {
    // The Adaptive start has always seeded {cadence_s, interval, stop_condition,
    // condition_value, monitoring_mode}. Nothing else — so nothing else is required.
    const plan = P.fromStructure({ cadence_s: 120, interval: 120, stop_condition: 'manual',
                                   condition_value: null, monitoring_mode: 'idle' });
    assert.equal(plan.intervalSeconds, 120);
    assert.equal(plan.dic.enabled, false);
    assert.deepEqual(plan.overrides, []);
    assert.match(P.describe(plan, SUBJECTS), /^Every 2 min: SPIM volumes \(50 slices · 10 ms\) of 4 embryos · until stopped\.$/);
});

test('the stop spec parses back to what the pane offers', () => {
    assert.deepEqual(P.parseStopSpec('timepoints:12'), { kind: 'timepoints', value: 12 });
    assert.deepEqual(P.parseStopSpec('duration:6h'), { kind: 'duration', value: 6 });
    assert.deepEqual(P.parseStopSpec('hatching+3'), { kind: 'hatching', value: null });
    assert.deepEqual(P.parseStopSpec('something_else'), { kind: 'manual', value: null });
    assert.deepEqual(P.parseStopSpec(undefined), { kind: 'manual', value: null });
});

test('a DIC interval that is not a whole number of rounds rounds to one', () => {
    const plan = P.fromStructure({ cadence_s: 300, dic: { enabled: true, every_seconds: 700 } });
    assert.equal(plan.dic.everyRounds, 2);
});

test('the agent’s stage-based ending reads back as the pane’s own', () => {
    // A resumed session written by the agent ends at "stages(hatched,hatching)";
    // the pane has that ending, by name. It used to read back as "until stopped".
    assert.deepEqual(P.parseStopSpec('stages(hatched,hatching)'), { kind: 'hatching', value: null });
    assert.deepEqual(P.parseStopSpec('stages(comma)'), { kind: 'comma', value: null });
    assert.deepEqual(P.parseStopSpec('stages(twofold)'), { kind: 'manual', value: null });
    assert.equal(P.fromStructure({ stop_condition: 'stages(hatched,hatching)' }).stop.kind, 'hatching');
});

test('the plan says which light the overview is taken under', () => {
    // The bottom camera drives no light of its own. A night of overview
    // frames came out dark because nothing said which light to use.
    const room = P.fromForm({ dic: true });
    assert.equal(room.dic.light, 'room', 'the room light is what this rig usually uses');
    assert.match(P.describe(room, SUBJECTS), /from the centroid, under the room light/);
    const led = P.fromForm({ dic: true, dicLight: 'led' });
    assert.equal(P.toPayload(led, IDS).dic.light, 'led');
    assert.match(P.describe(led, SUBJECTS), /under the LED/);
    const asIs = P.fromForm({ dic: true, dicLight: 'none' });
    assert.match(P.describe(asIs, SUBJECTS), /in the light as it is/);
    assert.equal(P.fromForm({ dic: true, dicLight: 'sunlight' }).dic.light, 'room');
    // saved and reloaded, the light comes back
    assert.equal(P.fromStructure(P.toStructure(led)).dic.light, 'led');
    // a plan saved before the light existed is taken under the room light
    assert.equal(P.fromStructure({ dic: { enabled: true, use_led: true } }).dic.light, 'room');
    assert.deepEqual(Object.keys(P.DIC_LIGHTS), ['room', 'led', 'none']);
});
