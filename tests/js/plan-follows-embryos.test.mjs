/**
 * A blank field is "as each embryo has".
 *
 *   node --test tests/js/plan-follows-embryos.test.mjs
 *
 * The SPIM fields on the Acquisition pane are derived from the embryos the
 * run targets. When those embryos disagree — the agent set one of them to 30
 * slices — the field cannot show one number, so it shows none, and a plan
 * with a blank field must send nothing for it: each embryo keeps its own.
 * A form with the field absent altogether (a template, an older page) still
 * gets the default, so nothing that worked before says less.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../gently/ui/web/static/js/acquisition-plan.js');

const SUBJECTS = [{ id: 'embryo_1', label: '1' }, { id: 'embryo_2', label: '2' }];
const IDS = SUBJECTS.map(s => s.id);

test('a blank field is left alone; an absent one takes the default', () => {
    assert.equal(P.fromForm({ slices: '' }).spim.slices, null);
    assert.equal(P.fromForm({ exposureMs: '' }).spim.exposureMs, null);
    assert.equal(P.fromForm({}).spim.slices, 50);
    assert.equal(P.fromForm({}).spim.exposureMs, 10);
    assert.equal(P.fromForm({ slices: undefined }).spim.slices, 50);
});

test('nothing is sent for a blank field, so each embryo keeps its own', () => {
    const body = P.toPayload(P.fromForm({ slices: '', exposureMs: 12 }), IDS);
    assert.ok(!('num_slices' in body));
    assert.equal(body.exposure_ms, 12);
    const both = P.toPayload(P.fromForm({ slices: '', exposureMs: '' }), IDS);
    assert.ok(!('num_slices' in both) && !('exposure_ms' in both));
});

test('the sentence says so', () => {
    const plan = P.fromForm({ interval: 5, intervalUnit: 'min', slices: '', exposureMs: '' });
    assert.match(P.describe(plan, SUBJECTS),
        /^Every 5 min: SPIM volumes \(slices as each embryo has · exposure as each embryo has\) of 2 embryos/);
    assert.match(P.describe(P.fromForm({ slices: '' , exposureMs: 8 }), SUBJECTS), /slices as each embryo has · 8 ms/);
});

test('a blank field is still a valid plan', () => {
    assert.deepEqual(P.validate(P.fromForm({ slices: '', exposureMs: '' }), IDS), []);
});
