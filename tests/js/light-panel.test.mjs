/**
 * Unit tests for the Light panel's derived state.
 *
 *   node --test tests/js/light-panel.test.mjs
 *
 * `emitting()` is the reason this panel exists. `LASER: ON` used to mean "an
 * HTTP request returned 200", so the UI reported ON over a dark microscope
 * (#106). The replacement computes emission from three read-back facts, and
 * the rule that matters most is the one about not knowing: an unread value
 * must produce `null`, never `false`. "No" and "I have not looked" are
 * different answers, and only one of them is safe to act on.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../gently/ui/web/static/js/panels/light.js');

const armed = { a: true, b: true };
const dark = { a: false, b: false };

test('wavelengthsOf: reads the lines out of a config name', () => {
    assert.deepEqual(P.wavelengthsOf('488 and 561'), [488, 561]);
    assert.deepEqual(P.wavelengthsOf('488 only'), [488]);
});

test('wavelengthsOf: ALL OFF routes nothing', () => {
    assert.deepEqual(P.wavelengthsOf('ALL OFF'), []);
});

test('wavelengthsOf: an unknown config means every line is in play', () => {
    // Not knowing which lines are routed is not the same as none being routed.
    assert.deepEqual(P.wavelengthsOf(null), [405, 488, 561, 637]);
});

test('wavelengthsOf: ignores wavelengths the hardware cannot address', () => {
    assert.deepEqual(P.wavelengthsOf('488 and 999'), [488]);
});

test('emitting: armed, routed and powered is emitting', () => {
    assert.equal(P.emitting({ beam: armed, config: '488 only', power: { 488: 4.0 } }), true);
});

test('emitting: a disarmed beam is not emitting, whatever else is set', () => {
    assert.equal(P.emitting({ beam: dark, config: '488 only', power: { 488: 6.0 } }), false);
});

test('emitting: armed but routed to nothing is not emitting', () => {
    // The exact state after "ALL OFF" — PLogic gates every line, beam still armed.
    assert.equal(P.emitting({ beam: armed, config: 'ALL OFF', power: { 488: 4.0 } }), false);
});

test('emitting: armed and routed at zero power is not emitting', () => {
    assert.equal(P.emitting({ beam: armed, config: '488 only', power: { 488: 0 } }), false);
});

test('emitting: one armed side is enough', () => {
    assert.equal(P.emitting({ beam: { a: true, b: false }, config: '488 only', power: { 488: 4 } }), true);
});

test('emitting: an unread beam is unknown, not off', () => {
    // The whole point. Returning false here would reproduce #106 with better
    // arithmetic — a confident "not emitting" over a scope nobody has asked.
    assert.equal(P.emitting({ beam: null, config: '488 only', power: { 488: 4 } }), null);
});

test('emitting: an unreadable side is unknown, not off', () => {
    assert.equal(P.emitting({ beam: { a: null, b: false }, config: '488 only', power: { 488: 4 } }), null);
});

test('emitting: unread power under an armed beam is unknown', () => {
    assert.equal(P.emitting({ beam: armed, config: '488 only', power: {} }), null);
    assert.equal(P.emitting({ beam: armed, config: '488 and 561', power: { 488: 4 } }), null);
});

test('emitting: nothing read at all is unknown', () => {
    assert.equal(P.emitting({}), null);
});

/**
 * `mode()` — the root of the panel's decision tree.
 *
 * The one rule with teeth: `both` must be reachable. LED and laser are two
 * independent Micro-Manager config groups, so nothing in the hardware stops
 * both being open, and #106 is that state going unnoticed for weeks — every
 * vision frame LED brightfield with a 50 ms laser gate on top, the detector
 * hunting nuclei in a DIC-like image. A mode selector that collapsed the pair
 * into an either/or would have made its own founding bug unrepresentable.
 */
test('mode: the LED alone is LED mode', () => {
    assert.equal(P.mode({ led: 'Open', config: 'ALL OFF' }), 'led');
});

test('mode: a routed line alone is laser mode', () => {
    assert.equal(P.mode({ led: 'Closed', config: '488 only' }), 'laser');
});

test('mode: neither is off', () => {
    assert.equal(P.mode({ led: 'Closed', config: 'ALL OFF' }), 'off');
});

test('mode: both open is a state, not an impossibility (#106)', () => {
    assert.equal(P.mode({ led: 'Open', config: '488 only' }), 'both');
});

test('mode: an unread source is unknown, never off', () => {
    assert.equal(P.mode({ led: null, config: 'ALL OFF' }), null);
    assert.equal(P.mode({ led: 'Closed', config: null }), null);
    assert.equal(P.mode({}), null);
});
