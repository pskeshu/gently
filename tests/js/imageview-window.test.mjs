/**
 * The display window's linear map.
 *
 *   node --test tests/js/imageview-window.test.mjs
 *
 * `out = (in - lo) / (hi - lo)`, expressed as an SVG feComponentTransfer
 * linear function. Worth testing because CSS cannot do this and the reason is
 * easy to forget: brightness() and contrast() are both multiplicative about
 * their own origin, so composing them gives no free intercept, and an
 * arbitrary window needs one. Getting the intercept wrong shifts every pixel
 * value while still looking plausible on screen.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const IV = require('../../gently/ui/web/static/js/panels/imageview.js');

const apply = (v, { slope, intercept }) => v * slope + intercept;
const close = (a, b) => Math.abs(a - b) < 1e-9;

test('a full window is the identity', () => {
    const t = IV.windowTransfer(0, 1);
    assert.ok(close(t.slope, 1));
    assert.ok(close(t.intercept, 0));
    for (const v of [0, 0.25, 0.5, 1]) assert.ok(close(apply(v, t), v));
});

test('the black point maps to 0 and the white point to 1', () => {
    const t = IV.windowTransfer(0.2, 0.8);
    assert.ok(close(apply(0.2, t), 0), 'black point must land on 0');
    assert.ok(close(apply(0.8, t), 1), 'white point must land on 1');
});

test('the midpoint of the window lands mid-grey', () => {
    const t = IV.windowTransfer(0.2, 0.8);
    assert.ok(close(apply(0.5, t), 0.5));
});

test('a narrow window stretches faint detail', () => {
    // The case that matters: dim nuclei sitting in a thin slice of the range.
    const t = IV.windowTransfer(0.6, 0.7);
    assert.ok(close(t.slope, 10), 'a 10% window is a 10x stretch');
    assert.ok(close(apply(0.65, t), 0.5));
});

test('an offset window needs a non-zero intercept', () => {
    // Precisely what CSS brightness/contrast cannot express.
    const t = IV.windowTransfer(0.3, 1.0);
    assert.ok(t.intercept < 0, 'a raised black point must pull values down');
    assert.ok(close(apply(0.3, t), 0));
});

test('values outside the window fall outside 0..1 and are clamped by the filter', () => {
    const t = IV.windowTransfer(0.4, 0.6);
    assert.ok(apply(0.1, t) < 0, 'below black is negative before clamping');
    assert.ok(apply(0.9, t) > 1, 'above white exceeds 1 before clamping');
});

test('MIN_SPAN keeps the slope finite', () => {
    const t = IV.windowTransfer(0.5, 0.5 + IV.MIN_SPAN);
    assert.ok(Number.isFinite(t.slope));
    assert.ok(Number.isFinite(t.intercept));
    assert.ok(t.slope <= 1 / IV.MIN_SPAN + 1e-9);
});
