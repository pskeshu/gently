/**
 * The uint16 viewer's arithmetic.
 *
 *   node --test tests/js/u16-math.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const M = require('../../gently/ui/web/static/js/u16-math.js');

test('percentiles read values off the histogram at the asked fractions', () => {
    const px = new Uint16Array(1000);
    for (let i = 0; i < 1000; i++) px[i] = i * 10;          // 0, 10, …, 9990
    const h = M.histogram(px);
    const [p1, p50, p99] = M.percentiles(h, px.length, [0.01, 0.5, 0.99]);
    assert.equal(p1, 90);
    assert.equal(p50, 4990);
    assert.equal(p99, 9890);
    assert.deepEqual(M.percentiles(h, 0, [0.5]), [0]);
});

test('the lookup table is 0 below black, 255 above white, monotonic between', () => {
    const t = M.lut(100, 1100, 1);
    assert.equal(t[0], 0); assert.equal(t[100], 0);
    assert.equal(t[1100], 255); assert.equal(t[65535], 255);
    assert.equal(t[600], 128);
    for (let v = 101; v <= 1100; v++) assert.ok(t[v] >= t[v - 1]);
    // gamma below 1 lifts the middle; white and black are unmoved
    const g = M.lut(100, 1100, 0.5);
    assert.ok(g[600] > t[600]);
    assert.equal(g[100], 0); assert.equal(g[1100], 255);
    // a white at or below black still gives a usable table
    assert.equal(M.lut(500, 500, 1)[501], 255);
});

test('shade writes one grey per pixel with full alpha', () => {
    const px = new Uint16Array([0, 500, 1000]);
    const rgba = M.shade(px, M.lut(0, 1000, 1), new Uint8ClampedArray(12));
    assert.deepEqual([...rgba], [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
});

test('headers from gently.core.u16 parse into numbers, missing ones to null', () => {
    const h = { 'X-Image-Width': '512', 'X-Image-Height': '256', 'X-Image-Scale': '4', 'X-Image-Min': '12', 'X-Image-Max': '4000' };
    const m = M.parseHeaders(k => h[k]);
    assert.equal(m.width, 512); assert.equal(m.height, 256); assert.equal(m.scale, 4);
    assert.equal(m.min, 12); assert.equal(m.max, 4000); assert.equal(m.planes, null);
    assert.equal(M.parseHeaders(() => undefined).scale, 1);
});
