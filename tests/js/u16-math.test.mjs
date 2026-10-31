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

test("auto contrast is ImageJ's: the background bin ignored, the first bin over pixels/5000 from each end", () => {
    // 1000 pixels: 900 at 100 (more than a tenth: ignored), 50 at 1000, 50 at 2000.
    const px = new Uint16Array(1000).fill(100);
    px.fill(1000, 900, 950); px.fill(2000, 950);
    const r = M.imagejAuto(M.histogram(px), px.length, 0);
    // min 100, max 2000, so bins of 1901/256; 1000 lands in bin 121 and 2000 in bin 255,
    // and each end is read at its bin's lower edge
    const bin = 1901 / 256;
    assert.equal(r.lo, 100 + 121 * bin);
    assert.equal(r.hi, 100 + 255 * bin);
    assert.equal(r.autoThreshold, 5000);
});

test('a repeat press halves the threshold and cuts tighter; under 10 it starts over at 5000', () => {
    // 100000 pixels: 30 at 100, 90000 at 500 (ignored), 4985 at 2000, 4985 at 3000.
    const px = new Uint16Array(100000).fill(500);
    px.fill(100, 0, 30); px.fill(2000, 30, 5015); px.fill(3000, 5015, 10000);
    const h = M.histogram(px), bin = 2901 / 256;
    const first = M.imagejAuto(h, px.length, 0);                        // over 20 pixels: the 30 at 100 count
    assert.equal(first.lo, 100);
    assert.equal(first.hi, 100 + 255 * bin);
    const second = M.imagejAuto(h, px.length, first.autoThreshold);     // over 40: they no longer do
    assert.equal(second.autoThreshold, 2500);
    assert.equal(second.lo, 100 + 167 * bin);
    assert.equal(second.hi, 100 + 255 * bin);
    assert.equal(M.imagejAuto(h, px.length, 19).autoThreshold, 9);
    assert.equal(M.imagejAuto(h, px.length, 9).autoThreshold, 5000);
});

test('a flat frame gives its one value as both ends, as ImageJ resets to the data range', () => {
    const px = new Uint16Array(50).fill(777);
    const r = M.imagejAuto(M.histogram(px), px.length, 0);
    assert.equal(r.lo, 777); assert.equal(r.hi, 777);
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
