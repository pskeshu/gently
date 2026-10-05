/**
 * One embryo-stage palette for every view.
 *
 *   node --test tests/js/stage-colors.test.mjs
 *
 * embryos.js and timepoint-player.js used to carry their own stage→colour
 * maps that spelled the keys differently, so the same stage was blue in one
 * view and violet in the other, and misspelled stages fell back to grey.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { stageColor } = require('../../gently/ui/web/static/js/stage-colors.js');

test('every spelling of a stage lands on the same colour', () => {
    assert.equal(stageColor('1.5fold'), stageColor('1_5_fold'));
    assert.equal(stageColor('1.5 Fold'), stageColor('15fold'));
    assert.equal(stageColor('2fold'), stageColor('2_fold'));
    assert.equal(stageColor('3fold'), stageColor('pretzel'));
    assert.equal(stageColor('Bean'), stageColor('bean'));
});

test('the ramp is ordinal: lightness rises with developmental stage', () => {
    const L = hex => {
        const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const order = ['early', 'bean', 'comma', '1_5_fold', '2_fold', '3_fold', 'hatching', 'hatched'];
    const ls = order.map(s => L(stageColor(s)));
    for (let i = 1; i < ls.length; i++) assert.ok(ls[i] > ls[i - 1], `${order[i]} darker than ${order[i - 1]}`);
});

test('unknown and missing stages fall back to grey', () => {
    assert.equal(stageColor('bogus'), '#8b949e');
    assert.equal(stageColor(null), '#8b949e');
});
