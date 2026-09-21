/**
 * The calibration progress panel's reading of a frame.
 *
 *   node --test tests/js/calprogress.test.mjs
 *
 * (Pass the file, not the directory — see operate-math.test.mjs.)
 *
 * The panel's whole claim is that the frames already on the wire say enough to
 * narrate a calibration. That claim lives in two pure functions, so they are
 * what is tested: given the metadata `agent.push_viz` actually attaches, does
 * the panel say something an operator can act on?
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const P = require('../../gently/ui/web/static/js/panels/calprogress.js');

// Exactly the shape gently/app/tools/calibration_tools.py pushes.
const EDGE = {
    data_type: 'edge_detection',
    metadata: { embryo_id: 'embryo_1', galvo: -0.15, piezo: -15.0, visible: false, feature_score: 3 },
};
const SWEEP = {
    data_type: 'focus_sweep',
    metadata: {
        embryo_id: 'embryo_1', sweep: 'dense', galvo_name: 'top',
        galvo: -0.1, piezo: 12.4, score: 0.83, peak_detected: true,
    },
};
const PLOT = {
    data_type: 'focus_plot',
    metadata: { embryo_id: 'embryo_1', galvo_name: 'bottom', best_piezo: 13.44, r_squared: 0.9721 },
};
const SUMMARY = {
    data_type: 'calibration_summary',
    metadata: { embryo_id: 'embryo_1', slope: 101.23, r_squared_top: 0.972, r_squared_bottom: 0.951 },
};

test('each phase names itself from the frame, not from a counter', () => {
    assert.match(P.phaseOf(EDGE), /edges/i);
    assert.equal(P.phaseOf(SWEEP), 'Focus sweep · top dense');
    assert.equal(P.phaseOf(PLOT), 'Fitting the bottom focus curve');
    assert.equal(P.phaseOf(SUMMARY), 'Calibrated');
});

test('an edge frame says where it looked and what was there', () => {
    const cap = P.captionOf(EDGE);
    assert.match(cap, /-0\.150°/);      // where the galvo was
    assert.match(cap, /nothing there/);  // Claude's verdict, the reason the sweep stops
    assert.match(cap, /features 3\/10/);
});

test('a visible edge frame reads differently from an empty one', () => {
    const seen = P.captionOf({ ...EDGE, metadata: { ...EDGE.metadata, visible: true } });
    assert.match(seen, /embryo visible/);
    assert.doesNotMatch(seen, /nothing there/);
});

test('a focus frame carries the number the sweep is maximising', () => {
    const cap = P.captionOf(SWEEP);
    assert.match(cap, /piezo 12\.4 µm/);
    assert.match(cap, /sharpness 0\.83/);
    assert.match(cap, /peak found/);
});

test('the fit reports its R², which is what makes a calibration trustworthy', () => {
    assert.match(P.captionOf(PLOT), /R² 0\.972/);
    assert.match(P.captionOf(SUMMARY), /101\.2 µm\/deg/);
    assert.match(P.captionOf(SUMMARY), /R² 0\.972 top \/ 0\.951 bottom/);
});

test('a frame missing its numbers degrades to silence, never to NaN', () => {
    // Older runs, partial metadata, a phase that pushes an image with nothing
    // attached: a caption of "galvo NaN°" would be worse than no caption.
    for (const t of ['edge_detection', 'focus_sweep', 'focus_plot', 'calibration_summary']) {
        const cap = P.captionOf({ data_type: t, metadata: {} });
        assert.doesNotMatch(cap, /NaN|undefined|null/);
    }
    assert.equal(P.captionOf({ data_type: 'something_else', metadata: { galvo: 1 } }), '');
});

test('an unknown image type is never narrated as a calibration', () => {
    // The panel listens to a broadcast every image on the system rides on, so
    // a snapshot or a volume projection must not be mistaken for a frame.
    assert.equal(P.phaseOf({ data_type: 'snapshot', metadata: {} }), 'Calibrating');
});
