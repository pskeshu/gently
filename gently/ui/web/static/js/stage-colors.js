/**
 * One ordinal stage ramp shared by every surface that paints a developmental
 * stage. Viridis-like, dark -> light, so order survives colorblind simulation.
 *
 * Classic script (window global `stageColor`); also CJS-exported for node tests.
 */
// "1.5fold" / "1.5 fold" / "15fold" / "1_5fold" -> "1_5_fold"; "2fold" -> "2_fold"; "2cell" -> "2_cell".
// Python emits both spellings; every map keyed by stage goes through this.
const stageKey = (name) => String(name ?? '').toLowerCase().replace(/\s+/g, '').replace(/\./g, '_')
    .replace(/^(\d)_?(\d)_?fold$/, '$1_$2_fold')
    .replace(/^(\d)(fold|cell)$/, '$1_$2');

const stageColor = (() => {
    const RAMP = {
        early: '#440154', '1_cell': '#440154', '2_cell': '#440154', '4_cell': '#440154',
        bean: '#3b528b',
        comma: '#21918c',
        '1_5_fold': '#27ad81',
        '2_fold': '#5ec962',
        pretzel: '#aadc32', '3_fold': '#aadc32',
        hatching: '#fde725',
        hatched: '#fef3c7',
    };
    const FALLBACK = '#8b949e';
    return (name) => RAMP[stageKey(name)] || FALLBACK;
})();

if (typeof module !== 'undefined' && module.exports) module.exports = { stageColor, stageKey };
