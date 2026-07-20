/**
 * Operate — pure geometry, banding and interlock logic.
 *
 * These four functions decide where an embryo is, how far the F-drive may move,
 * where the gauge marker sits, and whether XY motion is safe. They take explicit
 * arguments and touch no DOM, so they can be unit-tested (tests/js/operate-math.test.mjs)
 * — which the rest of the surface, being wiring over already-covered endpoints,
 * does not need.
 *
 * Loaded as a plain script in the browser (window.OperateMath) and required by
 * the Node test runner; there is no root package.json, so `.js` here is CJS.
 */
const OperateMath = (function () {
    // pixel_size_um / objective_mag, before frame downsampling.
    const BASE_UM_PER_PX = 6.5 / 10.0;

    function umPerPx(frame, base) {
        const b = (base == null) ? BASE_UM_PER_PX : base;
        return b * ((frame && frame.downsample) || 1);
    }

    /**
     * Frame pixel → absolute stage µm.
     *
     * `captureStage` is the absolute XY the frame was taken at and must be real —
     * callers block marking rather than defaulting it to [0,0], because that
     * silently converts clicks into offsets from stage origin and lands embryos
     * hundreds of µm away. Returns null rather than guessing.
     *
     * Stage +Y is up, image +Y is down, hence the sign flip on Y only.
     */
    function frameToStage(fx, fy, frame, captureStage, base) {
        if (!frame || !Number.isFinite(frame.w) || !Number.isFinite(frame.h)) return null;
        if (!Array.isArray(captureStage) || captureStage.length !== 2) return null;
        if (!Number.isFinite(captureStage[0]) || !Number.isFinite(captureStage[1])) return null;
        const u = umPerPx(frame, base);
        return [
            captureStage[0] + (fx - frame.w / 2) * u,
            captureStage[1] - (fy - frame.h / 2) * u,
        ];
    }

    // The F-drive travels from ~25000 µm down onto a sample sitting around 50-60.
    // Operators close that in bands: a big jump, then thousands, hundreds, tens.
    // Offering ±1 at 25000 µm is 2500 clicks; offering ±1000 at 200 µm is a crash.
    // So the steps on offer follow the current height.
    const FD_BANDS = [
        { above: 10000, steps: [5000, 1000], label: 'coarse approach' },
        { above: 2000, steps: [1000, 500], label: 'approach' },
        { above: 1000, steps: [500, 100], label: 'near sample' },
        { above: 200, steps: [100, 50], label: 'close' },
        { above: -Infinity, steps: [50, 10, 5], label: 'fine — at sample' },
    ];

    // With no position yet the finest band is the safe answer. Note `>` is strict:
    // exactly 200 falls through to 'fine', not 'close'.
    function fdBand(pos) {
        if (pos == null || !Number.isFinite(pos)) return FD_BANDS[FD_BANDS.length - 1];
        return FD_BANDS.find(b => pos > b.above) || FD_BANDS[FD_BANDS.length - 1];
    }

    /**
     * Absolute stage µm → frame pixel. The inverse of frameToStage.
     *
     * Markers are stored in stage coordinates, not pixel coordinates, so they
     * stay attached to the sample rather than to the viewport: the frame can
     * keep streaming and the stage can move, and each marker re-projects onto
     * whatever frame is current. That is what removes the old freeze-the-frame
     * marking mode. Returns null on the same conditions frameToStage does.
     */
    function stageToFrame(sx, sy, frame, captureStage, base) {
        if (!frame || !Number.isFinite(frame.w) || !Number.isFinite(frame.h)) return null;
        if (!Array.isArray(captureStage) || captureStage.length !== 2) return null;
        if (!Number.isFinite(captureStage[0]) || !Number.isFinite(captureStage[1])) return null;
        const u = umPerPx(frame, base);
        if (!(u > 0)) return null;
        return [
            frame.w / 2 + (sx - captureStage[0]) / u,
            frame.h / 2 - (sy - captureStage[1]) / u,
        ];
    }

    /**
     * May this nudge be offered, given the remaining travel to the floor?
     *
     * Banding (fdBand) picks step sizes proportionate to height; this is the
     * separate safety gate. Up-steps are always offered — the server fences the
     * ceiling. A down-step may not exceed the travel that is left.
     *
     * With no telemetry the answer is "allow": the UI must not pretend to know a
     * distance it has not been told, and the real backstop is the server-side
     * fence (F_DRIVE_MIN_UM in hardware/dispim/devices/piezo.py), which cannot be
     * removed from here.
     */
    function stepAllowed(delta, distanceToFloor) {
        if (!Number.isFinite(delta)) return false;
        if (delta >= 0) return true;
        if (distanceToFloor == null || !Number.isFinite(distanceToFloor)) return true;
        return Math.abs(delta) <= distanceToFloor;
    }

    /**
     * Where the marker sits on a travel track, 0 (min) to 1 (max). Null means
     * "don't draw a marker" — a marker parked at the bottom reads as "at the
     * limit", which is a lie when the truth is that nothing is known yet.
     *
     * The F-drive uses scale 'log'. Linearly, the last 200 µm of a 30-25000 µm
     * axis — the entire approach-and-crash region — is 0.7% of the track, under
     * one pixel. Log makes the approach legible where it matters.
     */
    function gaugeFraction(pos, min, max, scale) {
        if (pos == null || min == null || max == null) return null;
        if (!Number.isFinite(pos) || !Number.isFinite(min) || !Number.isFinite(max)) return null;
        const span = max - min;
        if (!(span > 0)) return null;
        const clamp = v => Math.min(1, Math.max(0, v));
        if (scale === 'log') {
            const denom = Math.log10(span + 1);
            if (!(denom > 0)) return null;
            return clamp(Math.log10(Math.max(0, pos - min) + 1) / denom);
        }
        return clamp((pos - min) / span);
    }

    // RIG-NOTE: 1000 µm is the band where operators switch to hundred-µm steps.
    // Confirm against the real geometry before trusting it on the rig.
    const ENGAGED_WITHIN_UM = 1000;

    /**
     * Is the sample close enough to the objective that XY motion is unsafe?
     *
     * `latch` is a BELIEF (persisted: we commanded the head down). `floor` is a
     * MEASUREMENT (distance_to_floor from telemetry). Measurement wins when it is
     * decisive in either direction; the latch only decides inside the hysteresis
     * band, and is the sole signal when there is no measurement at all.
     *
     * Fail safe: with no telemetry, a set latch locks. The `> within * 2` clear is
     * what lets someone raising the head at the controller box be noticed; the 2x
     * band stops it chattering at the boundary.
     */
    function isEngaged(latch, floor, within) {
        const w = (within == null) ? ENGAGED_WITHIN_UM : within;
        if (floor == null || !Number.isFinite(floor)) return !!latch;
        if (floor < w) return true;
        if (floor > w * 2) return false;
        return !!latch;
    }

    return {
        BASE_UM_PER_PX, ENGAGED_WITHIN_UM, FD_BANDS,
        umPerPx, frameToStage, stageToFrame, fdBand, stepAllowed, gaugeFraction, isEngaged,
    };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = OperateMath;
