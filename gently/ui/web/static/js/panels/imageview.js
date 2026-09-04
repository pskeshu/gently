/**
 * ImageView — zoom and pan for a camera surface.
 *
 * The second panel built to docs/architecture/PANELS.md.
 *
 *     ImageView.attach('op-cam-bottom');
 *
 * WHY
 *
 * The bottom camera had no zoom. Ryan, on the 2026-08-07 walkthrough, on why
 * two-point calibration was hard: "the embryo is a little small, so we'd
 * probably zoom in on this like in Micro-Manager to make it easier to see
 * nuclei... effectively like using the magnifying glass in ImageJ". #113 also
 * measured the click-to-mark surface rendering at roughly 20x20 px, under a
 * caption inviting you to click it precisely.
 *
 * HOW IT STAYS CORRECT
 *
 * Zoom is one CSS transform on the fit box, so the frame and the marker canvas
 * inside it scale as one object and markers cannot drift off the embryos they
 * mark. The existing geometry already anticipated this and needs no change:
 * `renderedRect()` measures `offsetWidth`, which a transform does not affect,
 * and `onCanvasClick()` divides the live ancestor scale out of every click
 * (`rect.width / offsetWidth`) — so clicks land on the same pixel at any zoom.
 *
 * NOT a camera ROI. This magnifies pixels already read out; it does not change
 * what the sensor reads. #125 is the ROI ask, and it is a different, better
 * thing for seeing nuclei — more signal per pixel, not bigger pixels.
 */
const ImageView = (() => {
    'use strict';

    // RIG-NOTE: 8x is well past the point where bottom-camera pixels turn to
    // mush, but zoom is free and an operator hunting a nucleus can decide that
    // for themselves. Tune if it proves useless past some smaller number.
    const MIN_Z = 1;
    const MAX_Z = 8;
    const WHEEL_STEP = 1.15;

    const views = new Map();   // containerId -> {z, tx, ty, fit, badge}

    function attach(containerId) {
        if (views.has(containerId)) return;
        const box = document.getElementById(containerId);
        if (!box) return;
        const fit = box.querySelector('.op-cam-fit');
        if (!fit) return;

        const v = { z: 1, tx: 0, ty: 0, fit, box, badge: null, drag: null };
        views.set(containerId, v);

        // Zoom about the cursor, so the thing under the pointer stays under it.
        box.addEventListener('wheel', e => {
            e.preventDefault();
            const prev = v.z;
            const next = clamp(prev * (e.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP));
            if (next === prev) return;
            const r = fit.getBoundingClientRect();
            // Cursor position within the untransformed box.
            const ox = (e.clientX - r.left) / prev;
            const oy = (e.clientY - r.top) / prev;
            v.z = next;
            v.tx -= ox * (next - prev);
            v.ty -= oy * (next - prev);
            apply(v);
        }, { passive: false });

        // Pan only once there is somewhere to pan to, or a plain click on the
        // frame would start a drag and marking would fight it.
        box.addEventListener('pointerdown', e => {
            if (v.z <= 1 || e.button !== 0) return;
            v.drag = { x: e.clientX, y: e.clientY, tx: v.tx, ty: v.ty, moved: false };
            box.setPointerCapture(e.pointerId);
        });
        box.addEventListener('pointermove', e => {
            if (!v.drag) return;
            const dx = e.clientX - v.drag.x, dy = e.clientY - v.drag.y;
            if (Math.hypot(dx, dy) > 3) v.drag.moved = true;
            v.tx = v.drag.tx + dx;
            v.ty = v.drag.ty + dy;
            apply(v);
        });
        box.addEventListener('pointerup', e => {
            const moved = v.drag && v.drag.moved;
            v.drag = null;
            try { box.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ }
            // A drag that panned must not also register as a mark.
            if (moved) { e.stopPropagation(); e.preventDefault(); }
        }, true);

        // Reset needs to exist or a zoomed-in operator is stranded, but it does
        // not need chrome. Double-click is the same gesture every image viewer
        // uses (PANELS.md rule 6 — nothing to open or close).
        box.addEventListener('dblclick', e => { e.preventDefault(); reset(containerId); });

        apply(v);
    }

    const clamp = z => Math.min(MAX_Z, Math.max(MIN_Z, z));

    function apply(v) {
        if (v.z <= 1) { v.z = 1; v.tx = 0; v.ty = 0; }
        v.fit.style.transform = v.z === 1 ? '' : `translate(${v.tx}px, ${v.ty}px) scale(${v.z})`;
        v.fit.style.transformOrigin = '0 0';
        v.box.style.cursor = v.z > 1 ? 'grab' : '';
        badge(v);
    }

    // Presence-driven: the readout exists only while zoomed, because at 1x it
    // would be a permanent label saying nothing happened.
    function badge(v) {
        if (v.z === 1) {
            if (v.badge) { v.badge.remove(); v.badge = null; }
            return;
        }
        if (!v.badge) {
            v.badge = document.createElement('div');
            v.badge.className = 'iv-badge';
            v.badge.title = 'Double-click to reset';
            v.box.appendChild(v.badge);
        }
        v.badge.textContent = `${v.z.toFixed(1)}x`;
    }

    function reset(containerId) {
        const v = views.get(containerId);
        if (!v) return;
        v.z = 1; v.tx = 0; v.ty = 0;
        apply(v);
    }

    function zoomOf(containerId) {
        const v = views.get(containerId);
        return v ? v.z : 1;
    }

    return { attach, reset, zoomOf, MIN_Z, MAX_Z, WHEEL_STEP, _clamp: clamp };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ImageView;
