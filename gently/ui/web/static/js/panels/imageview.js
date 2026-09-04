/**
 * ImageView — zoom, pan and display range for a camera surface.
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
    // Below this, a press is a click, not a pan. Generous enough to survive
    // the hand wobble of clicking a 13 px marker on a zoomed frame.
    const DRAG_SLOP = 4;

    const views = new Map();   // containerId -> {z, tx, ty, fit, badge}

    function attach(containerId) {
        if (views.has(containerId)) return;
        const box = document.getElementById(containerId);
        if (!box) return;
        const fit = box.querySelector('.op-cam-fit');
        if (!fit) return;

        const v = { z: 1, tx: 0, ty: 0, fit, box, badge: null, drag: null,
                    lo: 0, hi: 1, bar: null, filterId: `iv-win-${containerId}` };
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
        //
        // Capture is taken LAZILY, only once the pointer has actually moved past
        // the threshold. Capturing on pointerdown retargets the subsequent
        // `click` to the capturing element, so it never reaches the marker
        // canvas — which made it impossible to click a spurious marker away
        // while zoomed in. A press that does not move must stay an ordinary
        // click.
        box.addEventListener('pointerdown', e => {
            if (v.z <= 1 || e.button !== 0) return;
            v.drag = { x: e.clientX, y: e.clientY, tx: v.tx, ty: v.ty,
                       moved: false, captured: false, id: e.pointerId };
        });
        box.addEventListener('pointermove', e => {
            if (!v.drag) return;
            const dx = e.clientX - v.drag.x, dy = e.clientY - v.drag.y;
            if (!v.drag.moved && Math.hypot(dx, dy) <= DRAG_SLOP) return;
            if (!v.drag.captured) {
                v.drag.moved = true;
                v.drag.captured = true;
                try { box.setPointerCapture(v.drag.id); } catch (_) { /* fine without it */ }
            }
            v.tx = v.drag.tx + dx;
            v.ty = v.drag.ty + dy;
            apply(v);
        });
        box.addEventListener('pointerup', e => {
            const d = v.drag;
            v.drag = null;
            if (!d) return;
            if (d.captured) {
                try { box.releasePointerCapture(d.id); } catch (_) { /* already gone */ }
            }
            // A drag that panned must not also register as a mark. A press that
            // never moved is left completely alone, so the click runs.
            //
            // preventDefault on pointerup does NOT suppress the click the
            // browser then synthesises, and relying on pointer capture to
            // retarget it is what broke clicking markers away in the first
            // place. So the next click is swallowed explicitly, in the capture
            // phase on the container — which runs before the canvas's own
            // handler — and the listener removes itself either way.
            if (d.moved) {
                e.stopPropagation();
                const eat = ev => { ev.stopPropagation(); ev.preventDefault(); };
                box.addEventListener('click', eat, { capture: true, once: true });
                // If no click follows (some pointer types, or the press ended
                // off-target), the listener must not linger and eat a real one.
                setTimeout(() => box.removeEventListener('click', eat, true), 300);
            }
        }, true);

        // Reset needs to exist or a zoomed-in operator is stranded, but it does
        // not need chrome. Double-click is the same gesture every image viewer
        // uses (PANELS.md rule 6 — nothing to open or close).
        box.addEventListener('dblclick', e => { e.preventDefault(); reset(containerId); });

        buildBar(v, containerId);
        apply(v);
    }

    /**
     * The display window: a black point and a white point on one track.
     *
     * This replaced two abstract "contrast" and "brightness" sliders. Those are
     * not how anyone reading a microscope image thinks — a microscopist sets a
     * display range, the way ImageJ's Brightness/Contrast or Micro-Manager's
     * histogram does, and asking them to reach that through two multiplicative
     * knobs is arithmetic homework.
     *
     * It also cannot be done with CSS. `brightness()` and `contrast()` are both
     * multiplicative about their own origin, so composing them gives
     * `out = in·b·c + k` where k is pinned by c — there is no free intercept,
     * and an arbitrary window needs one. An SVG feComponentTransfer with
     * type="linear" has slope AND intercept, so it maps the window exactly:
     *
     *     out = (in - lo) / (hi - lo)   →   slope = 1/(hi-lo), intercept = -lo·slope
     *
     * `color-interpolation-filters="sRGB"` matters: the SVG default is
     * linearRGB, which would silently apply a gamma the operator did not ask
     * for and make the readout a lie.
     */
    function buildBar(v, containerId) {
        ensureFilter(v);

        const bar = document.createElement('div');
        bar.className = 'iv-bar';
        bar.innerHTML = `
          <span class="iv-lbl">display</span>
          <div class="iv-win" data-win tabindex="0" role="group"
               aria-label="Display range: black point and white point">
            <div class="iv-win-track"></div>
            <div class="iv-win-span" data-span></div>
            <div class="iv-win-h iv-win-lo" data-h="lo" role="slider" tabindex="0"
                 aria-label="Black point" aria-valuemin="0" aria-valuemax="100"></div>
            <div class="iv-win-h iv-win-hi" data-h="hi" role="slider" tabindex="0"
                 aria-label="White point" aria-valuemin="0" aria-valuemax="100"></div>
          </div>
          <span class="iv-read" data-read></span>
          <button type="button" class="iv-reset" title="Reset zoom and display range">reset</button>`;

        ['pointerdown', 'pointerup', 'click', 'dblclick', 'wheel'].forEach(ev =>
            bar.addEventListener(ev, e => e.stopPropagation()));

        const win = bar.querySelector('[data-win]');
        let dragging = null;

        const fromX = clientX => {
            const r = win.getBoundingClientRect();
            return Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width)));
        };

        win.addEventListener('pointerdown', e => {
            const h = e.target.closest('[data-h]');
            // Grab the nearer handle when the track itself is pressed, so the
            // operator does not have to hit an 8 px target.
            const at = fromX(e.clientX);
            dragging = h ? h.dataset.h
                : (Math.abs(at - v.lo) <= Math.abs(at - v.hi) ? 'lo' : 'hi');
            moveHandle(v, dragging, at);
            win.setPointerCapture(e.pointerId);
        });
        win.addEventListener('pointermove', e => {
            if (!dragging) return;
            moveHandle(v, dragging, fromX(e.clientX));
        });
        win.addEventListener('pointerup', e => {
            dragging = null;
            try { win.releasePointerCapture(e.pointerId); } catch (_) { /* fine */ }
        });

        // Keyboard: the window is a real control, not a mouse-only flourish.
        win.addEventListener('keydown', e => {
            const step = e.shiftKey ? 0.10 : 0.02;
            const which = (document.activeElement && document.activeElement.dataset.h) || 'hi';
            if (e.key === 'ArrowLeft') { moveHandle(v, which, v[which] - step); e.preventDefault(); }
            else if (e.key === 'ArrowRight') { moveHandle(v, which, v[which] + step); e.preventDefault(); }
        });

        bar.querySelector('.iv-reset').addEventListener('click', () => {
            v.lo = 0; v.hi = 1;
            reset(containerId);
        });

        v.bar = bar;
        v.box.appendChild(bar);
    }

    // Handles cannot cross, and cannot meet: hi === lo is an infinite slope.
    const MIN_SPAN = 0.02;

    function moveHandle(v, which, value) {
        const at = Math.min(1, Math.max(0, value));
        if (which === 'lo') v.lo = Math.min(at, v.hi - MIN_SPAN);
        else v.hi = Math.max(at, v.lo + MIN_SPAN);
        apply(v);
    }

    /** One SVG filter per view, since each carries its own window. */
    function ensureFilter(v) {
        if (document.getElementById(v.filterId)) return;
        let host = document.getElementById('iv-filters');
        if (!host) {
            host = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            host.id = 'iv-filters';
            host.setAttribute('aria-hidden', 'true');
            host.setAttribute('width', '0');
            host.setAttribute('height', '0');
            host.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden';
            document.body.appendChild(host);
        }
        const NS = 'http://www.w3.org/2000/svg';
        const f = document.createElementNS(NS, 'filter');
        f.id = v.filterId;
        // Without this the transfer runs in linearRGB and applies a gamma the
        // operator never asked for.
        f.setAttribute('color-interpolation-filters', 'sRGB');
        const t = document.createElementNS(NS, 'feComponentTransfer');
        ['feFuncR', 'feFuncG', 'feFuncB'].forEach(n => {
            const fn = document.createElementNS(NS, n);
            fn.setAttribute('type', 'linear');
            fn.setAttribute('slope', '1');
            fn.setAttribute('intercept', '0');
            t.appendChild(fn);
        });
        f.appendChild(t);
        host.appendChild(f);
    }

    function applyWindow(v) {
        const img = v.fit.querySelector('.op-cam-img');
        const wide = v.lo === 0 && v.hi === 1;
        if (img) img.style.filter = wide ? '' : `url(#${CSS.escape(v.filterId)})`;

        const f = document.getElementById(v.filterId);
        if (f) {
            const slope = 1 / (v.hi - v.lo);
            const intercept = -v.lo * slope;
            f.querySelectorAll('feFuncR, feFuncG, feFuncB').forEach(fn => {
                fn.setAttribute('slope', String(slope));
                fn.setAttribute('intercept', String(intercept));
            });
        }

        if (!v.bar) return;
        const lo = v.bar.querySelector('[data-h="lo"]');
        const hi = v.bar.querySelector('[data-h="hi"]');
        const span = v.bar.querySelector('[data-span]');
        const read = v.bar.querySelector('[data-read]');
        if (lo) { lo.style.left = `${v.lo * 100}%`; lo.setAttribute('aria-valuenow', Math.round(v.lo * 100)); }
        if (hi) { hi.style.left = `${v.hi * 100}%`; hi.setAttribute('aria-valuenow', Math.round(v.hi * 100)); }
        if (span) { span.style.left = `${v.lo * 100}%`; span.style.right = `${(1 - v.hi) * 100}%`; }
        // Presence-driven: the numbers appear only once the window is not full.
        if (read) read.textContent = wide ? '' : `${Math.round(v.lo * 100)}–${Math.round(v.hi * 100)}`;
    }

    const clamp = z => Math.min(MAX_Z, Math.max(MIN_Z, z));

    function apply(v) {
        if (v.z <= 1) { v.z = 1; v.tx = 0; v.ty = 0; }
        v.fit.style.transform = v.z === 1 ? '' : `translate(${v.tx}px, ${v.ty}px) scale(${v.z})`;
        v.fit.style.transformOrigin = '0 0';
        // On the <img>, not the fit box: the marker canvas is a sibling inside
        // it and must NOT be filtered, or green markers dim with the frame.
        applyWindow(v);
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

    function displayOf(containerId) {
        const v = views.get(containerId);
        return v ? { lo: v.lo, hi: v.hi } : null;
    }

    /** The exact linear map the window applies — `out = in * slope + intercept`. */
    function windowTransfer(lo, hi) {
        const slope = 1 / (hi - lo);
        return { slope, intercept: -lo * slope };
    }

    return { attach, reset, zoomOf, displayOf, windowTransfer,
             MIN_Z, MAX_Z, WHEEL_STEP, MIN_SPAN, _clamp: clamp };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ImageView;
