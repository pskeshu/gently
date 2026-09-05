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

    function attach(containerId, opts) {
        if (views.has(containerId)) return;
        const box = document.getElementById(containerId);
        if (!box) return;
        const fit = box.querySelector('.op-cam-fit');
        if (!fit) return;

        const v = { z: 1, tx: 0, ty: 0, fit, box, badge: null, drag: null,
                    lo: 0, hi: 1, panel: null, hist: null,
                    filterId: `iv-win-${containerId}` };
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

        // Controls go in a host OUTSIDE the frame. Nothing sits on the pixels:
        // the image is the instrument, and an overlay both hides data and eats
        // clicks meant for markers.
        const host = opts && opts.controlsHost && document.getElementById(opts.controlsHost);
        if (host) buildPanel(v, containerId, host);
        apply(v);
    }

    /**
     * The display panel: a histogram with the transfer line drawn across it.
     *
     * Modelled on ImageJ/Fiji's Brightness/Contrast, which is the tool every
     * microscopist already knows. Its insight is that the histogram and the
     * mapping belong in one picture: you see where the data actually sits, and
     * you see the line that decides what black and white mean, together. That
     * is why B&C needs no numeric status readout — the artefact is the status.
     *
     * Two abstract "contrast" and "brightness" sliders were the opposite of
     * this: they asked the operator to guess where the data was and then reach
     * it through multiplication.
     *
     * It also cannot be done with CSS. `brightness()` and `contrast()` are both
     * multiplicative about their own origin, so composing them gives
     * `out = in·b·c + k` with k pinned by c — no free intercept, and a window
     * needs one. An SVG feComponentTransfer with type="linear" has slope AND
     * intercept, so it maps the window exactly:
     *
     *     out = (in - lo) / (hi - lo)   →   slope = 1/(hi-lo), intercept = -lo·slope
     *
     * `color-interpolation-filters="sRGB"` matters: the SVG default is
     * linearRGB, which would apply a gamma the operator did not ask for.
     *
     * The histogram is of the frame as displayed, which is 8-bit and already
     * percentile-stretched by the device layer before JPEG encoding. It is not
     * raw camera counts, and it is labelled as such — the honest fix for that
     * is #149, and this panel is what will make the case visible.
     */
    function buildPanel(v, containerId, host) {
        ensureFilter(v);

        host.innerHTML = `
          <div class="lp">
            <div class="lp-head">
              <span class="lp-title">Display</span>
              <span class="iv-src" title="Histogram of the displayed 8-bit frame, which the device layer has already percentile-stretched. Not raw camera counts (#149).">as displayed</span>
            </div>
            <div class="iv-hist" data-hist>
              <canvas class="iv-hist-c" data-hist-c width="512" height="96"></canvas>
              <div class="iv-h iv-h-lo" data-h="lo" role="slider" tabindex="0"
                   aria-label="Black point" aria-valuemin="0" aria-valuemax="100"></div>
              <div class="iv-h iv-h-hi" data-h="hi" role="slider" tabindex="0"
                   aria-label="White point" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            <div class="iv-acts">
              <button type="button" class="lp-btn" data-auto>Auto</button>
              <button type="button" class="lp-btn" data-reset>Reset</button>
            </div>
          </div>`;

        const hist = host.querySelector('[data-hist]');
        let dragging = null;
        const fromX = clientX => {
            const r = hist.getBoundingClientRect();
            return Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width)));
        };

        hist.addEventListener('pointerdown', e => {
            const h = e.target.closest('[data-h]');
            const at = fromX(e.clientX);
            // Grab the nearer handle when the histogram itself is pressed, so
            // the operator does not have to hit an 8 px target.
            dragging = h ? h.dataset.h
                : (Math.abs(at - v.lo) <= Math.abs(at - v.hi) ? 'lo' : 'hi');
            moveHandle(v, dragging, at);
            hist.setPointerCapture(e.pointerId);
        });
        hist.addEventListener('pointermove', e => {
            if (dragging) moveHandle(v, dragging, fromX(e.clientX));
        });
        hist.addEventListener('pointerup', e => {
            dragging = null;
            try { hist.releasePointerCapture(e.pointerId); } catch (_) { /* fine */ }
        });
        hist.addEventListener('keydown', e => {
            const step = e.shiftKey ? 0.10 : 0.02;
            const which = (document.activeElement && document.activeElement.dataset.h) || 'hi';
            if (e.key === 'ArrowLeft') { moveHandle(v, which, v[which] - step); e.preventDefault(); }
            else if (e.key === 'ArrowRight') { moveHandle(v, which, v[which] + step); e.preventDefault(); }
        });

        host.querySelector('[data-auto]').addEventListener('click', () => autoWindow(v));
        host.querySelector('[data-reset]').addEventListener('click', () => {
            v.lo = 0; v.hi = 1;
            reset(containerId);
        });

        v.panel = host;
        // Frames stream, so recompute on arrival but not on every one.
        const img = v.fit.querySelector('.op-cam-img');
        if (img) img.addEventListener('load', () => scheduleHistogram(v));
        scheduleHistogram(v);
    }

    /* ── histogram ───────────────────────────────────────────────────────── */

    // RIG-NOTE: 400ms between recomputes. The frame rate is higher than any eye
    // needs from a histogram, and each pass is a full readback of the decoded
    // image. Lower it if the histogram feels laggy while hunting focus.
    const HIST_MS = 400;
    let histTimer = null;
    let histPending = null;

    function scheduleHistogram(v) {
        histPending = v;
        if (histTimer) return;
        histTimer = setTimeout(() => {
            histTimer = null;
            const target = histPending;
            histPending = null;
            if (target) computeHistogram(target);
        }, HIST_MS);
    }

    function computeHistogram(v) {
        const img = v.fit.querySelector('.op-cam-img');
        if (!img || !img.naturalWidth) { v.hist = null; drawHistogram(v); return; }
        try {
            // Downsampled: a histogram does not need every pixel, and reading
            // back a full frame every 400ms would be the expensive part.
            const W = 256, H = Math.max(1, Math.round(256 * img.naturalHeight / img.naturalWidth));
            const c = histCanvas(W, H);
            const ctx = c.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(img, 0, 0, W, H);
            const d = ctx.getImageData(0, 0, W, H).data;
            const bins = new Uint32Array(256);
            for (let i = 0; i < d.length; i += 4) {
                // Rec. 601 luma. The frames are greyscale, so any channel would
                // do, but this stays right if a colour overlay ever arrives.
                bins[(d[i] * 77 + d[i + 1] * 150 + d[i + 2] * 29) >> 8]++;
            }
            v.hist = bins;
        } catch (_) {
            v.hist = null;      // tainted or not yet decodable
        }
        drawHistogram(v);
    }

    let _histCanvas = null;
    function histCanvas(w, h) {
        if (!_histCanvas) _histCanvas = document.createElement('canvas');
        if (_histCanvas.width !== w || _histCanvas.height !== h) {
            _histCanvas.width = w; _histCanvas.height = h;
        }
        return _histCanvas;
    }

    function drawHistogram(v) {
        if (!v.panel) return;
        const c = v.panel.querySelector('[data-hist-c]');
        if (!c) return;
        const ctx = c.getContext('2d');
        const W = c.width, H = c.height;
        ctx.clearRect(0, 0, W, H);

        if (v.hist) {
            // Log counts: a microscopy histogram is dominated by background, and
            // on a linear axis the signal that matters is a flat line at zero.
            let max = 0;
            for (let i = 0; i < 256; i++) max = Math.max(max, v.hist[i]);
            const norm = Math.log1p(max) || 1;
            ctx.fillStyle = 'rgba(255,255,255,0.55)';
            for (let i = 0; i < 256; i++) {
                const h = (Math.log1p(v.hist[i]) / norm) * (H - 2);
                if (h > 0) ctx.fillRect((i / 256) * W, H - h, Math.ceil(W / 256), h);
            }
        } else {
            ctx.fillStyle = 'rgba(255,255,255,0.25)';
            ctx.font = '11px system-ui, sans-serif';
            ctx.fillText('no frame', 6, H / 2);
        }

        // The transfer line: what black and white currently mean, drawn over the
        // data it applies to. This is the readout — ImageJ's B&C insight.
        ctx.strokeStyle = 'rgba(96,165,250,0.95)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, H - 1);
        ctx.lineTo(v.lo * W, H - 1);
        ctx.lineTo(v.hi * W, 1);
        ctx.lineTo(W, 1);
        ctx.stroke();
    }

    /**
     * Auto: put the window around the data actually present.
     *
     * The 0.1st and 99.9th percentiles of the displayed histogram, not min/max
     * — a handful of hot pixels should not define white.
     */
    function autoWindow(v) {
        if (!v.hist) return;
        let total = 0;
        for (let i = 0; i < 256; i++) total += v.hist[i];
        if (!total) return;
        const lowCut = total * 0.001, highCut = total * 0.999;
        let acc = 0, lo = 0, hi = 255;
        for (let i = 0; i < 256; i++) {
            acc += v.hist[i];
            if (acc >= lowCut) { lo = i; break; }
        }
        acc = 0;
        for (let i = 0; i < 256; i++) {
            acc += v.hist[i];
            if (acc >= highCut) { hi = i; break; }
        }
        v.lo = lo / 255;
        v.hi = Math.max(v.lo + MIN_SPAN, hi / 255);
        apply(v);
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

        if (!v.panel) return;
        // The handles are the only numeric readout there is, and the transfer
        // line over the histogram is the rest — ImageJ's B&C carries no status
        // text either, because the picture already says it.
        const lo = v.panel.querySelector('[data-h="lo"]');
        const hi = v.panel.querySelector('[data-h="hi"]');
        if (lo) { lo.style.left = `${v.lo * 100}%`; lo.setAttribute('aria-valuenow', Math.round(v.lo * 100)); }
        if (hi) { hi.style.left = `${v.hi * 100}%`; hi.setAttribute('aria-valuenow', Math.round(v.hi * 100)); }
    }

    const clamp = z => Math.min(MAX_Z, Math.max(MIN_Z, z));

    function apply(v) {
        if (v.z <= 1) { v.z = 1; v.tx = 0; v.ty = 0; }
        v.fit.style.transform = v.z === 1 ? '' : `translate(${v.tx}px, ${v.ty}px) scale(${v.z})`;
        v.fit.style.transformOrigin = '0 0';
        // On the <img>, not the fit box: the marker canvas is a sibling inside
        // it and must NOT be filtered, or green markers dim with the frame.
        applyWindow(v);
        drawHistogram(v);
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
