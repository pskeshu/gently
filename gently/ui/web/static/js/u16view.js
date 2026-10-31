/**
 * U16View — the one viewer for pixels as they are.
 *
 * Fetches a frame as uint16 bytes (gently.core.u16), keeps the array, draws
 * it on a canvas through a lookup table it owns, and answers the cursor with
 * the value under it. Black, white and gamma are sliders that redraw from
 * memory; nothing goes back to the server to change the contrast. Auto sets
 * black and white as ImageJ's Brightness/Contrast Auto button does, and a
 * second press cuts tighter, as there.
 *
 * ponytail: a 2D canvas and a per-pixel JS loop; a 1024² frame redraws in
 * ~10 ms, a 2048² one in ~40 ms. A WebGL texture with the table in a shader
 * is the upgrade if a full-resolution live stream wants more.
 */
const U16View = {
    /** Mount a viewer in `host`. Returns the view; call view.load(url) to show a frame. */
    create(host, opts = {}) {
        const view = {
            host, opts, data: null, meta: null, w: 0, h: 0, lo: 0, hi: 65535, gamma: 1,
            _seq: 0, _raf: null, _lut: null, _img: null, _onPixel: opts.onPixel || null,
        };
        host.classList.add('u16');
        host.innerHTML = `
            <div class="u16-stage"><canvas class="u16-c"></canvas><div class="u16-read" hidden></div></div>
            <div class="u16-bar${opts.bar === false ? ' is-hidden' : ''}">
                <button type="button" class="u16-auto" title="Auto contrast, as ImageJ's Auto button · press again for a tighter cut">auto</button>
                <label class="u16-l">black <input class="u16-lo" type="range" min="0" max="65535" step="1"><output class="u16-lo-v">0</output></label>
                <label class="u16-l">white <input class="u16-hi" type="range" min="0" max="65535" step="1"><output class="u16-hi-v">65535</output></label>
                <label class="u16-l">γ <input class="u16-g" type="range" min="0.3" max="3" step="0.05" value="1"><output class="u16-g-v">1.0</output></label>
                <span class="u16-info"></span>
            </div>`;
        const $ = sel => host.querySelector(sel);
        view.canvas = $('.u16-c'); view.ctx = view.canvas.getContext('2d');
        view.read = $('.u16-read'); view.info = $('.u16-info'); view.bar = $('.u16-bar');
        // The bar's parts are taken once: a host may move the bar out of the
        // stage (the overview stage puts it in its side column), and after
        // that nothing under `host` would answer a query for them.
        const lo = $('.u16-lo'), hi = $('.u16-hi'), g = $('.u16-g');
        const loV = $('.u16-lo-v'), hiV = $('.u16-hi-v'), gV = $('.u16-g-v');
        const paintBar = () => {
            lo.value = view.lo; hi.value = view.hi; g.value = view.gamma;
            loV.textContent = view.lo; hiV.textContent = view.hi; gV.textContent = Number(view.gamma).toFixed(2).replace(/0$/, '');
        };
        lo.oninput = () => { view.lo = Math.min(+lo.value, view.hi - 1); paintBar(); view.redraw(); };
        hi.oninput = () => { view.hi = Math.max(+hi.value, view.lo + 1); paintBar(); view.redraw(); };
        g.oninput = () => { view.gamma = +g.value; paintBar(); view.redraw(); };
        $('.u16-auto').onclick = () => { view.auto(); paintBar(); view.redraw(); };
        view._paintBar = paintBar;

        // The cursor reads the array: the value under it, and where that is in
        // the source frame when the bytes were averaged down.
        /** The readout for pixel (x, y) of the frame held; off the frame, nothing. A
         *  host whose own overlay takes the mouse (the Operate marker canvas) calls this. */
        view.readAt = function (x, y) {
            if (!view.data) return;
            if (x < 0 || y < 0 || x >= view.w || y >= view.h) { view.read.hidden = true; return; }
            const v = view.data[y * view.w + x], s = view.meta.scale || 1;
            const where = s > 1 ? `${x * s}, ${y * s} (mean of ${s}×${s})` : `${x}, ${y}`;
            view.read.textContent = `${where} · ${v}`;
            view.read.hidden = false;
            if (view._onPixel) view._onPixel({ x, y, value: v, scale: s });
        };
        view.readOff = function () { view.read.hidden = true; };
        view.canvas.addEventListener('mousemove', ev => {
            if (!view.data) return;
            const r = view.canvas.getBoundingClientRect();
            view.readAt(Math.floor((ev.clientX - r.left) / r.width * view.w), Math.floor((ev.clientY - r.top) / r.height * view.h));
        });
        view.canvas.addEventListener('mouseleave', () => { view.read.hidden = true; });

        /** Fetch and show a frame. A later load wins over a slower earlier one. */
        view.load = async function (url) {
            const seq = ++view._seq;
            const t0 = performance.now();
            const r = await fetch(url);
            if (!r.ok) throw new Error(`${r.status} for ${url}`);
            const meta = U16Math.parseHeaders(k => r.headers.get(k));
            const buf = await r.arrayBuffer();
            if (seq !== view._seq) return null;
            view.set(new Uint16Array(buf), meta);
            view._lastLoadMs = performance.now() - t0;
            return meta;
        };

        /** Show an array already in hand (the live stream hands frames in this way). */
        view.set = function (u16, meta) {
            const first = !view.data || view.w !== meta.width || view.h !== meta.height;
            view.data = u16; view.meta = meta; view.w = meta.width; view.h = meta.height;
            view._autoThreshold = 0;   // a new image: ImageJ starts its Auto over
            if (first) {
                view.canvas.width = view.w; view.canvas.height = view.h;
                view._img = view.ctx.createImageData(view.w, view.h);
                view.auto();
                paintBar();
            }
            view.info.textContent = `${meta.sourceWidth || view.w}×${meta.sourceHeight || view.h}` + (meta.scale > 1 ? ` · shown ÷${meta.scale}` : '') + ` · ${meta.min}–${meta.max}`;
            // The sliders span the frame's own range, widened to keep a range already set;
            // a camera using 3% of 16 bits would otherwise get 3% of the slider.
            if (meta.min != null && meta.max != null) {
                const a = Math.min(meta.min, view.lo, +lo.min || meta.min), b = Math.max(meta.max, view.hi, +hi.max === 65535 ? meta.max : +hi.max);
                lo.min = hi.min = a; lo.max = hi.max = b; paintBar();
            }
            view.redraw(true);
        };

        /** Black and white from the frame itself, as ImageJ's Auto sets them; pressed again, a tighter cut. */
        view.auto = function () {
            if (!view.data) return;
            const r = U16Math.imagejAuto(U16Math.histogram(view.data), view.data.length, view._autoThreshold || 0);
            view._autoThreshold = r.autoThreshold;
            view.lo = Math.round(r.lo); view.hi = Math.max(Math.round(r.hi), view.lo + 1); view._lut = null;
            view._paintBar();
        };

        view.setRange = function (black, white, gamma) {
            view.lo = black; view.hi = Math.max(white, black + 1); if (gamma) view.gamma = gamma;
            view._lut = null; paintBar(); view.redraw();
        };

        /** Redraw through the table; slider moves are coalesced to one frame. */
        view.redraw = function (now) {
            if (!view.data) return;
            const go = () => {
                view._raf = null;
                const t0 = performance.now();
                view._lut = U16Math.lut(view.lo, view.hi, view.gamma);
                U16Math.shade(view.data, view._lut, view._img.data);
                view.ctx.putImageData(view._img, 0, 0);
                view._lastDrawMs = performance.now() - t0;
            };
            if (now) { if (view._raf) cancelAnimationFrame(view._raf); go(); return; }
            if (!view._raf) view._raf = requestAnimationFrame(go);
        };

        view.destroy = function () { view._seq++; host.innerHTML = ''; host.classList.remove('u16'); view.data = null; };
        return view;
    },

    /**
     * A thumbnail: a bar-less view in `host` loading `url` (ask with ?max=N
     * for the size the tile has). If the counts cannot be had, an <img> of
     * `fallback` takes the host's place, so a tile is never blank.
     */
    tile(host, url, fallback, alt) {
        const view = U16View.create(host, { bar: false });
        view.load(url).catch(() => {
            const img = document.createElement('img');
            img.src = fallback || ''; img.alt = alt || ''; img.loading = 'lazy';
            host.replaceWith(img);
        });
        return view;
    },

    /** A frame over the page, full size, with the bar: for "look at this one". */
    overlay(url, title) {
        document.querySelectorAll('.u16-overlay').forEach(o => o.remove());
        const o = document.createElement('div');
        o.className = 'u16-overlay';
        o.innerHTML = `<div class="u16-overlay-head"><span class="u16-overlay-title">${title ? String(title).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])) : ''}</span><button type="button" class="u16-overlay-close" aria-label="Close">×</button></div><div class="u16-overlay-body"></div>`;
        document.body.appendChild(o);
        const close = () => { o.remove(); document.removeEventListener('keydown', onKey); };
        const onKey = e => { if (e.key === 'Escape') close(); };
        o.querySelector('.u16-overlay-close').onclick = close;
        o.addEventListener('click', e => { if (e.target === o) close(); });
        document.addEventListener('keydown', onKey);
        const view = U16View.create(o.querySelector('.u16-overlay-body'));
        view.load(url).catch(err => { view.info.textContent = `could not load: ${err.message}`; });
        return view;
    },
};
