/**
 * The overview stage: the brightfield frames of a run, one large, with a
 * timeline beneath.
 *
 *     OverviewStage.mount('embryos-overview-stage', {
 *         frames: () => EmbryosManager._dicFrames,     // oldest first
 *         references: () => EmbryosManager.run?.dic?.references,
 *     });
 *     OverviewStage.framesChanged();                   // after a frame lands
 *
 * In a brightfield-only run the frames ARE the experiment, so they get the
 * space: the newest frame large, a 44 px frame-indexed bar under it (every
 * frame an equal slot; a pause in the run is a seam, not a gap), a hover
 * preview above the bar, press-and-drag to scrub, ←/→ to step (Shift ×10),
 * Space to play, N for newest, C for corrected. Hover never moves the big
 * frame — a room may be watching it — only a drag does. "Following newest"
 * is an explicit state: any navigation disengages it, the Newest button
 * re-engages it, and a tag on the image says which it is.
 *
 * Scrubbing shows the ?max=640 PNG; on settle the full frame is swapped in.
 * ?corrected=1 asks the server to divide the session's dark and flat out
 * (gently.app.brightfield); the toggle is live only when the frame has them.
 * Compare shows frame 1 under a draggable divider on the same pixels, which
 * is how drift and growth are seen.
 */
const OverviewStage = (() => {
    'use strict';

    const SCRUB_MAX = 640;
    const PREVIEW_MAX = 128;
    const SETTLE_MS = 350;

    let host = null;
    let opts = { frames: () => [], references: () => null, onTakeReferences: null, onFold: null };
    let cur = -1;               // index into frames(); -1 = none yet
    let follow = true;
    let playing = false;
    let loop = false;
    let corrected = false;
    let compare = false;        // A/B: frame 1 under a divider
    let divider = 0.5;
    let hover = -1;
    let fps = 10;
    let timer = null;
    let settleTimer = null;
    let loadSeq = 0;
    const cache = new Map();    // url -> HTMLImageElement (decoded), LRU by insertion
    const CACHE_MAX = 160;
    let lastKey = '';           // what the file buttons and references line were last built for

    const $ = id => host && host.querySelector('#' + id);
    const esc = s => (typeof escapeHtml === 'function') ? escapeHtml(String(s == null ? '' : s)) : String(s == null ? '' : s);
    const fmtT = iso => { const d = new Date(iso); return isNaN(d) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }); };
    const fmtEl = ms => `${Math.floor(ms / 36e5)}h ${String(Math.floor(ms / 6e4) % 60).padStart(2, '0')}m`;
    const frames = () => opts.frames() || [];

    function urlFor(f, max) {
        if (!f || !f.url) return f && f.thumb ? f.thumb : '';
        const q = [];
        if (max) q.push(`max=${max}`);
        if (corrected && f.correctable) q.push('corrected=1');
        return f.url + (q.length ? `?${q.join('&')}` : '');
    }

    function load(url) {
        // One decoded Image per URL, evicting the oldest; the newest frame is
        // never the oldest entry for long.
        if (!url) return Promise.resolve(null);
        if (cache.has(url)) { const im = cache.get(url); cache.delete(url); cache.set(url, im); return Promise.resolve(im); }
        return new Promise(resolve => {
            const im = new Image();
            im.onload = () => {
                cache.set(url, im); while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
                // decoded before it is swapped in, or the swap itself flashes
                (im.decode ? im.decode().catch(() => {}) : Promise.resolve()).then(() => resolve(im));
            };
            im.onerror = () => resolve(null);
            im.src = url;
        });
    }

    function render() {
        if (!host) return;
        const all = frames();
        if (!all.length) {
            host.innerHTML = '';
            return;
        }
        if (cur < 0 || cur >= all.length) cur = all.length - 1;
        const f = all[cur];
        const refs = opts.references ? opts.references() : null;
        const canCorrect = !!f.correctable;
        if (!host.querySelector('.ov')) {
            lastKey = '';
            host.innerHTML = `
                <div class="ov" tabindex="0" aria-label="Overview frames: drag the bar or use arrow keys to scrub, space to play">
                    <div class="ov-stage" id="ov-stage">
                        <img class="ov-img" id="ov-img" alt="Overview frame">
                        <img class="ov-img ov-img-b" id="ov-img-b" alt="Overview frame 1, for comparison" hidden>
                        <div class="ov-divider" id="ov-divider" hidden><span></span></div>
                        <span class="ov-live" id="ov-live"></span>
                    </div>
                    <div class="ov-side">
                        <div class="ov-meta" id="ov-meta"></div>
                        <div class="ov-controls">
                            <button type="button" class="op-btn" id="ov-play" title="Space"></button>
                            <select id="ov-fps" class="op-sel" title="Frames per second"><option>2</option><option>5</option><option selected>10</option><option>20</option></select><span class="ov-keys">fps</span>
                            <button type="button" class="op-btn" id="ov-loop" title="Play again from the start">Loop</button>
                            <button type="button" class="op-btn" id="ov-corr" title="C — divide the session's dark and flat out"></button>
                            <button type="button" class="op-btn" id="ov-cmp" title="Frame 1 under a divider you can drag — drift and growth show at the line">Compare with first</button>
                            <button type="button" class="op-btn" id="ov-newest" title="N — the newest frame, and the ones that follow"></button>
                            ${opts.onFold ? '<button type="button" class="op-btn" id="ov-fold" title="Back to the row of thumbnails">Fold ⌃</button>' : ''}
                        </div>
                        <div class="ov-controls">
                            <span class="ov-reveal" id="ov-reveal"></span>
                            <span class="ov-refs" id="ov-refs"></span>
                        </div>
                        <span class="ov-keys"><kbd>←</kbd><kbd>→</kbd> step · <kbd>Shift</kbd> ×10 · <kbd>Space</kbd> play · <kbd>N</kbd> newest · <kbd>C</kbd> corrected</span>
                        <div class="ov-bar-wrap">
                            <div class="ov-preview" id="ov-preview"><img id="ov-pv-img" alt=""><span id="ov-pv-cap"></span></div>
                            <canvas class="ov-bar" id="ov-bar" height="44" aria-hidden="true"></canvas>
                        </div>
                    </div>
                </div>`;
            // The frame keeps its own aspect: as tall as the stage, as wide as that makes it.
            $('ov-img').addEventListener('load', ev => {
                const im = ev.target;
                if (im.naturalWidth && im.naturalHeight) $('ov-stage').style.aspectRatio = `${im.naturalWidth} / ${im.naturalHeight}`;
            });
            wire();
        }
        $('ov-live').textContent = follow ? 'FOLLOWING NEWEST' : `FRAME ${cur + 1} / ${all.length}`;
        $('ov-live').classList.toggle('is-on', follow);
        $('ov-play').textContent = playing ? '❚❚ Pause' : '▶ Play';
        $('ov-loop').classList.toggle('is-on', loop);
        const corr = $('ov-corr');
        corr.textContent = corrected ? 'Corrected' : 'Raw';
        corr.classList.toggle('is-on', corrected);
        corr.disabled = !canCorrect;
        corr.title = canCorrect ? 'C — divide the session\'s dark and flat out' : 'No dark and flat for this frame yet';
        $('ov-cmp').classList.toggle('is-on', compare);
        $('ov-newest').textContent = follow ? '● Following newest' : 'Newest ⤓';
        $('ov-newest').classList.toggle('is-follow', follow);
        $('ov-img-b').hidden = !compare;
        $('ov-divider').hidden = !compare;
        if (compare) {
            $('ov-img-b').style.clipPath = `inset(0 ${(100 - divider * 100).toFixed(2)}% 0 0)`;
            $('ov-divider').style.left = `${(divider * 100).toFixed(2)}%`;
        }
        renderMeta(f, all);
        // Rebuilt only when the frame (or its correction) changes: rebuilding
        // these every tick of playback made the controls flicker.
        const key = `${f.stem}|${canCorrect}|${corrected}|${refs && refs.record}`;
        if (!playing && key !== lastKey) {
            lastKey = key;
            renderRefs(f, refs);
            if (typeof Reveal !== 'undefined' && $('ov-reveal')) {
                Reveal.fill($('ov-reveal'), f.stem ? { what: 'dic', stem: f.stem } : null);
            }
        }
        drawBar();
        show(f);
    }

    function renderMeta(f, all) {
        const first = all[0];
        const prev = cur > 0 ? all[cur - 1] : null;
        const t = f.when ? new Date(f.when) : null;
        const t1 = first && first.when ? new Date(first.when) : null;
        const gap = prev && prev.when && f.when ? new Date(f.when) - new Date(prev.when) : 0;
        const typical = all.length > 2 && first && first.when && all[1].when ? Math.abs(new Date(all[1].when) - new Date(first.when)) : 0;
        const bits = [
            `<span class="ov-now">${t && !isNaN(t) ? `<b>${fmtT(f.when)}</b> · ` : ''}frame <b>${cur + 1}</b>/${all.length}</span>`,
            t && t1 && !isNaN(t) && !isNaN(t1) ? `<span>elapsed <b>${fmtEl(t - t1)}</b></span>` : '',
            f.round != null ? `<span>round <b>${esc(f.round)}</b></span>` : '',
            f.position && f.position.x != null ? `<span>stage <b>${Math.round(f.position.x)}, ${Math.round(f.position.y)}</b> µm</span>` : '',
            f.exposure_ms != null ? `<span>exposure <b>${esc(f.exposure_ms)} ms</b>${f.light ? ` · ${esc(f.light === 'led' ? `LED${f.led_intensity_pct != null ? ` ${f.led_intensity_pct}%` : ''}` : f.light === 'room' ? 'room light' : 'light as it is')}` : ''}</span>` : '',
            `<span>${corrected && f.correctable ? 'corrected' : 'raw'}</span>`,
            typical && gap > typical * 2 ? `<span class="ov-warn">paused ${fmtEl(gap)} before this frame</span>` : '',
        ];
        $('ov-meta').innerHTML = bits.filter(Boolean).join('');
    }

    function renderRefs(f, refs) {
        const el = $('ov-refs');
        if (!el) return;
        if (f.correctable || (refs && refs.dark && refs.flat)) {
            el.innerHTML = refs && refs.record ? `<span class="ov-ok">✓ dark and flat: ${esc(refs.record)}</span>` : '<span class="ov-ok">✓ dark and flat on file</span>';
            return;
        }
        el.innerHTML = `<span class="ov-missing">✕ No dark and flat for these frames yet.</span> ` +
            `<button type="button" class="ov-link" id="ov-take">Take them from Acquisition</button> <span class="ov-keys">— they can be taken after the run too; the frames will still be matched.</span>`;
        const b = $('ov-take');
        if (b) b.addEventListener('click', () => {
            if (typeof opts.onTakeReferences === 'function') opts.onTakeReferences();
        });
    }

    function show(f) {
        // The scrub size now; the full frame once the pointer settles.
        const seq = ++loadSeq;
        const img = $('ov-img');
        const quick = urlFor(f, SCRUB_MAX);
        load(quick).then(im => { if (seq === loadSeq && im) img.src = im.src; });
        clearTimeout(settleTimer);
        const all = frames();
        if (playing) {
            // Playing shows the scrub size only — swapping each frame to its
            // full-size file a moment later made every frame draw twice. The
            // full frame comes when playback stops. The next frames load now.
            [cur + 1, cur + 2, cur + 3].forEach(i => { if (all[i]) load(urlFor(all[i], SCRUB_MAX)); });
            return;
        }
        settleTimer = setTimeout(() => {
            if (seq !== loadSeq) return;
            load(urlFor(f, null)).then(im => { if (seq === loadSeq && im) img.src = im.src; });
            // neighbours, for the next step
            [cur - 1, cur + 1, cur + 2].forEach(i => { if (all[i]) load(urlFor(all[i], SCRUB_MAX)); });
        }, SETTLE_MS);
        if (compare) {
            const first = frames()[0];
            load(urlFor(first, SCRUB_MAX)).then(im => { if (im) $('ov-img-b').src = im.src; });
        }
    }

    function drawBar() {
        const bar = $('ov-bar');
        if (!bar) return;
        const all = frames(), n = all.length;
        if (!n) return;
        // Drawn while the tab was hidden, the canvas has no width and a 1 px
        // bitmap stretches to a solid bar. Wait for size; the observer redraws.
        if (!bar.clientWidth) return;
        const dpr = window.devicePixelRatio || 1;
        const W = bar.width = bar.clientWidth * dpr, H = bar.height = 44 * dpr, slot = W / n;
        const ctx = bar.getContext('2d');
        const css = getComputedStyle(document.documentElement), tok = v => css.getPropertyValue(v).trim() || '#888';
        ctx.fillStyle = tok('--bg-hover'); ctx.fillRect(0, 0, W, H);
        if (slot >= 4 * dpr) {
            ctx.fillStyle = tok('--border');
            for (let i = 0; i < n; i++) ctx.fillRect(i * slot + slot / 2 - 0.5, H * 0.35, 1, H * 0.3);
        }
        // a pause in the run: a seam between the frames it separates
        let typical = 0;
        if (n > 2 && all[0].when && all[1].when) typical = Math.abs(new Date(all[1].when) - new Date(all[0].when));
        if (typical) {
            ctx.fillStyle = tok('--text-muted');
            for (let i = 1; i < n; i++) {
                if (all[i].when && all[i - 1].when && new Date(all[i].when) - new Date(all[i - 1].when) > typical * 2) ctx.fillRect(i * slot - 1, 0, 2, H);
            }
        }
        ctx.fillStyle = tok('--accent-green'); ctx.fillRect(W - Math.max(slot, 3 * dpr), 0, Math.max(slot, 3 * dpr), H);
        if (hover >= 0) { ctx.fillStyle = tok('--text-muted'); ctx.fillRect(hover * slot, 0, Math.max(slot, 2 * dpr), H); }
        ctx.fillStyle = tok('--accent'); ctx.fillRect(cur * slot, 0, Math.max(slot, 3 * dpr), H);
    }

    function go(i, byUser) {
        const n = frames().length;
        if (!n) return;
        cur = Math.max(0, Math.min(n - 1, i));
        if (byUser) follow = false;
        render();
    }

    function idxAt(ev) {
        const bar = $('ov-bar'), r = bar.getBoundingClientRect(), n = frames().length;
        return Math.min(n - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * n)));
    }

    function preview(ev, i) {
        const bar = $('ov-bar'), r = bar.getBoundingClientRect(), f = frames()[i];
        hover = i;
        const pv = $('ov-preview');
        pv.style.display = 'block';
        pv.style.left = `${Math.min(r.width - 56, Math.max(56, ev.clientX - r.left))}px`;
        const url = f.thumb && f.thumb.startsWith('data:') ? f.thumb : urlFor(f, PREVIEW_MAX);
        load(url).then(im => { if (im && hover === i) $('ov-pv-img').src = im.src; });
        $('ov-pv-cap').textContent = `${f.frame ?? i + 1}${f.when ? ` · ${fmtT(f.when)}` : ''}`;
        drawBar();
    }

    function tick() {
        const n = frames().length;
        if (cur >= n - 1) { if (loop) go(0, false); else { stop(); return; } } else go(cur + 1, false);
    }
    function play() {
        if (playing) return;
        playing = true; follow = false;
        if (cur >= frames().length - 1) cur = 0;
        timer = setInterval(tick, 1000 / fps);
        render();
    }
    function stop() { playing = false; clearInterval(timer); timer = null; render(); }
    function newest() { stop(); follow = true; go(frames().length - 1, false); }
    function toggleCorrected() { const f = frames()[cur]; if (!f || !f.correctable) return; corrected = !corrected; render(); }

    function onKey(ev) {
        if (!host || !host.offsetParent) return;                  // the tab is not showing
        const t = ev.target, tag = t && t.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
        const k = ev.key, step = ev.shiftKey ? 10 : 1;
        if (k === 'ArrowRight' || k === '.') { stop(); go(cur + step, true); }
        else if (k === 'ArrowLeft' || k === ',') { stop(); go(cur - step, true); }
        else if (k === 'Home') { stop(); go(0, true); }
        else if (k === 'End') { stop(); go(frames().length - 1, true); }
        else if (k === ' ') { playing ? stop() : play(); }
        else if (k === 'n' || k === 'N') newest();
        else if (k === 'c' || k === 'C') toggleCorrected();
        else return;
        ev.preventDefault();
    }

    function wire() {
        const bar = $('ov-bar');
        bar.addEventListener('pointermove', ev => { const i = idxAt(ev); preview(ev, i); if (ev.buttons) { stop(); go(i, true); } });
        bar.addEventListener('pointerdown', ev => { bar.setPointerCapture(ev.pointerId); host.querySelector('.ov').focus(); stop(); go(idxAt(ev), true); });
        bar.addEventListener('pointerleave', () => { hover = -1; $('ov-preview').style.display = 'none'; drawBar(); });
        $('ov-play').addEventListener('click', () => playing ? stop() : play());
        $('ov-loop').addEventListener('click', () => { loop = !loop; render(); });
        $('ov-corr').addEventListener('click', toggleCorrected);
        $('ov-cmp').addEventListener('click', () => { compare = !compare; render(); });
        $('ov-newest').addEventListener('click', newest);
        const fold = $('ov-fold');
        if (fold) fold.addEventListener('click', () => { stop(); if (typeof opts.onFold === 'function') opts.onFold(); });
        $('ov-fps').addEventListener('change', ev => { fps = Number(ev.target.value) || 10; if (playing) { stop(); play(); } });
        // the compare divider
        const stage = $('ov-stage');
        let dragging = false;
        stage.addEventListener('pointerdown', ev => { if (!compare) return; dragging = true; stage.setPointerCapture(ev.pointerId); });
        stage.addEventListener('pointermove', ev => { if (!compare || !dragging) return; const r = stage.getBoundingClientRect(); divider = Math.min(0.98, Math.max(0.02, (ev.clientX - r.left) / r.width)); render(); });
        stage.addEventListener('pointerup', () => { dragging = false; });
        // press-and-hold the picture plays it (touch), release stays
        let hold = null;
        stage.addEventListener('pointerdown', ev => { if (compare || ev.pointerType !== 'touch') return; hold = setTimeout(play, 350); });
        stage.addEventListener('pointerup', () => { clearTimeout(hold); if (playing) stop(); });
        document.addEventListener('keydown', onKey);
        window.addEventListener('resize', drawBar);
        if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => drawBar()).observe(bar);
    }

    function mount(hostId, options) {
        host = document.getElementById(hostId);
        opts = Object.assign(opts, options || {});
        if (!host) return;
        render();
    }

    /** A frame landed, or the list was reloaded. Following newest jumps to it. */
    function framesChanged() {
        if (!host) return;
        const n = frames().length;
        if (follow || cur < 0) cur = n - 1;
        render();
    }

    return { mount, framesChanged, go, newest, play, stop, current: () => cur, following: () => follow };
})();

if (typeof module !== 'undefined') module.exports = OverviewStage;
