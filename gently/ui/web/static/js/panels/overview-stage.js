/**
 * A stage: a series of frames, one large, with a timeline beneath.
 *
 * Built for the brightfield overview, and the same thing an embryo's
 * projections want: "the show latest feature etc. the player was rather
 * nice." So it is a factory; each surface mounts its own instance.
 *
 *     const stage = createOverviewStage();
 *     stage.mount('embryos-overview', {
 *         frames: () => EmbryosManager._bfFrames,     // oldest first
 *         references: () => EmbryosManager.run?.bf?.references,
 *     });
 *     stage.framesChanged();                           // after a frame lands
 *
 * `OverviewStage` is the instance the brightfield overview has always used.
 *
 * The newest frame large, a 44 px frame-indexed bar under it (every frame an
 * equal slot; a pause in the run is a seam, not a gap), a hover preview above
 * the bar, press-and-drag to scrub, ←/→ to step (Shift ×10), Space to play,
 * N for newest, C for corrected. Hover never moves the big frame — a room may
 * be watching it — only a drag does. "Following newest" is an explicit state:
 * any navigation disengages it, the Newest button re-engages it, and a tag on
 * the image says which it is. With two stages on screen the keys go to the
 * one last touched.
 *
 * Options beyond `frames`:
 *   tools      which controls exist: {corrected, compare, references, reveal,
 *              fields} — all on for the overview, most off for projections.
 *   meta(f, i, all)  the caption under the controls, as HTML; the default
 *              reads the overview frame's record.
 *   color(f)   a colour per frame for the bar, e.g. the stage call at that
 *              timepoint; null leaves the slot plain.
 *   onFrame(i, {playing, byUser})  after the stage moves, so a detail pane
 *              can follow.
 *
 * Scrubbing shows the ?max=640 PNG; on settle the full frame is swapped in.
 * ?corrected=1 asks the server to divide the session's dark and flat out
 * (gently.app.brightfield); the toggle is live only when the frame has them.
 * Compare shows frame 1 under a draggable divider on the same pixels, which
 * is how drift and growth are seen.
 */
let _activeStage = null;   // the instance the keyboard talks to: the last mounted or touched

function createOverviewStage() {
    'use strict';

    const SCRUB_MAX = 640;
    const PREVIEW_MAX = 128;
    const SETTLE_MS = 350;
    const TOOLS = { corrected: true, compare: true, references: true, reveal: true, fields: true };

    let host = null;
    let opts = { frames: () => [], references: () => null, onTakeReferences: null, onFold: null, fields: null, onField: null,
                 tools: TOOLS, meta: null, color: null, onFrame: null, label: 'Overview frames' };
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
    let wired = false;          // document and window listeners, once per instance
    let observer = null;
    const cache = new Map();    // url -> HTMLImageElement (decoded), LRU by insertion
    const CACHE_MAX = 160;
    let lastKey = '';           // what the file buttons and references line were last built for
    let api = null;

    const $ = id => host && host.querySelector('#' + id);
    const esc = s => (typeof escapeHtml === 'function') ? escapeHtml(String(s == null ? '' : s)) : String(s == null ? '' : s);
    const fmtT = iso => { const d = new Date(iso); return isNaN(d) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }); };
    const fmtEl = ms => `${Math.floor(ms / 36e5)}h ${String(Math.floor(ms / 6e4) % 60).padStart(2, '0')}m`;
    const frames = () => opts.frames() || [];
    const tools = () => Object.assign({}, TOOLS, opts.tools || {});
    const supportsMax = f => !!(f && f.url && f.scalable !== false);

    function urlFor(f, max) {
        if (!f || !f.url) return f && f.thumb ? f.thumb : '';
        const q = [];
        if (max && supportsMax(f)) q.push(`max=${max}`);
        if (corrected && f.correctable) q.push('corrected=1');
        return f.url + (q.length ? `${f.url.includes('?') ? '&' : '?'}${q.join('&')}` : '');
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
        const t = tools();
        const refs = t.references && opts.references ? opts.references() : null;
        const canCorrect = t.corrected && !!f.correctable;
        if (!host.querySelector('.ov')) {
            lastKey = '';
            host.innerHTML = `
                <div class="ov" tabindex="0" aria-label="${esc(opts.label)}: drag the bar or use arrow keys to scrub, space to play">
                    <div class="ov-stage" id="ov-stage">
                        <img class="ov-img" id="ov-img" alt="${esc(opts.label)}">
                        <img class="ov-img ov-img-b" id="ov-img-b" alt="Frame 1, for comparison" hidden>
                        <div class="ov-divider" id="ov-divider" hidden><span></span></div>
                        <span class="ov-live" id="ov-live"></span>
                    </div>
                    <div class="ov-side">
                        <div class="ov-meta" id="ov-meta"></div>
                        <div class="ov-controls ov-fields" id="ov-fields" hidden></div>
                        <div class="ov-controls">
                            <button type="button" class="op-btn" id="ov-play" title="Space"></button>
                            <select id="ov-fps" class="op-sel" title="Frames per second"><option>2</option><option>5</option><option selected>10</option><option>20</option></select><span class="ov-keys">fps</span>
                            <button type="button" class="op-btn" id="ov-loop" title="Play again from the start">Loop</button>
                            ${t.corrected ? '<button type="button" class="op-btn" id="ov-corr" title="C — divide the session\'s dark and flat out"></button>' : ''}
                            ${t.compare ? '<button type="button" class="op-btn" id="ov-cmp" title="Frame 1 under a divider you can drag — drift and growth show at the line">Compare with first</button>' : ''}
                            <button type="button" class="op-btn" id="ov-newest" title="N — the newest frame, and the ones that follow"></button>
                            ${opts.onFold ? '<button type="button" class="op-btn" id="ov-fold" title="Back to the row of thumbnails">Fold ⌃</button>' : ''}
                        </div>
                        ${t.reveal || t.references ? `<div class="ov-controls">
                            <span class="ov-reveal" id="ov-reveal"></span>
                            <span class="ov-refs" id="ov-refs"></span>
                        </div>` : ''}
                        <span class="ov-keys"><kbd>←</kbd><kbd>→</kbd> step · <kbd>Shift</kbd> ×10 · <kbd>Space</kbd> play · <kbd>N</kbd> newest${t.corrected ? ' · <kbd>C</kbd> corrected' : ''}</span>
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
        // Taken from more than one position: one field on the stage at a
        // time, the others a click away.
        const fl = t.fields && opts.fields ? opts.fields() : null;
        const fieldsEl = $('ov-fields');
        if (fl && fl.list && fl.list.length > 1) {
            const key = `${fl.list.join(',')}|${fl.current}`;
            if (fieldsEl.dataset.key !== key) {
                fieldsEl.dataset.key = key;
                fieldsEl.innerHTML = '<span class="ov-keys">Field</span>' + fl.list.map(n =>
                    `<button type="button" class="op-btn${n === fl.current ? ' is-on' : ''}" data-ov-field="${n}" title="The frames taken from position ${n}">${n}</button>`).join('');
            }
            fieldsEl.hidden = false;
        } else {
            fieldsEl.hidden = true;
        }
        $('ov-live').textContent = (follow ? 'FOLLOWING NEWEST' : `FRAME ${cur + 1} / ${all.length}`) + (fl && fl.list && fl.list.length > 1 ? ` · FIELD ${fl.current}` : '');
        $('ov-live').classList.toggle('is-on', follow);
        $('ov-play').textContent = playing ? '❚❚ Pause' : '▶ Play';
        $('ov-loop').classList.toggle('is-on', loop);
        const corr = $('ov-corr');
        if (corr) {
            corr.textContent = corrected ? 'Corrected' : 'Raw';
            corr.classList.toggle('is-on', corrected);
            corr.disabled = !canCorrect;
            corr.title = canCorrect ? 'C — divide the session\'s dark and flat out' : 'No dark and flat for this frame yet';
        }
        if ($('ov-cmp')) $('ov-cmp').classList.toggle('is-on', compare);
        $('ov-newest').textContent = follow ? '● Following newest' : 'Newest ⤓';
        $('ov-newest').classList.toggle('is-follow', follow);
        $('ov-img-b').hidden = !compare;
        $('ov-divider').hidden = !compare;
        if (compare) {
            $('ov-img-b').style.clipPath = `inset(0 ${(100 - divider * 100).toFixed(2)}% 0 0)`;
            $('ov-divider').style.left = `${(divider * 100).toFixed(2)}%`;
        }
        if (typeof opts.meta === 'function') $('ov-meta').innerHTML = opts.meta(f, cur, all) || '';
        else renderMeta(f, all);
        // Rebuilt only when the frame (or its correction) changes: rebuilding
        // these every tick of playback made the controls flicker.
        const key = `${f.stem}|${canCorrect}|${corrected}|${refs && refs.record}`;
        if (!playing && key !== lastKey) {
            lastKey = key;
            if (t.references) renderRefs(f, refs);
            if (t.reveal && typeof Reveal !== 'undefined' && $('ov-reveal')) {
                Reveal.fill($('ov-reveal'), f.stem ? { what: 'bf', stem: f.stem } : null);
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
        // A colour per frame — the stage call at that timepoint — as a band
        // through the middle, so the series reads before any frame is opened.
        if (typeof opts.color === 'function') {
            for (let i = 0; i < n; i++) {
                const c = opts.color(all[i]);
                if (!c) continue;
                ctx.fillStyle = c; ctx.fillRect(i * slot, H * 0.3, Math.max(slot, 1), H * 0.4);
            }
        } else if (slot >= 4 * dpr) {
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
        if (typeof opts.onFrame === 'function') opts.onFrame(cur, { playing, byUser: !!byUser });
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
        $('ov-pv-cap').textContent = `${f.frame ?? i + 1}${f.when ? ` · ${fmtT(f.when)}` : ''}${f.caption ? ` · ${f.caption}` : ''}`;
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
    function stop() {
        const was = playing;
        playing = false; clearInterval(timer); timer = null; render();
        if (was && typeof opts.onFrame === 'function') opts.onFrame(cur, { playing: false, byUser: true });
    }
    function newest() { stop(); follow = true; go(frames().length - 1, false); }
    function toggleCorrected() { const f = frames()[cur]; if (!f || !f.correctable || !tools().corrected) return; corrected = !corrected; render(); }

    function onKey(ev) {
        if (!host || !host.offsetParent) return;                  // the tab is not showing
        if (_activeStage && _activeStage !== api) return;         // another stage was touched last
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
        if ($('ov-corr')) $('ov-corr').addEventListener('click', toggleCorrected);
        if ($('ov-cmp')) $('ov-cmp').addEventListener('click', () => { compare = !compare; render(); });
        $('ov-fields').addEventListener('click', ev => {
            const b = ev.target.closest('[data-ov-field]');
            if (b && opts.onField) { stop(); opts.onField(Number(b.dataset.ovField)); }
        });
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
        // whichever stage was touched last owns the keys
        host.addEventListener('pointerdown', () => { _activeStage = api; });
        if (observer) observer.disconnect();
        if (typeof ResizeObserver !== 'undefined') { observer = new ResizeObserver(() => drawBar()); observer.observe(bar); }
        if (!wired) {
            wired = true;
            document.addEventListener('keydown', onKey);
            window.addEventListener('resize', drawBar);
        }
    }

    function mount(hostId, options) {
        const el = document.getElementById(hostId);
        opts = Object.assign(opts, options || {});
        if (!el) return;
        if (host !== el) { host = el; _activeStage = api; }   // a new host takes the keys; a re-mount keeps them where they are
        render();
    }

    /** A frame landed, or the list was reloaded. Following newest jumps to it. */
    function framesChanged() {
        if (!host) return;
        const n = frames().length;
        if (follow || cur < 0) cur = n - 1;
        render();
    }

    /** A different series on the same stage: start at its newest, following. */
    function reset() {
        stop();
        cur = -1; follow = true; compare = false; hover = -1; lastKey = '';
        if (host) host.innerHTML = '';
    }

    api = { mount, framesChanged, go, newest, play, stop, reset, current: () => cur, following: () => follow, playing: () => playing };
    return api;
}

/** The brightfield overview's stage, as it has always been reached. */
const OverviewStage = createOverviewStage();

if (typeof module !== 'undefined') { module.exports = OverviewStage; module.exports.createOverviewStage = createOverviewStage; }
