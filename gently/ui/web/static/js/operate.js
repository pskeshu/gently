/**
 * Operate — three independent instrument surfaces.
 *
 *   Bottom cam   see the dish, focus the bottom objective, find embryos
 *   SPIM head    bring the objectives to height over the sample
 *   Acquisition  the embryo roster and what to run on it
 *
 * There are no steps, no phases and no progress ladder. Every pane is fully
 * live whenever it is visible.
 *
 * THE INVARIANT: no control anywhere reads `_pane`. It is consulted by exactly
 * two things — render dispatch, and which camera stream owns MMCore. If a
 * `disabled`, `hidden` or early-return ever starts keying off `_pane`, the step
 * model has grown back and this rewrite has been undone.
 *
 * Gating comes from live hardware state only: the XY interlock (moveStageTo /
 * OperateMath.isEngaged) and the F-drive floor (OperateMath.stepAllowed).
 * Selection is a cursor — it parameterises requests and never disables anything.
 *
 * Pure geometry, banding and the interlock predicate live in operate-math.js so
 * they can be unit-tested (tests/js/operate-math.test.mjs).
 */
const OperateManager = (function () {
    const M = (typeof OperateMath !== 'undefined') ? OperateMath : null;
    const MARK_HIT_PX = 14;

    let _wired = false, _active = false;

    // ── navigation: the ONLY navigation state in this file ──────────────────
    let _pane = 'bottom';

    // ── session facts (owned by the bus, mirrored here) ─────────────────────
    let _embryos = [];
    // A CURSOR, not a step. It parameterises request bodies and readouts. It
    // must never appear in a `disabled` or visibility expression.
    let _selected = null;

    // ── device state ────────────────────────────────────────────────────────
    let _xy = null;                 // {x, y} from DEVICE_STATE_UPDATE
    // Last bottom-cam frame. Kept because the marking canvas needs its geometry
    // and capture position; the SPIM frame is only ever displayed, so it isn't.
    let _lastBottom = null;

    // ── stream ownership, per pane ──────────────────────────────────────────
    let _bottomOn = false, _bottomWasOn = false;
    let _spimOn = false, _spimWasOn = false;

    // ── the interlock latch ─────────────────────────────────────────────────
    // Retract is a RELATIVE move, so there is no absolute "safe height" to
    // derive this from — asking the hardware cannot answer it. Persist it, so
    // the ordinary reaction to something looking stuck (F5) doesn't come back
    // claiming the head is up and re-enable an absolute XY move with the
    // objective down. Telemetry can still clear it (see OperateMath.isEngaged).
    const HEAD_KEY = 'gently.operate.headLowered';
    function loadHeadLowered() {
        try { return sessionStorage.getItem(HEAD_KEY) === '1'; } catch (_) { return false; }
    }
    let _headLowered = loadHeadLowered();
    function setHeadLowered(v) {
        _headLowered = !!v;
        try { sessionStorage.setItem(HEAD_KEY, _headLowered ? '1' : '0'); } catch (_) {}
        renderLock();
    }

    // ── marking ─────────────────────────────────────────────────────────────
    // Markers are held in STAGE coordinates and re-projected onto whatever frame
    // is current, so they stay attached to the sample instead of the viewport.
    // That is what lets marking be always-on rather than a mode you enter.
    let _markers = [];

    // ── emitters / run ──────────────────────────────────────────────────────
    let _ledOn = false, _acquiring = false;
    let _galvo = 0.0, _piezo = 50.0;
    let _mode = 'single';
    let _selectedLib = null;
    let _runPaused = false;

    // ── primitives ──────────────────────────────────────────────────────────
    function $(id) { return document.getElementById(id); }
    function toast(m) { if (typeof showGentlyToast === 'function') showGentlyToast(m); }
    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, c =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    async function postJSON(url, body) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        const text = await res.text().catch(() => '');
        let data = {};
        try { data = text ? JSON.parse(text) : {}; } catch (_) { /* not JSON */ }
        if (!res.ok) {
            const e = new Error(`${res.status} ${data.error || text}`);
            e.status = res.status;
            // A fenced nudge answers 400 but still reports where the axis
            // actually is. Carry the body so the caller can re-absorb it
            // instead of leaving the gauge stale.
            e.data = data;
            throw e;
        }
        return data;
    }
    async function getJSON(url) {
        const res = await fetch(url);
        const text = await res.text().catch(() => '');
        let data = {};
        try { data = text ? JSON.parse(text) : {}; } catch (_) { /* not JSON */ }
        if (!res.ok) { const e = new Error(String(res.status)); e.status = res.status; e.data = data; throw e; }
        return data;
    }

    function labelFor(emb) {
        const m = emb && emb.id && String(emb.id).match(/(\d+)/);
        return m ? m[1] : '?';
    }
    function resolveXY(emb) {
        const f = emb && emb.position_fine;
        if (f && Number.isFinite(f.x) && Number.isFinite(f.y)) return { x: f.x, y: f.y };
        const c = emb && emb.position_coarse;
        if (c && Number.isFinite(c.x) && Number.isFinite(c.y)) return { x: c.x, y: c.y };
        return null;
    }

    // ══ THE XY CHOKEPOINT ═══════════════════════════════════════════════════
    // The ONLY caller of /api/devices/stage/move in this file. The server does
    // NOT interlock XY against the F-drive — routes/data.py validates that x and
    // y are floats and nothing else — so this predicate is the whole guard.
    // Do not add a second fetch of this endpoint; grep the URL string before
    // touching XY motion.
    async function moveStageTo(x, y, why) {
        if (isEngaged()) {
            toast('Sample is at the objective — back it off before moving XY');
            return false;
        }
        try {
            await postJSON('/api/devices/stage/move', { x, y });
            if (why) toast(why);
            return true;
        } catch (e) {
            toast(`Move failed (${e.status || e.message})`);
            return false;
        }
    }
    function isEngaged() {
        return M ? M.isEngaged(_headLowered, fd.floor()) : _headLowered;
    }

    // ══ Z INSTRUMENT ════════════════════════════════════════════════════════
    // One factory, two instances. The axes differ enormously in scale — the
    // bottom objective has a ~200 µm throw, the F-drive runs 30→25000 µm onto a
    // sample sitting at ~50 — so the F-drive gets position-banded steps and a
    // log track, and the bottom axis a fixed ladder on a linear one.
    function makeZAxis(cfg) {
        const st = { pos: null, min: null, max: null, floor: null, status: 'unknown' };
        let busy = false;

        const num = v => (v == null || !Number.isFinite(Number(v))) ? null : Number(v);

        function absorb(d) {
            if (!d) return;
            if (num(d.position) != null) st.pos = num(d.position);
            if (num(d.min) != null) st.min = num(d.min);
            if (num(d.max) != null) st.max = num(d.max);
            if (num(d.distance_to_floor) != null) st.floor = num(d.distance_to_floor);
            else if (st.pos != null && st.min != null) st.floor = st.pos - st.min;
            if (st.pos != null) st.status = 'ok';
            render();
        }
        // 1 Hz position stream. Without this the gauge goes stale the moment
        // someone drives the axis at the controller box instead of from here.
        function absorbTelemetry(val) {
            if (!Number.isFinite(val)) return;
            st.pos = val;
            if (st.min != null) st.floor = val - st.min;
            if (st.status !== 'absent') st.status = 'ok';
            render();
        }
        function fail(e) {
            const msg = String((e && e.data && e.data.error) || (e && e.message) || '');
            // An axis this rig does not have is a FACT, not an error: the device
            // layer 503s with "device not found". Render it as absent, quietly.
            st.status = (e && e.status === 503 && /not found/i.test(msg)) ? 'absent' : 'error';
            render();
        }

        function steps() {
            if (!cfg.bands) return cfg.steps;
            return (M ? M.fdBand(st.pos) : { steps: cfg.steps }).steps;
        }
        function bandLabel() {
            if (!cfg.bands) return '';
            if (st.pos == null) return 'position unknown';
            return (M ? M.fdBand(st.pos).label : '');
        }

        function renderNudges() {
            const host = $(cfg.root + '-nudge');
            if (!host) return;
            if (st.status === 'absent') { host.innerHTML = ''; host.dataset.band = ''; return; }
            const s = steps();
            const key = s.join(',');
            if (host.dataset.band !== key) {
                host.dataset.band = key;
                host.style.gridTemplateColumns = `repeat(${s.length}, minmax(0, 1fr))`;
                // Ups on the top row, downs on the bottom, columns aligned by
                // magnitude, so the control maps to the motion.
                host.innerHTML =
                    s.map(v => `<button class="op-nbtn" data-nudge="${v}" type="button">▲&nbsp;${v}</button>`).join('') +
                    s.map(v => `<button class="op-nbtn" data-nudge="${-v}" type="button">▼&nbsp;${v}</button>`).join('');
            }
            host.querySelectorAll('[data-nudge]').forEach(b => {
                const d = Number(b.dataset.nudge);
                b.disabled = busy || (M ? !M.stepAllowed(d, st.floor) : false);
            });
        }

        function renderTicks() {
            const host = $(cfg.root + '-ticks');
            if (!host || !cfg.ticks) return;
            const key = `${st.min},${st.max}`;
            if (host.dataset.k === key) return;
            host.dataset.k = key;
            if (st.status !== 'ok' || st.min == null || st.max == null) { host.innerHTML = ''; return; }
            host.innerHTML = cfg.ticks
                .filter(t => t > st.min && t < st.max)
                .map(t => {
                    const f = M ? M.gaugeFraction(t, st.min, st.max, cfg.scale) : null;
                    return f == null ? '' : `<span style="bottom:${(f * 100).toFixed(2)}%">${t}</span>`;
                }).join('');
        }

        function render() {
            const g = $(cfg.gauge);
            if (g) {
                g.dataset.status = st.status;
                g.classList.toggle('is-near-floor',
                    st.status === 'ok' && st.floor != null && st.floor < 100);
            }
            const read = $(cfg.root + '-pos');
            if (read) {
                read.textContent = st.status === 'absent' ? 'n/a'
                    : (st.pos == null ? '—' : st.pos.toFixed(1));
            }
            const mark = $(cfg.root + '-mark');
            // Null fraction means "do not draw a marker" — a marker parked at
            // the bottom of the track reads as "at the limit", which is a lie
            // when the truth is that nothing is known yet.
            const frac = (st.status === 'ok' && M)
                ? M.gaugeFraction(st.pos, st.min, st.max, cfg.scale) : null;
            if (mark) {
                if (frac == null) mark.style.display = 'none';
                else { mark.style.display = 'block'; mark.style.bottom = `${(frac * 100).toFixed(2)}%`; }
            }
            const lo = $(cfg.root + '-min'), hi = $(cfg.root + '-max');
            if (lo) lo.textContent = st.min == null ? '—' : Math.round(st.min);
            if (hi) hi.textContent = st.max == null ? '—' : Math.round(st.max);

            const track = $(cfg.root + '-track');
            if (track) {
                track.setAttribute('aria-valuetext',
                    st.status === 'absent' ? 'axis not present'
                        : st.pos == null ? 'unknown' : `${st.pos.toFixed(1)} micrometres`);
            }
            // An absent or unreachable axis says so wherever this gauge has room
            // for a line of text — the banded axis uses its band caption, the
            // plain one its foot.
            const statusText = st.status === 'absent' ? 'axis not present on this rig'
                : st.status === 'error' ? 'position unavailable' : null;
            const band = $(cfg.root + '-band');
            if (band) band.textContent = statusText || bandLabel();
            const floor = $(cfg.root + '-floor');
            if (floor) floor.textContent = st.floor == null ? '—' : Math.round(st.floor);
            const foot = $(cfg.root + '-foot');
            if (foot) foot.textContent = statusText || '';
            renderTicks();
            renderNudges();
            renderLock();
        }

        async function nudge(delta) {
            if (M && !M.stepAllowed(delta, st.floor)) {
                toast('Too close to the floor for that step');
                return;
            }
            busy = true; renderNudges();
            try {
                absorb(await postJSON(cfg.nudge, { delta }));
                if (cfg.onDown && delta < 0) cfg.onDown();
                if (cfg.onUp && delta > 0) cfg.onUp(st);
            } catch (e) {
                // Even a refused nudge reports the real position — take it.
                if (e && e.data && e.data.position != null) absorb(e.data);
                toast(`${cfg.label} nudge blocked (${e.status || e.message})`);
            } finally { busy = false; renderNudges(); }
        }

        async function refresh() {
            try { absorb(await getJSON(cfg.get)); } catch (e) { fail(e); }
        }

        function wire() {
            const host = $(cfg.root + '-nudge');
            if (host) {
                host.addEventListener('click', e => {
                    const b = e.target.closest('[data-nudge]');
                    if (b && !b.disabled) nudge(Number(b.dataset.nudge));
                });
            }
            const track = $(cfg.root + '-track');
            if (track) {
                track.addEventListener('keydown', e => {
                    const s = steps();
                    const fine = s[s.length - 1], coarse = s[0];
                    const map = { ArrowUp: fine, ArrowDown: -fine, PageUp: coarse, PageDown: -coarse };
                    if (map[e.key] == null) return;
                    e.preventDefault();
                    nudge(map[e.key]);
                });
            }
        }

        return {
            wire, refresh, absorb, absorbTelemetry, render, nudge,
            floor: () => st.floor,
            status: () => st.status,
        };
    }

    const bz = makeZAxis({
        root: 'op-bz', gauge: 'op-gauge-bz', label: 'Bottom-Z',
        get: '/api/devices/stage/bottom_z',
        nudge: '/api/devices/stage/bottom_z/nudge',
        steps: [10, 1], scale: 'linear', bands: false,
    });
    const fd = makeZAxis({
        root: 'op-fd', gauge: 'op-gauge-fd', label: 'F-drive',
        get: '/api/devices/spim/fdrive',
        nudge: '/api/devices/spim/fdrive/nudge',
        bands: true, scale: 'log', ticks: [100, 1000, 10000],
        steps: [50, 10, 5],
        onDown: () => setHeadLowered(true),
    });

    // ══ THE INTERLOCK, MADE VISIBLE ═════════════════════════════════════════
    // Enforcement alone is not enough: with click-to-center always available,
    // the operator must be able to see WHY a click will not do anything. The
    // affordance withdraws itself (locked cursor) and the banner says so.
    function renderLock() {
        const eng = isEngaged();
        const d = fd.floor();
        ['bottom', 'spim'].forEach(p => {
            const el = $(`op-lock-${p}`);
            if (el) el.hidden = !eng;
            const dd = $(`op-lock-${p}-d`);
            if (dd) dd.textContent = d == null ? '—' : Math.round(d);
        });
        const cam = $('op-cam-bottom');
        if (cam) cam.classList.toggle('is-locked', eng);
        drawMarkers();
    }
    // Always reachable, unlike the old design where clearing the latch lived on
    // a step you might never arrive at — lower the head, never reach it, and XY
    // stayed locked forever with no escape short of clearing sessionStorage.
    async function backOff() {
        try {
            fd.absorb(await postJSON('/api/devices/spim/fdrive/nudge', { delta: 100 }));
            // A retract that FAILS must not report the head as up: that is the
            // state the XY chokepoint trusts before commanding an absolute move.
            setHeadLowered(false);
            toast('Backed off 100 µm');
        } catch (e) {
            if (e && e.data && e.data.position != null) fd.absorb(e.data);
            toast(`Back-off failed (${e.status || e.message}) — head still down`);
        }
    }

    // ══ VIEWPORT ════════════════════════════════════════════════════════════
    function frameOf(p) {
        if (!p || !Array.isArray(p.shape)) return null;
        return { w: p.shape[1], h: p.shape[0], downsample: p.downsample || 1 };
    }
    function stageOf(p) {
        if (p && Array.isArray(p.stage_position) && p.stage_position.length === 2) return p.stage_position;
        // The device layer OMITS stage_position rather than defaulting it to
        // [0,0] when it cannot know it. Honour that: fall back to the position
        // stream, never to the origin.
        if (_xy && Number.isFinite(_xy.x) && Number.isFinite(_xy.y)) return [_xy.x, _xy.y];
        return null;
    }
    function setImg(imgId, phId, p) {
        const img = $(imgId), ph = $(phId);
        if (!img || !p || !p.jpeg_b64) return;
        img.src = `data:${p.mime || 'image/jpeg'};base64,${p.jpeg_b64}`;
        if (!img.classList.contains('has-frame')) {
            img.classList.add('has-frame');
            if (ph) ph.style.display = 'none';
        }
        // Match the viewport box to the frame's aspect so the border hugs the
        // image. naturalWidth is 0 until the data URL decodes, so fall back to a
        // one-shot load listener.
        if (img.naturalWidth && img.naturalHeight) setCamAspect(img);
        else img.addEventListener('load', () => setCamAspect(img), { once: true });
    }
    function setCamAspect(img) {
        const fit = img.closest('.op-cam-fit');
        if (fit && img.naturalWidth && img.naturalHeight) {
            fit.style.setProperty('--cam-ar', `${img.naturalWidth} / ${img.naturalHeight}`);
        }
    }
    function clearImg(imgId, phId, text) {
        const img = $(imgId), ph = $(phId);
        if (img) img.classList.remove('has-frame');
        if (ph) { ph.style.display = ''; if (text) ph.textContent = text; }
    }
    // Stopping the stream freezes the last frame in place rather than clearing
    // it: an operator wants to keep reading what is on the sample surface after
    // ending live view. Only fall back to the placeholder when no frame was
    // ever shown. The "LIVE" badge dropping (renderSubnavMeta) is the cue that
    // the frame is now static.
    function freezeImg(imgId, phId, text) {
        const img = $(imgId);
        if (img && img.classList.contains('has-frame')) return;
        clearImg(imgId, phId, text);
    }

    // Letterbox geometry for an object-fit: contain image, in CSS pixels.
    // Measured off the CANVAS, not its host: it is the element being drawn into,
    // and using one source for geometry and for the backing store keeps them
    // from disagreeing.
    function renderedRect() {
        const c = $('op-mark-canvas');
        if (!c) return null;
        const sb = c.getBoundingClientRect();
        if (!(sb.width > 0 && sb.height > 0)) return null;
        const f = frameOf(_lastBottom);
        const fw = f ? f.w : sb.width, fh = f ? f.h : sb.height;
        const ar = fw / fh, sar = sb.width / sb.height;
        let w, h;
        if (ar > sar) { w = sb.width; h = sb.width / ar; }
        else { h = sb.height; w = sb.height * ar; }
        return { x: (sb.width - w) / 2, y: (sb.height - h) / 2, w, h, fw, fh, sb };
    }
    function canvasCtx() {
        const c = $('op-mark-canvas');
        if (!c) return null;
        const r = c.getBoundingClientRect();
        if (!(r.width > 0 && r.height > 0)) return null;
        // The backing store must track the CSS box or everything drawn is
        // scaled — a stale height renders circles as ellipses. Scaling by dpr
        // keeps it crisp on fractional-ratio displays; the transform then lets
        // every drawing call stay in CSS pixels.
        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
        if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
        const ctx = c.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, r.width, r.height);
        return ctx;
    }
    // Project a stage-space marker onto the current frame, then onto the canvas.
    function markerToCanvas(m, r) {
        const f = frameOf(_lastBottom), cap = stageOf(_lastBottom);
        if (!f || !cap || !M) return null;
        const px = M.stageToFrame(m.stageX, m.stageY, f, cap);
        if (!px) return null;
        return { cx: r.x + (px[0] / r.fw) * r.w, cy: r.y + (px[1] / r.fh) * r.h, px };
    }
    // Registered embryos, projected onto the current frame. These are the
    // click-to-center targets; pending markers are a separate, editable set.
    function embryoPoints(r) {
        const f = frameOf(_lastBottom), cap = stageOf(_lastBottom);
        if (!f || !cap || !M) return [];
        const out = [];
        _embryos.forEach(emb => {
            const xy = resolveXY(emb);
            if (!xy) return;
            const px = M.stageToFrame(xy.x, xy.y, f, cap);
            if (!px) return;
            out.push({ emb, cx: r.x + (px[0] / r.fw) * r.w, cy: r.y + (px[1] / r.fh) * r.h });
        });
        return out;
    }

    function drawMarkers() {
        // Render dispatch, not gating: the canvas has no size while its pane is
        // hidden, so there is nothing to draw onto.
        if (_pane !== 'bottom') return;
        const ctx = canvasCtx();
        if (!ctx) return;
        const r = renderedRect();
        if (!r) return;

        // Registered embryos first, so pending markers draw over them.
        embryoPoints(r).forEach(({ emb, cx, cy }) => {
            const sel = emb.id === _selected;
            ctx.save();
            ctx.strokeStyle = isEngaged() ? '#7d8899' : (sel ? '#93c5fd' : '#60a5fa');
            ctx.fillStyle = ctx.strokeStyle;
            ctx.lineWidth = sel ? 2 : 1.2;
            if (isEngaged()) ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.arc(cx, cy, 13, 0, Math.PI * 2); ctx.stroke();
            ctx.setLineDash([]);
            ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fill();
            ctx.font = '600 11px Inter Tight, sans-serif';
            ctx.fillText(labelFor(emb), cx + 16, cy + 4);
            ctx.restore();
        });
        const locked = isEngaged();
        const colour = locked ? '#7d8899' : '#4ade80';
        _markers.forEach((m, i) => {
            const p = markerToCanvas(m, r);
            if (!p) return;
            const { cx, cy } = p;
            ctx.save();
            ctx.strokeStyle = colour;
            ctx.lineWidth = locked ? 1.2 : 2;
            if (locked) ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.arc(cx, cy, 11, 0, Math.PI * 2); ctx.stroke();
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
            ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
            ctx.stroke();
            ctx.fillStyle = colour;
            ctx.font = '600 11px Inter Tight, sans-serif';
            ctx.fillText(String(i + 1), cx + 13, cy - 8);
            ctx.restore();
        });
    }

    function onCanvasClick(e) {
        const r = renderedRect();
        const c = $('op-mark-canvas');
        if (!r || !c) return;
        const rect = c.getBoundingClientRect();
        const cxv = e.clientX - rect.left, cyv = e.clientY - rect.top;

        // Click a pending marker to remove it.
        for (let i = 0; i < _markers.length; i++) {
            const p = markerToCanvas(_markers[i], r);
            if (p && Math.hypot(cxv - p.cx, cyv - p.cy) <= MARK_HIT_PX) {
                // Editing the set invalidates the "N candidates added" note — it
                // was the detection-time count, not the current one.
                _markers.splice(i, 1); drawMarkers(); renderMarkCount(); setDetectNote(''); return;
            }
        }
        // Click a registered embryo to select it and centre the stage on it.
        // Always available — the interlock lives in moveStageTo, not here.
        for (const p of embryoPoints(r)) {
            if (Math.hypot(cxv - p.cx, cyv - p.cy) <= MARK_HIT_PX) {
                centerOnEmbryo(p.emb);
                return;
            }
        }
        if (cxv < r.x || cxv > r.x + r.w || cyv < r.y || cyv > r.y + r.h) return;

        const f = frameOf(_lastBottom), cap = stageOf(_lastBottom);
        if (!f) { toast('Start the camera first'); return; }
        // NEVER default the capture position to [0,0]: that silently converts
        // clicks into offsets from stage origin, so embryos land hundreds of µm
        // away and calibration images empty field. Refuse to mark instead.
        if (!cap) { toast('Stage position unknown — wait for the readout, then mark'); return; }

        const fx = ((cxv - r.x) / r.w) * r.fw, fy = ((cyv - r.y) / r.h) * r.fh;
        const s = M && M.frameToStage(fx, fy, f, cap);
        if (!s) { toast('Cannot place a marker without a stage position'); return; }
        _markers.push({ stageX: s[0], stageY: s[1], source: 'manual' });
        drawMarkers(); renderMarkCount();
    }

    async function centerOnEmbryo(emb) {
        const xy = resolveXY(emb);
        if (!xy) { toast('That embryo has no recorded position'); return; }
        selectEmbryo(emb.id);
        await moveStageTo(xy.x, xy.y, `Centred on embryo ${labelFor(emb)}`);
    }

    function renderMarkCount() {
        const n = _markers.length;
        const c = $('op-mark-count'); if (c) c.textContent = n;
        const ok = $('op-confirm'); if (ok) ok.disabled = n === 0;
        const cl = $('op-clear'); if (cl) cl.disabled = n === 0;
    }
    function setDetectNote(text) {
        const note = $('op-detect-note'); if (note) note.textContent = text || '';
    }

    // ══ BOTTOM PANE ═════════════════════════════════════════════════════════
    async function toggleBottomCam() {
        const b = $('op-cam-toggle'); if (b) b.disabled = true;
        try {
            const ep = _bottomOn ? '/api/devices/bottom_camera/stream/stop'
                : '/api/devices/bottom_camera/stream/start';
            const d = await postJSON(ep, {});
            applyBottomCam(!!d.streaming);
            _bottomWasOn = _bottomOn;
        } catch (e) { toast(`Camera toggle failed (${e.status || e.message})`); }
        finally { if (b) b.disabled = false; }
    }
    function applyBottomCam(on) {
        _bottomOn = on;
        const b = $('op-cam-toggle');
        if (b) { b.textContent = on ? 'Stop camera' : 'Start camera'; b.classList.toggle('is-on', on); }
        if (!on) freezeImg('op-img-bottom', 'op-ph-bottom', 'Camera off');
        renderSubnavMeta();
    }

    function setBusyText(t) {
        const el = document.querySelector('#op-busy-bottom .op-cam-busy-txt');
        if (el) el.textContent = t;
    }
    async function runDetect() {
        const b = $('op-detect');
        if (b) { b.disabled = true; b.textContent = 'Detecting…'; }
        const busy = $('op-busy-bottom'); if (busy) busy.hidden = false;
        const note = $('op-detect-note');
        // Detect on the frame already on screen when there is one — the operator
        // is looking at it, and re-capturing would disturb the LED/room light.
        const shown = $('op-img-bottom');
        let hasFrame = !!(shown && shown.classList.contains('has-frame'));
        try {
            // Phase 1 — when the viewport is empty, capture and SHOW the image
            // FIRST (no SAM yet), so the operator sees what detection will run on
            // before it runs, rather than the image appearing only at the end.
            if (!hasFrame) {
                setBusyText('Capturing…');
                const cap = await postJSON('/api/devices/detect_embryos', { capture_only: true });
                if (cap.frame && cap.frame.jpeg_b64) {
                    _lastBottom = cap.frame;
                    setImg('op-img-bottom', 'op-ph-bottom', cap.frame);
                    hasFrame = true;
                }
            }
            // Phase 2 — run SAM on the frame now on screen, then overlay results.
            setBusyText('Detecting…');
            const d = await postJSON('/api/devices/detect_embryos', { use_last_frame: hasFrame });
            if (d.frame && d.frame.jpeg_b64) {
                _lastBottom = d.frame;
                setImg('op-img-bottom', 'op-ph-bottom', d.frame);
            }
            const cands = Array.isArray(d.embryos) ? d.embryos : [];
            const f = frameOf(_lastBottom);
            const cap = d.stage_position || stageOf(_lastBottom);
            // A fresh detection REPLACES the previous auto-detected set rather
            // than piling onto it (re-running would otherwise double the marks).
            // Manual marks are kept — only 'sam' ones are cleared.
            _markers = _markers.filter(m => m.source !== 'sam');
            let added = 0;
            cands.forEach(c => {
                let sx = c.stage_x_um, sy = c.stage_y_um;
                if ((sx == null || sy == null) && f && cap && M && c.pixel_x != null && c.pixel_y != null) {
                    const s = M.frameToStage(c.pixel_x / f.downsample, c.pixel_y / f.downsample, f, cap);
                    if (s) { sx = s[0]; sy = s[1]; }
                }
                if (sx == null || sy == null) return;
                _markers.push({ stageX: sx, stageY: sy, source: 'sam' });
                added++;
            });
            drawMarkers(); renderMarkCount();
            if (note) note.textContent = `${added} candidate${added === 1 ? '' : 's'} added — edit them on the image, then register.`;
            toast(`Detected ${added} candidate${added === 1 ? '' : 's'}`);
        } catch (e) {
            if (e.status === 503) {
                if (note) note.textContent = 'Automatic detection is unavailable on this rig — mark by clicking the image.';
            } else {
                toast(`Detect failed (${e.status || e.message})`);
            }
        } finally {
            if (b) { b.disabled = false; b.textContent = 'Detect automatically'; }
            if (busy) busy.hidden = true;
        }
    }

    async function confirmMarks() {
        if (!_markers.length) return;
        const f = frameOf(_lastBottom), cap = stageOf(_lastBottom);
        if (!cap) { toast('Stage position unknown — cannot register markers'); return; }
        const b = $('op-confirm'); if (b) b.disabled = true;
        try {
            // The payload shape is a hard contract with _persist_detection_labels
            // (routes/data.py), which turns every confirm into training data for
            // the localiser that will replace SAM. Pixel coords are projected
            // from stage space against the frame being submitted.
            const markers = _markers.map(m => {
                const px = (f && M) ? M.stageToFrame(m.stageX, m.stageY, f, cap) : null;
                return {
                    stage_x_um: m.stageX, stage_y_um: m.stageY,
                    pixel_x: px ? px[0] : undefined, pixel_y: px ? px[1] : undefined,
                    source: m.source,
                };
            });
            const d = await postJSON('/api/devices/embryos/confirm', {
                markers,
                image_b64: _lastBottom ? _lastBottom.jpeg_b64 : undefined,
                frame: f ? { w: f.w, h: f.h, downsample: f.downsample } : undefined,
                stage_position: cap,
            });
            const n = (d.registered || []).length;
            _markers = [];
            drawMarkers(); renderMarkCount();
            setDetectNote(`Registered ${n} embryo${n === 1 ? '' : 's'} — they're in the roster and on the SPIM head.`);
            toast(`Registered ${n} embryo${n === 1 ? '' : 's'}`);
        } catch (e) {
            toast(`Register failed (${e.status || e.message})`);
            if (b) b.disabled = false;
        }
    }

    // ══ SPIM PANE ═══════════════════════════════════════════════════════════
    async function toggleSpim() {
        const b = $('op-spim-toggle'); if (b) b.disabled = true;
        try {
            const ep = _spimOn ? '/api/devices/lightsheet/live/stop' : '/api/devices/lightsheet/live/start';
            const d = await postJSON(ep, {});
            applySpim(!!d.streaming);
            _spimWasOn = _spimOn;
        } catch (e) { toast(`SPIM view toggle failed (${e.status || e.message})`); }
        finally { if (b) b.disabled = false; }
    }
    function applySpim(on) {
        _spimOn = on;
        const b = $('op-spim-toggle');
        if (b) { b.textContent = on ? 'Stop view' : 'Start view'; b.classList.toggle('is-on', on); }
        if (!on) freezeImg('op-img-spim', 'op-ph-spim', 'View off');
        renderSubnavMeta();
    }

    let _lsTimer = null;
    function postLsParams() {
        if (_lsTimer) clearTimeout(_lsTimer);
        _lsTimer = setTimeout(() => {
            postJSON('/api/devices/lightsheet/live/params',
                { galvo: _galvo, piezo: _piezo, exposure: 20, side: 'A' }).catch(() => {});
        }, 120);
    }
    function nudgeGalvo(d) {
        _galvo = Math.max(-5, Math.min(5, _galvo + d));
        const el = $('op-gv'); if (el) el.textContent = _galvo.toFixed(1);
        postLsParams();
    }
    function nudgePiezo(d) {
        _piezo = Math.max(0, Math.min(200, _piezo + d));
        const el = $('op-pz'); if (el) el.textContent = _piezo.toFixed(0);
        postLsParams();
    }
    async function toggleLed() {
        _ledOn = !_ledOn;
        applyLed();
        try { await postJSON('/api/devices/led/set', { state: _ledOn ? 'Open' : 'Closed' }); }
        catch (e) { toast(`LED failed (${e.status || e.message})`); }
    }
    function applyLed() {
        const b = $('op-led');
        if (b) {
            b.setAttribute('aria-pressed', _ledOn ? 'true' : 'false');
            b.classList.toggle('is-emitting', _ledOn);
        }
        renderSubnavMeta();
    }
    async function forceLedOff() {
        if (!_ledOn) return;
        _ledOn = false; applyLed();
        try { await postJSON('/api/devices/led/set', { state: 'Closed' }); } catch (_) {}
    }

    async function calibrateSelected() {
        if (!_selected) { toast('Select an embryo first'); return; }
        const b = $('op-calibrate'), out = $('op-cal-result');
        if (b) { b.disabled = true; b.textContent = 'Calibrating…'; }
        if (out) out.textContent = 'sweeping…';
        try {
            const d = await postJSON(`/api/devices/embryos/${_selected}/calibrate`, {});
            const cal = d.calibration || {};
            const slope = cal.slope_um_per_deg, r2 = cal.r_squared;
            if (out) {
                out.textContent = (slope != null)
                    ? `${Number(slope).toFixed(1)} µm/deg${r2 != null ? ` · R² ${Number(r2).toFixed(2)}` : ''}`
                    : 'done';
            }
        } catch (e) {
            if (out) out.textContent = 'failed';
            toast(`Calibrate failed (${e.status || e.message})`);
        } finally { if (b) { b.disabled = false; b.textContent = 'Calibrate piezo–galvo'; } }
    }

    function renderSpimTarget() {
        const el = $('op-spim-target');
        if (!el) return;
        const emb = _embryos.find(e => e.id === _selected);
        el.textContent = emb ? `Selected: embryo ${labelFor(emb)}` : 'No embryo selected';
    }

    // ══ ACQUISITION PANE ════════════════════════════════════════════════════
    function selectEmbryo(id) {
        _selected = id;
        renderRoster(); renderSpimTarget(); renderSingle(); renderEmbryoRail();
    }

    // Shared embryo list, left of every instrument surface. Reads the canonical
    // _embryos (bootstrapped from /api/embryos/current, kept live by
    // EMBRYOS_UPDATE), so it is the same set on Bottom / SPIM / Acquire and it
    // survives a refresh.
    function renderEmbryoRail() {
        const host = $('op-erail-list');
        const count = $('op-erail-count');
        if (count) count.textContent = _embryos.length;
        if (!host) return;
        host.innerHTML = '';
        if (!_embryos.length) {
            const box = document.createElement('div');
            box.className = 'op-erail-empty';
            box.textContent = 'No embryos yet — detect on the bottom camera, then register.';
            host.appendChild(box);
            return;
        }
        _embryos.forEach(emb => {
            const xy = resolveXY(emb);
            const row = document.createElement('div');
            row.className = 'op-erow' + (emb.id === _selected ? ' is-sel' : '');
            row.tabIndex = 0;
            row.dataset.embryo = emb.id;
            row.innerHTML =
                '<span class="op-erow-main">' +
                `<span class="op-erow-label">Embryo ${escapeHtml(labelFor(emb))}</span>` +
                `<span class="op-erow-xy">${xy ? `${xy.x.toFixed(0)}, ${xy.y.toFixed(0)}` : '—'}</span>` +
                '</span>' +
                `<button class="op-erow-del" type="button" title="Remove this embryo (false positive)" ` +
                `data-del="${escapeHtml(emb.id)}">×</button>`;
            host.appendChild(row);
        });
    }

    async function deleteEmbryo(id) {
        try {
            const res = await fetch(`/api/embryos/${encodeURIComponent(id)}`, { method: 'DELETE' });
            if (!res.ok) throw Object.assign(new Error('delete failed'), { status: res.status });
            // EMBRYOS_UPDATE will reconcile every view; prune optimistically so
            // the row disappears immediately even before the event lands.
            _embryos = _embryos.filter(e => e.id !== id);
            if (_selected === id) _selected = _embryos.length ? _embryos[0].id : null;
            renderEmbryoRail(); renderRoster(); renderSpimTarget(); renderSingle(); drawMarkers();
        } catch (e) {
            toast(`Delete failed (${e.status || e.message})`);
        }
    }

    function renderRoster() {
        const host = $('op-roster');
        const count = $('op-roster-count');
        if (count) count.textContent = _embryos.length;
        if (!host) return;
        host.innerHTML = '';
        if (!_embryos.length) {
            const box = document.createElement('div');
            box.className = 'op-empty';
            box.innerHTML = 'No embryos marked yet.' +
                '<button class="op-btn" type="button" data-goto="bottom">Go to Bottom cam</button>';
            host.appendChild(box);
            return;
        }
        _embryos.forEach(emb => {
            const xy = resolveXY(emb);
            const role = (emb.role && emb.role !== 'unassigned') ? emb.role : 'test';
            const row = document.createElement('div');
            row.className = 'op-rrow' + (emb.id === _selected ? ' is-sel' : '');
            row.tabIndex = 0;
            row.dataset.embryo = emb.id;
            row.innerHTML =
                `<span class="op-rlabel">Embryo ${escapeHtml(labelFor(emb))}</span>` +
                `<span class="op-rxy">${xy ? `${xy.x.toFixed(0)}, ${xy.y.toFixed(0)}` : '—'}</span>` +
                `<button class="op-rrole${role === 'calibration' ? ' is-reference' : ''}" type="button" ` +
                `data-role-for="${escapeHtml(emb.id)}">${role === 'calibration' ? 'ref' : 'subj'}</button>` +
                `<button class="op-rcenter" type="button" title="Centre the stage on this embryo" ` +
                `data-center="${escapeHtml(emb.id)}">Centre</button>`;
            host.appendChild(row);
        });
    }

    // Roles are read from the canonical embryo list and written through the
    // endpoint — deliberately NOT mirrored in a local map, which in the old
    // design drifted from _embryos and needed a reconciliation loop.
    async function toggleRole(id) {
        const emb = _embryos.find(e => e.id === id);
        if (!emb) return;
        const cur = (emb.role && emb.role !== 'unassigned') ? emb.role : 'test';
        const next = cur === 'calibration' ? 'test' : 'calibration';
        emb.role = next;
        renderRoster();
        const roles = {};
        _embryos.forEach(e => { roles[e.id] = (e.role && e.role !== 'unassigned') ? e.role : 'test'; });
        try { await postJSON('/api/embryos/roles', { roles }); }
        catch (e) { toast(`Roles failed (${e.status || e.message})`); }
    }

    function setMode(m) {
        _mode = m;
        document.querySelectorAll('#op-modes [data-mode]').forEach(b =>
            b.classList.toggle('is-on', b.dataset.mode === m));
        ['single', 'adaptive', 'library', 'agent'].forEach(k => {
            const p = $(`op-panel-${k}`);
            if (p) p.hidden = k !== m;
        });
        if (m === 'library') loadLibrary();
        renderSingle();
    }

    function renderSingle() {
        const t = $('op-single-target'), d = $('op-single-delta');
        const emb = _embryos.find(e => e.id === _selected);
        if (t) t.textContent = emb ? `embryo ${labelFor(emb)}` : 'none selected';
        if (!d) return;
        const xy = emb ? resolveXY(emb) : null;
        // A fact, not a gate: the operator is told how far off the stage is and
        // decides for themselves. Acquire is never disabled on this.
        if (!xy || !_xy) { d.textContent = '—'; return; }
        const dx = xy.x - _xy.x, dy = xy.y - _xy.y;
        d.textContent = `${Math.round(Math.hypot(dx, dy))} µm away`;
    }

    async function loadLibrary() {
        const host = $('op-lib-list');
        if (!host) return;
        try {
            const d = await getJSON('/api/tactic_library');
            const items = (d && d.tactics) || [];
            if (!items.length) { host.innerHTML = '<div class="op-empty">No saved tactics</div>'; return; }
            host.innerHTML = items.map(t =>
                `<button class="op-libitem${t.id === _selectedLib ? ' is-sel' : ''}" type="button" ` +
                `data-lib="${escapeHtml(t.id)}">${escapeHtml(t.name || t.id)}` +
                `<small>${escapeHtml(t.kind || '')}</small></button>`).join('');
        } catch (_) { host.innerHTML = '<div class="op-empty">Library unavailable</div>'; }
    }

    function subjectIds() {
        const subs = _embryos.filter(e => e.role !== 'calibration').map(e => e.id);
        return subs.length ? subs : _embryos.map(e => e.id);
    }

    async function startRun() {
        const b = $('op-run-start');
        const done = () => { if (b) { b.disabled = false; b.textContent = 'Start'; } };
        if (b) { b.disabled = true; b.textContent = 'Starting…'; }
        try {
            if (_mode === 'single') {
                if (!_selected) { toast('Select an embryo first'); return; }
                _acquiring = true; renderSubnavMeta();
                try {
                    await postJSON('/api/devices/acquire/volume', {
                        num_slices: Math.max(1, Number(($('op-vol-slices') || {}).value) || 50),
                        exposure_ms: Math.max(1, Number(($('op-vol-exp') || {}).value) || 10),
                    });
                    toast('Volume acquired');
                } finally { _acquiring = false; await forceLedOff(); renderSubnavMeta(); }
                return;
            }
            if (_mode === 'adaptive') {
                const interval = Math.max(1, Number(($('op-tl-interval') || {}).value) || 120);
                const sel = ($('op-tl-stop') || {}).value || 'manual';
                const val = Math.max(1, Number(($('op-tl-condval') || {}).value) || 1);
                // The orchestrator parses the COMBINED form ('timepoints:N' /
                // 'duration:Xh'). A bare 'timepoints' silently degrades to manual,
                // i.e. a timelapse that never stops.
                let stop_condition = sel;
                if (sel === 'timepoints') stop_condition = `timepoints:${val}`;
                else if (sel === 'duration') stop_condition = `duration:${val}h`;
                await postJSON('/api/devices/timelapse/start', {
                    embryo_ids: subjectIds(),
                    interval_seconds: interval,
                    stop_condition,
                    monitoring_mode: ($('op-tl-monitor') || {}).value || 'idle',
                });
                toast('Adaptive timelapse started');
                renderRun();
                return;
            }
            if (_mode === 'library') {
                if (!_selectedLib) { toast('Pick a saved tactic'); return; }
                const d = await postJSON('/api/operate/run-tactic',
                    { library_id: _selectedLib, embryo_ids: subjectIds() });
                if (d.success) { toast('Tactic started'); renderRun(); }
                else toast(`Run failed: ${(d.result && d.result.message) || '?'}`);
                return;
            }
            if (_mode === 'agent') {
                const roster = _embryos.map(e => {
                    const xy = resolveXY(e);
                    const r = e.role === 'calibration' ? 'reference' : 'subject';
                    return `${labelFor(e)}${xy ? ` (${xy.x.toFixed(0)},${xy.y.toFixed(0)})` : ''} [${r}]`;
                }).join(', ');
                const prompt = `I marked ${_embryos.length} embryos: ${roster}. ` +
                    'Propose and start an Operation Plan to image them.';
                if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
                    AgentChat.togglePanel(true);
                    if (AgentChat.runCommand) setTimeout(() => AgentChat.runCommand(prompt), 300);
                } else toast('Agent chat unavailable');
            }
        } catch (e) {
            toast(`Start failed (${e.status || e.message})`);
        } finally { done(); }
    }

    // Run presence is DERIVED from the server, not from a client flag. The old
    // design kept it in memory, so F5 during a running timelapse lost the whole
    // panel — and left a client state machine that could re-grow into steps.
    async function renderRun() {
        const host = $('op-runspine'), actions = $('op-run-actions');
        if (!host) return;
        let tactics = [];
        try {
            const d = await getJSON('/api/operation_plan');
            tactics = (d && d.plan && d.plan.tactics) || [];
        } catch (_) { /* leave empty */ }
        const live = tactics.filter(t => t.state === 'active' || t.state === 'paused');
        if (actions) actions.hidden = live.length === 0;
        if (!tactics.length) {
            host.innerHTML = '<div class="op-empty">Nothing running.</div>';
            return;
        }
        host.innerHTML = tactics.map(tacticCard).join('');
        _runPaused = live.some(t => t.state === 'paused');
        const p = $('op-run-pause');
        if (p) p.textContent = _runPaused ? 'Resume' : 'Pause';
    }
    function tacticCard(t) {
        const state = t.state || 'planned';
        const struct = t.structure || {};
        const meta = [];
        if (struct.cadence_s != null) meta.push(`${struct.cadence_s}s`);
        if (struct.interval != null) meta.push(`${struct.interval}s`);
        if (struct.status) meta.push(struct.status);
        if (t.live && t.live.signal != null) meta.push(`signal ${t.live.signal}`);
        return `<div class="op-tcard st-${escapeHtml(state)}">` +
            `<div class="op-tcard-head"><span class="op-tcard-name">${escapeHtml(t.name || t.id)}</span>` +
            `<span class="op-tcard-state">${escapeHtml(state)}</span></div>` +
            `<div class="op-tcard-kind">${escapeHtml(t.kind || '')}</div>` +
            (meta.length ? `<div class="op-tcard-meta">${escapeHtml(meta.join(' · '))}</div>` : '') +
            (t.rationale ? `<div class="op-tcard-meta">${escapeHtml(t.rationale)}</div>` : '') +
            '</div>';
    }
    async function pauseRun() {
        try {
            await postJSON(_runPaused ? '/api/devices/timelapse/resume' : '/api/devices/timelapse/pause', {});
            toast(_runPaused ? 'Resumed' : 'Paused');
        } catch (e) { toast(`Pause/resume failed (${e.status || e.message})`); }
        renderRun();
    }
    async function stopRun() {
        if (!window.confirm('Stop the run?')) return;
        try { await postJSON('/api/devices/timelapse/stop', { reason: 'operator' }); toast('Run stopped'); }
        catch (e) { toast(`Stop failed (${e.status || e.message})`); }
        renderRun();
    }

    // ══ PANES ═══════════════════════════════════════════════════════════════
    // Camera ownership is keyed on VISIBILITY, not on a step. Both cameras
    // contend for MMCore, and the client swaps .src per frame with no throttle,
    // so two live decoders is the condition that risks a Video-TDR freeze. "The
    // camera is live while you are looking at it" guarantees at most one.
    const PANES = {
        bottom: {
            onEnter() { if (_bottomWasOn && !_bottomOn) toggleBottomCam(); drawMarkers(); },
            onLeave() { _bottomWasOn = _bottomOn; if (_bottomOn) stopBottom(); },
            render() { renderMarkCount(); drawMarkers(); bz.render(); },
        },
        spim: {
            onEnter() { if (_spimWasOn && !_spimOn) toggleSpim(); },
            onLeave() { _spimWasOn = _spimOn; if (_spimOn) stopSpim(); forceLedOff(); },
            render() { renderSpimTarget(); fd.render(); },
        },
        acquire: {
            onEnter() { renderRun(); },
            onLeave() {},
            render() { renderRoster(); renderSingle(); },
        },
    };
    function stopBottom() {
        fetch('/api/devices/bottom_camera/stream/stop', { method: 'POST' }).catch(() => {});
        applyBottomCam(false);
    }
    function stopSpim() {
        fetch('/api/devices/lightsheet/live/stop', { method: 'POST' }).catch(() => {});
        applySpim(false);
    }

    function showPane(name) {
        if (!PANES[name] || name === _pane) return;
        const prev = _pane;
        _pane = name;
        if (PANES[prev]) PANES[prev].onLeave();
        ['bottom', 'spim', 'acquire'].forEach(p => {
            const el = $(`op-pane-${p}`);
            if (el) el.hidden = p !== name;
        });
        // Drives the CSS that hides the shared rail on Acquisition (its own
        // roster is richer) while keeping it on Bottom / SPIM.
        const body = $('op-body'); if (body) body.dataset.pane = name;
        if (typeof updateViewButtons === 'function') updateViewButtons('operate-subtab-switcher', name);
        PANES[name].onEnter();
        PANES[name].render();
        renderEmbryoRail();
        renderLock();
    }

    function renderSubnavMeta() {
        const el = $('op-subnav-meta');
        if (!el) return;
        const bits = [];
        if (_bottomOn) bits.push('BOTTOM ● LIVE');
        if (_spimOn) bits.push('SPIM ● LIVE');
        if (_ledOn) bits.push('LED EMITTING');
        if (_acquiring) bits.push('LASER EMITTING');
        el.textContent = bits.join('  ·  ');
    }

    // ══ EVENTS ══════════════════════════════════════════════════════════════
    function onBottomFrame(p) {
        // Bail when hidden, or a hidden Operate keeps base64-decoding every
        // frame behind whatever is on screen and races for stream ownership.
        if (!_active || _pane !== 'bottom' || !p || !p.jpeg_b64) return;
        _lastBottom = p;
        if (p.focus_score != null) {
            const el = $('op-bz-score');
            if (el) el.textContent = Number(p.focus_score).toFixed(3);
        }
        setImg('op-img-bottom', 'op-ph-bottom', p);
        drawMarkers();
    }
    function onSpimFrame(p) {
        if (!_active || _pane !== 'spim' || !p || !p.jpeg_b64) return;
        if (p.focus_score != null) {
            const el = $('op-spim-score');
            if (el) el.textContent = Number(p.focus_score).toFixed(3);
        }
        setImg('op-img-spim', 'op-ph-spim', p);
    }
    function onEmbryosUpdate(p) {
        _embryos = (p && Array.isArray(p.embryos)) ? p.embryos : [];
        if (_selected && !_embryos.some(e => e.id === _selected)) _selected = null;
        // The embryo list is shared across all three panes; keep a live
        // selection whenever it is non-empty so SPIM/Acquire aren't a dead-end
        // ("No embryo selected") right after registering. The operator can still
        // switch by clicking a registered embryo (bottom) or a roster row.
        if (!_selected && _embryos.length) _selected = _embryos[0].id;
        // Render the shared rail even when the Operate view isn't the active tab,
        // so switching to it (or refreshing) shows the list immediately rather
        // than waiting for the next mutation event.
        renderEmbryoRail();
        if (!_active) return;
        renderRoster(); renderSpimTarget(); renderSingle();
    }

    function wire() {
        if (_wired) return;
        _wired = true;

        if (typeof initViewSwitcher === 'function') {
            // Click delegation only — deliberately NO `views` option.
            //
            // initViewSwitcher's `views` binds BARE number keys on document, and
            // 1-6 are already global main-tab navigation (KeyboardShortcuts in
            // app.js), with system/calibration/plan switchers claiming 1-3 too.
            // Binding them here either steals a main-tab key or, with a guard
            // tight enough to be safe, never fires at all — verified in-browser:
            // app.js handles the keypress first and moves state.tab, so the
            // guard then correctly refuses. Three peer surfaces are fine to
            // click between; a shortcut would need a modifier to be honest.
            initViewSwitcher('operate-subtab-switcher', showPane);
        }

        bz.wire(); fd.wire();

        const cam = $('op-cam-toggle'); if (cam) cam.addEventListener('click', toggleBottomCam);
        const det = $('op-detect'); if (det) det.addEventListener('click', runDetect);
        const ok = $('op-confirm'); if (ok) ok.addEventListener('click', confirmMarks);
        const cl = $('op-clear');
        if (cl) cl.addEventListener('click', () => { _markers = []; drawMarkers(); renderMarkCount(); setDetectNote(''); });
        const canvas = $('op-mark-canvas'); if (canvas) canvas.addEventListener('click', onCanvasClick);

        // Shared embryo rail: click a row to select (delete button is guarded
        // first so removing a false positive doesn't also select it).
        const erail = $('op-erail-list');
        if (erail) {
            erail.addEventListener('click', (e) => {
                const del = e.target.closest('[data-del]');
                if (del) { e.stopPropagation(); deleteEmbryo(del.dataset.del); return; }
                const row = e.target.closest('[data-embryo]');
                if (row) selectEmbryo(row.dataset.embryo);
            });
            erail.addEventListener('keydown', (e) => {
                if (e.key !== 'Enter' && e.key !== ' ') return;
                const row = e.target.closest('[data-embryo]');
                if (row) { e.preventDefault(); selectEmbryo(row.dataset.embryo); }
            });
        }

        const sp = $('op-spim-toggle'); if (sp) sp.addEventListener('click', toggleSpim);
        const led = $('op-led'); if (led) led.addEventListener('click', toggleLed);
        const cal = $('op-calibrate'); if (cal) cal.addEventListener('click', calibrateSelected);
        document.querySelectorAll('[data-gv]').forEach(b =>
            b.addEventListener('click', () => nudgeGalvo(Number(b.dataset.gv))));
        document.querySelectorAll('[data-pz]').forEach(b =>
            b.addEventListener('click', () => nudgePiezo(Number(b.dataset.pz))));
        document.querySelectorAll('[data-backoff]').forEach(b =>
            b.addEventListener('click', backOff));

        const modes = $('op-modes');
        if (modes) {
            modes.addEventListener('click', e => {
                const b = e.target.closest('[data-mode]');
                if (b) setMode(b.dataset.mode);
            });
        }
        const stop = $('op-tl-stop');
        if (stop) {
            stop.addEventListener('change', () => {
                const w = $('op-tl-condwrap');
                if (w) w.hidden = stop.value === 'manual';
            });
        }
        const lib = $('op-lib-list');
        if (lib) {
            lib.addEventListener('click', e => {
                const b = e.target.closest('[data-lib]');
                if (b) { _selectedLib = b.dataset.lib; loadLibrary(); }
            });
        }
        const roster = $('op-roster');
        if (roster) {
            roster.addEventListener('click', e => {
                const r = e.target.closest('[data-role-for]');
                if (r) { e.stopPropagation(); toggleRole(r.dataset.roleFor); return; }
                const c = e.target.closest('[data-center]');
                if (c) {
                    e.stopPropagation();
                    const emb = _embryos.find(x => x.id === c.dataset.center);
                    if (emb) centerOnEmbryo(emb);
                    return;
                }
                const g = e.target.closest('[data-goto]');
                if (g) { showPane(g.dataset.goto); return; }
                const row = e.target.closest('[data-embryo]');
                if (row) selectEmbryo(row.dataset.embryo);
            });
            roster.addEventListener('keydown', e => {
                if (e.key !== 'Enter' && e.key !== ' ') return;
                const row = e.target.closest('[data-embryo]');
                if (row) { e.preventDefault(); selectEmbryo(row.dataset.embryo); }
            });
        }
        const start = $('op-run-start'); if (start) start.addEventListener('click', startRun);
        const pause = $('op-run-pause'); if (pause) pause.addEventListener('click', pauseRun);
        const stopb = $('op-run-stop'); if (stopb) stopb.addEventListener('click', stopRun);

        window.addEventListener('resize', () => { if (_active && _pane === 'bottom') drawMarkers(); });
        // The viewport also changes size without a window resize — revealing the
        // pane, the agent panel opening, the first frame arriving. Without this
        // the overlay keeps whatever size it was first drawn at and every marker
        // sits in the wrong place until the next frame happens to redraw it.
        const camBox = $('op-cam-bottom');
        if (camBox && typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(() => { if (_active && _pane === 'bottom') drawMarkers(); }).observe(camBox);
        }

        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('BOTTOM_CAMERA_FRAME', onBottomFrame);
            ClientEventBus.on('LIGHTSHEET_FRAME', onSpimFrame);
            ClientEventBus.on('EMBRYOS_UPDATE', onEmbryosUpdate);
            ClientEventBus.on('DEVICE_STATE_UPDATE', p => {
                const pos = p && p.positions;
                if (!pos) return;
                if (Array.isArray(pos.xy_stage)) _xy = { x: pos.xy_stage[0], y: pos.xy_stage[1] };
                if (!_active) return;
                for (const v of Object.values(pos)) {
                    if (!v || typeof v !== 'object' || v.Position == null) continue;
                    const val = Number(v.Position);
                    if (!Number.isFinite(val)) continue;
                    if (v.kind === 'fdrive') fd.absorbTelemetry(val);
                    else if (v.kind === 'bottom_z') bz.absorbTelemetry(val);
                }
                if (_pane === 'acquire') renderSingle();
            });
        }
    }

    async function activate() {
        wire();
        if (_active) return;
        _active = true;
        showPaneInitial();
        try { onEmbryosUpdate(await getJSON('/api/embryos/current')); } catch (_) {}
        await Promise.all([bz.refresh(), fd.refresh()]);
        renderLock();
        renderSubnavMeta();
    }
    function showPaneInitial() {
        ['bottom', 'spim', 'acquire'].forEach(p => {
            const el = $(`op-pane-${p}`);
            if (el) el.hidden = p !== _pane;
        });
        if (typeof updateViewButtons === 'function') updateViewButtons('operate-subtab-switcher', _pane);
        if (PANES[_pane]) { PANES[_pane].onEnter(); PANES[_pane].render(); }
    }
    function deactivate() {
        if (!_active) return;
        _active = false;
        // Remember what was running so returning restores it, but leave nothing
        // decoding behind a hidden view.
        _bottomWasOn = _bottomOn; _spimWasOn = _spimOn;
        if (_bottomOn) stopBottom();
        if (_spimOn) stopSpim();
        forceLedOff();
    }

    return { activate, deactivate };
})();
