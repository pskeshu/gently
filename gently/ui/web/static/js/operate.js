/**
 * Operate view — "The Operator Spine".
 *
 * One guided surface for the bottom-cam → SPIM workflow. A single state driver
 * (`renderStep`) reads the selected embryo + its progress and sets EVERYTHING in
 * lockstep: the header stepper ("you are here"), the always-on safety status
 * strip, the left worklist + dish mini-map, the one live viewport (its camera
 * source + on-image overlays), and which single control group the right rail
 * shows. Exactly one camera and one step are live at a time.
 *
 * Phase A (no embryo selected): A1 focus bottom objective → A2 mark all (one
 * frozen FOV, positions only) → Confirm into the canonical list.
 * Phase B (per selected embryo): B1 center → B2 lower SPIM head (fenced, floor) →
 * B3 focus SPIM (LED) → B4 acquire → B5 retract & advance.
 *
 * Safety: manual fenced nudges only (no autofocus); F-drive floor honored and
 * down-nudges auto-grey near it; XY centering blocked while the head is lowered;
 * 'focused' is earned at B3 (never on a stray F-drive nudge); LED is force-closed
 * on step-leave and view-leave.
 */
const OperateManager = (function () {
    const BASE_UM_PER_PX = 6.5 / 10.0;   // pixel_size_um / objective_mag
    const MARK_HIT_PX = 14;
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const ROLE_NEUTRAL = '#8a8f98';
    const STATE_RANK = { marked: 0, centered: 1, lowering: 2, focused: 3, calibrated: 4, imaged: 5 };

    let _wired = false, _active = false;

    // workflow state
    let _selected = null;            // embryo id, or null = Phase A (survey)
    let _step = null;                // explicit B-step for the selected embryo
    let _marking = false;
    const _states = {};              // id -> marked|centered|lowering|focused|imaged
    let _embryos = [];               // canonical EMBRYOS_UPDATE list
    let _headLowered = false;        // global: is the SPIM head below safe travel?
    let _acquiring = false;

    // viewport / streams
    let _camOn = false, _spimOn = false, _camStarting = false;
    let _lastFrame = null;           // last live bottom-cam payload
    let _lastSpimFrame = null;
    let _bzScore = null, _spimScore = null;

    // marking
    let _frozenSrc = null, _frozenFrame = null, _captureStage = [0, 0], _markers = [];

    // device telemetry
    let _lastXY = null;              // {X,Y}
    let _fdPos = null, _fdFloor = null;
    let _galvo = 0.0, _piezo = 50.0, _ledOn = false;

    // Phase C "Run": _runState null = not in Run; 'choose' = chooser; 'running' = live.
    // Every Run mode emits one tactic scoped to the marked set. _roles maps each
    // embryo to a role ('test'=subject default, 'calibration'=reference).
    let _runState = null;
    let _runMode = 'adaptive';
    const _roles = {};
    let _runMeta = null;             // summary of the active run (for the run-spine)
    let _runPlan = null;             // operation plan {tactics:[]} for the run-spine
    let _runPaused = false;
    let _selectedLib = null, _selectedPlan = null;

    // DOM (cached on wire)
    const D = {};

    function $(id) { return document.getElementById(id); }
    function toast(m) { if (typeof showGentlyToast === 'function') showGentlyToast(m); }

    async function postJSON(url, body) {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
        if (!res.ok) { const t = await res.text().catch(() => ''); const e = new Error(`${res.status} ${t}`); e.status = res.status; throw e; }
        return res.json().catch(() => ({}));
    }
    async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(String(r.status)); return r.json(); }

    function cacheDom() {
        [
            'op-stepper', 'op-loop-emb', 'op-st-head', 'op-st-floor', 'op-st-led', 'op-st-led-wrap',
            'op-st-laser', 'op-st-laser-wrap', 'op-st-cam', 'op-minimap', 'op-board', 'op-board-count',
            'op-survey-btn', 'op-badge', 'op-cam-stage', 'op-cam-img', 'op-mark-canvas', 'op-cam-ph',
            'op-rail', 'op-rail-head', 'op-cam-toggle', 'op-bz-pos', 'op-bz-score', 'op-bz-nudge',
            'op-tomark', 'op-detect', 'op-confirm', 'op-mark-count', 'op-clear', 'op-center', 'op-center-hint',
            'op-fd-pos', 'op-fd-floor', 'op-fd-nudge', 'op-fd-d100', 'op-fd-d10', 'op-tofocus',
            'op-spim-toggle', 'op-led', 'op-gv', 'op-pz', 'op-spim-score', 'op-infocus',
            'op-calibrate', 'op-cal-result', 'op-cal-skip', 'op-acquire', 'op-retract',
            'op-rolechips', 'op-modes', 'op-panel-adaptive', 'op-panel-library', 'op-panel-plan', 'op-panel-agent',
            'op-tl-interval', 'op-tl-stop', 'op-tl-condval', 'op-tl-monitor', 'op-lib-list', 'op-plan-list',
            'op-run-start', 'op-runspine', 'op-run-pause', 'op-run-stop', 'op-run-open',
        ].forEach(id => { D[id] = $(id); });
    }

    // ── step model ───────────────────────────────────────────────────────
    function stepForState(st) {
        return { marked: 'b1', centered: 'b2', lowering: 'b2', focused: 'bc', calibrated: 'b4', imaged: 'b5' }[st] || 'b1';
    }
    function effectiveStep() {
        if (_marking) return 'a2';
        if (_runState === 'running') return 'running';
        if (_runState === 'choose') return 'c0';
        if (_selected) return _step || 'b1';
        return 'a1';
    }
    function cameraForStep(step) {
        if (step === 'a1' || step === 'a2' || step === 'b1') return 'bottom';
        if (step === 'b2' || step === 'b3' || step === 'bc' || step === 'b4' || step === 'b5') return 'spim';
        return 'none';  // c0 / running are non-camera (the dish map carries context)
    }

    const RAIL_HEADS = {
        a1: 'Focus the bottom objective', a2: 'Mark all embryos',
        b1: 'Center the embryo', b2: 'Lower the SPIM head', b3: 'Focus the SPIM objective',
        bc: 'Calibrate piezo-galvo', b4: 'Acquire the volume', b5: 'Retract & advance',
        c0: 'Run — choose how to image', running: 'Run — live',
    };
    const STEP_NODE = { a1: 'a1', a2: 'a2', b1: 'b1', b2: 'b2', b3: 'b3', bc: 'bc', b4: 'b4', b5: 'b4', c0: 'run', running: 'run' };

    function setStep(s) { _step = s; renderStep(); }

    // The single driver — every state change routes through here.
    function renderStep() {
        const step = effectiveStep();
        const cam = cameraForStep(step);

        // rail group + head
        if (D['op-rail']) D['op-rail'].dataset.active = step;
        if (D['op-rail-head']) {
            const emb = _embryos.find(e => e.id === _selected);
            D['op-rail-head'].textContent = emb
                ? `Embryo ${labelFor(emb)} · ${RAIL_HEADS[step]}`
                : RAIL_HEADS[step];
        }

        // stepper nodes (done/active/locked) driven by selected embryo's state
        const st = _selected ? (_states[_selected] || 'marked') : null;
        const rank = st ? STATE_RANK[st] : -1;
        const order = { b1: 0, b2: 1, b3: 2, bc: 3, b4: 4 };
        D['op-stepper'].querySelectorAll('.op-node').forEach(n => {
            const node = n.dataset.node;
            n.classList.remove('is-active', 'is-done', 'is-locked');
            const active = STEP_NODE[step] === node;
            if (node === 'a1' || node === 'a2') {
                if (active) n.classList.add('is-active');
                else if (_embryos.length) n.classList.add('is-done');
            } else if (node === 'run') {
                if (active) n.classList.add('is-active');
                else if (_embryos.length && _selected) n.classList.add('is-done');
                else if (!_embryos.length) n.classList.add('is-locked');
            } else {
                const oi = order[node];
                if (active) n.classList.add('is-active');
                else if (rank > oi) n.classList.add('is-done');
                else if (!_selected || rank < oi) n.classList.add('is-locked');
            }
        });
        if (D['op-loop-emb']) {
            const emb = _embryos.find(e => e.id === _selected);
            const idx = emb ? _embryos.indexOf(emb) + 1 : 0;
            D['op-loop-emb'].textContent = emb ? ` · ${idx}/${_embryos.length}` : '';
        }

        // viewport: badge + stop the inactive camera
        if (D['op-badge']) {
            D['op-badge'].textContent = cam === 'none'
                ? RAIL_HEADS[step]
                : (cam === 'bottom' ? 'Bottom cam' : 'SPIM · side A') + ' — ' + RAIL_HEADS[step].toLowerCase();
        }
        ensureInactiveCameraStopped(cam);
        // B1 (Center) needs a live bottom view but has no Start button (the
        // chooser stops all cameras) — auto-start it so centering has feedback.
        if (step === 'b1' && !_camOn && !_camStarting && !_marking) {
            _camStarting = true;
            fetch('/api/devices/bottom_camera/stream/start', { method: 'POST' })
                .then(r => r.json()).then(d => { _camStarting = false; applyCam(!!d.streaming); })
                .catch(() => { _camStarting = false; });
        }
        // show the right frame
        if (cam === 'bottom' && !_marking && _lastFrame) setImg(_lastFrame);
        else if (cam === 'spim' && _lastSpimFrame) setImg(_lastSpimFrame);
        else if (cam === 'spim' && !_spimOn) { D['op-cam-img'].classList.remove('has-frame'); D['op-cam-ph'].style.display = ''; D['op-cam-ph'].textContent = 'SPIM view off'; }
        else if (cam === 'bottom' && !_camOn && !_marking) { D['op-cam-img'].classList.remove('has-frame'); D['op-cam-ph'].style.display = ''; D['op-cam-ph'].textContent = 'Camera off'; }
        else if (cam === 'none') {
            D['op-cam-img'].classList.remove('has-frame'); D['op-cam-ph'].style.display = '';
            D['op-cam-ph'].textContent = step === 'running'
                ? `Run live — ${_embryos.length} embryo${_embryos.length === 1 ? '' : 's'} (see the dish map)`
                : `${_embryos.length} embryo${_embryos.length === 1 ? '' : 's'} marked — choose how to image`;
        }

        // gating
        if (D['op-center']) {
            if (_headLowered) { D['op-center'].textContent = 'Retract head first'; D['op-center'].disabled = true; }
            else { D['op-center'].textContent = 'Center stage on embryo'; D['op-center'].disabled = false; }
        }
        gateFdriveNudges();

        drawOverlay(step);
        renderStatus();
        renderBoard();
        renderMiniMap();
        if (step === 'c0') renderChooser();
        else if (step === 'running') renderRunSpine();
    }

    function ensureInactiveCameraStopped(cam) {
        if (cam !== 'bottom' && _camOn) { fetch('/api/devices/bottom_camera/stream/stop', { method: 'POST' }).catch(() => {}); applyCam(false); }
        if (cam !== 'spim' && _spimOn) { fetch('/api/devices/lightsheet/live/stop', { method: 'POST' }).catch(() => {}); _spimOn = false; D['op-spim-toggle'].textContent = 'Start view'; D['op-spim-toggle'].classList.remove('op-btn-on'); }
    }
    function setImg(p) {
        if (!p || !p.jpeg_b64) return;
        D['op-cam-img'].src = `data:${p.mime || 'image/jpeg'};base64,${p.jpeg_b64}`;
        if (!D['op-cam-img'].classList.contains('has-frame')) { D['op-cam-img'].classList.add('has-frame'); D['op-cam-ph'].style.display = 'none'; }
    }

    // ── status strip ──────────────────────────────────────────────────────
    function renderStatus() {
        D['op-st-head'].textContent = _headLowered ? '▼ lowered' : '▲ up';
        D['op-st-head'].parentElement.classList.toggle('is-down', _headLowered);
        D['op-st-floor'].textContent = _fdFloor != null ? `${Math.round(_fdFloor)} µm` : '—';
        D['op-st-led'].textContent = _ledOn ? 'EMITTING' : 'OFF';
        D['op-st-led-wrap'].classList.toggle('is-emitting', _ledOn);
        D['op-st-laser'].textContent = _acquiring ? 'EMITTING' : 'OFF';
        D['op-st-laser-wrap'].classList.toggle('is-emitting', _acquiring);
        const cam = cameraForStep(effectiveStep());
        const on = cam === 'bottom' ? _camOn : _spimOn;
        D['op-st-cam'].textContent = `${cam === 'bottom' ? 'bottom' : 'SPIM'} ${on ? '● live' : '○ off'}`;
    }

    // ── left spine: worklist + mini-map ────────────────────────────────────
    function resolveXY(emb) {
        const f = emb && emb.position_fine;
        if (f && Number.isFinite(f.x) && Number.isFinite(f.y)) return { x: f.x, y: f.y };
        const c = emb && emb.position_coarse;
        if (c && Number.isFinite(c.x) && Number.isFinite(c.y)) return { x: c.x, y: c.y };
        return null;
    }
    function labelFor(emb) { const m = emb && emb.id && String(emb.id).match(/(\d+)/); return m ? m[1] : '?'; }

    function renderBoard() {
        if (!D['op-board']) return;
        const imaged = _embryos.filter(e => _states[e.id] === 'imaged').length;
        D['op-board-count'].textContent = `${imaged} / ${_embryos.length} imaged`;
        D['op-board'].innerHTML = '';
        if (!_embryos.length) { const e = document.createElement('div'); e.className = 'op-empty'; e.textContent = 'No embryos yet'; D['op-board'].appendChild(e); return; }
        _embryos.forEach(emb => {
            const st = _states[emb.id] || 'marked';
            const rank = STATE_RANK[st];
            const xy = resolveXY(emb);
            const row = document.createElement('div');
            row.className = 'op-brow' + (emb.id === _selected ? ' is-sel' : '');
            row.addEventListener('click', () => selectEmbryo(emb.id));

            const dot = document.createElement('span');
            dot.className = 'op-bdot'; dot.textContent = labelFor(emb);
            if (st === 'imaged') dot.style.background = 'var(--accent-green)';
            row.appendChild(dot);

            const meta = document.createElement('span'); meta.className = 'op-bmeta';
            const lab = document.createElement('span'); lab.className = 'op-blabel';
            lab.textContent = xy ? `(${xy.x.toFixed(0)}, ${xy.y.toFixed(0)})` : `embryo ${labelFor(emb)}`;
            meta.appendChild(lab);
            const track = document.createElement('span'); track.className = 'op-track';
            ['centered', 'lowering', 'focused', 'imaged'].forEach((k, i) => {
                const t = document.createElement('span'); t.className = 'op-tnode';
                if (rank >= STATE_RANK[k]) t.classList.add('on-' + k);
                track.appendChild(t);
            });
            meta.appendChild(track);
            row.appendChild(meta);

            const sc = document.createElement('span'); sc.className = 'op-bstate'; sc.textContent = st;
            row.appendChild(sc);
            D['op-board'].appendChild(row);
        });
    }

    function renderMiniMap() {
        const svg = D['op-minimap']; if (!svg) return;
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        const pts = _embryos.map(e => resolveXY(e)).filter(Boolean);
        if (_lastXY) pts.push({ x: _lastXY.X, y: _lastXY.Y });
        if (!pts.length) return;
        let xMin = Math.min(...pts.map(p => p.x)), xMax = Math.max(...pts.map(p => p.x));
        let yMin = Math.min(...pts.map(p => p.y)), yMax = Math.max(...pts.map(p => p.y));
        const span = Math.max(xMax - xMin, yMax - yMin, 100), padf = span * 0.18;
        const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2, half = span / 2 + padf;
        const toX = x => ((x - (cx - half)) / (2 * half)) * 100;
        const toY = y => 100 - ((y - (cy - half)) / (2 * half)) * 100;  // flip Y (stage +Y up)
        const r = 2.4;
        _embryos.forEach(emb => {
            const xy = resolveXY(emb); if (!xy) return;
            const st = _states[emb.id] || 'marked';
            const c = document.createElementNS(SVG_NS, 'circle');
            c.setAttribute('cx', toX(xy.x)); c.setAttribute('cy', toY(xy.y)); c.setAttribute('r', r);
            const col = st === 'imaged' ? 'var(--accent-green)' : (st === 'focused' || st === 'calibrated') ? 'var(--accent-orange)'
                : (st === 'centered' || st === 'lowering') ? 'var(--accent)' : ROLE_NEUTRAL;
            c.setAttribute('fill', col);
            if (emb.id === _selected) { c.setAttribute('stroke', 'var(--text)'); c.setAttribute('stroke-width', '1.2'); }
            svg.appendChild(c);
        });
        if (_lastXY) {  // stage crosshair
            const X = toX(_lastXY.X), Y = toY(_lastXY.Y);
            [[X - 4, Y, X + 4, Y], [X, Y - 4, X, Y + 4]].forEach(([x1, y1, x2, y2]) => {
                const l = document.createElementNS(SVG_NS, 'line');
                l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2);
                l.setAttribute('stroke', 'var(--accent-cyan)'); l.setAttribute('stroke-width', '0.8');
                svg.appendChild(l);
            });
        }
    }

    // ── viewport overlays ──────────────────────────────────────────────────
    function renderedRect() {
        const sb = D['op-cam-stage'].getBoundingClientRect();
        const fw = _frozenFrame ? _frozenFrame.w : (_lastFrame ? _lastFrame.shape[1] : sb.width);
        const fh = _frozenFrame ? _frozenFrame.h : (_lastFrame ? _lastFrame.shape[0] : sb.height);
        const ar = fw / fh, sar = sb.width / sb.height;
        let w, h;
        if (ar > sar) { w = sb.width; h = sb.width / ar; } else { h = sb.height; w = sb.height * ar; }
        return { x: (sb.width - w) / 2, y: (sb.height - h) / 2, w, h, fw, fh, sb };
    }
    function canvasCtx() {
        const sb = D['op-cam-stage'].getBoundingClientRect();
        D['op-mark-canvas'].width = Math.round(sb.width); D['op-mark-canvas'].height = Math.round(sb.height);
        const ctx = D['op-mark-canvas'].getContext('2d');
        ctx.clearRect(0, 0, D['op-mark-canvas'].width, D['op-mark-canvas'].height);
        return ctx;
    }
    function drawOverlay(step) {
        if (!D['op-mark-canvas']) return;
        if (step === 'a2') return drawMarkers();
        const ctx = canvasCtx();
        if (step === 'b1') drawReticle(ctx);
        else if (step === 'b2') drawFloorGauge(ctx);
    }
    function drawMarkers() {
        const r = renderedRect(); const ctx = canvasCtx();
        _markers.forEach((m, i) => {
            const cx = r.x + (m.fx / r.fw) * r.w, cy = r.y + (m.fy / r.fh) * r.h;
            ctx.beginPath(); ctx.arc(cx, cy, 11, 0, 7); ctx.lineWidth = 2; ctx.strokeStyle = '#34d399'; ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();
            ctx.fillStyle = '#34d399'; ctx.font = '600 11px Inter Tight, sans-serif'; ctx.fillText(String(i + 1), cx + 13, cy - 8);
        });
    }
    function drawReticle(ctx) {
        const sb = D['op-mark-canvas']; const cx = sb.width / 2, cy = sb.height / 2;
        ctx.strokeStyle = 'rgba(96,165,250,0.85)'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(cx - 22, cy); ctx.lineTo(cx + 22, cy); ctx.moveTo(cx, cy - 22); ctx.lineTo(cx, cy + 22); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx, cy, 10, 0, 7); ctx.stroke();
        // SPIM-FOV footprint box (light-sheet FOV << bottom FOV)
        const r = renderedRect(); const bw = r.w * 0.22, bh = r.h * 0.22;
        ctx.setLineDash([5, 3]); ctx.strokeStyle = 'rgba(34,211,238,0.7)';
        ctx.strokeRect(cx - bw / 2, cy - bh / 2, bw, bh); ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(34,211,238,0.85)'; ctx.font = '600 10px Inter Tight, sans-serif';
        ctx.fillText('SPIM FOV', cx - bw / 2, cy - bh / 2 - 4);
    }
    function drawFloorGauge(ctx) {
        const sb = D['op-mark-canvas']; const pad = 24, h = 26, y = sb.height - pad - h, w = sb.width - 2 * pad, x = pad;
        const floor = _fdFloor;
        ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
        // fill: full when far from floor, shrinks toward floor
        const MAXD = 500;
        const frac = floor == null ? 0 : Math.max(0, Math.min(1, floor / MAXD));
        const col = floor == null ? '#555' : floor <= 30 ? '#ef4444' : floor < 150 ? '#fb923c' : '#4ade80';
        ctx.fillStyle = col; ctx.fillRect(x + 2, y + 2, (w - 4) * frac, h - 4);
        ctx.fillStyle = '#fff'; ctx.font = '700 13px Inter Tight, sans-serif';
        ctx.fillText(floor == null ? 'distance to floor —' : `${Math.round(floor)} µm to floor (30 µm hard floor)`, x + 8, y + h - 8);
    }

    // ── bottom-cam frames ──────────────────────────────────────────────────
    function onBottomFrame(p) {
        if (!p || !p.jpeg_b64) return;
        _lastFrame = p;
        if (p.focus_score != null) { _bzScore = p.focus_score; if (D['op-bz-score']) D['op-bz-score'].textContent = Number(p.focus_score).toFixed(3); }
        const cam = cameraForStep(effectiveStep());
        if (cam === 'bottom' && !_marking) setImg(p);
    }
    function onLightsheetFrame(p) {
        if (!p || !p.jpeg_b64) return;
        _lastSpimFrame = p;
        if (p.focus_score != null) { _spimScore = p.focus_score; if (D['op-spim-score']) D['op-spim-score'].textContent = Number(p.focus_score).toFixed(3); }
        if (cameraForStep(effectiveStep()) === 'spim') setImg(p);
    }

    // ── A1 focus bottom ────────────────────────────────────────────────────
    async function toggleCamera() {
        D['op-cam-toggle'].disabled = true;
        try {
            const ep = _camOn ? '/api/devices/bottom_camera/stream/stop' : '/api/devices/bottom_camera/stream/start';
            const d = await postJSON(ep, {}); applyCam(!!d.streaming);
        } catch (e) { toast(`Camera toggle failed (${e.status || e.message})`); }
        finally { D['op-cam-toggle'].disabled = false; }
    }
    function applyCam(on) {
        _camOn = on;
        D['op-cam-toggle'].textContent = on ? 'Stop camera' : 'Start camera';
        D['op-cam-toggle'].classList.toggle('op-btn-on', on);
        if (!on && !_marking) { D['op-cam-img'].classList.remove('has-frame'); D['op-cam-ph'].style.display = ''; D['op-cam-ph'].textContent = 'Camera off'; }
        renderStatus();
    }
    async function nudgeBottomZ(delta) {
        if (_marking) { toast('Finish marking first'); return; }
        try { const d = await postJSON('/api/devices/stage/bottom_z/nudge', { delta }); if (d.position != null) D['op-bz-pos'].textContent = Number(d.position).toFixed(1); }
        catch (e) { toast(`Bottom-Z nudge blocked (${e.status || e.message})`); }
    }

    // ── A2 mark ────────────────────────────────────────────────────────────
    function umPerPxDisplay() { return BASE_UM_PER_PX * ((_frozenFrame && _frozenFrame.downsample) || 1); }
    function frameToStage(fx, fy) {
        const u = umPerPxDisplay(), cx = _frozenFrame.w / 2, cy = _frozenFrame.h / 2;
        return [_captureStage[0] + (fx - cx) * u, _captureStage[1] - (fy - cy) * u];
    }
    function enterMarking(cands) {
        if (!_lastFrame) { toast('Start the camera first'); return false; }
        // Absolute stage origin for pixel→stage conversion. Prefer the XY
        // stamped on the live frame by the device layer; fall back to the
        // position stream. NEVER default to [0, 0] — that silently converts
        // clicks to offsets from stage origin, so embryos land hundreds of µm
        // off and calibration images empty field. Block marking instead.
        const capStage = (Array.isArray(_lastFrame.stage_position) && _lastFrame.stage_position.length === 2)
            ? _lastFrame.stage_position
            : (_lastXY ? [_lastXY.X, _lastXY.Y] : null);
        if (!capStage) { toast('Stage position unknown — wait for the position readout, then mark'); return false; }
        _marking = true;
        _frozenFrame = { w: _lastFrame.shape[1], h: _lastFrame.shape[0], downsample: _lastFrame.downsample || 1 };
        _captureStage = capStage;
        _frozenSrc = `data:${_lastFrame.mime || 'image/jpeg'};base64,${_lastFrame.jpeg_b64}`;
        D['op-cam-img'].src = _frozenSrc; D['op-cam-img'].classList.add('has-frame'); D['op-cam-ph'].style.display = 'none';
        _markers = [];
        (cands || []).forEach(c => {
            const ds = _frozenFrame.downsample;
            const fx = c.pixel_x != null ? c.pixel_x / ds : _frozenFrame.w / 2;
            const fy = c.pixel_y != null ? c.pixel_y / ds : _frozenFrame.h / 2;
            const s = (c.stage_x_um != null && c.stage_y_um != null) ? [c.stage_x_um, c.stage_y_um] : frameToStage(fx, fy);
            _markers.push({ fx, fy, stageX: s[0], stageY: s[1], source: 'sam' });
        });
        D['op-cam-stage'].classList.add('is-marking');
        renderStep(); updateMarkCount();
        return true;
    }
    function exitMarking() {
        _marking = false; _markers = []; _frozenSrc = null; _frozenFrame = null;
        D['op-cam-stage'].classList.remove('is-marking');
    }
    function updateMarkCount() {
        const n = _markers.length;
        if (D['op-mark-count']) D['op-mark-count'].textContent = `(${n})`;
        if (D['op-confirm']) D['op-confirm'].disabled = n === 0;
        if (D['op-clear']) D['op-clear'].disabled = n === 0;
    }
    function onCanvasClick(e) {
        if (!_marking) return;
        const r = renderedRect(), rect = D['op-mark-canvas'].getBoundingClientRect();
        const cxv = e.clientX - rect.left, cyv = e.clientY - rect.top;
        for (let i = 0; i < _markers.length; i++) {
            const mx = r.x + (_markers[i].fx / r.fw) * r.w, my = r.y + (_markers[i].fy / r.fh) * r.h;
            if (Math.hypot(cxv - mx, cyv - my) <= MARK_HIT_PX) { _markers.splice(i, 1); drawMarkers(); updateMarkCount(); return; }
        }
        if (cxv < r.x || cxv > r.x + r.w || cyv < r.y || cyv > r.y + r.h) return;
        const fx = ((cxv - r.x) / r.w) * r.fw, fy = ((cyv - r.y) / r.h) * r.fh, s = frameToStage(fx, fy);
        _markers.push({ fx, fy, stageX: s[0], stageY: s[1], source: 'manual' });
        drawMarkers(); updateMarkCount();
    }
    async function runDetect() {
        if (!_marking && !enterMarking([])) return;
        D['op-detect'].disabled = true; D['op-detect'].textContent = 'Detecting…';
        try {
            const d = await postJSON('/api/devices/detect_embryos', {});
            const cands = Array.isArray(d.embryos) ? d.embryos : [];
            if (_lastFrame && d.stage_position) { _lastFrame.stage_position = d.stage_position; _captureStage = d.stage_position; }
            cands.forEach(c => {
                const ds = _frozenFrame.downsample;
                const fx = c.pixel_x != null ? c.pixel_x / ds : _frozenFrame.w / 2;
                const fy = c.pixel_y != null ? c.pixel_y / ds : _frozenFrame.h / 2;
                const s = (c.stage_x_um != null && c.stage_y_um != null) ? [c.stage_x_um, c.stage_y_um] : frameToStage(fx, fy);
                _markers.push({ fx, fy, stageX: s[0], stageY: s[1], source: 'sam' });
            });
            drawMarkers(); updateMarkCount();
            toast(`Detected ${cands.length} candidate${cands.length === 1 ? '' : 's'}`);
        } catch (e) { toast(`Detect failed (${e.status || e.message})`); }
        finally { D['op-detect'].disabled = false; D['op-detect'].textContent = 'Detect (SAM)'; }
    }
    async function confirmMarks() {
        if (!_markers.length) return;
        if (!Array.isArray(_captureStage) || _captureStage.length !== 2) {
            toast('Stage position unknown — cannot register markers'); return;
        }
        D['op-confirm'].disabled = true;
        try {
            const markers = _markers.map(m => ({ stage_x_um: m.stageX, stage_y_um: m.stageY, pixel_x: m.fx, pixel_y: m.fy, source: m.source }));
            const d = await postJSON('/api/devices/embryos/confirm', {
                markers, image_b64: _frozenSrc ? _frozenSrc.split(',')[1] : undefined,
                frame: _frozenFrame ? { w: _frozenFrame.w, h: _frozenFrame.h, downsample: _frozenFrame.downsample } : undefined,
                stage_position: _captureStage,
            });
            toast(`Registered ${(d.registered || []).length} embryo${(d.registered || []).length === 1 ? '' : 's'}`);
            exitMarking();  // EMBRYOS_UPDATE refreshes the board, then auto-select first
        } catch (e) { toast(`Confirm failed (${e.status || e.message})`); D['op-confirm'].disabled = false; }
    }

    // ── B1 center ──────────────────────────────────────────────────────────
    function selectEmbryo(id) {
        _selected = id;
        if (id) _step = stepForState(_states[id] || 'marked');
        renderStep();
    }
    async function centerOnSelected() {
        if (_headLowered) { toast('Retract the SPIM head first'); return; }
        const emb = _embryos.find(e => e.id === _selected), xy = emb ? resolveXY(emb) : null;
        if (!xy) return;
        D['op-center'].disabled = true; D['op-center'].textContent = 'Centering…';
        try {
            await postJSON('/api/devices/stage/move', { x: xy.x, y: xy.y });
            advanceState(_selected, 'centered'); setStep('b2');
            toast(`Centred on embryo ${labelFor(emb)}`);
        } catch (e) { toast(`Center failed (${e.status || e.message})`); D['op-center'].disabled = false; D['op-center'].textContent = 'Center stage on embryo'; }
    }
    function advanceState(id, st) {
        if (!id) return;
        if (STATE_RANK[st] > STATE_RANK[_states[id] || 'marked']) _states[id] = st;
        renderStep();
    }

    // ── B2 lower (F-drive, fenced) ─────────────────────────────────────────
    function gateFdriveNudges() {
        // auto-grey down-nudges that would exceed remaining distance-to-floor
        if (D['op-fd-d100']) D['op-fd-d100'].disabled = _fdFloor != null && _fdFloor < 100;
        if (D['op-fd-d10']) D['op-fd-d10'].disabled = _fdFloor != null && _fdFloor < 10;
    }
    async function nudgeFdrive(delta) {
        if (delta < 0 && _fdFloor != null && Math.abs(delta) > _fdFloor) { toast('Too close to the floor for that step'); return; }
        try {
            const d = await postJSON('/api/devices/spim/fdrive/nudge', { delta });
            if (d.position != null) { _fdPos = d.position; D['op-fd-pos'].textContent = Number(d.position).toFixed(1); }
            if (d.distance_to_floor != null) { _fdFloor = d.distance_to_floor; D['op-fd-floor'].textContent = Number(d.distance_to_floor).toFixed(0); }
            if (delta < 0) { _headLowered = true; if (_selected) advanceState(_selected, 'lowering'); }
            renderStep();  // refresh gauge + gates + status
        } catch (e) { toast(`F-drive nudge blocked (${e.status || e.message})`); }
    }

    // ── B3 focus SPIM ──────────────────────────────────────────────────────
    async function toggleSpim() {
        D['op-spim-toggle'].disabled = true;
        try {
            const ep = _spimOn ? '/api/devices/lightsheet/live/stop' : '/api/devices/lightsheet/live/start';
            const d = await postJSON(ep, {}); _spimOn = !!d.streaming;
            D['op-spim-toggle'].textContent = _spimOn ? 'Stop view' : 'Start view';
            D['op-spim-toggle'].classList.toggle('op-btn-on', _spimOn);
            if (!_spimOn) { D['op-cam-img'].classList.remove('has-frame'); D['op-cam-ph'].style.display = ''; }
            renderStatus();
        } catch (e) { toast(`SPIM view toggle failed (${e.status || e.message})`); }
        finally { D['op-spim-toggle'].disabled = false; }
    }
    let _lsTimer = null;
    function postLsParams() {
        if (_lsTimer) clearTimeout(_lsTimer);
        _lsTimer = setTimeout(() => { postJSON('/api/devices/lightsheet/live/params', { galvo: _galvo, piezo: _piezo, exposure: 20, side: 'A' }).catch(() => {}); }, 120);
    }
    function nudgeGalvo(d) { _galvo = Math.max(-5, Math.min(5, _galvo + d)); D['op-gv'].textContent = _galvo.toFixed(1); postLsParams(); }
    function nudgePiezo(d) { _piezo = Math.max(0, Math.min(200, _piezo + d)); D['op-pz'].textContent = _piezo.toFixed(0); postLsParams(); }
    async function toggleLed() {
        _ledOn = !_ledOn;
        D['op-led'].classList.toggle('op-btn-toggle', true); D['op-led'].setAttribute('aria-pressed', _ledOn ? 'true' : 'false');
        D['op-led'].classList.toggle('op-btn-on', _ledOn);
        try { await postJSON('/api/devices/led/set', { state: _ledOn ? 'Open' : 'Closed' }); } catch (e) { toast(`LED failed (${e.status || e.message})`); }
        renderStatus();
    }
    async function forceLedOff() {
        if (!_ledOn) return;
        _ledOn = false; D['op-led'].setAttribute('aria-pressed', 'false'); D['op-led'].classList.remove('op-btn-on');
        try { await postJSON('/api/devices/led/set', { state: 'Closed' }); } catch (_) {}
        renderStatus();
    }
    function markInFocus() {
        if (!_selected) return;
        advanceState(_selected, 'focused'); forceLedOff(); setStep('bc');
        toast(`Embryo ${labelFor(_embryos.find(e => e.id === _selected))} in focus`);
    }

    // ── B-cal calibrate (piezo-galvo) ──────────────────────────────────────
    async function calibrateSelected() {
        if (!_selected) return;
        if (STATE_RANK[_states[_selected] || 'marked'] < STATE_RANK.focused) { toast('Focus the embryo first'); return; }
        D['op-calibrate'].disabled = true; D['op-calibrate'].textContent = 'Calibrating…';
        if (D['op-cal-result']) D['op-cal-result'].textContent = 'sweeping…';
        try {
            const d = await postJSON(`/api/devices/embryos/${_selected}/calibrate`, {});
            const cal = d.calibration || {};
            const slope = cal.slope_um_per_deg, r2 = cal.r_squared;
            if (D['op-cal-result']) {
                D['op-cal-result'].textContent = (slope != null)
                    ? `${Number(slope).toFixed(1)} µm/deg${r2 != null ? ` · R² ${Number(r2).toFixed(2)}` : ''}`
                    : 'done';
            }
            advanceState(_selected, 'calibrated'); setStep('b4');
            toast(`Calibrated embryo ${labelFor(_embryos.find(e => e.id === _selected))}`);
        } catch (e) {
            if (D['op-cal-result']) D['op-cal-result'].textContent = 'failed';
            toast(`Calibrate failed (${e.status || e.message})`);
        } finally {
            D['op-calibrate'].disabled = false; D['op-calibrate'].textContent = 'Calibrate this embryo';
        }
    }
    function skipCalibration() {
        if (!_selected) return;
        advanceState(_selected, 'calibrated'); setStep('b4');
        toast('Calibration skipped');
    }

    // ── B4 acquire ─────────────────────────────────────────────────────────
    async function acquireSelected() {
        if (!_selected || STATE_RANK[_states[_selected] || 'marked'] < STATE_RANK.focused) { toast('Focus the embryo first'); return; }
        D['op-acquire'].disabled = true; D['op-acquire'].textContent = 'Acquiring…'; _acquiring = true; renderStatus();
        try {
            await postJSON('/api/devices/acquire/volume', { num_slices: 50, exposure_ms: 10.0 });
            advanceState(_selected, 'imaged'); setStep('b5');
            toast('Volume acquired');
        } catch (e) { toast(`Acquire failed (${e.status || e.message})`); }
        finally { _acquiring = false; D['op-acquire'].disabled = false; D['op-acquire'].textContent = 'Acquire volume'; await forceLedOff(); renderStatus(); }
    }

    // ── B5 retract & advance ───────────────────────────────────────────────
    async function retractAndAdvance() {
        D['op-retract'].disabled = true;
        try {
            await postJSON('/api/devices/spim/fdrive/nudge', { delta: 100 }).catch(() => {});
            _headLowered = false; _fdFloor = null;
            const next = _embryos.find(e => _states[e.id] !== 'imaged');
            if (next) { selectEmbryo(next.id); toast(`Next: embryo ${labelFor(next)}`); }
            else {
                // Done imaging — return to the Run chooser (not Focus) so another
                // run mode can be chosen for the same marked set.
                _selected = null; _step = null; _runState = 'choose'; renderStep();
                toast('All embryos imaged');
            }
        } finally { D['op-retract'].disabled = false; }
    }

    // ── embryo SSOT ────────────────────────────────────────────────────────
    function onEmbryosUpdate(p) {
        const wasEmpty = _embryos.length === 0;
        _embryos = (p && Array.isArray(p.embryos)) ? p.embryos : [];
        const ids = new Set(_embryos.map(e => e.id));
        Object.keys(_states).forEach(id => { if (!ids.has(id)) delete _states[id]; });
        Object.keys(_roles).forEach(id => { if (!ids.has(id)) delete _roles[id]; });
        _embryos.forEach(e => {
            if (!_states[e.id]) _states[e.id] = 'marked';
            // seed role from the embryo if present, else default to subject ('test')
            if (!_roles[e.id]) _roles[e.id] = (e.role && e.role !== 'unassigned') ? e.role : 'test';
        });
        if (_selected && !ids.has(_selected)) { _selected = null; _step = null; }
        // After the first marking confirm, enter the Phase C Run chooser
        // (NOT the old auto-dive into the manual loop).
        if (wasEmpty && _embryos.length && !_selected && !_marking && _runState === null) {
            _runState = 'choose';
        }
        renderStep();
    }

    // ── Phase C: Run chooser + run-spine ───────────────────────────────────
    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function renderChooser() {
        if (D['op-rolechips']) {
            D['op-rolechips'].innerHTML = '';
            _embryos.forEach(emb => {
                const role = _roles[emb.id] || 'test';
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'op-rolechip' + (role === 'calibration' ? ' is-reference' : '');
                chip.textContent = `${labelFor(emb)} · ${role === 'calibration' ? 'ref' : 'subj'}`;
                chip.title = 'Click to toggle subject / reference';
                chip.addEventListener('click', () => {
                    _roles[emb.id] = (_roles[emb.id] === 'calibration') ? 'test' : 'calibration';
                    renderChooser();
                });
                D['op-rolechips'].appendChild(chip);
            });
        }
        document.querySelectorAll('.op-modepanel').forEach(p => p.classList.toggle('is-shown', p.dataset.mode === _runMode));
        if (D['op-tl-stop'] && D['op-tl-condval']) {
            const s = D['op-tl-stop'].value;
            D['op-tl-condval'].style.display = (s === 'timepoints' || s === 'duration') ? '' : 'none';
        }
        if (_runMode === 'library') loadLibrary();
        if (_runMode === 'plan') loadPlanItems();
    }

    async function loadLibrary() {
        if (!D['op-lib-list']) return;
        try {
            const d = await getJSON('/api/tactic_library');
            const items = (d && d.tactics) || [];
            if (!items.length) { D['op-lib-list'].innerHTML = '<div class="op-empty">No saved tactics</div>'; return; }
            D['op-lib-list'].innerHTML = '';
            items.forEach(t => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'op-libitem' + (t.id === _selectedLib ? ' is-sel' : '');
                b.innerHTML = `${escapeHtml(t.name || t.id)}<small>${escapeHtml(t.kind || '')}</small>`;
                b.addEventListener('click', () => { _selectedLib = t.id; loadLibrary(); });
                D['op-lib-list'].appendChild(b);
            });
        } catch (_) { D['op-lib-list'].innerHTML = '<div class="op-empty">Library unavailable</div>'; }
    }
    function loadPlanItems() {
        // Plan resolution lives in the agent (resume_plan + execute_plan_item);
        // Start hands the roster to the agent to attach + continue the right item.
        if (D['op-plan-list']) {
            D['op-plan-list'].innerHTML = '<div class="op-empty">Start hands these embryos to the agent to attach a plan item and continue imaging.</div>';
        }
    }

    async function applyRoles() {
        const roles = {};
        _embryos.forEach(e => { roles[e.id] = _roles[e.id] || 'test'; });
        try { await postJSON('/api/embryos/roles', { roles }); } catch (e) { toast(`Roles failed (${e.status || e.message})`); }
    }

    async function startRun() {
        // Persist the chooser's role toggles for EVERY mode (manual included).
        await applyRoles();
        if (_runMode === 'manual') {
            // record a cosmetic oneshot tactic so even a manual sweep shows on the spine
            postJSON('/api/operate/run-tactic', {
                tactic: { kind: 'oneshot', name: 'Manual sweep', structure: { note: 'manual one-by-one' } },
                embryo_ids: _embryos.map(e => e.id),
            }).catch(() => {});
            _runState = null;
            if (_embryos.length) selectEmbryo(_embryos[0].id);
            return;
        }
        const subjects = _embryos.filter(e => (_roles[e.id] || 'test') !== 'calibration').map(e => e.id);
        const embryo_ids = subjects.length ? subjects : _embryos.map(e => e.id);
        if (_runMode === 'adaptive') {
            const interval = Math.max(1, Number(D['op-tl-interval'].value) || 120);
            const stopSel = D['op-tl-stop'].value;
            const monitoring = D['op-tl-monitor'].value;
            // Send the combined stop form the orchestrator parser understands
            // ('timepoints:N' / 'duration:Xh'); a bare 'timepoints' silently
            // degrades to manual (never stops).
            let stop_condition = stopSel;
            if (stopSel === 'timepoints') stop_condition = `timepoints:${Math.max(1, Number(D['op-tl-condval'].value) || 1)}`;
            else if (stopSel === 'duration') stop_condition = `duration:${Math.max(1, Number(D['op-tl-condval'].value) || 1)}h`;
            const body = { embryo_ids, interval_seconds: interval, stop_condition, monitoring_mode: monitoring };
            D['op-run-start'].disabled = true; D['op-run-start'].textContent = 'Starting…';
            try {
                await postJSON('/api/devices/timelapse/start', body);
                _runMeta = { mode: 'adaptive', interval, stop: stop_condition, monitoring, n: embryo_ids.length };
                _runState = 'running'; _runPaused = false;
                toast(`Adaptive timelapse started — ${embryo_ids.length} subject${embryo_ids.length === 1 ? '' : 's'}`);
                renderStep();
            } catch (e) {
                toast(`Start failed (${e.status || e.message})`);
            } finally { D['op-run-start'].disabled = false; D['op-run-start'].textContent = 'Start run'; }
            return;
        }
        const roster = _embryos.map(e => {
            const xy = resolveXY(e);
            const r = _roles[e.id] === 'calibration' ? 'reference' : 'subject';
            return `${labelFor(e)}${xy ? ` (${xy.x.toFixed(0)},${xy.y.toFixed(0)})` : ''} [${r}]`;
        }).join(', ');

        if (_runMode === 'library') {
            if (!_selectedLib) { toast('Pick a saved tactic'); return; }
            D['op-run-start'].disabled = true; D['op-run-start'].textContent = 'Starting…';
            try {
                const d = await postJSON('/api/operate/run-tactic', { library_id: _selectedLib, embryo_ids });
                if (d.success) {
                    _runMeta = { mode: 'library', n: embryo_ids.length };
                    _runState = 'running'; _runPaused = false; toast('Tactic started'); renderStep();
                } else { toast(`Run failed: ${(d.result && d.result.message) || '?'}`); }
            } catch (e) { toast(`Start failed (${e.status || e.message})`); }
            finally { D['op-run-start'].disabled = false; D['op-run-start'].textContent = 'Start run'; }
            return;
        }
        if (_runMode === 'plan' || _runMode === 'agent') {
            // The agent owns plan resolution + composed tactics; hand off the roster.
            const prompt = _runMode === 'plan'
                ? `Continue a plan on these ${_embryos.length} marked embryos — attach this session to the right plan item and start imaging: ${roster}.`
                : `I marked ${_embryos.length} embryos: ${roster}. Propose and start an Operation Plan to image them.`;
            if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
                AgentChat.togglePanel(true);
                if (AgentChat.runCommand) setTimeout(() => AgentChat.runCommand(prompt), 300);
            } else { toast('Agent chat unavailable'); }
            _runState = 'running'; renderStep();
            return;
        }
    }

    async function renderRunSpine() {
        if (!D['op-runspine']) return;
        let tactics = [];
        try { const d = await getJSON('/api/operation_plan'); tactics = (d && d.plan && d.plan.tactics) || []; } catch (_) {}
        D['op-runspine'].innerHTML = '';
        if (tactics.length) {
            tactics.forEach(t => D['op-runspine'].appendChild(tacticCard(t)));
        } else if (_runMeta) {
            // Fallback card when the plan fetch is empty/failed — shaped per mode
            // (a library run carries no interval/stop/monitoring).
            const n = _runMeta.n || 0;
            const subj = `${n} subject${n === 1 ? '' : 's'}`;
            const card = document.createElement('div');
            card.className = 'op-tcard st-active';
            if (_runMeta.mode === 'adaptive') {
                const stopTxt = (_runMeta.stop === 'manual' || _runMeta.stop == null) ? 'until stopped' : _runMeta.stop;
                const ivl = _runMeta.interval != null ? `${escapeHtml(String(_runMeta.interval))}s · ` : '';
                card.innerHTML = '<div class="op-tcard-head"><span class="op-tcard-name">Adaptive timelapse</span><span class="op-tcard-state">active</span></div>'
                    + `<div class="op-tcard-kind">standing_timelapse · ${escapeHtml(_runMeta.monitoring || 'idle')}</div>`
                    + `<div class="op-tcard-meta">${ivl}${subj} · ${escapeHtml(stopTxt)}</div>`;
            } else {
                card.innerHTML = '<div class="op-tcard-head"><span class="op-tcard-name">Tactic</span><span class="op-tcard-state">active</span></div>'
                    + `<div class="op-tcard-meta">${subj}</div>`;
            }
            D['op-runspine'].appendChild(card);
        } else {
            D['op-runspine'].innerHTML = '<div class="op-empty">Run active.</div>';
        }
        if (D['op-run-pause']) D['op-run-pause'].textContent = _runPaused ? 'Resume' : 'Pause';
    }
    function tacticCard(t) {
        const card = document.createElement('div');
        const state = t.state || 'planned';
        card.className = 'op-tcard st-' + state;
        const struct = t.structure || {};
        const meta = [];
        if (struct.cadence_s != null) meta.push(`${struct.cadence_s}s`);
        if (struct.interval != null) meta.push(`${struct.interval}s`);
        if (struct.status) meta.push(struct.status);
        if (t.live && t.live.signal != null) meta.push(`signal ${t.live.signal}`);
        card.innerHTML = `<div class="op-tcard-head"><span class="op-tcard-name">${escapeHtml(t.name || t.id)}</span><span class="op-tcard-state">${escapeHtml(state)}</span></div>`
            + `<div class="op-tcard-kind">${escapeHtml(t.kind || '')}</div>`
            + (meta.length ? `<div class="op-tcard-meta">${escapeHtml(meta.join(' · '))}</div>` : '')
            + (t.rationale ? `<div class="op-tcard-meta">${escapeHtml(t.rationale)}</div>` : '');
        return card;
    }

    async function pauseRun() {
        try {
            if (_runPaused) { await postJSON('/api/devices/timelapse/resume', {}); _runPaused = false; toast('Resumed'); }
            else { await postJSON('/api/devices/timelapse/pause', {}); _runPaused = true; toast('Paused'); }
            renderRunSpine();
        } catch (e) { toast(`Pause/resume failed (${e.status || e.message})`); }
    }
    async function stopRun() {
        if (!window.confirm('Stop the run?')) return;
        try { await postJSON('/api/devices/timelapse/stop', { reason: 'operator' }); } catch (e) { toast(`Stop failed (${e.status || e.message})`); }
        _runState = 'choose'; _runMeta = null; _runPaused = false;
        toast('Run stopped'); renderStep();
    }
    function openInOperations() {
        const nav = [...document.querySelectorAll('[data-tab]')].find(n => /operations/i.test(n.textContent || ''));
        if (nav) nav.click(); else toast('Operations tab not found');
    }

    // ── lifecycle ──────────────────────────────────────────────────────────
    function surveyMore() { _selected = null; _step = null; _runState = null; if (_marking) exitMarking(); renderStep(); }

    function wire() {
        if (_wired) return; _wired = true;
        cacheDom();
        D['op-cam-toggle'].addEventListener('click', toggleCamera);
        D['op-mark-canvas'].addEventListener('click', onCanvasClick);
        D['op-bz-nudge'].addEventListener('click', e => { const b = e.target.closest('[data-bz]'); if (b) nudgeBottomZ(Number(b.dataset.bz)); });
        D['op-tomark'].addEventListener('click', () => { if (enterMarking([])) renderStep(); });
        D['op-detect'].addEventListener('click', runDetect);
        D['op-confirm'].addEventListener('click', confirmMarks);
        D['op-clear'].addEventListener('click', () => { _markers = []; drawMarkers(); updateMarkCount(); });
        D['op-center'].addEventListener('click', centerOnSelected);
        D['op-fd-nudge'].addEventListener('click', e => { const b = e.target.closest('[data-fd]'); if (b) nudgeFdrive(Number(b.dataset.fd)); });
        D['op-tofocus'].addEventListener('click', () => setStep('b3'));
        D['op-spim-toggle'].addEventListener('click', toggleSpim);
        D['op-led'].addEventListener('click', toggleLed);
        document.querySelectorAll('[data-gv]').forEach(b => b.addEventListener('click', () => nudgeGalvo(Number(b.dataset.gv))));
        document.querySelectorAll('[data-pz]').forEach(b => b.addEventListener('click', () => nudgePiezo(Number(b.dataset.pz))));
        D['op-infocus'].addEventListener('click', markInFocus);
        if (D['op-calibrate']) D['op-calibrate'].addEventListener('click', calibrateSelected);
        if (D['op-cal-skip']) D['op-cal-skip'].addEventListener('click', skipCalibration);
        D['op-acquire'].addEventListener('click', acquireSelected);
        D['op-retract'].addEventListener('click', retractAndAdvance);
        D['op-survey-btn'].addEventListener('click', surveyMore);
        // Re-enter the Run chooser by clicking the "Run" stepper node — so a
        // different run mode can be chosen for an already-marked set (e.g. after
        // a manual sweep). Without this the chooser is a one-time gate.
        if (D['op-stepper']) D['op-stepper'].addEventListener('click', e => {
            const n = e.target.closest('.op-node');
            if (n && n.dataset.node === 'run' && _embryos.length && !_marking && _runState !== 'running') {
                _selected = null; _step = null; _runState = 'choose'; renderStep();
            }
        });
        // Phase C: Run chooser + run-spine
        document.querySelectorAll('input[name="op-mode"]').forEach(r =>
            r.addEventListener('change', () => { _runMode = r.value; renderChooser(); }));
        if (D['op-tl-stop']) D['op-tl-stop'].addEventListener('change', renderChooser);
        if (D['op-run-start']) D['op-run-start'].addEventListener('click', startRun);
        if (D['op-run-pause']) D['op-run-pause'].addEventListener('click', pauseRun);
        if (D['op-run-stop']) D['op-run-stop'].addEventListener('click', stopRun);
        if (D['op-run-open']) D['op-run-open'].addEventListener('click', openInOperations);
        window.addEventListener('resize', () => { if (_active) drawOverlay(effectiveStep()); });

        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('BOTTOM_CAMERA_FRAME', onBottomFrame);
            ClientEventBus.on('LIGHTSHEET_FRAME', onLightsheetFrame);
            ClientEventBus.on('EMBRYOS_UPDATE', onEmbryosUpdate);
            ClientEventBus.on('DEVICE_STATE_UPDATE', p => {
                const pos = p && p.positions; if (!pos) return;
                if (pos.xy_stage && Array.isArray(pos.xy_stage)) { _lastXY = { X: pos.xy_stage[0], Y: pos.xy_stage[1] }; if (_active) renderMiniMap(); }
            });
        }
    }

    async function activate() {
        wire();
        if (_active) return; _active = true;
        try { const s = await getJSON('/api/embryos/current'); onEmbryosUpdate(s); } catch (_) { renderStep(); }
        try { const b = await getJSON('/api/devices/stage/bottom_z'); if (b.position != null) D['op-bz-pos'].textContent = Number(b.position).toFixed(1); } catch (_) {}
        try { const f = await getJSON('/api/devices/spim/fdrive'); if (f.position != null) { _fdPos = f.position; D['op-fd-pos'].textContent = Number(f.position).toFixed(1); } if (f.distance_to_floor != null) { _fdFloor = f.distance_to_floor; D['op-fd-floor'].textContent = Number(f.distance_to_floor).toFixed(0); } } catch (_) {}
        renderStep();
    }
    function deactivate() {
        if (!_active) return; _active = false;
        if (_camOn) { fetch('/api/devices/bottom_camera/stream/stop', { method: 'POST' }).catch(() => {}); applyCam(false); }
        if (_spimOn) { fetch('/api/devices/lightsheet/live/stop', { method: 'POST' }).catch(() => {}); _spimOn = false; D['op-spim-toggle'].textContent = 'Start view'; D['op-spim-toggle'].classList.remove('op-btn-on'); }
        forceLedOff();
        if (_marking) exitMarking();
    }

    return { activate, deactivate };
})();
