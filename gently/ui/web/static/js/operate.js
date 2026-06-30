/**
 * Operate view — guided bottom-cam → SPIM operator surface.
 *
 * Three zones, mirroring the physical workflow:
 *   1. Survey & mark  — live bottom cam, fenced bottom-Z focus nudge, Detect /
 *                       click-to-mark all embryos on a frozen frame, Confirm.
 *   2. Embryos        — the single canonical experiment.embryos list (EMBRYOS_UPDATE),
 *                       each row a per-embryo state chip + select.
 *   3. Acquire        — for the selected embryo: Center → lower SPIM head (F-drive,
 *                       fenced) → focus SPIM (live + galvo/piezo/LED nudge) → Acquire.
 *
 * Single source of truth: marking confirms into experiment.embryos via the
 * agent-free /api/devices/embryos/confirm endpoint; the list refreshes from the
 * canonical EMBRYOS_UPDATE event. No autonomous Z moves — every focus move is a
 * bounded, server-fenced nudge.
 */
const OperateManager = (function () {
    // Camera transform constants (match gently/core/coordinates.py defaults).
    const BASE_UM_PER_PX = 6.5 / 10.0;  // pixel_size_um / objective_mag
    const MARK_HIT_PX = 14;             // click-to-remove radius (canvas px)

    let _wired = false;
    let _active = false;

    // DOM
    let _camImg, _camPh, _camStage, _markCanvas, _camToggle;
    let _bzPos, _bzScore, _bzNudge;
    let _detectBtn, _confirmBtn, _clearBtn, _markCount, _markHint;
    let _embryoList, _embryoCount;
    let _selLabel, _spimImg, _spimPh, _spimToggle, _ledBtn;
    let _fdPos, _fdFloor, _fdNudge, _centerBtn, _acquireBtn;
    let _gvVal, _pzVal, _spimScore;

    // State
    let _camOn = false, _spimOn = false;
    let _lastFrame = null;        // last live bottom-cam payload {jpeg_b64, shape, downsample, focus_score}
    let _marking = false;
    let _frozenSrc = null;        // frozen image data URL
    let _frozenFrame = null;      // {w, h, downsample} of the frozen frame (frame-space)
    let _captureStage = [0, 0];   // stage XY at freeze time (µm)
    let _markers = [];            // [{fx, fy, stageX, stageY, source}] fx/fy in frame-space px
    let _embryos = [];            // canonical list (EMBRYOS_UPDATE shape)
    let _selected = null;         // selected embryo id
    const _states = {};           // embryo_id -> 'marked'|'centered'|'focused'|'imaged'
    let _lastXY = null;           // {X, Y} from DEVICE_STATE_UPDATE
    let _galvo = 0.0, _piezo = 50.0, _ledOn = false;

    const _ROLE_COLOR = { test: '#ff66cc', calibration: '#00cccc', unassigned: '#8a8f98' };
    const _STATE_LABEL = { marked: 'marked', centered: 'centered', focused: 'focused', imaged: 'imaged' };

    function $(id) { return document.getElementById(id); }
    function toast(msg) { if (typeof showGentlyToast === 'function') showGentlyToast(msg); }

    async function postJSON(url, body) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        if (!res.ok) {
            const detail = await res.text().catch(() => '');
            const err = new Error(`${res.status} ${detail}`);
            err.status = res.status;
            throw err;
        }
        return res.json().catch(() => ({}));
    }
    async function getJSON(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
    }

    function cacheDom() {
        _camImg = $('op-cam-img'); _camPh = $('op-cam-ph'); _camStage = $('op-cam-stage');
        _markCanvas = $('op-mark-canvas'); _camToggle = $('op-cam-toggle');
        _bzPos = $('op-bz-pos'); _bzScore = $('op-bz-score'); _bzNudge = $('op-bz-nudge');
        _detectBtn = $('op-detect'); _confirmBtn = $('op-confirm'); _clearBtn = $('op-clear');
        _markCount = $('op-mark-count'); _markHint = $('op-mark-hint');
        _embryoList = $('op-embryo-list'); _embryoCount = $('op-embryo-count');
        _selLabel = $('op-sel-label'); _spimImg = $('op-spim-img'); _spimPh = $('op-spim-ph');
        _spimToggle = $('op-spim-toggle'); _ledBtn = $('op-led');
        _fdPos = $('op-fd-pos'); _fdFloor = $('op-fd-floor'); _fdNudge = $('op-fd-nudge');
        _centerBtn = $('op-center'); _acquireBtn = $('op-acquire');
        _gvVal = $('op-gv'); _pzVal = $('op-pz'); _spimScore = $('op-spim-score');
    }

    // ---- Bottom-camera survey -------------------------------------------
    async function toggleCamera() {
        _camToggle.disabled = true;
        try {
            const ep = _camOn ? '/api/devices/bottom_camera/stream/stop'
                              : '/api/devices/bottom_camera/stream/start';
            const data = await postJSON(ep, {});
            applyCameraState(!!data.streaming);
        } catch (err) {
            toast(`Camera toggle failed (${err.status || err.message})`);
        } finally {
            _camToggle.disabled = false;
        }
    }
    function applyCameraState(on) {
        _camOn = on;
        _camToggle.textContent = on ? 'Stop camera' : 'Start camera';
        _camToggle.classList.toggle('op-btn-on', on);
        if (!on && !_marking) { _camImg.classList.remove('has-frame'); _camPh.style.display = ''; _camPh.textContent = 'Camera off'; }
        else if (on && !_marking) { _camPh.textContent = 'waiting…'; }
    }
    function onBottomFrame(p) {
        if (!p || !p.jpeg_b64) return;
        _lastFrame = p;
        if (p.focus_score != null) _bzScore.textContent = Number(p.focus_score).toFixed(3);
        if (_marking) return;  // keep the frozen frame visible while marking
        _camImg.src = `data:${p.mime || 'image/jpeg'};base64,${p.jpeg_b64}`;
        if (!_camImg.classList.contains('has-frame')) { _camImg.classList.add('has-frame'); _camPh.style.display = 'none'; }
    }

    // ---- Marking --------------------------------------------------------
    function umPerPxDisplay() {
        const ds = (_frozenFrame && _frozenFrame.downsample) || 1;
        return BASE_UM_PER_PX * ds;
    }
    // frame-space (downsampled px) -> stage µm, using the frozen capture pose.
    function frameToStage(fx, fy) {
        const u = umPerPxDisplay();
        const cx = _frozenFrame.w / 2, cy = _frozenFrame.h / 2;
        return [_captureStage[0] + (fx - cx) * u, _captureStage[1] - (fy - cy) * u];
    }
    function enterMarking(candidates) {
        if (!_lastFrame) { toast('Start the camera first'); return false; }
        _marking = true;
        _frozenFrame = { w: _lastFrame.shape[1], h: _lastFrame.shape[0], downsample: _lastFrame.downsample || 1 };
        _captureStage = _lastFrame.stage_position || (_lastXY ? [_lastXY.X, _lastXY.Y] : [0, 0]);
        _frozenSrc = `data:${_lastFrame.mime || 'image/jpeg'};base64,${_lastFrame.jpeg_b64}`;
        _camImg.src = _frozenSrc; _camImg.classList.add('has-frame'); _camPh.style.display = 'none';
        _markers = [];
        (candidates || []).forEach(c => {
            const ds = _frozenFrame.downsample;
            const fx = (c.pixel_x != null) ? c.pixel_x / ds : _frozenFrame.w / 2;
            const fy = (c.pixel_y != null) ? c.pixel_y / ds : _frozenFrame.h / 2;
            const stage = (c.stage_x_um != null && c.stage_y_um != null)
                ? [c.stage_x_um, c.stage_y_um] : frameToStage(fx, fy);
            _markers.push({ fx, fy, stageX: stage[0], stageY: stage[1], source: 'sam' });
        });
        _camStage.classList.add('op-marking');
        _markHint.textContent = 'Click to add an embryo · click a marker to remove · Confirm when done.';
        drawMarkers(); updateMarkActions();
        return true;
    }
    function exitMarking() {
        _marking = false; _markers = []; _frozenSrc = null; _frozenFrame = null;
        _camStage.classList.remove('op-marking');
        clearCanvas();
        updateMarkActions();
        _markHint.textContent = 'Start the camera, then Detect — or click the image to mark embryos.';
        if (_camOn && _lastFrame) onBottomFrame(_lastFrame);
    }
    // rendered image rect (object-fit: contain) within the stage box.
    function renderedRect() {
        const sb = _camStage.getBoundingClientRect();
        const fw = _frozenFrame ? _frozenFrame.w : (_lastFrame ? _lastFrame.shape[1] : sb.width);
        const fh = _frozenFrame ? _frozenFrame.h : (_lastFrame ? _lastFrame.shape[0] : sb.height);
        const ar = fw / fh, sar = sb.width / sb.height;
        let w, h;
        if (ar > sar) { w = sb.width; h = sb.width / ar; } else { h = sb.height; w = sb.height * ar; }
        return { x: (sb.width - w) / 2, y: (sb.height - h) / 2, w, h, fw, fh, sb };
    }
    function clearCanvas() {
        if (!_markCanvas) return;
        const ctx = _markCanvas.getContext('2d');
        ctx && ctx.clearRect(0, 0, _markCanvas.width, _markCanvas.height);
    }
    function drawMarkers() {
        if (!_markCanvas || !_marking) return;
        const r = renderedRect();
        _markCanvas.width = Math.round(r.sb.width); _markCanvas.height = Math.round(r.sb.height);
        const ctx = _markCanvas.getContext('2d');
        ctx.clearRect(0, 0, _markCanvas.width, _markCanvas.height);
        _markers.forEach((m, i) => {
            const cx = r.x + (m.fx / r.fw) * r.w;
            const cy = r.y + (m.fy / r.fh) * r.h;
            ctx.beginPath(); ctx.arc(cx, cy, 11, 0, 7);
            ctx.lineWidth = 2; ctx.strokeStyle = '#34d399'; ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
            ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();
            ctx.fillStyle = '#34d399'; ctx.font = '600 11px Inter Tight, system-ui, sans-serif';
            ctx.fillText(String(i + 1), cx + 13, cy - 8);
        });
    }
    function onCanvasClick(e) {
        if (!_marking) return;
        const r = renderedRect();
        const rect = _markCanvas.getBoundingClientRect();
        const cxv = e.clientX - rect.left, cyv = e.clientY - rect.top;
        // hit-test existing markers (remove)
        for (let i = 0; i < _markers.length; i++) {
            const mx = r.x + (_markers[i].fx / r.fw) * r.w;
            const my = r.y + (_markers[i].fy / r.fh) * r.h;
            if (Math.hypot(cxv - mx, cyv - my) <= MARK_HIT_PX) {
                _markers.splice(i, 1); drawMarkers(); updateMarkActions(); return;
            }
        }
        // add — convert canvas px -> frame-space px -> stage µm
        if (cxv < r.x || cxv > r.x + r.w || cyv < r.y || cyv > r.y + r.h) return;
        const fx = ((cxv - r.x) / r.w) * r.fw;
        const fy = ((cyv - r.y) / r.h) * r.fh;
        const stage = frameToStage(fx, fy);
        _markers.push({ fx, fy, stageX: stage[0], stageY: stage[1], source: 'manual' });
        drawMarkers(); updateMarkActions();
    }
    function updateMarkActions() {
        const n = _markers.length;
        _markCount.textContent = `(${n})`;
        _confirmBtn.disabled = !_marking || n === 0;
        _clearBtn.disabled = !_marking || n === 0;
    }
    async function runDetect() {
        _detectBtn.disabled = true; _detectBtn.textContent = 'Detecting…';
        try {
            const data = await postJSON('/api/devices/detect_embryos', {});
            const cands = Array.isArray(data.embryos) ? data.embryos : [];
            if (_lastFrame && data.stage_position) _lastFrame.stage_position = data.stage_position;
            if (enterMarking(cands)) toast(`Detected ${cands.length} candidate${cands.length === 1 ? '' : 's'} — review & confirm`);
        } catch (err) {
            toast(`Detect failed (${err.status || err.message})`);
        } finally {
            _detectBtn.disabled = false; _detectBtn.textContent = 'Detect';
        }
    }
    async function confirmMarks() {
        if (!_markers.length) return;
        _confirmBtn.disabled = true;
        try {
            // Stage coords register the embryos; pixel coords + the frozen frame
            // are persisted as localization labels (sub-project B) server-side.
            const markers = _markers.map(m => ({
                stage_x_um: m.stageX, stage_y_um: m.stageY,
                pixel_x: m.fx, pixel_y: m.fy, source: m.source,
            }));
            const data = await postJSON('/api/devices/embryos/confirm', {
                markers,
                image_b64: _frozenSrc ? _frozenSrc.split(',')[1] : undefined,
                frame: _frozenFrame ? { w: _frozenFrame.w, h: _frozenFrame.h, downsample: _frozenFrame.downsample } : undefined,
                stage_position: _captureStage,
            });
            const n = (data.registered || []).length;
            toast(`Registered ${n} embryo${n === 1 ? '' : 's'}`);
            exitMarking();  // list refreshes via EMBRYOS_UPDATE
        } catch (err) {
            toast(`Confirm failed (${err.status || err.message})`);
            _confirmBtn.disabled = false;
        }
    }

    // ---- Embryo list (single source of truth) ---------------------------
    function resolveXY(emb) {
        const f = emb && emb.position_fine;
        if (f && Number.isFinite(f.x) && Number.isFinite(f.y)) return { x: f.x, y: f.y };
        const c = emb && emb.position_coarse;
        if (c && Number.isFinite(c.x) && Number.isFinite(c.y)) return { x: c.x, y: c.y };
        return null;
    }
    function labelFor(emb, i) {
        const m = emb.id && String(emb.id).match(/(\d+)/);
        return m ? m[1] : String(i + 1);
    }
    function onEmbryosUpdate(p) {
        _embryos = (p && Array.isArray(p.embryos)) ? p.embryos : [];
        // prune state for embryos that no longer exist
        const ids = new Set(_embryos.map(e => e.id));
        Object.keys(_states).forEach(id => { if (!ids.has(id)) delete _states[id]; });
        _embryos.forEach(e => { if (!_states[e.id]) _states[e.id] = 'marked'; });
        if (_selected && !ids.has(_selected)) selectEmbryo(null);
        renderList();
    }
    function renderList() {
        if (!_embryoList) return;
        _embryoCount.textContent = _embryos.length ? `(${_embryos.length})` : '';
        _embryoList.innerHTML = '';
        if (!_embryos.length) {
            const e = document.createElement('div'); e.className = 'op-empty';
            e.textContent = 'No embryos yet — mark some in step 1.';
            _embryoList.appendChild(e); return;
        }
        _embryos.forEach((emb, i) => {
            const xy = resolveXY(emb);
            const row = document.createElement('div');
            row.className = 'op-erow' + (emb.id === _selected ? ' op-erow-sel' : '');
            row.addEventListener('click', () => selectEmbryo(emb.id));

            const dot = document.createElement('span');
            dot.className = 'op-edot'; dot.textContent = labelFor(emb, i);
            dot.style.background = _ROLE_COLOR[emb.role] || _ROLE_COLOR.unassigned;
            row.appendChild(dot);

            const coord = document.createElement('span');
            coord.className = 'op-ecoord';
            coord.textContent = xy ? `(${xy.x.toFixed(0)}, ${xy.y.toFixed(0)}) µm` : 'no position';
            row.appendChild(coord);

            const chip = document.createElement('span');
            const st = _states[emb.id] || 'marked';
            chip.className = `op-echip op-echip-${st}`;
            chip.textContent = _STATE_LABEL[st] || st;
            row.appendChild(chip);

            _embryoList.appendChild(row);
        });
    }
    function setState(id, st) { if (id) { _states[id] = st; renderList(); } }

    // ---- Acquire (selected embryo) --------------------------------------
    function selectEmbryo(id) {
        _selected = id;
        const emb = _embryos.find(e => e.id === id);
        _selLabel.textContent = emb ? `— embryo ${labelFor(emb, _embryos.indexOf(emb))}` : '— select an embryo';
        _centerBtn.disabled = !emb;
        _acquireBtn.disabled = !emb || (_states[id] !== 'focused' && _states[id] !== 'imaged');
        renderList();
    }
    async function centerOnSelected() {
        const emb = _embryos.find(e => e.id === _selected);
        const xy = emb ? resolveXY(emb) : null;
        if (!xy) return;
        _centerBtn.disabled = true; _centerBtn.textContent = 'Centering…';
        try {
            await postJSON('/api/devices/stage/move', { x: xy.x, y: xy.y });
            setState(_selected, 'centered');
            toast(`Centred on embryo ${labelFor(emb, _embryos.indexOf(emb))}`);
        } catch (err) {
            toast(`Center failed (${err.status || err.message})`);
        } finally {
            _centerBtn.disabled = false; _centerBtn.textContent = 'Center stage on embryo';
        }
    }
    async function acquireSelected() {
        if (!_selected) return;
        _acquireBtn.disabled = true; _acquireBtn.textContent = 'Acquiring…';
        try {
            await postJSON('/api/devices/acquire/volume', { num_slices: 50, exposure_ms: 10.0 });
            setState(_selected, 'imaged');
            toast('Volume acquired');
        } catch (err) {
            toast(`Acquire failed (${err.status || err.message})`);
        } finally {
            _acquireBtn.disabled = false; _acquireBtn.textContent = 'Acquire volume';
        }
    }

    // ---- Focus Z nudges (fenced) ----------------------------------------
    async function nudgeBottomZ(delta) {
        try {
            const d = await postJSON('/api/devices/stage/bottom_z/nudge', { delta });
            if (d.position != null) _bzPos.textContent = Number(d.position).toFixed(1);
        } catch (err) { toast(`Bottom-Z nudge blocked (${err.status || err.message})`); }
    }
    async function nudgeFdrive(delta) {
        try {
            const d = await postJSON('/api/devices/spim/fdrive/nudge', { delta });
            if (d.position != null) _fdPos.textContent = Number(d.position).toFixed(1);
            if (d.distance_to_floor != null) _fdFloor.textContent = Number(d.distance_to_floor).toFixed(0);
            if (_selected) setState(_selected, 'focused');
            selectEmbryo(_selected);  // re-eval acquire enable
        } catch (err) { toast(`F-drive nudge blocked (${err.status || err.message})`); }
    }
    async function refreshZReadouts() {
        try { const b = await getJSON('/api/devices/stage/bottom_z'); if (b.position != null) _bzPos.textContent = Number(b.position).toFixed(1); } catch (_) {}
        try {
            const f = await getJSON('/api/devices/spim/fdrive');
            if (f.position != null) _fdPos.textContent = Number(f.position).toFixed(1);
            if (f.distance_to_floor != null) _fdFloor.textContent = Number(f.distance_to_floor).toFixed(0);
        } catch (_) {}
    }

    // ---- SPIM focus -----------------------------------------------------
    async function toggleSpim() {
        _spimToggle.disabled = true;
        try {
            const ep = _spimOn ? '/api/devices/lightsheet/live/stop' : '/api/devices/lightsheet/live/start';
            const data = await postJSON(ep, {});
            _spimOn = !!data.streaming;
            _spimToggle.textContent = _spimOn ? 'Stop view' : 'Start view';
            _spimToggle.classList.toggle('op-btn-on', _spimOn);
            if (!_spimOn) { _spimImg.classList.remove('has-frame'); _spimPh.style.display = ''; }
            else _spimPh.textContent = 'waiting…';
        } catch (err) { toast(`SPIM view toggle failed (${err.status || err.message})`); }
        finally { _spimToggle.disabled = false; }
    }
    function onLightsheetFrame(p) {
        if (!p || !p.jpeg_b64) return;
        _spimImg.src = `data:${p.mime || 'image/jpeg'};base64,${p.jpeg_b64}`;
        if (!_spimImg.classList.contains('has-frame')) { _spimImg.classList.add('has-frame'); _spimPh.style.display = 'none'; }
        if (p.focus_score != null) _spimScore.textContent = Number(p.focus_score).toFixed(3);
    }
    let _lsTimer = null;
    function postLsParams() {
        if (_lsTimer) clearTimeout(_lsTimer);
        _lsTimer = setTimeout(() => {
            postJSON('/api/devices/lightsheet/live/params',
                { galvo: _galvo, piezo: _piezo, exposure: 20, side: 'A' }).catch(() => {});
        }, 120);
    }
    function nudgeGalvo(d) { _galvo = Math.max(-5, Math.min(5, _galvo + d)); _gvVal.textContent = _galvo.toFixed(1); postLsParams(); }
    function nudgePiezo(d) { _piezo = Math.max(0, Math.min(200, _piezo + d)); _pzVal.textContent = _piezo.toFixed(0); postLsParams(); }
    async function toggleLed() {
        _ledOn = !_ledOn;
        _ledBtn.classList.toggle('op-btn-on', _ledOn); _ledBtn.setAttribute('aria-pressed', _ledOn ? 'true' : 'false');
        try { await postJSON('/api/devices/led/set', { state: _ledOn ? 'Open' : 'Closed' }); }
        catch (err) { toast(`LED failed (${err.status || err.message})`); }
    }

    // ---- Wiring / lifecycle --------------------------------------------
    function wire() {
        if (_wired) return; _wired = true;
        cacheDom();
        _camToggle.addEventListener('click', toggleCamera);
        _markCanvas.addEventListener('click', onCanvasClick);
        _detectBtn.addEventListener('click', runDetect);
        _confirmBtn.addEventListener('click', confirmMarks);
        _clearBtn.addEventListener('click', () => { _markers = []; drawMarkers(); updateMarkActions(); });
        _bzNudge.addEventListener('click', e => { const b = e.target.closest('[data-bz]'); if (b) nudgeBottomZ(Number(b.dataset.bz)); });
        _fdNudge.addEventListener('click', e => { const b = e.target.closest('[data-fd]'); if (b) nudgeFdrive(Number(b.dataset.fd)); });
        _centerBtn.addEventListener('click', centerOnSelected);
        _acquireBtn.addEventListener('click', acquireSelected);
        _spimToggle.addEventListener('click', toggleSpim);
        _ledBtn.addEventListener('click', toggleLed);
        document.querySelectorAll('[data-gv]').forEach(b => b.addEventListener('click', () => nudgeGalvo(Number(b.dataset.gv))));
        document.querySelectorAll('[data-pz]').forEach(b => b.addEventListener('click', () => nudgePiezo(Number(b.dataset.pz))));
        window.addEventListener('resize', () => { if (_marking) drawMarkers(); });

        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('BOTTOM_CAMERA_FRAME', onBottomFrame);
            ClientEventBus.on('LIGHTSHEET_FRAME', onLightsheetFrame);
            ClientEventBus.on('EMBRYOS_UPDATE', onEmbryosUpdate);
            ClientEventBus.on('EMBRYO_DETECTED', () => {});  // bulk EMBRYOS_UPDATE covers it
            ClientEventBus.on('DEVICE_STATE_UPDATE', p => {
                const pos = p && p.positions;
                if (pos && pos.xy_stage && Array.isArray(pos.xy_stage)) _lastXY = { X: pos.xy_stage[0], Y: pos.xy_stage[1] };
            });
        }
    }

    async function activate() {
        wire();
        if (_active) return;
        _active = true;
        // Bootstrap the canonical list (mid-session open).
        try { const s = await getJSON('/api/embryos/current'); onEmbryosUpdate(s); } catch (_) {}
        refreshZReadouts();
    }
    function deactivate() {
        if (!_active) return;
        _active = false;
        // Stop streams so they don't run while the view is hidden.
        if (_camOn) { fetch('/api/devices/bottom_camera/stream/stop', { method: 'POST' }).catch(() => {}); applyCameraState(false); }
        if (_spimOn) { fetch('/api/devices/lightsheet/live/stop', { method: 'POST' }).catch(() => {}); _spimOn = false; _spimToggle.textContent = 'Start view'; _spimToggle.classList.remove('op-btn-on'); }
        if (_marking) exitMarking();
    }

    return { activate, deactivate };
})();
