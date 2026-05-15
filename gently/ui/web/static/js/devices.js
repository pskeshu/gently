/**
 * Devices tab — read-only stream of MMCore device state.
 *
 * Two views:
 *   Map     — top-down XY stage map with the firmware-fence "optimal" box
 *             on a hatched "forbidden" envelope (default)
 *   Details — position cards + filterable properties table
 *
 * Both views are driven by the same DEVICE_STATE_UPDATE payload forwarded
 * from the device-layer SSE stream by DeviceStateMonitor. No write controls
 * — the optimal-zone outline reflects whatever the XYStage device reports
 * as its hardware soft limits (LowerLimX/UpperLimX/LowerLimY/UpperLimY in
 * the ASI adapter), which is the single source of truth.
 */
const DevicesManager = (function () {
    const STALE_AFTER_MS = 4000;
    const VIEWS = ['map', 'details'];
    const SVG_NS = 'http://www.w3.org/2000/svg';

    // Status / details DOM
    let _statusPill, _statusMeta, _statusDot;
    let _posX, _posY, _piezoZ, _galvoA, _galvoB;
    let _tbody, _filter;

    // Map DOM
    let _mapSvg, _mapBg, _mapGridMinor, _mapGridMajor, _mapAxisEmphasis;
    let _mapBeyond, _mapCoverslip;
    let _mapZones, _mapZoneLabels, _mapOrigin, _mapAxes;
    let _mapEmbryos;
    let _mapMarker, _mapMarkerPulse, _mapMarkerRing, _mapMarkerDot;
    let _mapReadoutX, _mapReadoutY;
    let _mapWrap;
    let _scalebarLabel;

    // Embryo waypoints — driven by EMBRYOS_UPDATE events (the canonical bulk
    // mutation broadcast added by the embryos-broadcast commit) and the
    // initial /api/embryos/current snapshot. Each entry mirrors
    // EmbryoState.to_dict() (id, position_coarse, position_fine,
    // has_fine_position, nickname, role, ...). Role drives marker color
    // (mirrors the marking-window legend: magenta=test, cyan=calibration,
    // grey=unassigned). EMBRYO_DETECTED / STATUS_CHANGED listeners stay
    // hooked as a belt-and-braces refresh path.
    let _embryos = [];
    const _ROLE_COLOR = {
        test: '#ff66cc',
        calibration: '#00cccc',
        unassigned: '#888888',
    };

    // Map-side edit state. _selectedEmbryoId means "picked up": the next
    // click on empty map space drops it there (with a confirm), Delete /
    // Backspace removes it (with a confirm), Escape clears the selection.
    let _selectedEmbryoId = null;

    // Bottom-camera panel DOM + state
    let _camPanel, _camToggle, _camImg, _camPlaceholder, _camLed, _camMeta;
    let _camStreaming = false;
    let _camLastFrameTs = 0;
    let _camHasFrame = false;
    let _camStaleTimer = null;
    const _CAM_FPS_WINDOW = 12;
    let _camFrameTimes = [];

    let _lastTs = 0;
    let _previousTs = 0;
    let _lastWallTs = 0;
    let _staleTimer = null;
    let _filterText = '';
    let _lastPropertyMap = {};
    let _lastXY = null;             // {X, Y} in stage µm, last seen
    let _currentView = 'map';

    // Map geometry
    //   _optimalBox: { x: [min, max], y: [min, max] } in stage µm, derived
    //   from the live XYStage firmware-fence properties. null until first
    //   DEVICE_STATE_UPDATE delivers them.
    //   _coverslip: optional {center_um, size_mm} from GET /api/devices/coverslip
    //   _viewBox: { xMin, xMax, yMin, yMax } in stage µm
    let _optimalBox = null;
    let _coverslip = null;
    let _viewBox = null;

    function cacheDom() {
        if (_statusPill) return;
        _statusPill = document.getElementById('devices-stream-pill');
        _statusMeta = document.getElementById('devices-status-meta');
        _statusDot  = document.getElementById('devices-status-dot');
        _posX       = document.getElementById('dev-pos-x');
        _posY       = document.getElementById('dev-pos-y');
        _piezoZ     = document.getElementById('dev-pos-piezo');
        _galvoA     = document.getElementById('dev-pos-galvo-a');
        _galvoB     = document.getElementById('dev-pos-galvo-b');
        _tbody      = document.getElementById('devices-tbody');
        _filter     = document.getElementById('devices-filter');

        _mapSvg           = document.getElementById('devices-map-svg');
        _mapBg            = document.getElementById('devices-map-bg');
        _mapBeyond        = document.getElementById('devices-map-beyond');
        _mapGridMinor     = document.getElementById('devices-map-grid-minor');
        _mapGridMajor     = document.getElementById('devices-map-grid-major');
        _mapAxisEmphasis  = document.getElementById('devices-map-axis-emphasis');
        _mapCoverslip     = document.getElementById('devices-map-coverslip');
        _mapZones         = document.getElementById('devices-map-zones');
        _mapZoneLabels    = document.getElementById('devices-map-zone-labels');
        _mapOrigin        = document.getElementById('devices-map-origin');
        _mapAxes          = document.getElementById('devices-map-axes');
        _mapEmbryos       = document.getElementById('devices-map-embryos');
        _mapMarker        = document.getElementById('devices-map-marker');
        _mapMarkerPulse   = document.getElementById('devices-map-marker-pulse');
        _mapMarkerRing    = document.getElementById('devices-map-marker-ring');
        _mapMarkerDot     = document.getElementById('devices-map-marker-dot');
        _mapReadoutX      = document.getElementById('devices-map-x');
        _mapReadoutY      = document.getElementById('devices-map-y');
        _mapWrap          = document.getElementById('devices-map-wrap');
        _scalebarLabel    = document.getElementById('devices-scalebar-value');

        _camPanel        = document.getElementById('devices-camera-panel');
        _camToggle       = document.getElementById('devices-camera-toggle');
        _camImg          = document.getElementById('devices-camera-img');
        _camPlaceholder  = document.getElementById('devices-camera-placeholder');
        _camLed          = document.getElementById('devices-camera-led');
        _camMeta         = document.getElementById('devices-camera-meta');

        // Recompute the scale bar caption whenever the canvas resizes.
        if (_mapSvg && window.ResizeObserver) {
            new ResizeObserver(() => updateScalebar()).observe(_mapSvg);
        }

        if (_filter) {
            _filter.addEventListener('input', () => {
                _filterText = _filter.value.trim().toLowerCase();
                renderPropertiesTable(_lastPropertyMap);
            });
        }
    }

    // =====================================================================
    // Status / formatting helpers
    // =====================================================================

    function fmtNumber(v, digits = 2) {
        if (v === null || v === undefined || Number.isNaN(v)) return '—';
        if (typeof v !== 'number') return String(v);
        return v.toFixed(digits);
    }

    function setStatus(klass, label, meta) {
        if (!_statusPill) return;
        _statusPill.classList.remove('live', 'stale', 'paused', 'error');
        _statusPill.classList.add(klass);
        _statusPill.textContent = label;
        if (_statusMeta) _statusMeta.textContent = meta || '';
        if (_statusDot) {
            _statusDot.classList.remove('live', 'stale', 'paused', 'error');
            _statusDot.classList.add(klass);
        }
    }

    function scheduleStaleCheck() {
        if (_staleTimer) clearTimeout(_staleTimer);
        _staleTimer = setTimeout(() => {
            const age = (Date.now() - _lastWallTs) / 1000;
            setStatus('stale', 'stale', `last ${age.toFixed(1)}s ago`);
        }, STALE_AFTER_MS);
    }

    function flashValue(el) {
        if (!el) return;
        el.classList.add('updating');
        setTimeout(() => el.classList.remove('updating'), 400);
    }

    function setAxis(el, value, digits) {
        if (!el) return;
        const formatted = fmtNumber(value, digits);
        if (el.textContent !== formatted) {
            el.textContent = formatted;
            flashValue(el);
        }
    }

    // =====================================================================
    // Live state handlers
    // =====================================================================

    function renderPositions(positions) {
        if (!positions) return;
        for (const devName of Object.keys(positions)) {
            const entry = positions[devName] || {};
            switch (entry.kind) {
                case 'xy_stage':
                    setAxis(_posX, entry.X, 2);
                    setAxis(_posY, entry.Y, 2);
                    _lastXY = { X: entry.X, Y: entry.Y };
                    if (computeViewBox()) renderMap();
                    else updateMapMarker();
                    if (_mapReadoutX) _mapReadoutX.textContent = fmtNumber(entry.X, 1);
                    if (_mapReadoutY) _mapReadoutY.textContent = fmtNumber(entry.Y, 1);
                    break;
                case 'piezo':
                    setAxis(_piezoZ, entry.Position, 3);
                    break;
                case 'galvo':
                    setAxis(_galvoA, entry.A, 4);
                    setAxis(_galvoB, entry.B, 4);
                    break;
            }
        }
    }

    // Extract the optimal-box outline from the XYStage device's firmware-fence
    // properties (LowerLimX/UpperLimX/LowerLimY/UpperLimY, all in mm on the
    // ASI adapter). Single source of truth: these are what the controller
    // enforces against every motion source (joystick, MMCore, plans). When
    // their values change, the map redraws.
    function extractOptimalBox(propsByDevice) {
        if (!propsByDevice) return null;
        for (const name of Object.keys(propsByDevice)) {
            const p = propsByDevice[name] || {};
            const xMinMm = parseFloat(p['LowerLimX(mm)']);
            const xMaxMm = parseFloat(p['UpperLimX(mm)']);
            const yMinMm = parseFloat(p['LowerLimY(mm)']);
            const yMaxMm = parseFloat(p['UpperLimY(mm)']);
            if (isFinite(xMinMm) && isFinite(xMaxMm) &&
                isFinite(yMinMm) && isFinite(yMaxMm)) {
                return {
                    x: [xMinMm * 1000, xMaxMm * 1000],   // mm → µm
                    y: [yMinMm * 1000, yMaxMm * 1000],
                };
            }
        }
        return null;
    }

    function applyOptimalBoxFromProperties(propsByDevice) {
        const next = extractOptimalBox(propsByDevice);
        if (!next) return;
        // Treat differences below 1 µm as noise — don't churn redraws.
        const prev = _optimalBox;
        const changed = !prev
            || Math.abs(prev.x[0] - next.x[0]) > 1
            || Math.abs(prev.x[1] - next.x[1]) > 1
            || Math.abs(prev.y[0] - next.y[0]) > 1
            || Math.abs(prev.y[1] - next.y[1]) > 1;
        if (!changed) return;
        _optimalBox = next;
        computeViewBox();
        renderMap();
    }

    // =====================================================================
    // Coverslip (decorative reference)
    // =====================================================================

    async function loadCoverslip() {
        try {
            const res = await fetch('/api/devices/coverslip');
            if (!res.ok) return;
            const data = await res.json();
            _coverslip = data.coverslip || null;
            computeViewBox();
            renderMap();
        } catch (err) {
            console.debug('coverslip fetch failed:', err);
        }
    }

    // Initial embryo snapshot — closes the gap for clients that connect
    // mid-session, after the last EMBRYOS_UPDATE has already been broadcast
    // and aged out of history. Subsequent updates arrive over the event bus.
    async function loadEmbryosSnapshot() {
        try {
            const res = await fetch('/api/embryos/current');
            if (!res.ok) return;
            const data = await res.json();
            handleEmbryosUpdate(data);
        } catch (err) {
            console.debug('embryos snapshot fetch failed:', err);
        }
    }

    function handleEmbryosUpdate(payload) {
        _embryos = (payload && Array.isArray(payload.embryos)) ? payload.embryos : [];
        if (!_viewBox) {
            computeViewBox();
            renderMap();
        } else {
            renderEmbryos();
        }
    }

    // =====================================================================
    // Properties table (Details view)
    // =====================================================================

    const DEVICE_TYPE_LABEL = {
        0: 'Unknown', 1: 'Any', 2: 'Camera', 3: 'Shutter',
        4: 'XY', 5: 'Stage', 6: 'State', 7: 'Serial',
        8: 'Generic', 9: 'AutoFocus', 10: 'Core', 11: 'Image',
        12: 'Signal IO', 13: 'Magnifier', 14: 'SLM', 15: 'Hub',
        16: 'Galvo',
    };

    function flattenProperties(propsByDevice) {
        const rows = [];
        const prevMap = _lastPropertyMap;
        const devNames = Object.keys(propsByDevice).sort();
        for (const dev of devNames) {
            const bundle = propsByDevice[dev] || {};
            const typeCode = bundle.__type__;
            const type = (typeCode in DEVICE_TYPE_LABEL) ? DEVICE_TYPE_LABEL[typeCode] : String(typeCode ?? '');
            const prev = prevMap[dev] || {};
            const propNames = Object.keys(bundle).filter(k => k !== '__type__').sort();
            for (const prop of propNames) {
                const value = bundle[prop];
                const changed = prev[prop] !== undefined && prev[prop] !== value;
                rows.push({ device: dev, type, property: prop, value, changed });
            }
        }
        return rows;
    }

    function renderPropertiesTable(propsByDevice) {
        if (!_tbody || !propsByDevice) return;
        const rows = flattenProperties(propsByDevice);
        const filtered = _filterText
            ? rows.filter(r =>
                r.device.toLowerCase().includes(_filterText) ||
                r.property.toLowerCase().includes(_filterText) ||
                String(r.value).toLowerCase().includes(_filterText))
            : rows;
        if (!filtered.length) {
            _tbody.innerHTML = '<tr class="devices-empty-row"><td colspan="4">' +
                (rows.length ? 'No matches for filter' : 'No properties yet') + '</td></tr>';
            return;
        }
        const html = filtered.map(r =>
            '<tr>' +
                '<td class="dev-col-device">' + escapeHtml(r.device) + '</td>' +
                '<td class="dev-col-type">' + escapeHtml(r.type) + '</td>' +
                '<td class="dev-col-prop">' + escapeHtml(r.property) + '</td>' +
                '<td class="dev-col-value' + (r.changed ? ' changed' : '') + '">' + escapeHtml(r.value) + '</td>' +
            '</tr>'
        ).join('');
        _tbody.innerHTML = html;
    }

    // =====================================================================
    // Map view — geometry + rendering
    // =====================================================================

    // Compute the viewport extent. Fits the union of:
    //   - the optimal box (firmware fence), if known
    //   - the live stage XY, if known
    // Returns true if the viewBox changed meaningfully (≥0.5% drift) so the
    // caller knows to rebuild zones/grid/axes; cheap micro-jitter doesn't
    // trigger a full redraw.
    function computeViewBox() {
        let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
        if (_optimalBox) {
            xMin = Math.min(xMin, _optimalBox.x[0]);
            xMax = Math.max(xMax, _optimalBox.x[1]);
            yMin = Math.min(yMin, _optimalBox.y[0]);
            yMax = Math.max(yMax, _optimalBox.y[1]);
        }
        if (_lastXY) {
            xMin = Math.min(xMin, _lastXY.X); xMax = Math.max(xMax, _lastXY.X);
            yMin = Math.min(yMin, _lastXY.Y); yMax = Math.max(yMax, _lastXY.Y);
        }
        if (!isFinite(xMin) || !isFinite(yMin)) {
            xMin = -100; xMax = 100; yMin = -100; yMax = 100;
        }
        if (xMax - xMin < 1) { const c = (xMax + xMin) / 2; xMin = c - 100; xMax = c + 100; }
        if (yMax - yMin < 1) { const c = (yMax + yMin) / 2; yMin = c - 100; yMax = c + 100; }
        const span = Math.max(xMax - xMin, yMax - yMin);
        const pad = span * 0.08;
        const next = {
            xMin: xMin - pad, xMax: xMax + pad,
            yMin: yMin - pad, yMax: yMax + pad,
        };
        const changed = !_viewBox ||
            Math.abs(next.xMin - _viewBox.xMin) > span * 0.005 ||
            Math.abs(next.xMax - _viewBox.xMax) > span * 0.005 ||
            Math.abs(next.yMin - _viewBox.yMin) > span * 0.005 ||
            Math.abs(next.yMax - _viewBox.yMax) > span * 0.005;
        _viewBox = next;
        return changed;
    }

    // Stage Y is positive-up; SVG Y is positive-down. Convert by negation.
    function svgY(stageY) { return -stageY; }

    function renderMap() {
        if (!_mapSvg || !_viewBox) return;
        const { xMin, xMax, yMin, yMax } = _viewBox;
        const w = xMax - xMin, h = yMax - yMin;
        _mapSvg.setAttribute('viewBox', `${xMin} ${-yMax} ${w} ${h}`);

        const span = Math.max(w, h);
        if (_mapMarkerDot)   _mapMarkerDot.setAttribute('r',   span * 0.0045);
        if (_mapMarkerRing)  _mapMarkerRing.setAttribute('r',  span * 0.014);
        if (_mapMarkerPulse) _mapMarkerPulse.setAttribute('r', span * 0.014);

        renderBackground();
        renderBeyond();
        renderGrid();
        renderAxisEmphasis();
        renderCoverslip();
        renderZones();
        renderZoneLabels();
        renderOrigin();
        renderAxes();
        renderEmbryos();
        updateMapMarker();
        updateScalebar();
    }

    function renderEmbryos() {
        if (!_mapEmbryos || !_viewBox) return;
        _mapEmbryos.innerHTML = '';
        if (!_embryos || _embryos.length === 0) return;

        const { xMin, xMax, yMin, yMax } = _viewBox;
        const span = Math.max(xMax - xMin, yMax - yMin);
        const r = span * 0.006;              // embryo dot radius (stage µm)
        const ringR = r * 1.9;               // accent ring
        const labelFontSize = span * 0.012;

        for (const emb of _embryos) {
            if (emb.x == null || emb.y == null) continue;
            const color = _ROLE_COLOR[emb.role] || _ROLE_COLOR.test;
            // Group so we can attach the title (tooltip).
            const g = document.createElementNS(SVG_NS, 'g');
            g.setAttribute('class', `devices-embryo devices-embryo-${emb.role || 'test'}`);

            // Soft outer ring — makes the marker visible even on dense
            // background grids without overpowering the position marker.
            const ring = document.createElementNS(SVG_NS, 'circle');
            ring.setAttribute('cx', emb.x);
            ring.setAttribute('cy', svgY(emb.y));
            ring.setAttribute('r', ringR);
            ring.setAttribute('fill', 'none');
            ring.setAttribute('stroke', color);
            ring.setAttribute('stroke-opacity', '0.45');
            ring.setAttribute('stroke-width', r * 0.35);
            g.appendChild(ring);

            const dot = document.createElementNS(SVG_NS, 'circle');
            dot.setAttribute('cx', emb.x);
            dot.setAttribute('cy', svgY(emb.y));
            dot.setAttribute('r', r);
            dot.setAttribute('fill', color);
            dot.setAttribute('fill-opacity', '0.9');
            dot.setAttribute('stroke', '#000');
            dot.setAttribute('stroke-opacity', '0.5');
            dot.setAttribute('stroke-width', r * 0.18);
            g.appendChild(dot);

            // Label — embryo id, small, just above the dot.
            const label = document.createElementNS(SVG_NS, 'text');
            label.setAttribute('x', emb.x);
            label.setAttribute('y', svgY(emb.y + r * 2.2));
            label.setAttribute('font-size', labelFontSize);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('class', 'devices-embryo-label');
            label.setAttribute('fill', color);
            label.textContent = emb.user_label || emb.embryo_id || '';
            g.appendChild(label);

            const title = document.createElementNS(SVG_NS, 'title');
            const role = emb.role || 'test';
            const parts = [
                `${emb.user_label || emb.embryo_id}`,
                `role: ${role}`,
                `(${emb.x.toFixed(1)}, ${emb.y.toFixed(1)}) µm`,
            ];
            if (emb.cadence_phase) parts.push(`phase: ${emb.cadence_phase}`);
            title.textContent = parts.join('\n');
            g.appendChild(title);

            _mapEmbryos.appendChild(g);
        }
    }

    async function loadEmbryos() {
        try {
            const res = await fetch('/api/embryos/positions');
            if (!res.ok) return;
            const data = await res.json();
            _embryos = Array.isArray(data.embryos) ? data.embryos : [];
            renderEmbryos();
        } catch (err) {
            console.debug('embryo positions fetch failed:', err);
        }
    }

    function _upsertEmbryo(payload) {
        const eid = payload && payload.embryo_id;
        if (!eid) return false;
        const idx = _embryos.findIndex(e => e.embryo_id === eid);
        const existing = idx >= 0 ? _embryos[idx] : null;
        const merged = Object.assign({}, existing || {}, {
            embryo_id: eid,
            role: payload.role || (existing && existing.role) || 'test',
        });
        if (payload.x != null) merged.x = payload.x;
        if (payload.y != null) merged.y = payload.y;
        if (payload.user_label !== undefined) merged.user_label = payload.user_label;
        if (payload.confidence !== undefined) merged.confidence = payload.confidence;
        if (payload.cadence_phase !== undefined) merged.cadence_phase = payload.cadence_phase;
        if (idx >= 0) {
            _embryos[idx] = merged;
        } else {
            _embryos.push(merged);
        }
        return true;
    }

    function handleEmbryoDetected(payload) {
        if (_upsertEmbryo(payload)) renderEmbryos();
    }

    function handleStatusChanged(payload) {
        // Only the role-assignment variant is relevant to the map.
        if (!payload || payload.change !== 'role_assigned') return;
        const eid = payload.embryo_id;
        const newRole = payload.new_role;
        if (!eid || !newRole) return;
        const idx = _embryos.findIndex(e => e.embryo_id === eid);
        if (idx >= 0) {
            _embryos[idx] = Object.assign({}, _embryos[idx], { role: newRole });
            renderEmbryos();
        } else {
            // Embryo we haven't seen yet — refetch to be safe.
            loadEmbryos();
        }
    }

    function renderBackground() {
        if (!_mapBg || !_viewBox) return;
        _mapBg.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        const grain = document.createElementNS(SVG_NS, 'rect');
        grain.setAttribute('x', xMin);
        grain.setAttribute('y', svgY(yMax));
        grain.setAttribute('width',  xMax - xMin);
        grain.setAttribute('height', yMax - yMin);
        grain.setAttribute('fill', 'url(#devices-map-grain)');
        _mapBg.appendChild(grain);

        const vignette = document.createElementNS(SVG_NS, 'rect');
        vignette.setAttribute('x', xMin);
        vignette.setAttribute('y', svgY(yMax));
        vignette.setAttribute('width',  xMax - xMin);
        vignette.setAttribute('height', yMax - yMin);
        vignette.setAttribute('fill', 'url(#devices-map-vignette)');
        vignette.setAttribute('pointer-events', 'none');
        _mapBg.appendChild(vignette);
    }

    // "Forbidden" is implicit: paint the whole viewport with the red hatch
    // pattern. The optimal zone rect paints ABOVE this so the operator's safe
    // window looks carved out of a hatched danger envelope.
    function renderBeyond() {
        if (!_mapBeyond || !_viewBox) return;
        _mapBeyond.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        const span = Math.max(xMax - xMin, yMax - yMin);
        const tile = Math.max(8, span / 50);

        const pattern = document.getElementById('devices-map-hatch-red');
        if (pattern) {
            pattern.setAttribute('width',  tile);
            pattern.setAttribute('height', tile);
            const line = pattern.querySelector('line');
            if (line) line.setAttribute('y2', tile);
        }

        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', xMin);
        rect.setAttribute('y', svgY(yMax));
        rect.setAttribute('width',  xMax - xMin);
        rect.setAttribute('height', yMax - yMin);
        rect.setAttribute('class', 'devices-beyond-fill');
        _mapBeyond.appendChild(rect);
    }

    function renderCoverslip() {
        if (!_mapCoverslip) return;
        _mapCoverslip.innerHTML = '';
        const cs = _coverslip;
        if (!cs || !Array.isArray(cs.center_um) || !Array.isArray(cs.size_mm)) return;
        const [cx, cy] = cs.center_um;
        const [wMm, hMm] = cs.size_mm;
        const w = wMm * 1000, h = hMm * 1000;
        const x0 = cx - w / 2;
        const y1 = cy + h / 2;

        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', x0);
        rect.setAttribute('y', svgY(y1));
        rect.setAttribute('width',  w);
        rect.setAttribute('height', h);
        rect.setAttribute('class', 'devices-coverslip-outline');
        _mapCoverslip.appendChild(rect);

        const span = Math.max(_viewBox.xMax - _viewBox.xMin, _viewBox.yMax - _viewBox.yMin);
        const fs = span * 0.015;
        const label = document.createElementNS(SVG_NS, 'text');
        label.setAttribute('x', x0 + w - span * 0.008);
        label.setAttribute('y', svgY(y1) + fs * 1.1);
        label.setAttribute('class', 'devices-coverslip-label');
        label.setAttribute('font-size', fs);
        label.setAttribute('text-anchor', 'end');
        label.textContent = `coverslip · ${wMm}×${hMm} mm`;
        _mapCoverslip.appendChild(label);
    }

    function renderZones() {
        if (!_mapZones) return;
        _mapZones.innerHTML = '';
        if (!_optimalBox) return;
        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', _optimalBox.x[0]);
        rect.setAttribute('y', svgY(_optimalBox.y[1]));
        rect.setAttribute('width',  _optimalBox.x[1] - _optimalBox.x[0]);
        rect.setAttribute('height', _optimalBox.y[1] - _optimalBox.y[0]);
        rect.setAttribute('class', 'devices-zone devices-zone-green');
        _mapZones.appendChild(rect);
    }

    function renderZoneLabels() {
        if (!_mapZoneLabels || !_viewBox || !_optimalBox) return;
        _mapZoneLabels.innerHTML = '';
        const zw = _optimalBox.x[1] - _optimalBox.x[0];
        const zh = _optimalBox.y[1] - _optimalBox.y[0];
        const span = Math.max(_viewBox.xMax - _viewBox.xMin, _viewBox.yMax - _viewBox.yMin);
        const fontSize = span * 0.018;
        const inset    = span * 0.010;
        const text = 'OPTIMAL';
        const estTextWidth = text.length * fontSize * 0.78;
        if (zw < estTextWidth * 1.15 || zh < fontSize * 2.0) return;
        const tx = _optimalBox.x[0] + inset;
        const ty = _optimalBox.y[1] - inset;
        const t = document.createElementNS(SVG_NS, 'text');
        t.setAttribute('x', tx);
        t.setAttribute('y', svgY(ty) + fontSize * 0.85);
        t.setAttribute('class', 'devices-zone-label devices-zone-green');
        t.setAttribute('font-size', fontSize);
        t.setAttribute('text-anchor', 'start');
        t.textContent = text;
        _mapZoneLabels.appendChild(t);
    }

    function renderOrigin() {
        if (!_mapOrigin || !_viewBox) return;
        _mapOrigin.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        if (xMin > 0 || xMax < 0 || yMin > 0 || yMax < 0) return;
        const span = Math.max(xMax - xMin, yMax - yMin);
        const arm = span * 0.012;
        const path = document.createElementNS(SVG_NS, 'path');
        path.setAttribute('d',
            `M ${-arm} 0 L ${arm} 0  M 0 ${-arm} L 0 ${arm}`);
        path.setAttribute('class', 'devices-origin-mark');
        _mapOrigin.appendChild(path);
        const label = document.createElementNS(SVG_NS, 'text');
        label.setAttribute('x', arm * 1.4);
        label.setAttribute('y', -arm * 0.6);
        label.setAttribute('class', 'devices-origin-label');
        label.setAttribute('font-size', span * 0.013);
        label.textContent = '0,0';
        _mapOrigin.appendChild(label);
    }

    function niceStep(span, divisions = 6) {
        const target = span / divisions;
        const pow = Math.pow(10, Math.floor(Math.log10(target)));
        const norm = target / pow;
        const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
        return step * pow;
    }

    function renderGrid() {
        if (!_mapGridMinor || !_mapGridMajor || !_viewBox) return;
        _mapGridMinor.innerHTML = '';
        _mapGridMajor.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        const span = Math.max(xMax - xMin, yMax - yMin);
        const major = niceStep(span);
        const minor = major / 5;
        for (let x = Math.ceil(xMin / minor) * minor; x <= xMax; x += minor) {
            const isMajor = Math.abs(x % major) < minor * 0.001 || Math.abs(x % major - major) < minor * 0.001;
            const line = document.createElementNS(SVG_NS, 'line');
            line.setAttribute('x1', x); line.setAttribute('x2', x);
            line.setAttribute('y1', svgY(yMax)); line.setAttribute('y2', svgY(yMin));
            line.setAttribute('class', isMajor ? 'devices-grid-major' : 'devices-grid-minor');
            (isMajor ? _mapGridMajor : _mapGridMinor).appendChild(line);
        }
        for (let y = Math.ceil(yMin / minor) * minor; y <= yMax; y += minor) {
            const isMajor = Math.abs(y % major) < minor * 0.001 || Math.abs(y % major - major) < minor * 0.001;
            const line = document.createElementNS(SVG_NS, 'line');
            line.setAttribute('y1', svgY(y)); line.setAttribute('y2', svgY(y));
            line.setAttribute('x1', xMin); line.setAttribute('x2', xMax);
            line.setAttribute('class', isMajor ? 'devices-grid-major' : 'devices-grid-minor');
            (isMajor ? _mapGridMajor : _mapGridMinor).appendChild(line);
        }
    }

    function renderAxisEmphasis() {
        if (!_mapAxisEmphasis || !_viewBox) return;
        _mapAxisEmphasis.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        if (xMin <= 0 && xMax >= 0) {
            const l = document.createElementNS(SVG_NS, 'line');
            l.setAttribute('x1', 0); l.setAttribute('x2', 0);
            l.setAttribute('y1', svgY(yMax)); l.setAttribute('y2', svgY(yMin));
            l.setAttribute('class', 'devices-axis-emphasis');
            _mapAxisEmphasis.appendChild(l);
        }
        if (yMin <= 0 && yMax >= 0) {
            const l = document.createElementNS(SVG_NS, 'line');
            l.setAttribute('y1', svgY(0)); l.setAttribute('y2', svgY(0));
            l.setAttribute('x1', xMin); l.setAttribute('x2', xMax);
            l.setAttribute('class', 'devices-axis-emphasis');
            _mapAxisEmphasis.appendChild(l);
        }
    }

    function renderAxes() {
        if (!_mapAxes || !_viewBox) return;
        _mapAxes.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        const span = Math.max(xMax - xMin, yMax - yMin);
        const step = niceStep(span);
        const tickLen = span * 0.010;
        const fontSize = span * 0.018;
        const bottomSvg = svgY(yMin);
        for (let x = Math.ceil(xMin / step) * step; x <= xMax; x += step) {
            const t = document.createElementNS(SVG_NS, 'line');
            t.setAttribute('x1', x); t.setAttribute('x2', x);
            t.setAttribute('y1', bottomSvg); t.setAttribute('y2', bottomSvg - tickLen);
            t.setAttribute('class', 'devices-axis-tick');
            _mapAxes.appendChild(t);
            const label = document.createElementNS(SVG_NS, 'text');
            label.setAttribute('x', x);
            label.setAttribute('y', bottomSvg - tickLen * 1.8);
            label.setAttribute('class', 'devices-axis-label-svg');
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', fontSize);
            label.textContent = formatTickValue(x);
            _mapAxes.appendChild(label);
        }
        for (let y = Math.ceil(yMin / step) * step; y <= yMax; y += step) {
            const ySvg = svgY(y);
            const t = document.createElementNS(SVG_NS, 'line');
            t.setAttribute('y1', ySvg); t.setAttribute('y2', ySvg);
            t.setAttribute('x1', xMin); t.setAttribute('x2', xMin + tickLen);
            t.setAttribute('class', 'devices-axis-tick');
            _mapAxes.appendChild(t);
            const label = document.createElementNS(SVG_NS, 'text');
            label.setAttribute('x', xMin + tickLen * 1.8);
            label.setAttribute('y', ySvg);
            label.setAttribute('class', 'devices-axis-label-svg');
            label.setAttribute('text-anchor', 'start');
            label.setAttribute('dominant-baseline', 'middle');
            label.setAttribute('font-size', fontSize);
            label.textContent = formatTickValue(y);
            _mapAxes.appendChild(label);
        }
    }

    function formatTickValue(v) {
        const a = Math.abs(v);
        if (a >= 10000) return (v / 1000).toFixed(a >= 100000 ? 0 : 1) + 'k';
        return Math.round(v).toString();
    }

    // =====================================================================
    // Embryo waypoints
    // =====================================================================

    // "embryo_007" / "embryo_7" -> 7. Falls back to a 1-based index from the
    // caller so the label always shows *something*, even for stray ids.
    function embryoLabelText(id, fallbackIndex) {
        const m = id && String(id).match(/(\d+)/);
        if (m) {
            const n = parseInt(m[1], 10);
            if (Number.isFinite(n)) return String(n);
        }
        return String(fallbackIndex + 1);
    }

    // Resolve XY for rendering — fine if SPIM-aligned, else coarse. Returns
    // null when neither stage carries usable values so the entry is skipped
    // (e.g. an embryo whose detection record came in malformed).
    function embryoResolvedXY(emb) {
        const f = emb && emb.position_fine;
        if (f && Number.isFinite(f.x) && Number.isFinite(f.y)) return { x: f.x, y: f.y };
        const c = emb && emb.position_coarse;
        if (c && Number.isFinite(c.x) && Number.isFinite(c.y)) return { x: c.x, y: c.y };
        return null;
    }

    function renderEmbryos() {
        if (!_mapEmbryos || !_viewBox) return;
        _mapEmbryos.innerHTML = '';
        if (!_embryos || !_embryos.length) return;
        const span = Math.max(_viewBox.xMax - _viewBox.xMin,
                              _viewBox.yMax - _viewBox.yMin);
        const radius = span * 0.012;
        const fontSize = span * 0.015;

        _embryos.forEach((emb, i) => {
            const xy = embryoResolvedXY(emb);
            if (!xy) return;

            const isFine = !!emb.has_fine_position;
            const isSelected = _selectedEmbryoId !== null
                            && emb.id === _selectedEmbryoId;

            // Wrap circle + label in a group so a single closest() lookup
            // finds the embryo regardless of which child the click hit.
            const group = document.createElementNS(SVG_NS, 'g');
            group.setAttribute('class',
                'devices-embryo-group' + (isSelected ? ' devices-embryo-selected' : ''));
            group.setAttribute('data-embryo-id', emb.id || '');
            group.setAttribute('data-embryo-stage', isFine ? 'fine' : 'coarse');

            const circle = document.createElementNS(SVG_NS, 'circle');
            circle.setAttribute('cx', xy.x);
            circle.setAttribute('cy', svgY(xy.y));
            circle.setAttribute('r', radius);
            circle.setAttribute('class',
                isFine ? 'devices-embryo-disc' : 'devices-embryo-ring');
            group.appendChild(circle);

            const label = document.createElementNS(SVG_NS, 'text');
            label.setAttribute('x', xy.x);
            label.setAttribute('y', svgY(xy.y));
            label.setAttribute('class', 'devices-embryo-label');
            label.setAttribute('font-size', fontSize);
            label.textContent = embryoLabelText(emb.id, i);
            group.appendChild(label);

            _mapEmbryos.appendChild(group);
        });
    }

    // ---- Map-side edit interactions ------------------------------------
    // Convert a pointer event's client coords into stage µm. SVG y axis is
    // positive-down and stage y is positive-up, so the y component is
    // negated to match the convention used elsewhere in this module.
    function eventToStageXY(event) {
        if (!_mapSvg || !_mapSvg.getScreenCTM) return null;
        const ctm = _mapSvg.getScreenCTM();
        if (!ctm) return null;
        const pt = _mapSvg.createSVGPoint();
        pt.x = event.clientX;
        pt.y = event.clientY;
        const local = pt.matrixTransform(ctm.inverse());
        return { x: local.x, y: -local.y };
    }

    function findEmbryoIdAt(target) {
        if (!target) return null;
        const node = target.closest && target.closest('[data-embryo-id]');
        return node ? node.getAttribute('data-embryo-id') : null;
    }

    function embryoById(id) {
        return _embryos.find(e => e.id === id) || null;
    }

    function embryoNumberFor(emb) {
        return embryoLabelText(emb.id, _embryos.indexOf(emb));
    }

    function setSelectedEmbryo(id) {
        if (_selectedEmbryoId === id) return;
        _selectedEmbryoId = id;
        renderEmbryos();
    }

    function clearSelection() {
        if (_selectedEmbryoId === null) return;
        _selectedEmbryoId = null;
        renderEmbryos();
    }

    async function attemptMoveSelected(targetStage) {
        const id = _selectedEmbryoId;
        if (!id) return;
        const emb = embryoById(id);
        if (!emb) { clearSelection(); return; }
        const cur = embryoResolvedXY(emb);
        const num = embryoNumberFor(emb);
        const oldStr = cur ? `(${cur.x.toFixed(1)}, ${cur.y.toFixed(1)})` : '(unknown)';
        const newStr = `(${targetStage.x.toFixed(1)}, ${targetStage.y.toFixed(1)})`;
        if (!window.confirm(`Move embryo ${num} from ${oldStr} to ${newStr}?`)) {
            return;  // keep the embryo picked up so they can try again
        }
        try {
            const res = await fetch(`/api/embryos/${encodeURIComponent(id)}/position`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: targetStage.x, y: targetStage.y }),
            });
            if (!res.ok) {
                window.alert(`Move failed (${res.status}): ${await res.text()}`);
                return;
            }
            // EMBRYOS_UPDATE will arrive over the bus and refresh the layer;
            // dropping clears the picked-up state regardless.
            clearSelection();
        } catch (err) {
            console.error('move embryo:', err);
            window.alert(`Move failed: ${err.message}`);
        }
    }

    async function attemptDeleteSelected() {
        const id = _selectedEmbryoId;
        if (!id) return;
        const emb = embryoById(id);
        const num = emb ? embryoNumberFor(emb) : id;
        if (!window.confirm(`Remove embryo ${num}?`)) return;
        try {
            const res = await fetch(`/api/embryos/${encodeURIComponent(id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                window.alert(`Delete failed (${res.status}): ${await res.text()}`);
                return;
            }
            // The embryo is gone from the server snapshot; EMBRYOS_UPDATE
            // will arrive and drop it from _embryos. Clear locally too.
            _selectedEmbryoId = null;
        } catch (err) {
            console.error('delete embryo:', err);
            window.alert(`Delete failed: ${err.message}`);
        }
    }

    function onMapPointerDown(event) {
        // Ignore non-primary buttons so right-clicks etc. don't trigger UI.
        if (event.button !== undefined && event.button !== 0) return;
        const id = findEmbryoIdAt(event.target);
        if (id) {
            setSelectedEmbryo(id);
            return;
        }
        // Empty-space click: drop the picked-up embryo here.
        if (_selectedEmbryoId !== null) {
            const stage = eventToStageXY(event);
            if (stage) attemptMoveSelected(stage);
        }
    }

    function onMapKeyDown(event) {
        // Only honour keys when the operator is actually looking at the Map:
        // not on another top-level tab, not on the Details subview, and not
        // typing into an input / textarea / select / contenteditable.
        if (typeof state !== 'undefined' && typeof TABS !== 'undefined'
                && state.tab !== TABS.DEVICES) {
            return;
        }
        if (_currentView !== 'map') return;
        const a = document.activeElement;
        if (a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' ||
                  a.tagName === 'SELECT' || a.isContentEditable)) {
            return;
        }
        if (event.key === 'Escape') {
            if (_selectedEmbryoId !== null) {
                clearSelection();
                event.preventDefault();
            }
            return;
        }
        if (_selectedEmbryoId === null) return;
        if (event.key === 'Delete' || event.key === 'Backspace') {
            event.preventDefault();  // Backspace would otherwise navigate back
            attemptDeleteSelected();
        }
    }

    function updateMapMarker() {
        if (!_mapMarker || !_lastXY) return;
        const sx = _lastXY.X;
        const sy = svgY(_lastXY.Y);
        _mapMarkerDot.setAttribute('cx',   sx); _mapMarkerDot.setAttribute('cy',   sy);
        _mapMarkerRing.setAttribute('cx',  sx); _mapMarkerRing.setAttribute('cy',  sy);
        _mapMarkerPulse.setAttribute('cx', sx); _mapMarkerPulse.setAttribute('cy', sy);
        _mapMarker.classList.remove('hidden');
    }

    function updateScalebar() {
        if (!_scalebarLabel || !_mapSvg || !_viewBox) return;
        const rect = _mapSvg.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const w = _viewBox.xMax - _viewBox.xMin;
        const h = _viewBox.yMax - _viewBox.yMin;
        const scale = Math.min(rect.width / w, rect.height / h);
        if (!isFinite(scale) || scale <= 0) return;
        const targetPx = 120;
        const rawUm = targetPx / scale;
        const pow = Math.pow(10, Math.floor(Math.log10(rawUm)));
        const norm = rawUm / pow;
        const snap = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
        const um = snap * pow;
        _scalebarLabel.textContent = formatScale(um);
    }
    function formatScale(um) {
        if (um >= 1000) return (um / 1000).toFixed(um >= 10000 ? 0 : 1).replace(/\.0$/, '') + ' mm';
        return Math.round(um) + ' µm';
    }

    // =====================================================================
    // Bottom-camera panel
    // =====================================================================

    async function toggleCameraStream() {
        if (!_camToggle) return;
        _camToggle.disabled = true;
        try {
            const endpoint = _camStreaming
                ? '/api/devices/bottom_camera/stream/stop'
                : '/api/devices/bottom_camera/stream/start';
            const res = await fetch(endpoint, { method: 'POST' });
            if (!res.ok) {
                const detail = await res.text();
                console.error('Bottom-camera toggle failed:', detail);
                _camMeta.textContent = `error: ${res.status}`;
                return;
            }
            const data = await res.json();
            applyCameraState(!!data.streaming);
        } catch (err) {
            console.error('Bottom-camera toggle failed:', err);
            _camMeta.textContent = `error: ${err}`;
        } finally {
            _camToggle.disabled = false;
        }
    }

    function applyCameraState(streaming) {
        _camStreaming = streaming;
        if (_camToggle) {
            _camToggle.textContent = streaming ? 'Stop' : 'Start';
            _camToggle.classList.toggle('active', streaming);
        }
        if (_camLed) {
            _camLed.classList.toggle('live', streaming);
            _camLed.classList.remove('stale');
        }
        if (!streaming) {
            _camHasFrame = false;
            _camFrameTimes = [];
            if (_camImg) _camImg.classList.remove('has-frame');
            if (_camPlaceholder) _camPlaceholder.hidden = false;
            if (_camMeta) _camMeta.textContent = 'stream off';
            if (_camStaleTimer) { clearTimeout(_camStaleTimer); _camStaleTimer = null; }
        } else {
            _camFrameTimes = [];
            if (_camMeta) _camMeta.textContent = 'waiting…';
        }
    }

    function handleCameraFrame(payload) {
        if (!payload || !payload.jpeg_b64 || !_camImg) return;
        _camImg.src = `data:${payload.mime || 'image/jpeg'};base64,${payload.jpeg_b64}`;
        if (!_camHasFrame) {
            _camHasFrame = true;
            _camImg.classList.add('has-frame');
            if (_camPlaceholder) _camPlaceholder.hidden = true;
        }
        const now = performance.now();
        _camLastFrameTs = Date.now();
        _camFrameTimes.push(now);
        if (_camFrameTimes.length > _CAM_FPS_WINDOW) _camFrameTimes.shift();
        if (_camLed) {
            _camLed.classList.add('live');
            _camLed.classList.remove('stale');
        }
        if (_camMeta) {
            const shape = payload.shape || [];
            const dims = shape.length === 2 ? `${shape[1]}×${shape[0]}` : '';
            const fps = computeCameraFps();
            _camMeta.textContent = dims
                ? `${dims}  ·  ${fps != null ? fps.toFixed(1) + ' fps' : '…'}`
                : (fps != null ? `${fps.toFixed(1)} fps` : 'live');
        }
        scheduleCameraStaleCheck();
    }

    function computeCameraFps() {
        const n = _camFrameTimes.length;
        if (n < 2) return null;
        const span = _camFrameTimes[n - 1] - _camFrameTimes[0];
        if (span <= 0) return null;
        return ((n - 1) * 1000) / span;
    }

    function scheduleCameraStaleCheck() {
        if (_camStaleTimer) clearTimeout(_camStaleTimer);
        _camStaleTimer = setTimeout(() => {
            const age = (Date.now() - _camLastFrameTs) / 1000;
            if (_camMeta) _camMeta.textContent = `last frame ${age.toFixed(1)}s ago`;
            if (_camLed) _camLed.classList.add('stale');
        }, 1500);
    }

    async function syncInitialCameraState() {
        try {
            const res = await fetch('/api/devices/bottom_camera/status');
            if (!res.ok) return;
            const data = await res.json();
            applyCameraState(!!data.streaming);
        } catch (err) {
            console.debug('bottom-camera status check failed:', err);
        }
    }

    function setupCameraWiring() {
        if (!_camToggle) return;
        _camToggle.addEventListener('click', toggleCameraStream);
        applyCameraState(false);
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('BOTTOM_CAMERA_FRAME', handleCameraFrame);
        }
    }

    // =====================================================================
    // View switching
    // =====================================================================

    function switchView(viewName) {
        if (!VIEWS.includes(viewName)) return;
        _currentView = viewName;
        for (const v of VIEWS) {
            const el = document.getElementById(`devices-view-${v}`);
            if (el) el.style.display = (v === viewName) ? '' : 'none';
        }
        if (typeof updateViewButtons === 'function') {
            updateViewButtons('devices-view-switcher', viewName);
        }
    }

    function setupViewSwitcher() {
        if (typeof initViewSwitcher === 'function') {
            initViewSwitcher('devices-view-switcher', switchView);
        }
        document.addEventListener('keydown', (e) => {
            if (typeof state !== 'undefined' && typeof TABS !== 'undefined' && state.tab !== TABS.DEVICES) return;
            if (e.target.matches('input, textarea, select, [contenteditable]')) return;
            if (e.key === 'm') { e.preventDefault(); switchView('map'); }
            else if (e.key === 'd') { e.preventDefault(); switchView('details'); }
        });
    }

    // =====================================================================
    // Top-level event handlers
    // =====================================================================

    function handlePayload(payload) {
        if (!payload) return;
        cacheDom();
        _lastTs = payload.t || 0;
        _lastWallTs = Date.now();

        if (payload.heartbeat || payload.paused) {
            setStatus('paused', 'paused', payload.reason ? `reason: ${payload.reason}` : 'updates paused');
        } else {
            const interval = _lastTs && _previousTs ? (_lastTs - _previousTs).toFixed(2) + 's' : '';
            setStatus('live', 'live', interval ? `Δt ${interval}` : 'streaming');
        }
        _previousTs = _lastTs;

        renderPositions(payload.positions);
        if (payload.properties) {
            applyOptimalBoxFromProperties(payload.properties);
            renderPropertiesTable(payload.properties);
            _lastPropertyMap = payload.properties;
        }
        scheduleStaleCheck();
    }

    function init() {
        cacheDom();
        setupViewSwitcher();
        setupCameraWiring();
        loadCoverslip();
        loadEmbryosSnapshot();
        switchView(_currentView);
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('DEVICE_STATE_UPDATE', handlePayload);
            ClientEventBus.on('EMBRYOS_UPDATE', handleEmbryosUpdate);
            // Belt-and-braces: also listen for the fine-grained events that
            // existed before EMBRYOS_UPDATE so direct emitters still refresh.
            ClientEventBus.on('EMBRYO_DETECTED', handleEmbryoDetected);
            ClientEventBus.on('STATUS_CHANGED', handleStatusChanged);
        }
        // Map-side edit handlers. Pointer events on the SVG cover both
        // "click an embryo" (selects it) and "click empty map" (drops the
        // selected embryo). Keyboard listener is document-wide but guards
        // against firing while an input is focused.
        if (_mapSvg) {
            _mapSvg.addEventListener('pointerdown', onMapPointerDown);
        }
        document.addEventListener('keydown', onMapKeyDown);
        setStatus('stale', 'waiting', 'no payload yet');
        syncInitialCameraState();
        // Stop the camera stream if the tab is closed while it's running,
        // so MMCore isn't held by a disconnected browser.
        window.addEventListener('beforeunload', () => {
            if (_camStreaming) {
                try {
                    navigator.sendBeacon('/api/devices/bottom_camera/stream/stop');
                } catch (_) {}
            }
        });
    }

    return { init, handlePayload, switchView };
})();

document.addEventListener('DOMContentLoaded', () => DevicesManager.init());
