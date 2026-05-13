/**
 * Devices tab — read-only stream of MMCore device state.
 *
 * Two views:
 *   Map     — top-down XY stage map with safe/caution/forbidden zones (default)
 *   Details — position cards + filterable properties table
 *
 * Both views are driven by the same DEVICE_STATE_UPDATE payload forwarded
 * from the device-layer SSE stream by DeviceStateMonitor. No write controls.
 */
const DevicesManager = (function () {
    const STALE_AFTER_MS = 4000;
    const VIEWS = ['map', 'details'];
    const SVG_NS = 'http://www.w3.org/2000/svg';

    // Status / details DOM
    let _statusPill, _statusMeta, _statusDot;
    let _posX, _posY, _piezoZ, _galvoA, _galvoB;
    let _tbody, _filter;

    // Map DOM — Navigator layout
    let _mapSvg, _mapBg, _mapGridMinor, _mapGridMajor, _mapAxisEmphasis;
    let _mapBeyond, _mapCoverslip;
    let _mapZones, _mapZoneLabels, _mapOrigin, _mapAxes;
    let _mapMarker, _mapMarkerPulse, _mapMarkerRing, _mapMarkerDot;
    let _mapReadoutX, _mapReadoutY;
    let _mapWrap, _editOverlay;
    let _scalebarLabel;
    let _editToggleBtn, _editZonePicker, _editInstruction, _editActions, _editSaveBtn, _editCancelBtn, _editCaptureBtn;
    let _editFlow;

    // Bottom-camera panel DOM + state
    let _camPanel, _camToggle, _camImg, _camPlaceholder, _camLed, _camMeta;
    let _camStreaming = false;
    let _camLastFrameTs = 0;
    let _camHasFrame = false;
    let _camStaleTimer = null;
    // Rolling FPS over the last N frame timestamps (perf.now ms).
    const _CAM_FPS_WINDOW = 12;
    let _camFrameTimes = [];

    // Edit-zones state
    let _editMode = false;
    let _draftZones = null;          // working copy of zones during editing
    let _activeColor = null;         // which zone is currently being redefined
    let _firstClick = null;          // {X, Y} stage µm of first click, or null
    let _mouseMoveBound = null;      // bound mousemove handler for cleanup
    let _svgClickBound = null;       // bound click handler for cleanup

    let _lastTs = 0;
    let _previousTs = 0;
    let _lastWallTs = 0;
    let _staleTimer = null;
    let _filterText = '';
    let _lastPropertyMap = {};
    let _lastXY = null;             // {X, Y} in stage µm, last seen
    let _currentView = 'map';

    // Map geometry (filled once /api/devices/zones loads)
    let _zoneCfg = null;            // { units, zones: [{color, x:[min,max], y:[min,max]}] }
    let _viewBox = null;            // { xMin, xMax, yMin, yMax } in stage µm

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
        _mapMarker        = document.getElementById('devices-map-marker');
        _mapMarkerPulse   = document.getElementById('devices-map-marker-pulse');
        _mapMarkerRing    = document.getElementById('devices-map-marker-ring');
        _mapMarkerDot     = document.getElementById('devices-map-marker-dot');
        _mapReadoutX      = document.getElementById('devices-map-x');
        _mapReadoutY      = document.getElementById('devices-map-y');
        _mapWrap          = document.getElementById('devices-map-wrap');
        _editOverlay      = document.getElementById('devices-map-edit-overlay');
        _scalebarLabel    = document.getElementById('devices-scalebar-value');

        _editToggleBtn   = document.getElementById('devices-edit-toggle');
        _editZonePicker  = document.getElementById('devices-edit-zone-picker');
        _editInstruction = document.getElementById('devices-edit-instruction');
        _editActions     = document.getElementById('devices-edit-actions');
        _editSaveBtn     = document.getElementById('devices-edit-save');
        _editCancelBtn   = document.getElementById('devices-edit-cancel');
        _editCaptureBtn  = document.getElementById('devices-edit-capture');
        _editFlow        = document.querySelector('.devices-edit-flow');

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

    function renderPositions(positions) {
        if (!positions) return;
        for (const devName of Object.keys(positions)) {
            const entry = positions[devName] || {};
            switch (entry.kind) {
                case 'xy_stage':
                    setAxis(_posX, entry.X, 2);
                    setAxis(_posY, entry.Y, 2);
                    _lastXY = { X: entry.X, Y: entry.Y };
                    // Autoscale: if the live position moves enough to change
                    // the bounding box, redraw zones/grid/axes too.
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

    // MMCore DeviceType enum -> short label
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
    // Map view
    // =====================================================================

    async function loadZones() {
        try {
            const res = await fetch('/api/devices/zones');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            _zoneCfg = await res.json();
        } catch (err) {
            console.error('Failed to load device zones:', err);
            _zoneCfg = { units: 'um', zones: [] };
        }
        computeViewBox();
        renderMap();
    }

    // Compute the viewport extent. Always fits the union of:
    //   - all configured zones
    //   - the live stage XY (if known)
    // No hardcoded axis limits — anything the operator drives onto/into the
    // scene grows the bbox. Returns true if the computed viewBox differs
    // meaningfully from the previous one (caller should re-render zones/axes).
    function computeViewBox() {
        const zones = _draftZones || _zoneCfg?.zones || [];
        let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
        for (const z of zones) {
            if (z.x) { xMin = Math.min(xMin, z.x[0]); xMax = Math.max(xMax, z.x[1]); }
            if (z.y) { yMin = Math.min(yMin, z.y[0]); yMax = Math.max(yMax, z.y[1]); }
        }
        if (_lastXY) {
            xMin = Math.min(xMin, _lastXY.X); xMax = Math.max(xMax, _lastXY.X);
            yMin = Math.min(yMin, _lastXY.Y); yMax = Math.max(yMax, _lastXY.Y);
        }
        if (!isFinite(xMin) || !isFinite(yMin)) {
            // Nothing to show yet — pick a small symmetric default
            xMin = -100; xMax = 100; yMin = -100; yMax = 100;
        }
        // Ensure a non-degenerate span if all inputs collapse to a point
        if (xMax - xMin < 1) { const c = (xMax + xMin) / 2; xMin = c - 100; xMax = c + 100; }
        if (yMax - yMin < 1) { const c = (yMax + yMin) / 2; yMin = c - 100; yMax = c + 100; }

        // Symmetric padding ~8% of the larger span so labels and the marker
        // halo don't kiss the edge of the viewport.
        const span = Math.max(xMax - xMin, yMax - yMin);
        const pad = span * 0.08;
        const next = {
            xMin: xMin - pad, xMax: xMax + pad,
            yMin: yMin - pad, yMax: yMax + pad,
        };

        // Only flag a meaningful change so a millisecond-scale stage jitter
        // doesn't keep redrawing zones/grid/axes (each redraw rebuilds DOM).
        const changed = !_viewBox ||
            Math.abs(next.xMin - _viewBox.xMin) > span * 0.005 ||
            Math.abs(next.xMax - _viewBox.xMax) > span * 0.005 ||
            Math.abs(next.yMin - _viewBox.yMin) > span * 0.005 ||
            Math.abs(next.yMax - _viewBox.yMax) > span * 0.005;
        _viewBox = next;
        return changed;
    }

    // Stage Y is positive-up; SVG Y is positive-down. Convert by negation.
    // Every drawn coordinate goes through stageToSvg() so there are no
    // nested transforms to reason about (especially nice for text).
    function svgY(stageY) { return -stageY; }

    function renderMap() {
        if (!_mapSvg || !_viewBox) return;
        const { xMin, xMax, yMin, yMax } = _viewBox;
        const w = xMax - xMin, h = yMax - yMin;
        _mapSvg.setAttribute('viewBox', `${xMin} ${-yMax} ${w} ${h}`);

        // Marker sized in user-space — keep it small. A surveyor's pip, not
        // a glow blob: ~0.45% of the visible span for the dot.
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
        updateMapMarker();
        updateScalebar();
    }

    // "Beyond" is implicit: paint the whole viewport with the red hatch
    // pattern. The optimal/maximal zone rects paint ABOVE this so the
    // operator's safe windows look carved out of a hatched danger envelope.
    //
    // The hatch is drawn via the <pattern id="devices-map-hatch-red"> in defs,
    // but the pattern's tile size is in user-space (stage µm), so we resize
    // it on each renderMap to keep ~50 stripes visible regardless of zoom.
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

    // Coverslip is an optional spatial reference — a thin dashed rect showing
    // the 50x24 mm slide outline around its centre. Drawn only when the config
    // includes a coverslip block.
    function renderCoverslip() {
        if (!_mapCoverslip) return;
        _mapCoverslip.innerHTML = '';
        const cs = _zoneCfg?.coverslip;
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

        // Tiny caption at top-right of the coverslip outline
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

    function renderBackground() {
        if (!_mapBg || !_viewBox) return;
        _mapBg.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        // Grain pattern + radial vignette layered on the canvas
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

    function renderZones() {
        if (!_mapZones) return;
        _mapZones.innerHTML = '';
        // Only optimal (green) and maximal (orange) are explicit zones now.
        // "Beyond" is the implicit hatched layer painted in renderBeyond().
        const zones = _zoneCfg?.zones || [];
        for (const z of zones) {
            if (!z.x || !z.y) continue;
            if (z.color === 'red') continue; // belt-and-suspenders: GET should already filter
            const rect = document.createElementNS(SVG_NS, 'rect');
            rect.setAttribute('x', z.x[0]);
            rect.setAttribute('y', svgY(z.y[1]));
            rect.setAttribute('width',  z.x[1] - z.x[0]);
            rect.setAttribute('height', z.y[1] - z.y[0]);
            rect.setAttribute('class', `devices-zone devices-zone-${z.color}`);
            _mapZones.appendChild(rect);
        }
    }

    function renderZoneLabels() {
        if (!_mapZoneLabels || !_viewBox) return;
        _mapZoneLabels.innerHTML = '';
        const zones = _zoneCfg?.zones || [];
        if (!zones.length) return;
        drawZoneLabels(zones);
    }

    // Place a zone's caption at its top-left corner, but only if the zone is
    // big enough relative to the viewport that the text won't overflow it.
    // Tiny zones lose their in-canvas label — the legend at bottom-right still
    // disambiguates them.
    function drawZoneLabels(zones) {
        if (!_mapZoneLabels || !_viewBox) return;
        const span = Math.max(_viewBox.xMax - _viewBox.xMin, _viewBox.yMax - _viewBox.yMin);
        const fontSize = span * 0.018;
        const inset    = span * 0.010;
        const labelMap = { green: 'OPTIMAL', orange: 'MAXIMAL', red: 'FORBIDDEN' };
        for (const z of zones) {
            if (!z.x || !z.y) continue;
            const zw = z.x[1] - z.x[0];
            const zh = z.y[1] - z.y[0];
            // Each character is ~0.62em wide with tight tracking + letter-
            // spacing; require the zone to comfortably fit the label.
            const text = labelMap[z.color] || z.color.toUpperCase();
            const estTextWidth = text.length * fontSize * 0.78;
            if (zw < estTextWidth * 1.15 || zh < fontSize * 2.0) continue;

            const tx = z.x[0] + inset;
            const ty = z.y[1] - inset;
            const t = document.createElementNS(SVG_NS, 'text');
            t.setAttribute('x', tx);
            t.setAttribute('y', svgY(ty) + fontSize * 0.85);
            t.setAttribute('class', `devices-zone-label devices-zone-${z.color}`);
            t.setAttribute('font-size', fontSize);
            t.setAttribute('text-anchor', 'start');
            t.textContent = text;
            _mapZoneLabels.appendChild(t);
        }
    }

    function renderOrigin() {
        if (!_mapOrigin || !_viewBox) return;
        _mapOrigin.innerHTML = '';
        const { xMin, xMax, yMin, yMax } = _viewBox;
        // Only draw if (0, 0) is within view
        if (xMin > 0 || xMax < 0 || yMin > 0 || yMax < 0) return;
        const span = Math.max(xMax - xMin, yMax - yMin);
        const arm = span * 0.012;
        const path = document.createElementNS(SVG_NS, 'path');
        // Cross at origin, in SVG coords (origin maps to (0, 0))
        path.setAttribute('d',
            `M ${-arm} 0 L ${arm} 0  M 0 ${-arm} L 0 ${arm}`);
        path.setAttribute('class', 'devices-origin-mark');
        _mapOrigin.appendChild(path);

        // Small "0,0" tag, offset to upper right of origin
        const label = document.createElementNS(SVG_NS, 'text');
        label.setAttribute('x', arm * 1.4);
        label.setAttribute('y', -arm * 0.6);
        label.setAttribute('class', 'devices-origin-label');
        label.setAttribute('font-size', span * 0.013);
        label.textContent = '0,0';
        _mapOrigin.appendChild(label);
    }

    // niceStep with 1/2/5 family at the chosen power of ten.
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

        // Vertical lines
        for (let x = Math.ceil(xMin / minor) * minor; x <= xMax; x += minor) {
            const isMajor = Math.abs(x % major) < minor * 0.001 || Math.abs(x % major - major) < minor * 0.001;
            const line = document.createElementNS(SVG_NS, 'line');
            line.setAttribute('x1', x); line.setAttribute('x2', x);
            line.setAttribute('y1', svgY(yMax)); line.setAttribute('y2', svgY(yMin));
            line.setAttribute('class', isMajor ? 'devices-grid-major' : 'devices-grid-minor');
            (isMajor ? _mapGridMajor : _mapGridMinor).appendChild(line);
        }
        // Horizontal lines
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
        // X=0 line
        if (xMin <= 0 && xMax >= 0) {
            const l = document.createElementNS(SVG_NS, 'line');
            l.setAttribute('x1', 0); l.setAttribute('x2', 0);
            l.setAttribute('y1', svgY(yMax)); l.setAttribute('y2', svgY(yMin));
            l.setAttribute('class', 'devices-axis-emphasis');
            _mapAxisEmphasis.appendChild(l);
        }
        // Y=0 line
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

    // Tick labels: use "k" suffix for values past ±10000, keep small numbers
    // sharp — keeps long axis labels from crowding the canvas.
    function formatTickValue(v) {
        const a = Math.abs(v);
        if (a >= 10000) return (v / 1000).toFixed(a >= 100000 ? 0 : 1) + 'k';
        return Math.round(v).toString();
    }

    function updateMapMarker() {
        if (_editMode && _firstClick && _lastXY) {
            drawPreviewRect(_firstClick, _lastXY);
        }
        if (!_mapMarker || !_lastXY) return;
        const sx = _lastXY.X;
        const sy = svgY(_lastXY.Y);
        _mapMarkerDot.setAttribute('cx',   sx); _mapMarkerDot.setAttribute('cy',   sy);
        _mapMarkerRing.setAttribute('cx',  sx); _mapMarkerRing.setAttribute('cy',  sy);
        _mapMarkerPulse.setAttribute('cx', sx); _mapMarkerPulse.setAttribute('cy', sy);
        _mapMarker.classList.remove('hidden');
    }

    // Scale bar caption: figure out how many stage-µm the fixed 120-px
    // scale bar spans at the current zoom, then round to a nice 1/2/5 value.
    function updateScalebar() {
        if (!_scalebarLabel || !_mapSvg || !_viewBox) return;
        const rect = _mapSvg.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const w = _viewBox.xMax - _viewBox.xMin;
        const h = _viewBox.yMax - _viewBox.yMin;
        // preserveAspectRatio="meet" — pick the smaller scale factor
        const scale = Math.min(rect.width / w, rect.height / h);
        if (!isFinite(scale) || scale <= 0) return;
        const targetPx = 120; // visible scale-bar width on screen
        const rawUm = targetPx / scale;
        // Snap to nearest 1/2/5 × 10^k
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
    // Edit-zones mode
    // =====================================================================

    // Convert a DOM mouse event to stage µm using the SVG CTM.
    function eventToStageXY(evt) {
        const pt = _mapSvg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        const inverse = _mapSvg.getScreenCTM().inverse();
        const local = pt.matrixTransform(inverse);
        // SVG uses flipped Y; convert back to stage µm
        return { X: local.x, Y: -local.y };
    }

    function setEditMode(on) {
        _editMode = on;
        if (!_mapWrap) return;
        _mapWrap.classList.toggle('edit-mode', on);
        _editToggleBtn.classList.toggle('active', on);
        if (_editFlow) _editFlow.hidden = !on;
        _editInstruction.hidden = !on;
        if (_editCaptureBtn) _editCaptureBtn.hidden = true;  // shown after a zone is picked

        if (on) {
            _draftZones = (_zoneCfg?.zones || []).map(z => ({
                color: z.color,
                x: [...(z.x || [])],
                y: [...(z.y || [])],
            }));
            _activeColor = null;
            _firstClick = null;
            _editInstruction.textContent = 'Pick a zone, then capture two opposite corners — drive the stage to each one and press Capture.';
            attachSvgEditHandlers();
        } else {
            _draftZones = null;
            _activeColor = null;
            _firstClick = null;
            clearEditOverlay();
            detachSvgEditHandlers();
            renderZones();
            renderZoneLabels();
        }

        document.querySelectorAll('.devices-edit-zone-chip').forEach(b =>
            b.classList.toggle('active', b.dataset.zone === _activeColor));
    }

    function attachSvgEditHandlers() {
        if (!_mapSvg) return;
        detachSvgEditHandlers();
        _svgClickBound = onSvgClick;
        _mouseMoveBound = onSvgMouseMove;
        _mapSvg.addEventListener('click', _svgClickBound);
        _mapSvg.addEventListener('mousemove', _mouseMoveBound);
    }
    function detachSvgEditHandlers() {
        if (!_mapSvg) return;
        if (_svgClickBound)    _mapSvg.removeEventListener('click', _svgClickBound);
        if (_mouseMoveBound)   _mapSvg.removeEventListener('mousemove', _mouseMoveBound);
        _svgClickBound = _mouseMoveBound = null;
    }

    function pickZone(color) {
        _activeColor = color;
        _firstClick = null;
        _editInstruction.textContent =
            `Move stage to the first corner of the ${zoneLabel(color)} zone, ` +
            `then click "Capture current". (Or click on the map.)`;
        document.querySelectorAll('.devices-edit-zone-chip').forEach(b =>
            b.classList.toggle('active', b.dataset.zone === color));
        if (_editCaptureBtn) _editCaptureBtn.hidden = false;
        clearEditOverlay();
    }

    function zoneLabel(color) {
        return color === 'green' ? 'optimal' : color === 'orange' ? 'maximal' : 'beyond';
    }

    // Single corner-capture path used by both SVG clicks and the
    // "Capture current position" button.
    function captureCorner(X, Y) {
        if (!_editMode || !_activeColor) return;
        if (_firstClick === null) {
            _firstClick = { X, Y };
            _editInstruction.textContent =
                `First corner captured at (${X.toFixed(1)}, ${Y.toFixed(1)}). ` +
                `Move stage to the opposite corner, or click on the map.`;
            drawFirstPoint(_firstClick);
            // Show a preview rect from the first corner to the live XY so the
            // operator can see the zone forming as they drive the stage.
            if (_lastXY) drawPreviewRect(_firstClick, _lastXY);
        } else {
            const x = [Math.min(_firstClick.X, X), Math.max(_firstClick.X, X)];
            const y = [Math.min(_firstClick.Y, Y), Math.max(_firstClick.Y, Y)];
            const idx = _draftZones.findIndex(z => z.color === _activeColor);
            if (idx >= 0) _draftZones[idx] = { color: _activeColor, x, y };
            _firstClick = null;
            _editInstruction.textContent = `Updated ${zoneLabel(_activeColor)}. Pick another zone or click Save.`;
            renderDraftZones();
            clearEditOverlay();
        }
    }

    function onSvgClick(evt) {
        if (!_editMode || !_activeColor) return;
        const { X, Y } = eventToStageXY(evt);
        captureCorner(X, Y);
    }

    function onSvgMouseMove(evt) {
        if (!_editMode || !_firstClick) return;
        const { X, Y } = eventToStageXY(evt);
        drawPreviewRect(_firstClick, { X, Y });
    }

    function captureCurrentPosition() {
        if (!_editMode) return;
        if (!_activeColor) {
            _editInstruction.textContent = 'Pick a zone first (Safe / Caution / Forbidden).';
            return;
        }
        if (!_lastXY) {
            _editInstruction.textContent = 'No live XY yet — waiting for the device stream.';
            return;
        }
        captureCorner(_lastXY.X, _lastXY.Y);
    }

    function drawFirstPoint(pt) {
        clearEditOverlay();
        if (!_editOverlay || !_viewBox) return;
        const r = Math.max(_viewBox.xMax - _viewBox.xMin, _viewBox.yMax - _viewBox.yMin) * 0.008;
        const c = document.createElementNS(SVG_NS, 'circle');
        c.setAttribute('cx', pt.X);
        c.setAttribute('cy', svgY(pt.Y));
        c.setAttribute('r', r);
        c.setAttribute('class', 'devices-edit-first-point');
        _editOverlay.appendChild(c);
    }

    function drawPreviewRect(a, b) {
        // Keep the first-click marker, replace any prior preview rect
        if (!_editOverlay) return;
        Array.from(_editOverlay.querySelectorAll('.devices-edit-preview-rect')).forEach(el => el.remove());
        const x0 = Math.min(a.X, b.X), x1 = Math.max(a.X, b.X);
        const y0 = Math.min(a.Y, b.Y), y1 = Math.max(a.Y, b.Y);
        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', x0);
        rect.setAttribute('y', svgY(y1));
        rect.setAttribute('width',  x1 - x0);
        rect.setAttribute('height', y1 - y0);
        rect.setAttribute('class', 'devices-edit-preview-rect');
        _editOverlay.appendChild(rect);
    }

    function clearEditOverlay() {
        if (_editOverlay) _editOverlay.innerHTML = '';
    }

    function renderDraftZones() {
        // Paint the working copy without touching the persisted _zoneCfg.
        if (!_mapZones || !_mapZoneLabels) return;
        _mapZones.innerHTML = '';
        _mapZoneLabels.innerHTML = '';
        const zones = _draftZones || _zoneCfg?.zones || [];
        for (const z of zones) {
            if (!z.x || !z.y) continue;
            if (z.color === 'red') continue;
            const rect = document.createElementNS(SVG_NS, 'rect');
            rect.setAttribute('x', z.x[0]);
            rect.setAttribute('y', svgY(z.y[1]));
            rect.setAttribute('width',  z.x[1] - z.x[0]);
            rect.setAttribute('height', z.y[1] - z.y[0]);
            rect.setAttribute('class', `devices-zone devices-zone-${z.color}`);
            _mapZones.appendChild(rect);
        }
        drawZoneLabels(zones);
    }

    async function saveEdits() {
        if (!_draftZones) { setEditMode(false); return; }
        try {
            const res = await fetch('/api/devices/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    units: _zoneCfg?.units || 'um',
                    zones: _draftZones,
                }),
            });
            if (!res.ok) {
                const detail = await res.text();
                _editInstruction.textContent = `Save failed: ${detail}`;
                return;
            }
            const data = await res.json();
            _zoneCfg = data;
            computeViewBox();
            renderMap();
            setEditMode(false);
        } catch (err) {
            _editInstruction.textContent = `Save failed: ${err}`;
        }
    }

    function setupEditWiring() {
        if (!_editToggleBtn) return;
        _editToggleBtn.addEventListener('click', () => setEditMode(!_editMode));
        _editCancelBtn.addEventListener('click', () => setEditMode(false));
        _editSaveBtn.addEventListener('click', saveEdits);
        if (_editCaptureBtn) _editCaptureBtn.addEventListener('click', captureCurrentPosition);
        document.querySelectorAll('.devices-edit-zone-chip').forEach(b =>
            b.addEventListener('click', () => pickZone(b.dataset.zone)));
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
            // Stream paused — drop the last frame so the operator doesn't
            // confuse a stale image with a live one.
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

        // Rolling FPS: keep the last N frame arrival timestamps and use
        // their span. Resilient to occasional gaps; reads as live.
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
        // Cancel any in-progress edit when leaving the Map view; otherwise the
        // SVG click handlers would keep firing against a hidden tree.
        if (viewName !== 'map' && _editMode) setEditMode(false);
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
        // Letter-key shortcuts when the Devices tab is active.
        // Number keys are reserved by the global tab switcher.
        document.addEventListener('keydown', (e) => {
            if (typeof state !== 'undefined' && typeof TABS !== 'undefined' && state.tab !== TABS.DEVICES) return;
            if (e.target.matches('input, textarea, select, [contenteditable]')) return;
            if (e.key === 'm') { e.preventDefault(); switchView('map'); }
            else if (e.key === 'd') { e.preventDefault(); switchView('details'); }
            else if (e.key === 'e' && _currentView === 'map') {
                e.preventDefault();
                setEditMode(!_editMode);
            } else if (e.key === 'Escape' && _editMode) {
                e.preventDefault();
                setEditMode(false);
            }
        });
    }

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
            renderPropertiesTable(payload.properties);
            _lastPropertyMap = payload.properties;
        }
        scheduleStaleCheck();
    }

    function init() {
        cacheDom();
        setupViewSwitcher();
        setupEditWiring();
        setupCameraWiring();
        loadZones();
        switchView(_currentView);
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('DEVICE_STATE_UPDATE', handlePayload);
        }
        setStatus('stale', 'waiting', 'no payload yet');
        // Whatever state the agent reports (possibly already streaming from a
        // previous browser session) — sync the toggle.
        syncInitialCameraState();
        // Safety: if the page is being closed and the camera was started by
        // this browser, tell the agent to stop so MMCore doesn't keep
        // grabbing frames into the void.
        window.addEventListener('beforeunload', () => {
            if (_camStreaming) {
                // sendBeacon is fire-and-forget but reliable on unload.
                try {
                    navigator.sendBeacon('/api/devices/bottom_camera/stream/stop');
                } catch (_) {}
            }
        });
    }

    return { init, handlePayload, switchView };
})();

document.addEventListener('DOMContentLoaded', () => DevicesManager.init());
