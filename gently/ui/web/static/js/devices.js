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
    const VIEWS = ['map', 'details', 'optical3d', 'manual'];
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
    let _camStage, _camCrosshair, _camCrosshairGroup;
    let _camStreaming = false;
    let _camLastFrameTs = 0;
    let _camHasFrame = false;
    let _camStaleTimer = null;
    const _CAM_FPS_WINDOW = 12;
    let _camFrameTimes = [];

    // Camera zoom / pan. Identity transform = (zoom 1, tx 0, ty 0); pan only
    // engages once zoom > 1. Reset on double-click and on stream-off.
    let _camZoom = 1;
    let _camTx = 0;
    let _camTy = 0;
    let _camPanLast = null;  // {x, y} clientX/Y of last pointermove during pan
    const _CAM_ZOOM_MIN = 1;
    const _CAM_ZOOM_MAX = 8;
    const _CAM_ZOOM_STEP = 1.15;  // multiplicative per wheel notch

    // Lightsheet live panel DOM + state (Manual view)
    let _lsToggle, _lsImg, _lsPlaceholder, _lsLed, _lsMeta, _lsStage;
    let _lsStreaming = false;
    let _lsLastFrameTs = 0;
    let _lsHasFrame = false;
    let _lsStaleTimer = null;
    const _LS_FPS_WINDOW = 12;
    let _lsFrameTimes = [];

    // Lightsheet zoom / pan (mirrors camera zoom/pan)
    let _lsZoom = 1;
    let _lsTx = 0;
    let _lsTy = 0;
    let _lsPanLast = null;

    // Lightsheet live params — debounced POST to /api/devices/lightsheet/live/params
    let _lsGalvo = 0;
    let _lsPiezo = 50;
    let _lsExposure = 20;  // matches device-layer _ls_params default (20 ms)
    let _lsSide = 'A';     // SPIM side — 'A' (HamCam1) or 'B' (HamCam2 if present)
    let _lsParamTimer = null;

    // Lightsheet control inputs (rail)
    let _lsGalvoSlider, _lsGalvoNum, _lsPiezoSlider, _lsPiezoNum, _lsExposureNum;
    let _lsLedToggle, _lsCamLed, _lsRoomLightBtn;
    let _lsLedIsOpen = false;  // LED toggle state: false = Closed (safe default)
    let _lsCamLedOn = false;
    let _lsLaserToggle;
    let _lsLaserOn = false;    // Laser toggle state: false = OFF (entry-safe default)
    let _lsSnapVolBtn, _lsBurstBtn, _lsLastcap, _lsLastcapRef;
    let _lsLaserStatus;   // span inside .ls-laser-indicator — driven by actual laser/off calls
    let _lsLaserPreset;   // <select id="devices-laser-preset"> — populated on manual-view entry
    let _lsSideSelect;    // <select id="devices-ls-side"> — shown only when camera_b present
    let _lsTempInput, _lsTempSet;

    // Timelapse form DOM refs (Manual view — #devices-tl-group)
    let _tlToggle, _tlBody;
    let _tlInterval, _tlStop, _tlCondRow, _tlCondLabel, _tlCondVal;
    let _tlEmbryos, _tlMode;
    let _tlSlices, _tlExposure, _tlGalvoAmp, _tlGalvoCtr, _tlPiezoAmp, _tlPiezoCtr, _tlLaser;
    let _tlStart, _tlStatus, _tlStatusText;
    // Accordion active-state per section: { sched, targets, geom }
    let _tlTouched = { sched: false, targets: false, geom: false };

    // Room-light toggle (header). Drives the SwitchBot Bot that switches the
    // diSPIM room light. State is the bot's cached on/off; hidden until the
    // device layer reports the accessory is configured.
    let _roomLightToggle, _roomLightLabel;
    let _roomLightState = 'unknown';
    let _roomLightAvailable = false;
    let _roomLightBusy = false;
    let _roomLightTimer = null;

    // Temperature-controller panel DOM + state
    let _tempEl, _tempReadout, _tempInput, _tempSet;
    let _tempState = 'unknown';
    let _tempAvailable = false;
    let _tempBusy = false;
    let _tempTimer = null;

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
        _camStage        = _camPanel ? _camPanel.querySelector('.devices-camera-stage') : null;
        _camCrosshair    = document.getElementById('devices-camera-crosshair');
        _camCrosshairGroup = document.getElementById('devices-camera-crosshair-group');
        _camLed          = document.getElementById('devices-camera-led');
        _camMeta         = document.getElementById('devices-camera-meta');

        // Manual / lightsheet panel
        _lsToggle        = document.getElementById('devices-ls-toggle');
        _lsImg           = document.getElementById('devices-ls-img');
        _lsPlaceholder   = document.getElementById('devices-ls-placeholder');
        _lsStage         = document.getElementById('devices-ls-stage');
        _lsLed           = document.getElementById('devices-ls-led');
        _lsMeta          = document.getElementById('devices-ls-meta');
        _lsGalvoSlider   = document.getElementById('devices-ls-galvo-slider');
        _lsGalvoNum      = document.getElementById('devices-ls-galvo');
        _lsPiezoSlider   = document.getElementById('devices-ls-piezo-slider');
        _lsPiezoNum      = document.getElementById('devices-ls-piezo');
        _lsExposureNum   = document.getElementById('devices-ls-exposure');
        _lsLedToggle     = document.getElementById('devices-ls-led-toggle');
        _lsCamLed        = document.getElementById('devices-ls-cam-led');
        _lsRoomLightBtn  = document.getElementById('devices-ls-room-light-btn');
        _lsLaserToggle   = document.getElementById('devices-ls-laser-toggle');
        _lsSnapVolBtn    = document.getElementById('devices-ls-snap-volume');
        _lsBurstBtn      = document.getElementById('devices-ls-burst');
        _lsLastcap       = document.getElementById('devices-ls-lastcap');
        _lsLastcapRef    = document.getElementById('devices-ls-lastcap-ref');
        _lsLaserStatus   = document.getElementById('devices-ls-laser-status');
        _lsLaserPreset   = document.getElementById('devices-laser-preset');
        _lsSideSelect    = document.getElementById('devices-ls-side');
        _lsTempInput     = document.getElementById('devices-ls-temp-input');
        _lsTempSet       = document.getElementById('devices-ls-temp-set');

        // Timelapse form
        _tlToggle     = document.getElementById('devices-tl-toggle');
        _tlBody       = document.getElementById('devices-tl-body');
        _tlInterval   = document.getElementById('devices-tl-interval');
        _tlStop       = document.getElementById('devices-tl-stop');
        _tlCondRow    = document.getElementById('devices-tl-cond-row');
        _tlCondLabel  = document.getElementById('devices-tl-cond-label');
        _tlCondVal    = document.getElementById('devices-tl-cond-val');
        _tlEmbryos    = document.getElementById('devices-tl-embryos');
        _tlMode       = document.getElementById('devices-tl-mode');
        _tlSlices     = document.getElementById('devices-tl-slices');
        _tlExposure   = document.getElementById('devices-tl-exposure');
        _tlGalvoAmp   = document.getElementById('devices-tl-galvo-amp');
        _tlGalvoCtr   = document.getElementById('devices-tl-galvo-ctr');
        _tlPiezoAmp   = document.getElementById('devices-tl-piezo-amp');
        _tlPiezoCtr   = document.getElementById('devices-tl-piezo-ctr');
        _tlLaser      = document.getElementById('devices-tl-laser');
        _tlStart      = document.getElementById('devices-tl-start');
        _tlStatus     = document.getElementById('devices-tl-status');
        _tlStatusText = document.getElementById('devices-tl-status-text');

        _roomLightToggle = document.getElementById('devices-room-light-toggle');
        _roomLightLabel  = document.getElementById('devices-room-light-label');

        _tempEl      = document.getElementById('devices-temp');
        _tempReadout = document.getElementById('devices-temp-readout');
        _tempInput   = document.getElementById('devices-temp-input');
        _tempSet     = document.getElementById('devices-temp-set');

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
            // Operator may have zoomed in; reset so the next stream session
            // starts at 1× rather than inheriting a stale view.
            resetCameraZoom();
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

    // ---- Camera zoom / pan ---------------------------------------------
    function applyCameraTransform() {
        if (!_camImg) return;
        _camImg.style.transform =
            `translate(${_camTx}px, ${_camTy}px) scale(${_camZoom})`;
        // Reticle uses an SVG transform attribute on the inner <g> instead
        // of a CSS transform on the SVG element — same geometric effect,
        // but the SVG renderer re-rasterises at the new zoom so the 1px
        // strokes stay crisp instead of getting bitmap-scaled.
        if (_camCrosshairGroup && _camStage) {
            const rect = _camStage.getBoundingClientRect();
            // Convert pixel-space translation to viewBox units (viewBox is
            // 0..100 in both axes, preserveAspectRatio=none).
            const txV = rect.width  > 0 ? (_camTx * 100) / rect.width  : 0;
            const tyV = rect.height > 0 ? (_camTy * 100) / rect.height : 0;
            // translate(50+tx, 50+ty) scale(zoom) translate(-50, -50) keeps
            // the viewBox centre (50, 50) as the zoom anchor and offsets by
            // the converted pixel translation.
            _camCrosshairGroup.setAttribute(
                'transform',
                `translate(${50 + txV} ${50 + tyV}) ` +
                `scale(${_camZoom}) ` +
                `translate(-50 -50)`
            );
        }
    }

    function resetCameraZoom() {
        _camZoom = 1;
        _camTx = 0;
        _camTy = 0;
        applyCameraTransform();
        if (_camStage) _camStage.classList.remove('camera-zoomed', 'camera-panning');
    }

    // Keep at least the image centre within the visible window so the
    // operator can't accidentally pan the entire frame off-screen. At
    // zoom 1 this collapses to (0, 0).
    function clampCameraPan() {
        if (!_camStage) return;
        const rect = _camStage.getBoundingClientRect();
        const maxX = (rect.width  * (_camZoom - 1)) / 2;
        const maxY = (rect.height * (_camZoom - 1)) / 2;
        _camTx = Math.max(-maxX, Math.min(maxX, _camTx));
        _camTy = Math.max(-maxY, Math.min(maxY, _camTy));
    }

    function onCameraWheel(event) {
        if (!_camStage) return;
        // Always preventDefault so the page doesn't scroll under the
        // operator while they're framing a sample.
        event.preventDefault();
        const rect = _camStage.getBoundingClientRect();
        const cx = event.clientX - rect.left - rect.width  / 2;
        const cy = event.clientY - rect.top  - rect.height / 2;
        const oldZoom = _camZoom;
        const factor = event.deltaY < 0 ? _CAM_ZOOM_STEP : 1 / _CAM_ZOOM_STEP;
        const newZoom = Math.max(_CAM_ZOOM_MIN,
                                 Math.min(_CAM_ZOOM_MAX, oldZoom * factor));
        if (newZoom === oldZoom) return;

        // Keep the image point under the cursor anchored under the cursor
        // across the zoom: cursor_new = cursor_old after the transform
        // change, which means newT = cursor - (cursor - oldT) * (new/old).
        const ratio = newZoom / oldZoom;
        _camTx = cx - (cx - _camTx) * ratio;
        _camTy = cy - (cy - _camTy) * ratio;
        _camZoom = newZoom;

        if (Math.abs(_camZoom - 1) < 0.001) {
            resetCameraZoom();
            return;
        }
        clampCameraPan();
        applyCameraTransform();
        _camStage.classList.add('camera-zoomed');
    }

    function onCameraPointerDown(event) {
        if (event.button !== 0) return;
        if (_camZoom <= 1) return;
        _camPanLast = { x: event.clientX, y: event.clientY };
        try { _camStage.setPointerCapture(event.pointerId); } catch (_) {}
        _camStage.classList.add('camera-panning');
        event.preventDefault();
    }

    function onCameraPointerMove(event) {
        if (!_camPanLast) return;
        _camTx += event.clientX - _camPanLast.x;
        _camTy += event.clientY - _camPanLast.y;
        _camPanLast = { x: event.clientX, y: event.clientY };
        clampCameraPan();
        applyCameraTransform();
    }

    function onCameraPointerEnd(event) {
        if (!_camPanLast) return;
        _camPanLast = null;
        try { _camStage.releasePointerCapture(event.pointerId); } catch (_) {}
        if (_camStage) _camStage.classList.remove('camera-panning');
    }

    function onCameraDoubleClick(event) {
        if (_camZoom !== 1 || _camTx !== 0 || _camTy !== 0) {
            event.preventDefault();
            resetCameraZoom();
        }
    }

    function setupCameraWiring() {
        if (!_camToggle) return;
        _camToggle.addEventListener('click', toggleCameraStream);
        applyCameraState(false);
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('BOTTOM_CAMERA_FRAME', handleCameraFrame);
        }
        // Camera zoom/pan. wheel needs passive:false so we can preventDefault
        // and stop the page from scrolling beneath the FOV.
        if (_camStage) {
            _camStage.addEventListener('wheel', onCameraWheel, { passive: false });
            _camStage.addEventListener('pointerdown', onCameraPointerDown);
            _camStage.addEventListener('pointermove', onCameraPointerMove);
            _camStage.addEventListener('pointerup', onCameraPointerEnd);
            _camStage.addEventListener('pointercancel', onCameraPointerEnd);
            _camStage.addEventListener('dblclick', onCameraDoubleClick);
        }
    }

    // =====================================================================
    // Lightsheet live panel (Manual view)
    // =====================================================================

    /** Gate ALL lasers off via the Laser "ALL OFF" config-group preset.
     *  Updates the indicator span from the actual API result (not a static label).
     *  Fire-and-forget safe — failure shows a warning, never throws. */
    async function setLaserOff() {
        cacheDom();
        try {
            const res = await fetch('/api/devices/laser/off', { method: 'POST' });
            if (_lsLaserStatus) {
                _lsLaserStatus.textContent = res.ok ? 'OFF (brightfield)' : 'warning: state unknown';
            }
            if (res.ok) _setLaserToggleState(false);
        } catch (err) {
            if (_lsLaserStatus) _lsLaserStatus.textContent = 'warning: state unknown';
            console.debug('laser off call failed:', err);
        }
    }

    /** Fetch laser config-group presets and populate the #devices-laser-preset select.
     *  Selects "ALL OFF" by default (entry safety preset).
     *  Wires the change handler to POST the selected preset.
     *  Fire-and-forget safe — failure leaves the fallback "ALL OFF" option in place. */
    async function populateLaserPresets() {
        cacheDom();
        if (!_lsLaserPreset) return;
        try {
            const res = await fetch('/api/devices/laser/configs');
            if (!res.ok) return;
            const data = await res.json();
            // data may be an array of preset names or {configs: [...]}
            const presets = Array.isArray(data) ? data : (data.configs || []);
            if (!presets.length) return;
            // Rebuild option list
            _lsLaserPreset.innerHTML = '';
            for (const name of presets) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                _lsLaserPreset.appendChild(opt);
            }
            // Default to "ALL OFF" — entry safety state
            if (presets.includes('ALL OFF')) _lsLaserPreset.value = 'ALL OFF';
            // Wire change handler — only POST if laser is currently ON; if OFF, it's
            // just a selection that will be activated when the toggle is pressed.
            _lsLaserPreset.onchange = () => { if (_lsLaserOn) setLaserPreset(_lsLaserPreset.value); };
        } catch (err) {
            console.debug('laser preset fetch failed:', err);
        }
    }

    /** Fetch SPIM camera roles and show the Side A/B selector if camera_b is present.
     *  Called on manual-view entry.  Hides the selector on single-camera rigs.
     *  Fire-and-forget safe — failure leaves the selector hidden (safe default). */
    async function populateCameraRoles() {
        cacheDom();
        const group = document.getElementById('devices-ls-side-group');
        try {
            const res = await fetch('/api/devices/cameras');
            if (!res.ok) return;
            const data = await res.json();
            // data may be {cameras: [...]} or a raw array
            const cameras = Array.isArray(data) ? data : (data.cameras || []);
            const hasSideB = cameras.includes('B');
            if (group) group.style.display = hasSideB ? '' : 'none';
            if (_lsSideSelect && hasSideB) {
                _lsSideSelect.onchange = () => {
                    _lsSide = _lsSideSelect.value;
                    postLightsheetParams();
                };
            }
        } catch (err) {
            console.debug('camera roles fetch failed:', err);
        }
    }

    /** POST a named laser preset to the device layer.
     *  Updates the status indicator on success.
     *  Fire-and-forget safe — failure shows a warning, never throws. */
    async function setLaserPreset(config) {
        cacheDom();
        if (!config) return;
        try {
            const res = await fetch('/api/devices/laser/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config }),
            });
            if (_lsLaserStatus) {
                _lsLaserStatus.textContent = res.ok ? config : 'warning: state unknown';
            }
            if (res.ok) _setLaserToggleState(config !== 'ALL OFF');
            if (!res.ok) console.debug('laser preset set failed:', await res.text());
        } catch (err) {
            if (_lsLaserStatus) _lsLaserStatus.textContent = 'warning: state unknown';
            console.debug('laser preset set failed:', err);
        }
    }

    // =====================================================================
    // Timelapse config form (Manual view)
    // =====================================================================

    // ── Accordion section summary builders ───────────────────────────────────

    function _tlSchedSummary() {
        const interval = (_tlInterval && _tlInterval.value) ? _tlInterval.value : '120';
        const stop     = (_tlStop    && _tlStop.value)     ? _tlStop.value     : 'manual';
        const condVal  = (_tlCondVal && _tlCondVal.value)  ? _tlCondVal.value  : '10';
        if (stop === 'timepoints') return `${interval} s · ${condVal} frames`;
        if (stop === 'duration')   return `${interval} s · ${condVal} h`;
        return `${interval} s · manual`;
    }

    function _tlTargetsSummary() {
        const embryos  = (_tlEmbryos && _tlEmbryos.value.trim())
            ? _tlEmbryos.value.trim()
            : 'all';
        const modeEl   = _tlMode;
        const modeText = (modeEl && modeEl.value)
            ? modeEl.options[modeEl.selectedIndex].text
            : 'none';
        return `${embryos} · ${modeText}`;
    }

    function _tlGeomSummary() {
        const slices   = (_tlSlices   && _tlSlices.value)   ? _tlSlices.value   : '50';
        const exposure = (_tlExposure && _tlExposure.value) ? _tlExposure.value : '10';
        const laser    = (_tlLaser    && _tlLaser.value)    ? _tlLaser.value    : 'ALL OFF';
        return `${slices} sl · ${exposure} ms · ${laser}`;
    }

    /** Update a section's header active state and summary text, then sync the
     *  outer panel dot and the start button.  sec = 'sched'|'targets'|'geom'. */
    function _tlUpdateSection(sec) {
        const head    = document.getElementById(`devices-tlacc-${sec}-head`);
        const summary = document.getElementById(`devices-tlacc-${sec}-sum`);
        const touched = _tlTouched[sec];

        if (head)    head.classList.toggle('is-active', touched);
        if (summary) {
            summary.hidden = !touched;
            if (touched) {
                if      (sec === 'sched')   summary.textContent = _tlSchedSummary();
                else if (sec === 'targets') summary.textContent = _tlTargetsSummary();
                else if (sec === 'geom')    summary.textContent = _tlGeomSummary();
            }
        }

        // Outer panel dot + start button "ready" state
        const anyActive = Object.values(_tlTouched).some(Boolean);
        const outerDot  = document.getElementById('devices-tl-outer-dot');
        if (outerDot) outerDot.classList.toggle('is-active', anyActive);
        if (_tlStart) _tlStart.classList.toggle('is-ready', anyActive);
    }

    /** Wire the timelapse panel: outer collapsible toggle, accordion section
     *  toggles, touch listeners, and the submit button.
     *  Safe to call multiple times (re-assigns handlers idempotently). */
    function initTlForm() {
        cacheDom();

        // Reset touched state on each init (re-entering the manual view = fresh)
        _tlTouched = { sched: false, targets: false, geom: false };
        // Clear any leftover active-state visuals from a previous visit
        ['sched', 'targets', 'geom'].forEach(sec => _tlUpdateSection(sec));

        // ── Outer collapsible toggle ──────────────────────────────────────────
        if (_tlToggle && _tlBody) {
            _tlToggle.onclick = () => {
                const open = _tlBody.hidden;
                _tlBody.hidden = !open;
                _tlToggle.setAttribute('aria-expanded', String(open));
                const arrow = _tlToggle.querySelector('.ls-collapsible-arrow');
                if (arrow) arrow.textContent = open ? '▼' : '▶';
            };
        }

        // ── Accordion section toggles ─────────────────────────────────────────
        ['sched', 'targets', 'geom'].forEach(sec => {
            const head = document.getElementById(`devices-tlacc-${sec}-head`);
            const body = document.getElementById(`devices-tlacc-${sec}-body`);
            if (!head || !body) return;
            head.onclick = () => {
                const open = body.hidden;
                body.hidden = !open;
                head.setAttribute('aria-expanded', String(open));
                const arrow = head.querySelector('.ls-acc-arrow');
                if (arrow) arrow.textContent = open ? '▼' : '▶';
            };
        });

        // ── Touch listeners ───────────────────────────────────────────────────
        const markTouched = sec => {
            _tlTouched[sec] = true;
            _tlUpdateSection(sec);
        };

        // Schedule — interval and stop condition drive summary; cond-row visibility unchanged
        [_tlInterval, _tlCondVal].forEach(el => {
            if (el) el.addEventListener('input', () => markTouched('sched'));
        });
        if (_tlStop) {
            _tlStop.addEventListener('change', () => {
                const v = _tlStop.value;
                const show = v === 'timepoints' || v === 'duration';
                if (_tlCondRow)   _tlCondRow.hidden = !show;
                if (_tlCondLabel) _tlCondLabel.textContent = v === 'duration' ? 'Hours' : 'Count';
                markTouched('sched');
            });
        }

        // Targets
        if (_tlEmbryos) _tlEmbryos.addEventListener('input',  () => markTouched('targets'));
        if (_tlMode)    _tlMode.addEventListener('change',    () => markTouched('targets'));

        // Volume geometry
        [_tlSlices, _tlExposure, _tlGalvoAmp, _tlGalvoCtr, _tlPiezoAmp, _tlPiezoCtr].forEach(el => {
            if (el) el.addEventListener('input', () => markTouched('geom'));
        });
        if (_tlLaser) _tlLaser.addEventListener('change', () => markTouched('geom'));

        // ── Submit ────────────────────────────────────────────────────────────
        if (_tlStart) _tlStart.onclick = startTimelapse;
    }

    /** Populate timelapse volume-geometry defaults from GET /api/devices/scan_geometry,
     *  and populate the laser preset select from GET /api/devices/laser/configs.
     *  Fire-and-forget safe — failure leaves form-coded defaults in place. */
    async function populateTlDefaults() {
        cacheDom();
        // Geometry defaults
        try {
            const res = await fetch('/api/devices/scan_geometry');
            if (res.ok) {
                const data = await res.json();
                const scan = data.scan || {};
                if (_tlSlices    && scan.num_slices    != null) _tlSlices.value    = scan.num_slices;
                if (_tlExposure  && scan.exposure_ms   != null) _tlExposure.value  = scan.exposure_ms;
                if (_tlGalvoAmp  && scan.galvo_amplitude_deg != null) _tlGalvoAmp.value = scan.galvo_amplitude_deg;
                if (_tlGalvoCtr  && scan.galvo_center_deg    != null) _tlGalvoCtr.value = scan.galvo_center_deg;
                if (_tlPiezoAmp  && scan.piezo_amplitude_um  != null) _tlPiezoAmp.value = scan.piezo_amplitude_um;
                if (_tlPiezoCtr  && scan.piezo_center_um     != null) _tlPiezoCtr.value = scan.piezo_center_um;
            }
        } catch (err) {
            console.debug('tl scan_geometry fetch failed:', err);
        }
        // Laser presets — reuse the shared endpoint; mirror populateLaserPresets()
        if (!_tlLaser) return;
        try {
            const res = await fetch('/api/devices/laser/configs');
            if (!res.ok) return;
            const data = await res.json();
            const presets = Array.isArray(data) ? data : (data.configs || []);
            if (!presets.length) return;
            _tlLaser.innerHTML = '';
            for (const name of presets) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                _tlLaser.appendChild(opt);
            }
            if (presets.includes('ALL OFF')) _tlLaser.value = 'ALL OFF';
        } catch (err) {
            console.debug('tl laser configs fetch failed:', err);
        }
    }

    /** Gather form values, POST to /api/devices/timelapse/start, show result. */
    async function startTimelapse() {
        cacheDom();
        if (!_tlStart) return;
        _tlStart.disabled = true;

        // Build payload
        const interval = parseFloat(_tlInterval ? _tlInterval.value : '120') || 120;
        const stop_condition = _tlStop ? _tlStop.value : 'manual';
        const condRaw = _tlCondVal ? _tlCondVal.value : '';
        const condition_value = condRaw ? parseInt(condRaw, 10) : null;
        const embryoRaw = _tlEmbryos ? _tlEmbryos.value.trim() : '';
        const embryo_ids = embryoRaw
            ? embryoRaw.split(',').map(s => s.trim()).filter(Boolean)
            : null;
        const monitoring_mode = _tlMode ? (_tlMode.value || null) : null;

        const payload = {
            interval_seconds: interval,
            stop_condition,
            embryo_ids,
            condition_value,
            monitoring_mode,
            num_slices:      _tlSlices    ? parseInt(_tlSlices.value,    10) : 50,
            exposure_ms:     _tlExposure  ? parseFloat(_tlExposure.value)    : 10.0,
            galvo_amplitude: _tlGalvoAmp  ? parseFloat(_tlGalvoAmp.value)    : 0.5,
            galvo_center:    _tlGalvoCtr  ? parseFloat(_tlGalvoCtr.value)    : 0.0,
            piezo_amplitude: _tlPiezoAmp  ? parseFloat(_tlPiezoAmp.value)    : 25.0,
            piezo_center:    _tlPiezoCtr  ? parseFloat(_tlPiezoCtr.value)    : 50.0,
            laser_config:    _tlLaser     ? (_tlLaser.value || null)          : null,
        };

        if (_tlStatus)     _tlStatus.hidden = false;
        if (_tlStatusText) _tlStatusText.textContent = 'Starting…';

        try {
            const res = await fetch('/api/devices/timelapse/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const body = await res.json().catch(() => ({}));
            if (res.ok) {
                const msg = body.result || 'Timelapse started.';
                if (_tlStatusText) _tlStatusText.textContent = msg;
            } else {
                const detail = body.detail || `error ${res.status}`;
                if (_tlStatusText) _tlStatusText.textContent = `Error: ${detail}`;
                console.debug('timelapse start failed:', body);
            }
        } catch (err) {
            if (_tlStatusText) _tlStatusText.textContent = `Network error: ${err.message}`;
            console.debug('timelapse start failed:', err);
        } finally {
            if (_tlStart) _tlStart.disabled = false;
        }
    }

    async function toggleLightsheetStream() {
        if (!_lsToggle) return;
        _lsToggle.disabled = true;
        try {
            const starting = !_lsStreaming;
            const endpoint = _lsStreaming
                ? '/api/devices/lightsheet/live/stop'
                : '/api/devices/lightsheet/live/start';
            const res = await fetch(endpoint, { method: 'POST' });
            if (!res.ok) {
                const detail = await res.text();
                console.error('Lightsheet toggle failed:', detail);
                if (_lsMeta) _lsMeta.textContent = `error: ${res.status}`;
                return;
            }
            const data = await res.json();
            applyLightsheetState(!!data.streaming);
            // Gate lasers off whenever live starts — brightfield-safe by default.
            if (starting && data.streaming) setLaserOff();
        } catch (err) {
            console.error('Lightsheet toggle failed:', err);
            if (_lsMeta) _lsMeta.textContent = `error: ${err}`;
        } finally {
            _lsToggle.disabled = false;
        }
    }

    function applyLightsheetState(streaming) {
        _lsStreaming = streaming;
        if (_lsToggle) {
            _lsToggle.textContent = streaming ? 'Stop' : 'Start';
            _lsToggle.classList.toggle('active', streaming);
        }
        if (_lsLed) {
            _lsLed.classList.toggle('live', streaming);
            _lsLed.classList.remove('stale');
        }
        if (!streaming) {
            _lsHasFrame = false;
            _lsFrameTimes = [];
            if (_lsImg) _lsImg.classList.remove('has-frame');
            if (_lsPlaceholder) _lsPlaceholder.hidden = false;
            if (_lsMeta) _lsMeta.textContent = 'stream off';
            if (_lsStaleTimer) { clearTimeout(_lsStaleTimer); _lsStaleTimer = null; }
            resetLightsheetZoom();
        } else {
            _lsFrameTimes = [];
            if (_lsMeta) _lsMeta.textContent = 'waiting…';
        }
    }

    function handleLightsheetFrame(payload) {
        if (!payload || !payload.jpeg_b64 || !_lsImg) return;
        _lsImg.src = `data:${payload.mime || 'image/jpeg'};base64,${payload.jpeg_b64}`;
        if (!_lsHasFrame) {
            _lsHasFrame = true;
            _lsImg.classList.add('has-frame');
            if (_lsPlaceholder) _lsPlaceholder.hidden = true;
        }
        const now = performance.now();
        _lsLastFrameTs = Date.now();
        _lsFrameTimes.push(now);
        if (_lsFrameTimes.length > _LS_FPS_WINDOW) _lsFrameTimes.shift();
        if (_lsLed) {
            _lsLed.classList.add('live');
            _lsLed.classList.remove('stale');
        }
        if (_lsMeta) {
            const shape = payload.shape || [];
            const dims = shape.length === 2 ? `${shape[1]}×${shape[0]}` : '';
            const fps = computeLightsheetFps();
            _lsMeta.textContent = dims
                ? `${dims}  ·  ${fps != null ? fps.toFixed(1) + ' fps' : '…'}`
                : (fps != null ? `${fps.toFixed(1)} fps` : 'live');
        }
        scheduleLightsheetStaleCheck();
    }

    function computeLightsheetFps() {
        const n = _lsFrameTimes.length;
        if (n < 2) return null;
        const span = _lsFrameTimes[n - 1] - _lsFrameTimes[0];
        if (span <= 0) return null;
        return ((n - 1) * 1000) / span;
    }

    function scheduleLightsheetStaleCheck() {
        if (_lsStaleTimer) clearTimeout(_lsStaleTimer);
        _lsStaleTimer = setTimeout(() => {
            const age = (Date.now() - _lsLastFrameTs) / 1000;
            if (_lsMeta) _lsMeta.textContent = `last frame ${age.toFixed(1)}s ago`;
            if (_lsLed) _lsLed.classList.add('stale');
        }, 1500);
    }

    async function syncInitialLightsheetState() {
        try {
            const res = await fetch('/api/devices/lightsheet/live/status');
            if (!res.ok) return;
            const data = await res.json();
            applyLightsheetState(!!data.streaming);
        } catch (err) {
            console.debug('lightsheet status check failed:', err);
        }
    }

    // ---- Lightsheet zoom / pan (mirrors camera zoom/pan) ----------------
    function applyLightsheetTransform() {
        if (!_lsImg) return;
        _lsImg.style.transform =
            `translate(${_lsTx}px, ${_lsTy}px) scale(${_lsZoom})`;
    }

    function resetLightsheetZoom() {
        _lsZoom = 1;
        _lsTx = 0;
        _lsTy = 0;
        applyLightsheetTransform();
        if (_lsStage) _lsStage.classList.remove('camera-zoomed', 'camera-panning');
    }

    function clampLightsheetPan() {
        if (!_lsStage) return;
        const rect = _lsStage.getBoundingClientRect();
        const maxX = (rect.width  * (_lsZoom - 1)) / 2;
        const maxY = (rect.height * (_lsZoom - 1)) / 2;
        _lsTx = Math.max(-maxX, Math.min(maxX, _lsTx));
        _lsTy = Math.max(-maxY, Math.min(maxY, _lsTy));
    }

    function onLightsheetWheel(event) {
        if (!_lsStage) return;
        event.preventDefault();
        const rect = _lsStage.getBoundingClientRect();
        const cx = event.clientX - rect.left - rect.width  / 2;
        const cy = event.clientY - rect.top  - rect.height / 2;
        const oldZoom = _lsZoom;
        const factor = event.deltaY < 0 ? _CAM_ZOOM_STEP : 1 / _CAM_ZOOM_STEP;
        const newZoom = Math.max(_CAM_ZOOM_MIN, Math.min(_CAM_ZOOM_MAX, oldZoom * factor));
        if (newZoom === oldZoom) return;
        const ratio = newZoom / oldZoom;
        _lsTx = cx - (cx - _lsTx) * ratio;
        _lsTy = cy - (cy - _lsTy) * ratio;
        _lsZoom = newZoom;
        if (Math.abs(_lsZoom - 1) < 0.001) { resetLightsheetZoom(); return; }
        clampLightsheetPan();
        applyLightsheetTransform();
        _lsStage.classList.add('camera-zoomed');
    }

    function onLightsheetPointerDown(event) {
        if (event.button !== 0) return;
        if (_lsZoom <= 1) return;
        _lsPanLast = { x: event.clientX, y: event.clientY };
        try { _lsStage.setPointerCapture(event.pointerId); } catch (_) {}
        _lsStage.classList.add('camera-panning');
        event.preventDefault();
    }

    function onLightsheetPointerMove(event) {
        if (!_lsPanLast) return;
        _lsTx += event.clientX - _lsPanLast.x;
        _lsTy += event.clientY - _lsPanLast.y;
        _lsPanLast = { x: event.clientX, y: event.clientY };
        clampLightsheetPan();
        applyLightsheetTransform();
    }

    function onLightsheetPointerEnd(event) {
        if (!_lsPanLast) return;
        _lsPanLast = null;
        try { _lsStage.releasePointerCapture(event.pointerId); } catch (_) {}
        if (_lsStage) _lsStage.classList.remove('camera-panning');
    }

    function onLightsheetDoubleClick(event) {
        if (_lsZoom !== 1 || _lsTx !== 0 || _lsTy !== 0) {
            event.preventDefault();
            resetLightsheetZoom();
        }
    }

    // ---- Lightsheet live params (debounced) -----------------------------
    function postLightsheetParams() {
        fetch('/api/devices/lightsheet/live/params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ galvo: _lsGalvo, piezo: _lsPiezo, exposure: _lsExposure, side: _lsSide }),
        }).catch(err => console.debug('lightsheet params post failed:', err));
    }

    function scheduleLightsheetParamPost() {
        if (_lsParamTimer) clearTimeout(_lsParamTimer);
        _lsParamTimer = setTimeout(postLightsheetParams, 120);
    }

    function onGalvoInput(src) {
        const v = parseFloat(src.value);
        if (isNaN(v)) return;
        _lsGalvo = v;
        // Sync the sibling control
        if (src === _lsGalvoSlider && _lsGalvoNum) _lsGalvoNum.value = v;
        if (src === _lsGalvoNum   && _lsGalvoSlider) _lsGalvoSlider.value = v;
        scheduleLightsheetParamPost();
    }

    function onPiezoInput(src) {
        const v = parseFloat(src.value);
        if (isNaN(v)) return;
        _lsPiezo = v;
        if (src === _lsPiezoSlider && _lsPiezoNum) _lsPiezoNum.value = v;
        if (src === _lsPiezoNum   && _lsPiezoSlider) _lsPiezoSlider.value = v;
        scheduleLightsheetParamPost();
    }

    function onExposureInput() {
        const v = parseFloat(_lsExposureNum && _lsExposureNum.value);
        if (isNaN(v) || v < 1) return;
        _lsExposure = v;
        scheduleLightsheetParamPost();
    }

    // ---- Illumination toggles -------------------------------------------
    async function postLedPreset(preset) {
        try {
            await fetch('/api/devices/led/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: preset }),
            });
        } catch (err) { console.debug('LED preset failed:', err); }
    }

    /** Single LED toggle — mirrors Cam LED / Room Light aria-pressed pattern.
     *  Flips between Open (active) and Closed (inactive/safe default). */
    async function toggleLedPreset() {
        _lsLedIsOpen = !_lsLedIsOpen;
        if (_lsLedToggle) {
            _lsLedToggle.classList.toggle('ls-illum-btn--active', _lsLedIsOpen);
            _lsLedToggle.setAttribute('aria-pressed', _lsLedIsOpen ? 'true' : 'false');
            _lsLedToggle.textContent = _lsLedIsOpen ? 'LED: Open' : 'LED: Closed';
        }
        await postLedPreset(_lsLedIsOpen ? 'Open' : 'Closed');
    }

    /** Update laser toggle button + dot to reflect on/off state.
     *  Called by setLaserOff() and setLaserPreset() after a successful API call. */
    function _setLaserToggleState(on) {
        _lsLaserOn = on;
        if (_lsLaserToggle) {
            _lsLaserToggle.classList.toggle('ls-illum-btn--active', on);
            _lsLaserToggle.setAttribute('aria-pressed', on ? 'true' : 'false');
            _lsLaserToggle.textContent = on ? 'Laser: ON' : 'Laser: OFF';
        }
        const dot = document.querySelector('.ls-laser-dot');
        if (dot) dot.classList.toggle('ls-laser-dot--on', on);
    }

    /** Laser on/off toggle — OFF fires laser/off; ON applies the selected preset.
     *  If selected preset is "ALL OFF", picks the first non-"ALL OFF" option.
     *  Entry safety: starts OFF (setLaserOff fires on manual-view entry). */
    async function toggleLaser() {
        if (_lsLaserOn) {
            await setLaserOff();
        } else {
            let config = _lsLaserPreset ? _lsLaserPreset.value : null;
            if (!config || config === 'ALL OFF') {
                const opts = _lsLaserPreset ? Array.from(_lsLaserPreset.options) : [];
                const first = opts.find(o => o.value !== 'ALL OFF');
                if (first) {
                    config = first.value;
                    _lsLaserPreset.value = config;
                } else {
                    if (_lsLaserStatus) _lsLaserStatus.textContent = 'select a laser line first';
                    return;
                }
            }
            await setLaserPreset(config);
        }
    }

    async function toggleCamLedMode() {
        _lsCamLedOn = !_lsCamLedOn;
        if (_lsCamLed) {
            _lsCamLed.classList.toggle('ls-illum-btn--active', _lsCamLedOn);
            _lsCamLed.setAttribute('aria-pressed', _lsCamLedOn ? 'true' : 'false');
        }
        try {
            await fetch('/api/devices/camera/led_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ use_led: _lsCamLedOn }),
            });
        } catch (err) { console.debug('cam LED mode failed:', err); }
    }

    async function toggleManualRoomLight() {
        const nextState = _roomLightState === 'on' ? 'off' : 'on';
        if (_lsRoomLightBtn) {
            _lsRoomLightBtn.classList.toggle('ls-illum-btn--active', nextState === 'on');
            _lsRoomLightBtn.setAttribute('aria-pressed', nextState === 'on' ? 'true' : 'false');
        }
        try {
            const res = await fetch('/api/devices/room_light/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: nextState }),
            });
            if (res.ok) {
                const data = await res.json();
                _roomLightState = data.state || nextState;
                if (_lsRoomLightBtn) {
                    const on = _roomLightState === 'on';
                    _lsRoomLightBtn.classList.toggle('ls-illum-btn--active', on);
                    _lsRoomLightBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
                }
            }
        } catch (err) { console.debug('manual room light toggle failed:', err); }
    }

    // ---- Acquire --------------------------------------------------------
    async function runLightsheetAcquire(mode) {
        const btn = mode === 'burst' ? _lsBurstBtn : _lsSnapVolBtn;
        if (btn) { btn.disabled = true; btn.textContent = 'acquiring…'; }
        try {
            let res;
            if (mode === 'burst') {
                res = await fetch('/api/devices/acquire/burst', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frames: 10, mode: 'brightfield',
                                          num_slices: 50, exposure_ms: _lsExposure,
                                          laser_config: 'ALL OFF',
                                          piezo_center: _lsPiezo,
                                          galvo_center: _lsGalvo }),
                });
            } else {
                res = await fetch('/api/devices/acquire/volume', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_slices: 50, exposure_ms: _lsExposure,
                                          laser_config: 'ALL OFF',
                                          piezo_center: _lsPiezo,
                                          galvo_center: _lsGalvo }),
                });
            }
            if (!res.ok) {
                console.error('acquire failed:', res.status, await res.text());
                return;
            }
            const data = await res.json();
            if (_lsLastcap) _lsLastcap.hidden = false;
            if (_lsLastcapRef) {
                _lsLastcapRef.textContent = data.volume_path || data.path || data.id || 'done';
            }
            // Show confirmation toast — no inline preview to keep manual mode uncluttered
            if (typeof showGentlyToast === 'function') {
                const label = mode === 'burst' ? 'Burst acquired' : 'Volume acquired';
                showGentlyToast(label, 'View in Gallery', () => {
                    if (typeof switchTab === 'function' && typeof TABS !== 'undefined') {
                        switchTab(TABS.GALLERY);
                    }
                });
            }
        } catch (err) {
            console.error('acquire error:', err);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = mode === 'burst' ? 'Burst' : 'Snap Volume';
            }
        }
    }

    // ---- Temperature set (rail copy, delegates to same API) -------------
    async function setLightsheetTemperature() {
        if (!_lsTempInput) return;
        const target = parseFloat(_lsTempInput.value);
        if (isNaN(target) || target < 0 || target > 99.9) return;
        try {
            await fetch('/api/devices/temperature/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_c: target }),
            });
        } catch (err) { console.debug('ls temp set failed:', err); }
    }

    function setupManualWiring() {
        if (!_lsToggle) return;
        _lsToggle.addEventListener('click', toggleLightsheetStream);
        applyLightsheetState(false);

        // Param controls — slider ↔ number sync + debounced POST
        if (_lsGalvoSlider) _lsGalvoSlider.addEventListener('input', () => onGalvoInput(_lsGalvoSlider));
        if (_lsGalvoNum)    _lsGalvoNum.addEventListener('input',    () => onGalvoInput(_lsGalvoNum));
        if (_lsPiezoSlider) _lsPiezoSlider.addEventListener('input', () => onPiezoInput(_lsPiezoSlider));
        if (_lsPiezoNum)    _lsPiezoNum.addEventListener('input',    () => onPiezoInput(_lsPiezoNum));
        if (_lsExposureNum) _lsExposureNum.addEventListener('input', onExposureInput);

        // Illumination
        if (_lsLedToggle)    _lsLedToggle.addEventListener('click',    toggleLedPreset);
        if (_lsCamLed)       _lsCamLed.addEventListener('click',       toggleCamLedMode);
        if (_lsRoomLightBtn) _lsRoomLightBtn.addEventListener('click', toggleManualRoomLight);
        if (_lsLaserToggle)  _lsLaserToggle.addEventListener('click',  toggleLaser);

        // Acquire
        if (_lsSnapVolBtn) _lsSnapVolBtn.addEventListener('click', () => runLightsheetAcquire('volume'));
        if (_lsBurstBtn)   _lsBurstBtn.addEventListener('click',   () => runLightsheetAcquire('burst'));

        // Temperature
        if (_lsTempSet) _lsTempSet.addEventListener('click', setLightsheetTemperature);
        if (_lsTempInput) {
            _lsTempInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); setLightsheetTemperature(); }
            });
        }

        // Zoom / pan
        if (_lsStage) {
            _lsStage.addEventListener('wheel', onLightsheetWheel, { passive: false });
            _lsStage.addEventListener('pointerdown', onLightsheetPointerDown);
            _lsStage.addEventListener('pointermove', onLightsheetPointerMove);
            _lsStage.addEventListener('pointerup', onLightsheetPointerEnd);
            _lsStage.addEventListener('pointercancel', onLightsheetPointerEnd);
            _lsStage.addEventListener('dblclick', onLightsheetDoubleClick);
        }

        // Subscribe to LIGHTSHEET_FRAME events
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('LIGHTSHEET_FRAME', handleLightsheetFrame);
        }

        syncInitialLightsheetState();
    }

    // =====================================================================
    // Room-light toggle
    // =====================================================================

    function applyRoomLight(state, available) {
        _roomLightState = state || 'unknown';
        _roomLightAvailable = !!available;
        if (!_roomLightToggle) return;
        _roomLightToggle.hidden = !_roomLightAvailable;
        _roomLightToggle.disabled = !_roomLightAvailable || _roomLightBusy;
        const on = _roomLightState === 'on';
        _roomLightToggle.classList.toggle('is-on', on);
        _roomLightToggle.setAttribute('aria-pressed', on ? 'true' : 'false');
        if (_roomLightLabel && !_roomLightBusy) {
            _roomLightLabel.textContent = on ? 'Room light: on'
                : (_roomLightState === 'off' ? 'Room light: off' : 'Room light');
        }
    }

    async function loadRoomLightStatus() {
        if (!_roomLightToggle || _roomLightBusy) return;
        try {
            const res = await fetch('/api/devices/room_light/status');
            if (!res.ok) { applyRoomLight('unknown', false); return; }
            const data = await res.json();
            applyRoomLight(data.state, data.available);
        } catch (err) {
            console.debug('room light status fetch failed:', err);
            applyRoomLight('unknown', false);
        }
    }

    async function toggleRoomLight() {
        if (!_roomLightToggle || _roomLightBusy || !_roomLightAvailable) return;
        const next = _roomLightState === 'on' ? 'off' : 'on';
        _roomLightBusy = true;
        _roomLightToggle.classList.add('is-busy');
        _roomLightToggle.disabled = true;
        if (_roomLightLabel) {
            _roomLightLabel.textContent = next === 'on' ? 'Turning on…' : 'Turning off…';
        }

        // Settle back to the resolved state, or surface a transient message
        // (insufficient control / error) for 2 s before reverting.
        const finish = (msg) => {
            _roomLightBusy = false;
            _roomLightToggle.classList.remove('is-busy');
            if (msg) {
                if (_roomLightLabel) _roomLightLabel.textContent = msg;
                _roomLightToggle.disabled = false;
                setTimeout(() => applyRoomLight(_roomLightState, _roomLightAvailable), 2000);
            } else {
                applyRoomLight(_roomLightState, _roomLightAvailable);
            }
        };

        try {
            const res = await fetch('/api/devices/room_light/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: next }),
            });
            if (res.status === 401 || res.status === 403) { finish('Need control'); return; }
            if (!res.ok) {
                console.error('room light set failed:', await res.text());
                finish('Error');
                return;
            }
            const data = await res.json();
            _roomLightState = data.state || next;
            finish(null);
        } catch (err) {
            console.error('room light toggle failed:', err);
            finish('Error');
        }
    }

    function setupRoomLight() {
        if (!_roomLightToggle) return;
        _roomLightToggle.addEventListener('click', toggleRoomLight);
        loadRoomLightStatus();
        // Light periodic refresh: state can also change from agent plans
        // (e.g. brightfield imaging turns it on). Status read is cached at the
        // device layer (no BLE), so polling is cheap; it also makes the toggle
        // appear automatically once the device layer connects.
        if (_roomLightTimer) clearInterval(_roomLightTimer);
        _roomLightTimer = setInterval(loadRoomLightStatus, 15000);
    }

    // =====================================================================
    // Temperature controller (ACUITYnano) — readout + setpoint
    // =====================================================================

    function fmtTemp(v) {
        return (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(1) + '°';
    }

    function applyTemperature(data) {
        _tempAvailable = !!(data && data.available);
        if (!_tempEl) return;
        _tempEl.hidden = !_tempAvailable;
        if (!_tempAvailable) return;
        _tempState = (data && data.state) || 'unknown';
        const locked = /LOCK/i.test(_tempState);
        _tempEl.classList.toggle('is-locked', locked);
        if (_tempBusy) return;  // a set() is in flight; leave its transient label
        const cur = fmtTemp(data.temperature_c);
        const hasSp = data.setpoint_c !== null && data.setpoint_c !== undefined;
        const sp = hasSp ? fmtTemp(data.setpoint_c) : null;
        _tempReadout.textContent = sp ? (cur + ' → ' + sp) : cur;
        _tempReadout.title = 'Water ' + cur + (sp ? (', setpoint ' + sp) : '')
            + (locked ? ' (locked)' : '');
        // Seed the input with the current setpoint once, while untouched, so the
        // operator sees where it is before nudging it.
        if (_tempInput && document.activeElement !== _tempInput && _tempInput.value === '' && hasSp) {
            _tempInput.value = Number(data.setpoint_c).toFixed(1);
        }
    }

    async function loadTemperatureStatus() {
        if (!_tempEl || _tempBusy) return;
        try {
            const res = await fetch('/api/devices/temperature/status');
            if (!res.ok) { applyTemperature({ available: false }); return; }
            applyTemperature(await res.json());
        } catch (err) {
            console.debug('temperature status fetch failed:', err);
            applyTemperature({ available: false });
        }
    }

    async function setTemperature() {
        if (!_tempEl || _tempBusy || !_tempAvailable) return;
        const target = parseFloat(_tempInput && _tempInput.value);
        if (isNaN(target) || target < 0 || target > 99.9) {
            _tempReadout.textContent = '0–99.9 °C';
            setTimeout(loadTemperatureStatus, 1500);
            return;
        }
        _tempBusy = true;
        _tempEl.classList.add('is-busy');
        if (_tempSet) _tempSet.disabled = true;
        _tempReadout.textContent = 'Set ' + target.toFixed(1) + '°…';

        // Settle back to the resolved state, or surface a transient message
        // (insufficient control / error) for 2 s before reverting.
        const finish = (msg) => {
            _tempBusy = false;
            _tempEl.classList.remove('is-busy');
            if (_tempSet) _tempSet.disabled = false;
            if (msg) {
                _tempReadout.textContent = msg;
                setTimeout(loadTemperatureStatus, 2000);
            } else {
                loadTemperatureStatus();  // controller ramps; poll shows progress
            }
        };

        try {
            const res = await fetch('/api/devices/temperature/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_c: target }),
            });
            if (res.status === 401 || res.status === 403) { finish('Need control'); return; }
            if (!res.ok) {
                console.error('temperature set failed:', await res.text());
                finish('Error');
                return;
            }
            await res.json();
            finish(null);
        } catch (err) {
            console.error('temperature set failed:', err);
            finish('Error');
        }
    }

    function setupTemperature() {
        if (!_tempEl) return;
        if (_tempSet) _tempSet.addEventListener('click', setTemperature);
        if (_tempInput) {
            _tempInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); setTemperature(); }
            });
        }
        loadTemperatureStatus();
        // Periodic refresh: the setpoint can also change from agent plans, and a
        // commanded ramp settles over time. Status is cached at the device layer,
        // so polling is cheap; it also reveals the control once the layer connects.
        if (_tempTimer) clearInterval(_tempTimer);
        _tempTimer = setInterval(loadTemperatureStatus, 15000);
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
        // The 3D optical-space view owns its own WebGL module. Build it lazily
        // on first activation (the panel was display:none, so its container had
        // no size until now); init() is idempotent and resizes on re-entry.
        if (viewName === 'optical3d' && typeof Occupancy3DManager !== 'undefined') {
            Occupancy3DManager.init();
        }
        // Initialize temperature graph for the active view. The TemperatureGraph
        // is a singleton; reinit on view switch ensures only one graph target
        // is live at a time. ClientEventBus.off/on in TemperatureGraph.init
        // makes re-init safe (idempotent).
        if (window.TemperatureGraph) {
            if (viewName === 'manual') {
                const el = document.getElementById('devices-ls-tempgraph');
                if (el) TemperatureGraph.init(el, 'current');
            } else {
                const el = document.getElementById('devices-temp-graph');
                if (el) TemperatureGraph.init(el, 'current');
            }
        }
        // Entering Manual view — gate lasers off immediately (brightfield-safe).
        // populateLaserPresets() runs after setLaserOff() so the select is always
        // seeded with the entry-safety state first.
        if (viewName === 'manual') { setLaserOff(); populateLaserPresets(); populateCameraRoles(); initTlForm(); populateTlDefaults(); }
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
            else if (e.key === 'v') { e.preventDefault(); switchView('optical3d'); }
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
        setupManualWiring();
        setupRoomLight();
        setupTemperature();
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
        // Stop the camera and lightsheet streams if the tab is closed while
        // running, so MMCore isn't held by a disconnected browser.
        window.addEventListener('beforeunload', () => {
            if (_camStreaming) {
                try { navigator.sendBeacon('/api/devices/bottom_camera/stream/stop'); } catch (_) {}
            }
            if (_lsStreaming) {
                try { navigator.sendBeacon('/api/devices/lightsheet/live/stop'); } catch (_) {}
            }
        });
    }

    return { init, handlePayload, switchView };
})();

document.addEventListener('DOMContentLoaded', () => DevicesManager.init());
