/**
 * Devices tab — read-only stream of MMCore device state.
 *
 * Subscribes to DEVICE_STATE_UPDATE events on the client event bus
 * (forwarded from the device-layer SSE stream by DeviceStateMonitor) and
 * renders positions + properties into the Devices tab.
 *
 * No write controls — this surface is purely observational.
 */
const DevicesManager = (function () {
    const STALE_AFTER_MS = 4000;   // pill flips to "stale" if no payload for this long

    // Cached DOM refs (filled lazily on first update)
    let _statusPill, _statusMeta, _statusDot;
    let _posX, _posY, _piezoZ, _galvoA, _galvoB;
    let _tbody, _filter;

    let _lastTs = 0;
    let _previousTs = 0;
    let _lastWallTs = 0;
    let _staleTimer = null;
    let _filterText = '';
    let _lastPropertyMap = {};   // device -> { property -> value, __type__ -> str }

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
        // Payload is keyed by device name with a `kind` tag:
        //   { "<name>": {X, Y, kind: "xy_stage"}, ... }
        for (const devName of Object.keys(positions)) {
            const entry = positions[devName] || {};
            switch (entry.kind) {
                case 'xy_stage':
                    setAxis(_posX, entry.X, 2);
                    setAxis(_posY, entry.Y, 2);
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

    // MMCore DeviceType enum -> short label. Values per
    // pymmcore.DeviceType / MMDeviceConstants.h
    const DEVICE_TYPE_LABEL = {
        0: 'Unknown', 1: 'Any', 2: 'Camera', 3: 'Shutter',
        4: 'XY', 5: 'Stage', 6: 'State', 7: 'Serial',
        8: 'Generic', 9: 'AutoFocus', 10: 'Core', 11: 'Image',
        12: 'Signal IO', 13: 'Magnifier', 14: 'SLM', 15: 'Hub',
        16: 'Galvo',
    };

    function flattenProperties(propsByDevice) {
        // Returns sorted array of { device, type, property, value, changed }
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
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('DEVICE_STATE_UPDATE', handlePayload);
        }
        setStatus('stale', 'waiting', 'no payload yet');
    }

    return { init, handlePayload };
})();

document.addEventListener('DOMContentLoaded', () => DevicesManager.init());
