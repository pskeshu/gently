// ══════════════════════════════════════════════════════════
//  Shared utilities – loaded before all other app scripts
// ══════════════════════════════════════════════════════════

// Tab and view name constants
const TABS = { HOME: 'home', EMBRYOS: 'embryos', CALIBRATION: 'calibration', EVENTS: 'events', PLANS: 'plans', SESSIONS: 'sessions', DEVICES: 'devices', EXPERIMENT: 'experiment', NOTEBOOK: 'notebook', GALLERY: 'gallery' };

/**
 * Extract the XY firmware fence (the addressable stage box) from a device-state
 * properties map. The ASI adapter exposes LowerLimX/UpperLimX/LowerLimY/
 * UpperLimY in mm; we convert to µm. Single source of truth for both the 2D
 * devices map and the 3D optical-space view.
 *
 * @param {Object} propsByDevice - payload.properties from DEVICE_STATE_UPDATE
 * @returns {{x:[number,number], y:[number,number]}|null}
 */
function extractFirmwareBox(propsByDevice) {
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

/**
 * Build a coordinate mapper from device microns to Three.js scene units.
 * Centers each axis on its range midpoint and divides by the LARGEST span so
 * the whole scene fits a ~[-0.5, 0.5] cube while keeping axes proportional
 * (anisotropic Z vs XY preserved). Returns helpers used by all scene objects
 * so a single scale governs geometry and camera distance.
 *
 * @param {{xRange:[number,number], yRange:[number,number], zRange:[number,number]}} ranges (µm)
 */
function makeSceneScaler(ranges) {
    const xc = (ranges.xRange[0] + ranges.xRange[1]) / 2;
    const yc = (ranges.yRange[0] + ranges.yRange[1]) / 2;
    const zc = (ranges.zRange[0] + ranges.zRange[1]) / 2;
    const xs = Math.abs(ranges.xRange[1] - ranges.xRange[0]);
    const ys = Math.abs(ranges.yRange[1] - ranges.yRange[0]);
    const zs = Math.abs(ranges.zRange[1] - ranges.zRange[0]);
    const maxExtent = Math.max(xs, ys, zs, 1e-6);
    return {
        maxExtent,
        center: { x: xc, y: yc, z: zc },
        // Map an absolute µm position on one axis into scene space.
        toScene(um, axis) {
            const c = axis === 'x' ? xc : axis === 'y' ? yc : zc;
            return (um - c) / maxExtent;
        },
        // Map a µm length (span) into scene units (no centering).
        scaleLen(um) {
            return um / maxExtent;
        },
    };
}

/**
 * HTML-escape a string (safe for insertion into innerHTML).
 * Uses the browser's built-in text node escaping.
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

/**
 * Format an ISO date string for display.
 * @param {string} isoStr
 * @returns {string} e.g. "Mar 14, 2:30 PM"
 */
function formatDate(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
            ', ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    } catch { return isoStr; }
}

/**
 * Initialize a view-switcher: click delegation + keyboard shortcuts.
 *
 * @param {string}   containerId  - ID of the `.view-switcher` container element.
 * @param {Function} callback     - Called with the view name string when a view is selected.
 * @param {Object}   [opts]
 * @param {string[]} [opts.views] - Ordered view names for keyboard shortcuts (keys 1-N).
 * @param {Function} [opts.guard] - Return true to allow shortcuts (e.g., check active tab).
 */
function initViewSwitcher(containerId, callback, opts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.addEventListener('click', e => {
        const btn = e.target.closest('[data-view]');
        if (btn) callback(btn.dataset.view);
    });
    const views = opts?.views;
    const guard = opts?.guard;
    if (views && views.length) {
        document.addEventListener('keydown', e => {
            if (guard && !guard()) return;
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const idx = parseInt(e.key) - 1;
            if (idx >= 0 && idx < views.length) callback(views[idx]);
        });
    }
}

/**
 * Update view-switcher button active states.
 * @param {string} containerId - ID of the `.view-switcher` container.
 * @param {string} activeView  - The data-view value to mark active.
 */
function updateViewButtons(containerId, activeView) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    container.querySelector(`[data-view="${activeView}"]`)?.classList.add('active');
}

/**
 * Toggle a dropdown element's visibility and register a one-shot
 * outside-click handler to close it.
 *
 * @param {HTMLElement} dropdownEl  - The element to show/hide (via the `hidden` class).
 * @param {Event}       event      - The triggering click event (will be stopPropagated).
 */
const toggleDropdown = (() => {
    // Track the current close handler so double-opens don't leak listeners.
    let _activeClose = null;

    return function toggleDropdown(dropdownEl, event) {
        event.stopPropagation();
        if (!dropdownEl) return;

        // If there is already a close handler for a *different* dropdown, fire it first.
        if (_activeClose && _activeClose._dropdown !== dropdownEl) {
            _activeClose._dropdown.classList.add('hidden');
            document.removeEventListener('click', _activeClose);
            _activeClose = null;
        }

        dropdownEl.classList.toggle('hidden');

        if (!dropdownEl.classList.contains('hidden')) {
            const close = (e) => {
                if (!dropdownEl.contains(e.target)) {
                    dropdownEl.classList.add('hidden');
                    document.removeEventListener('click', close);
                    _activeClose = null;
                }
            };
            close._dropdown = dropdownEl;
            _activeClose = close;
            setTimeout(() => document.addEventListener('click', close), 0);
        } else {
            // Toggled closed manually – clean up any pending listener.
            if (_activeClose && _activeClose._dropdown === dropdownEl) {
                document.removeEventListener('click', _activeClose);
                _activeClose = null;
            }
        }
    };
})();
