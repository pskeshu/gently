// ══════════════════════════════════════════════════════════
//  Shared utilities – loaded before all other app scripts
// ══════════════════════════════════════════════════════════

// Tab and view name constants
const TABS = { EMBRYOS: 'embryos', CALIBRATION: 'calibration', EVENTS: 'events', PLANS: 'plans', SESSIONS: 'sessions', DEVICES: 'devices', EXPERIMENT: 'experiment' };

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
