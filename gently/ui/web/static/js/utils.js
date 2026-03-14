// ══════════════════════════════════════════════════════════
//  Shared utilities – loaded before all other app scripts
// ══════════════════════════════════════════════════════════

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
