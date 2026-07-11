// control-auth.js — friendly, actionable hint when a control action is denied.
//
// Control routes (POST/PUT/DELETE that move hardware or mutate state) return 403
// when the session lacks the control role — e.g. account mode with no operator
// logged in. Without this, the only signal is a bare 403 in the console and a
// button that silently does nothing. This installs a single global fetch wrapper
// that surfaces a throttled "Control required — Log in" toast instead. It only
// reads res.status (never consumes the body), so callers behave unchanged.
(function () {
    if (window.__gentlyControlAuthPatched) return;
    window.__gentlyControlAuthPatched = true;

    const origFetch = window.fetch.bind(window);
    let lastHintAt = 0;

    function isControlRequest(input, init) {
        let method = (init && init.method) || (typeof input === 'object' && input && input.method) || 'GET';
        method = String(method).toUpperCase();
        if (method === 'GET' || method === 'HEAD' || method === 'OPTIONS') return false;
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        return url.includes('/api/');
    }

    function showLoginHint() {
        const now = Date.now();
        if (now - lastHintAt < 4000) return;  // throttle so repeated clicks don't stack
        lastHintAt = now;
        const msg = 'Control required — you are in view-only mode.';
        if (typeof showGentlyToast === 'function') {
            showGentlyToast(msg, 'Log in', () => { window.location.href = '/login'; }, 7000);
        } else {
            console.warn(msg + ' Log in at /login to drive hardware.');
        }
    }

    window.fetch = async function (input, init) {
        const res = await origFetch(input, init);
        try {
            if (res.status === 403 && isControlRequest(input, init)) {
                showLoginHint();
            }
        } catch (_) {
            // Never let the hint interfere with the actual response.
        }
        return res;
    };
})();
