/**
 * Boot banner — a small, always-visible, non-modal summary of the device-layer
 * boot, so the operator can follow MMCore startup from anywhere on the dashboard
 * (the Devices panel is the on-demand console; this is its compact form).
 *
 * Polls /api/device-layer/status globally — fast (1s) while booting, calm (6s)
 * otherwise so it also catches a device layer started later from the Devices
 * panel. Also publishes a readiness signal (window.gentlyDeviceReady +
 * ClientEventBus 'DEVICE_LAYER_STATE') that hardware-only controls can gate on.
 */
const BootBanner = (function () {
    const STAGE_TOTAL = 5;

    let _el, _text, _details, _retry, _close;
    let _timer = null;
    let _pollMs = 0;
    let _lastState = null;
    let _dom = false;

    function cacheDom() {
        if (_dom) return;
        _el = document.getElementById('gently-boot-banner');
        if (!_el) return;
        _text = document.getElementById('boot-banner-text');
        _details = document.getElementById('boot-banner-details');
        _retry = document.getElementById('boot-banner-retry');
        _close = document.getElementById('boot-banner-close');
        _dom = true;
    }

    function init() {
        cacheDom();
        if (!_el) return;
        _details.addEventListener('click', () => {
            // The v2 landing overlay covers the workspace — dismiss it first,
            // or the tab switch below happens invisibly behind it and the
            // click feels dead (found via session replay: the operator
            // clicked Details and saw nothing for over two minutes).
            const landing = document.getElementById('v2-landing');
            if (landing && !landing.classList.contains('dismissed')) {
                const skip = document.getElementById('v2-landing-skip');
                if (skip) skip.click(); else landing.classList.add('dismissed');
            }
            if (typeof switchTab === 'function' && typeof TABS !== 'undefined') switchTab(TABS.DEVICES);
        });
        _retry.addEventListener('click', onRetry);
        _close.addEventListener('click', hide);
        setPoll(1500);
    }

    function setPoll(ms) {
        if (ms === _pollMs && _timer) return;
        _pollMs = ms;
        if (_timer) clearInterval(_timer);
        _timer = setInterval(poll, _pollMs);
    }

    async function poll() {
        try {
            const r = await fetch('/api/device-layer/status');
            if (!r.ok) return;
            render(await r.json());
        } catch (e) {
            /* keep last-known UI */
        }
    }

    function render(d) {
        const state = d.state;

        // Publish a readiness signal for hardware-only controls to gate on.
        const ready = state === 'ready' || state === 'external';
        window.gentlyDeviceReady = ready;
        if (state !== _lastState && typeof ClientEventBus !== 'undefined') {
            ClientEventBus.emit('DEVICE_LAYER_STATE', { state, ready });
        }

        if (state === 'starting' || state === 'initializing') {
            show('booting');
            const p = d.progress || {};
            const step = p.i ? `step ${p.i}/${p.n || STAGE_TOTAL} · ` : '';
            _text.textContent = `Microscope warming up — ${step}${p.label || 'starting…'}`;
            btns({ details: true, retry: false, close: false });
            setPoll(1000);
        } else if (state === 'ready') {
            setPoll(6000);
            const justFinished = _lastState === 'starting' || _lastState === 'initializing';
            if (justFinished) {
                // Flash "ready" briefly, then auto-dismiss.
                show('ready');
                _text.textContent = 'Microscope ready';
                btns({ details: false, retry: false, close: false });
                setTimeout(() => {
                    if (_el.classList.contains('ready')) hide();
                }, 3500);
            } else if (!(_el.classList.contains('ready') && !_el.hidden)) {
                hide(); // already ready on load, or the ready-flash was dismissed
            }
        } else if (state === 'failed' || state === 'crashed') {
            show('failed');
            _text.textContent =
                (d.failure && d.failure.summary) ||
                (state === 'crashed'
                    ? 'The device layer stopped unexpectedly.'
                    : "The microscope didn't start.");
            btns({ details: true, retry: true, close: true });
            setPoll(6000);
        } else {
            // stopped (software-only session) or external with nothing to add.
            hide();
            setPoll(6000);
        }
        _lastState = state;
    }

    function show(kind) {
        _el.hidden = false;
        _el.className = 'boot-banner ' + kind;
    }
    function hide() {
        _el.hidden = true;
    }
    function btns({ details, retry, close }) {
        _details.hidden = !details;
        _retry.hidden = !retry;
        _close.hidden = !close;
    }

    async function onRetry() {
        _retry.disabled = true;
        try {
            await fetch('/api/device-layer/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{}',
            });
        } catch (e) {
            /* the status poll will reflect the outcome */
        } finally {
            _retry.disabled = false;
            setTimeout(poll, 400);
            poll();
        }
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', () => BootBanner.init());
