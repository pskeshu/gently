/**
 * Device-layer supervision card (Devices tab).
 *
 * The runtime mirror of the launch gate's hardware block: shows whether the
 * device layer is running / stopped / external / crashed, lets an operator
 * Start or Stop it from the UI, and tails its console. Consumes the routes
 * added in gently/ui/web/routes/device_layer.py:
 *
 *   GET  /api/device-layer/status  -> {state, managed, pid, port, port_open,
 *                                       sam_device, uptime_seconds, log_tail[]}
 *   GET  /api/device-layer/log?limit=N -> {lines: [...]}
 *   POST /api/device-layer/start   (control-gated)  body {sam_device?, config_path?}
 *   POST /api/device-layer/stop    (control-gated)  body {confirm?, force?}
 *                                   -> 409 {blocked:true,...} mid-acquisition
 *
 * Self-contained IIFE (no devices.js internals). Polls only while the Devices
 * tab is visible (ClientEventBus 'TAB_CHANGED'). Control auth is transparent:
 * a same-origin fetch carries the gently_session cookie, and control-auth.js
 * already toasts on a bare 403, so Start/Stop need no credential code.
 */
const DeviceLayerCard = (function () {
    let _pollMs = 5000;   // adaptive: 1s while booting, 5s otherwise

    // state -> [pill modifier class, label].
    const PILL = {
        ready:        ['live',   'ready'],
        running:      ['live',   'running'],   // legacy alias
        starting:     ['paused', 'starting'],
        initializing: ['paused', 'starting'],
        external:     ['paused', 'external'],
        stopped:      ['',       'stopped'],
        crashed:      ['error',  'crashed'],
        failed:       ['error',  'failed'],
    };

    let _card, _pill, _meta, _start, _stop, _logToggle, _log, _hint;
    let _timer = null;
    let _busy = false;   // a start/stop is in flight
    let _dom = false;
    let _autoLog = false;         // log pane was auto-opened for the boot phase
    let _logUserOverride = false; // user toggled the log → stop auto-managing it this cycle
    let _uptimeTimer = null;      // 1s local ticker so uptime reads live between 5s polls
    let _uptimeBase = 0;          // uptime_seconds from the most recent poll
    let _uptimeAnchor = 0;        // performance.now() when that poll landed
    let _samLabel = '';           // cached accelerator label for the local re-render

    function cacheDom() {
        if (_dom) return;
        _card = document.getElementById('devices-layer-card');
        if (!_card) return;              // markup absent -> stay inert
        _pill      = document.getElementById('devices-layer-pill');
        _meta      = document.getElementById('devices-layer-meta');
        _start     = document.getElementById('devices-layer-start');
        _stop      = document.getElementById('devices-layer-stop');
        _logToggle = document.getElementById('devices-layer-log-toggle');
        _log       = document.getElementById('devices-layer-log');
        _hint      = document.getElementById('devices-layer-hint');
        _dom = true;
    }

    function init() {
        cacheDom();
        if (!_card) return;
        _start.addEventListener('click', onStart);
        _stop.addEventListener('click', onStop);
        _logToggle.addEventListener('click', toggleLog);

        // Poll only while the Devices tab is showing. switchTab() emits
        // TAB_CHANGED with the *new* tab id (no leave signal), so compare
        // against TABS.DEVICES to derive both enter and exit.
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('TAB_CHANGED', (tab) => {
                if (tab === TABS.DEVICES) startPoll();
                else stopPoll();
            });
        }
        // Seed if Devices is already the active tab (reload / deep link) —
        // TAB_CHANGED does not fire for the landing tab.
        const content = document.getElementById('devices-content');
        if (content && content.classList.contains('active')) startPoll();
    }

    // ── polling ──────────────────────────────────────────────────────────

    function startPoll() {
        loadStatus();                       // immediate
        if (_timer) clearInterval(_timer);  // clear-before-set
        _timer = setInterval(loadStatus, _pollMs);
    }

    function stopPoll() {
        if (_timer) { clearInterval(_timer); _timer = null; }
        stopUptimeTicker();   // no local ticking while the tab is hidden
    }

    async function loadStatus() {
        if (!_card) return;
        try {
            const res = await fetch('/api/device-layer/status');
            if (!res.ok) return;            // keep last-known UI
            apply(await res.json());
        } catch (e) {
            console.debug('device-layer status poll failed', e);
        }
    }

    // ── render ───────────────────────────────────────────────────────────

    function apply(d) {
        _card.hidden = false;
        const [mod, label] = PILL[d.state] || ['stale', d.state || 'unknown'];
        _pill.className = 'devices-status-pill' + (mod ? ' ' + mod : '');
        _pill.textContent = label;

        const booting = d.state === 'starting' || d.state === 'initializing';

        // Meta line: during boot show the current stage; otherwise uptime + SAM.
        if (booting) {
            stopUptimeTicker();
            _meta.textContent = (d.progress && d.progress.i && d.progress.n)
                ? 'step ' + d.progress.i + '/' + d.progress.n + ' · ' + (d.progress.label || 'starting…')
                : 'starting…';
        } else {
            // Operator-friendly status line: uptime + the SAM accelerator in
            // plain terms. pid, raw port, and "closed" are debug details (in the
            // Log / status payload), kept off the glanceable line.
            _samLabel = !d.sam_device ? ''
                : d.sam_device === 'cuda' ? 'GPU'
                : d.sam_device === 'cpu' ? 'CPU' : d.sam_device;
            if (typeof d.uptime_seconds === 'number' && d.uptime_seconds > 0) {
                // Anchor to the server's value, then tick locally every second so
                // uptime counts smoothly instead of stepping by the 5s poll gap.
                _uptimeBase = d.uptime_seconds;
                _uptimeAnchor = (typeof performance !== 'undefined' ? performance.now() : 0);
                renderSteadyMeta();
                startUptimeTicker();
            } else {
                _uptimeBase = 0;
                stopUptimeTicker();
                renderSteadyMeta();
            }
        }

        // Hint line: failure reason, external note, or a reassurance during the
        // slow MMCore step (step 2) so a long wait doesn't read as frozen.
        if (d.state === 'failed' && d.failure) {
            _hint.hidden = false;
            const h = (d.failure.hints && d.failure.hints[0]) ? ' — ' + d.failure.hints[0] : '';
            _hint.textContent = (d.failure.summary || 'Startup failed') + h;
        } else if (d.state === 'external') {
            _hint.hidden = false;
            _hint.textContent = 'running externally — not managed by gently';
        } else if (booting && d.progress && d.progress.i === 2) {
            _hint.hidden = false;
            _hint.textContent = 'Initializing Micro-Manager — this can take a minute.';
        } else {
            _hint.hidden = true;
            _hint.textContent = '';
        }

        // Enable/disable (a hint only; the server 403/409 is the real gate).
        const canStart = d.state === 'stopped' || d.state === 'crashed' || d.state === 'failed';
        const canStop =
            d.state === 'ready' || d.state === 'running' || booting;
        _start.disabled = _busy || !canStart;
        _stop.disabled = _busy || !canStop;

        // Adaptive cadence: poll fast (1s) while booting, calm (5s) otherwise.
        const want = booting ? 1000 : 5000;
        if (want !== _pollMs && _timer) {
            _pollMs = want;
            clearInterval(_timer);
            _timer = setInterval(loadStatus, _pollMs);
        }

        // During boot, auto-surface the trailing init log so the long MMCore
        // step visibly progresses ("something is happening") instead of sitting
        // on a static hint. Reuses the Log pane; auto-collapses once boot ends —
        // unless the user has taken the log's open/closed state into their hands.
        if (booting && _log.hidden && !_logUserOverride) {
            _log.hidden = false;
            _logToggle.setAttribute('aria-expanded', 'true');
            _logToggle.classList.add('active');
            _autoLog = true;
        } else if (!booting && _autoLog && !_logUserOverride && !_log.hidden) {
            _log.hidden = true;
            _logToggle.setAttribute('aria-expanded', 'false');
            _logToggle.classList.remove('active');
            _autoLog = false;
        }

        if (Array.isArray(d.log_tail) && d.log_tail.length && !_log.hidden) {
            renderLog(d.log_tail);
        }
    }

    function fmtUptime(s) {
        s = Math.max(0, Math.floor(s));
        if (s < 60) return s + 's';
        if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
        return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
    }

    // Re-render the steady-state meta from cached values, computing uptime live
    // off the last poll's anchor so the local 1s ticker counts smoothly.
    function renderSteadyMeta() {
        if (!_meta) return;
        const bits = [];
        if (_uptimeBase > 0) {
            const now = (typeof performance !== 'undefined' ? performance.now() : 0);
            bits.push('up ' + fmtUptime(_uptimeBase + Math.max(0, now - _uptimeAnchor) / 1000));
        }
        if (_samLabel) bits.push('SAM: ' + _samLabel);
        _meta.textContent = bits.join(' · ');
    }

    function startUptimeTicker() {
        if (_uptimeTimer) return;                        // already ticking
        _uptimeTimer = setInterval(renderSteadyMeta, 1000);
    }

    function stopUptimeTicker() {
        if (_uptimeTimer) { clearInterval(_uptimeTimer); _uptimeTimer = null; }
    }

    // ── start / stop ─────────────────────────────────────────────────────

    async function onStart() {
        if (_start.disabled) return;
        // Fresh boot cycle — let the log pane auto-manage again.
        _logUserOverride = false;
        _autoLog = false;
        _busy = true;
        _start.disabled = true;
        try {
            await postJSON('/api/device-layer/start', {});
            toast('Device layer starting…');
        } catch (e) {
            // 403 already surfaced by control-auth.js; don't double-toast.
            if (e.status !== 403) toast('Start failed: ' + e.message);
        } finally {
            _busy = false;
            // The layer takes a moment to bind its port; re-poll shortly.
            setTimeout(loadStatus, 800);
            loadStatus();
        }
    }

    async function onStop() {
        if (_stop.disabled) return;
        if (!window.confirm('Stop the device layer?')) return;
        _busy = true;
        _stop.disabled = true;
        try {
            await postJSON('/api/device-layer/stop', { confirm: true });
            toast('Device layer stopping…');
        } catch (e) {
            if (e.status === 409 && e.payload && e.payload.blocked) {
                // Mid-acquisition soft block — offer an explicit force-stop.
                if (window.confirm('A run is active. Force-stop the device layer anyway?')) {
                    try {
                        await postJSON('/api/device-layer/stop', { confirm: true, force: true });
                        toast('Force-stopping…');
                    } catch (e2) {
                        if (e2.status !== 403) toast('Force-stop failed: ' + e2.message);
                    }
                }
            } else if (e.status !== 403) {
                toast('Stop failed: ' + e.message);
            }
        } finally {
            _busy = false;
            setTimeout(loadStatus, 500);
            loadStatus();
        }
    }

    // POST JSON; on !ok throw an Error carrying .status and parsed .payload so
    // callers can branch on 409/403 (mirrors operate.js's postJSON convention).
    async function postJSON(url, body) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        if (!res.ok) {
            let payload = {};
            try { payload = await res.json(); } catch (_) { /* non-JSON */ }
            const err = new Error(payload.error || payload.reason || res.statusText);
            err.status = res.status;
            err.payload = payload;
            throw err;
        }
        return res.json();
    }

    // ── log tail ─────────────────────────────────────────────────────────

    function toggleLog() {
        _logUserOverride = true;   // user owns the pane's open/closed state now
        _autoLog = false;
        const show = _log.hidden;
        _log.hidden = !show;
        _logToggle.setAttribute('aria-expanded', String(show));
        _logToggle.classList.toggle('active', show);
        if (show) loadFullLog();
    }

    async function loadFullLog() {
        try {
            const res = await fetch('/api/device-layer/log?limit=200');
            if (!res.ok) return;
            const d = await res.json();
            if (Array.isArray(d.lines)) renderLog(d.lines);
        } catch (e) {
            console.debug('device-layer log fetch failed', e);
        }
    }

    function renderLog(lines) {
        if (!_log) return;
        _log.textContent = lines.join('\n');
        _log.scrollTop = _log.scrollHeight;   // pin to newest
    }

    function toast(msg) {
        if (typeof showGentlyToast === 'function') showGentlyToast(msg);
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', () => DeviceLayerCard.init());
