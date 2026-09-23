/**
 * Rig menu — the microscope in the header chrome.
 *
 * The rig is not a page. Whether the device layer is up, what the water is
 * doing, whether the room light is on, and the two buttons that change any of
 * it, used to live on a card inside the Devices tab: invisible from everywhere
 * else, and eating the vertical space the bottom camera needed. They live here
 * now — one chip that always shows the rig's state, one menu behind it.
 *
 * Reuses rather than duplicates:
 *   - boot-banner.js already polls /api/device-layer/status globally and
 *     publishes DEVICE_LAYER_STATE on ClientEventBus. This subscribes; it does
 *     not add a second poll.
 *   - LogConsole is the one log surface (agent + device layer tabs), so "Log"
 *     opens that drawer instead of tailing the log again here.
 *   - Water and room light poll only while the menu is open — they are cheap
 *     cached reads, but a closed menu has no reason to ask.
 *
 * Control auth is transparent: same-origin fetch carries the session cookie and
 * control-auth.js toasts on a bare 403.
 */
const RigMenu = (function () {
    const STATE_LABEL = {
        ready: 'Rig ready',
        running: 'Rig ready',
        starting: 'Starting',
        initializing: 'Starting',
        external: 'Rig ready',
        stopped: 'Rig off',
        crashed: 'Rig crashed',
        failed: 'Rig failed',
    };
    const STATE_CLASS = {
        ready: 'connected', running: 'connected', external: 'connected',
        starting: 'partial', initializing: 'partial',
        stopped: '', crashed: 'error', failed: 'error',
    };

    let _dom = false;
    let _el = {};
    let _last = null;        // last device-layer status payload
    let _busy = false;       // a start/stop is in flight
    let _pollTimer = null;   // water/light poll, only while open

    function $(id) { return document.getElementById(id); }

    function cacheDom() {
        if (_dom) return true;
        _el = {
            chip: $('status-button'), dot: $('status-dot'), text: $('status-text'),
            sep: $('rig-chip-sep'), temp: $('rig-chip-temp'),
            menu: $('status-popover'),
            dlState: $('rig-dl-state'), dlMeta: $('rig-dl-meta'),
            dlProgress: $('rig-dl-progress'), dlFill: $('rig-dl-progress-fill'),
            dlProgressText: $('rig-dl-progress-text'),
            start: $('rig-dl-start'), stop: $('rig-dl-stop'), log: $('rig-dl-log'),
            hint: $('rig-dl-hint'),
            tempSec: $('rig-temp-sec'), tempNow: $('rig-temp-now'),
            tempState: $('rig-temp-state'), tempInput: $('rig-temp-input'),
            tempApply: $('rig-temp-apply'),
            lightSec: $('rig-light-sec'), lightState: $('rig-light-state'),
            lightToggle: $('rig-light-toggle'),
        };
        _dom = !!(_el.chip && _el.menu);
        return _dom;
    }

    function isOpen() { return _el.menu && !_el.menu.classList.contains('hidden'); }

    // ── chip + menu rendering ────────────────────────────────────────────
    function renderDeviceLayer(s) {
        if (!cacheDom() || !s) return;
        _last = s;
        const state = String(s.state || 'stopped');
        const label = STATE_LABEL[state] || state;
        const booting = state === 'starting' || state === 'initializing';

        // The chip is the only always-visible rig readout, so a boot shows its
        // stage there rather than only inside the menu.
        if (booting && s.progress && s.progress.n) {
            _el.text.textContent = `Booting ${s.progress.i}/${s.progress.n}`;
        } else {
            _el.text.textContent = label;
        }
        _el.dot.classList.remove('connected', 'partial', 'error');
        const cls = STATE_CLASS[state];
        if (cls) _el.dot.classList.add(cls);

        _el.dlState.textContent = state;
        _el.dlState.className = 'rig-state rig-state-' + (cls || 'off');

        const meta = [];
        if (s.uptime_seconds != null && (state === 'ready' || state === 'running')) {
            meta.push('up ' + fmtUptime(s.uptime_seconds));
        }
        if (s.sam_device) meta.push('SAM ' + String(s.sam_device).toUpperCase());
        if (state === 'external') meta.push('started outside Gently');
        _el.dlMeta.textContent = meta.join(' · ');

        if (booting && s.progress && s.progress.n) {
            _el.dlProgress.hidden = false;
            _el.dlFill.style.width = Math.round((s.progress.i / s.progress.n) * 100) + '%';
            _el.dlProgressText.textContent =
                `${s.progress.i}/${s.progress.n} · ${s.progress.label || ''}`.trim();
        } else {
            _el.dlProgress.hidden = true;
        }

        const up = state === 'ready' || state === 'running' || state === 'external' || booting;
        _el.start.disabled = _busy || up;
        // An external device layer is not ours to stop — the supervisor refuses,
        // and a button that cannot work should say so by being disabled.
        _el.stop.disabled = _busy || !up || state === 'external';

        if (state === 'crashed' || state === 'failed') {
            _el.hint.hidden = false;
            _el.hint.textContent = (s.failure && s.failure.detail)
                ? String(s.failure.detail).slice(0, 160)
                : 'The device layer stopped unexpectedly. Log has the reason.';
        } else {
            _el.hint.hidden = true;
        }
    }

    function fmtUptime(sec) {
        const s = Math.max(0, Math.round(sec));
        if (s < 60) return s + 's';
        const m = Math.floor(s / 60);
        if (m < 60) return m + 'm';
        const h = Math.floor(m / 60);
        return h + 'h ' + (m % 60) + 'm';
    }

    // ── water + room light (polled only while the menu is open) ──────────
    async function refreshAmbient() {
        try {
            const r = await fetch('/api/devices/temperature/status');
            const d = r.ok ? await r.json() : null;
            if (d && d.available && d.temperature_c != null) {
                const t = Number(d.temperature_c).toFixed(1);
                _el.tempSec.hidden = false;
                _el.tempNow.textContent = t + '°';
                _el.tempState.textContent = shortTempState(d.state, d.temperature_c, d.setpoint_c);
                _el.sep.hidden = false;
                _el.temp.hidden = false;
                _el.temp.textContent = t + '°';
                if (document.activeElement !== _el.tempInput && d.setpoint_c != null) {
                    _el.tempInput.value = Number(d.setpoint_c).toFixed(1);
                }
            } else {
                _el.tempSec.hidden = true;
                _el.sep.hidden = true;
                _el.temp.hidden = true;
            }
        } catch (e) { /* leave the last reading on screen */ }

        try {
            const r = await fetch('/api/devices/room_light/status');
            const d = r.ok ? await r.json() : null;
            if (d && d.available) {
                const on = String(d.state).toLowerCase() === 'on';
                _el.lightSec.hidden = false;
                _el.lightState.textContent = on ? 'on' : 'off';
                _el.lightToggle.setAttribute('aria-checked', String(on));
                _el.lightToggle.classList.toggle('on', on);
            } else {
                _el.lightSec.hidden = true;
            }
        } catch (e) { /* same */ }
    }

    // The controller's state string is a debug dump — "(OFFLINE) (SIM)
    // [ SYSTEM LOCKED ]". What an operator needs is whether the water is where
    // they set it; the rest is noise in a 316px menu.
    function shortTempState(state, now, target) {
        if (now != null && target != null && Math.abs(Number(now) - Number(target)) <= 0.3) {
            return 'at setpoint';
        }
        const s = String(state || '').toLowerCase();
        if (s.includes('lock')) return 'locked';
        if (now != null && target != null) {
            return (Number(now) > Number(target) ? 'above ' : 'below ') + Number(target).toFixed(1) + '°';
        }
        return '';
    }

    function startAmbientPoll() {
        stopAmbientPoll();
        refreshAmbient();
        _pollTimer = setInterval(refreshAmbient, 10000);
    }
    function stopAmbientPoll() {
        if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    }

    // ── actions ──────────────────────────────────────────────────────────
    async function post(url, body) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        const text = await res.text().catch(() => '');
        let data = {};
        try { data = text ? JSON.parse(text) : {}; } catch (e) { /* not JSON */ }
        if (!res.ok) {
            const err = new Error(data.detail || data.error || (res.status + ''));
            err.status = res.status;
            err.data = data;
            throw err;
        }
        return data;
    }

    function say(msg, bad) {
        if (typeof showToast === 'function') showToast(msg, bad ? 'error' : 'success');
        else if (bad) console.error(msg);
    }

    async function startLayer() {
        _busy = true; renderDeviceLayer(_last);
        try {
            await post('/api/device-layer/start', {});
            say('Starting the microscope…');
        } catch (e) {
            say('Could not start the microscope (' + e.message + ')', true);
        } finally { _busy = false; renderDeviceLayer(_last); }
    }

    async function stopLayer() {
        _busy = true; renderDeviceLayer(_last);
        try {
            await post('/api/device-layer/stop', {});
            say('Microscope stopped');
        } catch (e) {
            // 409 is the supervisor refusing mid-acquisition — that is a real
            // answer, not a failure, and the operator needs the reason verbatim.
            say(e.status === 409
                ? ('Not stopped: ' + (e.data && e.data.detail ? e.data.detail : 'something is running'))
                : ('Could not stop the microscope (' + e.message + ')'), true);
        } finally { _busy = false; renderDeviceLayer(_last); }
    }

    async function applyTemp() {
        const v = parseFloat(_el.tempInput.value);
        if (!Number.isFinite(v)) { say('Enter a target temperature', true); return; }
        try {
            await post('/api/devices/temperature/set', { temperature_c: v });
            say('Water target set to ' + v.toFixed(1) + '°');
            refreshAmbient();
        } catch (e) { say('Could not set the temperature (' + e.message + ')', true); }
    }

    async function toggleLight() {
        const on = _el.lightToggle.getAttribute('aria-checked') === 'true';
        try {
            await post('/api/devices/room_light/set', { state: on ? 'off' : 'on' });
            refreshAmbient();
        } catch (e) { say('Could not switch the room light (' + e.message + ')', true); }
    }

    function init() {
        if (!cacheDom()) return;

        _el.start.addEventListener('click', startLayer);
        _el.stop.addEventListener('click', stopLayer);
        _el.log.addEventListener('click', () => {
            if (typeof LogConsole === 'undefined' || !LogConsole.open) return;
            LogConsole.open();
            if (LogConsole.selectSource) LogConsole.selectSource('device');
        });
        _el.tempApply.addEventListener('click', applyTemp);
        _el.tempInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); applyTemp(); }
        });
        _el.lightToggle.addEventListener('click', toggleLight);

        // XY limits, from the same store the Devices map renders, so the two
        // cannot disagree about whether this controller is fencing anyone.
        const limSec = document.getElementById('rig-limits-sec');
        const limState = document.getElementById('rig-limits-state');
        const limToggle = document.getElementById('rig-limits-toggle');
        if (limSec && limState && limToggle && typeof XYLimitsState !== 'undefined') {
            XYLimitsState.subscribe(s => {
                // Rule 6: nothing to say about a controller nobody can reach.
                limSec.hidden = s.enforced === null;
                limState.textContent = s.busy ? 'writing…' : (s.enforced ? 'enforced' : 'OFF');
                limToggle.setAttribute('aria-checked', s.enforced ? 'true' : 'false');
                limToggle.disabled = !!s.busy;
            });
            limToggle.addEventListener('click', () => {
                XYLimitsState.write(limToggle.getAttribute('aria-checked') !== 'true');
            });
        }

        // boot-banner.js owns the global device-layer poll; ride its signal.
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('DEVICE_LAYER_STATE', (s) => renderDeviceLayer(s));
        }
        fetch('/api/device-layer/status')
            .then(r => r.ok ? r.json() : null)
            .then(s => s && renderDeviceLayer(s))
            .catch(() => {});

        // Ambient readings follow the menu: a closed menu polls nothing.
        const sync = () => { isOpen() ? startAmbientPoll() : stopAmbientPoll(); };
        _el.chip.addEventListener('click', () => setTimeout(sync, 0));
        document.addEventListener('click', () => setTimeout(sync, 0));
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') stopAmbientPoll(); });
    }

    document.addEventListener('DOMContentLoaded', init);
    return { refreshAmbient };
})();
