/**
 * Camera — exposure for one camera.
 *
 * Third panel under docs/architecture/PANELS.md.
 *
 *     CameraPanel.mount('op-cam-panel-bottom', { camera: 'bottom' });
 *     CameraPanel.mount('op-cam-panel-spim',   { camera: 'spim'   });
 *
 * WHY
 *
 * Ryan asked for exposure on the 2026-08-07 walkthrough (#110): "in
 * Micro-Manager sometimes there's a default like very low exposure time, and
 * that can make it a little more challenging to find the embryos". There was no
 * exposure control anywhere in Operate, on either surface.
 *
 * Exposure briefly lived in the Light panel. It is a camera property, and the
 * bottom camera needs it without needing laser lines or beam arming, so it
 * moved here before the boundary set.
 *
 * SCOPE
 *
 * Deliberately not stream start/stop. Those buttons are wired into operate.js's
 * streaming lifecycle, they work, and moving them would be a risky refactor
 * with nothing visible to show for it. This panel adds the control that was
 * missing and reads back the one that existed.
 *
 * The two cameras reach exposure by different routes — the bottom camera has a
 * device-layer endpoint of its own, the light-sheet takes it as a live
 * streaming parameter — so the endpoint is per-camera config rather than an
 * assumption baked into the panel.
 */
const CameraPanel = (() => {
    'use strict';

    const CAMERAS = {
        bottom: {
            label: 'Bottom camera',
            read: '/api/devices/camera/exposure',
            write: '/api/devices/camera/exposure',
            body: ms => ({ exposure_ms: ms }),
        },
        spim: {
            label: 'Light sheet',
            // The light-sheet streamer takes exposure with its other live
            // params; there is no separate read, so the panel shows what was
            // last set and says so rather than implying a hardware read.
            read: null,
            write: '/api/devices/lightsheet/live_params',
            body: ms => ({ exposure: ms }),
        },
    };

    const mounts = new Map();   // hostId -> {cfg, key, exposureMs, readAt, confirmed}

    async function mount(hostId, opts) {
        const key = (opts && opts.camera) || 'bottom';
        const cfg = CAMERAS[key];
        if (!cfg || mounts.has(hostId)) return;
        // `titled: false` when mounted inside a block that already names the
        // camera — two headings for one device is the duplication this policy
        // exists to remove, just quieter.
        const titled = !(opts && opts.titled === false);
        mounts.set(hostId, { cfg, key, titled, exposureMs: null, readAt: null, confirmed: false });
        render(hostId);
        await read(hostId);
    }

    async function read(hostId) {
        const m = mounts.get(hostId);
        if (!m || !m.cfg.read) { render(hostId); return; }
        try {
            const r = await fetch(m.cfg.read);
            if (r.ok) {
                const d = await r.json();
                // The device layer answers `{success: false, error: ...}` with
                // HTTP 200 when the camera is absent, so `r.ok` is not enough.
                const ms = d && d.success !== false && d.exposure_ms != null
                    ? Number(d.exposure_ms) : null;
                m.exposureMs = Number.isFinite(ms) ? ms : null;
                m.confirmed = m.exposureMs != null;
                m.readAt = Date.now();
            } else {
                m.exposureMs = null;
                m.confirmed = false;
            }
        } catch (_) {
            m.exposureMs = null;
            m.confirmed = false;
        }
        render(hostId);
    }

    async function write(hostId, ms) {
        const m = mounts.get(hostId);
        if (!m) return;
        try {
            const r = await fetch(m.cfg.write, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(m.cfg.body(ms)),
            });
            const text = await r.text().catch(() => '');
            let d = {}; try { d = text ? JSON.parse(text) : {}; } catch (_) { /* not JSON */ }
            if (!r.ok) {
                const msg = d.detail || d.error || `${r.status}`;
                if (typeof showGentlyToast === 'function') {
                    showGentlyToast(String(msg), null, null, 10000, 'error');
                }
            } else if (!m.cfg.read) {
                // No read-back route on this camera. Record what was sent, but
                // mark it unconfirmed — a sent value is not a read one, and the
                // panel must not present it as though it were (PANELS.md 3).
                m.exposureMs = ms;
                m.confirmed = false;
                m.readAt = Date.now();
            }
        } catch (_) { /* toasted above where possible */ }
        await read(hostId);
    }

    function render(hostId) {
        const m = mounts.get(hostId);
        const el = document.getElementById(hostId);
        if (!m || !el) return;
        const val = m.exposureMs == null ? '' : m.exposureMs;
        const note = m.exposureMs == null ? '—'
            : m.confirmed ? 'ms'
                : 'ms · sent';       // not read back on this camera
        el.innerHTML = `
          <div class="lp">
            ${m.titled ? `<div class="lp-head"><span class="lp-title">${m.cfg.label}</span></div>` : ''}
            <div class="lp-row">
              <span class="lp-label">Exposure</span>
              <input class="lp-num" type="number" data-exposure min="1" max="10000" step="1"
                     value="${val}" placeholder="—"
                     aria-label="${m.cfg.label} exposure in milliseconds">
              <span class="lp-val">${note}</span>
            </div>
          </div>`;
        const input = el.querySelector('[data-exposure]');
        if (input) input.onchange = () => {
            const ms = Number(input.value);
            if (!(ms > 0 && ms <= 10000)) { render(hostId); return; }
            write(hostId, ms);
        };
    }

    /**
     * Re-read every mounted camera.
     *
     * The panel used to read once, at mount. The device supervisor spawns the
     * device layer ~30s after the web server, so a panel mounted before that
     * showed an em dash for the rest of the session with nothing to prompt
     * another attempt. Reported from the rig.
     */
    async function refreshAll() {
        await Promise.all([...mounts.keys()].map(read));
    }

    // The device layer becoming available is exactly when a first read becomes
    // possible, so take that moment.
    if (typeof ClientEventBus !== 'undefined') {
        ClientEventBus.on('DEVICE_LAYER_AVAILABILITY', () => refreshAll());
    }

    function unmount(hostId) { mounts.delete(hostId); }

    return { mount, unmount, refresh: read, refreshAll, CAMERAS };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = CameraPanel;
