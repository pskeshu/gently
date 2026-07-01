/**
 * Dashboard Settings Page
 * Reads/writes gently-dashboard-config in localStorage
 */

const SettingsManager = {
    STORAGE_KEY: 'gently-dashboard-config',

    defaults: {
        defaultView: 'default',
        board: {
            columns: ['stage', 'confidence', 'rate', 'eta', 'sparkline', 'alert'],
            sparklineLength: 20,
            warnOvertimeRatio: 1.5,
            criticalOvertimeRatio: 2.5
        },
        filmstrip: {
            thumbnailSize: 56,
            showStageLabels: true,
            skipInterval: 1,
            borderEncoding: 'stage'
        },
        vitals: {
            temperatureModel: '20C',
            showExpectedLine: true,
            timeAxis: 'elapsed'
        },
        detail: {
            imageSplitRatio: 40,
            autoAdvance: false,
            showContrastive: true
        },
        ambient: {
            enabled: true,
            sensitivity: 'normal',
            audioTick: false
        }
    },

    config: null,

    init() {
        this.config = this.load();
        this.populateForm();
        this.setupNavigation();
        this.setupListeners();
    },

    load() {
        try {
            const stored = localStorage.getItem(this.STORAGE_KEY);
            if (stored) {
                return this.deepMerge(this.defaults, JSON.parse(stored));
            }
        } catch (e) {
            console.warn('Failed to load settings:', e);
        }
        return { ...this.defaults };
    },

    save() {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.config));
            this.showSaveStatus('Settings saved');
        } catch (e) {
            this.showSaveStatus('Failed to save');
        }
    },

    deepMerge(target, source) {
        const result = { ...target };
        for (const key of Object.keys(source)) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this.deepMerge(target[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        return result;
    },

    showSaveStatus(msg) {
        const el = document.getElementById('settings-save-status');
        if (el) {
            el.textContent = msg;
            el.classList.add('visible');
            setTimeout(() => el.classList.remove('visible'), 2000);
        }
    },

    // Populate form from config
    populateForm() {
        const c = this.config;

        // Default view
        this.setRadio('defaultView', c.defaultView);

        // Alerts
        this.setInput('cfg-warnOvertimeRatio', c.board.warnOvertimeRatio);
        this.setInput('cfg-criticalOvertimeRatio', c.board.criticalOvertimeRatio);

        // Ambient
        this.setCheckbox('cfg-ambientEnabled', c.ambient.enabled);
        this.setCheckbox('cfg-audioTick', c.ambient.audioTick);
        this.setRadio('ambientSensitivity', c.ambient.sensitivity);

        // Board
        const colCheckboxes = document.querySelectorAll('#cfg-boardColumns input[type="checkbox"]');
        colCheckboxes.forEach(cb => {
            cb.checked = c.board.columns.includes(cb.value);
        });
        this.setInput('cfg-sparklineLength', c.board.sparklineLength);

        // Filmstrip
        this.setRadio('thumbnailSize', String(c.filmstrip.thumbnailSize));
        this.setCheckbox('cfg-showStageLabels', c.filmstrip.showStageLabels);
        this.setInput('cfg-skipInterval', c.filmstrip.skipInterval);
        this.setRadio('borderEncoding', c.filmstrip.borderEncoding);

        // Vitals
        this.setRadio('temperatureModel', c.vitals.temperatureModel);
        this.setCheckbox('cfg-showExpectedLine', c.vitals.showExpectedLine);
        this.setRadio('timeAxis', c.vitals.timeAxis);

        // Default view settings
        const splitRange = document.getElementById('cfg-imageSplitRatio');
        if (splitRange) {
            splitRange.value = c.detail.imageSplitRatio;
            this.updateSplitDisplay(c.detail.imageSplitRatio);
        }
        this.setCheckbox('cfg-autoAdvance', c.detail.autoAdvance);
        this.setCheckbox('cfg-showContrastive', c.detail.showContrastive);
    },

    setRadio(name, value) {
        const radio = document.querySelector(`input[name="${name}"][value="${value}"]`);
        if (radio) radio.checked = true;
    },

    setCheckbox(id, value) {
        const el = document.getElementById(id);
        if (el) el.checked = value;
    },

    setInput(id, value) {
        const el = document.getElementById(id);
        if (el) el.value = value;
    },

    updateSplitDisplay(value) {
        const display = document.getElementById('cfg-imageSplitRatio-display');
        if (display) display.textContent = `${value} / ${100 - value}`;
    },

    // Navigation
    setupNavigation() {
        const nav = document.getElementById('settings-nav');
        if (!nav) return;
        nav.addEventListener('click', (e) => {
            const item = e.target.closest('.settings-nav-item');
            if (!item) return;
            e.preventDefault();
            const section = item.dataset.section;
            // Update nav active state
            nav.querySelectorAll('.settings-nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            // Scroll to section
            const sectionEl = document.getElementById(`section-${section}`);
            if (sectionEl) {
                sectionEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    },

    // Auto-save on any change
    setupListeners() {
        const content = document.getElementById('settings-content');
        if (!content) return;

        // Ignore the server-backed hardware sections — they own their own
        // handlers (ThermalizerSettings) and must not trigger the localStorage
        // save or its "Settings saved" toast.
        const isHardware = (t) => t && t.closest && t.closest('#section-thermalizer, #section-effective');
        content.addEventListener('change', (e) => {
            if (isHardware(e.target)) return;
            this.readFormAndSave();
        });
        content.addEventListener('input', (e) => {
            // Live update for range sliders
            if (e.target.id === 'cfg-imageSplitRatio') {
                this.updateSplitDisplay(e.target.value);
            }
        });
    },

    readFormAndSave() {
        const c = this.config;

        // Default view
        c.defaultView = this.getRadio('defaultView') || 'default';

        // Alerts
        c.board.warnOvertimeRatio = parseFloat(document.getElementById('cfg-warnOvertimeRatio')?.value) || 1.5;
        c.board.criticalOvertimeRatio = parseFloat(document.getElementById('cfg-criticalOvertimeRatio')?.value) || 2.5;

        // Ambient
        c.ambient.enabled = document.getElementById('cfg-ambientEnabled')?.checked ?? true;
        c.ambient.audioTick = document.getElementById('cfg-audioTick')?.checked ?? false;
        c.ambient.sensitivity = this.getRadio('ambientSensitivity') || 'normal';

        // Board columns
        const colCheckboxes = document.querySelectorAll('#cfg-boardColumns input[type="checkbox"]');
        c.board.columns = Array.from(colCheckboxes).filter(cb => cb.checked).map(cb => cb.value);
        c.board.sparklineLength = parseInt(document.getElementById('cfg-sparklineLength')?.value) || 20;

        // Filmstrip
        c.filmstrip.thumbnailSize = parseInt(this.getRadio('thumbnailSize')) || 56;
        c.filmstrip.showStageLabels = document.getElementById('cfg-showStageLabels')?.checked ?? true;
        c.filmstrip.skipInterval = parseInt(document.getElementById('cfg-skipInterval')?.value) || 1;
        c.filmstrip.borderEncoding = this.getRadio('borderEncoding') || 'stage';

        // Vitals
        c.vitals.temperatureModel = this.getRadio('temperatureModel') || '20C';
        c.vitals.showExpectedLine = document.getElementById('cfg-showExpectedLine')?.checked ?? true;
        c.vitals.timeAxis = this.getRadio('timeAxis') || 'elapsed';

        // Default view
        c.detail.imageSplitRatio = parseInt(document.getElementById('cfg-imageSplitRatio')?.value) || 40;
        c.detail.autoAdvance = document.getElementById('cfg-autoAdvance')?.checked ?? false;
        c.detail.showContrastive = document.getElementById('cfg-showContrastive')?.checked ?? true;

        this.save();
    },

    getRadio(name) {
        const checked = document.querySelector(`input[name="${name}"]:checked`);
        return checked ? checked.value : null;
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => SettingsManager.init());


/**
 * ThermalizerSettings — server-backed hardware config, isolated from the
 * localStorage SettingsManager above. Reads/writes the ACUITYnano connection
 * via /api/devices/temperature/config{,/test}. Mock backend is dev-only
 * (revealed via ?dev=1 or localStorage 'gently-dev').
 */
const ThermalizerSettings = {
    el(id) { return document.getElementById(id); },
    devMode() {
        try {
            return new URLSearchParams(location.search).has('dev')
                || localStorage.getItem('gently-dev') === '1';
        } catch (_) { return false; }
    },

    async init() {
        if (!this.el('section-thermalizer')) return;
        if (this.devMode()) {
            const m = document.querySelector('.th-mock-opt');
            if (m) m.style.display = '';
        }
        // backend radio toggles the field groups
        document.querySelectorAll('input[name="th-backend"]').forEach(r =>
            r.addEventListener('change', () => this.applyBackendVisibility(r.value)));
        const test = this.el('th-test'), apply = this.el('th-apply');
        if (test) test.addEventListener('click', () => this.test());
        if (apply) apply.addEventListener('click', () => this.apply());
        await this.load();
        await this.loadEffective();
    },

    applyBackendVisibility(backend) {
        const s = this.el('th-serial'), m = this.el('th-mqtt');
        if (s) s.style.display = backend === 'serial' ? '' : 'none';
        if (m) m.style.display = backend === 'mqtt' ? '' : 'none';
    },

    setForm(cfg) {
        cfg = cfg || {};
        const backend = cfg.backend || 'serial';
        const radio = document.querySelector(`input[name="th-backend"][value="${backend}"]`);
        if (radio) radio.checked = true;
        this.applyBackendVisibility(backend);
        const set = (id, v) => { const e = this.el(id); if (e && v != null) e.value = v; };
        set('th-com', cfg.com_port); set('th-baud', cfg.baud_rate);
        set('th-broker', cfg.broker); set('th-port', cfg.port); set('th-user', cfg.user);
        set('th-stabilize', cfg.stabilize_timeout);
        const pel = this.el('th-peltier'); if (pel) pel.checked = !!cfg.feedback_peltier;
        // password intentionally left blank (write-only)
    },

    readForm() {
        const backend = (document.querySelector('input[name="th-backend"]:checked') || {}).value || 'serial';
        const cfg = { backend };
        const num = id => { const v = this.el(id) && this.el(id).value; return v === '' || v == null ? null : Number(v); };
        const str = id => { const v = this.el(id) && this.el(id).value; return v == null ? '' : v.trim(); };
        if (backend === 'serial') {
            cfg.com_port = str('th-com');
            if (num('th-baud') != null) cfg.baud_rate = num('th-baud');
        } else if (backend === 'mqtt') {
            if (str('th-broker')) cfg.broker = str('th-broker');
            if (num('th-port') != null) cfg.port = num('th-port');
            if (str('th-user')) cfg.user = str('th-user');
            const pw = this.el('th-pass') && this.el('th-pass').value;
            if (pw) cfg.password = pw;  // blank = keep stored (server preserves)
        }
        if (num('th-stabilize') != null) cfg.stabilize_timeout = num('th-stabilize');
        cfg.feedback_peltier = !!(this.el('th-peltier') && this.el('th-peltier').checked);
        return cfg;
    },

    renderStatus(d) {
        const el = this.el('th-status'); if (!el) return;
        if (!d || d.available === false) { el.textContent = 'Controller not available (device layer offline or no thermalizer configured).'; return; }
        const st = d.state || {};
        const parts = [];
        if (d.live_backend) parts.push(`backend: ${d.live_backend}`);
        if (st.temperature_c != null) parts.push(`water: ${st.temperature_c} °C`);
        if (st.setpoint_c != null) parts.push(`setpoint: ${st.setpoint_c} °C`);
        if (st.state) parts.push(st.state);
        el.textContent = parts.length ? parts.join(' · ') : 'active';
    },

    async load() {
        try {
            const res = await fetch('/api/devices/temperature/config');
            const d = await res.json();
            this.renderStatus(d);
            if (d && d.config) this.setForm(d.config);
        } catch (e) { this.renderStatus(null); }
    },

    result(msg, ok) {
        const el = this.el('th-result'); if (!el) return;
        el.textContent = msg;
        el.className = 'settings-result ' + (ok ? 'is-ok' : 'is-err');
    },

    async test() {
        const btn = this.el('th-test'); btn.disabled = true; const old = btn.textContent; btn.textContent = 'Testing…';
        try {
            const res = await fetch('/api/devices/temperature/config/test', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.readForm()),
            });
            if (res.status === 401 || res.status === 403) { this.result('Need control to test.', false); return; }
            const d = await res.json();
            if (d.success) {
                const r = d.result || {};
                this.result(`OK — ${r.backend || 'connected'}${r.state ? ' · ' + r.state : ''}${r.temperature_c != null ? ' · ' + r.temperature_c + ' °C' : ''}`, true);
            } else { this.result(`Failed: ${d.error || res.status}`, false); }
        } catch (e) { this.result(`Error: ${e.message}`, false); }
        finally { btn.disabled = false; btn.textContent = old; }
    },

    async apply() {
        if (!window.confirm('Apply this thermalizer config to the rig?')) return;
        const btn = this.el('th-apply'); btn.disabled = true; const old = btn.textContent; btn.textContent = 'Applying…';
        try {
            const res = await fetch('/api/devices/temperature/config', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.readForm()),
            });
            if (res.status === 401 || res.status === 403) { this.result('Need control to apply.', false); return; }
            const d = await res.json();
            // The device-layer 409 (run/ramp active) is flattened to 200 by the
            // proxy, so detect it via the body flag, not the HTTP status.
            if (d.blocked) { this.result(`Blocked: ${d.error || 'a run/ramp is active'}`, false); return; }
            if (d.success && d.applied) { this.result('Applied live.', true); await this.load(); }
            else if (d.restart_required) { this.result(`Saved — restart the device layer to apply. (${d.error || ''})`, false); }
            else { this.result(`Failed: ${d.error || (d.detail || res.status)}`, false); }
        } catch (e) { this.result(`Error: ${e.message}`, false); }
        finally { btn.disabled = false; btn.textContent = old; }
    },

    async loadEffective() {
        const el = this.el('effective-config'); if (!el) return;
        try {
            const res = await fetch('/api/config/effective');
            if (!res.ok) { el.textContent = 'Unavailable.'; return; }
            const d = await res.json();
            el.textContent = JSON.stringify(d, null, 2);
        } catch (e) { el.textContent = 'Unavailable.'; }
    },
};

document.addEventListener('DOMContentLoaded', () => ThermalizerSettings.init());
