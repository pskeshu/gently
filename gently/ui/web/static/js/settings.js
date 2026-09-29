/**
 * Settings — a tab of the app, drawn from the settings registry.
 *
 * GET /api/settings/schema says what every setting is about, how far it
 * reaches, when a change takes effect and where it is kept. This draws them.
 * A new setting is an entry in gently/ui/web/settings_registry.py, not a
 * control built here.
 *
 * Three hardware blocks are not single values and keep their own markup and
 * code: the thermalizer's connection, the device layer's port and SAM device,
 * and the joystick lock. The registry names them as `custom` and this places
 * them in their category.
 */
const SettingsTab = (function () {
    'use strict';

    const $ = id => document.getElementById(id);
    const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
        c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    let _schema = null;
    let _category = 'views';
    let _inited = false;

    // ── what a setting reaches, in words ────────────────────────────────────
    function badge(s) {
        const reach = s.reach === 'rig' ? 'this rig' : 'this browser';
        const when = { load: 'next page load', launch: 'next launch', restart: 'needs restart' }[s.applies];
        return when ? `${reach} · ${when}` : reach;
    }

    // ── reading and writing, by where a setting is kept ─────────────────────
    function readValue(s) {
        const store = s.store || '';
        if (store === 'theme') {
            try { return localStorage.getItem('gently-theme') || s.default; } catch (_) { return s.default; }
        }
        if (store.startsWith('prefs:')) return SettingsStore.get(store.slice(6), s.default);
        return s.value !== undefined ? s.value : s.default;
    }

    async function postJSON(url, method, body) {
        const res = await fetch(url, {
            method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            const err = new Error(res.status === 401 || res.status === 403
                ? 'Sign in with control to change this'
                : (data.detail || `Failed (${res.status})`));
            err.status = res.status;
            throw err;
        }
        return data;
    }

    /**
     * A browser's preference lives in that browser, so the history on the rig
     * learns of a change only by being told. Told and forgotten: a history
     * that could not be written never stops the change it records.
     */
    function report(key, old, value) {
        let clientId = null;
        try { clientId = localStorage.getItem('gently-client-id'); } catch (_) { /* private mode */ }
        fetch('/api/settings/history', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, old, new: value, client_id: clientId }),
        }).catch(() => {});
    }

    async function writeValue(s, value) {
        const store = s.store || '';
        if (store === 'theme' || store.startsWith('prefs:')) report(s.key, readValue(s), value);
        if (store === 'theme') {
            if (typeof ThemeManager !== 'undefined' && ThemeManager.setTheme) ThemeManager.setTheme(value);
            else {
                document.documentElement.setAttribute('data-theme', value);
                document.body.setAttribute('data-theme', value);
                try { localStorage.setItem('gently-theme', value); } catch (_) { /* private mode */ }
            }
            return 'Saved';
        }
        if (store.startsWith('prefs:')) {
            SettingsStore.set(store.slice(6), value);
            return s.applies === 'load' ? 'Saved. Applies the next time the page loads.' : 'Saved';
        }
        if (store.startsWith('env:')) {
            await postJSON('/api/config/settings-overrides', 'PUT', { [store.slice(4)]: value });
            s.value = value; s.overridden = true;
            return 'Saved. Applies after Gently restarts.';
        }
        if (store.startsWith('launch:')) {
            await postJSON('/api/launch/prefs', 'POST', { [store.slice(7)]: value });
            s.value = value;
            return 'Saved. Applies the next time Gently is launched.';
        }
        throw new Error('This setting has nowhere to be kept');
    }

    // ── controls ────────────────────────────────────────────────────────────
    function control(s) {
        const v = readValue(s);
        const id = `set-${s.key.replace(/[^a-z0-9]+/gi, '-')}`;
        const unit = s.unit ? `<span class="settings-unit">${esc(s.unit)}</span>` : '';
        if (s.type === 'bool') {
            return `<label class="settings-checkbox"><input type="checkbox" id="${id}" data-set="${esc(s.key)}"${v ? ' checked' : ''}> ${esc(s.label)}</label>`;
        }
        if (s.type === 'choice') {
            const opts = (s.choices || []).map(c =>
                `<label class="settings-radio"><input type="radio" name="${id}" data-set="${esc(s.key)}" value="${esc(c.value)}"${String(c.value) === String(v) ? ' checked' : ''}> ${esc(c.label)}</label>`);
            return `<div class="settings-radio-group">${opts.join('')}</div>`;
        }
        if (s.type === 'multichoice') {
            const on = Array.isArray(v) ? v.map(String) : [];
            const opts = (s.choices || []).map(c =>
                `<label class="settings-checkbox"><input type="checkbox" data-set="${esc(s.key)}" data-multi="1" value="${esc(c.value)}"${on.includes(String(c.value)) ? ' checked' : ''}> ${esc(c.label)}</label>`);
            return `<div class="settings-checkbox-group">${opts.join('')}</div>`;
        }
        if (s.type === 'int' || s.type === 'float') {
            const attrs = ['min', 'max', 'step'].filter(a => s[a] != null).map(a => `${a}="${s[a]}"`).join(' ');
            const step = s.step == null && s.type === 'float' ? ' step="any"' : '';
            return `<div class="settings-input-row"><input type="number" class="settings-input" id="${id}" data-set="${esc(s.key)}" ${attrs}${step} value="${esc(v)}">${unit}</div>`;
        }
        if (s.type === 'text') {
            return `<div class="settings-input-row"><input type="text" class="settings-input settings-input-wide" id="${id}" data-set="${esc(s.key)}" value="${esc(v)}"></div>`;
        }
        if (s.type === 'readonly') {
            return `<div class="settings-readonly">${esc(v == null ? '—' : v)}</div>`;
        }
        if (s.type === 'link') {
            const go = s.href
                ? `<button type="button" class="settings-btn settings-go" data-go="${esc(s.href)}">Open ${esc(s.href === 'devices' ? 'Devices' : s.href)}</button>`
                : '';
            return `<div class="settings-where">Set in <b>${esc(s.where)}</b>${go}</div>`;
        }
        return '';
    }

    function row(s) {
        if (s.type === 'custom') return `<div class="settings-custom" data-block="${esc(s.block)}"></div>`;
        const label = s.type === 'bool' ? '' : `<label class="settings-label">${esc(s.label)}</label>`;
        const help = s.help ? `<div class="settings-hint">${esc(s.help)}</div>` : '';
        const flag = s.overridden ? '<span class="settings-badge is-set">set here</span>' : '';
        return `<div class="settings-field settings-row" data-row="${esc(s.key)}">` +
            `<div class="settings-row-main">${label}${control(s)}${help}</div>` +
            `<div class="settings-row-side"><span class="settings-badge is-${esc(s.reach)}">${esc(badge(s))}</span>${flag}` +
            `<span class="settings-row-status" aria-live="polite"></span></div></div>`;
    }

    // ── the page ────────────────────────────────────────────────────────────
    function renderNav() {
        const nav = $('settings-nav');
        if (!nav || !_schema) return;
        nav.innerHTML = _schema.categories.map(c =>
            `<a class="settings-nav-item${c.id === _category ? ' active' : ''}" data-category="${esc(c.id)}" href="#settings:${esc(c.id)}">${esc(c.label)}</a>`).join('');
    }

    function parkBlocks() {
        // Custom blocks live in the page; before a redraw they go back to the
        // shelf so the redraw does not take them, and their state, with it.
        const shelf = $('settings-blocks');
        if (!shelf) return;
        document.querySelectorAll('#settings-body .settings-block').forEach(b => shelf.appendChild(b));
    }

    function renderBody() {
        const body = $('settings-body');
        if (!body || !_schema) return;
        parkBlocks();
        const cat = _schema.categories.find(c => c.id === _category) || _schema.categories[0];
        const mine = _schema.settings.filter(s => s.category === cat.id);
        const groups = [];
        mine.forEach(s => {
            const g = s.group || '';
            let at = groups.find(x => x.name === g);
            if (!at) { at = { name: g, items: [] }; groups.push(at); }
            at.items.push(s);
        });
        let html = `<section class="settings-section"><h2 class="settings-section-title">${esc(cat.label)}</h2>` +
            `<p class="settings-blurb">${esc(cat.blurb)}</p>`;
        groups.forEach(g => {
            if (g.name) html += `<div class="settings-subhead">${esc(g.name)}</div>`;
            html += g.items.map(row).join('');
        });
        html += '</section>';
        if (mine.some(s => (s.store || '').startsWith('prefs:'))) html += defaultsBar();
        body.innerHTML = html;
        body.querySelectorAll('.settings-custom').forEach(host => {
            const block = $(host.dataset.block);
            if (block) { host.appendChild(block); block.hidden = false; }
        });
        body.scrollTop = 0;
    }

    function defaultsBar() {
        return '<div class="settings-save-bar"><span class="settings-save-status" id="settings-save-status"></span>' +
            '<span class="settings-defaults-bar">' +
            '<button class="settings-btn" id="pref-save-defaults" type="button" title="Make what this browser shows the default for every browser on this rig">Save as rig defaults</button>' +
            '<button class="settings-btn" id="pref-reset" type="button" title="Forget this browser\'s choices and use the rig\'s">Reset to defaults</button>' +
            '<button class="settings-btn" id="pref-export" type="button">Export</button>' +
            '<button class="settings-btn" id="pref-import" type="button">Import</button>' +
            '<input type="file" id="pref-import-file" accept="application/json" hidden></span></div>';
    }

    function say(msg) {
        const el = $('settings-save-status');
        if (!el) return;
        el.textContent = msg;
        el.classList.add('visible');
        setTimeout(() => el.classList.remove('visible'), 2500);
    }

    function show(category) {
        if (_schema && _schema.categories.some(c => c.id === category)) _category = category;
        renderNav();
        renderBody();
    }

    // ── changes ─────────────────────────────────────────────────────────────
    function valueOf(s, input) {
        if (s.type === 'bool') return !!input.checked;
        if (s.type === 'multichoice') {
            return Array.from(document.querySelectorAll(`#settings-body [data-set="${CSS.escape(s.key)}"]`))
                .filter(i => i.checked).map(i => i.value);
        }
        if (s.type === 'int') return parseInt(input.value, 10);
        if (s.type === 'float') return parseFloat(input.value);
        if (s.type === 'choice') {
            const hit = (s.choices || []).find(c => String(c.value) === String(input.value));
            return hit ? hit.value : input.value;
        }
        return input.value;
    }

    function within(s, v) {
        if (s.type !== 'int' && s.type !== 'float') return true;
        if (!Number.isFinite(v)) return false;
        if (s.min != null && v < s.min) return false;
        if (s.max != null && v > s.max) return false;
        return true;
    }

    async function onChange(e) {
        const input = e.target.closest('[data-set]');
        if (!input || !_schema) return;
        const s = _schema.settings.find(x => x.key === input.dataset.set);
        if (!s) return;
        const rowEl = input.closest('.settings-row');
        const status = rowEl ? rowEl.querySelector('.settings-row-status') : null;
        const tell = (msg, bad) => {
            if (!status) return;
            status.textContent = msg;
            status.classList.toggle('is-err', !!bad);
        };
        const value = valueOf(s, input);
        if (!within(s, value)) {
            tell(`Between ${s.min} and ${s.max}`, true);
            return;
        }
        try {
            tell(await writeValue(s, value), false);
        } catch (err) {
            tell(err.message, true);
            // Put the control back: what is on screen is what is in effect.
            if (input.type === 'checkbox' && !input.dataset.multi) input.checked = !input.checked;
        }
    }

    function onClick(e) {
        const nav = e.target.closest('[data-category]');
        if (nav) { e.preventDefault(); show(nav.dataset.category); return; }
        const go = e.target.closest('[data-go]');
        if (go) { if (typeof switchTab === 'function') switchTab(go.dataset.go); return; }
        const id = e.target.id;
        if (id === 'pref-save-defaults') saveRigDefaults();
        else if (id === 'pref-reset') resetPrefs();
        else if (id === 'pref-export') exportPrefs();
        else if (id === 'pref-import') { const f = $('pref-import-file'); if (f) f.click(); }
    }

    async function saveRigDefaults() {
        try {
            const prefs = SettingsStore.merged({});
            await postJSON('/api/config/dashboard-defaults', 'PUT', prefs);
            SettingsStore.setRigDefaults(prefs);
            say('Saved as this rig\'s defaults');
        } catch (err) { say(err.message); }
    }

    function resetPrefs() {
        if (!window.confirm('Forget this browser\'s choices and use the rig\'s defaults?')) return;
        report('views.reset', SettingsStore.local(), {});
        SettingsStore.clear();
        renderBody();
        say('Reset');
    }

    function exportPrefs() {
        const blob = new Blob([JSON.stringify(SettingsStore.merged({}), null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'gently-preferences.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    async function importPrefs(file) {
        try {
            const obj = JSON.parse(await file.text());
            if (!obj || typeof obj !== 'object' || Array.isArray(obj)) throw new Error('not an object');
            report('views.import', SettingsStore.local(), obj);
            SettingsStore.replaceAll(obj);
            renderBody();
            say('Imported');
        } catch (_) { say('Import failed: that is not a preferences file'); }
    }

    // ── start ───────────────────────────────────────────────────────────────
    async function load() {
        const res = await fetch('/api/settings/schema');
        if (!res.ok) throw new Error(`schema ${res.status}`);
        _schema = await res.json();
        SettingsStore.setRigDefaults(_schema.rig_defaults || {});
    }

    async function init(category) {
        const host = $('settings-content');
        if (!host) return;
        if (!_inited) {
            _inited = true;
            host.addEventListener('click', onClick);
            host.addEventListener('change', e => {
                if (e.target.id === 'pref-import-file') {
                    const f = e.target.files[0];
                    if (f) importPrefs(f);
                    e.target.value = '';
                    return;
                }
                onChange(e);
            });
            ThermalizerSettings.init();
            DeviceLayerSettings.init();
            JoystickSetting.init();
        }
        try {
            await load();
        } catch (err) {
            const body = $('settings-body');
            if (body) body.innerHTML = `<div class="settings-hint">Settings could not be loaded (${esc(err.message)}).</div>`;
            return;
        }
        show(category || _category);
        RecordingInfo.load();
        EffectiveConfig.load();
        SettingsHistory.load();
    }

    return { init, show, badge, schema: () => _schema };
})();


/**
 * ThermalizerSettings — the ACUITYnano's connection, kept on the device layer.
 * Reads and writes /api/devices/temperature/config{,/test}. The mock backend
 * is for development (?dev=1, or localStorage 'gently-dev').
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
        if (!this.el('settings-block-thermalizer')) return;
        if (this.devMode()) {
            const m = document.querySelector('.th-mock-opt');
            if (m) m.hidden = false;
        }
        document.querySelectorAll('input[name="th-backend"]').forEach(r =>
            r.addEventListener('change', () => this.applyBackendVisibility(r.value)));
        const test = this.el('th-test'), apply = this.el('th-apply');
        if (test) test.addEventListener('click', () => this.test());
        if (apply) apply.addEventListener('click', () => this.apply());
        await this.load();
    },

    applyBackendVisibility(backend) {
        const s = this.el('th-serial'), m = this.el('th-mqtt');
        if (s) s.hidden = backend !== 'serial';
        if (m) m.hidden = backend !== 'mqtt';
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
        // The password is write-only: it is never sent back, so never shown.
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
            if (pw) cfg.password = pw;  // blank keeps the stored one
        }
        if (num('th-stabilize') != null) cfg.stabilize_timeout = num('th-stabilize');
        cfg.feedback_peltier = !!(this.el('th-peltier') && this.el('th-peltier').checked);
        return cfg;
    },

    renderStatus(d) {
        const el = this.el('th-status'); if (!el) return;
        if (!d || d.available === false) { el.textContent = 'Controller not available (device layer offline, or no thermalizer configured).'; return; }
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
            // The device layer's 409 (a run or a ramp is active) reaches here as
            // a 200, so it is read from the body.
            if (d.blocked) { this.result(`Blocked: ${d.error || 'a run or ramp is active'}`, false); return; }
            if (d.success && d.applied) { this.result('Applied live.', true); await this.load(); }
            else if (d.restart_required) { this.result(`Saved — restart the device layer to apply. (${d.error || ''})`, false); }
            else { this.result(`Failed: ${d.error || (d.detail || res.status)}`, false); }
        } catch (e) { this.result(`Error: ${e.message}`, false); }
        finally { btn.disabled = false; btn.textContent = old; }
    },
};


/** The device layer's port, and what SAM runs on. Applies at its next start. */
const DeviceLayerSettings = {
    init() {
        const port = document.getElementById('dl-port');
        const detected = document.getElementById('dl-sam-detected');
        const status = document.getElementById('dl-status');
        if (!port) return;
        let loaded = false;
        fetch('/api/launch/prefs').then(r => (r.ok ? r.json() : null)).then(p => {
            if (!p) return;
            port.value = (p.port != null ? p.port : '');
            const sam = p.sam_device_raw || 'auto';
            const radio = document.querySelector(`#dl-sam input[value="${sam}"]`)
                || document.querySelector('#dl-sam input[value="auto"]');
            if (radio) radio.checked = true;
            if (detected) {
                detected.textContent = 'Detected: ' + (p.sam_detected === 'cuda'
                    ? 'GPU (CUDA)'
                    : 'CPU — no GPU found, so image analysis will be slower');
            }
            loaded = true;
        }).catch(() => { /* offline: the block says nothing */ });
        const save = async () => {
            if (!loaded) return;
            const sam = (document.querySelector('#dl-sam input:checked') || {}).value || 'auto';
            const body = { sam_device: sam };
            const pv = parseInt(port.value, 10);
            if (pv) body.port = pv;
            try {
                const r = await fetch('/api/launch/prefs', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                if (status) status.textContent = r.ok
                    ? 'Saved — applies the next time the device layer starts'
                    : (r.status === 403 ? 'Sign in with control to change this' : 'Failed to save');
            } catch (e) { if (status) status.textContent = 'Failed to save'; }
        };
        port.addEventListener('change', save);
        document.querySelectorAll('#dl-sam input').forEach(r => r.addEventListener('change', save));
    },
};


/** The joystick lock: written to the controller and read back from it. */
const JoystickSetting = {
    init() {
        const cb = document.getElementById('hw-joystick-lock');
        const status = document.getElementById('hw-joystick-status');
        if (!cb) return;
        const show = (enabled) => {
            cb.checked = !enabled;
            status.textContent = enabled ? 'Joystick enabled at the controller' : 'Joystick LOCKED at the controller';
        };
        const offline = () => { cb.disabled = true; status.textContent = 'Microscope not connected'; };
        fetch('/api/devices/stage/joystick').then(r => r.ok ? r.json() : null).then(d => {
            if (d && d.success !== false) show(!!d.enabled);
            else offline();
        }).catch(offline);
        cb.addEventListener('change', async () => {
            const enabled = !cb.checked;
            cb.disabled = true; status.textContent = 'Writing to the controller…';
            try {
                const r = await fetch('/api/devices/stage/joystick', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled }),
                });
                const d = await r.json().catch(() => ({}));
                if (r.ok) show(!!d.enabled);
                else { cb.checked = !cb.checked; status.textContent = r.status === 403 ? 'Sign in with control to change this' : (d.detail || `Failed (${r.status})`); }
            } catch (e) { cb.checked = !cb.checked; status.textContent = 'Failed: ' + e.message; }
            finally { cb.disabled = false; }
        });
    },
};


/** How much has been recorded, and where to watch it. */
const RecordingInfo = {
    async load() {
        const el = document.getElementById('rec-usage');
        if (!el) return;
        try {
            const res = await fetch('/replay/api/recordings');
            const d = res.ok ? await res.json() : null;
            const recs = (d && d.recordings) || [];
            const bytes = recs.reduce((n, r) => n + (r.bytes || 0), 0);
            el.textContent = recs.length
                ? `${recs.length} recording${recs.length === 1 ? '' : 's'} on disk, ${(bytes / 1048576).toFixed(0)} MB together.`
                : 'Nothing has been recorded yet.';
        } catch (_) { el.textContent = 'Could not read the recordings.'; }
    },
};


/** Every change to a setting, newest first. The file is kept for good. */
const SettingsHistory = {
    esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g,
            c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    },
    say(v) {
        if (v === null || v === undefined || v === '') return '—';
        if (typeof v === 'object') return JSON.stringify(v);
        return String(v);
    },
    async load() {
        const host = document.getElementById('settings-history');
        const where = document.getElementById('settings-history-file');
        if (!host) return;
        try {
            const res = await fetch('/api/settings/history?limit=200');
            const d = res.ok ? await res.json() : null;
            const rows = (d && d.changes) || [];
            if (where && d) {
                where.textContent = `${d.total} change${d.total === 1 ? '' : 's'} kept in ${d.file}`;
                // textContent took the button with it; the file is on disk
                // once there is a change in it.
                if (d.total && typeof Reveal !== 'undefined') {
                    where.insertAdjacentHTML('beforeend', Reveal.button(
                        { what: 'settings_history' }, 'show',
                        { label: 'Show the file', title: 'Show the history file in its folder' }));
                }
            }
            if (!rows.length) { host.innerHTML = '<div class="settings-hint">No setting has been changed yet.</div>'; return; }
            host.innerHTML = '<table class="settings-history"><thead><tr>' +
                '<th>When</th><th>Setting</th><th>From</th><th>To</th><th>Reach</th><th>By</th></tr></thead><tbody>' +
                rows.map(r => {
                    const when = r.at ? new Date(r.at) : null;
                    const t = when && !isNaN(when) ? when.toLocaleString() : this.say(r.at);
                    const who = [r.by, r.client].filter(Boolean).join(' · ');
                    return `<tr title="${this.esc(r.via || '')}"><td>${this.esc(t)}</td>` +
                        `<td>${this.esc(r.label || r.key)}<div class="settings-history-key">${this.esc(r.key)}</div></td>` +
                        `<td>${this.esc(this.say(r.old))}</td><td>${this.esc(this.say(r.new))}</td>` +
                        `<td>${this.esc(r.reach === 'rig' ? 'this rig' : 'a browser')}</td>` +
                        `<td>${this.esc(who || '—')}</td></tr>`;
                }).join('') + '</tbody></table>';
        } catch (_) { host.innerHTML = '<div class="settings-hint">The history could not be read.</div>'; }
    },
};


/** Everything the server has in effect, secrets left out. Read-only. */
const EffectiveConfig = {
    async load() {
        const el = document.getElementById('effective-config');
        if (!el) return;
        try {
            const res = await fetch('/api/config/effective');
            el.textContent = res.ok ? JSON.stringify(await res.json(), null, 2) : 'Unavailable.';
        } catch (e) { el.textContent = 'Unavailable.'; }
    },
};
