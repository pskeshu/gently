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

        content.addEventListener('change', () => this.readFormAndSave());
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
