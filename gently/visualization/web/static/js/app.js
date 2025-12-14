/**
 * Main application state and initialization for Gently Visualization
 */

// State
const state = {
    ws: null,
    connected: false,
    tab: 'embryos',  // Default to Embryos tab
    snapshots: [],
    calibration: [],
    embryos: [],
    volumes3d: [],
    currentImage: null,
    current3dVolume: null,  // Currently displayed 3D volume
    currentZ: 0,
    // Events tab state
    allEvents: [],
    eventTypeFilter: '',
    eventSourceFilter: '',
    eventSources: new Set()
};

// Data type classification
const CALIBRATION_TYPES = ['focus_sweep', 'focus_plot', 'edge_detection', 'calibration_summary',
                           'focus_snap', 'focus_coarse', 'focus_curve', 'focus_assess'];
const ANALYSIS_TYPES = ['segmentation', 'detection', 'classification', 'tracking',
                        'roi_detection', 'cropped_roi', 'vision_prepared', 'timeline', 'cv_visualization'];
const VOLUME_TYPES = ['volume', 'volume_projection', 'z_stack', 'timelapse'];

// UI Update functions
function updateMainCount() {
    const el = document.getElementById('main-count');
    if (el) el.textContent = state.snapshots.length;
}

function updateCalibrationCount() {
    const calCount = state.calibration.length;
    const vol3dCount = state.volumes3d.length;
    document.getElementById('calibration-count').textContent = calCount + vol3dCount;
}

function switchTab(tabName) {
    state.tab = tabName;

    // Update tab styling
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');

    // Show/hide content
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`${tabName}-content`).classList.add('active');

    // Render galleries
    if (tabName === 'calibration') renderCalibrationGallery();
    if (tabName === 'events') renderEventsTable();

    // Clear detection badge when viewing Embryos tab
    if (tabName === 'embryos' && typeof EmbryosManager !== 'undefined') {
        EmbryosManager.clearDetectionBadge();
    }
}

function logEvent(type, message) {
    const log = document.getElementById('event-log');
    const div = document.createElement('div');
    div.className = 'event-item';
    // CV events get special styling
    const isCvEvent = type.startsWith('CV_') || type === 'SEGMENTATION_COMPLETED' || type === 'STAGE_DETECTED';
    const typeClass = isCvEvent ? 'event-type cv-event' : 'event-type';
    div.innerHTML = `<span class="event-time">${new Date().toLocaleTimeString()}</span>
                    <span class="${typeClass}">${type}</span>: ${message}`;
    log.insertBefore(div, log.firstChild);
    while (log.children.length > 50) log.removeChild(log.lastChild);
}

/**
 * Tooltip System - Shows helpful hints on hover
 */
const Tooltips = {
    current: null,
    timeout: null,

    init() {
        // Use event delegation for better performance
        document.addEventListener('mouseenter', (e) => {
            if (!e.target || !e.target.closest) return;
            const target = e.target.closest('[data-tooltip]');
            if (target) this.show(target);
        }, true);

        document.addEventListener('mouseleave', (e) => {
            if (!e.target || !e.target.closest) return;
            const target = e.target.closest('[data-tooltip]');
            if (target) this.hide();
        }, true);

        // Hide on scroll
        document.addEventListener('scroll', () => this.hide(), true);
    },

    show(target) {
        const text = target.dataset.tooltip;
        if (!text) return;

        // Small delay before showing
        this.timeout = setTimeout(() => {
            // Remove any existing tooltip
            this.hide();

            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = text;
            document.body.appendChild(tooltip);

            // Position below the target element
            const rect = target.getBoundingClientRect();
            const tooltipRect = tooltip.getBoundingClientRect();

            // Default position below, centered
            let top = rect.bottom + 8;
            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);

            // Keep within viewport bounds
            if (left < 8) left = 8;
            if (left + tooltipRect.width > window.innerWidth - 8) {
                left = window.innerWidth - tooltipRect.width - 8;
            }

            // If below would go off-screen, show above
            if (top + tooltipRect.height > window.innerHeight - 8) {
                top = rect.top - tooltipRect.height - 8;
                tooltip.classList.add('tooltip-above');
            }

            tooltip.style.top = `${top}px`;
            tooltip.style.left = `${left}px`;

            this.current = tooltip;
        }, 150);  // 150ms delay - faster for better responsiveness
    },

    hide() {
        if (this.timeout) {
            clearTimeout(this.timeout);
            this.timeout = null;
        }
        if (this.current) {
            this.current.remove();
            this.current = null;
        }
    }
};

/**
 * Theme Manager - Dark/Light mode toggle
 */
const ThemeManager = {
    storageKey: 'gently-theme',

    init() {
        // Load saved theme or default to light
        const savedTheme = localStorage.getItem(this.storageKey) || 'light';
        this.setTheme(savedTheme);

        // Setup toggle button
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggle());
        }
    },

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(this.storageKey, theme);
    },

    toggle() {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        this.setTheme(next);
    },

    getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    }
};

/**
 * Keyboard Shortcuts Handler
 */
const KeyboardShortcuts = {
    enabled: true,

    shortcuts: {
        '1': () => switchTab('embryos'),     // Embryos
        '2': () => switchTab('events'),      // System
        '3': () => switchTab('main'),        // Live View
        '4': () => switchTab('calibration'), // Setup
        'ArrowUp': () => KeyboardShortcuts.adjustZSlider(1),
        'ArrowDown': () => KeyboardShortcuts.adjustZSlider(-1),
        '?': () => KeyboardShortcuts.showHelp(),
        't': () => ThemeManager.toggle(),
    },

    init() {
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
    },

    handleKeyDown(e) {
        // Ignore if typing in input or lightbox is open
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (typeof Lightbox !== 'undefined' && Lightbox.isOpen) return;

        const handler = this.shortcuts[e.key];
        if (handler) {
            e.preventDefault();
            handler();
        }
    },

    adjustZSlider(direction) {
        if (state.tab !== 'main' || !state.current3dVolume) return;

        const slider = document.getElementById('z-slider');
        if (!slider) return;

        const newVal = parseInt(slider.value) + direction;
        if (newVal >= parseInt(slider.min) && newVal <= parseInt(slider.max)) {
            slider.value = newVal;
            slider.dispatchEvent(new Event('input'));
        }
    },

    showHelp() {
        // Create help modal if it doesn't exist
        let modal = document.getElementById('shortcuts-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'shortcuts-modal';
            modal.className = 'shortcuts-modal';
            modal.innerHTML = `
                <div class="shortcuts-content">
                    <div class="shortcuts-header">
                        <h3>Keyboard Shortcuts</h3>
                        <button class="shortcuts-close" onclick="KeyboardShortcuts.hideHelp()">&times;</button>
                    </div>
                    <div class="shortcuts-body">
                        <div class="shortcut-group">
                            <h4>Navigation</h4>
                            <div class="shortcut"><kbd>1</kbd> Embryos tab</div>
                            <div class="shortcut"><kbd>2</kbd> System tab</div>
                            <div class="shortcut"><kbd>3</kbd> Live View tab</div>
                            <div class="shortcut"><kbd>4</kbd> Setup tab</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>3D Volume</h4>
                            <div class="shortcut"><kbd>&uarr;</kbd> <kbd>&darr;</kbd> Navigate Z-slices</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>Lightbox</h4>
                            <div class="shortcut"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> Prev/Next image</div>
                            <div class="shortcut"><kbd>Esc</kbd> Close lightbox</div>
                            <div class="shortcut"><kbd>+</kbd> <kbd>-</kbd> Zoom in/out</div>
                            <div class="shortcut"><kbd>0</kbd> Reset zoom</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>Events</h4>
                            <div class="shortcut"><kbd>Ctrl</kbd>+<kbd>F</kbd> Search events</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>General</h4>
                            <div class="shortcut"><kbd>t</kbd> Toggle dark/light theme</div>
                            <div class="shortcut"><kbd>?</kbd> Show this help</div>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);

            // Close on backdrop click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.hideHelp();
            });
        }

        modal.classList.add('active');
    },

    hideHelp() {
        const modal = document.getElementById('shortcuts-modal');
        if (modal) modal.classList.remove('active');
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltip system
    Tooltips.init();

    // Initialize theme manager
    ThemeManager.init();

    // Initialize keyboard shortcuts
    KeyboardShortcuts.init();

    // Tab click handlers
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Z-slider event listener
    document.getElementById('z-slider').addEventListener('input', (e) => {
        if (!state.current3dVolume) return;
        state.currentZ = parseInt(e.target.value);
        updateZSliderDisplay();
        loadZSlice(state.current3dVolume.uid, state.currentZ);
    });

    // Initialize events tab
    initEventsTab();

    // Start WebSocket connection
    connectWebSocket();
});
