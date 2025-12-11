/**
 * Main application state and initialization for Gently Visualization
 */

// State
const state = {
    ws: null,
    connected: false,
    tab: 'main',
    embryoFilter: '',
    snapshots: [],
    volumes: [],
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
function updateEmbryoFilter() {
    const select = document.getElementById('embryo-filter');
    const currentValue = select.value;
    select.innerHTML = '<option value="">All Embryos</option>' +
        state.embryos.map(e => `<option value="${e}">${e}</option>`).join('');
    select.value = currentValue;
}

function updateMainCount() {
    document.getElementById('main-count').textContent = filterByEmbryo(state.snapshots).length;
}

function updateVolumesCount() {
    document.getElementById('volumes-count').textContent = filterByEmbryo(state.volumes).length;
}

function updateCalibrationCount() {
    const calCount = filterByEmbryo(state.calibration).length;
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
    if (tabName === 'volumes') renderVolumesGallery();
    if (tabName === 'calibration') renderCalibrationGallery();
    if (tabName === 'events') renderEventsTable();
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
 * Theme Manager - Dark/Light mode toggle
 */
const ThemeManager = {
    storageKey: 'gently-theme',

    init() {
        // Load saved theme or default to dark
        const savedTheme = localStorage.getItem(this.storageKey) || 'dark';
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
        '1': () => switchTab('main'),
        '2': () => switchTab('volumes'),
        '3': () => switchTab('calibration'),
        '4': () => switchTab('events'),
        '5': () => switchTab('tasks'),
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
                            <div class="shortcut"><kbd>1</kbd> Main tab</div>
                            <div class="shortcut"><kbd>2</kbd> Volumes tab</div>
                            <div class="shortcut"><kbd>3</kbd> Calibration tab</div>
                            <div class="shortcut"><kbd>4</kbd> Events tab</div>
                            <div class="shortcut"><kbd>5</kbd> Tasks tab</div>
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
    // Initialize theme manager
    ThemeManager.init();

    // Initialize keyboard shortcuts
    KeyboardShortcuts.init();

    // Tab click handlers
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Embryo filter change handler
    document.getElementById('embryo-filter').addEventListener('change', (e) => {
        state.embryoFilter = e.target.value;
        updateMainCount();
        updateVolumesCount();
        updateCalibrationCount();
        renderRecentList();
        if (state.tab === 'volumes') renderVolumesGallery();
        if (state.tab === 'calibration') renderCalibrationGallery();
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
