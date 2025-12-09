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

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
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
