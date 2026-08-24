/**
 * Image viewer and Z-slider functionality for Gently Visualization
 */

/**
 * Main Viewer Zoom Controller - delegates to ZoomPanController
 */
const MainViewerZoom = {
    _ctrl: null,

    init() {
        const container = document.getElementById('main-image-container');
        const image = document.getElementById('main-image');
        const indicator = document.getElementById('main-zoom-indicator');

        if (!container || !image) return;

        this._ctrl = new ZoomPanController({
            minScale: 0.25,
            maxScale: 8,
            target: image,
            container: container,
            indicator: indicator,
            onZoomChange: (scale) => {
                if (scale > 1) {
                    image.classList.add('zoomed');
                } else {
                    image.classList.remove('zoomed');
                }
            }
        });
        this._ctrl.bind();

        // Button controls
        document.getElementById('zoom-in')?.addEventListener('click', () => this._ctrl.handleZoomDelta(0.25));
        document.getElementById('zoom-out')?.addEventListener('click', () => this._ctrl.handleZoomDelta(-0.25));
        document.getElementById('zoom-reset')?.addEventListener('click', () => this.reset());
    },

    handleZoomDelta(delta) {
        this._ctrl?.handleZoomDelta(delta);
    },

    reset() {
        this._ctrl?.reset();
        document.getElementById('main-image')?.classList.remove('zoomed');
    }
};

// Initialize zoom on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => MainViewerZoom.init());
} else {
    MainViewerZoom.init();
}

function handleNew3DVolume(data) {
    // Add to volumes3d list
    state.volumes3d.push(data);
    // Update counts and galleries
    updateCalibrationCount();
    if (state.tab === 'calibration') {
        CalibrationManager.render();
    }
}

function hideZSlider() {
    const container = document.getElementById('z-slider-container');
    if (container) container.classList.remove('active');
    state.current3dVolume = null;
}

function handleNewImage(data) {
    const dataType = data.data_type;
    const embryoId = data.metadata?.embryo_id;

    // Route to appropriate list
    if (CALIBRATION_TYPES.includes(dataType) || ANALYSIS_TYPES.includes(dataType)) {
        state.calibration.push(data);
        updateCalibrationCount();
        // Use incremental update instead of full gallery refresh
        if (state.tab === 'calibration') {
            CalibrationManager.handleNewImage(data);
        }
    } else if (VOLUME_TYPES.includes(dataType)) {
        // Volume images go to snapshots (Gallery tab removed)
        state.snapshots.push(data);
        updateMainCount();
        renderRecentList();
    } else {
        state.snapshots.push(data);
        updateMainCount();
        renderRecentList();
    }

    // Update embryo list if new
    if (embryoId && !state.embryos.includes(embryoId)) {
        state.embryos.push(embryoId);
    }

    // Show on main viewer if main tab
    if (state.tab === 'main') {
        displayImage(data);
    }
}

function displayImage(data) {
    state.currentImage = data;

    const img = document.getElementById('main-image');
    if (!img) {
        // Live View tab removed — just update recent strip
        renderRecentList();
        return;
    }

    // Hide Z slider when showing regular 2D images
    hideZSlider();

    // Reset zoom when showing new image
    MainViewerZoom.reset();

    const placeholder = document.getElementById('placeholder');
    const viewerInfo = document.getElementById('viewer-info');

    if (data.base64_png) {
        img.src = 'data:image/png;base64,' + data.base64_png;
        img.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
        if (viewerInfo) viewerInfo.style.display = 'flex';
    }

    const embryoId = data.metadata?.embryo_id || '-';
    const infoEmbryo = document.getElementById('info-embryo');
    if (infoEmbryo) infoEmbryo.textContent = embryoId;
    const infoType = document.getElementById('info-type');
    if (infoType) infoType.textContent = data.data_type;
    const infoUid = document.getElementById('info-uid');
    if (infoUid) infoUid.textContent = data.uid.slice(0, 12) + '...';
    const imageTime = document.getElementById('image-time');
    if (imageTime) imageTime.textContent = new Date(data.timestamp).toLocaleTimeString();

    // Update recent strip to show active state
    renderRecentList();
}
