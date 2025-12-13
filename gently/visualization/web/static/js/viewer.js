/**
 * Image viewer and Z-slider functionality for Gently Visualization
 */

/**
 * Main Viewer Zoom Controller
 */
const MainViewerZoom = {
    zoom: { scale: 1, offsetX: 0, offsetY: 0, isDragging: false, startX: 0, startY: 0 },
    container: null,
    image: null,
    indicator: null,
    indicatorTimeout: null,

    init() {
        this.container = document.getElementById('main-image-container');
        this.image = document.getElementById('main-image');
        this.indicator = document.getElementById('main-zoom-indicator');

        if (!this.container || !this.image) return;

        this.bindEvents();
    },

    bindEvents() {
        // Mouse wheel zoom
        this.container.addEventListener('wheel', (e) => {
            if (this.image.style.display === 'none') return;
            e.preventDefault();
            this.handleZoomDelta(e.deltaY > 0 ? -0.15 : 0.15);
        }, { passive: false });

        // Double-click reset
        this.container.addEventListener('dblclick', () => this.reset());

        // Pan
        this.container.addEventListener('mousedown', (e) => this.startPan(e));
        document.addEventListener('mousemove', (e) => this.doPan(e));
        document.addEventListener('mouseup', () => this.endPan());

        // Button controls
        document.getElementById('zoom-in')?.addEventListener('click', () => this.handleZoomDelta(0.25));
        document.getElementById('zoom-out')?.addEventListener('click', () => this.handleZoomDelta(-0.25));
        document.getElementById('zoom-reset')?.addEventListener('click', () => this.reset());
    },

    handleZoomDelta(delta) {
        const newScale = Math.max(0.25, Math.min(8, this.zoom.scale + delta));
        if (newScale !== this.zoom.scale) {
            this.zoom.scale = newScale;
            this.apply();
            this.showIndicator();
        }
    },

    startPan(e) {
        if (this.zoom.scale <= 1) return;
        if (e.target.closest('.zoom-controls')) return; // Don't pan when clicking controls

        this.zoom.isDragging = true;
        this.zoom.startX = e.clientX - this.zoom.offsetX;
        this.zoom.startY = e.clientY - this.zoom.offsetY;
        this.image.classList.add('zoomed');
    },

    doPan(e) {
        if (!this.zoom.isDragging) return;
        this.zoom.offsetX = e.clientX - this.zoom.startX;
        this.zoom.offsetY = e.clientY - this.zoom.startY;
        this.apply();
    },

    endPan() {
        this.zoom.isDragging = false;
    },

    reset() {
        this.zoom = { scale: 1, offsetX: 0, offsetY: 0, isDragging: false, startX: 0, startY: 0 };
        this.apply();
        this.image?.classList.remove('zoomed');
    },

    apply() {
        if (this.image) {
            this.image.style.transform =
                `translate(${this.zoom.offsetX}px, ${this.zoom.offsetY}px) scale(${this.zoom.scale})`;

            if (this.zoom.scale > 1) {
                this.image.classList.add('zoomed');
            } else {
                this.image.classList.remove('zoomed');
            }
        }
    },

    showIndicator() {
        if (this.indicator) {
            this.indicator.textContent = `${Math.round(this.zoom.scale * 100)}%`;
            this.indicator.classList.add('visible');

            clearTimeout(this.indicatorTimeout);
            this.indicatorTimeout = setTimeout(() => {
                this.indicator?.classList.remove('visible');
            }, 1200);
        }
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
    logEvent('segmentation', `3D: ${data.num_cells} cells, ${data.num_slices} slices`);

    // Update counts and galleries
    updateCalibrationCount();
    if (state.tab === 'calibration') {
        render3DVolumesGallery();
    }

    // Auto-display the new 3D volume
    if (state.tab === 'main') {
        display3DVolume(data);
    }
}

function display3DVolume(data) {
    state.current3dVolume = data;
    state.currentZ = Math.floor(data.num_slices / 2);  // Start in middle

    // Setup slider
    const slider = document.getElementById('z-slider');
    const sliderContainer = document.getElementById('z-slider-container');

    slider.min = 0;
    slider.max = data.num_slices - 1;
    slider.value = state.currentZ;

    // Show slider
    sliderContainer.classList.add('active');

    // Update info
    updateZSliderDisplay();

    // Load the slice
    loadZSlice(data.uid, state.currentZ);

    // Update image info
    document.getElementById('info-type').textContent = '3D Segmentation';
    document.getElementById('info-shape').textContent = data.shape.join(' x ');
    document.getElementById('info-uid').textContent = data.uid.slice(0, 16) + '...';
    document.getElementById('info-embryo').textContent = data.metadata?.embryo_id || '-';
    document.getElementById('image-info').textContent = `3D Seg: ${data.num_cells} cells`;
    document.getElementById('image-time').textContent = new Date(data.timestamp).toLocaleTimeString();
}

function loadZSlice(uid, z) {
    const img = document.getElementById('main-image');
    const placeholder = document.getElementById('placeholder');

    // Add cache buster to force reload
    img.src = `/api/volumes3d/${uid}/slice/${z}?t=${Date.now()}`;
    img.style.display = 'block';
    placeholder.style.display = 'none';
}

function updateZSliderDisplay() {
    const data = state.current3dVolume;
    if (!data) return;

    document.getElementById('z-slider-value').textContent = state.currentZ;
    document.getElementById('z-slider-info').textContent = `${state.currentZ + 1} / ${data.num_slices}`;
}

function hideZSlider() {
    document.getElementById('z-slider-container').classList.remove('active');
    state.current3dVolume = null;
}

function handleNewImage(data) {
    const dataType = data.data_type;
    const embryoId = data.metadata?.embryo_id;

    // Route to appropriate list
    if (CALIBRATION_TYPES.includes(dataType) || ANALYSIS_TYPES.includes(dataType)) {
        state.calibration.push(data);
        updateCalibrationCount();
        if (state.tab === 'calibration') renderCalibrationGallery();
        const eventType = ANALYSIS_TYPES.includes(dataType) ? 'analysis' : 'calibration';
        logEvent(eventType, `${dataType}${embryoId ? ' ' + embryoId : ''}`);
    } else if (VOLUME_TYPES.includes(dataType)) {
        // Volume images go to snapshots (Gallery tab removed)
        state.snapshots.push(data);
        updateMainCount();
        renderRecentList();
        logEvent('volume', `${dataType}${embryoId ? ' ' + embryoId : ''}`);
    } else {
        state.snapshots.push(data);
        updateMainCount();
        renderRecentList();
        logEvent('image', `${dataType}${embryoId ? ' ' + embryoId : ''}`);
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

    // Hide Z slider when showing regular 2D images
    hideZSlider();

    // Reset zoom when showing new image
    MainViewerZoom.reset();

    const img = document.getElementById('main-image');
    const placeholder = document.getElementById('placeholder');

    if (data.base64_png) {
        img.src = 'data:image/png;base64,' + data.base64_png;
        img.style.display = 'block';
        placeholder.style.display = 'none';
    }

    const embryoId = data.metadata?.embryo_id || '-';
    document.getElementById('info-embryo').textContent = embryoId;
    document.getElementById('info-type').textContent = data.data_type;
    document.getElementById('info-shape').textContent = data.shape ? data.shape.join(' x ') : '-';
    document.getElementById('info-uid').textContent = data.uid.slice(0, 16) + '...';
    document.getElementById('image-info').textContent = data.data_type;
    document.getElementById('image-time').textContent = new Date(data.timestamp).toLocaleTimeString();
}

function show3DVolume(uid) {
    const vol = state.volumes3d.find(v => v.uid === uid);
    if (vol) {
        display3DVolume(vol);
        switchTab('main');
    }
}

function showInModal(uid, source) {
    const img = state.calibration.find(i => i.uid === uid);
    if (img) displayImage(img);
    // Switch to main tab to show the image
    switchTab('main');
}
