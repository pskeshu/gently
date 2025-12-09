/**
 * Image viewer and Z-slider functionality for Gently Visualization
 */

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
        state.volumes.push(data);
        updateVolumesCount();
        if (state.tab === 'volumes') renderVolumesGallery();
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
        updateEmbryoFilter();
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

    const img = document.getElementById('main-image');
    const placeholder = document.getElementById('placeholder');

    if (data.base64_png) {
        img.src = 'data:image/png;base64,' + data.base64_png;
        img.style.display = 'block';
        placeholder.style.display = 'none';
    }

    const embryoId = data.metadata?.embryo_id || '-';
    document.getElementById('current-embryo').textContent = embryoId !== '-' ? embryoId : '';
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
    const list = source === 'volumes' ? state.volumes : state.calibration;
    const img = list.find(i => i.uid === uid);
    if (img) displayImage(img);
    // Switch to main tab to show the image
    switchTab('main');
}
