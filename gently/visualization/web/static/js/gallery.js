/**
 * Gallery rendering functionality for Gently Visualization
 */

function renderRecentList() {
    const list = document.getElementById('recent-list');
    const filtered = filterByEmbryo(state.snapshots);

    list.innerHTML = filtered.slice(-15).reverse().map(img => `
        <div class="gallery-item" style="margin-bottom: 0.5rem;" onclick="displayImage(state.snapshots.find(i => i.uid === '${img.uid}'))">
            <div class="gallery-info">
                <div class="gallery-type">${img.data_type}</div>
                <div class="gallery-meta">${img.uid.slice(0, 8)}...</div>
            </div>
        </div>
    `).join('');
}

function renderVolumesGallery() {
    const gallery = document.getElementById('volumes-gallery');
    const filtered = filterByEmbryo(state.volumes);

    if (filtered.length === 0) {
        gallery.innerHTML = '<div class="empty-state">No volume images yet</div>';
        return;
    }

    gallery.innerHTML = filtered.slice(-50).reverse().map(img => `
        <div class="gallery-item" onclick="showInModal('${img.uid}', 'volumes')">
            <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
            <div class="gallery-info">
                <div class="gallery-type">${img.data_type}</div>
                <div class="gallery-meta">${img.metadata?.embryo_id || 'unknown'}</div>
            </div>
        </div>
    `).join('');
}

function renderCalibrationGallery() {
    const gallery = document.getElementById('calibration-gallery');
    const filtered = filterByEmbryo(state.calibration);

    if (filtered.length === 0) {
        gallery.innerHTML = '<div class="empty-state">No calibration images yet</div>';
        return;
    }

    gallery.innerHTML = filtered.slice(-50).reverse().map(img => `
        <div class="gallery-item" onclick="showInModal('${img.uid}', 'calibration')">
            <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
            <div class="gallery-info">
                <div class="gallery-type">${img.data_type}</div>
                <div class="gallery-meta">${img.metadata?.embryo_id || ''} ${formatMeta(img.metadata)}</div>
            </div>
        </div>
    `).join('');

    // Also render 3D volumes
    render3DVolumesGallery();
}

function render3DVolumesGallery() {
    const gallery = document.getElementById('volumes3d-gallery');
    if (!gallery) return;

    if (state.volumes3d.length === 0) {
        gallery.innerHTML = '<div class="empty-state">No 3D segmentations yet</div>';
        return;
    }

    gallery.innerHTML = state.volumes3d.slice(-20).reverse().map(vol => `
        <div class="gallery-item" onclick="show3DVolume('${vol.uid}')" style="min-width: 120px;">
            <div class="gallery-info" style="padding: 0.75rem; text-align: center;">
                <div class="gallery-type">3D Seg</div>
                <div class="gallery-meta" style="font-size: 1.1rem; color: var(--accent-green);">${vol.num_cells} cells</div>
                <div class="gallery-meta">${vol.num_slices} slices</div>
                <div class="gallery-meta" style="font-size: 0.65rem;">${vol.uid.slice(0, 12)}...</div>
            </div>
        </div>
    `).join('');
}

function formatMeta(meta) {
    if (!meta) return '';
    if (meta.focus_score) return `score: ${meta.focus_score.toFixed(2)}`;
    if (meta.piezo_um) return `${meta.piezo_um.toFixed(1)}um`;
    return '';
}

function filterByEmbryo(list) {
    if (!state.embryoFilter) return list;
    return list.filter(img => img.metadata?.embryo_id === state.embryoFilter);
}
