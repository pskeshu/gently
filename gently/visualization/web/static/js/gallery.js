/**
 * Gallery rendering functionality for Gently Visualization
 */

function renderRecentList() {
    const list = document.getElementById('recent-list');
    if (!list) return;

    const filtered = filterByEmbryo(state.snapshots);
    const displayList = filtered.slice(-20).reverse();

    if (displayList.length === 0) {
        list.innerHTML = '<div class="recent-strip-empty">No images yet</div>';
        return;
    }

    list.innerHTML = displayList.map((img, index) => `
        <div class="recent-strip-item ${state.currentImage?.uid === img.uid ? 'active' : ''}"
             onclick="openSnapshotsLightbox(${index})"
             title="${img.data_type} - ${img.uid.slice(0, 8)}">
            <img src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
        </div>
    `).join('');
}

function openSnapshotsLightbox(index) {
    const filtered = filterByEmbryo(state.snapshots);
    const displayList = filtered.slice(-20).reverse();
    Lightbox.open(displayList, index, 'snapshots');
}

/**
 * CalibrationManager - Two-column layout for calibration images
 */
const CalibrationManager = {
    selectedEmbryoId: null,
    _lastImageCount: 0,  // Track for incremental updates

    render() {
        this.renderSidebar();
        this.renderPanel();
        this._lastImageCount = state.calibration.length;
    },

    /**
     * Incremental update when a single image is added
     * Returns true if handled incrementally, false if full render needed
     */
    handleNewImage(newImage) {
        const embryoId = newImage.metadata?.embryo_id || 'General';

        // Update sidebar count for this embryo (or add new card)
        const updated = this._updateSidebarCount(embryoId);
        if (!updated) {
            // New embryo - need to add card, do full sidebar render
            this.renderSidebar();
        }

        // Only update panel if this embryo is selected
        if (this.selectedEmbryoId === embryoId) {
            this._prependImageToPanel(newImage);
        }

        // Update lightbox if it's open and showing calibration images for this embryo
        if (Lightbox.isOpen && Lightbox.source === 'calibration' && this.selectedEmbryoId === embryoId) {
            const images = state.calibration.filter(img =>
                (img.metadata?.embryo_id || 'General') === this.selectedEmbryoId
            );
            const displayList = images.slice(-50).reverse();
            Lightbox.updateImageList(displayList);
        }

        this._lastImageCount = state.calibration.length;
        return true;
    },

    /**
     * Update just the count badge for an embryo card
     */
    _updateSidebarCount(embryoId) {
        const cardsContainer = document.getElementById('calibration-embryo-cards');
        if (!cardsContainer) return false;

        const cards = cardsContainer.querySelectorAll('.calibration-embryo-card');
        for (const card of cards) {
            const nameEl = card.querySelector('.card-name');
            if (nameEl && nameEl.textContent === embryoId) {
                const countEl = card.querySelector('.card-count');
                if (countEl) {
                    // Count images for this embryo
                    const count = state.calibration.filter(img =>
                        (img.metadata?.embryo_id || 'General') === embryoId
                    ).length;
                    countEl.textContent = `${count} image${count !== 1 ? 's' : ''}`;
                    return true;
                }
            }
        }
        return false;  // Card not found, need full render
    },

    /**
     * Prepend a single image to the panel grid without full re-render
     */
    _prependImageToPanel(newImage) {
        const grid = document.querySelector('#calibration-panel .calibration-image-grid');
        if (!grid) {
            // No grid yet, do full panel render
            this.renderPanel();
            return;
        }

        // Create new image element
        const div = document.createElement('div');
        div.className = 'gallery-item';
        div.onclick = () => CalibrationManager.openLightbox(0);  // Newest is index 0
        div.innerHTML = `
            <img class="gallery-img" src="data:image/png;base64,${newImage.base64_png}" alt="${newImage.data_type}">
            <div class="gallery-info">
                <div class="gallery-type">${newImage.data_type}</div>
                <div class="gallery-meta">${formatMeta(newImage.metadata)}</div>
            </div>
        `;

        // Insert at beginning (newest first)
        grid.insertBefore(div, grid.firstChild);

        // Update onclick indices for existing items
        const items = grid.querySelectorAll('.gallery-item');
        items.forEach((item, idx) => {
            item.onclick = () => CalibrationManager.openLightbox(idx);
        });

        // Remove excess items (keep last 50)
        // Note: items is a static NodeList, so use grid.children.length for live count
        while (grid.children.length > 50) {
            grid.removeChild(grid.lastChild);
        }
    },

    renderSidebar() {
        const cardsContainer = document.getElementById('calibration-embryo-cards');
        if (!cardsContainer) return;

        // Group images by embryo_id
        const grouped = {};
        state.calibration.forEach(img => {
            const eid = img.metadata?.embryo_id || 'General';
            if (!grouped[eid]) grouped[eid] = [];
            grouped[eid].push(img);
        });

        // Sort embryo IDs (General last)
        const sortedKeys = Object.keys(grouped).sort((a, b) => {
            if (a === 'General') return 1;
            if (b === 'General') return -1;
            return a.localeCompare(b);
        });

        if (sortedKeys.length === 0) {
            cardsContainer.innerHTML = '<div class="empty-state-small">No calibration images yet</div>';
        } else {
            cardsContainer.innerHTML = sortedKeys.map(embryoId => {
                const imgs = grouped[embryoId];
                const isSelected = this.selectedEmbryoId === embryoId;
                const safeId = embryoId.replace(/'/g, "\\'");
                return `
                    <div class="calibration-embryo-card ${isSelected ? 'selected' : ''}"
                         onclick="CalibrationManager.selectEmbryo('${safeId}')">
                        <div class="card-name">${embryoId}</div>
                        <div class="card-count">${imgs.length} image${imgs.length !== 1 ? 's' : ''}</div>
                    </div>
                `;
            }).join('');

            // Auto-select first if none selected
            if (!this.selectedEmbryoId && sortedKeys.length > 0) {
                this.selectedEmbryoId = sortedKeys[0];
                this.render();
                return;
            }
        }
    },

    renderPanel() {
        const panel = document.getElementById('calibration-panel');
        if (!panel) return;

        if (!this.selectedEmbryoId) {
            panel.innerHTML = `
                <div class="calibration-empty">
                    <div class="calibration-empty-icon">&#x1F4F7;</div>
                    <div class="calibration-empty-text">Select an embryo to view calibration images</div>
                </div>
            `;
            return;
        }

        // Get images for selected embryo
        const images = state.calibration.filter(img =>
            (img.metadata?.embryo_id || 'General') === this.selectedEmbryoId
        );

        if (images.length === 0) {
            panel.innerHTML = `
                <div class="calibration-empty">
                    <div class="calibration-empty-icon">&#x1F4F7;</div>
                    <div class="calibration-empty-text">No images for ${this.selectedEmbryoId}</div>
                </div>
            `;
            return;
        }

        const displayList = images.slice(-50).reverse();

        // Build 3D segments section (only if there are any)
        let segments3dHtml = '';
        if (state.volumes3d.length > 0) {
            segments3dHtml = `
                <div class="calibration-3d-section">
                    <div class="calibration-section-header">3D Segmentations</div>
                    <div class="calibration-3d-grid">
                        ${state.volumes3d.slice(-10).reverse().map(vol => `
                            <div class="calibration-3d-card" onclick="show3DVolume('${vol.uid}')">
                                <div class="card-cells">${vol.num_cells} cells</div>
                                <div class="card-slices">${vol.num_slices} slices</div>
                                <div class="card-uid">${vol.uid.slice(0, 10)}...</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        panel.innerHTML = `
            <div class="calibration-panel-header">${this.selectedEmbryoId}</div>
            <div class="calibration-image-grid">
                ${displayList.map((img, idx) => `
                    <div class="gallery-item" onclick="CalibrationManager.openLightbox(${idx})">
                        <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
                        <div class="gallery-info">
                            <div class="gallery-type">${img.data_type}</div>
                            <div class="gallery-meta">${formatMeta(img.metadata)}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
            ${segments3dHtml}
        `;
    },

    selectEmbryo(embryoId) {
        this.selectedEmbryoId = embryoId;
        this.render();
    },

    openLightbox(index) {
        const images = state.calibration.filter(img =>
            (img.metadata?.embryo_id || 'General') === this.selectedEmbryoId
        );
        const displayList = images.slice(-50).reverse();
        Lightbox.open(displayList, index, 'calibration');
    }
};

// Legacy wrappers kept for backward compatibility
function renderCalibrationGallery() { CalibrationManager.render(); }

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
