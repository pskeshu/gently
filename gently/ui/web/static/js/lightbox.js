/**
 * Lightbox module for Gently Visualization
 * Provides full-screen image viewing with navigation, zoom, and thumbnails
 */

const Lightbox = {
    // State
    isOpen: false,
    currentIndex: 0,
    imageList: [],
    source: null,
    _zoomCtrl: null,

    // DOM references (cached on init)
    els: {},

    init() {
        // Cache DOM elements
        this.els = {
            overlay: document.getElementById('lightbox-overlay'),
            image: document.getElementById('lightbox-image'),
            imageWrapper: document.getElementById('lightbox-image-wrapper'),
            title: document.getElementById('lightbox-title'),
            position: document.getElementById('lightbox-position'),
            thumbnails: document.getElementById('lightbox-thumbnails'),
            zoomIndicator: document.getElementById('lightbox-zoom-indicator'),
            prevBtn: document.getElementById('lightbox-prev'),
            nextBtn: document.getElementById('lightbox-next'),
            closeBtn: document.getElementById('lightbox-close'),
            infoType: document.getElementById('lb-info-type'),
            infoEmbryo: document.getElementById('lb-info-embryo'),
            infoShape: document.getElementById('lb-info-shape'),
            infoTime: document.getElementById('lb-info-time'),
        };

        if (!this.els.overlay) {
            console.warn('Lightbox: overlay element not found');
            return;
        }

        // Initialize zoom/pan controller
        if (this.els.imageWrapper && this.els.image) {
            this._zoomCtrl = new ZoomPanController({
                minScale: 0.5,
                maxScale: 6,
                target: this.els.image,
                container: this.els.imageWrapper,
                indicator: this.els.zoomIndicator,
                onZoomChange: (scale) => {
                    if (this.els.imageWrapper) {
                        this.els.imageWrapper.style.cursor = scale > 1 ? 'grab' : 'default';
                    }
                }
            });
            this._zoomCtrl.bind();
        }

        this.bindEvents();
    },

    bindEvents() {
        // Navigation
        this.els.prevBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.navigate(-1);
        });
        this.els.nextBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.navigate(1);
        });
        this.els.closeBtn?.addEventListener('click', () => this.close());

        // Close on overlay click (but not on content click)
        this.els.overlay?.addEventListener('click', (e) => {
            if (e.target === this.els.overlay) this.close();
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!this.isOpen) return;

            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.navigate(-1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.navigate(1);
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.close();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.goTo(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this.goTo(this.imageList.length - 1);
                    break;
                case '0':
                case 'r':
                    e.preventDefault();
                    this.resetZoom();
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    this._zoomCtrl?.handleZoomDelta(0.25);
                    break;
                case '-':
                    e.preventDefault();
                    this._zoomCtrl?.handleZoomDelta(-0.25);
                    break;
            }
        });

        // Prevent context menu on image
        this.els.image?.addEventListener('contextmenu', (e) => e.preventDefault());
    },

    open(imageList, startIndex = 0, source = null) {
        if (!imageList || imageList.length === 0) {
            console.warn('Lightbox: no images to display');
            return;
        }

        this.imageList = imageList;
        this.currentIndex = Math.max(0, Math.min(startIndex, imageList.length - 1));
        this.source = source;
        this.isOpen = true;

        this.els.overlay?.classList.add('active');
        document.body.style.overflow = 'hidden';

        this.resetZoom();
        this.showImage(this.currentIndex);
        this.renderThumbnails();
    },

    /**
     * Open lightbox with a sequence of images loaded by UID
     * Enables left/right navigation through the sequence
     */
    openWithSequence(imageList, startIndex = 0, source = null) {
        if (!imageList || imageList.length === 0) {
            console.warn('Lightbox: no images to display');
            return;
        }

        this.imageList = imageList;
        this.currentIndex = Math.max(0, Math.min(startIndex, imageList.length - 1));
        this.source = source;
        this.isOpen = true;

        this.els.overlay?.classList.add('active');
        document.body.style.overflow = 'hidden';

        this.resetZoom();
        this.showImageByUid(this.currentIndex);
        this.renderThumbnailsByUid();

        // Show nav buttons for multi-image sequences
        if (this.els.prevBtn) this.els.prevBtn.style.display = imageList.length > 1 ? '' : 'none';
        if (this.els.nextBtn) this.els.nextBtn.style.display = imageList.length > 1 ? '' : 'none';
        if (this.els.thumbnails) this.els.thumbnails.style.display = imageList.length > 1 ? '' : 'none';
    },

    showImageByUid(index) {
        const img = this.imageList[index];
        if (!img) return;

        // Load image from API by UID
        if (this.els.image && img.uid) {
            this.els.image.classList.add('transitioning');
            this.els.image.src = `/api/images/${img.uid}/png`;
            setTimeout(() => {
                this.els.image.classList.remove('transitioning');
            }, 150);
        }

        // Update info - use timepoint from metadata if available
        const timepoint = img.metadata?.timepoint;
        const title = timepoint !== undefined ? `T${timepoint}` : (img.data_type || 'Image');
        if (this.els.title) this.els.title.textContent = title;
        if (this.els.position) this.els.position.textContent = `${index + 1} of ${this.imageList.length}`;

        // Update info panel
        if (this.els.infoType) this.els.infoType.textContent = img.data_type || '-';
        if (this.els.infoEmbryo) this.els.infoEmbryo.textContent = img.metadata?.embryo_id || '-';
        if (this.els.infoShape) {
            const shape = img.shape || img.metadata?.shape;
            this.els.infoShape.textContent = Array.isArray(shape) ? shape.join(' x ') : (shape || '-');
        }
        if (this.els.infoTime) {
            if (timepoint !== undefined) {
                this.els.infoTime.textContent = `T${timepoint}`;
            } else if (img.timestamp) {
                this.els.infoTime.textContent = new Date(img.timestamp).toLocaleTimeString();
            } else {
                this.els.infoTime.textContent = '-';
            }
        }

        // Update nav button states
        if (this.els.prevBtn) this.els.prevBtn.disabled = index === 0;
        if (this.els.nextBtn) this.els.nextBtn.disabled = index === this.imageList.length - 1;
    },

    renderThumbnailsByUid() {
        if (!this.els.thumbnails) return;

        this.els.thumbnails.innerHTML = '';

        // Show subset of thumbnails centered on current
        const visible = 9;
        const half = Math.floor(visible / 2);
        let start = Math.max(0, this.currentIndex - half);
        let end = Math.min(this.imageList.length, start + visible);

        if (end - start < visible) {
            start = Math.max(0, end - visible);
        }

        for (let i = start; i < end; i++) {
            const img = this.imageList[i];
            const thumb = document.createElement('div');
            thumb.className = `lightbox-thumb ${i === this.currentIndex ? 'active' : ''}`;
            thumb.dataset.index = i;

            // Load thumbnail from API
            if (img.uid) {
                thumb.innerHTML = `<img src="/api/images/${img.uid}/png" alt="T${img.metadata?.timepoint ?? i}">`;
            } else {
                thumb.innerHTML = `<span class="thumb-placeholder">T${img.metadata?.timepoint ?? i}</span>`;
            }

            thumb.addEventListener('click', () => this.goToByUid(i));
            this.els.thumbnails.appendChild(thumb);
        }
    },

    goToByUid(index) {
        if (index < 0 || index >= this.imageList.length) return;

        this.currentIndex = index;
        this.resetZoom();
        this.showImageByUid(index);

        // Update thumbnail highlight
        const thumbs = this.els.thumbnails?.querySelectorAll('.lightbox-thumb');
        if (thumbs) {
            let needsRerender = true;
            thumbs.forEach((thumb) => {
                const i = parseInt(thumb.dataset.index);
                const isActive = i === this.currentIndex;
                thumb.classList.toggle('active', isActive);
                if (isActive) needsRerender = false;
            });
            if (needsRerender) {
                this.renderThumbnailsByUid();
            }
        }
    },

    /**
     * Open lightbox with a single image by UID
     * Nav buttons will be hidden since there's only one image
     */
    openByUid(uid, metadata = {}) {
        if (!uid) return;

        // Create image object with UID and optional metadata
        const img = {
            uid: uid,
            base64_png: null,  // Will load via src
            data_type: metadata.data_type || 'Image',
            metadata: {
                embryo_id: metadata.embryo_id,
                timepoint: metadata.timepoint,
                shape: metadata.shape
            }
        };

        this.imageList = [img];
        this.currentIndex = 0;
        this.source = 'single';
        this.isOpen = true;

        this.els.overlay?.classList.add('active');
        document.body.style.overflow = 'hidden';

        this.resetZoom();

        // Load image directly from API
        if (this.els.image) {
            this.els.image.src = `/api/images/${uid}/png`;
        }

        // Update UI for single image
        const title = metadata.timepoint !== undefined
            ? `T${metadata.timepoint}`
            : 'Image';
        if (this.els.title) this.els.title.textContent = title;
        if (this.els.position) this.els.position.textContent = '';

        // Update info panel
        const infoType = document.getElementById('lb-info-type');
        const infoEmbryo = document.getElementById('lb-info-embryo');
        const infoShape = document.getElementById('lb-info-shape');
        const infoTime = document.getElementById('lb-info-time');

        if (infoType) infoType.textContent = metadata.data_type || 'Image';
        if (infoEmbryo) infoEmbryo.textContent = metadata.embryo_id || '-';
        if (infoTime) infoTime.textContent = metadata.timepoint !== undefined ? `T${metadata.timepoint}` : '-';

        // Hide shape if not available
        if (infoShape) {
            const shapeItem = infoShape.closest('.lb-info-item');
            if (metadata.shape && metadata.shape !== '-') {
                infoShape.textContent = metadata.shape;
                if (shapeItem) shapeItem.style.display = '';
            } else {
                infoShape.textContent = '-';
                if (shapeItem) shapeItem.style.display = 'none';
            }
        }

        // Hide nav buttons for single image
        if (this.els.prevBtn) this.els.prevBtn.style.display = 'none';
        if (this.els.nextBtn) this.els.nextBtn.style.display = 'none';

        // Hide thumbnails for single image
        if (this.els.thumbnails) this.els.thumbnails.style.display = 'none';
    },

    close() {
        this.isOpen = false;
        this.els.overlay?.classList.remove('active');
        document.body.style.overflow = '';
        this.resetZoom();

        // Restore nav buttons visibility for next open
        if (this.els.prevBtn) this.els.prevBtn.style.display = '';
        if (this.els.nextBtn) this.els.nextBtn.style.display = '';
        if (this.els.thumbnails) this.els.thumbnails.style.display = '';
    },

    /**
     * Update the image list while lightbox is open (e.g., when new images arrive)
     * Preserves current position if possible, re-renders thumbnails
     */
    updateImageList(newImageList) {
        if (!this.isOpen || !newImageList || newImageList.length === 0) return;

        // Find the UID of the currently displayed image to try to preserve position
        const currentImage = this.imageList[this.currentIndex];
        const currentUid = currentImage?.uid;

        this.imageList = newImageList;

        // Try to find the same image in the new list
        if (currentUid) {
            const newIndex = newImageList.findIndex(img => img.uid === currentUid);
            if (newIndex >= 0) {
                this.currentIndex = newIndex;
            } else {
                // Image no longer in list, clamp to valid range
                this.currentIndex = Math.min(this.currentIndex, newImageList.length - 1);
            }
        } else {
            this.currentIndex = Math.min(this.currentIndex, newImageList.length - 1);
        }

        // Update position display
        if (this.els.position) {
            this.els.position.textContent = `${this.currentIndex + 1} of ${this.imageList.length}`;
        }

        // Re-render thumbnails
        this.renderThumbnails();
    },

    navigate(direction) {
        const newIndex = this.currentIndex + direction;
        if (newIndex >= 0 && newIndex < this.imageList.length) {
            this.goTo(newIndex);
        }
    },

    goTo(index) {
        if (index < 0 || index >= this.imageList.length) return;

        this.currentIndex = index;
        this.resetZoom();

        // Use UID-based loading for reasoning panel sequences
        if (this.source === 'reasoning') {
            this.showImageByUid(index);
            // Update thumbnail highlight for UID-based thumbnails
            const thumbs = this.els.thumbnails?.querySelectorAll('.lightbox-thumb');
            if (thumbs) {
                let needsRerender = true;
                thumbs.forEach((thumb) => {
                    const i = parseInt(thumb.dataset.index);
                    const isActive = i === this.currentIndex;
                    thumb.classList.toggle('active', isActive);
                    if (isActive) needsRerender = false;
                });
                if (needsRerender) {
                    this.renderThumbnailsByUid();
                }
            }
        } else {
            this.showImage(index);
            this.updateThumbnailHighlight();
        }
    },

    showImage(index) {
        const img = this.imageList[index];
        if (!img) return;

        // Animate image transition
        if (this.els.image) {
            this.els.image.classList.add('transitioning');

            setTimeout(() => {
                // A record without base64_png used to leave the PREVIOUS image on
                // screen while the title and metadata updated to the new item —
                // you saw image A captioned as item B, which is worse than
                // seeing nothing. Volume3DData.to_info_dict carries no
                // base64_png, and events.js:155 opens volumes through here.
                if (img.base64_png) {
                    this.els.image.src = 'data:image/png;base64,' + img.base64_png;
                } else if (img.num_slices) {
                    // a 3D volume: show its middle slice, which the API renders
                    const mid = Math.floor(img.num_slices / 2);
                    this.els.image.src = `/api/volumes3d/${img.uid}/slice/${mid}`;
                } else if (img.uid) {
                    this.els.image.src = `/api/images/${img.uid}/png`;
                } else {
                    this.els.image.removeAttribute('src');   // never a stale frame
                }
                this.els.image.classList.remove('transitioning');
            }, 150);
        }

        // Update info
        if (this.els.title) this.els.title.textContent = img.data_type || 'Image';
        if (this.els.position) this.els.position.textContent = `${index + 1} of ${this.imageList.length}`;
        if (this.els.infoType) this.els.infoType.textContent = img.data_type || '-';
        if (this.els.infoEmbryo) this.els.infoEmbryo.textContent = img.metadata?.embryo_id || '-';
        if (this.els.infoShape) this.els.infoShape.textContent = img.shape ? img.shape.join(' x ') : '-';
        if (this.els.infoTime) this.els.infoTime.textContent = img.timestamp ? new Date(img.timestamp).toLocaleTimeString() : '-';

        // Update nav button states
        if (this.els.prevBtn) this.els.prevBtn.disabled = index === 0;
        if (this.els.nextBtn) this.els.nextBtn.disabled = index === this.imageList.length - 1;
    },

    renderThumbnails() {
        if (!this.els.thumbnails) return;

        this.els.thumbnails.innerHTML = '';

        // Show subset of thumbnails centered on current
        const visible = 9;
        const half = Math.floor(visible / 2);
        let start = Math.max(0, this.currentIndex - half);
        let end = Math.min(this.imageList.length, start + visible);

        if (end - start < visible) {
            start = Math.max(0, end - visible);
        }

        for (let i = start; i < end; i++) {
            const img = this.imageList[i];
            const thumb = document.createElement('div');
            thumb.className = `lightbox-thumb ${i === this.currentIndex ? 'active' : ''}`;
            thumb.dataset.index = i;

            if (img.base64_png) {
                thumb.innerHTML = `<img src="data:image/png;base64,${img.base64_png}" alt="${img.data_type || 'Image'}">`;
            }

            thumb.addEventListener('click', () => this.goTo(i));
            this.els.thumbnails.appendChild(thumb);
        }
    },

    updateThumbnailHighlight() {
        if (!this.els.thumbnails) return;

        const thumbs = this.els.thumbnails.querySelectorAll('.lightbox-thumb');
        let needsRerender = true;

        thumbs.forEach((thumb) => {
            const index = parseInt(thumb.dataset.index);
            const isActive = index === this.currentIndex;
            thumb.classList.toggle('active', isActive);
            if (isActive) needsRerender = false;
        });

        // Re-render if current is not visible
        if (needsRerender) {
            this.renderThumbnails();
        }
    },

    // Zoom delegation to ZoomPanController
    resetZoom() {
        this._zoomCtrl?.reset();
    }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => Lightbox.init());
} else {
    Lightbox.init();
}
