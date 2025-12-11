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
    zoom: { scale: 1, offsetX: 0, offsetY: 0, isDragging: false, startX: 0, startY: 0 },
    zoomTimeout: null,

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
                    this.handleZoomDelta(0.25);
                    break;
                case '-':
                    e.preventDefault();
                    this.handleZoomDelta(-0.25);
                    break;
            }
        });

        // Zoom controls
        this.els.imageWrapper?.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.handleZoom(e);
        }, { passive: false });

        this.els.imageWrapper?.addEventListener('dblclick', () => this.resetZoom());

        // Pan controls
        this.els.imageWrapper?.addEventListener('mousedown', (e) => this.startPan(e));
        document.addEventListener('mousemove', (e) => this.doPan(e));
        document.addEventListener('mouseup', () => this.endPan());

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

    close() {
        this.isOpen = false;
        this.els.overlay?.classList.remove('active');
        document.body.style.overflow = '';
        this.resetZoom();
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
        this.showImage(index);
        this.updateThumbnailHighlight();
    },

    showImage(index) {
        const img = this.imageList[index];
        if (!img) return;

        // Animate image transition
        if (this.els.image) {
            this.els.image.classList.add('transitioning');

            setTimeout(() => {
                if (img.base64_png) {
                    this.els.image.src = 'data:image/png;base64,' + img.base64_png;
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

    // Zoom methods
    handleZoom(e) {
        const delta = e.deltaY > 0 ? -0.15 : 0.15;
        this.handleZoomDelta(delta);
    },

    handleZoomDelta(delta) {
        const newScale = Math.max(0.5, Math.min(6, this.zoom.scale + delta));

        if (newScale !== this.zoom.scale) {
            this.zoom.scale = newScale;
            this.applyTransform();
            this.showZoomIndicator();
        }
    },

    showZoomIndicator() {
        if (this.els.zoomIndicator) {
            this.els.zoomIndicator.textContent = `${Math.round(this.zoom.scale * 100)}%`;
            this.els.zoomIndicator.classList.add('visible');

            clearTimeout(this.zoomTimeout);
            this.zoomTimeout = setTimeout(() => {
                this.els.zoomIndicator?.classList.remove('visible');
            }, 1200);
        }
    },

    startPan(e) {
        if (this.zoom.scale <= 1) return;

        this.zoom.isDragging = true;
        this.zoom.startX = e.clientX - this.zoom.offsetX;
        this.zoom.startY = e.clientY - this.zoom.offsetY;

        if (this.els.imageWrapper) {
            this.els.imageWrapper.style.cursor = 'grabbing';
        }
    },

    doPan(e) {
        if (!this.zoom.isDragging) return;

        this.zoom.offsetX = e.clientX - this.zoom.startX;
        this.zoom.offsetY = e.clientY - this.zoom.startY;
        this.applyTransform();
    },

    endPan() {
        this.zoom.isDragging = false;

        if (this.els.imageWrapper) {
            this.els.imageWrapper.style.cursor = this.zoom.scale > 1 ? 'grab' : 'default';
        }
    },

    resetZoom() {
        this.zoom = { scale: 1, offsetX: 0, offsetY: 0, isDragging: false, startX: 0, startY: 0 };
        this.applyTransform();

        if (this.els.imageWrapper) {
            this.els.imageWrapper.style.cursor = 'default';
        }
    },

    applyTransform() {
        if (this.els.image) {
            this.els.image.style.transform =
                `translate(${this.zoom.offsetX}px, ${this.zoom.offsetY}px) scale(${this.zoom.scale})`;
        }
    }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => Lightbox.init());
} else {
    Lightbox.init();
}
