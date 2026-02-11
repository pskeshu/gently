/**
 * ZoomPanController - Shared zoom/pan logic for image viewers
 *
 * Used by both MainViewerZoom (main image) and Lightbox (fullscreen viewer).
 *
 * @param {Object} options
 * @param {number} options.minScale - Minimum zoom level (default: 0.5)
 * @param {number} options.maxScale - Maximum zoom level (default: 6)
 * @param {HTMLElement} options.target - Element to apply transform to
 * @param {HTMLElement} options.container - Element to listen for wheel/mouse events
 * @param {HTMLElement} [options.indicator] - Element to show zoom percentage
 * @param {function} [options.onZoomChange] - Callback when zoom changes (receives scale)
 */
class ZoomPanController {
    constructor({ minScale = 0.5, maxScale = 6, target, container, indicator, onZoomChange }) {
        this.minScale = minScale;
        this.maxScale = maxScale;
        this.target = target;
        this.container = container;
        this.indicator = indicator;
        this.onZoomChange = onZoomChange || null;

        this.zoom = { scale: 1, offsetX: 0, offsetY: 0, isDragging: false, startX: 0, startY: 0 };
        this._indicatorTimeout = null;

        // Bound handlers for cleanup
        this._onMouseMove = (e) => this._doPan(e);
        this._onMouseUp = () => this._endPan();
    }

    bind() {
        if (!this.container) return;

        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.handleZoomDelta(e.deltaY > 0 ? -0.15 : 0.15);
        }, { passive: false });

        this.container.addEventListener('dblclick', () => this.reset());
        this.container.addEventListener('mousedown', (e) => this._startPan(e));
        document.addEventListener('mousemove', this._onMouseMove);
        document.addEventListener('mouseup', this._onMouseUp);
    }

    unbind() {
        document.removeEventListener('mousemove', this._onMouseMove);
        document.removeEventListener('mouseup', this._onMouseUp);
    }

    handleZoomDelta(delta) {
        const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.zoom.scale + delta));
        if (newScale !== this.zoom.scale) {
            this.zoom.scale = newScale;
            this._apply();
            this._showIndicator();
            if (this.onZoomChange) this.onZoomChange(this.zoom.scale);
        }
    }

    reset() {
        this.zoom = { scale: 1, offsetX: 0, offsetY: 0, isDragging: false, startX: 0, startY: 0 };
        this._apply();
        if (this.onZoomChange) this.onZoomChange(this.zoom.scale);
    }

    get scale() {
        return this.zoom.scale;
    }

    _startPan(e) {
        if (this.zoom.scale <= 1) return;
        // Don't start pan if clicking on controls
        if (e.target.closest && e.target.closest('.zoom-controls, .ctrl-btn')) return;

        this.zoom.isDragging = true;
        this.zoom.startX = e.clientX - this.zoom.offsetX;
        this.zoom.startY = e.clientY - this.zoom.offsetY;
    }

    _doPan(e) {
        if (!this.zoom.isDragging) return;
        this.zoom.offsetX = e.clientX - this.zoom.startX;
        this.zoom.offsetY = e.clientY - this.zoom.startY;
        this._apply();
    }

    _endPan() {
        this.zoom.isDragging = false;
    }

    _apply() {
        if (this.target) {
            this.target.style.transform =
                `translate(${this.zoom.offsetX}px, ${this.zoom.offsetY}px) scale(${this.zoom.scale})`;
        }
    }

    _showIndicator() {
        if (this.indicator) {
            this.indicator.textContent = `${Math.round(this.zoom.scale * 100)}%`;
            this.indicator.classList.add('visible');

            clearTimeout(this._indicatorTimeout);
            this._indicatorTimeout = setTimeout(() => {
                this.indicator?.classList.remove('visible');
            }, 1200);
        }
    }
}
