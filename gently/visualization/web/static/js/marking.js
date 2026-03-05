/**
 * Embryo Marking - Web-based interactive embryo position marking
 *
 * Replaces napari-based marking with browser-based click-to-mark workflow.
 * User clicks on bottom camera overview image to place numbered markers,
 * positions are sent back to server via WebSocket.
 */

const MarkingManager = {
    // State
    active: false,           // Marking session in progress
    markers: [],             // Array of { number, pixelX, pixelY, timestamp }
    imageWidth: 0,
    imageHeight: 0,
    sessionId: null,         // Server-assigned marking session ID

    // DOM refs (set in init)
    canvas: null,
    img: null,
    container: null,
    listEl: null,

    init() {
        this.canvas = document.getElementById('marking-canvas');
        this.img = document.getElementById('marking-image');
        this.container = document.getElementById('marking-image-container');
        this.listEl = document.getElementById('marking-list');

        if (!this.canvas || !this.img) return;

        // Click to place marker
        this.canvas.addEventListener('click', (e) => this._onCanvasClick(e));

        // Redraw overlay when image loads
        this.img.addEventListener('load', () => {
            this._syncCanvasSize();
            this._redraw();
        });

        // Handle resize
        new ResizeObserver(() => {
            this._syncCanvasSize();
            this._redraw();
        }).observe(this.container);

        // Listen for marking image from server
        this._subscribeToEvents();
    },

    _subscribeToEvents() {
        // Server pushes marking image via WebSocket
        // Handled in websocket.js handleMessage
    },

    // Called when server sends a marking_image message
    handleMarkingImage(data) {
        this.sessionId = data.session_id;
        this.imageWidth = data.width;
        this.imageHeight = data.height;
        this.markers = [];
        this.active = true;

        // Display the image
        this.img.src = 'data:image/png;base64,' + data.image_b64;
        this.img.style.display = 'block';

        // Show the marking UI
        document.getElementById('marking-placeholder').style.display = 'none';
        document.getElementById('marking-active').style.display = 'flex';

        // Update instructions
        document.getElementById('marking-instructions').textContent =
            'Click on each embryo center. Press Done when finished.';

        this._renderList();
    },

    _syncCanvasSize() {
        if (!this.img || !this.canvas || !this.imageWidth || !this.imageHeight) return;

        // The img uses object-fit:contain, so compute the actual rendered area
        const containerRect = this.container.getBoundingClientRect();
        const imgAspect = this.imageWidth / this.imageHeight;
        const containerAspect = containerRect.width / containerRect.height;

        let renderW, renderH;
        if (imgAspect > containerAspect) {
            // Image wider than container — limited by width
            renderW = containerRect.width;
            renderH = containerRect.width / imgAspect;
        } else {
            // Image taller — limited by height
            renderH = containerRect.height;
            renderW = containerRect.height * imgAspect;
        }

        this.canvas.width = renderW;
        this.canvas.height = renderH;
        this.canvas.style.width = renderW + 'px';
        this.canvas.style.height = renderH + 'px';
    },

    _onCanvasClick(e) {
        if (!this.active) return;

        const rect = this.canvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        // Convert to image pixel coordinates
        const scaleX = this.imageWidth / this.canvas.width;
        const scaleY = this.imageHeight / this.canvas.height;
        const pixelX = canvasX * scaleX;
        const pixelY = canvasY * scaleY;

        const marker = {
            number: this.markers.length + 1,
            pixelX: Math.round(pixelX * 10) / 10,
            pixelY: Math.round(pixelY * 10) / 10,
            timestamp: new Date().toISOString()
        };
        this.markers.push(marker);

        // Send to server
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'embryo_marked',
                session_id: this.sessionId,
                marker: marker
            }));
        }

        this._redraw();
        this._renderList();
    },

    _redraw() {
        if (!this.canvas) return;
        const ctx = this.canvas.getContext('2d');
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (!this.active || this.markers.length === 0) return;

        const scaleX = w / this.imageWidth;
        const scaleY = h / this.imageHeight;

        for (const m of this.markers) {
            const x = m.pixelX * scaleX;
            const y = m.pixelY * scaleY;
            const size = Math.max(10, Math.min(w, h) * 0.015);

            // Crosshair
            ctx.strokeStyle = '#00e5ff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
            ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
            ctx.stroke();

            // Circle
            ctx.beginPath();
            ctx.arc(x, y, size * 2, 0, Math.PI * 2);
            ctx.stroke();

            // Number label
            ctx.font = `bold ${Math.max(12, size)}px sans-serif`;
            ctx.fillStyle = '#00e5ff';
            ctx.textAlign = 'center';
            ctx.fillText(m.number, x, y - size * 2.5);
        }
    },

    _renderList() {
        if (!this.listEl) return;

        if (this.markers.length === 0) {
            this.listEl.innerHTML = '<div class="marking-list-empty">No embryos marked yet</div>';
            return;
        }

        this.listEl.innerHTML = this.markers.map(m =>
            `<div class="marking-list-item">
                <span class="marking-number">${m.number}</span>
                <span class="marking-coords">(${m.pixelX}, ${m.pixelY})</span>
                <button class="marking-remove-btn" onclick="MarkingManager.removeMarker(${m.number})" title="Remove">&times;</button>
            </div>`
        ).join('');

        // Update badge count
        const countEl = document.getElementById('marking-count');
        if (countEl) {
            countEl.textContent = this.markers.length;
            countEl.style.display = this.markers.length > 0 ? '' : 'none';
        }
    },

    removeMarker(number) {
        this.markers = this.markers.filter(m => m.number !== number);
        // Renumber
        this.markers.forEach((m, i) => m.number = i + 1);

        // Notify server
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'marking_update',
                session_id: this.sessionId,
                markers: this.markers
            }));
        }

        this._redraw();
        this._renderList();
    },

    clearAll() {
        if (!this.active) return;
        if (this.markers.length > 0 && !confirm('Clear all marked embryos?')) return;

        this.markers = [];

        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'marking_update',
                session_id: this.sessionId,
                markers: []
            }));
        }

        this._redraw();
        this._renderList();
    },

    done() {
        if (!this.active) return;

        if (this.markers.length === 0) {
            if (!confirm('No embryos marked. Finish anyway?')) return;
        }

        // Send completion to server
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'marking_done',
                session_id: this.sessionId,
                markers: this.markers
            }));
        }

        this.active = false;
        document.getElementById('marking-instructions').textContent =
            `Marking complete - ${this.markers.length} embryo(s) marked.`;

        // Disable buttons
        document.querySelectorAll('.marking-action-btn').forEach(btn => btn.disabled = true);
    },

    // Switch between monitoring and marking subtabs
    switchSubtab(subtab) {
        document.querySelectorAll('.embryos-subtab').forEach(t => t.classList.remove('active'));
        document.querySelector(`.embryos-subtab[data-subtab="${subtab}"]`).classList.add('active');

        document.getElementById('embryos-monitoring').style.display = subtab === 'monitoring' ? '' : 'none';
        document.getElementById('embryos-marking').style.display = subtab === 'marking' ? '' : 'none';
    }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => MarkingManager.init());
} else {
    MarkingManager.init();
}
