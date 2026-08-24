/**
 * Map view — embryo detection, marking, role assignment.
 *
 * Single spatial GUI: SAM-detected markers arrive pre-placed and editable;
 * user can add/remove markers, toggle each between Test (magenta) and
 * Calibration (cyan), and click "Re-detect" to recapture + re-run SAM.
 * Replaces the deprecated napari-based marker.
 *
 * WS message types:
 *   incoming  marking_image       — image + initial markers + default role
 *   outgoing  embryo_marked       — single new marker added (clicked)
 *   outgoing  marking_update      — full marker list (after edit/role-cycle)
 *   outgoing  marking_done        — finalize + commit roles
 *   outgoing  marking_redetect    — recapture + re-run SAM, replace markers
 */

const ROLE_CYCLE = ['test', 'calibration', 'unassigned'];
const ROLE_COLORS = {
    test: '#ff66cc',           // magenta — biological subject (precious)
    calibration: '#00cccc',    // cyan    — reference / staging anchor
    unassigned: '#888888',     // grey    — not yet classified
};
const ROLE_LABEL = {
    test: 'Test',
    calibration: 'Cal',
    unassigned: '—',
};

const MarkingManager = {
    // State
    active: false,
    markers: [],             // [{ number, pixelX, pixelY, role, source, embryo_id?, confidence?, timestamp }]
    imageWidth: 0,
    imageHeight: 0,
    sessionId: null,
    defaultRole: 'test',

    // Coordinate / overlay state
    stageXUm: 0,             // current stage position (um)
    stageYUm: 0,
    umPerPixel: 0.65,        // effective um per pixel on sample
    coverslip: null,         // {center_um: [x,y], size_mm: [w,h]} from /api/devices/coverslip

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

        this.canvas.addEventListener('click', (e) => this._onCanvasClick(e));

        this.img.addEventListener('load', () => {
            this._syncCanvasSize();
            this._redraw();
        });

        new ResizeObserver(() => {
            this._syncCanvasSize();
            this._redraw();
        }).observe(this.container);

        // Load coverslip outline once (small static metadata)
        fetch('/api/devices/coverslip')
            .then(r => r.ok ? r.json() : null)
            .then(d => { if (d && d.coverslip) this.coverslip = d.coverslip; })
            .catch(() => { /* coverslip overlay is optional */ });
    },

    // Called when server sends a marking_image message
    handleMarkingImage(data) {
        this.sessionId = data.session_id;
        this.imageWidth = data.width;
        this.imageHeight = data.height;
        this.defaultRole = data.default_role || 'test';
        this.stageXUm = data.stage_x_um != null ? data.stage_x_um : 0;
        this.stageYUm = data.stage_y_um != null ? data.stage_y_um : 0;
        this.umPerPixel = data.pixel_size_um || 0.65;

        // Auto-switch to the marking subtab so the user doesn't have to
        // hunt for it when a session starts.
        try { this.switchSubtab('marking'); } catch (_) { /* tabs not ready */ }

        // Hydrate any initial markers (e.g. SAM detections) — already
        // normalized server-side to {number, pixelX, pixelY, role, source, ...}.
        this.markers = (data.initial_markers || []).map(m => ({
            number: m.number,
            pixelX: m.pixelX,
            pixelY: m.pixelY,
            role: m.role || this.defaultRole,
            source: m.source || 'sam',
            embryo_id: m.embryo_id || null,
            confidence: m.confidence != null ? m.confidence : null,
            timestamp: m.timestamp || new Date().toISOString(),
        }));
        this.active = true;

        this.img.src = 'data:image/png;base64,' + data.image_b64;
        this.img.style.display = 'block';

        const placeholder = document.getElementById('marking-placeholder');
        const activeEl = document.getElementById('marking-active');
        if (placeholder) placeholder.style.display = 'none';
        if (activeEl) activeEl.style.display = 'flex';

        const instructions = document.getElementById('marking-instructions');
        if (instructions) {
            const n = this.markers.length;
            instructions.textContent = n > 0
                ? `${n} marker(s) loaded. Click to add, click a role chip to cycle Test/Calibration, press Done when finished.`
                : 'Click on each embryo center. Click a role chip to switch Test/Calibration. Press Done when finished.';
        }

        // Re-enable action buttons in case a previous session disabled them.
        document.querySelectorAll('.marking-actions .marking-action-btn').forEach(btn => btn.disabled = false);

        this._renderList();
    },

    _syncCanvasSize() {
        if (!this.img || !this.canvas || !this.imageWidth || !this.imageHeight) return;

        const containerRect = this.container.getBoundingClientRect();
        const imgAspect = this.imageWidth / this.imageHeight;
        const containerAspect = containerRect.width / containerRect.height;

        let renderW, renderH;
        if (imgAspect > containerAspect) {
            renderW = containerRect.width;
            renderH = containerRect.width / imgAspect;
        } else {
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

        const scaleX = this.imageWidth / this.canvas.width;
        const scaleY = this.imageHeight / this.canvas.height;
        const pixelX = canvasX * scaleX;
        const pixelY = canvasY * scaleY;

        const marker = {
            number: this.markers.length + 1,
            pixelX: Math.round(pixelX * 10) / 10,
            pixelY: Math.round(pixelY * 10) / 10,
            role: this.defaultRole,
            source: 'manual',
            embryo_id: null,
            confidence: null,
            timestamp: new Date().toISOString(),
        };
        this.markers.push(marker);

        this._send('embryo_marked', { marker });

        this._redraw();
        this._renderList();
    },

    _redraw() {
        if (!this.canvas) return;
        const ctx = this.canvas.getContext('2d');
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (!this.active) return;

        const scaleX = w / this.imageWidth;
        const scaleY = h / this.imageHeight;

        // Coverslip outline (drawn first so markers sit on top).
        // The coverslip is much larger than the FOV; we draw it relative to
        // the current stage position so the user sees where in the slide
        // they're looking.
        if (this.coverslip && this.umPerPixel > 0) {
            const csCx = (this.coverslip.center_um && this.coverslip.center_um[0]) || 0;
            const csCy = (this.coverslip.center_um && this.coverslip.center_um[1]) || 0;
            const csW = ((this.coverslip.size_mm && this.coverslip.size_mm[0]) || 0) * 1000;
            const csH = ((this.coverslip.size_mm && this.coverslip.size_mm[1]) || 0) * 1000;
            if (csW > 0 && csH > 0) {
                // Coverslip extents in stage µm
                const x0um = csCx - csW / 2;
                const y0um = csCy - csH / 2;
                const x1um = csCx + csW / 2;
                const y1um = csCy + csH / 2;
                // Map each corner to pixel coords (image center IS stage_*Um)
                const imgCx = this.imageWidth / 2;
                const imgCy = this.imageHeight / 2;
                const toPxX = (xum) => (imgCx + (xum - this.stageXUm) / this.umPerPixel) * scaleX;
                const toPxY = (yum) => (imgCy + (yum - this.stageYUm) / this.umPerPixel) * scaleY;
                const left = toPxX(x0um);
                const top = toPxY(y0um);
                const right = toPxX(x1um);
                const bottom = toPxY(y1um);
                ctx.save();
                ctx.strokeStyle = 'rgba(255, 220, 0, 0.55)';
                ctx.setLineDash([6, 6]);
                ctx.lineWidth = 1.5;
                ctx.strokeRect(left, top, right - left, bottom - top);
                ctx.setLineDash([]);
                ctx.restore();
            }
        }

        // Stage-center crosshair (current XY) — image center IS the current
        // stage position by construction. Subtle so it doesn't fight markers.
        {
            const cx = (this.imageWidth / 2) * scaleX;
            const cy = (this.imageHeight / 2) * scaleY;
            const r = Math.max(6, Math.min(w, h) * 0.008);
            ctx.save();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy);
            ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r);
            ctx.stroke();
            ctx.restore();
        }

        if (this.markers.length === 0) return;

        for (const m of this.markers) {
            const x = m.pixelX * scaleX;
            const y = m.pixelY * scaleY;
            const size = Math.max(10, Math.min(w, h) * 0.015);
            const color = ROLE_COLORS[m.role] || ROLE_COLORS.test;

            // Crosshair (role-colored)
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
            ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
            ctx.stroke();

            // Circle (role-colored)
            ctx.beginPath();
            ctx.arc(x, y, size * 2, 0, Math.PI * 2);
            ctx.stroke();

            // Number label above
            ctx.font = `bold ${Math.max(12, size)}px sans-serif`;
            ctx.fillStyle = color;
            ctx.textAlign = 'center';
            ctx.fillText(m.number, x, y - size * 2.5);

            // Role short label below
            ctx.font = `${Math.max(10, size * 0.85)}px sans-serif`;
            ctx.fillText(ROLE_LABEL[m.role] || m.role, x, y + size * 3);
        }
    },

    _renderList() {
        if (!this.listEl) return;

        if (this.markers.length === 0) {
            this.listEl.innerHTML = '<div class="marking-list-empty">No embryos marked yet</div>';
            this._updateCount(0);
            return;
        }

        this.listEl.innerHTML = this.markers.map(m => {
            const color = ROLE_COLORS[m.role] || ROLE_COLORS.test;
            const label = ROLE_LABEL[m.role] || m.role;
            const src = m.source ? `<span class="marking-source" title="${m.source}">${m.source}</span>` : '';
            return `<div class="marking-list-item">
                <span class="marking-number" style="color:${color}">${m.number}</span>
                <button class="marking-role-chip"
                        style="background:${color}26;color:${color};border:1px solid ${color}"
                        title="Click to cycle role"
                        onclick="MarkingManager.cycleRole(${m.number})">${label}</button>
                ${src}
                <span class="marking-coords">(${m.pixelX}, ${m.pixelY})</span>
                <button class="marking-remove-btn" onclick="MarkingManager.removeMarker(${m.number})" title="Remove">&times;</button>
            </div>`;
        }).join('');

        this._updateCount(this.markers.length);
    },

    _updateCount(n) {
        const countEl = document.getElementById('marking-count');
        if (countEl) {
            countEl.textContent = n;
            countEl.style.display = n > 0 ? '' : 'none';
        }
    },

    cycleRole(number) {
        if (!this.active) return;
        const m = this.markers.find(x => x.number === number);
        if (!m) return;

        const idx = ROLE_CYCLE.indexOf(m.role);
        const next = ROLE_CYCLE[(idx + 1) % ROLE_CYCLE.length];
        m.role = next;

        this._send('marking_update', { markers: this.markers });
        this._redraw();
        this._renderList();
    },

    removeMarker(number) {
        this.markers = this.markers.filter(m => m.number !== number);
        // Renumber so labels stay 1..N
        this.markers.forEach((m, i) => m.number = i + 1);

        this._send('marking_update', { markers: this.markers });

        this._redraw();
        this._renderList();
    },

    clearAll() {
        if (!this.active) return;
        if (this.markers.length > 0 && !confirm('Clear all marked embryos?')) return;

        this.markers = [];
        this._send('marking_update', { markers: [] });

        this._redraw();
        this._renderList();
    },

    redetect() {
        if (!this.active) return;
        if (this.markers.length > 0 && !confirm('Recapture image and re-run SAM detection? Current markers will be replaced.')) return;

        this._send('marking_redetect', {});

        const instructions = document.getElementById('marking-instructions');
        if (instructions) {
            instructions.textContent = 'Recapturing and re-running detection…';
        }
    },

    done() {
        if (!this.active) return;

        if (this.markers.length === 0) {
            if (!confirm('No embryos marked. Finish anyway?')) return;
        }

        this._send('marking_done', { markers: this.markers });

        this.active = false;
        const counts = this.markers.reduce((acc, m) => {
            acc[m.role] = (acc[m.role] || 0) + 1;
            return acc;
        }, {});
        const summary = Object.entries(counts)
            .map(([r, n]) => `${n} ${r}`)
            .join(', ');
        const instructions = document.getElementById('marking-instructions');
        if (instructions) {
            instructions.textContent = `Marking complete — ${this.markers.length} embryo(s)${summary ? ': ' + summary : ''}.`;
        }

        document.querySelectorAll('.marking-actions .marking-action-btn').forEach(btn => btn.disabled = true);

        // Auto-switch back to monitoring after the user sees the
        // "marking complete" confirmation. Without this the marker
        // window stays put even after the agent has started a
        // timelapse, and the user has to manually switch tabs.
        setTimeout(() => {
            try { this.switchSubtab('monitoring'); } catch (_) { /* tabs may be gone */ }
            // Reset placeholder/active visibility so the next session
            // starts fresh on this tab.
            const placeholder = document.getElementById('marking-placeholder');
            const activeEl = document.getElementById('marking-active');
            if (placeholder) placeholder.style.display = '';
            if (activeEl) activeEl.style.display = 'none';
            this.markers = [];
            this._updateCount(0);
            this._redraw();
            this._renderList();
        }, 1500);
    },

    _send(type, payload) {
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
        state.ws.send(JSON.stringify({
            type,
            session_id: this.sessionId,
            ...payload,
        }));
    },

    // Switch between monitoring and marking subtabs
    switchSubtab(subtab) {
        document.querySelectorAll('.embryos-subtab').forEach(t => t.classList.remove('active'));
        const tab = document.querySelector(`.embryos-subtab[data-subtab="${subtab}"]`);
        if (tab) tab.classList.add('active');

        const monitoring = document.getElementById('embryos-monitoring');
        const marking = document.getElementById('embryos-marking');
        if (monitoring) monitoring.style.display = subtab === 'monitoring' ? '' : 'none';
        if (marking) marking.style.display = subtab === 'marking' ? '' : 'none';
    }
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => MarkingManager.init());
} else {
    MarkingManager.init();
}
