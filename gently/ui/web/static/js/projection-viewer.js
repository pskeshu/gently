/**
 * Projection Viewer - Multi-view projection display and 3D volume viewer
 * Shows all projection types plus interactive 3D volume rendering
 */

const ProjectionViewer = {
    isOpen: false,
    currentEmbryoId: null,
    currentTimepoint: null,
    projections: [],
    selectedMethod: null,

    // 3D viewer state
    scene3d: null,
    camera3d: null,
    renderer3d: null,
    sliceGroup: null,
    volumeData: null,
    volumeShape: null,
    // Physical voxel size (dz, dy, dx) in microns, for isometric rendering.
    // Default matches gently.core.imaging.projection_three_view.
    voxelSizeUm: [1.0, 0.1625, 0.1625],
    savedRotation: { x: -0.5, y: 0.5 },
    savedZoom: 0.9,
    isDragging: false,
    prevMouse: { x: 0, y: 0 },
    animationId: null,
    contrast: 1.0,
    threshold: 30,

    init() {
        // Create modal if it doesn't exist
        if (!document.getElementById('projection-viewer-modal')) {
            this.createModal();
        }
        this.bindEvents();
    },

    createModal() {
        const modal = document.createElement('div');
        modal.id = 'projection-viewer-modal';
        modal.className = 'projection-viewer-modal';
        modal.innerHTML = `
            <div class="projection-viewer-container">
                <div class="projection-viewer-header">
                    <h3 class="projection-viewer-title">Volume Projections</h3>
                    <div class="projection-viewer-info">
                        <span id="pv-embryo-id"></span>
                        <span class="pv-separator">·</span>
                        <span id="pv-timepoint"></span>
                        <span class="pv-separator">·</span>
                        <span id="pv-shape"></span>
                    </div>
                    <button class="projection-viewer-close" id="pv-close" aria-label="Close">&times;</button>
                </div>
                <div class="projection-viewer-body">
                    <div class="projection-viewer-loading" id="pv-loading">
                        <div class="spinner"></div>
                        <span>Loading projections...</span>
                    </div>
                    <div class="projection-viewer-error" id="pv-error" style="display: none;">
                        <span id="pv-error-text"></span>
                    </div>
                    <div class="projection-viewer-content" id="pv-content" style="display: none;">
                        <!-- 3D Viewer Section -->
                        <div class="pv-3d-section">
                            <div class="pv-3d-header">
                                <span class="pv-3d-title">3D Volume Viewer</span>
                                <div class="pv-3d-controls">
                                    <label>Threshold:</label>
                                    <input type="range" id="pv-threshold" min="0" max="100" value="30">
                                    <span id="pv-threshold-val">0.30</span>
                                    <label>Contrast:</label>
                                    <input type="range" id="pv-contrast" min="50" max="300" value="100">
                                    <span id="pv-contrast-val">1.0</span>
                                </div>
                            </div>
                            <div class="pv-3d-info">
                                Drag: rotate X/Y | Shift+Drag: rotate Z | Scroll: zoom | Double-click: reset
                            </div>
                            <div class="pv-3d-container" id="pv-3d-container"></div>
                        </div>
                        <!-- 2D Projections Grid -->
                        <div class="projection-viewer-grid" id="pv-grid"></div>
                    </div>
                </div>
                <div class="projection-viewer-footer">
                    <div class="projection-method-tabs" id="pv-tabs"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    },

    bindEvents() {
        const modal = document.getElementById('projection-viewer-modal');
        const closeBtn = document.getElementById('pv-close');

        // Close on button click
        closeBtn?.addEventListener('click', () => this.close());

        // Close on backdrop click
        modal?.addEventListener('click', (e) => {
            if (e.target === modal) this.close();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });
    },

    async open(embryoId, timepoint) {
        this.currentEmbryoId = embryoId;
        this.currentTimepoint = timepoint;
        this.projections = [];
        this.selectedMethod = null;
        this.isOpen = true;

        const modal = document.getElementById('projection-viewer-modal');
        const loading = document.getElementById('pv-loading');
        const error = document.getElementById('pv-error');
        const content = document.getElementById('pv-content');
        const grid = document.getElementById('pv-grid');

        // Show modal
        modal.classList.add('active');
        loading.style.display = 'flex';
        error.style.display = 'none';
        content.style.display = 'none';
        grid.innerHTML = '';

        // Update header
        document.getElementById('pv-embryo-id').textContent = embryoId;
        document.getElementById('pv-timepoint').textContent = `T${timepoint}`;
        document.getElementById('pv-shape').textContent = '';

        try {
            // Load projections and volume data in parallel
            const [projResponse, volResponse] = await Promise.all([
                fetch(`/api/projections/${encodeURIComponent(embryoId)}/${timepoint}`),
                fetch(`/api/volume-raw/${encodeURIComponent(embryoId)}/${timepoint}`)
            ]);

            if (!projResponse.ok) {
                const data = await projResponse.json().catch(() => ({}));
                throw new Error(data.detail || `HTTP ${projResponse.status}`);
            }

            const projData = await projResponse.json();
            this.projections = projData.projections || [];

            // Update shape info
            if (projData.volume_shape) {
                const [z, h, w] = projData.volume_shape;
                document.getElementById('pv-shape').textContent = `${w}×${h}×${z}`;
            }

            // Load volume data for 3D viewer
            if (volResponse.ok) {
                const volData = await volResponse.json();
                this.volumeShape = volData.shape;
                // Physical voxel size (dz, dy, dx) in microns. Falls back to
                // the diSPIM default if the backend didn't send one.
                if (Array.isArray(volData.voxel_size_um) && volData.voxel_size_um.length === 3) {
                    this.voxelSizeUm = volData.voxel_size_um;
                }

                // Decode base64 volume data
                const raw = atob(volData.data);
                this.volumeData = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) {
                    this.volumeData[i] = raw.charCodeAt(i);
                }
            }

            loading.style.display = 'none';
            content.style.display = 'block';

            this.renderProjections();
            this.renderTabs();

            // Initialize 3D viewer after content is visible
            if (this.volumeData) {
                setTimeout(() => this.init3DViewer(), 100);
            }

        } catch (err) {
            loading.style.display = 'none';
            error.style.display = 'block';
            document.getElementById('pv-error-text').textContent = `Failed to load projections: ${err.message}`;
        }
    },

    renderProjections() {
        const grid = document.getElementById('pv-grid');

        if (this.projections.length === 0) {
            grid.innerHTML = '<div class="pv-empty">No projections available</div>';
            return;
        }

        // Method display names
        const methodNames = {
            'three_view': 'Three View',
            'dual_view': 'Dual View',
            'depth_colored': 'Depth Colored',
            'multi_slice': 'Multi Slice',
            'spin_3d': '3D Spin'
        };

        // If a specific method is selected, show only that
        const toShow = this.selectedMethod
            ? this.projections.filter(p => p.method === this.selectedMethod)
            : this.projections;

        grid.innerHTML = toShow.map(proj => `
            <div class="projection-card ${this.selectedMethod === proj.method ? 'selected' : ''}"
                 data-method="${proj.method}"
                 onclick="ProjectionViewer.selectProjection('${proj.method}')">
                <div class="projection-card-header">
                    <span class="projection-method-name">${methodNames[proj.method] || proj.method}</span>
                </div>
                <div class="projection-card-image">
                    <img src="data:image/png;base64,${proj.data}" alt="${proj.method}" />
                </div>
                <div class="projection-card-desc">${proj.description || ''}</div>
            </div>
        `).join('');
    },

    renderTabs() {
        const tabs = document.getElementById('pv-tabs');

        const methodNames = {
            'three_view': 'Three View',
            'dual_view': 'Dual View',
            'depth_colored': 'Depth',
            'multi_slice': 'Slices',
            'spin_3d': '3D Spin'
        };

        const has3DViewer = this.volumeData !== null;

        tabs.innerHTML = `
            <button class="pv-tab ${!this.selectedMethod ? 'active' : ''}"
                    onclick="ProjectionViewer.selectMethod(null)">All</button>
            ${has3DViewer ? `
                <button class="pv-tab ${this.selectedMethod === '3d_viewer' ? 'active' : ''}"
                        onclick="ProjectionViewer.selectMethod('3d_viewer')">3D Viewer</button>
            ` : ''}
            ${this.projections.map(proj => `
                <button class="pv-tab ${this.selectedMethod === proj.method ? 'active' : ''}"
                        onclick="ProjectionViewer.selectMethod('${proj.method}')">
                    ${methodNames[proj.method] || proj.method}
                </button>
            `).join('')}
        `;
    },

    updateViewerVisibility() {
        const viewer3d = document.querySelector('.pv-3d-section');
        const grid = document.getElementById('pv-grid');

        if (!viewer3d || !grid) return;

        if (this.selectedMethod === '3d_viewer') {
            viewer3d.style.display = 'block';
            grid.style.display = 'none';
        } else if (this.selectedMethod && this.selectedMethod !== '3d_viewer') {
            viewer3d.style.display = 'none';
            grid.style.display = 'grid';
        } else {
            // Show all
            viewer3d.style.display = 'block';
            grid.style.display = 'grid';
        }
    },

    selectMethod(method) {
        this.selectedMethod = method;
        this.renderProjections();
        this.renderTabs();
        this.updateViewerVisibility();
    },

    selectProjection(method) {
        // Toggle selection
        if (this.selectedMethod === method) {
            this.selectedMethod = null;
        } else {
            this.selectedMethod = method;
        }
        this.renderProjections();
        this.renderTabs();
        this.updateViewerVisibility();
    },

    // 3D Viewer Methods
    init3DViewer() {
        const container = document.getElementById('pv-3d-container');
        if (!container || !this.volumeData || !this.volumeShape) return;

        // Clean up previous viewer
        this.cleanup3DViewer();

        const w = container.clientWidth || 500;
        const h = 400;

        this.scene3d = new THREE.Scene();
        this.camera3d = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
        this.camera3d.position.z = this.savedZoom;

        this.renderer3d = new THREE.WebGLRenderer({ antialias: true });
        this.renderer3d.setSize(w, h);
        this.renderer3d.setClearColor(0x000000);
        container.innerHTML = '';
        container.appendChild(this.renderer3d.domElement);

        this.sliceGroup = new THREE.Group();
        this.sliceGroup.rotation.x = this.savedRotation.x;
        this.sliceGroup.rotation.y = this.savedRotation.y;
        this.sliceGroup.rotation.z = 0;
        this.sliceGroup.scale.y = -1;  // Flip Y to match orientation
        this.scene3d.add(this.sliceGroup);

        this.threshold = 30;
        this.contrast = 1.0;
        this.buildSlices(32);

        // Threshold control
        const threshSlider = document.getElementById('pv-threshold');
        const threshDisplay = document.getElementById('pv-threshold-val');

        threshSlider.addEventListener('input', (e) => {
            this.threshold = parseInt(e.target.value);
            this.buildSlices(32);
            threshDisplay.textContent = (this.threshold / 100).toFixed(2);
        });

        // Contrast control
        const contrastSlider = document.getElementById('pv-contrast');
        const contrastDisplay = document.getElementById('pv-contrast-val');

        contrastSlider.addEventListener('input', (e) => {
            this.contrast = parseInt(e.target.value) / 100;
            this.buildSlices(32);
            contrastDisplay.textContent = this.contrast.toFixed(1);
        });

        // Mouse controls
        this.renderer3d.domElement.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.prevMouse = { x: e.clientX, y: e.clientY };
        });

        this.renderer3d.domElement.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            if (e.shiftKey) {
                this.sliceGroup.rotation.z += (e.clientX - this.prevMouse.x) * 0.01;
            } else {
                this.sliceGroup.rotation.y += (e.clientX - this.prevMouse.x) * 0.01;
                this.sliceGroup.rotation.x += (e.clientY - this.prevMouse.y) * 0.01;
            }
            this.savedRotation.x = this.sliceGroup.rotation.x;
            this.savedRotation.y = this.sliceGroup.rotation.y;
            this.prevMouse = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener('mouseup', () => this.isDragging = false);

        this.renderer3d.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera3d.position.z = Math.max(0.5, Math.min(5, this.camera3d.position.z + e.deltaY * 0.002));
            this.savedZoom = this.camera3d.position.z;
        });

        // Double-click to reset
        this.renderer3d.domElement.addEventListener('dblclick', () => {
            this.sliceGroup.rotation.x = -0.5;
            this.sliceGroup.rotation.y = 0.5;
            this.sliceGroup.rotation.z = 0;
            this.camera3d.position.z = 0.9;
            this.savedRotation = { x: -0.5, y: 0.5 };
            this.savedZoom = 0.9;
        });

        // Animation loop
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            this.renderer3d.render(this.scene3d, this.camera3d);
        };
        animate();
    },

    createSliceTexture(zIndex, threshold, contrast) {
        const [zd, h, w] = this.volumeShape;
        const sliceSize = w * h;
        const offset = zIndex * sliceSize;
        const rgba = new Uint8Array(w * h * 4);

        for (let i = 0; i < sliceSize; i++) {
            let val = this.volumeData[offset + i];
            if (val > threshold) {
                // Apply contrast adjustment around midpoint
                val = Math.round(((val - 128) * contrast) + 128);
                val = Math.max(0, Math.min(255, val));
                rgba[i * 4] = val;
                rgba[i * 4 + 1] = val;
                rgba[i * 4 + 2] = val;
                rgba[i * 4 + 3] = Math.min(255, (this.volumeData[offset + i] - threshold) * 2);
            } else {
                rgba[i * 4 + 3] = 0;
            }
        }

        const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
        tex.needsUpdate = true;
        return tex;
    },

    buildSlices(numSlices) {
        if (!this.volumeShape || !this.sliceGroup) return;
        const [zd, h, w] = this.volumeShape;

        // Clear old slices
        while (this.sliceGroup.children.length > 0) {
            const c = this.sliceGroup.children[0];
            c.geometry.dispose();
            c.material.dispose();
            this.sliceGroup.remove(c);
        }

        // Isometric scaling: derive the plane dimensions and Z spread from
        // the physical voxel size, matching the math in
        // gently.core.imaging.projection_three_view (z_scale = dz / dx).
        // With the default diSPIM voxel (1.0, 0.1625, 0.1625) each Z voxel
        // covers ~6.15x more physical distance than an XY pixel, so without
        // this correction volumes look squished in Z.
        const [dz, dy, dx] = this.voxelSizeUm;
        // Physical extents in microns
        const xExtentUm = w * dx;
        const yExtentUm = h * dy;
        const zExtentUm = zd * dz;
        // Normalize so the largest extent becomes 1 three.js unit.
        const maxExtentUm = Math.max(xExtentUm, yExtentUm, zExtentUm);
        const planeW = xExtentUm / maxExtentUm;
        const planeH = yExtentUm / maxExtentUm;
        const zScale = zExtentUm / maxExtentUm;

        for (let i = 0; i < numSlices; i++) {
            const zIndex = Math.floor(i * zd / numSlices);
            const zPos = (i / numSlices - 0.5) * zScale;
            const tex = this.createSliceTexture(zIndex, this.threshold, this.contrast);
            const mat = new THREE.MeshBasicMaterial({
                map: tex,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false
            });
            const geo = new THREE.PlaneGeometry(planeW, planeH);
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.z = zPos;
            this.sliceGroup.add(mesh);
        }
    },

    cleanup3DViewer() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        if (this.sliceGroup) {
            while (this.sliceGroup.children.length > 0) {
                const c = this.sliceGroup.children[0];
                c.geometry?.dispose();
                c.material?.dispose();
                this.sliceGroup.remove(c);
            }
        }
        if (this.renderer3d) {
            this.renderer3d.dispose();
            this.renderer3d = null;
        }
        this.scene3d = null;
        this.camera3d = null;
        this.sliceGroup = null;
    },

    close() {
        this.isOpen = false;
        this.cleanup3DViewer();
        this.volumeData = null;
        this.volumeShape = null;
        const modal = document.getElementById('projection-viewer-modal');
        modal?.classList.remove('active');
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    ProjectionViewer.init();
});
