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

        // Root group that holds all three axis-aligned slice stacks. Only
        // the stack most perpendicular to the current view direction is
        // rendered each frame (see _updateStackVisibility), which eliminates
        // the gaps you see with a single-axis stack when viewed edge-on.
        this.sliceGroup = new THREE.Group();
        this.sliceGroup.rotation.x = this.savedRotation.x;
        this.sliceGroup.rotation.y = this.savedRotation.y;
        this.sliceGroup.rotation.z = 0;
        this.sliceGroup.scale.y = -1;  // Flip Y to match image orientation
        this.scene3d.add(this.sliceGroup);

        this.zStack = new THREE.Group();  // XY planes at varying Z
        this.yStack = new THREE.Group();  // XZ planes at varying Y
        this.xStack = new THREE.Group();  // YZ planes at varying X
        this.sliceGroup.add(this.zStack);
        this.sliceGroup.add(this.yStack);
        this.sliceGroup.add(this.xStack);

        this.threshold = 30;
        this.contrast = 1.0;
        this.buildAllStacks();

        // Threshold control
        const threshSlider = document.getElementById('pv-threshold');
        const threshDisplay = document.getElementById('pv-threshold-val');

        threshSlider.addEventListener('input', (e) => {
            this.threshold = parseInt(e.target.value);
            this.buildAllStacks();
            threshDisplay.textContent = (this.threshold / 100).toFixed(2);
        });

        // Contrast control
        const contrastSlider = document.getElementById('pv-contrast');
        const contrastDisplay = document.getElementById('pv-contrast-val');

        contrastSlider.addEventListener('input', (e) => {
            this.contrast = parseInt(e.target.value) / 100;
            this.buildAllStacks();
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

        // Animation loop with view-aligned stack selection.
        this._viewDir = new THREE.Vector3();
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            this._updateStackVisibility();
            this.renderer3d.render(this.scene3d, this.camera3d);
        };
        animate();
    },

    // Pick the slice stack whose slice normals are most aligned with the
    // camera view direction, and hide the other two. This is the classic
    // "object-aligned slices" technique for gap-free volume rendering with
    // plain MeshBasicMaterial and no custom shaders.
    _updateStackVisibility() {
        if (!this.sliceGroup || !this.camera3d) return;
        if (!this.zStack || !this.yStack || !this.xStack) return;

        // Camera forward direction in world space
        this.camera3d.getWorldDirection(this._viewDir);

        // Transform view direction into sliceGroup's local space by inverting
        // the group rotation (the sliceGroup is what the user rotates).
        const inv = new THREE.Matrix4().copy(this.sliceGroup.matrixWorld).invert();
        const localDir = this._viewDir.clone().transformDirection(inv);

        const ax = Math.abs(localDir.x);
        const ay = Math.abs(localDir.y);
        const az = Math.abs(localDir.z);

        // Stack normal that is MOST aligned with view direction = stack we
        // should show. Z-stack slices are XY planes (normal = z), etc.
        this.xStack.visible = ax > ay && ax > az;
        this.yStack.visible = ay > ax && ay > az;
        this.zStack.visible = az >= ax && az >= ay;
    },

    // ======== SLICE EXTRACTION HELPERS ========
    // Each helper returns an RGBA Uint8Array for a 2D slice at the given
    // index along one axis. Values below threshold are alpha=0, above get
    // contrast-adjusted intensity + alpha. These are the expensive CPU-side
    // part of the 3D build.

    _applyContrast(val) {
        return Math.max(0, Math.min(255, Math.round(((val - 128) * this.contrast) + 128)));
    },

    createZSliceTexture(zIndex) {
        // XY plane at constant Z. Texture is (w x h), row = y.
        const [zd, h, w] = this.volumeShape;
        const offset = zIndex * w * h;
        const rgba = new Uint8Array(w * h * 4);
        const th = this.threshold;
        for (let i = 0; i < w * h; i++) {
            const raw = this.volumeData[offset + i];
            if (raw > th) {
                const v = this._applyContrast(raw);
                const di = i * 4;
                rgba[di] = v; rgba[di + 1] = v; rgba[di + 2] = v;
                rgba[di + 3] = Math.min(255, (raw - th) * 2);
            }
        }
        const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
        tex.needsUpdate = true;
        return tex;
    },

    createYSliceTexture(yIndex) {
        // XZ plane at constant Y. Texture is (w x zd), col = x, row = z.
        const [zd, h, w] = this.volumeShape;
        const rgba = new Uint8Array(w * zd * 4);
        const th = this.threshold;
        for (let z = 0; z < zd; z++) {
            const srcRow = z * w * h + yIndex * w;
            const dstRow = z * w;
            for (let x = 0; x < w; x++) {
                const raw = this.volumeData[srcRow + x];
                if (raw > th) {
                    const v = this._applyContrast(raw);
                    const di = (dstRow + x) * 4;
                    rgba[di] = v; rgba[di + 1] = v; rgba[di + 2] = v;
                    rgba[di + 3] = Math.min(255, (raw - th) * 2);
                }
            }
        }
        const tex = new THREE.DataTexture(rgba, w, zd, THREE.RGBAFormat);
        tex.needsUpdate = true;
        return tex;
    },

    createXSliceTexture(xIndex) {
        // YZ plane at constant X. After the X-stack plane is rotated by
        // PI/2 around Y, its local U points along Z and its local V points
        // along Y, so the DataTexture layout needs rows=Y (height), cols=Z
        // (width). Without this the slice appears transposed relative to
        // the other two stacks and you get a visible pop when view-aligned
        // stack selection swaps to the X-stack.
        const [zd, h, w] = this.volumeShape;
        const rgba = new Uint8Array(h * zd * 4);
        const th = this.threshold;
        for (let y = 0; y < h; y++) {
            const dstRow = y * zd;  // stride = zd (cols per row)
            for (let z = 0; z < zd; z++) {
                const raw = this.volumeData[z * w * h + y * w + xIndex];
                if (raw > th) {
                    const v = this._applyContrast(raw);
                    const di = (dstRow + z) * 4;
                    rgba[di] = v; rgba[di + 1] = v; rgba[di + 2] = v;
                    rgba[di + 3] = Math.min(255, (raw - th) * 2);
                }
            }
        }
        // width = zd (U/Z), height = h (V/Y)
        const tex = new THREE.DataTexture(rgba, zd, h, THREE.RGBAFormat);
        tex.needsUpdate = true;
        return tex;
    },

    _disposeStack(group) {
        if (!group) return;
        while (group.children.length > 0) {
            const c = group.children[0];
            c.geometry?.dispose();
            if (c.material?.map) c.material.map.dispose();
            c.material?.dispose();
            group.remove(c);
        }
    },

    buildAllStacks() {
        if (!this.volumeShape || !this.zStack || !this.yStack || !this.xStack) return;
        const [zd, h, w] = this.volumeShape;

        // Dispose previously-built meshes/textures so threshold/contrast
        // updates don't leak GPU memory.
        this._disposeStack(this.zStack);
        this._disposeStack(this.yStack);
        this._disposeStack(this.xStack);

        // Physical extents in microns, normalized so the largest axis
        // becomes 1 three.js unit. Matches the voxel_size_um math in
        // gently.core.imaging.projection_three_view.
        const [dz, dy, dx] = this.voxelSizeUm;
        const xExtentUm = w * dx;
        const yExtentUm = h * dy;
        const zExtentUm = zd * dz;
        const maxExtentUm = Math.max(xExtentUm, yExtentUm, zExtentUm);
        const planeW = xExtentUm / maxExtentUm;  // size of volume along X
        const planeH = yExtentUm / maxExtentUm;  // size of volume along Y
        const zScale = zExtentUm / maxExtentUm;  // size of volume along Z

        // Cap slice count per axis to limit CPU extraction time and VRAM.
        // At this cap we get gapless rendering from any angle on typical
        // diSPIM volumes without noticeable lag on threshold changes.
        const MAX_SLICES_PER_AXIS = 96;
        const numZ = Math.min(zd, MAX_SLICES_PER_AXIS);
        const numY = Math.min(h, MAX_SLICES_PER_AXIS);
        const numX = Math.min(w, MAX_SLICES_PER_AXIS);

        const makeMat = (tex) => new THREE.MeshBasicMaterial({
            map: tex,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        // Z-stack: XY planes at varying Z positions.
        for (let i = 0; i < numZ; i++) {
            const zIndex = Math.floor(i * zd / numZ);
            const zPos = (zIndex / (zd - 1) - 0.5) * zScale;
            const tex = this.createZSliceTexture(zIndex);
            const geo = new THREE.PlaneGeometry(planeW, planeH);
            const mesh = new THREE.Mesh(geo, makeMat(tex));
            mesh.position.z = zPos;
            this.zStack.add(mesh);
        }

        // Y-stack: XZ planes at varying Y positions. A PlaneGeometry lies
        // in the XY plane by default; rotating PI/2 around X reorients it
        // to the XZ plane so its normal points along Y.
        for (let i = 0; i < numY; i++) {
            const yIndex = Math.floor(i * h / numY);
            const yPos = (yIndex / (h - 1) - 0.5) * planeH;
            const tex = this.createYSliceTexture(yIndex);
            const geo = new THREE.PlaneGeometry(planeW, zScale);
            const mesh = new THREE.Mesh(geo, makeMat(tex));
            mesh.rotation.x = Math.PI / 2;
            mesh.position.y = yPos;
            this.yStack.add(mesh);
        }

        // X-stack: YZ planes at varying X positions. Rotate PI/2 around Y
        // so the plane normal points along X.
        for (let i = 0; i < numX; i++) {
            const xIndex = Math.floor(i * w / numX);
            const xPos = (xIndex / (w - 1) - 0.5) * planeW;
            const tex = this.createXSliceTexture(xIndex);
            const geo = new THREE.PlaneGeometry(zScale, planeH);
            const mesh = new THREE.Mesh(geo, makeMat(tex));
            mesh.rotation.y = Math.PI / 2;
            mesh.position.x = xPos;
            this.xStack.add(mesh);
        }
    },

    cleanup3DViewer() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        // Dispose each axis stack's meshes + textures
        this._disposeStack(this.zStack);
        this._disposeStack(this.yStack);
        this._disposeStack(this.xStack);
        if (this.sliceGroup) {
            // sliceGroup itself may still hold the three (now empty) stacks
            while (this.sliceGroup.children.length > 0) {
                this.sliceGroup.remove(this.sliceGroup.children[0]);
            }
        }
        if (this.renderer3d) {
            this.renderer3d.dispose();
            this.renderer3d = null;
        }
        this.scene3d = null;
        this.camera3d = null;
        this.sliceGroup = null;
        this.zStack = null;
        this.yStack = null;
        this.xStack = null;
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
