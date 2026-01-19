/**
 * Projections 3D Viewer for Gently Visualization
 * Displays 3D slice-based volume viewer in a modal
 */

const ProjectionsViewer = {
    scene: null,
    camera: null,
    renderer: null,
    sliceGroup: null,
    volumeData: null,
    volumeShape: null,
    savedRotation: { x: -0.5, y: 0.5 },
    savedZoom: 0.9,
    isInitialized: false,
    currentUid: null,
    isDragging: false,
    prevMouse: { x: 0, y: 0 },

    /**
     * Open the projections modal for a specific volume
     */
    async open(uid, embryoId, timepoint) {
        this.currentUid = uid;
        this.showModal();
        this.showLoading();

        try {
            // Fetch volume data from the API
            const res = await fetch(`/api/volume-data/${uid}`);
            if (!res.ok) {
                throw new Error(`Failed to load volume: ${res.statusText}`);
            }
            const data = await res.json();

            this.volumeShape = data.shape;

            // Decode base64 volume data
            const raw = atob(data.data);
            this.volumeData = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) {
                this.volumeData[i] = raw.charCodeAt(i);
            }

            // Update modal header
            this.updateHeader(embryoId, timepoint);

            // Initialize or update 3D viewer
            this.init3DViewer();
        } catch (err) {
            console.error('Failed to load projections:', err);
            this.showError(err.message);
        }
    },

    showModal() {
        let modal = document.getElementById('projections-modal');
        if (!modal) {
            this.createModal();
            modal = document.getElementById('projections-modal');
        }
        modal.classList.add('visible');
        document.body.classList.add('modal-open');
    },

    hideModal() {
        const modal = document.getElementById('projections-modal');
        if (modal) {
            modal.classList.remove('visible');
        }
        document.body.classList.remove('modal-open');
    },

    createModal() {
        const modal = document.createElement('div');
        modal.id = 'projections-modal';
        modal.className = 'projections-modal';
        modal.innerHTML = `
            <div class="projections-modal-backdrop" onclick="ProjectionsViewer.hideModal()"></div>
            <div class="projections-modal-content">
                <div class="projections-modal-header">
                    <h2 id="projections-title">3D Volume Viewer</h2>
                    <button class="projections-close-btn" onclick="ProjectionsViewer.hideModal()">&times;</button>
                </div>
                <div class="projections-modal-body">
                    <div id="projections-loading" class="projections-loading">
                        <div class="loading-spinner"></div>
                        <span>Loading volume data...</span>
                    </div>
                    <div id="projections-error" class="projections-error" style="display:none;"></div>
                    <div id="projections-viewer-wrapper" style="display:none;">
                        <div class="projections-controls">
                            <div class="control-group">
                                <label>Threshold:</label>
                                <input type="range" id="projections-threshold" min="0" max="100" value="30">
                                <span id="projections-thresh-display">0.30</span>
                            </div>
                            <div class="control-group">
                                <span id="projections-angle-display">angle_y: 0.50, angle_x: -0.50</span>
                            </div>
                            <div class="control-help">
                                Drag: rotate X/Y | Shift+Drag: rotate Z | Scroll: zoom | Double-click: reset
                            </div>
                        </div>
                        <div id="projections-3d-container"></div>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    },

    showLoading() {
        const loading = document.getElementById('projections-loading');
        const error = document.getElementById('projections-error');
        const wrapper = document.getElementById('projections-viewer-wrapper');
        if (loading) loading.style.display = 'flex';
        if (error) error.style.display = 'none';
        if (wrapper) wrapper.style.display = 'none';
    },

    showError(message) {
        const loading = document.getElementById('projections-loading');
        const error = document.getElementById('projections-error');
        const wrapper = document.getElementById('projections-viewer-wrapper');
        if (loading) loading.style.display = 'none';
        if (error) {
            error.style.display = 'block';
            error.innerHTML = `<p>Error: ${message}</p>`;
        }
        if (wrapper) wrapper.style.display = 'none';
    },

    updateHeader(embryoId, timepoint) {
        const title = document.getElementById('projections-title');
        if (title) {
            title.textContent = `3D Volume Viewer - ${embryoId || 'Unknown'} T${timepoint ?? '?'}`;
        }
    },

    init3DViewer() {
        const loading = document.getElementById('projections-loading');
        const wrapper = document.getElementById('projections-viewer-wrapper');
        if (loading) loading.style.display = 'none';
        if (wrapper) wrapper.style.display = 'block';

        const container = document.getElementById('projections-3d-container');
        if (!container) return;

        // Clear previous renderer if exists
        if (this.renderer) {
            container.innerHTML = '';
        }

        const w = container.clientWidth || 600;
        const h = 500;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
        this.camera.position.z = this.savedZoom;

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(w, h);
        this.renderer.setClearColor(0x000000);
        container.appendChild(this.renderer.domElement);

        this.sliceGroup = new THREE.Group();
        this.sliceGroup.rotation.x = this.savedRotation.x;
        this.sliceGroup.rotation.y = this.savedRotation.y;
        this.sliceGroup.rotation.z = 0;
        this.sliceGroup.scale.y = -1; // Flip Y to match projection orientation
        this.scene.add(this.sliceGroup);

        // Build initial slices
        this.buildSlices(32, 30);

        // Bind controls
        this.bindControls();

        // Start animation loop
        this.animate();

        this.isInitialized = true;
        console.log('3D projections viewer initialized');
    },

    createSliceTexture(zIndex, threshold) {
        const [zd, h, w] = this.volumeShape;
        const sliceSize = w * h;
        const offset = zIndex * sliceSize;
        const rgba = new Uint8Array(w * h * 4);

        for (let i = 0; i < sliceSize; i++) {
            const val = this.volumeData[offset + i];
            if (val > threshold) {
                rgba[i * 4] = val;
                rgba[i * 4 + 1] = val;
                rgba[i * 4 + 2] = val;
                rgba[i * 4 + 3] = Math.min(255, (val - threshold) * 2);
            } else {
                rgba[i * 4 + 3] = 0;
            }
        }

        const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
        tex.needsUpdate = true;
        return tex;
    },

    buildSlices(numSlices, threshold) {
        if (!this.volumeShape || !this.sliceGroup) return;
        const [zd, h, w] = this.volumeShape;

        // Clear old slices
        while (this.sliceGroup.children.length > 0) {
            const c = this.sliceGroup.children[0];
            c.geometry.dispose();
            c.material.dispose();
            this.sliceGroup.remove(c);
        }

        const aspect = w / h;
        const zScale = (zd / w) * 3; // Exaggerate Z for depth perception

        for (let i = 0; i < numSlices; i++) {
            const zIndex = Math.floor(i * zd / numSlices);
            const zPos = (i / numSlices - 0.5) * zScale;
            const tex = this.createSliceTexture(zIndex, threshold);
            const mat = new THREE.MeshBasicMaterial({
                map: tex,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false
            });
            const geo = new THREE.PlaneGeometry(1, 1 / aspect);
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.z = zPos;
            this.sliceGroup.add(mesh);
        }
    },

    bindControls() {
        const container = document.getElementById('projections-3d-container');
        const thresholdInput = document.getElementById('projections-threshold');

        // Threshold slider
        if (thresholdInput) {
            thresholdInput.addEventListener('input', (e) => {
                const threshVal = parseInt(e.target.value);
                this.buildSlices(32, threshVal);
                document.getElementById('projections-thresh-display').textContent = (threshVal / 100).toFixed(2);
            });
        }

        // Mouse controls on renderer
        const canvas = this.renderer.domElement;

        canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.prevMouse = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            if (e.shiftKey) {
                // Shift+drag: rotate around Z axis
                this.sliceGroup.rotation.z += (e.clientX - this.prevMouse.x) * 0.01;
            } else {
                // Normal drag: rotate around X and Y
                this.sliceGroup.rotation.y += (e.clientX - this.prevMouse.x) * 0.01;
                this.sliceGroup.rotation.x += (e.clientY - this.prevMouse.y) * 0.01;
            }
            this.savedRotation.x = this.sliceGroup.rotation.x;
            this.savedRotation.y = this.sliceGroup.rotation.y;

            document.getElementById('projections-angle-display').textContent =
                'angle_y: ' + this.sliceGroup.rotation.y.toFixed(2) +
                ', angle_x: ' + this.sliceGroup.rotation.x.toFixed(2);
            this.prevMouse = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera.position.z = Math.max(0.5, Math.min(5, this.camera.position.z + e.deltaY * 0.002));
            this.savedZoom = this.camera.position.z;
        }, { passive: false });

        // Double-click to reset view
        canvas.addEventListener('dblclick', () => {
            this.savedRotation = { x: -0.5, y: 0.5 };
            this.savedZoom = 0.9;
            this.sliceGroup.rotation.x = this.savedRotation.x;
            this.sliceGroup.rotation.y = this.savedRotation.y;
            this.sliceGroup.rotation.z = 0;
            this.camera.position.z = this.savedZoom;
            document.getElementById('projections-angle-display').textContent =
                'angle_y: 0.50, angle_x: -0.50';
        });
    },

    animate() {
        if (!this.renderer || !this.scene || !this.camera) return;

        const modal = document.getElementById('projections-modal');
        if (modal && modal.classList.contains('visible')) {
            requestAnimationFrame(() => this.animate());
            this.renderer.render(this.scene, this.camera);
        }
    },

    handleResize() {
        const container = document.getElementById('projections-3d-container');
        if (!container || !this.renderer || !this.camera) return;

        const w = container.clientWidth || 600;
        const h = 500;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }
};

// Handle window resize
window.addEventListener('resize', () => ProjectionsViewer.handleResize());
