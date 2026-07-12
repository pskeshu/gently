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
        // Clear any volume from a previous open so a failed /api/volume-raw fetch
        // can't leave the prior embryo/timepoint's 3D data bound (stale-render).
        this.volumeData = null;
        this.volumeShape = null;

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
        // If the 3D view is requested but no volume loaded (e.g. /api/volume-raw
        // failed while projections succeeded), fall back to the projections grid
        // rather than showing an empty, never-initialized 3D panel.
        if (method === '3d_viewer' && !this.volumeData) method = null;
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

    // Resize the WebGL canvas + camera to the container's current width.
    // (Height is fixed at 400px; only width tracks the layout.) The animation
    // loop handles re-rendering.
    _resize3D() {
        const container = document.getElementById('pv-3d-container');
        if (!container || !this.renderer3d || !this.camera3d) return;
        const w = container.clientWidth || 500;
        const h = 400;
        this.renderer3d.setSize(w, h);
        this.camera3d.aspect = w / h;
        this.camera3d.updateProjectionMatrix();
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

        // Keep the WebGL canvas in sync with its container width — the chat
        // panel can dock/resize and the window can resize. The animation loop
        // re-renders every frame, so on a size change we only need to resize the
        // renderer + camera (coalesced to one rAF). Also listen for the explicit
        // layout-change event the chat dock fires on collapse/expand + resize.
        if (this._resizeObserver) this._resizeObserver.disconnect();
        this._resizeObserver = new ResizeObserver(() => {
            if (this._resizeRaf) cancelAnimationFrame(this._resizeRaf);
            this._resizeRaf = requestAnimationFrame(() => this._resize3D());
        });
        this._resizeObserver.observe(container);
        if (!this._onLayoutChanged) {
            this._onLayoutChanged = () => this._resize3D();
            window.addEventListener('gently:layout-changed', this._onLayoutChanged);
        }

        // Root group is the object the user rotates. Raymarched volume
        // mesh is added here. The group scale flips Y so the image
        // orientation matches 2D projections.
        this.sliceGroup = new THREE.Group();
        this.sliceGroup.rotation.x = this.savedRotation.x;
        this.sliceGroup.rotation.y = this.savedRotation.y;
        this.sliceGroup.rotation.z = 0;
        this.sliceGroup.scale.y = -1;
        this.scene3d.add(this.sliceGroup);

        this.threshold = 30;
        this.contrast = 1.0;
        this._buildVolumeCube();

        // Threshold / contrast are now shader uniforms - updating them
        // is instant, no slice-stack rebuild.
        const threshSlider = document.getElementById('pv-threshold');
        const threshDisplay = document.getElementById('pv-threshold-val');
        threshSlider.addEventListener('input', (e) => {
            this.threshold = parseInt(e.target.value);
            if (this.volumeMaterial) {
                this.volumeMaterial.uniforms.uThreshold.value = this.threshold / 255.0;
            }
            threshDisplay.textContent = (this.threshold / 100).toFixed(2);
        });

        const contrastSlider = document.getElementById('pv-contrast');
        const contrastDisplay = document.getElementById('pv-contrast-val');
        contrastSlider.addEventListener('input', (e) => {
            this.contrast = parseInt(e.target.value) / 100;
            if (this.volumeMaterial) {
                this.volumeMaterial.uniforms.uContrast.value = this.contrast;
            }
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

        // Animation loop. Raymarching fragment shader needs the camera
        // position in the volume cube's local space each frame.
        this._cameraObjectPos = new THREE.Vector3();
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            if (this.volumeMesh && this.volumeMaterial) {
                // Camera position in cube-local coordinates (accounts for
                // sliceGroup's rotation AND the Y scale flip).
                this._cameraObjectPos.copy(this.camera3d.position);
                this.volumeMesh.worldToLocal(this._cameraObjectPos);
                this.volumeMaterial.uniforms.uCameraObjectPos.value.copy(this._cameraObjectPos);
            }
            this.renderer3d.render(this.scene3d, this.camera3d);
        };
        animate();
    },

    // Build the raymarched volume mesh: a Data3DTexture holding the full
    // volume, a BoxGeometry with the physical extents of the volume, and
    // a ShaderMaterial that marches rays from the camera through the cube
    // and composites front-to-back. This replaces the triple-axis slice
    // stacks and eliminates the edge-on line artifacts because it samples
    // the volume continuously along the view direction at any angle.
    _buildVolumeCube() {
        if (!this.volumeShape || !this.volumeData) return;
        const [zd, h, w] = this.volumeShape;

        // Upload the full volume as a 3D texture. RedFormat + UnsignedByte
        // keeps it to 1 byte per voxel. Data layout is (depth, height,
        // width) in the order z*h*w + y*w + x, which matches how the
        // backend packs it. sampler3D uvw coords are (x, y, z).
        const tex3d = new THREE.DataTexture3D(this.volumeData, w, h, zd);
        tex3d.format = THREE.RedFormat;
        tex3d.type = THREE.UnsignedByteType;
        tex3d.minFilter = THREE.LinearFilter;
        tex3d.magFilter = THREE.LinearFilter;
        tex3d.wrapR = THREE.ClampToEdgeWrapping;
        tex3d.wrapS = THREE.ClampToEdgeWrapping;
        tex3d.wrapT = THREE.ClampToEdgeWrapping;
        tex3d.unpackAlignment = 1;
        tex3d.needsUpdate = true;
        this.volumeTexture3D = tex3d;

        // Physical extents normalized to the largest axis so the cube
        // fits inside a unit sphere. Matches the earlier slice-stack math.
        const [dz, dy, dx] = this.voxelSizeUm;
        const xExtentUm = w * dx;
        const yExtentUm = h * dy;
        const zExtentUm = zd * dz;
        const maxExtentUm = Math.max(xExtentUm, yExtentUm, zExtentUm);
        const boxW = xExtentUm / maxExtentUm;
        const boxH = yExtentUm / maxExtentUm;
        const boxD = zExtentUm / maxExtentUm;
        const boxSize = new THREE.Vector3(boxW, boxH, boxD);

        const vertexShader = `
            out vec3 vObjectPos;
            void main() {
                vObjectPos = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        // Front-to-back raymarching fragment shader. Works at any camera
        // angle because it samples the 3D texture continuously along the
        // view ray using hardware trilinear interpolation - no slice
        // planes to go edge-on, no gaps between samples.
        const fragmentShader = `
            precision highp float;
            precision highp sampler3D;

            uniform sampler3D uVolume;
            uniform vec3 uBoxSize;
            uniform float uThreshold;
            uniform float uContrast;
            uniform vec3 uCameraObjectPos;
            uniform int uMaxSteps;

            in vec3 vObjectPos;
            out vec4 outColor;

            // Returns true and (tMin, tMax) if the ray (ro, rd) intersects
            // the AABB [boxMin, boxMax]. Slab method.
            bool rayBoxIntersect(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax, out float tMin, out float tMax) {
                vec3 invD = 1.0 / rd;
                vec3 t1 = (boxMin - ro) * invD;
                vec3 t2 = (boxMax - ro) * invD;
                vec3 tmn = min(t1, t2);
                vec3 tmx = max(t1, t2);
                tMin = max(max(tmn.x, tmn.y), tmn.z);
                tMax = min(min(tmx.x, tmx.y), tmx.z);
                return tMax > max(tMin, 0.0);
            }

            // Pseudo-random hash, used to jitter the ray start position.
            // Without jittering, the fixed step size beats against the
            // voxel grid and produces visible "wood grain" bands in the
            // final image. Jittering by up to one step size decorrelates
            // the sampling pattern across neighboring pixels so the bands
            // break up into noise the eye reads as smooth.
            float hash12(vec2 p) {
                vec3 p3 = fract(vec3(p.xyx) * 0.1031);
                p3 += dot(p3, p3.yzx + 33.33);
                return fract((p3.x + p3.y) * p3.z);
            }

            void main() {
                vec3 boxHalf = uBoxSize * 0.5;
                vec3 ro = uCameraObjectPos;
                vec3 rd = normalize(vObjectPos - uCameraObjectPos);

                float tMin, tMax;
                if (!rayBoxIntersect(ro, rd, -boxHalf, boxHalf, tMin, tMax)) {
                    discard;
                }
                tMin = max(tMin, 0.0);

                // Step size scales with ray traversal length so resolution
                // is consistent regardless of volume aspect ratio.
                float totalLen = tMax - tMin;
                float stepSize = totalLen / float(uMaxSteps);

                // Per-pixel jitter to break up wood-grain sampling bands.
                float jitter = hash12(gl_FragCoord.xy) * stepSize;
                vec3 pos = ro + rd * (tMin + jitter);
                vec3 step = rd * stepSize;

                // Nominal step count - used to normalize per-step opacity
                // so the overall look is roughly independent of uMaxSteps.
                // If we bump step count for quality, we don't also change
                // how dense the volume looks.
                const float NOMINAL_STEPS = 192.0;
                float opacityScale = NOMINAL_STEPS / float(uMaxSteps);

                vec4 accum = vec4(0.0);
                for (int i = 0; i < 512; i++) {
                    if (i >= uMaxSteps) break;

                    // Convert object-space position to [0,1] UVW texture coords
                    vec3 uvw = (pos + boxHalf) / uBoxSize;
                    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
                        pos += step;
                        continue;
                    }

                    float sampleVal = texture(uVolume, uvw).r;

                    // Smooth transfer function: density ramps continuously
                    // from 0 at uThreshold to 1 at max intensity. smoothstep
                    // has no hard cutoff (unlike the old "if > threshold"
                    // branch), so the volume's surface doesn't show
                    // quantization rings where the threshold cuts into
                    // a gradient.
                    float density = smoothstep(uThreshold, min(uThreshold + 0.45, 1.0), sampleVal);

                    if (density > 0.001) {
                        // Contrast around midpoint
                        float v = clamp((sampleVal - 0.5) * uContrast + 0.5, 0.0, 1.0);
                        vec3 color = vec3(v);

                        // Per-step opacity is intentionally LOW so density
                        // accumulates smoothly over many samples instead of
                        // saturating within one or two steps. Without this
                        // the first dense voxel a ray hits slams alpha to
                        // ~1.0 and early-terminates, producing black-centered
                        // concentric rings (contour lines of ray-integral
                        // density) instead of a smooth volume.
                        float alpha = density * 0.18 * opacityScale;

                        // Front-to-back over compositing
                        accum.rgb += (1.0 - accum.a) * color * alpha;
                        accum.a += (1.0 - accum.a) * alpha;
                    }

                    pos += step;

                    // Only early-terminate when effectively fully opaque.
                    // Raising the threshold from 0.995 to 0.999 keeps more
                    // of the far side of the volume in the final image.
                    if (accum.a > 0.999) break;
                }

                if (accum.a < 0.005) discard;
                outColor = accum;
            }
        `;

        const material = new THREE.ShaderMaterial({
            glslVersion: THREE.GLSL3,
            uniforms: {
                uVolume: { value: tex3d },
                uBoxSize: { value: boxSize },
                uThreshold: { value: this.threshold / 255.0 },
                uContrast: { value: this.contrast },
                uCameraObjectPos: { value: new THREE.Vector3() },
                uMaxSteps: { value: 256 },
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            side: THREE.BackSide,  // Render back faces so rays always start inside the cube
            depthWrite: false,
        });

        const geo = new THREE.BoxGeometry(boxW, boxH, boxD);
        const mesh = new THREE.Mesh(geo, material);
        this.sliceGroup.add(mesh);

        this.volumeMesh = mesh;
        this.volumeMaterial = material;
    },

    cleanup3DViewer() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this._resizeRaf) {
            cancelAnimationFrame(this._resizeRaf);
            this._resizeRaf = null;
        }
        if (this._onLayoutChanged) {
            window.removeEventListener('gently:layout-changed', this._onLayoutChanged);
            this._onLayoutChanged = null;
        }
        // Dispose the volume cube's geometry, material, and 3D texture.
        if (this.volumeMesh) {
            this.volumeMesh.geometry?.dispose();
            this.volumeMesh = null;
        }
        if (this.volumeMaterial) {
            this.volumeMaterial.dispose();
            this.volumeMaterial = null;
        }
        if (this.volumeTexture3D) {
            this.volumeTexture3D.dispose();
            this.volumeTexture3D = null;
        }
        if (this.sliceGroup) {
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
