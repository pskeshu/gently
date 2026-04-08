// Projections Viewer JavaScript
// Expects SESSION, EMBRYO, INDEX to be defined globally before this script loads

let scene3d, camera3d, renderer3d, sliceGroup;
let volumeData, volumeShape;
let savedRotation = { x: -0.5, y: 0.5 };
let savedZoom = 0.9;

async function loadProjections() {
    try {
        const res = await fetch(`/api/projections/${SESSION}/${EMBRYO}/${INDEX}`);
        const data = await res.json();
        renderProjections(data);
    } catch (err) {
        document.getElementById('content').innerHTML = `
            <div class="card">
                <h3>Error</h3>
                <p>Failed to load projections: ${err.message}</p>
            </div>
        `;
    }
}

function renderProjections(data) {
    const container = document.getElementById('content');

    let html = `
        <div class="info" style="margin-bottom: 15px;">
            Volume shape: ${data.volume_shape.join(' x ')}
            ${data.ground_truth_stage ? ` | Ground Truth: <strong>${data.ground_truth_stage}</strong>` : ''}
        </div>
        <div class="grid">
    `;

    // Add each projection
    for (const proj of data.projections) {
        html += `
            <div class="card">
                <h3>${proj.method.replace('_', ' ').toUpperCase()}</h3>
                <p class="info">${proj.description}</p>
                <img src="data:image/jpeg;base64,${proj.data}" alt="${proj.method}">
            </div>
        `;
    }

    // Add 3D viewer card
    html += `
        <div class="card" style="grid-column: span 2;">
            <h3>3D VOLUME VIEWER</h3>
            <p class="info">Drag: rotate X/Y | Shift+Drag: rotate Z | Scroll: zoom |
                Threshold: <input type="range" id="thresh3d" min="0" max="100" value="30" style="width:80px;vertical-align:middle;">
                <span id="thresh-display" style="font-family:monospace;color:#58a6ff;min-width:35px;display:inline-block;">0.30</span>
                | <span id="angle-display" style="font-family:monospace;color:#58a6ff;">angle_y: 0.50, angle_x: -0.50</span>
            </p>
            <div id="viewer3d-container" style="height:500px;background:#000;"></div>
        </div>
    `;

    html += '</div>';
    container.innerHTML = html;

    // Load 3D volume data
    load3DViewer();
}

async function load3DViewer() {
    try {
        const res = await fetch(`/api/volume/${SESSION}/${EMBRYO}/${INDEX}`);
        const volData = await res.json();
        volumeShape = volData.shape;

        const raw = atob(volData.data);
        volumeData = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) {
            volumeData[i] = raw.charCodeAt(i);
        }

        init3DViewer();
    } catch (err) {
        console.error('Failed to load 3D volume:', err);
    }
}

function createSliceTex(zIndex, threshold) {
    const [zd, h, w] = volumeShape;
    const sliceSize = w * h;
    const offset = zIndex * sliceSize;
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < sliceSize; i++) {
        const val = volumeData[offset + i];
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
}

// Physical voxel size (dz, dy, dx) in microns, matching the default in
// gently.core.imaging.projection_three_view. Used for isometric Z rescaling
// so volumes aren't squished in Z.
const DISPIM_VOXEL_SIZE_UM = [1.0, 0.1625, 0.1625];

function buildSlices3d(numSlices, threshold) {
    if (!volumeShape) return;
    const [zd, h, w] = volumeShape;

    // Clear old slices
    while (sliceGroup.children.length > 0) {
        const c = sliceGroup.children[0];
        c.geometry.dispose();
        c.material.dispose();
        sliceGroup.remove(c);
    }

    // Isometric scaling from physical voxel dimensions (mirrors the fix in
    // projection-viewer.js). Previously a hardcoded (zd/w)*3 multiplier
    // under-scaled Z by ~2x for typical diSPIM volumes.
    const [dz, dy, dx] = DISPIM_VOXEL_SIZE_UM;
    const xExtentUm = w * dx;
    const yExtentUm = h * dy;
    const zExtentUm = zd * dz;
    const maxExtentUm = Math.max(xExtentUm, yExtentUm, zExtentUm);
    const planeW = xExtentUm / maxExtentUm;
    const planeH = yExtentUm / maxExtentUm;
    const zScale = zExtentUm / maxExtentUm;

    for (let i = 0; i < numSlices; i++) {
        const zIndex = Math.floor(i * zd / numSlices);
        const zPos = (i / numSlices - 0.5) * zScale;
        const tex = createSliceTex(zIndex, threshold);
        const mat = new THREE.MeshBasicMaterial({
            map: tex,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
        const geo = new THREE.PlaneGeometry(planeW, planeH);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.z = zPos;
        sliceGroup.add(mesh);
    }
}

function init3DViewer() {
    const container = document.getElementById('viewer3d-container');
    const w = container.clientWidth || 600;
    const h = container.clientHeight || 500;

    scene3d = new THREE.Scene();
    camera3d = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    camera3d.position.z = savedZoom;

    renderer3d = new THREE.WebGLRenderer({ antialias: true });
    renderer3d.setSize(w, h);
    renderer3d.setClearColor(0x000000);
    container.appendChild(renderer3d.domElement);

    sliceGroup = new THREE.Group();
    sliceGroup.rotation.x = savedRotation.x;
    sliceGroup.rotation.y = savedRotation.y;
    sliceGroup.rotation.z = 0;
    sliceGroup.scale.y = -1;  // Flip Y to match dual_view orientation
    scene3d.add(sliceGroup);

    buildSlices3d(32, 30);

    // Threshold control
    document.getElementById('thresh3d').addEventListener('input', (e) => {
        const threshVal = parseInt(e.target.value);
        buildSlices3d(32, threshVal);
        document.getElementById('thresh-display').textContent = (threshVal / 100).toFixed(2);
    });

    // Mouse controls
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };

    renderer3d.domElement.addEventListener('mousedown', (e) => {
        isDragging = true;
        prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => isDragging = false);

    renderer3d.domElement.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        if (e.shiftKey) {
            // Shift+drag: rotate around Z axis
            sliceGroup.rotation.z += (e.clientX - prevMouse.x) * 0.01;
        } else {
            // Normal drag: rotate around X and Y
            sliceGroup.rotation.y += (e.clientX - prevMouse.x) * 0.01;
            sliceGroup.rotation.x += (e.clientY - prevMouse.y) * 0.01;
        }
        savedRotation.x = sliceGroup.rotation.x;
        savedRotation.y = sliceGroup.rotation.y;

        document.getElementById('angle-display').textContent =
            'angle_y: ' + sliceGroup.rotation.y.toFixed(2) +
            ', angle_x: ' + sliceGroup.rotation.x.toFixed(2);
        prevMouse = { x: e.clientX, y: e.clientY };
    });

    renderer3d.domElement.addEventListener('wheel', (e) => {
        e.preventDefault();
        camera3d.position.z = Math.max(0.5, Math.min(5, camera3d.position.z + e.deltaY * 0.002));
        savedZoom = camera3d.position.z;
    });

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        renderer3d.render(scene3d, camera3d);
    }
    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        const w = container.clientWidth || 600;
        const h = container.clientHeight || 500;
        camera3d.aspect = w / h;
        camera3d.updateProjectionMatrix();
        renderer3d.setSize(w, h);
    });

    console.log('3D slice viewer initialized');
}

// Start loading when DOM is ready
document.addEventListener('DOMContentLoaded', loadProjections);
