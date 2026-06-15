// ══════════════════════════════════════════════════════════════════════
//  3D Optical Space — live digital-twin of the addressable imaging volume
//
//  Renders the acquisition cuboid (the box of voxels being scanned) with the
//  live light-sheet plane inside it, plus a Z-neighbourhood reference frame.
//  An HTML overlay (mode badge + readouts + a top-down minimap) carries the
//  GLOBAL context: where in the addressable XY stage range this cuboid sits,
//  and the embryos around it.
//
//  Why two representations: the addressable stage XY (~tens of mm), the cuboid
//  footprint (~hundreds of µm) and the piezo Z range (~µm) differ by ~100x, so
//  a single literal-scale 3D box would draw the cuboid invisibly small. The 3D
//  scene therefore stays in one µm scale around the cuboid; the minimap (2D)
//  handles the much larger stage extent. Some scales are local by design — see
//  FOV_UM / the outer-frame sizing below.
//
//  Data:
//    DEVICE_STATE_UPDATE  → live Piezo.Position (sheet Z), Galvo.A/B, XYStage,
//                            and the firmware box (minimap extent).
//    SCAN_GEOMETRY_UPDATE → cuboid extents, num_slices, pencil/sheet mode.
//    EMBRYOS_UPDATE       → minimap markers.
//  Bootstrap via /api/devices/scan_geometry + /api/embryos/current.
//
//  Mirrors the DevicesManager IIFE pattern (devices.js) and reuses the
//  Three.js scaffold + drag-orbit from projection-viewer.js.
// ══════════════════════════════════════════════════════════════════════

const Occupancy3DManager = (function () {
    'use strict';

    // --- Tunables / approximations (v1) --------------------------------
    // Camera FOV footprint of the SPIM cuboid in µm. SPIM is 0.1625 µm/px;
    // a ~2048px sCMOS ROI ≈ 333 µm. Not currently streamed, so we use a
    // constant until SCAN_GEOMETRY_UPDATE carries fov_um. (Documented approx.)
    const FOV_UM = 333.0;
    const MAX_SLICE_LINES = 30;       // cap drawn slice outlines for perf
    const COLORS = {
        outer: 0x33414d,
        cuboid: 0x14b8c4,
        cuboidFace: 0x14b8c4,
        sheet: 0x39d0ff,
        slice: 0x2a6f78,
        beam: 0xffd166,
    };

    // --- Module state --------------------------------------------------
    let _initialized = false;
    let _scene = null, _camera = null, _renderer = null, _root = null;
    let _animationId = null, _resizeObserver = null, _resizeRaf = null, _onLayoutChanged = null;
    let _isDragging = false, _prevMouse = { x: 0, y: 0 };
    const _rot = { x: -0.6, y: 0.6 };
    let _zoom = 1.7;

    // Live data caches
    let _geom = null;                 // last SCAN_GEOMETRY_UPDATE.data
    let _firmwareBox = null;          // {x:[min,max], y:[min,max]} µm
    let _stage = { x: null, y: null };
    let _piezoZ = null;               // live axial position (µm)
    let _galvo = { a: null, b: null };
    let _embryos = [];                // [{x,y,role,id}]
    let _scaler = null;

    // Scene object handles (rebuilt as geometry changes)
    let _outerBox = null, _cuboid = null, _cuboidEdges = null;
    let _sheet = null, _beam = null, _sliceGroup = null;

    // DOM
    let _container = null, _modeEl = null, _readoutsEl = null, _minimapEl = null, _demoBtn = null;
    let _demoTimer = null;

    // ===================================================================
    // Init / scene scaffold
    // ===================================================================
    function init() {
        if (_initialized) { _resize(); return; }
        if (typeof THREE === 'undefined') {
            console.warn('[occupancy3d] THREE not loaded');
            return;
        }
        _container = document.getElementById('occ3d-container');
        _modeEl = document.getElementById('occ3d-mode');
        _readoutsEl = document.getElementById('occ3d-readouts');
        _minimapEl = document.getElementById('occ3d-minimap');
        _demoBtn = document.getElementById('occ3d-demo-btn');
        if (!_container) return;

        _buildScene();
        _wireInteraction();
        if (_demoBtn) _demoBtn.addEventListener('click', toggleDemo);

        // Subscribe to live data (mirror devices.js:1553-1559)
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('DEVICE_STATE_UPDATE', handleDeviceState);
            ClientEventBus.on('SCAN_GEOMETRY_UPDATE', handleScanGeometry);
            ClientEventBus.on('EMBRYOS_UPDATE', handleEmbryos);
        }
        _bootstrap();

        _initialized = true;
        _rebuildSceneObjects();
        _renderReadouts();
        _renderMinimap();
        _animate();
        // Container is 0×0 while the tab is hidden; size once it's visible.
        requestAnimationFrame(_resize);
    }

    function _buildScene() {
        const w = _container.clientWidth || 600;
        const h = _container.clientHeight || 460;

        _scene = new THREE.Scene();
        _camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
        _camera.position.set(0, 0, _zoom);

        _renderer = new THREE.WebGLRenderer({ antialias: true });
        _renderer.setSize(w, h);
        _renderer.setClearColor(0x0a0e12);
        _container.innerHTML = '';
        _container.appendChild(_renderer.domElement);

        _root = new THREE.Group();
        _root.rotation.x = _rot.x;
        _root.rotation.y = _rot.y;
        _scene.add(_root);

        // Keep the canvas in sync with its container (chat dock / window resize).
        if (_resizeObserver) _resizeObserver.disconnect();
        _resizeObserver = new ResizeObserver(() => {
            if (_resizeRaf) cancelAnimationFrame(_resizeRaf);
            _resizeRaf = requestAnimationFrame(_resize);
        });
        _resizeObserver.observe(_container);
        if (!_onLayoutChanged) {
            _onLayoutChanged = () => _resize();
            window.addEventListener('gently:layout-changed', _onLayoutChanged);
        }
    }

    function _resize() {
        if (!_renderer || !_container) return;
        const w = _container.clientWidth || 600;
        const h = _container.clientHeight || 460;
        if (w === 0 || h === 0) return;
        _camera.aspect = w / h;
        _camera.updateProjectionMatrix();
        _renderer.setSize(w, h);
    }

    function _wireInteraction() {
        const el = _renderer.domElement;
        el.addEventListener('mousedown', (e) => {
            _isDragging = true; _prevMouse = { x: e.clientX, y: e.clientY };
        });
        el.addEventListener('mousemove', (e) => {
            if (!_isDragging) return;
            _root.rotation.y += (e.clientX - _prevMouse.x) * 0.01;
            _root.rotation.x += (e.clientY - _prevMouse.y) * 0.01;
            _rot.x = _root.rotation.x; _rot.y = _root.rotation.y;
            _prevMouse = { x: e.clientX, y: e.clientY };
        });
        window.addEventListener('mouseup', () => { _isDragging = false; });
        el.addEventListener('wheel', (e) => {
            e.preventDefault();
            _zoom = Math.max(0.4, Math.min(6, _zoom + e.deltaY * 0.002));
            _camera.position.z = _zoom;
        }, { passive: false });
        el.addEventListener('dblclick', () => {
            _rot.x = -0.6; _rot.y = 0.6; _zoom = 1.7;
            _root.rotation.x = _rot.x; _root.rotation.y = _rot.y;
            _camera.position.z = _zoom;
        });
    }

    function _animate() {
        _animationId = requestAnimationFrame(_animate);
        if (_renderer && _scene && _camera) _renderer.render(_scene, _camera);
    }

    // ===================================================================
    // Scene geometry (rebuilt when scan geometry changes)
    // ===================================================================
    function _disposeObj(obj) {
        if (!obj) return;
        _root.remove(obj);
        obj.traverse?.((c) => {
            c.geometry?.dispose?.();
            if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach(m => m.dispose());
        });
        obj.geometry?.dispose?.();
        if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => m.dispose());
    }

    function _currentGeom() {
        // Fall back to nominal defaults so the scene is never empty.
        const g = _geom || {};
        const scan = g.scan || {};
        const derived = g.derived || {};
        const piezoCenter = scan.piezo_center_um != null ? scan.piezo_center_um : 50.0;
        const zExtent = derived.z_extent_um != null ? derived.z_extent_um : 50.0;
        return {
            numSlices: scan.num_slices != null ? scan.num_slices : 50,
            piezoCenter,
            zExtent,
            mode: g.mode || 'sheet',
        };
    }

    function _rebuildSceneObjects() {
        if (!_root) return;
        [_outerBox, _cuboid, _cuboidEdges, _sheet, _beam, _sliceGroup].forEach(_disposeObj);
        _outerBox = _cuboid = _cuboidEdges = _sheet = _beam = _sliceGroup = null;

        const g = _currentGeom();
        const fov = FOV_UM;
        // Outer Z neighbourhood centred on the cuboid so it's always framed.
        const halfZ = Math.max(g.zExtent * 2.5, 75);
        const zMin = g.piezoCenter - halfZ;
        const zMax = g.piezoCenter + halfZ;
        const halfXY = fov * 1.5;

        _scaler = makeSceneScaler({
            xRange: [-halfXY, halfXY],
            yRange: [-halfXY, halfXY],
            zRange: [zMin, zMax],
        });
        const L = (um) => _scaler.scaleLen(um);
        const Z = (um) => _scaler.toScene(um, 'z');

        // --- Outer reference frame (addressable Z × local XY) ----------
        _outerBox = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(L(2 * halfXY), L(zMax - zMin), L(2 * halfXY))),
            new THREE.LineBasicMaterial({ color: COLORS.outer })
        );
        _outerBox.position.y = Z(g.piezoCenter); // box centred on its own midpoint == piezoCenter
        _root.add(_outerBox);

        // --- Acquisition cuboid (footprint × z-extent) -----------------
        // Three.js Y is our axial (Z µm) axis; X/Z are the lateral footprint.
        const cw = L(fov), cd = L(fov), ch = L(g.zExtent);
        _cuboid = new THREE.Mesh(
            new THREE.BoxGeometry(cw, ch, cd),
            new THREE.MeshBasicMaterial({
                color: COLORS.cuboidFace, transparent: true, opacity: 0.06,
                depthWrite: false, side: THREE.DoubleSide,
            })
        );
        _cuboid.position.y = Z(g.piezoCenter);
        _root.add(_cuboid);
        _cuboidEdges = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(cw, ch, cd)),
            new THREE.LineBasicMaterial({ color: COLORS.cuboid })
        );
        _cuboidEdges.position.y = Z(g.piezoCenter);
        _root.add(_cuboidEdges);

        // --- Slice planes (faint outlines through the cuboid) ----------
        _sliceGroup = new THREE.Group();
        const n = Math.max(1, Math.min(g.numSlices, MAX_SLICE_LINES));
        const sliceMat = new THREE.LineBasicMaterial({ color: COLORS.slice, transparent: true, opacity: 0.5 });
        for (let i = 0; i < n; i++) {
            const frac = n === 1 ? 0.5 : i / (n - 1);
            const zUm = (g.piezoCenter - g.zExtent / 2) + frac * g.zExtent;
            const ring = new THREE.LineLoop(_rectXZ(cw, cd), sliceMat);
            ring.position.y = Z(zUm);
            _sliceGroup.add(ring);
        }
        _root.add(_sliceGroup);

        // --- Light sheet / pencil beam ---------------------------------
        if (g.mode === 'pencil') {
            // Pencil: a thin beam along the lateral axis through cuboid centre.
            _beam = new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(-cw / 2, 0, 0), new THREE.Vector3(cw / 2, 0, 0),
                ]),
                new THREE.LineBasicMaterial({ color: COLORS.beam })
            );
            _root.add(_beam);
        } else {
            _sheet = new THREE.Mesh(
                new THREE.PlaneGeometry(cw, cd),
                new THREE.MeshBasicMaterial({
                    color: COLORS.sheet, transparent: true, opacity: 0.35,
                    side: THREE.DoubleSide, depthWrite: false,
                })
            );
            _sheet.rotation.x = -Math.PI / 2; // lie in the lateral (X-Z) plane
            _root.add(_sheet);
        }
        _updateSheetPosition();
    }

    // A rectangle outline in the lateral (X-Z) plane, centred at origin.
    function _rectXZ(w, d) {
        const hw = w / 2, hd = d / 2;
        return new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-hw, 0, -hd), new THREE.Vector3(hw, 0, -hd),
            new THREE.Vector3(hw, 0, hd), new THREE.Vector3(-hw, 0, hd),
        ]);
    }

    // Move the sheet/beam to the live axial position (piezo µm), clamped to
    // the cuboid extent. Falls back to the cuboid centre when no live value.
    function _updateSheetPosition() {
        if (!_scaler) return;
        const g = _currentGeom();
        const zMin = g.piezoCenter - g.zExtent / 2;
        const zMax = g.piezoCenter + g.zExtent / 2;
        let zUm = _piezoZ != null ? _piezoZ : g.piezoCenter;
        zUm = Math.max(zMin, Math.min(zMax, zUm));
        const y = _scaler.toScene(zUm, 'z');
        if (_sheet) _sheet.position.y = y;
        if (_beam) _beam.position.y = y;
    }

    // ===================================================================
    // Event handlers
    // ===================================================================
    function handleDeviceState(payload) {
        if (!payload) return;
        const pos = payload.positions || {};
        for (const name of Object.keys(pos)) {
            const e = pos[name] || {};
            if (e.kind === 'xy_stage') {
                if (e.X != null) _stage.x = e.X;
                if (e.Y != null) _stage.y = e.Y;
            } else if (e.kind === 'piezo') {
                if (e.Position != null) _piezoZ = e.Position;
            } else if (e.kind === 'galvo') {
                if (e.A != null) _galvo.a = e.A;
                if (e.B != null) _galvo.b = e.B;
            }
        }
        const box = extractFirmwareBox(payload.properties);
        if (box) _firmwareBox = box;
        _updateSheetPosition();
        _renderReadouts();
        _renderMinimap();
    }

    function handleScanGeometry(payload) {
        if (!payload) return;
        _geom = payload;
        if (payload.stage_position_um) {
            if (payload.stage_position_um.x != null) _stage.x = payload.stage_position_um.x;
            if (payload.stage_position_um.y != null) _stage.y = payload.stage_position_um.y;
        }
        _rebuildSceneObjects();
        _renderReadouts();
        _renderMinimap();
    }

    function handleEmbryos(payload) {
        if (!payload || !Array.isArray(payload.embryos)) return;
        _embryos = payload.embryos.map((e) => {
            const fine = e.position_fine || {};
            const coarse = e.position_coarse || {};
            const x = fine.x != null ? fine.x : coarse.x;
            const y = fine.y != null ? fine.y : coarse.y;
            return { x, y, role: e.role, id: e.id };
        }).filter((e) => e.x != null && e.y != null);
        _renderMinimap();
    }

    async function _bootstrap() {
        try {
            const r = await fetch('/api/devices/scan_geometry');
            if (r.ok) handleScanGeometry(await r.json());
        } catch (_) { /* offline — demo button covers it */ }
        try {
            const r = await fetch('/api/embryos/current');
            if (r.ok) handleEmbryos(await r.json());
        } catch (_) { /* ignore */ }
    }

    // ===================================================================
    // HTML overlay: mode badge, readouts, minimap
    // ===================================================================
    function _fmt(v, digits = 1, unit = '') {
        return v == null ? '—' : (Number(v).toFixed(digits) + unit);
    }

    function _renderReadouts() {
        const g = _currentGeom();
        if (_modeEl) {
            _modeEl.textContent = g.mode === 'pencil' ? 'PENCIL' : 'SHEET';
            _modeEl.classList.toggle('is-pencil', g.mode === 'pencil');
        }
        if (!_readoutsEl) return;
        const scan = (_geom && _geom.scan) || {};
        const derived = (_geom && _geom.derived) || {};
        const rows = [
            ['stage X', _fmt(_stage.x, 0, ' µm')],
            ['stage Y', _fmt(_stage.y, 0, ' µm')],
            ['piezo Z', _fmt(_piezoZ, 1, ' µm')],
            ['galvo A/B', `${_fmt(_galvo.a, 3)} / ${_fmt(_galvo.b, 3)}°`],
            ['slices', scan.num_slices != null ? String(scan.num_slices) : '—'],
            ['Z extent', _fmt(derived.z_extent_um, 1, ' µm')],
            ['slice step', _fmt(derived.slice_spacing_um, 3, ' µm')],
        ];
        _readoutsEl.innerHTML = rows
            .map(([k, v]) => `<div class="occ3d-row"><span>${k}</span><b>${escapeHtml(v)}</b></div>`)
            .join('');
    }

    function _renderMinimap() {
        if (!_minimapEl) return;
        const VB = { w: 200, h: 120, pad: 8 };
        const box = _firmwareBox || { x: [-25000, 25000], y: [-12000, 12000] };
        const bw = box.x[1] - box.x[0], bh = box.y[1] - box.y[0];
        if (!(bw > 0 && bh > 0)) return;
        const sx = (VB.w - 2 * VB.pad) / bw;
        const sy = (VB.h - 2 * VB.pad) / bh;
        const s = Math.min(sx, sy);
        const ox = VB.pad + (VB.w - 2 * VB.pad - bw * s) / 2;
        const oy = VB.pad + (VB.h - 2 * VB.pad - bh * s) / 2;
        const px = (x) => ox + (x - box.x[0]) * s;
        const py = (y) => oy + (box.y[1] - y) * s; // flip Y for screen

        const parts = [];
        parts.push(`<rect x="${ox.toFixed(1)}" y="${oy.toFixed(1)}" width="${(bw * s).toFixed(1)}" height="${(bh * s).toFixed(1)}" class="occ3d-mm-box"/>`);
        for (const e of _embryos) {
            parts.push(`<circle cx="${px(e.x).toFixed(1)}" cy="${py(e.y).toFixed(1)}" r="2" class="occ3d-mm-embryo"/>`);
        }
        if (_stage.x != null && _stage.y != null) {
            const fovPx = FOV_UM * s;
            parts.push(`<rect x="${(px(_stage.x) - fovPx / 2).toFixed(1)}" y="${(py(_stage.y) - fovPx / 2).toFixed(1)}" width="${Math.max(fovPx, 3).toFixed(1)}" height="${Math.max(fovPx, 3).toFixed(1)}" class="occ3d-mm-cuboid"/>`);
            parts.push(`<circle cx="${px(_stage.x).toFixed(1)}" cy="${py(_stage.y).toFixed(1)}" r="2.5" class="occ3d-mm-stage"/>`);
        }
        _minimapEl.innerHTML = parts.join('');
    }

    // ===================================================================
    // Demo driver — develop without live hardware (launch_gently.py --offline)
    // ===================================================================
    function toggleDemo() {
        if (_demoTimer) {
            clearInterval(_demoTimer); _demoTimer = null;
            if (_demoBtn) _demoBtn.classList.remove('is-on');
            return;
        }
        if (_demoBtn) _demoBtn.classList.add('is-on');
        // Seed a firmware box, a scan geometry, and a few embryos.
        _firmwareBox = { x: [-25000, 25000], y: [-12000, 12000] };
        handleScanGeometry({
            embryo_id: 'demo_2',
            stage_position_um: { x: 4200, y: -1800 },
            scan: {
                num_slices: 60, exposure_ms: 5.0,
                galvo_amplitude_deg: 0.5, galvo_center_deg: 0.0,
                piezo_amplitude_um: 25.0, piezo_center_um: 50.0,
            },
            derived: { z_extent_um: 50.0, slice_spacing_um: 50 / 59, z_min_um: 25, z_max_um: 75 },
            mode: 'sheet', ts: 0,
        });
        handleEmbryos({
            embryos: [
                { id: 'demo_1', role: 'test', position_coarse: { x: 4200, y: -1800 } },
                { id: 'demo_2', role: 'control', position_coarse: { x: -8000, y: 5200 } },
                { id: 'demo_3', role: 'test', position_coarse: { x: 12000, y: 2400 } },
            ],
        });
        // Sweep the sheet in Z to animate the plane.
        let t = 0;
        _demoTimer = setInterval(() => {
            t += 0.08;
            const g = _currentGeom();
            _piezoZ = g.piezoCenter + (g.zExtent / 2) * Math.sin(t);
            _galvo.a = 0.5 * Math.sin(t);
            _updateSheetPosition();
            _renderReadouts();
        }, 60);
    }

    function cleanup() {
        if (_animationId) cancelAnimationFrame(_animationId);
        if (_demoTimer) { clearInterval(_demoTimer); _demoTimer = null; }
        if (_resizeObserver) _resizeObserver.disconnect();
        if (_renderer) { _renderer.dispose(); }
    }

    return { init, cleanup, toggleDemo, handleDeviceState, handleScanGeometry, handleEmbryos };
})();

document.addEventListener('DOMContentLoaded', () => {
    // Build lazily on first tab activation (container is 0×0 while hidden),
    // so init() is invoked from app.js switchTab(), not here.
});
