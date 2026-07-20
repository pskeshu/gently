/**
 * WebSocket connection management for Gently Visualization
 */

// Reconnect state for exponential backoff
let _wsReconnectDelay = 1000;  // Start at 1s
const _WS_MAX_DELAY = 30000;   // Max 30s

/**
 * Load initial data via parallel REST calls (much faster than WebSocket).
 * Called on page load and on WebSocket reconnect.
 */
let _initialDataLoaded = false;

function loadInitialData() {
    if (_initialDataLoaded) return;
    _initialDataLoaded = true;

    // Fire all three in parallel
    fetch('/api/snapshots')
        .then(r => r.json())
        .then(data => {
            state.snapshots = data.snapshots || [];
            updateMainCount();
            renderRecentList();
            if (typeof updateStatusbar === 'function') updateStatusbar();
        })
        .catch(e => console.warn('Failed to load snapshots:', e));

    fetch('/api/calibration')
        .then(r => r.json())
        .then(data => {
            state.calibration = data.calibration || [];
            updateCalibrationCount();
            renderCalibrationGallery();
        })
        .catch(e => console.warn('Failed to load calibration:', e));

    // Bootstrap from the CURRENT (in-memory experiment) embryos, not the disk
    // store: embryos registered this session live in memory until a volume is
    // acquired, so /api/embryos (disk-backed) reads 0 and every "N embryos"
    // count shows 0 on load. Map to id strings to keep state.embryos's shape
    // (viewer.js does .includes()/.push() on it).
    fetch('/api/embryos/current')
        .then(r => r.json())
        .then(data => {
            const list = data.embryos || [];
            state.embryos = list.map(e => (e && e.id) ? e.id : e);
            if (typeof updateStatusbar === 'function') updateStatusbar();
            // Fan out so every count-renderer (header strip, etc.) refreshes off
            // the bootstrap, not just the footer statusbar. Shape matches the
            // server-pushed EMBRYOS_UPDATE ({embryos: [...]}).
            if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('EMBRYOS_UPDATE', { embryos: list });
        })
        .catch(e => console.warn('Failed to load embryos:', e));
}

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(`${protocol}//${location.host}/ws`);

    state.ws.onopen = () => {
        state.connected = true;
        _wsReconnectDelay = 1000;  // Reset backoff on success
        updateGentlyStatus(true);

        // Send join message with presence info
        if (typeof PresenceManager !== 'undefined') {
            PresenceManager.sendJoin();
        }

        // Load initial data via REST (parallel, fast) if not already loaded
        loadInitialData();
    };

    state.ws.onclose = () => {
        state.connected = false;
        _initialDataLoaded = false;  // Allow reload on reconnect
        updateGentlyStatus(false);
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (capped)
        setTimeout(connectWebSocket, _wsReconnectDelay);
        _wsReconnectDelay = Math.min(_wsReconnectDelay * 2, _WS_MAX_DELAY);
    };

    state.ws.onerror = () => {};

    state.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
}

function handleMessage(msg) {
    if (msg.type === 'image') {
        handleNewImage(msg.data);
        ClientEventBus.emit('IMAGE_RECEIVED', msg.data);
    } else if (msg.type === 'volume_3d') {
        handleNew3DVolume(msg.data);
    } else if (msg.type === 'snapshots') {
        state.snapshots = msg.data || [];
        updateMainCount();
        renderRecentList();
        if (typeof updateStatusbar === 'function') updateStatusbar();
    } else if (msg.type === 'calibration') {
        state.calibration = msg.data || [];
        updateCalibrationCount();
        renderCalibrationGallery();
    } else if (msg.type === 'embryos') {
        state.embryos = msg.data || [];
        if (typeof updateStatusbar === 'function') updateStatusbar();
    } else if (msg.type === 'event') {
        // High-volume telemetry: skip the events-tab table (which DOM-creates a row
        // per event and would lag every other handler), but still emit to the
        // client event bus so the Devices tab gets the payload.
        if (msg.event_type !== 'DEVICE_STATE_UPDATE' &&
            msg.event_type !== 'BOTTOM_CAMERA_FRAME' &&
            msg.event_type !== 'TEMPERATURE_UPDATE' &&
            msg.event_type !== 'LIGHTSHEET_FRAME') {
            handleFullEvent({
                event_type: msg.event_type,
                data: msg.data,
                source: msg.source || 'unknown',
                timestamp: msg.timestamp || new Date().toISOString(),
                event_id: msg.event_id || ''
            });
        }

        // Broadcast via client event bus - managers subscribe at init time
        ClientEventBus.emit(msg.event_type, msg.data);

    } else if (msg.type === 'timelapse_state') {
        ClientEventBus.emit('TIMELAPSE_STATE', msg.data);
    } else if (msg.type === 'marking_image') {
        // Server is requesting embryo marking
        if (typeof MarkingManager !== 'undefined') {
            MarkingManager.handleMarkingImage(msg.data);
            // Auto-switch to marking subtab
            MarkingManager.switchSubtab('marking');
            // Switch to embryos tab if not already there
            if (state.tab !== 'embryos') switchTab('embryos');
        }
    } else if (msg.type === 'open_volume') {
        // The agent asked us to open the in-browser volume viewer — the
        // web-native replacement for the old desktop napari window.
        if (typeof ProjectionViewer !== 'undefined' && msg.embryo_id != null) {
            const view = msg.view || '3d_viewer';
            Promise.resolve(ProjectionViewer.open(msg.embryo_id, msg.timepoint))
                .then(() => {
                    // Default to the 3D viewer tab when the agent opens it.
                    if (view && typeof ProjectionViewer.selectMethod === 'function') {
                        ProjectionViewer.selectMethod(view);
                    }
                })
                .catch((e) => console.warn('open_volume failed', e));
        }
    } else if (msg.type === 'session_changed') {
        // The live agent switched sessions (resume from the Sessions tab) —
        // reload so every client picks up the new session's state + transcript.
        window.location.href = '/';
    } else if (msg.type === 'ping') {
        state.ws.send(JSON.stringify({type: 'pong'}));
    } else if (msg.type === 'presence') {
        ClientEventBus.emit('PRESENCE_UPDATE', msg.clients);
    }
}
