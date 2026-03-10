/**
 * WebSocket connection management for Gently Visualization
 */

// Reconnect state for exponential backoff
let _wsReconnectDelay = 1000;  // Start at 1s
const _WS_MAX_DELAY = 30000;   // Max 30s

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(`${protocol}//${location.host}/ws`);

    state.ws.onopen = () => {
        state.connected = true;
        _wsReconnectDelay = 1000;  // Reset backoff on success
        document.getElementById('status-text').textContent = 'Connected';
        document.getElementById('status-dot').classList.add('connected');

        // Send join message with presence info
        if (typeof PresenceManager !== 'undefined') {
            PresenceManager.sendJoin();
        }

        // Request initial data
        state.ws.send(JSON.stringify({type: 'get_embryos'}));
        state.ws.send(JSON.stringify({type: 'get_snapshots'}));
        state.ws.send(JSON.stringify({type: 'get_calibration'}));
    };

    state.ws.onclose = () => {
        state.connected = false;
        document.getElementById('status-text').textContent = 'Disconnected';
        document.getElementById('status-dot').classList.remove('connected');
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
    } else if (msg.type === 'calibration') {
        state.calibration = msg.data || [];
        updateCalibrationCount();
        if (state.tab === 'calibration') renderCalibrationGallery();
    } else if (msg.type === 'embryos') {
        state.embryos = msg.data || [];
    } else if (msg.type === 'event') {
        // Add to events tab (full event data)
        handleFullEvent({
            event_type: msg.event_type,
            data: msg.data,
            source: msg.source || 'unknown',
            timestamp: msg.timestamp || new Date().toISOString(),
            event_id: msg.event_id || ''
        });

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
    } else if (msg.type === 'ping') {
        state.ws.send(JSON.stringify({type: 'pong'}));
    } else if (msg.type === 'presence') {
        ClientEventBus.emit('PRESENCE_UPDATE', msg.clients);
    }
}
