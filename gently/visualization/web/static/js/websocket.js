/**
 * WebSocket connection management for Gently Visualization
 */

// Connect WebSocket
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(`${protocol}//${location.host}/ws`);

    state.ws.onopen = () => {
        state.connected = true;
        document.getElementById('status-text').textContent = 'Connected';
        document.getElementById('status-dot').classList.add('connected');
        logEvent('system', 'Connected to server');

        // Request initial data
        state.ws.send(JSON.stringify({type: 'get_embryos'}));
        state.ws.send(JSON.stringify({type: 'get_snapshots'}));
        state.ws.send(JSON.stringify({type: 'get_volumes'}));
        state.ws.send(JSON.stringify({type: 'get_calibration'}));
    };

    state.ws.onclose = () => {
        state.connected = false;
        document.getElementById('status-text').textContent = 'Disconnected';
        document.getElementById('status-dot').classList.remove('connected');
        logEvent('system', 'Disconnected');
        setTimeout(connectWebSocket, 3000);
    };

    state.ws.onerror = () => logEvent('error', 'Connection error');

    state.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
}

function handleMessage(msg) {
    if (msg.type === 'image') {
        handleNewImage(msg.data);
    } else if (msg.type === 'volume_3d') {
        handleNew3DVolume(msg.data);
    } else if (msg.type === 'snapshots') {
        state.snapshots = msg.data || [];
        updateMainCount();
        renderRecentList();
    } else if (msg.type === 'volumes') {
        state.volumes = msg.data || [];
        updateVolumesCount();
        if (state.tab === 'volumes') renderVolumesGallery();
    } else if (msg.type === 'calibration') {
        state.calibration = msg.data || [];
        updateCalibrationCount();
        if (state.tab === 'calibration') renderCalibrationGallery();
    } else if (msg.type === 'embryos') {
        state.embryos = msg.data || [];
        updateEmbryoFilter();
    } else if (msg.type === 'event') {
        // Add to events tab (full event data)
        handleFullEvent({
            event_type: msg.event_type,
            data: msg.data,
            source: msg.source || 'unknown',
            timestamp: msg.timestamp || new Date().toISOString(),
            event_id: msg.event_id || ''
        });

        // Route timelapse/task events to TasksManager
        if (typeof TasksManager !== 'undefined') {
            if (msg.event_type === 'ACQUISITION_STARTED') {
                TasksManager.handleAcquisitionStarted(msg.data);
            } else if (msg.event_type === 'ACQUISITION_COMPLETED') {
                TasksManager.handleAcquisitionCompleted(msg.data);
            } else if (msg.event_type === 'VOLUME_ACQUIRED') {
                TasksManager.handleVolumeAcquired(msg.data);
            } else if (msg.event_type === 'DETECTOR_EVALUATED') {
                // All detector evaluations (with reasoning) - for reasoning panel
                TasksManager.handleDetectorEvaluated(msg.data);
            } else if (msg.event_type === 'DETECTION_TRIGGERED') {
                // Positive detection - update embryo status
                TasksManager.handleDetectionTriggered(msg.data);
            } else if (msg.event_type === 'STATUS_CHANGED') {
                TasksManager.handleStatusChanged(msg.data);
            } else if (msg.event_type === 'HATCHING_DETECTED') {
                // Hatching is a positive detection
                TasksManager.handleDetectionTriggered({
                    embryo_id: msg.data.embryo_id,
                    detector_name: 'hatching',
                    ...msg.data
                });
            }
        }

        // Format CV events nicely for sidebar log
        let eventMsg;
        if (msg.event_type === 'CV_AGENT_THINKING') {
            const thinking = msg.data.thinking || '';
            const preview = thinking.length > 40 ? thinking.slice(0, 40) + '...' : thinking;
            eventMsg = `iter ${msg.data.iteration}: ${preview}`;
        } else if (msg.event_type === 'CV_TASK_QUEUED') {
            eventMsg = `${msg.data.intent} (${msg.data.embryo_id})`;
        } else if (msg.event_type === 'CV_TASK_COMPLETED') {
            eventMsg = `${msg.data.intent} done in ${(msg.data.processing_time_ms/1000).toFixed(1)}s`;
        } else if (msg.event_type === 'CV_TASK_FAILED') {
            eventMsg = `${msg.data.intent} failed: ${msg.data.error?.slice(0, 30) || 'unknown'}`;
        } else if (msg.event_type === 'ACQUISITION_STARTED') {
            const embryoCount = msg.data.embryo_ids?.length || 0;
            eventMsg = `Timelapse started with ${embryoCount} embryo${embryoCount !== 1 ? 's' : ''}`;
        } else if (msg.event_type === 'ACQUISITION_COMPLETED') {
            eventMsg = 'Timelapse completed';
        } else {
            eventMsg = JSON.stringify(msg.data).slice(0, 50);
        }
        logEvent(msg.event_type, eventMsg);
    } else if (msg.type === 'timelapse_state') {
        // Server sending authoritative timelapse state on connect
        if (typeof TasksManager !== 'undefined') {
            TasksManager.reconcileWithServerState(msg.data);
        }
    } else if (msg.type === 'ping') {
        state.ws.send(JSON.stringify({type: 'pong'}));
    }
}
