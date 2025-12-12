/**
 * Tasks Tab - Timelapse Task Tracking
 * Displays active timelapse tasks with per-embryo breakdown
 */

const TasksManager = {
    // Session ID for tracking experiment boundaries
    // When session_id changes, all state is cleared (new experiment)
    currentSessionId: null,

    state: {
        status: 'IDLE', // IDLE, RUNNING, PAUSED, COMPLETED, FAILED
        startedAt: null,
        embryos: {},  // embryo_id -> EmbryoTaskState
        totalTimepoints: 0,
        baseInterval: 120
    },

    // Detection reasoning cache (per-embryo)
    detectionReasoning: {},  // embryo_id -> list of detection results with reasoning

    // Currently selected embryo for detail view
    selectedEmbryoId: null,

    // Expanded image states
    expandedImages: {},  // detection index -> true/false

    countdownInterval: null,
    storageKey: 'gently-tasks-state',

    init() {
        // Restore state from localStorage
        this.loadState();
        // Start countdown update timer
        this.startCountdownUpdates();
        // Initial render
        this.render();
    },

    // ==========================================
    // State Persistence
    // ==========================================

    saveState() {
        try {
            const toSave = {
                sessionId: this.currentSessionId,  // Include session ID for validation on load
                status: this.state.status,
                startedAt: this.state.startedAt ? this.state.startedAt.toISOString() : null,
                embryos: JSON.parse(JSON.stringify(this.state.embryos)),  // Deep clone
                totalTimepoints: this.state.totalTimepoints,
                baseInterval: this.state.baseInterval,
                detectionReasoning: this.detectionReasoning,
                selectedEmbryoId: this.selectedEmbryoId,
                savedAt: new Date().toISOString()
            };
            // Convert Date objects in embryos
            Object.values(toSave.embryos).forEach(e => {
                if (e.lastAcquired instanceof Date) {
                    e.lastAcquired = e.lastAcquired.toISOString();
                }
            });
            localStorage.setItem(this.storageKey, JSON.stringify(toSave));
        } catch (err) {
            console.warn('Failed to save tasks state:', err);
        }
    },

    loadState() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (!saved) return;

            const data = JSON.parse(saved);

            // Check if state is stale (older than 24 hours)
            if (data.savedAt) {
                const savedAge = Date.now() - new Date(data.savedAt).getTime();
                if (savedAge > 24 * 60 * 60 * 1000) {
                    console.log('Tasks state too old, discarding');
                    localStorage.removeItem(this.storageKey);
                    return;
                }
            }

            // Restore session ID (will be validated against server on connect)
            this.currentSessionId = data.sessionId || null;

            this.state.status = data.status || 'IDLE';
            this.state.startedAt = data.startedAt ? new Date(data.startedAt) : null;
            this.state.totalTimepoints = data.totalTimepoints || 0;
            this.state.baseInterval = data.baseInterval || 120;
            this.state.embryos = data.embryos || {};

            // Restore Date objects in embryos
            Object.values(this.state.embryos).forEach(e => {
                if (e.lastAcquired && typeof e.lastAcquired === 'string') {
                    e.lastAcquired = new Date(e.lastAcquired);
                }
            });

            // Restore detection reasoning and selection
            this.detectionReasoning = data.detectionReasoning || {};
            this.selectedEmbryoId = data.selectedEmbryoId || null;

            // Auto-select first embryo if none selected
            if (!this.selectedEmbryoId && Object.keys(this.state.embryos).length > 0) {
                this.selectedEmbryoId = Object.keys(this.state.embryos)[0];
            }

            console.log('Restored tasks state:', this.state.status, Object.keys(this.state.embryos).length, 'embryos', 'session:', this.currentSessionId);
            this.updateTasksCount();
        } catch (err) {
            console.warn('Failed to load tasks state:', err);
        }
    },

    clearSavedState() {
        localStorage.removeItem(this.storageKey);
    },

    // ==========================================
    // Server State Reconciliation
    // ==========================================

    reconcileWithServerState(serverState) {
        const serverSessionId = serverState.session_id;
        const isNewSession = serverSessionId && serverSessionId !== this.currentSessionId;

        if (isNewSession) {
            console.log(`Session changed: ${this.currentSessionId} → ${serverSessionId}`);
            this.clearAllState();
        } else {
            console.log('Reconciling with server state:', serverState.status, 'session:', serverSessionId);
        }

        // Update session ID
        this.currentSessionId = serverSessionId;

        // Server state is authoritative - replace everything
        this.state.status = serverState.status || 'IDLE';
        this.state.startedAt = serverState.started_at ? new Date(serverState.started_at) : null;
        this.state.totalTimepoints = serverState.total_timepoints || 0;
        this.state.baseInterval = serverState.base_interval || 120;

        // Replace embryo states entirely
        this.state.embryos = {};
        for (const [eid, embryoData] of Object.entries(serverState.embryos || {})) {
            this.state.embryos[eid] = {
                embryoId: eid,
                stopCondition: embryoData.stop_condition || 'unknown',
                intervalSeconds: embryoData.interval_seconds || this.state.baseInterval,
                timepoints: embryoData.timepoints || 0,
                isComplete: embryoData.is_complete || false,
                completionReason: null,
                lastAcquired: embryoData.last_acquired ? new Date(embryoData.last_acquired) : null,
                detections: embryoData.detections || {},
                errorCount: 0,
                lastError: null
            };
        }

        // Replace detection reasoning from server
        this.detectionReasoning = serverState.detection_reasoning || {};
        this.expandedImages = {};  // Clear expanded image state

        // Clear selection if selected embryo no longer exists, then auto-select first
        if (this.selectedEmbryoId && !this.state.embryos[this.selectedEmbryoId]) {
            this.selectedEmbryoId = null;
        }
        if (!this.selectedEmbryoId && Object.keys(this.state.embryos).length > 0) {
            this.selectedEmbryoId = Object.keys(this.state.embryos)[0];
        }

        this.updateTasksCount();
        this.render();
        this.saveState();
    },

    // Clear all state (for session boundary)
    clearAllState() {
        this.state = {
            status: 'IDLE',
            startedAt: null,
            embryos: {},
            totalTimepoints: 0,
            baseInterval: 120
        };
        this.detectionReasoning = {};
        this.selectedEmbryoId = null;
        this.expandedImages = {};
    },

    // ==========================================
    // Event Handlers
    // ==========================================

    handleAcquisitionStarted(data) {
        // Clear all state for fresh experiment
        this.state.status = 'RUNNING';
        this.state.startedAt = new Date();
        this.state.totalTimepoints = 0;
        this.state.baseInterval = data.interval_seconds || 120;
        this.state.embryos = {};
        this.detectionReasoning = {};  // Clear old detection data
        this.selectedEmbryoId = null;
        this.expandedImages = {};

        // Initialize embryo states
        const embryoIds = data.embryo_ids || [];
        embryoIds.forEach(id => {
            this.state.embryos[id] = {
                embryoId: id,
                stopCondition: data.stop_condition || 'manual',
                intervalSeconds: data.interval_seconds || 120,
                timepoints: 0,
                isComplete: false,
                completionReason: null,
                lastAcquired: null,
                detections: {},
                errorCount: 0,
                lastError: null
            };
        });

        this.updateTasksCount();
        this.render();
        this.saveState();
    },

    handleVolumeAcquired(data) {
        const embryoId = data.embryo_id;
        if (!embryoId) return;

        // If we don't have this embryo yet, add it
        if (!this.state.embryos[embryoId]) {
            this.state.embryos[embryoId] = {
                embryoId: embryoId,
                stopCondition: 'unknown',
                intervalSeconds: this.state.baseInterval,
                timepoints: 0,
                isComplete: false,
                completionReason: null,
                lastAcquired: null,
                detections: {},
                errorCount: 0,
                lastError: null
            };
            // If we're receiving volumes, we're running
            if (this.state.status === 'IDLE') {
                this.state.status = 'RUNNING';
                this.state.startedAt = new Date();
            }
        }

        const embryo = this.state.embryos[embryoId];
        const newTimepoints = (data.timepoint !== undefined) ? data.timepoint + 1 : embryo.timepoints + 1;

        // Only update totalTimepoints if this is actually a new timepoint
        if (newTimepoints > embryo.timepoints) {
            this.state.totalTimepoints += (newTimepoints - embryo.timepoints);
        }
        embryo.timepoints = newTimepoints;
        embryo.lastAcquired = new Date();

        this.updateTasksCount();
        this.updateEmbryoCard(embryoId);
        this.updateSummary();
        this.saveState();
    },

    handleAcquisitionCompleted(data) {
        this.state.status = 'COMPLETED';

        // Mark any remaining embryos as complete
        Object.values(this.state.embryos).forEach(embryo => {
            if (!embryo.isComplete) {
                embryo.isComplete = true;
                embryo.completionReason = embryo.completionReason || 'Timelapse completed';
            }
        });

        this.render();
        this.saveState();
    },

    handleDetectorEvaluated(data) {
        // All detector evaluations (with reasoning) - populates reasoning panel
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        const detectorName = data.detector_name;
        const detected = data.detected;

        // Update detection status
        embryo.detections[detectorName] = {
            detected: detected,
            confidence: data.confidence,
            timepoint: data.timepoint
        };

        // Store detection reasoning for the panel (avoid duplicates)
        if (!this.detectionReasoning[embryoId]) {
            this.detectionReasoning[embryoId] = [];
        }
        // Check if we already have this detection (same timepoint + detector)
        const isDuplicate = this.detectionReasoning[embryoId].some(
            r => r.timepoint === data.timepoint && r.detector_name === detectorName
        );
        if (!isDuplicate) {
            this.detectionReasoning[embryoId].push({
                detector_name: detectorName,
                detected: detected,
                confidence: data.confidence,
                reasoning: data.reasoning,
                timepoint: data.timepoint,
                volume_uid: data.volume_uid,
                projection_uid: data.projection_uid,
                timestamp: new Date().toISOString()
            });
        }

        this.updateEmbryoCard(embryoId);
        // Update reasoning panel if this embryo is selected
        if (this.selectedEmbryoId === embryoId) {
            this.renderReasoningPanel();
        }
        this.saveState();
    },

    handleDetectionTriggered(data) {
        // Positive detection only - update embryo completion status
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        const detectorName = data.detector_name;
        const condition = embryo.stopCondition?.toLowerCase() || '';

        // Check if this detection completes the embryo
        if (detectorName === 'hatching' && condition.includes('hatching')) {
            embryo.isComplete = true;
            embryo.completionReason = 'Hatching detected';
        } else if (detectorName === 'comma' && condition.includes('comma')) {
            embryo.isComplete = true;
            embryo.completionReason = 'Comma stage detected';
        }

        this.updateEmbryoCard(embryoId);
        this.saveState();
    },

    handleStatusChanged(data) {
        // Handle interval changes
        if (data.embryo_id && data.new_interval_seconds) {
            const embryo = this.state.embryos[data.embryo_id];
            if (embryo) {
                embryo.intervalSeconds = data.new_interval_seconds;
                this.updateEmbryoCard(data.embryo_id);
            }
        }

        // Handle pause/resume
        if (data.status) {
            this.state.status = data.status;
            this.render();
        }
    },

    // ==========================================
    // Rendering
    // ==========================================

    render() {
        this.renderStatusBadge();
        this.renderSummary();
        this.renderEmbryoCards();
        this.renderReasoningPanel();
    },

    renderStatusBadge() {
        const statusEl = document.getElementById('timelapse-status');
        const textEl = document.getElementById('timelapse-status-text');
        const durationEl = document.getElementById('timelapse-duration');

        if (!statusEl) return;

        // Remove all status classes
        statusEl.classList.remove('running', 'paused', 'completed', 'idle');

        if (this.state.status === 'IDLE' || Object.keys(this.state.embryos).length === 0) {
            statusEl.classList.add('idle');
            textEl.textContent = 'No active timelapse';
            durationEl.textContent = '';
        } else {
            statusEl.classList.add(this.state.status.toLowerCase());
            textEl.textContent = this.state.status === 'RUNNING' ? 'Running' :
                                 this.state.status === 'PAUSED' ? 'Paused' :
                                 this.state.status === 'COMPLETED' ? 'Completed' : this.state.status;

            if (this.state.startedAt) {
                durationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
            }
        }
    },

    renderSummary() {
        const summaryEl = document.getElementById('tasks-summary');
        if (!summaryEl) return;

        const embryos = Object.values(this.state.embryos);
        if (embryos.length === 0) {
            summaryEl.classList.add('hidden');
            return;
        }

        summaryEl.classList.remove('hidden');

        const active = embryos.filter(e => !e.isComplete).length;
        const completed = embryos.filter(e => e.isComplete).length;

        summaryEl.innerHTML = `
            <div class="summary-stat">
                <span class="stat-value">${this.state.totalTimepoints}</span>
                <span class="stat-label">Total Timepoints</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">${active}</span>
                <span class="stat-label">Active Embryos</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">${completed}</span>
                <span class="stat-label">Completed</span>
            </div>
            ${this.state.startedAt ? `
            <div class="summary-stat">
                <span class="stat-value" id="summary-duration">${this.formatDuration(Date.now() - this.state.startedAt.getTime())}</span>
                <span class="stat-label">Duration</span>
            </div>
            ` : ''}
        `;
    },

    updateSummary() {
        // Quick update just for the duration
        const durationEl = document.getElementById('summary-duration');
        if (durationEl && this.state.startedAt) {
            durationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }
        // Update stats
        this.renderSummary();
    },

    renderEmbryoCards() {
        const container = document.getElementById('embryo-cards');
        if (!container) return;

        const embryos = Object.values(this.state.embryos);

        if (embryos.length === 0) {
            container.innerHTML = '<div class="empty-state">No active timelapse tasks</div>';
            return;
        }

        // Sort: running first, then by embryo ID
        embryos.sort((a, b) => {
            if (a.isComplete !== b.isComplete) return a.isComplete ? 1 : -1;
            return a.embryoId.localeCompare(b.embryoId);
        });

        container.innerHTML = embryos.map(embryo => this.renderEmbryoCard(embryo)).join('');

        // Add click handlers for selection
        container.querySelectorAll('.embryo-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectEmbryo(card.dataset.embryoId);
            });
        });
    },

    // Full embryo card for sidebar (with selection support)
    renderEmbryoCard(embryo) {
        const status = embryo.isComplete ? 'complete' :
                       embryo.lastError ? 'error' :
                       this.state.status === 'PAUSED' ? 'paused' : 'running';

        const statusIcon = status === 'complete' ? '&#x2714;' :
                          status === 'error' ? '&#x2718;' :
                          status === 'paused' ? '&#x23F8;' : '&#x25CF;';

        const statusText = status === 'complete' ? 'Complete' :
                          status === 'error' ? 'Error' :
                          status === 'paused' ? 'Paused' : 'Running';

        const isSelected = this.selectedEmbryoId === embryo.embryoId;

        // Calculate progress percentage
        const maxTimepoints = this.getMaxTimepoints(embryo);
        const progressPct = maxTimepoints > 0 ? Math.min(100, (embryo.timepoints / maxTimepoints) * 100) : 0;

        // Reasoning count badge
        const reasoningCount = (this.detectionReasoning[embryo.embryoId] || []).length;
        const reasoningBadge = reasoningCount > 0 ?
            `<span class="reasoning-count" title="${reasoningCount} detection evaluations">${reasoningCount}</span>` : '';

        // Calculate countdown
        let countdownHtml = '';
        if (!embryo.isComplete && embryo.lastAcquired) {
            const secondsUntilNext = this.getSecondsUntilNext(embryo);
            countdownHtml = `
                <div class="next-acquisition">
                    <span class="next-label">Next in:</span>
                    <span class="mini-countdown" data-embryo="${embryo.embryoId}">${this.formatCountdown(secondsUntilNext)}</span>
                    <span class="interval-info">(every ${this.formatInterval(embryo.intervalSeconds)})</span>
                </div>
            `;
        } else if (!embryo.isComplete) {
            countdownHtml = `
                <div class="next-acquisition">
                    <span class="next-label">Waiting for first acquisition...</span>
                </div>
            `;
        }

        // Detection status
        let detectionsHtml = '';
        const detectorNames = Object.keys(embryo.detections);
        if (detectorNames.length > 0) {
            detectionsHtml = `
                <div class="detection-status">
                    ${detectorNames.map(name => {
                        const det = embryo.detections[name];
                        const detected = det.detected;
                        return `
                            <div class="detector-item ${detected ? 'detected' : ''}">
                                <span class="detector-icon">${detected ? '&#x2714;' : '&#x2022;'}</span>
                                <span>${this.formatDetectorName(name)}: ${detected ? 'Detected' : 'Not detected'}</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }

        // Completion or error info
        let completionHtml = '';
        if (embryo.isComplete) {
            const duration = embryo.lastAcquired && this.state.startedAt ?
                this.formatDuration(embryo.lastAcquired.getTime() - this.state.startedAt.getTime()) : '';
            completionHtml = `
                <div class="completion-info">
                    <div class="completion-reason">${embryo.completionReason || 'Completed'}</div>
                    ${duration ? `<div class="completion-duration">Duration: ${duration}</div>` : ''}
                </div>
            `;
        } else if (embryo.lastError) {
            completionHtml = `
                <div class="error-info">
                    <div class="error-message">${embryo.lastError}</div>
                </div>
            `;
        }

        return `
            <div class="embryo-card sidebar-card ${status} ${isSelected ? 'selected' : ''}" data-embryo-id="${embryo.embryoId}">
                <div class="embryo-header">
                    <div class="embryo-header-left">
                        <span class="embryo-name">${embryo.embryoId}</span>
                        ${reasoningBadge}
                    </div>
                    <span class="embryo-status ${status}">
                        <span class="embryo-status-icon">${statusIcon}</span>
                        ${statusText}
                    </span>
                </div>

                <div class="stop-condition">
                    <span class="condition-icon">${this.getConditionIcon(embryo.stopCondition)}</span>
                    <span class="condition-text">${this.formatStopCondition(embryo.stopCondition)}</span>
                </div>

                <div class="progress-section">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressPct}%"></div>
                    </div>
                    <div class="progress-stats">
                        <span class="timepoints">${embryo.timepoints}${maxTimepoints > 0 ? ' / ' + maxTimepoints : ''} timepoints</span>
                        <span class="elapsed">${embryo.lastAcquired ? this.formatDuration(Date.now() - (this.state.startedAt?.getTime() || Date.now())) : ''}</span>
                    </div>
                </div>

                ${countdownHtml}
                ${detectionsHtml}
                ${completionHtml}
            </div>
        `;
    },

    selectEmbryo(embryoId) {
        this.selectedEmbryoId = embryoId;

        // Update card selection styles
        document.querySelectorAll('.embryo-card').forEach(card => {
            card.classList.toggle('selected', card.dataset.embryoId === embryoId);
        });

        // Render reasoning panel
        this.renderReasoningPanel();
    },

    renderReasoningPanel() {
        const panel = document.getElementById('reasoning-panel');
        if (!panel) return;

        // If no embryo selected or no embryos at all
        if (!this.selectedEmbryoId || !this.state.embryos[this.selectedEmbryoId]) {
            panel.innerHTML = `
                <div class="reasoning-empty">
                    <div class="reasoning-empty-icon">&#x1F50D;</div>
                    <div class="reasoning-empty-text">Select an embryo to view detection history</div>
                </div>
            `;
            return;
        }

        const embryo = this.state.embryos[this.selectedEmbryoId];
        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];

        // Embryo info header
        const statusIcon = embryo.isComplete ? '&#x2714;' :
                          embryo.lastError ? '&#x2718;' : '&#x25CF;';
        const statusClass = embryo.isComplete ? 'complete' :
                           embryo.lastError ? 'error' : 'running';

        // Detection history (most recent first)
        const sortedReasoning = [...reasoning].reverse();

        let reasoningListHtml = '';
        if (sortedReasoning.length === 0) {
            reasoningListHtml = `
                <div class="no-detections">
                    <div class="no-detections-icon">&#x1F9EC;</div>
                    <div class="no-detections-text">No detection evaluations yet</div>
                    <div class="no-detections-hint">
                        Detector reasoning will appear here as each timepoint is analyzed.<br>
                        All evaluations are shown, regardless of detection result.
                    </div>
                </div>
            `;
        } else {
            reasoningListHtml = sortedReasoning.map((r, idx) => this.renderDetectionCard(r, idx)).join('');
        }

        panel.innerHTML = `
            <div class="reasoning-header">
                <div class="reasoning-embryo-info">
                    <span class="reasoning-status-dot ${statusClass}">${statusIcon}</span>
                    <span class="reasoning-embryo-name">${embryo.embryoId}</span>
                    <span class="reasoning-condition">${this.formatStopCondition(embryo.stopCondition)}</span>
                </div>
                <div class="reasoning-stats">
                    <span class="stat">${embryo.timepoints} timepoints</span>
                    <span class="stat">${reasoning.length} detections</span>
                </div>
            </div>
            <div class="reasoning-list" id="reasoning-list">
                ${reasoningListHtml}
            </div>
        `;
    },

    renderDetectionCard(detection, index) {
        const isExpanded = this.expandedImages[index] || false;
        // Use projection_uid (proper DataStore UID) with fallback to volume_uid
        const imageUid = detection.projection_uid || detection.volume_uid;
        const hasImage = !!imageUid;
        const timestamp = detection.timestamp ? new Date(detection.timestamp).toLocaleTimeString() : '';

        // Confidence styling
        const confidenceClass = detection.confidence ? detection.confidence.toLowerCase() : '';

        return `
            <div class="detection-card ${detection.detected ? 'detected' : ''}">
                <div class="detection-card-header">
                    <div class="detection-meta">
                        <span class="detector-badge">${this.formatDetectorName(detection.detector_name)}</span>
                        <span class="detection-result ${detection.detected ? 'positive' : 'negative'}">
                            ${detection.detected ? 'Detected' : 'Not detected'}
                        </span>
                        ${detection.confidence ? `<span class="confidence-badge ${confidenceClass}">${detection.confidence}</span>` : ''}
                    </div>
                    <div class="detection-timing">
                        <span class="timepoint-badge">TP ${detection.timepoint ?? '?'}</span>
                        <span class="detection-time">${timestamp}</span>
                    </div>
                </div>
                ${detection.reasoning ? `
                    <div class="detection-reasoning-text">
                        ${this.escapeHtml(detection.reasoning)}
                    </div>
                ` : ''}
                ${hasImage ? `
                    <div class="detection-image-section">
                        <button class="toggle-image-btn" onclick="TasksManager.toggleImage(${index}, '${imageUid}')">
                            <span class="toggle-icon">${isExpanded ? '&#x25BC;' : '&#x25B6;'}</span>
                            ${isExpanded ? 'Hide' : 'Show'} Volume Projection
                        </button>
                        <div class="detection-image-container ${isExpanded ? 'expanded' : ''}" id="detection-image-${index}">
                            ${isExpanded ? `<img src="/api/images/${imageUid}/png" alt="Volume projection" class="detection-image" />` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    },

    toggleImage(index, volumeUid) {
        this.expandedImages[index] = !this.expandedImages[index];

        const container = document.getElementById(`detection-image-${index}`);
        const btn = container?.previousElementSibling;

        if (container && btn) {
            if (this.expandedImages[index]) {
                container.classList.add('expanded');
                container.innerHTML = `<img src="/api/images/${volumeUid}/png" alt="Volume projection" class="detection-image" />`;
                btn.innerHTML = '<span class="toggle-icon">&#x25BC;</span> Hide Volume Projection';
            } else {
                container.classList.remove('expanded');
                container.innerHTML = '';
                btn.innerHTML = '<span class="toggle-icon">&#x25B6;</span> Show Volume Projection';
            }
        }
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    updateEmbryoCard(embryoId) {
        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        const card = document.querySelector(`.embryo-card[data-embryo-id="${embryoId}"]`);
        if (card) {
            // Re-render just this card, preserving selection state
            const wasSelected = card.classList.contains('selected');
            const newCard = document.createElement('div');
            newCard.innerHTML = this.renderEmbryoCard(embryo);
            const renderedCard = newCard.firstElementChild;

            if (wasSelected) {
                renderedCard.classList.add('selected');
            }

            card.replaceWith(renderedCard);

            // Re-add click handler
            renderedCard.addEventListener('click', () => {
                this.selectEmbryo(embryoId);
            });
        } else {
            // Card doesn't exist, re-render all
            this.renderEmbryoCards();
        }
    },

    // ==========================================
    // Countdown Updates
    // ==========================================

    startCountdownUpdates() {
        // Update countdowns every second
        this.countdownInterval = setInterval(() => {
            this.updateCountdowns();
        }, 1000);
    },

    updateCountdowns() {
        if (this.state.status !== 'RUNNING') return;

        // Update main duration
        const durationEl = document.getElementById('timelapse-duration');
        if (durationEl && this.state.startedAt) {
            durationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }

        // Update summary duration
        const summaryDurationEl = document.getElementById('summary-duration');
        if (summaryDurationEl && this.state.startedAt) {
            summaryDurationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }

        // Update per-embryo countdowns (compact cards use mini-countdown class)
        Object.values(this.state.embryos).forEach(embryo => {
            if (embryo.isComplete) return;

            const countdownEl = document.querySelector(`.mini-countdown[data-embryo="${embryo.embryoId}"]`);
            if (countdownEl) {
                const seconds = this.getSecondsUntilNext(embryo);
                countdownEl.textContent = this.formatCountdown(seconds);
            }
        });
    },

    // ==========================================
    // Helpers
    // ==========================================

    getSecondsUntilNext(embryo) {
        if (!embryo.lastAcquired) return embryo.intervalSeconds;
        const elapsed = (Date.now() - embryo.lastAcquired.getTime()) / 1000;
        return Math.max(0, embryo.intervalSeconds - elapsed);
    },

    getMaxTimepoints(embryo) {
        // Try to parse max from stop condition
        const condition = embryo.stopCondition.toLowerCase();
        const match = condition.match(/(\d+)\s*timepoints?/);
        if (match) return parseInt(match[1]);

        // Duration-based: estimate based on interval
        const durationMatch = condition.match(/(\d+)\s*h(?:ours?)?/);
        if (durationMatch && embryo.intervalSeconds) {
            const hours = parseInt(durationMatch[1]);
            return Math.ceil((hours * 3600) / embryo.intervalSeconds);
        }

        return 0; // Unknown
    },

    formatStopCondition(condition) {
        if (!condition) return 'Manual stop';

        const c = condition.toLowerCase();

        // Handle composite conditions
        if (c.includes('|')) {
            const parts = c.split('|').map(p => this.formatSingleCondition(p.trim()));
            return parts.join(' OR ');
        }

        return this.formatSingleCondition(c);
    },

    formatSingleCondition(condition) {
        if (condition === 'manual') return 'Manual stop only';
        if (condition.includes('hatching')) return 'Until hatching';
        if (condition.includes('comma')) return 'Until comma stage';

        // Parse "duration:10h" or "10h" or "10 hours"
        const durationMatch = condition.match(/(?:duration:)?(\d+)\s*h(?:ours?)?/);
        if (durationMatch) return `${durationMatch[1]} hours`;

        // Parse "fixed_timepoints:100" or "100 timepoints"
        const tpMatch = condition.match(/(?:fixed_timepoints:)?(\d+)\s*(?:timepoints?)?/);
        if (tpMatch && !durationMatch) return `${tpMatch[1]} timepoints`;

        return condition;
    },

    getConditionIcon(condition) {
        if (!condition) return '&#x270B;'; // Hand

        const c = condition.toLowerCase();
        if (c.includes('|')) return '&#x1F3AF;'; // Target (composite)
        if (c.includes('hatching')) return '&#x1F423;'; // Hatching chick
        if (c.includes('comma')) return '&#x1F52C;'; // Microscope
        if (c.includes('timepoint')) return '&#x1F522;'; // Numbers
        if (c.includes('duration') || c.includes('hour')) return '&#x23F1;'; // Stopwatch
        return '&#x270B;'; // Hand (manual)
    },

    formatDetectorName(name) {
        return name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' ');
    },

    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    },

    formatCountdown(seconds) {
        if (seconds <= 0) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    },

    formatInterval(seconds) {
        if (seconds >= 3600) {
            return `${(seconds / 3600).toFixed(1)}h`;
        } else if (seconds >= 60) {
            return `${Math.round(seconds / 60)} min`;
        } else {
            return `${seconds}s`;
        }
    },

    updateTasksCount() {
        const badge = document.getElementById('tasks-count');
        if (badge) {
            const activeCount = Object.values(this.state.embryos).filter(e => !e.isComplete).length;
            badge.textContent = activeCount;
        }
    },

    // Reset state (e.g., when starting fresh)
    reset() {
        this.state = {
            status: 'IDLE',
            startedAt: null,
            embryos: {},
            totalTimepoints: 0,
            baseInterval: 120
        };
        this.updateTasksCount();
        this.render();
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    TasksManager.init();
});
