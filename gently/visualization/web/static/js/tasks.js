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

    // Expanded reasoning states (for collapsible reasoning text)
    expandedReasoning: {},  // detection index -> true/false

    // Expanded range states (for collapsed timepoint ranges)
    expandedRanges: {},  // range key -> true/false

    // Filter state for detection panel
    detectionFilter: 'all',  // 'all', 'detections', 'high-confidence'

    // Number of items to show in expanded ranges
    rangeLoadLimit: 10,  // Initial items to show
    rangeLoadMore: {},  // range key -> number of items loaded

    countdownInterval: null,
    storageKey: 'gently-tasks-state',

    init() {
        // Restore state from localStorage
        this.loadState();
        // Load detection agreements
        this.loadAgreements();
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
        this.expandedReasoning = {};
        this.expandedRanges = {};
        this.detectionFilter = 'all';
        this.rangeLoadMore = {};
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

        // Detection counts for badge
        const reasoning = this.detectionReasoning[embryo.embryoId] || [];
        const positiveCount = reasoning.filter(r => r.detected).length;
        const totalCount = reasoning.length;

        // Show positive detections prominently, or total evaluations if none
        let detectionBadge = '';
        if (positiveCount > 0) {
            detectionBadge = `<span class="reasoning-count" style="background: var(--accent-green);" title="${positiveCount} positive detection${positiveCount > 1 ? 's' : ''}">${positiveCount} detected</span>`;
        } else if (totalCount > 0) {
            detectionBadge = `<span class="reasoning-count" title="${totalCount} detection evaluations">${totalCount}</span>`;
        }

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

        // Play button for viewing all timepoints as video
        const hasTimepoints = embryo.timepoints > 0;
        const playButton = hasTimepoints ? `
            <button class="embryo-play-btn" onclick="event.stopPropagation(); TasksManager.playEmbryoTimelapse('${embryo.embryoId}')" title="Play all timepoints">
                <span class="play-icon">▶</span>
            </button>
        ` : '';

        return `
            <div class="embryo-card sidebar-card ${status} ${isSelected ? 'selected' : ''}" data-embryo-id="${embryo.embryoId}">
                <div class="embryo-header">
                    <div class="embryo-header-left">
                        <span class="embryo-name">${embryo.embryoId}</span>
                        ${detectionBadge}
                        ${playButton}
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

        // Calculate statistics
        const totalEvaluations = reasoning.length;
        const positiveDetections = reasoning.filter(r => r.detected);
        const highConfidence = reasoning.filter(r => r.confidence?.toLowerCase() === 'high');

        // Embryo info header
        const statusIcon = embryo.isComplete ? '&#x2714;' :
                          embryo.lastError ? '&#x2718;' : '&#x25CF;';
        const statusClass = embryo.isComplete ? 'complete' :
                           embryo.lastError ? 'error' : 'running';

        // Build quick jump badges for positive detections
        const quickJumpsHtml = positiveDetections.length > 0
            ? positiveDetections.map(d => `
                <span class="quick-jump-badge" onclick="TasksManager.scrollToDetection(${d.timepoint}, '${d.detector_name}')" title="Jump to detection">
                    <span class="detector-icon">${this.getDetectorIcon(d.detector_name)}</span>
                    ${this.formatDetectorName(d.detector_name)} @ T${d.timepoint}
                </span>
            `).join('')
            : '<span style="font-size: 0.8rem; color: var(--text-muted);">No positive detections yet</span>';

        // Build filter buttons with counts
        const filterButtonsHtml = `
            <div class="detection-filter-group">
                <button class="filter-btn ${this.detectionFilter === 'all' ? 'active' : ''}"
                        onclick="TasksManager.setDetectionFilter('all')">
                    All<span class="count">${totalEvaluations}</span>
                </button>
                <button class="filter-btn ${this.detectionFilter === 'detections' ? 'active' : ''}"
                        onclick="TasksManager.setDetectionFilter('detections')">
                    Detected<span class="count">${positiveDetections.length}</span>
                </button>
                <button class="filter-btn ${this.detectionFilter === 'high-confidence' ? 'active' : ''}"
                        onclick="TasksManager.setDetectionFilter('high-confidence')">
                    High Conf<span class="count">${highConfidence.length}</span>
                </button>
            </div>
        `;

        // Build detection list based on filter
        let detectionListHtml = '';
        if (totalEvaluations === 0) {
            detectionListHtml = `
                <div class="no-detections">
                    <div class="no-detections-icon">&#x1F9EC;</div>
                    <div class="no-detections-text">No detection evaluations yet</div>
                    <div class="no-detections-hint">
                        Detector reasoning will appear here as each timepoint is analyzed.
                    </div>
                </div>
            `;
        } else {
            detectionListHtml = this.renderDetectionListWithCollapse(reasoning);
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
                    <span class="stat">${totalEvaluations} evaluations</span>
                </div>
            </div>
            <div class="detection-summary-strip">
                <div class="detection-summary-stat">
                    <span class="stat-value ${positiveDetections.length > 0 ? 'has-detections' : ''}">${positiveDetections.length}</span>
                    <span class="stat-label">Detections</span>
                </div>
                <div class="detection-summary-stat">
                    <span class="stat-value">${totalEvaluations}</span>
                    <span class="stat-label">Evaluations</span>
                </div>
                <div class="detection-quick-jumps">
                    ${quickJumpsHtml}
                </div>
            </div>
            ${totalEvaluations > 0 ? this.renderTimelineSparkline(reasoning, embryo.timepoints) : ''}
            <div class="detection-controls">
                ${filterButtonsHtml}
            </div>
            <div class="detection-list-container" id="detection-list-container">
                ${detectionListHtml}
            </div>
        `;
    },

    // Render detections with range collapse for "not detected" sequences
    renderDetectionListWithCollapse(reasoning) {
        // Sort by timepoint descending (newest first)
        const sorted = [...reasoning].sort((a, b) => (b.timepoint ?? 0) - (a.timepoint ?? 0));

        // Apply filter
        let filtered = sorted;
        if (this.detectionFilter === 'detections') {
            filtered = sorted.filter(r => r.detected);
        } else if (this.detectionFilter === 'high-confidence') {
            filtered = sorted.filter(r => r.confidence?.toLowerCase() === 'high');
        }

        // If showing only detections, render them directly without collapse
        if (this.detectionFilter !== 'all') {
            if (filtered.length === 0) {
                return `
                    <div class="detection-empty-filtered">
                        <div class="icon">&#x1F50E;</div>
                        <div class="message">No ${this.detectionFilter === 'detections' ? 'positive detections' : 'high confidence evaluations'} found</div>
                        <div class="hint">Try selecting "All" to see all evaluations</div>
                    </div>
                `;
            }
            return filtered.map((r, idx) => this.renderDetectionCard(r, idx, true)).join('');
        }

        // For "all" view, use range collapse
        // Group consecutive "not detected" items, but always show "detected" items expanded
        const groups = this.groupDetectionsForCollapse(sorted);

        let html = '';

        // First, render positive detections section if any exist
        const positiveDetections = sorted.filter(r => r.detected);
        if (positiveDetections.length > 0) {
            html += `
                <div class="positive-detections-section">
                    <div class="section-header">
                        <span class="icon">&#x2714;</span>
                        <span class="title">Positive Detections</span>
                        <span class="count">${positiveDetections.length} found</span>
                    </div>
                    ${positiveDetections.map((r, idx) => this.renderDetectionCard(r, `pos-${idx}`, true)).join('')}
                </div>
            `;
        }

        // Then render the timeline with collapsed ranges
        html += `<div class="section-header" style="margin-top: 1rem;">
            <span class="icon">&#x1F4C5;</span>
            <span class="title" style="color: var(--text-muted);">Evaluation Timeline</span>
            <span class="count">${sorted.length} total</span>
        </div>`;

        groups.forEach((group, groupIdx) => {
            if (group.type === 'positive') {
                // Already rendered above in positive section, just show a marker
                html += `
                    <div class="detection-card positive-highlight" style="padding: 0.75rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="color: var(--accent-green);">&#x2714;</span>
                            <span class="timepoint-badge">T${group.items[0].timepoint}</span>
                            <span class="detector-badge">${this.formatDetectorName(group.items[0].detector_name)}</span>
                            <span style="color: var(--accent-green); font-weight: 500;">DETECTED</span>
                        </div>
                    </div>
                `;
            } else {
                // Collapsed range of "not detected" items
                const rangeKey = `range-${groupIdx}`;
                const isExpanded = this.expandedRanges[rangeKey] || false;
                const items = group.items;
                const startTp = items[items.length - 1].timepoint ?? '?';
                const endTp = items[0].timepoint ?? '?';
                const rangeLabel = items.length === 1
                    ? `Timepoint ${startTp}`
                    : `Timepoints ${endTp} - ${startTp}`;

                html += `
                    <div class="collapsed-range ${isExpanded ? 'expanded' : ''}" onclick="TasksManager.toggleRange('${rangeKey}')">
                        <span class="range-indicator"></span>
                        <span class="range-label"><strong>${rangeLabel}</strong>: No detections</span>
                        <span class="range-count">${items.length} evaluation${items.length > 1 ? 's' : ''}</span>
                        <span class="range-chevron">&#x25BC;</span>
                    </div>
                    <div class="range-expansion ${isExpanded ? 'expanded' : ''}" id="${rangeKey}">
                        <div class="range-expansion-inner">
                            ${this.renderRangeItems(items, rangeKey)}
                        </div>
                    </div>
                `;
            }
        });

        return html;
    },

    // Group detections into positive singles and negative ranges
    groupDetectionsForCollapse(sorted) {
        const groups = [];
        let currentNegatives = [];

        sorted.forEach(item => {
            if (item.detected) {
                // Flush any accumulated negatives
                if (currentNegatives.length > 0) {
                    groups.push({ type: 'negative', items: currentNegatives });
                    currentNegatives = [];
                }
                // Add positive as its own group
                groups.push({ type: 'positive', items: [item] });
            } else {
                currentNegatives.push(item);
            }
        });

        // Flush remaining negatives
        if (currentNegatives.length > 0) {
            groups.push({ type: 'negative', items: currentNegatives });
        }

        return groups;
    },

    // Render items inside an expanded range
    renderRangeItems(items, rangeKey) {
        const loadedCount = this.rangeLoadMore[rangeKey] || this.rangeLoadLimit;
        const visibleItems = items.slice(0, loadedCount);
        const hasMore = items.length > loadedCount;

        let html = visibleItems.map(item => `
            <div class="detection-row-compact" onclick="TasksManager.showDetectionDetail('${item.detector_name}', ${item.timepoint})">
                <span class="tp-badge">T${item.timepoint ?? '?'}</span>
                <span class="detector-name">${this.formatDetectorName(item.detector_name)}</span>
                <span class="result">Not detected</span>
                ${item.confidence ? `<span class="confidence ${item.confidence.toLowerCase()}">${item.confidence}</span>` : ''}
            </div>
        `).join('');

        if (hasMore) {
            html += `
                <button class="load-more-btn" onclick="event.stopPropagation(); TasksManager.loadMoreInRange('${rangeKey}', ${items.length})">
                    Load more (${items.length - loadedCount} remaining)
                </button>
            `;
        }

        return html;
    },

    // Toggle a collapsed range
    toggleRange(rangeKey) {
        this.expandedRanges[rangeKey] = !this.expandedRanges[rangeKey];

        const rangeHeader = document.querySelector(`.collapsed-range[onclick*="${rangeKey}"]`);
        const expansion = document.getElementById(rangeKey);

        if (rangeHeader && expansion) {
            rangeHeader.classList.toggle('expanded', this.expandedRanges[rangeKey]);
            expansion.classList.toggle('expanded', this.expandedRanges[rangeKey]);
        }
    },

    // Load more items in a range
    loadMoreInRange(rangeKey, totalItems) {
        const current = this.rangeLoadMore[rangeKey] || this.rangeLoadLimit;
        this.rangeLoadMore[rangeKey] = Math.min(current + this.rangeLoadLimit, totalItems);
        this.renderReasoningPanel();
    },

    // Set detection filter
    setDetectionFilter(filter) {
        this.detectionFilter = filter;
        this.renderReasoningPanel();
    },

    // Scroll to a specific detection
    scrollToDetection(timepoint, detectorName) {
        // Set filter to "all" to ensure the detection is visible
        this.detectionFilter = 'all';
        this.renderReasoningPanel();

        // Find and scroll to the detection card
        setTimeout(() => {
            const container = document.getElementById('detection-list-container');
            if (container) {
                // Look for the positive detection card with matching timepoint
                const cards = container.querySelectorAll('.detection-card.positive-highlight');
                cards.forEach(card => {
                    if (card.textContent.includes(`T${timepoint}`) && card.textContent.includes(this.formatDetectorName(detectorName))) {
                        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        // Add a brief highlight effect
                        card.style.animation = 'none';
                        card.offsetHeight; // Trigger reflow
                        card.style.animation = 'highlightPulse 1s ease-out';
                    }
                });
            }
        }, 100);
    },

    // Show detail for a compact row (placeholder for future enhancement)
    showDetectionDetail(detectorName, timepoint) {
        console.log(`Show detail for ${detectorName} at timepoint ${timepoint}`);
        // Could open a modal or expand inline - for now, just log
    },

    // Get icon for detector type
    getDetectorIcon(detectorName) {
        const name = detectorName?.toLowerCase() || '';
        if (name.includes('hatching')) return '&#x1F423;';
        if (name.includes('comma')) return '&#x1F52C;';
        if (name.includes('twofold')) return '&#x1F9EC;';
        return '&#x1F50D;';
    },

    renderDetectionCard(detection, index, isPositiveHighlight = false) {
        const imageExpanded = this.expandedImages[index] || false;
        const reasoningExpanded = this.expandedReasoning[index] || false;
        // Use projection_uid (proper DataStore UID) with fallback to volume_uid
        const imageUid = detection.projection_uid || detection.volume_uid;
        const hasImage = !!imageUid;
        const timestamp = detection.timestamp ? new Date(detection.timestamp).toLocaleTimeString() : '';

        // Confidence styling
        const confidenceClass = detection.confidence ? detection.confidence.toLowerCase() : '';

        // For positive detections, show context (confidence trend from previous timepoints)
        let contextHtml = '';
        if (detection.detected && isPositiveHighlight) {
            contextHtml = this.renderDetectionContext(detection);
        }

        // Card class based on detection status
        const cardClass = detection.detected
            ? (isPositiveHighlight ? 'detection-card positive-highlight' : 'detection-card detected')
            : 'detection-card';

        // Reasoning section - collapsible for non-detected, always visible for detected
        // Use linkifyTimepoints to make timepoint references clickable for video playback
        let reasoningHtml = '';
        if (detection.reasoning) {
            // Build context for the linkifier
            const linkContext = {
                detectionPoint: detection.detected ? detection.timepoint : null,
                reasoningText: detection.reasoning
            };
            const embryoId = this.selectedEmbryoId || '';
            const linkedReasoning = this.linkifyTimepoints(detection.reasoning, embryoId, linkContext);

            if (detection.detected) {
                // For positive detections, always show reasoning with clickable timepoints
                reasoningHtml = `
                    <div class="detection-reasoning-text">
                        ${linkedReasoning}
                    </div>
                `;
            } else {
                // For negative detections, make reasoning collapsible
                reasoningHtml = `
                    <button class="reasoning-toggle ${reasoningExpanded ? 'expanded' : ''}"
                            onclick="event.stopPropagation(); TasksManager.toggleReasoning('${index}')">
                        <span class="chevron">&#x25B6;</span>
                        ${reasoningExpanded ? 'Hide' : 'Show'} VLM reasoning
                    </button>
                    <div class="reasoning-content ${reasoningExpanded ? 'expanded' : ''}" id="reasoning-${index}">
                        <div class="detection-reasoning-text">
                            ${linkedReasoning}
                        </div>
                    </div>
                `;
            }
        }

        return `
            <div class="${cardClass}" data-timepoint="${detection.timepoint}" data-detector="${detection.detector_name}">
                <div class="detection-card-header">
                    <div class="detection-meta">
                        <span class="detector-badge">${this.formatDetectorName(detection.detector_name)}</span>
                        <span class="detection-result ${detection.detected ? 'positive' : 'negative'}">
                            ${detection.detected ? 'Detected' : 'Not detected'}
                        </span>
                        ${detection.confidence ? `<span class="confidence-badge ${confidenceClass}">${detection.confidence}</span>` : ''}
                    </div>
                    <div class="detection-timing">
                        <span class="timepoint-badge">T${detection.timepoint ?? '?'}</span>
                        <span class="detection-time">${timestamp}</span>
                    </div>
                </div>
                ${contextHtml}
                ${reasoningHtml}
                ${hasImage ? `
                    <div class="detection-image-section">
                        <button class="toggle-image-btn" onclick="event.stopPropagation(); TasksManager.toggleImage('${index}', '${imageUid}')">
                            <span class="toggle-icon">${imageExpanded ? '&#x25BC;' : '&#x25B6;'}</span>
                            ${imageExpanded ? 'Hide' : 'Show'} Volume Projection
                        </button>
                        <div class="detection-image-container ${imageExpanded ? 'expanded' : ''}" id="detection-image-${index}">
                            ${imageExpanded ? `<img src="/api/images/${imageUid}/png" alt="Volume projection" class="detection-image" />` : ''}
                        </div>
                    </div>
                ` : ''}
                ${detection.detected ? this.renderAgreeDisagreeButtons(detection, index) : ''}
            </div>
        `;
    },

    // Render agree/disagree buttons for detection feedback
    renderAgreeDisagreeButtons(detection, index) {
        const agreement = this.detectionAgreements[`${detection.detector_name}-${detection.timepoint}`];
        const agreedClass = agreement === true ? 'agreed' : '';
        const disagreedClass = agreement === false ? 'disagreed' : '';

        return `
            <div class="vlm-actions">
                <button class="vlm-action-btn agree ${agreedClass}"
                        onclick="event.stopPropagation(); TasksManager.markAgreement('${detection.detector_name}', ${detection.timepoint}, true)">
                    ${agreement === true ? '&#x2714; Agreed' : 'I Agree'}
                </button>
                <button class="vlm-action-btn disagree ${disagreedClass}"
                        onclick="event.stopPropagation(); TasksManager.markAgreement('${detection.detector_name}', ${detection.timepoint}, false)">
                    ${agreement === false ? '&#x2718; Disagreed' : 'I Disagree'}
                </button>
                <button class="vlm-action-btn" onclick="event.stopPropagation(); TasksManager.compareDetection('${detection.detector_name}', ${detection.timepoint})">
                    Compare
                </button>
            </div>
        `;
    },

    // Track user agreement/disagreement with detections
    detectionAgreements: {},  // key: "{detector}-{timepoint}" -> true/false

    markAgreement(detectorName, timepoint, agrees) {
        const key = `${detectorName}-${timepoint}`;
        const current = this.detectionAgreements[key];

        // Toggle off if clicking same button
        if (current === agrees) {
            delete this.detectionAgreements[key];
        } else {
            this.detectionAgreements[key] = agrees;
        }

        // Save to localStorage
        try {
            localStorage.setItem('gently-detection-agreements', JSON.stringify(this.detectionAgreements));
        } catch (e) {
            console.warn('Failed to save detection agreements:', e);
        }

        // Log for potential future analytics
        console.log(`Detection feedback: ${detectorName} at T${timepoint} - ${agrees ? 'agreed' : 'disagreed'}`);

        // Re-render the panel to update button states
        this.renderReasoningPanel();
    },

    // Load saved agreements
    loadAgreements() {
        try {
            const saved = localStorage.getItem('gently-detection-agreements');
            if (saved) {
                this.detectionAgreements = JSON.parse(saved);
            }
        } catch (e) {
            console.warn('Failed to load detection agreements:', e);
        }
    },

    // Compare detection to previous timepoint (placeholder for future enhancement)
    compareDetection(detectorName, timepoint) {
        console.log(`Compare detection: ${detectorName} at T${timepoint} vs T${timepoint - 1}`);
        // TODO: Open comparison modal showing current vs previous timepoint
        // For now, just show a notification
        const msg = `Comparison view coming soon. This will show ${detectorName} at T${timepoint} vs T${timepoint - 1}`;
        alert(msg);
    },

    // Render a timeline sparkline showing detection distribution
    renderTimelineSparkline(reasoning, totalTimepoints) {
        if (!reasoning || reasoning.length === 0 || totalTimepoints <= 0) {
            return '';
        }

        // Find max timepoint for scaling
        const maxTp = Math.max(totalTimepoints, ...reasoning.map(r => r.timepoint || 0));

        // Group by timepoint and determine status
        const timepoints = {};
        reasoning.forEach(r => {
            const tp = r.timepoint ?? 0;
            if (!timepoints[tp]) {
                timepoints[tp] = { detected: false, confidence: 'low' };
            }
            if (r.detected) {
                timepoints[tp].detected = true;
            }
            // Keep highest confidence
            const confOrder = { 'high': 3, 'medium': 2, 'low': 1 };
            const existingConf = confOrder[timepoints[tp].confidence] || 0;
            const newConf = confOrder[r.confidence?.toLowerCase()] || 0;
            if (newConf > existingConf) {
                timepoints[tp].confidence = r.confidence?.toLowerCase() || 'low';
            }
        });

        // Generate timeline points
        let pointsHtml = '';
        Object.entries(timepoints).forEach(([tp, data]) => {
            const position = (parseInt(tp) / maxTp) * 100;
            const isPositive = data.detected;
            const className = isPositive ? 'positive' : '';
            const title = isPositive
                ? `T${tp}: DETECTED`
                : `T${tp}: Not detected (${data.confidence})`;

            pointsHtml += `<div class="timeline-point ${className}"
                               style="left: ${position}%"
                               title="${title}"
                               onclick="TasksManager.scrollToDetection(${tp}, '')"></div>`;
        });

        // Labels for timeline
        const midTp = Math.floor(maxTp / 2);

        return `
            <div class="detection-timeline">
                <div class="timeline-track">
                    ${pointsHtml}
                </div>
                <div class="timeline-labels">
                    <span>T1</span>
                    <span>T${midTp}</span>
                    <span>T${maxTp}</span>
                </div>
            </div>
        `;
    },

    // Render context showing confidence trend leading up to a positive detection
    renderDetectionContext(detection) {
        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        const detectorName = detection.detector_name;
        const timepoint = detection.timepoint;

        // Get previous evaluations for same detector
        const previousEvals = reasoning
            .filter(r => r.detector_name === detectorName && (r.timepoint ?? 0) < timepoint)
            .sort((a, b) => (b.timepoint ?? 0) - (a.timepoint ?? 0))
            .slice(0, 3);  // Last 3 before detection

        if (previousEvals.length === 0) return '';

        const dots = previousEvals.reverse().map(e => {
            const conf = e.confidence?.toLowerCase() || 'low';
            return `<span class="context-dot ${conf}" title="T${e.timepoint}: ${e.confidence || 'Unknown'}"></span>`;
        }).join('');

        return `
            <div class="detection-context">
                <span>Confidence trend:</span>
                <div class="context-dots">${dots}</div>
                <span style="color: var(--accent-green);">&#x2714;</span>
                <span style="font-size: 0.7rem; color: var(--text-muted);">(${previousEvals.length} prior evaluations)</span>
            </div>
        `;
    },

    // Toggle reasoning visibility for a detection
    toggleReasoning(index) {
        this.expandedReasoning[index] = !this.expandedReasoning[index];

        const toggle = document.querySelector(`.reasoning-toggle[onclick*="'${index}'"]`);
        const content = document.getElementById(`reasoning-${index}`);

        if (toggle && content) {
            toggle.classList.toggle('expanded', this.expandedReasoning[index]);
            content.classList.toggle('expanded', this.expandedReasoning[index]);
            toggle.innerHTML = `
                <span class="chevron">&#x25B6;</span>
                ${this.expandedReasoning[index] ? 'Hide' : 'Show'} VLM reasoning
            `;
        }
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

    /**
     * Parse VLM reasoning text and convert timepoint references to clickable links
     * that trigger video playback when clicked.
     *
     * Recognizes patterns like:
     * - "timepoint 73", "timepoints 73-81"
     * - "T73", "T73-81", "T73-T81"
     * - "t=73"
     *
     * @param {string} text - The reasoning text to parse
     * @param {string} embryoId - The embryo ID for video playback
     * @param {object} context - Optional context {detectionPoint, detectorName}
     * @returns {string} HTML with clickable timepoint links
     */
    linkifyTimepoints(text, embryoId, context = {}) {
        if (!text) return '';

        // First escape HTML to prevent XSS
        let html = this.escapeHtml(text);

        // Pattern for "timepoint 73" or "timepoints 73-81" or "timepoints 73 - 81"
        html = html.replace(
            /timepoints?\s+(\d+)(?:\s*[-–]\s*(\d+))?/gi,
            (match, start, end) => {
                const startTp = parseInt(start);
                const endTp = end ? parseInt(end) : startTp;
                return this.createTimepointLink(match, embryoId, startTp, endTp, context);
            }
        );

        // Pattern for "T73" or "T73-81" or "T73-T81"
        html = html.replace(
            /(?<![a-zA-Z0-9])T(\d+)(?:\s*[-–]\s*T?(\d+))?(?![a-zA-Z0-9])/gi,
            (match, start, end) => {
                const startTp = parseInt(start);
                const endTp = end ? parseInt(end) : startTp;
                return this.createTimepointLink(match, embryoId, startTp, endTp, context);
            }
        );

        // Pattern for "t=73" (common in some scientific writing)
        html = html.replace(
            /t\s*=\s*(\d+)/gi,
            (match, tp) => {
                const timepoint = parseInt(tp);
                return this.createTimepointLink(match, embryoId, timepoint, timepoint, context);
            }
        );

        return html;
    },

    /**
     * Create a clickable timepoint link element
     */
    createTimepointLink(text, embryoId, start, end, context = {}) {
        const { detectionPoint, reasoningText } = context;

        // Build the onclick handler parameters
        const params = {
            embryoId,
            start,
            end,
            detectionPoint: detectionPoint ?? null,
            reasoningText: reasoningText ?? null
        };

        // Escape for use in onclick attribute
        const paramsStr = JSON.stringify(params).replace(/'/g, "\\'").replace(/"/g, '&quot;');

        return `<span class="timepoint-link"
                      data-start="${start}"
                      data-end="${end}"
                      data-embryo="${embryoId}"
                      onclick="TasksManager.playTimepointRange(${paramsStr})"
                      title="Click to play T${start}${end !== start ? '-T' + end : ''}">${text}</span>`;
    },

    /**
     * Handle click on a timepoint link - opens video player
     */
    async playTimepointRange(params) {
        const { embryoId, start, end, detectionPoint, reasoningText } = params;

        if (typeof TimepointPlayer !== 'undefined') {
            await TimepointPlayer.openSequence(embryoId, start, end, {
                vlmRange: { start, end },
                detectionPoint,
                reasoningText,
                bufferPercent: 0.2  // 20% buffer on each side
            });
        } else {
            console.warn('TimepointPlayer not available');
        }
    },

    /**
     * Play all timepoints for an embryo as a video timelapse
     */
    async playEmbryoTimelapse(embryoId) {
        if (typeof TimepointPlayer !== 'undefined') {
            // Get detection info if available
            const reasoning = this.detectionReasoning[embryoId] || [];
            const positiveDetections = reasoning.filter(r => r.detected);
            const latestDetection = positiveDetections.length > 0
                ? positiveDetections[positiveDetections.length - 1]
                : null;

            await TimepointPlayer.openSequence(embryoId, 0, null, {
                vlmRange: null,  // No specific VLM range for "play all"
                detectionPoint: latestDetection?.timepoint ?? null,
                reasoningText: latestDetection?.reasoning ?? null,
                bufferPercent: 0  // No buffer for "play all"
            });

            // Auto-play when opening from gallery
            TimepointPlayer.play();
        } else {
            console.warn('TimepointPlayer not available');
        }
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

/**
 * Global Experiment Status Strip Manager
 * Shows persistent experiment status at a glance
 */
const ExperimentStrip = {
    lastCheck: null,  // Track when user last viewed
    newDetections: [],  // Detections since last check

    init() {
        // Load last check time from localStorage
        const saved = localStorage.getItem('gently-last-check');
        if (saved) {
            this.lastCheck = new Date(saved);
        }
        this.update();
    },

    update() {
        const strip = document.getElementById('experiment-strip');
        if (!strip) return;

        const state = TasksManager.state;
        const embryoCount = Object.keys(state.embryos).length;

        // Show/hide strip based on whether there's an experiment
        if (embryoCount === 0 && state.status === 'IDLE') {
            strip.classList.add('hidden');
            return;
        }
        strip.classList.remove('hidden');

        // Update status indicator
        const indicator = document.getElementById('strip-indicator');
        const statusText = document.getElementById('strip-status');
        if (indicator && statusText) {
            indicator.className = 'strip-indicator ' + state.status.toLowerCase();
            statusText.textContent = this.formatStatus(state.status);
        }

        // Update duration
        const durationEl = document.getElementById('strip-duration');
        if (durationEl && state.startedAt) {
            durationEl.textContent = TasksManager.formatDuration(Date.now() - state.startedAt.getTime());
        }

        // Update embryo count
        const embryosEl = document.getElementById('strip-embryos');
        if (embryosEl) {
            const activeCount = Object.values(state.embryos).filter(e => !e.isComplete).length;
            const totalCount = embryoCount;
            embryosEl.textContent = `${activeCount}/${totalCount}`;
        }

        // Update next countdown
        const countdownEl = document.getElementById('strip-countdown');
        if (countdownEl) {
            const nextSeconds = this.getNextAcquisitionSeconds();
            countdownEl.textContent = nextSeconds > 0 ? TasksManager.formatCountdown(nextSeconds) : '--:--';
        }

        // Update detection alert
        this.updateDetectionAlert();
    },

    getNextAcquisitionSeconds() {
        const embryos = Object.values(TasksManager.state.embryos);
        const activeEmbryos = embryos.filter(e => !e.isComplete);
        if (activeEmbryos.length === 0) return 0;

        // Find the next acquisition (minimum countdown)
        let minSeconds = Infinity;
        activeEmbryos.forEach(embryo => {
            if (embryo.lastAcquired) {
                const elapsed = (Date.now() - new Date(embryo.lastAcquired).getTime()) / 1000;
                const remaining = Math.max(0, embryo.intervalSeconds - elapsed);
                minSeconds = Math.min(minSeconds, remaining);
            }
        });

        return minSeconds === Infinity ? 0 : Math.floor(minSeconds);
    },

    formatStatus(status) {
        switch (status) {
            case 'RUNNING': return 'Running';
            case 'PAUSED': return 'Paused';
            case 'COMPLETED': return 'Completed';
            case 'FAILED': return 'Failed';
            default: return 'Idle';
        }
    },

    updateDetectionAlert() {
        const alert = document.getElementById('strip-alert');
        if (!alert) return;

        // Count positive detections across all embryos
        let totalDetections = 0;
        let latestDetection = null;

        Object.entries(TasksManager.detectionReasoning).forEach(([embryoId, reasoning]) => {
            const positives = reasoning.filter(r => r.detected);
            totalDetections += positives.length;

            positives.forEach(d => {
                if (!latestDetection || new Date(d.timestamp) > new Date(latestDetection.timestamp)) {
                    latestDetection = { ...d, embryoId };
                }
            });
        });

        if (totalDetections === 0) {
            alert.classList.add('hidden');
            return;
        }

        alert.classList.remove('hidden');
        alert.classList.toggle('success', totalDetections > 0);

        const badge = document.getElementById('strip-alert-badge');
        const text = document.getElementById('strip-alert-text');

        if (badge) badge.textContent = totalDetections;
        if (text && latestDetection) {
            text.textContent = `${TasksManager.formatDetectorName(latestDetection.detector_name)} detected`;
        }
    },

    handleAlertClick() {
        // Switch to Tasks tab and select the embryo with the latest detection
        switchTab('tasks');

        // Find embryo with most recent detection
        let latestDetection = null;
        let latestEmbryoId = null;

        Object.entries(TasksManager.detectionReasoning).forEach(([embryoId, reasoning]) => {
            const positives = reasoning.filter(r => r.detected);
            positives.forEach(d => {
                if (!latestDetection || new Date(d.timestamp) > new Date(latestDetection.timestamp)) {
                    latestDetection = d;
                    latestEmbryoId = embryoId;
                }
            });
        });

        if (latestEmbryoId) {
            TasksManager.selectEmbryo(latestEmbryoId);
        }
    },

    markChecked() {
        this.lastCheck = new Date();
        localStorage.setItem('gently-last-check', this.lastCheck.toISOString());
        this.newDetections = [];
        this.updateDetectionAlert();
    }
};

/**
 * Narrative Summary Manager
 * Manages AI-generated experiment summaries
 */
const NarrativeManager = {
    isLoading: false,
    lastNarrative: null,
    isCollapsed: false,

    init() {
        // Load collapsed state from localStorage
        const collapsed = localStorage.getItem('gently-narrative-collapsed');
        this.isCollapsed = collapsed === 'true';
        this.applyCollapseState();
    },

    toggle() {
        this.isCollapsed = !this.isCollapsed;
        localStorage.setItem('gently-narrative-collapsed', this.isCollapsed.toString());
        this.applyCollapseState();
    },

    applyCollapseState() {
        const panel = document.getElementById('narrative-panel');
        if (panel) {
            panel.classList.toggle('collapsed', this.isCollapsed);
        }
    },

    async refresh() {
        if (this.isLoading) return;

        this.isLoading = true;
        this.showLoading(true);

        try {
            const response = await fetch('/api/narrative');
            if (response.ok) {
                const narrative = await response.json();
                this.lastNarrative = narrative;
                this.renderNarrative(narrative);
            } else {
                this.renderError('Failed to generate summary');
            }
        } catch (error) {
            console.error('Failed to fetch narrative:', error);
            this.renderError('Connection error');
        } finally {
            this.isLoading = false;
            this.showLoading(false);
        }
    },

    async showSinceLastCheck() {
        if (this.isLoading) return;

        const lastCheck = ExperimentStrip.lastCheck;
        if (!lastCheck) {
            this.renderLocalSummary();
            return;
        }

        this.isLoading = true;
        this.showLoading(true);

        try {
            const response = await fetch(`/api/narrative?since=${lastCheck.toISOString()}`);
            if (response.ok) {
                const narrative = await response.json();
                this.lastNarrative = narrative;
                this.renderNarrative(narrative);
            } else {
                this.renderLocalSummary();
            }
        } catch (error) {
            this.renderLocalSummary();
        } finally {
            this.isLoading = false;
            this.showLoading(false);
            ExperimentStrip.markChecked();
        }
    },

    renderLocalSummary() {
        // Generate a simple local summary when API isn't available
        const state = TasksManager.state;
        const embryoCount = Object.keys(state.embryos).length;

        if (embryoCount === 0) {
            this.updateNarrativeUI({
                status: 'normal',
                headline: 'No Active Experiment',
                details: ['Start a timelapse to see AI-generated summaries here.']
            });
            return;
        }

        const activeCount = Object.values(state.embryos).filter(e => !e.isComplete).length;
        const completedCount = Object.values(state.embryos).filter(e => e.isComplete).length;

        // Count detections
        let totalDetections = 0;
        let detectionDetails = [];
        Object.entries(TasksManager.detectionReasoning).forEach(([embryoId, reasoning]) => {
            const positives = reasoning.filter(r => r.detected);
            totalDetections += positives.length;
            positives.forEach(d => {
                detectionDetails.push(`${embryoId}: ${TasksManager.formatDetectorName(d.detector_name)} at T${d.timepoint}`);
            });
        });

        const details = [];
        if (activeCount > 0) details.push(`${activeCount} embryo${activeCount !== 1 ? 's' : ''} actively imaging`);
        if (completedCount > 0) details.push(`${completedCount} embryo${completedCount !== 1 ? 's' : ''} completed`);
        details.push(`${state.totalTimepoints} total timepoints acquired`);

        if (detectionDetails.length > 0) {
            details.push(`${totalDetections} positive detection${totalDetections !== 1 ? 's' : ''}: ${detectionDetails.slice(0, 3).join(', ')}${detectionDetails.length > 3 ? '...' : ''}`);
        }

        const status = totalDetections > 0 ? 'notable' :
                      completedCount > 0 ? 'normal' : 'normal';

        const headline = totalDetections > 0 ?
            `${totalDetections} Detection${totalDetections !== 1 ? 's' : ''} Found` :
            completedCount > 0 ?
            `${completedCount}/${embryoCount} Embryos Complete` :
            'Experiment In Progress';

        this.updateNarrativeUI({ status, headline, details });
    },

    renderNarrative(narrative) {
        this.updateNarrativeUI({
            status: narrative.status || 'normal',
            headline: narrative.headline || 'Experiment Summary',
            summary: narrative.summary,
            details: narrative.details || []
        });
    },

    renderError(message) {
        this.updateNarrativeUI({
            status: 'attention',
            headline: 'Summary Unavailable',
            details: [message, 'Showing local summary instead.']
        });
        // Fall back to local summary
        setTimeout(() => this.renderLocalSummary(), 2000);
    },

    updateNarrativeUI({ status, headline, summary, details }) {
        // Update badge
        const badge = document.getElementById('narrative-badge');
        if (badge) {
            badge.className = `narrative-status-badge ${status}`;
            badge.innerHTML = status === 'normal' ? '&#x2714;' :
                             status === 'notable' ? '&#x1F514;' : '&#x26A0;';
        }

        // Update headline
        const headlineEl = document.getElementById('narrative-headline');
        if (headlineEl) headlineEl.textContent = headline;

        // Update meta
        const metaEl = document.getElementById('narrative-meta');
        if (metaEl) metaEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;

        // Update body
        const bodyEl = document.getElementById('narrative-body');
        if (bodyEl) {
            let html = '';

            if (summary) {
                html += `
                    <div class="narrative-section">
                        <div class="narrative-section-title">Summary</div>
                        <p class="narrative-text">${this.escapeHtml(summary)}</p>
                    </div>
                `;
            }

            if (details && details.length > 0) {
                html += `
                    <div class="narrative-section">
                        <div class="narrative-section-title">Details</div>
                        <ul class="narrative-bullets">
                            ${details.map(d => `<li>${this.escapeHtml(d)}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            bodyEl.innerHTML = html || '<p class="narrative-text">No summary available.</p>';
        }
    },

    showLoading(show) {
        const loading = document.getElementById('narrative-loading');
        const body = document.getElementById('narrative-body');
        if (loading) loading.style.display = show ? 'flex' : 'none';
        if (body) body.style.display = show ? 'none' : 'block';
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    TasksManager.init();
    ExperimentStrip.init();
    NarrativeManager.init();

    // Update experiment strip every second
    setInterval(() => ExperimentStrip.update(), 1000);

    // Generate initial narrative summary
    setTimeout(() => NarrativeManager.renderLocalSummary(), 500);
});
