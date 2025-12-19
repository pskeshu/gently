/**
 * Embryos Tab - Timelapse Task Tracking
 * Displays active timelapse tasks with per-embryo breakdown
 */

const EmbryosManager = {
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

    // Number of items to show in expanded ranges (high number = show all)
    rangeLoadLimit: 500,
    rangeLoadMore: {},  // range key -> number of items loaded

    // Expanded range item states (for inline expansion within ranges)
    expandedRangeItems: {},  // "{rangeKey}-{timepoint}" -> true/false

    // Detail panel state
    currentDetailItem: null,  // Currently viewed item in detail panel
    detailPanelVisible: false,

    // Badge state for new detection notifications
    newDetectionCount: 0,  // Count of NEW detections since user last viewed
    lastSeenDetectionTime: null,  // When user last viewed the Embryos tab

    countdownInterval: null,
    storageKey: 'gently-tasks-state',

    init() {
        // Restore state from localStorage
        this.loadState();
        // Load detection agreements
        this.loadAgreements();
        // Load badge state (new detection count)
        this.loadBadgeState();
        // Start countdown update timer
        this.startCountdownUpdates();
        // Initial render
        this.render();
        // Update badge on init
        this.updateDetectionBadge();
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
                if (e.firstAcquired instanceof Date) {
                    e.firstAcquired = e.firstAcquired.toISOString();
                }
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
                if (e.firstAcquired && typeof e.firstAcquired === 'string') {
                    e.firstAcquired = new Date(e.firstAcquired);
                }
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
            this.updateEmbryosCount();
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
        // Clear state if: server has new session ID, OR server is IDLE/fresh and client has stale data
        const serverHasNewSession = serverSessionId && serverSessionId !== this.currentSessionId;
        const serverIsIdleButClientHasData = !serverSessionId && this.currentSessionId;

        if (serverHasNewSession || serverIsIdleButClientHasData) {
            console.log(`Session changed: ${this.currentSessionId} → ${serverSessionId || '(none)'}`);
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
                firstAcquired: embryoData.first_acquired ? new Date(embryoData.first_acquired) : null,
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

        this.updateEmbryosCount();
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
        // Clear badge state for new session
        this.newDetectionCount = 0;
        this.lastSeenDetectionTime = null;
        this.saveBadgeState();
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
                firstAcquired: null,
                lastAcquired: null,
                detections: {},
                errorCount: 0,
                lastError: null
            };
        });

        this.updateEmbryosCount();
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
                firstAcquired: null,
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
        const now = new Date();
        if (!embryo.firstAcquired) {
            embryo.firstAcquired = now;
        }
        embryo.lastAcquired = now;

        this.updateEmbryosCount();
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
        // All detector/perception evaluations (with reasoning) - populates reasoning panel
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        const detectorName = data.detector_name;
        // Handle both legacy "detected" and perception "is_hatching"
        const detected = data.detected ?? data.is_hatching ?? false;
        const stage = data.stage;

        // Update detection status
        embryo.detections[detectorName] = {
            detected: detected,
            confidence: data.confidence,
            timepoint: data.timepoint,
            stage: stage,
        };

        // Update current_stage if this is a perception result
        if (detectorName === 'perception' && stage) {
            embryo.current_stage = stage;
        }

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
                timestamp: new Date().toISOString(),
                // Perception-specific fields
                stage: stage,
                is_hatching: data.is_hatching,
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

        // Increment the new detection badge count
        this.incrementDetectionBadge();

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
    // Verification Event Handlers
    // ==========================================

    handleVerificationStarted(data) {
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        embryo.verification = {
            status: 'running',
            consecutive_count: data.consecutive_count || 0,
            required_count: data.required_count || 5,
            strategies_complete: 0,
            total_strategies: 5,
            strategies: {}
        };

        // Add to detection reasoning as a verification event
        if (!this.detectionReasoning[embryoId]) {
            this.detectionReasoning[embryoId] = [];
        }
        this.detectionReasoning[embryoId].push({
            type: 'verification_started',
            detector_name: data.detector_name || 'hatching',
            timepoint: data.round_number,
            consecutive_count: data.consecutive_count,
            required_count: data.required_count,
            timestamp: new Date().toISOString()
        });

        this.updateEmbryoCard(embryoId);
        this.saveState();
    },

    handleVerificationStrategy(data) {
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo || !embryo.verification) return;

        // Update strategy result
        embryo.verification.strategies[data.strategy] = {
            passed: data.passed,
            summary: data.summary
        };

        this.updateEmbryoCard(embryoId);
    },

    handleVerificationProgress(data) {
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo || !embryo.verification) return;

        embryo.verification.strategies_complete = data.strategies_complete;
        embryo.verification.total_strategies = data.total_strategies;

        this.updateEmbryoCard(embryoId);
    },

    handleVerificationCompleted(data) {
        const embryoId = data.embryo_id;
        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        // Update verification state
        embryo.verification = {
            status: 'completed',
            consensus: data.consensus,
            reasoning: data.reasoning,
            strategies: data.strategies,
            ensemble_votes: data.ensemble_votes,
            duration_seconds: data.duration_seconds
        };

        // Update consecutive count
        if (data.consensus) {
            embryo.consecutive_verified = (embryo.consecutive_verified || 0) + 1;
        } else {
            embryo.consecutive_verified = 0;
        }

        // Add to detection reasoning
        if (!this.detectionReasoning[embryoId]) {
            this.detectionReasoning[embryoId] = [];
        }
        this.detectionReasoning[embryoId].push({
            type: 'verification_completed',
            detector_name: data.detector_name || 'hatching',
            consensus: data.consensus,
            reasoning: data.reasoning,
            strategies: data.strategies,
            ensemble_votes: data.ensemble_votes,
            duration_seconds: data.duration_seconds,
            consecutive_verified: embryo.consecutive_verified,
            timestamp: new Date().toISOString()
        });

        this.updateEmbryoCard(embryoId);
        if (this.selectedEmbryoId === embryoId) {
            this.renderReasoningPanel();
        }
        this.saveState();
    },

    // ==========================================
    // Rendering
    // ==========================================

    render() {
        this.renderStatusBadge();
        this.renderSummary();
        this.renderEmbryoCards();
        this.renderReasoningPanel();
        // Show first-run hints after a short delay to let the UI settle
        setTimeout(() => this.showFirstRunHints(), 500);
    },

    renderStatusBadge() {
        const statusEl = document.getElementById('timelapse-status');
        const textEl = document.getElementById('timelapse-status-text');
        const durationEl = document.getElementById('timelapse-duration');
        const sessionIdEl = document.getElementById('session-id');

        if (!statusEl) return;

        // Remove all status classes
        statusEl.classList.remove('running', 'paused', 'completed', 'idle');

        if (this.state.status === 'IDLE' || Object.keys(this.state.embryos).length === 0) {
            statusEl.classList.add('idle');
            textEl.textContent = 'No active timelapse';
            durationEl.textContent = '';
            if (sessionIdEl) sessionIdEl.textContent = '';
        } else {
            statusEl.classList.add(this.state.status.toLowerCase());
            textEl.textContent = this.state.status === 'RUNNING' ? 'Running' :
                                 this.state.status === 'PAUSED' ? 'Paused' :
                                 this.state.status === 'COMPLETED' ? 'Completed' : this.state.status;

            if (this.state.startedAt) {
                durationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
            }

            // Display session ID
            if (sessionIdEl && this.currentSessionId) {
                sessionIdEl.textContent = this.currentSessionId;
            }
        }
    },

    renderSummary() {
        const summaryEl = document.getElementById('embryos-summary');
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
                <span class="stat-label">Active</span>
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
            ${active > 0 ? `
            <div class="summary-stat">
                <span class="stat-value" id="summary-next-countdown">${this.getNextCountdown()}</span>
                <span class="stat-label">Next</span>
            </div>
            ` : ''}
        `;
    },

    getNextCountdown() {
        const embryos = Object.values(this.state.embryos);
        const activeEmbryos = embryos.filter(e => !e.isComplete);
        if (activeEmbryos.length === 0) return '--:--';

        let minSeconds = Infinity;
        activeEmbryos.forEach(embryo => {
            if (embryo.lastAcquired) {
                const elapsed = (Date.now() - new Date(embryo.lastAcquired).getTime()) / 1000;
                const remaining = Math.max(0, embryo.intervalSeconds - elapsed);
                minSeconds = Math.min(minSeconds, remaining);
            } else if (embryo.intervalSeconds) {
                minSeconds = Math.min(minSeconds, embryo.intervalSeconds);
            }
        });

        if (minSeconds === Infinity) return '--:--';
        return this.formatCountdown(Math.floor(minSeconds));
    },

    updateSummary() {
        // Quick update for duration and countdown
        const durationEl = document.getElementById('summary-duration');
        if (durationEl && this.state.startedAt) {
            durationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }
        const countdownEl = document.getElementById('summary-next-countdown');
        if (countdownEl) {
            countdownEl.textContent = this.getNextCountdown();
        }
        // Update stats
        this.renderSummary();
    },

    renderEmbryoCards() {
        const container = document.getElementById('embryo-cards');
        if (!container) return;

        const embryos = Object.values(this.state.embryos);

        if (embryos.length === 0) {
            // Use smart empty state based on experiment status
            const emptyType = this.state.status === 'RUNNING' ? 'waiting-first' : 'no-embryos';
            container.innerHTML = this.renderSmartEmptyState(emptyType);
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

        // Current developmental stage (from perception system)
        let stageHtml = '';
        if (embryo.current_stage) {
            const stageIcon = this.getStageIcon(embryo.current_stage);
            stageHtml = `
                <div class="current-stage">
                    <span class="stage-icon">${stageIcon}</span>
                    <span class="stage-label">Stage:</span>
                    <span class="stage-value">${this.formatStageName(embryo.current_stage)}</span>
                </div>
            `;
        }

        // Detection status (legacy detectors)
        let detectionsHtml = '';
        const detectorNames = Object.keys(embryo.detections).filter(n => n !== 'perception');
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

        // Verification status (shows during/after verification rounds)
        let verificationHtml = '';
        if (embryo.verification || embryo.consecutive_verified > 0) {
            verificationHtml = this.renderVerificationStatus(embryo);
        }

        // Completion or error info
        let completionHtml = '';
        if (embryo.isComplete) {
            completionHtml = `
                <div class="completion-info">
                    <span class="completion-reason">${embryo.completionReason || 'Completed'}</span>
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
            <button class="embryo-play-btn" onclick="event.stopPropagation(); EmbryosManager.playEmbryoTimelapse('${embryo.embryoId}')" data-tooltip="Play timelapse video">
                <span class="play-icon">▶</span>
            </button>
        ` : '';

        return `
            <div class="embryo-card sidebar-card ${status} ${isSelected ? 'selected' : ''}" data-embryo-id="${embryo.embryoId}">
                <div class="embryo-header">
                    <div class="embryo-header-left">
                        <span class="embryo-name">${embryo.embryoId}</span>
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
                        <span class="imaging-duration">${this.getEmbryoImagingDuration(embryo)}</span>
                    </div>
                </div>

                ${countdownHtml}
                ${stageHtml}
                ${detectionsHtml}
                ${verificationHtml}
                ${completionHtml}
            </div>
        `;
    },

    // Get icon for developmental stage
    getStageIcon(stage) {
        const icons = {
            'early': '🥚',
            'bean': '🫘',
            'comma': '🌙',
            '1.5fold': '🔄',
            '2fold': '🔁',
            '3fold': '🔃',
            'pretzel': '🥨',
            'hatching': '🐣',
            'hatched': '🐛',
        };
        return icons[stage?.toLowerCase()] || '🔬';
    },

    // Format stage name for display
    formatStageName(stage) {
        if (!stage) return 'Unknown';
        const names = {
            'early': 'Early',
            'bean': 'Bean',
            'comma': 'Comma',
            '1.5fold': '1.5-Fold',
            '2fold': '2-Fold',
            '3fold': '3-Fold',
            'pretzel': 'Pretzel',
            'hatching': 'Hatching',
            'hatched': 'Hatched',
        };
        return names[stage.toLowerCase()] || stage;
    },

    // Render verification status for embryo card
    renderVerificationStatus(embryo) {
        const v = embryo.verification;
        const consecutiveCount = embryo.consecutive_verified || 0;
        const requiredCount = v?.required_count || 5;

        // If verification is running
        if (v && v.status === 'running') {
            const strategiesComplete = v.strategies_complete || 0;
            const totalStrategies = v.total_strategies || 5;
            const progressPct = (strategiesComplete / totalStrategies) * 100;

            // Build strategy status icons
            const strategies = v.strategies || {};
            const strategyIcons = ['adversarial', 'independent', 'temporal', 'ensemble', 'hardware_context']
                .map(name => {
                    if (strategies[name] !== undefined) {
                        return strategies[name].passed
                            ? `<span class="strategy-icon passed" title="${name}: passed">✓</span>`
                            : `<span class="strategy-icon failed" title="${name}: failed">✗</span>`;
                    }
                    return `<span class="strategy-icon pending" title="${name}: pending">○</span>`;
                }).join('');

            return `
                <div class="verification-status running">
                    <div class="verification-header">
                        <span class="verification-icon">🔍</span>
                        <span class="verification-label">Verifying...</span>
                        <span class="verification-count">(${consecutiveCount}/${requiredCount})</span>
                    </div>
                    <div class="verification-progress">
                        <div class="verification-progress-bar">
                            <div class="verification-progress-fill" style="width: ${progressPct}%"></div>
                        </div>
                        <span class="verification-progress-text">${strategiesComplete}/${totalStrategies}</span>
                    </div>
                    <div class="verification-strategies">
                        ${strategyIcons}
                    </div>
                </div>
            `;
        }

        // If verification completed (show result)
        if (v && v.status === 'completed') {
            const passed = v.consensus;
            const icon = passed ? '✓' : '✗';
            const statusClass = passed ? 'passed' : 'failed';
            const statusText = passed ? 'Verified' : 'Failed';

            return `
                <div class="verification-status ${statusClass}">
                    <div class="verification-header">
                        <span class="verification-icon">${icon}</span>
                        <span class="verification-label">${statusText}</span>
                        <span class="verification-count">(${consecutiveCount}/${requiredCount} consecutive)</span>
                    </div>
                    ${v.ensemble_votes ? `<div class="verification-detail">Ensemble: ${v.ensemble_votes}</div>` : ''}
                </div>
            `;
        }

        // If we have consecutive count but no active verification
        if (consecutiveCount > 0) {
            return `
                <div class="verification-status summary">
                    <div class="verification-header">
                        <span class="verification-icon">🔍</span>
                        <span class="verification-label">Verified</span>
                        <span class="verification-count">${consecutiveCount}/${requiredCount} consecutive</span>
                    </div>
                </div>
            `;
        }

        return '';
    },

    selectEmbryo(embryoId) {
        // Clear per-embryo UI state when switching embryos
        if (this.selectedEmbryoId !== embryoId) {
            this.currentDetailItem = null;
            this.expandedImages = {};
        }

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
            // Show contextual empty state
            const hasEmbryos = Object.keys(this.state.embryos).length > 0;
            if (hasEmbryos) {
                panel.innerHTML = `
                    <div class="reasoning-empty">
                        <div class="reasoning-empty-icon">&#x1F441;</div>
                        <div class="reasoning-empty-text">Select an embryo to view its detection analysis</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
                            Click on any embryo card in the left panel
                        </div>
                    </div>
                `;
            } else {
                panel.innerHTML = this.renderSmartEmptyState('no-embryos');
            }
            return;
        }

        const embryo = this.state.embryos[this.selectedEmbryoId];
        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];

        // Calculate statistics
        const totalEvaluations = reasoning.length;
        const positiveDetections = reasoning.filter(r => r.detected);
        const highConfidence = reasoning.filter(r => r.confidence?.toLowerCase() === 'high');

        // Check if this is perception data (has stage field)
        const isPerceptionData = reasoning.some(r => r.stage);

        // Get stage progression for perception data
        let stageTransitions = [];
        let currentStage = null;
        if (isPerceptionData) {
            const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));
            sorted.forEach(r => {
                if (r.stage && r.stage !== currentStage) {
                    stageTransitions.push({
                        stage: r.stage,
                        timepoint: r.timepoint,
                        detector_name: r.detector_name || 'perception'
                    });
                    currentStage = r.stage;
                }
            });
            // Get latest stage
            currentStage = sorted.length > 0 ? sorted[sorted.length - 1].stage : null;
        }

        // Embryo info header
        const statusIcon = embryo.isComplete ? '&#x2714;' :
                          embryo.lastError ? '&#x2718;' : '&#x25CF;';
        const statusClass = embryo.isComplete ? 'complete' :
                           embryo.lastError ? 'error' : 'running';

        // Build quick jump badges (stage transitions for perception, positive detections for legacy)
        let quickJumpsHtml;
        if (isPerceptionData && stageTransitions.length > 0) {
            quickJumpsHtml = stageTransitions.map(t => `
                <span class="quick-jump-badge stage-jump" onclick="EmbryosManager.scrollToDetection(${t.timepoint}, '${t.detector_name}')" title="Jump to ${this.formatStageName(t.stage)}">
                    <span class="stage-icon">${this.getStageIcon(t.stage)}</span>
                    ${this.formatStageName(t.stage)} @ T${t.timepoint}
                </span>
            `).join('');
        } else if (positiveDetections.length > 0) {
            quickJumpsHtml = positiveDetections.map(d => `
                <span class="quick-jump-badge" onclick="EmbryosManager.scrollToDetection(${d.timepoint}, '${d.detector_name}')" title="Jump to detection">
                    <span class="detector-icon">${this.getDetectorIcon(d.detector_name)}</span>
                    ${this.formatDetectorName(d.detector_name)} @ T${d.timepoint}
                </span>
            `).join('');
        } else {
            quickJumpsHtml = '<span style="font-size: 0.8rem; color: var(--text-muted);">No stage transitions yet</span>';
        }

        // Build empty state if no evaluations
        const emptyStateHtml = totalEvaluations === 0 ? `
            <div class="no-detections">
                <div class="no-detections-icon">&#x1F9EC;</div>
                <div class="no-detections-text">No stage evaluations yet</div>
                <div class="no-detections-hint">
                    Stage analysis will appear here as the embryo develops.
                </div>
            </div>
        ` : `
            <div class="eval-hint">
                <span>Click on any evaluation dot above to view the full VLM analysis</span>
            </div>
        `;

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
                ${isPerceptionData ? `
                    <div class="detection-summary-stat stage-stat">
                        <span class="stat-value stage-value">
                            ${currentStage ? `<span class="stage-icon">${this.getStageIcon(currentStage)}</span>` : ''}
                            ${currentStage ? this.formatStageName(currentStage) : 'Unknown'}
                        </span>
                        <span class="stat-label">Current Stage</span>
                    </div>
                    <div class="detection-summary-stat">
                        <span class="stat-value ${stageTransitions.length > 0 ? 'has-detections' : ''}">${stageTransitions.length}</span>
                        <span class="stat-label">Transitions</span>
                    </div>
                ` : `
                    <div class="detection-summary-stat">
                        <span class="stat-value ${positiveDetections.length > 0 ? 'has-detections' : ''}">${positiveDetections.length}</span>
                        <span class="stat-label">Detections</span>
                    </div>
                `}
                <div class="detection-summary-stat">
                    <span class="stat-value">${totalEvaluations}</span>
                    <span class="stat-label">Evaluations</span>
                </div>
                <div class="detection-quick-jumps">
                    ${quickJumpsHtml}
                </div>
            </div>
            ${this.renderTimelineSparkline(reasoning, embryo.timepoints)}
            ${emptyStateHtml}
            <div class="inline-detail-container" id="inline-detail-container"></div>
        `;

        // Re-render current detail if one was open
        if (this.currentDetailItem) {
            this.renderInlineDetail(this.currentDetailItem);
        }
    },

    // Render detections with range collapse for "not detected" sequences
    renderDetectionListWithCollapse(reasoning) {
        // Separate verification events from regular detections
        const verificationEvents = reasoning.filter(r => r.type === 'verification_started' || r.type === 'verification_completed');
        const detectionEvents = reasoning.filter(r => !r.type || (!r.type.startsWith('verification_')));

        // Sort by timepoint/timestamp descending (newest first)
        const sorted = [...detectionEvents].sort((a, b) => (b.timepoint ?? 0) - (a.timepoint ?? 0));
        const sortedVerifications = [...verificationEvents].sort((a, b) => {
            const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
            const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
            return timeB - timeA;
        });

        // Apply filter
        let filtered = sorted;
        if (this.detectionFilter === 'detections') {
            filtered = sorted.filter(r => r.detected);
        } else if (this.detectionFilter === 'high-confidence') {
            filtered = sorted.filter(r => r.confidence?.toLowerCase() === 'high');
        }

        // If showing only detections, render them directly without collapse
        if (this.detectionFilter !== 'all') {
            if (filtered.length === 0 && sortedVerifications.length === 0) {
                return `
                    <div class="detection-empty-filtered">
                        <div class="icon">&#x1F50E;</div>
                        <div class="message">No ${this.detectionFilter === 'detections' ? 'positive detections' : 'high confidence evaluations'} found</div>
                        <div class="hint">Try selecting "All" to see all evaluations</div>
                    </div>
                `;
            }
            let html = filtered.map((r, idx) => this.renderDetectionCard(r, idx, true)).join('');
            // Also show verification events in filtered view
            if (sortedVerifications.length > 0) {
                html += sortedVerifications.map((v, idx) => this.renderVerificationCard(v, `ver-${idx}`)).join('');
            }
            return html;
        }

        // For "all" view, use range collapse
        // Group consecutive "not detected" items, but always show "detected" items expanded
        const groups = this.groupDetectionsForCollapse(sorted);

        let html = '';

        // First, render verification section if any exist
        if (sortedVerifications.length > 0) {
            const completedVerifications = sortedVerifications.filter(v => v.type === 'verification_completed');
            html += `
                <div class="verification-section">
                    <div class="section-header">
                        <span class="icon">🔍</span>
                        <span class="title">Verification Rounds</span>
                        <span class="count">${completedVerifications.length} completed</span>
                    </div>
                    ${sortedVerifications.map((v, idx) => this.renderVerificationCard(v, `ver-${idx}`)).join('')}
                </div>
            `;
        }

        // Then, render positive detections section if any exist
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

        let firstNegativeRangeAutoExpanded = false;

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

                // Auto-expand first negative range so user sees evaluations immediately
                let isExpanded = this.expandedRanges[rangeKey];
                if (isExpanded === undefined && !firstNegativeRangeAutoExpanded) {
                    isExpanded = true;
                    firstNegativeRangeAutoExpanded = true;
                }
                isExpanded = isExpanded || false;
                const items = group.items;
                const startTp = items[items.length - 1].timepoint ?? '?';
                const endTp = items[0].timepoint ?? '?';
                const rangeLabel = items.length === 1
                    ? `Timepoint ${startTp}`
                    : `Timepoints ${endTp} - ${startTp}`;

                html += `
                    <div class="collapsed-range ${isExpanded ? 'expanded' : ''}" onclick="EmbryosManager.toggleRange('${rangeKey}')">
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

    // Render items inside an expanded range with thumbnails, confidence dots, and inline expansion
    renderRangeItems(items, rangeKey) {
        const loadedCount = this.rangeLoadMore[rangeKey] || this.rangeLoadLimit;
        const visibleItems = items.slice(0, loadedCount);
        const hasMore = items.length > loadedCount;

        // Get all items in this embryo for interest score calculation
        const allReasoning = this.detectionReasoning[this.selectedEmbryoId] || [];

        let html = visibleItems.map(item => {
            const itemKey = `${rangeKey}-${item.timepoint}`;
            const isExpanded = this.expandedRangeItems[itemKey] || false;
            const imageUid = item.projection_uid || item.volume_uid;
            const isInteresting = this.calculateInterestScore(item, allReasoning) > 0.5;

            return `
                <div class="detection-row-compact ${isExpanded ? 'expanded' : ''} ${isInteresting ? 'flagged' : ''}"
                     data-range="${rangeKey}" data-timepoint="${item.timepoint}">
                    <div class="compact-row-header" onclick="EmbryosManager.toggleRangeItem('${rangeKey}', ${item.timepoint})">
                        ${imageUid
                            ? `<img class="compact-thumbnail" src="/api/images/${imageUid}/png" alt="T${item.timepoint}" loading="lazy" />`
                            : '<div class="compact-thumbnail-placeholder"></div>'}
                        <span class="tp-badge">T${item.timepoint ?? '?'}</span>
                        <span class="detector-name">${this.formatDetectorName(item.detector_name)}</span>
                        <span class="result">Not detected</span>
                        ${this.renderConfidenceDots(item.confidence)}
                        ${isInteresting ? '<span class="interest-flag" title="Near detection boundary">&#x2691;</span>' : ''}
                        <span class="row-affordance">&#x22EF;</span>
                        <span class="row-chevron">&#x25B8;</span>
                    </div>
                    ${isExpanded ? this.renderInlineExpansion(item) : ''}
                </div>
            `;
        }).join('');

        if (hasMore) {
            html += `
                <button class="load-more-btn" onclick="event.stopPropagation(); EmbryosManager.loadMoreInRange('${rangeKey}', ${items.length})">
                    Load more (${items.length - loadedCount} remaining)
                </button>
            `;
        }

        return html;
    },

    // Toggle expansion of an individual item within a range
    toggleRangeItem(rangeKey, timepoint) {
        const itemKey = `${rangeKey}-${timepoint}`;
        this.expandedRangeItems[itemKey] = !this.expandedRangeItems[itemKey];

        const row = document.querySelector(`.detection-row-compact[data-range="${rangeKey}"][data-timepoint="${timepoint}"]`);
        if (row) {
            row.classList.toggle('expanded', this.expandedRangeItems[itemKey]);

            // Re-render the row content
            const allReasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
            const item = allReasoning.find(r => r.timepoint === timepoint);
            if (item) {
                const expansionContainer = row.querySelector('.inline-expansion');
                if (this.expandedRangeItems[itemKey] && !expansionContainer) {
                    // Add expansion
                    row.insertAdjacentHTML('beforeend', this.renderInlineExpansion(item));
                } else if (!this.expandedRangeItems[itemKey] && expansionContainer) {
                    // Remove expansion
                    expansionContainer.remove();
                }
            }
        }
    },

    // Render inline expansion with thumbnail and truncated reasoning
    renderInlineExpansion(item) {
        const imageUid = item.projection_uid || item.volume_uid;
        const reasoning = item.reasoning || 'No reasoning provided';
        const truncatedReasoning = reasoning.length > 250
            ? reasoning.substring(0, 250) + '...'
            : reasoning;

        return `
            <div class="inline-expansion">
                <div class="expansion-content">
                    ${imageUid ? `
                        <img class="expansion-image"
                             src="/api/images/${imageUid}/png"
                             alt="T${item.timepoint}"
                             onclick="event.stopPropagation(); EmbryosManager.openDetailPanel('${item.detector_name}', ${item.timepoint})" />
                    ` : '<div class="expansion-image-placeholder">No image</div>'}
                    <div class="expansion-text">
                        <div class="expansion-reasoning">${this.escapeHtml(truncatedReasoning)}</div>
                        <button class="expansion-link" onclick="event.stopPropagation(); EmbryosManager.openDetailPanel('${item.detector_name}', ${item.timepoint})">
                            View full analysis &#x2192;
                        </button>
                    </div>
                </div>
            </div>
        `;
    },

    // Render confidence level as dots (5-dot scale)
    renderConfidenceDots(confidence) {
        const level = confidence?.toLowerCase() || 'unknown';
        const filled = level === 'high' ? 4 : level === 'medium' ? 3 : level === 'low' ? 1 : 2;

        let dots = '';
        for (let i = 0; i < 5; i++) {
            dots += `<span class="confidence-dot ${i < filled ? 'filled' : ''}"></span>`;
        }
        return `<div class="confidence-dots" title="${confidence || 'Unknown'} confidence">${dots}</div>`;
    },

    // Calculate interest score for flagging "interesting" negatives
    calculateInterestScore(item, allItems) {
        if (!item || item.detected) return 0;

        let score = 0;

        // Find nearest positive detection
        const positives = allItems.filter(i => i.detected);
        if (positives.length > 0) {
            const distances = positives.map(p => Math.abs((p.timepoint || 0) - (item.timepoint || 0)));
            const nearestDistance = Math.min(...distances);

            // Within 3 timepoints of a detection
            if (nearestDistance <= 3) {
                score += 0.4 * (1 - nearestDistance / 3);
            }
        }

        // High confidence negative (VLM was very sure it wasn't detected)
        if (item.confidence?.toLowerCase() === 'high') {
            score += 0.3;
        }

        return score;
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

    // Scroll to a specific detection - now opens detail panel directly
    scrollToDetection(timepoint, detectorName) {
        this.openDetailPanel(detectorName, timepoint);
    },

    // Show detail for a compact row (legacy - redirects to openDetailPanel)
    showDetectionDetail(detectorName, timepoint) {
        this.openDetailPanel(detectorName, timepoint);
    },

    // Open the detail panel inline in the reasoning panel
    openDetailPanel(detectorName, timepoint) {
        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        const item = reasoning.find(r =>
            r.detector_name === detectorName && r.timepoint === timepoint
        );

        if (!item) {
            console.warn(`Detail panel: item not found for ${detectorName} at T${timepoint}`);
            return;
        }

        this.currentDetailItem = item;
        this.detailPanelVisible = true;

        this.renderInlineDetail(item);
    },

    // Render detail content inline in the reasoning panel
    renderInlineDetail(item) {
        const container = document.getElementById('inline-detail-container');
        if (!container) return;

        container.innerHTML = this.renderDetailPanel(item);
        container.classList.add('visible');

        // Scroll the detail into view
        container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    },

    // Render the detail panel content
    renderDetailPanel(item) {
        const imageUid = item.projection_uid || item.volume_uid;
        const reasoning = item.reasoning || 'No reasoning provided';

        // Linkify timepoints in reasoning
        const linkedReasoning = this.linkifyTimepoints(reasoning, this.selectedEmbryoId, {
            detectionPoint: item.detected ? item.timepoint : null,
            reasoningText: reasoning
        });

        // Build metadata for lightbox
        const lightboxMeta = {
            embryo_id: this.selectedEmbryoId,
            timepoint: item.timepoint,
            data_type: 'Volume Projection',
            shape: item.shape || ''
        };
        const metaJson = JSON.stringify(lightboxMeta).replace(/'/g, "\\'").replace(/"/g, '&quot;');

        // Build image HTML - will be loaded async if UID not available
        const imageHtml = imageUid
            ? `<img src="/api/images/${imageUid}/png"
                    alt="T${item.timepoint}"
                    onclick="Lightbox.openByUid && Lightbox.openByUid('${imageUid}', ${metaJson})" />`
            : `<div class="detail-image-loading" id="detail-image-placeholder"
                    data-embryo="${this.selectedEmbryoId}"
                    data-timepoint="${item.timepoint}">Loading image...</div>`;

        // Fetch image async if no UID
        if (!imageUid) {
            this.fetchDetailImage(this.selectedEmbryoId, item.timepoint);
        }

        return `
            <div class="detail-panel-header">
                <span class="detail-title">T${item.timepoint} - ${this.formatDetectorName(item.detector_name)}</span>
                <button class="detail-close" onclick="EmbryosManager.closeDetailPanel()">&times;</button>
            </div>
            <div class="detail-image-container">
                ${imageHtml}
            </div>
            <div class="detail-verdict ${item.detected ? 'detected' : ''}">
                ${item.detected ? 'DETECTED' : 'Not detected'} - ${item.confidence || 'Unknown'} confidence
            </div>
            <div class="detail-reasoning">
                <div class="reasoning-label">VLM Analysis</div>
                <div class="reasoning-text">${linkedReasoning}</div>
            </div>
            <div class="detail-actions">
                <button class="detail-nav" onclick="EmbryosManager.navigateDetail(-1)">&#x2190; Previous</button>
                <button class="detail-nav" onclick="EmbryosManager.navigateDetail(1)">Next &#x2192;</button>
            </div>
        `;
    },

    // Fetch image for detail panel using sequence API
    // Tries multiple data types as fallbacks
    async fetchDetailImage(embryoId, timepoint) {
        const placeholder = document.getElementById('detail-image-placeholder');
        if (!placeholder) return;

        // Try these data types in order of preference
        const dataTypes = ['volume_projection', 'volume', 'image'];

        for (const dataType of dataTypes) {
            try {
                const resp = await fetch(`/api/sequence/${embryoId}?start=${timepoint}&end=${timepoint}&data_type=${dataType}&buffer_percent=0`);
                const data = await resp.json();

                if (data.sequence && data.sequence.length > 0) {
                    const imgData = data.sequence[0];
                    const uid = imgData.uid;
                    // Build metadata for lightbox
                    const meta = {
                        embryo_id: embryoId,
                        timepoint: timepoint,
                        data_type: dataType,
                        shape: imgData.shape ? `${imgData.shape[0]}x${imgData.shape[1]}` : ''
                    };
                    const metaJson = JSON.stringify(meta).replace(/"/g, '&quot;');
                    placeholder.outerHTML = `<img src="/api/images/${uid}/png"
                                                  alt="T${timepoint}"
                                                  onclick="Lightbox.openByUid && Lightbox.openByUid('${uid}', ${metaJson})" />`;
                    return; // Success - exit
                }
            } catch (err) {
                console.warn(`Failed to fetch ${dataType} for T${timepoint}:`, err);
            }
        }

        // All types failed
        placeholder.outerHTML = '<div class="no-image">No image available for this timepoint</div>';
    },

    // Close the detail panel
    closeDetailPanel() {
        const container = document.getElementById('inline-detail-container');
        if (container) {
            container.classList.remove('visible');
            container.innerHTML = '';
        }
        this.detailPanelVisible = false;
        this.currentDetailItem = null;
    },

    // Navigate to previous/next item in detail panel
    navigateDetail(direction) {
        if (!this.currentDetailItem) return;

        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        const currentIdx = reasoning.findIndex(r =>
            r.detector_name === this.currentDetailItem.detector_name &&
            r.timepoint === this.currentDetailItem.timepoint
        );

        if (currentIdx === -1) return;

        const newIdx = currentIdx + direction;
        if (newIdx >= 0 && newIdx < reasoning.length) {
            const newItem = reasoning[newIdx];
            this.openDetailPanel(newItem.detector_name, newItem.timepoint);
        }
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

        // Stage badge for perception results
        const isPerception = detection.detector_name === 'perception';
        const stageHtml = detection.stage
            ? `<span class="stage-badge" title="Developmental stage">${this.getStageIcon(detection.stage)} ${this.formatStageName(detection.stage)}</span>`
            : '';

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
                            onclick="event.stopPropagation(); EmbryosManager.toggleReasoning('${index}')">
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
                        ${stageHtml}
                        <span class="detection-result ${detection.detected ? 'positive' : 'negative'}">
                            ${isPerception ? (detection.is_hatching ? 'Hatching!' : '') : (detection.detected ? 'Detected' : 'Not detected')}
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
                        <button class="toggle-image-btn" onclick="event.stopPropagation(); EmbryosManager.toggleImage('${index}', '${imageUid}')">
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
                        onclick="event.stopPropagation(); EmbryosManager.markAgreement('${detection.detector_name}', ${detection.timepoint}, true)"
                        data-tooltip="Confirm this detection is correct">
                    ${agreement === true ? '&#x2714; Agreed' : 'I Agree'}
                </button>
                <button class="vlm-action-btn disagree ${disagreedClass}"
                        onclick="event.stopPropagation(); EmbryosManager.markAgreement('${detection.detector_name}', ${detection.timepoint}, false)"
                        data-tooltip="Mark this detection as incorrect">
                    ${agreement === false ? '&#x2718; Disagreed' : 'I Disagree'}
                </button>
                <button class="vlm-action-btn" onclick="event.stopPropagation(); EmbryosManager.compareDetection('${detection.detector_name}', ${detection.timepoint})"
                        data-tooltip="Compare with previous timepoint">
                    Compare
                </button>
            </div>
        `;
    },

    // Render a verification event card for the reasoning panel
    renderVerificationCard(verification, index) {
        const isCompleted = verification.type === 'verification_completed';
        const timestamp = verification.timestamp ? new Date(verification.timestamp).toLocaleTimeString() : '';

        if (verification.type === 'verification_started') {
            return `
                <div class="detection-card verification-card started" data-timepoint="${verification.timepoint}">
                    <div class="detection-card-header">
                        <div class="detection-meta">
                            <span class="detector-badge verification">🔍 Verification</span>
                            <span class="detection-result verification-started">Started</span>
                        </div>
                        <div class="detection-timing">
                            <span class="timepoint-badge">Round ${verification.timepoint}</span>
                            <span class="detection-time">${timestamp}</span>
                        </div>
                    </div>
                    <div class="verification-info">
                        <span class="consecutive-badge">${verification.consecutive_count}/${verification.required_count} consecutive</span>
                    </div>
                </div>
            `;
        }

        // verification_completed
        const passed = verification.consensus;
        const strategies = verification.strategies || {};
        const strategyNames = ['adversarial', 'independent', 'temporal', 'ensemble', 'hardware_context'];

        const strategyRows = strategyNames.map(name => {
            const result = strategies[name];
            if (result === undefined || result === null) return '';
            const icon = result ? '✓' : '✗';
            const statusClass = result ? 'passed' : 'failed';
            const label = name.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            return `
                <div class="strategy-row ${statusClass}">
                    <span class="strategy-icon">${icon}</span>
                    <span class="strategy-name">${label}</span>
                </div>
            `;
        }).filter(Boolean).join('');

        return `
            <div class="detection-card verification-card ${passed ? 'passed' : 'failed'}" data-timepoint="${verification.timepoint}">
                <div class="detection-card-header">
                    <div class="detection-meta">
                        <span class="detector-badge verification">🔍 Verification</span>
                        <span class="detection-result ${passed ? 'positive' : 'negative'}">
                            ${passed ? 'PASSED' : 'FAILED'}
                        </span>
                    </div>
                    <div class="detection-timing">
                        <span class="consecutive-badge ${passed ? 'passed' : 'failed'}">
                            ${verification.consecutive_verified}/${5} consecutive
                        </span>
                        <span class="detection-time">${timestamp}</span>
                    </div>
                </div>

                <div class="verification-strategies-detail">
                    ${strategyRows}
                </div>

                ${verification.ensemble_votes ? `
                    <div class="verification-ensemble">
                        <span class="ensemble-label">Ensemble:</span>
                        <span class="ensemble-votes">${verification.ensemble_votes}</span>
                    </div>
                ` : ''}

                ${verification.reasoning ? `
                    <div class="verification-reasoning">
                        <span class="reasoning-label">Consensus:</span>
                        <span class="reasoning-text">${verification.reasoning}</span>
                    </div>
                ` : ''}

                ${verification.duration_seconds ? `
                    <div class="verification-duration">
                        Completed in ${verification.duration_seconds.toFixed(1)}s
                    </div>
                ` : ''}
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

    // ==========================================
    // Badge State Management
    // ==========================================

    // Load badge state from localStorage
    loadBadgeState() {
        try {
            const saved = localStorage.getItem('gently-badge-state');
            if (saved) {
                const data = JSON.parse(saved);
                this.newDetectionCount = data.newDetectionCount || 0;
                this.lastSeenDetectionTime = data.lastSeenDetectionTime
                    ? new Date(data.lastSeenDetectionTime)
                    : null;
            }
        } catch (e) {
            console.warn('Failed to load badge state:', e);
        }
    },

    // Save badge state to localStorage
    saveBadgeState() {
        try {
            const data = {
                newDetectionCount: this.newDetectionCount,
                lastSeenDetectionTime: this.lastSeenDetectionTime
                    ? this.lastSeenDetectionTime.toISOString()
                    : null
            };
            localStorage.setItem('gently-badge-state', JSON.stringify(data));
        } catch (e) {
            console.warn('Failed to save badge state:', e);
        }
    },

    // Update the detection badge - applies .has-new class when there are new detections
    updateDetectionBadge() {
        const badge = document.getElementById('embryos-count');
        if (!badge) return;

        if (this.newDetectionCount > 0) {
            badge.textContent = this.newDetectionCount;
            badge.classList.add('has-new');
            badge.title = `${this.newDetectionCount} new detection${this.newDetectionCount > 1 ? 's' : ''} since last viewed`;
        } else {
            // Show active embryo count when no new detections
            const activeCount = Object.values(this.state.embryos).filter(e => !e.isComplete).length;
            badge.textContent = activeCount;
            badge.classList.remove('has-new');
            badge.title = `${activeCount} active embryo${activeCount !== 1 ? 's' : ''}`;
        }
    },

    // Clear the detection badge - called when user views the Embryos tab
    clearDetectionBadge() {
        this.newDetectionCount = 0;
        this.lastSeenDetectionTime = new Date();
        this.saveBadgeState();
        this.updateDetectionBadge();
    },

    // Increment detection count for a new positive detection
    incrementDetectionBadge() {
        this.newDetectionCount++;
        this.saveBadgeState();
        this.updateDetectionBadge();
    },

    // ==========================================
    // First-Run Contextual Hints
    // ==========================================

    /**
     * Show a first-run hint if it hasn't been dismissed
     * @param {string} key - Unique key for this hint
     * @param {string} title - Hint title
     * @param {string} message - Hint message
     * @param {string} targetSelector - CSS selector for where to insert the hint
     */
    showFirstRunHint(key, title, message, targetSelector) {
        // Check if already dismissed
        if (localStorage.getItem(`gently-hint-${key}`)) return;

        const target = document.querySelector(targetSelector);
        if (!target) return;

        // Don't show if already showing a hint here
        if (target.querySelector('.first-run-hint')) return;

        const hint = document.createElement('div');
        hint.className = 'first-run-hint';
        hint.innerHTML = `
            <div class="hint-header">
                <span class="hint-icon">&#x1F4A1;</span>
                <span class="hint-title">${title}</span>
            </div>
            <div class="hint-message">${message}</div>
            <div class="hint-actions">
                <button class="hint-dismiss" onclick="EmbryosManager.dismissHint('${key}')">Got it</button>
            </div>
        `;

        target.insertAdjacentElement('afterbegin', hint);
    },

    /**
     * Dismiss a hint and save to localStorage
     * @param {string} key - Hint key to dismiss
     */
    dismissHint(key) {
        localStorage.setItem(`gently-hint-${key}`, 'true');
        const hint = document.querySelector('.first-run-hint');
        if (hint) {
            hint.style.animation = 'hintSlideOut 0.2s ease forwards';
            setTimeout(() => hint.remove(), 200);
        }
    },

    // ==========================================
    // Smart Empty States
    // ==========================================

    /**
     * Render a smart empty state with contextual messaging
     * @param {string} type - Type of empty state
     * @returns {string} HTML for the empty state
     */
    renderSmartEmptyState(type) {
        const states = {
            'no-embryos': {
                icon: '&#x1F52C;',  // Microscope
                title: 'No embryos yet',
                message: 'Start a timelapse acquisition to begin tracking embryos. Configure your experiment in the Setup tab.',
                action: { label: 'Go to Setup', tab: 'calibration' }
            },
            'no-detections': {
                icon: '&#x1F441;',  // Eye
                title: 'No detections yet',
                message: 'The AI will analyze each timepoint and notify you when developmental events are detected. Typical first detection: 2-4 hours after start.',
                action: null
            },
            'experiment-idle': {
                icon: '&#x23F8;',  // Pause
                title: 'Experiment not running',
                message: 'No active timelapse. Configure and start an experiment to begin automated embryo monitoring.',
                action: { label: 'Setup Experiment', tab: 'calibration' }
            },
            'waiting-first': {
                icon: '&#x23F3;',  // Hourglass
                title: 'Waiting for first acquisition',
                message: 'The timelapse has started. The first volume should arrive shortly.',
                action: null
            }
        };

        const state = states[type] || states['no-embryos'];

        const actionHtml = state.action
            ? `<button class="empty-action" onclick="switchTab('${state.action.tab}')">${state.action.label}</button>`
            : '';

        return `
            <div class="smart-empty-state">
                <div class="empty-icon">${state.icon}</div>
                <div class="empty-title">${state.title}</div>
                <div class="empty-message">${state.message}</div>
                ${actionHtml}
            </div>
        `;
    },

    /**
     * Show appropriate first-run hints based on current state
     */
    showFirstRunHints() {
        // Hint for the embryo list
        if (!localStorage.getItem('gently-hint-embryos') && Object.keys(this.state.embryos).length > 0) {
            this.showFirstRunHint(
                'embryos',
                'Welcome to Embryo Monitoring',
                'This is where you\'ll track your experiment progress. Each embryo card shows acquisition status and AI-detected events. Click an embryo to see detailed detection reasoning.',
                '.embryo-list'
            );
        }

        // Hint for detection panel when there are detections
        const hasDetections = Object.values(this.detectionReasoning).some(r => r && r.length > 0);
        if (!localStorage.getItem('gently-hint-detections') && hasDetections) {
            this.showFirstRunHint(
                'detections',
                'AI Detection Analysis',
                'The AI evaluates each timepoint for developmental events. Positive detections are highlighted. Click "View full analysis" to see the AI\'s reasoning with the actual image.',
                '.reasoning-panel'
            );
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

    // Render a timeline sparkline showing detection/perception distribution
    renderTimelineSparkline(reasoning, totalTimepoints) {
        if (!reasoning || reasoning.length === 0) {
            return '';
        }

        // Separate verification events from regular detections/perceptions
        const verificationEvents = reasoning.filter(r => r.type === 'verification_completed');
        const detectionEvents = reasoning.filter(r => !r.type || !r.type.startsWith('verification_'));

        // Check if this is perception data (has stage field)
        const isPerceptionData = detectionEvents.some(r => r.stage);

        // Group by timepoint
        const timepoints = {};
        detectionEvents.forEach(r => {
            const tp = r.timepoint ?? 0;
            if (!timepoints[tp]) {
                timepoints[tp] = {
                    detected: false,
                    confidence: 'low',
                    type: 'detection',
                    stage: null,
                    is_hatching: false
                };
            }
            if (r.detected || r.is_hatching) {
                timepoints[tp].detected = true;
            }
            if (r.stage) {
                timepoints[tp].stage = r.stage;
            }
            if (r.is_hatching) {
                timepoints[tp].is_hatching = true;
            }
            // Keep highest confidence
            const confOrder = { 'high': 3, 'medium': 2, 'low': 1 };
            const existingConf = confOrder[timepoints[tp].confidence] || 0;
            const newConf = confOrder[r.confidence?.toLowerCase()] || 0;
            if (newConf > existingConf) {
                timepoints[tp].confidence = r.confidence?.toLowerCase() || 'low';
            }
        });

        // Add verification events as special timepoints
        verificationEvents.forEach((v, idx) => {
            const verKey = `v${idx}`;
            timepoints[verKey] = {
                type: 'verification',
                passed: v.consensus,
                consecutiveCount: v.consecutive_verified || 0,
                timestamp: v.timestamp
            };
        });

        // Sort: regular timepoints first, then verifications by timestamp
        const regularTps = Object.keys(timepoints)
            .filter(k => !k.startsWith('v'))
            .map(Number)
            .sort((a, b) => a - b);

        const verificationTps = Object.keys(timepoints)
            .filter(k => k.startsWith('v'))
            .sort((a, b) => {
                const timeA = new Date(timepoints[a].timestamp).getTime();
                const timeB = new Date(timepoints[b].timestamp).getTime();
                return timeA - timeB;
            });

        // Generate dots for regular timepoints
        let dotsHtml = regularTps.map(tp => {
            const data = timepoints[tp];
            const stage = data.stage;
            const isHatching = data.is_hatching;

            // For perception data, use stage-based coloring
            if (isPerceptionData && stage) {
                const stageClass = `stage-${stage.toLowerCase().replace('.', '')}`;
                const stageIcon = this.getStageIcon(stage);
                const title = isHatching
                    ? `T${tp}: ${this.formatStageName(stage)} - HATCHING!`
                    : `T${tp}: ${this.formatStageName(stage)}`;

                const detectorName = detectionEvents.find(r => r.timepoint === tp)?.detector_name || 'perception';
                return `<div class="eval-dot ${stageClass} ${isHatching ? 'hatching' : ''}"
                             title="${title}"
                             onclick="EmbryosManager.openDetailPanel('${detectorName}', ${tp})">
                            <span class="eval-dot-icon">${stageIcon}</span>
                        </div>`;
            }

            // Legacy detection behavior
            const isPositive = data.detected;
            const confClass = data.confidence || 'low';
            const className = isPositive ? 'positive' : confClass;
            const title = isPositive
                ? `T${tp}: DETECTED`
                : `T${tp}: Not detected (${data.confidence})`;

            const detectorName = detectionEvents.find(r => r.timepoint === tp)?.detector_name || '';
            return `<div class="eval-dot ${className}"
                         title="${title}"
                         onclick="EmbryosManager.openDetailPanel('${detectorName}', ${tp})">
                        <span class="eval-dot-label">T${tp}</span>
                    </div>`;
        }).join('');

        // Generate dots for verification events (distinct purple color)
        const verificationDotsHtml = verificationTps.map((key, idx) => {
            const data = timepoints[key];
            const passed = data.passed;
            const className = passed ? 'verification-passed' : 'verification-failed';
            const title = passed
                ? `Verification ${idx + 1}: PASSED (${data.consecutiveCount}/5)`
                : `Verification ${idx + 1}: FAILED`;

            return `<div class="eval-dot ${className}"
                         title="${title}">
                        <span class="eval-dot-label">V${idx + 1}</span>
                    </div>`;
        }).join('');

        const evalCount = regularTps.length;
        const positiveCount = regularTps.filter(tp => timepoints[tp].detected).length;
        const verificationCount = verificationTps.length;
        const passedCount = verificationTps.filter(k => timepoints[k].passed).length;

        // Count stage transitions for perception data
        const uniqueStages = [...new Set(regularTps.map(tp => timepoints[tp].stage).filter(Boolean))];
        const hatchingCount = regularTps.filter(tp => timepoints[tp].is_hatching).length;

        // Build count text
        let countText = `${evalCount} checked`;
        if (isPerceptionData) {
            if (uniqueStages.length > 0) countText += `, ${uniqueStages.length} stages`;
            if (hatchingCount > 0) countText += `, hatching detected`;
        } else {
            if (positiveCount > 0) countText += `, ${positiveCount} detected`;
        }
        if (verificationCount > 0) countText += `, ${passedCount}/${verificationCount} verified`;

        return `
            <div class="evaluation-dots">
                <div class="eval-dots-header">
                    <span class="eval-dots-title">Evaluations</span>
                    <span class="eval-dots-count">${countText}</span>
                </div>
                <div class="eval-dots-track">
                    ${dotsHtml}
                    ${verificationDotsHtml ? `<span class="eval-dots-separator"></span>${verificationDotsHtml}` : ''}
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
        const { detectionPoint } = context;
        // Note: reasoningText is NOT included in onclick to avoid bloated HTML and parsing issues

        // Build the onclick handler parameters (minimal - no long text)
        const params = {
            embryoId,
            start,
            end,
            detectionPoint: detectionPoint ?? null
        };

        // Escape for use in onclick attribute
        const paramsStr = JSON.stringify(params).replace(/'/g, "\\'").replace(/"/g, '&quot;');

        return `<span class="timepoint-link"
                      data-start="${start}"
                      data-end="${end}"
                      data-embryo="${embryoId}"
                      onclick="EmbryosManager.playTimepointRange(${paramsStr})"
                      title="Click to play timepoint ${start}${end !== start ? '-' + end : ''}">${text}</span>`;
    },

    /**
     * Handle click on a timepoint link - opens video player
     */
    async playTimepointRange(params) {
        const { embryoId, start, end, detectionPoint, reasoningText, stage, isHatching } = params;

        if (typeof TimepointPlayer !== 'undefined') {
            // For single timepoints, add context window of 5 frames before/after
            const contextWindow = 5;
            const actualStart = (start === end) ? Math.max(0, start - contextWindow) : start;
            const actualEnd = (start === end) ? end + contextWindow : end;

            await TimepointPlayer.openSequence(embryoId, actualStart, actualEnd, {
                vlmRange: { start, end },
                detectionPoint: detectionPoint ?? start,  // Highlight the clicked timepoint
                reasoningText,
                stage,
                isHatching: isHatching || false,
                bufferPercent: 0.15
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

            // Check for perception data (has stage) or legacy detections
            const hasPerceptionData = reasoning.some(r => r.stage);

            let latestEvent = null;
            if (hasPerceptionData) {
                // Get latest stage info, prefer hatching
                const hatchingEvent = reasoning.find(r => r.is_hatching);
                const latestReasoning = reasoning.filter(r => r.stage).slice(-1)[0];
                latestEvent = hatchingEvent || latestReasoning;
            } else {
                // Legacy: get latest positive detection
                const positiveDetections = reasoning.filter(r => r.detected);
                latestEvent = positiveDetections.length > 0
                    ? positiveDetections[positiveDetections.length - 1]
                    : null;
            }

            await TimepointPlayer.openSequence(embryoId, 0, null, {
                vlmRange: null,  // No specific VLM range for "play all"
                detectionPoint: latestEvent?.timepoint ?? null,
                reasoningText: latestEvent?.reasoning ?? null,
                stage: latestEvent?.stage ?? null,
                isHatching: latestEvent?.is_hatching ?? false,
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

    getEmbryoImagingDuration(embryo) {
        // Show per-embryo imaging duration (time from first to last acquisition)
        if (!embryo.firstAcquired) {
            return '';
        }
        const endTime = embryo.isComplete ? embryo.lastAcquired : new Date();
        const durationMs = endTime.getTime() - embryo.firstAcquired.getTime();
        return this.formatDuration(durationMs);
    },

    updateEmbryosCount() {
        // Use the new detection badge system which handles both
        // new detection notifications and active embryo counts
        this.updateDetectionBadge();
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
        this.updateEmbryosCount();
        this.render();
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    EmbryosManager.init();
});
