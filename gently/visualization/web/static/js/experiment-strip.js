/**
 * Experiment Status Strip Manager
 * Shows persistent experiment status at a glance in the header
 */

const ExperimentStrip = {
    lastCheck: null,  // Track when user last viewed
    newDetections: [],  // Detections since last check

    init() {
        // Load last check time from localStorage
        const saved = localStorage.getItem('gently-last-check');
        if (saved) {
            this.lastCheck = new Date(saved);
        } else {
            // Set to current time so existing detections aren't marked as "new"
            this.lastCheck = new Date();
            localStorage.setItem('gently-last-check', this.lastCheck.toISOString());
        }
        this.update();

        // Subscribe to events via ClientEventBus
        ClientEventBus.on('IMAGE_RECEIVED', (data) => {
            if (data?.uid) this.updateLatestFrame(data.uid, data.embryo_id);
        });
        ClientEventBus.on('VOLUME_ACQUIRED', (data) => {
            const projUid = data.projection_uid || data.volume_uid;
            if (projUid) this.updateLatestFrame(projUid, data.embryo_id);
        });
    },

    update() {
        const headerStatus = document.getElementById('header-status');
        if (!headerStatus) return;

        const state = EmbryosManager.state;
        const embryoCount = Object.keys(state.embryos).length;

        // Show/hide status based on whether there's an experiment
        if (embryoCount === 0 && state.status === 'IDLE') {
            headerStatus.classList.add('hidden');
            return;
        }
        headerStatus.classList.remove('hidden');

        // Update status indicator
        const indicator = document.getElementById('header-indicator');
        const statusText = document.getElementById('header-status-text');
        if (indicator && statusText) {
            indicator.className = 'status-indicator ' + state.status.toLowerCase();
            statusText.textContent = this.formatStatus(state.status);
        }

        // Update duration
        const durationEl = document.getElementById('header-duration');
        if (durationEl) {
            if (state.startedAt) {
                durationEl.textContent = EmbryosManager.formatDuration(Date.now() - state.startedAt.getTime());
            } else {
                durationEl.textContent = '';
            }
        }

        // Update embryo count
        const embryosEl = document.getElementById('header-embryos');
        if (embryosEl) {
            const activeCount = Object.values(state.embryos).filter(e => !e.isComplete).length;
            const totalCount = embryoCount;
            embryosEl.textContent = `${activeCount}/${totalCount}`;
        }

        // Update next countdown
        const countdownEl = document.getElementById('header-countdown');
        if (countdownEl) {
            const nextSeconds = this.getNextAcquisitionSeconds();
            countdownEl.textContent = nextSeconds > 0 ? EmbryosManager.formatCountdown(nextSeconds) : '--:--';
        }

        // Update detection alert
        this.updateDetectionAlert();
    },

    getNextAcquisitionSeconds() {
        const embryos = Object.values(EmbryosManager.state.embryos);
        const activeEmbryos = embryos.filter(e => !e.isComplete);
        if (activeEmbryos.length === 0) return 0;

        // Find the next acquisition (minimum countdown)
        let minSeconds = Infinity;
        activeEmbryos.forEach(embryo => {
            if (embryo.lastAcquired) {
                // Calculate remaining time based on last acquisition
                const elapsed = (Date.now() - new Date(embryo.lastAcquired).getTime()) / 1000;
                const remaining = Math.max(0, embryo.intervalSeconds - elapsed);
                minSeconds = Math.min(minSeconds, remaining);
            } else if (embryo.intervalSeconds) {
                // No acquisition yet - use full interval as estimate
                // This shows countdown before first acquisition
                minSeconds = Math.min(minSeconds, embryo.intervalSeconds);
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
        const alert = document.getElementById('header-alert');
        if (!alert) return;

        // Check if using perception system
        let isPerception = false;
        let newEventCount = 0;
        let latestEvent = null;
        let hatchingDetected = false;

        Object.entries(EmbryosManager.detectionReasoning).forEach(([embryoId, reasoning]) => {
            // Check for perception data (has stage field)
            const hasStages = reasoning.some(r => r.stage);
            if (hasStages) isPerception = true;

            // For perception: count hatching events as notable
            // For legacy: count detected events
            const notableEvents = hasStages
                ? reasoning.filter(r => r.is_hatching)
                : reasoning.filter(r => r.detected);

            notableEvents.forEach(d => {
                const eventTime = d.timestamp ? new Date(d.timestamp) : null;
                const isNew = !this.lastCheck || (eventTime && eventTime > this.lastCheck);

                if (isNew) {
                    newEventCount++;
                    if (d.is_hatching) hatchingDetected = true;
                }

                if (!latestEvent || (eventTime && eventTime > new Date(latestEvent.timestamp))) {
                    latestEvent = { ...d, embryoId };
                }
            });
        });

        if (newEventCount === 0) {
            alert.classList.add('hidden');
            return;
        }

        alert.classList.remove('hidden');

        const badge = document.getElementById('header-alert-badge');
        const text = document.getElementById('header-alert-text');

        if (badge) badge.textContent = newEventCount;
        if (text && latestEvent) {
            if (isPerception && hatchingDetected) {
                text.textContent = 'Hatching detected!';
            } else if (isPerception && latestEvent.stage) {
                text.textContent = `Stage: ${EmbryosManager.formatStageName(latestEvent.stage)}`;
            } else {
                text.textContent = `${EmbryosManager.formatDetectorName(latestEvent.detector_name)} detected`;
            }
        }
    },

    handleAlertClick() {
        // Switch to Tasks tab and select the embryo with the latest detection
        switchTab('tasks');

        // Find embryo with most recent detection
        let latestDetection = null;
        let latestEmbryoId = null;

        Object.entries(EmbryosManager.detectionReasoning).forEach(([embryoId, reasoning]) => {
            const positives = reasoning.filter(r => r.detected);
            positives.forEach(d => {
                if (!latestDetection || new Date(d.timestamp) > new Date(latestDetection.timestamp)) {
                    latestDetection = d;
                    latestEmbryoId = embryoId;
                }
            });
        });

        if (latestEmbryoId) {
            EmbryosManager.selectEmbryo(latestEmbryoId);
        }

        // Mark detections as seen so alert hides
        this.markChecked();
    },

    markChecked() {
        this.lastCheck = new Date();
        localStorage.setItem('gently-last-check', this.lastCheck.toISOString());
        this.newDetections = [];
        this.updateDetectionAlert();
    },

    // Latest frame tracking
    latestFrameUid: null,
    latestFrameTime: null,

    /**
     * Update the latest frame preview in the experiment strip
     * Called when a new volume/image is acquired
     * @param {string} uid - The image UID for the thumbnail
     * @param {string} embryoId - Optional embryo ID for context
     */
    updateLatestFrame(uid, embryoId = null) {
        if (!uid) return;

        this.latestFrameUid = uid;
        this.latestFrameTime = new Date();

        const thumb = document.getElementById('latest-frame-thumb');
        const placeholder = document.getElementById('latest-frame-placeholder');
        const timeEl = document.getElementById('latest-frame-time');

        if (thumb && placeholder) {
            thumb.src = `/api/images/${uid}/png?size=96`;
            thumb.style.display = 'block';
            placeholder.style.display = 'none';
        }

        if (timeEl) {
            this.updateLatestFrameTime();
        }
    },

    /**
     * Update the "X ago" timestamp for the latest frame
     */
    updateLatestFrameTime() {
        const timeEl = document.getElementById('latest-frame-time');
        if (!timeEl || !this.latestFrameTime) {
            if (timeEl) timeEl.textContent = '--';
            return;
        }

        const elapsed = Date.now() - this.latestFrameTime.getTime();
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);

        if (minutes > 60) {
            const hours = Math.floor(minutes / 60);
            timeEl.textContent = `${hours}h ago`;
        } else if (minutes > 0) {
            timeEl.textContent = `${minutes}m ago`;
        } else if (seconds > 5) {
            timeEl.textContent = `${seconds}s ago`;
        } else {
            timeEl.textContent = 'now';
        }
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    ExperimentStrip.init();
    // Update experiment strip every second
    setInterval(() => ExperimentStrip.update(), 1000);
});
