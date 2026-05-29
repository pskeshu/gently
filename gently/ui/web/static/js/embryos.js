/**
 * Embryos Tab - Timelapse Task Tracking
 * Displays active timelapse tasks with per-embryo breakdown
 */

const EmbryosManager = {
    // Session ID for tracking experiment boundaries
    // When session_id changes, all state is cleared (new experiment)
    currentSessionId: null,

    state: {
        status: 'IDLE', // IDLE, RUNNING, PAUSED, COMPLETED, STOPPED, FAILED
        startedAt: null,
        embryos: {},  // embryo_id -> EmbryoTaskState
        totalTimepoints: 0,
        baseInterval: 120
    },

    // Detection reasoning cache (per-embryo)
    detectionReasoning: {},  // embryo_id -> list of detection results with reasoning
    MAX_REASONING_PER_EMBRYO: 200,

    // Track if we've reconciled with server (prevents showing stale cached data)
    hasReconciledWithServer: false,

    // Currently selected embryo for detail view
    selectedEmbryoId: null,

    // Expanded image states
    expandedImages: {},  // detection index -> true/false

    // Cached DOM element references for countdown updates (invalidated on re-render)
    _countdownCache: null,

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

    // Multi-view system
    currentView: 'default',  // 'default' | 'board' | 'filmstrip' | 'vitals'

    // Dashboard config (loaded from localStorage)
    dashboardConfig: {
        defaultView: 'default',
        board: {
            columns: ['stage', 'clock', 'stereo', 'pace', 'eta', 'sparkline', 'alert'],
            sparklineLength: 20,
            warnOvertimeRatio: 1.5,
            criticalOvertimeRatio: 2.5
        },
        filmstrip: {
            thumbnailSize: 56,
            showStageLabels: true,
            skipInterval: 1,
            borderEncoding: 'stage'
        },
        vitals: {
            temperatureModel: '20C',
            showExpectedLine: true,
            timeAxis: 'elapsed'
        },
        detail: {
            imageSplitRatio: 40,
            autoAdvance: false,
            showContrastive: true
        },
        ambient: {
            enabled: true,
            sensitivity: 'normal',
            audioTick: false
        }
    },

    // Stage timing reference (20C, minutes from first cell division)
    STAGE_TIMING: {
        'early': 0, '1_cell': 0, '2_cell': 40, '4_cell': 70, 'bean': 120,
        'comma': 180, '1_5_fold': 240, '2_fold': 360,
        'pretzel': 420, '3_fold': 420, 'hatching': 510, 'hatched': 570
    },

    // Stage ordinal for sparkline/vitals
    STAGE_ORDINAL: {
        'early': 0, '1_cell': 0, '2_cell': 1, '4_cell': 2, 'bean': 3,
        'comma': 4, '1_5_fold': 5, '2_fold': 6,
        'pretzel': 7, '3_fold': 7, 'hatching': 8, 'hatched': 9
    },

    STAGE_COLORS: {
        'early': '#8b949e', '1_cell': '#8b949e', '2_cell': '#8b949e', '4_cell': '#8b949e',
        'bean': '#60a5fa', 'comma': '#60a5fa',
        '1_5_fold': '#4ade80', '2_fold': '#4ade80',
        'pretzel': '#c084fc', '3_fold': '#c084fc',
        'hatching': '#fb923c', 'hatched': '#f472b6'
    },

    // Badge state for new detection notifications
    newDetectionCount: 0,  // Count of NEW detections since user last viewed
    lastSeenDetectionTime: null,  // When user last viewed the Embryos tab

    countdownInterval: null,
    storageKey: 'gently-tasks-state',

    // Helper to normalize confidence values (handles both numeric 0.95 and string "high")
    normalizeConfidence(confidence) {
        if (typeof confidence === 'number') {
            return confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
        }
        if (typeof confidence === 'string') {
            return confidence.toLowerCase();
        }
        return 'unknown';
    },

    init() {
        // Load dashboard config from localStorage
        this.loadDashboardConfig();
        // Restore state from localStorage
        this.loadState();
        // Load detection agreements
        this.loadAgreements();
        // Load badge state (new detection count)
        this.loadBadgeState();
        // Start countdown update timer
        this.startCountdownUpdates();
        // Set initial view from config
        this.currentView = this.dashboardConfig.defaultView || 'default';
        // Initial render (reasoning will show loading until server reconciles)
        this.render();
        // Don't auto-open detail until we've reconciled with server
        // (prevents showing stale cached data from previous session)
        // Update badge on init
        this.updateDetectionBadge();
        // Setup view switcher
        this._setupViewSwitcher();
        // Setup keyboard shortcuts for views
        this._setupViewKeyboard();
        // Start ambient pulse updates
        this._startAmbientPulse();

        // Subscribe to events via ClientEventBus
        this._subscribeToEvents();
    },

    _subscribeToEvents() {
        ClientEventBus.on('ACQUISITION_STARTED', (data) => this.handleAcquisitionStarted(data));
        ClientEventBus.on('ACQUISITION_COMPLETED', (data) => this.handleAcquisitionCompleted(data));
        ClientEventBus.on('VOLUME_ACQUIRED', (data) => this.handleVolumeAcquired(data));
        ClientEventBus.on('DETECTOR_EVALUATED', (data) => this.handleDetectorEvaluated(data));
        ClientEventBus.on('DETECTION_TRIGGERED', (data) => this.handleDetectionTriggered(data));
        ClientEventBus.on('STATUS_CHANGED', (data) => this.handleStatusChanged(data));
        ClientEventBus.on('HATCHING_DETECTED', (data) => this.handleDetectionTriggered({
            embryo_id: data.embryo_id,
            detector_name: 'hatching',
            ...data
        }));
        ClientEventBus.on('VERIFICATION_STARTED', (data) => this.handleVerificationStarted(data));
        ClientEventBus.on('VERIFICATION_STRATEGY', (data) => this.handleVerificationStrategy(data));
        ClientEventBus.on('VERIFICATION_PROGRESS', (data) => this.handleVerificationProgress(data));
        ClientEventBus.on('VERIFICATION_COMPLETED', (data) => this.handleVerificationCompleted(data));
        ClientEventBus.on('TIMELAPSE_STATE', (data) => this.reconcileWithServerState(data));
    },

    // ==========================================
    // View Switching System
    // ==========================================

    _setupViewSwitcher() {
        const switcher = document.getElementById('view-switcher');
        if (!switcher) return;
        switcher.addEventListener('click', (e) => {
            const btn = e.target.closest('.view-btn');
            if (!btn) return;
            this.switchView(btn.dataset.view);
        });
        // Set initial active state
        this._updateViewButtons();
        this.switchView(this.currentView);
    },

    _setupViewKeyboard() {
        // Note: arrow-key timepoint navigation lives in
        // ``setupKeyboardNavigation()`` further down (it gates on
        // ``currentDetailItem``, which all views set on detail open).
        // Here we only handle view-switch hotkeys (1..4).
        document.addEventListener('keydown', (e) => {
            if (typeof state !== 'undefined' && state.tab !== TABS.EMBRYOS) return;
            if (e.target.matches('input, textarea, select, [contenteditable]')) return;
            const viewMap = { '1': 'default', '2': 'board', '3': 'filmstrip', '4': 'vitals' };
            if (viewMap[e.key]) {
                e.preventDefault();
                this.switchView(viewMap[e.key]);
            }
        });
    },

    switchView(viewName) {
        if (!['default', 'board', 'filmstrip', 'vitals'].includes(viewName)) return;
        this.currentView = viewName;
        // Hide all view containers
        ['default', 'board', 'filmstrip', 'vitals'].forEach(v => {
            const el = document.getElementById(`view-${v}`);
            if (el) el.style.display = 'none';
        });
        // Show active view
        const activeEl = document.getElementById(`view-${viewName}`);
        if (activeEl) {
            activeEl.style.display = viewName === 'default' ? 'flex' : '';
        }
        // Update buttons
        this._updateViewButtons();
        // Render the active view's content
        this._renderActiveView();
    },

    _updateViewButtons() {
        const switcher = document.getElementById('view-switcher');
        if (!switcher) return;
        switcher.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === this.currentView);
        });
    },

    _renderActiveView() {
        switch (this.currentView) {
            case 'default':
                this.renderEmbryoCards();
                this.renderReasoningPanel();
                break;
            case 'board':
                this.renderBoardView();
                break;
            case 'filmstrip':
                this.renderFilmstripView();
                break;
            case 'vitals':
                this.renderVitalsView();
                break;
        }
    },

    // ==========================================
    // Dashboard Config
    // ==========================================

    loadDashboardConfig() {
        try {
            const stored = localStorage.getItem('gently-dashboard-config');
            if (stored) {
                const parsed = JSON.parse(stored);
                // Deep merge with defaults
                this.dashboardConfig = this._deepMerge(this.dashboardConfig, parsed);
            }
            // Migrate legacy board columns: drop the never-populated
            // 'confidence' column and the misleading 'rate' column in
            // favour of clock/stereo/pace. Idempotent — runs on every load.
            const cols = this.dashboardConfig.board?.columns;
            if (Array.isArray(cols)) {
                const filtered = cols.filter(c => c !== 'confidence' && c !== 'rate');
                const ensure = (key, after) => {
                    if (filtered.includes(key)) return;
                    const idx = filtered.indexOf(after);
                    if (idx === -1) filtered.push(key);
                    else filtered.splice(idx + 1, 0, key);
                };
                ensure('clock', 'stage');
                ensure('stereo', 'clock');
                ensure('pace', 'stereo');
                this.dashboardConfig.board.columns = filtered;
            }
        } catch (e) {
            console.warn('Failed to load dashboard config:', e);
        }
    },

    saveDashboardConfig() {
        try {
            localStorage.setItem('gently-dashboard-config', JSON.stringify(this.dashboardConfig));
        } catch (e) {
            console.warn('Failed to save dashboard config:', e);
        }
    },

    _deepMerge(target, source) {
        const result = { ...target };
        for (const key of Object.keys(source)) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this._deepMerge(target[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        return result;
    },

    // ==========================================
    // Ambient Heartbeat Pulse
    // ==========================================

    _startAmbientPulse() {
        // Update every 10 seconds
        this._ambientInterval = setInterval(() => this.updateAmbientPulse(), 10000);
        this.updateAmbientPulse();
    },

    updateAmbientPulse() {
        const el = document.getElementById('ambient-pulse');
        if (!el) return;
        if (!this.dashboardConfig.ambient.enabled) {
            el.className = 'ambient-pulse';
            return;
        }
        const health = this.computeHealthScore();
        el.className = `ambient-pulse ${health}`;
    },

    computeHealthScore() {
        const embryos = Object.values(this.state.embryos);
        if (embryos.length === 0) return 'normal';

        const sensitivity = this.dashboardConfig.ambient.sensitivity;
        const warnThreshold = sensitivity === 'high' ? 1.2 : sensitivity === 'low' ? 2.0 : 1.5;

        for (const embryo of embryos) {
            const reasoning = this.detectionReasoning[embryo.embryoId];
            if (!reasoning?.length) continue;
            const latest = reasoning[reasoning.length - 1];

            // Check for hatching
            if (latest.stage === 'hatching' || latest.stage === 'hatched') return 'hatching';

            // Check for arrested
            const temporal = latest.temporal_analysis;
            if (temporal?.is_potentially_arrested) return 'critical';
        }

        // Check for slow/uncertain embryos
        for (const embryo of embryos) {
            const reasoning = this.detectionReasoning[embryo.embryoId];
            if (!reasoning?.length) continue;
            const latest = reasoning[reasoning.length - 1];
            const overtime = latest.temporal_analysis?.overtime_ratio;
            if (overtime && overtime > warnThreshold) return 'warning';
            if (this.normalizeConfidence(latest.confidence) === 'low') return 'warning';
        }

        return 'normal';
    },

    // ==========================================
    // Board View
    // ==========================================

    renderBoardView() {
        const container = document.getElementById('view-board');
        if (!container) return;

        const embryos = Object.values(this.state.embryos);
        if (embryos.length === 0) {
            container.innerHTML = `<div class="board-empty"><div class="reasoning-empty-icon">&#x1F4CA;</div><div class="reasoning-empty-text">No embryos to display</div></div>`;
            return;
        }

        // Sort embryos
        embryos.sort((a, b) => {
            if (a.isComplete !== b.isComplete) return a.isComplete ? 1 : -1;
            return a.embryoId.localeCompare(b.embryoId);
        });

        const cols = this.dashboardConfig.board.columns;
        const headerHtml = `
            <div class="board-header">
                <span class="board-col board-col-embryo">Embryo</span>
                ${cols.includes('stage') ? '<span class="board-col board-col-stage">Stage</span>' : ''}
                ${cols.includes('clock') ? '<span class="board-col board-col-clock" title="Clock time in current stage">Clock</span>' : ''}
                ${cols.includes('stereo') ? '<span class="board-col board-col-stereo" title="Stereotypic developmental position (20°C reference)">Stereo</span>' : ''}
                ${cols.includes('pace') ? '<span class="board-col board-col-pace" title="Clock / stereotypic time — 1.0× means on reference pace">Pace</span>' : ''}
                ${cols.includes('eta') ? '<span class="board-col board-col-eta" title="Estimated clock-time to hatch, pace-corrected">ETA</span>' : ''}
                ${cols.includes('sparkline') ? '<span class="board-col board-col-spark">Progression</span>' : ''}
                ${cols.includes('alert') ? '<span class="board-col board-col-alert">Alert</span>' : ''}
            </div>
        `;

        const rowsHtml = embryos.map(embryo => this._renderBoardRow(embryo)).join('');

        container.innerHTML = `
            <div class="board-table">
                ${headerHtml}
                <div class="board-rows">${rowsHtml}</div>
            </div>
            <div class="board-detail" id="board-detail"></div>
        `;

        // Add click handlers
        container.querySelectorAll('.board-row').forEach(row => {
            row.addEventListener('click', () => {
                const eid = row.dataset.embryoId;
                this.selectedEmbryoId = eid;
                // Toggle expansion
                const detail = document.getElementById('board-detail');
                const wasOpen = row.classList.contains('expanded');
                container.querySelectorAll('.board-row').forEach(r => r.classList.remove('expanded'));
                if (!wasOpen) {
                    row.classList.add('expanded');
                    this._renderBoardDetail(eid, detail);
                } else {
                    detail.innerHTML = '';
                }
            });
        });
    },

    _renderBoardRow(embryo) {
        const reasoning = this.detectionReasoning[embryo.embryoId] || [];
        const latest = reasoning.length > 0 ? reasoning[reasoning.length - 1] : null;
        const cols = this.dashboardConfig.board.columns;

        const stage = latest?.stage || embryo.current_stage || '—';
        const stageIcon = this.getStageIcon(stage);
        const stageName = this.formatStageName(stage);

        const align = this._computeAlignment(latest);
        const overtime = align?.overtime;

        const clockText = align ? this._formatMinutes(align.inStageClockMin) : '—';
        const stereoText = align ? this._formatStereoLabel(align) : '—';
        const pace = align ? this._formatPace(align) : { text: '—', className: '' };
        const eta = align ? this._formatEta(align) : '—';

        const sparklineSvg = cols.includes('sparkline') ? this._renderBoardSparkline(reasoning) : '';

        const arrested = latest?.temporal_analysis?.is_potentially_arrested;
        const slow = overtime && overtime > (this.dashboardConfig.board.warnOvertimeRatio || 1.5);
        let alertHtml = '<span class="board-alert-none">—</span>';
        if (arrested) {
            alertHtml = '<span class="board-alert board-alert-critical">⚠ arrested</span>';
        } else if (slow) {
            alertHtml = `<span class="board-alert board-alert-warn">⚠ slow ${overtime.toFixed(1)}×</span>`;
        }

        const status = embryo.isComplete ? 'complete' : embryo.lastError ? 'error' : 'running';
        const isExpanded = this.selectedEmbryoId === embryo.embryoId;

        return `
            <div class="board-row ${status} ${isExpanded ? 'expanded' : ''}" data-embryo-id="${embryo.embryoId}">
                <span class="board-col board-col-embryo">
                    <span class="board-status-dot ${status}">●</span>
                    <span class="board-embryo-name">${embryo.embryoId.replace(/embryo_?/i, 'E')}</span>
                </span>
                ${cols.includes('stage') ? `<span class="board-col board-col-stage"><span class="board-stage-badge" style="color:${this.STAGE_COLORS[stage] || 'var(--text)'}">${stageIcon} ${stageName}</span></span>` : ''}
                ${cols.includes('clock') ? `<span class="board-col board-col-clock">${clockText}</span>` : ''}
                ${cols.includes('stereo') ? `<span class="board-col board-col-stereo">${stereoText}</span>` : ''}
                ${cols.includes('pace') ? `<span class="board-col board-col-pace ${pace.className}">${pace.text}</span>` : ''}
                ${cols.includes('eta') ? `<span class="board-col board-col-eta">${eta}</span>` : ''}
                ${cols.includes('sparkline') ? `<span class="board-col board-col-spark">${sparklineSvg}</span>` : ''}
                ${cols.includes('alert') ? `<span class="board-col board-col-alert">${alertHtml}</span>` : ''}
            </div>
        `;
    },

    /** Compute clock↔stereotypic alignment from perception temporal_analysis.
     *
     * Definitions:
     *   inStageClockMin  — wall-clock minutes elapsed in current stage
     *   inStageStereoMin — stereotypic minutes "used" within the stage,
     *                      capped at the stage's expected duration. An
     *                      overdue embryo is stuck at the stage end in
     *                      stereo time while clock keeps ticking.
     *   overtime         — ratio inStageClockMin / expected_duration.
     *                      >1 means the embryo has spent more clock time
     *                      in the stage than the reference 20°C textbook
     *                      duration. <1 just means "still within stage" —
     *                      no slow/fast signal yet.
     *   stereoAgeMin     — total stereotypic age, anchored at the start
     *                      minute of the current stage in the reference
     *                      table plus the (capped) in-stage stereo offset.
     */
    _computeAlignment(latest) {
        const ta = latest?.temporal_analysis;
        if (!ta || !ta.current_stage) return null;
        const stage = ta.current_stage;
        const stageStart = this.STAGE_TIMING[stage];
        if (stageStart == null) return null;

        const expDur = Number(ta.expected_duration_min) || 0;
        const inClock = Number(ta.time_in_stage_min) || 0;
        const overtime = Number(ta.overtime_ratio) || 0;

        const inStereo = expDur > 0 ? Math.min(inClock, expDur) : inClock;
        const stereoAge = stageStart + inStereo;

        return {
            stage,
            stageStart,
            expDur,
            inStageClockMin: inClock,
            inStageStereoMin: inStereo,
            stereoAgeMin: stereoAge,
            overtime,
        };
    },

    /** Render the stereo cell: "≈early", "≈bean +12m", or "≈comma +88m ⚠"
     * when overdue (stereo capped at stage end while clock keeps running). */
    _formatStereoLabel(align) {
        const stageName = this.formatStageName(align.stage);
        const offsetMin = Math.round(align.inStageStereoMin);
        const overdue = align.expDur > 0 && align.inStageClockMin > align.expDur + 1;
        const offsetStr = offsetMin > 0 ? ` +${offsetMin}m` : '';
        const overdueMark = overdue ? ' <span class="stereo-overdue" title="Clock ran past expected stage duration">⚠</span>' : '';
        return `≈${stageName}${offsetStr}${overdueMark}`;
    },

    _formatPace(align) {
        // Only emit a pace signal once we have meaningful clock data.
        // Within the first few minutes the ratio is tiny and noisy — show
        // a dashed placeholder so the column doesn't lie about precision.
        const NORMAL_BAND = 1.05;
        const SLOW_BAND = 1.5;
        if (align.inStageClockMin < 1 || align.expDur <= 0) {
            return { text: '—', className: 'pace-unknown' };
        }
        const r = align.overtime;
        if (r <= NORMAL_BAND) {
            return { text: '1.0×', className: 'pace-normal' };
        }
        if (r <= SLOW_BAND) {
            return { text: `${r.toFixed(1)}× slow`, className: 'pace-slow' };
        }
        return { text: `⚠ ${r.toFixed(1)}×`, className: 'pace-slow-bad' };
    },

    /** ETA in hours from current stereotypic position to hatched, scaled
     * by observed pace when the embryo is demonstrably slow. */
    _formatEta(align) {
        const hatchStereo = this.STAGE_TIMING['hatched'] || 570;
        const remainStereo = hatchStereo - align.stereoAgeMin;
        if (remainStereo <= 0) return 'done';
        const paceFactor = align.overtime > 1.05 ? align.overtime : 1.0;
        const remainClockMin = remainStereo * paceFactor;
        return `~${(remainClockMin / 60).toFixed(1)}h`;
    },

    /** Compact minute formatter: "45s" / "10m" / "1h 22m" / "3h". */
    _formatMinutes(min) {
        if (min == null || !isFinite(min)) return '—';
        if (min < 1) return `${Math.round(min * 60)}s`;
        if (min < 60) return `${Math.round(min)}m`;
        const h = Math.floor(min / 60);
        const m = Math.round(min - h * 60);
        return m > 0 ? `${h}h ${m}m` : `${h}h`;
    },

    _renderBoardSparkline(reasoning) {
        if (!reasoning.length) return '';
        const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));
        const maxItems = this.dashboardConfig.board.sparklineLength || 20;
        const items = sorted.slice(-maxItems);

        const width = 160;
        const height = 28;
        const maxOrd = 9; // hatched
        const step = width / Math.max(items.length - 1, 1);

        let pathD = '';
        let lastColor = '#8b949e';
        items.forEach((item, i) => {
            const ord = this.STAGE_ORDINAL[item.stage] ?? 0;
            const x = i * step;
            const y = height - (ord / maxOrd) * (height - 4) - 2;
            pathD += i === 0 ? `M${x},${y}` : `L${x},${y}`;
            lastColor = this.STAGE_COLORS[item.stage] || '#8b949e';
        });

        return `<svg class="board-sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <path d="${pathD}" fill="none" stroke="${lastColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>
        </svg>`;
    },

    _renderBoardDetail(embryoId, container) {
        if (!container) return;
        const reasoning = this.detectionReasoning[embryoId] || [];
        if (reasoning.length === 0) {
            container.innerHTML = '<div class="board-detail-empty">No evaluations yet</div>';
            return;
        }
        // Show latest evaluation detail
        const sorted = [...reasoning].sort((a, b) => (b.timepoint ?? 0) - (a.timepoint ?? 0));
        const latest = sorted[0];
        this.currentDetailItem = latest;
        container.innerHTML = `<div class="board-detail-content">${this.renderDetailPanel(latest)}</div>`;
    },

    // ==========================================
    // Filmstrip View
    // ==========================================

    renderFilmstripView() {
        const container = document.getElementById('view-filmstrip');
        if (!container) return;

        // Capture scroll state before we blow away the DOM. If the user was
        // at (or near) the right edge — the "leading edge" showing the newest
        // frames — auto-follow to keep them pinned to newest. Otherwise
        // preserve their exact scroll position so the strip doesn't jump.
        const prevScroll = (() => {
            const sc = container.querySelector('.filmstrip-container');
            if (!sc) return null;
            const max = sc.scrollWidth - sc.clientWidth;
            const FOLLOW_THRESHOLD_PX = 24;  // within 24px of the right edge
            return {
                left: sc.scrollLeft,
                wasAtRightEdge: max <= 0 || sc.scrollLeft >= max - FOLLOW_THRESHOLD_PX,
            };
        })();

        const embryos = Object.values(this.state.embryos);
        if (embryos.length === 0) {
            container.innerHTML = `<div class="board-empty"><div class="reasoning-empty-icon">&#x1F3AC;</div><div class="reasoning-empty-text">No embryos to display</div></div>`;
            return;
        }

        embryos.sort((a, b) => a.embryoId.localeCompare(b.embryoId));
        const config = this.dashboardConfig.filmstrip;
        const thumbSize = config.thumbnailSize || 56;

        let html = '<div class="filmstrip-container">';
        for (const embryo of embryos) {
            const reasoning = this.detectionReasoning[embryo.embryoId] || [];
            const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));

            // Apply skip interval
            const skip = config.skipInterval || 1;
            const filtered = skip > 1 ? sorted.filter((_, i) => i % skip === 0 || i === sorted.length - 1) : sorted;

            const shortName = embryo.embryoId.replace(/embryo_?/i, 'E');
            const latestStage = sorted.length > 0 ? this.formatStageName(sorted[sorted.length - 1].stage) : '—';
            const isTerminated = !!embryo.isComplete;
            const termReason = embryo.completionReason || '';
            // Short label for the badge — humanise the no_object terminal
            // reason, otherwise keep the first clause of whatever the
            // backend sent so the user still gets a hint.
            const termBadge = isTerminated
                ? (termReason.includes('no_object') ? 'HATCHED?' : 'STOPPED')
                : '';
            const termTooltip = isTerminated
                ? `Terminated — ${termReason || 'no reason given'}`
                : '';

            html += `<div class="filmstrip-row${isTerminated ? ' terminated' : ''}"${termTooltip ? ` title="${termTooltip.replace(/"/g, '&quot;')}"` : ''}>`;
            html += `<div class="filmstrip-label">
                <span class="filmstrip-name">${shortName}</span>
                <span class="filmstrip-stage">${latestStage}</span>
                <span class="filmstrip-count">${reasoning.length} eval</span>
                ${isTerminated ? `<span class="filmstrip-terminated-badge">${termBadge}</span>` : ''}
            </div>`;
            html += `<div class="filmstrip-thumbs">`;

            for (const item of filtered) {
                const imageUid = item.image_uid || item.projection_uid;
                const isPending = item._pending === true;
                const stage = item.stage || (isPending ? 'pending' : '—');
                const stageColor = isPending
                    ? 'var(--text-muted)'
                    : (this.STAGE_COLORS[stage] || '#8b949e');
                const confNorm = isPending
                    ? 'analyzing…'
                    : this.normalizeConfidence(item.confidence);
                const stageLabel = isPending ? '…' : this.formatStageName(stage);
                const cellClass = `filmstrip-cell${isPending ? ' pending' : ''}`;

                html += `<div class="${cellClass}" data-embryo-id="${embryo.embryoId}" data-timepoint="${item.timepoint}" title="T${item.timepoint} — ${stageLabel} — ${confNorm}">`;
                if (imageUid) {
                    html += `<img class="filmstrip-thumb" src="/api/images/${imageUid}/png?size=${thumbSize * 2}" loading="lazy" width="${thumbSize}" height="${thumbSize}" style="border-color:${stageColor}"/>`;
                } else {
                    html += `<div class="filmstrip-placeholder" style="width:${thumbSize}px;height:${thumbSize}px;border-color:${stageColor}">T${item.timepoint}</div>`;
                }
                if (config.showStageLabels) {
                    html += `<span class="filmstrip-stage-label" style="color:${stageColor}">${stageLabel}</span>`;
                }
                html += `</div>`;
            }

            html += `</div></div>`;
        }
        html += '</div>';
        html += '<div class="filmstrip-detail" id="filmstrip-detail"></div>';
        container.innerHTML = html;

        // Click handlers
        container.querySelectorAll('.filmstrip-cell').forEach(cell => {
            cell.addEventListener('click', () => {
                const eid = cell.dataset.embryoId;
                const tp = parseInt(cell.dataset.timepoint);
                this.selectedEmbryoId = eid;
                // Find the matching item
                const reasoning = this.detectionReasoning[eid] || [];
                const item = reasoning.find(r => r.timepoint === tp);
                if (item) {
                    container.querySelectorAll('.filmstrip-cell').forEach(c => c.classList.remove('active'));
                    cell.classList.add('active');
                    this.currentDetailItem = item;
                    const detail = document.getElementById('filmstrip-detail');
                    if (detail) {
                        detail.innerHTML = `<div class="filmstrip-detail-content">${this.renderDetailPanel(item)}</div>`;
                        this.initChatPanel(eid, tp);
                    }
                }
            });
        });

        // Convert vertical scroll to horizontal on the single shared
        // filmstrip scroll container, BUT only while there's still
        // horizontal room to scroll in the wheel's direction. When the
        // container is at the left/right edge for the requested direction,
        // let the wheel event bubble so the page (and the detail panel
        // below the rows) can scroll vertically. Shift+wheel always
        // bubbles for vertical (standard convention).
        const scrollContainer = container.querySelector('.filmstrip-container');
        if (scrollContainer) {
            scrollContainer.addEventListener('wheel', (e) => {
                if (e.shiftKey || e.deltaY === 0) return;
                const maxScroll = scrollContainer.scrollWidth - scrollContainer.clientWidth;
                const atLeftEdge = scrollContainer.scrollLeft <= 0 && e.deltaY < 0;
                const atRightEdge = scrollContainer.scrollLeft >= maxScroll && e.deltaY > 0;
                if (atLeftEdge || atRightEdge || maxScroll <= 0) {
                    // Nothing left to scroll horizontally — let the wheel
                    // bubble so vertical page/detail scroll takes over.
                    return;
                }
                e.preventDefault();
                scrollContainer.scrollLeft += e.deltaY;
            }, { passive: false });
        }

        // Restore any open detail panel — re-render strips wipe the
        // filmstrip-detail container, so paint the previously-selected
        // item back in.
        if (this.currentDetailItem && this.selectedEmbryoId) {
            const tp = this.currentDetailItem.timepoint;
            const cell = container.querySelector(
                `.filmstrip-cell[data-embryo-id="${this.selectedEmbryoId}"][data-timepoint="${tp}"]`
            );
            if (cell) cell.classList.add('active');
            const detail = document.getElementById('filmstrip-detail');
            if (detail) {
                detail.innerHTML = `<div class="filmstrip-detail-content">${this.renderDetailPanel(this.currentDetailItem)}</div>`;
                this.initChatPanel(this.selectedEmbryoId, tp);
            }
        }

        // Restore horizontal scroll position. Done after innerHTML is set so
        // scrollWidth reflects the (possibly grown) new content.
        if (prevScroll && scrollContainer) {
            if (prevScroll.wasAtRightEdge) {
                scrollContainer.scrollLeft =
                    scrollContainer.scrollWidth - scrollContainer.clientWidth;
            } else {
                scrollContainer.scrollLeft = prevScroll.left;
            }
        }
    },

    // ==========================================
    // Vitals (Strip Chart) View
    // ==========================================

    renderVitalsView() {
        const container = document.getElementById('view-vitals');
        if (!container) return;

        const embryos = Object.values(this.state.embryos);
        if (embryos.length === 0) {
            container.innerHTML = `<div class="board-empty"><div class="reasoning-empty-icon">&#x1F4C8;</div><div class="reasoning-empty-text">No embryos to display</div></div>`;
            return;
        }

        embryos.sort((a, b) => a.embryoId.localeCompare(b.embryoId));

        let html = '<div class="vitals-container">';
        for (const embryo of embryos) {
            html += this._renderVitalsStrip(embryo);
        }
        html += '</div>';
        html += '<div class="vitals-detail" id="vitals-detail"></div>';
        container.innerHTML = html;

        // Click handlers on SVG data points
        container.querySelectorAll('.vitals-point').forEach(pt => {
            pt.addEventListener('click', (e) => {
                e.stopPropagation();
                const eid = pt.dataset.embryoId;
                const tp = parseInt(pt.dataset.timepoint);
                this.selectedEmbryoId = eid;
                const reasoning = this.detectionReasoning[eid] || [];
                const item = reasoning.find(r => r.timepoint === tp);
                if (item) {
                    this.currentDetailItem = item;
                    container.querySelectorAll('.vitals-point').forEach(p => p.classList.remove('active'));
                    pt.classList.add('active');
                    const detail = document.getElementById('vitals-detail');
                    if (detail) {
                        detail.innerHTML = `<div class="vitals-detail-content">${this.renderDetailPanel(item)}</div>`;
                        this.initChatPanel(eid, tp);
                    }
                }
            });
        });

        // Restore any open detail panel after a strip rebuild (e.g. when
        // a new timepoint arrives mid-conversation).
        if (this.currentDetailItem && this.selectedEmbryoId) {
            const tp = this.currentDetailItem.timepoint;
            const pt = container.querySelector(
                `.vitals-point[data-embryo-id="${this.selectedEmbryoId}"][data-timepoint="${tp}"]`
            );
            if (pt) pt.classList.add('active');
            const detail = document.getElementById('vitals-detail');
            if (detail) {
                detail.innerHTML = `<div class="vitals-detail-content">${this.renderDetailPanel(this.currentDetailItem)}</div>`;
                this.initChatPanel(this.selectedEmbryoId, tp);
            }
        }
    },

    _renderVitalsStrip(embryo) {
        const reasoning = this.detectionReasoning[embryo.embryoId] || [];
        const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));

        const shortName = embryo.embryoId.replace(/embryo_?/i, 'E');
        const latest = sorted.length > 0 ? sorted[sorted.length - 1] : null;
        const currentStage = latest?.stage || embryo.current_stage || '—';
        const conf = latest ? this.normalizeConfidence(latest.confidence) : 'unknown';
        const overtime = latest?.temporal_analysis?.overtime_ratio;
        const rate = overtime ? (1 / overtime).toFixed(1) + 'x' : '—';
        const arrested = latest?.temporal_analysis?.is_potentially_arrested;

        // Status badge
        let statusBadge = '<span class="vitals-status vitals-ok">ON TRACK</span>';
        if (arrested) {
            statusBadge = '<span class="vitals-status vitals-critical">ARRESTED</span>';
        } else if (overtime && overtime > 1.5) {
            statusBadge = `<span class="vitals-status vitals-warn">SLOW ${overtime.toFixed(1)}x</span>`;
        }

        // ETA
        let eta = '—';
        if (currentStage && this.STAGE_TIMING[currentStage] != null) {
            const remaining = (this.STAGE_TIMING['hatched'] || 570) - this.STAGE_TIMING[currentStage];
            if (remaining > 0) eta = `~${(remaining / 60).toFixed(1)}h`;
            else eta = 'done';
        }

        // SVG chart
        const svgWidth = 800;
        const svgHeight = 140;
        const padLeft = 60;
        const padRight = 20;
        const padTop = 10;
        const padBottom = 25;
        const chartW = svgWidth - padLeft - padRight;
        const chartH = svgHeight - padTop - padBottom;

        // Time range: 0 to max of (experiment duration, 10 hours)
        const experimentStart = this.state.startedAt ? this.state.startedAt.getTime() : Date.now();
        const maxMinutes = Math.max(
            sorted.length > 0 ? ((Date.now() - experimentStart) / 60000) : 60,
            120
        );

        // Stage labels for Y axis
        const stages = ['early', 'bean', 'comma', '1_5_fold', '2_fold', 'pretzel', 'hatching', 'hatched'];
        const stageLabels = ['Early', 'Bean', 'Comma', '1.5F', '2-Fold', 'Pretzel', 'Hatch', 'Done'];
        const stageY = (stageName) => {
            const ord = this.STAGE_ORDINAL[stageName] ?? 0;
            return padTop + chartH - (ord / 9) * chartH;
        };

        // Y-axis labels
        let yLabels = '';
        stages.forEach((s, i) => {
            const y = stageY(s);
            yLabels += `<text x="${padLeft - 5}" y="${y + 3}" text-anchor="end" fill="var(--text-muted)" font-size="9">${stageLabels[i]}</text>`;
            yLabels += `<line x1="${padLeft}" y1="${y}" x2="${svgWidth - padRight}" y2="${y}" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="2,4"/>`;
        });

        // X-axis labels (time)
        let xLabels = '';
        const timeStep = maxMinutes > 300 ? 60 : 30;
        for (let m = 0; m <= maxMinutes; m += timeStep) {
            const x = padLeft + (m / maxMinutes) * chartW;
            const label = m < 60 ? `${m}m` : `${(m / 60).toFixed(0)}h`;
            xLabels += `<text x="${x}" y="${svgHeight - 3}" text-anchor="middle" fill="var(--text-muted)" font-size="9">${label}</text>`;
            xLabels += `<line x1="${x}" y1="${padTop}" x2="${x}" y2="${svgHeight - padBottom}" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="2,4"/>`;
        }

        // Actual trace
        let actualPath = '';
        let pointsHtml = '';
        sorted.forEach((item, i) => {
            const timestamp = item.timestamp ? new Date(item.timestamp).getTime() : experimentStart + (item.timepoint || 0) * (this.state.baseInterval || 120) * 1000;
            const minutes = (timestamp - experimentStart) / 60000;
            const x = padLeft + (minutes / maxMinutes) * chartW;
            const y = stageY(item.stage);
            actualPath += i === 0 ? `M${x},${y}` : `L${x},${y}`;
            const color = this.STAGE_COLORS[item.stage] || '#8b949e';
            pointsHtml += `<circle class="vitals-point" cx="${x}" cy="${y}" r="4" fill="${color}" stroke="var(--bg-card)" stroke-width="1.5" data-embryo-id="${embryo.embryoId}" data-timepoint="${item.timepoint}" style="cursor:pointer"/>`;
        });

        // Expected trace
        let expectedPath = '';
        if (this.dashboardConfig.vitals.showExpectedLine) {
            const expectedStages = ['early', 'bean', 'comma', '1_5_fold', '2_fold', 'pretzel', 'hatching', 'hatched'];
            expectedStages.forEach((s, i) => {
                const m = this.STAGE_TIMING[s] ?? 0;
                if (m > maxMinutes) return;
                const x = padLeft + (m / maxMinutes) * chartW;
                const y = stageY(s);
                expectedPath += i === 0 ? `M${x},${y}` : `L${x},${y}`;
            });
        }

        const svg = `<svg class="vitals-chart" width="100%" viewBox="0 0 ${svgWidth} ${svgHeight}" preserveAspectRatio="xMidYMid meet">
            ${yLabels}
            ${xLabels}
            ${expectedPath ? `<path d="${expectedPath}" fill="none" stroke="var(--text-muted)" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.4"/>` : ''}
            ${actualPath ? `<path d="${actualPath}" fill="none" stroke="${this.STAGE_COLORS[currentStage] || '#60a5fa'}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>` : ''}
            ${pointsHtml}
        </svg>`;

        return `
            <div class="vitals-strip" data-embryo-id="${embryo.embryoId}">
                <div class="vitals-info">
                    <span class="vitals-name">${shortName}</span>
                    <span class="vitals-stage" style="color:${this.STAGE_COLORS[currentStage] || 'var(--text)'}">${this.formatStageName(currentStage)}</span>
                    <span class="vitals-conf">${conf}</span>
                    <span class="vitals-rate">${rate}</span>
                    <span class="vitals-eta">${eta}</span>
                    ${statusBadge}
                </div>
                <div class="vitals-chart-container">${svg}</div>
            </div>
        `;
    },

    // Header panel collapse state
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

    // Update the session ID link in the header
    updateSessionIdLink() {
        const link = document.getElementById('session-id-link');
        if (!link) return;

        if (this.currentSessionId) {
            link.textContent = this.currentSessionId;
            link.href = `/review?session=${this.currentSessionId}`;
        } else {
            link.textContent = '';
            link.href = '/review';
        }
    },

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
        this.updateSessionIdLink();

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
                completionReason: embryoData.completion_reason || null,
                firstAcquired: embryoData.first_acquired ? new Date(embryoData.first_acquired) : null,
                lastAcquired: embryoData.last_acquired ? new Date(embryoData.last_acquired) : null,
                detections: embryoData.detections || {},
                current_stage: embryoData.current_stage || null,  // Restore stage from server
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
        this.hasReconciledWithServer = true;  // Mark as reconciled - safe to show reasoning data
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
        // Also clear localStorage to prevent stale data on refresh
        this.clearSavedState();
        // Reset reconciliation flag (will be set true when we get fresh server state)
        this.hasReconciledWithServer = false;
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
        // data.timepoint is already the count (timepoints_acquired), not 0-indexed
        const newTimepoints = (data.timepoint !== undefined) ? data.timepoint : embryo.timepoints + 1;

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

        // Seed a pending detection entry with the projection image so every
        // view in the embryos tab (default, board, filmstrip, vitals) can
        // render the new frame immediately, without waiting for perception
        // to finish. handleDetectorEvaluated will upgrade this entry in
        // place once the detector result arrives (could be 20-40s later).
        //
        // detector_name is a sentinel '_pending' rather than 'perception'
        // because role-routed detection (Phase 2) means test embryos run
        // 'dopaminergic_signal' rather than 'perception'. The matcher in
        // handleDetectorEvaluated upgrades the sentinel to whatever
        // detector_name the result actually carries.
        if (data.projection_uid || data.volume_uid) {
            if (!this.detectionReasoning[embryoId]) {
                this.detectionReasoning[embryoId] = [];
            }
            const existing = this.detectionReasoning[embryoId].find(
                r => r.timepoint === data.timepoint
                  && (r.detector_name === '_pending' || r._pending === true)
            );
            if (!existing) {
                this.detectionReasoning[embryoId].push({
                    detector_name: '_pending',
                    timepoint: data.timepoint,
                    volume_uid: data.volume_uid,
                    projection_uid: data.projection_uid,
                    timestamp: new Date().toISOString(),
                    // Pending result — fields filled in by DETECTOR_EVALUATED
                    _pending: true,
                    stage: null,
                    confidence: null,
                    reasoning: null,
                    is_hatching: null,
                    is_transitional: null,
                    transition_between: null,
                    observed_features: null,
                    contrastive_reasoning: null,
                    reasoning_trace: null,
                    temporal_analysis: null,
                });
            }
        }

        this.updateEmbryosCount();
        this.updateEmbryoCard(embryoId);
        this.updateSummary();

        // Re-render whichever view is active so the new frame shows up now
        if (this.currentView !== 'default') {
            this._renderActiveView();
        }
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

        // Update current_stage. For role-routed Claude detectors (test
        // embryos) the orchestrator synthesizes a pseudo-stage like
        // "lit_up" / "lit_up_saturating" / "hatched" so the existing
        // stage-driven UI works without changes.
        if (stage && (detectorName === 'perception' || detectorName === 'dopaminergic_signal')) {
            embryo.current_stage = stage;
        }

        // Store detection reasoning for the panel.
        //
        // If handleVolumeAcquired already seeded a _pending entry for this
        // timepoint + detector (the common case now that we render cells
        // optimistically on VOLUME_ACQUIRED), mutate that entry in place
        // so views preserve ordering and don't have to reconcile a new
        // array element against the old one.
        if (!this.detectionReasoning[embryoId]) {
            this.detectionReasoning[embryoId] = [];
        }
        const perceptionFields = {
            detected: detected,
            confidence: data.confidence,
            reasoning: data.reasoning,
            // Perceiver prose from the two-stage dopaminergic detector.
            // Null for legacy single-call detectors / perception.
            description: data.description,
            volume_uid: data.volume_uid ?? null,
            projection_uid: data.projection_uid ?? null,
            timestamp: new Date().toISOString(),
            stage: stage,
            is_hatching: data.is_hatching,
            is_transitional: data.is_transitional,
            transition_between: data.transition_between,
            observed_features: data.observed_features,
            contrastive_reasoning: data.contrastive_reasoning,
            reasoning_trace: data.reasoning_trace,
            temporal_analysis: data.temporal_analysis,
            // Phase 2 Claude-detector findings: the test-embryo path
            // doesn't produce stage classifications; it produces
            // intensity_level / structure_quality / has_hatched. Keep
            // these alongside the perception fields so the reasoning
            // panel can render whichever ones are present.
            intensity_level: data.intensity_level,
            structure_quality: data.structure_quality,
            has_hatched: data.has_hatched,
            findings: data.findings,
            _pending: false,
        };
        // Match priority:
        //   1. Existing non-pending entry at (timepoint, detectorName)
        //      → upgrade in place (the detector fired again, e.g. burst).
        //   2. Pending placeholder at the same timepoint (any detector
        //      name) → claim it, overwriting detector_name. Handles the
        //      Phase 2 role-routed case where the seed says '_pending'
        //      but the result arrives as 'dopaminergic_signal'.
        //   3. Otherwise push a new entry.
        let existing = this.detectionReasoning[embryoId].find(
            r => r.timepoint === data.timepoint && r.detector_name === detectorName
        );
        if (!existing) {
            existing = this.detectionReasoning[embryoId].find(
                r => r.timepoint === data.timepoint
                  && (r._pending === true || r.detector_name === '_pending')
            );
            if (existing) {
                existing.detector_name = detectorName;
            }
        }
        if (existing) {
            // Only overwrite UIDs if the detector evaluation actually
            // shipped them - the optimistic seed from VOLUME_ACQUIRED
            // already knows the right projection_uid otherwise.
            if (perceptionFields.volume_uid == null) delete perceptionFields.volume_uid;
            if (perceptionFields.projection_uid == null) delete perceptionFields.projection_uid;
            Object.assign(existing, perceptionFields);
        } else {
            this.detectionReasoning[embryoId].push({
                detector_name: detectorName,
                timepoint: data.timepoint,
                ...perceptionFields,
            });
            // Cap per-embryo reasoning to prevent unbounded memory growth
            const arr = this.detectionReasoning[embryoId];
            if (arr.length > this.MAX_REASONING_PER_EMBRYO) {
                arr.splice(0, arr.length - this.MAX_REASONING_PER_EMBRYO);
            }
        }

        if (this.currentView === 'default') {
            this.updateEmbryoCard(embryoId);
            // Update reasoning panel if this embryo is selected
            if (this.selectedEmbryoId === embryoId) {
                this.renderReasoningPanel();
                // Only auto-open if no detail panel is currently visible
                // (avoid hijacking scroll when user is reading something)
                if (!this.detailPanelVisible) {
                    this.openLatestDetail();
                }
            }
        } else {
            this._renderActiveView();
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
        this._renderActiveView();
        this.updateAmbientPulse();
    },

    renderStatusBadge() {
        const statusEl = document.getElementById('timelapse-status');
        const textEl = document.getElementById('timelapse-status-text');

        if (!statusEl || !textEl) return;

        // Remove all status classes
        statusEl.classList.remove('running', 'paused', 'completed', 'idle');

        if (this.state.status === 'IDLE' || Object.keys(this.state.embryos).length === 0) {
            statusEl.classList.add('idle');
            textEl.textContent = 'No active timelapse';
        } else {
            statusEl.classList.add(this.state.status.toLowerCase());
            textEl.textContent = this.state.status === 'RUNNING' ? 'Running' :
                                 this.state.status === 'PAUSED' ? 'Paused' :
                                 this.state.status === 'COMPLETED' ? 'Completed' :
                                 this.state.status === 'STOPPED' ? 'Stopped' : this.state.status;
        }
    },

    renderSummary() {
        // Invalidate countdown cache since we're rebuilding DOM
        this._countdownCache = null;

        const statsEl = document.getElementById('header-stats');
        if (!statsEl) return;

        const embryos = Object.values(this.state.embryos);
        if (embryos.length === 0) {
            statsEl.innerHTML = '';
            return;
        }

        const active = embryos.filter(e => !e.isComplete).length;
        const completed = embryos.filter(e => e.isComplete).length;

        statsEl.innerHTML = `
            <div class="header-stat">
                <span class="stat-value">${this.state.totalTimepoints}</span>
                <span class="stat-label">TP</span>
            </div>
            <div class="header-stat">
                <span class="stat-value">${active}</span>
                <span class="stat-label">Active</span>
            </div>
            <div class="header-stat">
                <span class="stat-value">${completed}</span>
                <span class="stat-label">Done</span>
            </div>
            ${this.state.startedAt ? `
            <div class="header-stat">
                <span class="stat-value" id="summary-duration">${this.formatDuration(Date.now() - this.state.startedAt.getTime())}</span>
                <span class="stat-label">Dur</span>
            </div>
            ` : ''}
            ${active > 0 ? `
            <div class="header-stat">
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
        // Quick targeted update for duration and countdown only.
        // renderSummary() is intentionally NOT called here — it rebuilds the
        // entire header-stats innerHTML and would overwrite these same elements.
        // renderSummary() is called on full re-renders via render().
        const durationEl = document.getElementById('summary-duration');
        if (durationEl && this.state.startedAt) {
            durationEl.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }
        const countdownEl = document.getElementById('summary-next-countdown');
        if (countdownEl) {
            countdownEl.textContent = this.getNextCountdown();
        }
    },

    renderEmbryoCards() {
        // Invalidate countdown cache since we're rebuilding embryo card DOM
        this._countdownCache = null;

        const container = document.getElementById('embryo-cards');
        if (!container) return;

        const embryos = Object.values(this.state.embryos);

        if (embryos.length === 0) {
            container.innerHTML = '';
            return;
        }

        // Sort: running first, then by embryo ID
        embryos.sort((a, b) => {
            if (a.isComplete !== b.isComplete) return a.isComplete ? 1 : -1;
            return a.embryoId.localeCompare(b.embryoId);
        });

        container.innerHTML = embryos.map(embryo => this.renderEmbryoCard(embryo)).join('');

        // Add click handlers for selection
        container.querySelectorAll('.embryo-rail-item').forEach(card => {
            card.addEventListener('click', () => {
                this.selectEmbryo(card.dataset.embryoId);
            });
        });
    },

    // Compact rail item for embryo switcher
    renderEmbryoCard(embryo) {
        const status = embryo.isComplete ? 'complete' :
                       embryo.lastError ? 'error' :
                       this.state.status === 'PAUSED' ? 'paused' : 'running';

        const isSelected = this.selectedEmbryoId === embryo.embryoId;

        // Stage info
        const stageIcon = embryo.current_stage ? this.getStageIcon(embryo.current_stage) : '🔬';
        const stageName = embryo.current_stage ? this.formatStageName(embryo.current_stage) : 'Acquiring';

        // Short label: extract number from embryo_3 → "E3"
        const shortLabel = embryo.embryoId.replace(/embryo_?/i, 'E');

        return `
            <div class="embryo-rail-item ${status} ${isSelected ? 'selected' : ''}"
                 data-embryo-id="${embryo.embryoId}"
                 title="${embryo.embryoId} — ${stageName} — ${embryo.timepoints} TP">
                <span class="rail-icon">${stageIcon}</span>
                <span class="rail-label">${shortLabel}</span>
            </div>
        `;
    },

    // Legacy full embryo card (keeping for reference, not used)
    renderEmbryoCardFull(embryo) {
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
        const detectorNames = Object.keys(embryo.detections).filter(n => n !== 'perception' && n !== 'unknown');
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
            'bean': '🌱',  // seedling - more compatible than bean emoji
            'comma': '🌙',
            '1.5fold': '🔄',
            '2fold': '🔁',
            '3fold': '🔃',
            'pretzel': '🥨',
            'hatching': '🐣',
            'hatched': '🐛',
            'arrested': '⏸️',
            'no_object': '⬜',
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
            'arrested': 'Arrested',
            'no_object': 'Empty',
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
        const isNewEmbryo = this.selectedEmbryoId !== embryoId;
        if (isNewEmbryo) {
            this.expandedImages = {};
            this.expandedRangeItems = {};
            // Clear detail panel state to prevent stale embryoId in timepoint links
            this.currentDetailItem = null;
            this.detailPanelVisible = false;
        }

        this.selectedEmbryoId = embryoId;

        // Update card selection styles
        document.querySelectorAll('.embryo-rail-item').forEach(card => {
            card.classList.toggle('selected', card.dataset.embryoId === embryoId);
        });

        // Render reasoning panel
        this.renderReasoningPanel();

        // Auto-open latest evaluation's detail for this embryo
        if (isNewEmbryo) {
            this.openLatestDetail();
        }
    },

    // Open detail panel for the latest evaluation of the selected embryo
    openLatestDetail() {
        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        if (reasoning.length > 0) {
            // Sort by timepoint descending to find latest
            const sorted = [...reasoning].sort((a, b) => (b.timepoint ?? 0) - (a.timepoint ?? 0));
            const latest = sorted[0];
            this.openDetailPanel(latest.detector_name, latest.timepoint);
        }
    },

    renderReasoningPanel() {
        const panel = document.getElementById('reasoning-panel');
        if (!panel) return;

        // Show loading state until we've reconciled with server (prevents stale cached data)
        if (!this.hasReconciledWithServer) {
            panel.innerHTML = `
                <div class="reasoning-empty">
                    <div class="reasoning-empty-icon">&#x23F3;</div>
                    <div class="reasoning-empty-text">Connecting to server...</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
                        Syncing with experiment data
                    </div>
                </div>
            `;
            return;
        }

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
        // Handle both numeric confidence (perception: 0.95) and string confidence (legacy: "high")
        const highConfidence = reasoning.filter(r => {
            if (typeof r.confidence === 'number') return r.confidence >= 0.8;
            if (typeof r.confidence === 'string') return r.confidence.toLowerCase() === 'high';
            return false;
        });

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

        // Merged compact info bar: embryo name + stage + stats + quick-jumps
        const stageBadge = isPerceptionData && currentStage
            ? `<span class="stage-icon">${this.getStageIcon(currentStage)}</span> ${this.formatStageName(currentStage)}`
            : '';
        const transitionsText = isPerceptionData
            ? `${stageTransitions.length} transitions`
            : `${positiveDetections.length} detections`;

        panel.innerHTML = `
            <div class="reasoning-header">
                <div class="reasoning-embryo-info">
                    <span class="reasoning-status-dot ${statusClass}">${statusIcon}</span>
                    <span class="reasoning-embryo-name">${embryo.embryoId}</span>
                    ${stageBadge ? `<span class="reasoning-condition">${stageBadge}</span>` : ''}
                    <span class="stat" style="margin-left: 0.5rem;">${transitionsText}</span>
                    <span class="stat">${totalEvaluations} evals</span>
                    <span class="stat">${embryo.timepoints} tp</span>
                </div>
                <div class="detection-quick-jumps" style="display:flex;gap:0.35rem;flex-wrap:nowrap;overflow-x:auto;">
                    ${quickJumpsHtml}
                </div>
            </div>
            ${this.renderTimelineSparkline(reasoning, embryo.timepoints)}
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
            filtered = sorted.filter(r => this.normalizeConfidence(r.confidence) === 'high');
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
            // Only use stored UIDs - don't guess with fallback patterns as they can match wrong datasets
            const imageUid = item.projection_uid || item.volume_uid || null;
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
        // Only use stored UIDs - don't guess with fallback patterns as they can match wrong datasets
        const imageUid = item.projection_uid || item.volume_uid || null;
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
                             onclick="event.stopPropagation(); EmbryosManager.openDetailPanel('${item.detector_name}', ${item.timepoint}, true)" />
                    ` : '<div class="expansion-image-placeholder">No image</div>'}
                    <div class="expansion-text">
                        <div class="expansion-reasoning">${this.escapeHtml(truncatedReasoning)}</div>
                        <button class="expansion-link" onclick="event.stopPropagation(); EmbryosManager.openDetailPanel('${item.detector_name}', ${item.timepoint}, true)">
                            View full analysis &#x2192;
                        </button>
                    </div>
                </div>
            </div>
        `;
    },

    // Render confidence level as dots (5-dot scale)
    renderConfidenceDots(confidence) {
        const level = this.normalizeConfidence(confidence);
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
        if (this.normalizeConfidence(item.confidence) === 'high') {
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
        this.openDetailPanel(detectorName, timepoint, true);
    },

    // Show detail for a compact row (legacy - redirects to openDetailPanel)
    showDetectionDetail(detectorName, timepoint) {
        this.openDetailPanel(detectorName, timepoint, true);
    },

    // Open the detail panel inline in the reasoning panel
    // userInitiated: true when user clicked, false when auto-opened by new data
    openDetailPanel(detectorName, timepoint, userInitiated = false) {
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

        // Highlight the selected eval dot
        this.highlightEvalDot(timepoint, detectorName);

        this.renderInlineDetail(item, userInitiated);
    },

    // Highlight the currently selected eval dot in the timeline
    highlightEvalDot(timepoint, detectorName) {
        // Remove highlight from all dots
        document.querySelectorAll('.eval-dot.active').forEach(dot => {
            dot.classList.remove('active');
        });

        // Add highlight to the selected dot
        const selectedDot = document.querySelector(
            `.eval-dot[data-timepoint="${timepoint}"][data-detector="${detectorName}"]`
        );
        if (selectedDot) {
            selectedDot.classList.add('active');
        }
    },

    // Render detail content inline in the reasoning panel
    // scrollToIt: if true, scroll the detail into view (user-initiated actions only)
    renderInlineDetail(item, scrollToIt = false) {
        const container = document.getElementById('inline-detail-container');
        if (!container) return;

        container.innerHTML = this.renderDetailPanel(item);
        container.classList.add('visible');

        // Auto-scroll active dot into view in the horizontal strip
        const activeDot = document.querySelector('.eval-dot.active');
        if (activeDot) {
            activeDot.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }

        this.initChatPanel(this.selectedEmbryoId, item.timepoint);
    },

    // Render the detail panel content
    renderDetailPanel(item) {
        // Only use stored UIDs - don't guess with fallback patterns as they can match wrong datasets
        const imageUid = item.projection_uid || item.volume_uid || null;
        const reasoning = item.reasoning || 'No reasoning provided';

        // Linkify timepoints in reasoning
        const linkedReasoning = this.linkifyTimepoints(reasoning, this.selectedEmbryoId, {
            detectionPoint: item.detected ? item.timepoint : null,
            reasoningText: reasoning
        });

        // Two-stage detectors (dopaminergic_signal) ship a perceiver
        // description alongside the classifier reasoning. Render it as a
        // separate block above the classifier; relabel the classifier
        // block from "VLM Summary" to "Classifier" so the two-stage
        // architecture is visually explicit.
        const hasDescription = !!item.description;
        // Icon + label are wrapped in spans so CSS can flex-align them and
        // size the icon independently — emoji glyphs don't share text
        // baselines and look misaligned when interpolated as raw text.
        const perceiverDetailHtml = hasDescription
            ? `<div class="detail-perceiver">
                   <div class="reasoning-label perceiver-label"><span class="stage-icon" aria-hidden="true">&#x1F441;</span><span class="stage-text">Perceiver</span></div>
                   <div class="vlm-perceiver-text-detail">${this.escapeHtml(item.description)}</div>
               </div>`
            : '';
        const classifierLabel = hasDescription
            ? '<span class="stage-icon" aria-hidden="true">&#x2699;</span><span class="stage-text">Classifier</span>'
            : 'VLM Summary';
        // Pretty-printed JSON of the classifier output (only for the
        // two-stage path — single-call detectors don't carry these
        // structured fields). Newer dopaminergic_signal payloads nest the
        // fields under `findings`; older ones put them at the top level —
        // fall back to `findings` so the JSON view never silently drops
        // them when only one shape is present.
        const f = item.findings || {};
        const classifierJsonDetailHtml = hasDescription
            ? `<pre class="classifier-json"><code>${this.escapeHtml(JSON.stringify({
                   intensity_level: item.intensity_level ?? f.intensity_level ?? null,
                   structure_quality: item.structure_quality ?? f.structure_quality ?? null,
                   has_hatched: item.has_hatched ?? f.has_hatched ?? null,
                   reasoning: item.reasoning,
               }, null, 2))}</code></pre>`
            : '';

        // Build metadata for lightbox
        const lightboxMeta = {
            embryo_id: this.selectedEmbryoId,
            timepoint: item.timepoint,
            data_type: 'Volume Projection',
            shape: item.shape || ''
        };
        const metaJson = JSON.stringify(lightboxMeta).replace(/'/g, "\\'").replace(/"/g, '&quot;');

        // Build image HTML - will be loaded async if UID not available
        // Use openTimepointInLightbox for navigation through all timepoints
        const imageHtml = imageUid
            ? `<img src="/api/images/${imageUid}/png"
                    alt="T${item.timepoint}"
                    onclick="EmbryosManager.openTimepointInLightbox('${this.selectedEmbryoId}', ${item.timepoint})" />`
            : `<div class="detail-image-loading" id="detail-image-placeholder"
                    data-embryo="${this.selectedEmbryoId}"
                    data-timepoint="${item.timepoint}">Loading image...</div>`;

        // Fetch image async if no UID
        if (!imageUid) {
            this.fetchDetailImage(this.selectedEmbryoId, item.timepoint);
        }

        // Build Claude-detector findings block (test embryos / Phase 2).
        // Surfaces intensity_level + structure_quality + has_hatched so
        // the user can see what the per-volume Claude vision call
        // returned, separate from the perception-style "stage" semantics.
        let detectorFindingsHtml = '';
        const claudeIntensity = item.intensity_level || (item.findings && item.findings.intensity_level);
        const claudeStructure = item.structure_quality || (item.findings && item.findings.structure_quality);
        const claudeHatched = (item.has_hatched != null)
            ? item.has_hatched
            : (item.findings && item.findings.has_hatched);
        if (claudeIntensity || claudeStructure || claudeHatched !== undefined) {
            const intensityColors = {
                NONE: '#888', WEAK: '#7bb3d4', MEDIUM: '#ffba6b',
                STRONG: '#ff8c42', SATURATING: '#ff5252',
            };
            const structureColors = {
                NONE: '#888', PARTIAL: '#ffba6b', GOOD: '#4caf50',
            };
            const iColor = intensityColors[claudeIntensity] || '#aaa';
            const sColor = structureColors[claudeStructure] || '#aaa';
            const intensityChip = claudeIntensity
                ? `<span class="finding-chip" style="background:${iColor}26;color:${iColor};border:1px solid ${iColor}">intensity: ${claudeIntensity}</span>`
                : '';
            const structureChip = claudeStructure
                ? `<span class="finding-chip" style="background:${sColor}26;color:${sColor};border:1px solid ${sColor}">structure: ${claudeStructure}</span>`
                : '';
            const hatchedChip = claudeHatched
                ? `<span class="finding-chip" style="background:#ff525226;color:#ff5252;border:1px solid #ff5252">hatched</span>`
                : '';
            detectorFindingsHtml = `
                <div class="detail-claude-findings">
                    <div class="reasoning-label">Claude detector — ${this.escapeHtml(item.detector_name || 'unknown')}</div>
                    <div class="findings-row">${intensityChip}${structureChip}${hatchedChip}</div>
                </div>
            `;
        }

        // Build observed features section if available
        let observedFeaturesHtml = '';
        if (item.observed_features) {
            const f = item.observed_features;
            observedFeaturesHtml = `
                <div class="detail-observed-features">
                    <div class="reasoning-label">Observed Features</div>
                    <div class="features-grid">
                        ${f.shape ? `<span class="feature-item"><span class="feature-label">Shape:</span> ${this.escapeHtml(f.shape)}</span>` : ''}
                        ${f.curvature ? `<span class="feature-item"><span class="feature-label">Curvature:</span> ${this.escapeHtml(f.curvature)}</span>` : ''}
                        ${f.shell_status ? `<span class="feature-item"><span class="feature-label">Shell:</span> ${this.escapeHtml(f.shell_status)}</span>` : ''}
                        ${f.body_segments ? `<span class="feature-item"><span class="feature-label">Segments:</span> ${this.escapeHtml(f.body_segments)}</span>` : ''}
                        ${f.emergence ? `<span class="feature-item"><span class="feature-label">Emergence:</span> ${this.escapeHtml(f.emergence)}</span>` : ''}
                    </div>
                </div>
            `;
        }

        // Build contrastive reasoning section if available
        let contrastiveHtml = '';
        if (item.contrastive_reasoning) {
            const c = item.contrastive_reasoning;
            contrastiveHtml = `
                <div class="detail-contrastive">
                    <div class="reasoning-label">Contrastive Reasoning</div>
                    <div class="contrastive-grid">
                        ${c.why_not_previous ? `<div class="contrastive-item"><span class="contrastive-label">Not previous stage:</span> ${this.escapeHtml(c.why_not_previous)}</div>` : ''}
                        ${c.why_not_next ? `<div class="contrastive-item"><span class="contrastive-label">Not next stage:</span> ${this.escapeHtml(c.why_not_next)}</div>` : ''}
                    </div>
                </div>
            `;
        }

        // Build transitional indicator if applicable
        let transitionalHtml = '';
        if (item.is_transitional && item.transition_between) {
            const stages = item.transition_between.map(s => this.formatStageName(s)).join(' → ');
            transitionalHtml = `
                <div class="detail-transitional">
                    <span class="transitional-badge">TRANSITIONAL</span>
                    <span class="transitional-stages">${stages}</span>
                </div>
            `;
        }

        // Build reasoning trace section if available (interleaved reasoning with tool calls)
        let reasoningTraceHtml = '';
        if (item.reasoning_trace && item.reasoning_trace.steps && item.reasoning_trace.steps.length > 0) {
            const steps = item.reasoning_trace.steps;
            const toolCalls = item.reasoning_trace.total_tool_calls || 0;

            let stepsHtml = steps.map(step => {
                // Add arrest-warning class if content mentions arrest
                const hasArrestWarning = step.content && step.content.includes('ARREST WARNING');
                const stepClass = `trace-step trace-${step.step_type}${hasArrestWarning ? ' arrest-warning' : ''}`;
                let icon = '';
                let label = '';

                switch (step.step_type) {
                    case 'temporal_context':
                        icon = '⏱';
                        label = 'Temporal';
                        break;
                    case 'initial_analysis':
                        icon = '🔍';
                        label = 'Analysis';
                        break;
                    case 'tool_call':
                        icon = '🔧';
                        if (step.tool_name === 'view_previous_timepoint') {
                            label = `View T-${step.tool_input?.offset || '?'}`;
                        } else if (step.tool_name === 'view_embryo') {
                            const rx = step.tool_input?.rotation_x || 0;
                            const ry = step.tool_input?.rotation_y || 0;
                            label = `View 3D (${rx}°, ${ry}°)`;
                        } else if (step.tool_name === 'request_verification') {
                            const numComparisons = step.tool_input?.comparisons?.length || 0;
                            label = `Verify (${numComparisons} comparisons)`;
                        } else if (step.tool_name === 'get_reference') {
                            label = `Get ${step.tool_input?.stage || '?'} ref`;
                        } else {
                            label = step.tool_name || 'Tool call';
                        }
                        break;
                    case 'tool_result':
                        icon = '📷';
                        label = step.tool_result_summary || 'Result';
                        break;
                    case 'final_decision':
                        icon = '✓';
                        label = 'Decision';
                        break;
                    case 'verification_requested':
                        icon = '🔀';
                        label = 'Verification';
                        break;
                    case 'verification_subagent':
                        icon = '🤖';
                        label = step.tool_result_summary || 'Subagent';
                        break;
                    case 'verification_result':
                        icon = '📊';
                        label = 'Aggregation';
                        break;
                    default:
                        icon = '•';
                        label = step.step_type;
                }

                // Show preview with expand option for long content
                const fullContent = step.content || '';
                const isLong = fullContent.length > 200;
                const previewContent = isLong ? fullContent.substring(0, 200) + '...' : fullContent;
                const contentId = `trace-content-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

                // Use markdown rendering for analysis and decision steps
                const useMarkdown = step.step_type === 'final_decision' || step.step_type === 'initial_analysis';
                const renderFn = useMarkdown ? this.renderMarkdown.bind(this) : this.escapeHtml.bind(this);

                return `
                    <div class="${stepClass}">
                        <span class="trace-icon">${icon}</span>
                        <span class="trace-label">${label}</span>
                        ${fullContent ? `
                            <div class="trace-content ${isLong ? 'expandable' : ''} ${useMarkdown ? 'markdown' : ''}" id="${contentId}">
                                <div class="trace-content-preview">${useMarkdown ? renderFn(previewContent) : this.escapeHtml(previewContent)}</div>
                                ${isLong ? `
                                    <div class="trace-content-full" style="display: none;">${renderFn(fullContent)}</div>
                                    <button class="trace-expand-btn" onclick="EmbryosManager.toggleTraceContent('${contentId}')">Show more</button>
                                ` : ''}
                            </div>
                        ` : ''}
                    </div>
                `;
            }).join('');

            reasoningTraceHtml = `
                <div class="detail-reasoning-trace">
                    <div class="reasoning-label">
                        Reasoning Trace
                        ${toolCalls > 0 ? `<span class="trace-tool-count">${toolCalls} tool call${toolCalls > 1 ? 's' : ''}</span>` : ''}
                    </div>
                    <div class="trace-steps">
                        ${stepsHtml}
                    </div>
                </div>
            `;
        }

        // Format confidence display. Hide entirely when the detector
        // doesn't emit a probabilistic confidence (e.g. dopaminergic_signal
        // returns structured intensity/structure findings instead) — the
        // string "Unknown confidence" was actively confusing.
        const hasNumericConf = typeof item.confidence === 'number';
        const hasTextConf = typeof item.confidence === 'string' && item.confidence.trim() !== '';
        const confHtml = hasNumericConf
            ? `<span class="verdict-confidence">${Math.round(item.confidence * 100)}% confidence</span>`
            : hasTextConf
                ? `<span class="verdict-confidence">${item.confidence}</span>`
                : '';

        return `
            <div class="detail-panel-header">
                <span class="detail-title">T${item.timepoint} - ${this.formatDetectorName(item.detector_name)}</span>
                <button class="view-projections-btn header-projections-btn"
                        onclick="ProjectionViewer.open('${this.selectedEmbryoId}', ${item.timepoint})"
                        data-tooltip="View all projection types from the 3D volume">
                    Projections
                </button>
                <button class="detail-close" onclick="EmbryosManager.closeDetailPanel()">&times;</button>
            </div>
            <div class="detail-split">
                <div class="detail-split-left">
                    <div class="detail-image-container">
                        ${imageHtml}
                    </div>
                    <div class="detail-verdict ${item.detected ? 'detected' : ''}">
                        <span class="verdict-stage">${item.stage ? this.formatStageName(item.stage) : (item.detected ? 'DETECTED' : 'Not detected')}</span>
                        ${confHtml}
                        ${transitionalHtml}
                    </div>
                    ${detectorFindingsHtml}
                    ${observedFeaturesHtml}
                </div>
                <div class="detail-split-right">
                    ${contrastiveHtml}
                    ${reasoningTraceHtml}
                    ${perceiverDetailHtml}
                    <div class="detail-reasoning">
                        <div class="reasoning-label ${hasDescription ? 'classifier-label' : ''}">${classifierLabel}</div>
                        <div class="reasoning-text">${linkedReasoning}</div>
                        ${classifierJsonDetailHtml}
                    </div>
                    <div class="chat-panel"
                         data-embryo-id="${this.selectedEmbryoId}"
                         data-timepoint="${item.timepoint}">
                        <div class="chat-panel-label">Follow-up</div>
                        <div class="chat-thread" id="chat-thread"></div>
                        <form class="chat-input-row" onsubmit="return EmbryosManager.sendChat(event)">
                            <textarea class="chat-input" id="chat-input"
                                      placeholder="Ask a follow-up about this timepoint…"
                                      rows="2"
                                      onkeydown="EmbryosManager.handleChatKeydown(event)"></textarea>
                            <button type="submit" class="chat-send">Send</button>
                        </form>
                    </div>
                </div>
            </div>
            <div class="detail-actions">
                <button class="detail-nav" onclick="EmbryosManager.navigateDetail(-1)">&#x2190; Previous</button>
                <button class="detail-nav" onclick="EmbryosManager.navigateDetail(1)">Next &#x2192;</button>
            </div>
        `;
    },

    // ==========================================
    // Chat — per-timepoint VLM follow-up
    // ==========================================

    async initChatPanel(embryoId, timepoint) {
        const sessionId = this.currentSessionId;
        if (!sessionId) return;

        const thread = document.getElementById('chat-thread');
        if (!thread) return;
        thread.innerHTML = '';

        try {
            const resp = await fetch(
                `/api/perception/chat/${sessionId}/${embryoId}/${timepoint}`
            );
            if (!resp.ok) return;
            const data = await resp.json();
            for (const turn of (data.turns || [])) {
                this.appendChatMessage(turn.role, turn.content);
            }
        } catch (err) {
            console.warn('Failed to load chat history', err);
        }
    },

    appendChatMessage(role, content) {
        const thread = document.getElementById('chat-thread');
        if (!thread) return null;

        const el = document.createElement('div');
        el.className = `chat-message chat-message-${role}`;

        const label = document.createElement('div');
        label.className = 'chat-message-label';
        label.textContent = role === 'user' ? 'You' : 'Gently';
        el.appendChild(label);

        const contentEl = document.createElement('div');
        contentEl.className = 'chat-message-content';
        contentEl.textContent = content;
        el.appendChild(contentEl);

        thread.appendChild(el);
        thread.scrollTop = thread.scrollHeight;
        return el;
    },

    handleChatKeydown(event) {
        // Enter submits; Shift+Enter inserts a newline.
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            event.target.form.requestSubmit();
        }
    },

    async sendChat(event) {
        event.preventDefault();
        const panel = document.querySelector('.chat-panel');
        const input = document.getElementById('chat-input');
        if (!panel || !input) return false;

        const text = input.value.trim();
        if (!text) return false;

        const sessionId = this.currentSessionId;
        const embryoId = panel.dataset.embryoId;
        const timepoint = parseInt(panel.dataset.timepoint, 10);
        if (!sessionId || !embryoId || isNaN(timepoint)) {
            console.error('Chat: missing session/embryo/timepoint');
            return false;
        }

        this.appendChatMessage('user', text);
        input.value = '';
        input.disabled = true;
        const sendBtn = panel.querySelector('.chat-send');
        if (sendBtn) sendBtn.disabled = true;

        const assistantEl = this.appendChatMessage('assistant', '');
        const contentEl = assistantEl
            ? assistantEl.querySelector('.chat-message-content')
            : null;
        if (contentEl) contentEl.classList.add('streaming');

        try {
            const resp = await fetch(
                `/api/perception/chat/${sessionId}/${embryoId}/${timepoint}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text }),
                }
            );

            if (!resp.ok) {
                const errTxt = await resp.text();
                if (contentEl) {
                    contentEl.textContent = `Error: ${errTxt}`;
                    contentEl.classList.add('error');
                }
                return false;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let accumulated = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const events = buffer.split('\n\n');
                buffer = events.pop();

                for (const ev of events) {
                    const line = ev.trim();
                    if (!line.startsWith('data:')) continue;
                    const payloadStr = line.slice(5).trim();
                    try {
                        const payload = JSON.parse(payloadStr);
                        if (payload.type === 'delta') {
                            accumulated += payload.text;
                            if (contentEl) contentEl.textContent = accumulated;
                            const thread = document.getElementById('chat-thread');
                            if (thread) thread.scrollTop = thread.scrollHeight;
                        } else if (payload.type === 'error') {
                            if (contentEl) {
                                contentEl.textContent = `Error: ${payload.message}`;
                                contentEl.classList.add('error');
                            }
                        }
                    } catch (e) {
                        console.warn('Bad SSE payload', payloadStr);
                    }
                }
            }
        } catch (err) {
            console.error('Chat request failed', err);
            if (contentEl) {
                contentEl.textContent = `Error: ${err.message}`;
                contentEl.classList.add('error');
            }
        } finally {
            if (contentEl) contentEl.classList.remove('streaming');
            input.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            input.focus();
        }

        return false;
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
                    // Use openTimepointInLightbox for navigation through all timepoints
                    placeholder.outerHTML = `<img src="/api/images/${uid}/png"
                                                  alt="T${timepoint}"
                                                  onclick="EmbryosManager.openTimepointInLightbox('${embryoId}', ${timepoint})" />`;
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
        // Filmstrip side panel — clearing innerHTML lets the :empty CSS
        // rule collapse the panel and let the rows reclaim full width.
        const filmstripDetail = document.getElementById('filmstrip-detail');
        if (filmstripDetail) {
            filmstripDetail.innerHTML = '';
        }
        this.detailPanelVisible = false;
        this.currentDetailItem = null;

        // Clear eval dot + filmstrip cell highlight
        document.querySelectorAll('.eval-dot.active').forEach(dot => {
            dot.classList.remove('active');
        });
        document.querySelectorAll('.filmstrip-cell.active').forEach(cell => {
            cell.classList.remove('active');
        });
    },

    // Navigate to previous/next item in detail panel
    navigateDetail(direction) {
        if (!this.currentDetailItem) return;

        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        // Scope to the same detector and sort by timepoint so navigation
        // is monotonic — independent of insertion order in the array.
        // If multiple detectors fired per timepoint, we stay within the
        // currently-viewed detector instead of skipping over its siblings.
        const detectorName = this.currentDetailItem.detector_name;
        const sameDetector = reasoning
            .filter(r => r.detector_name === detectorName)
            .sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));
        const currentIdx = sameDetector.findIndex(
            r => r.timepoint === this.currentDetailItem.timepoint
        );
        if (currentIdx === -1) return;
        const newIdx = currentIdx + direction;
        if (newIdx >= 0 && newIdx < sameDetector.length) {
            const newItem = sameDetector[newIdx];
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
        const confidenceClass = this.normalizeConfidence(detection.confidence);

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

        // VLM analysis section. Two-stage detectors (dopaminergic_signal)
        // emit both a perceiver description (free prose, what the model
        // saw) and a classifier reasoning (one-line rubric pick). Show
        // them as two labelled blocks so the audience can see observation
        // vs classification separately. Single-call detectors just emit
        // reasoning and get the classifier block alone.
        let reasoningHtml = '';
        if (detection.description || detection.reasoning) {
            const embryoId = this.selectedEmbryoId || '';
            const linkContext = {
                detectionPoint: detection.detected ? detection.timepoint : null,
                reasoningText: detection.reasoning || ''
            };
            const linkedReasoning = detection.reasoning
                ? this.linkifyTimepoints(detection.reasoning, embryoId, linkContext)
                : '';

            const isTwoStage = !!detection.description;
            const perceiverBlock = detection.description
                ? `<div class="vlm-perceiver-text">
                       <span class="vlm-stage-label perceiver"><span class="stage-icon" aria-hidden="true">&#x1F441;</span><span class="stage-text">Perceiver</span></span>
                       <div class="vlm-stage-body">${this.escapeHtml(detection.description)}</div>
                   </div>`
                : '';
            // Pretty-printed JSON of the classifier's structured output —
            // shown alongside the prose so the audience can see the
            // exact decision the classifier emitted to the orchestrator.
            // Fall back to `findings` for fields the newer detector nests
            // there. Without this, missing top-level keys silently drop
            // from the JSON view.
            const df = detection.findings || {};
            const classifierJsonBlock = isTwoStage
                ? `<pre class="classifier-json"><code>${this.escapeHtml(JSON.stringify({
                       intensity_level: detection.intensity_level ?? df.intensity_level ?? null,
                       structure_quality: detection.structure_quality ?? df.structure_quality ?? null,
                       has_hatched: detection.has_hatched ?? df.has_hatched ?? null,
                       reasoning: detection.reasoning,
                   }, null, 2))}</code></pre>`
                : '';
            const classifierBlock = detection.reasoning
                ? `<div class="detection-reasoning-text ${isTwoStage ? 'two-stage' : ''}">
                       ${isTwoStage ? '<span class="vlm-stage-label classifier"><span class="stage-icon" aria-hidden="true">&#x2699;</span><span class="stage-text">Classifier</span></span>' : ''}
                       <div class="vlm-stage-body">${linkedReasoning}</div>
                       ${classifierJsonBlock}
                   </div>`
                : '';
            const inner = perceiverBlock + classifierBlock;

            if (detection.detected) {
                reasoningHtml = `<div class="vlm-analysis ${isTwoStage ? 'two-stage' : ''}">${inner}</div>`;
            } else {
                const label = isTwoStage ? 'analysis' : 'VLM reasoning';
                reasoningHtml = `
                    <button class="reasoning-toggle ${reasoningExpanded ? 'expanded' : ''}"
                            onclick="event.stopPropagation(); EmbryosManager.toggleReasoning('${index}')">
                        <span class="chevron">&#x25B6;</span>
                        ${reasoningExpanded ? 'Hide' : 'Show'} ${label}
                    </button>
                    <div class="reasoning-content ${reasoningExpanded ? 'expanded' : ''}" id="reasoning-${index}">
                        <div class="vlm-analysis ${isTwoStage ? 'two-stage' : ''}">${inner}</div>
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
                        <div class="detection-image-actions">
                            <button class="toggle-image-btn" onclick="event.stopPropagation(); EmbryosManager.toggleImage('${index}', '${imageUid}')">
                                <span class="toggle-icon">${imageExpanded ? '&#x25BC;' : '&#x25B6;'}</span>
                                ${imageExpanded ? 'Hide' : 'Show'} Volume Projection
                            </button>
                            <button class="view-projections-btn" onclick="event.stopPropagation(); ProjectionViewer.open('${this.selectedEmbryoId}', ${detection.timepoint})"
                                    data-tooltip="View all projection types from the 3D volume">
                                View All Projections
                            </button>
                        </div>
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
                message: 'Start a timelapse acquisition to begin tracking embryos. Configure your experiment in the Calibration tab.',
                action: { label: 'Go to Calibration', tab: 'calibration' }
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
                action: { label: 'Go to Calibration', tab: 'calibration' }
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
            // Keep highest confidence - handle both numeric (0.95) and string ("high") values
            const confOrder = { 'high': 3, 'medium': 2, 'low': 1 };
            const existingConf = confOrder[timepoints[tp].confidence] || 0;
            let newConfLevel;
            if (typeof r.confidence === 'number') {
                newConfLevel = r.confidence >= 0.8 ? 'high' : r.confidence >= 0.5 ? 'medium' : 'low';
            } else if (typeof r.confidence === 'string') {
                newConfLevel = r.confidence.toLowerCase();
            } else {
                newConfLevel = 'low';
            }
            const newConf = confOrder[newConfLevel] || 0;
            if (newConf > existingConf) {
                timepoints[tp].confidence = newConfLevel;
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
                             data-timepoint="${tp}"
                             data-detector="${detectorName}"
                             onclick="EmbryosManager.openDetailPanel('${detectorName}', ${tp}, true)">
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
                         data-timepoint="${tp}"
                         data-detector="${detectorName}"
                         onclick="EmbryosManager.openDetailPanel('${detectorName}', ${tp}, true)">
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
            const conf = this.normalizeConfidence(e.confidence);
            const confDisplay = typeof e.confidence === 'number' ? `${Math.round(e.confidence * 100)}%` : (e.confidence || 'Unknown');
            return `<span class="context-dot ${conf}" title="T${e.timepoint}: ${confDisplay}"></span>`;
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

    /**
     * Open the 3D volume viewer for a detection
     * Tries to find a matching 3D segmented volume for the given embryo/timepoint
     */
    async view3D(embryoId, timepoint) {
        try {
            // First, check if there are any 3D volumes available
            const response = await fetch('/api/volumes3d');
            const data = await response.json();

            if (!data.volumes || data.volumes.length === 0) {
                this.showToast('No 3D volumes available. Run segmentation to generate 3D data.', 'info');
                return;
            }

            // Try to find a volume matching this embryo
            let volumeUid = null;
            const matchingVolumes = data.volumes.filter(v =>
                v.metadata?.embryo_id === embryoId
            );

            if (matchingVolumes.length > 0) {
                // Use the most recent matching volume
                volumeUid = matchingVolumes[matchingVolumes.length - 1].uid;
            } else if (data.volumes.length > 0) {
                // Fallback to most recent volume
                volumeUid = data.volumes[data.volumes.length - 1].uid;
                this.showToast(`Showing most recent 3D volume (no match for ${embryoId})`, 'info');
            }

            if (volumeUid && typeof ProjectionViewer !== 'undefined') {
                ProjectionViewer.open(embryoId, timepoint);
            } else if (typeof ProjectionViewer === 'undefined') {
                console.error('ProjectionViewer not loaded');
                this.showToast('3D Viewer not available - refresh the page', 'error');
            }
        } catch (err) {
            console.error('Failed to load 3D volumes:', err);
            this.showToast('Failed to load 3D volumes', 'error');
        }
    },

    /**
     * Show a toast notification
     */
    showToast(message, type = 'info') {
        // Simple toast implementation
        let toast = document.getElementById('embryo-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'embryo-toast';
            toast.className = 'embryo-toast';
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.className = `embryo-toast ${type} visible`;
        setTimeout(() => {
            toast.classList.remove('visible');
        }, 3000);
    },

    // Toggle trace content expand/collapse
    toggleTraceContent(contentId) {
        const container = document.getElementById(contentId);
        if (!container) return;

        const preview = container.querySelector('.trace-content-preview');
        const full = container.querySelector('.trace-content-full');
        const btn = container.querySelector('.trace-expand-btn');

        if (preview && full && btn) {
            const isExpanded = full.style.display !== 'none';
            preview.style.display = isExpanded ? 'block' : 'none';
            full.style.display = isExpanded ? 'none' : 'block';
            btn.textContent = isExpanded ? 'Show more' : 'Show less';
        }
    },

    escapeHtml(text) { return escapeHtml(text); },

    // Simple markdown renderer for reasoning trace
    renderMarkdown(text) {
        if (!text) return '';

        // Helper to extract useful info from JSON content
        const extractJsonSummary = (jsonContent) => {
            const stageMatch = jsonContent.match(/"stage"\s*:\s*"([^"]+)"/);
            if (stageMatch) {
                return `**Classification: ${stageMatch[1]}**`;
            }
            const confMatch = jsonContent.match(/"confidence"\s*:\s*([\d.]+)/);
            if (confMatch) {
                return `[Confidence: ${Math.round(parseFloat(confMatch[1]) * 100)}%]`;
            }
            return null;
        };

        // Handle JSON code blocks - collapse them but keep a summary
        // Complete blocks: ```json ... ```
        let processedText = text.replace(/```json\s*([\s\S]*?)```/g, (match, jsonContent) => {
            const summary = extractJsonSummary(jsonContent);
            return summary ? `\n${summary}\n` : '';
        });

        // Incomplete JSON blocks (no closing ```) - extract what we can
        processedText = processedText.replace(/```json\s*([\s\S]*)$/g, (match, jsonContent) => {
            const summary = extractJsonSummary(jsonContent);
            return summary ? `\n${summary}\n` : '';
        });

        // Handle raw JSON objects embedded in text (VLM outputs JSON without code blocks)
        // Find JSON objects that contain stage classification using balanced brace matching
        const findJsonObjects = (text) => {
            const results = [];
            let i = 0;
            while (i < text.length) {
                if (text[i] === '{') {
                    let braceCount = 1;
                    let j = i + 1;
                    while (j < text.length && braceCount > 0) {
                        if (text[j] === '{') braceCount++;
                        if (text[j] === '}') braceCount--;
                        j++;
                    }
                    if (braceCount === 0) {
                        const jsonStr = text.slice(i, j);
                        // Only process if it contains a stage field
                        if (jsonStr.includes('"stage"')) {
                            results.push({ start: i, end: j, content: jsonStr });
                        }
                    }
                    i = j;
                } else {
                    i++;
                }
            }
            return results;
        };

        // Replace JSON objects from end to start (to preserve indices)
        const jsonObjects = findJsonObjects(processedText);
        for (let i = jsonObjects.length - 1; i >= 0; i--) {
            const obj = jsonObjects[i];
            const summary = extractJsonSummary(obj.content);
            if (summary) {
                processedText = processedText.slice(0, obj.start) + `\n\n${summary}\n` + processedText.slice(obj.end);
            }
        }

        // Escape HTML
        let html = this.escapeHtml(processedText);

        // Headers: ## Header -> <div class="md-h2">
        html = html.replace(/^## (.+)$/gm, '<div class="md-h2">$1</div>');
        html = html.replace(/^### (.+)$/gm, '<div class="md-h3">$1</div>');

        // Bold: **text** -> <strong>
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Bullet points: - item or * item -> list items
        // Group consecutive bullets into a list
        html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul class="md-list">$&</ul>');

        // Style bracketed notes
        html = html.replace(/\[([^\]]+)\]/g, '<span class="md-json-note">[$1]</span>');

        // Line breaks for readability
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        return `<div class="md-content"><p>${html}</p></div>`;
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

            // Build stage data map (timepoint -> stage)
            const stageData = {};
            reasoning.forEach(r => {
                if (r.timepoint !== undefined && r.stage) {
                    stageData[r.timepoint] = r.stage;
                }
            });

            await TimepointPlayer.openSequence(embryoId, 0, null, {
                vlmRange: null,  // No specific VLM range for "play all"
                detectionPoint: latestEvent?.timepoint ?? null,
                reasoningText: latestEvent?.reasoning ?? null,
                stage: latestEvent?.stage ?? null,
                isHatching: latestEvent?.is_hatching ?? false,
                bufferPercent: 0,  // No buffer for "play all"
                stageData: stageData  // Per-timepoint stage info for timeline coloring
            });

            // Auto-play when opening from gallery
            TimepointPlayer.play();
        } else {
            console.warn('TimepointPlayer not available');
        }
    },

    updateEmbryoCard(embryoId) {
        // Invalidate countdown cache since card DOM is being replaced
        this._countdownCache = null;

        const embryo = this.state.embryos[embryoId];
        if (!embryo) return;

        const card = document.querySelector(`[data-embryo-id="${embryoId}"]`);
        if (card) {
            // Re-render just this card, preserving selection state
            const wasSelected = card.classList.contains('selected');
            const newCard = document.createElement('div');
            newCard.innerHTML = this.renderEmbryoCard(embryo);
            const renderedCard = newCard.firstElementChild;

            if (!renderedCard) {
                console.warn('Failed to render embryo card for', embryoId);
                return;
            }

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

    _ensureCountdownCache() {
        if (this._countdownCache) return this._countdownCache;
        this._countdownCache = {
            timelapseDuration: document.getElementById('timelapse-duration'),
            summaryDuration: document.getElementById('summary-duration'),
            miniCountdowns: {},  // embryoId -> element
        };
        // Cache per-embryo mini-countdown elements
        document.querySelectorAll('.mini-countdown[data-embryo]').forEach(el => {
            this._countdownCache.miniCountdowns[el.dataset.embryo] = el;
        });
        return this._countdownCache;
    },

    updateCountdowns() {
        if (this.state.status !== 'RUNNING') return;

        const cache = this._ensureCountdownCache();

        // Update main duration
        if (cache.timelapseDuration && this.state.startedAt) {
            cache.timelapseDuration.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }

        // Update summary duration
        if (cache.summaryDuration && this.state.startedAt) {
            cache.summaryDuration.textContent = this.formatDuration(Date.now() - this.state.startedAt.getTime());
        }

        // Update the header NEXT countdown. This ticks every second
        // instead of only on VOLUME_ACQUIRED events, so it no longer
        // freezes during the long wait between rounds.
        const nextEl = document.getElementById('summary-next-countdown');
        if (nextEl) {
            nextEl.textContent = this.getNextCountdown();
        }

        // Update per-embryo countdowns (compact cards use mini-countdown class)
        Object.values(this.state.embryos).forEach(embryo => {
            if (embryo.isComplete) return;

            const countdownEl = cache.miniCountdowns[embryo.embryoId];
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
    },

    // Open lightbox for a timepoint with navigation through all timepoints for this embryo
    openTimepointInLightbox(embryoId, timepoint) {
        const reasoning = this.detectionReasoning[embryoId] || [];
        if (reasoning.length === 0) return;

        // Sort by timepoint to ensure correct order
        const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));

        // Build image list for Lightbox
        const imageList = sorted.map(item => {
            const uid = item.projection_uid || item.volume_uid ||
                        `volume_${embryoId}_t${String(item.timepoint).padStart(4, '0')}`;
            return {
                uid: uid,
                base64_png: null,  // Will be loaded via src
                data_type: item.stage ? `Stage: ${this.formatStageName(item.stage)}` : 'Volume Projection',
                metadata: {
                    embryo_id: embryoId,
                    timepoint: item.timepoint,
                    shape: item.shape || ''
                },
                // Include for info panel display
                shape: item.shape ? (Array.isArray(item.shape) ? item.shape : []) : [],
                timestamp: item.timestamp
            };
        });

        // Find the index of the clicked timepoint
        const startIndex = sorted.findIndex(item => item.timepoint === timepoint);
        const index = startIndex >= 0 ? startIndex : 0;

        // Open lightbox with full list and navigation
        if (imageList.length > 0 && typeof Lightbox !== 'undefined') {
            Lightbox.openWithSequence(imageList, index, 'reasoning');
        }
    },

    // Pick the right detail-renderer for the current view. The default
    // view uses openDetailPanel (full panel). Filmstrip and vitals
    // re-render their inline ``#filmstrip-detail`` / ``#vitals-detail``
    // containers. Used by arrow-key navigation so prev/next works
    // regardless of which view is active.
    _renderDetailForCurrentView(item) {
        if (!item) return;
        if (this.currentView === 'filmstrip') {
            const detail = document.getElementById('filmstrip-detail');
            if (detail) {
                // Update the highlighted cell
                document.querySelectorAll('.filmstrip-cell.active').forEach(c => c.classList.remove('active'));
                const cell = document.querySelector(
                    `.filmstrip-cell[data-embryo-id="${this.selectedEmbryoId}"][data-timepoint="${item.timepoint}"]`
                );
                if (cell) {
                    cell.classList.add('active');
                    cell.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                }
                this.currentDetailItem = item;
                detail.innerHTML = `<div class="filmstrip-detail-content">${this.renderDetailPanel(item)}</div>`;
                this.initChatPanel(this.selectedEmbryoId, item.timepoint);
            }
            return;
        }
        if (this.currentView === 'vitals') {
            const detail = document.getElementById('vitals-detail');
            if (detail) {
                this.currentDetailItem = item;
                detail.innerHTML = `<div class="vitals-detail-content">${this.renderDetailPanel(item)}</div>`;
                this.initChatPanel(this.selectedEmbryoId, item.timepoint);
            }
            return;
        }
        // Default view (and board): open the full detail panel.
        this.openDetailPanel(item.detector_name, item.timepoint, true);
    },

    // Navigate to previous timepoint in detail panel
    navigateToPrevTimepoint() {
        if (!this.currentDetailItem || !this.selectedEmbryoId) return;

        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        if (reasoning.length === 0) return;

        const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));
        const currentIdx = sorted.findIndex(r =>
            r.timepoint === this.currentDetailItem.timepoint &&
            r.detector_name === this.currentDetailItem.detector_name
        );

        if (currentIdx > 0) {
            this._renderDetailForCurrentView(sorted[currentIdx - 1]);
        }
    },

    // Navigate to next timepoint in detail panel
    navigateToNextTimepoint() {
        if (!this.currentDetailItem || !this.selectedEmbryoId) return;

        const reasoning = this.detectionReasoning[this.selectedEmbryoId] || [];
        if (reasoning.length === 0) return;

        const sorted = [...reasoning].sort((a, b) => (a.timepoint ?? 0) - (b.timepoint ?? 0));
        const currentIdx = sorted.findIndex(r =>
            r.timepoint === this.currentDetailItem.timepoint &&
            r.detector_name === this.currentDetailItem.detector_name
        );

        if (currentIdx >= 0 && currentIdx < sorted.length - 1) {
            this._renderDetailForCurrentView(sorted[currentIdx + 1]);
        }
    },

    // Setup keyboard navigation
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Arrow keys navigate timepoints whenever a detail item is
            // active in ANY view — default / board / filmstrip / vitals.
            // (Previously gated on detailPanelVisible which is only set
            // by openDetailPanel — so filmstrip arrow nav silently
            // didn't fire.)
            if (!this.currentDetailItem || !this.selectedEmbryoId) return;
            if (typeof state !== 'undefined' && state.tab !== TABS.EMBRYOS) return;
            if (e.target.matches('input, textarea, select, [contenteditable]')) return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                this.navigateToPrevTimepoint();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                this.navigateToNextTimepoint();
            }
        });
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    EmbryosManager.init();
    EmbryosManager.setupKeyboardNavigation();
});
