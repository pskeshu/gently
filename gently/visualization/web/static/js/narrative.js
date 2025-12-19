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
        const state = EmbryosManager.state;
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

        // Check if using perception system (has stage data) or legacy detectors
        let isPerception = false;
        const stageInfo = {};  // embryo_id -> current stage
        const hatchingEmbryos = [];
        const stageOrder = ['early', 'bean', 'comma', '1.5fold', '2fold', '3fold', 'pretzel', 'hatching', 'hatched'];

        // Count stages or detections
        let totalDetections = 0;
        let detectionDetails = [];

        Object.entries(EmbryosManager.detectionReasoning).forEach(([embryoId, reasoning]) => {
            // Check for perception data
            const stages = reasoning.map(r => r.stage).filter(Boolean);
            if (stages.length > 0) {
                isPerception = true;
                stageInfo[embryoId] = stages[stages.length - 1];  // Latest stage
                if (reasoning.some(r => r.is_hatching)) {
                    hatchingEmbryos.push(embryoId);
                }
            } else {
                // Legacy detection counting
                const positives = reasoning.filter(r => r.detected);
                totalDetections += positives.length;
                positives.forEach(d => {
                    detectionDetails.push(`${embryoId}: ${EmbryosManager.formatDetectorName(d.detector_name)} at T${d.timepoint}`);
                });
            }
        });

        const details = [];
        if (activeCount > 0) details.push(`${activeCount} embryo${activeCount !== 1 ? 's' : ''} actively imaging`);
        if (completedCount > 0) details.push(`${completedCount} embryo${completedCount !== 1 ? 's' : ''} completed`);
        details.push(`${state.totalTimepoints} total timepoints acquired`);

        let status, headline;

        if (isPerception && Object.keys(stageInfo).length > 0) {
            // Show stage distribution for perception
            const stageCounts = {};
            Object.values(stageInfo).forEach(stage => {
                stageCounts[stage] = (stageCounts[stage] || 0) + 1;
            });
            const sortedStages = Object.entries(stageCounts)
                .sort((a, b) => {
                    const idxA = stageOrder.indexOf(a[0].toLowerCase());
                    const idxB = stageOrder.indexOf(b[0].toLowerCase());
                    return (idxA >= 0 ? idxA : 99) - (idxB >= 0 ? idxB : 99);
                });
            const stageSummary = sortedStages.map(([stage, count]) => `${count} ${stage}`).join(', ');
            details.push(`Stages: ${stageSummary}`);

            if (hatchingEmbryos.length > 0) {
                details.push(`Hatching detected: ${hatchingEmbryos.join(', ')}`);
                status = 'notable';
                headline = `Hatching in ${hatchingEmbryos.length} Embryo${hatchingEmbryos.length !== 1 ? 's' : ''}`;
            } else {
                // Find most advanced stage
                const maxStageIdx = Math.max(...Object.values(stageInfo).map(s => {
                    const idx = stageOrder.indexOf(s.toLowerCase());
                    return idx >= 0 ? idx : 0;
                }));
                const maxStage = stageOrder[maxStageIdx].replace('fold', '-fold');
                status = 'normal';
                headline = `Most Advanced: ${maxStage.charAt(0).toUpperCase() + maxStage.slice(1)}`;
            }
        } else if (detectionDetails.length > 0) {
            details.push(`${totalDetections} positive detection${totalDetections !== 1 ? 's' : ''}: ${detectionDetails.slice(0, 3).join(', ')}${detectionDetails.length > 3 ? '...' : ''}`);
            status = 'notable';
            headline = `${totalDetections} Detection${totalDetections !== 1 ? 's' : ''} Found`;
        } else if (completedCount > 0) {
            status = 'normal';
            headline = `${completedCount}/${embryoCount} Embryos Complete`;
        } else {
            status = 'normal';
            headline = 'Experiment In Progress';
        }

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
    NarrativeManager.init();
    // Generate initial narrative summary
    setTimeout(() => NarrativeManager.renderLocalSummary(), 500);
});
