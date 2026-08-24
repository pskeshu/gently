/**
 * TimepointPlayer - Video playback for embryo timelapse sequences
 * Extends Lightbox with animated playback, timeline, and VLM context
 */
const TimepointPlayer = {
    // Playback state
    sequence: [],           // Array of {uid, timepoint, ...}
    currentIndex: 0,
    isPlaying: false,
    fps: 3,                 // Default speed (biology-appropriate)
    looping: true,          // Loop by default
    animationId: null,
    lastFrameTime: 0,

    // Context for VLM highlighting
    embryoId: null,
    vlmRange: null,         // {start: 73, end: 81}
    detectionPoint: null,   // Timepoint where detection occurred
    reasoningText: null,    // Original VLM text
    stage: null,            // Current developmental stage (perception)
    isHatching: false,      // Whether hatching was detected

    // Preloading
    preloadRadius: 5,       // Frames to preload ahead/behind
    imageCache: new Map(),  // uid -> Image object
    loadingSet: new Set(),  // UIDs currently loading

    // DOM elements (created on open)
    els: {},

    async openSequence(embryoId, start, end, options = {}) {
        /**
         * Open video player for a timepoint range
         *
         * @param {string} embryoId - Embryo to play
         * @param {number} start - Start timepoint
         * @param {number} end - End timepoint (optional)
         * @param {object} options - {vlmRange, detectionPoint, reasoningText, bufferPercent}
         */
        this.embryoId = embryoId;
        this.vlmRange = options.vlmRange || {start, end};
        this.detectionPoint = options.detectionPoint || null;
        this.reasoningText = options.reasoningText || null;
        this.stage = options.stage || null;
        this.isHatching = options.isHatching || false;
        this.stageData = options.stageData || {};  // Per-timepoint stage info

        // Fetch sequence from server - try multiple data types
        const bufferPercent = options.bufferPercent || 0.15;
        const dataTypes = ['volume_projection', 'volume', 'image'];

        let sequence = null;

        for (const dataType of dataTypes) {
            const params = new URLSearchParams({
                start: start,
                data_type: dataType,
                buffer_percent: bufferPercent
            });
            if (end !== null && end !== undefined) {
                params.set('end', end);
            }

            try {
                const resp = await fetch(`/api/sequence/${embryoId}?${params}`);
                const data = await resp.json();

                if (data.sequence && data.sequence.length > 0) {
                    sequence = data.sequence;
                    console.log(`TimepointPlayer: loaded ${sequence.length} frames using ${dataType}`);
                    break; // Found images, stop trying
                }
            } catch (err) {
                console.warn(`TimepointPlayer: failed to fetch ${dataType}:`, err);
            }
        }

        if (!sequence || sequence.length === 0) {
            console.warn('TimepointPlayer: no images found for any data type');
            return;
        }

        this.sequence = sequence;
        this.currentIndex = 0;
        this.imageCache.clear();
        this.loadingSet.clear();

        // Open in video mode
        this.openVideoMode();

        // Start preloading
        this.preloadAround(0);

        // Show first frame
        await this.showFrame(0);
    },

    openVideoMode() {
        /**
         * Transform lightbox into video mode with controls and context panel
         */
        const overlay = document.getElementById('lightbox-overlay');
        if (!overlay) return;

        overlay.classList.add('active', 'video-mode');
        document.body.style.overflow = 'hidden';

        // Inject video-specific UI if not already present
        if (!document.getElementById('video-controls')) {
            this.injectVideoUI(overlay);
        }

        // Show video UI
        document.getElementById('video-controls')?.classList.remove('hidden');
        document.getElementById('video-timeline')?.classList.remove('hidden');

        // Hide VLM context panel by default - users should focus on video first
        document.getElementById('video-context-panel')?.classList.add('hidden');

        // Hide standard lightbox nav and position counter
        document.querySelectorAll('.lightbox-nav').forEach(el => el.style.display = 'none');
        document.getElementById('lightbox-thumbnails')?.classList.add('hidden');
        const posEl = document.getElementById('lightbox-position');
        if (posEl) posEl.style.display = 'none';

        // Update context panel (for when it's shown)
        this.updateContextPanel();

        // Bind video-specific keys
        this.bindVideoKeys();

        Lightbox.isOpen = true;
    },

    injectVideoUI(overlay) {
        /**
         * Inject video player UI elements into the lightbox
         */
        const container = overlay.querySelector('.lightbox-container');
        if (!container) return;

        // Timeline
        const timeline = document.createElement('div');
        timeline.id = 'video-timeline';
        timeline.className = 'video-timeline';
        timeline.innerHTML = `
            <div class="timeline-track" id="timeline-track">
                <div class="timeline-buffer" id="timeline-buffer"></div>
                <div class="timeline-vlm-range" id="timeline-vlm-range"></div>
                <div class="timeline-detection-marker" id="timeline-detection-marker"></div>
                <div class="timeline-playhead" id="timeline-playhead"></div>
            </div>
            <div class="timeline-labels" id="timeline-labels"></div>
        `;
        container.appendChild(timeline);

        // Playback controls
        const controls = document.createElement('div');
        controls.id = 'video-controls';
        controls.className = 'video-controls';
        controls.innerHTML = `
            <div class="controls-left">
                <button class="ctrl-btn" id="ctrl-first" title="First frame (Home)">
                    <span class="ctrl-icon">⏮</span>
                </button>
                <button class="ctrl-btn" id="ctrl-step-back" title="Step back (←)">
                    <span class="ctrl-icon">◀</span>
                </button>
                <button class="ctrl-btn ctrl-play" id="ctrl-play-pause" title="Play/Pause (Space)">
                    <span class="ctrl-icon" id="play-icon">▶</span>
                </button>
                <button class="ctrl-btn" id="ctrl-step-forward" title="Step forward (→)">
                    <span class="ctrl-icon">▶</span>
                </button>
                <button class="ctrl-btn" id="ctrl-last" title="Last frame (End)">
                    <span class="ctrl-icon">⏭</span>
                </button>
            </div>
            <div class="controls-center">
                <button class="ctrl-btn ctrl-small ${this.looping ? 'active' : ''}" id="ctrl-loop" title="Toggle loop (L)">
                    <span class="ctrl-icon">↺</span>
                </button>
                <div class="speed-control">
                    <button class="ctrl-btn ctrl-small" id="ctrl-slower" title="Slower (-)">−</button>
                    <span class="speed-label" id="speed-label">${this.fps} fps</span>
                    <button class="ctrl-btn ctrl-small" id="ctrl-faster" title="Faster (+)">+</button>
                </div>
            </div>
            <div class="controls-right">
                <span class="frame-counter" id="frame-counter">T0 / 0</span>
            </div>
        `;
        container.appendChild(controls);

        // Context panel (side panel for VLM reasoning)
        const contextPanel = document.createElement('div');
        contextPanel.id = 'video-context-panel';
        contextPanel.className = 'video-context-panel';
        contextPanel.innerHTML = `
            <div class="context-header">
                <span class="context-title">VLM Analysis</span>
                <button class="context-close" id="context-panel-close">×</button>
            </div>
            <div class="context-verdict" id="context-verdict"></div>
            <div class="context-reasoning" id="context-reasoning"></div>
            <div class="context-info" id="context-info"></div>
        `;
        overlay.appendChild(contextPanel);

        // Bind control events
        this.bindControlEvents();
    },

    bindControlEvents() {
        // Play/Pause
        document.getElementById('ctrl-play-pause')?.addEventListener('click', () => {
            this.isPlaying ? this.pause() : this.play();
        });

        // Step and jump
        document.getElementById('ctrl-first')?.addEventListener('click', () => this.seekTo(0));
        document.getElementById('ctrl-step-back')?.addEventListener('click', () => this.stepFrame(-1));
        document.getElementById('ctrl-step-forward')?.addEventListener('click', () => this.stepFrame(1));
        document.getElementById('ctrl-last')?.addEventListener('click', () => this.seekTo(this.sequence.length - 1));

        // Loop
        document.getElementById('ctrl-loop')?.addEventListener('click', () => this.toggleLoop());

        // Speed
        document.getElementById('ctrl-slower')?.addEventListener('click', () => this.setSpeed(this.fps - 0.5));
        document.getElementById('ctrl-faster')?.addEventListener('click', () => this.setSpeed(this.fps + 0.5));

        // Timeline click
        document.getElementById('timeline-track')?.addEventListener('click', (e) => this.handleTimelineClick(e));

        // Context panel close
        document.getElementById('context-panel-close')?.addEventListener('click', () => {
            document.getElementById('video-context-panel')?.classList.add('collapsed');
        });
    },

    bindVideoKeys() {
        // Store handler reference for cleanup
        this._videoKeyHandler = (e) => {
            if (!Lightbox.isOpen) return;

            // Don't intercept if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch (e.key) {
                case ' ':  // Space - play/pause
                    e.preventDefault();
                    this.isPlaying ? this.pause() : this.play();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.stepFrame(-1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.stepFrame(1);
                    break;
                case '-':
                    e.preventDefault();
                    this.setSpeed(this.fps - 0.5);
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    this.setSpeed(this.fps + 0.5);
                    break;
                case 'l':
                case 'L':
                    e.preventDefault();
                    this.toggleLoop();
                    break;
                case 'd':
                case 'D':
                    e.preventDefault();
                    this.jumpToDetection();
                    break;
                case '[':
                    e.preventDefault();
                    this.jumpToVlmStart();
                    break;
                case ']':
                    e.preventDefault();
                    this.jumpToVlmEnd();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.seekTo(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this.seekTo(this.sequence.length - 1);
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.close();
                    break;
            }
        };

        // Remove existing and add new
        document.removeEventListener('keydown', this._videoKeyHandler);
        document.addEventListener('keydown', this._videoKeyHandler);
    },

    // Playback controls
    play() {
        if (this.isPlaying || this.sequence.length === 0) return;

        this.isPlaying = true;
        this.lastFrameTime = performance.now();
        this.updatePlayButton();
        this.animate();
    },

    pause() {
        this.isPlaying = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.updatePlayButton();
    },

    animate() {
        if (!this.isPlaying) return;

        const now = performance.now();
        const elapsed = now - this.lastFrameTime;
        const frameInterval = 1000 / this.fps;

        if (elapsed >= frameInterval) {
            this.lastFrameTime = now - (elapsed % frameInterval);

            // Advance frame
            let nextIndex = this.currentIndex + 1;
            if (nextIndex >= this.sequence.length) {
                if (this.looping) {
                    nextIndex = 0;
                } else {
                    this.pause();
                    return;
                }
            }

            this.showFrame(nextIndex);
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    },

    stepFrame(direction) {
        this.pause();
        let newIndex = this.currentIndex + direction;

        // Wrap around if looping
        if (newIndex < 0) {
            newIndex = this.looping ? this.sequence.length - 1 : 0;
        } else if (newIndex >= this.sequence.length) {
            newIndex = this.looping ? 0 : this.sequence.length - 1;
        }

        this.showFrame(newIndex);
    },

    setSpeed(fps) {
        this.fps = Math.max(0.5, Math.min(15, fps));
        document.getElementById('speed-label').textContent = `${this.fps} fps`;
    },

    toggleLoop() {
        this.looping = !this.looping;
        document.getElementById('ctrl-loop')?.classList.toggle('active', this.looping);
    },

    seekTo(index) {
        if (index < 0 || index >= this.sequence.length) return;
        this.showFrame(index);
    },

    jumpToDetection() {
        if (this.detectionPoint === null) return;
        const idx = this.sequence.findIndex(s => s.timepoint === this.detectionPoint);
        if (idx >= 0) this.seekTo(idx);
    },

    jumpToVlmStart() {
        if (!this.vlmRange) return;
        const idx = this.sequence.findIndex(s => s.timepoint >= this.vlmRange.start);
        if (idx >= 0) this.seekTo(idx);
    },

    jumpToVlmEnd() {
        if (!this.vlmRange) return;
        // Find last frame in VLM range
        for (let i = this.sequence.length - 1; i >= 0; i--) {
            if (this.sequence[i].timepoint <= this.vlmRange.end) {
                this.seekTo(i);
                return;
            }
        }
    },

    // Frame display
    async showFrame(index) {
        if (index < 0 || index >= this.sequence.length) return;

        this.currentIndex = index;
        const frame = this.sequence[index];

        // Get or load image
        let img = this.imageCache.get(frame.uid);
        if (!img) {
            img = await this.loadImage(frame.uid);
        }

        // Display
        const lightboxImg = document.getElementById('lightbox-image');
        if (lightboxImg && img) {
            lightboxImg.src = img.src;
        }

        // Update metadata panels
        const infoType = document.getElementById('lb-info-type');
        const infoEmbryo = document.getElementById('lb-info-embryo');
        const infoShape = document.getElementById('lb-info-shape');
        const infoTime = document.getElementById('lb-info-time');

        // Show stage for current timepoint if available
        const currentStage = this.stageData[frame.timepoint];
        if (infoType) {
            if (currentStage) {
                const stageIcon = this.getStageIcon(currentStage);
                const stageName = this.formatStageName(currentStage);
                infoType.textContent = `${stageIcon} ${stageName}`;
            } else {
                infoType.textContent = 'Timelapse';
            }
        }
        if (infoEmbryo) infoEmbryo.textContent = this.embryoId || '-';
        if (infoShape) infoShape.textContent = `Frame ${index + 1} of ${this.sequence.length}`;
        if (infoTime) {
            // Show timepoint
            if (frame.timepoint !== undefined) {
                infoTime.textContent = `T${frame.timepoint}`;
            } else {
                infoTime.textContent = `Frame ${index + 1}`;
            }
        }

        // Update UI
        this.updateFrameCounter();
        this.updatePlayhead();
        this.updateContextHighlight();

        // Preload nearby frames
        this.preloadAround(index);
    },

    async loadImage(uid) {
        // Return cached if available
        if (this.imageCache.has(uid)) {
            return this.imageCache.get(uid);
        }

        // Prevent duplicate loading
        if (this.loadingSet.has(uid)) {
            return new Promise((resolve) => {
                const check = () => {
                    if (this.imageCache.has(uid)) {
                        resolve(this.imageCache.get(uid));
                    } else if (!this.loadingSet.has(uid)) {
                        resolve(null);
                    } else {
                        setTimeout(check, 50);
                    }
                };
                check();
            });
        }

        this.loadingSet.add(uid);

        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                this.imageCache.set(uid, img);
                this.loadingSet.delete(uid);
                resolve(img);
            };
            img.onerror = () => {
                this.loadingSet.delete(uid);
                resolve(null);
            };
            img.src = `/api/images/${uid}/png`;
        });
    },

    preloadAround(centerIndex) {
        // Preload frames around current position
        for (let offset = -this.preloadRadius; offset <= this.preloadRadius; offset++) {
            const idx = centerIndex + offset;
            if (idx >= 0 && idx < this.sequence.length) {
                const frame = this.sequence[idx];
                if (!this.imageCache.has(frame.uid) && !this.loadingSet.has(frame.uid)) {
                    this.loadImage(frame.uid);  // Fire and forget
                }
            }
        }
    },

    // UI updates
    updatePlayButton() {
        const icon = document.getElementById('play-icon');
        if (icon) {
            icon.textContent = this.isPlaying ? '\u23F8' : '\u25B6';
        }
    },

    updateFrameCounter() {
        const counter = document.getElementById('frame-counter');
        const frame = this.sequence[this.currentIndex];
        if (counter && frame) {
            counter.textContent = `T${frame.timepoint} / ${this.sequence.length}`;
        }
    },

    updatePlayhead() {
        const playhead = document.getElementById('timeline-playhead');
        const track = document.getElementById('timeline-track');
        if (!playhead || !track || this.sequence.length === 0) return;

        const percent = (this.currentIndex / (this.sequence.length - 1)) * 100;
        playhead.style.left = `${percent}%`;
    },

    updateContextHighlight() {
        // Highlight timepoint references in VLM text that match current frame
        const frame = this.sequence[this.currentIndex];
        if (!frame || !this.reasoningText) return;

        const reasoningEl = document.getElementById('context-reasoning');
        if (!reasoningEl) return;

        // Re-render with current timepoint highlighted
        const currentTp = frame.timepoint;
        let html = this.escapeHtml(this.reasoningText);

        // Highlight ranges that include current timepoint
        html = html.replace(
            /timepoints?\s+(\d+)(?:\s*[-\u2013]\s*(\d+))?/gi,
            (match, start, end) => {
                const s = parseInt(start);
                const e = end ? parseInt(end) : s;
                const isActive = currentTp >= s && currentTp <= e;
                return `<mark class="${isActive ? 'active' : ''}">${match}</mark>`;
            }
        );

        // Also highlight T-format references
        html = html.replace(
            /T(\d+)(?:\s*[-\u2013]\s*T?(\d+))?/gi,
            (match, start, end) => {
                const s = parseInt(start);
                const e = end ? parseInt(end) : s;
                const isActive = currentTp >= s && currentTp <= e;
                return `<mark class="${isActive ? 'active' : ''}">${match}</mark>`;
            }
        );

        reasoningEl.innerHTML = html;
    },

    updateContextPanel() {
        const verdictEl = document.getElementById('context-verdict');
        const reasoningEl = document.getElementById('context-reasoning');
        const infoEl = document.getElementById('context-info');

        if (verdictEl) {
            if (this.isHatching) {
                verdictEl.textContent = `HATCHING at T${this.detectionPoint || '?'}`;
                verdictEl.className = 'context-verdict detected hatching';
            } else if (this.stage) {
                // Perception system - show stage
                const stageIcon = this.getStageIcon(this.stage);
                const stageName = this.formatStageName(this.stage);
                verdictEl.innerHTML = `${stageIcon} Stage: ${stageName}`;
                verdictEl.className = 'context-verdict stage';
            } else if (this.detectionPoint !== null) {
                verdictEl.textContent = `DETECTED at T${this.detectionPoint}`;
                verdictEl.className = 'context-verdict detected';
            } else if (this.reasoningText) {
                // We have reasoning but no detection - negative result
                verdictEl.textContent = 'No detection';
                verdictEl.className = 'context-verdict';
            } else {
                // No data at all - don't show misleading text
                verdictEl.textContent = '';
                verdictEl.className = 'context-verdict';
            }
        }

        if (reasoningEl && this.reasoningText) {
            reasoningEl.textContent = this.reasoningText;
        }

        if (infoEl) {
            const firstTp = this.sequence[0]?.timepoint || '?';
            const lastTp = this.sequence[this.sequence.length - 1]?.timepoint || '?';
            infoEl.textContent = `Embryo: ${this.embryoId} | Range: T${firstTp}-T${lastTp}`;
        }

        // Update timeline markers
        this.renderTimeline();
    },

    // Stage color mapping
    stageColors: {
        'early': '#6b7280',      // gray
        'bean': '#8b5cf6',       // violet
        'comma': '#3b82f6',      // blue
        '1.5fold': '#06b6d4',    // cyan
        '2fold': '#10b981',      // emerald
        '3fold': '#22c55e',      // green
        'hatching': '#f59e0b',   // amber
        'hatched': '#ef4444',    // red
    },

    renderStageSegments(track, firstTp, range) {
        /**
         * Render colored segments on timeline based on VLM stage per timepoint
         * Creates contiguous segments for each developmental stage
         */
        // Remove any existing stage segments
        track.querySelectorAll('.timeline-stage-segment').forEach(el => el.remove());

        if (!this.stageData || Object.keys(this.stageData).length === 0) return;

        // Build contiguous segments from stageData
        // Group consecutive timepoints with the same stage
        const segments = [];
        let currentSegment = null;

        // Sort sequence by timepoint to ensure correct order
        const sortedSeq = [...this.sequence].sort((a, b) => a.timepoint - b.timepoint);

        for (const frame of sortedSeq) {
            const tp = frame.timepoint;
            const stage = this.stageData[tp];

            if (!stage) {
                // No stage data for this timepoint - close current segment
                if (currentSegment) {
                    segments.push(currentSegment);
                    currentSegment = null;
                }
                continue;
            }

            if (currentSegment && currentSegment.stage === stage) {
                // Extend current segment
                currentSegment.endTp = tp;
            } else {
                // Close previous segment and start new one
                if (currentSegment) {
                    segments.push(currentSegment);
                }
                currentSegment = { stage, startTp: tp, endTp: tp };
            }
        }

        // Don't forget the last segment
        if (currentSegment) {
            segments.push(currentSegment);
        }

        // Render each segment as a colored div
        for (const seg of segments) {
            const color = this.stageColors[seg.stage.toLowerCase()] || '#6b7280';
            const startPct = ((seg.startTp - firstTp) / range) * 100;
            const endPct = ((seg.endTp - firstTp) / range) * 100;
            // Add small margin for segment visibility (at least 1% width)
            const width = Math.max(1, endPct - startPct);

            const segmentEl = document.createElement('div');
            segmentEl.className = 'timeline-stage-segment';
            segmentEl.style.cssText = `
                position: absolute;
                left: ${startPct}%;
                width: ${width}%;
                height: 100%;
                background-color: ${color};
                opacity: 0.4;
                pointer-events: none;
                z-index: 0;
            `;
            segmentEl.title = `${seg.stage}: T${seg.startTp}-T${seg.endTp}`;

            track.appendChild(segmentEl);
        }
    },

    renderTimeline() {
        const track = document.getElementById('timeline-track');
        const labelsEl = document.getElementById('timeline-labels');
        const vlmRangeEl = document.getElementById('timeline-vlm-range');
        const detectionMarker = document.getElementById('timeline-detection-marker');

        if (!track || this.sequence.length === 0) return;

        const firstTp = this.sequence[0]?.timepoint ?? 0;
        const lastTp = this.sequence[this.sequence.length - 1]?.timepoint ?? 0;
        const range = lastTp - firstTp || 1;  // Avoid division by zero for single timepoint

        // Render stage segments on timeline
        this.renderStageSegments(track, firstTp, range);

        // VLM range highlight
        if (vlmRangeEl && this.vlmRange && this.vlmRange.start != null && this.vlmRange.end != null) {
            const startPct = ((this.vlmRange.start - firstTp) / range) * 100;
            const endPct = ((this.vlmRange.end - firstTp) / range) * 100;
            vlmRangeEl.style.left = `${Math.max(0, startPct)}%`;
            vlmRangeEl.style.width = `${Math.min(100, endPct) - Math.max(0, startPct)}%`;
        }

        // Detection point marker - clickable to show context panel
        if (detectionMarker && this.detectionPoint !== null) {
            const pct = ((this.detectionPoint - firstTp) / range) * 100;
            detectionMarker.style.left = `${pct}%`;
            detectionMarker.style.display = 'block';
            detectionMarker.title = `Detection at T${this.detectionPoint} - Click for details`;
            detectionMarker.style.cursor = 'pointer';
            // Remove old handler and add new one
            detectionMarker.onclick = (e) => {
                e.stopPropagation();  // Don't trigger timeline click
                this.showDetectionPanel();
            };
        } else if (detectionMarker) {
            detectionMarker.style.display = 'none';
            detectionMarker.onclick = null;
        }

        // Labels
        if (labelsEl) {
            const vlmStartLabel = (this.vlmRange && this.vlmRange.start != null)
                ? `<span class="timeline-label" style="left: ${((this.vlmRange.start - firstTp) / range) * 100}%">T${this.vlmRange.start}</span>` : '';
            const vlmEndLabel = (this.vlmRange && this.vlmRange.end != null)
                ? `<span class="timeline-label" style="left: ${((this.vlmRange.end - firstTp) / range) * 100}%">T${this.vlmRange.end}</span>` : '';
            labelsEl.innerHTML = `
                <span class="timeline-label" style="left: 0">T${firstTp}</span>
                ${vlmStartLabel}
                ${vlmEndLabel}
                <span class="timeline-label" style="left: 100%">T${lastTp}</span>
            `;
        }
    },

    handleTimelineClick(e) {
        const track = document.getElementById('timeline-track');
        if (!track || this.sequence.length === 0) return;

        const rect = track.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        const index = Math.round(pct * (this.sequence.length - 1));
        this.seekTo(Math.max(0, Math.min(this.sequence.length - 1, index)));
    },

    escapeHtml(text) { return escapeHtml(text); },

    showDetectionToast() {
        // Remove any existing toast
        document.querySelector('.detection-toast')?.remove();

        const container = document.querySelector('.lightbox-container');
        if (!container) return;

        let toastText, toastIcon;

        if (this.isHatching) {
            toastIcon = '\uD83D\uDC23';
            toastText = `Hatching detected at T${this.detectionPoint}`;
        } else if (this.stage) {
            toastIcon = this.getStageIcon(this.stage);
            toastText = `Stage: ${this.formatStageName(this.stage)}`;
        } else {
            const detectorName = this.formatDetectorName(this.detectorName || 'hatching');
            toastIcon = '\uD83D\uDD2C';
            toastText = `${detectorName} detected at T${this.detectionPoint}`;
        }

        const toast = document.createElement('div');
        toast.className = 'detection-toast';
        toast.innerHTML = `
            <span class="toast-icon">${toastIcon}</span>
            <span class="toast-text">${toastText}</span>
            <span class="toast-hint">Click timeline marker for details</span>
            <button class="toast-dismiss" aria-label="Dismiss">&times;</button>
        `;

        // Dismiss button handler
        toast.querySelector('.toast-dismiss').onclick = () => toast.remove();

        container.appendChild(toast);

        // Auto-dismiss after 6 seconds
        setTimeout(() => toast.remove(), 6000);
    },

    formatDetectorName(name) {
        if (!name) return 'Detection';
        return name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' ');
    },

    getStageIcon(stage) {
        const icons = {
            'early': '\uD83E\uDD5A',
            'bean': '\uD83E\uDED8',
            'comma': '\uD83C\uDF19',
            '1.5fold': '\uD83D\uDD04',
            '2fold': '\uD83D\uDD01',
            '3fold': '\uD83D\uDD03',
            'pretzel': '\uD83E\uDD68',
            'hatching': '\uD83D\uDC23',
            'hatched': '\uD83D\uDC1B',
        };
        return icons[stage?.toLowerCase()] || '\uD83D\uDD2C';
    },

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

    showDetectionPanel() {
        // Jump to detection frame
        this.jumpToDetection();

        // Pause playback
        this.pause();

        // Show the context panel
        const panel = document.getElementById('video-context-panel');
        if (panel) {
            panel.classList.remove('hidden');
            panel.classList.add('overlay-mode');
        }

        // Remove toast if still visible
        document.querySelector('.detection-toast')?.remove();
    },

    hideDetectionPanel() {
        const panel = document.getElementById('video-context-panel');
        if (panel) {
            panel.classList.add('hidden');
            panel.classList.remove('overlay-mode');
        }
    },

    close() {
        this.pause();
        this.imageCache.clear();
        this.loadingSet.clear();
        this.sequence = [];

        // Remove video mode class
        const overlay = document.getElementById('lightbox-overlay');
        overlay?.classList.remove('video-mode');

        // Hide video UI
        document.getElementById('video-controls')?.classList.add('hidden');
        document.getElementById('video-timeline')?.classList.add('hidden');
        document.getElementById('video-context-panel')?.classList.add('hidden');

        // Show standard lightbox nav and position counter
        document.querySelectorAll('.lightbox-nav').forEach(el => el.style.display = '');
        document.getElementById('lightbox-thumbnails')?.classList.remove('hidden');
        const posEl = document.getElementById('lightbox-position');
        if (posEl) posEl.style.display = '';

        // Close lightbox
        Lightbox.close();

        // Remove key handler
        if (this._videoKeyHandler) {
            document.removeEventListener('keydown', this._videoKeyHandler);
        }
    },

    // Convenience method for playing all timepoints of an embryo
    async playAll(embryoId) {
        await this.openSequence(embryoId, 0, null, {
            bufferPercent: 0
        });
        this.play();
    }
};

// Make available globally
window.TimepointPlayer = TimepointPlayer;
