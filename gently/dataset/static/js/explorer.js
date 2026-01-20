// Embryo Dataset Explorer JavaScript

// =============================================================================
// Presence Manager - Collaborative presence like Google Docs
// =============================================================================

const PresenceManager = {
    clientId: null,
    name: null,
    clients: [],

    // Animal names for anonymous users
    ANIMALS: [
        'Koala', 'Penguin', 'Fox', 'Owl', 'Panda', 'Tiger', 'Dolphin',
        'Eagle', 'Bear', 'Wolf', 'Rabbit', 'Deer', 'Otter', 'Falcon',
        'Hedgehog', 'Badger', 'Lynx', 'Seal', 'Raven', 'Crane', 'Gecko',
        'Meerkat', 'Lemur', 'Toucan', 'Sloth', 'Jaguar', 'Pelican', 'Moose'
    ],

    init() {
        // Load or generate client ID
        this.clientId = localStorage.getItem('explorer-client-id');
        if (!this.clientId) {
            this.clientId = this.generateId();
            localStorage.setItem('explorer-client-id', this.clientId);
        }

        // Load saved name or generate anonymous name
        this.name = localStorage.getItem('explorer-user-name');
        if (!this.name) {
            this.name = this.getAnonymousName();
        }
    },

    generateId() {
        return 'xxxx-xxxx'.replace(/x/g, () =>
            Math.floor(Math.random() * 16).toString(16)
        );
    },

    getAnonymousName() {
        // Use client ID to pick a consistent animal
        const hash = this.clientId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
        const animal = this.ANIMALS[hash % this.ANIMALS.length];
        return `Anonymous ${animal}`;
    },

    sendJoin() {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'join',
                client_id: this.clientId,
                name: this.name
            }));
        }
    },

    handlePresenceUpdate(clients) {
        this.clients = clients;
        this.render();
    },

    render() {
        const container = document.getElementById('presence-container');
        if (!container) return;

        // Clear existing
        container.innerHTML = '';

        // Sort: put "you" first
        const sorted = [...this.clients].sort((a, b) => {
            if (a.is_you) return -1;
            if (b.is_you) return 1;
            return 0;
        });

        // Render avatars (max 5 visible, then +N)
        const maxVisible = 5;
        const visible = sorted.slice(0, maxVisible);
        const overflow = sorted.length - maxVisible;

        visible.forEach(client => {
            const avatar = document.createElement('div');
            avatar.className = 'presence-avatar' + (client.is_you ? ' is-you' : '');
            avatar.style.backgroundColor = client.color;
            avatar.textContent = this.getInitials(client.name);
            avatar.title = client.is_you ? `${client.name} (you)` : client.name;

            // Click on your own avatar to change name
            if (client.is_you) {
                avatar.style.cursor = 'pointer';
                avatar.addEventListener('click', () => this.showNamePrompt());
            }

            container.appendChild(avatar);
        });

        // Show overflow count
        if (overflow > 0) {
            const more = document.createElement('div');
            more.className = 'presence-overflow';
            more.textContent = `+${overflow}`;
            more.title = `${overflow} more viewer${overflow > 1 ? 's' : ''}`;
            container.appendChild(more);
        }
    },

    getInitials(name) {
        if (!name) return '?';
        // For "Anonymous X", use animal initial
        if (name.startsWith('Anonymous ')) {
            return name.split(' ')[1]?.[0] || 'A';
        }
        // Otherwise use first letter of each word (max 2)
        const words = name.trim().split(/\s+/);
        if (words.length === 1) {
            return words[0][0].toUpperCase();
        }
        return (words[0][0] + words[1][0]).toUpperCase();
    },

    showNamePrompt() {
        const newName = prompt('Enter your display name:', this.name);
        if (newName && newName.trim() && newName !== this.name) {
            this.name = newName.trim();
            localStorage.setItem('explorer-user-name', this.name);
            // Send update to server
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                state.ws.send(JSON.stringify({
                    type: 'update_name',
                    name: this.name
                }));
            }
        }
    }
};

// =============================================================================
// WebSocket Connection
// =============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log('WebSocket connected');
        PresenceManager.sendJoin();
    };

    state.ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'presence') {
                PresenceManager.handlePresenceUpdate(msg.clients);
            }
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    state.ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// State
let state = {
    sessions: [],
    embryos: [],
    embryoUids: [],  // Cross-session embryos
    selectedSession: null,
    selectedEmbryo: null,
    selectedEmbryoUid: null,  // For cross-session view
    timeline: null,
    selectedIndex: null,
    currentImage: null,
    currentTab: 'sessions',  // 'sessions' or 'embryos'
    // Unified timeline for cross-session view
    unifiedTimeline: null,
    selectedUnifiedIndex: null,
    // Annotation state (session view)
    annotationMode: null,  // null, 'annotating'
    annotationStage: null,
    annotationStart: null,
    // Annotation state (unified view)
    unifiedAnnotationMode: null,  // null, 'annotating'
    unifiedAnnotationStage: null,
    unifiedAnnotationStart: null,
    // Undo history
    undoStack: [],
    // WebSocket
    ws: null,
};

// Theme management
function toggleTheme() {
    const body = document.body;
    body.classList.toggle('light-mode');
    const isLight = body.classList.contains('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
}

function loadTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'light') {
        document.body.classList.add('light-mode');
    }
}

// Tab switching
function switchTab(tab) {
    state.currentTab = tab;

    // Update tab buttons
    document.getElementById('tab-sessions').classList.toggle('active', tab === 'sessions');
    document.getElementById('tab-embryos').classList.toggle('active', tab === 'embryos');

    // Show/hide views
    document.getElementById('sessions-view').style.display = tab === 'sessions' ? 'block' : 'none';
    document.getElementById('embryos-view').style.display = tab === 'embryos' ? 'block' : 'none';

    // Load data if needed
    if (tab === 'embryos' && state.embryoUids.length === 0) {
        loadEmbryoUids();
    }
}

// Load cross-session embryos
async function loadEmbryoUids() {
    try {
        state.embryoUids = await api('embryos_with_multiple_sessions');
        renderEmbryoUids();
    } catch (err) {
        console.error('Failed to load embryo UIDs:', err);
        document.getElementById('embryo-uids-list').innerHTML =
            '<div class="info">Failed to load embryo UIDs</div>';
    }
}

function renderEmbryoUids() {
    const container = document.getElementById('embryo-uids-list');
    if (!state.embryoUids || state.embryoUids.length === 0) {
        container.innerHTML = '<div class="info">No embryos with multiple sessions found</div>';
        return;
    }

    container.innerHTML = state.embryoUids.map(e => `
        <div class="embryo-uid-item ${state.selectedEmbryoUid === e.embryo_uid ? 'selected' : ''}"
             onclick="selectEmbryoUid('${e.embryo_uid}')">
            <div class="uid-name">${e.embryo_uid}</div>
            <div class="uid-info">
                ${e.session_count} sessions | ${e.total_volumes} volumes
                ${e.has_ground_truth ? '<span class="badge badge-gt">GT</span>' : ''}
            </div>
        </div>
    `).join('');
}

async function selectEmbryoUid(embryoUid) {
    state.selectedEmbryoUid = embryoUid;
    state.selectedSession = null;
    state.selectedEmbryo = null;
    state.unifiedTimeline = null;
    state.selectedUnifiedIndex = null;
    // Clear unified annotation state when switching embryos
    clearUnifiedAnnotationMode();
    renderEmbryoUids();

    // Load unified timeline for this embryo UID
    try {
        const res = await fetch(`/api/unified_timeline/${encodeURIComponent(embryoUid)}`);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        const data = await res.json();
        console.log('Unified timeline data:', data.images?.slice(0, 5).map(img => ({ session: img.session_id, gt: img.ground_truth_stage })));
        state.unifiedTimeline = data;
        renderCrossSessionView(data);
    } catch (err) {
        console.error('Failed to load embryo UID data:', err);
        document.getElementById('content-area').innerHTML = `
            <div class="info" style="color: var(--danger);">
                Failed to load data for embryo UID: ${embryoUid}<br>
                <small>${err.message}</small>
            </div>
        `;
    }
}

function renderCrossSessionView(data) {
    const container = document.getElementById('content-area');
    const images = data.images || [];
    const sessions = data.sessions || [];

    // Group images by session for display info
    const sessionCounts = {};
    images.forEach(img => {
        sessionCounts[img.session_id] = (sessionCounts[img.session_id] || 0) + 1;
    });

    // Calculate annotation preview range for unified view
    let previewStart = null, previewEnd = null;
    if (state.unifiedAnnotationMode === 'annotating' && state.unifiedAnnotationStart !== null && state.selectedUnifiedIndex !== null) {
        previewStart = Math.min(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
        previewEnd = Math.max(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
    }

    container.innerHTML = `
        <h2>Embryo: ${data.embryo_uid}</h2>
        <div class="info">
            ${sessions.length} session${sessions.length !== 1 ? 's' : ''} |
            ${images.length} total images
        </div>
        <div class="info" style="font-size: 0.8em; color: var(--text-muted);">
            Sessions: ${sessions.map(s => `${s} (${sessionCounts[s] || 0})`).join(', ')}
        </div>

        <div class="annotation-status" id="unified-annotation-status"></div>

        <div class="timeline" id="unified-timeline" style="margin-top: 20px;">
            ${images.map((img, idx) => {
                // Determine stage class - use preview stage if in annotation range
                const inPreviewRange = previewStart !== null && idx >= previewStart && idx <= previewEnd;
                let stageClass;
                if (inPreviewRange && state.unifiedAnnotationStage) {
                    stageClass = state.unifiedAnnotationStage.replace('.', '_');
                } else {
                    stageClass = (img.ground_truth_stage || 'unknown').replace('.', '_');
                }
                const isSelected = state.selectedUnifiedIndex === idx;
                // Extract timepoint from file_path if not available
                let tp = img.timepoint;
                if (!tp && img.file_path) {
                    const match = img.file_path.match(/_t(\d+)_/);
                    if (match) tp = parseInt(match[1], 10);
                }
                return `
                <div class="timeline-item stage-${stageClass} ${isSelected ? 'selected' : ''} ${inPreviewRange ? 'preview-range' : ''}"
                     onclick="handleUnifiedTimelineClick(${idx})"
                     title="#${idx + 1}: ${img.session_id} T${tp || '?'} - ${inPreviewRange ? state.unifiedAnnotationStage + ' (preview)' : (img.ground_truth_stage || 'unlabeled')}">
                    ${idx + 1}
                </div>
            `}).join('')}
        </div>

        <div class="image-viewer" id="unified-image-viewer" style="margin-top: 20px;">
            <div class="info">Click a timeline dot to view image</div>
        </div>
    `;

    updateUnifiedAnnotationStatus();
}

// Select image from unified timeline
async function selectUnifiedIndex(idx) {
    state.selectedUnifiedIndex = idx;
    const timeline = state.unifiedTimeline;
    if (!timeline || !timeline.images || idx >= timeline.images.length) return;

    const img = timeline.images[idx];

    // Update timeline selection
    document.querySelectorAll('#unified-timeline .timeline-item').forEach((el, i) => {
        el.classList.toggle('selected', i === idx);
    });

    // Load image from the appropriate session
    try {
        // Find the 0-based index within that session/embryo
        // Count how many images come before this one in the same session/embryo
        let sessionIndex = 0;
        for (let i = 0; i < idx; i++) {
            if (timeline.images[i].session_id === img.session_id &&
                timeline.images[i].embryo_id === img.embryo_id) {
                sessionIndex++;
            }
        }

        const imageData = await api(`image/${img.session_id}/${img.embryo_id}/${sessionIndex}`);

        // Try to load perception trace from JSON file (1-based timepoint)
        // Extract timepoint from file_path if not available (e.g., embryo_5_t0001_... -> 1)
        let trace = null;
        let timepoint = img.timepoint;
        if (!timepoint && img.file_path) {
            const match = img.file_path.match(/_t(\d+)_/);
            if (match) {
                timepoint = parseInt(match[1], 10);
            }
        }
        if (timepoint) {
            try {
                const traceRes = await fetch(`/api/trace/${img.session_id}/${img.embryo_id}/${timepoint}`);
                if (traceRes.ok) {
                    trace = await traceRes.json();
                }
            } catch (e) {
                // Trace not available
            }
        }

        // Add parsed timepoint to imgMeta for display
        img.parsedTimepoint = timepoint;
        renderUnifiedImageViewer(idx, img, imageData, trace);
    } catch (err) {
        console.error('Failed to load image:', err);
        document.getElementById('unified-image-viewer').innerHTML = `
            <div class="info" style="color: var(--danger);">Failed to load image: ${err.message}</div>
        `;
    }
}

function renderUnifiedImageViewer(idx, imgMeta, imageData, trace) {
    const viewer = document.getElementById('unified-image-viewer');
    if (!viewer) return;
    console.log('renderUnifiedImageViewer - imgMeta.ground_truth_stage:', imgMeta.ground_truth_stage);

    // Build annotation list from unified timeline (grouped by session/embryo)
    const annotationsBySession = {};
    if (state.unifiedTimeline && state.unifiedTimeline.images) {
        // Group images by session/embryo/stage
        const groups = {};
        state.unifiedTimeline.images.forEach((img, i) => {
            if (!img.ground_truth_stage) return;
            const key = `${img.session_id}|${img.embryo_id}|${img.ground_truth_stage}`;
            if (!groups[key]) {
                groups[key] = {
                    session_id: img.session_id,
                    embryo_id: img.embryo_id,
                    stage: img.ground_truth_stage,
                    indices: []
                };
            }
            groups[key].indices.push(i);
        });

        // Convert to array and find ranges
        for (const [key, group] of Object.entries(groups)) {
            const sessionKey = `${group.session_id}|${group.embryo_id}`;
            if (!annotationsBySession[sessionKey]) {
                annotationsBySession[sessionKey] = {
                    session_id: group.session_id,
                    embryo_id: group.embryo_id,
                    annotations: []
                };
            }
            // Find contiguous ranges (use first/last indices)
            const indices = group.indices.sort((a, b) => a - b);
            annotationsBySession[sessionKey].annotations.push({
                stage: group.stage,
                start: indices[0] + 1,  // 1-indexed for display
                end: indices[indices.length - 1] + 1
            });
        }
    }
    const hasAnnotations = Object.keys(annotationsBySession).length > 0;

    // Build perception display
    let predictionHtml = '';
    const source = trace;
    if (source) {
        const confidence = source.confidence ? (source.confidence * 100).toFixed(0) + '%' : 'N/A';
        const isCorrect = source.predicted_stage === imgMeta.ground_truth_stage;
        const predColor = isCorrect ? '#2ecc71' : (imgMeta.ground_truth_stage ? '#e74c3c' : '#00d4ff');

        // Build features display from trace
        let featuresHtml = '';
        if (trace && trace.observed_features) {
            const features = trace.observed_features;
            featuresHtml = `
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; color: var(--text-secondary);">Observed Features</summary>
                    <div style="margin-top: 8px; padding: 10px; background: var(--bg-primary); border-radius: 5px; font-size: 0.85em;">
                        ${features.shape ? `<div><strong>Shape:</strong> ${features.shape}</div>` : ''}
                        ${features.curvature ? `<div><strong>Curvature:</strong> ${features.curvature}</div>` : ''}
                        ${features.shell_status ? `<div><strong>Shell:</strong> ${features.shell_status}</div>` : ''}
                        ${features.emergence ? `<div><strong>Emergence:</strong> ${features.emergence}</div>` : ''}
                    </div>
                </details>
            `;
        }

        // Build contrastive reasoning display
        let contrastiveHtml = '';
        if (trace && trace.contrastive_reasoning) {
            const cr = trace.contrastive_reasoning;
            contrastiveHtml = `
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; color: var(--text-secondary);">Contrastive Reasoning</summary>
                    <div style="margin-top: 8px; padding: 10px; background: var(--bg-primary); border-radius: 5px; font-size: 0.85em;">
                        ${cr.why_not_previous ? `<div style="margin-bottom: 5px;"><strong>Why not previous:</strong> ${cr.why_not_previous}</div>` : ''}
                        ${cr.why_not_next ? `<div><strong>Why not next:</strong> ${cr.why_not_next}</div>` : ''}
                    </div>
                </details>
            `;
        }

        predictionHtml = `
            <div style="margin-top: 15px; padding: 10px; background: var(--bg-primary); border-radius: 8px;">
                <h3 style="color: var(--accent); margin-bottom: 8px;">Perception (trace)</h3>
                <div style="display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap;">
                    <div>Stage: <span style="color: ${predColor}; font-weight: bold;">${source.predicted_stage}</span></div>
                    <div>Confidence: <span style="color: var(--warning);">${confidence}</span></div>
                    ${imgMeta.ground_truth_stage ? `<div>${isCorrect ? '✓ Correct' : '✗ Wrong'}</div>` : ''}
                </div>
                ${source.reasoning ? `
                    <details style="margin-top: 10px;" open>
                        <summary style="cursor: pointer; color: var(--text-secondary);">Reasoning</summary>
                        <div style="margin-top: 8px; padding: 10px; background: var(--bg-primary); border-radius: 5px; font-size: 0.85em; line-height: 1.5; max-height: 200px; overflow-y: auto;">
                            ${source.reasoning.replace(/\n/g, '<br>')}
                        </div>
                    </details>
                ` : ''}
                ${featuresHtml}
                ${contrastiveHtml}
            </div>
        `;
    }

    // Calculate session-relative index for projection viewer
    const timeline = state.unifiedTimeline;
    let sessionIndex = 0;
    for (let i = 0; i < idx; i++) {
        if (timeline.images[i].session_id === imgMeta.session_id &&
            timeline.images[i].embryo_id === imgMeta.embryo_id) {
            sessionIndex++;
        }
    }

    viewer.innerHTML = `
        <div class="image-container" style="flex: 1;">
            <div style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>Image #${idx + 1}</strong>
                    <span style="color: var(--text-secondary); font-size: 0.9em;"> - ${imgMeta.session_id} / ${imgMeta.embryo_id} / T${imgMeta.parsedTimepoint || imgMeta.timepoint || '?'}</span>
                    ${imgMeta.ground_truth_stage ? `<span class="current-gt"> - GT: ${imgMeta.ground_truth_stage}</span>` : ''}
                </div>
                <button onclick="openUnifiedProjections('${imgMeta.session_id}', '${imgMeta.embryo_id}', ${sessionIndex})" style="padding: 5px 12px; background: #3498db; border: none; border-radius: 5px; color: #fff; cursor: pointer; font-size: 0.9em;">
                    View More →
                </button>
            </div>
            <div style="font-size: 0.8em; color: var(--text-muted); margin-bottom: 10px;">${imgMeta.timestamp || ''}</div>
            ${imageData.image_b64 ? `<img src="data:image/jpeg;base64,${imageData.image_b64}" alt="Image ${idx}">` : '<div class="info">Image not available</div>'}
            ${predictionHtml}
        </div>
        <div class="annotation-panel" style="width: 220px;">
            <h3 style="color: var(--accent); margin-bottom: 10px;">Annotate Stage</h3>
            <p class="info" style="font-size: 0.8em; margin-bottom: 8px;">
                ${state.unifiedAnnotationMode === 'annotating'
                    ? 'Use ← → arrows to extend range, Enter to finish, Esc to cancel'
                    : 'Click stage to start, then extend range with arrows'}
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-bottom: 10px;">
                ${STAGES.map((stage, i) => {
                    const isActive = state.unifiedAnnotationStage === stage;
                    return `
                    <button class="stage-btn stage-${stage.replace('.', '_')} ${isActive ? 'active' : ''}"
                            onclick="startUnifiedAnnotation('${stage}')"
                            style="padding: 6px; font-size: 0.85em;"
                            ${state.unifiedAnnotationMode === 'annotating' && !isActive ? 'disabled' : ''}>
                        ${stage}
                    </button>
                `}).join('')}
            </div>

            ${imgMeta.ground_truth_stage ? `
                <button class="stage-btn" style="background: #e74c3c; font-size: 0.85em;" onclick="deleteUnifiedAnnotation('${imgMeta.session_id}', '${imgMeta.embryo_id}', '${imgMeta.ground_truth_stage}')">
                    Delete "${imgMeta.ground_truth_stage}"
                </button>
            ` : ''}

            ${hasAnnotations ? `
                <div style="margin-top: 15px; border-top: 1px solid var(--border); padding-top: 10px;">
                    <h3 style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 8px;">Current Annotations</h3>
                    <div style="display: flex; flex-direction: column; gap: 5px; max-height: 180px; overflow-y: auto;">
                        ${Object.values(annotationsBySession).map(sessionGroup => `
                            <div style="font-size: 0.7em; color: var(--text-muted); margin-top: 3px;">${sessionGroup.session_id}</div>
                            ${sessionGroup.annotations.map(ann => `
                                <div style="display: flex; align-items: center; gap: 6px; padding: 4px; background: var(--bg-tertiary); border-radius: 4px;">
                                    <span class="stage-${ann.stage.replace('.', '_')}" style="padding: 2px 5px; border-radius: 3px; font-size: 0.8em;">${ann.stage}</span>
                                    <span style="flex: 1; font-size: 0.75em; color: var(--text-secondary);">#${ann.start}-${ann.end}</span>
                                    <button onclick="deleteUnifiedAnnotation('${sessionGroup.session_id}', '${sessionGroup.embryo_id}', '${ann.stage}')" style="padding: 2px 6px; background: #e74c3c; border: none; border-radius: 3px; color: #fff; cursor: pointer; font-size: 0.7em;">×</button>
                                </div>
                            `).join('')}
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            <div style="margin-top: 15px; border-top: 1px solid var(--border); padding-top: 10px;">
                <div style="display: flex; gap: 5px; margin-bottom: 10px;">
                    <button class="stage-btn" style="background: #555; flex: 1; font-size: 0.85em;" onclick="prevUnifiedImage()">← Prev</button>
                    <button class="stage-btn" style="background: #555; flex: 1; font-size: 0.85em;" onclick="nextUnifiedImage()">Next →</button>
                </div>
                <button class="stage-btn" style="background: var(--accent); font-size: 0.85em;" onclick="openUnifiedSession('${imgMeta.session_id}', '${imgMeta.embryo_id}')">
                    Open Full Session →
                </button>
            </div>
        </div>
    `;
}

// Navigation for unified timeline
async function prevUnifiedImage() {
    if (!state.unifiedTimeline || state.selectedUnifiedIndex === null) return;
    if (state.selectedUnifiedIndex > 0) {
        await selectUnifiedIndex(state.selectedUnifiedIndex - 1);
    }
}

async function nextUnifiedImage() {
    if (!state.unifiedTimeline || state.selectedUnifiedIndex === null) return;
    if (state.selectedUnifiedIndex < state.unifiedTimeline.images.length - 1) {
        await selectUnifiedIndex(state.selectedUnifiedIndex + 1);
    }
}

function openUnifiedSession(sessionId, embryoId) {
    // Switch to sessions tab and open this session/embryo
    switchTab('sessions');
    selectSession(sessionId).then(() => {
        selectEmbryo(embryoId);
    });
}

function openUnifiedProjections(sessionId, embryoId, sessionIndex) {
    // Open projection viewer for this image
    const url = `/projections/${sessionId}/${embryoId}/${sessionIndex}`;
    window.open(url, '_blank');
}

// Handle unified timeline click
async function handleUnifiedTimelineClick(idx) {
    await selectUnifiedIndex(idx);
    // If in annotation mode, update the timeline preview
    if (state.unifiedAnnotationMode === 'annotating') {
        updateUnifiedTimelinePreview();
    }
}

// Start annotation mode in unified view
function startUnifiedAnnotation(stage) {
    console.log('startUnifiedAnnotation called, stage:', stage, 'selectedUnifiedIndex:', state.selectedUnifiedIndex);
    if (state.selectedUnifiedIndex === null) {
        alert('Please select a timepoint first');
        return;
    }
    state.unifiedAnnotationStage = stage;
    state.unifiedAnnotationMode = 'annotating';
    state.unifiedAnnotationStart = state.selectedUnifiedIndex;
    console.log('Annotation mode set:', state.unifiedAnnotationMode, 'stage:', state.unifiedAnnotationStage, 'start:', state.unifiedAnnotationStart);

    // Re-render to show preview
    renderCrossSessionView(state.unifiedTimeline);
    selectUnifiedIndex(state.selectedUnifiedIndex);
}

// Update unified annotation status bar
function updateUnifiedAnnotationStatus() {
    const statusEl = document.getElementById('unified-annotation-status');
    if (!statusEl) return;

    if (state.unifiedAnnotationMode === 'annotating') {
        const start = Math.min(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
        const end = Math.max(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
        statusEl.innerHTML = `
            <div class="status-message selecting" style="display: flex; align-items: center; gap: 10px; padding: 10px; background: var(--bg-tertiary); border-radius: 5px; margin: 10px 0;">
                <span>Annotating <strong class="stage-${state.unifiedAnnotationStage.replace('.', '_')}" style="padding: 2px 8px; border-radius: 3px;">${state.unifiedAnnotationStage}</strong></span>
                <span style="color: var(--text-secondary);">#${start + 1} - #${end + 1}</span>
                <button onclick="finishUnifiedAnnotation()" style="padding: 5px 15px; background: #2ecc71; border: none; border-radius: 5px; color: #fff; cursor: pointer; font-weight: bold;">Finish</button>
                <button onclick="cancelUnifiedAnnotation()" style="padding: 5px 15px; background: #e74c3c; border: none; border-radius: 5px; color: #fff; cursor: pointer;">Cancel</button>
            </div>
        `;
        statusEl.style.display = 'block';
    } else {
        statusEl.style.display = 'none';
    }
}

// Update unified timeline dots in-place for preview
function updateUnifiedTimelinePreview() {
    if (!state.unifiedTimeline) return;

    const timelineEl = document.getElementById('unified-timeline');
    if (!timelineEl) return;

    // Calculate preview range
    let previewStart = null, previewEnd = null;
    if (state.unifiedAnnotationMode === 'annotating' && state.unifiedAnnotationStart !== null && state.selectedUnifiedIndex !== null) {
        previewStart = Math.min(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
        previewEnd = Math.max(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
    }

    // Update each timeline dot
    const dots = timelineEl.querySelectorAll('.timeline-item');
    dots.forEach((dot, idx) => {
        const img = state.unifiedTimeline.images[idx];
        const inPreviewRange = previewStart !== null && idx >= previewStart && idx <= previewEnd;

        // Update selection
        dot.classList.toggle('selected', state.selectedUnifiedIndex === idx);
        dot.classList.toggle('preview-range', inPreviewRange);

        // Update stage class for preview
        dot.className = dot.className.replace(/stage-\w+/g, '');

        let stageClass;
        if (inPreviewRange && state.unifiedAnnotationStage) {
            stageClass = state.unifiedAnnotationStage.replace('.', '_');
        } else {
            stageClass = (img.ground_truth_stage || 'unknown').replace('.', '_');
        }
        dot.classList.add(`stage-${stageClass}`);
    });

    // Update annotation status bar
    updateUnifiedAnnotationStatus();
}

// Finish unified annotation - save the range
async function finishUnifiedAnnotation() {
    console.log('finishUnifiedAnnotation called, mode:', state.unifiedAnnotationMode, 'start:', state.unifiedAnnotationStart);
    if (state.unifiedAnnotationMode !== 'annotating' || state.unifiedAnnotationStart === null) {
        console.log('Early return - not in annotation mode');
        return;
    }

    const startIdx = Math.min(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
    const endIdx = Math.max(state.unifiedAnnotationStart, state.selectedUnifiedIndex);
    const indexToRestore = state.selectedUnifiedIndex;
    const embryoUidToRestore = state.selectedEmbryoUid;

    // Get session/embryo info for each image in range and group by session/embryo
    // Use ROW INDEX within each session/embryo (not v.timepoint!)
    const annotationGroups = {};
    for (let i = startIdx; i <= endIdx; i++) {
        const img = state.unifiedTimeline.images[i];
        const key = `${img.session_id}|${img.embryo_id}`;
        if (!annotationGroups[key]) {
            annotationGroups[key] = {
                session_id: img.session_id,
                embryo_id: img.embryo_id,
                indices: []
            };
        }
        // Calculate session-relative index (row index within this session/embryo)
        let sessionIndex = 0;
        for (let j = 0; j < i; j++) {
            if (state.unifiedTimeline.images[j].session_id === img.session_id &&
                state.unifiedTimeline.images[j].embryo_id === img.embryo_id) {
                sessionIndex++;
            }
        }
        annotationGroups[key].indices.push(sessionIndex);
    }

    // Save annotations for each session/embryo group
    try {
        for (const group of Object.values(annotationGroups)) {
            if (group.indices.length === 0) {
                console.warn('No indices for group:', group.session_id, group.embryo_id);
                continue;
            }
            const minIdx = Math.min(...group.indices);
            const maxIdx = Math.max(...group.indices);
            console.log('Saving annotation:', group.session_id, group.embryo_id, state.unifiedAnnotationStage, 'indices:', minIdx, '-', maxIdx);
            const result = await apiPost('ground_truth', {
                session_id: group.session_id,
                embryo_id: group.embryo_id,
                stage: state.unifiedAnnotationStage,
                start_timepoint: minIdx,
                end_timepoint: maxIdx + 1,  // exclusive
                annotator: 'web_explorer',
            });
            console.log('Save result:', result);
        }
    } catch (err) {
        console.error('Failed to save annotation:', err);
        alert('Failed to save annotation: ' + err.message);
        return;
    }

    clearUnifiedAnnotationMode();
    loadStats();
    // Reload embryo UIDs list to update GT badges
    loadEmbryoUids();

    // Reload unified timeline
    if (embryoUidToRestore) {
        console.log('Reloading embryo UID:', embryoUidToRestore);
        await selectEmbryoUid(embryoUidToRestore);
        console.log('After reload, unifiedTimeline:', state.unifiedTimeline?.images?.length, 'images');
        // Make sure we have a valid timeline before selecting an index
        if (state.unifiedTimeline && state.unifiedTimeline.images && indexToRestore !== null) {
            const maxIdx = state.unifiedTimeline.images.length - 1;
            const safeIdx = Math.min(indexToRestore, maxIdx);
            console.log('Restoring index:', safeIdx, 'of', maxIdx);
            if (safeIdx >= 0) {
                await selectUnifiedIndex(safeIdx);
            }
        } else {
            console.log('Could not restore index - timeline:', !!state.unifiedTimeline, 'images:', state.unifiedTimeline?.images?.length, 'indexToRestore:', indexToRestore);
        }
    } else {
        console.log('No embryoUidToRestore');
    }
}

// Cancel unified annotation
function cancelUnifiedAnnotation() {
    clearUnifiedAnnotationMode();
    renderCrossSessionView(state.unifiedTimeline);
    if (state.selectedUnifiedIndex !== null) {
        selectUnifiedIndex(state.selectedUnifiedIndex);
    }
}

// Clear unified annotation mode
function clearUnifiedAnnotationMode() {
    state.unifiedAnnotationMode = null;
    state.unifiedAnnotationStage = null;
    state.unifiedAnnotationStart = null;
}

// Delete annotation in unified view
async function deleteUnifiedAnnotation(sessionId, embryoId, stage) {
    if (!confirm(`Delete "${stage}" annotation for ${embryoId}?`)) return;

    const idxToRestore = state.selectedUnifiedIndex;

    await apiDelete('ground_truth', {
        session_id: sessionId,
        embryo_id: embryoId,
        stage: stage,
    });

    // Refresh stats and embryo UIDs list
    loadStats();
    loadEmbryoUids();

    // Reload the unified timeline
    await selectEmbryoUid(state.selectedEmbryoUid);

    // Restore selected index
    if (idxToRestore !== null) {
        await selectUnifiedIndex(idxToRestore);
    }
}

const STAGES = ['early', 'bean', 'comma', '1.5fold', '2fold', 'pretzel', 'hatching', 'hatched'];

// API helpers
async function api(endpoint) {
    const res = await fetch('/api/' + endpoint);
    return res.json();
}

async function apiPost(endpoint, data) {
    const res = await fetch('/api/' + endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    return res.json();
}

async function apiDelete(endpoint, data) {
    const res = await fetch('/api/' + endpoint, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    return res.json();
}

// Load stats
async function loadStats() {
    const stats = await api('stats');
    document.getElementById('stat-sessions').textContent = stats.sessions?.toLocaleString() || 0;
    document.getElementById('stat-embryos').textContent = stats.embryos?.toLocaleString() || 0;
    document.getElementById('stat-volumes').textContent = stats.volumes?.toLocaleString() || 0;
    document.getElementById('stat-images').textContent = stats.images?.toLocaleString() || 0;
    document.getElementById('stat-ground_truth').textContent = stats.ground_truth?.toLocaleString() || 0;
}

// Load sessions
async function loadSessions() {
    state.sessions = await api('sessions');
    renderSessions();
}

function renderSessions() {
    const container = document.getElementById('sessions-list');
    const filterWithEmbryos = document.getElementById('filter-with-embryos').checked;
    const filterWithData = document.getElementById('filter-with-data').checked;

    let filteredSessions = state.sessions;
    if (filterWithEmbryos) {
        filteredSessions = filteredSessions.filter(s => s.embryo_count > 0);
    }
    if (filterWithData) {
        filteredSessions = filteredSessions.filter(s => s.volume_count > 0);
    }

    container.innerHTML = filteredSessions.map(s => {
        const date = s.created_at ? s.created_at.split('T')[0] : '';
        return `
        <div class="session-item ${state.selectedSession === s.session_id ? 'selected' : ''}"
             onclick="selectSession('${s.session_id}')">
            <div><strong>${s.session_id}</strong></div>
            <div style="font-size: 0.8em; color: #888;">
                ${date} | ${s.embryo_count} embryos | ${s.volume_count || 0} vol
                ${s.has_ground_truth ? '<span class="badge badge-gt">GT</span>' : ''}
            </div>
        </div>
    `}).join('');
}

function filterSessions() {
    renderSessions();
}

// Select session
async function selectSession(sessionId) {
    state.selectedSession = sessionId;
    state.selectedEmbryo = null;
    state.timeline = null;
    clearAnnotationMode();
    renderSessions();

    const embryos = await api(`embryos?session_id=${sessionId}`);
    state.embryos = embryos;
    renderEmbryoGrid();
}

function renderEmbryoGrid() {
    const container = document.getElementById('content-area');
    if (!state.embryos || state.embryos.length === 0) {
        container.innerHTML = '<div class="info">No embryos in this session</div>';
        return;
    }

    container.innerHTML = `
        <div class="content-layout">
            <div class="embryo-panel">
                <h2>Session ${state.selectedSession}</h2>
                <div class="embryo-list">
                    ${state.embryos.map(e => `
                        <div class="embryo-card ${state.selectedEmbryo === e.embryo_id ? 'selected' : ''}"
                             onclick="selectEmbryo('${e.embryo_id}')">
                            <div class="embryo-card-title">${e.embryo_id}</div>
                            <div class="embryo-card-info">
                                ${e.num_volumes} vol
                                ${e.has_ground_truth ? '<span class="badge badge-gt">GT</span>' : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div class="timeline-panel" id="timeline-panel">
                <div class="info">Select an embryo to view timeline</div>
            </div>
        </div>
    `;
}

// Select embryo
async function selectEmbryo(embryoId) {
    state.selectedEmbryo = embryoId;
    state.selectedIndex = null;
    clearAnnotationMode();

    // Update embryo selection visually
    document.querySelectorAll('.embryo-card').forEach(el => {
        el.classList.toggle('selected', el.querySelector('.embryo-card-title')?.textContent === embryoId);
    });

    const timeline = await api(`timeline/${state.selectedSession}/${embryoId}`);
    state.timeline = timeline;

    renderTimeline();
}

function renderTimeline() {
    if (!state.timeline) return;

    const t = state.timeline;
    const container = document.getElementById('timeline-panel');
    if (!container) return;

    // Build stage summary with ranges
    let stageRanges = t.ground_truth.map(g => {
        const endStr = g.end_timepoint ? `-${g.end_timepoint}` : '+';
        return `${g.stage}(${g.start_timepoint}${endStr})`;
    }).join(' → ');

    // Calculate annotation preview range
    let previewStart = null, previewEnd = null;
    if (state.annotationMode === 'annotating' && state.annotationStart !== null && state.selectedIndex !== null) {
        previewStart = Math.min(state.annotationStart, state.selectedIndex);
        previewEnd = Math.max(state.annotationStart, state.selectedIndex);
    }

    container.innerHTML = `
        <h2>${t.embryo_id}</h2>

        <div class="info">
            ${t.images.length} images |
            GT: ${stageRanges || 'None'}
        </div>

        <div class="annotation-status" id="annotation-status"></div>

        <div class="timeline" id="timeline">
            ${t.images.map((img, idx) => {
                // Determine stage class - use preview stage if in annotation range
                let stageClass;
                const inPreviewRange = previewStart !== null && idx >= previewStart && idx <= previewEnd;
                if (inPreviewRange && state.annotationStage) {
                    stageClass = state.annotationStage.replace('.', '_');
                } else {
                    stageClass = (img.ground_truth_stage || 'unknown').replace('.', '_');
                }
                return `
                <div class="timeline-item stage-${stageClass} ${state.selectedIndex === idx ? 'selected' : ''} ${inPreviewRange ? 'preview-range' : ''}"
                     onclick="handleTimelineClick(${idx})"
                     title="#${idx + 1}: ${inPreviewRange ? state.annotationStage + ' (preview)' : (img.ground_truth_stage || 'unlabeled')}">
                    ${idx + 1}
                </div>
            `}).join('')}
        </div>

        <div class="image-viewer" id="image-viewer">
            <div class="info">Click a timeline dot to view image</div>
        </div>
    `;

    updateAnnotationStatus();
}

function updateAnnotationStatus() {
    const statusEl = document.getElementById('annotation-status');
    if (!statusEl) return;

    if (state.annotationMode === 'annotating') {
        const start = Math.min(state.annotationStart, state.selectedIndex);
        const end = Math.max(state.annotationStart, state.selectedIndex);
        statusEl.innerHTML = `
            <div class="status-message selecting" style="display: flex; align-items: center; gap: 10px; padding: 10px; background: var(--bg-tertiary); border-radius: 5px; margin: 10px 0;">
                <span>Annotating <strong class="stage-${state.annotationStage.replace('.', '_')}" style="padding: 2px 8px; border-radius: 3px;">${state.annotationStage}</strong></span>
                <span style="color: var(--text-secondary);">#${start + 1} - #${end + 1}</span>
                <button onclick="finishAnnotation()" style="padding: 5px 15px; background: #2ecc71; border: none; border-radius: 5px; color: #fff; cursor: pointer; font-weight: bold;">Finish</button>
                <button onclick="cancelAnnotation()" style="padding: 5px 15px; background: #e74c3c; border: none; border-radius: 5px; color: #fff; cursor: pointer;">Cancel</button>
            </div>
        `;
        statusEl.style.display = 'block';
    } else {
        statusEl.style.display = 'none';
    }
}

// Update timeline dots in-place without rebuilding the whole panel
// This is used during annotation navigation to update preview colors
function updateTimelinePreview() {
    if (!state.timeline) return;

    const timelineEl = document.getElementById('timeline');
    if (!timelineEl) return;

    // Calculate preview range
    let previewStart = null, previewEnd = null;
    if (state.annotationMode === 'annotating' && state.annotationStart !== null && state.selectedIndex !== null) {
        previewStart = Math.min(state.annotationStart, state.selectedIndex);
        previewEnd = Math.max(state.annotationStart, state.selectedIndex);
    }

    // Update each timeline dot
    const dots = timelineEl.querySelectorAll('.timeline-item');
    dots.forEach((dot, idx) => {
        const img = state.timeline.images[idx];
        const inPreviewRange = previewStart !== null && idx >= previewStart && idx <= previewEnd;

        // Update selection
        dot.classList.toggle('selected', state.selectedIndex === idx);
        dot.classList.toggle('preview-range', inPreviewRange);

        // Update stage class for preview
        // First remove all stage classes
        dot.className = dot.className.replace(/stage-\w+/g, '');

        // Add appropriate stage class
        let stageClass;
        if (inPreviewRange && state.annotationStage) {
            stageClass = state.annotationStage.replace('.', '_');
        } else {
            stageClass = (img.ground_truth_stage || 'unknown').replace('.', '_');
        }
        dot.classList.add(`stage-${stageClass}`);
    });

    // Update annotation status bar
    updateAnnotationStatus();
}

// Handle timeline click - select image (annotation starts from stage buttons)
async function handleTimelineClick(idx) {
    await selectIndex(idx);
    // If in annotation mode, update the timeline preview (not full rebuild)
    if (state.annotationMode === 'annotating') {
        updateTimelinePreview();
    }
}

// Start annotation mode for a stage at current position
function startAnnotation(stage) {
    if (state.selectedIndex === null) {
        alert('Please select a timepoint first');
        return;
    }
    state.annotationStage = stage;
    state.annotationMode = 'annotating';
    state.annotationStart = state.selectedIndex;
    renderTimeline();
    renderImageViewer();
}

// Finish annotation - save the range
async function finishAnnotation() {
    if (state.annotationMode !== 'annotating' || state.annotationStart === null) return;

    const startIdx = Math.min(state.annotationStart, state.selectedIndex);
    const endIdx = Math.max(state.annotationStart, state.selectedIndex);
    const indexToRestore = state.selectedIndex;  // Save before clearing

    // Use indices as timepoints (database timepoints may be null)
    await saveAnnotation(state.annotationStage, startIdx, endIdx);
    clearAnnotationMode();

    // Reload timeline to show saved annotation
    await selectEmbryo(state.selectedEmbryo);
    // Restore the selected index (selectEmbryo resets it to null)
    if (indexToRestore !== null) {
        await selectIndex(indexToRestore);
    }
}

// Cancel annotation without saving
function cancelAnnotation() {
    clearAnnotationMode();
    renderTimeline();
    renderImageViewer();
}

// Clear annotation mode
function clearAnnotationMode() {
    state.annotationMode = null;
    state.annotationStage = null;
    state.annotationStart = null;
}

// Delete GT at current position
async function deleteCurrentGT() {
    if (!state.timeline || state.selectedIndex === null) return;

    const img = state.timeline.images[state.selectedIndex];
    if (!img.ground_truth_stage) {
        alert('No ground truth at this position');
        return;
    }

    if (!confirm(`Delete ${img.ground_truth_stage} annotation?`)) return;

    await deleteAnnotation(img.ground_truth_stage);
}

// Save annotation to backend (caller is responsible for reload)
async function saveAnnotation(stage, start, end) {
    // Save for undo
    const prevGroundTruth = [...state.timeline.ground_truth];
    state.undoStack.push({
        type: 'annotation',
        session_id: state.selectedSession,
        embryo_id: state.selectedEmbryo,
        previous: prevGroundTruth,
    });

    await apiPost('ground_truth', {
        session_id: state.selectedSession,
        embryo_id: state.selectedEmbryo,
        stage: stage,
        start_timepoint: start,
        end_timepoint: end + 1,  // end is exclusive
        annotator: 'web_explorer',
    });

    // Refresh stats
    loadStats();
}

// Delete a stage annotation (with confirmation)
async function deleteAnnotation(stage) {
    if (!confirm(`Delete ${stage} annotation?`)) return;
    await deleteAnnotationByStage(stage);
}

// Delete annotation by stage name (from the annotation list)
async function deleteAnnotationByStage(stage) {
    const indexToRestore = state.selectedIndex;  // Save before reload

    await apiDelete('ground_truth', {
        session_id: state.selectedSession,
        embryo_id: state.selectedEmbryo,
        stage: stage,
    });

    // Reload timeline
    await selectEmbryo(state.selectedEmbryo);
    if (indexToRestore !== null) {
        await selectIndex(indexToRestore);
    }

    loadStats();
}

// Undo last action
async function undo() {
    if (state.undoStack.length === 0) {
        alert('Nothing to undo');
        return;
    }

    const action = state.undoStack.pop();

    // Delete all current ground truth for this embryo
    await apiDelete('ground_truth', {
        session_id: action.session_id,
        embryo_id: action.embryo_id,
    });

    // Restore previous ground truth
    for (const gt of action.previous) {
        await apiPost('ground_truth', {
            session_id: action.session_id,
            embryo_id: action.embryo_id,
            stage: gt.stage,
            start_timepoint: gt.start_timepoint,
            end_timepoint: gt.end_timepoint,
            annotator: gt.annotator,
        });
    }

    // Reload
    await selectEmbryo(state.selectedEmbryo);
    if (state.selectedIndex !== null) {
        await selectIndex(state.selectedIndex);
    }

    loadStats();
}

// Select image by index
async function selectIndex(idx) {
    state.selectedIndex = idx;

    // Update timeline selection
    document.querySelectorAll('.timeline-item').forEach((el, i) => {
        el.classList.toggle('selected', i === idx);
    });

    // Load image by index
    try {
        const img = await api(`image/${state.selectedSession}/${state.selectedEmbryo}/${idx}`);
        state.currentImage = img;
        state.currentImage.index = idx;

        // Look up prediction for this index if available (from DB)
        if (state.timeline && state.timeline.predictions) {
            const pred = state.timeline.predictions.find(p => p.timepoint === idx);
            state.currentImage.prediction = pred || null;
        }

        // Try to load perception trace from JSON file (1-based timepoint)
        try {
            const traceRes = await fetch(`/api/trace/${state.selectedSession}/${state.selectedEmbryo}/${idx + 1}`);
            if (traceRes.ok) {
                state.currentImage.trace = await traceRes.json();
            } else {
                state.currentImage.trace = null;
            }
        } catch (e) {
            state.currentImage.trace = null;
        }

        renderImageViewer();
    } catch (err) {
        console.error('Failed to load image:', err);
    }
}

function renderImageViewer() {
    const img = state.currentImage;
    if (!img) return;

    const idx = state.selectedIndex;
    const pred = img.prediction;
    const trace = img.trace;
    const viewer = document.getElementById('image-viewer');

    // Build perception display - prefer trace (from JSON) over prediction (from DB)
    let predictionHtml = '';
    const source = trace || pred;
    if (source) {
        const confidence = source.confidence ? (source.confidence * 100).toFixed(0) + '%' : 'N/A';
        const isCorrect = source.predicted_stage === img.ground_truth_stage;
        const predColor = isCorrect ? '#2ecc71' : (img.ground_truth_stage ? '#e74c3c' : '#00d4ff');

        // Build features display from trace
        let featuresHtml = '';
        if (trace && trace.observed_features) {
            const features = trace.observed_features;
            featuresHtml = `
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; color: var(--text-secondary);">Observed Features</summary>
                    <div style="margin-top: 8px; padding: 10px; background: var(--bg-primary); border-radius: 5px; font-size: 0.85em;">
                        ${features.shape ? `<div><strong>Shape:</strong> ${features.shape}</div>` : ''}
                        ${features.curvature ? `<div><strong>Curvature:</strong> ${features.curvature}</div>` : ''}
                        ${features.shell_status ? `<div><strong>Shell:</strong> ${features.shell_status}</div>` : ''}
                        ${features.emergence ? `<div><strong>Emergence:</strong> ${features.emergence}</div>` : ''}
                    </div>
                </details>
            `;
        }

        // Build contrastive reasoning display
        let contrastiveHtml = '';
        if (trace && trace.contrastive_reasoning) {
            const cr = trace.contrastive_reasoning;
            contrastiveHtml = `
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; color: var(--text-secondary);">Contrastive Reasoning</summary>
                    <div style="margin-top: 8px; padding: 10px; background: var(--bg-primary); border-radius: 5px; font-size: 0.85em;">
                        ${cr.why_not_previous ? `<div style="margin-bottom: 5px;"><strong>Why not previous:</strong> ${cr.why_not_previous}</div>` : ''}
                        ${cr.why_not_next ? `<div><strong>Why not next:</strong> ${cr.why_not_next}</div>` : ''}
                    </div>
                </details>
            `;
        }

        predictionHtml = `
            <div style="margin-top: 15px; padding: 10px; background: var(--bg-primary); border-radius: 8px;">
                <h3 style="color: var(--accent); margin-bottom: 8px;">Perception ${trace ? '(trace)' : '(DB)'}</h3>
                <div style="display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap;">
                    <div>Stage: <span style="color: ${predColor}; font-weight: bold;">${source.predicted_stage}</span></div>
                    <div>Confidence: <span style="color: var(--warning);">${confidence}</span></div>
                    ${img.ground_truth_stage ? `<div>${isCorrect ? '✓ Correct' : '✗ Wrong'}</div>` : ''}
                </div>
                ${source.reasoning ? `
                    <details style="margin-top: 10px;" open>
                        <summary style="cursor: pointer; color: var(--text-secondary);">Reasoning</summary>
                        <div style="margin-top: 8px; padding: 10px; background: var(--bg-primary); border-radius: 5px; font-size: 0.85em; line-height: 1.5; max-height: 200px; overflow-y: auto;">
                            ${source.reasoning.replace(/\n/g, '<br>')}
                        </div>
                    </details>
                ` : ''}
                ${featuresHtml}
                ${contrastiveHtml}
            </div>
        `;
    }

    // Check if in annotation mode
    const isAnnotating = state.annotationMode === 'annotating';

    viewer.innerHTML = `
        <div class="image-container">
            <div style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>Image #${idx + 1}</strong>
                    ${img.ground_truth_stage ? `<span class="current-gt"> - GT: ${img.ground_truth_stage}</span>` : ''}
                    <span style="color: #666; font-size: 0.9em;"> ${img.timestamp || ''}</span>
                </div>
                <button onclick="openProjections()" style="padding: 5px 12px; background: #3498db; border: none; border-radius: 5px; color: #fff; cursor: pointer; font-size: 0.9em;">
                    View More →
                </button>
            </div>
            ${img.image_b64 ? `<img src="data:image/jpeg;base64,${img.image_b64}" alt="Image ${idx}">` : '<div class="info">Image not available</div>'}
            ${predictionHtml}
        </div>

        <div class="annotation-panel">
            <h2>Annotate Stage</h2>
            <p class="info" style="font-size: 0.85em; margin-bottom: 10px;">
                ${isAnnotating
                    ? 'Use ← → arrows to extend range, Enter to finish, Esc to cancel'
                    : 'Click stage to start, Delete to remove GT at current position'}
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
                ${STAGES.map((stage, i) => {
                    const existing = state.timeline?.ground_truth?.find(g => g.stage === stage);
                    const isActive = state.annotationStage === stage;
                    return `
                    <button class="stage-btn stage-${stage.replace('.', '_')} ${isActive ? 'active' : ''}"
                            onclick="startAnnotation('${stage}')"
                            style="padding: 8px; font-size: 0.9em;"
                            ${isAnnotating && !isActive ? 'disabled' : ''}>
                        ${i + 1}. ${stage} ${existing ? '✓' : ''}
                    </button>
                `}).join('')}
            </div>

            ${state.timeline?.ground_truth?.length > 0 ? `
                <div style="margin-top: 15px; border-top: 1px solid #333; padding-top: 10px;">
                    <h3 style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 8px;">Current Annotations</h3>
                    <div style="display: flex; flex-direction: column; gap: 5px; max-height: 150px; overflow-y: auto;">
                        ${state.timeline.ground_truth.map(gt => {
                            const endStr = gt.end_timepoint ? gt.end_timepoint - 1 : '...';
                            return `
                            <div style="display: flex; align-items: center; gap: 8px; padding: 5px; background: var(--bg-tertiary); border-radius: 4px;">
                                <span class="stage-${gt.stage.replace('.', '_')}" style="padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">${gt.stage}</span>
                                <span style="flex: 1; font-size: 0.8em; color: var(--text-secondary);">#${gt.start_timepoint + 1}-${typeof endStr === 'number' ? endStr + 1 : endStr}</span>
                                <button onclick="deleteAnnotationByStage('${gt.stage}')" style="padding: 2px 8px; background: #e74c3c; border: none; border-radius: 3px; color: #fff; cursor: pointer; font-size: 0.75em;">×</button>
                            </div>
                        `}).join('')}
                    </div>
                </div>
            ` : ''}

            <div style="margin-top: 15px; border-top: 1px solid #333; padding-top: 15px;">
                <div style="display: flex; gap: 5px;">
                    <button class="stage-btn" style="background: #555; flex: 1;" onclick="prevImage()">← Prev</button>
                    <button class="stage-btn" style="background: #555; flex: 1;" onclick="nextImage()">Next →</button>
                </div>
            </div>
        </div>
    `;
}

function openProjections() {
    const url = `/projections/${state.selectedSession}/${state.selectedEmbryo}/${state.selectedIndex}`;
    window.open(url, '_blank');
}

// Navigation
async function prevImage() {
    if (!state.timeline || state.selectedIndex === null) return;
    if (state.selectedIndex > 0) {
        await selectIndex(state.selectedIndex - 1);
    }
}

async function nextImage() {
    if (!state.timeline || state.selectedIndex === null) return;
    if (state.selectedIndex < state.timeline.images.length - 1) {
        await selectIndex(state.selectedIndex + 1);
    }
}

// Keyboard navigation
document.addEventListener('keydown', async (e) => {
    // Check if we're in unified view
    const inUnifiedView = state.unifiedTimeline && state.selectedUnifiedIndex !== null;

    if (e.key === 'ArrowLeft') {
        if (inUnifiedView) {
            await prevUnifiedImage();
            // Update unified timeline preview if in annotation mode
            if (state.unifiedAnnotationMode === 'annotating') {
                updateUnifiedTimelinePreview();
            }
        } else {
            await prevImage();
            // Update timeline preview if in annotation mode (don't rebuild)
            if (state.annotationMode === 'annotating') {
                updateTimelinePreview();
            }
        }
    }
    if (e.key === 'ArrowRight') {
        if (inUnifiedView) {
            await nextUnifiedImage();
            // Update unified timeline preview if in annotation mode
            if (state.unifiedAnnotationMode === 'annotating') {
                updateUnifiedTimelinePreview();
            }
        } else {
            await nextImage();
            // Update timeline preview if in annotation mode (don't rebuild)
            if (state.annotationMode === 'annotating') {
                updateTimelinePreview();
            }
        }
    }
    if (e.key === 'Escape') {
        if (state.unifiedAnnotationMode === 'annotating') {
            cancelUnifiedAnnotation();
        } else if (state.annotationMode === 'annotating') {
            cancelAnnotation();
        }
    }
    if (e.key === 'Enter') {
        // Finish annotation with Enter key
        if (state.unifiedAnnotationMode === 'annotating') {
            finishUnifiedAnnotation();
        } else if (state.annotationMode === 'annotating') {
            finishAnnotation();
        }
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
        // Delete GT at current position (only if not in annotation mode)
        if (state.annotationMode !== 'annotating' && state.unifiedAnnotationMode !== 'annotating' && state.selectedIndex !== null) {
            e.preventDefault();
            deleteCurrentGT();
        }
    }
    // Number keys for quick stage annotation (only if not already annotating)
    if (e.key >= '1' && e.key <= '8' && state.annotationMode !== 'annotating' && state.unifiedAnnotationMode !== 'annotating') {
        const stageIdx = parseInt(e.key) - 1;
        if (stageIdx < STAGES.length) {
            if (inUnifiedView) {
                startUnifiedAnnotation(STAGES[stageIdx]);
            } else if (state.selectedIndex !== null) {
                startAnnotation(STAGES[stageIdx]);
            }
        }
    }
    // Ctrl+Z for undo
    if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        undo();
    }
});

// Server status indicator
async function checkServerStatus() {
    const indicator = document.getElementById('server-status');
    try {
        const start = Date.now();
        await api('stats');
        const latency = Date.now() - start;
        indicator.innerHTML = `<span style="color: #2ecc71;">● Online</span> <span style="color: #666; font-size: 0.8em;">(${latency}ms)</span>`;
        indicator.title = 'Server is responding';
    } catch (err) {
        indicator.innerHTML = '<span style="color: #e74c3c;">● Offline</span>';
        indicator.title = 'Cannot reach server: ' + err.message;
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    loadStats();
    loadSessions();
    checkServerStatus();

    // Initialize presence manager and connect WebSocket
    PresenceManager.init();
    connectWebSocket();

    // Attach event listeners
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    document.getElementById('tab-sessions').addEventListener('click', () => switchTab('sessions'));
    document.getElementById('tab-embryos').addEventListener('click', () => switchTab('embryos'));
    document.getElementById('filter-with-embryos').addEventListener('change', filterSessions);
    document.getElementById('filter-with-data').addEventListener('change', filterSessions);

    // Periodically check server status
    setInterval(checkServerStatus, 30000);
});
