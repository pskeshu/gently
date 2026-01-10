// Embryo Dataset Explorer JavaScript

// State
let state = {
    sessions: [],
    embryos: [],
    selectedSession: null,
    selectedEmbryo: null,
    timeline: null,
    selectedIndex: null,
    currentImage: null,
};

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

    container.innerHTML = `
        <h2>${t.embryo_id}</h2>

        <div class="info">
            ${t.images.length} images |
            GT: ${t.ground_truth.length > 0 ? t.ground_truth.map(g => g.stage).join(' -> ') : 'None'}
        </div>

        <div class="timeline" id="timeline">
            ${t.images.map((img, idx) => `
                <div class="timeline-item stage-${(img.ground_truth_stage || 'unknown').replace('.', '_')} ${state.selectedIndex === idx ? 'selected' : ''}"
                     onclick="selectIndex(${idx})"
                     title="#${idx}: ${img.ground_truth_stage || 'unlabeled'}">
                </div>
            `).join('')}
        </div>

        <div class="image-viewer" id="image-viewer">
            <div class="info">Click a timeline dot to view image</div>
        </div>
    `;
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

        // Look up prediction for this index if available
        if (state.timeline && state.timeline.predictions) {
            const pred = state.timeline.predictions.find(p => p.timepoint === idx);
            state.currentImage.prediction = pred || null;
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
    const viewer = document.getElementById('image-viewer');

    // Build prediction display if available
    let predictionHtml = '';
    if (pred) {
        const confidence = pred.confidence ? (pred.confidence * 100).toFixed(0) + '%' : 'N/A';
        const isCorrect = pred.predicted_stage === img.ground_truth_stage;
        const predColor = isCorrect ? '#2ecc71' : '#e74c3c';
        predictionHtml = `
            <div style="margin-top: 15px; padding: 10px; background: #1a1a2e; border-radius: 8px;">
                <h3 style="color: #00d4ff; margin-bottom: 8px;">Prediction</h3>
                <div style="display: flex; gap: 15px; margin-bottom: 10px;">
                    <div>Stage: <span style="color: ${predColor}; font-weight: bold;">${pred.predicted_stage}</span></div>
                    <div>Confidence: <span style="color: #f1c40f;">${confidence}</span></div>
                    ${img.ground_truth_stage ? `<div>${isCorrect ? '✓ Correct' : '✗ Wrong'}</div>` : ''}
                </div>
                ${pred.reasoning ? `
                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; color: #888;">View Reasoning</summary>
                        <div style="margin-top: 8px; padding: 10px; background: #0d1117; border-radius: 5px; font-size: 0.9em; white-space: pre-wrap; max-height: 200px; overflow-y: auto;">
                            ${pred.reasoning}
                        </div>
                    </details>
                ` : ''}
            </div>
        `;
    }

    viewer.innerHTML = `
        <div class="image-container">
            <div style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>Image #${idx}</strong>
                    ${img.ground_truth_stage ? `<span class="current-gt"> - GT: ${img.ground_truth_stage}</span>` : ''}
                    <span style="color: #666; font-size: 0.9em;"> ${img.timestamp || ''}</span>
                </div>
                <button onclick="openProjections()" style="padding: 5px 12px; background: #3498db; border: none; border-radius: 5px; color: #fff; cursor: pointer; font-size: 0.9em;">
                    View More ->
                </button>
            </div>
            ${img.image_b64 ? `<img src="data:image/jpeg;base64,${img.image_b64}" alt="Image ${idx}">` : '<div class="info">Image not available</div>'}
            ${predictionHtml}
        </div>

        <div class="annotation-panel">
            <h2>Set Stage Transition</h2>
            <p class="info">Click to mark when this stage STARTS at index ${idx}</p>
            ${STAGES.map(stage => `
                <button class="stage-btn stage-${stage.replace('.', '_')} ${img.ground_truth_stage === stage ? 'active' : ''}"
                        onclick="setGroundTruth('${stage}', ${idx})">
                    ${stage}
                </button>
            `).join('')}

            <div style="margin-top: 20px;">
                <h2>Navigation</h2>
                <button class="stage-btn" style="background: #555;" onclick="prevImage()">← Previous</button>
                <button class="stage-btn" style="background: #555;" onclick="nextImage()">Next →</button>
            </div>
        </div>
    `;
}

function openProjections() {
    const url = `/projections/${state.selectedSession}/${state.selectedEmbryo}/${state.selectedIndex}`;
    window.open(url, '_blank');
}

// Set ground truth
async function setGroundTruth(stage, idx) {
    await apiPost('ground_truth', {
        session_id: state.selectedSession,
        embryo_id: state.selectedEmbryo,
        stage: stage,
        start_timepoint: idx,  // Using index as timepoint
        annotator: 'web_explorer',
    });

    // Reload timeline
    await selectEmbryo(state.selectedEmbryo);
    await selectIndex(idx);

    // Refresh stats
    loadStats();
}

// Navigation
function prevImage() {
    if (!state.timeline || state.selectedIndex === null) return;
    if (state.selectedIndex > 0) {
        selectIndex(state.selectedIndex - 1);
    }
}

function nextImage() {
    if (!state.timeline || state.selectedIndex === null) return;
    if (state.selectedIndex < state.timeline.images.length - 1) {
        selectIndex(state.selectedIndex + 1);
    }
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prevImage();
    if (e.key === 'ArrowRight') nextImage();
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
    loadStats();
    loadSessions();
    checkServerStatus();
    // Periodically check server status
    setInterval(checkServerStatus, 30000);
});
