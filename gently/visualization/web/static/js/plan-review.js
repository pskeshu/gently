/**
 * Plan Review — three-panel plan document viewer.
 *
 * Fetches /api/campaigns/{id}/document, renders outline + document + detail.
 * Supports version viewing, scroll-spy, print, and responsive layout.
 */

const TYPE_ICONS = {
    imaging: '\u{1F4F7}', bench: '\u{1F9EA}', genetics: '\u{1F9EC}',
    analysis: '\u{1F4CA}', decision_point: '\u{1F6A6}',
};
const STATUS_DOTS = {
    planned: '\u25CB', in_progress: '\u25D1', completed: '\u25CF',
    skipped: '\u2298', blocked: '\u2297',
};
const STATUS_LABELS = {
    active: 'Active', completed: 'Completed', paused: 'Paused',
    planned: 'Planned', in_progress: 'In Progress',
    skipped: 'Skipped', blocked: 'Blocked',
};

// ── State ────────────────────────────────────────────────
let campaignId = window.CAMPAIGN_ID;
let docData = null;       // full document response
let versions = [];        // snapshot list
let allItemsFlat = {};    // id → item (for quick lookup)
let selectedItemId = null;
let viewingSnapshotId = null;

// ── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadDocument();
    loadVersions();

    document.getElementById('theme-toggle')?.addEventListener('click', () => {
        const next = document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        document.body.setAttribute('data-theme', next);
        localStorage.setItem('gently-theme', next);
    });

    document.getElementById('print-btn')?.addEventListener('click', () => window.print());
    document.getElementById('detail-close')?.addEventListener('click', closeDetail);
    document.getElementById('version-btn')?.addEventListener('click', toggleVersionDropdown);
    document.getElementById('snapshot-banner-close')?.addEventListener('click', backToCurrent);

    // Hamburger for responsive outline
    document.getElementById('outline-hamburger')?.addEventListener('click', () => {
        document.getElementById('plan-outline')?.classList.toggle('open');
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeDetail();
            document.getElementById('version-dropdown')?.classList.add('hidden');
            document.getElementById('plan-outline')?.classList.remove('open');
        }
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        const wrap = document.getElementById('version-dropdown-wrap');
        if (wrap && !wrap.contains(e.target)) {
            document.getElementById('version-dropdown')?.classList.add('hidden');
        }
    });

    // Scroll-spy: track which phase is visible
    const doc = document.getElementById('plan-document');
    if (doc) {
        doc.addEventListener('scroll', onDocScroll, { passive: true });
    }
});

// ── Data loading ─────────────────────────────────────────
async function loadDocument() {
    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/document`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        docData = await res.json();
        buildItemIndex(docData.document);
        renderAll();
    } catch (err) {
        console.error('Failed to load document:', err);
        document.getElementById('doc-loading').innerHTML =
            '<p style="color:var(--accent-orange)">Failed to load plan document.</p>';
    }
}

async function loadVersions() {
    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/versions`);
        if (!res.ok) return;
        const data = await res.json();
        versions = data.versions || [];
    } catch (err) {
        console.error('Failed to load versions:', err);
    }
}

function buildItemIndex(node) {
    for (const item of (node.items || [])) {
        allItemsFlat[item.id] = item;
    }
    for (const child of (node.children || [])) {
        buildItemIndex(child);
    }
}

// ── Render all panels ────────────────────────────────────
function renderAll() {
    if (!docData) return;
    document.getElementById('doc-loading')?.classList.add('hidden');
    const tree = docData.document;
    const campaign = tree.campaign;

    // Header title
    const titleEl = document.getElementById('plan-title');
    if (titleEl) titleEl.textContent = campaign.description || 'Plan Review';

    renderOutline(tree);
    renderDocument(tree);
    renderFooter();
}

// ── Outline panel ────────────────────────────────────────
function renderOutline(tree) {
    const container = document.getElementById('outline-content');
    if (!container) return;

    const campaign = tree.campaign;
    let html = '';

    html += `<div class="outline-campaign-title" onclick="scrollToElement('plan-title-card')">${esc(campaign.shorthand || campaign.description)}</div>`;

    // Phases (children)
    const children = tree.children || [];
    children.forEach((child, idx) => {
        const phaseNum = idx + 1;
        const phaseName = child.campaign.shorthand || child.campaign.description;
        const items = child.items || [];

        html += `<div class="outline-phase">
            <div class="outline-phase-header" onclick="toggleOutlinePhase(this)">
                <svg class="outline-phase-chevron open" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="9,18 15,12 9,6"></polyline>
                </svg>
                <span class="outline-phase-name">${esc(phaseName)}</span>
            </div>
            <div class="outline-items">`;

        items.forEach((item, iIdx) => {
            const taskNum = `${phaseNum}.${iIdx + 1}`;
            const dot = STATUS_DOTS[item.status] || '\u25CB';
            const dotClass = item.status || 'planned';
            html += `<div class="outline-item" data-item-id="${item.id}" onclick="navigateToItem('${item.id}')">
                <span class="outline-item-dot dot-${dotClass}">${dot}</span>
                <span class="outline-item-num">${esc(taskNum)}</span>
                <span class="outline-item-title">${esc(item.title)}</span>
            </div>`;
        });

        html += `</div></div>`;
    });

    // Items directly on root (no phases)
    if (children.length === 0 && (tree.items || []).length > 0) {
        tree.items.forEach((item, idx) => {
            const dot = STATUS_DOTS[item.status] || '\u25CB';
            const dotClass = item.status || 'planned';
            html += `<div class="outline-item" data-item-id="${item.id}" onclick="navigateToItem('${item.id}')">
                <span class="outline-item-dot dot-${dotClass}">${dot}</span>
                <span class="outline-item-num">${idx + 1}</span>
                <span class="outline-item-title">${esc(item.title)}</span>
            </div>`;
        });
    }

    // Bottom links
    const refCount = (docData.bibliography || []).length;
    html += `<div class="outline-divider"></div>`;
    html += `<div class="outline-link" onclick="scrollToElement('bibliography-section')">
        Refs <span class="outline-link-count">(${refCount})</span>
    </div>`;
    html += `<div class="outline-link" onclick="scrollToElement('versions-section')">
        History <span class="outline-link-count">(${versions.length})</span>
    </div>`;

    container.innerHTML = html;
}

// ── Center document ──────────────────────────────────────
function renderDocument(tree) {
    const container = document.getElementById('doc-content');
    if (!container) return;

    const campaign = tree.campaign;
    const status = docData.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const inProgress = status.in_progress || 0;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

    let html = '';

    // Title card
    html += `<div class="plan-title-card" id="plan-title-card">
        <div class="plan-title-card-header">
            ${campaign.shorthand ? `<span class="plan-title-card-shorthand">${esc(campaign.shorthand)}</span>` : ''}
            <span class="plan-title-card-name">${esc(campaign.description)}</span>
            <span class="plan-title-card-status">
                <span class="status-badge status-${campaign.status}">${STATUS_LABELS[campaign.status] || campaign.status}</span>
            </span>
        </div>
        ${campaign.target ? `<div class="plan-title-card-target">${esc(campaign.target)}</div>` : ''}
        <div class="plan-title-card-stats">
            <span>${total} items</span>
            <span>${completed} done</span>
            <span>${inProgress} active</span>
        </div>
        ${total > 0 ? `
        <div class="plan-progress-bar">
            <div class="plan-progress-fill" style="width: ${pct}%"></div>
        </div>
        <div class="plan-progress-label">${pct}%</div>` : ''}
    </div>`;

    // Phase sections
    const children = tree.children || [];
    if (children.length > 0) {
        children.forEach((child, idx) => {
            html += renderPhaseBlock(child, idx + 1);
        });
    } else if ((tree.items || []).length > 0) {
        // Items directly on root
        html += `<div class="phase-block">`;
        tree.items.forEach((item, idx) => {
            html += renderDocItem(item, String(idx + 1));
        });
        html += `</div>`;
    }

    // Bibliography
    html += renderBibliography();

    // Version history
    html += renderVersionHistory();

    container.innerHTML = html;
}

function renderPhaseBlock(child, phaseNum) {
    const campaign = child.campaign;
    const items = child.items || [];
    const status = child.status || {};
    const total = status.total || 0;

    let html = `<div class="phase-block" id="phase-${phaseNum}" data-phase="${phaseNum}">
        <div class="phase-block-header">
            <span class="phase-block-num">Phase ${phaseNum}</span>
            <span class="phase-block-name">${esc(campaign.shorthand || campaign.description)}</span>
            <span class="phase-block-count">${total} item${total !== 1 ? 's' : ''}</span>
        </div>`;

    items.forEach((item, idx) => {
        html += renderDocItem(item, `${phaseNum}.${idx + 1}`);
    });

    html += `</div>`;
    return html;
}

function renderDocItem(item, taskNum) {
    const icon = TYPE_ICONS[item.type] || '\u{1F4CB}';
    const statusClass = item.status || 'planned';
    const dot = STATUS_DOTS[statusClass] || '\u25CB';

    // One-line spec
    let specLine = '';
    if (item.imaging_spec) {
        const s = item.imaging_spec;
        const parts = [];
        if (s.strain) parts.push(s.strain);
        if (s.num_embryos) parts.push(`${s.num_embryos} embryos`);
        if (s.interval_s) parts.push(`${s.interval_s}s interval`);
        specLine = parts.join(' \u00B7 ');
    } else if (item.bench_spec?.protocol) {
        specLine = item.bench_spec.protocol;
    } else if (item.description) {
        specLine = item.description.slice(0, 100);
    }

    // Reference badges
    let refBadges = '';
    if (item.ref_numbers && item.ref_numbers.length > 0) {
        refBadges = item.ref_numbers.map(n =>
            `<a class="ref-badge" onclick="event.stopPropagation(); scrollToRef(${n})">[${n}]</a>`
        ).join('');
    }

    // Dependency links
    let depLinks = '';
    if (item.dependencies && item.dependencies.length > 0) {
        const depStr = item.dependencies.map(d =>
            `<a onclick="event.stopPropagation(); navigateToItem('${d.id}')">${esc(d.title)}</a>`
        ).join(', ');
        depLinks += `<span class="doc-dep-link">\u2190 needs: ${depStr}</span>`;
    }
    if (item.dependents && item.dependents.length > 0) {
        const dntStr = item.dependents.map(d =>
            `<a onclick="event.stopPropagation(); navigateToItem('${d.id}')">${esc(d.title)}</a>`
        ).join(', ');
        depLinks += `<span class="doc-dep-link">\u2192 blocks: ${dntStr}</span>`;
    }

    const footer = (refBadges || depLinks)
        ? `<div class="doc-item-footer">${refBadges}${depLinks}</div>`
        : '';

    return `<div class="doc-item" id="item-${item.id}" data-item-id="${item.id}" onclick="selectItem('${item.id}')">
        <span class="doc-item-status dot-${statusClass}">${dot}</span>
        <span class="doc-item-icon type-${item.type}">${icon}</span>
        <div class="doc-item-body">
            <div class="doc-item-title-row">
                <span class="doc-item-num">${esc(taskNum)}</span>
                <span class="doc-item-title">${esc(item.title)}</span>
            </div>
            ${specLine ? `<div class="doc-item-spec">${esc(specLine)}</div>` : ''}
            ${footer}
        </div>
    </div>`;
}

function renderBibliography() {
    const bib = docData.bibliography || [];
    let html = `<div class="bibliography-section" id="bibliography-section">
        <div class="bibliography-title">References</div>`;

    if (bib.length === 0) {
        html += `<div class="no-versions">No references cited.</div>`;
    } else {
        bib.forEach(ref => {
            const num = ref.number;
            let text = '';
            let link = '';
            const source = (ref.source || '').toLowerCase();

            if (source === 'pubmed' || ref.pmid) {
                const pmid = ref.pmid || ref.key || ref.id || '';
                text = ref.title || ref.citation || `PMID:${pmid}`;
                if (pmid) link = `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
            } else if (source === 'wormbase') {
                text = `WormBase: ${ref.key || ref.id || ref.title || ''}`;
                if (ref.key) link = `https://wormbase.org/search/all/${ref.key}`;
            } else if (source === 'cgc') {
                text = `CGC: ${ref.key || ref.strain || ref.title || ''}`;
            } else if (source === 'claude') {
                text = ref.note || ref.title || ref.key || 'Agent note';
            } else {
                text = ref.title || ref.citation || ref.key || ref.note || JSON.stringify(ref);
            }

            const sourceTag = ref.source ? ` <span class="bib-source">${esc(ref.source)}</span>` : '';
            const linkHtml = link
                ? `<a href="${esc(link)}" target="_blank" rel="noopener">${esc(text)}</a>`
                : esc(text);

            html += `<div class="bib-entry" id="ref-${num}">
                <span class="bib-num">[${num}]</span>
                <span class="bib-text">${linkHtml}${sourceTag}</span>
            </div>`;
        });
    }

    html += `</div>`;
    return html;
}

function renderVersionHistory() {
    let html = `<div class="versions-section" id="versions-section">
        <div class="versions-title">Version History</div>`;

    if (versions.length === 0) {
        html += `<div class="no-versions">No snapshots yet.</div>`;
    } else {
        // Show newest first
        const sorted = [...versions].sort((a, b) => (b.version_number || 0) - (a.version_number || 0));
        sorted.forEach((v, idx) => {
            const isCurrent = idx === 0;
            const label = v.label || v.summary || 'Snapshot';
            const date = v.created_at ? formatDate(v.created_at) : '';
            html += `<div class="version-entry ${isCurrent ? 'current' : ''}" onclick="viewVersion('${v.version_id}', ${isCurrent})">
                <span class="version-entry-num">v${v.version_number || '?'}</span>
                <span class="version-entry-label">${esc(label)}</span>
                ${isCurrent ? '<span class="version-entry-current">Current</span>' : ''}
                <span class="version-entry-date">${esc(date)}</span>
            </div>`;
        });
    }

    html += `</div>`;
    return html;
}

// ── Detail panel ─────────────────────────────────────────
async function selectItem(itemId) {
    selectedItemId = itemId;

    // Highlight in document
    document.querySelectorAll('.doc-item.selected').forEach(el => el.classList.remove('selected'));
    const docEl = document.getElementById(`item-${itemId}`);
    if (docEl) docEl.classList.add('selected');

    // Highlight in outline
    document.querySelectorAll('.outline-item.active').forEach(el => el.classList.remove('active'));
    const outlineEl = document.querySelector(`.outline-item[data-item-id="${itemId}"]`);
    if (outlineEl) outlineEl.classList.add('active');

    // Open detail panel
    document.getElementById('plan-review')?.classList.add('detail-open');

    // Fetch enriched data
    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(itemId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderDetail(data);
    } catch (err) {
        console.error('Failed to load item detail:', err);
        const body = document.getElementById('detail-body');
        if (body) body.innerHTML = '<p style="color:var(--accent-orange)">Failed to load item details.</p>';
    }
}

function renderDetail(data) {
    const item = data.item;
    const deps = data.dependencies || [];
    const dnts = data.dependents || [];
    const sessions = data.sessions || [];

    const titleEl = document.getElementById('detail-title');
    const statusEl = document.getElementById('detail-status');
    const body = document.getElementById('detail-body');

    if (titleEl) titleEl.textContent = item.title;
    if (statusEl) {
        const sc = item.status === 'completed' ? 'completed' : item.status === 'in_progress' ? 'active' : 'planned';
        statusEl.className = `detail-status-badge status-${sc}`;
        statusEl.textContent = STATUS_LABELS[item.status] || item.status;
    }

    let html = '';

    // Type + ID
    const icon = TYPE_ICONS[item.type] || '\u{1F4CB}';
    html += `<div class="detail-meta">
        <span class="detail-type">${icon} ${item.type.replace('_', ' ')}</span>
        <span class="detail-id">${item.id}</span>
    </div>`;

    // Description
    if (item.description) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Description</div>
            <div class="detail-section-content">${esc(item.description)}</div>
        </div>`;
    }

    // Outcome
    if (item.outcome) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Outcome</div>
            <div class="detail-section-content">${esc(item.outcome)}</div>
        </div>`;
    }

    // Imaging spec
    if (item.imaging_spec) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Imaging Specification</div>
            <table class="spec-table">${renderSpecTable(item.imaging_spec)}</table>
        </div>`;
    }

    // Bench spec
    if (item.bench_spec) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Bench Specification</div>
            <table class="spec-table">${renderSpecTable(item.bench_spec)}</table>
        </div>`;
    }

    // Dependencies
    if (deps.length > 0) {
        const chips = deps.map(d => {
            const dot = STATUS_DOTS[d.status] || '\u25CB';
            const dotClass = d.status || 'planned';
            return `<span class="dep-chip" onclick="navigateToItem('${d.id}')">
                <span class="dep-chip-dot dot-${dotClass}">${dot}</span>
                ${esc(d.title)}
            </span>`;
        }).join('');
        html += `<div class="detail-section">
            <div class="detail-section-title">Depends on</div>
            <div class="dep-list">${chips}</div>
        </div>`;
    }

    // Dependents
    if (dnts.length > 0) {
        const chips = dnts.map(d => {
            const dot = STATUS_DOTS[d.status] || '\u25CB';
            const dotClass = d.status || 'planned';
            return `<span class="dep-chip" onclick="navigateToItem('${d.id}')">
                <span class="dep-chip-dot dot-${dotClass}">${dot}</span>
                ${esc(d.title)}
            </span>`;
        }).join('');
        html += `<div class="detail-section">
            <div class="detail-section-title">Blocks</div>
            <div class="dep-list">${chips}</div>
        </div>`;
    }

    // References
    if (item.references && item.references.length > 0) {
        let refHtml = '';
        item.references.forEach((ref, idx) => {
            const source = (ref.source || '').toLowerCase();
            let text = ref.title || ref.citation || ref.key || ref.note || '';
            let link = '';
            if (source === 'pubmed' || ref.pmid) {
                const pmid = ref.pmid || ref.key || '';
                if (pmid) link = `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
            } else if (source === 'wormbase' && ref.key) {
                link = `https://wormbase.org/search/all/${ref.key}`;
            }
            const linkHtml = link ? `<a href="${esc(link)}" target="_blank" rel="noopener">${esc(text)}</a>` : esc(text);
            refHtml += `<div class="detail-ref">
                <span class="detail-ref-num">[${idx + 1}]</span>
                ${linkHtml}
            </div>`;
        });
        html += `<div class="detail-section">
            <div class="detail-section-title">References</div>
            ${refHtml}
        </div>`;
    }

    // Sessions
    if (sessions.length > 0) {
        let sessHtml = '';
        sessions.forEach(s => {
            sessHtml += `<div class="detail-session">
                <div class="detail-session-title">${esc(s.planned_intent || s.id)}</div>
                <div class="detail-session-date">${s.created_at ? formatDate(s.created_at) : ''}</div>
            </div>`;
        });
        html += `<div class="detail-section">
            <div class="detail-section-title">Sessions</div>
            ${sessHtml}
        </div>`;
    } else {
        html += `<div class="detail-section">
            <div class="detail-section-title">Sessions</div>
            <div class="detail-section-content" style="color:var(--text-muted);font-style:italic">No linked sessions</div>
        </div>`;
    }

    if (body) body.innerHTML = html;
}

function renderSpecTable(spec) {
    const LABELS = {
        strain: 'Strain', genotype: 'Genotype', reporter: 'Reporter',
        sample_prep: 'Sample Prep', temperature_c: 'Temperature',
        num_embryos: 'Embryos', num_slices: 'Z Slices', exposure_ms: 'Exposure',
        laser_wavelength_nm: 'Laser', laser_power_pct: 'Power', interval_s: 'Interval',
        target_window: 'Dev. Window', start_stage: 'Start Stage',
        stop_condition: 'Stop Condition', estimated_duration_h: 'Duration',
        success_criteria: 'Success Criteria', comparison_to: 'Compare To',
        protocol: 'Protocol', reagents: 'Reagents', strains: 'Strains',
        target_genotype: 'Target Genotype', estimated_days: 'Est. Days', notes: 'Notes',
    };
    const UNITS = {
        temperature_c: '\u00B0C', exposure_ms: ' ms', laser_wavelength_nm: ' nm',
        laser_power_pct: '%', interval_s: 's', estimated_duration_h: ' hrs',
        estimated_days: ' days',
    };

    let rows = '';
    for (const [key, value] of Object.entries(spec)) {
        if (value == null || key.startsWith('_')) continue;
        const label = LABELS[key] || key;
        let display = Array.isArray(value) ? value.join(', ') : String(value);
        if (UNITS[key]) display += UNITS[key];
        rows += `<tr><td>${esc(label)}</td><td>${esc(display)}</td></tr>`;
    }
    return rows;
}

// ── Footer ───────────────────────────────────────────────
function renderFooter() {
    const status = docData?.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    const phases = (docData?.document?.children || []).length;

    const statsEl = document.getElementById('footer-stats');
    if (statsEl) {
        statsEl.textContent = `Plan: ${total} items \u00B7 ${pct}% complete \u00B7 ${phases} phase${phases !== 1 ? 's' : ''}`;
    }

    const snapEl = document.getElementById('footer-snapshot');
    if (snapEl && versions.length > 0) {
        const latest = versions.reduce((a, b) => (b.version_number || 0) > (a.version_number || 0) ? b : a, versions[0]);
        const date = latest.created_at ? formatDate(latest.created_at) : '';
        snapEl.textContent = `Last snapshot: v${latest.version_number || '?'} ${date}`;
    }
}

// ── Version viewing ──────────────────────────────────────
function toggleVersionDropdown() {
    const dd = document.getElementById('version-dropdown');
    if (!dd) return;

    if (dd.classList.contains('hidden')) {
        // Populate
        let html = `<div class="version-item ${!viewingSnapshotId ? 'active' : ''}" onclick="backToCurrent()">
            <span class="version-num">Current</span>
            <div class="version-info"><div class="version-label-text">Live data</div></div>
        </div>`;

        const sorted = [...versions].sort((a, b) => (b.version_number || 0) - (a.version_number || 0));
        sorted.forEach(v => {
            const active = viewingSnapshotId === v.version_id;
            const label = v.label || v.summary || 'Snapshot';
            const date = v.created_at ? formatDate(v.created_at) : '';
            html += `<div class="version-item ${active ? 'active' : ''}" onclick="viewVersion('${v.version_id}', false)">
                <span class="version-num">v${v.version_number || '?'}</span>
                <div class="version-info">
                    <div class="version-label-text">${esc(label)}</div>
                    <div class="version-date">${esc(date)}</div>
                </div>
            </div>`;
        });

        dd.innerHTML = html;
        dd.classList.remove('hidden');
    } else {
        dd.classList.add('hidden');
    }
}

async function viewVersion(versionId, isCurrent) {
    document.getElementById('version-dropdown')?.classList.add('hidden');

    if (isCurrent) {
        backToCurrent();
        return;
    }

    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/versions/${encodeURIComponent(versionId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const snapshot = data.version;

        if (snapshot && snapshot.snapshot_json) {
            viewingSnapshotId = versionId;
            // Replace document data with snapshot
            const snapshotTree = snapshot.snapshot_json;
            // Use snapshot tree for rendering (it has the same structure)
            docData.document = snapshotTree;
            allItemsFlat = {};
            buildItemIndex(snapshotTree);
            renderAll();

            // Show banner
            const banner = document.getElementById('snapshot-banner');
            const bannerText = document.getElementById('snapshot-banner-text');
            if (banner) banner.classList.remove('hidden');
            if (bannerText) bannerText.textContent = `Viewing v${snapshot.version_number || '?'}: ${snapshot.label || 'Snapshot'}`;

            // Update version button label
            const label = document.getElementById('version-label');
            if (label) label.textContent = `v${snapshot.version_number || '?'}`;
        }
    } catch (err) {
        console.error('Failed to load version:', err);
    }
}

function backToCurrent() {
    document.getElementById('version-dropdown')?.classList.add('hidden');
    viewingSnapshotId = null;

    // Reload current data
    const banner = document.getElementById('snapshot-banner');
    if (banner) banner.classList.add('hidden');
    const label = document.getElementById('version-label');
    if (label) label.textContent = 'Current';

    loadDocument();
}

// ── Navigation helpers ───────────────────────────────────
function navigateToItem(itemId) {
    // Scroll center document to item
    const el = document.getElementById(`item-${itemId}`);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    // Select it (opens detail)
    selectItem(itemId);
    // Close outline on mobile
    document.getElementById('plan-outline')?.classList.remove('open');
}

function scrollToElement(id) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    document.getElementById('plan-outline')?.classList.remove('open');
}

function scrollToRef(num) {
    const el = document.getElementById(`ref-${num}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function closeDetail() {
    selectedItemId = null;
    document.getElementById('plan-review')?.classList.remove('detail-open');
    document.querySelectorAll('.doc-item.selected').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.outline-item.active').forEach(el => el.classList.remove('active'));
}

function toggleOutlinePhase(el) {
    const chevron = el.querySelector('.outline-phase-chevron');
    const items = el.nextElementSibling;
    if (items) items.classList.toggle('collapsed');
    if (chevron) chevron.classList.toggle('open');
}

// ── Scroll-spy ───────────────────────────────────────────
function onDocScroll() {
    const doc = document.getElementById('plan-document');
    if (!doc) return;
    const scrollTop = doc.scrollTop;
    const offset = 100;

    // Find the visible item closest to top
    let closestItem = null;
    let closestDist = Infinity;
    document.querySelectorAll('.doc-item').forEach(el => {
        const rect = el.getBoundingClientRect();
        const docRect = doc.getBoundingClientRect();
        const relTop = rect.top - docRect.top;
        if (relTop < offset && Math.abs(relTop) < closestDist) {
            closestDist = Math.abs(relTop);
            closestItem = el;
        }
    });

    if (closestItem && !selectedItemId) {
        const itemId = closestItem.dataset.itemId;
        document.querySelectorAll('.outline-item.active').forEach(el => el.classList.remove('active'));
        const outlineEl = document.querySelector(`.outline-item[data-item-id="${itemId}"]`);
        if (outlineEl) outlineEl.classList.add('active');
    }
}

// ── Utilities ────────────────────────────────────────────
function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function formatDate(isoStr) {
    try {
        const d = new Date(isoStr);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
            ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
        return isoStr;
    }
}
