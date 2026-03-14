/**
 * Campaigns — Unified workspace for campaign browsing and plan review.
 *
 * Views:
 *   dashboard — all campaigns, navigator shows campaign list, canvas shows cards
 *   plan      — single campaign, navigator shows outline, canvas shows plan doc
 */

// Wrapped in IIFE to avoid global state collision with app.js
(function() {
'use strict';

// ── Constants ────────────────────────────────────────────
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
const SPEC_LABELS = {
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
const SPEC_UNITS = {
    temperature_c: '\u00B0C', exposure_ms: ' ms', laser_wavelength_nm: ' nm',
    laser_power_pct: '%', interval_s: 's', estimated_duration_h: ' hrs',
    estimated_days: ' days',
};

// ── State ────────────────────────────────────────────────
const state = {
    planView: 'doc',            // 'doc' | 'graph' | 'board' | 'decide' | 'matrix'
    activeCampaignId: null,
    selectedItemId: null,
    allCampaigns: [],           // full tree list from /api/campaigns
    docData: null,              // plan document data for plan view
    versions: [],               // snapshots list
    viewingSnapshotId: null,
    allItemsFlat: {},           // id → item for quick lookup
};

// ── DOM refs (cached on init) ────────────────────────────
let $workspace, $navContent, $canvasContent, $canvasLoading;
let $inspectorTitle, $inspectorStatus, $inspectorBody;
let $headerTitle, $headerBreadcrumb;
let $versionWrap, $versionBtn, $versionLabel, $versionDropdown;
let $printBtn, $snapshotBanner, $snapshotBannerText;
let $statusLeft, $statusRight;

// ── Init ─────────────────────────────────────────────────
let _initialized = false;
function boot() {
    if (_initialized) return;
    _initialized = true;
    // Cache DOM
    $workspace      = document.getElementById('workspace');
    $navContent     = document.getElementById('nav-content');
    $canvasContent  = document.getElementById('canvas-content');
    $canvasLoading  = document.getElementById('canvas-loading');
    $inspectorTitle  = document.getElementById('inspector-title');
    $inspectorStatus = document.getElementById('inspector-status');
    $inspectorBody   = document.getElementById('inspector-body');
    $headerTitle     = document.getElementById('header-title');
    $headerBreadcrumb = document.getElementById('header-breadcrumb');
    $versionWrap    = document.getElementById('version-wrap');
    $versionBtn     = document.getElementById('version-btn');
    $versionLabel   = document.getElementById('version-label');
    $versionDropdown = document.getElementById('version-dropdown');
    $printBtn       = document.getElementById('print-btn');
    $snapshotBanner = document.getElementById('snapshot-banner');
    $snapshotBannerText = document.getElementById('snapshot-banner-text');
    $statusLeft     = document.getElementById('status-left');
    $statusRight    = document.getElementById('status-right');

    // Event listeners (theme toggle handled by _header.html)
    document.getElementById('inspector-close')?.addEventListener('click', closeInspector);
    $printBtn?.addEventListener('click', () => window.print());
    $versionBtn?.addEventListener('click', toggleVersionDropdown);
    document.getElementById('snapshot-close')?.addEventListener('click', backToCurrent);

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            closeInspector();
            $versionDropdown?.classList.add('hidden');
        }
    });

    // Close dropdown on outside click
    document.addEventListener('click', e => {
        if ($versionWrap && !$versionWrap.contains(e.target)) {
            $versionDropdown?.classList.add('hidden');
        }
    });

    // Scroll-spy for plan view
    document.getElementById('canvas')?.addEventListener('scroll', onCanvasScroll, { passive: true });

    // Plan view switcher
    setupPlanViewSwitcher();

    // Load campaigns (will auto-select first or the specified one)
    const initialId = window.INITIAL_CAMPAIGN_ID;
    loadCampaigns().then(() => {
        if (initialId) openCampaign(initialId);
    });
}


// ══════════════════════════════════════════════════════════
//  DATA LOADING
// ══════════════════════════════════════════════════════════

async function loadCampaigns() {
    showLoading('Loading campaigns...');
    try {
        const res = await fetch('/api/campaigns');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        state.allCampaigns = data.campaigns || [];

        // Build flat item index across all campaigns
        state.allItemsFlat = {};
        function indexItems(tree) {
            for (const item of (tree.items || [])) {
                state.allItemsFlat[item.id] = item;
                item._rootCampaignId = tree.campaign.id;
            }
            for (const child of (tree.children || [])) indexItems(child);
        }
        state.allCampaigns.forEach(indexItems);

        hideLoading();
        renderNavigator();

        // Auto-select first campaign if none active
        if (!state.activeCampaignId && state.allCampaigns.length > 0) {
            openCampaign(state.allCampaigns[0].campaign.id);
        } else if (state.activeCampaignId) {
            renderAll();
        } else {
            $canvasContent.innerHTML = `<div class="empty-state">
                <p>No campaigns yet</p>
                <span class="empty-hint">Create a campaign plan to get started</span>
            </div>`;
            updateHeader();
            updateStatusbar();
        }
    } catch (err) {
        console.error('Failed to load campaigns:', err);
        hideLoading();
        $canvasContent.innerHTML = `<div class="empty-state">
            <p>Failed to load campaigns</p>
            <span class="empty-hint">${esc(err.message)}</span>
        </div>`;
    }
}

async function loadDocument(campaignId) {
    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/document`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.docData = await res.json();
        state.allItemsFlat = {};
        buildItemIndex(state.docData.document);
    } catch (err) {
        console.error('Failed to load document:', err);
        state.docData = null;
    }
}

async function loadVersions(campaignId) {
    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/versions`);
        if (!res.ok) return;
        const data = await res.json();
        state.versions = data.versions || [];
    } catch (err) {
        console.error('Failed to load versions:', err);
        state.versions = [];
    }
}

function buildItemIndex(node) {
    for (const item of (node.items || [])) {
        state.allItemsFlat[item.id] = item;
    }
    for (const child of (node.children || [])) {
        buildItemIndex(child);
    }
}

// ══════════════════════════════════════════════════════════
//  VIEW SWITCHING
// ══════════════════════════════════════════════════════════

async function openCampaign(campaignId) {
    if (state.activeCampaignId === campaignId && state.docData) {
        // Already loaded — just update sidebar highlight
        renderNavigator();
        return;
    }
    state.activeCampaignId = campaignId;
    state.selectedItemId = null;
    state.viewingSnapshotId = null;
    state.planView = 'doc';
    closeInspector();
    showLoading('Loading plan...');

    await Promise.all([
        loadDocument(campaignId),
        loadVersions(campaignId),
    ]);

    // If we don't have campaigns list yet (direct URL), load in background
    if (state.allCampaigns.length === 0) {
        loadCampaigns().catch(() => {});
    }

    hideLoading();

    if (!state.docData) {
        $canvasContent.innerHTML = `<div class="empty-state">
            <p>Failed to load plan document</p>
            <span class="empty-hint">Campaign may not exist</span>
        </div>`;
        updateHeader();
        return;
    }

    renderAll();
}

// Handle browser back/forward
window.addEventListener('popstate', e => {
    const s = e.state;
    if (s && s.campaignId) {
        openCampaign(s.campaignId);
    } else {
        goToDashboard();
    }
});

// ══════════════════════════════════════════════════════════
//  RENDER ALL
// ══════════════════════════════════════════════════════════

function renderAll() {
    renderNavigator();
    renderCanvas();
    updateHeader();
    updateStatusbar();
}

// ══════════════════════════════════════════════════════════
//  NAVIGATOR
// ══════════════════════════════════════════════════════════

function renderNavigator() {
    if (!$navContent) return;
    renderNavDashboard();
}

function renderNavDashboard() {
    let html = '<div class="nav-section-title">Campaigns</div>';

    if (state.allCampaigns.length === 0) {
        html += '<div style="padding:16px;color:var(--text-muted);font-size:0.78rem;font-style:italic">No campaigns</div>';
        $navContent.innerHTML = html;
        return;
    }

    for (const tree of state.allCampaigns) {
        const c = tree.campaign;
        const total = tree.status?.total || 0;
        const isActive = state.activeCampaignId === c.id;
        html += `<div class="nav-campaign-item ${isActive ? 'active' : ''}" onclick="openCampaign('${esc(c.id)}')">
            <span class="nav-status-dot st-${c.status || 'active'}"></span>
            <span class="nav-campaign-name">${esc(c.shorthand || c.description)}</span>
            ${total > 0 ? `<span class="nav-campaign-count">${total}</span>` : ''}
        </div>`;
    }

    $navContent.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
//  CANVAS
// ══════════════════════════════════════════════════════════

function renderCanvas() {
    if (!$canvasContent) return;
    if (!state.docData) {
        $canvasContent.innerHTML = `<div class="empty-state">
            <p>Select a campaign</p>
        </div>`;
        return;
    }
    switch (state.planView) {
        case 'board': renderBoardView(); break;
        case 'decide': renderDecideView(); break;
        case 'graph': renderGraphView(); break;
        case 'matrix': renderMatrixView(); break;
        default: renderPlanDoc(); break;
    }
}


// ── Plan Document ────────────────────────────────────────

function renderPlanDoc() {
    if (!state.docData) return;
    const tree = state.docData.document;
    const campaign = tree.campaign;
    const status = state.docData.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const inProgress = status.in_progress || 0;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

    let html = '<div class="plan-doc">';

    // Title card
    const descParts = (campaign.description || '').split(' — ');
    const titleMain = descParts[0] || campaign.shorthand;
    const titleSub = descParts.length > 1 ? descParts.slice(1).join(' — ') : '';

    html += `<div class="plan-title-card" id="plan-title-card">
        <div class="plan-title-card-header">
            <div class="plan-title-card-top">
                ${campaign.shorthand ? `<span class="plan-title-card-shorthand">${esc(campaign.shorthand)}</span>` : ''}
                <span class="status-badge status-${campaign.status}">${STATUS_LABELS[campaign.status] || campaign.status}</span>
            </div>
            <span class="plan-title-card-name">${esc(titleMain)}</span>
            ${titleSub ? `<span class="plan-title-card-subtitle">${esc(titleSub)}</span>` : ''}
        </div>
        ${campaign.target ? `<div class="plan-title-card-target">${esc(campaign.target)}</div>` : ''}
        <div class="plan-title-card-stats">
            <span>${total} items</span>
            <span>${completed} done</span>
            <span>${inProgress} active</span>
        </div>
        ${total > 0 ? `
        <div class="plan-progress-bar">
            <div class="plan-progress-fill" style="width:${pct}%"></div>
        </div>
        <div class="plan-progress-label">${pct}%</div>` : ''}
    </div>`;

    // Root-level items (not assigned to any phase)
    const rootItems = tree.items || [];
    if (rootItems.length > 0) {
        html += '<div class="phase-block">';
        if ((tree.children || []).length > 0) {
            html += `<div class="phase-block-header">
                <span class="phase-block-name">Unassigned</span>
                <span class="phase-block-count">${rootItems.length} item${rootItems.length !== 1 ? 's' : ''}</span>
            </div>`;
        }
        rootItems.forEach((item, idx) => {
            html += renderDocItem(item, String(idx + 1));
        });
        html += '</div>';
    }

    // Phase sections (children)
    const children = tree.children || [];
    children.forEach((child, idx) => {
        html += renderPhaseBlock(child, idx + 1);
    });

    // Bibliography
    html += renderBibliography();

    // Version history
    html += renderVersionHistory();

    html += '</div>';
    $canvasContent.innerHTML = html;
}

function renderPhaseBlock(child, phaseNum) {
    const campaign = child.campaign;
    const items = child.items || [];
    const status = child.status || {};
    const total = status.total || 0;

    let html = `<div class="phase-block" id="phase-${phaseNum}" data-phase="${phaseNum}">
        <div class="phase-block-header">
            <span class="phase-block-num">Phase ${phaseNum}</span>
            <span class="phase-block-name">${esc(campaign.description || campaign.shorthand)}</span>
            <span class="phase-block-count">${total} item${total !== 1 ? 's' : ''}</span>
        </div>`;

    items.forEach((item, idx) => {
        html += renderDocItem(item, `${phaseNum}.${idx + 1}`);
    });

    html += '</div>';
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

    // Ref badges
    let refBadges = '';
    if (item.ref_numbers && item.ref_numbers.length > 0) {
        refBadges = item.ref_numbers.map(n =>
            `<a class="ref-badge" onclick="event.stopPropagation(); scrollCanvasTo('ref-${n}')">[${n}]</a>`
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
    const bib = state.docData?.bibliography || [];
    let html = `<div class="bibliography-section" id="bibliography-section">
        <div class="bibliography-title">References</div>`;

    if (bib.length === 0) {
        html += '<div class="no-data-msg">No references cited.</div>';
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

    html += '</div>';
    return html;
}

function renderVersionHistory() {
    let html = `<div class="versions-section" id="versions-section">
        <div class="versions-title">Version History</div>`;

    if (state.versions.length === 0) {
        html += '<div class="no-data-msg">No snapshots yet.</div>';
    } else {
        const sorted = [...state.versions].sort((a, b) =>
            (b.version_number || 0) - (a.version_number || 0)
        );
        sorted.forEach((v, idx) => {
            const isCurrent = idx === 0;
            const label = v.label || v.summary || 'Snapshot';
            const date = v.created_at ? formatDate(v.created_at) : '';
            // Extract item count from summary (e.g. "...12 items total...")
            const itemMatch = (v.summary || '').match(/(\d+)\s+items?\s+total/);
            const itemCount = itemMatch ? itemMatch[1] + ' items' : '';
            html += `<div class="version-entry ${isCurrent ? 'current' : ''}" onclick="viewVersion('${v.version_id}', ${isCurrent})">
                <span class="version-entry-num">v${v.version_number || '?'}</span>
                <span class="version-entry-label">${esc(label)}</span>
                ${isCurrent ? '<span class="version-entry-current">Current</span>' : ''}
                ${itemCount ? `<span class="version-entry-count">${itemCount}</span>` : ''}
                <span class="version-entry-date">${esc(date)}</span>
            </div>`;
        });
    }

    html += '</div>';
    return html;
}

// ══════════════════════════════════════════════════════════
//  INSPECTOR (detail panel)
// ══════════════════════════════════════════════════════════

async function selectItem(itemId) {
    state.selectedItemId = itemId;

    // Highlight in document
    document.querySelectorAll('.doc-item.selected').forEach(el => el.classList.remove('selected'));
    const docEl = document.getElementById(`item-${itemId}`);
    if (docEl) docEl.classList.add('selected');

    // Highlight in navigator
    document.querySelectorAll('.nav-item.active').forEach(el => el.classList.remove('active'));
    const navEl = document.querySelector(`.nav-item[data-item-id="${itemId}"]`);
    if (navEl) navEl.classList.add('active');

    // Open inspector
    $workspace?.classList.add('inspector-open');

    // Show loading state
    if ($inspectorTitle) $inspectorTitle.textContent = 'Loading...';
    if ($inspectorStatus) $inspectorStatus.textContent = '';
    if ($inspectorBody) $inspectorBody.innerHTML = '<div class="loading-spinner" style="margin:24px auto"></div>';

    // Fetch enriched data
    const campaignId = state.activeCampaignId;
    if (!campaignId) return;

    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(itemId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderInspector(data);
    } catch (err) {
        console.error('Failed to load item detail:', err);
        if ($inspectorBody) {
            $inspectorBody.innerHTML = `<p style="color:var(--accent-orange);font-size:0.82rem">Failed to load item details.</p>`;
        }
    }
}

function renderInspector(data) {
    const item = data.item;
    const deps = data.dependencies || [];
    const dnts = data.dependents || [];
    const sessions = data.sessions || [];

    if ($inspectorTitle) $inspectorTitle.textContent = item.title;
    if ($inspectorStatus) {
        const sc = item.status === 'completed' ? 'completed'
            : item.status === 'in_progress' ? 'active' : 'planned';
        $inspectorStatus.className = 'inspector-status';
        $inspectorStatus.innerHTML = `<span class="status-badge status-${sc}">${STATUS_LABELS[item.status] || item.status}</span>`;
    }

    let html = '';

    // Type + ID
    const icon = TYPE_ICONS[item.type] || '\u{1F4CB}';
    html += `<div class="detail-meta">
        <span class="detail-type">${icon} ${(item.type || '').replace('_', ' ')}</span>
        <span class="detail-id">${item.id}</span>
    </div>`;

    // Description
    if (item.description) {
        html += section('Description', `<div class="detail-section-content">${esc(item.description)}</div>`);
    }

    // Outcome
    if (item.outcome) {
        html += section('Outcome', `<div class="detail-section-content">${esc(item.outcome)}</div>`);
    }

    // Imaging spec
    if (item.imaging_spec) {
        html += section('Imaging Specification', `<table class="spec-table">${renderSpecTable(item.imaging_spec)}</table>`);
    }

    // Bench spec
    if (item.bench_spec) {
        html += section('Bench Specification', `<table class="spec-table">${renderSpecTable(item.bench_spec)}</table>`);
    }

    // Dependencies
    if (deps.length > 0) {
        const chips = deps.map(d => {
            const dot = STATUS_DOTS[d.status] || '\u25CB';
            const dc = d.status || 'planned';
            return `<span class="dep-chip" onclick="navigateToItem('${d.id}')">
                <span class="dep-chip-dot dot-${dc}">${dot}</span>
                ${esc(d.title)}
            </span>`;
        }).join('');
        html += section('Depends on', `<div class="dep-list">${chips}</div>`);
    }

    // Dependents
    if (dnts.length > 0) {
        const chips = dnts.map(d => {
            const dot = STATUS_DOTS[d.status] || '\u25CB';
            const dc = d.status || 'planned';
            return `<span class="dep-chip" onclick="navigateToItem('${d.id}')">
                <span class="dep-chip-dot dot-${dc}">${dot}</span>
                ${esc(d.title)}
            </span>`;
        }).join('');
        html += section('Blocks', `<div class="dep-list">${chips}</div>`);
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
            const linkHtml = link
                ? `<a href="${esc(link)}" target="_blank" rel="noopener">${esc(text)}</a>`
                : esc(text);
            refHtml += `<div class="detail-ref">
                <span class="detail-ref-num">[${idx + 1}]</span>
                ${linkHtml}
            </div>`;
        });
        html += section('References', refHtml);
    }

    // Sessions
    if (sessions.length > 0) {
        let sessHtml = '';
        sessions.forEach(s => {
            sessHtml += `<div class="detail-session">
                <span class="detail-session-title">${esc(s.planned_intent || s.id || 'Session')}</span>
                ${s.created_at ? `<span class="detail-session-date">${formatDate(s.created_at)}</span>` : ''}
            </div>`;
        });
        html += section('Sessions', sessHtml);
    } else {
        html += section('Sessions',
            '<div class="detail-section-content" style="color:var(--text-muted);font-style:italic">No linked sessions</div>');
    }

    if ($inspectorBody) $inspectorBody.innerHTML = html;
}

function closeInspector() {
    state.selectedItemId = null;
    $workspace?.classList.remove('inspector-open');
    document.querySelectorAll('.doc-item.selected').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.nav-item.active').forEach(el => el.classList.remove('active'));
}

// ══════════════════════════════════════════════════════════
//  VERSION VIEWING
// ══════════════════════════════════════════════════════════

function toggleVersionDropdown() {
    if (!$versionDropdown) return;

    if ($versionDropdown.classList.contains('hidden')) {
        let html = `<div class="version-item ${!state.viewingSnapshotId ? 'active' : ''}" onclick="backToCurrent()">
            <span class="version-num">Current</span>
            <div class="version-info"><div class="version-label-text">Live data</div></div>
        </div>`;

        const sorted = [...state.versions].sort((a, b) =>
            (b.version_number || 0) - (a.version_number || 0)
        );
        sorted.forEach(v => {
            const active = state.viewingSnapshotId === v.version_id;
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

        $versionDropdown.innerHTML = html;
        $versionDropdown.classList.remove('hidden');
    } else {
        $versionDropdown.classList.add('hidden');
    }
}

async function viewVersion(versionId, isCurrent) {
    $versionDropdown?.classList.add('hidden');

    if (isCurrent) {
        backToCurrent();
        return;
    }

    const campaignId = state.activeCampaignId;
    if (!campaignId) return;

    try {
        const res = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/versions/${encodeURIComponent(versionId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const snapshot = data.version;

        if (snapshot && snapshot.snapshot_json) {
            state.viewingSnapshotId = versionId;
            // Snapshot JSON is flat ({description, shorthand, items, children})
            // but the rendering code expects {campaign: {...}, items, children, status}
            const normalized = normalizeSnapshotTree(snapshot.snapshot_json);
            state.docData.document = normalized;
            state.docData.status = normalized.status;
            state.docData.bibliography = [];  // snapshots don't carry bibliography
            state.allItemsFlat = {};
            buildItemIndex(normalized);

            renderAll();

            // Show banner
            if ($snapshotBanner) $snapshotBanner.classList.remove('hidden');
            if ($snapshotBannerText) {
                $snapshotBannerText.textContent = `Viewing v${snapshot.version_number || '?'}: ${snapshot.label || 'Snapshot'}`;
            }
            if ($versionLabel) $versionLabel.textContent = `v${snapshot.version_number || '?'}`;
        }
    } catch (err) {
        console.error('Failed to load version:', err);
    }
}

function backToCurrent() {
    $versionDropdown?.classList.add('hidden');
    state.viewingSnapshotId = null;

    if ($snapshotBanner) $snapshotBanner.classList.add('hidden');
    if ($versionLabel) $versionLabel.textContent = 'Current';

    // Reload current data
    const campaignId = state.activeCampaignId;
    if (campaignId) {
        loadDocument(campaignId).then(() => {
            renderCanvas();
            renderNavigator();
        });
    }
}

// ══════════════════════════════════════════════════════════
//  HEADER & STATUSBAR
// ══════════════════════════════════════════════════════════

function updateHeader() {
    const $viewSwitcher = document.getElementById('plan-view-switcher');
    const campaign = state.docData?.document?.campaign;
    if ($headerTitle) $headerTitle.textContent = campaign?.shorthand || 'Campaigns';
    if ($headerBreadcrumb) $headerBreadcrumb.textContent = '';
    // Show view switcher and controls when a campaign is loaded
    if (state.docData) {
        if (state.versions.length > 0) {
            $versionWrap?.classList.remove('hidden');
        } else {
            $versionWrap?.classList.add('hidden');
        }
        $printBtn?.classList.remove('hidden');
        $viewSwitcher?.classList.remove('hidden');
        updateViewButtons('plan-view-switcher', state.planView);
    } else {
        $versionWrap?.classList.add('hidden');
        $printBtn?.classList.add('hidden');
        $viewSwitcher?.classList.add('hidden');
    }
}

function updateStatusbar() {
    const numCampaigns = state.allCampaigns.length;
    const status = state.docData?.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

    if ($statusLeft) {
        $statusLeft.textContent = state.docData
            ? `${numCampaigns} campaign${numCampaigns !== 1 ? 's' : ''} \u00B7 ${total} items \u00B7 ${pct}% complete`
            : `${numCampaigns} campaign${numCampaigns !== 1 ? 's' : ''}`;
    }
    if ($statusRight) {
        if (state.versions.length > 0) {
            const latest = state.versions.reduce((a, b) =>
                (b.version_number || 0) > (a.version_number || 0) ? b : a, state.versions[0]);
            const date = latest.created_at ? formatDate(latest.created_at) : '';
            $statusRight.textContent = `v${latest.version_number || '?'} ${date}`;
        } else {
            $statusRight.textContent = '';
        }
    }
}

// ══════════════════════════════════════════════════════════
//  NAVIGATION HELPERS
// ══════════════════════════════════════════════════════════

function navigateToItem(itemId) {
    // Scroll to item in canvas
    const el = document.getElementById(`item-${itemId}`);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    // Open detail
    selectItem(itemId);
}

function scrollCanvasTo(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleNavPhase(headerEl) {
    const chevron = headerEl.querySelector('.nav-phase-chevron');
    const items = headerEl.nextElementSibling;
    if (items) items.classList.toggle('collapsed');
    if (chevron) chevron.classList.toggle('open');
}

// ── Scroll-spy ───────────────────────────────────────────
function onCanvasScroll() {
    if (state.planView !== 'doc' || state.selectedItemId) return;

    const canvas = document.getElementById('canvas');
    if (!canvas) return;

    let closestItem = null;
    let closestDist = Infinity;

    document.querySelectorAll('.doc-item').forEach(el => {
        const rect = el.getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();
        const relTop = rect.top - canvasRect.top;
        if (relTop < 100 && Math.abs(relTop) < closestDist) {
            closestDist = Math.abs(relTop);
            closestItem = el;
        }
    });

    if (closestItem) {
        const itemId = closestItem.dataset.itemId;
        document.querySelectorAll('.nav-item.active').forEach(el => el.classList.remove('active'));
        const navEl = document.querySelector(`.nav-item[data-item-id="${itemId}"]`);
        if (navEl) navEl.classList.add('active');
    }
}

// ══════════════════════════════════════════════════════════
//  UTILITIES
// ══════════════════════════════════════════════════════════

/**
 * Normalize a snapshot tree into the format the rendering code expects.
 *
 * Snapshots store a compact format:
 *   {description, shorthand, target, items: [{type, title, spec, ...}], children: [...]}
 * Document API returns:
 *   {campaign: {id, description, ...}, items: [{id, status, imaging_spec, ...}], children, status}
 *
 * Key differences: snapshot items have `spec` (not `imaging_spec`/`bench_spec`),
 * no `id`, no `status` (null), and `depends_on_indices` instead of resolved deps.
 */
function normalizeSnapshotTree(node) {
    if (!node) return node;
    // Already in document format (has .campaign key)
    if (node.campaign) return node;

    const items = (node.items || []).map((item, idx) => {
        const normalized = { ...item };
        // Generate stable placeholder ID
        if (!normalized.id) {
            normalized.id = '_snap_' + idx + '_' + Math.random().toString(36).slice(2, 8);
        }
        // Default status
        if (!normalized.status) {
            normalized.status = 'planned';
        }
        // Snapshot uses generic `spec` — map to typed spec based on item type
        if (normalized.spec && !normalized.imaging_spec && !normalized.bench_spec) {
            if (normalized.type === 'imaging') {
                normalized.imaging_spec = normalized.spec;
            } else if (normalized.type === 'bench') {
                normalized.bench_spec = normalized.spec;
            } else if (normalized.type === 'genetics') {
                normalized.bench_spec = normalized.spec;
            }
        }
        return normalized;
    });

    const children = (node.children || []).map(c => normalizeSnapshotTree(c));

    // Count status from items + children
    let total = items.length, completed = 0, inProgress = 0, planned = 0;
    for (const it of items) {
        if (it.status === 'completed') completed++;
        else if (it.status === 'in_progress') inProgress++;
        else planned++;
    }
    for (const ch of children) {
        const s = ch.status || {};
        total += s.total || 0;
        completed += s.completed || 0;
        inProgress += s.in_progress || 0;
        planned += s.planned || 0;
    }

    return {
        campaign: {
            description: node.description || '',
            shorthand: node.shorthand || null,
            target: node.target || null,
            status: node.status || 'active',
            id: node.id || '',
        },
        items,
        children,
        status: { total, completed, in_progress: inProgress, planned },
    };
}

function showLoading(text) {
    if ($canvasLoading) {
        $canvasLoading.classList.remove('hidden');
        const p = $canvasLoading.querySelector('p');
        if (p) p.textContent = text || 'Loading...';
    }
    if ($canvasContent) $canvasContent.innerHTML = '';
}

function hideLoading() {
    $canvasLoading?.classList.add('hidden');
}

function section(title, content) {
    return `<div class="detail-section"><div class="detail-section-title">${title}</div>${content}</div>`;
}

function renderSpecTable(spec) {
    let rows = '';
    for (const [key, value] of Object.entries(spec)) {
        if (value == null || key.startsWith('_')) continue;
        const label = SPEC_LABELS[key] || key;
        let display = Array.isArray(value) ? value.join(', ') : String(value);
        if (SPEC_UNITS[key]) display += SPEC_UNITS[key];
        rows += `<tr><td>${esc(label)}</td><td>${esc(display)}</td></tr>`;
    }
    return rows;
}

function esc(str) { return escapeHtml(str); }

function formatDate(isoStr) {
    try {
        const d = new Date(isoStr);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
            ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
        return isoStr;
    }
}


// ══════════════════════════════════════════════════════════
//  PLAN VIEW SWITCHING
// ══════════════════════════════════════════════════════════

const STATUS_COLORS = {
    planned: 'var(--text-muted)',
    in_progress: 'var(--accent)',
    completed: 'var(--accent-green)',
    skipped: 'var(--text-muted)',
    blocked: '#f85149',
};

function setupPlanViewSwitcher() {
    initViewSwitcher('plan-view-switcher', switchPlanView, {
        views: ['doc', 'graph', 'board', 'decide', 'matrix'],
        guard: () => !!state.docData
    });
}

function switchPlanView(viewName) {
    if (!state.docData) return;
    state.planView = viewName;
    updateViewButtons('plan-view-switcher', viewName);
    renderCanvas();
}

// Collect all items from the document tree into a flat array
function collectAllItems(node, phase, phaseNum) {
    const result = [];
    function walk(n, ph, pn) {
        (n.items || []).forEach(item => {
            result.push({ ...item, _phase: ph || 'Unassigned', _phaseNum: pn || 0 });
        });
        (n.children || []).forEach((child, idx) => {
            const childName = child.campaign?.description || child.title || '';
            const cph = childName || ph;
            const cpn = pn || idx + 1;
            walk(child, cph, cpn);
        });
    }
    walk(node, phase, phaseNum);
    return result;
}

// ══════════════════════════════════════════════════════════
//  BOARD VIEW
// ══════════════════════════════════════════════════════════

function renderBoardView() {
    const doc = state.docData?.document;
    if (!doc) return;
    const items = collectAllItems(doc);
    const columns = ['planned', 'in_progress', 'completed', 'skipped', 'blocked'];
    const labels = { planned: 'Planned', in_progress: 'In Progress', completed: 'Completed', skipped: 'Skipped', blocked: 'Blocked' };
    const grouped = {};
    columns.forEach(c => grouped[c] = []);
    items.forEach(item => {
        const col = grouped[item.status] || grouped.planned;
        col.push(item);
    });

    let html = '<div class="board-view">';
    columns.forEach(col => {
        const colItems = grouped[col];
        html += `<div class="board-column">
            <div class="board-column-header">
                <span class="board-column-dot" style="background:${STATUS_COLORS[col]}"></span>
                ${labels[col]} <span class="board-column-count">${colItems.length}</span>
            </div>
            <div class="board-column-body">`;
        colItems.forEach(item => {
            const icon = TYPE_ICONS[item.type] || '';
            const spec = item.imaging_spec || item.bench_spec;
            const specLine = spec ? (spec.strain || spec.protocol || '') : '';
            html += `<div class="board-card" onclick="selectItem('${item.id}')">
                <div class="board-card-top">
                    <span class="board-card-icon">${icon}</span>
                    <span class="board-card-phase">P${item._phaseNum}</span>
                </div>
                <div class="board-card-title">${esc(item.title)}</div>
                ${specLine ? `<div class="board-card-spec">${esc(specLine)}</div>` : ''}
            </div>`;
        });
        html += '</div></div>';
    });
    html += '</div>';
    $canvasContent.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
//  DECIDE VIEW
// ══════════════════════════════════════════════════════════

function renderDecideView() {
    const doc = state.docData?.document;
    if (!doc) return;
    const items = collectAllItems(doc);
    const decisions = items.filter(i => i.type === 'decision_point');

    if (decisions.length === 0) {
        $canvasContent.innerHTML = `<div class="empty-state">
            <div style="font-size:2rem;margin-bottom:12px">🚦</div>
            <p>No decision points in this plan</p>
            <span class="empty-hint">Decision points help track key branching moments in your experiment</span>
        </div>`;
        return;
    }

    let html = '<div class="decide-view">';
    decisions.forEach(item => {
        const deps = (item.dependencies || []).map(d =>
            `<span class="dep-chip" onclick="selectItem('${d.id}')">${STATUS_DOTS[d.status] || '○'} ${esc(d.title)}</span>`
        ).join('');
        const dependents = (item.dependents || []).map(d =>
            `<span class="dep-chip" onclick="selectItem('${d.id}')">${esc(d.title)}</span>`
        ).join('');
        const statusClass = item.status === 'completed' ? 'completed' : item.status === 'in_progress' ? 'active' : '';

        html += `<div class="decide-card ${statusClass}" onclick="selectItem('${item.id}')">
            <div class="decide-phase">Phase ${item._phaseNum}</div>
            <div class="decide-header">
                <span class="decide-status">${STATUS_DOTS[item.status] || '○'}</span>
                <h3>${esc(item.title)}</h3>
            </div>
            ${item.description ? `<div class="decide-desc">${esc(item.description)}</div>` : ''}
            ${deps ? `<div class="decide-section"><div class="decide-section-label">Inputs</div>${deps}</div>` : ''}
            <div class="decide-section">
                <div class="decide-section-label">Outcome</div>
                <div class="decide-outcome ${item.outcome ? '' : 'pending'}">${item.outcome ? esc(item.outcome) : 'Pending — no outcome recorded yet'}</div>
            </div>
            ${dependents ? `<div class="decide-section"><div class="decide-section-label">Unblocks</div>${dependents}</div>` : ''}
        </div>`;
    });
    html += '</div>';
    $canvasContent.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
//  GRAPH VIEW
// ══════════════════════════════════════════════════════════

function renderGraphView() {
    const doc = state.docData?.document;
    if (!doc) return;
    const items = collectAllItems(doc);
    if (items.length === 0) {
        $canvasContent.innerHTML = '<div class="empty-state"><p>No items to graph</p></div>';
        return;
    }

    // Build adjacency and compute depths via BFS from roots
    const byId = {};
    items.forEach(i => byId[i.id] = i);
    const inDegree = {};
    const outEdges = {};
    items.forEach(i => {
        inDegree[i.id] = 0;
        outEdges[i.id] = [];
    });
    items.forEach(i => {
        (i.dependencies || []).forEach(d => {
            if (byId[d.id]) {
                outEdges[d.id].push(i.id);
                inDegree[i.id] = (inDegree[i.id] || 0) + 1;
            }
        });
    });

    // Topological layering
    const depth = {};
    const queue = items.filter(i => inDegree[i.id] === 0).map(i => i.id);
    queue.forEach(id => depth[id] = 0);
    let head = 0;
    while (head < queue.length) {
        const id = queue[head++];
        outEdges[id].forEach(next => {
            depth[next] = Math.max(depth[next] || 0, depth[id] + 1);
            inDegree[next]--;
            if (inDegree[next] === 0) queue.push(next);
        });
    }
    const maxDepth = Math.max(0, ...Object.values(depth));
    items.forEach(i => { if (depth[i.id] === undefined) depth[i.id] = maxDepth + 1; });

    // Group by depth layer
    const layers = {};
    items.forEach(i => {
        const d = depth[i.id];
        if (!layers[d]) layers[d] = [];
        layers[d].push(i);
    });

    // Collect phase info for background bands
    const phaseMap = {};
    items.forEach(i => {
        if (!phaseMap[i._phaseNum]) phaseMap[i._phaseNum] = { name: i._phase, items: [] };
        phaseMap[i._phaseNum].items.push(i);
    });

    // Node sizing — horizontal layout (left-to-right flow)
    const nodeW = 220, nodeH = 72, pad = 40;
    const gapX = 100, gapY = 28;
    const numLayers = Math.max(...Object.keys(layers).map(Number)) + 1;
    const maxPerLayer = Math.max(...Object.values(layers).map(l => l.length));
    const svgW = numLayers * nodeW + (numLayers - 1) * gapX + pad * 2;
    const svgH = maxPerLayer * nodeH + (maxPerLayer - 1) * gapY + pad * 2;

    // Assign positions
    const pos = {};
    Object.entries(layers).forEach(([d, layerItems]) => {
        const di = parseInt(d);
        const totalH = layerItems.length * (nodeH + gapY) - gapY;
        const startY = (svgH - totalH) / 2;
        layerItems.forEach((item, idx) => {
            pos[item.id] = {
                x: pad + di * (nodeW + gapX),
                y: startY + idx * (nodeH + gapY)
            };
        });
    });

    // Phase background bands
    let bands = '';
    const phaseColors = ['rgba(96,165,250,0.06)', 'rgba(52,211,153,0.06)', 'rgba(251,191,36,0.06)', 'rgba(167,139,250,0.06)', 'rgba(248,113,113,0.06)'];
    Object.entries(phaseMap).forEach(([pNum, info], ci) => {
        if (!info.items.length) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        info.items.forEach(i => {
            const p = pos[i.id];
            if (!p) return;
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x + nodeW);
            maxY = Math.max(maxY, p.y + nodeH);
        });
        const bpad = 16;
        const color = phaseColors[ci % phaseColors.length];
        bands += `<rect x="${minX - bpad}" y="${minY - bpad - 18}" width="${maxX - minX + nodeW + bpad * 2 - nodeW}" height="${maxY - minY + nodeH + bpad * 2 + 18}" rx="12" fill="${color}" />`;
        bands += `<text x="${minX - bpad + 8}" y="${minY - bpad}" class="graph-phase-label">P${pNum} — ${esc(info.name.split(' — ').pop())}</text>`;
    });

    // Render edges
    let edges = '';
    items.forEach(i => {
        (i.dependencies || []).forEach(d => {
            if (!pos[d.id] || !pos[i.id]) return;
            const from = pos[d.id];
            const to = pos[i.id];
            const x1 = from.x + nodeW, y1 = from.y + nodeH / 2;
            const x2 = to.x, y2 = to.y + nodeH / 2;
            const cx = (x1 + x2) / 2;
            edges += `<path d="M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}" class="graph-edge" data-from="${d.id}" data-to="${i.id}" />`;
        });
    });

    // Render nodes — full title with word wrap via foreignObject
    let nodes = '';
    items.forEach(i => {
        const p = pos[i.id];
        if (!p) return;
        const icon = TYPE_ICONS[i.type] || '';
        const color = STATUS_COLORS[i.status] || STATUS_COLORS.planned;
        const statusLabel = STATUS_LABELS[i.status] || i.status;
        nodes += `<g class="graph-node" onclick="selectItem('${i.id}')" data-item-id="${i.id}" data-item-type="${i.type}">
            <rect x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="10"
                  fill="var(--bg-card)" stroke="${color}" stroke-width="2" />
            <foreignObject x="${p.x + 10}" y="${p.y + 6}" width="${nodeW - 20}" height="${nodeH - 12}">
                <div xmlns="http://www.w3.org/1999/xhtml" class="graph-node-inner">
                    <div class="graph-node-top"><span>${icon}</span><span class="graph-node-phase">P${i._phaseNum}</span></div>
                    <div class="graph-node-title">${esc(i.title)}</div>
                    <div class="graph-node-status" style="color:${color}">${STATUS_DOTS[i.status]} ${statusLabel}</div>
                </div>
            </foreignObject>
        </g>`;
    });

    // Type filter toolbar
    const types = [...new Set(items.map(i => i.type))];
    const typeButtons = types.map(t =>
        `<button class="graph-filter-btn" data-filter-type="${t}" onclick="filterGraphByType('${t}')">${TYPE_ICONS[t] || ''} ${esc(t)}</button>`
    ).join('');

    $canvasContent.innerHTML = `<div class="graph-view" id="graph-view-container">
        <div class="graph-toolbar">
            <button class="graph-filter-btn active" data-filter-type="" onclick="filterGraphByType('')">All</button>
            ${typeButtons}
        </div>
        <svg width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">
            ${bands}${edges}${nodes}
        </svg>
    </div>`;

    // Convert vertical scroll to horizontal when graph doesn't need vertical scrolling
    const container = document.getElementById('graph-view-container');
    if (container) {
        container.addEventListener('wheel', (e) => {
            const needsVerticalScroll = container.scrollHeight > container.clientHeight;
            if (!needsVerticalScroll && Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
                e.preventDefault();
                container.scrollLeft += e.deltaY;
            }
        }, { passive: false });
    }
}

function filterGraphByType(type) {
    // Update button active states
    document.querySelectorAll('.graph-filter-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.filterType === type);
    });
    // Dim/highlight nodes and edges
    document.querySelectorAll('.graph-node').forEach(node => {
        const itemType = node.dataset.itemType || '';
        node.classList.toggle('dimmed', type !== '' && itemType !== type);
    });
    document.querySelectorAll('.graph-edge').forEach(edge => {
        edge.classList.toggle('dimmed', type !== '');
    });
    // Keep edges connected to highlighted nodes visible
    if (type) {
        const activeIds = new Set();
        document.querySelectorAll(`.graph-node:not(.dimmed)`).forEach(n => activeIds.add(n.dataset.itemId));
        document.querySelectorAll('.graph-edge').forEach(edge => {
            const from = edge.dataset.from;
            const to = edge.dataset.to;
            if (activeIds.has(from) || activeIds.has(to)) {
                edge.classList.remove('dimmed');
            }
        });
    }
}

// ══════════════════════════════════════════════════════════
//  MATRIX VIEW
// ══════════════════════════════════════════════════════════

function renderMatrixView() {
    const doc = state.docData?.document;
    if (!doc) return;
    const phases = (doc.children || []).filter(c => c.children);
    const types = ['imaging', 'bench', 'genetics', 'analysis', 'decision_point'];
    const typeLabels = { imaging: 'Imaging', bench: 'Bench', genetics: 'Genetics', analysis: 'Analysis', decision_point: 'Decisions' };

    let html = '<div class="matrix-view"><table class="matrix-table"><thead><tr><th>Phase</th>';
    types.forEach(t => html += `<th>${typeLabels[t]}</th>`);
    html += '<th>Total</th></tr></thead><tbody>';

    const colTotals = {};
    types.forEach(t => colTotals[t] = { total: 0, completed: 0, active: 0 });
    let grandTotal = 0;

    phases.forEach((phase, idx) => {
        const phaseName = phase.campaign?.description || phase.title || '';
        const shortName = phaseName.split(' — ').pop() || phaseName;
        const items = collectAllItems(phase, phaseName, idx + 1);
        let rowTotal = 0;
        html += `<tr><td class="matrix-phase">P${idx + 1} — ${esc(shortName)}</td>`;
        types.forEach(type => {
            const matching = items.filter(i => i.type === type);
            const done = matching.filter(i => i.status === 'completed').length;
            const active = matching.filter(i => i.status === 'in_progress').length;
            const count = matching.length;
            colTotals[type].total += count;
            colTotals[type].completed += done;
            colTotals[type].active += active;
            rowTotal += count;
            grandTotal += count;
            if (count === 0) {
                html += '<td class="matrix-cell empty">—</td>';
            } else {
                const dots = matching.map(i => `<span class="matrix-dot" style="background:${STATUS_COLORS[i.status]}" title="${esc(i.title)}"></span>`).join('');
                html += `<td class="matrix-cell">${dots}<span class="matrix-count">${count}</span></td>`;
            }
        });
        html += `<td class="matrix-total">${rowTotal}</td></tr>`;
    });

    // Totals row
    html += '<tr class="matrix-totals-row"><td>Total</td>';
    types.forEach(t => {
        html += `<td class="matrix-total">${colTotals[t].total}</td>`;
    });
    html += `<td class="matrix-total">${grandTotal}</td></tr>`;
    html += '</tbody></table></div>';
    $canvasContent.innerHTML = html;
}

// Expose init and onclick-referenced functions to global scope
window.CampaignsApp = { init: boot };
window.openCampaign = openCampaign;
window.navigateToItem = navigateToItem;
window.toggleNavPhase = toggleNavPhase;
window.scrollCanvasTo = scrollCanvasTo;
window.selectItem = selectItem;
window.viewVersion = viewVersion;
window.backToCurrent = backToCurrent;
window.switchPlanView = switchPlanView;
window.filterGraphByType = filterGraphByType;

// Auto-init on standalone campaigns page (detected via data-page attribute)
if (document.body?.dataset.page === 'campaigns') {
    boot();
} else {
    document.addEventListener('DOMContentLoaded', () => {
        if (document.body.dataset.page === 'campaigns') boot();
    });
}

})(); // end IIFE
