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
// Imaging-spec fields the inspector lets you edit/fill inline (ordered for the form).
// Empty ones still show \u2014 that's how you fill a TBD value like laser power.
const IMAGING_SPEC_FIELDS = [
    'strain', 'genotype', 'reporter', 'sample_prep', 'temperature_c', 'num_embryos',
    'num_slices', 'exposure_ms', 'laser_wavelength_nm', 'laser_power_pct', 'interval_s',
    'target_window', 'start_stage', 'stop_condition', 'estimated_duration_h',
    'success_criteria', 'comparison_to',
];
const SPEC_NUMERIC = new Set([
    'temperature_c', 'num_embryos', 'num_slices', 'exposure_ms', 'laser_wavelength_nm',
    'laser_power_pct', 'interval_s', 'estimated_duration_h',
]);

// ── State ────────────────────────────────────────────────
const state = {
    planView: 'doc',            // 'doc' | 'graph' | 'board' | 'decide' | 'matrix' | 'timeline'
    typeFilter: '',              // '' = all, or 'imaging', 'bench', etc.
    activeCampaignId: null,
    selectedItemId: null,
    allCampaigns: [],           // full tree list from /api/campaigns
    docData: null,              // plan document data for plan view
    versions: [],               // snapshots list
    viewingSnapshotId: null,
    allItemsFlat: {},           // id → item for quick lookup
    editingSpec: false,         // inspector imaging-spec edit mode
    _inspectorData: null,       // last item-detail payload (for re-render on edit toggle)
    _specError: '',             // inline save error in the spec editor
    _sessionPickerOpen: false,  // whether the session link picker is visible
    _availableSessions: null,   // null = not yet loaded, [] = loaded (for link picker)
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

    // Delegated click handler — replaces inline onclick attributes
    document.addEventListener('click', e => {
        const el = e.target.closest('[data-action]');
        if (!el) return;
        const action = el.dataset.action;
        const id = el.dataset.id;
        switch (action) {
            case 'select-item': selectItem(id); break;
            case 'open-campaign': openCampaign(id); break;
            case 'navigate-item': e.stopPropagation(); navigateToItem(id); break;
            case 'run-item': e.stopPropagation(); runPlanItem(id); break;
            case 'filter-type': applyTypeFilter(el.dataset.filterType); break;
            case 'view-version': viewVersion(el.dataset.versionId, el.dataset.isCurrent === 'true'); break;
            case 'back-to-current': backToCurrent(); break;
            case 'scroll-to': e.stopPropagation(); scrollCanvasTo(el.dataset.target); break;
            case 'toggle-phase': toggleNavPhase(el); break;
            case 'spec-edit': e.stopPropagation(); startSpecEdit(); break;
            case 'spec-cancel': e.stopPropagation(); cancelSpecEdit(); break;
            case 'spec-save': e.stopPropagation(); saveSpecEdit(); break;
            case 'session-picker-open': e.stopPropagation(); openSessionPicker(); break;
            case 'session-picker-cancel': e.stopPropagation(); cancelSessionPicker(); break;
            case 'session-picker-link': e.stopPropagation(); submitSessionLink(); break;
            case 'session-delink': e.stopPropagation(); handleSessionDelink(el.dataset.sessionId); break;
        }
    });

    // Scroll-spy for plan view
    document.getElementById('canvas')?.addEventListener('scroll', onCanvasScroll, { passive: true });

    // Plan view switcher
    setupPlanViewSwitcher();

    // Live refresh: re-fetch the active campaign when the plan changes (item status,
    // session link, new item, progress). The store emits PLAN_UPDATED, the server
    // broadcasts it to /ws, and websocket.js re-emits it on the client bus.
    if (typeof ClientEventBus !== 'undefined') {
        ClientEventBus.on('PLAN_UPDATED', () => scheduleCampaignRefresh());
    }

    // Load campaigns — auto-selects first, or the specified one
    const initialId = window.INITIAL_CAMPAIGN_ID;
    if (initialId) {
        state.activeCampaignId = initialId;
        loadCampaigns().then(() => openCampaign(initialId));
    } else {
        loadCampaigns();
    }
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
    state.typeFilter = '';
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

// Live refresh of the open campaign (debounced) — re-fetch its tree and re-render,
// preserving the selected item so the inspector reflects the change without a reload.
let _planRefreshTimer = null;
function scheduleCampaignRefresh() {
    if (_planRefreshTimer) clearTimeout(_planRefreshTimer);
    _planRefreshTimer = setTimeout(() => {
        _planRefreshTimer = null;
        refreshActiveCampaign().catch(() => {});
    }, 400);
}
async function refreshActiveCampaign() {
    if (!state.activeCampaignId) return;
    const keep = state.selectedItemId;
    await loadDocument(state.activeCampaignId);
    if (!state.docData) return;
    renderAll();
    // Don't clobber an in-progress spec edit with a re-fetch.
    if (keep && !state.editingSpec) selectItem(keep).catch(() => {});
}

// Handle browser back/forward
window.addEventListener('popstate', e => {
    const s = e.state;
    if (s && s.campaignId) {
        openCampaign(s.campaignId);
    } else if (state.allCampaigns.length > 0) {
        openCampaign(state.allCampaigns[0].campaign.id);
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
        html += `<div class="nav-campaign-item ${isActive ? 'active' : ''}" data-action="open-campaign" data-id="${esc(c.id)}">
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
        case 'timeline': renderTimelinePlanView(); break;
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
            `<a class="ref-badge" data-action="scroll-to" data-target="ref-${n}">[${n}]</a>`
        ).join('');
    }

    // Dependency links
    let depLinks = '';
    if (item.dependencies && item.dependencies.length > 0) {
        const depStr = item.dependencies.map(d =>
            `<a data-action="navigate-item" data-id="${d.id}">${esc(d.title)}</a>`
        ).join(', ');
        depLinks += `<span class="doc-dep-link">\u2190 needs: ${depStr}</span>`;
    }
    if (item.dependents && item.dependents.length > 0) {
        const dntStr = item.dependents.map(d =>
            `<a data-action="navigate-item" data-id="${d.id}">${esc(d.title)}</a>`
        ).join(', ');
        depLinks += `<span class="doc-dep-link">\u2192 blocks: ${dntStr}</span>`;
    }

    const footer = (refBadges || depLinks)
        ? `<div class="doc-item-footer">${refBadges}${depLinks}</div>`
        : '';

    return `<div class="doc-item" id="item-${item.id}" data-item-id="${item.id}" data-action="select-item" data-id="${item.id}">
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
            html += `<div class="version-entry ${isCurrent ? 'current' : ''}" data-action="view-version" data-version-id="${v.version_id}" data-is-current="${isCurrent}">
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
    // A re-fetch of the same item (e.g. after saving) keeps view mode; switching
    // to a different item always lands in read-only.
    if (itemId !== state.selectedItemId) {
        state.editingSpec = false;
        state._specError = '';
        state._sessionPickerOpen = false;
        state._availableSessions = null;
    }
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
    state._inspectorData = data;
    const item = data.item;
    const deps = data.dependencies || [];
    const dnts = data.dependents || [];
    // Build a metadata map from the campaign-level sessions included in the payload,
    // then derive the per-item sessions list from item.session_ids (the true source
    // of truth). data.sessions is the campaign pool; we only show sessions that are
    // actually linked to THIS item.
    const _sessionMeta = {};
    (data.sessions || []).forEach(s => { _sessionMeta[s.session_id || s.id] = s; });
    const sessions = (item.session_ids || []).map(sid => _sessionMeta[sid] || { session_id: sid, id: sid });

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

    // Run affordance — only for an actionable imaging item. Routes through the
    // agent (it applies this item's spec via execute_plan_item), in keeping with
    // the agent-first paradigm.
    if (item.type === 'imaging' && item.status === 'planned') {
        html += `<div class="detail-actions" style="margin:4px 0 14px">
            <button data-action="run-item" data-id="${item.id}"
                style="background:var(--accent,#2f6df6);color:#fff;border:0;border-radius:9px;padding:9px 16px;font:inherit;font-size:13px;font-weight:600;cursor:pointer">
                ▶ Run this imaging item
            </button>
            <span style="margin-left:10px;color:var(--text-muted,#94a3b8);font-size:12px">Hands it to the agent to apply the spec and start</span>
        </div>`;
    }

    // Description
    if (item.description) {
        html += section('Description', `<div class="detail-section-content">${esc(item.description)}</div>`);
    }

    // Outcome
    if (item.outcome) {
        html += section('Outcome', `<div class="detail-section-content">${esc(item.outcome)}</div>`);
    }

    // Imaging spec — view, or edit/fill inline (the laser-power loop). Shown for any
    // imaging item even when no spec is set yet, so empty fields can be filled.
    if (item.type === 'imaging' || item.imaging_spec) {
        const spec = item.imaging_spec || {};
        if (state.editingSpec) {
            html += section('Imaging Specification', renderSpecEditor(spec));
        } else {
            const rows = renderSpecTable(spec);
            const content = rows
                ? `<table class="spec-table">${rows}</table>`
                : '<div class="detail-section-content" style="color:var(--text-muted);font-style:italic">No parameters set yet</div>';
            const editBtn = '<button class="spec-edit-btn" data-action="spec-edit">✎ Edit</button>';
            html += section('Imaging Specification', content, editBtn);
        }
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
            return `<span class="dep-chip" data-action="navigate-item" data-id="${d.id}">
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
            return `<span class="dep-chip" data-action="navigate-item" data-id="${d.id}">
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

    // Sessions — item-scoped (item.session_ids), with link/delink controls.
    const _linkBtn = `<button class="session-link-btn" data-action="session-picker-open">+ link session</button>`;
    let sessHtml = '';
    if (sessions.length > 0) {
        sessions.forEach(s => {
            const sid = s.session_id || s.id || '';
            const name = s.name || s.planned_intent || sid || 'Session';
            sessHtml += `<div class="detail-session">
                <span class="detail-session-title">${esc(name)}</span>
                <span class="detail-session-right">
                    ${s.created_at ? `<span class="detail-session-date">${formatDate(s.created_at)}</span>` : ''}
                    <button class="session-delink-btn" data-action="session-delink"
                        data-session-id="${esc(sid)}" title="Delink session">×</button>
                </span>
            </div>`;
        });
    } else {
        sessHtml = '<div class="detail-session-empty">No linked sessions</div>';
    }
    // Inline link picker — rendered when openSessionPicker() has set state flag + loaded data.
    if (state._sessionPickerOpen) {
        if (state._availableSessions === null) {
            // Still loading — show spinner text; will re-render once fetch completes.
            sessHtml += `<div class="session-picker session-picker--loading">Loading sessions…</div>`;
        } else {
            const _linkedIds = new Set(item.session_ids || []);
            const _available = state._availableSessions.filter(s => !_linkedIds.has(s.session_id));
            const _opts = _available.length === 0
                ? `<option value="">No other sessions available</option>`
                : _available.map(s => `<option value="${esc(s.session_id)}">${esc(s.name || s.session_id)}</option>`).join('');
            sessHtml += `<div class="session-picker">
                <select class="session-picker-select" id="session-picker-select">${_opts}</select>
                <div class="session-picker-actions">
                    <button class="session-picker-link-btn" data-action="session-picker-link">Link</button>
                    <button class="session-picker-cancel-btn" data-action="session-picker-cancel">Cancel</button>
                </div>
            </div>`;
        }
    }
    html += section('Sessions', sessHtml, _linkBtn);

    if ($inspectorBody) $inspectorBody.innerHTML = html;
}

// Editable imaging-spec form. Lists every fillable field — empty ones included,
// flagged — so a TBD value (e.g. laser power) is obvious and one click away.
function renderSpecEditor(spec) {
    let rows = '';
    for (const key of IMAGING_SPEC_FIELDS) {
        const label = SPEC_LABELS[key] || key;
        const val = spec[key];
        const has = val != null && val !== '';
        const numeric = SPEC_NUMERIC.has(key);
        const unit = SPEC_UNITS[key]
            ? `<span class="spec-edit-unit">${esc(SPEC_UNITS[key].trim())}</span>` : '';
        const rowCls = has ? 'spec-edit-row' : 'spec-edit-row spec-edit-row--empty';
        rows += `<div class="${rowCls}">
            <label class="spec-edit-label" for="spec-${key}">${esc(label)}</label>
            <span class="spec-edit-field">
                <input id="spec-${key}" class="spec-edit-input" data-spec-key="${key}"
                    type="${numeric ? 'number' : 'text'}"${numeric ? ' step="any"' : ''}
                    value="${has ? esc(String(val)) : ''}" placeholder="not set">${unit}
            </span>
        </div>`;
    }
    const err = state._specError
        ? `<div class="spec-edit-error">${esc(state._specError)}</div>` : '';
    return `<div class="spec-editor">
        ${rows}
        ${err}
        <div class="spec-edit-actions">
            <button class="spec-save-btn" data-action="spec-save">Save</button>
            <button class="spec-cancel-btn" data-action="spec-cancel">Cancel</button>
        </div>
    </div>`;
}

function startSpecEdit() {
    if (!state._inspectorData) return;
    state.editingSpec = true;
    state._specError = '';
    renderInspector(state._inspectorData);
}

function cancelSpecEdit() {
    state.editingSpec = false;
    state._specError = '';
    if (state._inspectorData) renderInspector(state._inspectorData);
}

// Collect changed/filled fields and PATCH them. The store fires PLAN_UPDATED,
// which live-refreshes the plan; we also re-fetch the inspector for immediacy.
async function saveSpecEdit() {
    const data = state._inspectorData;
    const item = data && data.item;
    const campaignId = state.activeCampaignId;
    if (!item || !campaignId) return;

    const orig = item.imaging_spec || {};
    const specPatch = {};
    document.querySelectorAll('#inspector-body [data-spec-key]').forEach(inp => {
        const key = inp.dataset.specKey;
        const raw = inp.value.trim();
        const hadVal = orig[key] != null && orig[key] !== '';
        if (raw === '') {
            if (hadVal) specPatch[key] = '';   // cleared an existing value → unset
            return;                            // stayed empty → skip
        }
        let v = raw;
        if (SPEC_NUMERIC.has(key)) {
            const n = Number(raw);
            if (!Number.isNaN(n)) v = n;
        }
        if (String(orig[key] ?? '') !== String(v)) specPatch[key] = v;
    });

    state.editingSpec = false;
    state._specError = '';
    if (Object.keys(specPatch).length === 0) {
        selectItem(item.id).catch(() => {});   // nothing changed — just leave edit mode
        return;
    }

    try {
        const res = await fetch(
            `/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(item.id)}`,
            {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ spec: specPatch }),
            },
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        selectItem(item.id).catch(() => {});    // refresh inspector now; PLAN_UPDATED refreshes the plan
    } catch (err) {
        console.error('Failed to save spec:', err);
        state.editingSpec = true;
        state._specError = 'Could not save — try again.';
        renderInspector(data);
    }
}

// ── Session link / delink ─────────────────────────────────────────────────────

// Open the inline session picker. Fetches /api/sessions and re-renders with the
// picker shown. Two-phase: immediate re-render with loading state, then again
// once the fetch resolves (mirrors the pattern of selectItem loading state).
async function openSessionPicker() {
    state._sessionPickerOpen = true;
    state._availableSessions = null;   // triggers loading display
    if (state._inspectorData) renderInspector(state._inspectorData);
    try {
        const res = await fetch('/api/sessions');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        state._availableSessions = body.sessions || [];
    } catch (err) {
        console.error('Failed to load sessions for picker:', err);
        state._availableSessions = [];
    }
    if (state._sessionPickerOpen && state._inspectorData) {
        renderInspector(state._inspectorData);
    }
}

function cancelSessionPicker() {
    state._sessionPickerOpen = false;
    state._availableSessions = null;
    if (state._inspectorData) renderInspector(state._inspectorData);
}

// Read the picker <select>, POST to the link endpoint, then refresh the inspector.
async function submitSessionLink() {
    const select = document.getElementById('session-picker-select');
    const sessionId = select && select.value;
    if (!sessionId) return;
    const data = state._inspectorData;
    const item = data && data.item;
    const campaignId = state.activeCampaignId;
    if (!item || !campaignId) return;
    // Close picker before the async call so a re-render doesn't reopen it.
    state._sessionPickerOpen = false;
    state._availableSessions = null;
    try {
        const res = await fetch(
            `/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(item.id)}/sessions`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId }),
            },
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        selectItem(item.id).catch(() => {});   // re-fetch → re-render with updated session_ids
    } catch (err) {
        console.error('Failed to link session:', err);
        if (state._inspectorData) renderInspector(state._inspectorData);
    }
}

// DELETE the session→item edge then refresh the inspector.
async function handleSessionDelink(sessionId) {
    if (!sessionId) return;
    const data = state._inspectorData;
    const item = data && data.item;
    const campaignId = state.activeCampaignId;
    if (!item || !campaignId) return;
    try {
        const res = await fetch(
            `/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(item.id)}/sessions/${encodeURIComponent(sessionId)}`,
            { method: 'DELETE' },
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        selectItem(item.id).catch(() => {});
    } catch (err) {
        console.error('Failed to delink session:', err);
        renderInspector(state._inspectorData);
    }
}

// Hand a planned imaging item to the agent to execute. The agent resolves the
// item ref, applies its spec, and starts the timelapse (execute_plan_item). We
// open the chat so the user sees it pick up and can confirm/adjust.
function runPlanItem(id) {
    if (typeof AgentChat === 'undefined' || !AgentChat.runCommand) return;
    AgentChat.runCommand(`Start imaging for plan item ${id}`);
    if (AgentChat.togglePanel) AgentChat.togglePanel(true);
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
        let html = `<div class="version-item ${!state.viewingSnapshotId ? 'active' : ''}" data-action="back-to-current">
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
            html += `<div class="version-item ${active ? 'active' : ''}" data-action="view-version" data-version-id="${v.version_id}" data-is-current="false">
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

function section(title, content, headerAction) {
    const extra = headerAction ? `<span class="detail-section-action">${headerAction}</span>` : '';
    const titleCls = headerAction ? 'detail-section-title detail-section-title--row' : 'detail-section-title';
    return `<div class="detail-section"><div class="${titleCls}">${title}${extra}</div>${content}</div>`;
}

function renderSpecTable(spec) {
    let rows = '';
    for (const [key, value] of Object.entries(spec)) {
        if (value == null || key.startsWith('_')) continue;
        // Skip nested objects (e.g. provenance metadata) — they'd render as
        // "[object Object]". Field-level provenance isn't a spec value to show here.
        if (typeof value === 'object' && !Array.isArray(value)) continue;
        const label = SPEC_LABELS[key] || key;
        let display = Array.isArray(value) ? value.join(', ') : String(value);
        if (SPEC_UNITS[key]) display += SPEC_UNITS[key];
        rows += `<tr><td>${esc(label)}</td><td>${esc(display)}</td></tr>`;
    }
    return rows;
}

function esc(str) { return escapeHtml(str); }

// formatDate is now in utils.js


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

// Timeline-only: tint bars by task type so the filter highlight has something to pop.
const TYPE_COLORS = {
    imaging: '#3b82f6',          // blue
    bench: '#10b981',            // green
    genetics: '#a855f7',         // purple
    analysis: '#06b6d4',         // cyan
    decision_point: '#f59e0b',   // amber
};

function setupPlanViewSwitcher() {
    initViewSwitcher('plan-view-switcher', switchPlanView, {
        views: ['doc', 'graph', 'board', 'decide', 'matrix', 'timeline'],
        guard: () => !!state.docData
    });
}

function switchPlanView(viewName) {
    if (!state.docData) return;
    state.planView = viewName;
    updateViewButtons('plan-view-switcher', viewName);
    renderCanvas();
}

// Type filter — shared across views
function buildTypeFilterBar(items) {
    const types = [...new Set(items.map(i => i.type))];
    const buttons = types.map(t =>
        `<button class="graph-filter-btn ${state.typeFilter === t ? 'active' : ''}" data-action="filter-type" data-filter-type="${t}">${TYPE_ICONS[t] || ''} ${esc(t)}</button>`
    ).join('');
    return `<div class="type-filter-bar">
        <button class="graph-filter-btn ${state.typeFilter === '' ? 'active' : ''}" data-action="filter-type" data-filter-type="">All</button>
        ${buttons}
    </div>`;
}

function applyTypeFilter(type) {
    if (state.planView === 'graph') {
        // Graph uses CSS dimming to preserve layout
        filterGraphByType(type);
    } else {
        state.typeFilter = type;
        renderCanvas();
    }
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
    const allItems = collectAllItems(doc);
    const items = state.typeFilter ? allItems.filter(i => i.type === state.typeFilter) : allItems;
    const columns = ['planned', 'in_progress', 'completed', 'skipped', 'blocked'];
    const labels = { planned: 'Planned', in_progress: 'In Progress', completed: 'Completed', skipped: 'Skipped', blocked: 'Blocked' };
    const grouped = {};
    columns.forEach(c => grouped[c] = []);
    items.forEach(item => {
        const col = grouped[item.status] || grouped.planned;
        col.push(item);
    });

    let html = buildTypeFilterBar(allItems) + '<div class="board-view">';
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
            html += `<div class="board-card" data-action="select-item" data-id="${item.id}">
                <div class="board-card-top">
                    <span class="board-card-icon">${icon}</span>
                    <span class="board-card-phase">P${item._phaseNum}</span>
                </div>
                <div class="board-card-title">${esc(item.title)}</div>
                ${specLine ? `<div class="board-card-spec">${esc(specLine)}</div>` : ''}
                ${item.estimated_days ? `<div class="board-card-duration">${item.estimated_days}d</div>` : ''}
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
            `<span class="dep-chip" data-action="select-item" data-id="${d.id}">${STATUS_DOTS[d.status] || '○'} ${esc(d.title)}</span>`
        ).join('');
        const dependents = (item.dependents || []).map(d =>
            `<span class="dep-chip" data-action="select-item" data-id="${d.id}">${esc(d.title)}</span>`
        ).join('');
        const statusClass = item.status === 'completed' ? 'completed' : item.status === 'in_progress' ? 'active' : '';

        html += `<div class="decide-card ${statusClass}" data-action="select-item" data-id="${item.id}">
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

    // Group items by phase (swim lanes)
    const phases = {};
    const phaseOrder = [];
    items.forEach(i => {
        if (!phases[i._phaseNum]) {
            phases[i._phaseNum] = { name: i._phase, items: [] };
            phaseOrder.push(i._phaseNum);
        }
        phases[i._phaseNum].items.push(i);
    });

    // Sort items within each phase by phase_order (their natural sequence)
    Object.values(phases).forEach(p => p.items.sort((a, b) => (a.phase_order || 0) - (b.phase_order || 0)));

    // Layout constants
    const nodeW = 200, nodeH = 64, gapX = 40, gapY = 16;
    const laneGap = 20, lanePadX = 16, lanePadY = 28, pad = 20;

    // Compute positions: each phase is a swim lane row
    const pos = {};
    let currentY = pad;
    const laneYRanges = [];

    phaseOrder.forEach(pNum => {
        const phase = phases[pNum];
        const laneTop = currentY;
        const itemsY = currentY + lanePadY;

        // Lay items left-to-right within the lane
        phase.items.forEach((item, idx) => {
            pos[item.id] = {
                x: pad + lanePadX + idx * (nodeW + gapX),
                y: itemsY
            };
        });

        const laneH = lanePadY + nodeH + lanePadY;
        laneYRanges.push({ pNum, name: phase.name, top: laneTop, height: laneH });
        currentY += laneH + laneGap;
    });

    const maxItemsPerPhase = Math.max(...Object.values(phases).map(p => p.items.length));
    const svgW = pad * 2 + lanePadX * 2 + maxItemsPerPhase * (nodeW + gapX) - gapX;
    const svgH = currentY;

    // Render swim lane backgrounds
    const phaseColors = ['rgba(96,165,250,0.07)', 'rgba(52,211,153,0.07)', 'rgba(251,191,36,0.07)', 'rgba(167,139,250,0.07)', 'rgba(248,113,113,0.07)', 'rgba(236,72,153,0.07)'];
    let bands = '';
    laneYRanges.forEach((lane, ci) => {
        const color = phaseColors[ci % phaseColors.length];
        const shortName = lane.name.split(' — ').pop();
        bands += `<rect x="${pad}" y="${lane.top}" width="${svgW - pad * 2}" height="${lane.height}" rx="10" fill="${color}" />`;
        bands += `<text x="${pad + 10}" y="${lane.top + 16}" class="graph-phase-label">P${lane.pNum} — ${esc(shortName)}</text>`;
    });

    // Render edges (hidden by default, shown on node hover)
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

    // Render nodes
    let nodes = '';
    items.forEach(i => {
        const p = pos[i.id];
        if (!p) return;
        const icon = TYPE_ICONS[i.type] || '';
        const color = STATUS_COLORS[i.status] || STATUS_COLORS.planned;
        const statusLabel = STATUS_LABELS[i.status] || i.status;
        nodes += `<g class="graph-node" data-action="select-item" data-id="${i.id}" data-item-id="${i.id}" data-item-type="${i.type}">
            <rect x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="8"
                  fill="var(--bg-card)" stroke="${color}" stroke-width="2" />
            <foreignObject x="${p.x + 8}" y="${p.y + 4}" width="${nodeW - 16}" height="${nodeH - 8}">
                <div xmlns="http://www.w3.org/1999/xhtml" class="graph-node-inner">
                    <div class="graph-node-top"><span>${icon}</span><span class="graph-node-phase">P${i._phaseNum}</span></div>
                    <div class="graph-node-title">${esc(i.title)}</div>
                    <div class="graph-node-status" style="color:${color}">${STATUS_DOTS[i.status]} ${statusLabel}</div>
                </div>
            </foreignObject>
        </g>`;
    });

    $canvasContent.innerHTML = `<div class="graph-view" id="graph-view-container">
        ${buildTypeFilterBar(items)}
        <svg width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">
            ${bands}${edges}${nodes}
        </svg>
    </div>`;

    // Hover: show edges connected to hovered node
    const svg = $canvasContent.querySelector('svg');
    if (svg) {
        svg.addEventListener('mouseover', e => {
            const node = e.target.closest('.graph-node');
            if (!node) return;
            const id = node.dataset.itemId;
            svg.classList.add('graph-hover-active');
            node.classList.add('graph-node-hover');
            svg.querySelectorAll('.graph-edge').forEach(edge => {
                if (edge.dataset.from === id || edge.dataset.to === id) {
                    edge.classList.add('graph-edge-active');
                    // Highlight connected node
                    const otherId = edge.dataset.from === id ? edge.dataset.to : edge.dataset.from;
                    svg.querySelector(`.graph-node[data-item-id="${otherId}"]`)?.classList.add('graph-node-connected');
                }
            });
        });
        svg.addEventListener('mouseout', e => {
            const node = e.target.closest('.graph-node');
            if (!node) return;
            svg.classList.remove('graph-hover-active');
            svg.querySelectorAll('.graph-node-hover, .graph-node-connected').forEach(el => {
                el.classList.remove('graph-node-hover', 'graph-node-connected');
            });
            svg.querySelectorAll('.graph-edge-active').forEach(el => el.classList.remove('graph-edge-active'));
        });
    }
}

// filterGraphByType kept for graph-specific CSS dimming (preserves layout while filtering)
function filterGraphByType(type) {
    state.typeFilter = type;
    // Update shared filter bar
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
    if (type) {
        const activeIds = new Set();
        document.querySelectorAll(`.graph-node:not(.dimmed)`).forEach(n => activeIds.add(n.dataset.itemId));
        document.querySelectorAll('.graph-edge').forEach(edge => {
            if (activeIds.has(edge.dataset.from) || activeIds.has(edge.dataset.to)) {
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
    const allItems = collectAllItems(doc);
    const phases = (doc.children || []).filter(c => c.children);
    const types = state.typeFilter ? [state.typeFilter] : ['imaging', 'bench', 'genetics', 'analysis', 'decision_point'];
    const typeLabels = { imaging: 'Imaging', bench: 'Bench', genetics: 'Genetics', analysis: 'Analysis', decision_point: 'Decisions' };

    let html = buildTypeFilterBar(allItems) + '<div class="matrix-view"><table class="matrix-table"><thead><tr><th>Phase</th>';
    types.forEach(t => html += `<th>${typeLabels[t] || esc(t)}</th>`);
    if (types.length > 1) html += '<th>Total</th>';
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
        if (types.length > 1) html += `<td class="matrix-total">${rowTotal}</td>`;
        html += '</tr>';
    });

    // Totals row
    html += '<tr class="matrix-totals-row"><td>Total</td>';
    types.forEach(t => {
        html += `<td class="matrix-total">${colTotals[t].total}</td>`;
    });
    if (types.length > 1) html += `<td class="matrix-total">${grandTotal}</td>`;
    html += '</tr>';
    html += '</tbody></table></div>';
    $canvasContent.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
//  TIMELINE VIEW (Gantt-style relative timeline from Day 0)
// ══════════════════════════════════════════════════════════

function renderTimelinePlanView() {
    const doc = state.docData?.document;
    if (!doc) return;
    const allItems = collectAllItems(doc);
    if (allItems.length === 0) {
        $canvasContent.innerHTML = '<div class="empty-state"><p>No items to schedule</p></div>';
        return;
    }
    // Compute schedule using ALL items (dependency chain), then filter for display
    const items = allItems;

    // Build dependency map and compute earliest start days via topological sort
    const byId = {};
    items.forEach(i => byId[i.id] = i);
    const startDay = {};
    const endDay = {};

    // Topological order
    const inDeg = {};
    const out = {};
    items.forEach(i => { inDeg[i.id] = 0; out[i.id] = []; });
    items.forEach(i => {
        (i.dependencies || []).forEach(d => {
            if (byId[d.id]) {
                out[d.id].push(i.id);
                inDeg[i.id]++;
            }
        });
    });

    const queue = items.filter(i => inDeg[i.id] === 0).map(i => i.id);
    queue.forEach(id => startDay[id] = 0);
    let head = 0;
    while (head < queue.length) {
        const id = queue[head++];
        const dur = byId[id].estimated_days || 1;
        endDay[id] = startDay[id] + dur;
        out[id].forEach(next => {
            startDay[next] = Math.max(startDay[next] || 0, endDay[id]);
            inDeg[next]--;
            if (inDeg[next] === 0) queue.push(next);
        });
    }
    // Handle cycles
    items.forEach(i => {
        if (startDay[i.id] === undefined) startDay[i.id] = 0;
        if (endDay[i.id] === undefined) endDay[i.id] = (startDay[i.id] || 0) + (i.estimated_days || 1);
    });

    const totalDays = Math.max(1, ...Object.values(endDay));
    const dayWidth = 36;
    const rowHeight = 32;
    const labelWidth = 220;
    const headerHeight = 40;
    const chartWidth = totalDays * dayWidth;

    // Sort items by start day then phase. Filter dims non-matching rows rather than hiding them.
    const sorted = [...items].sort((a, b) => (startDay[a.id] || 0) - (startDay[b.id] || 0) || a._phaseNum - b._phaseNum);
    const isDim = (item) => state.typeFilter && item.type !== state.typeFilter;

    // Day header
    let dayHeaders = '';
    for (let d = 0; d < totalDays; d++) {
        const isWeek = d % 7 === 0;
        dayHeaders += `<div class="tl-day-header ${isWeek ? 'week' : ''}" style="left:${d * dayWidth}px;width:${dayWidth}px">
            ${isWeek ? `W${Math.floor(d / 7) + 1}` : d + 1}
        </div>`;
    }

    const chartHeight = sorted.length * rowHeight;

    $canvasContent.innerHTML = `${buildTypeFilterBar(allItems)}<div class="timeline-plan-view">
        <div class="tl-header-row">
            <div class="tl-corner">Task</div>
            <div class="tl-header-scroll" id="tl-header-scroll">
                <div class="tl-header-days" style="width:${chartWidth}px">${dayHeaders}</div>
            </div>
        </div>
        <div class="tl-body">
            <div class="tl-labels" style="height:${chartHeight}px">
                ${sorted.map((item, idx) => {
                    const icon = TYPE_ICONS[item.type] || '';
                    const dim = isDim(item) ? ' dim' : '';
                    return `<div class="tl-label${dim}" style="top:${idx * rowHeight}px" data-action="select-item" data-id="${item.id}" title="${esc(item.title)}">
                        <span class="tl-label-icon">${icon}</span>
                        <span class="tl-label-phase">P${item._phaseNum}</span>
                        <span class="tl-label-title">${esc(item.title)}</span>
                    </div>`;
                }).join('')}
            </div>
            <div class="tl-chart-scroll" id="tl-chart-scroll">
                <div class="tl-chart" style="width:${chartWidth}px;height:${chartHeight}px;background:repeating-linear-gradient(90deg,var(--border) 0,var(--border) 1px,transparent 1px,transparent ${dayWidth}px)">
                    ${sorted.map((item, idx) => {
                        const start = startDay[item.id] || 0;
                        const dur = item.estimated_days || 1;
                        const fill = TYPE_COLORS[item.type] || STATUS_COLORS.planned;
                        const statusAccent = item.status && item.status !== 'planned' ? (STATUS_COLORS[item.status] || '') : '';
                        const dim = isDim(item) ? ' dim' : '';
                        const styles = `top:${idx * rowHeight + 4}px;left:${start * dayWidth}px;width:${dur * dayWidth - 2}px;background:${fill}` +
                                       (statusAccent ? `;box-shadow:inset 4px 0 0 ${statusAccent}` : '');
                        return `<div class="tl-bar${dim}" style="${styles}"
                                     data-action="select-item" data-id="${item.id}" title="Day ${start + 1}–${start + dur}: ${esc(item.title)} (${dur}d)">
                            ${dur > 1 ? `<span class="tl-bar-text">${dur}d</span>` : ''}
                        </div>`;
                    }).join('')}
                </div>
            </div>
        </div>
        <div class="tl-summary">Total: ${totalDays} days (${Math.ceil(totalDays / 7)} weeks) \u00B7 Critical path determines minimum duration</div>
    </div>`;

    // Sync horizontal scroll between header and chart
    const chartScroll = document.getElementById('tl-chart-scroll');
    const headerScroll = document.getElementById('tl-header-scroll');
    if (chartScroll && headerScroll) {
        chartScroll.addEventListener('scroll', () => { headerScroll.scrollLeft = chartScroll.scrollLeft; });
        // Wheel scroll → horizontal pan. Leave native deltaX (trackpad) and shift+wheel alone.
        chartScroll.addEventListener('wheel', (e) => {
            if (e.deltaY !== 0 && e.deltaX === 0 && !e.shiftKey) {
                e.preventDefault();
                chartScroll.scrollLeft += e.deltaY;
            }
        }, { passive: false });
    }
}

// Expose only what's needed by other modules
// - CampaignsApp.init: called from app.js to boot the campaigns page
// - openCampaign: called from app.js hash router
// All other actions are handled via data-action event delegation above.
window.CampaignsApp = { init: boot };
window.openCampaign = openCampaign;

// Auto-init on standalone campaigns page (detected via data-page attribute)
if (document.body?.dataset.page === 'campaigns') {
    boot();
} else {
    document.addEventListener('DOMContentLoaded', () => {
        if (document.body.dataset.page === 'campaigns') boot();
    });
}

})(); // end IIFE
