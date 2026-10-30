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
    skipped: '\u2298', blocked: '\u2297', proposed: '\u25CC',
};
const STATUS_LABELS = {
    active: 'Active', completed: 'Completed', paused: 'Paused',
    planned: 'Planned', in_progress: 'In Progress',
    skipped: 'Skipped', blocked: 'Blocked', proposed: 'Suggested',
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
    editingSpec: false,         // inspector imaging-spec edit mode
    _inspectorData: null,       // last item-detail payload (for re-render on edit toggle)
    _specError: '',             // inline save error in the spec editor
    _sessionPickerOpen: false,  // whether the session link picker is visible
    _availableSessions: null,   // null = not yet loaded, [] = loaded (for link picker)
    // The project layer: a root campaign is a project; its page has sections.
    section: 'overview',        // 'overview' | 'plan' | 'sessions' | 'findings' | 'notes'
    ledger: null,               // /api/campaigns/{id}/ledger
    projSessions: null,         // /api/sessions?project_id=
    findings: null,             // finding notes of the project
    notes: null,                // all notes of the project
    sectionError: '',
    openForm: null,             // the one inline form or menu open, e.g. 'item', 'answer:<id>'
    newProjectOpen: false,
    editHead: false,       // the header's question and short name are inputs
    search: null,               // the add popover: {q, groups, sel, rows, ctrls, timer}
    searchMsg: '',
    planMsg: '',
    unfiled: null,              // sessions offered by "+ add a session"
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
// Home hands over the project to open (see home.js) through sessionStorage.
// app.js calls CampaignsApp.init() on every switch to this tab, so the handover
// is taken every time, not only at the first boot.
function takeHandover() {
    try {
        const wanted = sessionStorage.getItem('gently-open-project');
        if (wanted) sessionStorage.removeItem('gently-open-project');
        return wanted || null;
    } catch (_) { return null; }
}

function boot() {
    const handed = takeHandover();
    if (_initialized) {
        if (handed) openCampaign(handed).catch(() => {});
        return;
    }
    _initialized = true;
    if (handed) window.INITIAL_CAMPAIGN_ID = handed;
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
        // A ledger card is itself the way to its details; a click on one of
        // its controls, fields or menus is not.
        if (el.classList.contains('pj-entry') && e.target.closest('input, select, textarea, label, a, .pj-form, .pj-menu, .pj-act, .pj-tools')) return;
        switch (action) {
            case 'select-item':
                // A card already open in the side panel closes it on a second click.
                if (el.classList.contains('pj-entry') && state.selectedItemId === id && $workspace?.classList.contains('inspector-open')) closeInspector();
                else selectItem(id);
                break;
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
            default: projectAction(action, el, e); break;
        }
    });
    // The add popover and the inline forms: typing, keys, selects.
    document.addEventListener('input', e => {
        if (e.target && e.target.id === 'pj-search-input') onSearchInput(e.target.value);
        if (e.target && e.target.dataset && e.target.dataset.field === 'title' && e.target.closest('.pj-form[data-form="item"]')) previewIntent(e.target.value);
    });
    document.addEventListener('keydown', e => {
        if (e.target && e.target.id === 'pj-search-input') onSearchKey(e);
        else if (state.editHead && e.target.closest?.('#pj-head [data-field]')) {
            if (e.key === 'Enter') { e.preventDefault(); projectAction('head-save', e.target, e); }
            else if (e.key === 'Escape') projectAction('head-cancel', e.target, e);
        }
    });
    document.addEventListener('change', e => {
        const el = e.target.closest ? e.target.closest('select[data-action]') : null;
        if (el && el.value) projectAction(el.dataset.action, el, e);
    });
    if (typeof ClientEventBus !== 'undefined') {
        ClientEventBus.on('CONTEXT_UPDATED', () => scheduleCampaignRefresh());
    }
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
    showLoading('Loading projects...');
    try {
        const res = await fetch('/api/campaigns');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        state.allCampaigns = data.campaigns || [];

        hideLoading();
        renderNavigator();

        // Auto-select first campaign if none active
        if (!state.activeCampaignId && state.allCampaigns.length > 0) {
            openCampaign(state.allCampaigns[0].campaign.id);
        } else if (state.activeCampaignId) {
            renderAll();
        } else {
            $canvasContent.innerHTML = `<div class="empty-state">
                <p>No projects yet</p>
                <span class="empty-hint">“+ New project” above: one line saying what it asks. Or ask Gently to plan one.</span>
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
    state.section = 'overview';
    state.ledger = state.projSessions = state.findings = state.notes = null;
    state.sectionError = '';
    state.openForm = null;
    state.search = null;
    state.searchMsg = state.planMsg = '';
    closeInspector();
    showLoading('Loading project...');

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
            <p>Failed to load the project</p>
            <span class="empty-hint">It may not exist any more</span>
        </div>`;
        updateHeader();
        return;
    }

    renderAll();
    loadSection();
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
    await Promise.all([loadDocument(state.activeCampaignId), refreshSummaries()]);
    if (!state.docData) return;
    renderAll();
    loadSection();
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
    let html = '<div class="nav-section-title">Projects</div>';

    if (state.allCampaigns.length === 0) {
        html += '<div style="padding:16px;color:var(--text-muted);font-size:0.78rem;font-style:italic">No projects yet</div>';
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
            <p>Select a project</p>
        </div>`;
        $canvasContent.classList.remove('pj-with-header');
        return;
    }
    $canvasContent.classList.add('pj-with-header');
    if (state.section !== 'plan') {
        $canvasContent.innerHTML = renderProjectHeader() + renderSection();
        ProjectTools.mount();
        Stream.mount();
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
    // The project header sits above every plan view; the doc's own title
    // card is hidden under it (css .pj-with-header).
    $canvasContent.insertAdjacentHTML('afterbegin', renderProjectHeader() + renderPlanBar());
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

    // A suggestion from the agent: dashed, labelled, and not runnable until accepted.
    const suggested = statusClass === 'proposed';
    const suggestRow = suggested
        ? `<div class="doc-item-suggest"><span class="pj-chip ai">suggested by the agent</span>
            <button data-action="item-accept" data-id="${item.id}">Accept</button>
            <button data-action="select-item" data-id="${item.id}">Edit</button>
            <button data-action="item-remove" data-id="${item.id}">Dismiss</button></div>`
        : '';

    return `<div class="doc-item${suggested ? ' doc-item--suggested' : ''}" id="item-${item.id}" data-item-id="${item.id}" data-action="select-item" data-id="${item.id}">
        <span class="doc-item-status dot-${statusClass}">${dot}</span>
        <span class="doc-item-icon type-${item.type}">${icon}</span>
        <div class="doc-item-body">
            <div class="doc-item-title-row">
                <span class="doc-item-num">${esc(taskNum)}</span>
                <span class="doc-item-title">${esc(item.title)}</span>
                ${item.origin === 'manual' ? '<span class="pj-chip">manual</span>' : ''}
            </div>
            ${specLine ? `<div class="doc-item-spec">${esc(specLine)}</div>` : ''}
            ${suggestRow}
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
            const editBtn = '<button class="spec-edit-btn" data-action="spec-edit">✎ Edit</button>';
            const said = acquisitionSentence(spec);
            const setup = canSetUp(item) ? `<button class="pj-link" data-action="item-setup" data-id="${esc(item.id)}">Set up on Devices</button>` : '';
            const acqRows = renderAcquisitionRows(spec);
            const restRows = renderSpecTable(spec, ACQUISITION_FLAT_KEYS);
            let content = '';
            if (said) content += `<div class="spec-sentence">${esc(said)}${setup ? ` · ${setup}` : ''}</div>`;
            if (acqRows) content += `<div class="spec-sub">Acquisition <small>what Devices › Operate runs</small></div><table class="spec-table">${acqRows}</table>`;
            if (restRows) content += `<div class="spec-sub">Sample and intention</div><table class="spec-table">${restRows}</table>`;
            if (!acqRows && !restRows) content += '<div class="detail-section-content" style="color:var(--text-muted);font-style:italic">No parameters set yet</div>';
            html += section('Imaging Specification', content, editBtn);
            const tactics = renderTacticsList(spec);
            if (tactics) html += section('Tactics', tactics);
            const draft = renderDraftInspector(item);
            if (draft) html += section("Gently's draft", draft);
        }
    }

    // Bench spec
    if (item.bench_spec) {
        html += section('Bench Specification', `<table class="spec-table">${renderSpecTable(item.bench_spec)}</table>`);
    }

    // Dependencies — read here, and edited here: × takes one away, the select
    // adds one. Items that already wait on this one are left out (a cycle).
    {
        const depIds = new Set(deps.map(d => d.id));
        const dntIds = new Set(dnts.map(d => d.id));
        const chips = deps.map(d => {
            const dot = STATUS_DOTS[d.status] || '\u25CB';
            const dc = d.status || 'planned';
            return `<span class="dep-chip">
                <span class="dep-chip-dot dot-${dc}">${dot}</span>
                <a data-action="navigate-item" data-id="${d.id}">${esc(d.title)}</a>
                <button class="dep-remove" data-action="dep-remove" data-id="${esc(item.id)}" data-dep="${d.id}" title="This item no longer waits on that one">×</button>
            </span>`;
        }).join('');
        const candidates = allItems().filter(x => x.id !== item.id && !depIds.has(x.id) && !dntIds.has(x.id));
        const picker = candidates.length
            ? `<select class="dep-add" data-action="dep-add" data-id="${esc(item.id)}"><option value="">+ waits on…</option>${candidates.map(x => `<option value="${esc(x.id)}">${esc(x.title)}</option>`).join('')}</select>`
            : '<span class="pj-muted">nothing else in this project to wait on</span>';
        html += section('Depends on', `<div class="dep-list">${chips || '<span class="pj-muted">nothing</span>'}${picker}</div>`);
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
    if ($headerTitle) $headerTitle.textContent = 'Projects';
    if ($headerBreadcrumb) {
        $headerBreadcrumb.textContent = campaign ? `› ${campaign.shorthand || campaign.description}` : '';
    }
    // The plan's views, versions and print belong to the Plan section only.
    if (state.docData && state.section === 'plan') {
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
            ? `${numCampaigns} project${numCampaigns !== 1 ? 's' : ''} \u00B7 ${total} items \u00B7 ${pct}% complete`
            : `${numCampaigns} project${numCampaigns !== 1 ? 's' : ''}`;
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

// scrollIntoView scrolls every scrollable ancestor, so inside a transformed
// wrapper it pans the whole workspace. Scroll #canvas -- the intended container
// -- explicitly, converting the measured (transform-inclusive) offset back into
// the canvas's own scroll units.
function scrollCanvasToEl(el, centered) {
    const canvas = document.getElementById('canvas');
    if (!canvas) return;
    const canvasRect = canvas.getBoundingClientRect();
    const scale = canvasRect.height / (canvas.offsetHeight || canvasRect.height) || 1;
    const rect = el.getBoundingClientRect();
    const delta = rect.top - canvasRect.top - (centered ? (canvasRect.height - rect.height) / 2 : 0);
    canvas.scrollTo({ top: canvas.scrollTop + delta / scale, behavior: 'smooth' });
}

function navigateToItem(itemId) {
    // Scroll to item in canvas
    const el = document.getElementById(`item-${itemId}`);
    if (el) {
        scrollCanvasToEl(el, true);
    }
    // Open detail
    selectItem(itemId);
}

function scrollCanvasTo(elementId) {
    const el = document.getElementById(elementId);
    if (el) scrollCanvasToEl(el, false);
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

    // Measured rects are transform-inclusive; the 100 threshold below is in CSS
    // px, so divide the offset by the live scale before comparing.
    const canvasRect = canvas.getBoundingClientRect();
    const scale = canvasRect.height / (canvas.offsetHeight || canvasRect.height) || 1;

    document.querySelectorAll('.doc-item').forEach(el => {
        const rect = el.getBoundingClientRect();
        const relTop = (rect.top - canvasRect.top) / scale;
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

// The older flat fields the acquisition table already covers, left out of the
// sample-and-intention table when an acquisition is shown.
const ACQUISITION_FLAT_KEYS = new Set([
    'interval_s', 'num_slices', 'exposure_ms', 'laser_wavelength_nm', 'laser_power_pct',
    'galvo_amplitude', 'piezo_amplitude_um', 'stop_condition', 'adaptive_intervals',
    'acquisition', 'acquisition_derived', 'draft', 'tactics', 'provenance',
]);

function provenanceTag(spec, key) {
    const p = spec && spec.provenance && spec.provenance[key];
    if (!p) return '';
    const word = { sentence: 'from the sentence', model: 'drafted by Gently', typed: 'typed', accepted: 'accepted' }[p.source] || p.source;
    const cls = p.source === 'model' ? 'ai' : '';
    return `<span class="spec-prov ${cls}" title="${esc(p.text || '')}">${esc(word)}</span>`;
}

// The acquisition in the pane's own rows: cadence, channels, ending.
function renderAcquisitionRows(spec) {
    const a = spec && spec.acquisition;
    if (!a || typeof AcquisitionPlan === 'undefined') return '';
    const P = AcquisitionPlan.fromStructure(a);
    const tag = spec.acquisition_derived ? '<span class="spec-prov">read from the older fields</span>' : provenanceTag(spec, 'acquisition');
    const row = (k, v, extra) => `<tr><td>${esc(k)}</td><td>${v}${extra ? ' ' + extra : ''}</td></tr>`;
    const every = P.intervalSeconds % 60 === 0 && P.intervalSeconds >= 60 ? `${P.intervalSeconds / 60} min` : `${P.intervalSeconds} s`;
    let rows = row('Cadence', `every ${esc(every)}`, tag);
    if (P.spim.enabled === false) {
        rows += row('SPIM volumes', 'off — a brightfield-only run');
    } else {
        rows += row('SPIM volumes', [
            P.spim.slices == null ? 'slices as each embryo has' : `${P.spim.slices} slices`,
            P.spim.exposureMs == null ? 'exposure as each embryo has' : `${P.spim.exposureMs} ms`,
        ].map(esc).join(' · '));
        rows += row('Lasers', P.spim.laserConfig ? esc(P.spim.laserConfig) : '<span class="pj-muted">preset as the controller has</span>');
        const powers = Object.entries(P.spim.laserPowers || {});
        if (powers.length) rows += row('Laser power', powers.map(([wl, pct]) => `${esc(wl)} at ${esc(pct)} %`).join(' · '));
    }
    if (P.bf.enabled) {
        const lit = AcquisitionPlan.BF_LIGHTS[P.bf.light] ? AcquisitionPlan.BF_LIGHTS[P.bf.light].says : P.bf.light;
        rows += row('Brightfield overview', [
            P.spim.enabled === false ? 'the run' : (P.bf.everyRounds === 1 ? 'one frame per round' : `one frame every ${P.bf.everyRounds} rounds`),
            lit + (P.bf.light === 'led' && P.bf.ledPct != null ? ` at ${P.bf.ledPct} %` : ''),
            P.bf.exposureMs != null ? `${P.bf.exposureMs} ms` : '',
            P.bf.position === 'here' && P.bf.pins && P.bf.pins.length ? `from ${P.bf.pins.length} position${P.bf.pins.length === 1 ? '' : 's'}` : 'from the centroid',
        ].filter(Boolean).map(esc).join(' · '));
    } else if (P.spim.enabled !== false) {
        rows += row('Brightfield overview', '<span class="pj-muted">off</span>');
    }
    rows += row('Ends', esc(AcquisitionPlan.stopWords(P.stop.kind, P.stop.value, P.spim.enabled === false ? 'frame' : 'timepoint')));
    if (P.overrides && P.overrides.length) rows += row('Per-embryo endings', P.overrides.map(o => `${esc(o.embryoId)} ${esc(AcquisitionPlan.stopWords(o.kind, o.value))}`).join(', '));
    if (P.monitoringMode && P.monitoringMode !== 'idle') rows += row('Monitoring', esc(P.monitoringMode));
    return rows;
}

function renderTacticsList(spec) {
    const list = (spec && spec.tactics) || [];
    if (!list.length) return '';
    return `<div class="detail-section-content">${list.map(t => {
        const st = t.structure || {};
        const steps = Array.isArray(st.steps) ? st.steps : null;
        const stepsText = steps ? `${steps.length} step${steps.length === 1 ? '' : 's'}: ${steps.slice(0, 8).map(x => Object.entries(x).map(([k, v]) => `${k} ${v}`).join(' ')).join(', ')}${steps.length > 8 ? ', …' : ''}` : '';
        const rest = Object.entries(st).filter(([k]) => k !== 'steps').map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ');
        return `<div class="spec-tactic"><span class="pj-chip">${esc(t.kind || 'tactic')}</span> <b>${esc(t.name || '')}</b>${t.target ? ` <span class="pj-muted">· ${esc(t.target)}</span>` : ''}${t.scope ? ` <span class="pj-muted">· ${esc(t.scope)}</span>` : ''}${stepsText ? `<div class="pj-muted">${esc(stepsText)}</div>` : ''}${rest ? `<div class="pj-muted">${esc(rest)}</div>` : ''}</div>`;
    }).join('')}</div>`;
}

function renderDraftInspector(item) {
    const spec = item.imaging_spec || {};
    const d = spec.draft;
    if (!d) return '';
    const label = { drafting: 'Gently is drafting…', drafted: 'drafted by Gently, waiting for you', unavailable: 'no draft: ' + ((d.assumptions || [])[0] || 'no model to ask'), accepted: 'accepted' }[d.status] || d.status;
    const actions = d.status === 'drafted'
        ? `<div class="pj-act"><button data-action="item-accept-spec" data-id="${esc(item.id)}">Accept spec</button><button data-action="item-draft" data-id="${esc(item.id)}">Ask again</button></div>`
        : (d.status === 'unavailable' ? `<div class="pj-act"><button data-action="item-draft" data-id="${esc(item.id)}">Try again</button></div>` : '');
    return `<div class="detail-section-content">
        <div><span class="pj-chip ${d.status === 'accepted' ? '' : 'ai'}">${esc(label)}</span></div>
        ${d.rationale ? `<div class="spec-rationale">${esc(d.rationale)}</div>` : ''}
        ${(d.assumptions || []).length && d.status !== 'unavailable' ? `<ul class="spec-assumptions">${d.assumptions.map(a => `<li>${esc(a)}</li>`).join('')}</ul>` : ''}
        ${d.sentence && d.sentence !== item.title ? `<div class="pj-muted">from: “${esc(d.sentence)}”</div>` : ''}
        ${actions}
    </div>`;
}

function renderSpecTable(spec, skip) {
    let rows = '';
    for (const [key, value] of Object.entries(spec)) {
        if (value == null || key.startsWith('_')) continue;
        if (skip && skip.has(key)) continue;
        // Skip nested objects (e.g. provenance metadata) — they'd render as
        // "[object Object]". Field-level provenance isn't a spec value to show here.
        if (typeof value === 'object' && !Array.isArray(value)) continue;
        const label = SPEC_LABELS[key] || key;
        let display = Array.isArray(value) ? value.map(v => typeof v === 'object' ? JSON.stringify(v) : v).join(', ') : String(value);
        if (SPEC_UNITS[key]) display += SPEC_UNITS[key];
        rows += `<tr><td>${esc(label)}</td><td>${esc(display)} ${provenanceTag(spec, key)}</td></tr>`;
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
    in_progress: 'var(--accent-green)',
    completed: 'var(--accent)',
    skipped: 'var(--text-muted)',
    blocked: 'var(--accent-red)',
    proposed: 'var(--accent-purple)',
};

// Timeline-only: tint bars by task type so the filter highlight has something to pop.
// Okabe-Ito categorical set (colorblind-safe); bars also carry TYPE_ICONS.
const TYPE_COLORS = {
    imaging: '#0072b2',          // blue
    bench: '#009e73',            // bluish green
    genetics: '#cc79a7',         // reddish purple
    analysis: '#e69f00',         // orange
    decision_point: '#f0e442',   // yellow
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
                const dots = matching.map(i => `<span class="matrix-dot" style="background:none;width:auto;height:auto;color:${STATUS_COLORS[i.status]};font-size:10px;line-height:1" title="${esc(i.title)}">${STATUS_DOTS[i.status] || '\u25CB'}</span>`).join('');
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
                            <span class="tl-bar-text">${icon}${dur > 1 ? ` ${dur}d` : ''}</span>
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
        // Only consume the wheel when there is something to pan, so an enclosing
        // zoomable surface still sees the gesture otherwise.
        chartScroll.addEventListener('wheel', (e) => {
            const pannable = chartScroll.scrollWidth > chartScroll.clientWidth;
            if (pannable && e.deltaY !== 0 && e.deltaX === 0 && !e.shiftKey) {
                e.preventDefault();
                // deltaY is screen px; scrollLeft is in the element's own units.
                const rect = chartScroll.getBoundingClientRect();
                const scale = rect.width / (chartScroll.offsetWidth || rect.width) || 1;
                chartScroll.scrollLeft += e.deltaY / scale;
            }
        }, { passive: false });
    }
}

// ══════════════════════════════════════════════════════════
//  THE PROJECT LAYER
//  A root campaign is a project. Its page: a header with the question, the
//  strains and links it keeps, then sections. Every value is read back from
//  the store; nothing is remembered in the browser but the open section.
// ══════════════════════════════════════════════════════════

const enc = encodeURIComponent;

async function api(method, url, body) {
    const init = { method };
    if (body !== undefined) {
        init.headers = { 'Content-Type': 'application/json' };
        init.body = JSON.stringify(body);
    }
    const res = await fetch(url, init);
    if (!res.ok) {
        let detail = '';
        try { detail = (await res.json()).detail || ''; } catch (_) { /* not json */ }
        throw new Error(detail || `HTTP ${res.status}`);
    }
    return res.status === 204 ? null : res.json();
}

// An error is shown beside the control that caused it, never in an alert.
function showErr(el, msg) {
    const host = el?.closest?.('.pj-form, .pj-headform, .pj-entry, .pj-chip, .pj-head, .pj-card, .pj-newform, .pj-planbar') || el?.parentElement;
    if (!host) return;
    let box = host.querySelector(':scope > .pj-err');
    if (!box) {
        box = document.createElement('div');
        box.className = 'pj-err';
        host.appendChild(box);
    }
    box.textContent = msg;
}

function projectNameOf(id) {
    const t = (state.allCampaigns || []).find(t => (t.campaign || {}).id === id);
    return t ? (t.campaign.shorthand || t.campaign.description || id) : String(id).slice(0, 8);
}

function roleOf(s) {
    const link = (s.projects || []).find(x => x.id === state.activeCampaignId);
    const r = link && link.role;
    return r ? `<span class="pj-chip">${esc(r)}</span>` : '<span class="pj-muted">—</span>';
}

function fields(form) {
    const out = {};
    if (!form) return out;
    form.querySelectorAll('[data-field]').forEach(i => { out[i.dataset.field] = (i.value || '').trim(); });
    return out;
}

function focusFirst() {
    const el = document.querySelector('.pj-form [data-field], #new-project-form [data-field]');
    if (el) el.focus();
}

function fmtWhen(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function specProvenanceNote(s) {
    const prov = (s && s.provenance) || {};
    const fromSentence = Object.keys(prov).filter(k => prov[k] && prov[k].source === 'sentence');
    const fromModel = Object.keys(prov).filter(k => prov[k] && prov[k].source === 'model');
    const bits = [];
    if (fromSentence.length) bits.push(`${fromSentence.length} from the sentence`);
    if (fromModel.length) bits.push(`${fromModel.length} implied, by Gently`);
    return bits.length ? ` <span class="pj-prov" title="${esc(fromSentence.concat(fromModel).join(', '))}">· ${esc(bits.join(' · '))}</span>` : '';
}

// The pane's own sentence for the acquisition, when the spec carries one in
// the pane's shape; the older flat fields otherwise.
function acquisitionSentence(s) {
    if (!s || !s.acquisition || typeof AcquisitionPlan === 'undefined') return '';
    const n = Number(s.num_embryos) || 0;
    const subjects = Array.from({ length: n }, (_, i) => ({ id: `e${i + 1}`, label: `e${i + 1}` }));
    try { return AcquisitionPlan.describe(AcquisitionPlan.fromStructure(s.acquisition), subjects); } catch (_) { return ''; }
}

function specSummary(s) {
    if (!s) return '';
    const head = [s.strain, s.temperature_c != null ? `${s.temperature_c} °C` : ''].filter(Boolean).join(' · ');
    let body = acquisitionSentence(s);
    if (!body) {
        body = [
            s.num_embryos ? `${s.num_embryos} embryos` : '',
            s.interval_s ? `every ${s.interval_s} s` : '',
            s.laser_wavelength_nm ? `${s.laser_wavelength_nm} nm` : '',
            s.stop_condition ? `until ${s.stop_condition}` : '',
        ].filter(Boolean).join(' · ');
    }
    const tail = s.success_criteria ? `so that: ${s.success_criteria}` : '';
    return [head, body, tail].filter(Boolean).join(' · ');
}

// An item can be set up on Devices › Operate once its acquisition is the
// person's: typed, accepted, or never drafted.
function canSetUp(it) {
    const s = it.imaging_spec;
    if (!s || !s.acquisition || it.type !== 'imaging') return false;
    const d = s.draft;
    return !d || d.status === 'accepted' || (s.provenance && s.provenance.acquisition && s.provenance.acquisition.source === 'typed');
}

function setUpOnDevices(it) {
    const campaign = state.docData?.document?.campaign || {};
    const detail = {
        id: it.id,
        title: it.title,
        project: { id: state.activeCampaignId, name: campaign.shorthand || campaign.description || '' },
        structure: it.imaging_spec.acquisition,
    };
    try { sessionStorage.setItem('gently-setup-item', JSON.stringify(detail)); } catch (_) { /* fine */ }
    if (typeof switchTab === 'function') switchTab('devices');
    window.dispatchEvent(new CustomEvent('gently:setup-item', { detail }));
}

function treeOf() { return state.docData?.document || {}; }
function phasesOf() { return (treeOf().children || []).map(ch => ch.campaign).filter(Boolean); }
function allItems() {
    const t = treeOf();
    const out = [...(t.items || [])];
    (t.children || []).forEach(ch => out.push(...(ch.items || [])));
    return out;
}
// "1", "2.3": the numbers the Doc view gives items.
// The navigator's list and the header's counts: every root tree, re-read after
// anything that changes what a project holds.
async function refreshSummaries() {
    try {
        const data = await api('GET', '/api/campaigns');
        state.allCampaigns = data.campaigns || [];
    } catch (_) { /* the navigator keeps what it had */ }
}

function projectSummary() {
    const id = state.activeCampaignId;
    return (state.allCampaigns.find(t => t.campaign?.id === id) || {}).project || {};
}

// ── header ───────────────────────────────────────────────

function refChip(r, asStrain) {
    const label = asStrain ? (r.key || r.title) : (r.title || r.key);
    const name = r.url
        ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(label || '')}</a>`
        : `<span>${esc(label || '')}</span>`;
    const sub = r.subtitle || (asStrain && r.title && r.title !== r.key ? r.title : '');
    return `<span class="pj-chip pj-chip-src"><b>${esc(r.source || 'text')}</b>${name}${sub ? `<small>${esc(sub)}</small>` : ''}<i data-action="ref-remove" data-source="${esc(r.source || '')}" data-key="${esc(r.key || '')}" title="Remove from the project">×</i></span>`;
}

function renderProjectHeader() {
    const c = treeOf().campaign || {};
    const st = state.docData?.status || {};
    const summary = projectSummary();
    const dot = summary.live ? 'live' : c.status === 'completed' ? 'done' : c.status === 'paused' ? 'cut' : 'plan';
    const strains = c.strains || [];
    const refs = c.references || [];
    const tabs = [
        ['overview', 'Overview', null],
        ['plan', 'Plan', st.total || null],
        ['sessions', 'Sessions', summary.sessions || null],
        ['findings', 'Findings', null],
        ['notes', 'Notes', null],
    ];
    const stat = (n, word) => `<span><b>${n}</b> ${word}</span>`;
    const attr = v => esc(v).replace(/"/g, '&quot;');  // esc() leaves quotes alone; a value attribute cannot
    return `<div class="pj-head" id="pj-head">
        <div class="pj-row1">
            ${state.editHead
                ? `<input class="pj-head-in" data-field="shorthand" value="${attr(c.shorthand)}" placeholder="short name (optional)" title="The short name, as the navigator and the breadcrumb show it">`
                : `<h2>${esc(c.shorthand || c.description || 'Project')}</h2><button class="pj-add" data-action="head-edit" title="Change the question or the short name">edit</button>`}
            <span class="pj-state"><span class="pj-dot ${dot}"></span>${summary.live ? 'live now' : esc(c.status || 'active')}</span>
            <span class="pj-muted">since ${esc(formatDate(c.created_at))}</span>
            <span class="pj-stats">
                ${stat(summary.sessions || 0, 'session' + ((summary.sessions || 0) === 1 ? '' : 's'))}
                <span><b>${st.completed || 0}</b> / <b>${st.total || 0}</b> items</span>
                ${summary.proposed_items ? stat(summary.proposed_items, 'suggested') : ''}
                ${summary.proposed_findings ? stat(summary.proposed_findings, 'finding' + (summary.proposed_findings === 1 ? '' : 's') + ' to confirm') : ''}
            </span>
        </div>
        ${state.editHead
            ? `<div class="pj-headform"><input data-field="description" value="${attr(c.description)}" placeholder="What does the project ask? One line."><div class="pj-act"><button data-action="head-save">Save</button><button data-action="head-cancel">Cancel</button></div></div>`
            : `<div class="pj-q">${esc(c.description || '')}${c.target ? `<small>${esc(c.target)}</small>` : ''}</div>`}
        <div class="pj-facets">
            <span class="pj-k">Strains</span>
            <div class="pj-vals">${strains.map(s => refChip(s, true)).join('')}<button class="pj-add" data-action="search-open" data-as="strain">+ add</button></div>
            <span class="pj-k">Links</span>
            <div class="pj-vals">${refs.map(r => refChip(r, false)).join('')}<button class="pj-add" data-action="search-open" data-as="link">+ add</button></div>
        </div>
        ${state.search ? renderSearchPopover() : ''}
        <div class="pj-sections">${tabs.map(([k, label, n]) =>
            `<button class="pj-section${state.section === k ? ' on' : ''}" data-action="section" data-section="${k}">${label}${n != null ? `<i>${n}</i>` : ''}</button>`).join('')}</div>
    </div>`;
}

function renderNewProjectForm() {
    const host = document.getElementById('new-project-form');
    if (!host) return;
    host.classList.toggle('hidden', !state.newProjectOpen);
    host.innerHTML = state.newProjectOpen ? `
        <input data-field="description" placeholder="What does the project ask? One line." autofocus>
        <input data-field="shorthand" placeholder="short name (optional)">
        <div class="pj-act"><button data-action="new-project-submit">Create</button><button data-action="new-project-cancel">Cancel</button></div>` : '';
    if (state.newProjectOpen) focusFirst();
}

// ── sections ─────────────────────────────────────────────

function renderSection() {
    switch (state.section) {
        case 'sessions': return renderProjSessions();
        case 'findings': return renderFindingsSection();
        case 'notes': return renderNotesSection();
        default: return renderOverview();
    }
}

async function loadSection() {
    const id = state.activeCampaignId;
    if (!id) return;
    const P = enc(id);
    state.sectionError = '';
    try {
        if (state.section === 'overview') {
            if (!state.catalogue) state.catalogue = (await api('GET', '/api/tools').catch(() => ({}))).tools || [];
            state.ptools = (await api('GET', `/api/campaigns/${P}/tools`)).tools || [];
            state.stream = (await api('GET', `/api/campaigns/${P}/stream`)).notes || [];
        }
        else if (state.section === 'sessions') state.projSessions = (await api('GET', `/api/sessions?project_id=${P}`)).sessions || [];
        else if (state.section === 'findings') state.findings = (await api('GET', `/api/notebook/notes?kind=finding&project=${P}`)).notes || [];
        else if (state.section === 'notes') state.notes = (await api('GET', `/api/notebook/notes?project=${P}`)).notes || [];
        else return;
    } catch (err) {
        state.sectionError = err.message;
    }
    if (state.activeCampaignId === id) renderCanvas();
}

async function reloadProject() {
    await Promise.all([loadDocument(state.activeCampaignId), refreshSummaries()]);
    renderNavigator();
    await loadSection();
}

function loadingBox(what) {
    return `<div class="pj-card"><div class="pj-loading">${state.sectionError ? esc(state.sectionError) : `Reading ${what}…`}</div></div>`;
}

// ── overview: the stream, and the project's tools on top ──────────────
// The stream is the notebook filed to this project or its sessions: lines in
// the operator's own words, dated, no kind at write time. Above it sit the
// project-scoped tools the project imported (figures, reading), each mounted
// on the project the way a session tool mounts on a session.

function renderOverview() {
    const lines = state.stream, on = state.ptools;
    if (!lines || !on) return loadingBox('the project');
    const cat = (state.catalogue || []).filter(t => t.scope === 'project');
    const tools = new Set((state.catalogue || []).map(t => t.title));  // a tool's line wears its name; a person's headline leads the text
    const avail = cat.filter(t => !on.includes(t.name));
    const title = n => ((cat.find(t => t.name === n) || {}).title || n);
    const stream = lines.slice().sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    return `${on.map(n => `<div class="pj-card pj-tool" data-ptool="${esc(n)}">
        <div class="pj-card-head"><h3>${esc(title(n))}</h3><span class="pj-muted" data-ptool-head></span><span class="pj-tool-actions" data-ptool-actions></span><button class="pj-add pj-tool-drop" data-ptool-drop="${esc(n)}" title="Take this tool off the project; its files stay">remove</button></div>
        <div class="pj-tool-body" data-ptool-body><div class="pj-loading">Loading…</div></div>
    </div>`).join('')}
    <div class="pj-card" id="stream">
        <div class="pj-card-head"><h3>Stream</h3><span class="pj-muted">what you noticed, in your words; the reading sorts out what it was</span>
            ${avail.length ? `<span class="pj-tool-actions"><select id="ptool-pick">${avail.map(t => `<option value="${esc(t.name)}" title="${esc(t.summary)}">${esc(t.title)}</option>`).join('')}</select><button class="pj-add" data-ptool-import="1">+ tool</button></span>` : ''}
        </div>
        <div class="rd-add"><input id="stream-line" placeholder="One line, in your words. Enter files it." autocomplete="off"></div>
        <div class="pj-stack">${stream.length ? stream.map(n => `<div class="pj-entry rd-line">
            <span class="pj-muted rd-when">${esc(formatDate(n.created_at))}</span>
            ${n.author === 'agent' ? '<span class="pj-chip ai">agent</span>' : Who.chip(n.by)}${tools.has(n.title) ? `<span class="pj-chip">${esc(n.title)}</span>` : ''}${n.kind && n.kind !== 'observation' ? `<span class="pj-chip">${esc(n.kind)}</span>` : ''}
            <span class="rd-text">${n.title && !tools.has(n.title) ? `<b>${esc(n.title)}</b>: ` : ''}${esc(n.body)}</span>
            ${(n.sessions || []).length ? `<span class="pj-cite">${n.sessions.map(x => esc(String(x).slice(0, 8))).join(' ')}</span>` : ''}
        </div>`).join('') : '<div class="pj-none">Nothing written yet.</div>'}</div>
    </div>`;
}

const ProjectTools = {
    mount() {
        const pid = state.activeCampaignId;
        document.querySelectorAll('.pj-tool[data-ptool]').forEach(card => {
            const tool = (window.GentlyTools || {})[card.dataset.ptool];
            const body = card.querySelector('[data-ptool-body]');
            card.querySelector('[data-ptool-drop]').onclick = () => this.set((state.ptools || []).filter(n => n !== card.dataset.ptool));
            if (!tool) { body.innerHTML = '<div class="pj-none">This tool has no pane on this page.</div>'; return; }
            tool.mount(body, pid, { head: card.querySelector('[data-ptool-head]'), actions: card.querySelector('[data-ptool-actions]') });
        });
        const imp = document.querySelector('[data-ptool-import]');
        if (imp) imp.onclick = () => this.set([...(state.ptools || []), document.getElementById('ptool-pick').value]);
    },
    async set(tools) {
        try {
            state.ptools = (await api('PUT', `/api/campaigns/${enc(state.activeCampaignId)}/tools`, { tools })).tools;
        } catch (err) { showErr(err.message); return; }
        renderCanvas();
    },
};

const Stream = {
    mount() {
        const box = document.getElementById('stream-line');
        if (!box) return;
        box.onkeydown = e => { if (e.key === 'Enter' && box.value.trim()) this.add(box.value.trim()); };
    },
    async add(text) {
        try {
            await api('POST', '/api/notebook/notes', { body: text, projects: [state.activeCampaignId] });
        } catch (err) { showErr(err.message); return; }
        await loadSection();
    },
};

// ── the reading, as a project tool ─────────────────────────────────────

function readingHtml(text) {
    // Four words open four paragraphs; anything else is a paragraph as written.
    return text.split(/\n\s*\n/).map(p => {
        const m = p.match(/^\s*(KNOWN|DEVIATES|OPEN|NEXT)\.?\s*([\s\S]*)$/);
        return m ? `<p><b class="rd-k">${m[1]}</b> ${esc(m[2]).replace(/\n/g, '<br>')}</p>` : `<p>${esc(p).replace(/\n/g, '<br>')}</p>`;
    }).join('');
}

const ReadingTool = {
    el: null, pid: null, slots: {},
    async mount(el, pid, slots) {
        this.el = el; this.pid = pid; this.slots = slots || {};
        let r;
        try { r = await api('GET', `/api/campaigns/${enc(pid)}/reading`); } catch (err) { r = { error: err.message }; }
        this.render(r);
    },
    render(r) {
        const since = r.sessions_since || 0, n = ((r.meta && r.meta.sessions) || []).length;
        const when = r.meta && r.meta.written_at ? formatDate(r.meta.written_at) : '';
        if (this.slots.head) this.slots.head.innerHTML = r.text
            ? `written ${esc(when)} over ${n} session${n === 1 ? '' : 's'}${r.meta.asked_by ? ` · asked by ${Who.chip(r.meta.asked_by)}` : ''}${since ? ` · <b class="rd-stale">${since} session${since === 1 ? '' : 's'} since</b>` : ''}`
            : (r.error ? esc(r.error) : 'nothing read yet');
        if (this.slots.actions) {
            this.slots.actions.innerHTML = `<button class="pj-add" data-rd="read" title="Write the reading again over everything filed now">${r.text ? 'Read again' : 'Read'}</button>`;
            this.slots.actions.querySelector('[data-rd]').onclick = () => this.read();
        }
        this.el.innerHTML = `<div class="rd-body">${r.text ? readingHtml(r.text) : '<div class="pj-none">Ask for a reading once a session or two is filed and the stream has a line in it.</div>'}</div>`;
    },
    async read() {
        const btn = this.slots.actions && this.slots.actions.querySelector('[data-rd]');
        if (btn) { btn.disabled = true; btn.textContent = 'Reading…'; }
        let r;
        try { r = await api('POST', `/api/campaigns/${enc(this.pid)}/reading`, {}); } catch (err) { r = { error: err.message }; }
        this.render(r);
    },
};

// ── figures, as a project tool: recipes drawn in the browser ───────────
// A recipe is which tool, which phase or tag within this project, and what
// goes on x. The browser asks the pool for the rows each time the card is
// shown, so the figure is as current as the sessions; editing a chip is a
// PUT and a redraw. The agent's PNGs list underneath, unchanged.

const FiguresTool = {
    recipes: [], pngs: [], el: null, pid: null, slots: {},

    async mount(el, pid, slots) {
        this.el = el; this.pid = pid; this.slots = slots || {};
        const P = enc(pid);
        this.recipes = (await api('GET', `/api/campaigns/${P}/recipes`).catch(() => ({}))).recipes || [];
        this.pngs = (await api('GET', `/api/campaigns/${P}/figures`).catch(() => ({}))).figures || [];
        this.paint();
    },

    render() {
        const recipes = this.recipes || [];
        const projLabel = (treeOf().campaign || {}).shorthand || this.pid || '';
        const chip = (r, key, ph) => `<label class="lf-chip${r[key] ? ' on' : ''}"><span>${key}</span><input data-lf="${key}" data-name="${esc(r.name)}" value="${esc(r[key] || '')}" placeholder="${ph}" size="${Math.max(4, (r[key] || '').length + 1)}"></label>`;
        return `<div id="live-figures">
            ${recipes.length ? '' : '<div class="pj-none">No figure yet. "+ figure" draws twitch-to-hatch over this project.</div>'}
            ${recipes.map(r => `<div class="lf-card" data-recipe="${esc(r.name)}">
                <div class="lf-bar">
                    <input class="lf-title" data-lf="title" data-name="${esc(r.name)}" value="${esc(r.title || r.name)}">
                    ${r.edited && r.edited.by ? Who.chip(r.edited.by, { when: r.edited.at }) : (r.by ? Who.chip(r.by) : '')}
                    <span class="lf-chip fixed"><span>project</span><b>${esc(projLabel)}</b></span>
                    ${chip(r, 'phase', 'any')}${chip(r, 'tag', 'any')}
                    <select data-lf="x" data-name="${esc(r.name)}"><option value="session_name"${!['condition', 'role'].includes(r.x) ? ' selected' : ''}>a column per session</option><option value="role"${r.x === 'role' ? ' selected' : ''}>a column per role</option><option value="condition"${r.x === 'condition' ? ' selected' : ''}>one column</option></select>
                    <button class="pj-add lf-del" data-lf-del="${esc(r.name)}" title="Remove this figure">×</button>
                </div>
                <div class="lf-plot" data-plot="${esc(r.name)}"><div class="pj-loading">Pooling…</div></div>
            </div>`).join('')}
            ${(this.pngs || []).length ? `<div class="pj-figures">${this.pngs.map(p => `<a class="pj-figure" href="${p.url}" target="_blank" rel="noopener"><img src="${p.url}" alt="${esc(p.name)}"><span>${esc(p.name)}</span></a>`).join('')}</div>` : ''}
        </div>`;
    },

    paint() {
        if (this.slots.head) this.slots.head.textContent = 'drawn here from the pooled sessions, every time; edit the chips to re-pool';
        if (this.slots.actions) {
            this.slots.actions.innerHTML = '<button class="pj-add" data-lf-new="1">+ figure</button>';
            this.slots.actions.querySelector('[data-lf-new]').onclick = () => this.edit('twitch-to-hatch', 'title', 'Twitch to hatch');
        }
        this.el.innerHTML = this.render();
        this.wire();
    },

    wire() {
        const host = this.el.querySelector('#live-figures');
        if (!host) return;
        host.onchange = e => {
            const t = e.target;
            if (t.dataset.lf) this.edit(t.dataset.name, t.dataset.lf, t.value);
        };
        host.onclick = e => {
            const b = e.target.closest('[data-lf-del]');
            if (b) this.remove(b.dataset.lfDel);
        };
        (this.recipes || []).forEach(r => this.draw(r));
    },

    async edit(name, key, value) {
        const P = enc(this.pid);
        const cur = (this.recipes || []).find(r => r.name === name) || { name, tool: 'twitch_hatch' };
        const next = { ...cur, [key]: value || null };
        try {
            const saved = await api('PUT', `/api/campaigns/${P}/recipes/${enc(name)}`, next);
            this.recipes = [...(this.recipes || []).filter(r => r.name !== saved.name), saved].sort((a, b) => a.name.localeCompare(b.name));
        } catch (err) { showErr(err.message); return; }
        this.paint();
    },

    async remove(name) {
        const P = enc(this.pid);
        try { await api('DELETE', `/api/campaigns/${P}/recipes/${enc(name)}`); } catch (err) { showErr(err.message); return; }
        this.recipes = (this.recipes || []).filter(r => r.name !== name);
        this.paint();
    },

    async draw(r) {
        const q = new URLSearchParams({ project: this.pid });
        if (r.phase) q.set('phase', r.phase);
        if (r.tag) q.set('tag', r.tag);
        let d;
        try { d = await api('GET', `/api/tools/${enc(r.tool || 'twitch_hatch')}/pool?${q}`); }
        catch (err) { d = { error: err.message }; }
        const el = this.el.querySelector(`[data-plot="${CSS.escape(r.name)}"]`);
        if (!el) return;
        el.innerHTML = d.error ? `<div class="pj-none">${esc(d.error)}</div>` : this.svg(r, d);
    },

    // minutes on y, a dot per embryo, a column per session or one column, a
    // median tick per column, the pooled median across, Sulston as reference
    svg(r, d) {
        const rows = (d.rows || []).filter(x => x[r.y || 'minutes'] != null);
        if (!rows.length) return '<div class="pj-none">No rows yet under this selection. Mark embryos in a session\'s Tools tab.</div>';
        const yk = r.y || 'minutes', xk = ['condition', 'role'].includes(r.x) ? r.x : 'session_name';
        const cols = [...new Set(rows.map(x => String(x[xk])))];
        const median = a => { const s = [...a].sort((p, q) => p - q), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
        const all = rows.map(x => x[yk]), pooled = median(all), ref = d.sulston_min;
        const W = Math.max(440, 110 + cols.length * 150), H = 310, L = 54, Rm = 96, T = 16, B = 56;
        const lo = Math.min(...all, ref ?? Infinity) - 20, hi = Math.max(...all, ref ?? -Infinity) + 20;
        const y = v => T + (hi - v) / (hi - lo) * (H - T - B);
        const x = i => L + (i + 0.5) * (W - L - Rm) / cols.length;
        const step = (hi - lo) > 300 ? 100 : (hi - lo) > 120 ? 50 : 20;
        const ticks = []; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
        const dots = rows.map((row, i) => `<circle class="lf-dot" cx="${(x(cols.indexOf(String(row[xk]))) + ((i % 5) - 2) * 7).toFixed(1)}" cy="${y(row[yk]).toFixed(1)}" r="4"><title>${esc(row.session_name)} · embryo ${row.n} · ${row[yk]} min${row.temp_c != null ? ` · ${row.temp_c} °C` : ''}</title></circle>`).join('');
        const meds = cols.map((c, i) => { const m = median(rows.filter(row => String(row[xk]) === c).map(row => row[yk])); return `<line class="lf-med" x1="${x(i) - 22}" x2="${x(i) + 22}" y1="${y(m)}" y2="${y(m)}"/>`; }).join('');
        return `<svg class="lf-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(r.title || r.name)}">
            ${ticks.map(v => `<line class="lf-grid" x1="${L}" x2="${W - Rm}" y1="${y(v)}" y2="${y(v)}"/><text class="lf-tick" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join('')}
            ${ref != null ? `<line class="lf-ref" x1="${L}" x2="${W - Rm}" y1="${y(ref)}" y2="${y(ref)}"/><text class="lf-lab" x="${W - Rm + 6}" y="${y(ref) + 4}">Sulston ${ref}</text>` : ''}
            <line class="lf-pooled" x1="${L}" x2="${W - Rm}" y1="${y(pooled)}" y2="${y(pooled)}"/><text class="lf-lab" x="${W - Rm + 6}" y="${y(pooled) + 4}">median ${pooled}</text>
            ${meds}${dots}
            ${cols.map((c, i) => `<text class="lf-tick" x="${x(i)}" y="${H - B + 16}" text-anchor="end" transform="rotate(-15 ${x(i)} ${H - B + 16})">${esc(c.length > 22 ? c.slice(0, 21) + '…' : c)}</text>`).join('')}
            <text class="lf-tick" x="${L - 8}" y="${T - 4}" text-anchor="end">min</text>
        </svg><div class="lf-foot">${rows.length} embryo${rows.length === 1 ? '' : 's'} · ${cols.length} column${cols.length === 1 ? '' : 's'} · median <b>${pooled} min</b> · ${esc(d.label)}</div>`;
    },
};

// ── the ledger ───────────────────────────────────────────

function dotFor(status) {
    return { completed: 'done', in_progress: 'live', skipped: 'skip', blocked: 'err', proposed: 'plan' }[status] || 'plan';
}

function draftBlock(it) {
    const d = it.imaging_spec && it.imaging_spec.draft;
    const tactics = (it.imaging_spec && it.imaging_spec.tactics) || [];
    const asking = state.openForm === `redraft:${it.id}`;
    const askForm = asking ? `<div class="pj-form pj-redraft"><input data-field="note" placeholder="add a constraint, e.g. only 488, two embryos"><div class="pj-act"><button data-action="item-redraft" data-id="${esc(it.id)}">Ask again</button><button data-action="form-close">Cancel</button></div></div>` : '';
    if (!d) {
        if (it.type !== 'imaging' || specSummary(it.imaging_spec)) return '';
        return `<div class="pj-m pj-nospec">no spec yet · <button class="pj-link ai" data-action="item-draft" data-id="${esc(it.id)}" title="Ask Gently to draft the spec from the sentence">ask Gently to draft it</button></div>`;
    }
    const tacticLine = tactics.length ? `<div class="pj-m pj-tactics">${tactics.map(t => `<span class="pj-chip">${esc(t.kind || 'tactic')}</span> ${esc(t.name || '')}${t.structure && t.structure.steps ? ` · ${t.structure.steps.length} steps` : ''}`).join(' · ')}</div>` : '';
    const assumptions = (d.assumptions || []).length ? `<div class="pj-m pj-assume">${(d.assumptions || []).map(a => `<span>· ${esc(a)}</span>`).join(' ')}</div>` : '';
    if (d.status === 'drafting') {
        return `<div class="pj-m pj-draft"><span class="pj-dot live pulse"></span> Gently is drafting the spec from the sentence…</div>`;
    }
    if (d.status === 'unavailable') {
        return `<div class="pj-m pj-draft"><span class="pj-chip">no draft</span> ${esc((d.assumptions || [])[0] || 'no model to ask')} · <button class="pj-link ai" data-action="item-draft" data-id="${esc(it.id)}">try again</button></div>`;
    }
    if (d.status === 'accepted') {
        return `${tacticLine}<div class="pj-m pj-draft"><span class="pj-chip">spec accepted</span>${d.rationale ? ` ${esc(d.rationale)}` : ''}</div>`;
    }
    return `${tacticLine}
        <div class="pj-m pj-draft"><span class="pj-chip ai">drafted by Gently</span>${d.rationale ? ` ${esc(d.rationale)}` : ''}</div>
        ${assumptions}
        <div class="pj-act"><button data-action="item-accept-spec" data-id="${esc(it.id)}" title="Make the drafted fields yours">Accept spec</button><button data-action="select-item" data-id="${esc(it.id)}">Edit</button><button data-action="form-open" data-form="redraft:${esc(it.id)}">Ask again</button></div>
        ${askForm}`;
}

function itemMenu(it) {
    const phases = phasesOf().filter(p => p.id !== it.campaign_id);
    const others = allItems().filter(x => x.id !== it.id && !(it.depends_on || []).includes(x.id));
    return `<div class="pj-menu">
        <button data-action="item-draft" data-id="${esc(it.id)}" class="ai" title="Ask Gently to draft the spec from the sentence; it lists its assumptions">Draft spec with Gently</button>
        <button data-action="item-skip" data-id="${esc(it.id)}">Skip</button>
        <button data-action="item-remove" data-id="${esc(it.id)}" class="danger">Remove</button>
        ${phases.length ? `<label>Move to <select data-action="item-move" data-id="${esc(it.id)}"><option value="">a phase…</option>${phases.map(p => `<option value="${esc(p.id)}">${esc(p.description || p.shorthand)}</option>`).join('')}</select></label>` : ''}
        ${others.length ? `<label>Depends on <select data-action="item-depends" data-id="${esc(it.id)}"><option value="">an item…</option>${others.map(o => `<option value="${esc(o.id)}">${esc(o.title)}</option>`).join('')}</select></label>` : ''}
    </div>`;
}

function findingCard(n) {
    const dim = n.status === 'superseded';
    const sup = state.openForm === `supersede:${n.id}`;
    const cites = arr => (arr || []).map(b => `<span class="pj-cite">${esc(String(b).slice(0, 12))}</span>`).join(' ');
    return `<div class="pj-entry find${dim ? ' dim' : ''}">
        <div class="pj-t"><span class="pj-chip ${n.author === 'agent' ? 'ai' : ''}">finding · ${esc(n.status)}</span><span class="pj-muted">${n.author === 'agent' ? 'agent' : (n.by ? Who.chip(n.by) : esc(n.author))} · ${esc(formatDate(n.created_at))}</span></div>
        <div class="pj-body">${esc(n.body)}</div>
        ${(n.basis || []).length ? `<div class="pj-m">basis ${cites(n.basis)}</div>` : ''}
        ${(n.sessions || []).length ? `<div class="pj-m">sessions ${cites(n.sessions)}</div>` : ''}
        ${dim ? '' : `<div class="pj-act">${n.status === 'proposed' ? `<button data-action="note-status" data-id="${esc(n.id)}" data-status="confirmed">Confirm</button>` : ''}<button data-action="form-open" data-form="supersede:${esc(n.id)}">Supersede</button>${n.status === 'proposed' ? `<button data-action="note-status" data-id="${esc(n.id)}" data-status="superseded">Dismiss</button>` : ''}</div>`}
        ${sup ? `<div class="pj-form"><textarea data-field="body" rows="3" placeholder="The finding that replaces it"></textarea><div class="pj-act"><button data-action="note-supersede" data-id="${esc(n.id)}">Save</button><button data-action="form-close">Cancel</button></div></div>` : ''}
    </div>`;
}

function addRow(form, label, hint) {
    if (state.openForm === form) return addForm(form);
    return `<button class="pj-addrow" data-action="form-open" data-form="${form}">+ ${label}<small>${esc(hint || '')}</small></button>`;
}

function addForm(form) {
    const phases = phasesOf();
    const body = {
        item: `<select data-field="type"><option value="imaging">imaging</option><option value="bench">bench</option><option value="genetics">genetics</option><option value="analysis">analysis</option><option value="decision_point">decision point</option></select>
               <input data-field="title" placeholder="What to do, in a sentence"><div class="pj-intent-preview" id="pj-intent-preview"></div>
               ${phases.length ? `<select data-field="phase_id"><option value="">no phase</option>${phases.map(p => `<option value="${esc(p.id)}">${esc(p.description || p.shorthand)}</option>`).join('')}</select>` : ''}`,
        expectation: `<input data-field="target" placeholder="target: an embryo id, or project"><input data-field="prediction" placeholder="what will happen"><input data-field="expected_time" type="datetime-local"><input data-field="uncertainty" placeholder="± how much">`,
        finding: '<textarea data-field="body" rows="3" placeholder="A claim the data supports"></textarea>',
        question: '<input data-field="content" placeholder="What the data cannot settle yet">',
    }[form] || '';
    return `<div class="pj-form" data-form="${form}">${body}<div class="pj-act"><button data-action="form-submit" data-form="${form}">Add</button><button data-action="form-close">Cancel</button></div></div>`;
}

// the project tools register beside the session tools, by name
// ── the data, as a project tool: every measurement, pooled per tool ────
// One row per embryo across the project's sessions, with the session and its
// role on each row; the same rows as CSV. Every session tool that can be
// pooled answers here. Twitch-to-hatch has its columns named; any other
// tool's columns are its row keys.
const DataTool = {
    el: null, pid: null, slots: {}, pools: [],
    COLUMNS: {
        twitch_hatch: [['n', 'Embryo'], ['field', 'Field'], ['twitch_frame', 'Twitch frame'], ['hatch_frame', 'Hatch frame'],
            ['minutes', 'Minutes'], ['temp_c', '°C'], ['vs_sulston_min', 'vs Sulston'], ['by', 'Marked by']],
    },
    // What names the row, or what the named columns already say.
    HIDE: new Set(['session_id', 'session_name', 'role', 'condition', 'x', 'y', 'at', 'twitch_at', 'hatch_at']),

    async mount(el, pid, slots) {
        this.el = el; this.pid = pid; this.slots = slots || {};
        const pools = [];
        for (const t of (state.catalogue || []).filter(t => t.scope === 'session')) {
            let d;
            try { d = await api('GET', `/api/tools/${enc(t.name)}/pool?project=${enc(pid)}`); }
            catch (_) { continue; }   // a tool with nothing to pool answers 404
            pools.push({ tool: t.name, title: t.title, ...d });
        }
        this.pools = pools;
        this.paint();
    },

    columns(p) {
        if (this.COLUMNS[p.tool]) return this.COLUMNS[p.tool];
        const keys = [];
        (p.rows || []).forEach(r => Object.keys(r).forEach(k => { if (!this.HIDE.has(k) && !keys.includes(k)) keys.push(k); }));
        return keys.map(k => [k, k.replace(/_/g, ' ')]);
    },

    cell(r, key) {
        const v = r[key];
        if (v == null || v === '') return '<span class="z">—</span>';
        if (key === 'by') return Who.chip(v, { when: r.at });
        if (typeof v === 'number') return `<span class="num">${Number.isInteger(v) ? v : v.toFixed(1)}</span>`;
        return esc(String(v));
    },

    foot(p) {
        const done = (p.rows || []).map(r => r.minutes).filter(m => typeof m === 'number').sort((a, b) => a - b);
        if (p.tool !== 'twitch_hatch' || !done.length) return '';
        const med = done.length % 2 ? done[(done.length - 1) / 2] : (done[done.length / 2 - 1] + done[done.length / 2]) / 2;
        const ref = Object.entries(p.reference || {}).map(([k, v]) => `${k} ${v} min`).join(', ');
        return `<div class="dt-foot">Median ${med} min over ${done.length} embryo${done.length === 1 ? '' : 's'} with a hatching time${ref ? ` · ${ref}` : ''}.</div>`;
    },

    render() {
        const pools = this.pools.filter(p => (p.rows || []).length);
        if (!pools.length) return '<div class="pj-none">No measurement is filed under this project yet. A session tool\'s rows land here once a session in the project has some.</div>';
        return pools.map(p => {
            const cols = this.columns(p);
            return `<div class="dt-tool">
                <div class="dt-head">${esc(p.title)} <span class="pj-muted">${p.rows.length} row${p.rows.length === 1 ? '' : 's'}</span></div>
                <div class="dt-wrap"><table class="pj-obs"><thead><tr><th>Session</th><th>Role</th>${cols.map(([, label]) => `<th>${esc(label)}</th>`).join('')}</tr></thead><tbody>
                ${p.rows.map(r => `<tr data-sgo="${esc(r.session_id)}" title="Open the session">
                    <td class="name">${esc(r.session_name || r.session_id)}</td>
                    <td>${r.role && r.role !== 'unassigned' ? `<span class="pj-chip">${esc(r.role)}</span>` : '<span class="pj-muted">—</span>'}</td>
                    ${cols.map(([key]) => `<td>${this.cell(r, key)}</td>`).join('')}
                </tr>`).join('')}
                </tbody></table></div>
                ${this.foot(p)}
            </div>`;
        }).join('');
    },

    paint() {
        const pools = this.pools.filter(p => (p.rows || []).length);
        const rows = pools.reduce((n, p) => n + p.rows.length, 0);
        const sess = new Set(); pools.forEach(p => p.rows.forEach(r => sess.add(r.session_id)));
        if (this.slots.head) this.slots.head.textContent = rows
            ? `${rows} row${rows === 1 ? '' : 's'} from ${sess.size} session${sess.size === 1 ? '' : 's'}, one per embryo, pooled here every time`
            : 'nothing measured yet';
        if (this.slots.actions) this.slots.actions.innerHTML = pools.map(p =>
            `<a class="pj-add" href="/api/tools/${enc(p.tool)}/pool?project=${enc(this.pid)}&format=csv" download title="The same rows as CSV">CSV${pools.length > 1 ? ` · ${esc(p.title)}` : ''}</a>`).join(' ');
        this.el.innerHTML = this.render();
        this.el.onclick = e => { const tr = e.target.closest('[data-sgo]'); if (tr) goToSession(tr.dataset.sgo); };
    },
};

window.GentlyTools = Object.assign(window.GentlyTools || {}, { figures: FiguresTool, reading: ReadingTool, data: DataTool });

// ── the plan bar: today's canvas, plus the ways to add to it ──

function renderPlanBar() {
    const open = state.openForm === 'item';
    return `<div class="pj-planbar">
        <span class="pj-muted">The plan as it is: phases, items, dependencies and versions.</span>
        <span class="pj-spacer"></span>
        <button class="hdr-btn" data-action="form-open" data-form="item">+ item</button>
        <button class="hdr-btn pj-ai" data-action="plan-with-gently" title="Enter plan mode with this project as the focus">Plan with Gently</button>
        ${state.planMsg ? `<span class="pj-muted pj-planmsg">${esc(state.planMsg)}</span>` : ''}
        ${open ? addForm('item') : ''}
    </div>`;
}

// ── sessions: the project's observability, aggregated from each session's records ──

function runOf(s) {
    if (s.run && s.run.status) return s.run;
    const last = s.last_run;
    if (!last || !last.status) return null;
    const ended = ['completed', 'complete', 'stopped', 'idle'].includes(String(last.status));
    return Object.assign({}, last, { status: ended || !last.embryos_going ? 'complete' : last.status });
}

function endedOf(s) {
    if (s.active) return ['live', 'live now'];
    const run = runOf(s);
    if (!run) return ['plan', 'never ran'];
    const st = String(run.status);
    if (st === 'failed') return ['err', 'failed'];
    if (['interrupted', 'running', 'paused'].includes(st)) return ['cut', 'cut short'];
    return ['done', 'complete'];
}

function renderProjSessions() {
    const rows = state.projSessions;
    if (!rows) return loadingBox('the sessions');
    const adding = state.openForm === 'add-session';
    const addForm = adding ? `<div class="pj-form" data-form="add-session">
            ${state.unfiled === null ? '<span class="pj-muted">Finding sessions…</span>'
            : `<select data-field="session_id"><option value="">a session…</option>${state.unfiled.map(s => `<option value="${esc(s.session_id)}">${esc(s.name || s.suggested_name || s.session_id)} · ${esc(formatDate(s.created_at))}${s.project_id ? ' · in ' + esc(projectNameOf(s.project_id)) : ''}</option>`).join('')}</select>
            <select data-field="role"><option value="">— no role —</option>${['control', 'test', 'reference', 'pilot'].map(r => `<option value="${r}">${r}</option>`).join('')}</select>`}
            <div class="pj-act"><button data-action="session-add-submit">Link it here</button><button data-action="form-close">Cancel</button></div></div>` : '';
    const body = rows.length === 0
        ? '<div class="pj-none">No session is filed under this project yet.</div>'
        : `<table class="pj-obs"><thead><tr><th>When</th><th>Session</th><th>Role</th><th>Holds</th><th>Ended</th><th>Agent</th><th></th></tr></thead><tbody>
        ${rows.map(s => {
            const [cls, label] = endedOf(s);
            const run = runOf(s) || {};
            const sentence = [
                s.embryo_count ? `${s.embryo_count} emb` : '',
                run.interval_seconds ? `${run.interval_seconds} s` : '',
                run.total_timepoints ? `${run.total_timepoints} tp` : '',
            ].filter(Boolean).join(' · ');
            const plans = (s.plan_items || []).map(p => `<span class="pj-chip">plan: ${esc(p.title)}</span>`).join(' ');
            const holds = [
                s.embryo_count ? `${s.embryo_count} embryo${s.embryo_count !== 1 ? 's' : ''}` : '',
                s.timepoints ? `${s.timepoints} tp` : '',
                s.bf_frames ? `${s.bf_frames} brightfield` : '',
            ].filter(Boolean).join(' · ') || '—';
            const agent = s.turns != null
                ? `<b>${s.turns}</b> turn${s.turns !== 1 ? 's' : ''}${s.corrections ? ` · <span class="corr">${s.corrections} correction${s.corrections !== 1 ? 's' : ''}</span>` : ' · 0 corrections'}`
                : '<span class="z">—</span>';
            const chips = [s.active ? '<span class="pj-chip live">active</span>' : '', s.advanced_diagnostics ? '<span class="pj-chip diag">advanced diagnostics</span>' : ''].join('');
            return `<tr class="${s.active ? 'live' : ''}" data-action="session-go" data-session-id="${esc(s.session_id)}" title="Open in the Sessions tab">
                <td class="num">${esc(fmtWhen(s.created_at))}</td>
                <td><div class="name">${esc(s.name || s.suggested_name || s.session_id)}</div><div class="sub">${plans}${plans && sentence ? ' · ' : ''}${esc(sentence)}</div></td>
                <td>${roleOf(s)}</td>
                <td>${holds}</td>
                <td><span class="ended"><span class="pj-dot ${cls}"></span>${label}</span></td>
                <td class="ag">${agent}</td>
                <td>${chips}</td>
            </tr>`;
        }).join('')}
        </tbody></table>`;
    return `<div class="pj-card">
        <div class="pj-card-head"><h3>Sessions</h3><span class="pj-muted">${rows.length} in this project · aggregated from each session’s own records</span>
            <span class="pj-right"><button class="hdr-btn" data-action="session-add-open">+ add a session</button></span></div>
        ${addForm}${body}
    </div>`;
}

// The Sessions tab, on this session. ReviewApp is a top-level const there: a
// global binding, never a window property, so it is reached by name.
function goToSession(sid) {
    if (typeof switchTab === 'function') switchTab('sessions');
    setTimeout(() => { if (typeof ReviewApp !== 'undefined') ReviewApp.loadSession(sid); }, 80);
}

// ── findings and notes ───────────────────────────────────

function renderFindingsSection() {
    const list = state.findings;
    if (!list) return loadingBox('the findings');
    const live = list.filter(n => n.status !== 'superseded');
    const gone = list.filter(n => n.status === 'superseded');
    return `<div class="pj-card">
        <div class="pj-card-head"><h3>Findings</h3><span class="pj-muted">claims about this project; the agent’s arrive proposed, a person confirms</span></div>
        <div class="pj-stack">${live.map(findingCard).join('') || '<div class="pj-none">No finding yet.</div>'}${addRow('finding', 'finding', '')}</div>
        ${gone.length ? `<div class="pj-stack dimmed"><h4>Superseded</h4>${gone.map(findingCard).join('')}</div>` : ''}
    </div>`;
}

function renderNotesSection() {
    const list = state.notes;
    if (!list) return loadingBox('the notes');
    return `<div class="pj-card">
        <div class="pj-card-head"><h3>Notes</h3><span class="pj-muted">everything in the notebook filed to this project</span></div>
        <div class="pj-stack">${list.length ? list.map(n => `<div class="pj-entry">
            <div class="pj-t"><span class="pj-chip ${n.author === 'agent' ? 'ai' : ''}">${esc(n.kind)}${n.status && n.status !== 'confirmed' ? ` · ${esc(n.status)}` : ''}</span><span class="pj-muted">${n.author === 'agent' ? 'agent' : (n.by ? Who.chip(n.by) : esc(n.author))} · ${esc(formatDate(n.created_at))}</span></div>
            ${n.title ? `<div class="pj-t"><b>${esc(n.title)}</b></div>` : ''}
            <div class="pj-body">${esc(n.body)}</div>
            ${(n.sessions || []).length ? `<div class="pj-m">sessions ${n.sessions.map(s => `<span class="pj-cite">${esc(String(s).slice(0, 8))}</span>`).join(' ')}</div>` : ''}
        </div>`).join('') : '<div class="pj-none">No note is filed to this project yet.</div>'}</div>
    </div>`;
}

// ── the add popover: one box, five sources, each group lands when its source answers ──

const SEARCH_SOURCES = ['lab', 'genes', 'strains', 'papers', 'terms'];
const SEARCH_NAMES = {
    lab: ['This lab', 'strains and genes already named anywhere in gently'],
    genes: ['Genes', 'Alliance of Genome Resources, WormBase ids'],
    strains: ['Strains', 'CGC'],
    papers: ['Papers', 'Europe PMC · a DOI or PMID resolves directly'],
    terms: ['Terms', 'WBls life stages and WBbt anatomy, via OLS'],
};

function openSearch() {
    state.search = { q: '', groups: {}, sel: -1, rows: [], ctrls: {}, timer: null };
    state.searchMsg = '';
    renderCanvas();
    document.getElementById('pj-search-input')?.focus();
}

function closeSearch() {
    const S = state.search;
    if (S) Object.values(S.ctrls).forEach(c => c.abort());
    state.search = null;
    renderCanvas();
}

function onSearchInput(value) {
    const S = state.search;
    if (!S) return;
    S.q = value;
    S.sel = -1;
    clearTimeout(S.timer);
    S.timer = setTimeout(runSearch, 200);
}

function runSearch() {
    const S = state.search;
    if (!S) return;
    Object.values(S.ctrls).forEach(c => c.abort());
    S.ctrls = {};
    S.groups = {};
    const q = S.q.trim();
    if (q.length < 2) { paintSearch(); return; }
    SEARCH_SOURCES.forEach(src => {
        const ctrl = new AbortController();
        S.ctrls[src] = ctrl;
        S.groups[src] = { status: 'loading', results: [] };
        fetch(`/api/search/${src}?q=${enc(q)}`, { signal: ctrl.signal })
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then(data => { if (state.search === S && S.ctrls[src] === ctrl) { S.groups[src] = data; paintSearch(); } })
            .catch(err => {
                if (err.name === 'AbortError' || state.search !== S || S.ctrls[src] !== ctrl) return;
                S.groups[src] = { status: 'error', results: [], error: err.message };
                paintSearch();
            });
    });
    paintSearch();
}

function glyphFor(g) {
    if (!g) return ['plan', 'not asked'];
    if (g.status === 'loading') return ['cut', 'searching…'];
    if (g.status === 'error') return ['err', g.error ? `unreachable · ${g.error}` : 'unreachable'];
    if (g.status === 'empty' || !(g.results || []).length) return ['plan', 'none'];
    return ['live', `${g.results.length}${g.took_ms != null ? ` · ${g.took_ms} ms` : ''}`];
}

function renderSearchPopover() {
    const S = state.search;
    const q = S.q.trim();
    let idx = 0;
    const rows = [];
    const groups = SEARCH_SOURCES.map(src => {
        const g = S.groups[src];
        const [cls, label] = glyphFor(g);
        const items = (g?.results || []).map(r => {
            const i = idx++;
            rows.push(r);
            return `<div class="pj-srow${S.sel === i ? ' sel' : ''}" data-action="search-pick" data-idx="${i}">
                <span class="k">${esc(r.title || r.key)}<small>${esc(r.key)}${r.source ? ` · ${esc(r.source)}` : ''}</small></span>
                <span class="d">${esc(r.subtitle || '')}</span>
                <span class="to">files as <b>${esc(r.files_as || 'link')}</b></span></div>`;
        }).join('');
        return `<div class="pj-sgroup"><div class="gh">${SEARCH_NAMES[src][0]}<span class="src">· ${SEARCH_NAMES[src][1]}</span><span class="st"><span class="pj-dot ${cls}"></span>${esc(label)}</span></div>${items}</div>`;
    }).join('');
    S.rows = rows;
    S.keepIdx = idx;
    return `<div class="pj-search" id="pj-search">
        <div class="top"><input id="pj-search-input" value="${esc(S.q)}" placeholder="a strain, gene, paper, DOI, PMID or term" autocomplete="off" spellcheck="false">
            <span class="keys"><kbd>↑</kbd><kbd>↓</kbd> move · <kbd>↵</kbd> add and keep searching · <kbd>esc</kbd> close</span>
            <button data-action="search-close" title="Close">×</button></div>
        ${q.length < 2 ? '<div class="pj-none">Type at least two characters.</div>' : groups}
        ${q.length >= 2 ? `<div class="pj-srow keep${S.sel === idx ? ' sel' : ''}" data-action="search-keep" data-idx="${idx}">Keep “${esc(q)}” as plain text<small>· always offered, so a lost network never blocks a note</small></div>` : ''}
        <div class="pj-sfoot">${state.searchMsg ? esc(state.searchMsg) : 'The lab first, then the Alliance, CGC, Europe PMC and OLS. WormBase ids arrive through the Alliance and OLS; each chip links to wormbase.org.'}</div>
    </div>`;
}

// Repaint only the popover, so the caret stays where it was.
function paintSearch() {
    const el = document.getElementById('pj-search');
    if (!el || !state.search) return;
    const input = el.querySelector('input');
    const pos = input ? input.selectionStart : null;
    const focused = document.activeElement === input;
    el.outerHTML = renderSearchPopover();
    const inp = document.getElementById('pj-search-input');
    if (inp && focused) {
        inp.focus();
        if (pos != null) inp.setSelectionRange(pos, pos);
    }
}

function onSearchKey(e) {
    const S = state.search;
    if (!S) return;
    const last = S.keepIdx != null ? S.keepIdx : -1;
    if (e.key === 'ArrowDown') { e.preventDefault(); S.sel = Math.min(last, S.sel + 1); paintSearch(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); S.sel = Math.max(-1, S.sel - 1); paintSearch(); }
    else if (e.key === 'Escape') { e.preventDefault(); closeSearch(); }
    else if (e.key === 'Enter') {
        e.preventDefault();
        if (S.sel >= 0 && S.sel < S.rows.length) pickSearch(S.sel);
        else if (S.q.trim().length >= 2) keepText();
    }
}

async function addReference(ref) {
    const P = enc(state.activeCampaignId);
    await api('POST', `/api/campaigns/${P}/references`, {
        source: ref.source, key: ref.key, title: ref.title, url: ref.url || null,
        subtitle: ref.subtitle || null, files_as: ref.files_as || 'link',
    });
    state.searchMsg = `Kept ${ref.title || ref.key} as a ${ref.files_as === 'strain' ? 'strain' : 'link'}.`;
    await Promise.all([loadDocument(state.activeCampaignId), refreshSummaries()]);
    renderCanvas();
    document.getElementById('pj-search-input')?.focus();
}

async function pickSearch(i) {
    const S = state.search;
    const r = S && S.rows[i];
    if (!r) return;
    try { await addReference(r); } catch (err) { state.searchMsg = err.message; paintSearch(); }
}

async function keepText() {
    const S = state.search;
    const q = S && S.q.trim();
    if (!q) return;
    try { await addReference({ source: 'text', key: q, title: q, files_as: 'link' }); } catch (err) { state.searchMsg = err.message; paintSearch(); }
}

// ── the actions: every write through the same routes the agent's tools mirror ──

let _intentTimer = null;
function previewIntent(text) {
    if (_intentTimer) clearTimeout(_intentTimer);
    _intentTimer = setTimeout(async () => {
        const host = document.getElementById('pj-intent-preview');
        if (!host) return;
        if (!text || text.trim().length < 4) { host.innerHTML = ''; return; }
        try {
            const r = await fetch(`/api/plan/intent?text=${encodeURIComponent(text)}`);
            const d = await r.json();
            const read = specSummary(d.spec);
            host.innerHTML = read
                ? `reads as <b>${esc(read)}</b> <span class="pj-muted">· the rest stays to fill</span>`
                : '<span class="pj-muted">no numbers or stages in the sentence yet; the spec can be filled after, or asked of Gently</span>';
        } catch (_) { host.innerHTML = ''; }
    }, 200);
}

async function draftItem(id, note) {
    const P = enc(state.activeCampaignId);
    state.transcribeMsg = { id, text: 'Gently is drafting…' };
    state.openForm = null;
    renderCanvas();
    const r = await api('POST', `/api/campaigns/${P}/items/${enc(id)}/draft`, { note: note || null });
    const read = (r.from_sentence || []).length;
    const parts = [];
    if (r.status === 'unavailable') parts.push((r.assumptions || [])[0] || 'no model to ask');
    else parts.push(`${(r.filled || []).length} field${(r.filled || []).length !== 1 ? 's' : ''} in the spec, ${read} read from the sentence`);
    state.transcribeMsg = { id, text: parts.join(' · ') };
    await reloadProject();
}

async function submitForm(btn) {
    const form = btn.closest('.pj-form');
    const kind = btn.dataset.form;
    const d = fields(form);
    const cid = state.activeCampaignId;
    const P = enc(cid);
    if (kind === 'item') {
        if (!d.title) return showErr(form, 'A title is needed.');
        await api('POST', `/api/campaigns/${P}/items`, { type: d.type || 'imaging', title: d.title, phase_id: d.phase_id || null, origin: 'manual', draft: true });
    } else if (kind === 'expectation') {
        if (!d.prediction) return showErr(form, 'Say what will happen.');
        await api('POST', '/api/context/expectations', {
            target: d.target || 'project', prediction: d.prediction,
            expected_time: d.expected_time || new Date().toISOString().slice(0, 16),
            uncertainty: d.uncertainty || null, project_id: cid,
        });
    } else if (kind === 'finding') {
        if (!d.body) return showErr(form, 'Write the finding.');
        await api('POST', '/api/notebook/notes', { kind: 'finding', body: d.body, projects: [cid] });
    } else if (kind === 'question') {
        if (!d.content) return showErr(form, 'Write the question.');
        await api('POST', '/api/context/questions', { content: d.content, project_id: cid });
    } else {
        return;
    }
    state.openForm = null;
    await reloadProject();
}

async function projectAction(action, el, e) {
    const cid = state.activeCampaignId;
    const id = el.dataset.id;
    const P = enc(cid || '');
    try {
        switch (action) {
            case 'item-draft':
                await draftItem(id, null);
                return;
            case 'item-redraft': {
                const f = el.closest('.pj-form');
                const note = f ? (fields(f).note || '') : '';
                await draftItem(id, note);
                return;
            }
            case 'dep-add':
                if (!el.value) return;
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/depends`, { depends_on: el.value });
                await Promise.all([reloadProject(), selectItem(id)]);
                return;
            case 'dep-remove':
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/depends`, { depends_on: el.dataset.dep, remove: true });
                await Promise.all([reloadProject(), selectItem(id)]);
                return;
            case 'item-undepend':
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/depends`, { depends_on: el.dataset.dep, remove: true });
                await reloadProject();
                return;
            case 'item-setup': {
                const it = allItems().find(x => x.id === id);
                if (it) setUpOnDevices(it);
                return;
            }
            case 'item-accept-spec':
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/spec/accept`, {});
                state.transcribeMsg = null;
                await reloadProject();
                return;
            case 'section':
                state.section = el.dataset.section;
                state.openForm = null;
                renderAll();
                loadSection();
                return;
            case 'new-project-open':
                state.newProjectOpen = !state.newProjectOpen;
                renderNewProjectForm();
                return;
            case 'head-edit':
                state.editHead = true;
                renderCanvas();
                document.querySelector('#pj-head [data-field="description"]')?.focus();
                return;
            case 'head-cancel':
                state.editHead = false;
                renderCanvas();
                return;
            case 'head-save': {
                const d = fields(document.getElementById('pj-head'));
                if (!d.description) { showErr(el, 'Say what the project asks, in one line.'); return; }
                // An empty short name clears it: the route keeps every field it is sent and drops only the ones left out.
                await api('PATCH', `/api/campaigns/${P}`, { description: d.description, shorthand: d.shorthand });
                state.editHead = false;
                await refreshActiveCampaign();
                return;
            }
            case 'new-project-cancel':
                state.newProjectOpen = false;
                renderNewProjectForm();
                return;
            case 'new-project-submit': {
                const host = document.getElementById('new-project-form');
                const d = fields(host);
                if (!d.description) { showErr(host, 'Say what the project asks, in one line.'); return; }
                const r = await api('POST', '/api/campaigns', { description: d.description, shorthand: d.shorthand || null });
                state.newProjectOpen = false;
                renderNewProjectForm();
                await refreshSummaries();
                renderNavigator();
                await openCampaign(r.campaign.id);
                return;
            }
            case 'ref-remove':
                e.preventDefault();
                await api('DELETE', `/api/campaigns/${P}/references?source=${enc(el.dataset.source)}&key=${enc(el.dataset.key)}`);
                await Promise.all([loadDocument(cid), refreshSummaries()]);
                renderCanvas();
                return;
            case 'search-open': openSearch(); return;
            case 'search-close': closeSearch(); return;
            case 'search-pick': await pickSearch(Number(el.dataset.idx)); return;
            case 'search-keep': await keepText(); return;
            case 'item-accept':
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/status`, { status: 'planned' });
                await reloadProject();
                return;
            case 'item-skip':
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/status`, { status: 'skipped' });
                state.openForm = null;
                await reloadProject();
                return;
            case 'item-remove':
                await api('DELETE', `/api/campaigns/${P}/items/${enc(id)}`);
                state.openForm = null;
                if (state.selectedItemId === id) closeInspector();
                await reloadProject();
                return;
            case 'item-menu':
                state.openForm = state.openForm === `menu:${id}` ? null : `menu:${id}`;
                renderCanvas();
                return;
            case 'item-move':
                if (!el.value) return;  // the click that opens the select, not a choice
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/move`, { campaign_id: el.value });
                state.openForm = null;
                await reloadProject();
                return;
            case 'item-depends':
                if (!el.value) return;
                await api('POST', `/api/campaigns/${P}/items/${enc(id)}/depends`, { depends_on: el.value });
                state.openForm = null;
                await reloadProject();
                return;
            case 'form-open':
                state.openForm = el.dataset.form;
                renderCanvas();
                focusFirst();
                return;
            case 'form-close':
                state.openForm = null;
                renderCanvas();
                return;
            case 'form-submit': await submitForm(el); return;
            case 'outcome-save': {
                const d = fields(el.closest('.pj-form'));
                await api('PATCH', `/api/campaigns/${P}/items/${enc(id)}`, { outcome: d.outcome || '' });
                state.openForm = null;
                await reloadProject();
                return;
            }
            case 'exp-resolve':
                await api('POST', `/api/context/expectations/${enc(id)}/resolve`, { status: el.dataset.status });
                await reloadProject();
                return;
            case 'q-resolve': {
                const form = el.closest('.pj-form');
                const resolution = form ? (fields(form).resolution || '') : (el.dataset.resolution || '');
                await api('POST', `/api/context/questions/${enc(id)}/resolve`, { resolution });
                state.openForm = null;
                await reloadProject();
                return;
            }
            case 'note-status':
                await api('POST', `/api/notebook/notes/${enc(id)}/status`, { status: el.dataset.status });
                await reloadProject();
                return;
            case 'note-supersede': {
                const form = el.closest('.pj-form');
                const d = fields(form);
                if (!d.body) { showErr(form, 'Write the finding that replaces it.'); return; }
                await api('POST', '/api/notebook/notes', { kind: 'finding', body: d.body, projects: [cid], basis: [id] });
                await api('POST', `/api/notebook/notes/${enc(id)}/status`, { status: 'superseded' });
                state.openForm = null;
                await reloadProject();
                return;
            }
            case 'plan-with-gently': {
                const r = await api('POST', `/api/campaigns/${P}/plan-mode`);
                state.planMsg = r.ok ? (r.message || 'In plan mode.') : (r.detail || 'The agent is not running.');
                if (r.ok) document.getElementById('agent-rail-toggle')?.click();
                renderCanvas();
                return;
            }
            case 'session-go': goToSession(el.dataset.sessionId); return;
            case 'session-add-open': {
                state.openForm = 'add-session';
                state.unfiled = null;
                renderCanvas();
                // Every session not yet linked here, unfiled ones first; a
                // session informs several projects, so its home is no bar.
                const list = (await api('GET', '/api/sessions')).sessions || [];
                state.unfiled = list
                    .filter(s => !(s.projects || []).some(l => l.id === cid))
                    .sort((a, b) => (a.project_id ? 1 : 0) - (b.project_id ? 1 : 0));
                renderCanvas();
                return;
            }
            case 'session-add-submit': {
                const d = fields(el.closest('.pj-form'));
                if (!d.session_id) return;
                await api('POST', `/api/sessions/${enc(d.session_id)}/project`, { campaign_id: cid, role: d.role || null });
                state.openForm = null;
                await reloadProject();
                return;
            }
            default:
                return;
        }
    } catch (err) {
        showErr(el, err.message);
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
