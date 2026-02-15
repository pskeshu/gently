/**
 * Campaigns — Unified workspace for campaign browsing and plan review.
 *
 * Views:
 *   dashboard — all campaigns, navigator shows campaign list, canvas shows cards
 *   plan      — single campaign, navigator shows outline, canvas shows plan doc
 */

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
    view: 'dashboard',          // 'dashboard' | 'plan'
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
document.addEventListener('DOMContentLoaded', () => {
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

    // Event listeners
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
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

    // Route: auto-open campaign if ID is provided
    const initialId = window.INITIAL_CAMPAIGN_ID;
    if (initialId) {
        openCampaign(initialId);
    } else {
        loadCampaigns();
    }
});

// ── Theme ────────────────────────────────────────────────
function toggleTheme() {
    const next = document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', next);
    localStorage.setItem('gently-theme', next);
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
        renderAll();
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
    state.view = 'plan';
    state.activeCampaignId = campaignId;
    state.selectedItemId = null;
    state.viewingSnapshotId = null;
    closeInspector();
    showLoading('Loading plan...');

    // Load data in parallel
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

    // Update URL without reload
    const url = `/campaigns/${encodeURIComponent(campaignId)}/review`;
    if (window.location.pathname !== url) {
        history.pushState({ view: 'plan', campaignId }, '', url);
    }
}

function goToDashboard() {
    state.view = 'dashboard';
    state.activeCampaignId = null;
    state.selectedItemId = null;
    state.docData = null;
    state.versions = [];
    state.viewingSnapshotId = null;
    closeInspector();

    // Re-render with existing campaign data, or reload
    if (state.allCampaigns.length > 0) {
        renderAll();
    } else {
        loadCampaigns();
    }

    const url = '/campaigns';
    if (window.location.pathname !== url) {
        history.pushState({ view: 'dashboard' }, '', url);
    }
}

// Handle browser back/forward
window.addEventListener('popstate', e => {
    const s = e.state;
    if (s && s.view === 'plan' && s.campaignId) {
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
    if (state.view === 'dashboard') {
        renderNavDashboard();
    } else {
        renderNavOutline();
    }
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

function renderNavOutline() {
    if (!state.docData) return;
    const tree = state.docData.document;
    const campaign = tree.campaign;

    let html = `<div class="nav-back" onclick="goToDashboard()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
        All Campaigns
    </div>`;

    html += `<div class="nav-campaign-title">${esc(campaign.description)}</div>`;

    const children = tree.children || [];
    if (children.length > 0) {
        children.forEach((child, idx) => {
            const phaseNum = idx + 1;
            const phaseName = child.campaign.description || child.campaign.shorthand;
            const items = child.items || [];

            html += `<div class="nav-outline-phase">
                <div class="nav-phase-header" onclick="toggleNavPhase(this)">
                    <svg class="nav-phase-chevron open" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="9,18 15,12 9,6"></polyline>
                    </svg>
                    <span class="nav-phase-name">${esc(phaseName)}</span>
                </div>
                <div class="nav-phase-items">`;

            items.forEach((item, iIdx) => {
                const taskNum = `${phaseNum}.${iIdx + 1}`;
                const dot = STATUS_DOTS[item.status] || '\u25CB';
                const dotClass = item.status || 'planned';
                html += `<div class="nav-item" data-item-id="${item.id}" onclick="navigateToItem('${item.id}')">
                    <span class="nav-item-dot dot-${dotClass}">${dot}</span>
                    <span class="nav-item-num">${esc(taskNum)}</span>
                    <span class="nav-item-title">${esc(item.title)}</span>
                </div>`;
            });

            html += `</div></div>`;
        });
    } else if ((tree.items || []).length > 0) {
        tree.items.forEach((item, idx) => {
            const dot = STATUS_DOTS[item.status] || '\u25CB';
            const dotClass = item.status || 'planned';
            html += `<div class="nav-item" data-item-id="${item.id}" onclick="navigateToItem('${item.id}')">
                <span class="nav-item-dot dot-${dotClass}">${dot}</span>
                <span class="nav-item-num">${idx + 1}</span>
                <span class="nav-item-title">${esc(item.title)}</span>
            </div>`;
        });
    }

    // Bottom links
    const refCount = (state.docData.bibliography || []).length;
    html += `<div class="nav-divider"></div>`;
    html += `<div class="nav-link" onclick="scrollCanvasTo('bibliography-section')">
        Refs <span class="nav-link-count">(${refCount})</span>
    </div>`;
    html += `<div class="nav-link" onclick="scrollCanvasTo('versions-section')">
        History <span class="nav-link-count">(${state.versions.length})</span>
    </div>`;

    $navContent.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
//  CANVAS
// ══════════════════════════════════════════════════════════

function renderCanvas() {
    if (!$canvasContent) return;
    if (state.view === 'dashboard') {
        renderDashboard();
    } else {
        renderPlanDoc();
    }
}

function renderDashboard() {
    if (state.allCampaigns.length === 0) {
        $canvasContent.innerHTML = `<div class="empty-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                <polyline points="14,2 14,8 20,8"/>
            </svg>
            <p>No campaigns yet</p>
            <span class="empty-hint">Create a campaign from a chat session to get started</span>
        </div>`;
        return;
    }

    let html = '<div class="overview-grid">';
    for (const tree of state.allCampaigns) {
        html += renderCampaignCard(tree);
    }
    html += '</div>';
    $canvasContent.innerHTML = html;
}

function renderCampaignCard(tree) {
    const c = tree.campaign;
    const status = tree.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const inProgress = status.in_progress || 0;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    const children = tree.children || [];

    let phasesHtml = '';
    if (children.length > 0) {
        phasesHtml = '<div class="campaign-phases-preview">';
        children.forEach((child, idx) => {
            const ps = child.status || {};
            const pt = ps.total || 0;
            const pc = ps.completed || 0;
            const ppct = pt > 0 ? Math.round((pc / pt) * 100) : 0;
            const name = child.campaign.description || child.campaign.shorthand;
            phasesHtml += `<div class="phase-row">
                <span class="phase-num">P${idx + 1}</span>
                <span class="phase-name">${esc(name)}</span>
                ${pt > 0 ? `<div class="phase-mini-progress"><div class="phase-mini-fill" style="width:${ppct}%"></div></div>` : ''}
                <span class="phase-count">${pt > 0 ? `${pc}/${pt}` : ''}</span>
            </div>`;
        });
        phasesHtml += '</div>';
    }

    return `<div class="campaign-card" data-status="${c.status}">
        <div class="campaign-card-header">
            <div class="campaign-title-row">
                ${c.shorthand ? `<span class="campaign-shorthand">${esc(c.shorthand)}</span>` : ''}
                <span class="campaign-name">${esc(c.description)}</span>
                <span class="status-badge status-${c.status}">${STATUS_LABELS[c.status] || c.status}</span>
            </div>
        </div>
        ${c.target ? `<div class="campaign-target">${esc(c.target)}</div>` : ''}
        ${total > 0 ? `<div class="campaign-progress">
            <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
            <span class="progress-text">${completed}/${total} \u00B7 ${pct}%</span>
        </div>` : ''}
        ${phasesHtml}
        <div class="campaign-card-footer">
            <button class="view-plan-btn" onclick="openCampaign('${esc(c.id)}')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                </svg>
                View Plan
            </button>
        </div>
    </div>`;
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
            <div class="plan-progress-fill" style="width:${pct}%"></div>
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
        html += '<div class="phase-block">';
        tree.items.forEach((item, idx) => {
            html += renderDocItem(item, String(idx + 1));
        });
        html += '</div>';
    }

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
            html += `<div class="version-entry ${isCurrent ? 'current' : ''}" onclick="viewVersion('${v.version_id}', ${isCurrent})">
                <span class="version-entry-num">v${v.version_number || '?'}</span>
                <span class="version-entry-label">${esc(label)}</span>
                ${isCurrent ? '<span class="version-entry-current">Current</span>' : ''}
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
            state.docData.document = snapshot.snapshot_json;
            state.allItemsFlat = {};
            buildItemIndex(snapshot.snapshot_json);

            renderCanvas();
            renderNavigator();

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
    if (state.view === 'dashboard') {
        if ($headerTitle) $headerTitle.textContent = 'Campaigns';
        if ($headerBreadcrumb) $headerBreadcrumb.textContent = '';
        $versionWrap?.classList.add('hidden');
        $printBtn?.classList.add('hidden');
    } else {
        const campaign = state.docData?.document?.campaign;
        if ($headerTitle) $headerTitle.textContent = campaign?.description || campaign?.shorthand || 'Plan';
        if ($headerBreadcrumb) $headerBreadcrumb.textContent = '';
        // Show version dropdown if we have versions
        if (state.versions.length > 0) {
            $versionWrap?.classList.remove('hidden');
        } else {
            $versionWrap?.classList.add('hidden');
        }
        $printBtn?.classList.remove('hidden');
    }
}

function updateStatusbar() {
    if (state.view === 'dashboard') {
        // Aggregate stats
        let totalItems = 0, doneItems = 0, activeItems = 0;
        function countTree(tree) {
            const s = tree.status || {};
            totalItems += s.total || 0;
            doneItems += s.completed || 0;
            activeItems += s.in_progress || 0;
            (tree.children || []).forEach(countTree);
        }
        state.allCampaigns.forEach(countTree);

        const activeCampaigns = state.allCampaigns.filter(t => t.campaign.status === 'active').length;
        const pct = totalItems > 0 ? Math.round((doneItems / totalItems) * 100) : 0;

        if ($statusLeft) {
            $statusLeft.textContent = `${state.allCampaigns.length} campaign${state.allCampaigns.length !== 1 ? 's' : ''} \u00B7 ${totalItems} items \u00B7 ${pct}% complete`;
        }
        if ($statusRight) {
            $statusRight.textContent = `${activeCampaigns} active`;
        }
    } else {
        const status = state.docData?.status || {};
        const total = status.total || 0;
        const completed = status.completed || 0;
        const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
        const phases = (state.docData?.document?.children || []).length;

        if ($statusLeft) {
            $statusLeft.textContent = `Plan: ${total} items \u00B7 ${pct}% complete \u00B7 ${phases} phase${phases !== 1 ? 's' : ''}`;
        }
        if ($statusRight) {
            if (state.versions.length > 0) {
                const latest = state.versions.reduce((a, b) =>
                    (b.version_number || 0) > (a.version_number || 0) ? b : a, state.versions[0]);
                const date = latest.created_at ? formatDate(latest.created_at) : '';
                $statusRight.textContent = `Last snapshot: v${latest.version_number || '?'} ${date}`;
            } else {
                $statusRight.textContent = '';
            }
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
    if (state.view !== 'plan' || state.selectedItemId) return;

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
