/**
 * Campaigns page — browse experimental plans and campaign hierarchy.
 *
 * Fetches campaign trees from /api/campaigns, renders them as nested
 * cards with plan items, progress bars, and a detail panel.
 */

const TYPE_ICONS = {
    imaging: '📷',
    bench: '🧪',
    genetics: '🧬',
    analysis: '📊',
    decision_point: '🚦',
};

const STATUS_LABELS = {
    active: 'Active',
    completed: 'Completed',
    paused: 'Paused',
};

// ── State ─────────────────────────────────────────────────
let allCampaigns = [];
let allItemsById = {};  // flat map for quick lookups

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadCampaigns();

    // Theme toggle
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const next = document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            document.body.setAttribute('data-theme', next);
            localStorage.setItem('gently-theme', next);
        });
    }

    // Detail panel close
    document.getElementById('detail-close')?.addEventListener('click', closeDetail);

    // Close on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDetail();
    });

    // Auto-refresh every 30s
    setInterval(loadCampaigns, 30000);
});

// ── Data loading ──────────────────────────────────────────
async function loadCampaigns() {
    try {
        const res = await fetch('/api/campaigns');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        allCampaigns = data.campaigns || [];

        // Build flat item index
        allItemsById = {};
        function indexItems(tree) {
            for (const item of (tree.items || [])) {
                allItemsById[item.id] = item;
            }
            for (const child of (tree.children || [])) {
                indexItems(child);
            }
        }
        allCampaigns.forEach(indexItems);

        render();
    } catch (err) {
        console.error('Failed to load campaigns:', err);
        document.getElementById('loading-state')?.classList.add('hidden');
        document.getElementById('empty-state')?.classList.remove('hidden');
    }
}

// ── Rendering ─────────────────────────────────────────────
function render() {
    const loading = document.getElementById('loading-state');
    const empty = document.getElementById('empty-state');
    const list = document.getElementById('campaign-list');
    const count = document.getElementById('campaign-count');

    loading?.classList.add('hidden');

    if (allCampaigns.length === 0) {
        empty?.classList.remove('hidden');
        list.innerHTML = '';
        if (count) count.textContent = '';
        return;
    }

    empty?.classList.add('hidden');
    if (count) count.textContent = `${allCampaigns.length} campaign${allCampaigns.length !== 1 ? 's' : ''}`;

    list.innerHTML = allCampaigns.map(renderCampaignTree).join('');
}

function renderCampaignTree(tree) {
    const c = tree.campaign;
    const status = tree.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    const hasChildren = (tree.children || []).length > 0;
    const hasItems = (tree.items || []).length > 0;

    let phases = '';
    if (hasChildren) {
        phases = `<div class="campaign-phases">
            ${tree.children.map(renderPhase).join('')}
        </div>`;
    } else if (hasItems) {
        phases = `<div class="campaign-items">
            ${tree.items.map(renderPlanItem).join('')}
        </div>`;
    }

    return `
        <div class="campaign-card">
            <div class="campaign-header" onclick="toggleCampaign(this)">
                <div class="campaign-title-row">
                    ${c.shorthand ? `<span class="campaign-shorthand">${esc(c.shorthand)}</span>` : ''}
                    <span class="campaign-title">${esc(c.description)}</span>
                    <span class="status-badge status-${c.status}">${STATUS_LABELS[c.status] || c.status}</span>
                </div>
                ${c.target ? `<div class="campaign-target">${esc(c.target)}</div>` : ''}
            </div>
            ${total > 0 ? `
            <div class="campaign-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${pct}%"></div>
                </div>
                <span class="progress-text">${completed}/${total} items · ${pct}%</span>
            </div>` : ''}
            ${phases}
        </div>
    `;
}

function renderPhase(tree) {
    const c = tree.campaign;
    const items = tree.items || [];
    const status = tree.status || {};
    const total = status.total || 0;
    const completed = status.completed || 0;
    const countText = total > 0 ? `${completed}/${total}` : '';

    return `
        <div class="phase-section">
            <div class="phase-header" onclick="togglePhase(this)">
                <svg class="phase-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="9,18 15,12 9,6"></polyline>
                </svg>
                <span class="phase-name">${esc(c.shorthand || c.description)}</span>
                <span class="phase-count">${countText}</span>
            </div>
            <div class="plan-items collapsed">
                ${items.map(renderPlanItem).join('')}
                ${(tree.children || []).length > 0
                    ? tree.children.map(renderPhase).join('')
                    : ''}
            </div>
        </div>
    `;
}

function renderPlanItem(item) {
    const icon = TYPE_ICONS[item.type] || '📋';
    const statusClass = item.status || 'planned';

    // Build description hint
    let desc = '';
    if (item.imaging_spec?.strain) {
        desc = item.imaging_spec.strain;
        if (item.imaging_spec.num_embryos) desc += ` · ${item.imaging_spec.num_embryos} embryos`;
        if (item.imaging_spec.interval_s) desc += ` · ${item.imaging_spec.interval_s}s interval`;
    } else if (item.bench_spec?.protocol) {
        desc = item.bench_spec.protocol;
    } else if (item.description) {
        desc = item.description.slice(0, 80);
    }

    return `
        <div class="plan-item" onclick="showItemDetail('${item.id}')">
            <span class="item-icon type-${item.type}">${icon}</span>
            <div class="item-content">
                <div class="item-title">${esc(item.title)}</div>
                ${desc ? `<div class="item-desc">${esc(desc)}</div>` : ''}
            </div>
            <div class="item-status">
                <div class="item-status-dot dot-${statusClass}"></div>
            </div>
        </div>
    `;
}

// ── Interactions ───────────────────────────────────────────
function toggleCampaign(el) {
    const card = el.closest('.campaign-card');
    const phases = card.querySelector('.campaign-phases, .campaign-items');
    if (phases) phases.classList.toggle('collapsed');
}

function togglePhase(el) {
    const chevron = el.querySelector('.phase-chevron');
    const items = el.nextElementSibling;
    if (items) items.classList.toggle('collapsed');
    if (chevron) chevron.classList.toggle('open');
}

// ── Detail panel ──────────────────────────────────────────
function showItemDetail(itemId) {
    const item = allItemsById[itemId];
    if (!item) return;

    const panel = document.getElementById('detail-panel');
    const title = document.getElementById('detail-title');
    const status = document.getElementById('detail-status');
    const body = document.getElementById('detail-body');

    title.textContent = item.title;
    status.innerHTML = `<span class="status-badge status-${item.status === 'completed' ? 'completed' : item.status === 'in_progress' ? 'active' : 'paused'}">${item.status}</span>`;

    let html = '';

    // Type
    const icon = TYPE_ICONS[item.type] || '📋';
    html += `<div class="detail-section">
        <div class="detail-section-title">Type</div>
        <div class="detail-section-content">${icon} ${item.type.replace('_', ' ')}</div>
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
            <table class="spec-table">${renderSpec(item.imaging_spec)}</table>
        </div>`;
    }

    // Bench spec
    if (item.bench_spec) {
        html += `<div class="detail-section">
            <div class="detail-section-title">Bench Specification</div>
            <table class="spec-table">${renderSpec(item.bench_spec)}</table>
        </div>`;
    }

    // Dependencies
    if (item.depends_on && item.depends_on.length > 0) {
        const deps = item.depends_on.map(id => {
            const dep = allItemsById[id];
            return dep ? `<span class="dep-chip">${esc(dep.title)}</span>` : `<span class="dep-chip">${id.slice(0, 8)}</span>`;
        }).join('');
        html += `<div class="detail-section">
            <div class="detail-section-title">Depends on</div>
            <div class="dep-list">${deps}</div>
        </div>`;
    }

    body.innerHTML = html;
    panel.classList.remove('hidden');
}

function renderSpec(spec) {
    const LABELS = {
        strain: 'Strain',
        genotype: 'Genotype',
        reporter: 'Reporter',
        sample_prep: 'Sample Prep',
        temperature_c: 'Temperature',
        num_embryos: 'Embryos',
        num_slices: 'Z Slices',
        exposure_ms: 'Exposure',
        laser_wavelength_nm: 'Laser',
        laser_power_pct: 'Power',
        interval_s: 'Interval',
        target_window: 'Dev. Window',
        start_stage: 'Start Stage',
        stop_condition: 'Stop Condition',
        estimated_duration_h: 'Duration',
        success_criteria: 'Success Criteria',
        comparison_to: 'Compare To',
        protocol: 'Protocol',
        reagents: 'Reagents',
        strains: 'Strains',
        target_genotype: 'Target Genotype',
        estimated_days: 'Est. Days',
        notes: 'Notes',
    };

    const UNITS = {
        temperature_c: '°C',
        exposure_ms: ' ms',
        laser_wavelength_nm: ' nm',
        laser_power_pct: '%',
        interval_s: 's',
        estimated_duration_h: ' hrs',
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

function closeDetail() {
    document.getElementById('detail-panel')?.classList.add('hidden');
}

// ── Utilities ─────────────────────────────────────────────
function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}
