/**
 * Experiment Overview Tab — vector-graphics view of the live imaging tactics
 * (cadence patterns + reactive-monitoring rules) for the running experiment.
 *
 * Data source: GET /api/experiments/current/strategy — the live snapshot from
 * FileStore. When there is no active experiment (or the fetch isn't ready), the
 * view shows a calm empty state; it never renders stubbed/mock data.
 */


const ExperimentOverview = {
    initialized: false,
    expandedMode: null,
    activeView: 'overview',  // 'overview' | 'rules'
    activeStrategy: null,    // last fetched strategy snapshot (rules view)
    activePlan: null,        // last fetched/loaded operation plan (overview spine)
    isLive: false,           // true when data came from the API
    scenarioMode: false,     // true when ?scenario=<name> is active
    _subscribed: false,      // guard: prevents double-registration across tab re-clicks
    _planRefreshTimer: null, // debounce handle for tactic-event-driven refetch
    _tempUpdateHandler: null,// stored handler ref so it can be off()'d if needed
    _rosterEmbryos: [],      // embryos from /api/embryos/positions (D2 roster lens)
    _rolesMap: null,         // Map(role name → registry obj) from /api/roles (D2 roster lens)
    _currentSessionId: null, // session_id from /api/operation_plan/current (always, even idle)
    _planPickerOpen: false,  // whether the plan-link picker is visible
    _pickerItems: null,      // flat plan items for the picker (null=not loaded)
    _expandedTacticIds: new Set(), // tactic expand-keys for click-to-expand (survives refresh)

    async init() {
        console.log('[ExperimentOverview] init() called, view=', this.activeView);

        // Scenario dev mode: ?scenario=<name> renders a fixture with no fetch.
        // Guard against double-registration when the tab is clicked repeatedly.
        const scenarioParam = new URLSearchParams(location.search).get('scenario');
        if (scenarioParam && window.OPERATIONS_SCENARIOS &&
            Object.prototype.hasOwnProperty.call(window.OPERATIONS_SCENARIOS, scenarioParam)) {
            this.scenarioMode = true;
            this.activePlan = window.OPERATIONS_SCENARIOS[scenarioParam];
            this.activeStrategy = null;
            this.isLive = false;
            this.render(null);
            this.initialized = true;
            return;
        }

        this.scenarioMode = false;
        // Fetch plan (overview) and strategy (rules) in parallel so tab-switching
        // between the two views doesn't require a second round-trip.
        // D2: also fetch roster + roles for the roster lens.
        const [plan, strategy, rosterEmbryos, rolesMap] = await Promise.all([
            this.loadPlan(),
            this.loadStrategy(),
            this._loadRoster(),
            this._loadRolesMap(),
        ]);
        this.activePlan = plan;
        this.activeStrategy = strategy;
        this._rosterEmbryos = rosterEmbryos;
        this._rolesMap = rolesMap;
        this.isLive = plan !== null || strategy !== null;
        this.render(strategy);
        this.initialized = true;

        // Subscribe to tactic-state events once per page load.
        // Guard: _subscribed prevents double-registration across tab re-clicks.
        // Skip entirely in scenario mode — no live backend, no websocket.
        if (!this._subscribed) {
            this._subscribed = true;
            const refresh = () => this._debouncedRefresh();
            // Plan-changing events: re-fetch the whole plan after debounce.
            // CONTEXT_UPDATED fires when OperationPlanUpdater patches the plan.
            // The tactic-lifecycle events fire on transitions the updater also
            // reacts to, so they all funnel into the same debounced refetch.
            const TACTIC_EVENTS = [
                'CONTEXT_UPDATED',
                'TEMP_PROTOCOL_STARTED', 'TEMP_PROTOCOL_COMPLETED',
                'BURST_START', 'BURST_COMPLETE',
                'EMBRYO_CADENCE_CHANGED', 'TEMPERATURE_SETPOINT_CHANGED',
                'POWER_RAMP_STEP',
            ];
            TACTIC_EVENTS.forEach(ev => ClientEventBus.on(ev, refresh));

            // High-frequency temperature binding (~1 Hz).
            // Updates the active scripted_protocol tactic's temperature gauge
            // IN PLACE — no plan refetch, no full re-render.
            this._tempUpdateHandler = (data) => this._handleTempUpdate(data);
            ClientEventBus.on('TEMPERATURE_UPDATE', this._tempUpdateHandler);
        }
    },

    async loadStrategy() {
        try {
            const resp = await fetch('/api/experiments/current/strategy', {
                cache: 'no-store'
            });
            if (!resp.ok) {
                // No active experiment / not ready yet — show the empty state,
                // never stubbed data.
                console.warn('[ExperimentOverview] strategy fetch returned', resp.status);
                return null;
            }
            const data = await resp.json();
            return data;
        } catch (e) {
            console.warn('[ExperimentOverview] strategy fetch error:', e);
            return null;
        }
    },

    // Fetch the agent-authored Operation Plan for the current session.
    // Returns the plan object (plan.tactics etc.) or null when unavailable.
    // Always captures data.session_id in _currentSessionId so the Linked-plans
    // panel can work even when no operation plan is active.
    async loadPlan() {
        try {
            const resp = await fetch('/api/operation_plan/current', { cache: 'no-store' });
            if (!resp.ok) {
                console.warn('[ExperimentOverview] plan fetch returned', resp.status);
                return null;
            }
            const data = await resp.json();
            // Capture session_id regardless of plan availability — used by the
            // Linked-plans panel which is session-scoped, not plan-scoped.
            this._currentSessionId = data.session_id || null;
            if (!data.available) return null;
            return data.plan || null;
        } catch (e) {
            console.warn('[ExperimentOverview] plan fetch error:', e);
            return null;
        }
    },

    // Debounced plan refetch — coalesces rapid tactic-event bursts into a single
    // fetch+render.  500 ms window matches experiment-strip.js convention.
    // D2: also re-fetches the embryo roster so the lens stays current.
    _debouncedRefresh() {
        if (this._planRefreshTimer) clearTimeout(this._planRefreshTimer);
        this._planRefreshTimer = setTimeout(async () => {
            this._planRefreshTimer = null;
            const [plan, rosterEmbryos] = await Promise.all([
                this.loadPlan(),
                this._loadRoster(),
            ]);
            this.activePlan = plan;
            this._rosterEmbryos = rosterEmbryos;
            this.isLive = plan !== null;
            this.render(this.activeStrategy);
        }, 500);
    },

    // In-place temperature gauge update — called at ~1 Hz by TEMPERATURE_UPDATE.
    // Finds the active scripted_protocol tactic's temperature readout in the DOM
    // and rewrites only that element's value, never refetching the plan.
    // No-op when there is no active scripted_protocol tactic with temperature binding.
    _handleTempUpdate(data) {
        if (!data || !data.sample) return;
        const plan = this.activePlan;
        if (!plan || !Array.isArray(plan.tactics)) return;
        // Only act when an active scripted_protocol tactic declares temperature binding.
        const activeTactic = plan.tactics.find(
            t => t.state === 'active'
                && t.kind === 'scripted_protocol'
                && Array.isArray(t.live_bind)
                && t.live_bind.includes('temperature')
        );
        if (!activeTactic) return;

        const root = document.getElementById('experiment-overview-root');
        if (!root) return;
        // _renderOpsReadout stamps data-livebind="temperature" on the gauge div
        // when the readout label normalises to "temperature".
        const gauge = root.querySelector('.ops-node.active .ops-gauge[data-livebind="temperature"]');
        if (!gauge) return;
        const gv = gauge.querySelector('.ops-gv');
        if (!gv) return;

        const s = data.sample;
        const water = s.water_c != null
            ? parseFloat(s.water_c).toFixed(1) + '°C'
            : '—';
        const sp = s.setpoint_c != null
            ? ' → <span class="ops-set">'
                + parseFloat(s.setpoint_c).toFixed(1)
                + '°C</span>'
            : '';
        gv.innerHTML = water + sp;
    },

    setView(view) {
        if (view === this.activeView) return;
        this.activeView = view;
        // Update view-switcher button state
        document.querySelectorAll('[data-experiment-view]').forEach(b => {
            b.classList.toggle('active', b.dataset.experimentView === view);
        });
        // Re-render against the last fetched strategy (no re-fetch on tab
        // switch — refresh happens on tab activation in the bootstrap).
        this.render(this.activeStrategy);
    },

    render(s) {
        const root = document.getElementById('experiment-overview-root');
        if (!root) {
            console.error('[ExperimentOverview] #experiment-overview-root NOT FOUND in DOM');
            return;
        }
        // Tear down any prior ticker before we blow away the SVG it pointed at.
        this._stopNowTicker();
        // Reset plan-picker state on each full render so the picker doesn't persist
        // across tactic-event-driven re-renders.
        this._planPickerOpen = false;
        this._pickerItems = null;
        // Rules view requires the strategy snapshot; show an empty state when absent.
        // Overview view uses this.activePlan — the null/empty case is handled inside
        // _renderOperationSpine (it renders the idle state).
        if (this.activeView === 'rules' && !s) {
            root.innerHTML = '<div style="padding:32px;text-align:center;color:var(--text-muted,#94a3b8);font-size:13px;">' +
                'No active experiment — rules and monitoring modes will appear here once a run is live.</div>';
            return;
        }
        try {
            root.innerHTML = '';
            if (this.activeView === 'rules') {
                this._renderRulesView(root, s);
            } else {
                // Operation spine — data-driven tactic plan renderer.
                // The swimlane view is retired; this renders this.activePlan.
                this._renderOperationSpine(root, this.activePlan);
            }
            // Kick off the async Linked-plans panel (overview tab only).
            // Fire-and-forget: appends a placeholder immediately, fills after fetch.
            if (this.activeView === 'overview') {
                this._initLinkedPlansPanel(root).catch(e =>
                    console.warn('[ExperimentOverview] linked-plans panel error:', e));
            }
            console.log('[ExperimentOverview] rendered OK, view=', this.activeView);
        } catch (err) {
            console.error('[ExperimentOverview] render failed:', err);
            root.innerHTML = `<div style="padding:20px;color:#ef4444;font-family:monospace;font-size:12px;">
                Render error: ${err.message}<br>
                <pre style="margin-top:8px;font-size:11px;color:#888;white-space:pre-wrap;">${err.stack || ''}</pre>
            </div>`;
        }
    },


    _stopNowTicker() {
        if (this._nowTickerHandle) {
            clearTimeout(this._nowTickerHandle);
            this._nowTickerHandle = null;
        }
    },


    // =================================================================
    // Linked-plans panel — session ↔ plan items (F / Task 4)
    // Symmetric with the Plans-tab Sessions section (campaigns.js).
    // Endpoints: GET /api/sessions/{id}/plans
    //            POST /api/campaigns/{cid}/items/{iid}/sessions
    //            DELETE .../sessions/{session_id}
    // =================================================================

    // Initialise and append the Linked-plans panel to the ops-wrap in root.
    // Async: appends a loading placeholder immediately, fills after fetch.
    // No-op when no session_id is known (e.g. store not yet initialised).
    async _initLinkedPlansPanel(root) {
        if (!this._currentSessionId) return;
        const wrap = root.querySelector('.ops-wrap');
        if (!wrap) return;

        // Append placeholder before the async fetch so layout is stable.
        const panelEl = document.createElement('div');
        panelEl.className = 'ops-lp';
        panelEl.innerHTML = '<div class="ops-lp-loading">Loading linked plans…</div>';
        wrap.appendChild(panelEl);

        const sid = this._currentSessionId;
        try {
            const [linkedData, campaignsData] = await Promise.all([
                fetch(`/api/sessions/${encodeURIComponent(sid)}/plans`, { cache: 'no-store' })
                    .then(r => r.ok ? r.json() : { plans: [] })
                    .catch(() => ({ plans: [] })),
                fetch('/api/campaigns', { cache: 'no-store' })
                    .then(r => r.ok ? r.json() : { campaigns: [] })
                    .catch(() => ({ campaigns: [] })),
            ]);
            const plans = linkedData.plans || [];
            const campaignNameMap = {};
            this._flattenCampaignNames(campaignsData.campaigns || [], campaignNameMap);
            this._fillLinkedPlansPanel(panelEl, plans, campaignNameMap, sid);
        } catch (e) {
            console.warn('[ExperimentOverview] linked-plans init error:', e);
            panelEl.innerHTML = '<div class="ops-lp-loading">Could not load linked plans.</div>';
        }
    },

    // Build a map of campaign_id → display name from the /api/campaigns tree.
    _flattenCampaignNames(trees, map) {
        const walk = (tree) => {
            const c = tree.campaign;
            if (c && c.id) map[c.id] = c.description || c.shorthand || c.id;
            for (const child of (tree.children || [])) walk(child);
        };
        for (const tree of (trees || [])) walk(tree);
    },

    // Build a flat list of {id, title, status, campaign_id, campaign_name}
    // from the /api/campaigns tree, for the plan-item picker.
    _flattenCampaignItems(trees, campaignNameMap) {
        const items = [];
        const walk = (tree, inheritedCid, inheritedName) => {
            const c = tree.campaign;
            const cid  = (c && c.id)  || inheritedCid;
            const name = (c && (c.description || c.shorthand)) || inheritedName || cid;
            for (const item of (tree.items || [])) {
                items.push({
                    id:            item.id,
                    title:         item.title || item.id,
                    status:        typeof item.status === 'string' ? item.status
                                   : (item.status && item.status.value) || 'planned',
                    campaign_id:   cid,
                    campaign_name: name,
                });
            }
            for (const child of (tree.children || [])) walk(child, cid, name);
        };
        for (const tree of (trees || [])) walk(tree, null, null);
        return items;
    },

    // Render the linked-plans panel HTML into panelEl and wire button events.
    _fillLinkedPlansPanel(panelEl, plans, campaignNameMap, sessionId) {
        const ESC = this._opsESC.bind(this);

        // Header: section label + "+ link to a plan" button
        let html = `<div class="ops-lp-head">
            <span class="ops-lp-title">Linked plans</span>
            <button class="ops-lp-link-btn" id="ops-lp-link-btn">+ link to a plan</button>
        </div>`;

        // Linked plan-item rows (title · campaign · status · delink)
        if (plans.length > 0) {
            html += '<div class="ops-lp-list">';
            for (const p of plans) {
                const cname = campaignNameMap[p.campaign_id] || p.campaign_id || '—';
                const sCls = p.status === 'completed' ? 'done'
                    : p.status === 'in_progress'       ? 'active' : 'planned';
                html += `<div class="ops-lp-row">
                    <span class="ops-lp-row-title">${ESC(p.title || p.id)}</span>
                    <span class="ops-lp-row-campaign">${ESC(cname)}</span>
                    <span class="ops-lp-row-status ops-lp-status-${sCls}">${ESC(p.status || 'planned')}</span>
                    <button class="ops-lp-delink"
                        data-item-id="${ESC(p.id)}"
                        data-campaign-id="${ESC(p.campaign_id)}"
                        title="Delink this plan item">×</button>
                </div>`;
            }
            html += '</div>';
        } else {
            html += '<div class="ops-lp-empty">Not linked to any plan</div>';
        }

        // Inline picker — shown when _planPickerOpen is set
        if (this._planPickerOpen) {
            if (this._pickerItems === null) {
                // Still fetching — show loading state
                html += '<div class="ops-lp-picker ops-lp-picker--loading">Loading plan items…</div>';
            } else {
                const linkedIds = new Set(plans.map(p => p.id));
                const available = this._pickerItems.filter(it => !linkedIds.has(it.id));
                const opts = available.length === 0
                    ? '<option value="">No other plan items available</option>'
                    : available.map(it =>
                        `<option value="${ESC(it.campaign_id)}::${ESC(it.id)}">${ESC(it.campaign_name)}: ${ESC(it.title)}</option>`
                      ).join('');
                html += `<div class="ops-lp-picker">
                    <select class="ops-lp-picker-sel" id="ops-lp-picker-sel">${opts}</select>
                    <div class="ops-lp-picker-actions">
                        <button class="ops-lp-picker-link-btn" id="ops-lp-picker-link">Link</button>
                        <button class="ops-lp-picker-cancel-btn" id="ops-lp-picker-cancel">Cancel</button>
                    </div>
                </div>`;
            }
        }

        panelEl.innerHTML = html;

        // Wire events directly on the rendered buttons.
        const sid = sessionId;
        const linkBtn = panelEl.querySelector('#ops-lp-link-btn');
        if (linkBtn) {
            linkBtn.addEventListener('click', () =>
                this._openPlanPickerInPanel(panelEl, plans, campaignNameMap, sid));
        }
        panelEl.querySelectorAll('.ops-lp-delink').forEach(btn => {
            btn.addEventListener('click', () =>
                this._delinkPlanItem(panelEl, btn.dataset.itemId, btn.dataset.campaignId, sid));
        });
        const submitBtn = panelEl.querySelector('#ops-lp-picker-link');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this._submitPlanLink(panelEl, sid));
        }
        const cancelBtn = panelEl.querySelector('#ops-lp-picker-cancel');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
                this._planPickerOpen = false;
                this._pickerItems = null;
                this._fillLinkedPlansPanel(panelEl, plans, campaignNameMap, sid);
            });
        }
    },

    // Open the inline plan-item picker: set loading state, fetch /api/campaigns,
    // flatten to items, re-render with the picker populated.
    async _openPlanPickerInPanel(panelEl, plans, campaignNameMap, sessionId) {
        this._planPickerOpen = true;
        this._pickerItems = null;  // show loading
        this._fillLinkedPlansPanel(panelEl, plans, campaignNameMap, sessionId);
        try {
            const res = await fetch('/api/campaigns', { cache: 'no-store' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this._pickerItems = this._flattenCampaignItems(data.campaigns || [], campaignNameMap);
        } catch (err) {
            console.error('[ExperimentOverview] plan picker fetch error:', err);
            this._pickerItems = [];
        }
        if (this._planPickerOpen) {
            this._fillLinkedPlansPanel(panelEl, plans, campaignNameMap, sessionId);
        }
    },

    // POST the selected plan item → session link, then refetch and re-render.
    async _submitPlanLink(panelEl, sessionId) {
        const select = panelEl.querySelector('#ops-lp-picker-sel');
        const value  = select && select.value;
        if (!value || !value.includes('::')) return;
        const [campaignId, itemId] = value.split('::', 2);
        if (!campaignId || !itemId) return;

        this._planPickerOpen = false;
        this._pickerItems = null;
        panelEl.innerHTML = '<div class="ops-lp-loading">Linking…</div>';
        try {
            const res = await fetch(
                `/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(itemId)}/sessions`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId }),
                },
            );
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
        } catch (err) {
            console.error('[ExperimentOverview] plan link error:', err);
        }
        await this._refetchLinkedPlans(panelEl, sessionId);
    },

    // DELETE the plan-item → session edge, then refetch and re-render.
    async _delinkPlanItem(panelEl, itemId, campaignId, sessionId) {
        panelEl.innerHTML = '<div class="ops-lp-loading">Unlinking…</div>';
        try {
            const res = await fetch(
                `/api/campaigns/${encodeURIComponent(campaignId)}/items/${encodeURIComponent(itemId)}/sessions/${encodeURIComponent(sessionId)}`,
                { method: 'DELETE' },
            );
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
        } catch (err) {
            console.error('[ExperimentOverview] plan delink error:', err);
        }
        await this._refetchLinkedPlans(panelEl, sessionId);
    },

    // Refetch linked plans + campaigns after a link/delink op, then re-render.
    async _refetchLinkedPlans(panelEl, sessionId) {
        try {
            const [linkedData, campaignsData] = await Promise.all([
                fetch(`/api/sessions/${encodeURIComponent(sessionId)}/plans`, { cache: 'no-store' })
                    .then(r => r.ok ? r.json() : { plans: [] })
                    .catch(() => ({ plans: [] })),
                fetch('/api/campaigns', { cache: 'no-store' })
                    .then(r => r.ok ? r.json() : { campaigns: [] })
                    .catch(() => ({ campaigns: [] })),
            ]);
            const plans = linkedData.plans || [];
            const campaignNameMap = {};
            this._flattenCampaignNames(campaignsData.campaigns || [], campaignNameMap);
            this._fillLinkedPlansPanel(panelEl, plans, campaignNameMap, sessionId);
        } catch (e) {
            console.warn('[ExperimentOverview] linked-plans refetch error:', e);
            panelEl.innerHTML = '<div class="ops-lp-loading">Could not reload linked plans.</div>';
        }
    },



    _renderRulesView(root, s) {
        // Compact header echoing the session identity
        const header = el('div', 'expov-header');
        const metaRow = el('div', 'expov-header-row expov-header-row-meta');
        metaRow.appendChild(elText('span', 'expov-session-name', s.session_name));
        metaRow.appendChild(elText('span', 'expov-session-id', s.session_id));
        header.appendChild(metaRow);
        root.appendChild(header);

        // Active monitoring modes (context for the rules)
        root.appendChild(this._renderModes(s));
        root.appendChild(this._renderModeExpanded(s));

        // The rules table
        root.appendChild(this._renderRulesTable(s));
    },


    // -----------------------------------------------------------------
    // Monitoring mode chips + expanded panel
    // -----------------------------------------------------------------
    _renderModes(s) {
        const wrap = el('div', 'expov-modes');
        if (!s.monitoring_modes || s.monitoring_modes.length === 0) {
            const chip = el('div', 'expov-mode-chip idle');
            chip.appendChild(elText('span', 'expov-mode-name', 'Idle'));
            chip.appendChild(elText('span', 'expov-mode-desc',
                'no reactive monitoring installed'));
            wrap.appendChild(chip);
            return wrap;
        }
        s.monitoring_modes.forEach(m => {
            const chip = el('div', 'expov-mode-chip');
            chip.appendChild(elText('span', 'expov-mode-name',
                this._humanizeModeName(m.name)));
            chip.appendChild(elText('span', 'expov-mode-desc',
                this._modeSummary(m)));
            chip.appendChild(elText('span', 'expov-mode-scope',
                m.applies_to_roles.join(',')));
            chip.title = m.description;  // native tooltip for full text
            chip.addEventListener('click', () => {
                this.expandedMode = (this.expandedMode === m.name) ? null : m.name;
                this.render(s);
            });
            wrap.appendChild(chip);
        });
        return wrap;
    },

    _modeSummary(m) {
        // One-line param preview that fits inside the chip
        const p = m.params || {};
        if (m.name === 'expression_monitoring') {
            return `→ ${p.fast_interval}s on signal · 488 ↓ to ${p.rampdown_floor_pct}%`;
        }
        if (m.name === 'pre_terminal_monitoring') {
            return `→ ${p.fast_interval}s on pretzel`;
        }
        return Object.entries(p).map(([k, v]) => `${k}=${v}`).join(' · ');
    },

    _renderModeExpanded(s) {
        const wrap = el('div', 'expov-mode-expanded');
        if (!this.expandedMode) return wrap;
        const m = s.monitoring_modes.find(x => x.name === this.expandedMode);
        if (!m) return wrap;
        wrap.classList.add('show');
        const params = Object.entries(m.params || {})
            .map(([k, v]) => `<code>${k}=${v}</code>`).join(' ');
        wrap.innerHTML = `
            <strong>${this._humanizeModeName(m.name)}</strong> — ${m.description}<br>
            <span style="margin-top:6px;display:inline-block;">
                applies to roles: ${m.applies_to_roles.map(r => `<code>${r}</code>`).join(', ')}
                · params: ${params}
            </span>`;
        return wrap;
    },

    _humanizeModeName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    },


    // =================================================================
    // Operation Spine — data-driven plan renderer (replaces swimlanes)
    // =================================================================

    // Minimal HTML escaper — values in readouts may contain trusted HTML
    // (e.g. <span class="ops-set">32.0°C</span>) so they are rendered with
    // innerHTML; all other user/model strings go through _opsESC.
    _opsESC(s) {
        return String(s == null ? '' : s)
            .replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    },

    // Entry point: render the operation spine into `root`.
    // plan = the plan object (tactics array) or null for idle/unavailable.
    _renderOperationSpine(root, plan) {
        const ESC = this._opsESC.bind(this);

        if (!plan || !Array.isArray(plan.tactics) || plan.tactics.length === 0) {
            root.innerHTML = `
                <div class="ops-wrap">
                    <div class="ops-crumb">Operations</div>
                    <h1 class="ops-title">No operation running</h1>
                    <div class="ops-meta">Brief the agent — it will declare a tactic plan and the spine renders live.</div>
                    <div class="ops-setup-cta">
                        <button class="ops-brief-btn" data-ops-brief>Brief the agent</button>
                        <div class="ops-chips-label">— or start from a template</div>
                        <div class="ops-chips">
                            <button class="ops-chip" data-ops-prompt="Set up a temperature-strain operation: baseline monitoring, a temperature-change protocol on the test subjects, then recovery monitoring.">Temperature-strain run</button>
                            <button class="ops-chip" data-ops-prompt="Set up a standing timelapse on all embryos at 120 s cadence.">Standing timelapse</button>
                            <button class="ops-chip" data-ops-prompt="Arm a reactive monitor that watches for hatching on the most advanced embryos.">Watch for hatching</button>
                        </div>
                    </div>
                </div>`;

            // Wire CTA buttons — same open+send pattern as landing.js sendFreeform.
            function _opsOpenAgent(prompt) {
                if (typeof AgentChat === 'undefined' || !AgentChat.togglePanel) return;
                AgentChat.togglePanel(true);
                if (prompt && AgentChat.runCommand) setTimeout(() => AgentChat.runCommand(prompt), 300);
            }
            const briefBtn = root.querySelector('[data-ops-brief]');
            if (briefBtn) briefBtn.addEventListener('click', () => _opsOpenAgent(''));
            root.querySelectorAll('[data-ops-prompt]').forEach(chip => {
                chip.addEventListener('click', () => _opsOpenAgent(chip.dataset.opsPrompt));
            });
            return;
        }

        const tactics = plan.tactics;
        const hasActive = tactics.some(t => t.state === 'active');
        // Index of the first queued (planned) tactic — gets the "next" badge.
        const firstPlannedIdx = tactics.findIndex(t => t.state === 'planned');

        // Roster lens: visible when embryos + role metadata are both available.
        // Gracefully absent if either fetch failed or returned nothing (backward compat
        // with plans that pre-date D2 — spine renders exactly as before).
        const rosterEmbryos = this._rosterEmbryos || [];
        const rolesMap = this._rolesMap;
        const rosterCtx = (rosterEmbryos.length > 0 && rolesMap && rolesMap.size > 0)
            ? { embryos: rosterEmbryos, rolesMap }
            : null;
        const rosterHtml = rosterCtx
            ? `<div class="ops-section-label">Population roster — by class &amp; role</div>
               ${this._renderRosterLens(rosterEmbryos, rolesMap, plan, ESC)}`
            : '';
        const spineLabel = rosterCtx
            ? '<div class="ops-section-label">Tactic spine — role-scoped</div>'
            : '';

        const spineNodes = tactics
            .map((t, idx) => this._renderOpsTactic(t, idx, firstPlannedIdx, ESC, rosterCtx, tactics))
            .join('');

        root.innerHTML = `
            <div class="ops-wrap">
                <div class="ops-crumb">Operations · ${hasActive ? 'live' : 'idle'}</div>
                <h1 class="ops-title">${ESC(plan.title || '')}</h1>
                <div class="ops-meta">${ESC(plan.session_id || '')}${plan.goal ? ' · ' + ESC(plan.goal) : ''}</div>
                <div class="ops-legend">
                    <span><i style="background:var(--ops-done)"></i>done</span>
                    <span><i style="background:var(--ops-active)"></i>in use</span>
                    <span><i style="background:var(--ops-plan)"></i>queued</span>
                </div>
                ${rosterHtml}
                ${spineLabel}
                <div class="ops-spine">${spineNodes}</div>
            </div>`;
        this._wireSpineExpand(root);
    },

    // Render a single tactic node.
    // rosterCtx = { embryos, rolesMap } | null — when present, adds role-scope badge (D2).
    // tactics = full tactics array — used to resolve relation ids to names.
    _renderOpsTactic(t, idx, firstPlannedIdx, ESC, rosterCtx = null, tactics = []) {
        const STATE_LABEL = { done: 'done', active: 'in use', planned: 'queued', paused: 'paused' };
        const seq = String(t.seq || idx + 1).padStart(2, '0');
        const stateLabel = STATE_LABEL[t.state] || t.state;
        // First queued tactic gets a "next" badge — COCKED instrument marker.
        const isFirstQueued = t.state === 'planned' && idx === firstPlannedIdx;
        const nextBadge = isFirstQueued
            ? '<span class="ops-next-badge">next</span>'
            : '';

        const live = t.live || {};
        const target = live.target || '';
        const summary = live.summary || '';
        const desc = live.desc || '';

        // Scope chip/badge — compact chip for planned/done/paused (always visible, no embryo
        // list needed); full badge with embryo resolution for active state (D2 roster lens).
        const isExpandable = (t.state === 'planned' || t.state === 'done' || t.state === 'paused');
        const expandKey    = this._tacticExpandKey(t);
        const scopeBadge = isExpandable
            ? this._renderOpsScopeChip(t.scope, ESC)
            : (rosterCtx ? this._renderOpsScopeBadge(t.scope, rosterCtx.embryos, rosterCtx.rolesMap, ESC) : '');

        // Chevron toggle — only for expandable (planned / done / paused) tactics.
        const chevron = isExpandable
            ? `<button class="ops-expand-chevron${this._expandedTacticIds.has(expandKey) ? ' open' : ''}" aria-label="Toggle tactic details" title="Toggle details">›</button>`
            : '';

        // Header row: name · target · scope badge · summary · chevron
        let inner = `
            <div class="ops-row">
                <span class="ops-tname">${ESC(t.name)}</span>
                ${target ? `<span class="ops-target">${ESC(target)}</span>` : ''}
                ${scopeBadge}
                ${summary ? `<span class="ops-tsum">${ESC(summary)}</span>` : ''}
                ${chevron}
            </div>
            ${desc ? `<div class="ops-desc">${ESC(desc)}</div>` : ''}`;

        // AUDIT: FLATTEN the active card — readouts on the panel face, separated by
        // a hairline rule. No nested card-in-card boxes.
        if (t.state === 'active' && live.readouts && live.readouts.length) {
            inner += `<hr class="ops-rule">
                <div class="ops-live-strip">
                    ${live.readouts.map(r => this._renderOpsReadout(r, ESC)).join('')}
                </div>`;
        }

        // Kind-specific structure for the active state.
        if (t.state === 'active') {
            inner += this._renderOpsKindActive(t, live, ESC);
        } else if (t.state === 'planned') {
            inner += this._renderOpsKindPlanned(t, ESC);
            inner += this._renderOpsExpandBody(t, ESC, tactics);
        }

        // Fix #1: surface flat live.* telemetry keys not covered by structured
        // readouts/phases.  Render for active (in-progress telemetry) and done
        // (completion data such as sustained_hz, mp4_path, last_fired).
        // Skip planned — no live data is bound yet.
        if (t.state === 'active' || t.state === 'done') {
            const SKIP = new Set(['readouts', 'phases', 'target', 'summary', 'desc']);
            const flatEntries = Object.entries(live).filter(([k]) => !SKIP.has(k));
            if (flatEntries.length) {
                const humanKey = k => k.replace(/_/g, ' ');
                const pairs = flatEntries.map(([k, v]) => {
                    const vStr = v == null ? '—' : String(v);
                    return `<span class="ops-lf-pair"><span class="ops-lf-k">${ESC(humanKey(k))}</span><span class="ops-lf-v">${ESC(vStr)}</span></span>`;
                }).join('');
                inner += `<div class="ops-livefacts">${pairs}</div>`;
            }
        }

        // Expand body for done + paused tactics (appended after live-facts).
        if (t.state === 'done' || t.state === 'paused') {
            inner += this._renderOpsExpandBody(t, ESC, tactics);
        }

        const cardExpandAttr = isExpandable
            ? ` data-tactic-expand-id="${ESC(expandKey)}"`
            : '';

        return `
            <div class="ops-node ${ESC(t.state)}">
                <div class="ops-stagelab">${seq} · ${stateLabel}${nextBadge ? ' ' + nextBadge : ''}</div>
                <div class="ops-card"${cardExpandAttr}>${inner}</div>
            </div>`;
    },

    // -----------------------------------------------------------------
    // Expandable tactic detail — click-to-expand for queued/done/paused
    // -----------------------------------------------------------------

    // Stable key used to persist expand state across re-renders.
    _tacticExpandKey(t) {
        return String(t.id || t.seq || t.name || '');
    },

    // Compact inline scope chip — always shown in the collapsed header.
    // Works without embryo list; uses this._rolesMap for role color if available.
    _renderOpsScopeChip(scope, ESC) {
        const rolesMap = this._rolesMap;
        if (!scope || scope.mode === 'global') {
            return '<span class="ops-scope-chip ops-scope-global">global</span>';
        }
        if (scope.mode === 'role') {
            const role = scope.role || '';
            const roleInfo = rolesMap ? rolesMap.get(role) : null;
            const color  = (roleInfo && roleInfo.ui_color) || '#8b949e';
            const bg     = this._hexToRgba(color, 0.12);
            const border = this._hexToRgba(color, 0.35);
            return `<span class="ops-scope-chip" style="color:${ESC(color)};background:${bg};border-color:${border}">role: ${ESC(role)}</span>`;
        }
        if (scope.mode === 'embryos') {
            const ids = (scope.embryo_ids || []).join(', ');
            return `<span class="ops-scope-chip ops-scope-global">${ESC(ids) || '—'}</span>`;
        }
        return '';
    },

    // Expanded detail body — rationale, scope, structure, relations, live.readouts.
    // Rendered hidden by default; _wireSpineExpand toggles the `hidden` class.
    _renderOpsExpandBody(t, ESC, tactics) {
        const rows = [];

        // Rationale
        if (t.rationale) {
            rows.push(`<div class="ops-expand-row">
                <span class="ops-expand-key">rationale</span>
                <span class="ops-expand-val">${ESC(t.rationale)}</span>
            </div>`);
        }

        // Scope — resolved readable form
        const scope = t.scope;
        if (scope) {
            let scopeText = 'all embryos';
            if (scope.mode === 'role') {
                scopeText = `role: ${scope.role || ''}`;
            } else if (scope.mode === 'embryos') {
                scopeText = (scope.embryo_ids || []).join(', ') || '—';
            }
            rows.push(`<div class="ops-expand-row">
                <span class="ops-expand-key">scope</span>
                <span class="ops-expand-val">${ESC(scopeText)}</span>
            </div>`);
        }

        // Structure — kind-specific
        const struct = t.structure || {};
        if (t.kind === 'standing_timelapse' && struct.cadence_s != null) {
            rows.push(`<div class="ops-expand-row">
                <span class="ops-expand-key">cadence</span>
                <span class="ops-expand-val ops-expand-mono">${ESC(struct.cadence_s)}s</span>
            </div>`);
        }
        if (t.kind === 'scripted_protocol') {
            const phases = (struct.phases || []);
            if (phases.length) {
                const pHtml = phases.map(p =>
                    `<span class="ops-expand-phase">${ESC(p.name || p.state || '?')}</span>`
                ).join('<span class="ops-expand-arrow">→</span>');
                rows.push(`<div class="ops-expand-row">
                    <span class="ops-expand-key">phases</span>
                    <span class="ops-expand-val ops-expand-phases">${pHtml}</span>
                </div>`);
            }
        }
        if (t.kind === 'exclusive_burst' || t.kind === 'burst') {
            if (struct.frames != null) {
                rows.push(`<div class="ops-expand-row">
                    <span class="ops-expand-key">frames</span>
                    <span class="ops-expand-val ops-expand-mono">${ESC(struct.frames)}</span>
                </div>`);
            }
            if (struct.mode) {
                rows.push(`<div class="ops-expand-row">
                    <span class="ops-expand-key">mode</span>
                    <span class="ops-expand-val">${ESC(struct.mode)}</span>
                </div>`);
            }
        }
        if (t.kind === 'reactive_monitor' && struct.watch) {
            rows.push(`<div class="ops-expand-row">
                <span class="ops-expand-key">watch</span>
                <span class="ops-expand-val">${ESC(struct.watch)}</span>
            </div>`);
        }

        // Relations — resolve tactic IDs to names
        const relations = t.relations || {};
        const afterIds = Array.isArray(relations.after) ? relations.after : [];
        if (afterIds.length) {
            const tacticIdMap = {};
            for (const tac of (tactics || [])) {
                if (tac.id) tacticIdMap[tac.id] = tac.name || tac.id;
            }
            const names = afterIds.map(id => tacticIdMap[id] || id);
            rows.push(`<div class="ops-expand-row">
                <span class="ops-expand-key">runs after</span>
                <span class="ops-expand-val">${ESC(names.join(', '))}</span>
            </div>`);
        }

        // Live readouts (queued/done tactics may carry pre-set readout definitions)
        const live = t.live || {};
        if (live.readouts && live.readouts.length) {
            rows.push(`<div class="ops-expand-readouts">
                ${live.readouts.map(r => this._renderOpsReadout(r, ESC)).join('')}
            </div>`);
        }

        if (!rows.length) return '';

        const isOpen = this._expandedTacticIds.has(this._tacticExpandKey(t));
        return `<div class="ops-expand-body${isOpen ? '' : ' hidden'}">
            <hr class="ops-expand-rule">
            ${rows.join('\n')}
        </div>`;
    },

    // Wire click-to-expand on all expandable tactic cards in root after innerHTML is set.
    // Persist expand state in _expandedTacticIds so it survives debounced re-renders.
    _wireSpineExpand(root) {
        root.querySelectorAll('.ops-card[data-tactic-expand-id]').forEach(card => {
            const expandId = card.dataset.tacticExpandId;
            const body    = card.querySelector('.ops-expand-body');
            const chevron = card.querySelector('.ops-expand-chevron');
            if (!body) return;

            card.addEventListener('click', (e) => {
                // Let clicks on interactive elements inside the body bubble freely.
                if (e.target.closest('a, button:not(.ops-expand-chevron)')) return;
                const isExpanded = !body.classList.contains('hidden');
                if (isExpanded) {
                    body.classList.add('hidden');
                    if (chevron) chevron.classList.remove('open');
                    this._expandedTacticIds.delete(expandId);
                } else {
                    body.classList.remove('hidden');
                    if (chevron) chevron.classList.add('open');
                    this._expandedTacticIds.add(expandId);
                }
            });
        });
    },

    // Render a readout gauge. `r.value` may contain trusted HTML (span markup).
    // Stamps data-livebind on the outer div so _handleTempUpdate (and future
    // live-binding) can find the gauge in-place without a full re-render.
    // Priority: r.bind (explicit semantic key) > normalised r.label.
    _renderOpsReadout(r, ESC) {
        const bindKey = r.bind
            ? r.bind
            : (r.label
                ? r.label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
                : '');
        const bindAttr = bindKey ? ` data-livebind="${ESC(bindKey)}"` : '';
        return `<div class="ops-gauge"${bindAttr}>
            <div class="ops-gl">${ESC(r.label)}</div>
            <div class="ops-gv">${r.value}</div>
            ${r.bar != null
                ? `<div class="ops-tempbar"><i style="width:${Math.max(0, Math.min(100, r.bar))}%"></i></div>`
                : ''}
            ${r.sub ? `<div class="ops-gsub">${ESC(r.sub)}</div>` : ''}
        </div>`;
    },

    // Render one phase in the scripted_protocol stepper.
    // AUDIT: the active phase is the HEADLINE — CSS makes it larger.
    _renderOpsPhase(p, ESC) {
        const pips = (p.pips || [])
            .map(k => `<span class="ops-pip ${ESC(k)}"></span>`)
            .join('');
        const ic = p.state === 'done' ? '✓'
                 : p.state === 'active' ? '▶'
                 : (p.icon || '·');
        return `<div class="ops-ph ${ESC(p.state)}">
            <div class="ops-pht"><span class="ops-pi">${ic}</span>${ESC(p.name)}</div>
            <div class="ops-phc">${ESC(p.count || '')}</div>
            ${pips ? `<div class="ops-pips">${pips}</div>` : ''}
        </div>`;
    },

    // Kind-specific structure for ACTIVE tactics.
    _renderOpsKindActive(t, live, ESC) {
        if (!t.kind) return '';

        // scripted_protocol → before/during/after phase stepper.
        // Prefer live.phases (may carry pip/count state); fall back to structure.phases.
        if (t.kind === 'scripted_protocol') {
            const phases = live.phases || (t.structure && t.structure.phases) || [];
            if (!phases.length) return '';
            return `<div class="ops-phases">
                ${phases.map(p => this._renderOpsPhase(p, ESC)).join('')}
            </div>`;
        }

        // standing_timelapse → compact per-embryo cadence strip.
        if (t.kind === 'standing_timelapse') {
            const perEmbryo = t.structure && t.structure.per_embryo;
            if (!perEmbryo || !perEmbryo.length) return '';
            const rows = perEmbryo.map(e => {
                const intervalStr = e.interval_s != null ? `${ESC(e.interval_s)}s` : '—';
                return `<div class="ops-cadence-embryo">
                    <span class="ops-cadence-id">${ESC(e.embryo_id)}</span>
                    <span class="ops-cadence-phase ${ESC(e.cadence_phase)}">${ESC(e.cadence_phase)}</span>
                    <span class="ops-cadence-val">${intervalStr}</span>
                </div>`;
            }).join('');
            return `<hr class="ops-rule"><div class="ops-cadence-strip">${rows}</div>`;
        }

        // reactive_monitor → armed/watching/fired status badge.
        if (t.kind === 'reactive_monitor') {
            const st = (t.structure && t.structure.status) || 'armed';
            return `<div class="ops-monitor-status ops-monitor-${ESC(st)}">${ESC(st)}</div>`;
        }

        // exclusive_burst / oneshot / custom — readouts only (already rendered above).
        return '';
    },

    // Kind-specific structure for PLANNED (queued) tactics — compact hints.
    _renderOpsKindPlanned(t, ESC) {
        if (!t.kind) return '';

        if (t.kind === 'scripted_protocol') {
            const phases = (t.structure && t.structure.phases) || [];
            if (!phases.length) return '';
            return `<div class="ops-phases">
                ${phases.map(p => this._renderOpsPhase(p, ESC)).join('')}
            </div>`;
        }

        if (t.kind === 'standing_timelapse' && t.structure && t.structure.cadence_s) {
            return `<div class="ops-cadence-note">cadence · ${ESC(t.structure.cadence_s)}s</div>`;
        }

        if (t.kind === 'reactive_monitor' && t.structure && t.structure.watch) {
            return `<div class="ops-monitor-watch">watch · ${ESC(t.structure.watch)}</div>`;
        }

        if ((t.kind === 'oneshot' || t.kind === 'custom') && t.structure && t.structure.note) {
            return `<div class="ops-cadence-note">${ESC(t.structure.note)}</div>`;
        }

        return '';
    },

    // =================================================================
    // D2 — Roster Lens: embryo population by role class + role
    // =================================================================

    // Fetch the current embryo roster from /api/embryos/positions.
    // Returns [] on failure or when no embryos have positions yet.
    async _loadRoster() {
        try {
            const resp = await fetch('/api/embryos/positions', { cache: 'no-store' });
            if (!resp.ok) return [];
            const data = await resp.json();
            return Array.isArray(data.embryos) ? data.embryos : [];
        } catch (e) {
            console.warn('[ExperimentOverview] roster fetch error:', e);
            return [];
        }
    },

    // Fetch the roles registry from /api/roles.
    // Returns a Map(name → {ui_color, ui_icon, role_class, default_cadence_seconds}).
    // Returns empty Map on failure.
    async _loadRolesMap() {
        try {
            const resp = await fetch('/api/roles', { cache: 'no-store' });
            if (!resp.ok) return new Map();
            const data = await resp.json();
            const map = new Map();
            if (Array.isArray(data.roles)) {
                for (const r of data.roles) {
                    map.set(r.name, r);
                }
            }
            return map;
        } catch (e) {
            console.warn('[ExperimentOverview] roles fetch error:', e);
            return new Map();
        }
    },

    // Convert a hex color (#rrggbb) to rgba(r,g,b,alpha) for inline styles.
    _hexToRgba(hex, alpha) {
        const h = (hex || '#888888').replace('#', '');
        const r = parseInt(h.slice(0, 2), 16) || 0;
        const g = parseInt(h.slice(2, 4), 16) || 0;
        const b = parseInt(h.slice(4, 6), 16) || 0;
        return `rgba(${r},${g},${b},${alpha})`;
    },

    // Cross-reference: find the name of the active (or most recently done) tactic
    // that covers a given embryo (by embryo_id + role).
    // Scope resolution: global → covers all; role → covers matching role;
    // embryos → covers listed ids. Returns '' when no tactic found.
    // NOTE: mirrors resolve_scope_embryos() in gently/app/orchestration/role_scope.py — keep in sync if a new scope mode is added.
    _resolveCurrentTactic(embryoId, role, plan) {
        if (!plan || !Array.isArray(plan.tactics)) return '';
        const covers = (t) => {
            const scope = t.scope || { mode: 'global' };
            if (scope.mode === 'global') return true;
            if (scope.mode === 'role') return scope.role === role;
            if (scope.mode === 'embryos') {
                return Array.isArray(scope.embryo_ids) && scope.embryo_ids.includes(embryoId);
            }
            return false;
        };
        // Prefer the active tactic covering this embryo.
        const active = plan.tactics.find(t => t.state === 'active' && covers(t));
        if (active) return active.name;
        // Fall back to the most recently done tactic (last in array order).
        for (let i = plan.tactics.length - 1; i >= 0; i--) {
            if (plan.tactics[i].state === 'done' && covers(plan.tactics[i])) {
                return plan.tactics[i].name;
            }
        }
        return '';
    },

    // Render a scope badge for a tactic node in the spine.
    // Colors for role-scoped badges come from the roles registry (API), not CSS.
    // Global and explicit-embryos scopes use the static `.ops-scope-global` class.
    _renderOpsScopeBadge(scope, embryos, rolesMap, ESC) {
        if (!scope || scope.mode === 'global') {
            return '<span class="ops-scope-badge ops-scope-global">→ all embryos</span>';
        }
        if (scope.mode === 'role') {
            const role = scope.role || '';
            const roleInfo = rolesMap ? rolesMap.get(role) : null;
            const color = (roleInfo && roleInfo.ui_color) || '#8b949e';
            const matchIds = embryos
                .filter(e => e.role === role)
                .map(e => e.embryo_id)
                .join(', ');
            const label = matchIds
                ? `→ ${ESC(role)} · ${ESC(matchIds)}`
                : `→ ${ESC(role)}`;
            const bg = this._hexToRgba(color, 0.12);
            const border = this._hexToRgba(color, 0.35);
            return `<span class="ops-scope-badge" style="color:${ESC(color)};background:${bg};border-color:${border}">${label}</span>`;
        }
        if (scope.mode === 'embryos') {
            const ids = ESC((scope.embryo_ids || []).join(', '));
            return `<span class="ops-scope-badge ops-scope-global">→ ${ids}</span>`;
        }
        return '';
    },

    // Render the D2 roster lens: embryos grouped by role class then by role.
    // SUBJECTS section is foregrounded; REFERENCES section is compact/muted.
    // Role colors/icons come from the rolesMap (API data), not from CSS constants.
    // Returns empty string when embryos array is empty (backward compat).
    _renderRosterLens(embryos, rolesMap, plan, ESC) {
        if (!embryos || embryos.length === 0) return '';

        // Group embryos by role_class then by role, preserving first-seen order.
        const CLASS_ORDER_PREF = ['subject', 'reference'];
        const classOrder = [];
        const byClass = {};

        for (const emb of embryos) {
            const roleInfo = rolesMap.get(emb.role);
            const cls = (roleInfo && roleInfo.role_class) || 'subject';
            if (!byClass[cls]) {
                byClass[cls] = { roleOrder: [], byRole: {} };
                classOrder.push(cls);
            }
            const section = byClass[cls];
            if (!section.byRole[emb.role]) {
                section.byRole[emb.role] = [];
                section.roleOrder.push(emb.role);
            }
            section.byRole[emb.role].push(emb);
        }

        // Canonical class order: subjects first.
        classOrder.sort((a, b) => {
            const ai = CLASS_ORDER_PREF.indexOf(a);
            const bi = CLASS_ORDER_PREF.indexOf(b);
            return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
        });

        const totalCount = embryos.length;
        const totalRoles = classOrder.reduce((n, cls) => n + byClass[cls].roleOrder.length, 0);

        const classSections = classOrder.map(cls => {
            const { roleOrder, byRole } = byClass[cls];
            const isSubject = cls === 'subject';

            const classHeaderInner = isSubject
                ? `<span class="ops-class-live-dot"></span><span class="ops-class-label">Subjects</span><span class="ops-class-desc">— adaptive tactics / scenarios</span>`
                : `<span class="ops-class-label">References</span><span class="ops-class-desc">— steady acquisition</span>`;

            const roleGroups = roleOrder.map(role => {
                const roleEmbyros = byRole[role];
                const roleInfo = rolesMap.get(role);
                const uiColor = (roleInfo && roleInfo.ui_color) || '#8b949e';
                const uiIcon  = (roleInfo && roleInfo.ui_icon)  || '';
                const bgRgba  = this._hexToRgba(uiColor, 0.08);
                const ids = roleEmbyros.map(e => e.embryo_id).join(', ');

                const embryoRows = roleEmbyros.map(emb => {
                    const cadencePhase = emb.cadence_phase || 'normal';
                    const strain = emb.strain || '—';
                    const label  = emb.user_label || emb.embryo_id;
                    const tacticName = this._resolveCurrentTactic(emb.embryo_id, emb.role, plan);
                    const stateStr = emb.is_complete ? 'done'
                                   : cadencePhase === 'paused' ? 'paused'
                                   : 'active';
                    const compact = isSubject ? '' : ' compact';
                    const chipBg     = this._hexToRgba(uiColor, 0.15);
                    const chipBorder = this._hexToRgba(uiColor, 0.4);
                    return `<div class="ops-roster-embryo${compact}">
                        <span class="ops-rem-id">${ESC(label)}</span>
                        <span class="ops-rem-role-chip" style="background:${chipBg};color:${ESC(uiColor)};border-color:${chipBorder}">${uiIcon ? ESC(uiIcon) + ' ' : ''}${ESC(role)}</span>
                        <span class="ops-rem-strain">${ESC(strain)}</span>
                        <span class="ops-rem-phase ${ESC(cadencePhase)}">${ESC(cadencePhase)}</span>
                        <span class="ops-rem-tactic">${ESC(tacticName || '—')}</span>
                        <span class="ops-rem-state ${ESC(stateStr)}">${ESC(stateStr)}</span>
                    </div>`;
                }).join('');

                return `<div class="ops-role-group">
                    <div class="ops-role-header" style="border-left-color:${ESC(uiColor)};background:${bgRgba}">
                        <span class="ops-role-name" style="color:${ESC(uiColor)}">${ESC(role.toUpperCase())}</span>
                        <span class="ops-role-sep">·</span>
                        <span class="ops-role-count">${roleEmbyros.length} embryo${roleEmbyros.length !== 1 ? 's' : ''}</span>
                        <span class="ops-role-ids">${ESC(ids)}</span>
                    </div>
                    ${embryoRows}
                </div>`;
            }).join('');

            return `<div class="ops-class-section">
                <div class="ops-class-header ${ESC(cls)}">${classHeaderInner}</div>
                ${roleGroups}
            </div>`;
        }).join('');

        return `<div class="ops-roster">
            <div class="ops-roster-head">
                <span class="ops-roster-title">Population roster</span>
                <span class="ops-roster-count">${totalCount} embryo${totalCount !== 1 ? 's' : ''} · ${totalRoles} role${totalRoles !== 1 ? 's' : ''}</span>
            </div>
            ${classSections}
        </div>`;
    },

    // -----------------------------------------------------------------
    // Rules table — full subtab view, grouped by rule kind
    // -----------------------------------------------------------------
    _renderRulesTable(s) {
        const wrap = el('div', 'expov-rules-table');

        // Section title strip
        const title = el('div', 'expov-rules-title-row');
        title.appendChild(elText('h3', 'expov-rules-title', 'Reactive Rules'));
        title.appendChild(elText('span', 'expov-rules-count', `${s.triggers.length} active`));
        wrap.appendChild(title);

        // Group triggers by kind for readability
        const groups = {
            interval_rule: { label: 'Cadence rules', icon: '⏱', items: [] },
            power_rule:    { label: 'Laser power rules', icon: '☼', items: [] },
            burst:         { label: 'Burst rules', icon: '⚡', items: [] },
        };
        s.triggers.forEach(t => {
            const grp = groups[t.kind] || (groups[t.kind] = { label: t.kind, icon: '◇', items: [] });
            grp.items.push(t);
        });

        Object.values(groups).forEach(grp => {
            if (grp.items.length === 0) return;
            const section = el('div', 'expov-rules-section');
            const head = el('div', 'expov-rules-section-head');
            head.appendChild(elText('span', 'expov-rules-section-icon', grp.icon));
            head.appendChild(elText('span', 'expov-rules-section-label', grp.label));
            head.appendChild(elText('span', 'expov-rules-section-count', `${grp.items.length}`));
            section.appendChild(head);

            grp.items.forEach(t => {
                const row = el('div', 'expov-rule-row');
                // Column 1: trigger label
                const labelCol = el('div', 'expov-rule-col expov-rule-label');
                labelCol.appendChild(elClass('span', 'expov-trigger-diamond-inline'));
                labelCol.appendChild(elText('span', '', t.label));
                row.appendChild(labelCol);
                // Column 2: when
                const whenCol = el('div', 'expov-rule-col expov-rule-when');
                whenCol.appendChild(elText('span', 'expov-rule-col-lbl', 'WHEN'));
                whenCol.appendChild(elText('span', 'expov-rule-col-val', t.when_text));
                row.appendChild(whenCol);
                // Column 3: then
                const thenCol = el('div', 'expov-rule-col expov-rule-then');
                thenCol.appendChild(elText('span', 'expov-rule-col-lbl', 'THEN'));
                thenCol.appendChild(elText('span', 'expov-rule-col-val', t.then_text));
                row.appendChild(thenCol);
                // Column 4: scope + lifecycle
                const scopeCol = el('div', 'expov-rule-col expov-rule-scope');
                t.applies_to.forEach(role => {
                    scopeCol.appendChild(elText('span', 'expov-rule-scope-chip', role));
                });
                if (t.one_time) {
                    scopeCol.appendChild(elText('span', 'expov-rule-lifecycle one-time', 'one-time'));
                } else {
                    scopeCol.appendChild(elText('span', 'expov-rule-lifecycle persistent', 'persistent'));
                }
                row.appendChild(scopeCol);
                section.appendChild(row);
            });
            wrap.appendChild(section);
        });

        return wrap;
    }
};

// -----------------------------------------------------------------
// DOM helpers (tiny — avoid pulling in a framework just for SVG)
// -----------------------------------------------------------------
function el(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
}
function elText(tag, cls, text) {
    const e = el(tag, cls);
    e.textContent = text;
    return e;
}
function elHtml(tag, cls, html) {
    const e = el(tag, cls);
    e.innerHTML = html;
    return e;
}
function elClass(tag, cls) { return el(tag, cls); }
function svgEl(tag, attrs, text) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) {
        for (const k of Object.keys(attrs)) {
            e.setAttribute(k, attrs[k]);
        }
    }
    if (text !== undefined) e.textContent = text;
    return e;
}

// -----------------------------------------------------------------
// Self-bootstrap: wire up tab click + initial render fallback.
// This works even if app.js was cached and doesn't know about the
// Experiment tab lazy-init.
// -----------------------------------------------------------------
(function autoBootstrap() {
    function setup() {
        const tab = document.querySelector('.tab[data-tab="experiment"]');
        if (tab) {
            tab.addEventListener('click', () => {
                ExperimentOverview.init();
            });
        }
        // View-switcher buttons (Overview / Rules)
        document.querySelectorAll('[data-experiment-view]').forEach(btn => {
            btn.addEventListener('click', () => {
                ExperimentOverview.setView(btn.dataset.experimentView);
            });
        });
        // If tab is already active on page load (e.g. via /#experiment hash),
        // render immediately.
        const content = document.getElementById('experiment-content');
        if (content && content.classList.contains('active')) {
            ExperimentOverview.init();
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup);
    } else {
        setup();
    }
})();
