/**
 * Experiment Overview Tab — vector-graphics view of the planned timelapse.
 *
 * MOCKUP: data is stubbed below in STUB_STRATEGY. The real version will fetch
 * GET /api/experiments/{session_id}/strategy and pass the returned JSON
 * directly to ExperimentOverview.render(strategy).
 */

const STUB_STRATEGY = {
    session_id: "20260522_1430_dopaminergic_demo_a3f8e1c2",
    session_name: "dopaminergic-reporter demo",
    started_at: "2026-05-22T14:30:00",
    now_offset_s: 8100,         // 2h 15min into the run
    horizon_s: 14400,           // 4h total view window (past + projected)
    base_interval_s: 120,
    dose_budget_base_ms: 50000,
    per_timepoint_ms: 500,      // 50 slices × 10ms
    monitoring_modes: [
        {
            name: "expression_monitoring",
            description: "Anticipating fluorescent-reporter onset on Test embryos: accelerate to 60s on signal, ramp 488 down on saturation.",
            applies_to_roles: ["test"],
            params: {
                fast_interval: 60,
                rampdown_step_pct: 1.0,
                rampdown_floor_pct: 2.0,
                rampdown_ceiling_pct: 6.0
            }
        },
        {
            name: "pre_terminal_monitoring",
            description: "Anticipating organism pre-terminal stage (pretzel): accelerate to 30s on detection.",
            applies_to_roles: ["test"],
            params: { fast_interval: 30 }
        }
    ],
    triggers: [
        { id: "t1", kind: "interval_rule", label: "signal onset",
          when_text: "dopaminergic ≥ WEAK", then_text: "120s → 60s",
          applies_to: ["test"], one_time: true },
        { id: "t2", kind: "power_rule", label: "488 ramp down",
          when_text: "intensity = SATURATING (×3)", then_text: "488 ↓ 1%/step, floor 2%",
          applies_to: ["test"] },
        { id: "t3", kind: "burst", label: "structure-triggered burst",
          when_text: "structure_quality = GOOD", then_text: "burst 200 frames @ 20 Hz",
          applies_to: ["test"] },
        { id: "t4", kind: "interval_rule", label: "pre-terminal speedup",
          when_text: "stage = pretzel", then_text: "60s → 30s",
          applies_to: ["test"], one_time: true }
    ],
    embryos: [
        {
            id: "E1", role: "test", color: "#ff66cc", icon: "★",
            dose_used_ms: 12500, dose_budget_ms: 50000,
            tp_acquired: 25,
            stop_condition: "hatching+3 OR 24h duration",
            stop_kind: "bounded",
            laser_488_pct_now: 3.0,
            phases: [
                { mode: "base",     start: 0,    end: 1800, cadence_s: 120 },
                { mode: "fast",     start: 1800, end: 3600, cadence_s: 60 },
                { mode: "burst",    start: 3600, end: 3610, frames: 200, hz: 20 },
                { mode: "cooldown", start: 3610, end: 3640, cadence_s: 60 },
                { mode: "fast",     start: 3640, end: 8100, cadence_s: 60 }
            ],
            trigger_events: [
                { trigger_id: "t1", at: 1800 },
                { trigger_id: "t3", at: 3600 },
                { trigger_id: "t2", at: 5400, count: 3 }
            ],
            power_history_488: [
                { at: 0,    pct: 5.0 },
                { at: 5400, pct: 4.0 },
                { at: 5460, pct: 3.0 },
                { at: 8100, pct: 3.0 }
            ],
            // Future projection at current cadence (60s, fast). Hatching not
            // deterministic so projected_end_s is null — render fades to ∞.
            projected_cadence_s: 60,
            projected_end_s: null
        },
        {
            id: "E2", role: "test", color: "#ff66cc", icon: "★",
            dose_used_ms: 6500, dose_budget_ms: 50000,
            tp_acquired: 13,
            stop_condition: "hatching+3 OR 24h duration",
            stop_kind: "bounded",
            laser_488_pct_now: 5.0,
            phases: [
                { mode: "base", start: 0, end: 8100, cadence_s: 120 }
            ],
            trigger_events: [],
            power_history_488: [
                { at: 0,    pct: 5.0 },
                { at: 8100, pct: 5.0 }
            ],
            projected_cadence_s: 120,
            projected_end_s: null
        },
        {
            id: "E3", role: "test", color: "#ff66cc", icon: "★",
            dose_used_ms: 38000, dose_budget_ms: 50000,
            tp_acquired: 76,
            stop_condition: "manual",
            stop_kind: "open_ended",
            laser_488_pct_now: 5.0,
            phases: [
                { mode: "base", start: 0, end: 8100, cadence_s: 120 }
            ],
            trigger_events: [],
            power_history_488: [
                { at: 0,    pct: 5.0 },
                { at: 8100, pct: 5.0 }
            ],
            // Projected dose-exhaust horizon = 4.0h from now (warning condition)
            projected_cadence_s: 120,
            projected_end_s: null,
            dose_exhaust_at_s: 12000   // budget will run out at this elapsed time
        },
        {
            id: "C1", role: "calibration", color: "#22d3ee", icon: "◆",
            dose_used_ms: 33500, dose_budget_ms: 500000,   // 10× multiplier
            tp_acquired: 67,
            stop_condition: "manual",
            stop_kind: "open_ended",
            laser_488_pct_now: 5.0,
            phases: [
                { mode: "base", start: 0, end: 8100, cadence_s: 120 }
            ],
            trigger_events: [],
            power_history_488: [
                { at: 0,    pct: 5.0 },
                { at: 8100, pct: 5.0 }
            ],
            projected_cadence_s: 120,
            projected_end_s: null
        }
    ]
};

const ExperimentOverview = {
    initialized: false,
    expandedMode: null,
    activeView: 'overview',  // 'overview' | 'rules'

    init() {
        console.log('[ExperimentOverview] init() called, view=', this.activeView);
        this.render(STUB_STRATEGY);
        this.initialized = true;
    },

    setView(view) {
        if (view === this.activeView) return;
        this.activeView = view;
        // Update view-switcher button state
        document.querySelectorAll('[data-experiment-view]').forEach(b => {
            b.classList.toggle('active', b.dataset.experimentView === view);
        });
        this.render(STUB_STRATEGY);
    },

    render(s) {
        const root = document.getElementById('experiment-overview-root');
        if (!root) {
            console.error('[ExperimentOverview] #experiment-overview-root NOT FOUND in DOM');
            return;
        }
        try {
            root.innerHTML = '';
            if (this.activeView === 'rules') {
                this._renderRulesView(root, s);
            } else {
                this._renderOverviewView(root, s);
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

    _renderOverviewView(root, s) {
        root.appendChild(this._renderHeader(s));
        root.appendChild(this._renderModes(s));
        root.appendChild(this._renderModeExpanded(s));
        root.appendChild(this._renderSwimlanes(s));
    },

    _renderRulesView(root, s) {
        // Compact header echoing the session identity
        const header = el('div', 'expov-header');
        const metaRow = el('div', 'expov-header-row expov-header-row-meta');
        metaRow.appendChild(elText('span', 'expov-session-name', s.session_name));
        metaRow.appendChild(elText('span', 'expov-session-id', s.session_id));
        metaRow.appendChild(elText('span', 'expov-mockup-badge', 'mockup · stubbed data'));
        header.appendChild(metaRow);
        root.appendChild(header);

        // Active monitoring modes (context for the rules)
        root.appendChild(this._renderModes(s));
        root.appendChild(this._renderModeExpanded(s));

        // The rules table
        root.appendChild(this._renderRulesTable(s));
    },

    // -----------------------------------------------------------------
    // Header — session identification + key metrics strip
    // (page-level title lives in .experiment-header-bar above)
    // -----------------------------------------------------------------
    _renderHeader(s) {
        const elapsedH = Math.floor(s.now_offset_s / 3600);
        const elapsedM = Math.floor((s.now_offset_s % 3600) / 60);
        const wrap = el('div', 'expov-header');

        // Session identification line
        const metaRow = el('div', 'expov-header-row expov-header-row-meta');
        metaRow.appendChild(elText('span', 'expov-session-name', s.session_name));
        metaRow.appendChild(elText('span', 'expov-session-id', s.session_id));
        metaRow.appendChild(elText('span', 'expov-mockup-badge', 'mockup · stubbed data'));
        wrap.appendChild(metaRow);

        // Compact key-metric strip
        const roleCounts = {};
        s.embryos.forEach(e => { roleCounts[e.role] = (roleCounts[e.role] || 0) + 1; });
        const roleStr = Object.entries(roleCounts).map(([r, n]) => `${n} ${r}`).join(' · ');
        const metricsRow = el('div', 'expov-header-row expov-header-row-metrics');
        const metric = (label, val) => {
            const m = el('span', 'expov-metric');
            m.appendChild(elText('span', 'expov-metric-val', val));
            m.appendChild(elText('span', 'expov-metric-lbl', label));
            return m;
        };
        metricsRow.appendChild(metric('elapsed', `${elapsedH}h ${elapsedM}m`));
        metricsRow.appendChild(metric('base', `${s.base_interval_s}s`));
        metricsRow.appendChild(metric('budget', `${(s.dose_budget_base_ms/1000).toFixed(0)}s × role`));
        metricsRow.appendChild(metric('embryos', `${s.embryos.length} · ${roleStr}`));
        wrap.appendChild(metricsRow);

        return wrap;
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

    // -----------------------------------------------------------------
    // Legend
    // -----------------------------------------------------------------
    _renderLegend() {
        const wrap = el('div', 'expov-legend');
        const items = [
            ['base',     'base cadence'],
            ['fast',     'fast cadence'],
            ['burst',    'burst window'],
            ['cooldown', 'cooldown'],
            ['paused',   'paused'],
        ];
        items.forEach(([cls, label]) => {
            const item = el('span', 'expov-legend-item');
            item.appendChild(elClass('span', `expov-legend-swatch ${cls}`));
            item.appendChild(elText('span', '', label));
            wrap.appendChild(item);
        });
        const projItem = el('span', 'expov-legend-item');
        projItem.appendChild(elClass('span', 'expov-legend-swatch projected'));
        projItem.appendChild(elText('span', '', 'projected'));
        wrap.appendChild(projItem);

        const glyphs = [
            ['◇', 'trigger fired'],
            ['●', 'now'],
            ['■', 'stop condition'],
            ['∞', 'open-ended'],
            ['▲', 'burst start'],
            ['⚠', 'budget warning']
        ];
        glyphs.forEach(([g, label]) => {
            const item = el('span', 'expov-legend-item');
            item.appendChild(elText('span', 'expov-legend-glyph', g));
            item.appendChild(elText('span', '', label));
            wrap.appendChild(item);
        });
        return wrap;
    },

    // -----------------------------------------------------------------
    // Swimlanes SVG — the main visualization
    // -----------------------------------------------------------------
    _renderSwimlanes(s) {
        const wrap = el('div', 'expov-swimlanes-wrap');

        // Compact inline legend above the SVG
        const legend = el('div', 'expov-mini-legend');
        const swatches = [
            ['base', 'base'],
            ['fast', 'fast'],
            ['burst', 'burst'],
            ['cooldown', 'cooldown']
        ];
        swatches.forEach(([k, label]) => {
            const item = el('span', 'expov-mini-legend-item');
            const sw = el('span', `expov-mini-legend-swatch ${k}`);
            item.appendChild(sw);
            item.appendChild(elText('span', '', label));
            legend.appendChild(item);
        });
        const projItem = el('span', 'expov-mini-legend-item');
        projItem.appendChild(elClass('span', 'expov-mini-legend-swatch projected'));
        projItem.appendChild(elText('span', '', 'projected'));
        legend.appendChild(projItem);
        wrap.appendChild(legend);

        // Layout constants (logical pixels in the SVG viewBox)
        const LEFT  = 180;            // label gutter
        const RIGHT = 80;             // right gutter for stop icon + ∞
        const LANE_W = 900;           // lane drawing area
        const W = LEFT + LANE_W + RIGHT;

        const ROW_H        = 100;     // per-embryo row total height
        const LANE_H       = 28;      // cadence lane height
        const POWER_H      = 22;      // power strip height
        const DOSE_H       = 12;      // dose gauge height
        const ROW_PAD      = 14;      // top padding inside row
        const TOP_AXIS_H   = 36;      // top axis area (time labels + wall-clock)
        const BOTTOM_PAD   = 8;

        const rows = s.embryos.length;
        const H = TOP_AXIS_H + rows * ROW_H + BOTTOM_PAD;

        const svg = svgEl('svg', {
            class: 'expov-swimlanes-svg',
            viewBox: `0 0 ${W} ${H}`,
            preserveAspectRatio: 'xMinYMin meet'
        });

        // Time scale helpers
        const xForT = (t) => LEFT + (t / s.horizon_s) * LANE_W;
        const nowX = xForT(s.now_offset_s);

        // ----- top axis: hour ticks with wall-clock annotation
        const startedAt = new Date(s.started_at);
        const axisG = svgEl('g');
        for (let h = 0; h <= Math.ceil(s.horizon_s / 3600); h++) {
            const x = xForT(h * 3600);
            axisG.appendChild(svgEl('line', {
                x1: x, x2: x, y1: TOP_AXIS_H - 6, y2: H - BOTTOM_PAD,
                class: 'expov-svg-axis', 'stroke-opacity': h === 0 ? 0.55 : 0.12
            }));
            axisG.appendChild(svgEl('text', {
                x: x + 4, y: 12, class: 'expov-svg-axis-label'
            }, `+${h}h`));
            // Wall-clock subtitle
            const wallClock = new Date(startedAt.getTime() + h * 3600 * 1000);
            const hh = String(wallClock.getHours()).padStart(2, '0');
            const mm = String(wallClock.getMinutes()).padStart(2, '0');
            axisG.appendChild(svgEl('text', {
                x: x + 4, y: 22,
                class: 'expov-svg-axis-wallclock'
            }, `${hh}:${mm}`));
        }
        svg.appendChild(axisG);

        // ----- "now" line (spans all rows)
        const nowLine = svgEl('line', {
            x1: nowX, x2: nowX, y1: TOP_AXIS_H - 4, y2: H - BOTTOM_PAD,
            class: 'expov-svg-now-line'
        });
        svg.appendChild(nowLine);
        svg.appendChild(svgEl('text', {
            x: nowX + 4, y: 26, class: 'expov-svg-now-label'
        }, 'now'));

        // ----- one group per embryo
        s.embryos.forEach((emb, i) => {
            const rowTop = TOP_AXIS_H + i * ROW_H;
            const rowG = svgEl('g');
            rowG.appendChild(this._renderLaneRow(s, emb, {
                LEFT, LANE_W, RIGHT, W, ROW_H, LANE_H, POWER_H, DOSE_H, ROW_PAD,
                TOP_AXIS_H, rowTop, xForT, nowX
            }));
            svg.appendChild(rowG);
        });

        wrap.appendChild(svg);
        return wrap;
    },

    _renderLaneRow(s, emb, dim) {
        const { LEFT, LANE_W, W, ROW_H, LANE_H, POWER_H, DOSE_H, ROW_PAD,
                rowTop, xForT, nowX } = dim;
        const g = svgEl('g');

        // Hairline divider above each row (except first)
        if (rowTop > dim.TOP_AXIS_H) {
            g.appendChild(svgEl('line', {
                x1: 8, x2: W - 8, y1: rowTop, y2: rowTop,
                class: 'expov-svg-row-divider'
            }));
        }

        // ---- Left label gutter ---------------------------------------
        // Single header line + phase pill. Power/dose labels are at the
        // y-position of their respective sub-rows, right-aligned in the gutter.
        const labelY = rowTop + ROW_PAD + 12;

        // Header line: icon · ID · role (· 10× hint for calibration)
        g.appendChild(svgEl('text', {
            x: 14, y: labelY + 1, class: 'expov-svg-role-icon',
            fill: emb.color
        }, emb.icon));
        g.appendChild(svgEl('text', {
            x: 32, y: labelY, class: 'expov-svg-label'
        }, emb.id));
        // role tag (kept short — "10×" annotation moves to dose label)
        g.appendChild(svgEl('text', {
            x: 60, y: labelY, class: 'expov-svg-role-tag'
        }, emb.role));

        // Phase pill: current mode at glance
        const currentPhase = emb.phases[emb.phases.length - 1];
        const pillY = labelY + 7;
        const pillH = 14;
        const phaseLabel = currentPhase.mode === 'burst'
            ? `BURST · ${currentPhase.hz}Hz`
            : `${currentPhase.mode.toUpperCase()} · ${currentPhase.cadence_s}s`;
        const pillW = Math.max(70, phaseLabel.length * 6 + 12);
        const pillX = 32;
        const phaseColors = {
            base:     '#6b7280',
            fast:     '#fb923c',
            burst:    '#ef4444',
            cooldown: '#a78bfa',
            paused:   '#3b82f6'
        };
        const pillFill = phaseColors[currentPhase.mode] || '#6b7280';
        g.appendChild(svgEl('rect', {
            x: pillX, y: pillY, width: pillW, height: pillH, rx: 7,
            fill: pillFill, 'fill-opacity': 0.22,
            stroke: pillFill, 'stroke-opacity': 0.65, 'stroke-width': 1
        }));
        g.appendChild(svgEl('text', {
            x: pillX + pillW / 2, y: pillY + 10,
            'text-anchor': 'middle',
            fill: pillFill, 'font-size': 9.5, 'font-weight': 700,
            'font-family': "'JetBrains Mono', monospace"
        }, phaseLabel));

        // Tiny tp annotation under the pill (no stop — that's at lane right edge)
        g.appendChild(svgEl('text', {
            x: 32, y: pillY + pillH + 12,
            class: 'expov-svg-sublabel'
        }, `${emb.tp_acquired} tp acquired`));

        // ---- Cadence lane --------------------------------------------
        const laneY = rowTop + ROW_PAD;
        const laneMid = laneY + LANE_H / 2;
        const laneBottom = laneY + LANE_H;

        // Phases — solid colored rects, no ticks. Cadence is read from the
        // phase pill in the gutter and the optional inline cadence label.
        // Min 4px visual width so micro-phases (burst, cooldown) stay visible.
        emb.phases.forEach(ph => {
            const x0 = xForT(ph.start);
            const x1Raw = xForT(ph.end);
            const x1 = Math.max(x1Raw, x0 + 4);
            const width = x1 - x0;
            const cls = `expov-svg-phase-${ph.mode}`;
            g.appendChild(svgEl('rect', {
                x: x0, y: laneY, width, height: LANE_H, rx: 2,
                class: cls
            }));
            // Inline cadence label inside the rect, when there's room.
            // Reads "120s" / "60s" / "30s · cool" — small, centered, low-contrast.
            if (ph.mode !== 'burst' && ph.cadence_s && width >= 42) {
                const label = ph.mode === 'cooldown'
                    ? `${ph.cadence_s}s · cool`
                    : `${ph.cadence_s}s`;
                g.appendChild(svgEl('text', {
                    x: x0 + width / 2, y: laneMid + 3.5,
                    'text-anchor': 'middle',
                    class: 'expov-svg-phase-label'
                }, label));
            }
            // Burst: keep the bright block + balloon since it's the most
            // attention-worthy event in the lane
            if (ph.mode === 'burst') {
                const bx = (xForT(ph.start) + xForT(ph.end)) / 2;
                const balloonY = laneY - 12;
                const halfW = 30;
                g.appendChild(svgEl('rect', {
                    x: bx - halfW, y: balloonY - 10, width: halfW * 2, height: 12, rx: 3,
                    class: 'expov-svg-burst-balloon-bg'
                }));
                g.appendChild(svgEl('text', {
                    x: bx, y: balloonY - 1,
                    'text-anchor': 'middle',
                    class: 'expov-svg-burst-label'
                }, `${ph.frames}f · ${ph.hz}Hz`));
                g.appendChild(svgEl('line', {
                    x1: bx, x2: bx, y1: balloonY + 2, y2: laneY,
                    stroke: '#ef4444', 'stroke-width': 1, 'stroke-opacity': 0.7
                }));
            }
        });

        // Projected future segment (dashed) past 'now' to a horizon.
        // Color is grey for indefinite, amber when ending at dose exhaust.
        const projStartT = s.now_offset_s;
        let projEndT = s.horizon_s;
        let projEndsAtBudget = false;
        if (emb.projected_end_s) projEndT = Math.min(projEndT, emb.projected_end_s);
        if (emb.dose_exhaust_at_s && emb.dose_exhaust_at_s < projEndT) {
            projEndT = emb.dose_exhaust_at_s;
            projEndsAtBudget = true;
        }
        const xProjStart = xForT(projStartT);
        const xProjEnd = xForT(projEndT);
        if (xProjEnd > xProjStart) {
            g.appendChild(svgEl('line', {
                x1: xProjStart, y1: laneMid, x2: xProjEnd, y2: laneMid,
                class: projEndsAtBudget
                    ? 'expov-svg-projected-bar warn'
                    : 'expov-svg-projected-bar'
            }));
        }

        // Trigger diamonds — placed in the upper half of the lane to avoid
        // colliding with the burst balloon above
        (emb.trigger_events || []).forEach(te => {
            const x = xForT(te.at);
            const dy = laneY + 6;
            const size = 4;
            const trig = s.triggers.find(t => t.id === te.trigger_id);
            const label = trig ? trig.label : te.trigger_id;
            const dia = svgEl('polygon', {
                points: `${x},${dy-size} ${x+size},${dy} ${x},${dy+size} ${x-size},${dy}`,
                class: 'expov-svg-trigger-diamond expov-svg-tooltip-target'
            });
            const tooltip = svgEl('title');
            tooltip.textContent = `${label}\n${trig?.when_text || ''} → ${trig?.then_text || ''}` +
                                  (te.count ? ` (×${te.count})` : '');
            dia.appendChild(tooltip);
            g.appendChild(dia);
        });

        // Dose-exhaust warning: ⚠ + time-to-exhaust positioned ABOVE the lane
        // so it doesn't overlap with the dashed projected bar
        if (emb.dose_exhaust_at_s && emb.dose_exhaust_at_s < s.horizon_s) {
            const exhX = xForT(emb.dose_exhaust_at_s);
            const remain = emb.dose_exhaust_at_s - s.now_offset_s;
            const rh = Math.floor(remain / 3600);
            const rm = Math.floor((remain % 3600) / 60);
            const exhText = `⚠ budget exhausts in ${rh > 0 ? rh + 'h ' : ''}${rm}m`;
            g.appendChild(svgEl('text', {
                x: exhX - 4, y: laneY - 5,
                'text-anchor': 'end',
                fill: 'var(--accent-orange)',
                'font-size': 10,
                'font-weight': 600,
                'font-family': "'JetBrains Mono', monospace"
            }, exhText));
            // Small dotted vertical marker so the user can see WHERE on the lane
            g.appendChild(svgEl('line', {
                x1: exhX, x2: exhX, y1: laneY, y2: laneY + LANE_H,
                stroke: 'var(--accent-orange)',
                'stroke-width': 1.5,
                'stroke-dasharray': '2 2',
                opacity: 0.7
            }));
        }
        // Open-ended ∞ glyph at right edge
        if (emb.stop_kind === 'open_ended') {
            g.appendChild(svgEl('text', {
                x: LEFT + LANE_W + 8, y: laneMid + 5,
                class: 'expov-svg-infinity'
            }, '∞'));
        } else {
            g.appendChild(svgEl('text', {
                x: xForT(projEndT) + 6, y: laneMid + 4,
                class: 'expov-svg-stop-icon expov-svg-stop-hatch'
            }, '■'));
        }

        // ---- Power strip ---------------------------------------------
        // Visual encoding makes "steady" vs "ramping" obvious:
        //   • Steady segments  = thin grey horizontal line
        //   • Ramping segments = bright cyan line + filled area + dot at each step
        //   • Step transitions get a small arrow (↓ or ↑) and a delta tag
        const powerY = laneY + LANE_H + 6;
        const powerH = POWER_H;
        const powerYBase = powerY + powerH;
        g.appendChild(svgEl('line', {
            x1: LEFT, x2: LEFT + LANE_W, y1: powerYBase, y2: powerYBase,
            class: 'expov-svg-power-baseline'
        }));
        const yForPct = (pct) => powerYBase - (pct / 10) * powerH;

        const hist = emb.power_history_488 || [];
        if (hist.length > 1) {
            // Detect ramp clusters: consecutive change-steps with x-spacing <
            // CLUSTER_PX get grouped, annotated once at the cluster end.
            const CLUSTER_PX = 20;
            const stepEvents = [];   // each: {fromIdx, toIdx, isRamp}
            for (let k = 0; k < hist.length - 1; k++) {
                if (hist[k].pct !== hist[k+1].pct) {
                    stepEvents.push({ fromIdx: k, toIdx: k+1 });
                }
            }
            // Group consecutive close steps into clusters
            const clusters = [];
            stepEvents.forEach(step => {
                const last = clusters[clusters.length - 1];
                const stepX = xForT(hist[step.toIdx].at);
                if (last) {
                    const lastX = xForT(hist[last[last.length-1].toIdx].at);
                    if (Math.abs(stepX - lastX) < CLUSTER_PX) {
                        last.push(step);
                        return;
                    }
                }
                clusters.push([step]);
            });

            // Draw horizontal "steady" segments + vertical "step" lines for all
            // adjacent (hist[k], hist[k+1]) pairs.
            for (let k = 0; k < hist.length - 1; k++) {
                const x0 = xForT(hist[k].at);
                const x1 = xForT(hist[k+1].at);
                const y  = yForPct(hist[k].pct);
                const yNext = yForPct(hist[k+1].pct);
                g.appendChild(svgEl('line', {
                    x1: x0, x2: x1, y1: y, y2: y,
                    class: 'expov-svg-power-steady'
                }));
                if (hist[k].pct !== hist[k+1].pct) {
                    g.appendChild(svgEl('line', {
                        x1: x1, x2: x1, y1: y, y2: yNext,
                        class: 'expov-svg-power-step'
                    }));
                    g.appendChild(svgEl('circle', {
                        cx: x1, cy: yNext, r: 2.2,
                        class: 'expov-svg-power-stepdot'
                    }));
                }
            }
            // Final tail to lane right edge
            const last = hist[hist.length - 1];
            const lastX = xForT(last.at);
            const lastY = yForPct(last.pct);
            g.appendChild(svgEl('line', {
                x1: lastX, x2: LEFT + LANE_W, y1: lastY, y2: lastY,
                class: 'expov-svg-power-steady'
            }));

            // One annotation per cluster — bracket + "5% → 3%" label
            clusters.forEach(cluster => {
                const first = cluster[0];
                const tail = cluster[cluster.length - 1];
                const xStart = xForT(hist[first.fromIdx].at);
                const xEnd = xForT(hist[tail.toIdx].at);
                const pctStart = hist[first.fromIdx].pct;
                const pctEnd = hist[tail.toIdx].pct;
                const arrow = pctEnd < pctStart ? '↓' : '↑';
                const yMid = (yForPct(pctStart) + yForPct(pctEnd)) / 2;
                // Bracket: small horizontal line above the cluster steps
                const bracketY = Math.min(yForPct(pctStart), yForPct(pctEnd)) - 6;
                g.appendChild(svgEl('path', {
                    d: `M ${xStart} ${bracketY+3} L ${xStart} ${bracketY} L ${xEnd+2} ${bracketY} L ${xEnd+2} ${bracketY+3}`,
                    class: 'expov-svg-power-ramp-bracket'
                }));
                // Label "488 ↓ 5%→3%" anchored just right of the bracket end
                g.appendChild(svgEl('text', {
                    x: xEnd + 6, y: bracketY + 4,
                    class: 'expov-svg-power-ramp-label'
                }, `${arrow} ${pctStart}%→${pctEnd}%`));
            });

            // Subtle filled area under the curve — helps read overall level
            const areaPts = [`${xForT(hist[0].at)},${powerYBase}`];
            for (let k = 0; k < hist.length; k++) {
                const x = xForT(hist[k].at);
                const y = yForPct(hist[k].pct);
                areaPts.push(`${x},${y}`);
                if (k < hist.length - 1) {
                    const xNext = xForT(hist[k+1].at);
                    areaPts.push(`${xNext},${y}`);
                }
            }
            areaPts.push(`${LEFT + LANE_W},${yForPct(last.pct)}`);
            areaPts.push(`${LEFT + LANE_W},${powerYBase}`);
            g.appendChild(svgEl('polygon', {
                points: areaPts.join(' '),
                class: 'expov-svg-power-area'
            }));
        }
        // Power label with current value
        g.appendChild(svgEl('text', {
            x: LEFT - 8, y: powerY + powerH / 2 + 3,
            'text-anchor': 'end',
            class: 'expov-svg-sublabel'
        }, `488 · ${emb.laser_488_pct_now}%`));

        // ---- Dose gauge ----------------------------------------------
        const doseY = powerYBase + 6;
        const doseW = LANE_W;
        g.appendChild(svgEl('rect', {
            x: LEFT, y: doseY, width: doseW, height: DOSE_H, rx: 2,
            class: 'expov-svg-dose-track'
        }));
        const dosePct = emb.dose_used_ms / emb.dose_budget_ms;
        const fillCls = dosePct > 0.85 ? 'expov-svg-dose-fill-crit'
                      : dosePct > 0.60 ? 'expov-svg-dose-fill-warn'
                      : 'expov-svg-dose-fill-ok';
        g.appendChild(svgEl('rect', {
            x: LEFT, y: doseY, width: Math.max(1, doseW * dosePct), height: DOSE_H, rx: 2,
            class: fillCls
        }));
        // Dose label (shows 10× hint for calibration role)
        const doseLabel = emb.role === 'calibration' ? 'dose (10×)' : 'dose';
        g.appendChild(svgEl('text', {
            x: LEFT - 8, y: doseY + DOSE_H - 2,
            'text-anchor': 'end',
            class: 'expov-svg-sublabel'
        }, doseLabel));
        // Inside the bar: usage figure
        g.appendChild(svgEl('text', {
            x: LEFT + 6, y: doseY + DOSE_H - 2,
            class: 'expov-svg-dose-text'
        }, `${(emb.dose_used_ms/1000).toFixed(1)} / ${(emb.dose_budget_ms/1000).toFixed(1)} s`));

        return g;
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
