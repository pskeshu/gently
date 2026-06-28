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
    activeStrategy: null,    // last fetched/loaded snapshot
    isLive: false,           // true when activeStrategy came from the API

    async init() {
        console.log('[ExperimentOverview] init() called, view=', this.activeView);
        const strategy = await this.loadStrategy();
        this.activeStrategy = strategy;
        this.render(strategy);
        this.initialized = true;
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
                this.isLive = false;
                return null;
            }
            const data = await resp.json();
            this.isLive = true;
            return data;
        } catch (e) {
            console.warn('[ExperimentOverview] strategy fetch error:', e);
            this.isLive = false;
            return null;
        }
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
        if (!s) {
            // No active experiment — a calm empty state, never stubbed data.
            root.innerHTML = '<div style="padding:32px;text-align:center;color:var(--text-muted,#94a3b8);font-size:13px;">' +
                'No active experiment — the imaging tactics (cadence, reactive rules) will appear here once a run is live.</div>';
            return;
        }
        try {
            root.innerHTML = '';
            if (this.activeView === 'rules') {
                this._renderRulesView(root, s);
            } else {
                this._renderOverviewView(root, s);
                this._startNowTicker();
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

    // The "now" marker advances with wall-clock time and shows a countdown to
    // the next base-interval acquisition. We update only the marker group's
    // transform + the chip text, never re-rendering the whole SVG. Tick rate
    // is ~4 Hz which keeps the line motion visibly smooth without burning
    // cycles. Skipped while the tab is hidden.
    _startNowTicker() {
        this._stopNowTicker();
        const tick = () => {
            if (!this._nowTickerCtx) return;
            if (!document.hidden) this._updateNowMarker();
            this._nowTickerHandle = setTimeout(tick, 250);
        };
        this._nowTickerHandle = setTimeout(tick, 250);
    },

    _stopNowTicker() {
        if (this._nowTickerHandle) {
            clearTimeout(this._nowTickerHandle);
            this._nowTickerHandle = null;
        }
    },

    _updateNowMarker() {
        const ctx = this._nowTickerCtx;
        if (!ctx || !ctx.marker.isConnected) return;
        const elapsedRealS = (Date.now() - ctx.renderedAtMs) / 1000;
        const effOffsetS = Math.min(
            ctx.renderedOffsetS + elapsedRealS,
            ctx.horizonS
        );
        const x = ctx.xForT(effOffsetS);
        ctx.marker.setAttribute('transform', `translate(${x},0)`);

        // Wall-clock from session-anchored time so the line and the clock
        // can't drift apart even if the client clock is wrong.
        const wallMs = ctx.startedAtMs + effOffsetS * 1000;
        const d = new Date(wallMs);
        const hh = String(d.getHours()).padStart(2, '0');
        const mm = String(d.getMinutes()).padStart(2, '0');
        const ss = String(d.getSeconds()).padStart(2, '0');

        let label = `${hh}:${mm}:${ss}`;
        if (ctx.baseIntervalS > 0) {
            const nextTickS = Math.ceil(effOffsetS / ctx.baseIntervalS) * ctx.baseIntervalS;
            const remainS = Math.max(0, Math.round(nextTickS - effOffsetS));
            const rm = Math.floor(remainS / 60);
            const rs = String(remainS % 60).padStart(2, '0');
            label += ` · next ${rm}:${rs}`;
        }
        ctx.chipText.textContent = label;

        // Size the chip to fit; flip to the left of the line if we're near
        // the right edge so it stays on-screen.
        const textLen = label.length * 6.2 + 12;
        const nearEnd = x + textLen + 8 > ctx.laneRight;
        if (nearEnd) {
            ctx.chipBg.setAttribute('x', -textLen - 4);
            ctx.chipBg.setAttribute('width', textLen);
            ctx.chipText.setAttribute('x', -textLen + 2);
        } else {
            ctx.chipBg.setAttribute('x', 4);
            ctx.chipBg.setAttribute('width', textLen);
            ctx.chipText.setAttribute('x', 10);
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

        // Session identification — the navbar already carries the id on
        // every tab, so we only render a name line when it actually adds
        // info (i.e. a human label, not a hash). The data-source badge is
        // still useful and gets its own row so it stays visible.
        const metaRow = el('div', 'expov-header-row expov-header-row-meta');
        if (s.session_name && s.session_name !== s.session_id) {
            metaRow.appendChild(elText('span', 'expov-session-name', s.session_name));
        }
        metaRow.appendChild(elText('span', 'expov-live-badge', 'live'));
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
        const budgetText = (s.dose_budget_base_ms != null && isFinite(s.dose_budget_base_ms))
            ? `${(s.dose_budget_base_ms / 1000).toFixed(0)}s × role`
            : 'no limit';
        metricsRow.appendChild(metric('budget', budgetText));
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

        const TACTIC_BAND_H = 28;           // session-level tactic band height

        // Aggregate tactic protocol from embryos (first embryo with one wins)
        let tacticProto = null;
        let tacticSetpointChanges = [];
        for (const emb of s.embryos) {
            if (emb.temp_protocol) {
                tacticProto = emb.temp_protocol;
                tacticSetpointChanges = emb.setpoint_changes || [];
                break;
            }
        }
        const bandShift = tacticProto ? TACTIC_BAND_H : 0;

        const rows = s.embryos.length;
        const H = TOP_AXIS_H + bandShift + rows * ROW_H + BOTTOM_PAD;

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

        // ----- "now" marker: translatable group containing the vertical line
        // and a live clock chip. The chip advances every tick and shows the
        // countdown to the next base-interval acquisition window. The group
        // gets translated by _tickNow, so we don't rebuild SVG every second.
        const nowMarker = svgEl('g', { class: 'expov-svg-now-marker' });
        nowMarker.appendChild(svgEl('line', {
            x1: 0, x2: 0, y1: TOP_AXIS_H - 4, y2: H - BOTTOM_PAD,
            class: 'expov-svg-now-line'
        }));
        // Chip sits just below the axis labels (which live at y=12 and y=22)
        // so it doesn't sit on top of the "+0h / wallclock" annotation when
        // the now-line is near the start of the timeline.
        const chipBg = svgEl('rect', {
            x: 4, y: TOP_AXIS_H - 6, width: 120, height: 14, rx: 7,
            class: 'expov-svg-now-chip-bg'
        });
        const chipText = svgEl('text', {
            x: 10, y: TOP_AXIS_H + 4, class: 'expov-svg-now-label'
        }, '');
        nowMarker.appendChild(chipBg);
        nowMarker.appendChild(chipText);
        svg.appendChild(nowMarker);

        // ----- tactic band: session-level protocol pill + setpoint markers
        if (tacticProto) {
            const bandY = TOP_AXIS_H + 2;
            const bandContentH = TACTIC_BAND_H - 4;
            const bandMidY = bandY + bandContentH / 2;
            const bandG = svgEl('g', { class: 'expov-tactic-band-g' });

            // Background strip spanning the full lane area
            bandG.appendChild(svgEl('rect', {
                x: LEFT, y: bandY, width: LANE_W, height: bandContentH, rx: 3,
                class: 'expov-svg-tactic-band-bg'
            }));

            // Tactic pill: spans protocol start → end (or now if still running)
            const pillX0 = xForT(tacticProto.start);
            const pillX1 = xForT(tacticProto.end != null ? tacticProto.end : s.now_offset_s);
            const pillW = Math.max(4, pillX1 - pillX0);
            const pillLabel = `Temp-change burst → ${tacticProto.target_setpoint_c}°C`;
            bandG.appendChild(svgEl('rect', {
                x: pillX0, y: bandY + 2, width: pillW, height: bandContentH - 4, rx: 4,
                class: 'expov-svg-tactic-pill'
            }));
            if (pillW > 24) {
                bandG.appendChild(svgEl('text', {
                    x: pillX0 + 7, y: bandMidY + 3.5,
                    class: 'expov-svg-tactic-pill-label'
                }, pillLabel));
            }

            // Setpoint markers: vertical line + "→ N°C" chip (reuses chip idiom)
            tacticSetpointChanges.forEach(sc => {
                const scX = xForT(sc.t);
                const markerLabel = `→ ${sc.to}°C`;
                const chipW = markerLabel.length * 5.6 + 10;
                const chipY = bandY + 2;
                bandG.appendChild(svgEl('line', {
                    x1: scX, x2: scX, y1: bandY, y2: bandY + bandContentH,
                    class: 'expov-svg-tactic-setpoint-line'
                }));
                bandG.appendChild(svgEl('rect', {
                    x: scX - chipW / 2, y: chipY,
                    width: chipW, height: 12, rx: 3,
                    class: 'expov-svg-tactic-setpoint-chip-bg'
                }));
                bandG.appendChild(svgEl('text', {
                    x: scX, y: chipY + 9,
                    'text-anchor': 'middle',
                    class: 'expov-svg-tactic-setpoint-chip'
                }, markerLabel));
            });

            // Gutter label
            bandG.appendChild(svgEl('text', {
                x: LEFT - 8, y: bandMidY + 3.5,
                'text-anchor': 'end',
                class: 'expov-svg-tactic-band-label'
            }, 'tactic'));

            svg.appendChild(bandG);
        }

        // Stash the bits the ticker needs to update without re-rendering.
        this._nowTickerCtx = {
            marker: nowMarker,
            chipBg: chipBg,
            chipText: chipText,
            xForT,
            startedAtMs: new Date(s.started_at).getTime(),
            renderedAtMs: Date.now(),
            renderedOffsetS: s.now_offset_s,
            baseIntervalS: s.base_interval_s || 0,
            horizonS: s.horizon_s,
            laneLeft: LEFT,
            laneRight: LEFT + LANE_W,
        };
        this._updateNowMarker();

        // ----- one group per embryo
        s.embryos.forEach((emb, i) => {
            const rowTop = TOP_AXIS_H + bandShift + i * ROW_H;
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
        // role tag — eyebrow above the id (avoids colliding with long ids)
        g.appendChild(svgEl('text', {
            x: 14, y: rowTop + ROW_PAD - 1,
            class: 'expov-svg-role-tag'
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
            const cls = (ph.mode === 'burst' && ph.phase)
                ? `expov-svg-phase-${ph.mode} expov-burst-phase-${ph.phase}`
                : `expov-svg-phase-${ph.mode}`;
            g.appendChild(svgEl('rect', {
                x: x0, y: laneY, width, height: LANE_H, rx: 2,
                class: cls
            }));
            // Cadence text inside the rect is intentionally omitted — the
            // gutter pill (currentPhase) and the colored rect (mode) already
            // convey it. Keep an inline label only for cooldown, which is a
            // transient state the pill won't be showing.
            if (ph.mode === 'cooldown' && ph.cadence_s && width >= 42) {
                g.appendChild(svgEl('text', {
                    x: x0 + width / 2, y: laneMid + 3.5,
                    'text-anchor': 'middle',
                    class: 'expov-svg-phase-label'
                }, `${ph.cadence_s}s · cool`));
            }
            // Burst: keep the bright block + balloon since it's the most
            // attention-worthy event in the lane
            if (ph.mode === 'burst') {
                const bx = (xForT(ph.start) + xForT(ph.end)) / 2;
                const balloonY = laneY - 12;
                const halfW = 30;
                const balloonCls = ph.phase
                    ? `expov-svg-burst-balloon-bg expov-burst-phase-${ph.phase}`
                    : 'expov-svg-burst-balloon-bg';
                g.appendChild(svgEl('rect', {
                    x: bx - halfW, y: balloonY - 10, width: halfW * 2, height: 12, rx: 3,
                    class: balloonCls
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

        // ---- Acquisition density heatmap ----------------------------
        // Instead of one hairline per acquisition (reads as a barcode),
        // we encode acquisition density as a luminance gradient over
        // the past portion of the lane: sparse = lane stays muted,
        // dense = a brighter band. The eye reads acquisition rate as
        // brightness — no discrete marks, no clutter.
        //
        // Counts are derived from phase cadence then rescaled to match
        // the authoritative `tp_acquired`, so the gradient never lies
        // about how many timepoints fired even when the backend phase
        // history is stale or incorrect.
        const tickEnd = s.now_offset_s;
        const ackPhases = emb.phases
            .map((ph, i) => ({ ph, i }))
            .filter(({ ph }) => ph.mode !== 'burst' && ph.cadence_s);
        const predicted = ackPhases.map(({ ph }) => {
            const phEnd = Math.min(ph.end ?? tickEnd, tickEnd);
            const dur = phEnd - ph.start;
            return dur > 0 ? Math.max(1, Math.floor(dur / ph.cadence_s) + 1) : 0;
        });
        const predictedTotal = predicted.reduce((a, b) => a + b, 0);
        const actualTotal = Number.isFinite(emb.tp_acquired)
            ? emb.tp_acquired : predictedTotal;
        const scale = predictedTotal > 0 ? actualTotal / predictedTotal : 0;
        const acquisitions = [];
        ackPhases.forEach(({ ph }, idx) => {
            const phEnd = Math.min(ph.end ?? tickEnd, tickEnd);
            const dur = phEnd - ph.start;
            if (dur <= 0) return;
            const n = Math.max(1, Math.round(predicted[idx] * scale));
            for (let j = 0; j < n; j++) {
                acquisitions.push(ph.start + (dur * (j + 0.5)) / n);
            }
        });

        if (acquisitions.length > 0 && tickEnd > 0) {
            // Layer 1 (background): smoothed luminance gradient
            // encoding overall acquisition density along the lane.
            // The triangular kernel kills aliasing stripes caused by
            // evenly-spaced acquisitions falling into bins.
            const BINS = 64;
            const binSec = tickEnd / BINS;
            const raw = new Array(BINS).fill(0);
            for (const t of acquisitions) {
                const bin = Math.min(BINS - 1, Math.max(0, Math.floor(t / binSec)));
                raw[bin] += 1;
            }
            const kernel = [1, 2, 3, 4, 5, 4, 3, 2, 1];
            const kSum = kernel.reduce((a, b) => a + b, 0);
            const kOff = Math.floor(kernel.length / 2);
            const density = new Array(BINS).fill(0);
            for (let i = 0; i < BINS; i++) {
                let acc = 0, w = 0;
                for (let k = 0; k < kernel.length; k++) {
                    const j = i + k - kOff;
                    if (j < 0 || j >= BINS) continue;
                    acc += raw[j] * kernel[k];
                    w += kernel[k];
                }
                density[i] = w > 0 ? acc / w * (kSum / w) : 0;
            }
            const maxD = Math.max(...density, 1e-6);
            const gradId = `expov-density-${(emb.id || 'e').replace(/\W+/g, '_')}-r${rowTop}`;
            const grad = svgEl('linearGradient', {
                id: gradId, x1: '0%', x2: '100%', y1: '0%', y2: '0%'
            });
            for (let i = 0; i < BINS; i++) {
                const intensity = density[i] / maxD;
                // Lower ceiling than the heatmap-only version (0.22 vs
                // 0.38) because the dots above will carry the per-event
                // signal; the band just hints at rate.
                const alpha = 0.03 + 0.22 * intensity;
                grad.appendChild(svgEl('stop', {
                    offset: `${(i / (BINS - 1)) * 100}%`,
                    'stop-color': '#ffffff',
                    'stop-opacity': alpha.toFixed(3),
                }));
            }
            g.appendChild(grad);
            const pastW = xForT(tickEnd) - LEFT;
            if (pastW > 0) {
                g.appendChild(svgEl('rect', {
                    x: LEFT, y: laneY,
                    width: pastW, height: LANE_H,
                    fill: `url(#${gradId})`,
                    rx: 2,
                    'pointer-events': 'none'
                }));
            }
            // Layer 2 (foreground): one soft round dot per acquisition
            // along the top edge of the lane — keeps the per-event
            // temporal discreteness the heatmap alone hides.
            const dotY = laneY + 2;
            for (const t of acquisitions) {
                const tx = xForT(t);
                g.appendChild(svgEl('circle', {
                    cx: tx, cy: dotY, r: 1.3,
                    class: 'expov-svg-acq-dot'
                }));
            }
        }

        // ---- Cadence-change markers ---------------------------------
        // Where consecutive phases differ in cadence (or mode), drop a
        // vertical divider across the lane and a "300→60s · T34" chip
        // above so the change is named and time-stamped in the lane.
        // Apply the same scale factor used for tick rendering so the
        // T# stamps shown on diamonds and cadence chips agree with the
        // visible tick density and the authoritative tp_acquired count.
        // Otherwise the chip might say "T118" on an embryo where we
        // only drew 54 ticks — visually contradictory.
        const tpIndexAt = (atS) => {
            let count = 0;
            for (const ph of emb.phases) {
                if (!ph.cadence_s) continue;
                if (atS < ph.start) break;
                const phEnd = Math.min(atS, ph.end ?? atS);
                count += Math.floor((phEnd - ph.start) / ph.cadence_s) + 1;
            }
            return Math.max(1, Math.round(count * scale));
        };
        // Track placed chip x-positions to avoid stacking chips on top
        // of one another (e.g. the burst-balloon already sits there).
        const placedChipX = [];
        const burstXs = emb.phases
            .filter(p => p.mode === 'burst')
            .map(p => xForT(p.start));
        const isCollision = (x) => {
            const min = 70;  // px buffer
            if (burstXs.some(bx => Math.abs(bx - x) < min)) return true;
            return placedChipX.some(px => Math.abs(px - x) < min);
        };
        // Walk phases and detect transitions, but COLLAPSE consecutive
        // identical (mode + cadence) phases so a row of redundant phase
        // records doesn't generate redundant dividers/chips.
        let prevEffective = emb.phases[0];
        for (let i = 1; i < emb.phases.length; i++) {
            const curr = emb.phases[i];
            const prev = prevEffective;
            const sameCadence = prev.cadence_s === curr.cadence_s;
            const sameMode = prev.mode === curr.mode;
            if (sameCadence && sameMode) {
                continue;  // collapse: prevEffective stays the same
            }
            prevEffective = curr;
            if (prev.mode === 'burst' || curr.mode === 'burst') continue;
            const cx = xForT(curr.start);
            if (cx > xForT(s.now_offset_s)) continue;
            // Divider is cheap to keep even on collision; the chip is what
            // crowds the space, so we skip just the chip when crowded.
            g.appendChild(svgEl('line', {
                x1: cx, x2: cx, y1: laneY, y2: laneBottom,
                class: 'expov-svg-cadence-divider'
            }));
            if (isCollision(cx)) continue;
            const tp = tpIndexAt(curr.start);
            const prevS = prev.cadence_s ?? '?';
            const currS = curr.cadence_s ?? '?';
            const chipText = `${prevS}→${currS}s · T${tp}`;
            const chipW = chipText.length * 5.6 + 10;
            const chipY = laneY - 13;
            g.appendChild(svgEl('rect', {
                x: cx - chipW / 2, y: chipY,
                width: chipW, height: 12, rx: 3,
                class: 'expov-svg-cadence-chip-bg'
            }));
            g.appendChild(svgEl('text', {
                x: cx, y: chipY + 9,
                'text-anchor': 'middle',
                class: 'expov-svg-cadence-chip'
            }, chipText));
            placedChipX.push(cx);
        }

        // ---- Power-change chips -------------------------------------
        // Same visual language as cadence chips, but parked in a row
        // above so the two encodings stack neatly when they happen at
        // the same trigger. Each chip names the rule outcome
        // ("488 ↓ 5%→3%") and the timepoint it landed at.
        const placedPowerChipX = [];
        const hist488 = emb.power_history_488 || [];
        // Walk the history, collect actual transitions (pairs where pct
        // changes), then cluster consecutive close ones so a multi-step
        // ramp gets a single annotation.
        const transitions = [];
        for (let k = 0; k < hist488.length - 1; k++) {
            const a = hist488[k];
            const b = hist488[k + 1];
            if (a.pct === b.pct) continue;
            transitions.push({ from: a, to: b });
        }
        const CLUSTER_S = 60;
        const clusters = [];
        for (const tr of transitions) {
            const last = clusters[clusters.length - 1];
            if (last && tr.to.at - last[last.length - 1].to.at <= CLUSTER_S) {
                last.push(tr);
            } else {
                clusters.push([tr]);
            }
        }
        for (const cluster of clusters) {
            const first = cluster[0];
            const tail = cluster[cluster.length - 1];
            // Anchor the chip at the actual change time, not at any
            // trailing anchor record (those can land past `now` and
            // get hidden by the past-only guard).
            const atS = Math.min(tail.to.at, s.now_offset_s);
            const cx = xForT(atS);
            const arrow = tail.to.pct < first.from.pct ? '↓' : '↑';
            const chipText = `488 ${arrow} ${first.from.pct}%→${tail.to.pct}% · T${tpIndexAt(atS)}`;
            const chipW = chipText.length * 5.6 + 10;
            // Sit just above the burst-balloon band (which lives at
            // laneY-22..-10) so we stay within this row's vertical
            // budget — laneY-40 would have crossed into the row above.
            // Burst balloons live at separate x positions on every
            // case I've seen, so dropping the burst-collision check
            // lets the chip render even when a burst is on the same
            // lane elsewhere.
            const chipY = laneY - 25;
            const crowded =
                placedPowerChipX.some(px => Math.abs(px - cx) < 70);
            g.appendChild(svgEl('line', {
                x1: cx, x2: cx, y1: chipY + 12, y2: laneY,
                class: 'expov-svg-power-chip-stem'
            }));
            if (crowded) continue;
            g.appendChild(svgEl('rect', {
                x: cx - chipW / 2, y: chipY,
                width: chipW, height: 12, rx: 3,
                class: 'expov-svg-power-chip-bg'
            }));
            g.appendChild(svgEl('text', {
                x: cx, y: chipY + 9,
                'text-anchor': 'middle',
                class: 'expov-svg-power-chip'
            }, chipText));
            placedPowerChipX.push(cx);
        }

        // Projected future segment (dashed) past 'now' to a horizon —
        // skipped entirely when the embryo has been terminated, since
        // there's no future to project. projEndT is hoisted because
        // downstream code (stop-icon, dose-exhaust line) anchors to it.
        const isTerminated = emb.terminated_at_s != null
            && emb.terminated_at_s <= s.now_offset_s;
        const projStartT = s.now_offset_s;
        let projEndT = isTerminated ? emb.terminated_at_s : s.horizon_s;
        let projEndsAtBudget = false;
        if (!isTerminated) {
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
        }
        // Terminated cap: small vertical stop bar at the termination
        // point + a "■ DONE · T##" label below the lane so a finished
        // embryo doesn't look like it's still acquiring.
        if (isTerminated) {
            const termX = xForT(emb.terminated_at_s);
            g.appendChild(svgEl('line', {
                x1: termX, x2: termX,
                y1: laneY - 2, y2: laneBottom + 2,
                class: 'expov-svg-terminated-bar'
            }));
            g.appendChild(svgEl('rect', {
                x: termX - 2, y: laneY + LANE_H / 2 - 3,
                width: 6, height: 6,
                class: 'expov-svg-terminated-stop'
            }));
            const tp = tpIndexAt(emb.terminated_at_s);
            const capText = `DONE · T${tp}`;
            g.appendChild(svgEl('text', {
                x: termX + 6, y: laneBottom + 9,
                class: 'expov-svg-terminated-label'
            }, capText));
        }

        // Trigger diamonds — placed in the upper half of the lane to avoid
        // colliding with the burst balloon above. Each diamond gets a tiny
        // T# label below it so the user can see at which timepoint the
        // rule fired without hovering.
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
            g.appendChild(svgEl('text', {
                x: x, y: laneBottom + 9,
                'text-anchor': 'middle',
                class: 'expov-svg-trigger-tp'
            }, `T${tpIndexAt(te.at)}`));
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
        // Power label with current value — "@" reads as "at this power"
        // and avoids confusion with the bullet-separator used elsewhere
        g.appendChild(svgEl('text', {
            x: LEFT - 8, y: powerY + powerH / 2 + 3,
            'text-anchor': 'end',
            class: 'expov-svg-sublabel'
        }, `488 @ ${emb.laser_488_pct_now}%`));

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
        // Inside the bar: usage figure — "used of budget" is more scannable
        // than "x / y s" which reads like a fraction
        const usedS = (emb.dose_used_ms / 1000).toFixed(1);
        const budgetS = (emb.dose_budget_ms / 1000).toFixed(1);
        const doseText = emb.dose_budget_ms > 0
            ? `${usedS}s of ${budgetS}s (${Math.round(dosePct * 100)}%)`
            : `${usedS}s used`;
        g.appendChild(svgEl('text', {
            x: LEFT + 6, y: doseY + DOSE_H - 2,
            class: 'expov-svg-dose-text'
        }, doseText));

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
