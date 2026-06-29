/**
 * Gallery rendering functionality for Gently Visualization
 */

function renderRecentList() {
    const list = document.getElementById('recent-list');
    if (!list) return;

    const filtered = filterByEmbryo(state.snapshots);
    const displayList = filtered.slice(-20).reverse();

    if (displayList.length === 0) {
        list.innerHTML = '<div class="recent-strip-empty">No images yet</div>';
        return;
    }

    list.innerHTML = displayList.map((img, index) => `
        <div class="recent-strip-item ${state.currentImage?.uid === img.uid ? 'active' : ''}"
             onclick="openSnapshotsLightbox(${index})"
             title="${img.data_type} - ${img.uid.slice(0, 8)}">
            <img src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
        </div>
    `).join('');
}

function openSnapshotsLightbox(index) {
    const filtered = filterByEmbryo(state.snapshots);
    const displayList = filtered.slice(-20).reverse();
    Lightbox.open(displayList, index, 'snapshots');
}

// ==========================================
// Calibration Profile View - SVG Z-scan visualization
// ==========================================

// ==========================================
// CalibrationCharts - SVG chart rendering for focus curves & calibration summary
// ==========================================

const CalibrationCharts = {
    CHART_W: 380,
    CHART_H: 220,
    MARGIN: { top: 28, right: 20, bottom: 36, left: 44 },

    /**
     * Render a focus curve chart from sweep data points.
     * @param {Array} points - [{piezo, score}, ...] sorted by piezo
     * @param {string} galvoName - 'top' or 'bottom'
     * @param {number|null} bestPiezo - optimal piezo position
     * @param {number|null} rSquared - fit quality
     */
    renderFocusCurve(points, galvoName, bestPiezo, rSquared) {
        if (!points || points.length < 2) return '';

        const M = this.MARGIN;
        const W = this.CHART_W;
        const H = this.CHART_H;
        const pw = W - M.left - M.right;
        const ph = H - M.top - M.bottom;

        // Normalize scores to 0-1 range (absolute values are meaningless)
        const piezos = points.map(p => p.piezo);
        const rawScores = points.map(p => p.score);
        const rawMin = Math.min(...rawScores);
        const rawMax = Math.max(...rawScores);
        const rawRange = rawMax - rawMin || 1;
        const normPoints = points.map(p => ({
            piezo: p.piezo,
            score: (p.score - rawMin) / rawRange
        }));

        // Data ranges
        const pMin = Math.min(...piezos);
        const pMax = Math.max(...piezos);
        const pPad = (pMax - pMin) * 0.08 || 1;
        const pRange = [pMin - pPad, pMax + pPad];
        const sRange = [-0.08, 1.12]; // normalized 0-1 with padding

        const xScale = v => M.left + (v - pRange[0]) / (pRange[1] - pRange[0]) * pw;
        const yScale = v => M.top + ph - (v - sRange[0]) / (sRange[1] - sRange[0]) * ph;

        let svg = `<svg class="cal-chart-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;

        // Grid lines
        svg += this._renderGrid(M, pw, ph, pRange, sRange, xScale, yScale);

        // Gaussian fit curve (on normalized data)
        const fitParams = this._estimateGaussian(normPoints);
        if (fitParams) {
            const fitPoints = [];
            const steps = 80;
            for (let i = 0; i <= steps; i++) {
                const x = pRange[0] + (pRange[1] - pRange[0]) * i / steps;
                const y = fitParams.a * Math.exp(-((x - fitParams.mu) ** 2) / (2 * fitParams.sigma ** 2)) + fitParams.c;
                fitPoints.push(`${xScale(x).toFixed(1)},${yScale(y).toFixed(1)}`);
            }
            svg += `<polyline points="${fitPoints.join(' ')}" fill="none"
                     stroke="var(--accent-orange, #f97316)" stroke-width="2" stroke-opacity="0.8"/>`;
        }

        // Best position line
        if (bestPiezo != null && bestPiezo >= pRange[0] && bestPiezo <= pRange[1]) {
            const bx = xScale(bestPiezo);
            svg += `<line x1="${bx}" y1="${M.top}" x2="${bx}" y2="${M.top + ph}"
                     stroke="var(--accent-green, #22c55e)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8"/>`;
            // Clamp label position to stay within SVG
            const labelX = Math.min(Math.max(bx, M.left + 30), W - M.right - 30);
            svg += `<text x="${labelX}" y="${M.top - 6}" text-anchor="middle"
                     class="cal-chart-label" fill="var(--accent-green, #22c55e)"
                     font-size="9">${bestPiezo.toFixed(1)} µm</text>`;
        }

        // Data points (normalized)
        normPoints.forEach(p => {
            svg += `<circle cx="${xScale(p.piezo).toFixed(1)}" cy="${yScale(p.score).toFixed(1)}"
                     r="4" fill="var(--accent, #3b82f6)" stroke="var(--bg-card)" stroke-width="1"/>`;
        });

        // Title
        const label = galvoName ? galvoName.toUpperCase() : '';
        const r2Text = rSquared != null ? `  R²=${rSquared.toFixed(3)}` : '';
        svg += `<text x="${M.left + pw / 2}" y="16" text-anchor="middle"
                 class="cal-chart-title">${label} Focus Curve</text>`;
        if (r2Text) {
            svg += `<text x="${W - M.right}" y="16" text-anchor="end"
                     class="cal-chart-tick" font-size="9" opacity="0.6">${r2Text.trim()}</text>`;
        }

        // Axes labels
        svg += `<text x="${M.left + pw / 2}" y="${H - 4}" text-anchor="middle"
                 class="cal-chart-label">Piezo (µm)</text>`;
        svg += `<text x="10" y="${M.top + ph / 2}" text-anchor="middle"
                 class="cal-chart-label" transform="rotate(-90, 10, ${M.top + ph / 2})">Score</text>`;

        svg += '</svg>';
        return svg;
    },

    /**
     * Render calibration summary (piezo vs galvo linear fit).
     */
    renderCalibrationSummary(info) {
        if (info.focusTopGalvo == null || info.focusBotGalvo == null) return '';

        const M = this.MARGIN;
        const W = this.CHART_W;
        const H = this.CHART_H;
        const pw = W - M.left - M.right;
        const ph = H - M.top - M.bottom;

        // Get the two calibration points
        const g1 = info.focusTopGalvo, g2 = info.focusBotGalvo;
        // Compute piezo from slope/offset
        const p1 = info.slope * g1 + info.offset;
        const p2 = info.slope * g2 + info.offset;

        const galvos = [g1, g2];
        const piezos = [p1, p2];

        const gMin = Math.min(...galvos), gMax = Math.max(...galvos);
        const pMin = Math.min(...piezos), pMax = Math.max(...piezos);
        const gPad = (gMax - gMin) * 0.25 || 0.05;
        const pPad = (pMax - pMin) * 0.25 || 2;
        const gRange = [gMin - gPad, gMax + gPad];
        const pRange = [pMin - pPad, pMax + pPad];

        const xScale = v => M.left + (v - gRange[0]) / (gRange[1] - gRange[0]) * pw;
        const yScale = v => M.top + ph - (v - pRange[0]) / (pRange[1] - pRange[0]) * ph;

        let svg = `<svg class="cal-chart-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;

        // Grid
        svg += this._renderGrid(M, pw, ph, gRange, pRange, xScale, yScale);

        // Linear fit line (extend beyond points)
        const fitG0 = gRange[0], fitG1 = gRange[1];
        const fitP0 = info.slope * fitG0 + info.offset;
        const fitP1 = info.slope * fitG1 + info.offset;
        svg += `<line x1="${xScale(fitG0)}" y1="${yScale(fitP0)}" x2="${xScale(fitG1)}" y2="${yScale(fitP1)}"
                 stroke="var(--accent-orange, #f97316)" stroke-width="2" opacity="0.8"/>`;

        // Data points with labels
        galvos.forEach((g, i) => {
            const px = xScale(g), py = yScale(piezos[i]);
            const label = i === 0 ? 'Top' : 'Bottom';
            const r2 = i === 0 ? info.rSquaredTop : info.rSquaredBot;
            const r2Str = r2 != null ? `R²=${r2.toFixed(3)}` : '';
            svg += `<circle cx="${px}" cy="${py}" r="5"
                     fill="var(--accent, #3b82f6)" stroke="var(--bg-card)" stroke-width="1.5"/>`;
            // Position label left of point if near right edge
            const nearRight = px > M.left + pw * 0.65;
            const tx = nearRight ? px - 8 : px + 8;
            const anchor = nearRight ? 'end' : 'start';
            svg += `<text x="${tx}" y="${py + (i === 0 ? -8 : 14)}"
                     text-anchor="${anchor}" class="cal-chart-label" font-size="9">${label} ${r2Str}</text>`;
        });

        // Title
        svg += `<text x="${M.left + pw / 2}" y="16" text-anchor="middle"
                 class="cal-chart-title">Piezo-Galvo Calibration</text>`;

        // Equation
        const eq = `piezo = ${info.slope.toFixed(1)}·galvo + ${info.offset.toFixed(1)}`;
        svg += `<text x="${M.left + pw / 2}" y="${H - 4}" text-anchor="middle"
                 class="cal-chart-label" font-size="9" opacity="0.7">${eq}</text>`;

        // Axis labels
        svg += `<text x="${M.left + pw / 2}" y="${H - 16}" text-anchor="middle"
                 class="cal-chart-label">Galvo (deg)</text>`;
        svg += `<text x="12" y="${M.top + ph / 2}" text-anchor="middle"
                 class="cal-chart-label" transform="rotate(-90, 12, ${M.top + ph / 2})">Piezo (µm)</text>`;

        svg += '</svg>';
        return svg;
    },

    /** Estimate Gaussian params from data: y = a * exp(-(x-mu)^2/(2*sigma^2)) + c */
    _estimateGaussian(points) {
        if (points.length < 3) return null;

        const sorted = [...points].sort((a, b) => a.piezo - b.piezo);
        const scores = sorted.map(p => p.score);
        const piezos = sorted.map(p => p.piezo);

        const c = Math.min(...scores);
        const maxIdx = scores.indexOf(Math.max(...scores));
        const mu = piezos[maxIdx];
        const a = scores[maxIdx] - c;
        if (a <= 0) return null;

        // Estimate sigma from half-max width
        const halfMax = c + a / 2;
        let left = mu, right = mu;
        for (let i = maxIdx; i >= 0; i--) {
            if (scores[i] <= halfMax) { left = piezos[i]; break; }
            if (i === 0) left = piezos[0];
        }
        for (let i = maxIdx; i < scores.length; i++) {
            if (scores[i] <= halfMax) { right = piezos[i]; break; }
            if (i === scores.length - 1) right = piezos[i];
        }
        const fwhm = Math.abs(right - left) || (piezos[piezos.length - 1] - piezos[0]) * 0.3;
        const sigma = fwhm / 2.355; // FWHM = 2*sqrt(2*ln2)*sigma

        return sigma > 0 ? { a, mu, sigma, c } : null;
    },

    /** Render grid lines and tick labels */
    _renderGrid(M, pw, ph, xRange, yRange, xScale, yScale) {
        let svg = '';
        const xTicks = this._niceTicks(xRange[0], xRange[1], 5);
        const yTicks = this._niceTicks(yRange[0], yRange[1], 4);

        // Horizontal grid + Y labels
        yTicks.forEach(v => {
            const y = yScale(v);
            if (y < M.top || y > M.top + ph) return;
            svg += `<line x1="${M.left}" y1="${y}" x2="${M.left + pw}" y2="${y}"
                     stroke="var(--border)" stroke-width="0.5" opacity="0.4"/>`;
            svg += `<text x="${M.left - 6}" y="${y + 3}" text-anchor="end"
                     class="cal-chart-tick">${this._formatTick(v)}</text>`;
        });

        // Vertical grid + X labels
        xTicks.forEach(v => {
            const x = xScale(v);
            if (x < M.left || x > M.left + pw) return;
            svg += `<line x1="${x}" y1="${M.top}" x2="${x}" y2="${M.top + ph}"
                     stroke="var(--border)" stroke-width="0.5" opacity="0.4"/>`;
            svg += `<text x="${x}" y="${M.top + ph + 14}" text-anchor="middle"
                     class="cal-chart-tick">${this._formatTick(v)}</text>`;
        });

        // Border
        svg += `<rect x="${M.left}" y="${M.top}" width="${pw}" height="${ph}"
                 fill="none" stroke="var(--border)" stroke-width="0.5" opacity="0.3"/>`;

        return svg;
    },

    /** Generate nice round tick values */
    _niceTicks(min, max, targetCount) {
        const range = max - min;
        if (range <= 0) return [min];
        const rough = range / targetCount;
        const mag = Math.pow(10, Math.floor(Math.log10(rough)));
        const norm = rough / mag;
        let step;
        if (norm < 1.5) step = mag;
        else if (norm < 3.5) step = 2 * mag;
        else if (norm < 7.5) step = 5 * mag;
        else step = 10 * mag;

        const ticks = [];
        let v = Math.ceil(min / step) * step;
        while (v <= max) {
            ticks.push(v);
            v += step;
        }
        return ticks;
    },

    /** Format tick values: use SI-like abbreviation for large numbers */
    _formatTick(v) {
        const abs = Math.abs(v);
        if (abs >= 1e12) return (v / 1e12).toFixed(1) + 'T';
        if (abs >= 1e9) return (v / 1e9).toFixed(1) + 'G';
        if (abs >= 1e6) return (v / 1e6).toFixed(1) + 'M';
        if (abs >= 1e4) return (v / 1e3).toFixed(1) + 'k';
        if (abs >= 100) return v.toFixed(0);
        if (abs >= 1) return v.toFixed(1);
        if (abs >= 0.01) return v.toFixed(2);
        return v.toFixed(3);
    }
};

// ==========================================
// CalibrationProfileView - SVG Z-scan visualization
// ==========================================

const CalibrationProfileView = {
    SVG_WIDTH: 520,
    SVG_HEIGHT: 480,
    MARGIN: { top: 24, right: 60, bottom: 32, left: 56 },

    selectedSliceIndex: null,

    render(embryoId) {
        const images = state.calibration.filter(img =>
            (img.metadata?.embryo_id || 'General') === embryoId
        );

        // Separate by type
        const edgeImages = images.filter(i => i.data_type === 'edge_detection');
        const sweepImages = images.filter(i =>
            ['focus_sweep', 'focus_snap', 'focus_coarse'].includes(i.data_type)
        );
        const focusPlots = images.filter(i =>
            i.data_type === 'focus_plot' || i.data_type === 'focus_curve'
        );
        const summaries = images.filter(i => i.data_type === 'calibration_summary');

        // Collect edge detection images (have galvo positions along the scan axis)
        const edgeData = edgeImages
            .filter(img => img.metadata?.galvo != null)
            .map(img => ({
                galvo: img.metadata.galvo,
                piezo: img.metadata.piezo ?? null,
                visible: img.metadata.visible ?? true,
                feature_score: img.metadata.feature_score || 0,
                data_type: img.data_type,
                image: img
            }))
            .sort((a, b) => a.galvo - b.galvo);

        // Collect sweep data (grouped by galvo_name top/bottom)
        const sweepData = sweepImages
            .filter(img => img.metadata?.piezo != null)
            .map(img => ({
                galvo: img.metadata.galvo ?? 0,
                galvo_name: img.metadata.galvo_name || '',
                piezo: img.metadata.piezo,
                score: img.metadata.score || 0,
                sweep: img.metadata.sweep || '',
                image: img
            }));

        // Try to determine edge boundaries and calibration positions from data
        const calibInfo = this._extractCalibInfo(edgeData, sweepData, images);

        // If no positional data at all, show fallback
        if (edgeData.length === 0 && sweepData.length === 0) {
            return this._renderFallback(images, focusPlots, summaries);
        }

        // Metrics-first 2x2 layout: numbers on top, four chart quadrants
        // below (Focus TOP / Focus BOTTOM / Calibration Summary / Galvo
        // Profile sphere). SPIM live is a compact LED+thumb in the strip
        // — it only earns chart real estate when actively sweeping, and
        // even then we keep it tiny so the metrics dominate.
        const byGalvo = {};
        sweepData.forEach(s => {
            const name = s.galvo_name || 'unknown';
            if (!byGalvo[name]) byGalvo[name] = [];
            byGalvo[name].push({ piezo: s.piezo, score: s.score });
        });

        const focusMeta = {};
        focusPlots.forEach(fp => {
            const m = fp.metadata || {};
            if (m.galvo_name) {
                focusMeta[m.galvo_name] = {
                    bestPiezo: m.best_piezo ?? null,
                    rSquared: m.r_squared ?? null,
                };
            }
        });

        const focusTop = this._buildFocusCurveSvg(byGalvo['top'], 'top', focusMeta['top']);
        const focusBot = this._buildFocusCurveSvg(byGalvo['bottom'], 'bottom', focusMeta['bottom']);

        let summarySvg = '';
        if (calibInfo.slope != null && calibInfo.focusTopGalvo != null) {
            summarySvg = CalibrationCharts.renderCalibrationSummary(calibInfo);
        } else if (summaries.length > 0) {
            const latest = summaries[summaries.length - 1];
            summarySvg = `<img class="cal-quad-img" src="data:image/png;base64,${latest.base64_png}" alt="calibration summary"/>`;
        }

        return `
            <div class="cal-profile-v2">
                ${this._renderMetricsStrip(calibInfo)}
                <div class="cal-grid">
                    ${this._renderQuad('Galvo Profile', this._renderSVG(edgeData, sweepData, calibInfo), 'galvo')}
                    ${this._renderQuad('Calibration Summary', summarySvg || this._emptyQuadBody('Awaiting calibration results'), 'summary')}
                    ${this._renderQuad('Focus Curve — TOP', focusTop, 'focus-top')}
                    ${this._renderQuad('Focus Curve — BOTTOM', focusBot, 'focus-bot')}
                </div>
            </div>
        `;
    },

    /** Compact SPIM live indicator used inside the metrics strip.
     * Carries the same IDs as the old big preview so SpimLivePreview's
     * apply-on-render logic continues to work unchanged. The thumb is a
     * button — click to open the floating popout for a larger view. */
    _renderSpimIndicator() {
        return `
            <div class="cal-spim-indicator" id="cal-spim-preview">
                <span class="cal-spim-led idle" id="cal-spim-led"></span>
                <span class="cal-spim-label">SPIM</span>
                <button type="button" class="cal-spim-thumb-btn" id="cal-spim-thumb-btn"
                        title="Open live SPIM popout"
                        onclick="SpimPopout.open()">
                    <img class="cal-spim-img cal-spim-thumb" id="cal-spim-img" alt="">
                    <span class="cal-spim-expand-icon" aria-hidden="true">⤢</span>
                </button>
                <span class="cal-spim-placeholder" id="cal-spim-placeholder" hidden></span>
                <span class="cal-spim-meta" id="cal-spim-meta">—</span>
            </div>
        `;
    },

    _renderQuad(title, bodyHtml, area) {
        const areaStyle = area ? ` style="grid-area: ${area};"` : '';
        return `
            <div class="cal-quad cal-quad-${area || 'default'}"${areaStyle}>
                <div class="cal-quad-title">${title}</div>
                <div class="cal-quad-body">${bodyHtml}</div>
            </div>
        `;
    },

    _emptyQuadBody(msg) {
        return `<div class="cal-quad-empty">${msg}</div>`;
    },

    _buildFocusCurveSvg(points, name, meta) {
        if (!points || points.length < 2) {
            return this._emptyQuadBody('No sweep data');
        }
        const sorted = points.sort((a, b) => a.piezo - b.piezo);
        return CalibrationCharts.renderFocusCurve(
            sorted, name, meta?.bestPiezo, meta?.rSquared
        );
    },

    _extractCalibInfo(edgeData, sweepData, allImages) {
        const info = {
            edgeTop: null,
            edgeBottom: null,
            scanTop: null,
            scanBottom: null,
            focusTopGalvo: null,
            focusBotGalvo: null,
            slope: null,
            offset: null,
            rSquaredTop: null,
            rSquaredBot: null,
            coverage: null,
            bufferDeg: null
        };

        // Get edge boundaries from edge detection images
        if (edgeData.length > 0) {
            const galvos = edgeData.map(d => d.galvo);
            info.edgeTop = Math.min(...galvos);
            info.edgeBottom = Math.max(...galvos);
            info.coverage = info.edgeBottom - info.edgeTop;
        }

        // Extract calibration results from summary metadata
        // Fields: slope, offset, galvo_top, galvo_bottom, r_squared_top, r_squared_bottom
        for (const img of allImages) {
            const m = img.metadata || {};
            if (m.slope != null) info.slope = m.slope;
            if (m.offset != null) info.offset = m.offset;
            if (m.r_squared_top != null) info.rSquaredTop = m.r_squared_top;
            if (m.r_squared_bottom != null) info.rSquaredBot = m.r_squared_bottom;
            if (m.galvo_top != null) info.focusTopGalvo = m.galvo_top;
            if (m.galvo_bottom != null) info.focusBotGalvo = m.galvo_bottom;
        }

        // Estimate padding (25um default buffer / slope)
        if (info.slope) {
            info.bufferDeg = 25.0 / info.slope;
        } else {
            info.bufferDeg = 0.20;
        }

        // Compute scan boundaries = edge +/- buffer
        if (info.edgeTop != null) {
            info.scanTop = info.edgeTop - info.bufferDeg;
        }
        if (info.edgeBottom != null) {
            info.scanBottom = info.edgeBottom + info.bufferDeg;
        }

        return info;
    },

    _renderSVG(edgeData, sweepData, info) {
        const M = this.MARGIN;
        const W = this.SVG_WIDTH;
        const H = this.SVG_HEIGHT;
        const plotW = W - M.left - M.right;
        const plotH = H - M.top - M.bottom;

        // Fixed galvo Y-axis range so all embryos are directly comparable
        const yMin = -0.50;
        const yMax = 0.55;

        // Y-scale: galvo degrees → pixels (top=yMin, bottom=yMax)
        const yScale = (g) => M.top + plotH * ((g - yMin) / (yMax - yMin));

        // Ellipse representing the embryo
        const ellipseVisible = info.edgeTop != null && info.edgeBottom != null;
        const eTop = ellipseVisible ? yScale(info.edgeTop) : M.top + plotH * 0.2;
        const eBot = ellipseVisible ? yScale(info.edgeBottom) : M.top + plotH * 0.8;
        const cx = M.left + plotW * 0.45;
        const ry = (eBot - eTop) / 2;
        const rx = plotW * 0.32;
        const cy = eTop + ry;

        // Color mapping: use feature_score if available, else visible/invisible
        const hasFeatureScores = edgeData.some(d => d.feature_score > 0);

        const sliceColor = (d) => {
            if (hasFeatureScores && d.feature_score > 0) {
                const scores = edgeData.map(x => x.feature_score).filter(s => s > 0);
                const sMax = Math.max(...scores);
                const t = Math.min(d.feature_score / sMax, 1);
                const r = Math.round(139 + t * (74 - 139));
                const g = Math.round(148 + t * (222 - 148));
                const b = Math.round(158 + t * (128 - 158));
                return `rgb(${r}, ${g}, ${b})`;
            }
            // Fallback: visible = bright, invisible = dim
            return d.visible ? 'var(--accent)' : 'var(--text-muted)';
        };

        // Build SVG parts
        let svgParts = [];

        // Defs - richer gradients and a glow filter for a more refined look
        svgParts.push(`
            <defs>
                <!-- Embryo body: layered radial gradient with off-center highlight -->
                <radialGradient id="embryo-grd" cx="42%" cy="36%" r="80%">
                    <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.22"/>
                    <stop offset="40%" stop-color="var(--accent)" stop-opacity="0.11"/>
                    <stop offset="85%" stop-color="var(--accent)" stop-opacity="0.04"/>
                    <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.01"/>
                </radialGradient>
                <!-- Outer soft halo for the embryo -->
                <radialGradient id="embryo-halo" cx="50%" cy="50%" r="55%">
                    <stop offset="65%" stop-color="var(--accent)" stop-opacity="0"/>
                    <stop offset="88%" stop-color="var(--accent)" stop-opacity="0.07"/>
                    <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
                </radialGradient>
                <!-- Softer padding hatch -->
                <pattern id="pad-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <line x1="0" y1="0" x2="0" y2="8" stroke="var(--border)" stroke-width="0.7" opacity="0.28"/>
                </pattern>
                <!-- Glow filter for focus lines -->
                <filter id="focus-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="2.2" result="blur"/>
                    <feMerge>
                        <feMergeNode in="blur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
                <!-- Subtle drop shadow for the embryo ellipse -->
                <filter id="embryo-shadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
                    <feOffset dy="1" result="off"/>
                    <feComponentTransfer>
                        <feFuncA type="linear" slope="0.35"/>
                    </feComponentTransfer>
                    <feMerge>
                        <feMergeNode/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
        `);

        // Y-axis line and label
        svgParts.push(`
            <line x1="${M.left}" y1="${M.top}" x2="${M.left}" y2="${H - M.bottom}"
                  stroke="var(--border)" stroke-width="1"/>
            <text x="${M.left - 4}" y="${M.top - 10}" text-anchor="end"
                  fill="var(--text-muted)" font-size="10" font-weight="500">Galvo (deg)</text>
        `);

        // Y-axis ticks
        const range = yMax - yMin;
        const tickStep = range > 0.5 ? 0.1 : range > 0.2 ? 0.05 : 0.02;
        const firstTick = Math.ceil(yMin / tickStep) * tickStep;
        for (let g = firstTick; g <= yMax + 0.001; g += tickStep) {
            const y = yScale(g);
            svgParts.push(`
                <line x1="${M.left - 4}" y1="${y}" x2="${M.left}" y2="${y}"
                      stroke="var(--text-muted)" stroke-width="1"/>
                <text x="${M.left - 8}" y="${y + 3.5}" text-anchor="end"
                      fill="var(--text-muted)" font-size="10">${g.toFixed(2)}</text>
            `);
        }

        // Padding zones (hatched rectangles)
        if (info.scanTop != null && info.edgeTop != null) {
            const padTopY1 = yScale(info.scanTop);
            const padTopY2 = yScale(info.edgeTop);
            svgParts.push(`
                <rect x="${cx - rx - 10}" y="${padTopY1}" width="${rx * 2 + 20}" height="${padTopY2 - padTopY1}"
                      fill="url(#pad-hatch)" rx="4" opacity="0.5"/>
                <text x="${cx + rx + 16}" y="${(padTopY1 + padTopY2) / 2 + 3}"
                      fill="var(--text-muted)" font-size="9" font-style="italic">padding</text>
            `);
        }
        if (info.scanBottom != null && info.edgeBottom != null) {
            const padBotY1 = yScale(info.edgeBottom);
            const padBotY2 = yScale(info.scanBottom);
            svgParts.push(`
                <rect x="${cx - rx - 10}" y="${padBotY1}" width="${rx * 2 + 20}" height="${padBotY2 - padBotY1}"
                      fill="url(#pad-hatch)" rx="4" opacity="0.5"/>
                <text x="${cx + rx + 16}" y="${(padBotY1 + padBotY2) / 2 + 3}"
                      fill="var(--text-muted)" font-size="9" font-style="italic">padding</text>
            `);
        }

        // Scan boundary lines (dotted) with floating label chip
        const drawScanBoundary = (galvo, label) => {
            if (galvo == null) return;
            const y = yScale(galvo);
            svgParts.push(`
                <line x1="${M.left + 4}" y1="${y}" x2="${M.left + plotW}" y2="${y}"
                      stroke="var(--text-muted)" stroke-width="0.7"
                      stroke-dasharray="2 4" opacity="0.4"/>
                <rect x="${M.left + plotW + 2}" y="${y - 7}" width="44" height="13" rx="3"
                      fill="var(--bg-main)" stroke="var(--border)" stroke-width="0.6" opacity="0.9"/>
                <text x="${M.left + plotW + 24}" y="${y + 3}" text-anchor="middle"
                      fill="var(--text-muted)" font-size="9" letter-spacing="0.4">${label}</text>
            `);
        };
        drawScanBoundary(info.scanTop, 'scan top');
        drawScanBoundary(info.scanBottom, 'scan bot');

        // Embryo ellipse - layered for depth: halo, body, rim
        if (ellipseVisible) {
            // Outer soft halo
            svgParts.push(`
                <ellipse cx="${cx}" cy="${cy}" rx="${rx * 1.22}" ry="${ry * 1.32}"
                         fill="url(#embryo-halo)"/>
            `);
            // Main body with subtle drop shadow
            svgParts.push(`
                <ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"
                         fill="url(#embryo-grd)" stroke="var(--accent)"
                         stroke-width="1.4" stroke-opacity="0.55"
                         filter="url(#embryo-shadow)"/>
            `);
            // Inner highlight arc (subtle rim light at the top-left)
            svgParts.push(`
                <path d="M ${cx - rx * 0.75} ${cy - ry * 0.35}
                         A ${rx * 0.85} ${ry * 0.85} 0 0 1 ${cx - rx * 0.1} ${cy - ry * 0.9}"
                      fill="none" stroke="var(--accent)" stroke-width="0.9"
                      stroke-opacity="0.35" stroke-linecap="round"/>
            `);
            // Edge labels with connector line
            svgParts.push(`
                <line x1="${cx + rx - 2}" y1="${eTop}" x2="${cx + rx + 12}" y2="${eTop}"
                      stroke="var(--accent)" stroke-width="0.8" opacity="0.55"/>
                <text x="${cx + rx + 15}" y="${eTop + 3}"
                      fill="var(--accent)" font-size="9" font-weight="600"
                      letter-spacing="0.4">edge top</text>
                <line x1="${cx + rx - 2}" y1="${eBot}" x2="${cx + rx + 12}" y2="${eBot}"
                      stroke="var(--accent)" stroke-width="0.8" opacity="0.55"/>
                <text x="${cx + rx + 15}" y="${eBot + 3}"
                      fill="var(--accent)" font-size="9" font-weight="600"
                      letter-spacing="0.4">edge bot</text>
            `);
        }

        // Edge detection slice lines (inside/near embryo)
        edgeData.forEach((d, i) => {
            const y = yScale(d.galvo);
            const normalizedY = ellipseVisible ? (y - cy) / ry : 0;
            let x1, x2;

            if (ellipseVisible && Math.abs(normalizedY) <= 1) {
                const xHalf = rx * Math.sqrt(1 - normalizedY * normalizedY);
                x1 = cx - xHalf;
                x2 = cx + xHalf;
            } else {
                // Outside ellipse or no ellipse
                x1 = cx - rx * 0.6;
                x2 = cx + rx * 0.6;
            }

            const color = sliceColor(d);
            const isFocusTop = info.focusTopGalvo != null &&
                Math.abs(d.galvo - info.focusTopGalvo) < 0.01;
            const isFocusBot = info.focusBotGalvo != null &&
                Math.abs(d.galvo - info.focusBotGalvo) < 0.01;
            const isFocus = isFocusTop || isFocusBot;
            const isSelected = i === this.selectedSliceIndex;
            const sw = isFocus ? 2.5 : isSelected ? 2 : 1.2;

            svgParts.push(`
                <line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}"
                      stroke="${isSelected ? 'var(--accent)' : color}"
                      stroke-width="${sw}" stroke-linecap="round"
                      opacity="${isSelected ? 1 : 0.6}"
                      class="cal-slice-line ${isFocus ? 'focus-pos' : ''} ${isSelected ? 'selected' : ''}"
                      data-index="${i}" data-type="edge"
                      style="cursor:pointer"/>
            `);

            // Label on right: feature score or visibility (only on the outermost
            // slices + peak so the profile doesn't get cluttered by per-slice noise).
            // Always show the top-scoring slice and any "no embryo" markers.
            const showFeatureScore = d.feature_score > 0 && (
                d.feature_score >= 5 ||
                i === 0 ||
                i === edgeData.length - 1
            );
            if (showFeatureScore) {
                svgParts.push(`
                    <text x="${x2 + 8}" y="${y + 3}"
                          fill="${color}" font-size="8.5" font-weight="500"
                          opacity="0.75" letter-spacing="0.2">${d.feature_score}/10</text>
                `);
            } else if (!d.visible) {
                svgParts.push(`
                    <text x="${x2 + 8}" y="${y + 3}"
                          fill="var(--text-muted)" font-size="8.5"
                          font-style="italic" opacity="0.55">no embryo</text>
                `);
            }

            // Focus position marker (only when an edge slice happens to
            // coincide with the calibration position - the reliable focus
            // lines are drawn independently after this loop).
            if (isFocus) {
                const label = isFocusTop ? 'TOP FOCUS' : 'BOT FOCUS';
                svgParts.push(`
                    <polygon points="${x1 - 4},${y} ${x1 - 11},${y - 4.5} ${x1 - 11},${y + 4.5}"
                             fill="var(--accent-green)"/>
                    <text x="${x1 - 14}" y="${y + 3.5}" text-anchor="end"
                          fill="var(--accent-green)" font-size="9" font-weight="600">${label}</text>
                `);
            }
        });

        // Dedicated focus position lines. These always render at the actual
        // calib_top / calib_bottom galvo positions from the calibration
        // summary metadata, regardless of whether an edge detection slice
        // happens to coincide with them.
        const drawFocusLine = (galvo, label) => {
            if (galvo == null) return;
            const y = yScale(galvo);
            // Chord width: use ellipse intersection if inside, else default
            let x1, x2;
            if (ellipseVisible) {
                const normalizedY = (y - cy) / ry;
                if (Math.abs(normalizedY) <= 1) {
                    const xHalf = rx * Math.sqrt(1 - normalizedY * normalizedY);
                    x1 = cx - xHalf;
                    x2 = cx + xHalf;
                } else {
                    x1 = cx - rx * 0.6;
                    x2 = cx + rx * 0.6;
                }
            } else {
                x1 = cx - rx * 0.6;
                x2 = cx + rx * 0.6;
            }
            // Label chip (pill shape) on the left side
            const chipW = 54;
            const chipH = 14;
            const chipX = x1 - chipW - 8;
            svgParts.push(`
                <line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}"
                      stroke="var(--accent-green)" stroke-width="2.4"
                      stroke-linecap="round" opacity="0.98"
                      filter="url(#focus-glow)" class="cal-focus-line"/>
                <circle cx="${x1}" cy="${y}" r="2.5" fill="var(--accent-green)"/>
                <circle cx="${x2}" cy="${y}" r="2.5" fill="var(--accent-green)"/>
                <rect x="${chipX}" y="${y - chipH / 2}" width="${chipW}" height="${chipH}" rx="${chipH / 2}"
                      fill="var(--accent-green)" opacity="0.92"/>
                <text x="${chipX + chipW / 2}" y="${y + 3.5}" text-anchor="middle"
                      fill="#0a1419" font-size="9" font-weight="700"
                      letter-spacing="0.6">${label}</text>
            `);
        };
        drawFocusLine(info.focusTopGalvo, 'TOP FOCUS');
        drawFocusLine(info.focusBotGalvo, 'BOT FOCUS');

        return `
            <svg viewBox="0 0 ${W} ${H}" class="cal-profile-svg"
                 xmlns="http://www.w3.org/2000/svg"
                 onclick="CalibrationProfileView._handleSVGClick(event)">
                ${svgParts.join('')}
            </svg>
        `;
    },

    _renderMetricsStrip(info) {
        const slope = info.slope != null ? `${info.slope.toFixed(1)}` : '—';
        const slopeUnit = info.slope != null ? 'µm/deg' : '';
        const offset = info.offset != null ? `${info.offset.toFixed(1)}` : '—';
        const offsetUnit = info.offset != null ? 'µm' : '';
        const r2Top = info.rSquaredTop != null ? info.rSquaredTop.toFixed(3) : '—';
        const r2Bot = info.rSquaredBot != null ? info.rSquaredBot.toFixed(3) : '—';
        const coverage = (info.coverage != null && info.slope != null)
            ? `${(info.coverage * info.slope).toFixed(0)}` : (info.coverage != null ? info.coverage.toFixed(2) : '—');
        const coverageUnit = (info.coverage != null && info.slope != null) ? 'µm' : (info.coverage != null ? 'deg' : '');
        const scanRange = (info.scanTop != null && info.scanBottom != null)
            ? `${info.scanTop.toFixed(2)}…${info.scanBottom.toFixed(2)}` : '—';
        const scanRangeUnit = (info.scanTop != null && info.scanBottom != null) ? 'deg' : '';
        const padding = (info.bufferDeg != null && info.slope != null)
            ? `${(info.bufferDeg * info.slope).toFixed(0)}` : (info.bufferDeg != null ? info.bufferDeg.toFixed(2) : '—');
        const paddingUnit = (info.bufferDeg != null && info.slope != null) ? 'µm' : (info.bufferDeg != null ? 'deg' : '');

        const metric = (label, value, unit) => `
            <div class="cal-metric">
                <span class="cal-metric-label">${label}</span>
                <span class="cal-metric-value">${value}${unit ? `<span class="cal-metric-unit">${unit}</span>` : ''}</span>
            </div>
        `;

        return `
            <div class="cal-metrics-strip">
                ${metric('Slope', slope, slopeUnit)}
                ${metric('Offset', offset, offsetUnit)}
                ${metric('R² top', r2Top, '')}
                ${metric('R² bot', r2Bot, '')}
                ${metric('Coverage', coverage, coverageUnit)}
                ${metric('Scan range', scanRange, scanRangeUnit)}
                ${metric('Padding', padding, paddingUnit)}
                ${this._renderSpimIndicator()}
            </div>
        `;
    },

    _renderSliceDetail(sliceItem) {
        const img = sliceItem.image;
        const m = img.metadata || {};
        return `
            <div class="cal-detail-card">
                <div class="cal-detail-card-title">${img.data_type}</div>
                <img class="cal-detail-img" src="data:image/png;base64,${img.base64_png}"
                     alt="${img.data_type}"/>
                <div class="cal-detail-meta">
                    ${m.galvo != null ? `
                        <div class="cal-meta-item">
                            <span class="cal-meta-label">Galvo</span>
                            <span class="cal-meta-value">${m.galvo.toFixed(3)} deg</span>
                        </div>` : ''}
                    ${m.piezo != null ? `
                        <div class="cal-meta-item">
                            <span class="cal-meta-label">Piezo</span>
                            <span class="cal-meta-value">${m.piezo.toFixed(1)} um</span>
                        </div>` : ''}
                    ${m.visible != null ? `
                        <div class="cal-meta-item">
                            <span class="cal-meta-label">Visible</span>
                            <span class="cal-meta-value">${m.visible ? 'Yes' : 'No'}</span>
                        </div>` : ''}
                    ${m.feature_score ? `
                        <div class="cal-meta-item">
                            <span class="cal-meta-label">Feature</span>
                            <span class="cal-meta-value">${m.feature_score}/10</span>
                        </div>` : ''}
                </div>
            </div>
        `;
    },

    _renderFallback(allImages, focusPlots, summaries) {
        // No galvo position data — show focus plots and summaries prominently
        let html = '<div class="cal-profile-fallback">';
        html += '<div class="cal-profile-fallback-msg">No galvo position data available for profile view</div>';

        if (focusPlots.length > 0 || summaries.length > 0) {
            html += '<div class="cal-profile-fallback-grid">';
            [...summaries, ...focusPlots].slice(0, 4).forEach(img => {
                html += `
                    <div class="cal-detail-card">
                        <div class="cal-detail-card-title">${img.data_type}</div>
                        <img class="cal-detail-img" src="data:image/png;base64,${img.base64_png}"
                             alt="${img.data_type}"/>
                    </div>
                `;
            });
            html += '</div>';
        }

        html += '</div>';
        return html;
    },

    _handleSVGClick(event) {
        const line = event.target.closest('.cal-slice-line');
        if (!line) return;

        const index = parseInt(line.dataset.index);
        const type = line.dataset.type;

        this.selectedSliceIndex = index;

        // Get the corresponding data
        const embryoId = CalibrationManager.selectedEmbryoId;
        const images = state.calibration.filter(img =>
            (img.metadata?.embryo_id || 'General') === embryoId
        );
        const edgeData = images
            .filter(img => img.data_type === 'edge_detection' && img.metadata?.galvo != null)
            .sort((a, b) => a.metadata.galvo - b.metadata.galvo);

        if (edgeData[index]) {
            const sliceItem = {
                image: edgeData[index],
                galvo: edgeData[index].metadata.galvo,
                feature_score: edgeData[index].metadata.feature_score || edgeData[index].metadata.vision_score || 0
            };

            // Update detail panel
            const detail = document.getElementById('cal-profile-detail');
            if (detail) {
                detail.innerHTML = this._renderSliceDetail(sliceItem);
            }

            // Update selected state on SVG lines
            document.querySelectorAll('.cal-slice-line').forEach(l => l.classList.remove('selected'));
            line.classList.add('selected');
        }
    }
};


// ==========================================
// CalibrationManager - Two-column layout with Profile/Gallery views
// ==========================================

const CalibrationManager = {
    selectedEmbryoId: null,
    currentView: 'profile',  // 'profile' | 'gallery'
    _lastImageCount: 0,

    render() {
        this.renderSidebar();
        this.renderPanel();
        this._lastImageCount = state.calibration.length;
    },

    switchView(viewName) {
        if (!['profile', 'gallery'].includes(viewName)) return;
        this.currentView = viewName;
        CalibrationProfileView.selectedSliceIndex = null;
        this.renderPanel();
    },

    handleNewImage(newImage) {
        const embryoId = newImage.metadata?.embryo_id || 'General';

        const updated = this._updateSidebarCount(embryoId);
        if (!updated) {
            this.renderSidebar();
        }

        if (this.selectedEmbryoId === embryoId) {
            if (this.currentView === 'profile') {
                this.renderPanel();
                this._animateNewSlice(newImage);
            } else {
                this._prependImageToPanel(newImage);
            }
        }

        if (Lightbox.isOpen && Lightbox.source === 'calibration' && this.selectedEmbryoId === embryoId) {
            const images = state.calibration.filter(img =>
                (img.metadata?.embryo_id || 'General') === this.selectedEmbryoId
            );
            const displayList = images.slice(-50).reverse();
            Lightbox.updateImageList(displayList);
        }

        this._lastImageCount = state.calibration.length;
        return true;
    },

    _animateNewSlice(newImage) {
        if (newImage.metadata?.galvo == null) return;
        // Find the newest line and animate it
        requestAnimationFrame(() => {
            const lines = document.querySelectorAll('.cal-slice-line');
            const lastLine = lines[lines.length - 1];
            if (lastLine) {
                lastLine.style.opacity = '0';
                requestAnimationFrame(() => {
                    lastLine.style.transition = 'opacity 0.4s ease-out, filter 0.4s ease-out';
                    lastLine.style.opacity = '0.75';
                    lastLine.style.filter = 'drop-shadow(0 0 6px var(--accent-green))';
                    setTimeout(() => {
                        lastLine.style.transition = 'filter 0.6s ease-in';
                        lastLine.style.filter = 'none';
                    }, 800);
                });
            }
        });
    },

    _updateSidebarCount(embryoId) {
        const cardsContainer = document.getElementById('calibration-embryo-cards');
        if (!cardsContainer) return false;

        const cards = cardsContainer.querySelectorAll('.calibration-embryo-card');
        for (const card of cards) {
            const nameEl = card.querySelector('.card-name');
            if (nameEl && nameEl.textContent === embryoId) {
                const countEl = card.querySelector('.card-count');
                if (countEl) {
                    const count = state.calibration.filter(img =>
                        (img.metadata?.embryo_id || 'General') === embryoId
                    ).length;
                    countEl.textContent = `${count} image${count !== 1 ? 's' : ''}`;
                    return true;
                }
            }
        }
        return false;
    },

    _prependImageToPanel(newImage) {
        const grid = document.querySelector('#calibration-panel .calibration-image-grid');
        if (!grid) {
            this.renderPanel();
            return;
        }

        const div = document.createElement('div');
        div.className = 'gallery-item';
        div.onclick = () => CalibrationManager.openLightbox(0);
        div.innerHTML = `
            <img class="gallery-img" src="data:image/png;base64,${newImage.base64_png}" alt="${newImage.data_type}">
            <div class="gallery-info">
                <div class="gallery-type">${newImage.data_type}</div>
                <div class="gallery-meta">${formatMeta(newImage.metadata)}</div>
            </div>
        `;

        grid.insertBefore(div, grid.firstChild);

        const items = grid.querySelectorAll('.gallery-item');
        items.forEach((item, idx) => {
            item.onclick = () => CalibrationManager.openLightbox(idx);
        });

        while (grid.children.length > 50) {
            grid.removeChild(grid.lastChild);
        }
    },

    renderSidebar() {
        const cardsContainer = document.getElementById('calibration-embryo-cards');
        if (!cardsContainer) return;

        const grouped = {};
        state.calibration.forEach(img => {
            const eid = img.metadata?.embryo_id || 'General';
            if (!grouped[eid]) grouped[eid] = [];
            grouped[eid].push(img);
        });

        const sortedKeys = Object.keys(grouped).sort((a, b) => {
            if (a === 'General') return 1;
            if (b === 'General') return -1;
            return a.localeCompare(b);
        });

        if (sortedKeys.length === 0) {
            cardsContainer.innerHTML = '<div class="empty-state-small">No calibration images yet</div>';
        } else {
            cardsContainer.innerHTML = sortedKeys.map(embryoId => {
                const imgs = grouped[embryoId];
                const isSelected = this.selectedEmbryoId === embryoId;
                const safeId = embryoId.replace(/'/g, "\\'");

                // Count types for summary
                const edgeCount = imgs.filter(i => i.data_type === 'edge_detection').length;
                const sweepCount = imgs.filter(i =>
                    ['focus_sweep', 'focus_snap', 'focus_coarse'].includes(i.data_type)
                ).length;
                const hasPlot = imgs.some(i => ['focus_plot', 'focus_curve', 'calibration_summary'].includes(i.data_type));

                return `
                    <div class="calibration-embryo-card ${isSelected ? 'selected' : ''}"
                         onclick="CalibrationManager.selectEmbryo('${safeId}')">
                        <div class="card-name">${embryoId}</div>
                        <div class="card-count">${imgs.length} image${imgs.length !== 1 ? 's' : ''}</div>
                        <div class="card-types">
                            ${edgeCount ? `<span class="card-type-badge">edge:${edgeCount}</span>` : ''}
                            ${sweepCount ? `<span class="card-type-badge">sweep:${sweepCount}</span>` : ''}
                            ${hasPlot ? '<span class="card-type-badge done">cal done</span>' : ''}
                        </div>
                    </div>
                `;
            }).join('');

            if (!this.selectedEmbryoId && sortedKeys.length > 0) {
                this.selectedEmbryoId = sortedKeys[0];
                this.render();
                return;
            }
        }
    },

    renderPanel() {
        const panel = document.getElementById('calibration-panel');
        if (!panel) return;

        if (!this.selectedEmbryoId) {
            panel.innerHTML = `
                <div class="calibration-empty">
                    <div class="calibration-empty-icon">&#x1F4F7;</div>
                    <div class="calibration-empty-text">Select an embryo to view calibration images</div>
                </div>
            `;
            return;
        }

        const images = state.calibration.filter(img =>
            (img.metadata?.embryo_id || 'General') === this.selectedEmbryoId
        );

        if (images.length === 0) {
            panel.innerHTML = `
                <div class="calibration-empty">
                    <div class="calibration-empty-icon">&#x1F4F7;</div>
                    <div class="calibration-empty-text">No images for ${this.selectedEmbryoId}</div>
                </div>
            `;
            return;
        }

        // Header (view switcher is now in the static calibration header bar)
        const headerHtml = `
            <div class="calibration-panel-header">
                <span>${this.selectedEmbryoId}</span>
            </div>
        `;
        updateViewButtons('calibration-view-switcher', this.currentView);

        if (this.currentView === 'profile') {
            panel.innerHTML = headerHtml + CalibrationProfileView.render(this.selectedEmbryoId);
            // Re-apply the in-memory SPIM frame for this embryo: the
            // innerHTML reset above replaced the preview's fresh empty
            // DOM, so the cached frame needs to be painted back.
            if (typeof SpimLivePreview !== 'undefined') {
                SpimLivePreview.applyForEmbryo(this.selectedEmbryoId);
            }
        } else {
            panel.innerHTML = headerHtml + this._renderGalleryContent(images);
        }
    },

    _renderGalleryContent(images) {
        const displayList = images.slice(-50).reverse();

        // Extract best_piezo from focus_plot metadata to mark best-focus sweep images
        const bestPiezoByGalvo = {};
        images.forEach(img => {
            if ((img.data_type === 'focus_plot' || img.data_type === 'focus_curve') && img.metadata) {
                const name = img.metadata.galvo_name;
                if (name && img.metadata.best_piezo != null) {
                    bestPiezoByGalvo[name] = img.metadata.best_piezo;
                }
            }
        });

        let segments3dHtml = '';
        if (state.volumes3d.length > 0) {
            segments3dHtml = `
                <div class="calibration-3d-section">
                    <div class="calibration-section-header">3D Segmentations</div>
                    <div class="calibration-3d-grid">
                        ${state.volumes3d.slice(-10).reverse().map(vol => `
                            <div class="calibration-3d-card" onclick="show3DVolume('${vol.uid}')">
                                <div class="card-cells">${vol.num_cells} cells</div>
                                <div class="card-slices">${vol.num_slices} slices</div>
                                <div class="card-uid">${vol.uid.slice(0, 10)}...</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        return `
            <div class="calibration-image-grid">
                ${displayList.map((img, idx) => {
                    // Mark sweep images closest to best_piezo as best-focus
                    let isBestFocus = false;
                    const m = img.metadata || {};
                    if (m.piezo != null && m.galvo_name && bestPiezoByGalvo[m.galvo_name] != null) {
                        const best = bestPiezoByGalvo[m.galvo_name];
                        // Match within 0.5 µm (dense sweep step size)
                        isBestFocus = Math.abs(m.piezo - best) < 0.5;
                    }
                    return `
                    <div class="gallery-item${isBestFocus ? ' best-focus' : ''}" onclick="CalibrationManager.openLightbox(${idx})">
                        <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
                        <div class="gallery-info">
                            <div class="gallery-type">${img.data_type}${isBestFocus ? ' ★' : ''}</div>
                            <div class="gallery-meta">${formatMeta(img.metadata)}</div>
                        </div>
                    </div>`;
                }).join('')}
            </div>
            ${segments3dHtml}
        `;
    },

    selectEmbryo(embryoId) {
        this.selectedEmbryoId = embryoId;
        CalibrationProfileView.selectedSliceIndex = null;
        this.render();
    },

    openLightbox(index) {
        const images = state.calibration.filter(img =>
            (img.metadata?.embryo_id || 'General') === this.selectedEmbryoId
        );
        const displayList = images.slice(-50).reverse();
        Lightbox.open(displayList, index, 'calibration');
    }
};

// Init calibration view switcher (uses shared utility from utils.js)
// Defer calibration view-switcher init to DOMContentLoaded (needs DOM + utils.js ready)
document.addEventListener('DOMContentLoaded', () => {
    if (typeof initViewSwitcher === 'function') {
        initViewSwitcher('calibration-view-switcher', (view) => CalibrationManager.switchView(view), {
            views: ['profile', 'gallery'],
            guard: () => typeof state !== 'undefined' && state.tab === TABS.CALIBRATION
        });
    }
});

// ==========================================
// SPIM live preview (Calibration tab)
// ==========================================
// Subscribes to IMAGE_RECEIVED events and surfaces the latest SPIM-camera
// frame inside the Profile view, complementing the score-vs-piezo dots on
// the SVG. Keeps last frame per embryo in memory, so:
//   - sidebar switches show that embryo's last sighting immediately
//   - unrelated profile re-renders don't blank the preview
//   - live framing during a sweep + "last seen" framing when idle
// Always visible while the Profile view is up; shows an "awaiting
// calibration" placeholder until the first frame for the selected
// embryo arrives.
const SpimLivePreview = {
    // Frame types the calibration loop pushes via agent.push_viz as raw
    // camera captures (distinct from derived plots/summaries).
    //   edge_detection — Phase 1, embryo Z-extent sweep
    //   focus_sweep    — Phase 3, adaptive focus sweep
    //   focus_snap     — single focus probe
    //   focus_coarse   — coarse focus probe
    _ACCEPTED: new Set([
        'edge_detection',
        'focus_sweep',
        'focus_snap',
        'focus_coarse',
    ]),

    // {[embryo_id]: {base64_png, metadata, data_type, ts}}
    _latestByEmbryo: {},

    init() {
        if (typeof ClientEventBus === 'undefined') return;
        ClientEventBus.on('IMAGE_RECEIVED', (data) => this.handleImage(data));
    },

    handleImage(data) {
        if (!data || !data.base64_png) return;
        if (!this._ACCEPTED.has(data.data_type)) return;
        const md = data.metadata || {};
        const embId = md.embryo_id;
        if (!embId) return;

        // Replace any older entry for this embryo — only the most recent
        // capture is kept (no growth, no list).
        this._latestByEmbryo[embId] = {
            base64_png: data.base64_png,
            metadata: md,
            data_type: data.data_type,
            ts: Date.now(),
        };

        // If this frame is for the currently-selected embryo, paint
        // immediately. Otherwise it sits in memory and is applied next
        // time the operator switches to that embryo.
        const selected = (typeof CalibrationManager !== 'undefined')
            ? CalibrationManager.selectedEmbryoId : null;
        if (selected === embId) this.applyForEmbryo(embId);
    },

    /** Apply the most-recent frame for ``embryoId`` to the DOM. Called
     * after every CalibrationManager.renderPanel() so the in-memory
     * state survives the innerHTML reset that re-rendering causes. */
    applyForEmbryo(embryoId) {
        const img = document.getElementById('cal-spim-img');
        const placeholder = document.getElementById('cal-spim-placeholder');
        const metaEl = document.getElementById('cal-spim-meta');
        const led = document.getElementById('cal-spim-led');

        const latest = embryoId ? this._latestByEmbryo[embryoId] : null;

        if (img) {
            if (latest) {
                img.src = `data:image/png;base64,${latest.base64_png}`;
                img.classList.add('has-frame');
                if (placeholder) placeholder.hidden = true;
                if (metaEl) metaEl.textContent = this._formatMeta(latest);
                if (led) led.classList.remove('idle');
            } else {
                img.removeAttribute('src');
                img.classList.remove('has-frame');
                if (placeholder) placeholder.hidden = false;
                if (metaEl) metaEl.textContent = '—';
                if (led) led.classList.add('idle');
            }
        }

        // Mirror into popout if it's open — the popout lives outside the
        // calibration panel's innerHTML reset, so we paint it independently.
        if (typeof SpimPopout !== 'undefined') {
            SpimPopout.paint(latest ? {
                base64_png: latest.base64_png,
                meta: this._formatMeta(latest),
                embryoId,
            } : null);
        }
    },

    _formatMeta(latest) {
        const md = latest.metadata || {};
        const parts = [];
        if (latest.data_type === 'edge_detection') parts.push('edge');
        else if (md.sweep) parts.push(String(md.sweep));
        if (md.galvo_name) parts.push(String(md.galvo_name));
        if (md.piezo !== undefined && md.piezo !== null) {
            const n = Number(md.piezo);
            if (Number.isFinite(n)) parts.push(`p ${n.toFixed(1)}µm`);
        }
        if (md.score !== undefined && md.score !== null) {
            const s = Number(md.score);
            if (Number.isFinite(s)) parts.push(`s ${s.toExponential(1)}`);
        }
        // Age hint if the last frame is older than 10 s — distinguishes
        // "live sweep happening now" from "last sweep was a while ago."
        const age = (Date.now() - latest.ts) / 1000;
        if (age > 10) {
            parts.push(age < 60
                ? `${Math.round(age)}s ago`
                : `${Math.round(age / 60)}m ago`);
        }
        return parts.join('  ·  ') || '—';
    },
};

document.addEventListener('DOMContentLoaded', () => SpimLivePreview.init());

// ==========================================
// SPIM live popout (floating draggable window)
// ==========================================
// Lazy-built floating window that mirrors SpimLivePreview at a larger
// size. Draggable via the header bar, resizable from the bottom-right
// corner. Position and size persist in localStorage so the window
// re-opens where the operator last left it. Closes on Escape.
const SpimPopout = {
    _STORAGE_KEY: 'gently.spimPopout.v1',
    _root: null,
    _isOpen: false,

    _ensureBuilt() {
        if (this._root) return this._root;

        const el = document.createElement('div');
        el.className = 'cal-spim-popout';
        el.id = 'cal-spim-popout';
        el.hidden = true;
        el.innerHTML = `
            <div class="cal-spim-popout-header" id="cal-spim-popout-header">
                <span class="cal-spim-popout-led idle" id="cal-spim-popout-led"></span>
                <span class="cal-spim-popout-title">SPIM Live</span>
                <span class="cal-spim-popout-embryo" id="cal-spim-popout-embryo"></span>
                <span class="cal-spim-popout-spacer"></span>
                <button type="button" class="cal-spim-popout-close" id="cal-spim-popout-close"
                        title="Close (Esc)" aria-label="Close popout">×</button>
            </div>
            <div class="cal-spim-popout-body">
                <img class="cal-spim-popout-img" id="cal-spim-popout-img" alt="">
                <div class="cal-spim-popout-placeholder" id="cal-spim-popout-placeholder">
                    Awaiting SPIM frame…
                </div>
            </div>
            <div class="cal-spim-popout-footer">
                <span class="cal-spim-popout-meta" id="cal-spim-popout-meta">—</span>
            </div>
        `;
        document.body.appendChild(el);
        this._root = el;

        // Restore persisted geometry
        const saved = this._loadGeometry();
        if (saved) {
            el.style.left = `${saved.left}px`;
            el.style.top = `${saved.top}px`;
            el.style.width = `${saved.width}px`;
            el.style.height = `${saved.height}px`;
        }

        el.querySelector('#cal-spim-popout-close').addEventListener('click', () => this.close());
        this._wireDrag(el);
        this._wireResizeObserver(el);

        return el;
    },

    open() {
        const el = this._ensureBuilt();
        if (this._isOpen) return;
        el.hidden = false;
        this._isOpen = true;

        // Clamp into viewport in case window was resized while popout was hidden
        this._clampIntoViewport(el);

        // Paint current frame for the selected embryo
        const selected = (typeof CalibrationManager !== 'undefined')
            ? CalibrationManager.selectedEmbryoId : null;
        if (selected && typeof SpimLivePreview !== 'undefined') {
            const latest = SpimLivePreview._latestByEmbryo[selected];
            this.paint(latest ? {
                base64_png: latest.base64_png,
                meta: SpimLivePreview._formatMeta(latest),
                embryoId: selected,
            } : null);
        } else {
            this.paint(null);
        }

        document.addEventListener('keydown', this._onKey);
    },

    close() {
        if (!this._root || !this._isOpen) return;
        this._root.hidden = true;
        this._isOpen = false;
        document.removeEventListener('keydown', this._onKey);
    },

    toggle() {
        this._isOpen ? this.close() : this.open();
    },

    /** Called by SpimLivePreview whenever the current embryo's latest
     * frame changes. Frame is {base64_png, meta, embryoId} or null. */
    paint(frame) {
        if (!this._root || !this._isOpen) return;
        const img = this._root.querySelector('#cal-spim-popout-img');
        const placeholder = this._root.querySelector('#cal-spim-popout-placeholder');
        const meta = this._root.querySelector('#cal-spim-popout-meta');
        const embryoEl = this._root.querySelector('#cal-spim-popout-embryo');
        const led = this._root.querySelector('#cal-spim-popout-led');

        if (frame) {
            img.src = `data:image/png;base64,${frame.base64_png}`;
            img.classList.add('has-frame');
            placeholder.hidden = true;
            meta.textContent = frame.meta || '—';
            embryoEl.textContent = frame.embryoId || '';
            led.classList.remove('idle');
        } else {
            img.removeAttribute('src');
            img.classList.remove('has-frame');
            placeholder.hidden = false;
            meta.textContent = '—';
            embryoEl.textContent = '';
            led.classList.add('idle');
        }
    },

    _onKey: (e) => {
        if (e.key === 'Escape') SpimPopout.close();
    },

    _wireDrag(el) {
        const header = el.querySelector('#cal-spim-popout-header');
        let dragging = false;
        let startX = 0, startY = 0, startLeft = 0, startTop = 0;

        header.addEventListener('pointerdown', (e) => {
            // Don't start drag on the close button
            if (e.target.closest('.cal-spim-popout-close')) return;
            dragging = true;
            const rect = el.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            startLeft = rect.left;
            startTop = rect.top;
            // Switch to absolute positioning if currently default
            el.style.left = `${startLeft}px`;
            el.style.top = `${startTop}px`;
            el.style.right = 'auto';
            el.style.bottom = 'auto';
            header.setPointerCapture(e.pointerId);
            el.classList.add('dragging');
        });

        header.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            let nextLeft = startLeft + dx;
            let nextTop = startTop + dy;
            // Keep at least 40px of header on-screen
            const w = el.offsetWidth;
            const h = el.offsetHeight;
            nextLeft = Math.max(-(w - 80), Math.min(window.innerWidth - 80, nextLeft));
            nextTop = Math.max(0, Math.min(window.innerHeight - 40, nextTop));
            el.style.left = `${nextLeft}px`;
            el.style.top = `${nextTop}px`;
        });

        const endDrag = (e) => {
            if (!dragging) return;
            dragging = false;
            el.classList.remove('dragging');
            try { header.releasePointerCapture(e.pointerId); } catch (_) {}
            this._saveGeometry(el);
        };
        header.addEventListener('pointerup', endDrag);
        header.addEventListener('pointercancel', endDrag);
    },

    _wireResizeObserver(el) {
        if (typeof ResizeObserver === 'undefined') return;
        let saveTimer = null;
        const ro = new ResizeObserver(() => {
            if (!this._isOpen) return;
            clearTimeout(saveTimer);
            saveTimer = setTimeout(() => this._saveGeometry(el), 250);
        });
        ro.observe(el);
    },

    _clampIntoViewport(el) {
        const rect = el.getBoundingClientRect();
        if (rect.left + 80 > window.innerWidth || rect.top + 40 > window.innerHeight
            || rect.left < -(rect.width - 80) || rect.top < 0) {
            // Recenter
            const w = Math.min(rect.width || 520, window.innerWidth - 40);
            const h = Math.min(rect.height || 440, window.innerHeight - 40);
            el.style.width = `${w}px`;
            el.style.height = `${h}px`;
            el.style.left = `${Math.max(20, (window.innerWidth - w) / 2)}px`;
            el.style.top = `${Math.max(20, (window.innerHeight - h) / 2)}px`;
        }
    },

    _saveGeometry(el) {
        const rect = el.getBoundingClientRect();
        const data = {
            left: Math.round(rect.left),
            top: Math.round(rect.top),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
        };
        try { localStorage.setItem(this._STORAGE_KEY, JSON.stringify(data)); } catch (_) {}
    },

    _loadGeometry() {
        try {
            const raw = localStorage.getItem(this._STORAGE_KEY);
            if (!raw) return null;
            const data = JSON.parse(raw);
            if (typeof data.left !== 'number') return null;
            return data;
        } catch (_) { return null; }
    },
};

// Legacy wrappers kept for backward compatibility
function renderCalibrationGallery() { CalibrationManager.render(); }

function formatMeta(meta) {
    if (!meta) return '';
    if (meta.score) return `score: ${meta.score.toExponential(2)}`;
    if (meta.focus_score) return `score: ${meta.focus_score.toFixed(2)}`;
    if (meta.piezo != null && meta.galvo != null) return `${meta.galvo.toFixed(2)}° / ${meta.piezo.toFixed(1)}um`;
    if (meta.piezo != null) return `${meta.piezo.toFixed(1)}um`;
    if (meta.galvo != null) return `${meta.galvo.toFixed(3)}°`;
    return '';
}

function filterByEmbryo(list) {
    if (!state.embryoFilter) return list;
    return list.filter(img => img.metadata?.embryo_id === state.embryoFilter);
}

// ==========================================
// GalleryTab — top-level Gallery tab controller
// ==========================================

const GalleryTab = {
    _allItems: [],
    _embryoFilter: '',
    _typeFilter: '',

    async init() {
        const panel = document.getElementById('gallery-tab-body');
        if (!panel) return;

        // Wire filter controls (idempotent)
        const embryoSel = document.getElementById('gallery-embryo-filter');
        const typeSel   = document.getElementById('gallery-type-filter');
        const refreshBtn = document.getElementById('gallery-refresh-btn');
        if (embryoSel && !embryoSel._gtWired) {
            embryoSel._gtWired = true;
            embryoSel.addEventListener('change', () => {
                this._embryoFilter = embryoSel.value;
                this._renderGrid();
            });
        }
        if (typeSel && !typeSel._gtWired) {
            typeSel._gtWired = true;
            typeSel.addEventListener('change', () => {
                this._typeFilter = typeSel.value;
                this._renderGrid();
            });
        }
        if (refreshBtn && !refreshBtn._gtWired) {
            refreshBtn._gtWired = true;
            refreshBtn.addEventListener('click', () => this._load());
        }

        await this._load();
    },

    async _load() {
        const panel = document.getElementById('gallery-tab-body');
        if (!panel) return;
        panel.innerHTML = '<div class="gallery-tab-loading">Loading…</div>';
        try {
            const data = await fetch('/api/snapshots').then(r => r.json());
            const items = data.snapshots || [];
            // Sort newest first
            items.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));
            this._allItems = items;
            this._populateEmbroyFilter(items);
            this._renderGrid();
        } catch (err) {
            panel.innerHTML = '<div class="gallery-tab-error">Failed to load gallery — check connection.</div>';
        }
    },

    _populateEmbroyFilter(items) {
        const sel = document.getElementById('gallery-embryo-filter');
        if (!sel) return;
        const embryos = [...new Set(items.map(i => i.metadata?.embryo_id).filter(Boolean))].sort();
        const cur = sel.value;
        // Rebuild options beyond the "All embryos" placeholder
        while (sel.options.length > 1) sel.remove(1);
        embryos.forEach(eid => {
            const opt = document.createElement('option');
            opt.value = eid;
            opt.textContent = eid;
            sel.appendChild(opt);
        });
        if (cur && embryos.includes(cur)) sel.value = cur;
    },

    _renderGrid() {
        const panel = document.getElementById('gallery-tab-body');
        if (!panel) return;

        let items = this._allItems;
        if (this._embryoFilter) {
            items = items.filter(i => (i.metadata?.embryo_id || '') === this._embryoFilter);
        }
        if (this._typeFilter) {
            items = items.filter(i => i.data_type === this._typeFilter);
        }

        if (items.length === 0) {
            panel.innerHTML = '<div class="gallery-tab-empty">No images match the current filter.</div>';
            return;
        }

        const html = `<div class="gallery-tab-grid">${items.map((img, idx) => this._itemHtml(img, idx)).join('')}</div>`;
        panel.innerHTML = html;

        // Wire click handlers
        panel.querySelectorAll('.gallery-tab-item').forEach(el => {
            el.addEventListener('click', () => {
                const idx = parseInt(el.dataset.idx, 10);
                Lightbox.open(items, idx, 'gallery');
            });
        });
    },

    _itemHtml(img, idx) {
        const embryo = img.metadata?.embryo_id || '';
        const ts = img.timestamp ? img.timestamp.slice(0, 19).replace('T', ' ') : '';
        const typeLabel = img.data_type || 'image';
        const thumb = img.base64_png
            ? `<img class="gallery-tab-thumb" src="data:image/png;base64,${img.base64_png}" alt="${typeLabel}">`
            : `<div class="gallery-tab-thumb-placeholder">${typeLabel}</div>`;
        return `
            <div class="gallery-tab-item" data-idx="${idx}" title="${typeLabel}${embryo ? ' · ' + embryo : ''}">
                ${thumb}
                <div class="gallery-tab-item-info">
                    <div class="gallery-tab-item-type">${typeLabel}</div>
                    ${embryo ? `<div class="gallery-tab-item-embryo">${embryo}</div>` : ''}
                    ${ts ? `<div class="gallery-tab-item-ts">${ts}</div>` : ''}
                </div>
            </div>
        `;
    },
};

// ==========================================
// showGentlyToast — lightweight global toast (volume acquired, etc.)
// ==========================================

/**
 * Show a brief toast notification with an optional action link.
 * @param {string} message - Primary message text
 * @param {string|null} actionLabel - Label for the action button (null = no button)
 * @param {Function|null} actionFn - Callback invoked when the action is clicked
 * @param {number} [duration=6000] - Auto-dismiss delay in ms
 */
function showGentlyToast(message, actionLabel, actionFn, duration = 6000) {
    // Remove any existing gently-toast
    document.querySelectorAll('.gently-toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = 'gently-toast';

    const msgSpan = document.createElement('span');
    msgSpan.className = 'gently-toast-msg';
    msgSpan.textContent = message;
    toast.appendChild(msgSpan);

    if (actionLabel && actionFn) {
        const actionBtn = document.createElement('button');
        actionBtn.className = 'gently-toast-action';
        actionBtn.textContent = actionLabel;
        actionBtn.addEventListener('click', () => {
            actionFn();
            toast.remove();
        });
        toast.appendChild(actionBtn);
    }

    const dismiss = document.createElement('button');
    dismiss.className = 'gently-toast-dismiss';
    dismiss.setAttribute('aria-label', 'Dismiss');
    dismiss.textContent = '×';
    dismiss.addEventListener('click', () => toast.remove());
    toast.appendChild(dismiss);

    document.body.appendChild(toast);
    // Trigger transition
    requestAnimationFrame(() => toast.classList.add('visible'));

    const timer = setTimeout(() => toast.remove(), duration);
    dismiss.addEventListener('click', () => clearTimeout(timer));
}
