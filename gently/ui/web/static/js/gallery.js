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

        return `
            <div class="cal-profile-container">
                <div class="cal-profile-svg-area">
                    ${this._renderSVG(edgeData, sweepData, calibInfo)}
                    ${this._renderResultsCard(calibInfo)}
                </div>
                <div class="cal-profile-detail" id="cal-profile-detail">
                    ${this._renderDetailDefault(focusPlots, summaries, calibInfo)}
                </div>
            </div>
        `;
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

        // Determine galvo range for Y-axis
        let yMin, yMax;
        const allGalvos = edgeData.map(d => d.galvo);
        if (info.scanTop != null && info.scanBottom != null) {
            yMin = info.scanTop - 0.02;
            yMax = info.scanBottom + 0.02;
        } else if (allGalvos.length > 0) {
            const gMin = Math.min(...allGalvos);
            const gMax = Math.max(...allGalvos);
            const pad = (gMax - gMin) * 0.2 || 0.05;
            yMin = gMin - pad;
            yMax = gMax + pad;
        } else {
            yMin = -0.4;
            yMax = 0.4;
        }

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

        // Defs
        svgParts.push(`
            <defs>
                <radialGradient id="embryo-grd" cx="40%" cy="40%">
                    <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.07"/>
                    <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.02"/>
                </radialGradient>
                <pattern id="pad-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <line x1="0" y1="0" x2="0" y2="6" stroke="var(--border)" stroke-width="0.5" opacity="0.4"/>
                </pattern>
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

        // Scan boundary lines (dashed)
        if (info.scanTop != null) {
            const y = yScale(info.scanTop);
            svgParts.push(`
                <line x1="${M.left + 4}" y1="${y}" x2="${M.left + plotW}" y2="${y}"
                      stroke="var(--text-muted)" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.5"/>
                <text x="${M.left + plotW + 4}" y="${y + 3}"
                      fill="var(--text-muted)" font-size="9">scan top</text>
            `);
        }
        if (info.scanBottom != null) {
            const y = yScale(info.scanBottom);
            svgParts.push(`
                <line x1="${M.left + 4}" y1="${y}" x2="${M.left + plotW}" y2="${y}"
                      stroke="var(--text-muted)" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.5"/>
                <text x="${M.left + plotW + 4}" y="${y + 3}"
                      fill="var(--text-muted)" font-size="9">scan bot</text>
            `);
        }

        // Embryo ellipse
        if (ellipseVisible) {
            svgParts.push(`
                <ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"
                         fill="url(#embryo-grd)" stroke="var(--border)"
                         stroke-width="1.2" stroke-dasharray="6 3" opacity="0.7"/>
            `);
            // Edge labels
            svgParts.push(`
                <text x="${cx + rx + 16}" y="${eTop + 3}"
                      fill="var(--accent)" font-size="9" font-weight="500">edge top</text>
                <text x="${cx + rx + 16}" y="${eBot + 3}"
                      fill="var(--accent)" font-size="9" font-weight="500">edge bot</text>
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
                      stroke-width="${sw}" opacity="${isSelected ? 1 : 0.75}"
                      class="cal-slice-line ${isFocus ? 'focus-pos' : ''} ${isSelected ? 'selected' : ''}"
                      data-index="${i}" data-type="edge"
                      style="cursor:pointer"/>
            `);

            // Label on right: feature score or visibility
            if (d.feature_score > 0) {
                svgParts.push(`
                    <text x="${x2 + 6}" y="${y + 3.5}"
                          fill="${color}" font-size="9" opacity="0.8">${d.feature_score}/10</text>
                `);
            } else if (!d.visible) {
                svgParts.push(`
                    <text x="${x2 + 6}" y="${y + 3.5}"
                          fill="var(--text-muted)" font-size="9" opacity="0.6">no embryo</text>
                `);
            }

            // Focus position marker
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

        // Legend
        const legY = H - 10;
        svgParts.push(`
            <text x="${M.left}" y="${legY}" fill="var(--text-muted)" font-size="9">
                Line color = feature score (dim=low, bright=high)
            </text>
        `);

        return `
            <svg viewBox="0 0 ${W} ${H}" class="cal-profile-svg"
                 xmlns="http://www.w3.org/2000/svg"
                 onclick="CalibrationProfileView._handleSVGClick(event)">
                ${svgParts.join('')}
            </svg>
        `;
    },

    _renderResultsCard(info) {
        const slope = info.slope != null ? `${info.slope.toFixed(1)} um/deg` : '—';
        const offset = info.offset != null ? `${info.offset.toFixed(1)} um` : '—';
        const r2Top = info.rSquaredTop != null ? info.rSquaredTop.toFixed(3) : '—';
        const r2Bot = info.rSquaredBot != null ? info.rSquaredBot.toFixed(3) : '—';
        const coverage = info.coverage != null ? `${info.coverage.toFixed(2)} deg` : '—';
        const coverageUm = (info.coverage != null && info.slope != null)
            ? ` (~${(info.coverage * info.slope).toFixed(0)}um)` : '';
        const scanRange = (info.scanTop != null && info.scanBottom != null)
            ? `${info.scanTop.toFixed(2)} to ${info.scanBottom.toFixed(2)} deg` : '—';
        const padding = info.bufferDeg != null ? `${info.bufferDeg.toFixed(2)} deg` : '—';
        const paddingUm = (info.bufferDeg != null && info.slope != null)
            ? ` (~${(info.bufferDeg * info.slope).toFixed(0)}um)` : '';

        return `
            <div class="cal-results-card">
                <div class="cal-result">
                    <span class="cal-result-label">Slope</span>
                    <span class="cal-result-value">${slope}</span>
                </div>
                <div class="cal-result">
                    <span class="cal-result-label">Offset</span>
                    <span class="cal-result-value">${offset}</span>
                </div>
                <div class="cal-result">
                    <span class="cal-result-label">R² (top/bot)</span>
                    <span class="cal-result-value">${r2Top} / ${r2Bot}</span>
                </div>
                <div class="cal-result">
                    <span class="cal-result-label">Coverage</span>
                    <span class="cal-result-value">${coverage}${coverageUm}</span>
                </div>
                <div class="cal-result">
                    <span class="cal-result-label">Scan Range</span>
                    <span class="cal-result-value">${scanRange}</span>
                </div>
                <div class="cal-result">
                    <span class="cal-result-label">Padding</span>
                    <span class="cal-result-value">${padding}${paddingUm}</span>
                </div>
            </div>
        `;
    },

    _renderDetailDefault(focusPlots, summaries, info) {
        let html = '';

        // Show focus plots
        if (focusPlots.length > 0) {
            const latest = focusPlots.slice(-2).reverse(); // Show up to 2 (top + bottom)
            latest.forEach((img, i) => {
                const label = i === 0 ? 'Focus Curve (latest)' : 'Focus Curve';
                html += `
                    <div class="cal-detail-card">
                        <div class="cal-detail-card-title">${label}</div>
                        <img class="cal-detail-img" src="data:image/png;base64,${img.base64_png}"
                             alt="focus curve" onclick="CalibrationManager.openLightbox(0)"/>
                    </div>
                `;
            });
        }

        // Show calibration summary
        if (summaries.length > 0) {
            const latest = summaries[summaries.length - 1];
            html += `
                <div class="cal-detail-card">
                    <div class="cal-detail-card-title">Calibration Summary</div>
                    <img class="cal-detail-img" src="data:image/png;base64,${latest.base64_png}"
                         alt="calibration summary" onclick="CalibrationManager.openLightbox(0)"/>
                </div>
            `;
        }

        if (!html) {
            html = `
                <div class="cal-detail-empty">
                    <div class="cal-detail-empty-text">Click a slice line to view its image</div>
                </div>
            `;
        }

        return html;
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

        // Header with view switcher
        const headerHtml = `
            <div class="calibration-panel-header">
                <span>${this.selectedEmbryoId}</span>
                <div class="cal-view-switcher">
                    <button class="cal-view-btn ${this.currentView === 'profile' ? 'active' : ''}"
                            data-view="profile" onclick="CalibrationManager.switchView('profile')">Profile</button>
                    <button class="cal-view-btn ${this.currentView === 'gallery' ? 'active' : ''}"
                            data-view="gallery" onclick="CalibrationManager.switchView('gallery')">Gallery</button>
                </div>
            </div>
        `;

        if (this.currentView === 'profile') {
            panel.innerHTML = headerHtml + CalibrationProfileView.render(this.selectedEmbryoId);
        } else {
            panel.innerHTML = headerHtml + this._renderGalleryContent(images);
        }
    },

    _renderGalleryContent(images) {
        const displayList = images.slice(-50).reverse();

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
                ${displayList.map((img, idx) => `
                    <div class="gallery-item" onclick="CalibrationManager.openLightbox(${idx})">
                        <img class="gallery-img" src="data:image/png;base64,${img.base64_png}" alt="${img.data_type}">
                        <div class="gallery-info">
                            <div class="gallery-type">${img.data_type}</div>
                            <div class="gallery-meta">${formatMeta(img.metadata)}</div>
                        </div>
                    </div>
                `).join('')}
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
