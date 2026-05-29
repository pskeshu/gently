/**
 * Events tab functionality for Gently Visualization
 * Displays all EventBus events in a filterable table
 */

// Events state (added to global state in app.js)
// state.allEvents = []
// state.eventTypeFilter = ''
// state.eventSourceFilter = ''
// state.eventSources = new Set()

const MAX_EVENTS = 500;

// Search state
let searchQuery = '';
let searchDebounceTimer = null;

// Cached filtered count — maintained incrementally when adding events,
// recomputed fully only when filters change.
let _filteredCount = 0;
let _filteredCountDirty = true;  // true = needs full recount

// Image UID fields to look for in event data
const IMAGE_UID_FIELDS = ['volume_uid', 'image_uid', 'uid', 'visualization_uid', 'segmentation_uid', 'source_uid', 'mask_uid'];

// Event type categories for badge styling
const CV_EVENT_TYPES = ['CV_TASK_QUEUED', 'CV_TASK_COMPLETED', 'CV_TASK_FAILED', 'CV_AGENT_THINKING', 'CV_RESULT_READY'];
const PERCEPTION_EVENT_TYPES = ['DETECTOR_EVALUATED', 'HATCHING_DETECTED', 'PERCEPTION_COMPLETED', 'STAGE_TRANSITION'];
const ANALYSIS_EVENT_TYPES = ['SEGMENTATION_COMPLETED', 'STAGE_DETECTED', 'CELL_DIVISION_DETECTED', 'LINEAGE_UPDATED', 'ANOMALY_DETECTED'];
const ACQUISITION_EVENT_TYPES = ['VOLUME_ACQUIRED', 'IMAGE_ACQUIRED', 'ACQUISITION_STARTED', 'ACQUISITION_COMPLETED'];
const SESSION_EVENT_TYPES = ['SESSION_STARTED', 'SESSION_ENDED', 'SESSION_SAVED', 'SESSION_RESTORED'];
const ERROR_EVENT_TYPES = ['ERROR_OCCURRED', 'CV_TASK_FAILED', 'ACQUISITION_FAILED'];

function getEventBadgeClass(eventType) {
    if (CV_EVENT_TYPES.includes(eventType)) return 'cv';
    if (PERCEPTION_EVENT_TYPES.includes(eventType)) return 'perception';
    if (ANALYSIS_EVENT_TYPES.includes(eventType)) return 'analysis';
    if (ACQUISITION_EVENT_TYPES.includes(eventType)) return 'acquisition';
    if (SESSION_EVENT_TYPES.includes(eventType)) return 'session';
    if (ERROR_EVENT_TYPES.includes(eventType)) return 'error';
    return 'default';
}

// Log-record helpers --------------------------------------------------
// LOG_RECORD events come from the Python logging bridge. We collapse the
// generic "LOG_RECORD" type into the actual level (DEBUG / INFO / WARN /
// ERROR) so the table is readable -- otherwise every row in a busy
// session reads the same string in the Type column.
function isLogEvent(event) {
    return event && event.event_type === 'LOG_RECORD';
}

function logLevelLabel(d) {
    // levelname is fastest path; fall back to numeric mapping if missing.
    const lvl = (d && (d.level_name || '')).toString().toUpperCase();
    if (lvl) {
        if (lvl === 'WARNING') return 'WARN';
        if (lvl === 'CRITICAL') return 'CRIT';
        return lvl;
    }
    const n = d && Number(d.level);
    if (!isFinite(n)) return 'LOG';
    if (n >= 50) return 'CRIT';
    if (n >= 40) return 'ERROR';
    if (n >= 30) return 'WARN';
    if (n >= 20) return 'INFO';
    return 'DEBUG';
}

function logBadgeClass(d) {
    const label = logLevelLabel(d);
    if (label === 'DEBUG') return 'log-debug';
    if (label === 'INFO')  return 'log-info';
    if (label === 'WARN')  return 'log-warn';
    return 'log-error';  // ERROR / CRIT collapse together
}

// Search helper functions
function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function eventMatchesSearch(event) {
    if (!searchQuery) return true;

    const query = searchQuery.toLowerCase();

    // Search in event type
    if (event.event_type?.toLowerCase().includes(query)) return true;

    // Search in source
    if (event.source?.toLowerCase().includes(query)) return true;

    // Search in data (stringify and search)
    if (event.data) {
        const dataStr = JSON.stringify(event.data).toLowerCase();
        if (dataStr.includes(query)) return true;
    }

    return false;
}

function highlightSearchTerms(text) {
    // Escape first — event keys/values/messages are arbitrary text (perception
    // prose, file paths, agent output) and are inserted via innerHTML by the
    // callers. Escaping here closes the XSS hole at every call site; the
    // injected <mark> tags are the only markup we add.
    const safe = escapeHtml(text == null ? '' : String(text));
    if (!searchQuery) return safe;
    try {
        const regex = new RegExp(`(${escapeRegex(searchQuery)})`, 'gi');
        return safe.replace(regex, '<mark class="search-highlight">$1</mark>');
    } catch (e) {
        return safe;
    }
}

// Extract image UID from event data
function extractImageUid(eventData) {
    if (!eventData || typeof eventData !== 'object') return null;
    for (const field of IMAGE_UID_FIELDS) {
        if (eventData[field] && typeof eventData[field] === 'string') {
            return eventData[field];
        }
    }
    return null;
}

// Find image by UID across all image stores
function findImageByUid(uid) {
    if (!uid) return null;

    // Search in all image stores
    const allImages = [
        ...(state.volumes || []),
        ...(state.calibration || []),
        ...(state.snapshots || [])
    ];

    return allImages.find(img => img.uid === uid);
}

// Show event-linked image in lightbox
function showEventImage(uid) {
    const image = findImageByUid(uid);
    if (!image) {
        console.warn('Image not found for UID:', uid);
        return;
    }

    // Determine which list the image belongs to
    let source = 'snapshots';
    let list = state.snapshots || [];

    if (state.volumes?.find(i => i.uid === uid)) {
        source = 'volumes';
        list = state.volumes;
    } else if (state.calibration?.find(i => i.uid === uid)) {
        source = 'calibration';
        list = state.calibration;
    }

    const index = list.findIndex(i => i.uid === uid);
    if (index >= 0 && typeof Lightbox !== 'undefined') {
        Lightbox.open(list, index, source);
    }
}

function formatEventData(data) {
    if (!data || Object.keys(data).length === 0) return '-';

    // Format key-value pairs nicely
    const parts = [];
    for (const [key, value] of Object.entries(data)) {
        let displayValue = value;
        if (typeof value === 'object') {
            displayValue = JSON.stringify(value).slice(0, 50);
            if (JSON.stringify(value).length > 50) displayValue += '...';
        } else if (typeof value === 'string' && value.length > 40) {
            displayValue = value.slice(0, 40) + '...';
        }

        // Apply search highlighting
        const highlightedKey = highlightSearchTerms(key);
        const highlightedValue = highlightSearchTerms(String(displayValue));

        parts.push(`<span class="event-data-key">${highlightedKey}</span>=<span class="event-data-value">${highlightedValue}</span>`);
    }
    return parts.join(', ');
}

function formatEventTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 3
    }).replace(',', '.');
}

function addEventToTable(event, prepend = true) {
    // Store event
    if (prepend) {
        state.allEvents.unshift(event);
        // Trim to max
        if (state.allEvents.length > MAX_EVENTS) {
            state.allEvents.pop();
        }
    }

    // Track source
    if (event.source && !state.eventSources.has(event.source)) {
        state.eventSources.add(event.source);
        updateSourceFilter();
    }

    // Check filters
    const matchesFilter = (
        (!state.eventTypeFilter || event.event_type === state.eventTypeFilter) &&
        (!state.eventSourceFilter || event.source === state.eventSourceFilter) &&
        eventMatchesSearch(event)
    );

    // Incrementally update filtered count for newly prepended events
    if (prepend && matchesFilter && !_filteredCountDirty) {
        _filteredCount++;
        // If we trimmed an event, we can't know if it was filtered — mark dirty
        if (state.allEvents.length > MAX_EVENTS) {
            _filteredCountDirty = true;
        }
    }

    if (!matchesFilter) {
        // Still update the count display (total changed)
        updateEventsCount();
        return;
    }

    const tbody = document.getElementById('events-tbody');
    if (!tbody) return;

    // Check for linked image
    const imageUid = extractImageUid(event.data);
    const linkedImage = imageUid ? findImageByUid(imageUid) : null;
    const hasImage = !!linkedImage;

    const tr = document.createElement('tr');
    tr.className = prepend ? 'event-row-new' : '';
    if (hasImage) tr.classList.add('has-image');
    tr.dataset.eventId = event.event_id || '';

    if (isLogEvent(event)) {
        // Log rows have a compact, distinctive shape: level badge in the
        // Type column, logger name + message in the Data column. Click to
        // toggle a pre with the full payload (incl. exception trace).
        tr.classList.add('log-row');
        const d = event.data || {};
        const badgeCls = logBadgeClass(d);
        const label = logLevelLabel(d);
        const message = highlightSearchTerms(d.message || '');
        const loggerName = highlightSearchTerms(d.logger || '-');
        const excTag = d.exc_text ? '<span class="log-exc">  ⏎ trace…</span>' : '';
        tr.innerHTML = `
            <td class="col-time">${formatEventTime(event.timestamp)}</td>
            <td class="col-type"><span class="event-type-badge ${badgeCls}">${label}</span></td>
            <td class="col-source"><span class="event-source">${event.source || '-'}</span></td>
            <td class="col-data"><div class="event-data">
                <span class="log-logger">${loggerName}</span><span class="log-message">${message}</span>${excTag}
            </div></td>
        `;
        tr.addEventListener('click', () => {
            const dataDiv = tr.querySelector('.event-data');
            dataDiv.classList.toggle('expanded');
            if (dataDiv.classList.contains('expanded')) {
                const tracePart = d.exc_text
                    ? `\n\n${d.exc_text}` : '';
                dataDiv.innerHTML =
                    `<pre>${d.logger || ''}  ${d.func || ''}:${d.line || ''}\n` +
                    `${(d.message || '')}${tracePart}</pre>`;
            } else {
                dataDiv.innerHTML =
                    `<span class="log-logger">${loggerName}</span>` +
                    `<span class="log-message">${message}</span>${excTag}`;
            }
        });
    } else {
        const badgeClass = getEventBadgeClass(event.event_type);

        // Image indicator icon
        const imageIndicator = hasImage
            ? `<span class="event-image-indicator" title="Has linked image">
                 <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                   <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                   <circle cx="8.5" cy="8.5" r="1.5"></circle>
                   <polyline points="21,15 16,10 5,21"></polyline>
                 </svg>
               </span>`
            : '';

        // Thumbnail preview
        const thumbnailHtml = hasImage
            ? `<img class="event-thumbnail"
                   src="data:image/png;base64,${linkedImage.base64_png}"
                   onclick="event.stopPropagation(); showEventImage('${imageUid}')"
                   title="Click to view image"
                   alt="Event image">`
            : '';

        tr.innerHTML = `
            <td class="col-time">${formatEventTime(event.timestamp)}</td>
            <td class="col-type">${imageIndicator}<span class="event-type-badge ${badgeClass}">${event.event_type}</span></td>
            <td class="col-source"><span class="event-source">${event.source || '-'}</span></td>
            <td class="col-data">${thumbnailHtml}<div class="event-data">${formatEventData(event.data)}</div></td>
        `;

        // Click to expand data
        tr.addEventListener('click', () => {
            const dataDiv = tr.querySelector('.event-data');
            dataDiv.classList.toggle('expanded');
            if (dataDiv.classList.contains('expanded')) {
                dataDiv.innerHTML = `<pre>${JSON.stringify(event.data, null, 2)}</pre>`;
            } else {
                dataDiv.innerHTML = formatEventData(event.data);
            }
        });
    }

    if (prepend) {
        tbody.insertBefore(tr, tbody.firstChild);
        // Trim table rows
        while (tbody.children.length > MAX_EVENTS) {
            tbody.removeChild(tbody.lastChild);
        }
    } else {
        tbody.appendChild(tr);
    }

    updateEventsCount();
}

function renderEventsTable() {
    // Filters changed — mark count dirty so it gets recomputed
    _filteredCountDirty = true;

    const tbody = document.getElementById('events-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';

    const filtered = state.allEvents.filter(e => {
        if (state.eventTypeFilter && e.event_type !== state.eventTypeFilter) return false;
        if (state.eventSourceFilter && e.source !== state.eventSourceFilter) return false;
        if (!eventMatchesSearch(e)) return false;
        return true;
    });

    filtered.forEach(event => addEventToTable(event, false));
    updateEventsCount();
}

function updateSourceFilter() {
    const select = document.getElementById('event-source-filter');
    if (!select) return;

    const currentValue = select.value;
    select.innerHTML = '<option value="">All Sources</option>' +
        Array.from(state.eventSources).sort().map(s =>
            `<option value="${s}">${s}</option>`
        ).join('');
    select.value = currentValue;
}

function _recomputeFilteredCount() {
    _filteredCount = state.allEvents.filter(e => {
        if (state.eventTypeFilter && e.event_type !== state.eventTypeFilter) return false;
        if (state.eventSourceFilter && e.source !== state.eventSourceFilter) return false;
        if (!eventMatchesSearch(e)) return false;
        return true;
    }).length;
    _filteredCountDirty = false;
}

function updateEventsCount() {
    const badge = document.getElementById('events-count');
    const stats = document.getElementById('events-stats');

    const total = state.allEvents.length;

    // Only do a full recount when filters have changed
    if (_filteredCountDirty) {
        _recomputeFilteredCount();
    }

    if (badge) badge.textContent = total;
    if (stats) {
        if (_filteredCount === total) {
            stats.textContent = `${total} events`;
        } else {
            stats.textContent = `${_filteredCount} / ${total} events`;
        }
    }
}

function clearEvents() {
    state.allEvents = [];
    state.eventSources.clear();
    _filteredCount = 0;
    _filteredCountDirty = false;
    const tbody = document.getElementById('events-tbody');
    if (tbody) tbody.innerHTML = '';
    updateSourceFilter();
    updateEventsCount();
}

function handleFullEvent(event) {
    // Called when we receive a full event object from the server
    addEventToTable(event);
}

// Fetch initial events from the API
async function fetchInitialEvents() {
    try {
        const response = await fetch('/api/events?limit=100');
        const data = await response.json();

        if (data.events && data.events.length > 0) {
            // Add events in chronological order (oldest first, so newest ends up at top)
            data.events.reverse().forEach(event => {
                addEventToTable(event, true);
            });
        }
    } catch (error) {
        console.error('Failed to fetch initial events:', error);
    }
}

// ==========================================
// System View Switching
// ==========================================

let currentSystemView = 'log';

function switchSystemView(viewName) {
    if (!['log', 'timeline', 'summary'].includes(viewName)) return;
    currentSystemView = viewName;

    // Toggle view containers
    document.querySelectorAll('.system-view').forEach(el => {
        el.classList.toggle('active', el.id === `system-view-${viewName}`);
    });

    // Toggle buttons
    updateViewButtons('system-view-switcher', viewName);

    // Show/hide log-specific filters
    const filters = document.getElementById('events-toolbar-filters');
    if (filters) filters.style.display = viewName === 'log' ? '' : 'none';

    // Render active view
    if (viewName === 'timeline') renderTimelineView();
    else if (viewName === 'summary') renderSummaryView();
}

// ==========================================
// Timeline View
// ==========================================

function renderTimelineView() {
    const container = document.getElementById('timeline-container');
    if (!container) return;

    const events = state.allEvents;
    if (events.length === 0) {
        container.innerHTML = '<div style="text-align:center; color:var(--text-muted); padding:3rem; font-size:0.85rem;">No events yet</div>';
        return;
    }

    // Group by source
    const bySource = {};
    events.forEach((e, i) => {
        const src = e.source || 'unknown';
        if (!bySource[src]) bySource[src] = [];
        bySource[src].push({ ...e, _idx: i });
    });

    // Time range
    const timestamps = events.map(e => new Date(e.timestamp).getTime()).filter(t => !isNaN(t));
    const minT = Math.min(...timestamps);
    const maxT = Math.max(...timestamps);
    const range = maxT - minT || 1;

    let html = '';

    // Swim lanes per source
    for (const [source, srcEvents] of Object.entries(bySource).sort()) {
        html += `<div class="timeline-swim-lane">`;
        html += `<div class="timeline-lane-header">${source} <span style="opacity:0.5">(${srcEvents.length})</span></div>`;
        html += `<div class="timeline-lane-track">`;

        const sorted = [...srcEvents].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        for (const evt of sorted) {
            const badge = getEventBadgeClass(evt.event_type);
            const time = formatEventTime(evt.timestamp);
            const shortType = evt.event_type.replace(/_/g, ' ').toLowerCase()
                .split(' ').map(w => w[0]).join('').toUpperCase();
            html += `<div class="timeline-event ${badge}" data-evt-idx="${evt._idx}" title="${evt.event_type}\n${time}">${shortType}</div>`;
        }

        html += `</div></div>`;
    }

    // Time axis
    if (timestamps.length > 1) {
        const startTime = formatEventTime(new Date(minT).toISOString());
        const endTime = formatEventTime(new Date(maxT).toISOString());
        const midTime = formatEventTime(new Date(minT + range / 2).toISOString());
        html += `<div class="timeline-time-axis">
            <span class="timeline-time-label">${startTime}</span>
            <span class="timeline-time-label">${midTime}</span>
            <span class="timeline-time-label">${endTime}</span>
        </div>`;
    }

    // Detail panel (hidden until click)
    html += `<div class="timeline-detail" id="timeline-detail"></div>`;

    container.innerHTML = html;

    // Click handlers
    container.querySelectorAll('.timeline-event').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.evtIdx);
            const evt = state.allEvents[idx];
            if (!evt) return;

            // Toggle selection
            const wasSelected = el.classList.contains('selected');
            container.querySelectorAll('.timeline-event.selected').forEach(s => s.classList.remove('selected'));

            const detail = document.getElementById('timeline-detail');
            if (wasSelected) {
                detail.innerHTML = '';
                detail.classList.remove('visible');
                return;
            }

            el.classList.add('selected');
            const badge = getEventBadgeClass(evt.event_type);
            const dataHtml = evt.data && Object.keys(evt.data).length > 0
                ? `<pre>${JSON.stringify(evt.data, null, 2)}</pre>`
                : '<span style="color:var(--text-muted)">No data</span>';

            detail.innerHTML = `
                <div class="timeline-detail-header">
                    <span class="event-type-badge ${badge}">${evt.event_type}</span>
                    <span class="timeline-detail-meta">${evt.source} &middot; ${formatEventTime(evt.timestamp)}</span>
                </div>
                <div class="timeline-detail-body">${dataHtml}</div>
            `;
            detail.classList.add('visible');
        });
    });
}

// ==========================================
// Summary View
// ==========================================

function renderSummaryView() {
    const container = document.getElementById('summary-container');
    if (!container) return;

    const events = state.allEvents;
    const total = events.length;

    // Count by category
    const counts = { session: 0, acquisition: 0, perception: 0, cv: 0, analysis: 0, error: 0, other: 0 };
    events.forEach(e => {
        const cat = getEventBadgeClass(e.event_type);
        if (cat in counts) counts[cat]++;
        else counts.other++;
    });

    // Count by source
    const sourceCounts = {};
    events.forEach(e => {
        const src = e.source || 'unknown';
        sourceCounts[src] = (sourceCounts[src] || 0) + 1;
    });

    // Uptime
    const sessionStart = events.slice().reverse().find(e => e.event_type === 'SESSION_STARTED');
    let uptimeStr = '--';
    if (sessionStart) {
        const elapsed = (Date.now() - new Date(sessionStart.timestamp).getTime()) / 1000;
        if (elapsed < 60) uptimeStr = `${Math.floor(elapsed)}s`;
        else if (elapsed < 3600) uptimeStr = `${Math.floor(elapsed / 60)}m`;
        else uptimeStr = `${Math.floor(elapsed / 3600)}h ${Math.floor((elapsed % 3600) / 60)}m`;
    }

    // Latest event
    const latest = events[0];
    const latestStr = latest ? `${latest.event_type} (${formatEventTime(latest.timestamp)})` : '--';

    function card(label, count, cat, detail) {
        const pct = total > 0 ? (count / total * 100) : 0;
        return `<div class="summary-card ${cat}">
            <div class="summary-card-header"><span class="summary-card-label">${label}</span></div>
            <div class="summary-card-count">${count}</div>
            ${detail ? `<div class="summary-card-detail">${detail}</div>` : ''}
            <div class="summary-card-bar"><div class="summary-card-bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    }

    let html = '';

    // Uptime card
    html += `<div class="summary-card uptime">
        <div class="summary-card-header"><span class="summary-card-label">Uptime</span></div>
        <div class="summary-card-count">${uptimeStr}</div>
        <div class="summary-card-detail">Latest: ${latestStr}</div>
        <div class="summary-card-bar"><div class="summary-card-bar-fill" style="width:100%"></div></div>
    </div>`;

    html += card('Session', counts.session, 'session');
    html += card('Acquisition', counts.acquisition, 'acquisition');
    html += card('Perception', counts.perception, 'perception');
    if (counts.error > 0) html += card('Errors', counts.error, 'error');

    // Sources card
    const sourceList = Object.entries(sourceCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([name, ct]) => `<li class="summary-source-item"><span class="summary-source-name">${name}</span><span class="summary-source-count">${ct}</span></li>`)
        .join('');

    html += `<div class="summary-card sources">
        <div class="summary-card-header"><span class="summary-card-label">Sources</span></div>
        <div class="summary-card-count">${Object.keys(sourceCounts).length}</div>
        <ul class="summary-source-list">${sourceList}</ul>
        <div class="summary-card-bar"><div class="summary-card-bar-fill" style="width:100%"></div></div>
    </div>`;

    // Total card
    html += card('Total Events', total, 'session', `${MAX_EVENTS} max buffer`);

    container.innerHTML = html;
}

// Initialize events tab listeners
function initEventsTab() {
    // Initialize state
    state.allEvents = state.allEvents || [];
    state.eventTypeFilter = '';
    state.eventSourceFilter = '';
    state.eventSources = state.eventSources || new Set();

    // Search input
    const searchInput = document.getElementById('event-search');
    const searchClear = document.getElementById('search-clear');

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();

            // Show/hide clear button
            if (searchClear) {
                searchClear.classList.toggle('visible', query.length > 0);
            }

            // Debounce search
            clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(() => {
                searchQuery = query.toLowerCase();
                renderEventsTable();
            }, 150);
        });

        // Keyboard shortcut: Ctrl/Cmd + F to focus search when on events tab
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'f' && state.tab === TABS.EVENTS) {
                e.preventDefault();
                searchInput.focus();
                searchInput.select();
            }
        });
    }

    if (searchClear) {
        searchClear.addEventListener('click', () => {
            if (searchInput) {
                searchInput.value = '';
                searchInput.focus();
            }
            searchQuery = '';
            searchClear.classList.remove('visible');
            renderEventsTable();
        });
    }

    // Type filter
    const typeFilter = document.getElementById('event-type-filter');
    if (typeFilter) {
        typeFilter.addEventListener('change', (e) => {
            state.eventTypeFilter = e.target.value;
            renderEventsTable();
        });
    }

    // Source filter
    const sourceFilter = document.getElementById('event-source-filter');
    if (sourceFilter) {
        sourceFilter.addEventListener('change', (e) => {
            state.eventSourceFilter = e.target.value;
            renderEventsTable();
        });
    }

    // Clear button
    const clearBtn = document.getElementById('clear-events-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearEvents);
    }

    // System view switcher
    initViewSwitcher('system-view-switcher', switchSystemView, {
        views: ['log', 'timeline', 'summary'],
        guard: () => state.tab === TABS.EVENTS
    });
    updateViewButtons('system-view-switcher', 'log');

    // Fetch initial events from API
    fetchInitialEvents();
}
