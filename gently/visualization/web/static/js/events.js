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

// Event type categories for badge styling
const CV_EVENT_TYPES = ['CV_TASK_QUEUED', 'CV_TASK_COMPLETED', 'CV_TASK_FAILED', 'CV_AGENT_THINKING', 'CV_RESULT_READY'];
const ANALYSIS_EVENT_TYPES = ['SEGMENTATION_COMPLETED', 'STAGE_DETECTED', 'CELL_DIVISION_DETECTED', 'LINEAGE_UPDATED', 'ANOMALY_DETECTED'];
const ACQUISITION_EVENT_TYPES = ['VOLUME_ACQUIRED', 'IMAGE_ACQUIRED', 'ACQUISITION_STARTED', 'ACQUISITION_COMPLETED'];
const SESSION_EVENT_TYPES = ['SESSION_STARTED', 'SESSION_ENDED', 'SESSION_SAVED', 'SESSION_RESTORED'];
const ERROR_EVENT_TYPES = ['ERROR_OCCURRED', 'CV_TASK_FAILED', 'ACQUISITION_FAILED'];

function getEventBadgeClass(eventType) {
    if (CV_EVENT_TYPES.includes(eventType)) return 'cv';
    if (ANALYSIS_EVENT_TYPES.includes(eventType)) return 'analysis';
    if (ACQUISITION_EVENT_TYPES.includes(eventType)) return 'acquisition';
    if (SESSION_EVENT_TYPES.includes(eventType)) return 'session';
    if (ERROR_EVENT_TYPES.includes(eventType)) return 'error';
    return 'default';
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
        parts.push(`<span class="event-data-key">${key}</span>=<span class="event-data-value">${displayValue}</span>`);
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
    if (state.eventTypeFilter && event.event_type !== state.eventTypeFilter) return;
    if (state.eventSourceFilter && event.source !== state.eventSourceFilter) return;

    const tbody = document.getElementById('events-tbody');
    if (!tbody) return;

    const tr = document.createElement('tr');
    tr.className = prepend ? 'event-row-new' : '';
    tr.dataset.eventId = event.event_id || '';

    const badgeClass = getEventBadgeClass(event.event_type);

    tr.innerHTML = `
        <td class="col-time">${formatEventTime(event.timestamp)}</td>
        <td class="col-type"><span class="event-type-badge ${badgeClass}">${event.event_type}</span></td>
        <td class="col-source"><span class="event-source">${event.source || '-'}</span></td>
        <td class="col-data"><div class="event-data">${formatEventData(event.data)}</div></td>
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
    const tbody = document.getElementById('events-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';

    const filtered = state.allEvents.filter(e => {
        if (state.eventTypeFilter && e.event_type !== state.eventTypeFilter) return false;
        if (state.eventSourceFilter && e.source !== state.eventSourceFilter) return false;
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

function updateEventsCount() {
    const badge = document.getElementById('events-count');
    const stats = document.getElementById('events-stats');

    const total = state.allEvents.length;
    const filtered = state.allEvents.filter(e => {
        if (state.eventTypeFilter && e.event_type !== state.eventTypeFilter) return false;
        if (state.eventSourceFilter && e.source !== state.eventSourceFilter) return false;
        return true;
    }).length;

    if (badge) badge.textContent = total;
    if (stats) {
        if (filtered === total) {
            stats.textContent = `${total} events`;
        } else {
            stats.textContent = `${filtered} / ${total} events`;
        }
    }
}

function clearEvents() {
    state.allEvents = [];
    state.eventSources.clear();
    const tbody = document.getElementById('events-tbody');
    if (tbody) tbody.innerHTML = '';
    updateSourceFilter();
    updateEventsCount();
}

function handleFullEvent(event) {
    // Called when we receive a full event object from the server
    addEventToTable(event);
}

// Initialize events tab listeners
function initEventsTab() {
    // Initialize state
    state.allEvents = state.allEvents || [];
    state.eventTypeFilter = '';
    state.eventSourceFilter = '';
    state.eventSources = state.eventSources || new Set();

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
}
