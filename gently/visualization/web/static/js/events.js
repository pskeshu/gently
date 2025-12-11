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

// Image UID fields to look for in event data
const IMAGE_UID_FIELDS = ['volume_uid', 'image_uid', 'uid', 'visualization_uid', 'segmentation_uid', 'source_uid', 'mask_uid'];

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
    if (!searchQuery || !text) return text;
    try {
        const regex = new RegExp(`(${escapeRegex(searchQuery)})`, 'gi');
        return String(text).replace(regex, '<mark class="search-highlight">$1</mark>');
    } catch (e) {
        return text;
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
    if (state.eventTypeFilter && event.event_type !== state.eventTypeFilter) return;
    if (state.eventSourceFilter && event.source !== state.eventSourceFilter) return;
    if (!eventMatchesSearch(event)) return;

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

function updateEventsCount() {
    const badge = document.getElementById('events-count');
    const stats = document.getElementById('events-stats');

    const total = state.allEvents.length;
    const filtered = state.allEvents.filter(e => {
        if (state.eventTypeFilter && e.event_type !== state.eventTypeFilter) return false;
        if (state.eventSourceFilter && e.source !== state.eventSourceFilter) return false;
        if (!eventMatchesSearch(e)) return false;
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
            if ((e.ctrlKey || e.metaKey) && e.key === 'f' && state.tab === 'events') {
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

    // Fetch initial events from API
    fetchInitialEvents();
}
