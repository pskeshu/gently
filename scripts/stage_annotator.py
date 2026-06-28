"""
Stage Annotation Tool for C. elegans Embryo Images

Opens a local web interface to browse timepoints and label developmental stages.
Labeled images are saved to the perception examples folder.

Usage:
    python scripts/stage_annotator.py --session 3a4b0604
    python scripts/stage_annotator.py --session 3a4b0604 --embryo embryo_1
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import socketserver
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import tifffile
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
GENTLY_PATH = Path("D:/Gently")
EXAMPLES_PATH = Path(__file__).parent.parent / "gently" / "examples" / "stages"
STAGES = ["early", "comma", "1.5fold", "pretzel", "hatching", "hatched"]


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 uint8."""
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [1, 99])
    if p_high > p_low:
        img = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def load_and_project(tif_path: Path) -> np.ndarray:
    """Load TIFF and create dual-view projection (top + side).

    Data format: (1, Z, Y, X) where X = 2*width (View A | View B side-by-side)

    We use View A (left half) and create orthogonal projections:
    - TOP: Max projection along Z axis (looking down)
    - SIDE: Max projection along Y axis (looking from side)
    """
    vol = tifffile.imread(tif_path)
    vol = np.squeeze(vol)

    # Handle different volume shapes
    if vol.ndim == 3:
        # Shape: (Z, Y, X) where X may contain both views side-by-side
        z_depth, height, width = vol.shape

        # Check if width is roughly 4x height (dual-view format)
        if width > height * 2:
            # Extract View A (left half) - shape (Z, Y, X/2)
            view_a = vol[:, :, : width // 2]
        else:
            view_a = vol

        # TOP projection: max along Z axis (looking down at embryo)
        # Shape: (Y, X)
        top_proj = np.max(view_a, axis=0)

        # SIDE projection: max along Y axis (looking at embryo from side)
        # Shape: (Z, X) - this shows the embryo's profile
        side_proj = np.max(view_a, axis=1)

    elif vol.ndim == 2:
        # Already 2D - use same for both
        top_proj = vol
        side_proj = vol
    else:
        # Unexpected shape
        raise ValueError(f"Unexpected volume shape: {vol.shape}")

    # Normalize each view independently
    top_norm = normalize_image(top_proj)
    side_norm = normalize_image(side_proj)

    # Layout: TOP on left, SIDE on right
    # TOP: (Y, X) e.g. (512, 1024)
    # SIDE: (Z, X) e.g. (50, 1024) - need to rotate and scale

    target_height = top_norm.shape[0]  # 512

    # Rotate side view 90° clockwise so Z becomes horizontal
    # (Z, X) -> (X, Z) after rotation
    side_rotated = np.rot90(side_norm, k=-1)  # Now (1024, 50)

    # Scale to match top's height, make side view at least 150px wide for visibility
    pil_side = Image.fromarray(side_rotated)
    new_width = max(150, int(side_rotated.shape[1] * target_height / side_rotated.shape[0]))
    pil_side = pil_side.resize((new_width, target_height), Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)

    # Create vertical separator line
    sep_width = 4
    separator = np.ones((target_height, sep_width), dtype=np.uint8) * 128

    # Combine horizontally: TOP | separator | SIDE
    combined = np.concatenate([top_norm, separator, side_scaled], axis=1)

    return combined


def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 JPEG."""
    pil_img = Image.fromarray(img)

    # Convert to RGB for JPEG (grayscale works but RGB is more compatible)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def save_example(img: np.ndarray, stage: str, source_name: str) -> str:
    """Save image to examples folder."""
    stage_dir = EXAMPLES_PATH / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Find next example number
    existing = list(stage_dir.glob("example_*.jpg"))
    next_num = len(existing) + 1

    filepath = stage_dir / f"example_{next_num:03d}.jpg"
    Image.fromarray(img).save(filepath, quality=95)

    logger.info(f"Saved {stage} example: {filepath.name}")
    return str(filepath)


# Global state
class AppState:
    images: list = []  # List of (path, projection) tuples
    current_idx = 0
    session_id = ""
    embryos: dict = {}  # embryo_name -> [(path, proj), ...]
    current_embryo = None
    embryo_list: list = []


state = AppState()


class AnnotatorHandler(BaseHTTPRequestHandler):
    """HTTP handler for annotation interface."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # Suppress HTTP logging

    def do_GET(self):
        try:
            parsed = urlparse(self.path)

            if parsed.path == "/":
                self.send_html()
            elif parsed.path == "/api/image":
                self.send_current_image()
            elif parsed.path == "/api/status":
                self.send_status()
            elif parsed.path == "/api/next":
                state.current_idx = min(state.current_idx + 1, len(state.images) - 1)
                self.send_current_image()
            elif parsed.path == "/api/prev":
                state.current_idx = max(state.current_idx - 1, 0)
                self.send_current_image()
            elif parsed.path.startswith("/api/goto"):
                params = parse_qs(parsed.query)
                idx = int(params.get("idx", [0])[0])
                state.current_idx = max(0, min(idx, len(state.images) - 1))
                self.send_current_image()
            elif parsed.path.startswith("/api/embryo"):
                params = parse_qs(parsed.query)
                embryo = params.get("name", [""])[0]
                self.switch_embryo(embryo)
                self.send_current_image()
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_error(500, str(e))

    def do_POST(self):
        try:
            parsed = urlparse(self.path)

            if parsed.path == "/api/save_markers":
                # Read POST body
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())

                # Get labels from client
                labels = data.get("labels", {})
                count = self.save_with_labels(labels)
                self.send_json({"ok": True, "saved": count})
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            self.send_error(500, str(e))

    def send_html(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stage Annotator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { margin-bottom: 10px; }
        .status { color: #888; margin-bottom: 10px; }
        .embryo-selector { margin-bottom: 15px; }
        .embryo-selector select {
            padding: 8px 16px;
            font-size: 16px;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
        }
        .image-container {
            background: #000;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        img { max-width: 100%; height: auto; }
        .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .stage-btn { background: #333; color: #fff; border: 2px solid transparent; }
        .stage-btn:hover { filter: brightness(1.2); }
        .stage-btn.active { border-color: #fff; box-shadow: 0 0 10px rgba(255,255,255,0.5); }
        .nav-btn { background: #2196F3; color: #fff; }
        .nav-btn:hover { background: #1976D2; }
        .save-btn { background: #FF9800; color: #fff; }
        .save-btn:hover { background: #F57C00; }
        .nav-container { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .slider-container { flex: 1; display: flex; flex-direction: column; gap: 4px; }
        .slider { width: 100%; height: 20px; }
        .timeline-track {
            height: 16px;
            background: #333;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }
        .timeline-segment {
            position: absolute;
            height: 100%;
            top: 0;
        }
        .timeline-cursor {
            position: absolute;
            width: 2px;
            height: 100%;
            background: #fff;
            top: 0;
            z-index: 10;
        }
        .keyboard-hint { color: #666; font-size: 12px; margin-top: 5px; }
        .current-label {
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 6px;
            display: inline-block;
            margin-left: 20px;
            font-weight: bold;
        }
        .filename { color: #888; font-size: 12px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Stage Annotator</h1>
        <div class="status" id="status">Loading...</div>

        <div class="embryo-selector">
            <label>Embryo: </label>
            <select id="embryo-select" onchange="switchEmbryo(this.value)"></select>
        </div>

        <div class="nav-container">
            <button class="nav-btn" onclick="prev()">&larr; Prev (A)</button>
            <div class="slider-container">
                <input type="range" class="slider" id="slider" min="0" max="100"
                  value="0" oninput="gotoIdx(this.value)">
                <div class="timeline-track" id="timeline-track">
                    <div class="timeline-cursor" id="timeline-cursor"></div>
                </div>
            </div>
            <button class="nav-btn" onclick="next()">Next (D) &rarr;</button>
            <span id="position">0 / 0</span>
            <span class="current-label" id="current-label">Not labeled</span>
        </div>

        <div class="image-container">
            <img id="image" src="" alt="Loading...">
            <div class="filename" id="filename"></div>
        </div>

        <div class="controls">
            <button class="stage-btn" data-stage="early" onclick="label('early')">1: Early</button>
            <button class="stage-btn" data-stage="comma" onclick="label('comma')">2: Comma</button>
            <button class="stage-btn" data-stage="1.5fold"
              onclick="label('1.5fold')">3: 1.5-Fold</button>
            <button class="stage-btn" data-stage="pretzel"
              onclick="label('pretzel')">4: Pretzel</button>
            <button class="stage-btn" data-stage="hatching"
              onclick="label('hatching')">5: Hatching</button>
            <button class="stage-btn" data-stage="hatched"
              onclick="label('hatched')">6: Hatched</button>
            <button class="stage-btn" data-stage="" onclick="label('')">0: Clear</button>
        </div>

        <div class="controls">
            <button class="save-btn" onclick="saveAll()">Save Labeled Examples (S)</button>
        </div>

        <p class="keyboard-hint">
            <strong>How to use:</strong> Navigate to where a stage STARTS, then press the
            stage key (1-6).<br>
            All images from that point until the next marked stage will be labeled
            automatically.<br>
            Keys: 1-6 = mark stage start, 0 = clear mark, A/D or Arrows = navigate, S = save
        </p>

        <div id="stage-markers"
          style="margin-top: 15px; padding: 10px; background: #222; border-radius: 6px;">
            <strong>Stage Transitions:</strong>
            <div id="markers-list" style="margin-top: 8px; font-family: monospace;"></div>
        </div>
    </div>

    <script>
        let totalImages = 0;
        let currentIdx = 0;
        let paths = [];  // All image paths in order

        // Stage markers: index -> stage name (marks where a stage STARTS)
        let stageMarkers = {};

        const STAGES = ['early', 'comma', '1.5fold', 'pretzel', 'hatching', 'hatched'];

        // Color map for each stage (vibrant, distinguishable colors)
        const STAGE_COLORS = {
            'early': '#E91E63',    // Pink
            'comma': '#3F51B5',    // Indigo
            '1.5fold': '#00BCD4',  // Cyan
            'pretzel': '#4CAF50',  // Green
            'hatching': '#FF9800', // Orange
            'hatched': '#F44336'   // Red
        };

        // Apply colors to stage buttons on load
        function initStageButtons() {
            document.querySelectorAll('.stage-btn').forEach(btn => {
                const stage = btn.dataset.stage;
                if (stage && STAGE_COLORS[stage]) {
                    btn.style.background = STAGE_COLORS[stage];
                }
            });
        }

        // Update the timeline track with colored segments
        function updateTimelineTrack() {
            const track = document.getElementById('timeline-track');
            const cursor = document.getElementById('timeline-cursor');

            // Remove old segments (keep cursor)
            track.querySelectorAll('.timeline-segment').forEach(el => el.remove());

            if (totalImages === 0) return;

            // Get sorted marker indices
            let markerIndices = Object.keys(stageMarkers).map(Number).sort((a, b) => a - b);

            // Create segments for each stage range
            for (let i = 0; i < markerIndices.length; i++) {
                let startIdx = markerIndices[i];
                let endIdx = markerIndices[i + 1] || totalImages;
                let stage = stageMarkers[startIdx];

                let startPct = (startIdx / totalImages) * 100;
                let widthPct = ((endIdx - startIdx) / totalImages) * 100;

                let segment = document.createElement('div');
                segment.className = 'timeline-segment';
                segment.style.left = startPct + '%';
                segment.style.width = widthPct + '%';
                segment.style.background = STAGE_COLORS[stage] || '#666';
                track.appendChild(segment);
            }

            // Update cursor position
            let cursorPct = (currentIdx / Math.max(1, totalImages - 1)) * 100;
            cursor.style.left = cursorPct + '%';
        }

        // Get the stage for a given index based on markers
        function getStageAtIndex(idx) {
            // Find the most recent marker at or before this index
            let stage = null;
            let markerIndices = Object.keys(stageMarkers).map(Number).sort((a, b) => a - b);
            for (let markerIdx of markerIndices) {
                if (markerIdx <= idx) {
                    stage = stageMarkers[markerIdx];
                }
            }
            return stage;
        }

        // Update the markers list display
        function updateMarkersDisplay() {
            const list = document.getElementById('markers-list');
            let markerIndices = Object.keys(stageMarkers).map(Number).sort((a, b) => a - b);

            if (markerIndices.length === 0) {
                list.innerHTML = '<span style="color: #666;">'
                    + 'No markers set. Navigate to where a stage starts and press 1-7.'
                    + '</span>';
                return;
            }

            let html = '';
            for (let i = 0; i < markerIndices.length; i++) {
                let idx = markerIndices[i];
                let stage = stageMarkers[idx];
                let color = STAGE_COLORS[stage] || '#888';
                let nextIdx = markerIndices[i + 1] || totalImages;
                let count = nextIdx - idx;

                html += '<div style="margin: 4px 0; display: flex;'
                    + ' align-items: center; gap: 10px;">';
                html += '<span style="display: inline-block; width: 12px; height: 12px;'
                    + ' background: ' + color + '; border-radius: 2px;"></span>';
                html += '<span style="color: ' + color + '; font-weight: bold;'
                    + ' min-width: 70px;">' + stage.toUpperCase() + '</span>';
                html += '<span style="color: #888;">frame ' + (idx + 1) + '</span>';
                html += '<span style="color: #666;">(' + count + ' frames)</span>';
                html += '<button onclick="clearMarker(' + idx + ')"'
                    + ' style="padding: 2px 8px; background: #c00; color: white;'
                    + ' border: none; border-radius: 3px;'
                    + ' cursor: pointer; font-size: 11px;">×</button>';
                html += '<button onclick="gotoIdx(' + idx + ')"'
                    + ' style="padding: 2px 8px; background: #555; color: white;'
                    + ' border: none; border-radius: 3px;'
                    + ' cursor: pointer; font-size: 11px;">Go</button>';
                html += '</div>';
            }
            list.innerHTML = html;

            // Also update timeline track
            updateTimelineTrack();
        }

        // Compute all labels from markers (for saving)
        function computeLabelsFromMarkers() {
            let labels = {};
            for (let i = 0; i < paths.length; i++) {
                let stage = getStageAtIndex(i);
                if (stage) {
                    labels[paths[i]] = stage;
                }
            }
            return labels;
        }

        function updateUI(data) {
            if (data.error) {
                console.error(data.error);
                return;
            }

            document.getElementById('image').src = 'data:image/jpeg;base64,' + data.data;
            document.getElementById('filename').textContent = data.filename || '';
            document.getElementById('slider').max = data.total - 1;
            document.getElementById('slider').value = data.idx;
            document.getElementById('position').textContent = (data.idx + 1) + ' / ' + data.total;

            totalImages = data.total;
            currentIdx = data.idx;

            // Store path for current image
            if (data.all_paths) {
                paths = data.all_paths;
            }

            // Get current stage from markers
            const currentStage = getStageAtIndex(currentIdx);
            const isMarkerHere = stageMarkers[currentIdx] !== undefined;

            const labelEl = document.getElementById('current-label');
            if (currentStage) {
                const color = STAGE_COLORS[currentStage] || '#666';
                labelEl.textContent = currentStage.toUpperCase() + (isMarkerHere ? ' ★' : '');
                labelEl.style.background = color;
                labelEl.style.color = '#fff';
            } else {
                labelEl.textContent = 'Not labeled';
                labelEl.style.background = '#333';
                labelEl.style.color = '#888';
            }

            // Update timeline cursor position
            updateTimelineTrack();

            // Update button states - highlight if marker is at current position
            document.querySelectorAll('.stage-btn').forEach(btn => {
                btn.classList.remove('active');
                if (isMarkerHere && btn.dataset.stage === stageMarkers[currentIdx]) {
                    btn.classList.add('active');
                }
            });

            // Update status
            const labeledCount = Object.keys(computeLabelsFromMarkers()).length;
            document.getElementById('status').textContent =
                'Session: ' + data.session + ' | Embryo: ' + data.embryo + ' | ' +
                Object.keys(stageMarkers).length + ' markers | ' + labeledCount + ' frames labeled';

            // Update embryo selector
            const select = document.getElementById('embryo-select');
            if (data.embryo_list && select.options.length !== data.embryo_list.length) {
                select.innerHTML = '';
                data.embryo_list.forEach(e => {
                    const opt = document.createElement('option');
                    opt.value = e;
                    opt.textContent = e;
                    if (e === data.embryo) opt.selected = true;
                    select.appendChild(opt);
                });
            } else {
                select.value = data.embryo;
            }

            updateMarkersDisplay();
        }

        async function fetchAndUpdate(url) {
            try {
                const resp = await fetch(url);
                const data = await resp.json();
                updateUI(data);
            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        function next() { fetchAndUpdate('/api/next'); }
        function prev() { fetchAndUpdate('/api/prev'); }
        function gotoIdx(idx) { fetchAndUpdate('/api/goto?idx=' + idx); }
        function switchEmbryo(name) {
            stageMarkers = {};  // Clear markers when switching embryo
            fetchAndUpdate('/api/embryo?name=' + encodeURIComponent(name));
        }

        // Set a stage marker at current position
        function setMarker(stage) {
            if (stage) {
                stageMarkers[currentIdx] = stage;
            } else {
                delete stageMarkers[currentIdx];
            }
            updateMarkersDisplay();
            // Refresh to show updated label
            fetchAndUpdate('/api/image');
        }

        function clearMarker(idx) {
            delete stageMarkers[idx];
            updateMarkersDisplay();
            fetchAndUpdate('/api/image');
        }

        // Override the old label function to set markers
        function label(stage) {
            setMarker(stage);
        }

        async function saveAll() {
            try {
                // Send computed labels to server
                const labels = computeLabelsFromMarkers();
                const resp = await fetch('/api/save_markers', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({labels: labels})
                });
                const data = await resp.json();
                alert('Saved ' + (data.saved || 0) + ' examples!');
                fetchAndUpdate('/api/image');
            } catch (e) {
                alert('Save failed: ' + e);
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'SELECT') return;

            if (e.key >= '0' && e.key <= '6') {
                const stages = ['', 'early', 'comma', '1.5fold', 'pretzel', 'hatching', 'hatched'];
                setMarker(stages[parseInt(e.key)]);
            } else if (e.key === 'a' || e.key === 'ArrowLeft') {
                prev();
            } else if (e.key === 'd' || e.key === 'ArrowRight') {
                next();
            } else if (e.key === 's' && !e.ctrlKey) {
                saveAll();
            }
        });

        // Initial load
        initStageButtons();
        fetchAndUpdate('/api/image');
    </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(html))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(html.encode())

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_current_image(self):
        if not state.images:
            self.send_json({"error": "No images loaded"})
            return

        path, proj = state.images[state.current_idx]

        # No label overlay - labels are computed client-side from markers
        b64 = image_to_base64(proj)

        # Get all paths for client-side label computation
        all_paths = [str(p) for p, _ in state.images]

        self.send_json(
            {
                "data": b64,
                "path": str(path),
                "filename": path.name,
                "idx": state.current_idx,
                "total": len(state.images),
                "all_paths": all_paths,
                "session": state.session_id,
                "embryo": state.current_embryo,
                "embryo_list": state.embryo_list,
            }
        )

    def switch_embryo(self, embryo_name):
        if embryo_name in state.embryos:
            state.current_embryo = embryo_name
            state.images = state.embryos[embryo_name]
            state.current_idx = 0
            logger.info(f"Switched to {embryo_name} ({len(state.images)} images)")

    def save_with_labels(self, labels: dict):
        """Save labeled images to examples folder.

        Instead of saving every frame, we sample a few good examples per stage.
        """
        saved = 0

        # Group paths by stage
        stage_paths: dict = {}
        for path_str, stage in labels.items():
            if stage not in STAGES:
                continue
            if stage not in stage_paths:
                stage_paths[stage] = []
            stage_paths[stage].append(path_str)

        # For each stage, sample up to 3 representative examples
        # Take from beginning, middle, and end of each stage range
        for stage, paths in stage_paths.items():
            if not paths:
                continue

            # Sample indices
            n = len(paths)
            if n <= 3:
                sample_indices = list(range(n))
            else:
                # Beginning, middle, end
                sample_indices = [0, n // 2, n - 1]

            for idx in sample_indices:
                path_str = paths[idx]
                path = Path(path_str)

                # Find projection in current embryo images
                for p, proj in state.images:
                    if str(p) == path_str:
                        save_example(proj, stage, path.name)
                        saved += 1
                        break

        logger.info(f"Saved {saved} examples across {len(stage_paths)} stages")
        return saved


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""

    allow_reuse_address = True
    daemon_threads = True


def open_chrome(url):
    """Try to open URL in Chrome specifically."""
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]

    for chrome_path in chrome_paths:
        if os.path.exists(chrome_path):
            subprocess.Popen([chrome_path, url])
            return True

    # Fallback to default browser
    import webbrowser

    webbrowser.open(url)
    return False


def main():
    parser = argparse.ArgumentParser(description="Stage annotation tool")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--embryo", default=None, help="Start with specific embryo")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    # Find images
    images_path = GENTLY_PATH / "images" / args.session
    if not images_path.exists():
        logger.error(f"Session not found: {images_path}")
        return 1

    tif_files = sorted(images_path.glob("*.tif"))
    logger.info(f"Found {len(tif_files)} TIFF files")

    # Group by embryo
    embryos = {}
    for tif_path in tif_files:
        # Extract embryo name (e.g., "embryo_1" from "embryo_1_20251218_113636.tif")
        parts = tif_path.stem.split("_")
        if len(parts) >= 2:
            embryo_name = f"{parts[0]}_{parts[1]}"
        else:
            embryo_name = "unknown"

        if embryo_name not in embryos:
            embryos[embryo_name] = []
        embryos[embryo_name].append(tif_path)

    logger.info(f"Found {len(embryos)} embryos: {list(embryos.keys())}")

    # Load images for each embryo
    for embryo_name, files in embryos.items():
        logger.info(f"Loading {embryo_name} ({len(files)} timepoints)...")
        loaded = []
        for i, tif_path in enumerate(files):
            if i % 20 == 0 and i > 0:
                logger.info(f"  {i}/{len(files)}...")
            try:
                proj = load_and_project(tif_path)
                loaded.append((tif_path, proj))
            except Exception as e:
                logger.warning(f"Failed to load {tif_path.name}: {e}")
        embryos[embryo_name] = loaded
        logger.info(f"  Loaded {len(loaded)} images")

    # Setup state
    state.embryos = embryos
    state.embryo_list = sorted(embryos.keys())
    state.session_id = args.session

    # Select initial embryo
    if args.embryo and args.embryo in embryos:
        state.current_embryo = args.embryo
    else:
        state.current_embryo = state.embryo_list[0] if state.embryo_list else None

    if state.current_embryo:
        state.images = embryos[state.current_embryo]
    state.current_idx = 0

    # Setup signal handler for clean shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    server = ThreadedHTTPServer(("127.0.0.1", args.port), AnnotatorHandler)
    url = f"http://127.0.0.1:{args.port}"

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Annotation server running at: {url}")
    logger.info("Press Ctrl+C to stop")
    logger.info(f"{'=' * 50}\n")

    if not args.no_browser:
        open_chrome(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        logger.info("Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
