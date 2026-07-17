"""
Live Benchmark Viewer for Perception System.

Real-time web interface showing:
- Current image being analyzed
- Reasoning traces as they happen
- Verification subagent activity
- Tool calls and results
- Prediction vs ground truth

Run with: python -m benchmarks.perception.live_viewer --session <path> --ground-truth <path>
"""

import argparse
import asyncio
import json
import logging
import sys
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# FastAPI and websockets
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:
    print("Please install: pip install fastapi uvicorn websockets")
    sys.exit(1)

from .ground_truth import GroundTruth
from .testset import OfflineTestset

logger = logging.getLogger(__name__)

# Global state for websocket connections
connected_clients: list[WebSocket] = []
benchmark_state: dict[str, Any] = {
    "status": "idle",
    "current_embryo": None,
    "current_timepoint": None,
    "predictions": [],
    "current_trace": [],
    "verification_active": False,
}
is_paused: bool = False
pause_event: asyncio.Event | None = None  # Will be initialized on startup


async def broadcast(message: dict):
    """Broadcast message to all connected clients."""
    if not connected_clients:
        return

    text = json.dumps(message, default=str)
    disconnected = []

    for client in connected_clients:
        try:
            await client.send_text(text)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        connected_clients.remove(client)


app = FastAPI(title="Perception Benchmark Live Viewer")


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Perception Benchmark - Live Viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0f3460;
        }
        .header h1 { font-size: 1.4em; color: #e94560; }
        .status {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .status.running { background: #4caf50; }
        .status.idle { background: #607d8b; }
        .status.complete { background: #2196f3; }

        .main-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr;
            gap: 15px;
            padding: 15px;
            height: calc(100vh - 70px);
        }

        .top-left {
            display: flex;
            gap: 15px;
        }

        .top-right {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .trace-section {
            grid-column: 1 / -1;
            min-height: 300px;
        }

        .image-section {
            background: #16213e;
            border-radius: 8px;
            padding: 12px;
            flex: 1;
        }

        .image-section h2 {
            font-size: 0.95em;
            color: #e94560;
            margin-bottom: 8px;
        }

        .image-container {
            background: #0f0f1a;
            border-radius: 4px;
            min-height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .image-container img {
            max-width: 100%;
            max-height: 280px;
            object-fit: contain;
        }

        .image-info {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.9em;
        }

        .info-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .info-label { color: #888; }
        .info-value { font-weight: 500; }

        .predictions-section {
            background: #16213e;
            border-radius: 8px;
            padding: 12px;
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            max-height: 280px;
        }

        .predictions-section h2 {
            font-size: 0.95em;
            color: #e94560;
            margin-bottom: 8px;
        }

        .predictions-list {
            flex: 1;
            overflow-y: auto;
        }

        .prediction-row {
            display: grid;
            grid-template-columns: 50px 80px 80px 50px 1fr;
            gap: 8px;
            padding: 6px 8px;
            border-bottom: 1px solid #0f3460;
            font-size: 0.8em;
            align-items: center;
        }

        .prediction-row.header {
            font-weight: 600;
            color: #888;
            background: #0f3460;
            border-radius: 4px;
            position: sticky;
            top: 0;
        }

        .prediction-row.correct { background: rgba(76, 175, 80, 0.2); }
        .prediction-row.adjacent { background: rgba(255, 152, 0, 0.2); }
        .prediction-row.wrong { background: rgba(244, 67, 54, 0.2); }
        .prediction-row.current {
            background: rgba(33, 150, 243, 0.3);
            animation: pulse 1s infinite;
        }

        .prediction-row.selected {
            background: rgba(255, 255, 255, 0.2);
            border-left: 3px solid #fff;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .stage-badge {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
            text-align: center;
        }

        .stage-early { background: #1565c0; }
        .stage-bean { background: #7b1fa2; }
        .stage-comma { background: #2e7d32; }
        .stage-1\\.5fold { background: #ef6c00; }
        .stage-2fold { background: #c2185b; }
        .stage-pretzel { background: #6a1b9a; }
        .stage-hatching { background: #00838f; }
        .stage-hatched { background: #3f51b5; }

        .verification-badge {
            background: #9c27b0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            margin-left: 5px;
        }

        .trace-section {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .trace-section h2 {
            font-size: 1.1em;
            color: #e94560;
            margin-bottom: 10px;
        }

        .trace-list {
            flex: 1;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }

        .trace-step {
            padding: 8px 10px;
            margin-bottom: 5px;
            border-radius: 4px;
            border-left: 3px solid #444;
        }

        .trace-step.tool_call {
            background: rgba(33, 150, 243, 0.2);
            border-color: #2196f3;
        }

        .trace-step.tool_result {
            background: rgba(76, 175, 80, 0.2);
            border-color: #4caf50;
        }

        .trace-step.initial_analysis {
            background: rgba(156, 39, 176, 0.2);
            border-color: #9c27b0;
        }

        .trace-step.final_decision {
            background: rgba(255, 152, 0, 0.2);
            border-color: #ff9800;
        }

        .trace-step.verification_requested {
            background: rgba(233, 30, 99, 0.2);
            border-color: #e91e63;
        }

        .trace-step.verification_subagent {
            background: rgba(103, 58, 183, 0.2);
            border-color: #673ab7;
        }

        .trace-step.verification_result {
            background: rgba(0, 188, 212, 0.2);
            border-color: #00bcd4;
        }

        .trace-step-type {
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
            font-size: 0.75em;
            color: #888;
        }

        .trace-step-content {
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.85em;
            line-height: 1.4;
        }

        .stats-section {
            background: #16213e;
            border-radius: 8px;
            padding: 10px 12px;
        }

        .stats-section h2 {
            font-size: 0.9em;
            color: #e94560;
            margin-bottom: 8px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }

        .stat-card {
            background: #0f3460;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #e94560;
        }

        .stat-label {
            font-size: 0.7em;
            color: #888;
            margin-top: 2px;
        }

        .no-data {
            color: #666;
            text-align: center;
            padding: 40px;
        }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f0f1a; }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Perception Benchmark Live Viewer</h1>
        <div style="display: flex; align-items: center; gap: 15px;">
            <button id="pause-btn" onclick="togglePause()"
              style="padding: 8px 16px; font-size: 0.9em; cursor: pointer;
                     background: #4caf50; color: white; border: none;
                     border-radius: 4px;">⏸ Pause</button>
            <div id="status" class="status idle">Connecting...</div>
        </div>
    </div>

    <div class="main-container">
        <!-- Top Left: Image -->
        <div class="image-section">
            <h2>Current Image</h2>
            <div class="image-container" id="image-container">
                <div class="no-data">Waiting for images...</div>
            </div>
            <div class="image-info" id="image-info" style="display: none;">
                <div class="info-item">
                    <span class="info-label">Embryo:</span>
                    <span class="info-value" id="current-embryo">-</span>
                </div>
                <div class="info-item">
                    <span class="info-label">T:</span>
                    <span class="info-value" id="current-timepoint">-</span>
                </div>
                <div class="info-item">
                    <span class="info-label">GT:</span>
                    <span class="info-value" id="ground-truth">-</span>
                </div>
            </div>
        </div>

        <!-- Top Right: Stats + Predictions -->
        <div class="top-right">
            <div class="stats-section">
                <h2>Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="stat-accuracy">-</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat-adjacent">-</div>
                        <div class="stat-label">Adjacent</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat-total">0</div>
                        <div class="stat-label">Predictions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat-verified">0</div>
                        <div class="stat-label">Verified</div>
                    </div>
                </div>
            </div>

            <div class="predictions-section">
                <h2>Predictions (click to view trace)</h2>
                <div class="predictions-list" id="predictions-list">
                    <div class="prediction-row header">
                        <div>T</div>
                        <div>Pred</div>
                        <div>Truth</div>
                        <div>Conf</div>
                        <div>Details</div>
                    </div>
                    <div class="no-data">No predictions yet</div>
                </div>
            </div>
        </div>

        <!-- Bottom: Reasoning Trace (full width) -->
        <div class="trace-section">
            <h2>Reasoning Trace</h2>
            <div class="trace-list" id="trace-list">
                <div class="no-data">Waiting for reasoning...</div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let predictions = [];
        let traces = {};  // Store traces by timepoint
        let images = {};  // Store images by timepoint
        let currentTimepoint = null;
        let selectedTimepoint = null;  // For viewing historical traces
        let isPaused = false;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status idle';
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'status';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            switch(msg.type) {
                case 'status':
                    updateStatus(msg);
                    break;
                case 'image':
                    updateImage(msg);
                    break;
                case 'trace_step':
                    addTraceStep(msg);
                    break;
                case 'prediction':
                    addPrediction(msg);
                    break;
                case 'clear_trace':
                    clearTrace();
                    break;
                case 'stats':
                    updateStats(msg);
                    break;
            }
        }

        function updateStatus(msg) {
            const status = document.getElementById('status');
            status.textContent = msg.status;
            status.className = 'status ' + msg.status.toLowerCase();
        }

        function updateImage(msg) {
            currentTimepoint = msg.timepoint;
            // Store separate images if available
            images[msg.timepoint] = {
                combined: msg.image,
                top: msg.top_image,
                side: msg.side_image,
                embryoId: msg.embryo_id,
                groundTruth: msg.ground_truth
            };

            // Only update display if we're viewing the current timepoint
            if (selectedTimepoint === null || selectedTimepoint === msg.timepoint) {
                displayImage(msg.timepoint, msg.embryo_id, msg.ground_truth);
            }
        }

        function displayImage(timepoint, embryoId, groundTruth) {
            const container = document.getElementById('image-container');
            const imgData = images[timepoint];
            if (imgData) {
                // Display three-view combined image (XY+YZ+XZ orthogonal projections)
                container.innerHTML = `
                    <div style="text-align: center;">
                        <div style="color: #888; font-size: 12px; margin-bottom: 4px;">
                          THREE-VIEW (XY | YZ / XZ)</div>
                        <img src="data:image/jpeg;base64,${imgData.combined}"
                          alt="Three-View T${timepoint}" style="max-height: 450px;">
                    </div>`;
                // Also get embryoId and groundTruth from stored data if not provided
                embryoId = embryoId || imgData.embryoId;
                groundTruth = groundTruth || imgData.groundTruth;
            }
            document.getElementById('image-info').style.display = 'flex';
            document.getElementById('current-embryo').textContent = embryoId || '-';
            document.getElementById('current-timepoint').textContent = 'T' + timepoint;
            document.getElementById('ground-truth').textContent = groundTruth || '-';
        }

        function clearTrace() {
            // Start new trace for current timepoint
            if (currentTimepoint !== null) {
                traces[currentTimepoint] = [];
            }
            // Only clear display if viewing current trace
            if (selectedTimepoint === null) {
                document.getElementById('trace-list').innerHTML = '';
            }
        }

        function addTraceStep(msg) {
            // Store trace step
            if (currentTimepoint !== null) {
                if (!traces[currentTimepoint]) {
                    traces[currentTimepoint] = [];
                }
                traces[currentTimepoint].push(msg);
            }

            // Only update display if viewing current trace
            if (selectedTimepoint === null) {
                renderTraceStep(msg);
            }
        }

        function renderTraceStep(msg) {
            const list = document.getElementById('trace-list');
            if (list.querySelector('.no-data')) {
                list.innerHTML = '';
            }

            const step = document.createElement('div');
            step.className = `trace-step ${msg.step_type}`;
            step.innerHTML = `
                <div class="trace-step-type">${msg.step_type.replace(/_/g, ' ')}</div>
                <div class="trace-step-content">${escapeHtml(msg.content)}</div>
            `;
            list.appendChild(step);
            list.scrollTop = list.scrollHeight;
        }

        function selectTimepoint(timepoint) {
            selectedTimepoint = timepoint;
            renderPredictions();  // Update highlighting
            renderTraceForTimepoint(timepoint);

            // Show image for this timepoint
            const pred = predictions.find(p => p.timepoint === timepoint);
            if (pred) {
                displayImage(timepoint, pred.embryo_id || 'embryo_3', pred.ground_truth);
            }
        }

        function renderTraceForTimepoint(timepoint) {
            const list = document.getElementById('trace-list');
            const traceSteps = traces[timepoint] || [];

            // Update header
            document.querySelector('.trace-section h2').innerHTML =
                `Reasoning Trace - T${timepoint} <button onclick="clearSelection()"
                  style="float:right;font-size:0.8em;padding:2px 8px;
                         cursor:pointer;">Show Live</button>`;

            if (traceSteps.length === 0) {
                list.innerHTML = '<div class="no-data">No trace recorded for T'
                    + timepoint + '</div>';
                return;
            }

            list.innerHTML = '';
            for (const msg of traceSteps) {
                renderTraceStep(msg);
            }
        }

        function clearSelection() {
            selectedTimepoint = null;
            document.querySelector('.trace-section h2').textContent = 'Reasoning Trace';
            renderPredictions();

            // Show current trace
            const list = document.getElementById('trace-list');
            list.innerHTML = '';
            if (currentTimepoint !== null && traces[currentTimepoint]) {
                for (const msg of traces[currentTimepoint]) {
                    renderTraceStep(msg);
                }
            }
        }

        function addPrediction(msg) {
            msg.embryo_id = msg.embryo_id || 'embryo_3';
            predictions.push(msg);
            renderPredictions();
        }

        function renderPredictions() {
            const list = document.getElementById('predictions-list');

            let html = `
                <div class="prediction-row header">
                    <div>Time</div>
                    <div>Predicted</div>
                    <div>Truth</div>
                    <div>Conf</div>
                    <div>Details</div>
                </div>
            `;

            for (const pred of predictions.slice().reverse()) {
                const isSelected = pred.timepoint === selectedTimepoint;
                const rowClass = isSelected ? 'selected' :
                    (pred.is_correct ? 'correct' :
                    (pred.is_adjacent ? 'adjacent' : 'wrong'));

                const verifiedBadge = pred.verification_triggered ?
                    '<span class="verification-badge">verified</span>' : '';

                html += `
                    <div class="prediction-row ${rowClass}"
                      onclick="selectTimepoint(${pred.timepoint})" style="cursor:pointer;">
                        <div>T${pred.timepoint}</div>
                        <div><span class="stage-badge stage-${pred.predicted}"
                          >${pred.predicted}</span></div>
                        <div><span class="stage-badge stage-${pred.ground_truth}"
                          >${pred.ground_truth}</span></div>
                        <div>${(pred.confidence * 100).toFixed(0)}%</div>
                        <div>${pred.phase_count > 1
                          ? pred.phase_count + '-phase' : ''}${verifiedBadge}</div>
                    </div>
                `;
            }

            list.innerHTML = html;
        }

        function updateStats(msg) {
            document.getElementById('stat-accuracy').textContent =
                msg.accuracy !== null ? (msg.accuracy * 100).toFixed(0) + '%' : '-';
            document.getElementById('stat-adjacent').textContent =
                msg.adjacent !== null ? (msg.adjacent * 100).toFixed(0) + '%' : '-';
            document.getElementById('stat-total').textContent = msg.total;
            document.getElementById('stat-verified').textContent = msg.verified;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function togglePause() {
            isPaused = !isPaused;
            const btn = document.getElementById('pause-btn');
            if (isPaused) {
                btn.textContent = '▶ Resume';
                btn.style.background = '#ff9800';
            } else {
                btn.textContent = '⏸ Pause';
                btn.style.background = '#4caf50';
            }
            // Send pause/resume command to server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: isPaused ? 'pause' : 'resume' }));
            }
        }

        connect();
    </script>
</body>
</html>
"""


@app.get("/")
async def get_index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_paused, pause_event

    await websocket.accept()
    connected_clients.append(websocket)

    # Send current state
    await websocket.send_text(
        json.dumps(
            {
                "type": "status",
                "status": benchmark_state["status"],
            }
        )
    )

    try:
        while True:
            # Handle incoming messages (pause/resume commands)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("command") == "pause":
                    is_paused = True
                    if pause_event:
                        pause_event.clear()
                    await broadcast({"type": "status", "status": "Paused"})
                    logger.info("Benchmark paused")
                elif msg.get("command") == "resume":
                    is_paused = False
                    if pause_event:
                        pause_event.set()
                    await broadcast({"type": "status", "status": "Running"})
                    logger.info("Benchmark resumed")
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


class LiveBenchmarkRunner:
    """
    Benchmark runner with live websocket updates and trace persistence.
    """

    def __init__(
        self,
        testset: OfflineTestset,
        embryo_id: str,
        enable_verification: bool = True,
        start_timepoint: int = 0,
        max_timepoints: int | None = None,
        trace_dir: Path | None = None,
    ):
        self.testset = testset
        self.embryo_id = embryo_id
        self.enable_verification = enable_verification
        self.start_timepoint = start_timepoint
        self.max_timepoints = max_timepoints

        self.predictions: list[dict] = []
        self.correct_count = 0
        self.adjacent_count = 0
        self.verified_count = 0

        # Trace persistence
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_dir = trace_dir or Path("benchmarks/results/traces")
        self.run_dir = self.trace_dir / f"{self.run_id}_{embryo_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.traces: dict[int, list[dict]] = {}  # timepoint -> trace steps

        logger.info(f"Trace persistence enabled: {self.run_dir}")

    async def run(self):
        """Run the benchmark with live updates."""
        import anthropic
        from gently.harness.perception.engine import PerceptionEngine
        from gently.harness.perception.session import PerceptionSession
        from gently.harness.perception.stages import DevelopmentalStage

        # Update status
        benchmark_state["status"] = "running"
        benchmark_state["current_embryo"] = self.embryo_id
        await broadcast({"type": "status", "status": "Running"})

        # Create engine and session
        client = anthropic.Anthropic()

        examples_path = Path("gently/examples")
        if not examples_path.exists():
            examples_path = None

        engine = PerceptionEngine(
            claude_client=client,
            examples_path=examples_path,
            enable_verification=self.enable_verification,
        )

        session = PerceptionSession(self.embryo_id)

        # Iterate through timepoints
        # Calculate end_timepoint relative to start
        end_tp = None
        if self.max_timepoints is not None:
            end_tp = self.start_timepoint + self.max_timepoints

        for test_case in self.testset.iter_embryo(
            self.embryo_id,
            start_timepoint=self.start_timepoint,
            end_timepoint=end_tp,
        ):
            # Check for pause
            global is_paused, pause_event
            while is_paused:
                if pause_event:
                    await pause_event.wait()
                else:
                    await asyncio.sleep(0.5)

            benchmark_state["current_timepoint"] = test_case.timepoint

            # Send image(s)
            await broadcast(
                {
                    "type": "image",
                    "embryo_id": self.embryo_id,
                    "timepoint": test_case.timepoint,
                    "ground_truth": test_case.ground_truth_stage,
                    "image": test_case.image_b64,  # Combined for backward compat
                    "top_image": test_case.top_image_b64,
                    "side_image": test_case.side_image_b64,
                }
            )

            # Clear trace for new prediction
            await broadcast({"type": "clear_trace"})

            # Run perception with trace streaming
            result = await self._run_perception_with_streaming(engine, session, test_case)

            # Check accuracy
            is_correct = result.stage == test_case.ground_truth_stage
            is_adjacent = False

            if not is_correct and test_case.ground_truth_stage:
                try:
                    pred_order = DevelopmentalStage.get_order(result.stage)
                    gt_order = DevelopmentalStage.get_order(test_case.ground_truth_stage)
                    is_adjacent = abs(pred_order - gt_order) <= 1
                except ValueError:
                    pass

            if is_correct:
                self.correct_count += 1
            if is_adjacent or is_correct:
                self.adjacent_count += 1
            if result.verification_triggered:
                self.verified_count += 1

            # Send prediction
            pred_msg = {
                "type": "prediction",
                "timepoint": test_case.timepoint,
                "predicted": result.stage,
                "ground_truth": test_case.ground_truth_stage,
                "confidence": result.confidence,
                "is_correct": is_correct,
                "is_adjacent": is_adjacent,
                "verification_triggered": result.verification_triggered,
                "phase_count": result.phase_count,
                "is_current": False,
            }
            self.predictions.append(pred_msg)
            await broadcast(pred_msg)

            # Save trace for this timepoint
            self._save_timepoint_trace(
                test_case.timepoint, result, test_case, is_correct, is_adjacent
            )

            # Send updated stats
            total = len(self.predictions)
            await broadcast(
                {
                    "type": "stats",
                    "accuracy": self.correct_count / total if total > 0 else None,
                    "adjacent": self.adjacent_count / total if total > 0 else None,
                    "total": total,
                    "verified": self.verified_count,
                }
            )

            # Add observation to session with simulated timestamp
            # Typical diSPIM acquisition interval is ~4 minutes per timepoint
            simulated_timestamp = (
                datetime.now()
                - timedelta(minutes=(self.max_timepoints or 100) * 4)
                + timedelta(minutes=test_case.timepoint * 4)
            )

            session.add_observation(
                timepoint=test_case.timepoint,
                stage=result.stage,
                is_hatching=result.is_hatching,
                confidence=result.confidence,
                reasoning=result.reasoning,
                is_transitional=result.is_transitional,
                transition_between=result.transition_between,
                timestamp=simulated_timestamp,
            )

        # Save run summary
        self._save_run_summary()

        # Complete
        benchmark_state["status"] = "complete"
        await broadcast({"type": "status", "status": "Complete"})

    async def _run_perception_with_streaming(self, engine, session, test_case):
        """Run perception and stream trace steps."""

        # We need to hook into the reasoning trace
        # For now, run perception and stream the trace after
        # Use only the combined three-view image (don't pass separate top/side)
        result = await engine.perceive(
            image_b64=test_case.image_b64,
            session=session,
            timepoint=test_case.timepoint,
            volume=test_case.volume,
        )

        # Initialize trace storage for this timepoint
        self.traces[test_case.timepoint] = []

        # Stream and store trace steps
        if result.reasoning_trace:
            for step in result.reasoning_trace.steps:
                trace_step = {
                    "step_type": step.step_type,
                    "content": step.content,
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                    "tool_result_summary": step.tool_result_summary,
                }
                self.traces[test_case.timepoint].append(trace_step)

                await broadcast({"type": "trace_step", **trace_step})
                await asyncio.sleep(0.1)  # Small delay for visual effect

        return result

    def _save_timepoint_trace(
        self, timepoint: int, result, test_case, is_correct: bool, is_adjacent: bool
    ):
        """Save trace for a single timepoint to disk."""
        trace_data = {
            "timepoint": timepoint,
            "embryo_id": self.embryo_id,
            "ground_truth": test_case.ground_truth_stage,
            "predicted": result.stage,
            "confidence": result.confidence,
            "is_correct": is_correct,
            "is_adjacent": is_adjacent,
            "verification_triggered": result.verification_triggered,
            "phase_count": result.phase_count,
            "reasoning": result.reasoning,
            "trace_steps": self.traces.get(timepoint, []),
        }

        trace_path = self.run_dir / f"T{timepoint:03d}.json"
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)

    def _save_run_summary(self):
        """Save overall run summary."""
        total = len(self.predictions)
        summary = {
            "run_id": self.run_id,
            "embryo_id": self.embryo_id,
            "start_timepoint": self.start_timepoint,
            "max_timepoints": self.max_timepoints,
            "total_predictions": total,
            "correct_count": self.correct_count,
            "adjacent_count": self.adjacent_count,
            "verified_count": self.verified_count,
            "accuracy": self.correct_count / total if total > 0 else 0,
            "adjacent_accuracy": self.adjacent_count / total if total > 0 else 0,
            "predictions": self.predictions,
        }

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Run summary saved: {summary_path}")


async def run_benchmark_background(
    session_path: Path,
    ground_truth_path: Path,
    embryo_id: str,
    enable_verification: bool,
    start_timepoint: int = 0,
    max_timepoints: int | None = None,
):
    """Run benchmark in background after server starts."""
    print("[DEBUG] run_benchmark_background starting", flush=True)
    try:
        global pause_event
        pause_event = asyncio.Event()
        pause_event.set()  # Start in running state

        await asyncio.sleep(2)  # Wait for client to connect
        print("[DEBUG] Slept 2s, loading data...", flush=True)

        # Load data
        ground_truth = GroundTruth.from_json(ground_truth_path)
        print(
            f"[DEBUG] Loaded ground truth: {len(ground_truth.transitions)} embryos",
            flush=True,
        )
        testset = OfflineTestset(
            session_path=session_path,
            ground_truth=ground_truth,
            load_volumes=True,
        )
        print(f"[DEBUG] Created testset with embryos: {testset.embryo_ids}", flush=True)

        # Run benchmark
        runner = LiveBenchmarkRunner(
            testset=testset,
            embryo_id=embryo_id,
            enable_verification=enable_verification,
            start_timepoint=start_timepoint,
            max_timepoints=max_timepoints,
        )
        print("[DEBUG] Starting runner.run()", flush=True)

        await runner.run()
    except Exception as e:
        print(f"[DEBUG] ERROR in run_benchmark_background: {e}", flush=True)
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Live Perception Benchmark Viewer")
    parser.add_argument(
        "--session",
        required=True,
        help="Path to session directory with TIF volumes",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--embryo",
        default="embryo_3",
        help="Embryo ID to run (default: embryo_3)",
    )
    parser.add_argument(
        "--no-verification",
        action="store_true",
        help="Disable verification subagents",
    )
    parser.add_argument(
        "--start-timepoint",
        type=int,
        default=0,
        help="Timepoint to start from (default: 0)",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=None,
        help="Maximum number of timepoints to process (default: all)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to run server on (default: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    session_path = Path(args.session)
    if not session_path.exists():
        print(f"Session path not found: {session_path}")
        sys.exit(1)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Trace directory
    trace_dir = Path("benchmarks/results/traces")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 60}")
    print("Perception Benchmark Live Viewer")
    print(f"{'=' * 60}")
    print(f"Session: {session_path}")
    print(f"Embryo: {args.embryo}")
    print(f"Start timepoint: T{args.start_timepoint}")
    print(f"Max timepoints: {args.max_timepoints or 'all'}")
    print(f"Verification: {'disabled' if args.no_verification else 'enabled'}")
    print(f"Traces: {trace_dir / f'{run_id}_{args.embryo}'}")
    print(f"URL: http://localhost:{args.port}")
    print(f"{'=' * 60}\n")

    # Open browser
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    # Create background task for benchmark
    @app.on_event("startup")
    async def startup_event():
        print("[DEBUG] Startup event fired", flush=True)
        asyncio.create_task(
            run_benchmark_background(
                session_path=session_path,
                ground_truth_path=gt_path,
                embryo_id=args.embryo,
                enable_verification=not args.no_verification,
                start_timepoint=args.start_timepoint,
                max_timepoints=args.max_timepoints,
            )
        )
        print("[DEBUG] Background task created", flush=True)

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
