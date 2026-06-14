"""
Trace Viewer for Perception Benchmarks.

Generates HTML reports showing prediction details and reasoning traces.
"""

import argparse
import json
import sys
from pathlib import Path

from .metrics import PerceptionMetrics, format_metrics_summary

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Perception Benchmark Report</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1, h2, h3 { margin-top: 0; }
        h1 { color: #1a1a2e; }

        .summary-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #16213e;
        }

        .metric-label {
            color: #666;
            font-size: 0.9em;
        }

        .embryo-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .prediction-row {
            display: grid;
            grid-template-columns: 80px 120px 120px 80px 1fr;
            gap: 10px;
            padding: 10px;
            border-bottom: 1px solid #eee;
            align-items: center;
        }

        .prediction-row.header {
            font-weight: bold;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .prediction-row.correct { background: #e8f5e9; }
        .prediction-row.adjacent { background: #fff3e0; }
        .prediction-row.wrong { background: #ffebee; }

        .stage-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }

        .stage-early { background: #e3f2fd; color: #1565c0; }
        .stage-bean { background: #f3e5f5; color: #7b1fa2; }
        .stage-comma { background: #e8f5e9; color: #2e7d32; }
        .stage-1\\.5fold { background: #fff3e0; color: #ef6c00; }
        .stage-2fold { background: #fce4ec; color: #c2185b; }
        .stage-pretzel { background: #f3e5f5; color: #7b1fa2; }
        .stage-hatching { background: #e0f7fa; color: #00838f; }
        .stage-hatched { background: #e8eaf6; color: #3f51b5; }
        .stage-arrested { background: #ffebee; color: #c62828; }

        .confidence-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: #4caf50;
            transition: width 0.3s;
        }

        .reasoning-trace {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            display: none;
        }

        .reasoning-trace.visible { display: block; }

        .trace-step {
            margin-bottom: 10px;
            padding: 8px;
            border-left: 3px solid #ccc;
        }

        .trace-step.tool_call { border-color: #2196f3; background: #e3f2fd; }
        .trace-step.tool_result { border-color: #4caf50; background: #e8f5e9; }
        .trace-step.final_decision { border-color: #ff9800; background: #fff3e0; }
        .trace-step.verification_requested { border-color: #9c27b0; background: #f3e5f5; }
        .trace-step.verification_subagent { border-color: #673ab7; background: #ede7f6; }
        .trace-step.verification_result { border-color: #e91e63; background: #fce4ec; }

        .verification-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 500;
            background: #9c27b0;
            color: white;
            margin-left: 5px;
        }

        .phase-indicator {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 500;
            background: #607d8b;
            color: white;
            margin-left: 5px;
        }

        .expand-btn {
            cursor: pointer;
            color: #1976d2;
            font-size: 0.85em;
        }

        .expand-btn:hover { text-decoration: underline; }

        .confusion-matrix {
            overflow-x: auto;
            margin: 20px 0;
        }

        .confusion-matrix table {
            border-collapse: collapse;
            font-size: 0.9em;
        }

        .confusion-matrix th, .confusion-matrix td {
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: center;
        }

        .confusion-matrix th { background: #f5f5f5; }
        .confusion-matrix .diagonal { background: #e8f5e9; font-weight: bold; }
        .confusion-matrix .error { background: #ffebee; }

        .tabs {
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }

        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }

        .tab.active {
            border-color: #1976d2;
            color: #1976d2;
            font-weight: 500;
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Perception Benchmark Report</h1>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{accuracy:.1%}</div>
                <div class="metric-label">Exact Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{adjacent_accuracy:.1%}</div>
                <div class="metric-label">Adjacent Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_predictions}</div>
                <div class="metric-label">Total Predictions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{mean_confidence:.2f}</div>
                <div class="metric-label">Mean Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{tool_call_rate:.2f}</div>
                <div class="metric-label">Avg Tool Calls</div>
            </div>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="showTab('predictions')">Predictions</div>
            <div class="tab" onclick="showTab('metrics')">Detailed Metrics</div>
            <div class="tab" onclick="showTab('confusion')">Confusion Matrix</div>
        </div>

        <div id="predictions" class="tab-content active">
            {embryo_sections}
        </div>

        <div id="metrics" class="tab-content">
            <div class="summary-box">
                <pre>{detailed_metrics}</pre>
            </div>
        </div>

        <div id="confusion" class="tab-content">
            <div class="summary-box">
                <h2>Confusion Matrix</h2>
                {confusion_matrix_html}
            </div>
        </div>
    </div>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.tab[onclick="showTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}

        function toggleTrace(id) {{
            const trace = document.getElementById('trace-' + id);
            trace.classList.toggle('visible');
        }}
    </script>
</body>
</html>
"""


def generate_embryo_section(embryo_id: str, predictions: list[dict]) -> str:
    """Generate HTML for one embryo's predictions."""
    rows = [
        '<div class="prediction-row header">',
        "<div>Timepoint</div>",
        "<div>Predicted</div>",
        "<div>Ground Truth</div>",
        "<div>Confidence</div>",
        "<div>Details</div>",
        "</div>",
    ]

    for i, pred in enumerate(predictions):
        is_correct = pred.get("is_correct", False)
        is_adjacent = pred.get("is_adjacent_correct", False) and not is_correct

        row_class = "correct" if is_correct else ("adjacent" if is_adjacent else "wrong")

        pred_stage = pred.get("predicted_stage", "?")
        gt_stage = pred.get("ground_truth_stage", "?")
        confidence = pred.get("confidence", 0)
        timepoint = pred.get("timepoint", i)
        reasoning = pred.get("reasoning", "")

        row_id = f"{embryo_id}-{timepoint}"

        rows.append(f'<div class="prediction-row {row_class}">')
        rows.append(f"<div>T{timepoint}</div>")
        rows.append(f'<div><span class="stage-badge stage-{pred_stage}">{pred_stage}</span></div>')
        rows.append(f'<div><span class="stage-badge stage-{gt_stage}">{gt_stage}</span></div>')
        rows.append(f"""<div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%"></div>
            </div>
            <small>{confidence:.0%}</small>
        </div>""")

        # Details column with expand button
        tool_calls = pred.get("tool_calls", 0)
        is_transitional = pred.get("is_transitional", False)
        verification_triggered = pred.get("verification_triggered", False)
        phase_count = pred.get("phase_count", 1)

        details = []
        if tool_calls > 0:
            details.append(f"{tool_calls} tool calls")
        if is_transitional:
            details.append("transitional")

        details_str = ", ".join(details) if details else ""

        # Add verification and phase badges
        badges = ""
        if verification_triggered:
            badges += '<span class="verification-badge">verified</span>'
        if phase_count > 1:
            badges += f'<span class="phase-indicator">{phase_count}-phase</span>'

        rows.append(f"""<div>
            {details_str}
            {badges}
            <span class="expand-btn" onclick="toggleTrace('{row_id}')">
                [show trace]
            </span>
        </div>""")
        rows.append("</div>")

        # Reasoning trace (hidden by default)
        trace_html = format_reasoning_trace(pred.get("reasoning_trace"))
        rows.append(f"""<div id="trace-{row_id}" class="reasoning-trace">
            <strong>Reasoning:</strong> {reasoning}
            {trace_html}
        </div>""")

    return f"""
    <div class="embryo-section">
        <h2>{embryo_id}</h2>
        {"".join(rows)}
    </div>
    """


def format_reasoning_trace(trace: dict | None) -> str:
    """Format reasoning trace as HTML."""
    if not trace:
        return ""

    steps = trace.get("steps", [])
    if not steps:
        return ""

    html_parts = ["<hr><strong>Trace Steps:</strong>"]

    for step in steps:
        step_type = step.get("step_type", "unknown")
        content = step.get("content", "")

        if step_type == "tool_call":
            tool_name = step.get("tool_name", "")
            tool_input = step.get("tool_input", {})
            html_parts.append(f"""
                <div class="trace-step tool_call">
                    <strong>Tool Call:</strong> {tool_name}<br>
                    Input: {json.dumps(tool_input, indent=2)}
                </div>
            """)
        elif step_type == "tool_result":
            summary = step.get("tool_result_summary", content)
            html_parts.append(f"""
                <div class="trace-step tool_result">
                    <strong>Tool Result:</strong> {summary}
                </div>
            """)
        elif step_type == "final_decision":
            html_parts.append(f"""
                <div class="trace-step final_decision">
                    <strong>Final Decision:</strong><br>
                    {content[:500]}...
                </div>
            """)
        elif step_type == "verification_requested":
            html_parts.append(f"""
                <div class="trace-step verification_requested">
                    <strong>Verification Requested:</strong><br>
                    {content}
                </div>
            """)
        elif step_type == "verification_subagent":
            tool_input = step.get("tool_input", {})
            summary = step.get("tool_result_summary", content)
            html_parts.append(f"""
                <div class="trace-step verification_subagent">
                    <strong>Subagent:</strong> {tool_input.get("stage_a", "?")} vs
                    {tool_input.get("stage_b", "?")}<br>
                    Result: {summary}
                </div>
            """)
        elif step_type == "verification_result":
            html_parts.append(f"""
                <div class="trace-step verification_result">
                    <strong>Verification Result:</strong><br>
                    {content}
                </div>
            """)

    return "".join(html_parts)


def generate_confusion_matrix_html(confusion: dict[str, dict[str, int]]) -> str:
    """Generate HTML table for confusion matrix."""
    stages = [
        "early",
        "bean",
        "comma",
        "1.5fold",
        "2fold",
        "pretzel",
        "hatching",
        "hatched",
    ]

    # Filter to stages present in data
    present = set()
    for gt, preds in confusion.items():
        present.add(gt)
        for pred in preds:
            present.add(pred)

    stages = [s for s in stages if s in present]

    rows = ['<div class="confusion-matrix"><table>']

    # Header
    rows.append("<tr><th>GT \\ Pred</th>")
    for s in stages:
        rows.append(f"<th>{s}</th>")
    rows.append("</tr>")

    # Data rows
    for gt in stages:
        rows.append(f"<tr><th>{gt}</th>")
        for pred in stages:
            count = confusion.get(gt, {}).get(pred, 0)
            if gt == pred:
                rows.append(f'<td class="diagonal">{count}</td>')
            elif count > 0:
                rows.append(f'<td class="error">{count}</td>')
            else:
                rows.append("<td>.</td>")
        rows.append("</tr>")

    rows.append("</table></div>")
    return "".join(rows)


def generate_html_report(report_data: dict) -> str:
    """Generate complete HTML report from benchmark data."""
    # Extract summary metrics
    metrics = report_data.get("metrics", {})
    embryo_results = report_data.get("embryo_results", [])

    accuracy = metrics.get("accuracy", 0)
    adjacent_accuracy = metrics.get("adjacent_accuracy", 0)
    total_predictions = report_data.get("total_predictions", 0)
    mean_confidence = metrics.get("mean_confidence", 0)
    tool_call_rate = metrics.get("tool_call_rate", 0)

    # Generate embryo sections
    embryo_sections = []
    for er in embryo_results:
        embryo_sections.append(generate_embryo_section(er["embryo_id"], er["predictions"]))

    # Generate confusion matrix
    confusion = metrics.get("confusion_matrix", {})
    confusion_html = generate_confusion_matrix_html(confusion)

    # Generate detailed metrics
    if metrics:
        metrics_obj = PerceptionMetrics(
            accuracy=accuracy,
            adjacent_accuracy=adjacent_accuracy,
            stage_accuracy=metrics.get("stage_accuracy", {}),
            stage_counts=metrics.get("stage_counts", {}),
            confusion_matrix=confusion,
            mean_confidence=mean_confidence,
            confidence_when_correct=metrics.get("confidence_when_correct", 0),
            confidence_when_wrong=metrics.get("confidence_when_wrong", 0),
            expected_calibration_error=metrics.get("expected_calibration_error", 0),
            backward_transitions=metrics.get("backward_transitions", 0),
            total_tool_calls=metrics.get("total_tool_calls", 0),
            tool_call_rate=tool_call_rate,
            accuracy_with_tools=metrics.get("accuracy_with_tools", 0),
            accuracy_without_tools=metrics.get("accuracy_without_tools", 0),
            transitional_count=metrics.get("transitional_count", 0),
            transitional_rate=metrics.get("transitional_rate", 0),
            transitional_accuracy=metrics.get("transitional_accuracy", 0),
        )
        detailed_metrics = format_metrics_summary(metrics_obj)
    else:
        detailed_metrics = "No metrics available"

    return HTML_TEMPLATE.format(
        accuracy=accuracy,
        adjacent_accuracy=adjacent_accuracy,
        total_predictions=total_predictions,
        mean_confidence=mean_confidence,
        tool_call_rate=tool_call_rate,
        embryo_sections="".join(embryo_sections),
        detailed_metrics=detailed_metrics,
        confusion_matrix_html=confusion_html,
    )


def main():
    """CLI entry point for trace viewer."""
    parser = argparse.ArgumentParser(description="Generate HTML report from benchmark results")
    parser.add_argument(
        "--result",
        required=True,
        help="Path to benchmark result JSON file",
    )
    parser.add_argument(
        "--output",
        help="Output HTML file path (default: <result>_report.html)",
    )
    parser.add_argument(
        "--embryo",
        help="Show only specific embryo",
    )

    args = parser.parse_args()

    # Load result
    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Result file not found: {result_path}")
        sys.exit(1)

    with open(result_path) as f:
        report_data = json.load(f)

    # Filter embryo if specified
    if args.embryo:
        report_data["embryo_results"] = [
            er for er in report_data.get("embryo_results", []) if er["embryo_id"] == args.embryo
        ]

    # Generate HTML
    html = generate_html_report(report_data)

    # Write output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = result_path.with_suffix(".html")

    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
