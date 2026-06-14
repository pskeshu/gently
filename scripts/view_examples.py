"""
Simple web viewer for stage examples with descriptions.
"""

import base64
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "gently" / "examples" / "stages"
STAGES = ["early", "comma", "1.5fold", "pretzel", "hatching", "hatched"]


def load_examples():
    """Load all examples with metadata."""
    all_examples = {}

    for stage in STAGES:
        stage_dir = EXAMPLES_DIR / stage
        if not stage_dir.exists():
            continue

        # Load metadata
        metadata_path = stage_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        examples = []
        for img_path in sorted(stage_dir.glob("example_*.jpg")):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            description = metadata.get("examples", {}).get(img_path.name, "")
            examples.append(
                {
                    "filename": img_path.name,
                    "image_b64": img_b64,
                    "description": description,
                }
            )

        all_examples[stage] = {
            "description": metadata.get("description", ""),
            "examples": examples,
        }

    return all_examples


def generate_html():
    """Generate HTML page showing all examples."""
    examples = load_examples()

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Stage Examples Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #00d4ff;
        }
        .stage-section {
            margin-bottom: 40px;
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }
        .stage-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }
        .stage-name {
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
            text-transform: uppercase;
        }
        .stage-desc {
            color: #aaa;
            font-style: italic;
        }
        .examples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }
        .example-card {
            background: #0f0f23;
            border-radius: 8px;
            overflow: hidden;
        }
        .example-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .example-info {
            padding: 12px;
        }
        .example-filename {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .example-description {
            color: #ccc;
            line-height: 1.4;
        }
        .no-desc {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>C. elegans Stage Examples</h1>
"""

    for stage in STAGES:
        if stage not in examples:
            continue

        stage_data = examples[stage]
        html += f"""
    <div class="stage-section">
        <div class="stage-header">
            <span class="stage-name">{stage}</span>
            <span class="stage-desc">{stage_data["description"]}</span>
        </div>
        <div class="examples-grid">
"""

        for ex in stage_data["examples"]:
            desc_html = (
                ex["description"]
                if ex["description"]
                else '<span class="no-desc">No description</span>'
            )
            html += f"""
            <div class="example-card">
                <img src="data:image/jpeg;base64,{ex["image_b64"]}" alt="{ex["filename"]}">
                <div class="example-info">
                    <div class="example-filename">{ex["filename"]}</div>
                    <div class="example-description">{desc_html}</div>
                </div>
            </div>
"""

        html += """
        </div>
    </div>
"""

    html += """
</body>
</html>
"""
    return html


class ExamplesHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = generate_html()
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", len(html.encode()))
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_error(404)


def main():
    port = 8766
    server = HTTPServer(("127.0.0.1", port), ExamplesHandler)
    print(f"\n{'=' * 50}")
    print(f"Examples viewer running at: http://127.0.0.1:{port}")
    print(f"{'=' * 50}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
