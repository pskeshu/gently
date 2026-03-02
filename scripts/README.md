# Scripts

Utility scripts for perception system training data and stage annotation.

| Script | Description |
|--------|-------------|
| `auto_label_examples.py` | Sample timelapse images and auto-label developmental stages using Claude |
| `extract_stage_examples.py` | Extract stage examples from timelapse data with dual-view (TOP\|SIDE) projections |
| `gemini_stage_test.py` | Test Gemini video understanding for developmental stage classification |
| `generate_examples.py` | Generate compact stage example montages from embryo timelapses |
| `projection_explorer.py` | Interactive web UI to test and compare volume projection methods |
| `regenerate_examples_three_view.py` | Regenerate stage reference images in three-view projection format |
| `stage_annotator.py` | Web interface to browse timepoints and label developmental stages |
| `view_examples.py` | Simple web viewer for stage examples with descriptions |

## Usage

All scripts run from the repository root:

```bash
python scripts/stage_annotator.py --session 3a4b0604
python scripts/projection_explorer.py --session 59799c78
python scripts/auto_label_examples.py D:/path/to/timelapse --embryo 1
```
