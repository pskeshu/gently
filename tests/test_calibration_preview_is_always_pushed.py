"""A calibration sweep must produce a preview on every run, not only a bad one.

`hybrid_focus_selection` builds a labelled montage of the focus sweep. It used
to build it inside Stage 2 only — the branch taken when FFT scoring is
*ambiguous* and Claude Vision has to be consulted. So a calibration that went
well pushed no image at all, and the single run an operator could actually see
was one where the algorithm had already lost confidence.

That is backwards. The montage is the sweep made legible, and it is what the
Operate pane now seats under the Calibrate button while the routine runs
(`operate.js`, `PREVIEW_KINDS.focus_montage`). Ryan spent the 2026-08-07
walkthrough unable to tell what calibration was doing.

The confident path is the one that regresses, because it is the path where the
montage is not otherwise needed — nothing but this test would notice it going
away again.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from gently.app.tools.calibration_tools import hybrid_focus_selection


class _RecordingAgent:
    """Minimal stand-in for MicroscopyAgent — records what reaches the browser."""

    def __init__(self) -> None:
        self.viz_server = object()  # truthy: a client is attached
        self.pushes: list[dict] = []

    def push_viz(self, array, uid, data_type="image", metadata=None):  # noqa: ANN001
        self.pushes.append(
            {"array": array, "uid": uid, "data_type": data_type, "metadata": metadata or {}}
        )


def _sweep(sharp_at: int, n: int = 5) -> list[np.ndarray]:
    """A focus sweep whose sharpest frame is at `sharp_at`.

    A hard-edged disc, progressively blurred — real defocus, and it separates
    cleanly under `fft_bandpass` (second-best lands at 0.70 of best, inside the
    0.85 confidence threshold). A Gaussian blob does not: the metric scores the
    *defocused* frames higher than a tight one, so a blob sweep silently takes
    the ambiguous Vision path and tests nothing.
    """
    y, x = np.mgrid[0:96, 0:96]
    disc = ((((x - 48) ** 2 + (y - 48) ** 2) < 20**2) * 255).astype(np.uint8)
    imgs = []
    for k in range(n):
        im = Image.fromarray(disc)
        radius = abs(k - sharp_at) * 3.0
        if radius:
            im = im.filter(ImageFilter.GaussianBlur(radius))
        imgs.append(np.asarray(im).astype(np.float32) / 255.0)
    return imgs


OFFSETS = [-20.0, -10.0, 0.0, 10.0, 20.0]


@pytest.mark.asyncio
async def test_confident_run_still_pushes_the_montage() -> None:
    agent = _RecordingAgent()
    images = _sweep(sharp_at=2)

    idx, method, _ratio = await hybrid_focus_selection(
        images=images,
        offsets=OFFSETS,
        claude_vision=None,  # never consulted on the confident path
        agent=agent,
        embryo_id="embryo_1",
    )

    assert method == "fft", "test setup no longer exercises the confident path"
    assert idx == 2

    assert len(agent.pushes) == 1, (
        "a confident calibration pushed no preview — the montage has moved back "
        "inside the ambiguous branch and the operator sees nothing again"
    )
    push = agent.pushes[0]
    assert push["data_type"] == "focus_montage", "operate.js filters on this exact data_type"
    assert push["uid"] == "focus_montage_embryo_1"
    assert push["array"].ndim >= 2 and push["array"].size > 0


@pytest.mark.asyncio
async def test_the_push_carries_what_the_caption_reads() -> None:
    """operate.js renders pick/method/offsets; dropping one blanks the caption."""
    agent = _RecordingAgent()

    await hybrid_focus_selection(
        images=_sweep(sharp_at=2),
        offsets=OFFSETS,
        claude_vision=None,
        agent=agent,
        embryo_id="embryo_1",
    )

    md = agent.pushes[0]["metadata"]
    for key in ("embryo_id", "offsets", "labels", "pick", "method", "fft_scores"):
        assert key in md, f"metadata lost {key!r}, which the Operate caption reads"
    assert md["offsets"] == OFFSETS
    assert md["method"] == "fft"
    assert md["pick"] in md["labels"]
    assert len(md["labels"]) == len(OFFSETS)


@pytest.mark.asyncio
async def test_no_viz_server_is_not_an_error() -> None:
    """Headless runs (CLI, tests, agent with no UI attached) must not blow up."""
    agent = _RecordingAgent()
    agent.viz_server = None

    idx, method, _ = await hybrid_focus_selection(
        images=_sweep(sharp_at=1),
        offsets=OFFSETS,
        claude_vision=None,
        agent=agent,
        embryo_id="embryo_1",
    )
    assert method == "fft"
    assert idx == 1
    assert agent.pushes == []
