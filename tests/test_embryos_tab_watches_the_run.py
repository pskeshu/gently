"""The Embryos tab says what the run is and shows its frames as the experiment.

"the embryos tab - it has not been looked at in ages - can you see how we can
improve that view? especialy now that bf images are being sent there."

A brightfield-only run (volumes off, the brightfield channel on) has no embryos; the
tab used to call it "No active timelapse" under a strip of live frames. Now the
status comes from the run, the stats from what kind of run it is, the frames
get the stage, and the embryo rail is tiles on the stage ramp, not emoji.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
EMBRYOS = (WEB / "static" / "js" / "embryos.js").read_text(encoding="utf-8")
STAGE = (WEB / "static" / "js" / "panels" / "overview-stage.js").read_text(encoding="utf-8")
MAIN_CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")


def _fn(src: str, name: str, n: int = 2200) -> str:
    return src[src.index(name) :][:n]


class TestTheHeaderSaysWhatRuns:
    def test_status_comes_from_the_run_not_the_embryo_count(self):
        badge = _fn(EMBRYOS, "renderStatusBadge() {")
        assert "Object.keys(this.state.embryos).length === 0" not in badge
        assert "textEl.textContent = 'No run';" in badge
        assert "`${word} · brightfield`" in badge
        assert "status === 'COMPLETED' ? 'Done'" in badge

    def test_the_tab_asks_the_run_what_kind_it_is(self):
        assert "fetch('/api/devices/timelapse/status')" in EMBRYOS
        kind = _fn(EMBRYOS, "runKind() {", 900)
        assert "r.volumes === false && r.bf && r.bf.enabled) return 'brightfield'" in kind
        assert (
            "if (this._bfFrames.length && !embryos) return 'brightfield'" in kind
        )  # at rest, by what it holds

    def test_brightfield_stats_are_frames_cadence_next_and_references(self):
        summary = _fn(EMBRYOS, "renderSummary() {", 3200)
        for needle in ("'frames'", "'cadence'", "'next frame'", "'dark/flat'", "'elapsed'"):
            assert needle in summary, needle
        assert "'TP'" not in summary and "'Dur'" not in summary  # nouns, not abbreviations


class TestTheStage:
    def test_the_frames_get_the_stage_and_the_strip_is_its_folded_form(self):
        assert 'id="embryos-overview"' in INDEX and 'id="bf-strip-open"' in INDEX
        assert 'id="bf-viewer"' not in INDEX  # no modal
        strip = _fn(EMBRYOS, "renderBfStrip() {", 2400)
        assert "kind === 'brightfield' || this._overviewOpen" in strip
        assert "OverviewStage.mount('embryos-overview'" in EMBRYOS
        assert "is-overview-only" in strip and ".view-default.is-overview-only" in MAIN_CSS

    def test_the_stage_scrubs_plays_follows_and_corrects(self):
        for needle in (
            "setPointerCapture",  # drag to scrub
            "if (ev.buttons)",  # hover previews, only a drag moves the frame
            "'ArrowRight' || k === '.'",  # keys
            "ev.shiftKey ? 10 : 1",
            "FOLLOWING NEWEST",  # explicit follow state
            "if (byUser) follow = false;",
            "corrected=1",  # server-side correction
            "Compare with first",  # A/B on the same pixels
            "onTakeReferences",  # the prompt when none exist
            "ResizeObserver",  # drawn when it has a width, not before
        ):
            assert needle in STAGE, needle
        assert ".overview-stage[hidden] { display: none; }" in MAIN_CSS

    def test_the_frame_list_carries_light_exposure_and_correctability(self):
        remember = _fn(EMBRYOS, "async refreshBfStrip() {", 900)
        for needle in ("exposure_ms", "light", "led_intensity_pct", "correctable"):
            assert needle in remember, needle


class TestTheTiles:
    def test_tiles_replace_the_emoji_rail(self):
        card = _fn(EMBRYOS, "renderEmbryoCard(embryo) {", 3600)
        assert 'class="embryo-tile' in card and 'class="tile-dot ' in card
        assert 'class="stage-bar"' in card and "stageColor(r.stage)" in card
        assert 'class="tile-thumb"' in card and "/projection?embryo=" in card
        assert "getStageIcon" not in card  # no emoji on the tile
        assert "_wireTileScrub(" in EMBRYOS
        assert ".embryo-tile.is-selected" in MAIN_CSS and ".tile-dot.error" in MAIN_CSS

    def test_the_drawer_header_is_one_sentence_with_a_swatch(self):
        panel = _fn(EMBRYOS, "renderReasoningPanel() {", 9000)
        assert 'class="reasoning-sentence"' in panel
        assert "since t${since}" in panel
        assert "evals</span>" not in panel  # the transitions · evals · tp triple is gone
        assert 'type="button" class="quick-jump-badge' in panel

    def test_eval_dots_are_buttons_on_the_ramp(self):
        dots = _fn(EMBRYOS, 'class="eval-dot ${stageClass}', 600)
        assert dots.startswith('class="eval-dot')
        assert (
            "eval-dot-swatch" in EMBRYOS
            and 'return `<button type="button" class="eval-dot' in EMBRYOS
        )


class TestTheCopy:
    def test_states_say_what_is_true_and_what_happens_next(self):
        empty = _fn(EMBRYOS, "renderSmartEmptyState(type) {", 3000)
        for needle in (
            "Overview only.",
            "No embryos registered.",
            "No stage calls yet.",
            "No run.",
            "Run started.",
            "Open Operate",
        ):
            assert needle in empty, needle
        for dead in (
            "Go to Calibration",
            "Typical first detection",
            "Click on any",
            "Syncing with experiment",
            "No active timelapse",
        ):
            assert dead not in EMBRYOS, dead
        assert "No active timelapse" not in INDEX and "Embryo Monitoring" not in INDEX
        assert ">Watch<" in INDEX and ">Table<" in INDEX

    def test_the_stage_keys_agree_across_the_maps(self):
        assert "const stageKey" in (WEB / "static" / "js" / "stage-colors.js").read_text(
            encoding="utf-8"
        )
        assert "this.STAGE_TIMING[this._sk(stage)]" in EMBRYOS
        assert "this.STAGE_ORDINAL[this._sk(item.stage)]" in EMBRYOS
        assert (
            "${stageColor}" not in EMBRYOS
        )  # the function was once interpolated instead of called


class TestTheVolumeStatIsNotTheRoundCounter:
    """ "the gui in the embryos tab - it is showing -1 round"

    `current_round` is the orchestrator's acquisition counter: it starts at -1
    and bumps once per embryo volume, so it is neither a round nor a count.
    The tab shows the volumes taken instead and never that counter.
    """

    def test_the_summary_counts_volumes_not_the_round_counter(self):
        summary = _fn(EMBRYOS, "renderSummary() {", 3600)
        assert "stat(this.state.totalTimepoints, 'volumes')" in summary
        assert "r.current_round" not in summary
        assert "'round')" not in summary


class TestTheEmbryoHasTheSamePlayerAsTheOverview:
    """ "there is no nice timelapse viewing features that were there for the
    brightfield images. the show latest feature etc. the player was rather nice."

    The overview's stage is a factory now; the selected embryo mounts its own
    instance on its projections, with the bar coloured by each timepoint's
    call, following newest as the run goes, and the detail pane following it.
    """

    def test_the_stage_is_a_factory_and_the_overview_keeps_its_instance(self):
        assert "function createOverviewStage()" in STAGE
        assert "const OverviewStage = createOverviewStage();" in STAGE
        assert "OverviewStage.mount('embryos-overview'" in EMBRYOS

    def test_the_selected_embryo_gets_a_stage_of_its_projections(self):
        panel = _fn(EMBRYOS, "renderReasoningPanel() {", 8000)
        assert 'id="embryo-stage"' in panel
        assert "this._mountEmbryoStage(embryo.embryoId, !keepStage)" in panel
        mount = _fn(EMBRYOS, "_mountEmbryoStage(embryoId, fresh) {", 2600)
        assert "createOverviewStage()" in mount
        assert "corrected: false" in mount and "references: false" in mount
        frames = _fn(EMBRYOS, "_embryoFrames(embryoId) {", 1400)
        assert "/projection?embryo=" in frames and "scalable: false" in frames

    def test_the_stage_survives_a_re_render(self):
        panel = _fn(EMBRYOS, "renderReasoningPanel() {", 8000)
        assert "panel.querySelector('#embryo-stage').replaceWith(keepStage)" in panel

    def test_the_detail_follows_the_stage_and_the_stage_follows_a_chosen_call(self):
        mount = _fn(EMBRYOS, "_mountEmbryoStage(embryoId, fresh) {", 2600)
        assert "onFrame:" in mount and "this._syncDetailToFrame(i + 1)" in mount
        opened = _fn(
            EMBRYOS, "openDetailPanel(detectorName, timepoint, userInitiated = false) {", 1500
        )
        assert "this._embryoStage.go(timepoint - 1, true)" in opened
        assert "this._embryoStage.reset()" in _fn(EMBRYOS, "selectEmbryo(embryoId) {", 900)

    def test_the_arrows_step_the_stage_once_not_the_stage_and_the_detail(self):
        keys = _fn(EMBRYOS, "setupKeyboardNavigation() {", 1200)
        assert "document.querySelector('#embryo-stage .ov')" in keys
        assert "if (stage && stage.offsetParent) return;" in keys

    def test_with_two_stages_the_keys_go_to_the_one_last_touched(self):
        assert "if (_activeStage && _activeStage !== api) return;" in STAGE
        assert "host.addEventListener('pointerdown', () => { _activeStage = api; });" in STAGE

    def test_the_rail_cannot_run_under_the_panel(self):
        """ "here it is like something else is rendered on top kinda feeling":
        each tile sized itself to its content, 320 px in a 183 px rail."""
        cards = MAIN_CSS[MAIN_CSS.index(".embryo-cards { display: grid;") :][:120]
        assert "grid-template-columns: minmax(0, 1fr)" in cards
