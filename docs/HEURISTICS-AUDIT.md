# Heuristics audit — where to use the model (as a typed-output function) instead

Codebase sweep (5 parallel scanners + synthesis) for heuristics that **fake
judgment** an LLM would do better — in the spirit of the genotype→channel
refactor (drop the lookup table, let the model infer, keep a typed provenance
record + confirm-when-unsure). The flip side — logic that **must stay
deterministic** (safety, math, calibration, transport) — is listed at the end so
we don't mistakenly LLM-ify it.

The unifying move for every candidate: **LLM with a typed structured-output
schema + provenance + a confirm/UNCERTAIN escape**, never free-text-then-parse.

## Model candidates (ranked)

### High value

1. **Hatching / time-to-stage prediction** — `organisms/celegans/developmental_tracker.py`
   *(the closest twin of genotype→channel; medium effort)*
   Three hardcoded 20 °C lookup tables (`STAGE_TIMING_20C`, `TIME_TO_HATCHING`,
   `TIMING_VARIABILITY`) plus magic `{HIGH:1.0, MEDIUM:1.5, LOW:2.0}` uncertainty
   fudge factors. Structurally **can't use the rig's actual temperature** (we run
   a TEC), the strain, or the embryo's observed progression rate. Let the model
   produce a calibrated, explained interval; **keep the literature table as a
   deterministic sanity bracket** and flag when the estimate falls outside it.
   → `{ predicted_minutes_to_hatching, low, high, basis, assumptions{temperature_c,strain,used_observed_rate}, confidence, reasoning }`

2. **Citation → PubMed query** — `harness/plan_mode/tools/research.py` (`_search_pmid`)
   A regex that only handles "Surname et al YEAR …" + six hand-rolled query-
   relaxation strategies + a stopword/word-position ladder that drops load-bearing
   nouns. The model parses the sloppy citation and proposes relaxed queries; **code
   keeps the deterministic esearch call and never fabricates a PMID.**
   → `{ author_last, year, journal, topic_keywords[], organism, pubmed_query, alt_queries[], confidence }`

3. **Lab-history retrieval** — `harness/plan_mode/tools/lab_context.py`, `harness/memory/interface.py`
   Semantic recall faked by substring-OR over query tokens (matches "we"/"before",
   misses every paraphrase). Feed the model the candidate records and have it
   **rank/select from provided ids only** (no fabrication). Read-only, no
   acquisition risk.
   → `{ matches:[{kind,id,summary,relevance,why_relevant}], answer }`

4. **Stage-label parse via 22-entry synonym dict** — `developmental_tracker.py` (`_parse_stage_name`)
   *(small effort, pure robustness win)* The Vision call already classifies; the
   brittleness is a plain-text `STAGE:/CONFIDENCE:` block scraped line-by-line, with
   off-vocabulary phrasings silently collapsing to `UNKNOWN` (which kills the
   downstream hatching prediction). Constrained-enum structured output deletes the
   parser + synonym table.
   → `{ stage: enum(...), confidence: enum(high|medium|low), is_transitional, reasoning }`

### Medium value (mostly small — fix the output contract, not the judgment)

5. **Calibration Vision calls** — `hardware/dispim/claude_client.py`
   Four Vision calls return positional free text recovered by `'yes' in first_line`
   / `re.search(r'\d+')` / first-valid-letter, with silent defaults (so "no, this is
   not yes…" reads as *yes*). Typed output deletes the parse + silent-default layer.

6. **ML architecture ranking** — `ml/architectures.py` (`get_suitable_architectures`)
   Hard feasibility gates (VRAM / dataset) are correct **and stay**; the `+2/+1/+1`
   point-score ranking that follows discards the per-arch prose. Let the model rank
   the *pre-filtered feasible set* (ids constrained to that set).

7. **Training label normalization** — `ml/data_loader.py` (`build_labels_from_store`)
   Class space built by exact-string identity over free-text human annotations —
   "1.5-fold" and "1.5 fold" become different classes. Model normalizes to the
   canonical staging vocabulary, flags novel/ambiguous ones.

### Lower value

8. **"Plan has a control?"** — `plan_mode/tools/validation.py` — substring scan of a
   6-word keyword set; a scientific judgment over the whole plan. Non-blocking
   warning → safe for the model.
9. **CGC HTML scraping** — `research.py` (`_cgc_search`) — positional multi-group
   regex over fetched HTML; structured extraction the model does better (HTTP GET
   stays code; **mark strain names low-confidence to avoid sending someone to order
   a hallucinated strain**).

### Cross-cutting batch (small each): typed output for the detector/verifier cluster
`harness/detection/verifier.py`, `app/detectors/hatching.py`,
`app/detectors/dopaminergic_signal.py`, `hardware/dispim/sam_detection.py` — all
already make the right model call but reconstruct the verdict via
`startswith`/regex-JSON-scraping with silent defaults. A batch move to native
structured output **strictly reduces parse-induced false negatives** without
touching the deterministic vote-tally/consensus/enum-dispatch downstream.

**Reference implementations already in the repo (imitate, don't change):**
`dopaminergic_signal`'s perceiver→classifier rubric (typed enums, UNCERTAIN
escape, conservative-on-tie) and onboarding's `_extract_with_llm` (typed
extraction, degrade-to-verbatim fallback).

## Keep deterministic (do NOT LLM-ify)
Safety, math, calibration, and transport — where a hallucinated value is unsafe
or breaks reproducibility:
- Laser-power safety limits + wavelength→MM-property map (`hardware/dispim/devices/optical.py`)
- SPIM trigger-timing arithmetic, piezo–galvo calibration, MM framing (`dispim/config.py`)
- Calibration prior EMA + R²≥0.75 slope-lock gate (`dispim/calibration.py`)
- SwitchBot GATT byte commands / status decoding (`hardware/switchbot.py`)
- Temperature setpoint bound [0,99.9] °C + stabilization I/O (`hardware/temperature.py`)
- Autofocus signal-processing, curve fitting, adaptive-sweep stop rules (`analysis/core.py`, `analysis/focus.py`)
- Classical-CV ROI detection + pixel→stage coordinate transforms (`detection.py`, `sam_detection.py` geometry)
- Timelapse rule dispatch + `confirm_timepoints` debounce + monotonic power ramp (`app/orchestration/timelapse.py`)
- Volume→b64 dark/flat calibration + fixed brightness scaling (`dopaminergic_signal._volume_to_b64` — deliberately non-adaptive)
- Wake-router debounce/throttle/stage-transition gate (`app/wake_router.py`)
- Plan hardware limits, detector-preset membership, dependency-cycle DFS, stage-order normalization (`plan_mode/tools/validation.py`)
- Ensemble vote tally + 0.70 quorum / unanimity consensus (`detection/verifier.py`)
- ML metric/aggregation math: confusion matrix, F1, federated averaging (`ml/evaluation.py`, `federated.py`)
- Core imaging geometry (max-projection, crop bounds, Euler rotations) + UI event reduction/routing/security (`core/imaging.py`, `ui/web/*`)
- Device-state SSE watchdog/staleness timers (`app/device_state_monitor.py`)
- Reference-type dispatch (PMID/DOI/URL by canonical syntax), `os.path.isfile` checks (`research.py`)

## Note
`gap_assessment.conversation_weight` (the 0.25/0.1/0.05 readiness scalar) is now
largely **vestigial** — it only returns 'heavy' (lab onboarding) or 'none' — so
it's not worth an API call. Left off the candidate list.
