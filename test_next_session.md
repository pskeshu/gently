# Test Plan for Next Imaging Session

**Date:** December 2024 (Ryan mounting embryos)
**Purpose:** Validate fixes from overnight timelapse analysis

---

## Pre-Session Tests (Before Imaging)

### 1. Test `/detectors` Display Fix
```
> /detectors
```
**Expected:** Should show actual Runs and Detections counts (not 0/0 if hatching detector was used before)

**If fresh session:** Enable hatching detector first, then verify counts increment after detection runs.

---

### 2. Test `add_detector` Tool Fix

Ask copilot to add a custom detector:
```
> Add a detector called "no_embryo_visible" that detects when the embryo is no longer in the field of view.
  It should trigger automatically if detected.
```

**Expected:**
- Should succeed (previously failed with "technical issue")
- Should return: "Added detector 'no_embryo_visible' with action mode 'auto'"

**Verify:**
```
> /detectors
```
Should show the new detector in the list.

---

### 2b. Test Hatching Detector Auto-Stop (NEW FIX)

Enable the hatching detector and verify it will stop timelapse:
```
> Enable the hatching detector in auto mode
```

**Expected:**
- Should return message indicating it will stop timelapse when detected
- e.g., "Enabled 'hatching' detector... (will stop timelapse)"

**Or explicitly:**
```
> Add a hatching detector that stops the timelapse when hatching is detected
```

**Key Test:** Start a timelapse with hatching detector enabled. When hatching is detected, timelapse should STOP automatically (not continue for 40+ more timepoints like before).

---

### 3. Test `query_timeline_events` Tool

Ask copilot to query events:
```
> Show me the recent timeline events
```
or
```
> When was hatching detected in the last session?
```

**Expected:**
- Should return formatted list of events with timestamps
- If querying detection events: should show detector name, embryo, timepoint, confidence

---

## During Timelapse Tests

### 4. Test Context Injection (State Awareness)

**Setup:** Start a timelapse, let it run for a few timepoints.

**Test A - During acquisition:**
```
> What's the current status?
```
**Expected:** Copilot should know timelapse is running without you telling it.

**Test B - After stopping/completing:**
Wait until timelapse completes or stop it. Then start a new conversation:
```
> The hatching detector was triggered, what should I do?
```
**Expected:**
- Copilot should NOT offer to "stop the timelapse" if it already completed
- Should acknowledge timelapse is done and provide relevant next steps

---

### 5. Test Detection Reasoning Storage

**Setup:** Run with hatching detector enabled.

After a detection fires:
```
> What did the detector observe when hatching was detected?
```

**Expected:** Should be able to describe what the Vision API saw (reasoning is now stored)

---

## Quick Validation Checklist

| Test | Command/Action | Pass? |
|------|----------------|-------|
| `/detectors` shows counts | `/detectors` | [ ] |
| Add custom detector | "Add detector no_embryo_visible..." | [ ] |
| Query timeline events | "Show recent events" | [ ] |
| Context awareness (running) | Ask status during timelapse | [ ] |
| Context awareness (completed) | Ask about completed timelapse | [ ] |
| Detection reasoning | Ask what detector observed | [ ] |

---

## If Tests Fail

Note the exact error message and copilot response. Key things to capture:
1. Full error text
2. What command/question was asked
3. Current timelapse state (running/idle/completed)
4. Check viz server events log for comparison

---

## NEW FEATURES TO TEST

### 6. Test Composite Stop Conditions (OR Logic)

Start a timelapse with a duration limit, then add hatching detection:
```
> Start a timelapse with a 10 hour duration limit for embryo1
```

Then mid-run:
```
> Add a hatching stop condition to embryo1
```

**Expected:**
- Should succeed and show: "Stop conditions: 10h duration OR hatching"
- Timelapse should stop on EITHER condition (whichever comes first)

**Alternative:** Test the composite syntax directly:
```
> Start a timelapse for embryo1 with stop condition "hatching|duration:10h"
```

---

### 7. Test Challenger/Verifier Agent

**Setup:** Enable hatching detector in AUTO mode with stop_timelapse:
```
> Enable hatching detector in auto mode
```

When hatching is detected, the verifier should run automatically:

**Expected behavior:**
- Console should show: `[VERIFIER] Running verification for hatching detection on <embryo_id>`
- If all 3 strategies agree (consensus=True):
  - Console: `[VERIFIER] Verification passed, proceeding with stop`
  - Timelapse stops for that embryo
- If strategies disagree (consensus=False):
  - Console: `[VERIFICATION FAILED] hatching detection on <embryo_id>`
  - Shows which strategies disagreed
  - Falls back to RECOMMEND mode (asks user to confirm)

**To manually test verification failure:**
If you have a borderline case or false positive, the verifier should catch it and NOT auto-stop.

---

### 8. Test Verification Event Logging

After a detection with verification:
```
> Show me recent timeline events
```

**Expected:** Should see verification results in the event log:
- `verification_completed` action with full verification details
- If failed: `verification_failed` action with reason

---

## Quick Validation Checklist (Updated)

| Test | Command/Action | Pass? |
|------|----------------|-------|
| `/detectors` shows counts | `/detectors` | [ ] |
| Add custom detector | "Add detector no_embryo_visible..." | [ ] |
| Query timeline events | "Show recent events" | [ ] |
| Context awareness (running) | Ask status during timelapse | [ ] |
| Context awareness (completed) | Ask about completed timelapse | [ ] |
| Detection reasoning | Ask what detector observed | [ ] |
| Composite stop conditions | Add condition mid-run | [ ] |
| Verifier runs on AUTO | Check console for [VERIFIER] | [ ] |
| Verifier blocks false positive | Consensus failure fallback | [ ] |

---

## Notes

- Context summary uses Haiku and is cached for 5 minutes
- First message after 5 min gap will regenerate context (slight delay)
- Timeline events persist in `timeline.jsonl` in session storage
- Verifier uses claude-haiku-4-5-20251001 for speed/cost efficiency
- Verifier runs 3 strategies in parallel: adversarial, independent, temporal
