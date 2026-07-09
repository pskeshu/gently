export const meta = {
  name: 'session-postmortem',
  description: 'Forensic multi-lens analysis of a completed Gently run: fan out lenses over the session artifacts, adversarially verify the load-bearing findings, and return a draft post-mortem report.',
  phases: [
    { title: 'Read', detail: '6 parallel lenses: chat/decisions, raw event logs, perception, drift/termination, temperature, code-grounded bugs' },
    { title: 'Verify', detail: 'adversarially verify each high/critical negative finding against the data/code' },
    { title: 'Synthesize', detail: 'assemble a schema-conformant draft report from lenses + verdicts' },
  ],
}

// args: a session-directory path (string) OR { session_dir, repo }
const SESSION = typeof args === 'string' ? args : ((args && args.session_dir) || '')
const REPO = (args && args.repo) || 'C:/Users/dispim/Documents/github/gently'
const SCHEMA = `${REPO}/.claude/skills/session-postmortem/references/report_schema.yaml`
if (!SESSION) log('WARNING: no session_dir passed as args — lenses will have nothing to read.')

const PRE = `You are a forensic analyst producing a post-mortem of a completed Gently light-sheet microscopy run.
Session directory: ${SESSION}
Repo (for code-grounded checks): ${REPO}
ENCODING: default Python encoding here is cp1252 — always io.open(path, encoding='utf-8') and keep stdout ASCII (errors='replace'). Aggregate large JSONL with python; do NOT Read multi-MB files wholesale.
Be concrete and evidence-first: cite artifact path + key (line / timepoint). Return ONLY the structured object.`

const LENS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    lens: { type: 'string' },
    summary: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: {
        title: { type: 'string' },
        audience: { type: 'string', enum: ['agent', 'harness', 'hardware', 'biology'] },
        category: { type: 'string' },
        severity: { type: 'string', enum: ['info', 'low', 'medium', 'high', 'critical'] },
        polarity: { type: 'string', enum: ['negative', 'positive'] },
        detail: { type: 'string' },
        evidence: { type: 'array', items: { type: 'object', additionalProperties: false,
          properties: { artifact: { type: 'string' }, ref: { type: 'string' }, note: { type: 'string' } },
          required: ['artifact', 'ref'] } },
        proposed_fix: { type: 'string' },
        proposed_eval: { type: 'string' },
      }, required: ['title', 'audience', 'severity', 'polarity', 'detail', 'evidence'] } },
    metrics: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { key: { type: 'string' }, value: { type: 'string' }, unit: { type: 'string' }, source_artifact: { type: 'string' } },
      required: ['key', 'value'] } },
    data_gaps: { type: 'array', items: { type: 'string' } },
  },
  required: ['lens', 'summary', 'findings'],
}

const LENSES = [
  { key: 'chat-decisions', prompt: `${PRE}\n\nLENS: Agent chat & decision quality. Read chat_display.json (roles user/agent/tool/autonomous_start[trigger]/autonomous[text]) and conversation.json (full untruncated reasoning + exact tool I/O; decisions.jsonl is often empty — note it as a data gap). Assess: how autonomy was enabled; decision quality; in-session learning/self-correction; overconfident/unverified claims; setup confusion. Capture positive findings (good calls) too.` },
  { key: 'raw-events', effort: 'medium', prompt: `${PRE}\n\nLENS: Ground-truth event timeline. Aggregate transcript.jsonl, timeline.jsonl, events.jsonl (JSONL; vocab = EventType). Reconstruct start/end/duration, exactly how/when autonomy was triggered, tool errors with context, autonomous wake count + trigger breakdown, multi-window/control-handoff, and anything not surfaced in the display chat.` },
  { key: 'perception', effort: 'medium', prompt: `${PRE}\n\nLENS: Perception behavior. Aggregate embryos/*/predictions.jsonl (+ perception_runs.yaml). Per embryo: stage trajectory (run-length), oscillation / backward transitions, first no_object timepoint + wall-time + preceding stage (to test 'alive vs endpoint'), whether no_object onset was synchronous across embryos (global cause), and sample the perceiver's own no_object reasoning text.` },
  { key: 'drift-termination', prompt: `${PRE}\n\nLENS: Focus-drift & termination forensics. Sources: timelapse.yaml (completion_reason, total_exposure_ms, no_object_since_timepoint), embryos/*/embryo.yaml (calibration + focus_history), predictions (last real stage). Determine genuine endpoints vs false terminations (cross-ref last stage), quantify per-embryo drift magnitude+direction, photodose ratios (mid-run vs final), and whether refocus propagated to acquisition. Rank false-termination + non-sticking-refocus highest.` },
  { key: 'temperature', effort: 'medium', prompt: `${PRE}\n\nLENS: Temperature telemetry. Aggregate temperature.jsonl (fields t/water_c/setpoint_c/state). Confirm REAL vs '(SIM)'; sampling rate/span/coverage; water_c min/mean/max/stdev; setpoint ever active. CRITICAL: if temp was simulated and/or flat, any 'thermal drift' attribution in the chat is UNSUPPORTED — state this explicitly.` },
  { key: 'code-bugs', prompt: `${PRE}\n\nLENS: Code-grounded confirmation of harness defects the run exposed. Work in ${REPO}/gently. For each tool error / bad behavior seen in the chat, find the responsible file:line, root cause, and a minimal fix. Especially: no_object auto-termination for calibration role (roles.py / timelapse.py _check_stop_condition), whether fine_focus writes back to the acquisition center (focus_tools / state / timelapse acquire), any missing-method tool errors, and tool-arg type coercion in the ToolRegistry. Report file:line + fix + severity by user impact.` },
]

phase('Read')
const lenses = (await parallel(LENSES.map(L => () =>
  agent(L.prompt, { label: `read:${L.key}`, phase: 'Read', schema: LENS_SCHEMA, ...(L.effort ? { effort: L.effort } : {}) })
))).filter(Boolean)

// Load-bearing claims = the negative findings that would drive code changes. Verify each adversarially.
const claims = lenses.flatMap(l => (l.findings || [])
  .filter(f => f.polarity === 'negative' && (f.severity === 'high' || f.severity === 'critical' || f.severity === 'medium'))
  .map(f => ({ title: f.title, audience: f.audience, detail: f.detail, evidence: f.evidence })))
log(`Read done: ${lenses.length} lenses, ${claims.length} load-bearing claims to verify`)

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    title: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'PARTLY_CONFIRMED', 'REFUTED', 'UNVERIFIABLE'] },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    evidence: { type: 'string' },
    correction: { type: 'string' },
  },
  required: ['title', 'verdict', 'confidence', 'evidence'],
}

phase('Verify')
const verdicts = (await parallel(claims.map((c, i) => () =>
  agent(`${PRE}\n\nYou are an ADVERSARIAL verifier. Try to REFUTE or tighten this finding using the actual data/code. Default to skepticism; only CONFIRMED if evidence is solid. Report exact numbers/files/timepoints, and a correction if the finding is imprecise.\n\nFINDING: ${c.title}\nDETAIL: ${c.detail}\nCLAIMED EVIDENCE: ${JSON.stringify(c.evidence)}`,
    { label: `verify:${i + 1}`, phase: 'Verify', schema: VERDICT_SCHEMA })
))).filter(Boolean)

phase('Synthesize')
const report_draft = await agent(
  `${PRE}\n\nAssemble a DRAFT post-mortem report. Read the schema at ${SCHEMA} and conform to it exactly (schema_version 1).
Inputs:
LENSES: ${JSON.stringify(lenses)}
VERDICTS: ${JSON.stringify(verdicts)}
Rules: merge duplicate findings across lenses; set each finding's verdict/confidence from the matching VERDICT (map CONFIRMED->CONFIRMED, PARTLY_CONFIRMED->PLAUSIBLE, REFUTED->drop the finding, apply any correction into detail); derive a stable fingerprint 'audience:category:signature'; auto_actionable=true only when verdict=CONFIRMED and confidence=high; order findings most-severe first; keep positive findings; fill run_summary/timeline/metrics/action_items/data_gaps. Leave generated_at and generator.* as placeholders for the caller to stamp. Return ONLY the report object as JSON.`,
  { label: 'synthesize:report', phase: 'Synthesize' })

return { report_draft, lenses, verdicts }
