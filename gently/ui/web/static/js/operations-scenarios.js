/**
 * Operation Plan scenario fixtures — development and Chrome-MCP audit targets.
 *
 * Each entry is a plan object matching the real API schema returned by
 *   GET /api/operation_plan/{session_id} → { available, plan }
 * The `.plan` is what gets passed to the renderer. Active tactics carry a
 * `live` field (readouts + phases) that the API route merges from live
 * telemetry; here they are baked into the fixture.
 *
 * Scenario dev mode: load via ?scenario=<name> — ExperimentOverview reads
 * window.OPERATIONS_SCENARIOS[name] and skips all fetches.
 *
 * Scenarios:
 *   temp_strain        — scripted_protocol active (temp-change burst protocol)
 *   expression_onset   — reactive_monitor active (reporter rising)
 *   hatching_detect    — reactive_monitor active (watch=hatching, status=watching)
 *   transmission_survey — exclusive_burst active (brightfield only)
 *   decided_plan       — all planned, nothing run yet
 *   async_multi        — standing_timelapse per-embryo cadence + layered reactive_monitor
 *   idle               — null (no operation running)
 */
window.OPERATIONS_SCENARIOS = {

  /* ------------------------------------------------------------------ */
  temp_strain: {
    session_id: '20260628_1432_tempstrain_a',
    title: 'Temperature-strain run · E01',
    goal: 'Acquire volumes before, during, and after a +4 °C step to 32 °C; capture reporter response to heat stress.',
    tactics: [
      {
        id: 'ts-1', seq: 1,
        name: 'Monitor — low cadence',
        kind: 'standing_timelapse', state: 'done',
        scope: { mode: 'global' },
        rationale: 'Baseline acquisition before thermal perturbation.',
        structure: { cadence_s: 180 },
        live_bind: ['cadence'],
        live: { summary: '22 min · ended on signal' },
        relations: {}
      },
      {
        id: 'ts-2', seq: 2,
        name: 'Transmission burst — baseline',
        kind: 'exclusive_burst', state: 'done',
        scope: { mode: 'global' },
        rationale: 'Brightfield snapshot before setpoint change.',
        structure: { frames: 3, mode: 'brightfield' },
        live_bind: [],
        live: { summary: '3 bursts · brightfield' },
        relations: {}
      },
      {
        id: 'ts-3', seq: 3,
        name: 'Temp-change burst protocol',
        kind: 'scripted_protocol', state: 'active',
        scope: { mode: 'global' },
        rationale: 'Systematic volume capture before, during ramp, and after thermal lock. Laser off during ramp to limit phototoxicity.',
        structure: {
          phases: [
            { name: 'before', state: 'done',   count: '1/1 done' },
            { name: 'during', state: 'active',  count: '2 · awaiting lock' },
            { name: 'after',  state: 'todo',    count: '0/1' }
          ]
        },
        live_bind: ['temperature', 'current_burst'],
        live: {
          target: '→ 32.0 °C',
          summary: 'started 3m ago',
          desc: 'bursts before · setpoint change · bursts through ramp · bursts after lock — laser off',
          readouts: [
            {
              label: 'stage temp',
              bind: 'temperature',
              value: '29.4<span class="ops-u">→</span><span class="ops-set">32.0°C</span>',
              bar: 62
            },
            {
              label: 'current burst',
              value: '#3 <span class="ops-u">during</span>',
              sub: '60f · 1Hz · brightfield'
            }
          ],
          phases: [
            { name: 'before', state: 'done',   count: '1/1 done',          pips: ['before'] },
            { name: 'during', state: 'active',  count: '2 · awaiting lock', pips: ['during', 'during', 'pending'] },
            { name: 'after',  state: 'todo',    count: '0/1',               pips: ['pending'] }
          ]
        },
        relations: {}
      },
      {
        id: 'ts-4', seq: 4,
        name: 'Recovery monitor — low cadence',
        kind: 'standing_timelapse', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Resume gentle monitoring once temperature settles.',
        structure: { cadence_s: 180 },
        live_bind: ['cadence'],
        live: {
          summary: 'queued · 30 min after lock',
          desc: 'resume gentle monitoring once temperature settles'
        },
        relations: { after: ['ts-3'] }
      }
    ]
  },

  /* ------------------------------------------------------------------ */
  expression_onset: {
    session_id: '20260628_0915_onset_b',
    title: 'Reporter-onset watch · E04',
    goal: 'Detect first appearance of the fluorescent reporter; capture onset dynamics at high temporal resolution.',
    tactics: [
      {
        id: 'eo-1', seq: 1,
        name: 'Monitor — low cadence',
        kind: 'standing_timelapse', state: 'done',
        scope: { mode: 'global' },
        rationale: 'Baseline acquisition before signal appears.',
        structure: { cadence_s: 180 },
        live_bind: ['cadence'],
        live: { summary: '1h 40m · baseline' },
        relations: {}
      },
      {
        id: 'eo-2', seq: 2,
        name: 'Expression monitoring',
        kind: 'reactive_monitor', state: 'active',
        scope: { mode: 'global' },
        rationale: 'Accelerate cadence on signal, ramp 488 down on saturation, burst on stable structure.',
        structure: { watch: 'reporter onset', reaction: 'accelerate + ramp laser', status: 'watching' },
        live_bind: ['signal', 'cadence'],
        live: {
          target: 'reporter onset',
          summary: 'signal rising',
          desc: 'accelerate cadence on signal · ramp 488 down on saturation · burst on stable structure',
          readouts: [
            {
              label: 'reporter signal',
              value: '<span class="ops-set">rising</span>',
              sub: '+14% over 6 min',
              bar: 48
            },
            {
              label: 'cadence',
              value: '120s <span class="ops-u">→</span> 30s',
              sub: 'accelerated on onset'
            },
            {
              label: '488 power',
              value: '5% <span class="ops-u">→</span> 3%',
              sub: 'ramped to limit saturation'
            }
          ]
        },
        relations: {}
      },
      {
        id: 'eo-3', seq: 3,
        name: 'Burst on good structure',
        kind: 'exclusive_burst', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Capture a burst once the reporter pattern holds.',
        structure: { frames: 60, mode: 'fluorescence' },
        live_bind: [],
        live: {
          summary: 'queued · when structure stable',
          desc: 'capture a burst once the reporter pattern holds'
        },
        relations: { after: ['eo-2'] }
      }
    ]
  },

  /* ------------------------------------------------------------------ */
  hatching_detect: {
    session_id: '20260627_2210_hatch_c',
    title: 'Pre-hatching vigil · E11',
    goal: 'Detect and capture the hatching event; accelerate acquisition as hatching approaches.',
    tactics: [
      {
        id: 'hd-1', seq: 1,
        name: 'Pre-terminal monitoring',
        kind: 'reactive_monitor', state: 'active',
        scope: { mode: 'global' },
        rationale: 'Low cadence now; speed up as hatching approaches.',
        structure: { watch: 'hatching', reaction: 'accelerate near event', status: 'watching' },
        live_bind: ['cadence'],
        live: {
          target: 'hatching',
          summary: 'watching',
          desc: 'low cadence now · speed up as hatching approaches',
          readouts: [
            { label: 'est. time to hatch', value: '~38 min', sub: 'from motion + morphology' },
            { label: 'cadence',            value: '180s',    sub: 'will speed up near event' }
          ]
        },
        relations: {}
      },
      {
        id: 'hd-2', seq: 2,
        name: 'Hatching speedup',
        kind: 'standing_timelapse', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'High cadence through hatching.',
        structure: { cadence_s: 30 },
        live_bind: ['cadence'],
        live: {
          summary: 'queued · ~T-10 min',
          desc: 'high cadence through hatching'
        },
        relations: { after: ['hd-1'] }
      },
      {
        id: 'hd-3', seq: 3,
        name: 'Post-hatch monitor',
        kind: 'standing_timelapse', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Resume normal cadence after the event.',
        structure: { cadence_s: 120 },
        live_bind: ['cadence'],
        live: { summary: 'queued · after event' },
        relations: { after: ['hd-2'] }
      }
    ]
  },

  /* ------------------------------------------------------------------ */
  transmission_survey: {
    session_id: '20260628_1100_survey_a',
    title: 'Transmission survey · plate A',
    goal: 'Survey all embryos with brightfield only; no laser excitation.',
    tactics: [
      {
        id: 'srv-1', seq: 1,
        name: 'Transmission burst',
        kind: 'exclusive_burst', state: 'active',
        scope: { mode: 'global' },
        rationale: 'LED/brightfield bursts, no laser — DIC-like contrast.',
        structure: { frames: 30, mode: 'brightfield', phase: 'capturing' },
        live_bind: ['current_burst'],
        live: {
          summary: 'capturing',
          desc: 'LED/brightfield bursts, no laser — DIC-like contrast',
          readouts: [
            { label: 'bursts captured', value: '7',               sub: 'across 3 embryos' },
            { label: 'illumination',    value: 'LED · laser off', sub: 'brightfield' }
          ]
        },
        relations: {}
      },
      {
        id: 'srv-2', seq: 2,
        name: 'Volume at best plane',
        kind: 'oneshot', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Full volume capture at the best focal plane after operator review.',
        structure: { note: 'operator selects plane' },
        live_bind: [],
        live: { summary: 'queued · operator review' },
        relations: { after: ['srv-1'] }
      }
    ]
  },

  /* ------------------------------------------------------------------ */
  decided_plan: {
    session_id: '20260628_1500_tempstrain_b',
    title: 'Temperature-strain run · E02',
    goal: 'Repeat the thermal strain protocol on a second embryo cohort.',
    tactics: [
      {
        id: 'dp-1', seq: 1,
        name: 'Transmission burst — baseline',
        kind: 'exclusive_burst', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Brightfield baseline before any change.',
        structure: { frames: 3, mode: 'brightfield' },
        live_bind: [],
        live: {
          summary: 'queued · first',
          desc: 'brightfield baseline before any change'
        },
        relations: {}
      },
      {
        id: 'dp-2', seq: 2,
        name: 'Temp-change burst protocol',
        kind: 'scripted_protocol', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Thermal step to 30.0 °C with phased acquisition.',
        structure: {
          phases: [
            { name: 'before', state: 'todo', count: '0/1' },
            { name: 'during', state: 'todo', count: '0/3' },
            { name: 'after',  state: 'todo', count: '0/1' }
          ]
        },
        live_bind: ['temperature', 'current_burst'],
        live: { target: '→ 30.0 °C', summary: 'queued · second' },
        relations: { after: ['dp-1'] }
      },
      {
        id: 'dp-3', seq: 3,
        name: 'Recovery monitor',
        kind: 'standing_timelapse', state: 'planned',
        scope: { mode: 'global' },
        rationale: 'Low-cadence monitoring after temperature settles.',
        structure: { cadence_s: 180 },
        live_bind: ['cadence'],
        live: { summary: 'queued · last' },
        relations: { after: ['dp-2'] }
      }
    ]
  },

  /* ------------------------------------------------------------------ */
  async_multi: {
    session_id: '20260628_1630_async_multi',
    title: 'Async multi-embryo run · 4 embryos',
    goal: 'Per-embryo asynchronous acquisition with individual cadence phases; overlay a hatching watch on the two most advanced.',
    tactics: [
      {
        id: 'am-1', seq: 1,
        name: 'Async timelapse — per-embryo cadence',
        kind: 'standing_timelapse', state: 'active',
        scope: { mode: 'embryos', embryo_ids: ['E01', 'E02', 'E03', 'E04'] },
        rationale: 'Each embryo runs at its own cadence based on developmental stage and reporter state.',
        structure: {
          cadence_s: 120,
          per_embryo: [
            { embryo_id: 'E01', cadence_phase: 'normal', interval_s: 180 },
            { embryo_id: 'E02', cadence_phase: 'fast',   interval_s: 30  },
            { embryo_id: 'E03', cadence_phase: 'burst',  interval_s: 0   },
            { embryo_id: 'E04', cadence_phase: 'paused', interval_s: null }
          ]
        },
        live_bind: ['cadence'],
        live: {
          summary: 'running · 4 embryos',
          readouts: [
            { label: 'active embryos', value: '3 / 4', sub: 'E04 paused' },
            { label: 'cadence range',  value: '30–180s', sub: 'per-embryo mode' }
          ]
        },
        relations: {}
      },
      {
        id: 'am-2', seq: 2,
        name: 'Hatching watch — E01, E02',
        kind: 'reactive_monitor', state: 'active',
        scope: { mode: 'embryos', embryo_ids: ['E01', 'E02'] },
        rationale: 'Overlay a hatching detector on the two most advanced embryos.',
        structure: { watch: 'hatching', reaction: 'accelerate + alert', status: 'armed' },
        live_bind: ['signal'],
        live: {
          target: 'hatching',
          summary: 'armed · E01, E02',
          desc: 'watching for hatching onset on the two most advanced embryos',
          readouts: [
            { label: 'watch status', value: 'armed',      sub: 'no event yet' },
            { label: 'scope',        value: 'E01, E02',   sub: '2 of 4 embryos' }
          ]
        },
        relations: { layered_on: ['am-1'] }
      }
    ]
  },

  /* ------------------------------------------------------------------ */
  idle: null
};
