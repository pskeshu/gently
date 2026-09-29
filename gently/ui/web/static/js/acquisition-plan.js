/**
 * The acquisition plan — what a timelapse is, said once.
 *
 * A classical MDA asks for positions × channels × z × time in a fixed
 * order. Here two of those axes are already answered by the instrument and
 * are not asked: positions are the embryo roster, and z comes from each
 * embryo's calibration (the start route refuses an uncalibrated one). What
 * is left to ask is what this object holds — cadence, the two channels,
 * and how it ends — and it is said back as one sentence a biologist checks
 * before pressing Start:
 *
 *   Every 5 min: SPIM volumes (50 slices · 10 ms · 488 and 561) of 4 embryos
 *   + one DIC overview per round from the centroid · until stopped; embryo 2
 *   at hatching.
 *
 * Pure. Nothing here touches the DOM or the network: operate.js reads the
 * form into a plan, this module says it and turns it into the start
 * request, and the same object is what a saved tactic will store and the
 * agent will seed. One plan, three readers.
 */
const AcquisitionPlan = (() => {
    'use strict';

    /** How a run, or one embryo, ends. Keys are what the orchestrator parses. */
    const STOP_KINDS = {
        manual: { label: 'until stopped', needs: null },
        timepoints: { label: 'after N timepoints', needs: 'count' },
        duration: { label: 'after N hours', needs: 'hours' },
        hatching: { label: 'at hatching', needs: null },
        comma: { label: 'at comma stage', needs: null },
        all_test_hatched: { label: 'when every test embryo has hatched', needs: null },
    };

    // The light the DIC overview is taken under. The bottom camera drives
    // none of its own, so the plan has to say.
    const DIC_LIGHTS = {
        room: { label: 'room light', says: 'under the room light' },
        led: { label: 'LED', says: 'under the LED' },
        none: { label: 'as it is', says: 'in the light as it is' },
    };

    const num = (v, fallback) => {
        const n = Number(v);
        return Number.isFinite(n) ? n : fallback;
    };

    /** The combined spec string the orchestrator parses: "timepoints:12", "duration:6h". */
    function stopSpec(kind, value) {
        const k = STOP_KINDS[kind] ? kind : 'manual';
        if (k === 'timepoints') return `timepoints:${Math.max(1, Math.round(num(value, 1)))}`;
        if (k === 'duration') return `duration:${Math.max(0.1, num(value, 1))}h`;
        return k;
    }

    /** The same stop, in words. */
    function stopWords(kind, value) {
        const k = STOP_KINDS[kind] ? kind : 'manual';
        if (k === 'timepoints') {
            const n = Math.max(1, Math.round(num(value, 1)));
            return `after ${n} timepoint${n === 1 ? '' : 's'}`;
        }
        if (k === 'duration') {
            const h = Math.max(0.1, num(value, 1));
            return `after ${h} h`;
        }
        return STOP_KINDS[k].label;
    }

    /**
     * Build a plan from the form's values. Every field has a default, so a
     * form with nothing touched is still a complete, sayable plan.
     *
     * @param {object} f
     *   interval        number, in `intervalUnit`
     *   intervalUnit    's' | 'min'
     *   slices, exposureMs, laserConfig
     *   dic             bool
     *   dicEveryRounds  integer ≥ 1
     *   dicPosition     'centroid' | 'here'
     *   dicPin          {x, y} | null — the stage position captured for 'here'
     *   dicExposureMs   number | null
     *   stopKind, stopValue
     *   overrides       [{embryoId, kind, value}]  per-embryo terminations
     *   monitoringMode  string
     */
    function fromForm(f) {
        f = f || {};
        const unit = f.intervalUnit === 'min' ? 60 : 1;
        const intervalSeconds = Math.max(1, num(f.interval, 120) * unit);
        const overrides = (f.overrides || [])
            .filter(o => o && o.embryoId && o.kind && o.kind !== 'default')
            .map(o => ({ embryoId: String(o.embryoId), kind: o.kind, value: o.value == null ? null : o.value }));
        return {
            intervalSeconds,
            spim: {
                slices: Math.max(1, Math.round(num(f.slices, 50))),
                exposureMs: Math.max(1, num(f.exposureMs, 10)),
                laserConfig: f.laserConfig || null,
            },
            dic: {
                enabled: !!f.dic,
                everyRounds: Math.max(1, Math.round(num(f.dicEveryRounds, 1))),
                position: f.dicPosition === 'here' ? 'here' : 'centroid',
                pin: f.dicPin && Number.isFinite(f.dicPin.x) && Number.isFinite(f.dicPin.y)
                    ? { x: f.dicPin.x, y: f.dicPin.y } : null,
                exposureMs: f.dicExposureMs != null && f.dicExposureMs !== '' ? num(f.dicExposureMs, null) : null,
                light: DIC_LIGHTS[f.dicLight] ? f.dicLight : 'room',
            },
            stop: { kind: STOP_KINDS[f.stopKind] ? f.stopKind : 'manual', value: f.stopValue == null ? null : f.stopValue },
            overrides,
            monitoringMode: f.monitoringMode || 'idle',
        };
    }

    /** What is wrong with a plan, in the operator's words. Empty when nothing. */
    function validate(plan, subjectIds) {
        const problems = [];
        if (!subjectIds || !subjectIds.length) problems.push('No embryos to image — mark some first.');
        if (!(plan.intervalSeconds > 0)) problems.push('The interval must be positive.');
        if (plan.dic.enabled && plan.dic.position === 'here' && !plan.dic.pin) {
            problems.push('DIC overview: no stage position captured for "here" yet.');
        }
        if (plan.stop.kind === 'timepoints' && !(num(plan.stop.value, 0) >= 1)) {
            problems.push('Stop after how many timepoints?');
        }
        if (plan.stop.kind === 'duration' && !(num(plan.stop.value, 0) > 0)) {
            problems.push('Stop after how many hours?');
        }
        const ids = new Set(subjectIds || []);
        plan.overrides.forEach(o => {
            if (!ids.has(o.embryoId)) problems.push(`${o.embryoId} is not in this run.`);
        });
        return problems;
    }

    /** The start request. Only what the plan says goes on the wire. */
    function toPayload(plan, subjectIds) {
        const body = {
            embryo_ids: subjectIds.slice(),
            interval_seconds: plan.intervalSeconds,
            stop_condition: stopSpec(plan.stop.kind, plan.stop.value),
            monitoring_mode: plan.monitoringMode || 'idle',
            num_slices: plan.spim.slices,
            exposure_ms: plan.spim.exposureMs,
        };
        if (plan.spim.laserConfig) body.laser_config = plan.spim.laserConfig;
        if (plan.dic.enabled) {
            body.dic = {
                enabled: true,
                // The orchestrator schedules the overview on its own clock, in
                // seconds. "Every N rounds" is the operator's unit.
                every_seconds: plan.dic.everyRounds * plan.intervalSeconds,
                position: plan.dic.position === 'here' && plan.dic.pin
                    ? { x: plan.dic.pin.x, y: plan.dic.pin.y } : null,
                exposure_ms: plan.dic.exposureMs,
                light: plan.dic.light,
            };
        }
        if (plan.overrides.length) {
            body.stop_conditions = {};
            plan.overrides.forEach(o => { body.stop_conditions[o.embryoId] = stopSpec(o.kind, o.value); });
        }
        return body;
    }

    function intervalWords(seconds) {
        if (seconds % 3600 === 0 && seconds >= 3600) { const h = seconds / 3600; return `${h} h`; }
        if (seconds % 60 === 0 && seconds >= 60) return `${seconds / 60} min`;
        return `${seconds} s`;
    }

    /**
     * The plan as a sentence.
     *
     * @param plan       from fromForm()
     * @param subjects   [{id, label}] — the embryos this run will image
     */
    function describe(plan, subjects) {
        const n = (subjects || []).length;
        const labelOf = id => {
            const s = (subjects || []).find(e => e.id === id);
            return s ? s.label : id;
        };
        const spimBits = [`${plan.spim.slices} slices`, `${plan.spim.exposureMs} ms`];
        if (plan.spim.laserConfig) spimBits.push(plan.spim.laserConfig);
        const who = n === 0 ? 'no embryos' : n === 1 ? `embryo ${labelOf(subjects[0].id)}` : `${n} embryos`;
        let s = `Every ${intervalWords(plan.intervalSeconds)}: SPIM volumes (${spimBits.join(' · ')}) of ${who}`;
        if (plan.dic.enabled) {
            const every = plan.dic.everyRounds === 1 ? 'per round' : `every ${plan.dic.everyRounds} rounds`;
            const from = plan.dic.position === 'here' && plan.dic.pin
                ? `from ${plan.dic.pin.x.toFixed(0)}, ${plan.dic.pin.y.toFixed(0)}`
                : 'from the centroid';
            s += ` + one DIC overview ${every} ${from}, ${DIC_LIGHTS[plan.dic.light].says}`;
        }
        s += ` · ${stopWords(plan.stop.kind, plan.stop.value)}`;
        if (plan.overrides.length) {
            s += '; ' + plan.overrides
                .map(o => `${labelOf(o.embryoId)} ${stopWords(o.kind, o.value)}`)
                .join(', ');
        }
        return s + '.';
    }

    /** "timepoints:12" → {kind, value}; "duration:6h" → {kind, value}; "hatching+3" → {kind}. */
    function parseStopSpec(spec) {
        const s = String(spec || 'manual').trim().toLowerCase().replace(/\+\d+$/, '');
        if (s.startsWith('timepoints:')) return { kind: 'timepoints', value: Number(s.split(':')[1]) || 1 };
        if (s.startsWith('duration:')) return { kind: 'duration', value: Number(s.split(':')[1].replace(/h$/, '')) || 1 };
        // The agent's stage-based spec, "stages(hatched,hatching)": the pane
        // offers those endings by name. An unknown stage set stays manual,
        // and the sentence says "until stopped" rather than inventing an end.
        const m = s.match(/^stages?\(([^)]*)\)$/);
        if (m) {
            const stages = m[1].split(',').map(x => x.trim());
            if (stages.some(x => x === 'hatching' || x === 'hatched')) return { kind: 'hatching', value: null };
            if (stages.includes('comma')) return { kind: 'comma', value: null };
            return { kind: 'manual', value: null };
        }
        return { kind: STOP_KINDS[s] ? s : 'manual', value: null };
    }

    /**
     * The plan as a saved tactic's `structure` — the same shape the start
     * route seeds into the operation plan, so a template, a seeded tactic
     * and what the agent writes are one thing.
     */
    function toStructure(plan) {
        const st = {
            cadence_s: plan.intervalSeconds,
            interval: plan.intervalSeconds,
            stop_condition: stopSpec(plan.stop.kind, plan.stop.value),
            condition_value: null,
            monitoring_mode: plan.monitoringMode || 'idle',
            num_slices: plan.spim.slices,
            exposure_ms: plan.spim.exposureMs,
            laser_config: plan.spim.laserConfig || null,
            dic: null,
            stop_conditions: null,
        };
        const body = toPayload(plan, []);
        if (body.dic) st.dic = body.dic;
        if (body.stop_conditions) st.stop_conditions = body.stop_conditions;
        return st;
    }

    /** A plan from a saved tactic's `structure`. Anything missing takes the default. */
    function fromStructure(st) {
        st = st || {};
        const interval = Math.max(1, num(st.cadence_s, num(st.interval, 120)));
        const dic = st.dic && typeof st.dic === 'object' ? st.dic : null;
        const stop = parseStopSpec(st.stop_condition);
        const overrides = Object.entries(st.stop_conditions || {}).map(([embryoId, spec]) => {
            const p = parseStopSpec(typeof spec === 'string' ? spec
                : stopSpec((spec || {}).stop_condition, (spec || {}).condition_value));
            return { embryoId, kind: p.kind, value: p.value };
        });
        return fromForm({
            interval, intervalUnit: 's',
            slices: st.num_slices, exposureMs: st.exposure_ms, laserConfig: st.laser_config,
            dic: !!(dic && dic.enabled),
            dicEveryRounds: dic && dic.every_seconds ? Math.max(1, Math.round(dic.every_seconds / interval)) : 1,
            dicPosition: dic && dic.position ? 'here' : 'centroid',
            dicPin: dic && dic.position ? dic.position : null,
            dicExposureMs: dic ? dic.exposure_ms : null,
            dicLight: dic ? dic.light : 'room',
            stopKind: stop.kind, stopValue: stop.value,
            overrides,
            monitoringMode: st.monitoring_mode,
        });
    }

    return {
        STOP_KINDS, DIC_LIGHTS, stopSpec, stopWords, parseStopSpec, fromForm, fromStructure, toStructure,
        validate, toPayload, describe, intervalWords,
    };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = AcquisitionPlan;
