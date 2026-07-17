# Full Stack Microscopy

Gently should be documented as a microscopy system, not only as an agent or a
device controller. The useful view for a biologist or instrument developer is
the full path from experimental intent to stored evidence.

## Stack Map

| Stack | What it owns | Gently surface |
| --- | --- | --- |
| Experimental intent | Scientific question, hypothesis, controls, success criteria | plan mode, campaigns, plan items |
| Sample preparation | Organism, strain, treatment, mounting, perturbation | sample records, sample-tracking metrics |
| Hardware integration | Microscope devices, safety limits, device state, calibration | hardware profiles, device layer, profile templates |
| Acquisition | Snapshots, volumes, timepoints, illumination, temperature | acquisition tools, Bluesky plans, session metadata |
| Perception | Detection, classification, quality control, event recognition | perception traces, predictions, reasoning records |
| Closed-loop decisions | When to continue, stop, adapt, or ask the operator | agent tools, event logs, decision logs |
| Data and provenance | Raw data, derived data, logs, plans, exports | FileStore/GentlyStore, session directories, debug bundles |
| Operator experience | Setup, monitoring, intervention, recovery | web UI, chat, settings, docs/tutorials |

## Documentation Shape

Generated docs should include three kinds of material:

- Tutorials: task-focused paths such as "run without hardware", "add a hardware
  profile", and "start a safe timelapse".
- Concepts: the full-stack map, sample/hardware domain boundaries, and data
  provenance expectations.
- References: API surfaces, command-line flags, storage layouts, hardware
  profile checklists, and test markers.

## Hardware as One Stack

Hardware is a core stack, but it should not dominate the documentation model.
The device layer matters because it connects intent to physical state safely:
limits, calibration, timing, illumination, and temperature all shape what
scientific claims can be made from the data.

Hardware docs should therefore connect each device profile to:

- the sample state it can observe or change
- the safety boundaries it enforces
- the metadata it records
- the simulator or live-hardware tests that cover it
- the operator workflow for setup and recovery

## Tutorial Roadmap

Priority tutorials:

- run Gently offline and create a plan
- connect a local diSPIM device layer
- add a new hardware profile
- add a new organism/sample type
- inspect a stored session and export a debug bundle
- write a hardware contract test and an opt-in live hardware test
