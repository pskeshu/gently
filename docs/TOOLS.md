# Tool Reference

All tools use the `@tool` decorator and are auto-registered on import.
Source: `gently/agent/tools/` (run mode) and `gently/agent/plan_mode/tools/` (plan mode).

---

## Run Mode Tools

### Acquisition (`acquisition_tools.py`)

| Tool | Description |
|------|-------------|
| `acquire_volume` | Acquire a single 3D lightsheet volume for a specific embryo with calibration data |
| `capture_lightsheet` | Capture a single 2D lightsheet fluorescence image at specified piezo/galvo position |
| `batch_lightsheet` | Capture lightsheet images from ALL embryos and show them in the web UI viewer |

### Analysis (`analysis_tools.py`)

| Tool | Description |
|------|-------------|
| `analyze_volume` | Analyze embryo volume using Claude Vision API |
| `get_detection_summary` | Get summary of all detections across all embryos |

### Calibration (`calibration_tools.py`)

| Tool | Description |
|------|-------------|
| `calibrate_embryo` | Run full piezo-galvo calibration for a specific embryo using Claude vision |
| `calibrate_all_embryos` | Run piezo-galvo calibration for all detected embryos sequentially |

### Detection (`detection_tools.py`)

| Tool | Description |
|------|-------------|
| `detect_embryos` | Automatically detect embryos using brightness detection and SAM segmentation |
| `manual_mark_embryos` | Open interactive window to manually mark embryos by clicking |
| `edit_embryos` | Add/remove/move embryo positions in the web map view |
| `show_detected_embryos` | Capture fresh image and display all tracked embryos with labeled bounding boxes |

### Detectors (`detector_tools.py`)

| Tool | Description |
|------|-------------|
| `list_detectors` | List all registered detectors and their status |
| `enable_disable_detector` | Enable or disable a specific detector |
| `remove_detector` | Remove a detector from the registry |
| `add_detector` | Add a new detector with custom detection prompt |
| `enable_preset_detector` | Enable a preset detector (hatching, comma, pretzel, gastrulation, first_division) |
| `generate_detector_prompt` | Generate optimal detection prompt from a description |
| `test_detector` | Test a detector on a specific embryo's latest image |
| `query_timeline_events` | Query timeline for detection and timelapse events |

### Experiment (`experiment_tools.py`)

| Tool | Description |
|------|-------------|
| `get_experiment_summary` | Get comprehensive summary of current experiment |
| `query_embryo_status` | Query detailed status of a specific embryo |
| `skip_embryo` | Mark embryo to skip in future acquisitions |
| `remove_embryo` | Permanently remove embryo from experiment |
| `resume_embryo` | Resume imaging a previously skipped embryo |
| `assign_nickname` | Assign memorable nickname to an embryo |
| `modify_parameters` | Modify acquisition parameters for an embryo |

### Focus (`focus_tools.py`)

| Tool | Description |
|------|-------------|
| `fine_focus` | Perform fine focus adjustment by scanning piezo positions and finding optimal focus |
| `get_focus_score` | Calculate focus score for the current lightsheet image without moving the piezo |
| `get_focus_history` | Get focus history for an embryo showing all piezo-galvo measurements over time |

### Interaction (`interaction_tools.py`)

| Tool | Description |
|------|-------------|
| `ask_user_choice` | Present user with selectable options for discrete choices |

### LED (`led_tools.py`)

| Tool | Description |
|------|-------------|
| `set_led` | Set the LED illumination state |
| `get_led_status` | Get current LED illumination status |

### Plan Execution (`plan_execution_tools.py`)

| Tool | Description |
|------|-------------|
| `execute_plan_item` | Execute a planned imaging item — resolve spec, configure parameters, start timelapse |
| `complete_current_plan_item` | Mark the currently executing plan item as complete |

### Session (`session_tools.py`)

| Tool | Description |
|------|-------------|
| `assess_image_quality` | Assess image quality metrics (focus, brightness, noise) and suggest adjustments |
| `get_session_stats` | Get statistics for the current session (interactions, corrections, tool usage) |
| `compare_embryo_development` | Compare developmental progress across multiple embryos |
| `analyze_corrections` | Analyze user corrections from interaction logs to identify patterns |
| `export_interaction_log` | Export interaction logs for external analysis |
| `import_embryos_from_session` | Import embryos (positions, calibration, settings) from another session |
| `list_sessions` | List available sessions with IDs, embryo counts, and last active times |

### Stage (`stage_tools.py`)

| Tool | Description |
|------|-------------|
| `move_to_embryo` | Move XY stage to a specific embryo's stored position |
| `get_stage_position` | Get current XY stage position in micrometers |
| `move_stage` | Move XY stage to specific coordinates in micrometers |

### Timelapse (`timelapse_tools.py`)

| Tool | Description |
|------|-------------|
| `generate_bluesky_plan` | Generate a Bluesky acquisition plan from a scientific goal |
| `start_adaptive_timelapse` | Start an adaptive timelapse that runs in the background |
| `get_timelapse_status` | Get current status of the running timelapse including per-embryo progress |
| `modify_timelapse_embryo` | Modify parameters for a specific embryo during a running timelapse |
| `add_embryo_to_timelapse` | Add an embryo to an already running timelapse |
| `stop_timelapse_embryo` | Stop imaging a specific embryo (others continue) |
| `stop_timelapse` | Stop the entire timelapse acquisition |
| `pause_timelapse` | Pause the timelapse (can be resumed) |
| `resume_timelapse` | Resume a paused timelapse |
| `add_stop_condition` | Add an additional stop condition to a running timelapse (OR logic) |
| `add_interval_speedup_rule` | Add rule to speed up imaging when a developmental stage is reached |
| `enable_pre_hatching_speedup` | Enable automatic speedup when embryos approach hatching |
| `classify_embryo_stage` | Use Claude Vision to classify the current developmental stage |
| `get_stage_history` | Get the developmental stage progression history for an embryo |
| `predict_hatching` | Predict time-to-hatching with confidence intervals based on stage |

### Volume (`volume_tools.py`)

| Tool | Description |
|------|-------------|
| `view_image` | Capture and display current bottom camera widefield image |
| `view_volume` | Open a volume in the in-browser 3D viewer |
| `list_volumes` | List available volumes for an embryo or all embryos |

---

## Plan Mode Tools

### Planning (`planning.py`)

| Tool | Description |
|------|-------------|
| `create_campaign` | Create a research campaign or phase (top-level research goal) |
| `create_plan_item` | Create a plan item (imaging, bench, genetics, analysis, or decision_point) |
| `update_plan_item` | Update fields on an existing plan item |
| `link_plan_items` | Add a dependency between two plan items |
| `get_plan_item` | Get full details of a plan item by ID |
| `propose_plan` | Generate a complete experimental plan from a research goal |
| `get_plan_status` | Get campaign status with phase breakdown and progress |
| `batch_update_status` | Update status of multiple plan items at once |
| `batch_update_spec` | Update imaging/bench spec fields for multiple items |
| `move_plan_item` | Move a plan item to a different campaign or phase |
| `delete_plan_item` | Delete a plan item |
| `reorder_plan_items` | Reorder plan items within a campaign/phase |
| `update_phase` | Update phase metadata (title, description, phase_number) |
| `delete_phase` | Delete a phase and all its plan items |
| `export_plan` | Export a campaign as structured YAML/JSON |
| `snapshot_plan` | Create a named snapshot of the current plan state |
| `list_plan_versions` | List saved plan snapshots for a campaign |
| `restore_plan_version` | Restore a plan from a saved snapshot |

### Lab Context (`lab_context.py`)

| Tool | Description |
|------|-------------|
| `query_lab_history` | Search past sessions, campaigns, and learnings for relevant context |
| `check_hardware_capability` | Check if the current hardware supports a requested capability |

### Research (`research.py`)

| Tool | Description |
|------|-------------|
| `search_literature` | Search scientific literature (PubMed) for relevant papers |
| `search_strains` | Search strain/gene databases (WormBase, NCBI Gene) |
| `read_paper` | Fetch and summarize a paper by DOI or PubMed ID |

### Templates (`templates.py`)

| Tool | Description |
|------|-------------|
| `save_plan_template` | Save the current campaign as a reusable template |
| `list_templates` | List all saved plan templates |
| `apply_template` | Create new campaign from a saved template with optional overrides |

### Validation (`validation.py`)

| Tool | Description |
|------|-------------|
| `validate_plan` | Validate a plan for errors and warnings (hardware limits, dependencies, completeness) |
