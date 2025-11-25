"""
Claude tool definitions for Microscopy Copilot
"""

from typing import List, Dict


def get_tool_definitions() -> List[Dict]:
    """
    Get Claude tool (function calling) definitions

    These define what actions the copilot can take to interact with
    the microscope and query experiment state.

    Returns
    -------
    list of dict
        Tool definitions in Claude API format
    """
    return [
        {
            "name": "generate_bluesky_plan",
            "description": """Generate a Bluesky acquisition plan from a scientific goal.

This creates executable Python code for a Bluesky plan that can control the microscope.
Use this when the user asks to start imaging, create an experiment, or describes
what they want to accomplish.

The plan will be validated for safety before execution.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Scientific objective in natural language. Examples: 'Monitor all embryos for hatching', 'Image embryo 3 with high resolution', 'Track development with minimal photobleaching'"
                    },
                    "embryo_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which embryos to include (e.g., ['embryo_001', 'embryo_002']). Use all embryos if user says 'all'"
                    },
                    "plan_type": {
                        "type": "string",
                        "enum": ["adaptive_timelapse", "single_highres"],
                        "description": "Type of plan. adaptive_timelapse for monitoring over time, single_highres for detailed single acquisition"
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interval_seconds": {
                                "type": "number",
                                "description": "Time between timepoints in seconds (minimum 10s for hardware settle). Typical: 120-300s for normal monitoring, 30-60s for pre-hatching"
                            },
                            "num_timepoints": {
                                "type": "integer",
                                "description": "Maximum number of timepoints to acquire. Typical: 200-500 for full development cycle"
                            },
                            "num_slices": {
                                "type": "integer",
                                "description": "Number of Z slices per volume (10-200). Typical: 50 for normal embryos, 80-100 for elongated/hatching"
                            },
                            "exposure_ms": {
                                "type": "number",
                                "description": "Camera exposure time in milliseconds (5-100). Typical: 10ms. Lower for less photobleaching, higher for dimmer samples"
                            }
                        },
                        "description": "Acquisition parameters. If not specified, copilot will use intelligent defaults based on goal"
                    }
                },
                "required": ["goal", "embryo_ids"]
            }
        },
        {
            "name": "query_embryo_status",
            "description": """Get detailed status of a specific embryo.

Returns current state including:
- Last imaging time
- Acquisition parameters (interval, slices, exposure)
- Analysis results (hatching status, developmental stage)
- Recent observations

Use this when user asks about a specific embryo's status.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo identifier (e.g., 'embryo_001'). Can also accept user-friendly formats like 'embryo 1' or 'embryo_003'"
                    }
                },
                "required": ["embryo_id"]
            }
        },
        {
            "name": "analyze_volume",
            "description": """Analyze an embryo volume using Claude Vision API.

Sends the latest (or specified) volume to Claude for visual analysis. You can ask
specific questions like:
- "Is this embryo hatching?"
- "What developmental stage is this?"
- "Are there any abnormalities?"
- "Describe the morphology"

Optionally includes recent temporal context (previous images) for better analysis.

Use this when you need to examine images in detail to answer user questions or
make decisions about acquisition parameters.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to analyze"
                    },
                    "analysis_prompt": {
                        "type": "string",
                        "description": "Question to ask about the image. Be specific about what you're looking for. Examples: 'Is there any breach in the eggshell indicating hatching?', 'What is the approximate developmental stage?', 'Is the embryo elongated?'"
                    },
                    "use_recent_context": {
                        "type": "boolean",
                        "description": "Whether to include recent previous images for temporal context. True for questions about progression/changes, False for single-timepoint morphology"
                    },
                    "timepoint": {
                        "type": "integer",
                        "description": "Specific timepoint to analyze (optional). If not provided, uses most recent image"
                    }
                },
                "required": ["embryo_id", "analysis_prompt"]
            }
        },
        {
            "name": "modify_parameters",
            "description": """Modify acquisition parameters for a specific embryo mid-experiment.

Changes take effect at the next imaging timepoint. Use this to:
- Increase/decrease sampling rate (interval_seconds)
- Adjust Z-coverage (num_slices) for elongation
- Change exposure time
- Adjust priority (high priority embryos are imaged first and more frequently)

Always explain to the user why you're making these changes.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to modify"
                    },
                    "changes": {
                        "type": "object",
                        "properties": {
                            "interval_seconds": {
                                "type": "number",
                                "description": "New interval in seconds (min 10s)"
                            },
                            "num_slices": {
                                "type": "integer",
                                "description": "New number of Z slices (10-200)"
                            },
                            "exposure_ms": {
                                "type": "number",
                                "description": "New exposure time (5-100ms)"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "normal", "low"],
                                "description": "Acquisition priority. High priority embryos are imaged first and may be imaged more frequently"
                            }
                        },
                        "description": "Parameters to change. Only include parameters you want to modify"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you're making this change (for logging and user explanation)"
                    }
                },
                "required": ["embryo_id", "changes", "reason"]
            }
        },
        {
            "name": "get_experiment_summary",
            "description": """Get comprehensive summary of entire experiment.

Returns overview including:
- Experiment duration
- Status (idle/running/paused/completed)
- All embryos and their states
- Current acquisition plan
- Recent significant events

Use this when user asks broad questions like 'How's everything going?', 'What's the status?', 'Summarize the experiment'""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "skip_embryo",
            "description": """Mark an embryo to be skipped in future acquisitions.

Use this when:
- Embryo has hatched and confirmed (no need to keep imaging)
- Embryo has died
- Embryo is not interesting for current experimental goals
- User explicitly requests to stop imaging this embryo

The embryo can be re-enabled later if needed.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to skip"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why skipping this embryo (e.g., 'Hatched and confirmed', 'Development arrested', 'User requested')"
                    }
                },
                "required": ["embryo_id", "reason"]
            }
        },
        {
            "name": "resume_embryo",
            "description": """Resume imaging a previously skipped embryo.

Use this if user wants to re-enable imaging of an embryo that was skipped.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to resume"
                    }
                },
                "required": ["embryo_id"]
            }
        },
        {
            "name": "assign_nickname",
            "description": """Assign a memorable nickname to an embryo for easier reference.

Use this when you notice distinguishing characteristics. For example:
- "the fast developer" - developing faster than others
- "the slow one" - delayed development
- "top-left embryo" - spatial reference
- "early hatcher" - first to hatch

Nicknames make conversation more natural. User can always refer to embryos by ID too.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to nickname"
                    },
                    "nickname": {
                        "type": "string",
                        "description": "Memorable nickname (keep it short and descriptive)"
                    }
                },
                "required": ["embryo_id", "nickname"]
            }
        },
        {
            "name": "list_detectors",
            "description": """List all configured detectors in the system.

Shows detector name, description, enabled status, detection counts, and action mode.
Use when user asks 'what detectors do we have?', '/detectors', or wants to see detection status.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "enum": ["all", "enabled", "disabled"],
                        "description": "Filter detectors by status (default: all)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "add_detector",
            "description": """Add a new detector to the system.

The detector will automatically analyze volumes using Claude Vision API.
Use this when user wants to create a new detector for detecting specific events or stages.

You can either:
1. Create from preset (hatching, comma, pretzel, gastrulation, first_division)
2. Create custom with user-provided or generated prompt""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Detector name (unique identifier, lowercase with underscores, e.g., 'comma_stage')"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what this detector detects"
                    },
                    "detection_prompt": {
                        "type": "string",
                        "description": "Claude Vision API prompt for detection. Should instruct Claude to analyze image and respond with DETECTED: YES/NO, CONFIDENCE: HIGH/MEDIUM/LOW, REASONING: explanation"
                    },
                    "preset": {
                        "type": "string",
                        "enum": ["hatching", "comma", "pretzel", "gastrulation", "first_division"],
                        "description": "Create from preset instead of custom prompt (optional)"
                    },
                    "action_mode": {
                        "type": "string",
                        "enum": ["passive", "recommend", "auto"],
                        "description": "What to do when detected: passive (just log), recommend (suggest actions), auto (apply automatically)"
                    },
                    "parameter_changes": {
                        "type": "object",
                        "description": "Parameters to change when detected (for recommend/auto modes). e.g., {'interval_seconds': 60, 'num_slices': 80}"
                    },
                    "min_timepoint": {
                        "type": "integer",
                        "description": "Don't run before this timepoint (optional, e.g., 50 to skip early development)"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "generate_detector_prompt",
            "description": """Generate an optimal detection prompt from a natural language description.

Use this when user describes what they want to detect but doesn't provide a detailed prompt.
You'll use your knowledge of C. elegans biology and Claude Vision best practices to create
an effective detection prompt.

Example: User says "detect comma stage" → Generate prompt with key characteristics,
temporal context guidance, and structured output format.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detector_description": {
                        "type": "string",
                        "description": "User's description of what to detect (e.g., 'comma stage', 'when embryo starts moving', 'neural activity')"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the experiment or what to look for (optional)"
                    }
                },
                "required": ["detector_description"]
            }
        },
        {
            "name": "test_detector",
            "description": """Test a detector on a specific embryo's latest image.

Use this to verify a detector works correctly before enabling it for production use.
Shows detection result without storing it permanently.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detector_name": {
                        "type": "string",
                        "description": "Detector to test"
                    },
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to test on"
                    },
                    "timepoint": {
                        "type": "integer",
                        "description": "Specific timepoint to test (optional, defaults to latest)"
                    }
                },
                "required": ["detector_name", "embryo_id"]
            }
        },
        {
            "name": "enable_disable_detector",
            "description": """Enable or disable a detector.

Disabled detectors won't run on new volumes. Useful for temporarily turning off detectors
or focusing on specific detections.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detector_name": {
                        "type": "string",
                        "description": "Detector to enable/disable"
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable, False to disable"
                    }
                },
                "required": ["detector_name", "enabled"]
            }
        },
        {
            "name": "remove_detector",
            "description": """Remove a detector from the system permanently.

Use when user wants to delete a detector they no longer need.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detector_name": {
                        "type": "string",
                        "description": "Detector to remove"
                    }
                },
                "required": ["detector_name"]
            }
        },
        {
            "name": "get_detection_summary",
            "description": """Get summary of all detections across all embryos.

Shows which embryos have been detected by which detectors, detection counts, and timing.
Use when user asks about detection status or wants to compare embryos.""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "calibrate_embryo",
            "description": """Run full piezo-galvo calibration for a single embryo.

This performs the complete calibration workflow:
1. Moves stage to embryo position
2. Runs edge detection and focus sweeps
3. Performs 2-point linear fit
4. Stores calibration parameters

Use when:
- User requests to calibrate an embryo
- Setting up new embryos before volume acquisition
- Recalibration needed after hardware changes

Note: Requires hardware control (RunEngine and devices)""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to calibrate"
                    },
                    "piezo_positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Piezo Z positions for calibration (micrometers). Default: [40.0, 60.0] for 2-point calibration"
                    }
                },
                "required": ["embryo_id"]
            }
        },
        {
            "name": "acquire_volume",
            "description": """Acquire a single 3D volume for a specific embryo.

Uses hardware-triggered SPIM acquisition with synchronized scanner, piezo, and camera.
Applies embryo's calibration parameters if available.

Use when:
- User requests to image a specific embryo
- Taking a high-resolution snapshot
- Testing acquisition before starting timelapse

Note: Requires hardware control (RunEngine and devices)""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to image"
                    },
                    "num_slices": {
                        "type": "integer",
                        "description": "Number of Z slices (10-200). Default: 50"
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure in milliseconds. Default: 10.0"
                    },
                    "save": {
                        "type": "boolean",
                        "description": "Whether to save volume to disk. Default: true"
                    }
                },
                "required": ["embryo_id"]
            }
        },
        {
            "name": "move_to_embryo",
            "description": """Move XY stage to center on a specific embryo.

Uses embryo's stored calibrated position to move the stage.

Use when:
- User wants to view an embryo
- Preparing for manual intervention
- Verifying embryo positioning

Note: Requires hardware control (RunEngine and devices)""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_id": {
                        "type": "string",
                        "description": "Embryo to move to"
                    }
                },
                "required": ["embryo_id"]
            }
        },
        {
            "name": "start_multi_embryo_timelapse",
            "description": """Start multi-embryo time-lapse volume acquisition.

Orchestrates acquisition of multiple embryos over time with:
- Sequential positioning and imaging
- Automated detector analysis
- Adaptive parameter changes based on detections
- Progress tracking and reporting

Use when:
- User requests to start long-term monitoring
- Beginning the main experiment
- Restarting after pause

Note: Requires hardware control (RunEngine and devices)""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "embryo_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which embryos to include. Use all loaded embryos if not specified"
                    },
                    "num_timepoints": {
                        "type": "integer",
                        "description": "Maximum number of timepoints. Default: 500"
                    },
                    "interval_seconds": {
                        "type": "number",
                        "description": "Time between complete cycles (all embryos). Default: 120"
                    },
                    "num_slices": {
                        "type": "integer",
                        "description": "Number of Z slices per volume. Default: 50"
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure per slice. Default: 10.0"
                    },
                    "enable_detectors": {
                        "type": "boolean",
                        "description": "Whether to run detectors on acquired volumes. Default: true"
                    }
                },
                "required": []
            }
        },
        {
            "name": "pause_acquisition",
            "description": """Pause currently running acquisition.

Stops after completing current embryo. Can be resumed later.

Use when:
- User requests to pause
- Need to make adjustments
- Emergency stop needed

Note: Requires hardware control (RunEngine)""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "resume_acquisition",
            "description": """Resume previously paused acquisition.

Continues from where it left off.

Use when:
- User requests to resume
- Issues have been resolved

Note: Requires hardware control (RunEngine)""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "detect_embryos",
            "description": """Automatically detect embryos using brightness detection + SAM refinement.

Uses a hybrid approach:
1. Brightness thresholding finds embryo candidates (embryos are BRIGHT objects)
2. SAM refines segmentation using bounding box prompts

Returns detected positions for user confirmation before calibration.

Use when:
- Setting up new experiment
- User wants to find embryos automatically
- Adding more embryos to existing experiment

Parameters:
- brightness_percentile: 99.0 = fewer detections, 98.0 = more detections
- min_area/max_area: Filter by embryo size in pixels

Note: Requires hardware control (bottom camera) and SAM model""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "auto_calibrate": {
                        "type": "boolean",
                        "description": "Automatically calibrate all detected embryos after detection. Default: false"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0-1). Default: 0.7"
                    },
                    "use_claude_review": {
                        "type": "boolean",
                        "description": "Use Claude Vision to review and verify SAM detections. Slower but more accurate. Default: false"
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure time in milliseconds. Higher values (e.g., 100-500ms) improve contrast for better embryo detection. Default: uses current camera setting."
                    },
                    "brightness_percentile": {
                        "type": "number",
                        "description": "Percentile threshold for brightness-based detection. Embryos are detected as brightest objects. 99.0 = fewer confident detections, 98.0 = more detections. Default: 99.0"
                    },
                    "min_area": {
                        "type": "integer",
                        "description": "Minimum embryo area in pixels. Filters out small noise. Default: 5000"
                    },
                    "max_area": {
                        "type": "integer",
                        "description": "Maximum embryo area in pixels. Filters out large artifacts. Default: 150000"
                    }
                },
                "required": []
            }
        },
        {
            "name": "manual_mark_embryos",
            "description": """Manually mark embryos by clicking on an image.

Captures bottom camera image and opens a matplotlib window where
user can click on embryo centers. Close window when done marking.

Use when:
- Automatic detection (SAM) fails or gives poor results
- User wants precise control over embryo positions
- User explicitly asks to mark embryos manually

The tool converts pixel coordinates to stage coordinates automatically.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure time in milliseconds. Higher values (e.g., 100-500ms) improve contrast for better visibility."
                    }
                },
                "required": []
            }
        },
        {
            "name": "view_image",
            "description": """Capture and view the current bottom camera image.

Opens a matplotlib window to display the image, or saves to file if save_only is true.

Use when:
- User wants to see what the camera sees
- Checking sample position
- Debugging or verification""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Window title. Default: 'Bottom Camera Image'"
                    },
                    "exposure_ms": {
                        "type": "number",
                        "description": "Camera exposure time in milliseconds. Higher values improve contrast."
                    },
                    "save_only": {
                        "type": "boolean",
                        "description": "If true, saves image to file and returns immediately (non-blocking). Default: false"
                    }
                },
                "required": []
            }
        },
        {
            "name": "capture_lightsheet",
            "description": """Capture a single lightsheet image at the current position.

Takes a single lightsheet image (not a full volume) at specified piezo/galvo positions.
Useful for checking focus, alignment, or seeing the embryo with lightsheet illumination.

Use when:
- User wants to see a single lightsheet plane
- Checking lightsheet alignment
- Quick preview before full volume acquisition""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "piezo_position": {
                        "type": "number",
                        "description": "Piezo position in micrometers (Z position). Default: 50.0"
                    },
                    "galvo_position": {
                        "type": "number",
                        "description": "Galvo position in volts (lightsheet Y position). Default: 0.0"
                    },
                    "save_only": {
                        "type": "boolean",
                        "description": "If true, saves image to file without displaying. Default: false"
                    }
                },
                "required": []
            }
        },
        {
            "name": "show_detected_embryos",
            "description": """Show the last detected embryos with bounding boxes overlaid on the image.

Displays a visualization of all embryos from the most recent detection,
with colored bounding boxes and labels showing embryo ID and confidence.

Use when:
- User asks to see the detected embryos
- User wants to verify detection results
- User asks "can I see the embryos with bboxes"
- User wants to check which embryos were found

Note: Must run detect_embryos first to have results to display.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "save_to_file": {
                        "type": "boolean",
                        "description": "Save the visualization to a PNG file. Default: false"
                    },
                    "save_only": {
                        "type": "boolean",
                        "description": "If true, saves to file and returns immediately without showing window. Default: false"
                    }
                },
                "required": []
            }
        },
        {
            "name": "set_led",
            "description": """Control the LED light source.

Turn the LED on or off. The LED is used for brightfield imaging with the bottom camera.

Use when:
- User wants to turn LED on or off
- Troubleshooting bright/dark images
- Setting up illumination for bottom camera

States:
- 'Open' = LED ON (bright illumination)
- 'Closed' = LED OFF (no illumination)""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["Open", "Closed"],
                        "description": "LED state: 'Open' for ON, 'Closed' for OFF"
                    }
                },
                "required": ["state"]
            }
        },
        {
            "name": "get_led_status",
            "description": """Get current LED status and available configurations.

Returns current LED state and what preset configurations are available.
Use this to check if LED is on/off and troubleshoot lighting issues.""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        # Databroker tools for accessing run data
        {
            "name": "list_runs",
            "description": """List recent Bluesky runs from Databroker.

Shows run metadata including UID, start time, plan name, and any custom metadata
like embryo_id. Use this to find specific runs or see acquisition history.

Use when:
- User asks about recent acquisitions
- Need to find a specific run to analyze
- Want to see experiment history""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of runs to return. Default: 10"
                    },
                    "embryo_id": {
                        "type": "string",
                        "description": "Filter runs by embryo_id metadata (optional)"
                    },
                    "plan_name": {
                        "type": "string",
                        "description": "Filter runs by plan name (optional)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_run_data",
            "description": """Get data from a specific Bluesky run.

Retrieves the actual data (images, positions, etc.) from a run stored in Databroker.
Can get data by UID or relative index (-1 for last run, -2 for second to last, etc.).

Use when:
- Need to analyze data from a past acquisition
- User asks to see/compare previous images
- Want to extract specific measurements from a run""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run UID or relative index (e.g., '-1' for last run, '-2' for second to last)"
                    },
                    "data_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific data keys to retrieve (e.g., ['bottom_camera', 'xy_stage']). If not specified, returns all available keys."
                    },
                    "stream": {
                        "type": "string",
                        "description": "Data stream to read from. Default: 'primary'"
                    }
                },
                "required": ["run_id"]
            }
        },
        {
            "name": "get_run_image",
            "description": """Get an image from a Bluesky run for analysis.

Retrieves image data from a detector in a specific run. Returns image metadata
and optionally analyzes the image with Claude Vision.

Use when:
- User wants to see an image from a past acquisition
- Need to analyze historical images
- Comparing images across timepoints""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run UID or relative index (e.g., '-1' for last run)"
                    },
                    "detector": {
                        "type": "string",
                        "description": "Detector name (e.g., 'bottom_camera', 'volume_scanner'). Default: auto-detect"
                    },
                    "analyze": {
                        "type": "boolean",
                        "description": "Whether to analyze the image with Claude Vision. Default: false"
                    },
                    "analysis_prompt": {
                        "type": "string",
                        "description": "Prompt for Claude Vision analysis (required if analyze=true)"
                    }
                },
                "required": ["run_id"]
            }
        },
        {
            "name": "search_runs",
            "description": """Search Databroker runs by metadata criteria.

Flexible search across run metadata. Can search by time range, custom metadata fields,
plan type, etc.

Use when:
- Looking for runs matching specific criteria
- Finding all runs for a particular embryo
- Searching by time range""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "Start time for search (e.g., '1 hour ago', '2024-01-15', 'today')"
                    },
                    "until": {
                        "type": "string",
                        "description": "End time for search (optional)"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata key-value pairs to match (e.g., {'embryo_id': 'embryo_001'})"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return. Default: 20"
                    }
                },
                "required": []
            }
        }
    ]
