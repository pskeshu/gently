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
            "description": """Automatically detect embryos using Segment Anything Model (SAM).

Captures bottom camera image and uses computer vision to find embryos.
Returns detected positions for user confirmation before calibration.

Use when:
- Setting up new experiment
- User wants to find embryos automatically
- Adding more embryos to existing experiment

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
                    }
                },
                "required": []
            }
        }
    ]
