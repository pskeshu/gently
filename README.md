# DiSPIM Python Control with Vision Language Model Integration

A complete Python implementation for controlling DiSPIM (Dual-sided Selective Plane Illumination Microscopy) systems with integrated Vision Language Model (VLM) capabilities for intelligent, adaptive microscopy experiments.

## Overview

This project provides headless Python control of DiSPIM microscopes using pymmcore-plus, with device abstraction through Ophyd, experiment orchestration via Bluesky, and real-time image analysis using Vision Language Models. The implementation enables VLM-guided adaptive microscopy for automated experiment optimization.

## Features

- **Headless DiSPIM Control**: Direct hardware control using pymmcore-plus without GUI dependencies
- **Advanced Calibration**: Two-point light sheet alignment with multiple autofocus algorithms
- **Device Abstraction**: Ophyd integration for standardized device interfaces
- **Experiment Orchestration**: Bluesky plans for multi-dimensional acquisition protocols
- **VLM Integration**: Real-time image analysis with HuggingFace and OpenAI models
- **Adaptive Control**: Intelligent experiment modification based on VLM feedback
- **Safety Systems**: Position limits, error recovery, and timeout handling

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Bluesky       │────│   Ophyd Devices  │────│   DiSPIM Core   │
│   Plans         │    │   Abstraction    │    │   (pymmcore)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │               ┌──────────────────┐             │
         └───────────────│  VLM Processor   │─────────────┘
                         │  (Real-time AI)  │
                         └──────────────────┘
                                  │
                         ┌──────────────────┐
                         │  Hardware Layer  │
                         │  - ASI Tiger     │
                         │  - Piezo Stages  │
                         │  - Galvo Mirrors │
                         │  - Cameras       │
                         └──────────────────┘
```

## Installation

### Prerequisites

```bash
# Core microscopy control
pip install pymmcore-plus

# Device abstraction and experiment orchestration  
pip install ophyd bluesky

# VLM integration (choose based on your needs)
pip install transformers torch pillow  # For HuggingFace models
pip install openai                     # For GPT-4V integration

# Scientific computing
pip install numpy scipy
```

### Hardware Requirements

- ASI Tiger Controller with DiSPIM configuration
- Compatible piezo actuators and galvanometer/micromirror devices
- Camera supported by Micro-Manager
- Micro-Manager 2.0+ installation with device adapters

## Quick Start

### 1. Basic Setup

```python
from dispim_config import DiSPIMCore
from dispim_calibration import DiSPIMCalibration

# Initialize DiSPIM with your configuration
dispim = DiSPIMCore("/path/to/your/dispim_config.cfg")
dispim.initialize_devices()

# Setup calibration
calibration = DiSPIMCalibration(dispim)
```

### 2. Device Calibration

```python
# Run two-point calibration for light sheet alignment
result = calibration.two_point_calibration(
    point1_piezo=25.0,  # μm
    point2_piezo=75.0,  # μm
    auto_focus=True
)

print(f"Calibration: slope={result.slope:.4f}, offset={result.offset:.2f}")
```

### 3. Ophyd Device Integration

```python
from dispim_ophyd import create_dispim_devices

# Create device ensemble
light_sheet = create_dispim_devices("/path/to/config.cfg")

# Synchronized movement
light_sheet.synchronized_move(30.0)  # Move to 30 μm with calibration
```

### 4. Bluesky Experiment Plans

```python
from dispim_bluesky import setup_dispim_session, z_stack_scan, DiSPIMScanConfig

# Setup Bluesky environment
RE, light_sheet, callbacks = setup_dispim_session()

# Configure scan parameters
config = DiSPIMScanConfig(
    z_start=-25.0,
    z_stop=25.0,
    z_step=0.5,
    exposure_time=0.02
)

# Run Z-stack acquisition
RE(z_stack_scan(light_sheet, config.z_start, config.z_stop, config.z_step))
```

### 5. VLM-Guided Adaptive Microscopy

```python
from dispim_vlm import DiSPIMVLMProcessor, create_fast_vlm_config
from dispim_bluesky import adaptive_acquisition_with_vlm

# Setup VLM processor
vlm_config = create_fast_vlm_config()
vlm_processor = DiSPIMVLMProcessor(vlm_config)
vlm_processor.start_processing()

# Create VLM callback for adaptive control
vlm_callback = vlm_processor.create_bluesky_callback()

# Run adaptive experiment
RE(adaptive_acquisition_with_vlm(light_sheet, vlm_callback, config))
```

## Core Components

### DiSPIM Configuration (`dispim_config.py`)
- **DiSPIMCore**: Main control class using pymmcore-plus
- **DiSPIMConfig**: Configuration parameters and safety limits
- **Device Management**: Initialization, positioning, and safety checks

### Calibration System (`dispim_calibration.py`)  
- **DiSPIMCalibration**: Two-point calibration procedures
- **AutofocusConfig**: Configurable autofocus parameters
- **Focus Scoring**: VOLATH, Laplacian, Gradient, and Variance algorithms
- **Calibration Validation**: Multi-point accuracy testing

### Ophyd Devices (`dispim_ophyd.py`)
- **DiSPIMPiezo**: Piezo actuator control with safety limits
- **DiSPIMGalvo**: Galvanometer/micromirror positioning  
- **DiSPIMCamera**: Camera control with exposure management
- **DiSPIMLightSheet**: Composite device for synchronized control

### Bluesky Plans (`dispim_bluesky.py`)
- **Z-Stack Acquisition**: Single and multi-view volume imaging
- **Time-Lapse Experiments**: Long-term live imaging protocols
- **Multi-Position Scanning**: Automated sample survey
- **Adaptive Plans**: VLM-guided experiment modification

### VLM Integration (`dispim_vlm.py`)
- **Multiple Providers**: HuggingFace Transformers, OpenAI GPT-4V
- **Real-Time Processing**: Background image analysis threads
- **Decision Making**: Automated acquisition parameter adjustment
- **Performance Optimization**: Efficient image encoding and processing

## VLM Capabilities

### Supported Models
- **HuggingFace**: Git-base, LLaVA, and other vision-language models
- **OpenAI GPT-4V**: High-quality scientific image analysis
- **Local Models**: Custom trained models for specific applications

### Analysis Features
- **Focus Quality Assessment**: Automatic focus scoring and suggestions
- **Feature Detection**: Cell identification and structure analysis  
- **Acquisition Optimization**: Dynamic parameter adjustment
- **Experiment Adaptation**: Step size, exposure, and ROI modification

## Configuration

### Micro-Manager Setup
1. Install Micro-Manager 2.0+ with ASI device adapters
2. Create hardware configuration file (.cfg) for your DiSPIM system
3. Test device connectivity using Micro-Manager GUI
4. Export configuration for headless Python use

### VLM Configuration
```python
# Fast processing for real-time feedback
fast_config = create_fast_vlm_config()

# High-quality analysis for detailed assessment  
hq_config = create_high_quality_vlm_config()
hq_config.provider = VLMProvider.OPENAI_GPT4V  # Requires API key
```

## Advanced Usage

### Custom Acquisition Plans
```python
def custom_dispim_plan(light_sheet, parameters):
    """Create custom acquisition protocol"""
    yield from bps.open_run(md={'plan_name': 'custom_dispim'})
    
    # Your custom experiment logic here
    for z_pos in parameters.z_positions:
        yield from light_sheet.synchronized_move(z_pos)
        yield from bps.trigger_and_read([light_sheet.camera])
    
    yield from bps.close_run()
```

### VLM Decision Callbacks
```python
def smart_acquisition_callback(image_data, z_position, z_index):
    """Custom VLM decision logic"""
    # Analyze image with your specific criteria
    decision = analyze_with_domain_knowledge(image_data)
    
    return {
        'continue_scan': decision.interesting_features_found,
        'next_step': 0.3 if decision.high_detail_needed else 1.0,
        'suggest_autofocus': decision.focus_quality < 0.7
    }
```

## Safety Features

- **Position Limits**: Configurable safety bounds for all actuators
- **Timeout Protection**: Automatic recovery from stuck operations
- **Error Handling**: Graceful degradation and logging
- **Emergency Stops**: Safe shutdown procedures
- **Validation Checks**: Parameter verification before execution

## Performance Optimization

- **Parallel Processing**: Concurrent device operations
- **Background VLM**: Non-blocking image analysis
- **Memory Management**: Efficient image buffer handling
- **Adaptive Timing**: Dynamic acquisition rate adjustment

## Examples

Complete usage examples are provided in `example_dispim_vlm.py`:

1. **Basic Setup**: Device initialization and configuration
2. **Calibration**: Two-point alignment procedures  
3. **Ophyd Integration**: Device abstraction examples
4. **Bluesky Plans**: Multi-dimensional acquisition protocols
5. **VLM Integration**: Real-time image analysis setup
6. **Adaptive Experiments**: Complete VLM-guided workflows

## Contributing

Contributions are welcome! Areas of particular interest:

- Additional VLM model integrations
- Custom focus scoring algorithms
- Advanced calibration procedures
- Specialized acquisition protocols
- Performance optimizations
- GUI

## License

This project is open-source software. Please respect the licenses of dependencies including Micro-Manager, pymmcore-plus, Ophyd, and Bluesky.

## Support and Documentation

- **Hardware Setup**: Refer to [DiSPIM documentation](http://dispim.org/)
- **Micro-Manager**: See [Micro-Manager wiki](https://micro-manager.org/)
- **Bluesky**: Visit [Bluesky Project](https://blueskyproject.io/)
- **Issues**: Report bugs and request features via GitHub issues

