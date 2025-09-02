# DiSPIM Microscope Control

A simple test script for controlling a DiSPIM microscope using Micro-Manager and Python.

## Setup Instructions

### 1. Clone the Repository

1. Install [GitHub Desktop](https://desktop.github.com/)
2. Clone this repository using GitHub Desktop or git command line:
   ```bash
   git clone https://github.com/pskeshu/gently.git
   cd gently
   ```

### 2. Install Python

1. Download and install Python 3.8 or newer from [python.org](https://www.python.org/downloads/)
2. Ensure Python is added to your system PATH during installation

### 3. Set Up Virtual Environment and Install Libraries

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required libraries
pip install pymmcore pyyaml numpy
```

### 4. Configure Micro-Manager

1. Edit `config.yml` to point to your Micro-Manager installation directory
2. Ensure the `MMConfig_tracking_screening.cfg` file is present in the project directory
3. Update device names and settings in the configuration file as needed for your setup

## Usage

With the virtual environment activated:

```bash
python test_dispim_control.py
```

The script will:
1. Load the Micro-Manager configuration
2. Initialize devices
3. Read piezo positions
4. Capture an image and display array information  
5. Apply shutdown configuration

## Requirements

- Python 3.8+
- Micro-Manager installation
- Required Python packages (installed via pip above):
  - pymmcore
  - pyyaml  
  - numpy

## Configuration Files

- `config.yml` - Main configuration pointing to Micro-Manager files
- `MMConfig_tracking_screening.cfg` - Micro-Manager device configuration