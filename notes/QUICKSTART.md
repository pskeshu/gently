# Quick Start Guide

## First Time Setup

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
```

## Running the Application

### Option A: Using Batch Scripts (Windows)

**Terminal 1**: Double-click `start_backend.bat`
**Terminal 2**: Double-click `start_frontend.bat`

### Option B: Manual Commands

**Terminal 1** (Backend):
```bash
cd backend
python main.py
```

**Terminal 2** (Frontend):
```bash
cd frontend
npm run dev
```

## Access the Application

Open your browser to: **http://localhost:5173**

## Quick Workflow

1. **Create Session** → Enter name → Click "Create Session"
2. **Calibrate Embryos** → Capture Image → Mark Embryos → Click through calibration
3. **Run Volume Acquisition** → Select embryos → Configure parameters → Start

## Troubleshooting

**Backend won't start?**
- Ensure Micro-Manager is running
- Check that client.py can connect

**Frontend won't start?**
- Run `npm install` in the frontend folder
- Make sure Node.js is installed

**Can't see images?**
- Check that backend is running (http://localhost:8000)
- Verify hardware connections

For full documentation, see `README_GUI.md`
