# Multi-Embryo Calibration & Volume Acquisition GUI

A modern web-based interface for multi-embryo calibration and volume acquisition workflows on the DiSPIM microscope.

## Architecture

**Backend**: FastAPI (Python) + SQLite database
**Frontend**: React + TypeScript + Vite + TailwindCSS
**Pattern**: REST API + WebSocket for real-time updates

```
gently/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── database.py                # SQLAlchemy models
│   ├── hardware_control.py        # Micro-Manager wrappers
│   ├── models.py                  # Pydantic schemas
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── pages/                 # Main pages (Home, Calibration, Volume, History)
│   │   ├── components/            # Reusable components (EmbryoMarker)
│   │   ├── api/                   # API client
│   │   ├── hooks/                 # React hooks (WebSocket)
│   │   └── types/                 # TypeScript definitions
│   └── package.json               # Node dependencies
└── embryo_calibration.db          # SQLite database (created on first run)
```

## Features

### Session Management
- Create named experimental sessions
- Organize embryos by session
- Archive/delete old sessions
- View session history and statistics

### Multi-Embryo Calibration Workflow
1. **Capture Initial Image**: Bottom camera snapshot
2. **Mark All Embryos**: Interactive HTML5 canvas with click-to-mark
3. **Calibrate Each Embryo**:
   - Auto-center via stage movement
   - Run full piezo/galvo calibration
   - Save calibration parameters to database
4. **Summary**: View all calibrated embryos

### Volume Acquisition
- Select which embryos to image
- Configure parameters:
  - Number of Z-slices
  - Number of timepoints (timelapse support)
  - Interval between timepoints
- Real-time progress tracking
- Saves TIFF stacks to disk

### Real-Time Updates
- WebSocket connection for live progress
- Status messages during long operations
- No polling required

## Installation

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Ensure hardware access**:
   - Micro-Manager must be running
   - `client.py` must be able to connect (rpyc)
   - Bottom camera and SPIM camera configured

### Frontend Setup

1. **Install Node.js** (if not already installed):
   - Download from https://nodejs.org/ (LTS version recommended)

2. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

### Step 1: Start the Backend

```bash
cd backend
python main.py
```

The backend will start on **http://localhost:8000**

You should see:
```
✓ Database initialized: embryo_calibration.db
✓ FastAPI backend started
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start the Frontend

In a **separate terminal**:

```bash
cd frontend
npm run dev
```

The frontend will start on **http://localhost:5173**

You should see:
```
VITE v5.0.8  ready in 500 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Step 3: Access the Application

Open your browser to **http://localhost:5173**

## Usage Guide

### 1. Create a New Session

1. Click **"+ New Session"**
2. Enter a session name (e.g., `Sample_2025-01-15`)
3. Optionally add a description
4. Click **"Create Session"**

### 2. Calibrate Embryos

**Step 1: Capture Image**
- Position sample under bottom camera
- Click **"📷 Capture Image"**

**Step 2: Mark Embryos**
- Click on each embryo's center to mark it
- Numbers appear automatically (#1, #2, etc.)
- Pixel offsets from center shown
- Click **"Undo Last"** to remove latest marker
- Click **"Done"** when all embryos marked

**Step 3: Calibrate Each Embryo**
- For each embryo:
  1. Click **"Center Embryo"** → stage moves automatically
  2. Verification image appears
  3. Click **"Run Calibration"** → runs full piezo/galvo calibration
  4. Calibration results displayed (slope, offset, etc.)
  5. Automatically moves to next embryo

**Step 4: Summary**
- Table of all calibrated embryos
- Proceed to volume acquisition

### 3. Run Volume Acquisition

1. Navigate to **Volume Acquisition** page
2. Select which embryos to image (checkboxes)
3. Configure parameters:
   - **Slices**: 50 (default) for full embryo depth
   - **Timepoints**: 1 for single acquisition, >1 for timelapse
   - **Interval**: Time between timepoints (if timelapse)
4. Review summary (estimated time)
5. Click **"▶ Start Acquisition"**

### 4. View Session History

- Click **"View History"** from home page
- Filter by Active/Archived
- View statistics across all sessions
- Archive old sessions to clean up home page
- Delete sessions (permanently removes data)

## API Documentation

### REST Endpoints

**Sessions**:
- `POST /api/sessions` - Create session
- `GET /api/sessions` - List sessions
- `GET /api/sessions/{id}` - Get session details
- `DELETE /api/sessions/{id}` - Delete session

**Hardware**:
- `GET /api/hardware/status` - Get hardware status
- `POST /api/hardware/capture` - Capture image

**Embryos**:
- `POST /api/embryos/mark` - Mark embryo
- `GET /api/embryos` - List embryos
- `POST /api/embryos/{id}/center` - Center embryo
- `POST /api/embryos/{id}/calibrate` - Run calibration

**Volumes**:
- `POST /api/volumes/runs` - Create volume run
- `GET /api/volumes/runs` - List volume runs

**WebSocket**:
- `ws://localhost:8000/ws/calibration` - Real-time calibration updates

## Database Schema

**sessions**: Experimental sessions
**embryos**: Embryo records with calibration data
**images**: Base64-encoded PNG images
**volume_runs**: Volume acquisition runs
**volume_acquisitions**: Individual volume captures

## Troubleshooting

### Backend won't start
- Check that Micro-Manager is running
- Verify `client.py` can connect (test with existing scripts)
- Ensure port 8000 is not in use

### Frontend can't connect to backend
- Verify backend is running on http://localhost:8000
- Check browser console for errors (F12)
- Clear browser cache

### Hardware control errors
- Check hardware connections in Micro-Manager
- Verify device names match in `hardware_control.py`:
  - Bottom camera: "Bottom PCO"
  - SPIM camera: "HamCam1"
  - XY stage: "XYStage:XY:31"

### Image not displaying
- Check browser console for base64 decoding errors
- Verify image capture succeeded (check backend logs)
- Try recapturing the image

## Development

### Backend Development
```bash
cd backend
python main.py  # Auto-reloads on file changes
```

### Frontend Development
```bash
cd frontend
npm run dev  # Hot module replacement
```

### Building for Production
```bash
cd frontend
npm run build  # Creates optimized bundle in dist/
```

## Technology Stack

**Backend**:
- FastAPI 0.104.1 - Modern Python web framework
- SQLAlchemy 2.0.23 - ORM for database
- Pydantic 2.5.0 - Data validation
- WebSockets 12.0 - Real-time communication

**Frontend**:
- React 18.2.0 - UI library
- TypeScript 5.2.2 - Type safety
- Vite 5.0.8 - Build tool
- TailwindCSS 3.3.6 - Styling
- Axios 1.6.2 - HTTP client

## Future Enhancements

- [ ] Live video streaming from camera
- [ ] Export calibration data as CSV/JSON
- [ ] Download volume TIFFs as ZIP
- [ ] Real-time slice preview during acquisition
- [ ] User authentication/multi-user support
- [ ] Mobile-responsive design
- [ ] Dark/light theme toggle

## License

MIT License - See main repository for details.

## Support

For issues or questions:
1. Check existing scripts (`multi_embryo_calibration.py`, `run_multi_embryo_volumes.py`) for reference
2. Review browser console (F12) for frontend errors
3. Check backend terminal for Python tracebacks
4. Verify database integrity: `sqlite3 embryo_calibration.db .schema`
