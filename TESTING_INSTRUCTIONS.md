# Testing Instructions for Multi-Embryo Calibration GUI

## Pre-Flight Checklist

### Hardware Requirements
- [ ] Micro-Manager is running
- [ ] Bottom PCO camera is connected and configured
- [ ] HamCam1 (SPIM camera) is connected and configured
- [ ] XY stage (XYStage:XY:31) is operational
- [ ] Scanner (Scanner:AB:33) is configured
- [ ] Piezo (PiezoStage:P:34) is configured
- [ ] Lasers are functional (488 and 561 nm)
- [ ] Sample is mounted and positioned

### Software Requirements
- [ ] Python 3.8+ installed
- [ ] Node.js 18+ installed
- [ ] Backend dependencies installed (`pip install -r backend/requirements.txt`)
- [ ] Frontend dependencies installed (`cd frontend && npm install`)
- [ ] `client.py` can connect to Micro-Manager (test with existing scripts)
- [ ] `calibrate_embryo_piezo_galvo.py` is in root directory and working

---

## Test 1: Backend Startup

### Steps:
1. Open terminal in `backend/` directory
2. Run: `python main.py`

### Expected Output:
```
✓ Database initialized: embryo_calibration.db
✓ FastAPI backend started
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Troubleshooting:
- **Error: "ModuleNotFoundError"** → Run `pip install -r requirements.txt`
- **Error: "Failed to connect to Micro-Manager"** → Check `client.py` and Micro-Manager
- **Port 8000 already in use** → Kill existing process or change port in main.py

### Test API Health:
```bash
# In browser or curl:
http://localhost:8000/api/health

# Expected response:
{"status":"healthy","timestamp":"2025-01-15T..."}
```

---

## Test 2: Frontend Startup

### Steps:
1. Open NEW terminal in `frontend/` directory
2. Run: `npm run dev`

### Expected Output:
```
VITE v5.0.8  ready in 500 ms

➜  Local:   http://localhost:5173/
```

### Test Frontend:
1. Open browser to `http://localhost:5173`
2. Should see "Multi-Embryo Calibration" home page
3. Check browser console (F12) for errors

### Troubleshooting:
- **Error: "Module not found"** → Run `npm install`
- **Blank page** → Check browser console, verify backend is running
- **"Cannot connect to backend"** → Verify backend is at `http://localhost:8000`

---

## Test 3: Hardware Status

### Steps:
1. On home page, look at "Hardware Status" section
2. Should show:
   - ✅ Connected
   - Stage position: (X.X, Y.Y) µm

### If "Disconnected":
1. Check Micro-Manager is running
2. Test with existing script: `python test_simple_snap.py` (or similar)
3. Check backend terminal for error messages
4. Verify device names in `backend/hardware_control.py` match your hardware

---

## Test 4: Session Creation

### Steps:
1. Click **"+ New Session"**
2. Enter name: `Test_Session_1`
3. Enter description: `Testing GUI functionality`
4. Click **"Create Session"**

### Expected Result:
- Redirected to calibration wizard
- URL changes to `/calibration/1`
- Session name displayed at top

### Troubleshooting:
- **Error: "Session already exists"** → Use different name
- **Database error** → Check `embryo_calibration.db` permissions
- **500 error** → Check backend logs

---

## Test 5: Image Capture

### Steps:
1. Position sample under bottom camera
2. Click **"📷 Capture Image"**
3. Wait for image to appear (5-10 seconds)

### Expected Result:
- Loading spinner appears
- Image displays on screen
- Crosshair (red lines) shows center
- Grid lines visible

### Troubleshooting:
- **Timeout** → Check camera exposure settings
- **Black image** → Increase exposure or add light
- **No image** → Verify camera device name in hardware_control.py
- **Error in console** → Check backend logs for camera errors

---

## Test 6: Embryo Marking

### Steps:
1. Click on embryo #1 center
2. Green marker with "#1" should appear
3. Pixel offset displayed below marker
4. Click on embryo #2 center
5. Marker "#2" appears
6. Click on embryo #3 center
7. Marker "#3" appears

### Test Undo:
1. Click **"Undo Last"**
2. Marker #3 should disappear

### Test Removal:
1. Click directly on marker #2
2. Marker #2 should disappear
3. Remaining markers renumber automatically

### Expected Behavior:
- Markers appear instantly on click
- Numbers update automatically
- Crosshair helps center alignment
- Offset from center shown in pixels

---

## Test 7: Calibration Workflow

### Steps:
1. Mark 2-3 embryos
2. Click **"Done (3 embryos)"**
3. Wait for saving (should be fast)

### For Each Embryo:

#### Step 1: Centering
1. Click **"Center Embryo"**
2. Wait for stage movement
3. Verification image appears
4. Embryo should be at red crosshair center

**Check:**
- Stage moves smoothly
- Backend logs show movement
- New image captured automatically

#### Step 2: Calibration
1. Click **"Run Calibration"**
2. Watch backend terminal for progress
3. Wait 2-5 minutes for full calibration

**Expected Backend Output:**
```
========================================
RUNNING CALIBRATION FOR embryo_001
========================================
  Starting calibration workflow...
  Edge detection...
  Top focus sweep...
  Bottom focus sweep...
  Linear fit...
  ✓ Calibration complete:
    Slope: X.XXX µm/°
    Offset: XX.X µm
========================================
```

**Expected Frontend:**
- Progress messages appear
- Calibration results display:
  - Slope
  - Offset
  - Galvo top/bottom
  - Piezo range
- Green "✓ Calibration Complete" box

### Troubleshooting:
- **Stage doesn't move** → Check XY stage connection
- **Calibration fails** → Check `calibrate_embryo_piezo_galvo.py` works standalone
- **Missing calibration file** → Verify script generates `piezo_galvo_calibration_embryo.json`
- **Hardware errors** → Check Micro-Manager for device errors

---

## Test 8: Volume Acquisition

### Steps:
1. After calibration summary, click **"Proceed to Volume Acquisition →"**
2. Select embryos to image (checkboxes)
3. Configure parameters:
   - Slices: `20` (for quick test)
   - Timepoints: `1`
4. Review summary
5. Click **"▶ Start Acquisition"**

### Expected Behavior:
1. Button shows "Starting acquisition..."
2. Backend terminal shows:
   ```
   ======================================================================
   STARTING VOLUME ACQUISITION - RUN #1
   ======================================================================
     Embryos: 2
     Slices: 20
     Output: multi_embryo_volumes/20250115_143022
   ```

3. For each embryo:
   - Stage moves to position
   - Hardware configures (galvo, piezo, camera)
   - Slices acquired with hardware triggering
   - TIFF file saved

4. Check output directory:
   ```
   multi_embryo_volumes/
   └── 20250115_143022/
       ├── embryo_001_embryo001_t0000_20250115_143045.tif
       └── embryo_002_embryo002_t0000_20250115_143102.tif
   ```

### Verify TIFF Files:
1. Open TIFF in ImageJ/Fiji
2. Should be stack of 20 slices
3. Check dimensions: 2048 × 512 pixels (or your ROI)
4. Verify image quality and focus

### Troubleshooting:
- **No acquisition starts** → Check backend logs for errors
- **Hardware trigger fails** → Check SPIM state machine, galvo configuration
- **Empty TIFFs** → Check camera trigger settings
- **Wrong number of slices** → Verify buffer size, check for timeouts
- **Laser not on** → Check laser configuration in hardware_control.py

---

## Test 9: Timelapse Acquisition (Optional)

### Steps:
1. Start new volume run
2. Configure:
   - Slices: `20`
   - Timepoints: `3`
   - Interval: `0.5` minutes
3. Start acquisition

### Expected Behavior:
- Acquires all embryos for timepoint 0
- Waits 30 seconds
- Acquires all embryos for timepoint 1
- Waits 30 seconds
- Acquires all embryos for timepoint 2
- Completes

### Check Output:
```
multi_embryo_volumes/20250115_144500/
├── embryo_001_embryo001_t0000_....tif
├── embryo_001_embryo001_t0001_....tif
├── embryo_001_embryo001_t0002_....tif
├── embryo_002_embryo002_t0000_....tif
├── embryo_002_embryo002_t0001_....tif
└── embryo_002_embryo002_t0002_....tif
```

---

## Test 10: Session History

### Steps:
1. Navigate to home
2. Click **"View History"**
3. Should see all sessions in table
4. Test filtering: Active / Archived / All

### Test Archive:
1. Click **"Archive"** on a session
2. Should move to archived
3. Switch to "Archived" filter
4. Session appears there

### Test Delete:
1. Click **"Delete"** on test session
2. Confirm deletion
3. Session disappears
4. Database updated

---

## Test 11: Database Integrity

### Check Database:
```bash
sqlite3 embryo_calibration.db

.schema
# Should show: sessions, embryos, images, volume_runs, volume_acquisitions

SELECT * FROM sessions;
SELECT * FROM embryos;
SELECT * FROM volume_runs;
SELECT * FROM volume_acquisitions;

.exit
```

### Verify:
- All sessions present
- Embryos linked to sessions
- Calibration data stored as JSON
- Volume runs recorded
- Acquisition results saved

---

## Test 12: Error Handling

### Test Scenarios:

#### Hardware Disconnected:
1. Stop Micro-Manager
2. Try to capture image
3. Should see error message (not crash)

#### Invalid Calibration:
1. Rename `calibrate_embryo_piezo_galvo.py` temporarily
2. Try to run calibration
3. Should fail gracefully with error message

#### Disk Full (Simulation):
1. Set output dir to read-only location
2. Try volume acquisition
3. Should report error

---

## Performance Benchmarks

### Expected Timing:
- Session creation: < 1 second
- Image capture: 1-3 seconds
- Stage movement: 2-5 seconds
- Full calibration: 2-5 minutes per embryo
- Volume acquisition (50 slices): ~5-10 seconds per embryo
- TIFF save: < 1 second

---

## Common Issues & Solutions

### Issue: "Calibration file not generated"
**Solution:**
- Run `calibrate_embryo_piezo_galvo.py` standalone first
- Ensure it creates `piezo_galvo_calibration_embryo.json`
- Check hardware connections (camera, galvo, piezo)

### Issue: "Hardware trigger not working"
**Solution:**
- Verify camera is set to EXTERNAL trigger
- Check SPIM state machine configuration
- Ensure galvo SPIMState transitions properly
- Test with `test_spim_trigger_step_by_step.py`

### Issue: "Stage position doesn't update"
**Solution:**
- Check device name matches: `XYStage:XY:31`
- Verify stage is not manually controlled
- Test with: `core.setXYPosition(x, y)`

### Issue: "Frontend can't connect"
**Solution:**
- Verify backend is running on port 8000
- Check CORS settings in main.py
- Clear browser cache
- Check browser console for specific errors

---

## Success Criteria

✅ **All tests pass** if:
1. Backend starts without errors
2. Frontend displays correctly
3. Hardware status shows "Connected"
4. Sessions can be created
5. Images capture successfully
6. Embryos can be marked interactively
7. Calibration completes for all embryos
8. Volume acquisition produces valid TIFF files
9. Database stores all data correctly
10. Error handling prevents crashes

---

## Reporting Issues

If tests fail, collect:
1. **Backend logs** (full terminal output)
2. **Browser console** (F12 → Console tab)
3. **Database state** (`sqlite3 embryo_calibration.db .dump`)
4. **Hardware configuration** (Micro-Manager settings)
5. **Steps to reproduce**

Then check:
- Device names match your hardware
- Existing scripts work (`calibrate_embryo_piezo_galvo.py`)
- Permissions on directories
- Port conflicts (8000, 5173)

---

## Next Steps After Successful Testing

1. **Run with real samples** - Test full workflow with actual embryos
2. **Optimize parameters** - Tune slice count, exposure, etc.
3. **Add more embryos** - Test with 5-10 embryos
4. **Long timelapse** - Test overnight acquisition
5. **Data analysis** - Import TIFFs into analysis pipeline

🎉 **Congratulations!** You now have a working GUI for multi-embryo calibration and volume acquisition!
