# DiSPIMVolumeScanner Refactoring Plan

## Problem Statement

Current implementation blocks `trigger()` for ~23.6s total:
- Hardware config + triggering: ~2.7s
- Image retrieval via rpyc: ~22s ← **Bottleneck blocking trigger()**

This violates Ophyd semantics and prevents parallel multi-device operations in Bluesky.

## Root Cause: rpyc Network Transfer Bottleneck

Each `rpyc.classic.obtain(core.popNextImage())` takes ~220ms for 2304×2304 uint16 image:
- 100 slices × 220ms = 22 seconds unavoidable network transfer time
- Transfer happens in `trigger()`, blocking Status completion

## Recommended Solution: Lazy Retrieval Pattern

### Proper Ophyd Semantics

**trigger()**: Hardware configuration + buffer fill only
- Configure camera and SPIM controller (~1s)
- Start hardware-triggered acquisition (~1.7s)
- Wait for images to be IN CIRCULAR BUFFER
- Return Status.finished when buffer is full (~2.7s total)
- **Do NOT retrieve images**

**read()**: Image retrieval from buffer
- Called AFTER trigger() Status completes
- Retrieve images from MM circular buffer (~22s)
- Return volume data in Bluesky event format

### Benefits

1. ✅ **Parallel triggering**: Multiple devices can trigger concurrently
2. ✅ **Proper Status semantics**: "done" = ready to read, not "data retrieved"
3. ✅ **Non-blocking**: RunEngine can orchestrate other operations while images are in buffer
4. ✅ **Clear separation**: trigger = hardware control, read = data access

### Trade-offs

- Same total time (~24s) for single acquisition
- Enables better Bluesky integration and concurrent operations
- Proper Ophyd/Bluesky best practices

## Implementation Changes

### 1. Add stage()/unstage() Methods

```python
def stage(self):
    """
    Prepare device for acquisition series.
    Called once before multiple trigger/read cycles.
    """
    # Configure circular buffer size
    buffer_mb = int(self._num_slices * 2304 * 2304 * 2 * 2 / (1024**2))
    buffer_mb = max(buffer_mb, 2000)  # Minimum 2GB
    self.core.setCircularBufferMemoryFootprint(buffer_mb)

    # Wait for settings to apply
    self.core.waitForDevice(self.camera_name)
    self.core.waitForDevice(self.scanner_name)

    return super().stage()


def unstage(self):
    """Cleanup after acquisition series."""
    try:
        # Reset SPIM to idle
        self.core.setProperty(self.scanner_name, "SPIMState", "Idle")
    except:
        pass  # Best effort cleanup

    return super().unstage()
```

### 2. Refactor trigger() - Hardware Only

```python
def trigger(self):
    """
    Configure hardware and start triggered acquisition.

    Returns Status that completes when images are IN BUFFER.
    Does NOT retrieve images - that's done in read().

    Duration: ~2.7s (1s config + 1.7s acquisition)
    """

    def acquisition_thread():
        try:
            start_time = time.time()

            # 1. Configure camera and SPIM (~1s)
            self._configure_camera_for_hardware_trigger()
            self._configure_spim_timing()

            # 2. Clear circular buffer
            self.core.clearCircularBuffer()

            # 3. Start camera sequence (waiting for external triggers)
            self.core.prepareSequenceAcquisition(self.camera_name)
            self.core.startSequenceAcquisition(
                self.camera_name,
                self._num_slices,
                0,  # intervalMs
                True,  # stopOnOverflow
            )

            # 4. Trigger SPIM state machine (~1.7s for 100 slices @ 59fps)
            self.core.setProperty(self.scanner_name, "SPIMState", "Running")

            # 5. Wait for images to be IN BUFFER (not retrieved!)
            self._wait_for_buffer_fill(self._num_slices, timeout=5.0)

            # 6. Images are ready in buffer
            elapsed = time.time() - start_time
            self._acquisition_time = elapsed
            self._images_ready = True

            print(f"Hardware acquisition complete in {elapsed:.2f}s")
            print(f"Images in buffer: {self.core.getRemainingImageCount()}")

            status.set_finished()

        except Exception as e:
            # Cleanup on error
            try:
                self.core.stopSequenceAcquisition(self.camera_name)
                self.core.setProperty(self.scanner_name, "SPIMState", "Idle")
            except:
                pass

            self._images_ready = False
            status.set_exception(e)

    status = DeviceStatus(self)
    self._images_ready = False
    self._last_volume = None

    thread = threading.Thread(
        target=acquisition_thread, daemon=True, name=f"SPIM-Trigger-{self.name}"
    )
    thread.start()

    return status
```

### 3. Move Image Retrieval to read()

```python
def read(self):
    """
    Retrieve images from Micro-Manager circular buffer.

    This is the SLOW operation (~22s for 100 slices over rpyc).
    Called AFTER trigger() completes.

    Returns data in Bluesky event document format.
    """
    if not self._images_ready:
        raise RuntimeError(
            "read() called before trigger() completed. "
            "No images in buffer. Call trigger() first and wait for Status."
        )

    start_time = time.time()

    # Retrieve volume from buffer (SLOW: ~22s over rpyc)
    volume = self._retrieve_volume_from_buffer()

    retrieval_time = time.time() - start_time
    print(f"Retrieved {volume.shape} volume in {retrieval_time:.2f}s")

    # Cache for subsequent reads
    self._last_volume = volume
    self._last_volume_time = retrieval_time

    timestamp = time.time()

    return {self.name: {"value": volume, "timestamp": timestamp}}
```

### 4. Add Helper Methods

```python
def _wait_for_buffer_fill(self, expected_count, timeout):
    """
    Wait for circular buffer to contain all expected images.

    Uses two strategies:
    1. isSequenceRunning() - wait for hardware to finish
    2. getRemainingImageCount() - verify buffer has images
    """
    start_time = time.time()

    # Wait for sequence to finish
    while self.core.isSequenceRunning(self.camera_name):
        if time.time() - start_time > timeout:
            self.core.stopSequenceAcquisition(self.camera_name)
            raise TimeoutError(f"Sequence acquisition timeout after {timeout:.1f}s")
        time.sleep(0.01)

    # Verify buffer has all images
    actual_count = self.core.getRemainingImageCount()

    if actual_count < expected_count:
        raise RuntimeError(
            f"Incomplete acquisition: expected {expected_count} images, "
            f"but buffer contains only {actual_count}. "
            f"Check: buffer overflow, hardware triggers, camera frame drops."
        )

    if actual_count > expected_count:
        print(f"Warning: Buffer has {actual_count} images, expected {expected_count}")


def _retrieve_volume_from_buffer(self):
    """
    Retrieve all images from MM circular buffer.

    This is the rpyc bottleneck: ~22s for 100 images.
    Each popNextImage() + rpyc.classic.obtain() takes ~220ms.
    """
    images = []

    for i in range(self._num_slices):
        if self.core.getRemainingImageCount() == 0:
            raise RuntimeError(f"Buffer underrun at slice {i}/{self._num_slices}")

        # Pop and transfer over rpyc (SLOW: ~220ms per image)
        img = self.core.popNextImage()
        import rpyc

        img = rpyc.classic.obtain(img)

        images.append(img)

        # Progress logging
        if (i + 1) % 10 == 0:
            print(f"Retrieved {i + 1}/{self._num_slices} slices")

    return np.array(images)
```

### 5. Add State Tracking

```python
def __init__(self, scanner_device_name, camera_device_name, core, **kwargs):
    # ... existing init ...

    # State tracking
    self._images_ready = False
    self._acquisition_time = None
    self._last_volume = None
    self._last_volume_time = None

    # Thread safety
    self._acquisition_lock = threading.Lock()
```

## Testing Changes

### Update test_volume_bluesky.py

```python
# Current usage (works the same):
uid = RE(acquire_spim_volume(vol_scanner, num_slices=100))

# What happens internally (new behavior):
# 1. trigger() runs (~2.7s) - Status completes
# 2. read() runs (~22s) - retrieves volume
# Total: ~24.7s (same as before)

# NEW CAPABILITY: Parallel triggering
from bluesky.plans import trigger_and_read

# Trigger two volume scanners in parallel!
uid = RE(trigger_and_read([vol_scanner_A, vol_scanner_B]))
# Both trigger concurrently (~2.7s)
# Then both read sequentially (~22s + 22s = 44s total)
# vs old approach: 24.7s + 24.7s = 49.4s
```

## Future Optimizations (Eliminate rpyc Bottleneck)

### Long-term: 22s → 1-2s retrieval

**Option 1: Run Ophyd on MM Server** (Best)
- Eliminate rpyc entirely
- Direct numpy array access
- 100 images in ~1-2s instead of 22s

**Option 2: Custom MM Bulk Retrieval Plugin**
- Single rpyc call for all images
- Serialize/compress before transfer
- 22s → 5-10s

**Option 3: Async rpyc Pipelining**
- Overlap network transfers
- 22s → 15s

## Files to Modify

1. **gently/devices.py**
   - DiSPIMVolumeScanner class (lines 1001-1360)
   - Add stage()/unstage()
   - Refactor trigger() to only fill buffer
   - Move image retrieval to read()
   - Add _wait_for_buffer_fill() helper
   - Add state tracking

2. **test_volume_bluesky.py**
   - No changes needed (API remains the same)
   - Add example of parallel triggering

3. **gently/plans.py**
   - No changes needed (plans use trigger_and_read())

## Summary

**Immediate benefit**: Proper Ophyd/Bluesky semantics, enables parallel operations

**Same total time**: ~24s for single acquisition (hardware is the same)

**Future optimization path**: Clear separation enables rpyc elimination strategies

**Bluesky integration**: Follows Ophyd best practices for detector devices
