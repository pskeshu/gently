# DiSPIM Hardware Safety Checklist
## Colleague Consultation Guide

This checklist helps ensure safe deployment of the new Gently DiSPIM control software by gathering essential safety information from experienced users.

---

## 🚨 CRITICAL: Complete Before Any Hardware Testing

**Date:** ________________  
**Colleague Name:** ________________  
**Experience with this DiSPIM system:** ________________  

---

## 1. Device Safety Limits

### 1.1 Piezo Stages (Focus Control)

**Piezo A (Side A focus):**
- Safe position range: _______ μm to _______ μm
- Typical operating range: _______ μm to _______ μm  
- Home position (safe startup): _______ μm
- Emergency safe position: _______ μm
- Maximum safe speed: _______ μm/s
- Forbidden positions/zones: _________________
- **Sample collision risk at:** _______ μm
- **Objective collision risk at:** _______ μm

**Piezo B (Side B focus):**
- Safe position range: _______ μm to _______ μm
- Typical operating range: _______ μm to _______ μm
- Home position (safe startup): _______ μm  
- Emergency safe position: _______ μm
- Maximum safe speed: _______ μm/s
- Forbidden positions/zones: _________________
- **Sample collision risk at:** _______ μm
- **Objective collision risk at:** _______ μm

### 1.2 Galvo Mirrors (Beam Steering)

**Galvo A (Side A beam steering):**
- Safe angle range: _______ ° to _______ °
- Typical operating range: _______ ° to _______ °
- Home position (safe startup): _______ °
- Emergency safe position: _______ °
- Maximum safe speed: _______ °/s
- **Angles that hit detectors:** _________________
- **Angles that hit other surfaces:** _________________

**Galvo B (Side B beam steering):**
- Safe angle range: _______ ° to _______ °
- Typical operating range: _______ ° to _______ °
- Home position (safe startup): _______ °
- Emergency safe position: _______ °
- Maximum safe speed: _______ °/s
- **Angles that hit detectors:** _________________
- **Angles that hit other surfaces:** _________________

### 1.3 XY Stage (Sample Positioning)

**X-axis:**
- Safe position range: _______ μm to _______ μm
- Typical operating range: _______ μm to _______ μm
- Home position: _______ μm
- Emergency position: _______ μm
- Maximum safe speed: _______ μm/s

**Y-axis:**
- Safe position range: _______ μm to _______ μm  
- Typical operating range: _______ μm to _______ μm
- Home position: _______ μm
- Emergency position: _______ μm
- Maximum safe speed: _______ μm/s

**Sample holder collision zones:** _________________

---

## 2. Operational Safety

### 2.1 Daily Startup Procedure
What positions do you move devices to when starting up each day?

1. **First device to move:** _________________ to _______ 
2. **Second device to move:** _________________ to _______
3. **Third device to move:** _________________ to _______
4. **Order matters because:** _________________

### 2.2 Daily Shutdown Procedure  
What positions should devices be in when shutting down?

1. **Piezo A:** _______ μm
2. **Piezo B:** _______ μm
3. **Galvo A:** _______ °
4. **Galvo B:** _______ °
5. **XY Stage:** X=_______ μm, Y=_______ μm
6. **Reason for these positions:** _________________

### 2.3 Emergency Procedures
**If software crashes or hardware gets stuck:**

1. **Immediate action:** _________________
2. **Emergency stop button location:** _________________
3. **Safe manual positions:**
   - Piezo A: _______ μm
   - Piezo B: _______ μm  
   - Galvo A: _______ °
   - Galvo B: _______ °
4. **Who to contact:** _________________
5. **Hardware reset procedure:** _________________

---

## 3. Sample and Hardware Protection

### 3.1 Sample Protection
**What can damage samples?**

- **Piezo positions that crush samples:** _________________
- **Galvo angles that photo-damage:** _________________  
- **XY positions that hit sample holder:** _________________
- **Movement speeds that damage samples:** _________________

### 3.2 Hardware Protection
**What can damage the microscope?**

- **Piezo positions that hit objectives:** _________________
- **Galvo angles that damage detectors:** _________________
- **XY positions that collide with stages:** _________________
- **Movement sequences to avoid:** _________________

### 3.3 Optical Protection
**What can damage optics or detectors?**

- **Galvo angles that direct beam incorrectly:** _________________
- **Laser power settings that damage detectors:** _________________
- **Beam paths to avoid:** _________________

---

## 4. Typical Operating Parameters

### 4.1 Autofocus Parameters
**For typical autofocus scans:**

- **Piezo scan range:** ±_______ μm around current position
- **Typical step size:** _______ μm
- **Number of steps:** _______
- **Safe scan speed:** _______ μm/s

### 4.2 Calibration Parameters
**For two-point calibration:**

- **Typical calibration point 1:** _______ μm
- **Typical calibration point 2:** _______ μm  
- **Safe positions for calibration imaging:** _________________

### 4.3 Embryo Detection Parameters
**For embryo detection with bottom camera:**

- **XY scan range:** X: ±_______ μm, Y: ±_______ μm
- **Step size for detection:** _______ μm
- **Safe Z position during XY scan:** _______ μm

---

## 5. Hardware-Specific Notes

### 5.1 Known Issues
**Any known problems with this specific system:**

- **Devices that stick or drift:** _________________
- **Positions to avoid:** _________________  
- **Movement sequences that cause problems:** _________________
- **Hardware quirks:** _________________

### 5.2 Maintenance Positions
**Positions needed for cleaning/maintenance:**

- **For objective cleaning:** _________________
- **For stage cleaning:** _________________
- **For detector access:** _________________
- **For safe manual intervention:** _________________

---

## 6. Validation Questions

### 6.1 Movement Validation
**Before each automated sequence, what should be checked?**

- [ ] Sample is properly mounted and won't move
- [ ] Objectives are clean and properly positioned
- [ ] No tools or obstacles in movement path
- [ ] _________________
- [ ] _________________

### 6.2 Safety Indicators
**What indicates unsafe conditions?**

- **Visual indicators:** _________________
- **Sounds that indicate problems:** _________________  
- **Software warnings to take seriously:** _________________
- **When to stop immediately:** _________________

---

## 7. Testing Approval

### 7.1 Phase 1: Read-Only Testing
**Approval for reading device positions (no movements):**

- [ ] **Approved** - Safe to connect software and read positions
- [ ] **Not approved** - Concerns: _________________

**Colleague signature:** _________________ **Date:** _______

### 7.2 Phase 2: Write-Back Testing  
**Approval for writing back identical positions (no net movement):**

- [ ] **Approved** - Safe to write back same positions
- [ ] **Not approved** - Concerns: _________________

**Colleague signature:** _________________ **Date:** _______

### 7.3 Phase 3: Limited Movement Testing
**Approval for small test movements within safe limits:**

- [ ] **Approved** - Safe for limited movements with filled-in limits
- [ ] **Not approved** - Additional safety measures needed: _________________

**Colleague signature:** _________________ **Date:** _______

---

## 8. Contact Information

**Primary contact for hardware questions:** _________________  
**Phone:** _________________  
**Email:** _________________  

**Backup contact:** _________________  
**Phone:** _________________  
**Email:** _________________  

**Emergency contact (after hours):** _________________  
**Phone:** _________________  

---

## 9. Additional Notes

**Any other safety considerations, warnings, or advice:**

_________________________________________________________________  
_________________________________________________________________  
_________________________________________________________________  
_________________________________________________________________  
_________________________________________________________________  

---

**Checklist completed by:** _________________  
**Date:** _________________  
**Signature:** _________________  

---

## Next Steps After Completion

1. **Create limits configuration file:**
   ```bash
   python safe_limits_config.py create_template
   ```

2. **Fill in the generated template with information from this checklist**

3. **Test hardware connection (read-only):**
   ```bash
   python test_hardware_connection.py /path/to/micromanager config.cfg
   ```

4. **Test write-back verification:**
   ```bash  
   python test_read_write_back.py /path/to/micromanager config.cfg baseline_state.json
   ```

5. **Load colleague limits and begin careful testing:**
   ```bash
   python safe_limits_config.py load_colleague_file colleague_limits.json
   ```

**⚠️ DO NOT proceed to hardware testing until this checklist is complete and signed!**