import threading
import time

import serial


class AcuityNanoPrecisionThermalizerSerial:
    def __init__(self, com_port, baud_rate=115200):
        self.telemetry = {
            "target": 20.0,
            "water": 20.0,
            "peltier": 20.0,
            "state": "DISCONNECTED",
            "errors": "0",
        }
        self.running = True
        self.ser = serial.Serial(com_port, baud_rate, timeout=0.1)
        time.sleep(2)

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if "=" in line:
                        key, val = line.split("=", 1)
                        if key == "TARGET":
                            self.telemetry["target"] = float(val)
                        elif key == "WATER":
                            self.telemetry["water"] = float(val)
                        elif key == "ACTUAL":
                            self.telemetry["peltier"] = float(val)
                        elif key == "STATE":
                            self.telemetry["state"] = val
                        elif key == "ERRORS":
                            self.telemetry["errors"] = val
            except Exception:
                pass
            time.sleep(0.01)

    def close(self):
        self.running = False
        if self.ser.is_open:
            self.ser.close()

    def set_temperature(self, target_celsius):
        if 0.0 <= target_celsius <= 99.9:
            cmd = f"TEMP={target_celsius}\n"
            self.ser.write(cmd.encode("utf-8"))
        else:
            raise ValueError("Target must be between 0.0 and 99.9 C")

    def enable_tec(self, enable=True):
        val = "1" if enable else "0"
        cmd = f"ENABLE={val}\n"
        self.ser.write(cmd.encode("utf-8"))

    def set_feedback_sensor(self, use_peltier=False):
        val = "1" if use_peltier else "0"
        cmd = f"SENSOR={val}\n"
        self.ser.write(cmd.encode("utf-8"))

    def get_water_temp(self):
        return self.telemetry["water"]

    def get_system_state(self):
        return self.telemetry["state"]

    def wait_for_target(self, timeout_seconds=300):
        start = time.time()
        while time.time() - start < timeout_seconds:
            if "[ SYSTEM LOCKED ]" in self.telemetry["state"]:
                return True
            time.sleep(0.5)
        return False


if __name__ == "__main__":
    import time

    print("Connecting to ACUITYnano...")
    acuity = AcuityNanoPrecisionThermalizerSerial("COM8")

    print("Commanding 37.0 C...")
    acuity.set_temperature(37.0)
    acuity.enable_tec(True)

    print("Waiting for thermal stabilization...")
    if acuity.wait_for_target(timeout_seconds=600):
        print(f"System locked at {acuity.get_water_temp()} C!")
        # Trigger external camera or syringe pump here
    else:
        print("Timeout reached before system stabilized.")

    acuity.close()
