"""
ACUITYnano Third-Party Integration SDK
Provides a clean, object-oriented API for external software automation.
"""

import threading
import time

import paho.mqtt.client as mqtt


class AcuityNanoPrecisionThermalizerAPI:
    def __init__(
        self,
        broker="d0246aa97d194c9da52a19e6f46063eb.s1.eu.hivemq.cloud",
        port=8883,
        user="acuitynano",
        password="Bg984V!@wfhBrkp",
    ):
        self.prefix = "acuitynano_hhmi_shroff_diSPIM_001"
        self.telemetry = {
            "target": 20.0,
            "water": 20.0,
            "peltier": 20.0,
            "state": "DISCONNECTED",
            "errors": "0",
        }

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            self.client = mqtt.Client()

        self.client.username_pw_set(user, password)
        self.client.tls_set()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        time.sleep(2)

    def _start_loop(self):
        self.client.connect(broker, port, 60)  # noqa: F821
        self.client.loop_forever()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self.client.subscribe(f"{self.prefix}/telemetry/#")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic.split("/")[-1]
        payload = msg.payload.decode("utf-8")

        if topic == "target":
            self.telemetry["target"] = float(payload)
        elif topic == "water":
            self.telemetry["water"] = float(payload)
        elif topic == "actual":
            self.telemetry["peltier"] = float(payload)
        elif topic == "state":
            self.telemetry["state"] = payload
        elif topic == "errors":
            self.telemetry["errors"] = payload

    def set_temperature(self, target_celsius):
        if 0.0 <= target_celsius <= 99.9:
            self.client.publish(f"{self.prefix}/cmd/temp", str(target_celsius))
        else:
            raise ValueError("Target must be between 0.0 and 99.9 C")

    def enable_tec(self, enable=True):
        val = "1" if enable else "0"
        self.client.publish(f"{self.prefix}/cmd/enable", val)

    def set_feedback_sensor(self, use_peltier=False):
        val = "1" if use_peltier else "0"
        self.client.publish(f"{self.prefix}/cmd/sensor", val)

    def get_water_temp(self):
        return self.telemetry["water"]

    def get_peltier_temp(self):
        return self.telemetry["peltier"]

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

    from acuitynano_precision_thermalizer_api import (  # type: ignore[no-redef]
        AcuityNanoPrecisionThermalizerAPI,
    )

    acuity = AcuityNanoPrecisionThermalizerAPI()
    print("Commanding ACUITYnano to 37.0 C...")
    acuity.set_temperature(30.0)
    acuity.enable_tec(True)

    print("Waiting for thermal stabilization...")
    if acuity.wait_for_target(timeout_seconds=600):
        print(f"System locked at {acuity.get_water_temp()} C!")
        # Trigger image acquisition here
    else:
        print("Timeout reached.")
