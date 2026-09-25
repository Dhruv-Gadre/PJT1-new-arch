# Firmware notes

The active firmware is the Arduino sketch at `custom_espectre_csi/custom_espectre_csi.ino`.

The older ESP-IDF files in this directory are retained as reference only. Do not open or build them for the current workflow.

The first goal is to stream CSI windows to the host over serial, so that the motion/idle model can be trained and validated on real data.

## Recommended first firmware goal

- connect to the Tenda router over Wi-Fi in station mode
- read CSI packets from the ESP32 radio
- log amplitude and phase values for each packet
- print them as JSON to the USB serial port
- keep the code small and easy to debug

## Later extension

Once the motion/idle path is validated, the same firmware can be expanded to:

- trigger a motion window buffer
- run a small embedded inferencer
- perform running/walking/sitting classification on the device

The current codebase intentionally keeps the firmware layer separate from the training and feature pipeline, so the architecture can evolve without a rewrite.

## Configure and upload

1. Install Arduino IDE and the `esp32` board package by Espressif Systems.
2. Open `custom_espectre_csi/custom_espectre_csi.ino`.
3. Edit the SSID and password constants at the top of the sketch.
4. Select your ESP32-S3 board and its COM port.
5. Upload at `921600` baud.
6. Open Serial Monitor at `921600` baud.

The sketch prints its local IP address and then emits one JSON CSI packet per line. The Python collector consumes those JSON lines and saves 100 packets per `.npz` window.

If packet output is sparse, generate local UDP traffic directly to the ESP32 IP from a laptop on the same Tenda network. Internet access is not required. The ESP32 must be connected to the same 2.4 GHz network as the laptop, and client/AP isolation must be disabled if the devices cannot reach each other.
