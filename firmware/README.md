# Firmware Overview

This directory contains the firmware for the ESP32-S3 Wi-Fi CSI sensing pipeline.

There are two Arduino sketches depending on your workflow:

| Sketch | Path | Purpose |
| :--- | :--- | :--- |
| **Collector** | `custom_espectre_csi/custom_espectre_csi.ino` | Streams raw CSI JSON to host over USB serial for data collection and model training |
| **On-Device Inference** | `esp_inference/esp_inference.ino` | Runs Stage A (Math) + Stage B (Logistic Regression) natively on the ESP32-S3 |

> **Note:** The older ESP-IDF files in `main/` are retained for historical reference only. Use the Arduino sketches below for all current workflows.

---

## 1. Raw CSI Collector (`custom_espectre_csi/`)

Streams CSI packets as JSON lines to the host serial port. The host Python pipeline collects these into 100-packet `.npz` windows for calibration and training.

### Upload & Run
1. Open `custom_espectre_csi/custom_espectre_csi.ino` in Arduino IDE.
2. Edit `WIFI_SSID` and `WIFI_PASSWORD` for your 2.4 GHz network (e.g., Tenda router).
3. Select board **ESP32S3 Dev Module** (or your specific board variant) and port.
4. Upload at `921600` baud.
5. In Python, start the collector:
   ```bash
   python -m custom_espectre.collect --port COM3 --label idle --count 20
   ```

---

## 2. On-Device Native Inference (`esp_inference/`)

Runs both classification stages **completely on the ESP32-S3 hardware** without needing a Python process running on a host computer.

```
Wi-Fi CSI packets (100-packet window)
  │
  ▼
Stage A: Mathematics (ESPectre MVS + NBVI)
  │
  ├──► Score below threshold ─────────────► JSON: {"stage_a": "idle", ...}
  │
  └──► Score above threshold (MOTION)
         │
         ▼
       Stage B: Logistic Classifier (Compiled C array)
         │
         ├──► walking ──────────────────► JSON: {"stage_a": "motion", "stage_b": "walking", ...}
         └──► running ──────────────────► JSON: {"stage_a": "motion", "stage_b": "running", ...}
```

### Stage A: Mathematics (No ML Model Needed)
- **Moving Variance Score (MVS)**: Sliding-window variance over the mean amplitude envelope.
- **Narrowband Variance Indicator (NBVI)**: Per-subcarrier temporal variance normalized against the quiet-room baseline.
- **Automatic Boot Calibration**: The first 10 windows (`CALIBRATION_WINDOWS`) collected at power-on are treated as idle baseline to adapt to the physical room. Keep the room quiet for ~15 seconds after booting.

### Stage B: Logistic Regression (Optional On-Device Activity Classifier)
Stage B runs the 13 feature extraction metrics on-device and applies the trained weights + StandardScaler directly as C arrays.

#### How to export Stage B weights to the firmware:
1. Train Stage B on your collected walking/running data:
   ```bash
   $env:PYTHONPATH="src"
   python -m custom_espectre.train_activity --data-dir data/windows --out models/model_b.pkl
   ```
2. Export weights to C header:
   ```bash
   python -m custom_espectre.export_model_b --model models/model_b.pkl --out firmware/esp_inference/model_b_weights.h
   ```
3. In `esp_inference/esp_inference.ino`, enable Stage B:
   ```cpp
   #define ENABLE_STAGE_B 1
   ```
4. Flash the sketch to the ESP32-S3 at `921600` baud.

### Serial Monitor Output
Open Serial Monitor at `921600` baud to observe live JSON classification output:
```json
{"window": 1, "stage_a": "idle", "mvs": 0.000041, "nbvi": 1.02}
{"window": 2, "stage_a": "motion", "stage_b": "walking", "mvs": 0.012450, "nbvi": 5.41}
{"window": 3, "stage_a": "motion", "stage_b": "running", "mvs": 0.048910, "nbvi": 14.82}
```

---

## Wi-Fi Traffic Generation
CSI packets are only captured when Wi-Fi frames are transmitted. If output is sparse:
- Run the PowerShell UDP sender script from a laptop on the same Wi-Fi network:
  ```powershell
  .\udp.ps1 -TargetIp 192.168.0.xxx -Rate 60
  ```
- Ensure client/AP isolation is disabled on the router.
