# Custom ESPectre Minimal Prototype

This project is a deliberately small, custom Wi‑Fi CSI pipeline built around the ESP32-S3 and a Tenda router.

The aim is to get a working motion/idle detector first, while keeping the code structure extensible enough to add the later activity classifier for running, walking, and sitting.

## Project goal

1. Receive CSI packets on the ESP32-S3
2. Stream them to a host machine
3. Convert each packet window into features
4. Train a motion/idle classifier first
5. Extend the same structure later to classify running, walking, and sitting

This project intentionally avoids the full ESPectre framework and keeps the design minimal and debuggable. The collector uses the runtime CSI length reported by the ESP32 and normalizes each 100-packet window to the largest CSI shape observed within that window.

## Repository structure

```text
custom-espectre/
├── README.md                  # Project overview and run guide
├── requirements.txt           # Python dependencies
├── src/
│   └── custom_espectre/
│       ├── __init__.py
│       ├── config.py
│       ├── features.py
│       ├── data_io.py
│       ├── collect.py
│       ├── train_motion_idle.py
│       └── predict_motion_idle.py
├── data/
│   └── raw/
│       ├── idle/
│       └── motion/
├── models/
├── firmware/
│   ├── README.md
│   └── custom_espectre_csi/
│       └── custom_espectre_csi.ino
└── .gitignore
```

## Why this design

The first milestone is motion detection, because it is easier to debug and more robust than jumping straight into running/walking/sitting classification.

The code is structured so the same pipeline can later expand from binary classification to multi-class HAR with almost no architectural rewrite:

- `config.py` stores global shapes and settings
- `features.py` owns feature extraction
- `collect.py` owns CSI collection
- `train_motion_idle.py` owns the training/evaluation loop
- `predict_motion_idle.py` owns the inference loop

Adding a second stage later is just a matter of adding a new class label set and a new model training path.

## Recommended workflow

### Phase 1 — motion vs idle

Use this before trying to detect running, walking, and sitting.

1. Collect motion and idle windows from the ESP32
2. Train a simple classifier
3. Make sure confidence and false positives are acceptable
4. Only then move to the 3-class activity stage

### Phase 2 — activity classification

Later, extend the same pipeline with labels:

- running
- walking
- sitting

The model and training code can be copied and adapted without changing the surrounding architecture.

## Exact Step 1 test procedure

Step 1 is complete when the ESP32 joins the local Tenda Wi-Fi, emits CSI JSON, the host saves labelled windows, and a model is evaluated on real windows. This project uses Arduino IDE for the ESP32 firmware. Internet access is not required.

### 1) Install Arduino support

Install Arduino IDE, then open **Boards Manager** and install **esp32 by Espressif Systems**. Select your ESP32-S3 board under **Tools > Board > esp32**.

Connect the board with a USB data cable and select its port under **Tools > Port**.

### 2) Edit and upload the sketch

Open this file in Arduino IDE:

`firmware/custom_espectre_csi/custom_espectre_csi.ino`

Edit these two lines with your local Tenda Wi-Fi credentials:

```cpp
const char *WIFI_SSID = "REPLACE_WITH_TENDA_SSID";
const char *WIFI_PASSWORD = "REPLACE_WITH_TENDA_PASSWORD";
```

Choose the ESP32-S3 board, choose the correct COM port, select **Tools > Upload Speed > 921600**, and click **Upload**.

### 3) Open Serial Monitor

Open Arduino **Serial Monitor** and set the baud rate to `921600`. You must see:

```text
Wi-Fi connected; IP address: 192.168.x.x
CSI enabled.
```

Write down the printed ESP32 IP address.

### 4) Generate local Wi-Fi traffic

Connect the laptop to the same Tenda network. Verify local reachability:

```powershell
ping 192.168.0.123
```

Replace the address with the ESP32 IP. Then close Arduino Serial Monitor because the Python collector needs the COM port.

In a separate PowerShell, generate local UDP traffic directly to the ESP32:

```powershell
$espIp = "192.168.0.123"
$udp = [System.Net.Sockets.UdpClient]::new()
$bytes = [System.Text.Encoding]::ASCII.GetBytes("csi")
try {
	while ($true) {
		[void]$udp.Send($bytes, $bytes.Length, $espIp, 5005)
		Start-Sleep -Milliseconds 10
	}
}
finally {
	$udp.Dispose()
}
```

Replace `$espIp` with the actual ESP32 IP. Keep this running during data collection. Press `Ctrl+C` to stop it.

### 5) Prepare Python

From the custom project directory:

```powershell
cd C:\Users\dhruv\PROJECTS\PJT-1\custom-espectre
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

### 6) Collect idle data

Keep the room empty and run:

```powershell
..\.venv\Scripts\python.exe -m custom_espectre.collect --label idle --count 30
```

The collector defaults to `COM8`. Files should appear under `data\raw\idle\`.

### 7) Collect motion data

Run:

```powershell
..\.venv\Scripts\python.exe -m custom_espectre.collect --label motion --count 30
```

Walk repeatedly through the fixed sensing path. Files should appear under `data\raw\motion\`.

### 8) Train on real data

```powershell
..\.venv\Scripts\python.exe -m custom_espectre.train_motion_idle --data-dir data/raw
```

Use this real-data report, not the synthetic score, to judge Step 1.

### 9) Run live prediction

```powershell
..\.venv\Scripts\python.exe -m custom_espectre.predict_motion_idle --model models/motion_idle_model.pkl --count 20
```

Stay still for several windows, then move through the sensing path. Predictions should change consistently.

### Checkpoint before Step 2

Do not add running/walking/sitting yet. First confirm:

- Wi-Fi reconnects after reboot
- CSI JSON arrives continuously enough to form windows
- at least 30 idle and 30 motion windows exist
- real-data evaluation is better than chance
- live predictions are repeatable in the same room

Only after these checks pass should the label set be expanded for Step 2.

## Notes for real hardware

- The ESP32-S3 should be the CSI receiver.
- The Tenda router provides the local Wi-Fi network. The laptop generates local traffic directly to the ESP32; internet access is not required.
- Keep the room geometry fixed during initial testing.
- Collect idle and motion samples in the same conditions you will later demo in.
- Do not try to jump directly to the 3-class activity model until binary motion detection is stable.

## Next milestones

1. Build the custom ESP32 CSI collector and confirm packets are arriving
2. Collect a clean idle dataset
3. Collect a clean motion dataset
4. Train the motion/idle model
5. Extend to running/walking/sitting with the same architecture

This is the right order for a result that actually works.
