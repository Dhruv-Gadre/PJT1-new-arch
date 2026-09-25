import json
import time
import serial

PORT = "COM8"
BAUD = 921600
PACKETS = 1000

ser = serial.Serial(PORT, BAUD, timeout=2.0)

count = 0
start = None

print(f"Measuring CSI packet rate on {PORT}...")
print(f"Collecting {PACKETS} valid CSI packets...\n")

try:
    while count < PACKETS:
        line = ser.readline()

        if not line:
            continue

        try:
            payload = json.loads(
                line.decode("utf-8", errors="replace").strip()
            )

            amp = payload["a"]
            pha = payload["p"]

            if len(amp) != len(pha) or not amp:
                continue

        except Exception:
            continue

        if count == 0:
            start = time.perf_counter()

        count += 1

        if count % 100 == 0:
            print(f"{count}/{PACKETS} packets received")

finally:
    ser.close()

if start is None:
    raise RuntimeError("No valid CSI packets received")

elapsed = time.perf_counter() - start

rate = (PACKETS - 1) / elapsed

print("\n========================================")
print("CSI PACKET RATE")
print("========================================")
print(f"Packets : {PACKETS}")
print(f"Time    : {elapsed:.3f} seconds")
print(f"Rate    : {rate:.2f} packets/second")
print(f"Interval: {1000 / rate:.2f} ms")
print("========================================")