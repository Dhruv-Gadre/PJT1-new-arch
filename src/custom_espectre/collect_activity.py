"""Collect labelled walking/running CSI windows for Stage B."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


ACTIVITY_LABELS = ["walking", "running"]
WINDOW_SIZE = 100


def normalize_subcarriers(
    values: list[float],
    target_count: int,
) -> np.ndarray:
    """Normalize one packet to the largest shape observed in its window."""
    array = np.asarray(values, dtype=np.float32)

    if array.size == 0:
        raise ValueError("CSI packet contains no subcarriers")

    if array.size >= target_count:
        return array[:target_count]

    return np.pad(
        array,
        (0, target_count - array.size),
        mode="edge",
    )


def save_window(
    amplitudes: np.ndarray,
    phases: np.ndarray,
    label: str,
    out_dir: Path,
) -> Path:
    """Save one labelled Stage-B window."""
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    out_path = out_dir / f"{label}_{timestamp}.npz"

    np.savez(
        out_path,
        amplitudes=amplitudes.astype(np.float32),
        phases=phases.astype(np.float32),
        label=np.array(label),
        timestamp=np.array(timestamp, dtype=np.int64),
    )

    return out_path


def read_serial_windows(
    port: str,
    baud: int = 921600,
    count: int = 10,
):
    """Read CSI packets and yield 100-packet windows."""
    if serial is None:
        raise RuntimeError(
            "pyserial is required. Install with: pip install pyserial"
        )

    ser = serial.Serial(port, baud, timeout=2.0)

    try:
        packets_amp: list[list[float]] = []
        packets_phase: list[list[float]] = []
        yielded = 0

        while True:
            line = ser.readline()

            if not line:
                continue

            try:
                payload = json.loads(
                    line.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                )

                amp = payload["a"]
                pha = payload["p"]

                if len(amp) != len(pha) or not amp:
                    continue

            except Exception:
                continue

            packets_amp.append(amp)
            packets_phase.append(pha)

            if len(packets_amp) >= WINDOW_SIZE:
                target_count = max(
                    len(packet)
                    for packet in packets_amp[:WINDOW_SIZE]
                )

                window_amp = np.stack(
                    [
                        normalize_subcarriers(
                            packet,
                            target_count,
                        )
                        for packet in packets_amp[:WINDOW_SIZE]
                    ],
                    axis=0,
                )

                window_phase = np.stack(
                    [
                        normalize_subcarriers(
                            packet,
                            target_count,
                        )
                        for packet in packets_phase[:WINDOW_SIZE]
                    ],
                    axis=0,
                )

                packets_amp.clear()
                packets_phase.clear()

                yielded += 1

                yield window_amp, window_phase

                if count and yielded >= count:
                    break

    finally:
        ser.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Stage-B walking/running CSI windows"
    )

    parser.add_argument(
        "--port",
        type=str,
        default="COM8",
        help="Serial port for the ESP32-S3",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Serial baud rate",
    )

    parser.add_argument(
        "--label",
        choices=ACTIVITY_LABELS,
        required=True,
        help="Activity label",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of windows to collect",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/stage_b"),
        help="Stage-B data directory",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    out_dir = args.data_dir / args.label

    print(
        f"Collecting {args.count} {args.label} windows "
        f"from {args.port} ..."
    )

    for idx, (amp, pha) in enumerate(
        read_serial_windows(
            args.port,
            args.baud,
            count=args.count,
        ),
        start=1,
    ):
        save_path = save_window(
            amp,
            pha,
            args.label,
            out_dir,
        )

        print(
            f"[{idx}/{args.count}] saved -> {save_path}"
        )

    print("Collection complete.")


if __name__ == "__main__":
    main()