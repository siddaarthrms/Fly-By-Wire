"""Frame-by-frame demo telemetry for circuit-lightup visualization."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fly_brain import FlyBrain


class DemoTelemetryRecorder:
    """Writes meta.json, frames.csv, grids.npy, circuit_topology.npz, circuit_frames.npz."""

    FRAME_FIELDS = [
        "frame_idx",
        "t_sec",
        "carla_frame_id",
        "right_push",
        "left_push",
        "steer_raw",
        "steer",
        "brake",
        "throttle",
        "brake_drive",
        "brake_neuron",
        "brake_loom",
        "throttle_drive",
        "throttle_neuron",
        "throttle_flux",
        "loom_signal",
    ]

    def __init__(self, run_dir: str | Path, *, meta: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, Any] = dict(meta or {})
        self._meta.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        self._meta["run_dir"] = str(self.run_dir.resolve())

        self._frames: list[dict[str, float | int]] = []
        self._grids: list[np.ndarray] = []
        self._neuron_brightness: list[np.ndarray] = []
        self._right_contrib: list[np.ndarray] = []
        self._left_contrib: list[np.ndarray] = []
        self._brake_contrib: list[np.ndarray] = []
        self._throttle_contrib: list[np.ndarray] = []
        self._topology_written = False

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def write_topology(self, brain: FlyBrain) -> None:
        topo = brain.circuit_topology_arrays()
        np.savez_compressed(self.run_dir / "circuit_topology.npz", **topo)
        self._meta["neuron_count"] = int(len(topo["neuron_ids"]))
        self._meta["channel_input_counts"] = {
            "right_steer": int(len(topo["right_idx"])),
            "left_steer": int(len(topo["left_idx"])),
            "brake": int(len(topo["brake_idx"])),
            "throttle": int(len(topo["throttle_idx"])),
        }
        self._topology_written = True

    def record_frame(
        self,
        *,
        frame_idx: int,
        t_sec: float,
        grid_u8: np.ndarray,
        brain_diag: dict[str, np.ndarray | float],
        scalars: dict[str, float | int],
    ) -> None:
        row: dict[str, float | int] = {
            "frame_idx": int(frame_idx),
            "t_sec": float(t_sec),
        }
        for key in self.FRAME_FIELDS:
            if key in ("frame_idx", "t_sec"):
                continue
            if key in scalars:
                row[key] = scalars[key]
        if "right_push" not in row:
            row["right_push"] = float(brain_diag["right_push"])
        if "left_push" not in row:
            row["left_push"] = float(brain_diag["left_push"])
        if "brake_drive" not in row:
            row["brake_drive"] = float(brain_diag["brake_drive"])
        if "throttle_drive" not in row:
            row["throttle_drive"] = float(brain_diag["throttle_drive"])
        self._frames.append(row)
        self._grids.append(np.asarray(grid_u8, dtype=np.uint8).reshape(30, 30).copy())
        self._neuron_brightness.append(np.asarray(brain_diag["neuron_brightness"], dtype=np.float32))
        self._right_contrib.append(np.asarray(brain_diag["right_contrib"], dtype=np.float32))
        self._left_contrib.append(np.asarray(brain_diag["left_contrib"], dtype=np.float32))
        self._brake_contrib.append(np.asarray(brain_diag["brake_contrib"], dtype=np.float32))
        self._throttle_contrib.append(np.asarray(brain_diag["throttle_contrib"], dtype=np.float32))

    def attach_video_path(self, path: str | Path) -> None:
        self._meta["video_path"] = str(Path(path).resolve())

    def close(self) -> Path:
        if not self._frames:
            self._meta["frame_count"] = 0
            (self.run_dir / "meta.json").write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
            print(f"Telemetry run empty -> {self.run_dir.resolve()}", flush=True)
            return self.run_dir

        frames_path = self.run_dir / "frames.csv"
        with frames_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FRAME_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in self._frames:
                writer.writerow(row)

        grids = np.stack(self._grids, axis=0)
        np.save(self.run_dir / "grids.npy", grids)

        np.savez_compressed(
            self.run_dir / "circuit_frames.npz",
            neuron_brightness=np.stack(self._neuron_brightness, axis=0),
            right_contrib=np.stack(self._right_contrib, axis=0),
            left_contrib=np.stack(self._left_contrib, axis=0),
            brake_contrib=np.stack(self._brake_contrib, axis=0),
            throttle_contrib=np.stack(self._throttle_contrib, axis=0),
        )

        self._meta["frame_count"] = len(self._frames)
        self._meta["duration_sec"] = float(self._frames[-1]["t_sec"])
        self._meta["files"] = {
            "frames_csv": str(frames_path.name),
            "grids_npy": "grids.npy",
            "circuit_topology_npz": "circuit_topology.npz",
            "circuit_frames_npz": "circuit_frames.npz",
        }
        (self.run_dir / "meta.json").write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        print(
            f"Telemetry saved ({len(self._frames)} frames) -> {self.run_dir.resolve()}",
            flush=True,
        )
        return self.run_dir


def default_telemetry_run_dir(label: str, *, output_dir: str | Path = "demos/telemetry") -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{safe}_{ts}"


__all__ = ["DemoTelemetryRecorder", "default_telemetry_run_dir"]
