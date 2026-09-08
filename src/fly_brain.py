#!/usr/bin/env python3
"""
Fly "pilot" brain: 30x30 grayscale (same layout as training / vision_ingest)
→ raw right_push / left_push (same sums as neural_projector; bridge applies w=1.01 + R0 offset).

Loads CSVs with the stdlib ``csv`` module so a broken pandas/numpy ``read_csv`` does not
block the pilot loop.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Match neural_projector.py
RIGHT_STEER_ID = 720575940633816986
LEFT_STEER_ID = 720575940625653029


def _read_weights_rows(weights_path: str | Path) -> list[dict[str, int | float]]:
    path = Path(weights_path)
    rows: list[dict[str, int | float]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # FlyWire IDs are >2^53; never parse via float.
            rows.append(
                {
                    "pre_pt_root_id": int(str(row["pre_pt_root_id"]).strip()),
                    "post_pt_root_id": int(str(row["post_pt_root_id"]).strip()),
                    "synapse_count": float(row["synapse_count"]),
                }
            )
    return rows


def _read_pixel_map(
    pixel_map_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(pixel_map_path)
    ids: list[int] = []
    gx: list[int] = []
    gy: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(str(row["pt_root_id"]).strip()))
            gx.append(int(str(row["grid_x"]).strip()))
            gy.append(int(str(row["grid_y"]).strip()))
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(gx, dtype=np.int64),
        np.asarray(gy, dtype=np.int64),
    )


def _build_incoming(
    weight_rows: list[dict[str, int | float]],
    idx_map: dict[int, int],
    target_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    sender_ids: list[int] = []
    syn_counts: list[float] = []
    for row in weight_rows:
        if int(row["post_pt_root_id"]) != target_id:
            continue
        pre = int(row["pre_pt_root_id"])
        if pre not in idx_map:
            continue
        sender_ids.append(idx_map[pre])
        syn_counts.append(float(row["synapse_count"]))
    if not sender_ids:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    return (
        np.asarray(sender_ids, dtype=np.int64),
        np.asarray(syn_counts, dtype=np.float64),
    )


class FlyBrain:
    """
    Weights + pixel map once. process_frame matches neural_projector
    right_push / left_push. Optional DNg03 / DNg01/02 CSVs for brake and throttle.
    """

    def __init__(
        self,
        weights_path: str | Path,
        pixel_map_path: str | Path,
        brake_weights_path: str | Path | None = None,
        throttle_weights_path: str | Path | None = None,
    ) -> None:
        weight_rows = _read_weights_rows(weights_path)
        input_neuron_ids, self._grid_x, self._grid_y = _read_pixel_map(pixel_map_path)
        self._neuron_ids = np.asarray(input_neuron_ids, dtype=np.int64)
        idx_map = {int(nid): i for i, nid in enumerate(input_neuron_ids)}

        self._r_idx, self._r_syn = _build_incoming(weight_rows, idx_map, RIGHT_STEER_ID)
        self._l_idx, self._l_syn = _build_incoming(weight_rows, idx_map, LEFT_STEER_ID)
        self._b_idx = np.array([], dtype=np.int64)
        self._b_syn = np.array([], dtype=np.float64)
        self._b_total_syn = 0.0
        self._t_idx = np.array([], dtype=np.int64)
        self._t_syn = np.array([], dtype=np.float64)
        self._t_total_syn = 0.0
        if brake_weights_path is not None:
            p = Path(brake_weights_path)
            if p.is_file():
                counts: dict[int, int] = {}
                with p.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pre = row.get("pre_pt_root_id")
                        if pre is None or str(pre).strip() == "":
                            continue
                        pre_id = int(str(pre).strip())
                        counts[pre_id] = counts.get(pre_id, 0) + 1
                b_idx: list[int] = []
                b_syn: list[float] = []
                for pre_id, n_syn in counts.items():
                    if pre_id not in idx_map:
                        continue
                    b_idx.append(idx_map[pre_id])
                    b_syn.append(float(n_syn))
                if b_idx:
                    self._b_idx = np.asarray(b_idx, dtype=np.int64)
                    self._b_syn = np.asarray(b_syn, dtype=np.float64)
                    self._b_total_syn = float(np.sum(self._b_syn))
        if throttle_weights_path is not None:
            p = Path(throttle_weights_path)
            if p.is_file():
                counts: dict[int, int] = {}
                with p.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pre = row.get("pre_pt_root_id")
                        if pre is None or str(pre).strip() == "":
                            continue
                        pre_id = int(str(pre).strip())
                        counts[pre_id] = counts.get(pre_id, 0) + 1
                t_idx: list[int] = []
                t_syn: list[float] = []
                for pre_id, n_syn in counts.items():
                    if pre_id not in idx_map:
                        continue
                    t_idx.append(idx_map[pre_id])
                    t_syn.append(float(n_syn))
                if t_idx:
                    self._t_idx = np.asarray(t_idx, dtype=np.int64)
                    self._t_syn = np.asarray(t_syn, dtype=np.float64)
                    self._t_total_syn = float(np.sum(self._t_syn))

    @staticmethod
    def _grid_to_neuron_brightness(image_30x30_gray: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        g = np.asarray(image_30x30_gray)
        if g.shape != (30, 30):
            raise ValueError(f"Expected (30,30) grayscale, got {g.shape}")
        if g.dtype == np.float64 or g.dtype == np.float32 or (
            g.max() <= 1.0 + 1e-6 and g.min() >= -1e-6
        ):
            brightness = g.astype(np.float64)
            if brightness.max() > 1.0 + 1e-6:
                brightness = (brightness / 255.0).astype(np.float64)
        else:
            brightness = (g.astype(np.float64) / 255.0)
        return brightness[grid_y, grid_x]

    def process_frame(self, image_30x30_gray: np.ndarray) -> tuple[float, float]:
        """(30, 30) uint8 0..255 or float in [0, 1] -> (right_push, left_push)."""
        b = self._grid_to_neuron_brightness(image_30x30_gray, self._grid_x, self._grid_y)
        right_push = float(np.sum(b[self._r_idx] * self._r_syn)) if len(self._r_idx) else 0.0
        left_push = float(np.sum(b[self._l_idx] * self._l_syn)) if len(self._l_idx) else 0.0
        return right_push, left_push

    def process_frame_with_brake(self, image_30x30_gray: np.ndarray) -> tuple[float, float, float]:
        """
        Returns ``(right_push, left_push, brake_drive)`` where ``brake_drive`` is in ``[0, 1]``.

        ``brake_drive`` from mapped DNg03 partners in fly_brake_weights.csv:
        ``1 - weighted_mean_brightness`` (darker = more brake).
        """
        b = self._grid_to_neuron_brightness(image_30x30_gray, self._grid_x, self._grid_y)
        right_push = float(np.sum(b[self._r_idx] * self._r_syn)) if len(self._r_idx) else 0.0
        left_push = float(np.sum(b[self._l_idx] * self._l_syn)) if len(self._l_idx) else 0.0
        if len(self._b_idx) and self._b_total_syn > 0.0:
            brake_brightness = float(np.sum(b[self._b_idx] * self._b_syn) / self._b_total_syn)
            brake_drive = float(np.clip(1.0 - brake_brightness, 0.0, 1.0))
        else:
            brake_drive = 0.0
        return right_push, left_push, brake_drive

    def process_frame_with_longitudinal(
        self, image_30x30_gray: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        Returns ``(right_push, left_push, brake_drive, throttle_drive)``.

        ``brake_drive`` is darkness-biased from DNg03 upstream input:
          ``1 - weighted_mean_brightness``.
        ``throttle_drive`` is brightness-biased from DNg01/02 upstream input:
          ``weighted_mean_brightness``.
        """
        right_push, left_push, brake_drive = self.process_frame_with_brake(image_30x30_gray)
        b = self._grid_to_neuron_brightness(image_30x30_gray, self._grid_x, self._grid_y)
        if len(self._t_idx) and self._t_total_syn > 0.0:
            throttle_drive = float(np.sum(b[self._t_idx] * self._t_syn) / self._t_total_syn)
            throttle_drive = float(np.clip(throttle_drive, 0.0, 1.0))
        else:
            throttle_drive = 0.0
        return right_push, left_push, brake_drive, throttle_drive

    def frame_diagnostics(self, image_30x30_gray: np.ndarray) -> dict[str, np.ndarray | float]:
        """Per-frame neuron brightness and synapse-weighted channel contributions."""
        b = self._grid_to_neuron_brightness(image_30x30_gray, self._grid_x, self._grid_y)
        right_contrib = b[self._r_idx] * self._r_syn if len(self._r_idx) else np.array([], dtype=np.float64)
        left_contrib = b[self._l_idx] * self._l_syn if len(self._l_idx) else np.array([], dtype=np.float64)
        brake_contrib = b[self._b_idx] * self._b_syn if len(self._b_idx) else np.array([], dtype=np.float64)
        throttle_contrib = b[self._t_idx] * self._t_syn if len(self._t_idx) else np.array([], dtype=np.float64)
        right_push = float(np.sum(right_contrib)) if len(right_contrib) else 0.0
        left_push = float(np.sum(left_contrib)) if len(left_contrib) else 0.0
        if len(self._b_idx) and self._b_total_syn > 0.0:
            brake_brightness = float(np.sum(brake_contrib) / self._b_total_syn)
            brake_drive = float(np.clip(1.0 - brake_brightness, 0.0, 1.0))
        else:
            brake_drive = 0.0
        if len(self._t_idx) and self._t_total_syn > 0.0:
            throttle_drive = float(np.clip(float(np.sum(throttle_contrib) / self._t_total_syn), 0.0, 1.0))
        else:
            throttle_drive = 0.0
        return {
            "neuron_brightness": b.astype(np.float32, copy=False),
            "right_contrib": right_contrib.astype(np.float32, copy=False),
            "left_contrib": left_contrib.astype(np.float32, copy=False),
            "brake_contrib": brake_contrib.astype(np.float32, copy=False),
            "throttle_contrib": throttle_contrib.astype(np.float32, copy=False),
            "right_push": right_push,
            "left_push": left_push,
            "brake_drive": brake_drive,
            "throttle_drive": throttle_drive,
        }

    def circuit_topology_arrays(self) -> dict[str, np.ndarray]:
        """Static indices for telemetry / circuit-lightup renderers."""
        return {
            "neuron_ids": self._neuron_ids.copy(),
            "grid_x": self._grid_x.copy(),
            "grid_y": self._grid_y.copy(),
            "right_idx": self._r_idx.copy(),
            "right_syn": self._r_syn.copy(),
            "left_idx": self._l_idx.copy(),
            "left_syn": self._l_syn.copy(),
            "brake_idx": self._b_idx.copy(),
            "brake_syn": self._b_syn.copy(),
            "throttle_idx": self._t_idx.copy(),
            "throttle_syn": self._t_syn.copy(),
        }

    @property
    def brake_input_count(self) -> int:
        return int(len(self._b_idx))

    @property
    def throttle_input_count(self) -> int:
        return int(len(self._t_idx))


__all__ = ["FlyBrain", "RIGHT_STEER_ID", "LEFT_STEER_ID"]
