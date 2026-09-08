#!/usr/bin/env python3
"""
CARLA (or any) RGB -> 30x30 uint8 grayscale.
Same BGR/gray + INTER_AREA path as offline neural_projector validation.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import carla
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

GRID = 30
TARGET_FPS = 30.0


def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    """CARLA sensor BGRA -> HxWx3 BGR uint8 (matches OpenCV conventions)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))
    return arr[:, :, :3].copy()


def bgr_to_fly_grid_30(bgr: np.ndarray) -> np.ndarray:
    """
    BGR -> gray -> resize 30x30 INTER_AREA.
    Returns uint8 shape (30, 30).

    Uses OpenCV when it accepts numpy arrays; some cv2/numpy builds fail cvtColor
    and fall back to BT.601 gray + PIL BOX downscale (similar to INTER_AREA).
    """
    bgr = np.asarray(bgr, dtype=np.uint8, order="C")
    if cv2 is not None:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return cv2.resize(gray, (GRID, GRID), interpolation=cv2.INTER_AREA)
        except cv2.error:
            pass
    if Image is None:
        raise RuntimeError("Need opencv-python or Pillow for bgr_to_fly_grid_30")
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r).clip(0, 255).astype(np.uint8)
    try:
        resample = Image.Resampling.BOX
    except AttributeError:
        resample = Image.BOX  # type: ignore[attr-defined]
    im = Image.fromarray(gray, mode="L").resize((GRID, GRID), resample)
    return np.asarray(im, dtype=np.uint8)


def fly_grid_to_model_brightness(grid_u8: np.ndarray) -> np.ndarray:
    """0..255 -> 0..1 float64 brightness."""
    return (grid_u8.astype(np.float64) / 255.0)


class RetinotopicIngest:
    """Latest 30x30 retina frame. Write from the sensor callback; read from the main loop."""

    def __init__(
        self,
        *,
        target_fps: float = TARGET_FPS,
        drop_duplicate_frame_id: bool = True,
    ) -> None:
        self._target_fps = float(target_fps)
        self._min_interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.0
        self._drop_dup = drop_duplicate_frame_id
        self._lock = threading.Lock()
        self._grid_u8: np.ndarray | None = None
        self._frame_id: int = -1
        self._sim_time: float = 0.0
        self._ingest_count: int = 0
        self._last_ingest_mono: float = 0.0
        self._last_carla_frame: int = -1
        self._callbacks: list[Callable[[np.ndarray, int, float], Any]] = []

    def set_frame_callback(self, fn: Callable[[np.ndarray, int, float], Any] | None) -> None:
        with self._lock:
            self._callbacks = [fn] if fn is not None else []

    def add_frame_callback(self, fn: Callable[[np.ndarray, int, float], Any]) -> None:
        with self._lock:
            self._callbacks.append(fn)

    def ingest_carla_image(self, image: carla.Image) -> bool:
        """Store a grid if this frame is new and not faster than target_fps. True if stored."""
        now = time.monotonic()
        if self._min_interval > 0 and (now - self._last_ingest_mono) < self._min_interval:
            return False
        fid = int(image.frame)
        if self._drop_dup and fid == self._last_carla_frame:
            return False
        self._last_carla_frame = fid

        bgr = carla_image_to_bgr(image)
        grid = bgr_to_fly_grid_30(bgr)
        sim_time = float(image.timestamp)

        with self._lock:
            self._grid_u8 = grid
            self._frame_id = fid
            self._sim_time = sim_time
            self._ingest_count += 1
            self._last_ingest_mono = now
            cbs = list(self._callbacks)

        for cb in cbs:
            try:
                cb(grid, fid, sim_time)
            except Exception:
                pass
        return True

    def ingest_bgr_u8(self, bgr: np.ndarray, *, frame_id: int = 0, sim_time: float = 0.0) -> None:
        """Test hook: raw BGR numpy frame (HxWx3)."""
        grid = bgr_to_fly_grid_30(bgr)
        with self._lock:
            self._grid_u8 = grid
            self._frame_id = int(frame_id)
            self._sim_time = float(sim_time)
            self._ingest_count += 1
            self._last_ingest_mono = time.monotonic()
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(grid, int(frame_id), float(sim_time))
            except Exception:
                pass

    def get_latest_grid_u8(self) -> np.ndarray | None:
        """Copy of last 30x30 uint8 grayscale, or None if never ingested."""
        with self._lock:
            if self._grid_u8 is None:
                return None
            return self._grid_u8.copy()

    def get_latest_brightness_01(self) -> np.ndarray | None:
        """Per-pixel input (30,30) float64 in [0,1]."""
        g = self.get_latest_grid_u8()
        if g is None:
            return None
        return fly_grid_to_model_brightness(g)

    def get_meta(self) -> tuple[int, float, int]:
        """(frame_id, sim_timestamp, total_ingest_count) under lock snapshot."""
        with self._lock:
            return self._frame_id, self._sim_time, self._ingest_count

    @property
    def flat_index_row_major(self) -> str:
        """grid[y, x]; np.ravel(order='C') is a 900-vector."""
        return "grid[y,x] uint8; ravel('C') gives row-major flat vector length 900"


__all__ = [
    "GRID",
    "TARGET_FPS",
    "RetinotopicIngest",
    "bgr_to_fly_grid_30",
    "carla_image_to_bgr",
    "fly_grid_to_model_brightness",
]
