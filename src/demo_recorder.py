"""Chase-camera offline render: lossless PNG sequence -> MP4 (Blender-style)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import carla
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from demo_composite import render_demo_triptych, side_panel_width
from vision_ingest import carla_image_to_bgr


def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return str(bundled)
    except Exception:
        pass
    return None


class ChaseVideoRecorder:
    """
    Synchronous demo capture: one lossless PNG per sim tick, H.264 MP4 encoded after the run.

    When ``composite=True``, each PNG is a triptych: retina | chase | neuronal histogram.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 10,
        preset: str = "veryslow",
        keep_frames: bool = False,
        composite: bool = True,
        center_title: str = "Driving video",
    ) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for --record-video (pip install opencv-python)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.center_width = int(width)
        self.center_height = int(height)
        self.composite = bool(composite)
        self._center_title = str(center_title)
        self._side_width = side_panel_width(self.center_height) if self.composite else 0
        if self.composite:
            self.width = self.center_width + 2 * self._side_width
            self.height = self.center_height
        else:
            self.width = self.center_width
            self.height = self.center_height
        self.fps = float(max(1.0, fps))
        self._crf = int(max(0, min(51, crf)))
        self._preset = str(preset or "slow")
        self._keep_frames = bool(keep_frames)
        self.frames_dir = self.path.parent / f".{self.path.stem}_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.frame_count = 0
        self.dropped_ticks = 0
        self._last_carla_frame: int | None = None
        self._carla_frame_gaps = 0
        self._active = False
        self._latest_chase: np.ndarray | None = None
        self._chase_seq = 0

    def begin_recording(self) -> None:
        """Start counting/saving frames (call after sync warmup ticks)."""
        with self._lock:
            for p in self.frames_dir.glob("frame_*.png"):
                try:
                    p.unlink()
                except OSError:
                    pass
            self.frame_count = 0
            self.dropped_ticks = 0
            self._last_carla_frame = None
            self._carla_frame_gaps = 0
            self._latest_chase = None
            self._chase_seq = 0
            self._active = True

    def _resize_chase(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[0], bgr.shape[1]
        if w != self.center_width or h != self.center_height:
            return cv2.resize(
                bgr,
                (self.center_width, self.center_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return bgr

    def on_image(self, image: carla.Image) -> None:
        if not self._active:
            return
        bgr = self._resize_chase(carla_image_to_bgr(image))
        fid = int(image.frame)
        with self._lock:
            if self._last_carla_frame is not None and fid > self._last_carla_frame + 1:
                self._carla_frame_gaps += int(fid - self._last_carla_frame - 1)
            self._last_carla_frame = fid
            if self.composite:
                self._latest_chase = bgr
                self._chase_seq += 1
                return
            self.frame_count += 1
            out = self.frames_dir / f"frame_{self.frame_count:06d}.png"
        cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])

    def await_chase(self, prev_seq: int, *, timeout: float = 2.0) -> bool:
        """Block until the chase sensor delivers a frame for the tick that just ran."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            with self._lock:
                if self._chase_seq > int(prev_seq):
                    return True
            time.sleep(0.001)
        with self._lock:
            self.dropped_ticks += 1
        return False

    def commit_composite(
        self,
        grid_u8: np.ndarray,
        neuron_brightness: np.ndarray,
    ) -> bool:
        """Compose retina | chase | histogram and write one PNG for this sim tick."""
        if not self._active:
            return False
        with self._lock:
            if self._latest_chase is None:
                self.dropped_ticks += 1
                return False
            chase = self._latest_chase.copy()
            self._latest_chase = None
            self.frame_count += 1
            out = self.frames_dir / f"frame_{self.frame_count:06d}.png"
        frame = render_demo_triptych(
            chase,
            grid_u8,
            neuron_brightness,
            side_width=self._side_width,
            center_title=self._center_title,
        )
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        return True

    def await_frame(self, prev_count: int, *, timeout: float = 2.0) -> bool:
        """Block until ``on_image`` saves a frame for the tick that just ran."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            with self._lock:
                if self.frame_count > prev_count:
                    return True
            time.sleep(0.001)
        with self._lock:
            self.dropped_ticks += 1
        return False

    @property
    def chase_seq(self) -> int:
        with self._lock:
            return int(self._chase_seq)

    def _encode_mp4(self) -> None:
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required to assemble demo MP4 from PNG frames (install ffmpeg, add to PATH)."
            )
        pattern = str(self.frames_dir / "frame_%06d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{self.fps:.6f}",
            "-start_number",
            "1",
            "-i",
            pattern,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self._preset,
            "-crf",
            str(self._crf),
            "-profile:v",
            "high",
            "-tune",
            "film",
            "-x264-params",
            "ref=6:bframes=8:me=umh:subme=10:aq-mode=3:rc-lookahead=60",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg assemble failed (exit {proc.returncode}): {err or 'unknown'}")

    def close(self) -> None:
        with self._lock:
            n = int(self.frame_count)
            gaps = int(self._carla_frame_gaps)
            drops = int(self.dropped_ticks)
            self._active = False
        if n == 0:
            print("Warning: no chase frames captured; skipping MP4 encode.", flush=True)
            return
        label = "composite triptych" if self.composite else "chase"
        print(
            f"Rendering {label} MP4 from {n} lossless PNG frames "
            f"({self.width}x{self.height}, ffmpeg {self._preset} crf={self._crf})…",
            flush=True,
        )
        self._encode_mp4()
        if gaps or drops:
            print(
                f"Warning: recording gaps — sim ticks without PNG={drops}, "
                f"CARLA frame_id gaps={gaps}",
                flush=True,
            )
        if not self._keep_frames:
            try:
                shutil.rmtree(self.frames_dir)
            except OSError:
                pass
        else:
            print(f"Kept PNG sequence: {self.frames_dir.resolve()}", flush=True)
        print(
            f"Saved demo video ({n} frames @ {self.fps:.0f} fps): {self.path.resolve()}",
            flush=True,
        )


def default_recording_path(label: str, *, output_dir: str | Path = "demos/recordings") -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{safe}_{ts}.mp4"


__all__ = ["ChaseVideoRecorder", "default_recording_path"]
