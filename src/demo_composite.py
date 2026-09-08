"""Three-panel demo frame: retina | chase video | neuronal histogram."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

PANEL_BG = (18, 18, 22)
TITLE_COLOR = (235, 235, 240)
HIST_BAR = (200, 130, 255)
HIST_AXIS = (120, 120, 130)
BORDER = (55, 55, 62)

DRIVING_PANEL_TITLES: dict[str, str] = {
    "straight": "Cruising",
    "left_curve": "Turning",
    "straight_cone": "Obstacle Avoidance",
    "loom_wall": "Rapid Braking",
}


def driving_panel_title(preset: str, *, pure_neural: bool = False) -> str:
    key = str(preset or "").strip().lower()
    if pure_neural and key == "loom_wall":
        return "Neural Only"
    return DRIVING_PANEL_TITLES.get(key, "Driving video")


def _panel_title(img: np.ndarray, title: str) -> None:
    cv2.putText(
        img,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        TITLE_COLOR,
        2,
        cv2.LINE_AA,
    )


def _render_retina_panel(grid_u8: np.ndarray, *, width: int, height: int) -> np.ndarray:
    grid = np.asarray(grid_u8, dtype=np.uint8).reshape(30, 30)
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    scale = min((width - 24) / 30.0, (height - 56) / 30.0)
    rw = max(30, int(round(30.0 * scale)))
    rh = max(30, int(round(30.0 * scale)))
    retina = cv2.resize(grid, (rw, rh), interpolation=cv2.INTER_NEAREST)
    retina_bgr = cv2.cvtColor(retina, cv2.COLOR_GRAY2BGR)
    x0 = (width - rw) // 2
    y0 = 40 + (height - 40 - rh) // 2
    panel[y0 : y0 + rh, x0 : x0 + rw] = retina_bgr
    _panel_title(panel, "Artificial Retina")
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + rw, y0 + rh), BORDER, 1)
    return panel


def _render_histogram_panel(neuron_brightness: np.ndarray, *, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    vals = np.asarray(neuron_brightness, dtype=np.float32).ravel()
    _panel_title(panel, "Neuronal histogram")

    top = 44
    bottom = height - 28
    left = 28
    right = width - 12
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    cv2.line(panel, (left, bottom), (right, bottom), HIST_AXIS, 1, cv2.LINE_AA)
    cv2.line(panel, (left, top), (left, bottom), HIST_AXIS, 1, cv2.LINE_AA)

    if vals.size == 0:
        return panel

    counts, _edges = np.histogram(vals, bins=40, range=(0.0, 1.0))
    peak = float(max(counts.max(), 1))
    n_bins = len(counts)
    gap = 1
    bar_w = max(1, (plot_w - gap * (n_bins - 1)) // n_bins)
    for i, count in enumerate(counts):
        bar_h = int(round((float(count) / peak) * plot_h))
        x0 = left + i * (bar_w + gap)
        y1 = bottom
        y0 = max(top, y1 - bar_h)
        cv2.rectangle(panel, (x0, y0), (min(right, x0 + bar_w - 1), y1), HIST_BAR, -1)

    cv2.putText(
        panel,
        "brightness",
        (left, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        HIST_AXIS,
        1,
        cv2.LINE_AA,
    )
    return panel


def side_panel_width(center_height: int) -> int:
    return max(360, int(center_height // 2))


def render_demo_triptych(
    chase_bgr: np.ndarray,
    grid_u8: np.ndarray,
    neuron_brightness: np.ndarray,
    *,
    side_width: int | None = None,
    center_title: str = "Driving video",
) -> np.ndarray:
    """
    Left: 30x30 grayscale retina (nearest-neighbor upscale).
    Mid:  chase / driving RGB frame.
    Right: mapped-neuron brightness histogram.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for composite demo recording")

    center = np.asarray(chase_bgr, dtype=np.uint8)
    if center.ndim != 3 or center.shape[2] != 3:
        raise ValueError(f"chase_bgr must be HxWx3 BGR, got {center.shape}")
    h, w = center.shape[:2]
    side_w = int(side_width if side_width is not None else side_panel_width(h))

    left = _render_retina_panel(grid_u8, width=side_w, height=h)
    right = _render_histogram_panel(neuron_brightness, width=side_w, height=h)

    mid = center.copy()
    _panel_title(mid, str(center_title))
    return np.hstack([left, mid, right])


__all__ = ["render_demo_triptych", "side_panel_width", "driving_panel_title"]
