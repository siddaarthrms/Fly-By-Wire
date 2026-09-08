"""
Push -> steer in [-1, 1], plus 30x30 loom brake/throttle.

Keep steer/loom constants here stable; cameras and weather live in
carla_pilot.py / carla_fly_session.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# The "Fly-By-Wire" Bridge Logic (CARLA driver tune)
# ---------------------------------------------------------------------------
# Mild **right emphasis** via ``RIGHT_PUSH_SCALE_W = 1.01`` (vs symmetric w=1.0),
# with offset ``R0_RIGHT_MEAN`` so straight-road bias stays near zero:
#
#   bias = (w * right_push) - left_push - (w - 1) * R0
#
# When ``w > 1``, bias is zero for fixed straight-road means ``(r̄, l̄)`` if
#   R0 = (w * r̄ - l̄) / (w - 1).
# Legacy calibration clips used pushes ~3.27 (R0 ≈ 3.27386). CARLA live pilot
# uses much larger weighted sums (~14.4 / ~14.5); the default below matches a
# typical straight plateau so you do not sit at ~+0.067 steer forever.
# Re-tune with ``python carla_pilot.py --r0 ...`` from your own straight-run logs.
RIGHT_PUSH_SCALE_W = 1.01
R0_RIGHT_MEAN = 6.6414
GAIN = 2.0

# After ``bias * GAIN``, snap tiny residuals to 0 so slight left/right calibration
# noise does not move the wheel (CARLA still sees exact ±0.0001 as non-zero steer).
STEER_DEADZONE = 0.002

# When using **precomputed** bias from CSV (e.g. ``neural_projector`` columns that
# may use different offset formulas), ignore tiny straight residual (~1e-4)
# but keep below typical left-turn *mean* |bias| (~0.004).
DEADZONE = 0.003

# --- DNg03-style longitudinal control (center-grid looming / "dark obstacle") ---
# Central 10×10 on a 30×30 gray grid (same convention as ``bgr_to_fly_grid_30``).
# CARLA + INTER_AREA resize rarely yields many pixels <40 for a mid-gray wall — use a looser
# threshold and a mean-luminance “fill” term so brakes actually trip.
LOOM_CENTER_HALF = 5
LOOM_DARK_THRESHOLD_U8 = 68
# Second threshold for bright CARLA (asphalt + noon); counts “not bright sky” pixels.
LOOM_DARK_THRESHOLD_LOOSE_U8 = 108
LOOM_BLOCKED_FRACTION = 0.08
CRUISE_THROTTLE_DEFAULT = 0.3
LOOM_APPROACH_LOOSE_GAIN = 4.0

# Approach ROI mean vs “open road”: when a wall fills the lower windshield, mean gray drops
# even if few pixels are below LOOM_DARK_THRESHOLD_U8.
LOOM_APPROACH_MEAN_BRIGHT = 128.0
LOOM_APPROACH_MEAN_SPAN = 52.0

# Stateful “closure”: approach mean drops vs brightest mean seen this run (wall getting close).
LOOM_CLOSURE_PEAK_FRACTION = 0.13
LOOM_CLOSURE_DENOM_MIN = 20.0
# Closure can drift from tiny auto-exposure changes. Only arm closure when at least
# one direct visual cue is present (dark fractions / mean hazard / center dip), and
# ignore small closure values via a deadband.
LOOM_CLOSURE_ARM_EVIDENCE = 0.03
LOOM_CLOSURE_DEADBAND = 0.20

# Center column darker than left/right strips in approach ROI (central obstacle).
LOOM_CENTER_DIP_MIN = 6.0
LOOM_CENTER_DIP_DIV = 28.0

# Buffered looming: growth vs minimum in sliding window, one-frame jump, and window max (hysteresis).
LOOM_BUFFER_LEN = 8
LOOM_EXPANSION_DELTA = 0.03
LOOM_JUMP_DELTA = 0.02

# Steer vs looming: fly steering can aim the wall out of a tiny center ROI; damp / zero steer
# while hazard is high or brakes are applied so the car stops instead of dodging.
LOOM_BRAKE_STEER_MULT = 0.0
LOOM_HAZARD_STEER_DAMP_START = 0.05
LOOM_HAZARD_STEER_FULL_DAMP = 0.22
LOOM_HAZARD_STEER_MIN_MULT = 0.0

# --- DNg01/02-inspired global motion throttle adaptation ---
MOTION_BASE_THROTTLE = 0.5
MOTION_FLUX_SCALE = 0.1


def looming_fraction_center_dark(frame_30x30: np.ndarray) -> float:
    """Fraction of center-patch pixels below LOOM_DARK_THRESHOLD_U8 (0-1)."""
    g = np.asarray(frame_30x30)
    if g.ndim != 2:
        raise ValueError(f"looming_fraction_center_dark expects 2D grid, got shape {g.shape}")
    h, w = g.shape[0], g.shape[1]
    cy0, cy1 = h // 2 - LOOM_CENTER_HALF, h // 2 + LOOM_CENTER_HALF
    cx0, cx1 = w // 2 - LOOM_CENTER_HALF, w // 2 + LOOM_CENTER_HALF
    center_zone = g[cy0:cy1, cx0:cx1]
    if center_zone.size == 0:
        raise ValueError(
            f"Center slice empty for shape {g.shape}; need at least {(2 * LOOM_CENTER_HALF, 2 * LOOM_CENTER_HALF)}."
        )
    return float(np.mean(center_zone < LOOM_DARK_THRESHOLD_U8))


def _approach_zone_pixels(frame_30x30: np.ndarray) -> np.ndarray:
    """Lower-center windshield ROI (bottom 60 % height, central 2/3 width)."""
    g = np.asarray(frame_30x30)
    if g.ndim != 2:
        raise ValueError(f"_approach_zone_pixels expects 2D grid, got shape {g.shape}")
    h, w = g.shape[0], g.shape[1]
    r0 = max(0, (h * 2) // 5)
    r1 = h
    c0 = w // 6
    c1 = w - (w // 6)
    return g[r0:r1, c0:c1]


def looming_fraction_approach_loose(frame_30x30: np.ndarray) -> float:
    """Approach dark using LOOM_DARK_THRESHOLD_LOOSE_U8 (bright sun)."""
    zone = _approach_zone_pixels(frame_30x30)
    if zone.size == 0:
        return 0.0
    return float(np.mean(zone < LOOM_DARK_THRESHOLD_LOOSE_U8))


def looming_approach_center_vertical_dip(frame_30x30: np.ndarray) -> float:
    """
    Central obstacle: middle columns of the approach ROI darker than left/right (road at sides).
    """
    z = _approach_zone_pixels(frame_30x30).astype(np.float32)
    if z.size == 0:
        return 0.0
    h2, w2 = z.shape
    if h2 < 2 or w2 < 6:
        return 0.0
    tw = max(1, w2 // 3)
    left = z[:, :tw]
    right = z[:, -tw:]
    mid = z[:, tw : w2 - tw]
    if mid.size == 0:
        mid = z
    side_m = 0.5 * (float(np.mean(left)) + float(np.mean(right)))
    mid_m = float(np.mean(mid))
    dip = side_m - mid_m
    if dip <= LOOM_CENTER_DIP_MIN:
        return 0.0
    return float(np.clip(dip / LOOM_CENTER_DIP_DIV, 0.0, 1.0))


def looming_fraction_approach_dark(frame_30x30: np.ndarray) -> float:
    """
    Lower-center patch (dashcam: **bottom** rows = road ahead, obstacle loom).

    Survives aggressive steering better than a mid-frame square alone: the wall
    often stays in the lower FoV even when it leaves the geometric center.
    """
    zone = _approach_zone_pixels(frame_30x30)
    if zone.size == 0:
        return 0.0
    return float(np.mean(zone < LOOM_DARK_THRESHOLD_U8))


def looming_approach_mean_hazard(frame_30x30: np.ndarray) -> float:
    """
    Hazard from **overall dimming** of the approach ROI (wall / obstacle filling view).

    Open road in clear noon often sits around ``LOOM_APPROACH_MEAN_BRIGHT`` or above; a
    near fill drops the mean even when the surface is not “dark” per-pixel.
    """
    zone = _approach_zone_pixels(frame_30x30)
    if zone.size == 0:
        return 0.0
    m = float(np.mean(zone))
    if m >= LOOM_APPROACH_MEAN_BRIGHT:
        return 0.0
    span = max(1e-3, float(LOOM_APPROACH_MEAN_SPAN))
    return float(min(1.0, (LOOM_APPROACH_MEAN_BRIGHT - m) / span))


def looming_hazard_fraction(frame_30x30: np.ndarray) -> float:
    """Instant hazard in [0, 1]. Live runs should use LoomingController."""
    a_loose = looming_fraction_approach_loose(frame_30x30)
    return max(
        looming_fraction_center_dark(frame_30x30),
        looming_fraction_approach_dark(frame_30x30),
        float(np.clip(a_loose * LOOM_APPROACH_LOOSE_GAIN, 0.0, 1.0)),
        looming_approach_mean_hazard(frame_30x30),
        looming_approach_center_vertical_dip(frame_30x30),
    )


def get_adaptive_throttle(current_grid: np.ndarray, prev_grid: np.ndarray | None) -> float:
    """
    Global-motion adaptive throttle from frame-to-frame flux.

    ``motion_flux`` is the mean absolute pixel delta over the full 30x30 grid.
    Higher motion lowers throttle using:
      throttle = base / (1 + motion_flux * scale)

    First frame (no prev_grid) returns MOTION_BASE_THROTTLE.
    """
    if prev_grid is None:
        return float(MOTION_BASE_THROTTLE)

    cur = np.asarray(current_grid, dtype=np.float32)
    prv = np.asarray(prev_grid, dtype=np.float32)
    if cur.shape != prv.shape:
        raise ValueError(f"get_adaptive_throttle shape mismatch: {cur.shape} vs {prv.shape}")
    if cur.ndim != 2:
        raise ValueError(f"get_adaptive_throttle expects 2D grids, got {cur.shape}")

    motion_flux = float(np.mean(np.abs(cur - prv)))
    adaptation = 1.0 / (1.0 + motion_flux * float(MOTION_FLUX_SCALE))
    throttle = float(MOTION_BASE_THROTTLE * adaptation)
    return float(np.clip(throttle, 0.0, 1.0))


def blend_steer_with_loom(steer: float, brake: float, hazard_fraction: float) -> float:
    """Zero/damp steer while braking or when loom hazard is high."""
    st = float(steer)
    if float(brake) > 0.02:
        return float(np.clip(st * LOOM_BRAKE_STEER_MULT, -1.0, 1.0))
    h = float(hazard_fraction)
    if h <= LOOM_HAZARD_STEER_DAMP_START:
        return float(np.clip(st, -1.0, 1.0))
    span = max(1e-6, LOOM_HAZARD_STEER_FULL_DAMP - LOOM_HAZARD_STEER_DAMP_START)
    t = min(1.0, max(0.0, (h - LOOM_HAZARD_STEER_DAMP_START) / span))
    mult = 1.0 - t * (1.0 - LOOM_HAZARD_STEER_MIN_MULT)
    return float(np.clip(st * mult, -1.0, 1.0))


class LoomingController:
    """Buffered loom brake from the 30x30 grid (window max, expansion, jump)."""

    def __init__(
        self,
        *,
        buffer_len: int = LOOM_BUFFER_LEN,
        blocked_fraction: float = LOOM_BLOCKED_FRACTION,
        expansion_delta: float = LOOM_EXPANSION_DELTA,
        jump_delta: float = LOOM_JUMP_DELTA,
    ) -> None:
        self.buffer_len = int(buffer_len)
        self.blocked_fraction = float(blocked_fraction)
        self.expansion_delta = float(expansion_delta)
        self.jump_delta = float(jump_delta)
        self._buf: deque[float] = deque(maxlen=self.buffer_len)
        self._peak_approach_mean: float = -1.0
        self.last_signal: float = 0.0
        self.last_center: float = 0.0
        self.last_approach: float = 0.0
        self.last_approach_loose: float = 0.0
        self.last_mean_hazard: float = 0.0
        self.last_closure: float = 0.0
        self.last_center_dip: float = 0.0
        self.last_expansion: float = 0.0
        self.last_jump: float = 0.0
        self.last_window_max: float = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._peak_approach_mean = -1.0
        self.last_signal = 0.0
        self.last_center = 0.0
        self.last_approach = 0.0
        self.last_approach_loose = 0.0
        self.last_mean_hazard = 0.0
        self.last_closure = 0.0
        self.last_center_dip = 0.0
        self.last_expansion = 0.0
        self.last_jump = 0.0
        self.last_window_max = 0.0

    def step(
        self,
        frame_30x30: np.ndarray,
        *,
        cruise_throttle: float | None = None,
    ) -> tuple[float, float]:
        self.last_center = looming_fraction_center_dark(frame_30x30)
        self.last_approach = looming_fraction_approach_dark(frame_30x30)
        self.last_approach_loose = looming_fraction_approach_loose(frame_30x30)
        self.last_mean_hazard = looming_approach_mean_hazard(frame_30x30)
        self.last_center_dip = looming_approach_center_vertical_dip(frame_30x30)
        a_loose_scaled = float(np.clip(self.last_approach_loose * LOOM_APPROACH_LOOSE_GAIN, 0.0, 1.0))

        zone = _approach_zone_pixels(frame_30x30)
        app_mean = float(np.mean(zone)) if zone.size > 0 else 128.0
        if self._peak_approach_mean < 0.0:
            self._peak_approach_mean = app_mean
        else:
            self._peak_approach_mean = max(self._peak_approach_mean, app_mean)
        den = max(
            LOOM_CLOSURE_DENOM_MIN,
            float(self._peak_approach_mean) * LOOM_CLOSURE_PEAK_FRACTION,
        )
        closure_raw = float(np.clip((self._peak_approach_mean - app_mean) / den, 0.0, 1.0))
        evidence = max(
            self.last_center,
            self.last_approach,
            a_loose_scaled,
            self.last_mean_hazard,
            self.last_center_dip,
        )
        if evidence >= LOOM_CLOSURE_ARM_EVIDENCE:
            self.last_closure = float(
                np.clip(
                    (closure_raw - LOOM_CLOSURE_DEADBAND)
                    / max(1e-6, (1.0 - LOOM_CLOSURE_DEADBAND)),
                    0.0,
                    1.0,
                ),
            )
        else:
            self.last_closure = 0.0

        s = max(
            self.last_center,
            self.last_approach,
            a_loose_scaled,
            self.last_mean_hazard,
            self.last_closure,
            self.last_center_dip,
        )
        self.last_signal = s

        win_min_prior = min(self._buf) if self._buf else s
        prev = self._buf[-1] if self._buf else None
        self.last_expansion = float(s - win_min_prior)
        self.last_jump = float(s - prev) if prev is not None else 0.0

        self._buf.append(s)
        self.last_window_max = float(max(self._buf))

        cruise = float(CRUISE_THROTTLE_DEFAULT if cruise_throttle is None else cruise_throttle)
        cruise = float(np.clip(cruise, 0.0, 1.0))

        instant = s > self.blocked_fraction
        from_max_window = self.last_window_max > self.blocked_fraction
        from_expansion = len(self._buf) >= 2 and self.last_expansion >= self.expansion_delta
        from_jump = prev is not None and self.last_jump >= self.jump_delta

        if instant or from_max_window or from_expansion or from_jump:
            return 1.0, 0.0
        return 0.0, cruise


def get_longitudinal_control(
    frame_30x30: np.ndarray,
    *,
    cruise_throttle: float | None = None,
) -> tuple[float, float]:
    """
    Instant loom brake vs LOOM_BLOCKED_FRACTION.
    Live driving should use LoomingController.
    """
    looming_signal = looming_hazard_fraction(frame_30x30)
    cruise = float(CRUISE_THROTTLE_DEFAULT if cruise_throttle is None else cruise_throttle)
    cruise = float(np.clip(cruise, 0.0, 1.0))
    if looming_signal > LOOM_BLOCKED_FRACTION:
        return 1.0, 0.0
    return 0.0, cruise


def get_steer(right_push: float, left_push: float) -> float:
    """
    Raw pushes (same sums as neural_projector) -> steer in [-1, 1].
    bias = w*right - left - (w-1)*R0, then * GAIN; |steer| < STEER_DEADZONE -> 0.
    Positive is right, negative is left.
    """
    w = RIGHT_PUSH_SCALE_W
    r0 = R0_RIGHT_MEAN
    bias = (w * float(right_push)) - float(left_push) - ((w - 1.0) * r0)
    steer = float(bias * GAIN)
    dz = float(STEER_DEADZONE)
    if abs(steer) < dz:
        steer = 0.0
    return float(np.clip(steer, -1.0, 1.0))


def get_action(telemetry_bias: float) -> float:
    """DEADZONE then bias * GAIN, clip. For raw pushes use get_steer."""
    if abs(telemetry_bias) < DEADZONE:
        return 0.0
    steer_command = telemetry_bias * GAIN
    return float(np.clip(steer_command, -1.0, 1.0))


if __name__ == "__main__":
    w = RIGHT_PUSH_SCALE_W
    r0 = R0_RIGHT_MEAN
    print("Fly-By-Wire Bridge")
    print(f"  RIGHT_PUSH_SCALE_W = {w}  (1.01 = mild right bias vs symmetric 1.0)")
    print(f"  R0_RIGHT_MEAN      = {r0}")
    print(f"  GAIN               = {GAIN}  (tune for sim / vehicle)")
    print(f"  STEER_DEADZONE (get_steer) = {STEER_DEADZONE}  (|steer| below this -> 0)")
    print(f"  DEADZONE (get_action only) = {DEADZONE}")
    print()

    scenarios = [
        ("Straight CARLA plateau (zeros bias with default R0)", 14.4314, 14.5093),
        ("More right", 14.55, 14.40),
        ("More left", 14.30, 14.55),
    ]
    print("get_steer(right_push, left_push) -> steer in [-1, 1]")
    for label, rp, lp in scenarios:
        s = get_steer(rp, lp)
        bias = (w * rp) - lp - ((w - 1.0) * r0)
        print(f"  {label}")
        print(f"    bias={bias:+.6f}  ->  steer={s:+.4f}")
    print()

    print("get_action(telemetry_bias) -> steer (deadzone applied first)")
    for label, bias in [
        ("straight ~mean bias", -0.000098),
        ("right ~mean bias", 0.040719),
        ("left ~mean bias", -0.003849),
        ("sharp frame (~|bias| 0.09)", 0.09),
    ]:
        print(f"  {label}: bias={bias:+.6f} -> steer={get_action(bias):+.4f}")
    print()
    print("get_longitudinal_control (stateless instant) -> (brake, throttle)")
    bright = np.full((30, 30), 200, dtype=np.uint8)
    dark_center = bright.copy()
    dark_center[10:20, 10:20] = 20
    b0, t0 = get_longitudinal_control(bright)
    b1, t1 = get_longitudinal_control(dark_center)
    print(f"  mostly bright center: brake={b0} throttle={t0}")
    print(f"  dark 10x10 center:    brake={b1} throttle={t1}")
    print()
    print("LoomingController (buffered): bright road, center patch darkens then clears")
    lc = LoomingController()
    center_u8 = [220, 220, 220, 180, 120, 50, 25, 25, 220, 220, 220]
    for cv in center_u8:
        g = np.full((30, 30), 220, dtype=np.uint8)
        g[10:20, 10:20] = int(cv)
        br, th = lc.step(g, cruise_throttle=0.3)
        print(
            f"  center_u8={cv:3d} loom={lc.last_signal:.2f} exp={lc.last_expansion:+.2f} "
            f"jump={lc.last_jump:+.2f} win_max={lc.last_window_max:.2f} -> brake={br} thr={th}"
        )
