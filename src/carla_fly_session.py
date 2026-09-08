#!/usr/bin/env python3
"""
CARLA session: pygame for keys only, OpenDRIVE worlds by default.

  python carla_fly_session.py
  python carla_fly_session.py --launch
  python carla_fly_session.py --opendrive-preset left_curve|loom_wall
  python carla_fly_session.py --map Town10HD_Opt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import carla
import numpy as np

from vision_ingest import RetinotopicIngest


def _pygame_surface_from_carla_image(image: carla.Image):
    """BGRA sensor buffer -> pygame RGB surface."""
    import pygame

    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]
    return pygame.image.frombuffer(rgb.tobytes(), (image.width, image.height), "RGB").convert()

# Packaged-town fallback order when you pass ``--map`` without a full name list here.
DEFAULT_MAP_PREFERENCE = ["Town04_Opt", "Town04", "Town03_Opt", "Town03", "Town01_Opt", "Town01"]

# Minimal OpenDRIVE: single driving lane, straight reference along +X (CARLA import).
STRAIGHT_ROAD_LENGTH_M = 400.0
STRAIGHT_CONE_ROAD_LENGTH_M = 50.0

# Short straight + wall (``--opendrive-preset loom_wall``): quick loom / emergency-brake harness.
LOOM_TEST_ROAD_LENGTH_M = 50.0
# Wall placement along first road: ~0 = start, ~1 = end (by waypoint index along geometry).
LOOM_WALL_ROAD_FRACTION = 0.36
# Full 3.5 m lane coverage (meters along waypoint right: - = left, + = right of centerline).
LOOM_WALL_FULL_LANE_OFFSETS_M = (
    -1.68,
    -1.4,
    -1.12,
    -0.84,
    -0.56,
    -0.28,
    0.0,
    0.28,
    0.56,
    0.84,
    1.12,
    1.4,
    1.68,
)
# Second row along +forward (m) to close longitudinal gaps between props.
LOOM_WALL_ROW_STAGGER_M = (0.0, 1.05)

# Single arc segment: left bend (positive curvature = CCW / "left" from +X).
# R≈78 m keeps the connectome in-lane but reads clearly on video (R=120 m looked straight).
CURVED_ROAD_LENGTH_M = 220.0
CURVED_ARC_RADIUS_M = 78.0
# κ = 1/R; length L gives sweep L/R rad.
_CURV_KAPPA = 1.0 / CURVED_ARC_RADIUS_M
# Loose geographic bounds for <header> (CARLA only needs a rough box).
_CURVED_EAST = max(120.0, float(CURVED_ROAD_LENGTH_M) * 0.35)
_CURVED_NORTH = max(350.0, float(CURVED_ROAD_LENGTH_M) * 0.95)

MINIMAL_STRAIGHT_XODR = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="fbw_straight" version="1"
          north="0.0" south="0.0" east="{STRAIGHT_ROAD_LENGTH_M}" west="0.0" vendor="Fly-By-Wire"/>
  <road name="straight" length="{STRAIGHT_ROAD_LENGTH_M}" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="{STRAIGHT_ROAD_LENGTH_M}">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""

MINIMAL_STRAIGHT_CONE_XODR = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="fbw_straight_cone" version="1"
          north="0.0" south="0.0" east="{STRAIGHT_CONE_ROAD_LENGTH_M}" west="0.0" vendor="Fly-By-Wire"/>
  <road name="straight_cone" length="{STRAIGHT_CONE_ROAD_LENGTH_M}" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="{STRAIGHT_CONE_ROAD_LENGTH_M}">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""

MINIMAL_CURVED_XODR = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="fbw_curved" version="1"
          north="{_CURVED_NORTH}" south="-40.0" east="{_CURVED_EAST}" west="-40.0" vendor="Fly-By-Wire"/>
  <road name="curve" length="{CURVED_ROAD_LENGTH_M}" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="{CURVED_ROAD_LENGTH_M}">
        <arc curvature="{_CURV_KAPPA:.12f}"/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""

MINIMAL_LOOM_STRAIGHT_XODR = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="fbw_loom_straight" version="1"
          north="0.0" south="0.0" east="{LOOM_TEST_ROAD_LENGTH_M}" west="0.0" vendor="Fly-By-Wire"/>
  <road name="loom_straight" length="{LOOM_TEST_ROAD_LENGTH_M}" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="{LOOM_TEST_ROAD_LENGTH_M}">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""

# Longest built-in OpenDRIVE segment (for mesh tessellation max segment length).
BUILTIN_OPENDRIVE_MAX_LENGTH_M = max(
    STRAIGHT_ROAD_LENGTH_M,
    CURVED_ROAD_LENGTH_M,
    LOOM_TEST_ROAD_LENGTH_M,
)

# Good default for perception / driving stacks (exists in most 0.9.x builds).
VEHICLE_PREFERENCE = [
    "vehicle.mercedes.coupe_2020",
    "vehicle.tesla.model3",
    "vehicle.audi.tt",
    "vehicle.audi.etron",
    "vehicle.lincoln.mkz2017",
    "vehicle.nissan.patrol",
    "vehicle.dodge_charger.police",
]

# Vivid paint choices for demo recordings (CARLA "R,G,B" strings).
DEMO_VEHICLE_COLORS = [
    "255,28,28",
    "255,140,0",
    "255,215,0",
    "0,180,255",
    "180,0,255",
    "0,220,130",
]

RECORD_CAMERA_PRESETS: dict[str, dict[str, float]] = {
    # Rear three-quarter trailing rig (replaces the old high bird's-eye chase).
    "chase": {"x": -8.5, "y": -2.0, "z": 1.9, "pitch": -7.0, "yaw": 12.0, "fov": 92.0},
    "chase_low": {"x": -6.0, "y": 0.0, "z": 2.2, "pitch": -8.0, "yaw": 0.0, "fov": 92.0},
    "chase_side": {"x": -5.0, "y": 3.5, "z": 1.8, "pitch": -5.0, "yaw": -22.0, "fov": 92.0},
    "chase_oncoming": {
        "x": 16.0,
        "y": 0.0,
        "z": 1.35,
        "pitch": -6.0,
        "yaw": 180.0,
        "fov": 92.0,
    },
    "low_front": {"x": 2.8, "y": 0.0, "z": 0.95, "pitch": 4.0, "yaw": 0.0, "fov": 92.0},
    # straight demo jump-cut beats (StraightDriveCameraDirector)
    "straight_wide": {"x": -9.0, "y": 0.0, "z": 3.4, "pitch": -11.0, "yaw": 0.0, "fov": 90.0},
    "straight_profile": {"x": -5.0, "y": 4.5, "z": 1.5, "pitch": -4.0, "yaw": -24.0, "fov": 88.0},
    "straight_lead": {"x": 15.0, "y": 0.0, "z": 1.55, "pitch": -5.0, "yaw": 180.0, "fov": 86.0},
    # Low passenger-side 3/4 — front wheel + steer away from a left-edge cone.
    "chase_dodge": {
        "x": -1.4,
        "y": 3.85,
        "z": 0.5,
        "pitch": 2.0,
        "yaw": -28.0,
        "fov": 93.0,
    },
    # straight_cone scripted beats (ConeDodgeCameraDirector)
    "cone_high": {"x": -8.5, "y": 0.0, "z": 4.8, "pitch": -15.0, "yaw": 0.0, "fov": 88.0},
    "cone_wheel": {"x": -0.15, "y": 3.4, "z": 0.2, "pitch": -2.0, "yaw": -40.0, "fov": 98.0},
    # cone_bumper: computed each tick in cone_dodge_director (road-axis look-at)
    "cone_bumper": {"x": 0.0, "y": 0.0, "z": 1.1, "pitch": 0.0, "yaw": 180.0, "fov": 88.0},
    "cone_bird": {"x": -10.5, "y": 0.0, "z": 7.6, "pitch": -50.0, "yaw": 0.0, "fov": 84.0},
}

# Shared look for all demo recordings (launchers pass --weather %DEMO_WEATHER%).
DEMO_WEATHER = "clear"


def configure_rgb_camera_bp(
    bp: carla.ActorBlueprint,
    *,
    disable_motion_blur: bool = True,
    disable_postprocess: bool = False,
    disable_temporal_aa: bool = False,
    fast_shutter: bool = True,
) -> None:
    """Tune RGB sensor blueprint for stable demo footage.

    Keep ``disable_postprocess=False`` — turning postprocess off breaks exposure /
    tonemapping and yields black geometry with a blown-out white sky in MP4s.
    """
    if disable_motion_blur:
        for attr, val in (
            ("motion_blur_intensity", "0"),
            ("motion_blur_max_distortion", "0"),
            ("motion_blur_min_object_screen_size", "1.0"),
            ("blur_amount", "0"),
            ("blur_radius", "0"),
        ):
            if bp.has_attribute(attr):
                try:
                    bp.set_attribute(attr, val)
                except Exception:
                    pass
    if disable_temporal_aa:
        for attr, val in (
            ("temporal_aa", "False"),
            ("temporal_antialiasing", "False"),
            ("antialiasing", "FXAA"),
        ):
            if bp.has_attribute(attr):
                try:
                    bp.set_attribute(attr, val)
                except Exception:
                    pass
    if disable_postprocess and bp.has_attribute("enable_postprocess_effects"):
        try:
            bp.set_attribute("enable_postprocess_effects", "False")
        except Exception:
            pass
    if fast_shutter and bp.has_attribute("shutter_speed"):
        try:
            bp.set_attribute("shutter_speed", "2000")
        except Exception:
            pass


def weather_for_preset(name: str) -> carla.WeatherParameters:
    key = str(name or "clear").strip().lower()
    if key in {"clear", "noon", "clearnoon"}:
        return carla.WeatherParameters.ClearNoon
    if key == "sunset":
        return carla.WeatherParameters(
            cloudiness=35.0,
            sun_altitude_angle=18.0,
            sun_azimuth_angle=290.0,
            fog_density=0.0,
            wetness=0.0,
        )
    if key in {"cloudy", "overcast"}:
        return carla.WeatherParameters(
            cloudiness=85.0,
            sun_altitude_angle=42.0,
            sun_azimuth_angle=180.0,
            fog_density=0.0,
            wetness=10.0,
            precipitation_deposits=10.0,
        )
    if key in {"wet", "damp"}:
        return carla.WeatherParameters(
            cloudiness=70.0,
            sun_altitude_angle=35.0,
            sun_azimuth_angle=200.0,
            fog_density=0.0,
            wetness=45.0,
            precipitation_deposits=40.0,
        )
    raise ValueError(f"Unknown weather preset: {name!r} (try clear, sunset, cloudy, wet)")


def record_camera_spec(name: str) -> tuple[carla.Transform, float]:
    key = str(name or "chase_low").strip().lower()
    if key not in RECORD_CAMERA_PRESETS:
        choices = ", ".join(sorted(RECORD_CAMERA_PRESETS))
        raise ValueError(f"Unknown record camera preset: {name!r} (try {choices})")
    spec = RECORD_CAMERA_PRESETS[key]
    tf = carla.Transform(
        carla.Location(x=float(spec["x"]), y=float(spec["y"]), z=float(spec["z"])),
        carla.Rotation(pitch=float(spec["pitch"]), yaw=float(spec["yaw"])),
    )
    return tf, float(spec["fov"])


def apply_demo_vehicle_style(
    bp: carla.ActorBlueprint,
    *,
    explicit_color: str | None = None,
    rng: random.Random | None = None,
) -> None:
    if not bp.has_attribute("color"):
        return
    attr = bp.get_attribute("color")
    recommended = list(attr.recommended_values or [])
    if explicit_color:
        if explicit_color in recommended:
            bp.set_attribute("color", explicit_color)
        else:
            print(f"Vehicle color {explicit_color!r} not in blueprint list; picking demo color.", flush=True)
            pick = (rng or random).choice(DEMO_VEHICLE_COLORS)
            if pick in recommended:
                bp.set_attribute("color", pick)
            elif recommended:
                bp.set_attribute("color", recommended[0])
        return
    vivid = [c for c in DEMO_VEHICLE_COLORS if c in recommended]
    pool = vivid or recommended
    if pool:
        bp.set_attribute("color", (rng or random).choice(pool))


def spectator_follow_transform(
    vehicle_transform: carla.Transform, camera_preset: str
) -> carla.Transform:
    """Match live spectator to the record-camera offset for demo runs."""
    spec = RECORD_CAMERA_PRESETS.get(str(camera_preset).strip().lower(), RECORD_CAMERA_PRESETS["chase_low"])
    fwd = vehicle_transform.get_forward_vector()
    right = vehicle_transform.get_right_vector()
    loc = vehicle_transform.location
    loc = loc + fwd * float(spec["x"]) + right * float(spec["y"]) + carla.Location(z=float(spec["z"]))
    return carla.Transform(loc, carla.Rotation(pitch=float(spec["pitch"]), yaw=vehicle_transform.rotation.yaw + float(spec["yaw"])))


CHASE_CYCLE_PRESETS: tuple[str, ...] = ("chase_low", "chase", "chase_side", "chase_oncoming")

# Cone dodge: hold low approach, wheel-cut during pass, side exit, rear wrap.
CONE_DODGE_CYCLE_PRESETS: tuple[str, ...] = (
    "chase_low",
    "chase_dodge",
    "chase_side",
    "chase",
)


def _lerp_float(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def lerp_camera_preset(name_a: str, name_b: str, blend: float) -> tuple[carla.Transform, float]:
    """Blend two vehicle-relative chase rigs (``blend`` in [0, 1])."""
    sa = RECORD_CAMERA_PRESETS[str(name_a).strip().lower()]
    sb = RECORD_CAMERA_PRESETS[str(name_b).strip().lower()]
    u = _smoothstep01(blend)
    tf = carla.Transform(
        carla.Location(
            x=_lerp_float(sa["x"], sb["x"], u),
            y=_lerp_float(sa["y"], sb["y"], u),
            z=_lerp_float(sa["z"], sb["z"], u),
        ),
        carla.Rotation(
            pitch=_lerp_float(sa["pitch"], sb["pitch"], u),
            yaw=_lerp_angle(float(sa["yaw"]), float(sb["yaw"]), u),
        ),
    )
    fov = _lerp_float(sa["fov"], sb["fov"], u)
    return tf, fov


def _attach_to_world_transform(vehicle_transform: carla.Transform, attach: carla.Transform) -> carla.Transform:
    fwd = vehicle_transform.get_forward_vector()
    right = vehicle_transform.get_right_vector()
    loc = vehicle_transform.location
    loc = (
        loc
        + fwd * float(attach.location.x)
        + right * float(attach.location.y)
        + carla.Location(z=float(attach.location.z))
    )
    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=float(attach.rotation.pitch),
            yaw=float(vehicle_transform.rotation.yaw) + float(attach.rotation.yaw),
        ),
    )


def yaw_stabilized_transform(vehicle_transform: carla.Transform) -> carla.Transform:
    """Flatten vehicle pitch/roll so the chase rig ignores suspension wobble."""
    r = vehicle_transform.rotation
    return carla.Transform(
        vehicle_transform.location,
        carla.Rotation(pitch=0.0, yaw=float(r.yaw), roll=0.0),
    )


def chase_world_transform(
    vehicle_transform: carla.Transform,
    attach: carla.Transform,
    *,
    stabilize_yaw: bool = True,
) -> carla.Transform:
    """Vehicle-relative chase rig in world space (optional yaw-only stabilization)."""
    base = yaw_stabilized_transform(vehicle_transform) if stabilize_yaw else vehicle_transform
    return _attach_to_world_transform(base, attach)


class SpectatorSmoother:
    """Light EMA on live viewport only — does not affect the record sensor."""

    def __init__(self, alpha: float = 0.45) -> None:
        self._alpha = float(np.clip(alpha, 0.05, 1.0))
        self._tf: carla.Transform | None = None

    def smooth(self, target: carla.Transform) -> carla.Transform:
        if self._tf is None:
            self._tf = target
            return target
        a = self._alpha
        cur = self._tf
        tgt = target
        loc = carla.Location(
            x=cur.location.x + a * (tgt.location.x - cur.location.x),
            y=cur.location.y + a * (tgt.location.y - cur.location.y),
            z=cur.location.z + a * (tgt.location.z - cur.location.z),
        )
        pitch = cur.rotation.pitch + a * (tgt.rotation.pitch - cur.rotation.pitch)
        yaw = _lerp_angle(cur.rotation.yaw, tgt.rotation.yaw, a)
        self._tf = carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw))
        return self._tf


def _lerp_angle(a: float, b: float, t: float) -> float:
    delta = ((float(b) - float(a) + 180.0) % 360.0) - 180.0
    return float(a) + delta * float(t)


class DynamicChaseCamera:
    """
    Demo chase rig: hold each angle, smooth-blend vehicle-relative offsets.

    Offsets are applied as **relative** transforms on a Rigid-attached record camera.
    """

    HOLD_SEC = 7.5
    BLEND_SEC = 3.0
    SMOOTH_TAU_SEC = 0.6

    def __init__(
        self,
        presets: tuple[str, ...] = CHASE_CYCLE_PRESETS,
        *,
        start: str | None = None,
        hold_sec: float | None = None,
        blend_sec: float | None = None,
        smooth_tau_sec: float | None = None,
    ) -> None:
        names = [p for p in presets if p in RECORD_CAMERA_PRESETS]
        if not names:
            names = list(CHASE_CYCLE_PRESETS)
        start_key = str(start or names[0]).strip().lower()
        if start_key == "low_front":
            names = [p for p in ("low_front", "chase_low", "chase_side") if p in RECORD_CAMERA_PRESETS]
        elif start_key in names:
            idx = names.index(start_key)
            names = names[idx:] + names[:idx]
        self._names = names
        self._hold_sec = float(hold_sec if hold_sec is not None else self.HOLD_SEC)
        self._blend_sec = float(blend_sec if blend_sec is not None else self.BLEND_SEC)
        self._smooth_tau = float(smooth_tau_sec if smooth_tau_sec is not None else self.SMOOTH_TAU_SEC)
        self._attach_tf: carla.Transform | None = None
        self._attach_fov: float = 100.0
        self._last_t_sec: float | None = None

    def _preset_at(self, t_sec: float) -> tuple[carla.Transform, float]:
        seg = float(self._hold_sec + self._blend_sec)
        n = len(self._names)
        u = float(t_sec) % (seg * n)
        idx = int(u // seg) % n
        phase = u % seg
        if phase < self._hold_sec:
            return record_camera_spec(self._names[idx])
        blend = (phase - self._hold_sec) / max(1e-6, self._blend_sec)
        return lerp_camera_preset(self._names[idx], self._names[(idx + 1) % n], blend)

    def advance_attach(self, t_sec: float) -> tuple[carla.Transform, float]:
        """Update smoothed vehicle-relative rig from the demo timeline."""
        dt = 0.033333
        if self._last_t_sec is not None:
            dt = max(1e-4, float(t_sec) - float(self._last_t_sec))
        self._last_t_sec = float(t_sec)

        raw_tf, raw_fov = self._preset_at(t_sec)
        if self._attach_tf is None:
            self._attach_tf = raw_tf
            self._attach_fov = float(raw_fov)
        else:
            a = 1.0 - math.exp(-dt / max(1e-6, self._smooth_tau))
            cur = self._attach_tf
            self._attach_tf = carla.Transform(
                carla.Location(
                    x=cur.location.x + a * (raw_tf.location.x - cur.location.x),
                    y=cur.location.y + a * (raw_tf.location.y - cur.location.y),
                    z=cur.location.z + a * (raw_tf.location.z - cur.location.z),
                ),
                carla.Rotation(
                    pitch=cur.rotation.pitch + a * (raw_tf.rotation.pitch - cur.rotation.pitch),
                    yaw=_lerp_angle(cur.rotation.yaw, raw_tf.rotation.yaw, a),
                ),
            )
            self._attach_fov = self._attach_fov + a * (float(raw_fov) - self._attach_fov)
        return self._attach_tf, float(self._attach_fov)


class TransformShakeProbe:
    """
    Per-tick camera / vehicle motion probe for diagnosing viewport shake.

    Key metric: ``stale_lag_m`` — distance between where the chase camera *should*
    be (vehicle pose × attach offset) vs its actual world pose. Large values mean
    the record camera is not riding the vehicle (world-spawned + infrequent
    ``set_transform`` is the usual cause).
    """

    SHAKE_LOG_HEADER = (
        "tick,vehicle_dx,vehicle_dy,vehicle_dz,vehicle_dyaw,"
        "chase_dx,chase_dy,chase_dz,chase_dyaw,"
        "stale_lag_m,chase_updated,attach_mode"
    )

    def __init__(self) -> None:
        self._prev_vehicle: carla.Transform | None = None
        self._prev_chase: carla.Transform | None = None
        self._rows: list[str] = []

    @staticmethod
    def _delta(prev: carla.Transform, cur: carla.Transform) -> tuple[float, float, float, float]:
        dx = float(cur.location.x - prev.location.x)
        dy = float(cur.location.y - prev.location.y)
        dz = float(cur.location.z - prev.location.z)
        dyaw = float(((cur.rotation.yaw - prev.rotation.yaw + 180.0) % 360.0) - 180.0)
        return dx, dy, dz, dyaw

    @staticmethod
    def _dist(a: carla.Location, b: carla.Location) -> float:
        dx = float(a.x - b.x)
        dy = float(a.y - b.y)
        dz = float(a.z - b.z)
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)

    def sample(
        self,
        *,
        tick: int,
        vehicle_tf: carla.Transform,
        chase_tf: carla.Transform | None,
        attach_tf: carla.Transform | None,
        chase_updated: bool,
        attach_mode: str,
    ) -> dict[str, float | bool | int | str]:
        vdx = vdy = vdz = vdyaw = 0.0
        if self._prev_vehicle is not None:
            vdx, vdy, vdz, vdyaw = self._delta(self._prev_vehicle, vehicle_tf)
        self._prev_vehicle = vehicle_tf

        cdx = cdy = cdz = cdyaw = 0.0
        stale_lag_m = 0.0
        if chase_tf is not None:
            if self._prev_chase is not None:
                cdx, cdy, cdz, cdyaw = self._delta(self._prev_chase, chase_tf)
            self._prev_chase = chase_tf
            if attach_tf is not None:
                expected = _attach_to_world_transform(vehicle_tf, attach_tf)
                stale_lag_m = self._dist(expected.location, chase_tf.location)

        row = (
            f"{int(tick)},{vdx:.6f},{vdy:.6f},{vdz:.6f},{vdyaw:.6f},"
            f"{cdx:.6f},{cdy:.6f},{cdz:.6f},{cdyaw:.6f},"
            f"{stale_lag_m:.6f},{1 if chase_updated else 0},{attach_mode}"
        )
        self._rows.append(row)
        return {
            "tick": int(tick),
            "vehicle_dx": vdx,
            "vehicle_dy": vdy,
            "vehicle_dz": vdz,
            "vehicle_dyaw": vdyaw,
            "chase_dx": cdx,
            "chase_dy": cdy,
            "chase_dz": cdz,
            "chase_dyaw": cdyaw,
            "stale_lag_m": stale_lag_m,
            "chase_updated": bool(chase_updated),
            "attach_mode": attach_mode,
        }

    def write_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.SHAKE_LOG_HEADER + "\n" + "\n".join(self._rows) + "\n", encoding="utf-8")

    @staticmethod
    def summarize_csv(path: Path) -> dict[str, float]:
        """Offline RMS / peak summary for optional --shake-log CSVs."""
        import csv

        path = Path(path)
        rows: list[dict[str, str]] = []
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return {}

        def rms(key: str) -> float:
            vals = [float(r[key]) for r in rows]
            return float((sum(v * v for v in vals) / max(1, len(vals))) ** 0.5)

        def rms_xy(dx_key: str, dy_key: str) -> float:
            vals = [
                float(r[dx_key]) ** 2 + float(r[dy_key]) ** 2
                for r in rows
            ]
            return float((sum(vals) / max(1, len(vals))) ** 0.5)

        stale = [float(r["stale_lag_m"]) for r in rows]
        chase_jump = [
            float(r["chase_dx"]) ** 2 + float(r["chase_dy"]) ** 2 + float(r["chase_dz"]) ** 2
            for r in rows
            if int(r.get("chase_updated", "0")) == 1
        ]
        return {
            "n_ticks": float(len(rows)),
            "vehicle_rms_xy_m": rms_xy("vehicle_dx", "vehicle_dy"),
            "chase_rms_xy_m": rms_xy("chase_dx", "chase_dy"),
            "stale_lag_mean_m": float(sum(stale) / len(stale)),
            "stale_lag_max_m": float(max(stale)),
            "chase_jump_rms_m": float((sum(v for v in chase_jump) / max(1, len(chase_jump))) ** 0.5)
            if chase_jump
            else 0.0,
        }


def repo_root() -> Path:
    # src/ is a package directory; project root is one level up.
    return Path(__file__).resolve().parent.parent


def carla_root_config_file() -> Path:
    return repo_root() / "demos" / "carla_root.txt"


def _carla_install_candidates(explicit: Path | str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("CARLA_ROOT", "").strip()
    if env:
        candidates.append(Path(env))
    cfg = carla_root_config_file()
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            candidates.append(Path(line))
            break
    root = repo_root()
    candidates.extend(
        [
            root / "CARLA_0.9.16",
            root.parent / "Fly-By-Wire" / "CARLA_0.9.16",
            root.parent / "CARLA_0.9.16",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for raw in candidates:
        try:
            p = raw.expanduser().resolve()
        except OSError:
            p = raw.expanduser()
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def resolve_sim_dir(explicit: Path | str | None = None) -> Path:
    """Return the first candidate install dir that contains ``CarlaUE4.exe``."""
    tried: list[Path] = []
    for p in _carla_install_candidates(explicit):
        tried.append(p)
        if (p / "CarlaUE4.exe").is_file():
            return p
    lines = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        "CARLA 0.9.16 not found. Checked:\n"
        f"{lines}\n\n"
        "Fix one of:\n"
        "  1) set CARLA_ROOT=C:\\path\\to\\CARLA_0.9.16\n"
        "  2) copy demos/carla_root.example.txt -> demos/carla_root.txt and edit the path\n"
        "  3) run demos/Set_CARLA_Root.bat\n"
        "See docs/CARLA.md for install steps."
    )


def default_sim_dir() -> Path:
    """Best-effort CARLA path for help text / non-launch flows."""
    try:
        return resolve_sim_dir()
    except FileNotFoundError:
        env = os.environ.get("CARLA_ROOT", "").strip()
        if env:
            return Path(env)
        return repo_root() / "CARLA_0.9.16"


def advance_world_ticks(world: carla.World, n: int = 1) -> None:
    """Step the simulation ``n`` times (sync ``tick()`` or async ``wait_for_tick()``)."""
    count = max(0, int(n))
    if count == 0:
        return
    sync = bool(world.get_settings().synchronous_mode)
    for _ in range(count):
        if sync:
            world.tick()
        else:
            world.wait_for_tick()


def restore_world_async(world: carla.World) -> None:
    """Return world to default async stepping (call on shutdown)."""
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.0
    world.apply_settings(settings)


def configure_world_sync(world: carla.World, *, fps: float = 30.0) -> float:
    """
    Enable synchronous mode for deterministic sensor + control loops.

    Returns ``fixed_delta_seconds`` applied to the world.
    """
    dt = 1.0 / float(fps) if fps > 0 else 0.033333
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(dt)
    world.apply_settings(settings)
    return float(dt)


def server_is_up(host: str, port: int, *, timeout: float = 2.0) -> bool:
    """True when the CARLA RPC endpoint answers."""
    try:
        c = carla.Client(host, port)
        c.set_timeout(float(timeout))
        _ = c.get_server_version()
        return True
    except Exception:
        return False


def probe_server_world(host: str, port: int, *, timeout: float = 5.0) -> bool:
    try:
        c = carla.Client(host, port)
        c.set_timeout(float(timeout))
        _ = c.get_server_version()
        _ = c.get_world().get_map().name
        return True
    except Exception:
        return False


def carla_process_running() -> bool:
    if sys.platform == "win32":
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq CarlaUE4.exe"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "CarlaUE4.exe" in (r.stdout or "")
    r = subprocess.run(["pgrep", "-x", "CarlaUE4"], capture_output=True, text=True, check=False)
    return bool((r.stdout or "").strip())


def terminate_carla_processes() -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/IM", "CarlaUE4.exe", "/T"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        subprocess.run(["pkill", "-x", "CarlaUE4"], capture_output=True, text=True, check=False)


def wait_for_server(
    host: str,
    port: int,
    timeout: float = 180.0,
    *,
    initial_grace: float = 0.0,
) -> bool:
    if initial_grace > 0.0:
        print(f"Waiting {initial_grace:.0f}s for CARLA to start…", flush=True)
        time.sleep(float(initial_grace))
    t0 = time.time()
    last_log = 0.0
    while time.time() - t0 < timeout:
        try:
            c = carla.Client(host, port)
            c.set_timeout(2.0)
            _ = c.get_server_version()
            return True
        except Exception:
            now = time.time()
            if now - last_log >= 5.0:
                print(f"  …waiting for CARLA RPC ({now - t0:.0f}s)", flush=True)
                last_log = now
            time.sleep(0.25)
    return False


def wait_for_server_ready(
    host: str,
    port: int,
    timeout: float = 300.0,
    *,
    initial_grace: float = 8.0,
) -> bool:
    """
    Wait until CARLA RPC is up **and** responds to world/map queries.

    ``get_server_version()`` alone is not enough — the engine can answer RPC while
    still on the splash screen and before it can load maps.
    """
    if not wait_for_server(host, port, timeout=timeout, initial_grace=initial_grace):
        return False
    t0 = time.time()
    last_log = 0.0
    while time.time() - t0 < timeout:
        try:
            c = carla.Client(host, port)
            c.set_timeout(10.0)
            version = c.get_server_version()
            world = c.get_world()
            map_name = world.get_map().name
            print(f"CARLA engine ready (v{version}, map={map_name}).", flush=True)
            return True
        except Exception as ex:
            now = time.time()
            if now - last_log >= 5.0:
                print(f"  …waiting for CARLA world ({now - t0:.0f}s): {ex}", flush=True)
                last_log = now
            time.sleep(0.5)
    return False


def launch_simulator(
    sim_dir: Path,
    *,
    rhi: str = "d3d12",
    quality: str = "Medium",
    win_w: int = 1920,
    win_h: int = 1080,
) -> None:
    """
    Start CARLA. ``rhi`` selects the Unreal RHI (helps with some "D3D device lost" cases on DX11).

    - ``d3d12`` — pass ``-d3d12`` (default).
    - ``d3d11`` — pass ``-dx11`` (legacy).
    - ``default`` — no RHI flag; engine chooses.
    """
    exe = sim_dir / "CarlaUE4.exe"
    if not exe.is_file():
        raise FileNotFoundError(
            f"CarlaUE4.exe not found under {sim_dir}\n"
            "Set CARLA_ROOT, edit demos/carla_root.txt, or run demos/Set_CARLA_Root.bat — see docs/CARLA.md."
        )
    args = [str(exe)]
    if rhi == "d3d12":
        args.append("-d3d12")
    elif rhi == "d3d11":
        args.append("-dx11")
    elif rhi != "default":
        raise ValueError(f"launch_simulator: unknown rhi {rhi!r} (use d3d12, d3d11, default)")
    args.extend(
        [
            f"-quality-level={quality}",
            "-windowed",
            "-nosound",
            f"-ResX={win_w}",
            f"-ResY={win_h}",
        ]
    )
    print(f"Starting CARLA: {' '.join(args[1:])}", flush=True)
    subprocess.Popen(args, cwd=str(sim_dir), env=os.environ.copy())


def opendrive_generation_params(max_road_length: float | None = None) -> carla.OpendriveGenerationParameters:
    """
    ``max_road_length`` caps tessellation segment length; use at least your longest road geometry.
    """
    mrl = float(max_road_length) if max_road_length is not None else float(BUILTIN_OPENDRIVE_MAX_LENGTH_M)
    return carla.OpendriveGenerationParameters(
        vertex_distance=4.0,
        max_road_length=mrl,
        wall_height=0.0,
        additional_width=0.0,
        smooth_junctions=False,
        enable_mesh_visibility=True,
    )


def load_opendrive_string(
    client: carla.Client,
    xodr: str,
    *,
    description: str = "OpenDRIVE",
    max_road_length: float | None = None,
) -> carla.World:
    params = opendrive_generation_params(max_road_length=max_road_length)
    print(
        f"Generating world from OpenDRIVE ({description}) — first load can take 1–2 min…",
        flush=True,
    )
    client.set_timeout(180.0)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            world = client.generate_opendrive_world(xodr, params)
            restore_world_async(world)
            print("OpenDRIVE world loaded.", flush=True)
            return world
        except Exception as ex:
            last_err = ex
            print(f"OpenDRIVE load attempt {attempt}/3 failed: {ex}", flush=True)
            time.sleep(2.0)
    raise RuntimeError(f"generate_opendrive_world failed after 3 attempts: {last_err}") from last_err


def builtin_opendrive_for_preset(preset: str) -> tuple[str, str, float]:
    """
    Returns ``(xodr_xml, map_label, recommended_max_road_length_m)`` for built-in presets
    when not using ``--xodr-file``.
    """
    p = (preset or "straight").strip().lower()
    if p == "left_curve":
        return (
            MINIMAL_CURVED_XODR,
            f"OpenDRIVE:{int(CURVED_ROAD_LENGTH_M)}m arc R={int(CURVED_ARC_RADIUS_M)}m (1 lane)",
            float(CURVED_ROAD_LENGTH_M),
        )
    if p in ("loom_wall", "loom"):
        return (
            MINIMAL_LOOM_STRAIGHT_XODR,
            f"OpenDRIVE:loom {int(LOOM_TEST_ROAD_LENGTH_M)}m straight + wall (1 lane)",
            float(LOOM_TEST_ROAD_LENGTH_M),
        )
    if p == "straight_cone":
        return (
            MINIMAL_STRAIGHT_CONE_XODR,
            f"OpenDRIVE:{int(STRAIGHT_CONE_ROAD_LENGTH_M)}m straight + cone dodge (1 lane)",
            float(STRAIGHT_CONE_ROAD_LENGTH_M),
        )
    return (
        MINIMAL_STRAIGHT_XODR,
        f"OpenDRIVE:{int(STRAIGHT_ROAD_LENGTH_M)}m straight (1 lane)",
        float(STRAIGHT_ROAD_LENGTH_M),
    )


def _lift_spawn_z(t: carla.Transform, dz: float) -> carla.Transform:
    t2 = copy_transform(t)
    t2.location.z += dz
    return t2


def _waypoint_from_location(
    m: carla.Map,
    loc: carla.Location,
    *,
    lane_types: tuple = (carla.LaneType.Driving, carla.LaneType.Any),
) -> carla.Waypoint | None:
    """Lane-center waypoint for ``loc`` (use spawn / intended position, not a half-off vehicle)."""
    wp = None
    for lt in lane_types:
        try:
            wp = m.get_waypoint(loc, project_to_road=True, lane_type=lt)
        except TypeError:
            wp = m.get_waypoint(loc, project_to_road=True)
            break
        if wp is not None:
            break
    return wp


def priority_spawn_waypoints(
    world: carla.World,
    *,
    prefer_road_start: bool = False,
    start_window_m: float = 22.0,
    min_s_m: float = 3.0,
) -> list[carla.Waypoint]:
    """
    Ordered waypoints to try for hero spawn. For OpenDRIVE, ``min_s_m`` skips s≈0 where the mesh
    seam can mis-project; prefer a few meters into the road for a stable lane center.
    """
    m = world.get_map()
    pts = m.get_spawn_points()
    if pts:
        probe = [pts[0]] if prefer_road_start else pts[:32]
        out: list[carla.Waypoint] = []
        for p in probe:
            wp = _waypoint_from_location(m, p.location)
            if wp is not None:
                out.append(wp)
        if out:
            return out

    wps: list[carla.Waypoint] = []
    try:
        wps = list(m.generate_waypoints(2.0))
    except Exception:
        wps = []

    if not wps:
        return []

    if prefer_road_start:
        driving = [
            w
            for w in wps
            if w.lane_type == carla.LaneType.Driving and int(w.lane_id) < 0
        ]
        if not driving:
            driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
        if not driving:
            driving = wps
        by_road: dict[int, list[carla.Waypoint]] = {}
        for w in driving:
            by_road.setdefault(w.road_id, []).append(w)
        picked: list[carla.Waypoint] = []
        for rid in sorted(by_road.keys()):
            seg = sorted(by_road[rid], key=lambda w: float(w.s))
            # Skip the first meters (unstable projection / seam); then take up to start_window_m.
            seg_ok = [w for w in seg if float(w.s) >= min_s_m and float(w.s) <= start_window_m]
            picked.extend(seg_ok[:20])
        if not picked:
            for rid in sorted(by_road.keys()):
                seg = sorted(by_road[rid], key=lambda w: float(w.s))
                # Fallback: smallest s ≥ min_s_m, else absolute first
                seg2 = [w for w in seg if float(w.s) >= min_s_m]
                if seg2:
                    picked.append(seg2[0])
                elif seg:
                    picked.append(seg[0])
                if picked:
                    break
        return picked

    random.shuffle(wps)
    return wps[:120]


def spawn_transforms_for_world(
    world: carla.World,
    *,
    prefer_road_start: bool = False,
    start_window_m: float = 22.0,
    min_s_m: float = 3.0,
    only_fallback: bool = False,
) -> list[carla.Transform]:
    """
    Prefer real lane geometry (waypoints); OpenDRIVE imports usually support generate_waypoints.

    If ``prefer_road_start`` is True, pick transforms only from the **beginning** of each
    driving lane (low ``s`` along the road) so the vehicle starts near the origin and on
    the lane centerline (``align_vehicle_to_road`` then snaps tires to mesh).
    Packaged maps with native spawn points: when ``prefer_road_start``, use the **first**
    spawn point only (deterministic).

    If ``only_fallback`` is True, skip the priority waypoint list (used after
    spawn_hero_vehicle already tried those transforms).
    """
    if not only_fallback:
        wplist = priority_spawn_waypoints(
            world,
            prefer_road_start=prefer_road_start,
            start_window_m=start_window_m,
            min_s_m=min_s_m,
        )
        if wplist:
            return [_lift_spawn_z(w.transform, 0.08) for w in wplist]

    m = world.get_map()
    pts = m.get_spawn_points()
    if pts:
        if prefer_road_start:
            return [pts[0]]
        return list(pts)

    # CARLA builds a road mesh from XODR — waypoints sit on the lane centerline.
    wps: list[carla.Waypoint] = []
    try:
        wps = list(m.generate_waypoints(4.0))
    except Exception:
        wps = []

    if wps:
        road_transforms = [_lift_spawn_z(w.transform, 0.08) for w in wps]
        random.shuffle(road_transforms)
        return road_transforms[:120]

    # Last resort: coarse grid along +X (straight preset); left_curve should have waypoints.
    candidates = []
    for x in (8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 400.0, 600.0, 800.0, 950.0):
        for y in (-1.75, 1.75, -2.5, 2.5):
            for z in (0.15, 0.35, 0.55, 1.0):
                candidates.append(
                    carla.Transform(
                        carla.Location(x=x, y=y, z=z),
                        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                    )
                )
    return candidates


def spawn_hero_vehicle(
    world: carla.World,
    bp: carla.ActorBlueprint,
    *,
    prefer_road_start: bool,
    start_window_m: float = 22.0,
    min_s_m: float = 3.0,
) -> tuple[carla.Vehicle | None, carla.Waypoint | None, carla.Transform | None]:
    """
    Try priority lane waypoints first, then spawn_transforms_for_world.
    Returns (vehicle, reference_waypoint_or_None, attempt_transform).
    """
    wps = priority_spawn_waypoints(
        world,
        prefer_road_start=prefer_road_start,
        start_window_m=start_window_m,
        min_s_m=min_s_m,
    )
    for wp in wps:
        pt = _lift_spawn_z(wp.transform, 0.08)
        v = world.try_spawn_actor(bp, pt)
        if v:
            return v, wp, pt
    for pt in spawn_transforms_for_world(
        world,
        prefer_road_start=prefer_road_start,
        start_window_m=start_window_m,
        min_s_m=min_s_m,
        only_fallback=True,
    ):
        v = world.try_spawn_actor(bp, pt)
        if v:
            return v, None, pt
    return None, None, None


def align_vehicle_to_road(
    vehicle: carla.Vehicle,
    world: carla.World,
    *,
    settle_ticks: int = 3,
    reference_wp: carla.Waypoint | None = None,
    spawn_location: carla.Location | None = None,
    refine_passes: int = 2,
) -> carla.Transform:
    """
    Snap vehicle onto the **lane centerline**.

    If the vehicle is already half-off the road, ``get_waypoint(vehicle.get_location())``
    projects to the **road edge**, not the lane center — the car stays half-off and can fall.

    Prefer ``reference_wp`` (waypoint used for spawn) or ``spawn_location`` (intended spawn
    point) so the snap uses the true lane center. ``refine_passes`` repeats snap after ticks.
    """
    for _ in range(max(1, settle_ticks)):
        advance_world_ticks(world, 1)

    m = world.get_map()

    def _choose_wp(pass_idx: int) -> carla.Waypoint | None:
        # Lane-center waypoint from spawn intent — never project a half-off vehicle on pass 0
        # when we have a reference or spawn_location.
        if reference_wp is not None:
            return reference_wp
        if pass_idx == 0 and spawn_location is not None:
            wp0 = _waypoint_from_location(m, spawn_location)
            if wp0 is not None:
                return wp0
        loc = vehicle.get_location()
        return _waypoint_from_location(m, loc)

    last_t = copy_transform(vehicle.get_transform())
    for _pass in range(max(1, int(refine_passes))):
        wp = _choose_wp(_pass)
        if wp is None:
            return last_t

        t = copy_transform(wp.transform)
        # Actor origin ≈ body center; offset so tires sit on generated mesh (tune if floating/sinking).
        ext_z = float(vehicle.bounding_box.extent.z)
        t.location.z += ext_z + 0.06
        t.rotation.pitch = wp.transform.rotation.pitch
        t.rotation.yaw = wp.transform.rotation.yaw
        t.rotation.roll = wp.transform.rotation.roll

        vehicle.set_transform(t)
        clear_vehicle_motion(vehicle)
        advance_world_ticks(world, 1)
        last_t = copy_transform(vehicle.get_transform())

    return last_t


def pick_map(client: carla.Client, requested: str | None) -> str:
    names = {m.rstrip("/").split("/")[-1] for m in client.get_available_maps()}

    if requested:
        key = requested.rstrip("/").split("/")[-1]
        if key in names:
            return key
        print(f"Map {requested!r} missing. List: python carla_fly_session.py --list-maps")
        print("Falling back to default straight-road preference.")

    for name in DEFAULT_MAP_PREFERENCE:
        if name in names:
            return name
    towns = sorted(x for x in names if x.startswith("Town"))
    if towns:
        return towns[0]
    if names:
        return sorted(names)[0]
    raise RuntimeError("No maps reported by server.")


def copy_transform(t: carla.Transform) -> carla.Transform:
    return carla.Transform(
        carla.Location(t.location.x, t.location.y, t.location.z),
        carla.Rotation(t.rotation.pitch, t.rotation.yaw, t.rotation.roll),
    )


def clear_vehicle_motion(vehicle: carla.Vehicle) -> None:
    zero = carla.Vector3D(0.0, 0.0, 0.0)
    for name in ("set_velocity", "set_target_velocity"):
        fn = getattr(vehicle, name, None)
        if callable(fn):
            try:
                fn(zero)
            except Exception:
                pass
    for name in ("set_angular_velocity", "set_target_angular_velocity"):
        fn = getattr(vehicle, name, None)
        if callable(fn):
            try:
                fn(zero)
            except Exception:
                pass


def reset_vehicle_to_spawn(vehicle: carla.Vehicle, spawn_transform: carla.Transform) -> None:
    vehicle.set_transform(spawn_transform)
    clear_vehicle_motion(vehicle)


def pick_static_prop_blueprint(world: carla.World) -> carla.ActorBlueprint | None:
    """First available small static prop suitable as a road obstacle."""
    lib = world.get_blueprint_library()
    for bid in (
        "static.prop.constructioncone",
        "static.prop.trafficcone01",
        "static.prop.trafficcone02",
        "static.prop.trafficcone",
        "static.prop.barrier",
        "static.prop.plasticchair",
    ):
        found = list(lib.filter(bid))
        if found:
            return found[0]
    props = list(lib.filter("static.prop.*"))
    return props[0] if props else None


def pick_traffic_cone_blueprint(world: carla.World) -> carla.ActorBlueprint | None:
    """Orange traffic / construction cone for dodge demos."""
    lib = world.get_blueprint_library()
    for bid in (
        "static.prop.trafficcone01",
        "static.prop.trafficcone02",
        "static.prop.constructioncone",
        "static.prop.trafficcone",
    ):
        found = list(lib.filter(bid))
        if found:
            return found[0]
    return pick_static_prop_blueprint(world)


def _road_surface_z_at(m: carla.Map, loc: carla.Location) -> float:
    """Lane/road elevation at ``loc`` (projects lateral offsets to the mesh)."""
    wp = _waypoint_from_location(m, loc)
    if wp is not None:
        return float(wp.transform.location.z)
    return float(loc.z)


def _snap_actor_base_to_z(actor: carla.Actor, target_base_z: float) -> None:
    """Move a static prop so the bottom of its bounding box sits on ``target_base_z``."""
    bb = actor.bounding_box
    tf = actor.get_transform()
    bottom_world_z = float(tf.location.z) + float(bb.location.z) - float(bb.extent.z)
    dz = float(target_base_z) - bottom_world_z
    if abs(dz) > 1e-5:
        tf.location.z += dz
        actor.set_transform(tf)


def spawn_mid_road_obstacle_cluster(
    world: carla.World,
    *,
    road_fraction: float = 0.5,
    lateral_offsets_m: tuple[float, ...] = (1.55,),
    cone_only: bool = False,
) -> list[carla.Actor]:
    """
    Place static props at ~``road_fraction`` along the first road (driving lane ``lane_id < 0``).

    ``lateral_offsets_m`` are meters along the waypoint **right** vector (+ = toward the outer /
    driver's-right edge of the lane for typical OpenDRIVE lane -1). Default is **one** cone near
    the right edge (~1.55 m for a 3.5 m lane). Pass e.g. ``(-1.1, 0.0, 1.1)`` for three across.
    """
    bp = pick_traffic_cone_blueprint(world) if cone_only else pick_static_prop_blueprint(world)
    if bp is None:
        print("Warning: no static.prop blueprint for mid-road obstacle.")
        return []
    if cone_only and bp.has_attribute("simulate_physics"):
        bp.set_attribute("simulate_physics", "false")

    m = world.get_map()
    try:
        wps = [
            w
            for w in m.generate_waypoints(2.0)
            if w.lane_type == carla.LaneType.Driving and int(w.lane_id) < 0
        ]
    except Exception:
        wps = []

    if not wps:
        try:
            wps = [w for w in m.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
        except Exception:
            wps = []

    if not wps:
        print("Warning: no waypoints for mid-road obstacle.")
        return []

    wps.sort(key=lambda w: (w.road_id, float(w.s)))
    rid0 = wps[0].road_id
    same = [w for w in wps if w.road_id == rid0]
    if not same:
        return []
    frac = min(1.0, max(0.0, float(road_fraction)))
    idx = int(round((len(same) - 1) * frac))
    idx = max(0, min(idx, len(same) - 1))
    wp = same[idx]

    actors: list[carla.Actor] = []
    try:
        right = wp.transform.get_right_vector()
    except Exception:
        right = carla.Vector3D(0.0, 1.0, 0.0)

    for off in lateral_offsets_m:
        loc = carla.Location(
            wp.transform.location.x + float(right.x) * off,
            wp.transform.location.y + float(right.y) * off,
            wp.transform.location.z + float(right.z) * off,
        )
        if cone_only:
            loc.z = _road_surface_z_at(m, loc)
        else:
            loc.z += 0.12
        tf = carla.Transform(loc, wp.transform.rotation)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            if cone_only:
                advance_world_ticks(world, 1)
                ground_z = _road_surface_z_at(m, actor.get_location())
                _snap_actor_base_to_z(actor, ground_z + 0.015)
            actors.append(actor)
    if actors:
        print(
            f"Mid-road obstacle: {len(actors)}× {bp.id} at road_id={rid0} "
            f"s≈{float(wp.s):.1f} m (fraction={frac:.2f})."
        )
    return actors


def pick_loom_wall_blueprint(world: carla.World) -> carla.ActorBlueprint | None:
    """
    Prefer **narrow** props we can tile across the full lane.

    A single large mesh (e.g. container) is often pivot-heavy and reads as blocking
    only one side, so the car can dodge the other way; barriers tile edge-to-edge.
    """
    lib = world.get_blueprint_library()
    for bid in (
        "static.prop.barrier",
        "static.prop.streetbarrier",
        "static.prop.container",
    ):
        found = list(lib.filter(bid))
        if found:
            return found[0]
    return pick_static_prop_blueprint(world)


def spawn_loom_test_wall(
    world: carla.World,
    *,
    road_fraction: float | None = None,
    lateral_offsets_m: tuple[float, ...] | None = None,
    row_stagger_m: tuple[float, ...] | None = None,
) -> list[carla.Actor]:
    """
    Lane-blocking wall on the short ``loom_wall`` straight: **dense** props across
    the full lane width plus a **staggered** second row along the road so the car
    cannot slip past on the left or right.
    """
    frac_in = LOOM_WALL_ROAD_FRACTION if road_fraction is None else float(road_fraction)
    bp = pick_loom_wall_blueprint(world)
    if bp is None:
        print("Warning: no blueprint for loom wall.")
        return []

    if lateral_offsets_m is None:
        lateral_offsets_m = LOOM_WALL_FULL_LANE_OFFSETS_M
    if row_stagger_m is None:
        row_stagger_m = LOOM_WALL_ROW_STAGGER_M

    m = world.get_map()
    try:
        wps = [
            w
            for w in m.generate_waypoints(2.0)
            if w.lane_type == carla.LaneType.Driving and int(w.lane_id) < 0
        ]
    except Exception:
        wps = []

    if not wps:
        try:
            wps = [w for w in m.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
        except Exception:
            wps = []

    if not wps:
        print("Warning: no waypoints for loom wall.")
        return []

    wps.sort(key=lambda w: (w.road_id, float(w.s)))
    rid0 = wps[0].road_id
    same = [w for w in wps if w.road_id == rid0]
    if not same:
        return []
    frac = min(1.0, max(0.0, float(frac_in)))
    idx = int(round((len(same) - 1) * frac))
    idx = max(0, min(idx, len(same) - 1))
    wp = same[idx]

    actors: list[carla.Actor] = []
    try:
        right = wp.transform.get_right_vector()
    except Exception:
        right = carla.Vector3D(0.0, 1.0, 0.0)
    try:
        forward = wp.transform.get_forward_vector()
    except Exception:
        forward = carla.Vector3D(1.0, 0.0, 0.0)

    z_lift = 0.18 if "container" in bp.id.lower() else 0.12
    base = wp.transform.location
    for along in row_stagger_m:
        ax = float(base.x) + float(forward.x) * along
        ay = float(base.y) + float(forward.y) * along
        az = float(base.z) + float(forward.z) * along
        for off in lateral_offsets_m:
            loc = carla.Location(
                ax + float(right.x) * off,
                ay + float(right.y) * off,
                az + float(right.z) * off + z_lift,
            )
            tf = carla.Transform(loc, wp.transform.rotation)
            actor = world.try_spawn_actor(bp, tf)
            if actor is not None:
                actors.append(actor)
    if actors:
        n_rows = len(row_stagger_m)
        print(
            f"Loom wall: {len(actors)}× {bp.id} at road_id={rid0} s≈{float(wp.s):.1f} m "
            f"(fraction={frac:.2f}, {n_rows} row(s), full-lane lateral tiling)."
        )
    return actors


def bounds_mode_for_args(args: argparse.Namespace) -> str:
    """straight_xodr | left_curve_opendrive | custom_xodr | packaged"""
    if getattr(args, "map", None):
        return "packaged"
    if getattr(args, "xodr_file", None):
        return "custom_xodr"
    preset = getattr(args, "opendrive_preset", None) or "straight"
    if str(preset).lower() == "left_curve":
        return "left_curve_opendrive"
    if str(preset).lower() in ("loom_wall", "loom"):
        return "loom_wall_opendrive"
    if str(preset).lower() == "straight_cone":
        return "straight_cone_opendrive"
    return "straight_xodr"


def is_out_of_bounds(
    loc: carla.Location,
    spawn_loc: carla.Location,
    mode: str,
) -> bool:
    # Fell through world / below road
    if loc.z < spawn_loc.z - 12.0:
        return True
    if mode == "straight_xodr":
        # Minimal ODR: reference line +X for STRAIGHT_ROAD_LENGTH_M m; spawn ~x≈8, y≈±1.75
        dx = loc.x - spawn_loc.x
        dy = loc.y - spawn_loc.y
        # End of road at x ≈ STRAIGHT_ROAD_LENGTH_M; allow ~128 m past lane before reset
        max_forward_dx = STRAIGHT_ROAD_LENGTH_M - spawn_loc.x + 128.0
        if dx < -80.0 or dx > max_forward_dx:
            return True
        if abs(dy) > 28.0:
            return True
        return False
    if mode == "straight_cone_opendrive":
        dx = loc.x - spawn_loc.x
        dy = loc.y - spawn_loc.y
        max_forward_dx = STRAIGHT_CONE_ROAD_LENGTH_M - spawn_loc.x + 128.0
        if dx < -80.0 or dx > max_forward_dx:
            return True
        if abs(dy) > 28.0:
            return True
        return False
    if mode == "left_curve_opendrive":
        # Generated arc: stay within a generous radius of spawn (lane-following is loose).
        if loc.distance(spawn_loc) > 280.0:
            return True
        return False
    if mode == "loom_wall_opendrive":
        dx = loc.x - spawn_loc.x
        dy = loc.y - spawn_loc.y
        max_forward_dx = LOOM_TEST_ROAD_LENGTH_M - spawn_loc.x + 48.0
        if dx < -40.0 or dx > max_forward_dx:
            return True
        if abs(dy) > 22.0:
            return True
        return False
    # Packaged town or custom .xodr: radius + fall
    if loc.distance(spawn_loc) > 420.0:
        return True
    return False


def crazy_vibration_offsets(phase: float) -> tuple[float, float]:
    """Returns (steer_delta, throttle_delta); caller applies only on active inputs."""
    s = (
        0.42 * math.sin(phase * 58.0)
        + 0.28 * math.sin(phase * 103.0)
        + 0.18 * math.sin(phase * 187.0)
    )
    t = (
        0.14 * math.cos(phase * 67.0)
        + 0.10 * math.sin(phase * 131.0)
    )
    s += random.uniform(-0.22, 0.22)
    t += random.uniform(-0.10, 0.10)
    return s, t


def vibration_intensity(hand_brake: bool) -> float:
    """Full strength when allowed; never inject steering unless driver turns (see main loop)."""
    return 0.0 if hand_brake else 1.0


def print_spawn_probe(vehicle: carla.Vehicle, world: carla.World, map_label: str) -> None:
    t = vehicle.get_transform()
    loc, rot = t.location, t.rotation
    print("")
    print("=== PROBE SPAWN (verify coords) ===")
    print(f"  map_label     : {map_label}")
    print(f"  location (m)  : x={loc.x:+.4f}  y={loc.y:+.4f}  z={loc.z:+.4f}")
    print(f"  rotation (deg): pitch={rot.pitch:+.2f}  yaw={rot.yaw:+.2f}  roll={rot.roll:+.2f}")
    try:
        m = world.get_map()
        try:
            wp = m.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except TypeError:
            wp = m.get_waypoint(loc, project_to_road=True)
        if wp:
            lw = getattr(wp, "lane_width", float("nan"))
            print(
                f"  waypoint      : road_id={wp.road_id} lane_id={wp.lane_id} "
                f"s={wp.s:.3f} lane_width≈{lw:.2f}m"
            )
        else:
            print("  waypoint      : None (OpenDRIVE import often has no driving waypoint here — OK if car sits on mesh)")
    except Exception as ex:
        print(f"  waypoint      : ({type(ex).__name__}) {ex}")
    print("=== END PROBE ===")
    print("")


def resolve_vehicle_bp(world: carla.World, explicit: str | None) -> carla.ActorBlueprint:
    lib = world.get_blueprint_library()
    if explicit:
        bps = lib.filter(explicit)
        if bps:
            return bps[0]
        print(f"Blueprint {explicit!r} missing, using preference list.")
    for vid in VEHICLE_PREFERENCE:
        bps = lib.filter(vid)
        if bps:
            return bps[0]
    cars = [x for x in lib.filter("vehicle.*") if "two_wheels" not in x.id and "bicycle" not in x.id.lower()]
    if not cars:
        cars = list(lib.filter("vehicle.*"))
    return random.choice(cars)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("-p", "--port", type=int, default=2000)
    ap.add_argument(
        "--map",
        default=None,
        help="Packaged CARLA map, e.g. Town10HD_Opt (if set, skips built-in OpenDRIVE presets)",
    )
    ap.add_argument(
        "--xodr-file",
        type=Path,
        default=None,
        help="Load custom OpenDRIVE from file instead of built-in presets",
    )
    ap.add_argument(
        "--opendrive-preset",
        choices=("straight", "straight_cone", "left_curve", "loom_wall"),
        default="straight",
        help="Built-in OpenDRIVE when --map is omitted (ignored if --xodr-file is set). "
        "'straight_cone' = 1000 m straight + cone dodge, "
        "'left_curve' = single-lane arc, 'loom_wall' = short straight + wall; "
        "use --no-mid-road-obstacle to skip cones/wall.",
    )
    ap.add_argument(
        "--no-mid-road-obstacle",
        action="store_true",
        help="With curve/loom_wall/straight_cone presets, do not spawn static props / wall.",
    )
    ap.add_argument("--vehicle", default=None, help="Blueprint id, e.g. vehicle.tesla.model3")
    ap.add_argument("--ui-res", default="560x140", help="Small pygame window WxH (keyboard focus)")
    ap.add_argument("--list-maps", action="store_true", help="Print available maps and exit")
    ap.add_argument("--launch", action="store_true", help="Start CarlaUE4.exe before connecting")
    ap.add_argument(
        "--sim-dir",
        type=Path,
        default=None,
        help="CARLA install folder (default: $CARLA_ROOT or ./CARLA_0.9.16)",
    )
    ap.add_argument("--quality-level", default="Medium", help="Epic|High|Medium|Low (passed to simulator)")
    ap.add_argument("--sim-res", default="1920x1080", help="Simulator window WxH")
    ap.add_argument(
        "--rhi",
        choices=("d3d12", "d3d11", "default"),
        default="d3d12",
        help="Unreal graphics API: d3d12 (default, often stabler than DX11), d3d11, or default (no flag)",
    )
    ap.add_argument(
        "--no-dx11",
        action="store_true",
        help="Deprecated: same as --rhi default",
    )
    ap.add_argument(
        "--probe-spawn",
        action="store_true",
        help="Load world, spawn vehicle, print transform/waypoint to terminal, exit (needs running CARLA unless --launch)",
    )
    args = ap.parse_args()

    if args.launch:
        sim_dir = resolve_sim_dir(args.sim_dir)
        print(f"Using CARLA install: {sim_dir.resolve()}", flush=True)
    elif args.sim_dir:
        sim_dir = Path(args.sim_dir)
    else:
        sim_dir = default_sim_dir()

    if args.launch:
        sw, sh = [int(x) for x in args.sim_res.split("x")]
        rhi = "default" if args.no_dx11 else args.rhi
        reuse = False
        if server_is_up(args.host, args.port) and probe_server_world(args.host, args.port):
            print(f"CARLA already running on {args.host}:{args.port} — reusing.", flush=True)
            reuse = True
        elif carla_process_running() or server_is_up(args.host, args.port):
            print("Stopping stuck CARLA process before relaunch…", flush=True)
            terminate_carla_processes()
            time.sleep(3.0)
        if not reuse:
            print(
                f"Launching CARLA from {sim_dir} ({args.sim_res}, quality={args.quality_level}, rhi={rhi})…",
                flush=True,
            )
            launch_simulator(
                sim_dir,
                rhi=rhi,
                quality=args.quality_level,
                win_w=sw,
                win_h=sh,
            )
            if not wait_for_server_ready(
                args.host,
                args.port,
                timeout=300.0,
                initial_grace=8.0,
            ):
                print("Timed out waiting for CARLA. Is the install complete?", flush=True)
                return 1

    print("Connecting...", flush=True)
    client = carla.Client(args.host, args.port)
    # Fast fail for one-shot probe when sim is down
    client.set_timeout(15.0 if args.probe_spawn else 180.0)

    try:
        restore_world_async(client.get_world())
    except Exception:
        pass

    if args.list_maps:
        for m in sorted(client.get_available_maps()):
            print(m.replace("/Game/Carla/Maps/", ""))
        return 0

    map_label: str
    use_opendrive = not bool(args.map)
    max_odr_len: float | None = None
    if args.map:
        map_name = pick_map(client, args.map)
        print(f"Loading packaged map: {map_name}")
        client.load_world(map_name)
        world = client.get_world()
        map_label = map_name
    else:
        if args.xodr_file:
            path = Path(args.xodr_file)
            if not path.is_file():
                print(f"OpenDRIVE file not found: {path}")
                return 1
            xodr = path.read_text(encoding="utf-8")
            map_label = f"OpenDRIVE:{path.name}"
            max_odr_len = float(BUILTIN_OPENDRIVE_MAX_LENGTH_M)
        else:
            xodr, map_label, max_odr_len = builtin_opendrive_for_preset(args.opendrive_preset)
        world = load_opendrive_string(
            client,
            xodr,
            description=map_label,
            max_road_length=max_odr_len,
        )

    # Simple clear weather — no extra FX load.
    world.set_weather(carla.WeatherParameters.ClearNoon)

    bp = resolve_vehicle_bp(world, args.vehicle)
    bp.set_attribute("role_name", "hero")
    if bp.has_attribute("color") and bp.get_attribute("color").recommended_values:
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

    vehicle, spawn_ref_wp, attempt_tf = spawn_hero_vehicle(world, bp, prefer_road_start=use_opendrive)
    if not vehicle:
        print("Spawn failed.")
        return 1
    spawn_attempt_loc = attempt_tf.location if attempt_tf is not None else None

    print("Aligning vehicle to lane / road surface (lane-center snap)...")
    spawn_transform = align_vehicle_to_road(
        vehicle,
        world,
        reference_wp=spawn_ref_wp,
        spawn_location=None if spawn_ref_wp is not None else spawn_attempt_loc,
        settle_ticks=4,
    )
    bounds_mode = bounds_mode_for_args(args)

    obstacle_actors: list[carla.Actor] = []
    preset_key = str(args.opendrive_preset).lower()
    if (
        use_opendrive
        and not args.xodr_file
        and preset_key == "straight_cone"
        and not args.no_mid_road_obstacle
    ):
        obstacle_actors = spawn_mid_road_obstacle_cluster(
            world,
            road_fraction=0.48,
            lateral_offsets_m=(-1.68,),
            cone_only=True,
        )
    elif (
        use_opendrive
        and not args.xodr_file
        and preset_key == "loom_wall"
        and not args.no_mid_road_obstacle
    ):
        obstacle_actors = spawn_loom_test_wall(world)

    print_spawn_probe(vehicle, world, map_label)

    if args.probe_spawn:
        try:
            vehicle.destroy()
        except Exception:
            pass
        for obs in obstacle_actors:
            try:
                if obs.is_alive:
                    obs.destroy()
            except Exception:
                pass
        print("probe-spawn: OK (vehicle destroyed).")
        return 0

    print(
        f"Spawned {vehicle.type_id}. Click pygame: WASD, Space, R=reset spawn, ESC quit. "
        f"Bounds={bounds_mode}; vib only on keys you hold (W/S/A/D), not while coasting straight. Spectator follows car."
    )

    uw, uh = [int(x) for x in args.ui_res.split("x")]
    cam_w, cam_h = 640, 360
    cam_fps = 30
    cam_sensor_tick_s = 1.0 / float(cam_fps)
    win_w = max(uw, cam_w)
    win_h = uh + cam_h
    import pygame

    pygame.init()
    pygame.display.set_caption("KEYBOARD FOCUS — Fly-By-Wire CARLA")
    screen = pygame.display.set_mode((win_w, win_h), pygame.SWSURFACE)
    font = pygame.font.SysFont("consolas", 17)
    clock = pygame.time.Clock()
    control = carla.VehicleControl()
    steer_cache = 0.0
    spawn_loc = carla.Location(
        spawn_transform.location.x,
        spawn_transform.location.y,
        spawn_transform.location.z,
    )

    latest_cam: dict[str, pygame.Surface | None] = {"surf": None}
    cam_lock = threading.Lock()
    rgb_camera = None
    # 30x30 grayscale @ ~30 Hz, same preprocessing as neural_projector / training video.
    model_feed = RetinotopicIngest(target_fps=float(cam_fps), drop_duplicate_frame_id=True)

    def _on_camera_image(image: carla.Image) -> None:
        try:
            surf = _pygame_surface_from_carla_image(image)
            with cam_lock:
                latest_cam["surf"] = surf
            model_feed.ingest_carla_image(image)
        except Exception:
            pass

    try:
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(cam_w))
        cam_bp.set_attribute("image_size_y", str(cam_h))
        cam_bp.set_attribute("fov", "90")
        # Fixed 30 Hz capture (sim time between frames), not every Unreal frame.
        try:
            cam_bp.set_attribute("sensor_tick", f"{cam_sensor_tick_s:.6f}")
        except Exception:
            print("Warning: could not set sensor_tick; camera rate may not be 30 Hz.")
        configure_rgb_camera_bp(cam_bp, disable_postprocess=False)
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.5))
        rgb_camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        rgb_camera.listen(_on_camera_image)
        print(
            f"RGB camera: {cam_w}x{cam_h} @ {cam_fps} Hz (sensor_tick={cam_sensor_tick_s:.4f}s); "
            f"artificial retina 30x30 gray -> vision_ingest.RetinotopicIngest"
        )
    except Exception as ex:
        print(f"RGB camera feed unavailable: {ex}")

    try:
        running = True
        while running:
            # Match pygame/sim client cadence to camera FPS so the feed updates every frame, not stale dupes at 60 Hz.
            clock.tick(cam_fps)
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_vehicle_to_spawn(vehicle, spawn_transform)
                    # Re-snap to lane center (same reference as initial spawn).
                    spawn_transform = align_vehicle_to_road(
                        vehicle,
                        world,
                        reference_wp=spawn_ref_wp,
                        spawn_location=None if spawn_ref_wp is not None else spawn_attempt_loc,
                        settle_ticks=2,
                    )
                    spawn_loc = carla.Location(
                        spawn_transform.location.x,
                        spawn_transform.location.y,
                        spawn_transform.location.z,
                    )
                    steer_cache = 0.0
                    control = carla.VehicleControl()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                control.throttle = min(control.throttle + 0.08, 1.0)
            else:
                control.throttle = 0.0
            if keys[pygame.K_s]:
                control.brake = min(control.brake + 0.15, 1.0)
            else:
                control.brake = 0.0

            dt = clock.get_time()
            inc = 5e-4 * max(dt, 1)
            if keys[pygame.K_a]:
                steer_cache = max(-0.7, steer_cache - inc) if steer_cache <= 0 else 0.0
            elif keys[pygame.K_d]:
                steer_cache = min(0.7, steer_cache + inc) if steer_cache >= 0 else 0.0
            else:
                steer_cache = 0.0
            control.steer = round(steer_cache, 2)
            control.hand_brake = keys[pygame.K_SPACE]

            # Vibration must NOT touch steer/throttle unless that axis is actively driven.
            # (Speed-based gain was causing random steer while driving straight with W held.)
            intensity = vibration_intensity(control.hand_brake)
            steer_keys = keys[pygame.K_a] or keys[pygame.K_d]
            throttle_key = keys[pygame.K_w]
            brake_key = keys[pygame.K_s]
            if intensity > 0.0 and (steer_keys or throttle_key or brake_key):
                phase = time.monotonic()
                dv_steer, dv_thr = crazy_vibration_offsets(phase)
                if steer_keys:
                    control.steer = max(-1.0, min(1.0, control.steer + dv_steer * intensity))
                if throttle_key:
                    control.throttle = max(0.0, min(1.0, control.throttle + dv_thr * intensity))
                if brake_key:
                    control.brake = max(0.0, min(1.0, control.brake + abs(dv_thr) * intensity))

            vehicle.apply_control(control)
            world.wait_for_tick()

            t = vehicle.get_transform()
            if is_out_of_bounds(t.location, spawn_loc, bounds_mode):
                reset_vehicle_to_spawn(vehicle, spawn_transform)
                spawn_transform = align_vehicle_to_road(
                    vehicle,
                    world,
                    reference_wp=spawn_ref_wp,
                    spawn_location=None if spawn_ref_wp is not None else spawn_attempt_loc,
                    settle_ticks=2,
                )
                spawn_loc = carla.Location(
                    spawn_transform.location.x,
                    spawn_transform.location.y,
                    spawn_transform.location.z,
                )
                steer_cache = 0.0
                control = carla.VehicleControl()
            back = t.get_forward_vector() * (-10.0)
            loc = t.location + back + carla.Location(z=4.5)
            world.get_spectator().set_transform(
                carla.Transform(loc, carla.Rotation(pitch=-14.0, yaw=t.rotation.yaw))
            )

            screen.fill((18, 22, 32))
            with cam_lock:
                cam_surf = latest_cam["surf"]
            if cam_surf is not None:
                screen.blit(cam_surf, (0, 0))
            lines = [
                f"Map: {map_label}  |  {vehicle.type_id}  |  bounds={bounds_mode}",
                "W/S A/D Space | R=reset+snap | ESC | vib only on held keys",
                f"thr={control.throttle:.2f} brk={control.brake:.2f} str={control.steer:+.2f}",
            ]
            y = cam_h + 6
            for line in lines:
                screen.blit(font.render(line, True, (230, 228, 210)), (8, y))
                y += 24
            pygame.display.flip()

    finally:
        try:
            if rgb_camera is not None and rgb_camera.is_alive:
                rgb_camera.stop()
                rgb_camera.destroy()
        except Exception:
            pass
        try:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass
        for obs in obstacle_actors:
            try:
                if obs.is_alive:
                    obs.destroy()
            except Exception:
                pass
        pygame.quit()
        print("Session ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
