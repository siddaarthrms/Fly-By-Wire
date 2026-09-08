"""Scripted instant-cut camerawork for the straight_cone demo."""

from __future__ import annotations

import math
from typing import Any

import carla

import carla_fly_session as cfs

WHEEL_CUT_SIM_T = 3.0
NEAR_CONE_M = 10.5
PASS_AHEAD_M = 2.8

PHASE_HIGH = "cone_high"
PHASE_WHEEL = "cone_wheel"
PHASE_BUMPER = "cone_bumper"
PHASE_BIRD = "cone_bird"


def _loc(obj: Any) -> tuple[float, float, float]:
    if hasattr(obj, "x"):
        return float(obj.x), float(obj.y), float(obj.z)
    return float(obj[0]), float(obj[1]), float(obj[2])


def planar_distance_m(car_loc: Any, cone_loc: Any) -> float:
    cx, cy, _ = _loc(car_loc)
    ox, oy, _ = _loc(cone_loc)
    return float(math.hypot(cx - ox, cy - oy))


def car_passed_cone(car_loc: Any, cone_loc: Any, *, ahead_m: float = PASS_AHEAD_M) -> bool:
    cx, _, _ = _loc(car_loc)
    ox, _, _ = _loc(cone_loc)
    return float(cx) > float(ox) + float(ahead_m)


def next_cone_phase(
    phase: str,
    car_loc: Any,
    cone_loc: Any,
    sim_t: float,
) -> str:
    p = str(phase or PHASE_HIGH)
    if p == PHASE_HIGH and float(sim_t) >= WHEEL_CUT_SIM_T:
        return PHASE_WHEEL
    dist = planar_distance_m(car_loc, cone_loc)
    if p == PHASE_WHEEL and dist <= NEAR_CONE_M:
        return PHASE_BUMPER
    if p == PHASE_BUMPER and car_passed_cone(car_loc, cone_loc):
        return PHASE_BIRD
    return p


def _look_at_rotation(cam: carla.Location, target: carla.Location) -> carla.Rotation:
    dx = float(target.x - cam.x)
    dy = float(target.y - cam.y)
    dz = float(target.z - cam.z)
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.hypot(dx, dy)
    pitch = -math.degrees(math.atan2(dz, max(1e-6, horiz)))
    return carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=0.0)


def _world_to_attach(desired: carla.Transform, vehicle: carla.Transform) -> carla.Transform:
    """Express a world camera pose as a rigid child offset of the hero vehicle."""
    v = vehicle
    yaw = math.radians(float(v.rotation.yaw))
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx = float(desired.location.x - v.location.x)
    dy = float(desired.location.y - v.location.y)
    dz = float(desired.location.z - v.location.z)
    local_x = dx * cy + dy * sy
    local_y = -dx * sy + dy * cy
    local_z = dz
    dyaw = ((float(desired.rotation.yaw) - float(v.rotation.yaw) + 180.0) % 360.0) - 180.0
    return carla.Transform(
        carla.Location(x=local_x, y=local_y, z=local_z),
        carla.Rotation(
            pitch=float(desired.rotation.pitch) - float(v.rotation.pitch),
            yaw=float(dyaw),
            roll=0.0,
        ),
    )


def _bumper_attach_tf(
    vehicle_tf: carla.Transform,
    *,
    dist_to_cone_m: float,
) -> tuple[carla.Transform, float]:
    """
    Straight-on bumper beat: road-axis camera ahead of the car, centered on lane,
    elevated above the grille, pitched slightly down. Retreats as the hero closes in
    so the rig stays off the cone mesh (cone sits on the left edge).
    """
    road = cfs.yaw_stabilized_transform(vehicle_tf)
    yaw_rad = math.radians(float(road.rotation.yaw))
    cx, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    closeness = max(0.0, min(1.0, (float(NEAR_CONE_M) - dist_to_cone_m) / max(1e-6, float(NEAR_CONE_M))))
    ahead_m = 4.8 + closeness * 5.2  # 4.8 m → 10 m ahead as car approaches

    cam_loc = carla.Location(
        road.location.x + ahead_m * cx,
        road.location.y + ahead_m * sy,
        road.location.z + 1.14,
    )
    fwd = road.get_forward_vector()
    bumper = carla.Location(
        road.location.x + float(fwd.x) * 2.05,
        road.location.y + float(fwd.y) * 2.05,
        road.location.z + 0.48,
    )
    world_tf = carla.Transform(cam_loc, _look_at_rotation(cam_loc, bumper))
    return _world_to_attach(world_tf, vehicle_tf), 88.0


class ConeDodgeCameraDirector:
    """Cuts: cone_high, cone_wheel, cone_bumper, cone_bird."""

    def __init__(self, cone_location: carla.Location) -> None:
        self._cone = carla.Location(
            float(cone_location.x),
            float(cone_location.y),
            float(cone_location.z),
        )
        self._phase = "cone_high"
        self._phase_enter_sim_t = 0.0

    @property
    def phase(self) -> str:
        return str(self._phase)

    def advance(self, vehicle_tf: carla.Transform, sim_t: float) -> tuple[carla.Transform, float]:
        prev = self._phase
        self._phase = next_cone_phase(self._phase, vehicle_tf.location, self._cone, float(sim_t))
        if self._phase != prev:
            self._phase_enter_sim_t = float(sim_t)
        if self._phase == "cone_bumper":
            dist = planar_distance_m(vehicle_tf.location, self._cone)
            return _bumper_attach_tf(vehicle_tf, dist_to_cone_m=dist)
        return cfs.record_camera_spec(self._phase)
