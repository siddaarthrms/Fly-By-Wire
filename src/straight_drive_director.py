"""Scripted jump-cut camerawork for the straight OpenDRIVE demo."""

from __future__ import annotations

import carla

import carla_fly_session as cfs

# Three beats over a ~22 s demo: wide rear → side profile → far lead.
CUT_PROFILE_SIM_T = 7.0
CUT_LEAD_SIM_T = 14.0

PHASE_WIDE = "straight_wide"
PHASE_PROFILE = "straight_profile"
PHASE_LEAD = "straight_lead"


def straight_phase_at(sim_t: float) -> str:
    t = float(sim_t)
    if t < CUT_PROFILE_SIM_T:
        return PHASE_WIDE
    if t < CUT_LEAD_SIM_T:
        return PHASE_PROFILE
    return PHASE_LEAD


class StraightDriveCameraDirector:
    """Cuts: straight_wide, straight_profile, straight_lead."""

    def __init__(self) -> None:
        self._phase = PHASE_WIDE

    @property
    def phase(self) -> str:
        return str(self._phase)

    def advance(self, vehicle_tf: carla.Transform, sim_t: float) -> tuple[carla.Transform, float]:
        del vehicle_tf  # offsets are vehicle-relative presets only
        self._phase = straight_phase_at(float(sim_t))
        return cfs.record_camera_spec(self._phase)
