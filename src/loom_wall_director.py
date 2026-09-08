"""Jump-cut camerawork for the short loom_wall demo (50 m road, wall @ ~36%)."""

from __future__ import annotations

import carla_fly_session as cfs
from jump_cut_director import JumpCutChaseDirector

# ~14 s clip: cruise → wall loom/brake → stop at wall (matches 04/05 launchers).
LOOM_DEMO_RECORD_SECONDS = 14.0

# Sim-time hard cuts tuned for LOOM_TEST_ROAD_LENGTH_M=50, wall @ LOOM_WALL_ROAD_FRACTION.
LOOM_WALL_CUT_SIM_T = (3.5, 7.0, 10.0)


class LoomWallCameraDirector(JumpCutChaseDirector):
    """Cuts: chase_low, chase_oncoming, chase, chase_side."""

    def __init__(self, *, start: str | None = None) -> None:
        super().__init__(
            cfs.CHASE_CYCLE_PRESETS,
            start=start,
            cut_times=LOOM_WALL_CUT_SIM_T,
        )
