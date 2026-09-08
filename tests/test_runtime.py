from __future__ import annotations

from pathlib import Path

import numpy as np

from carla_bridge import R0_RIGHT_MEAN, RIGHT_PUSH_SCALE_W, get_steer
from fly_brain import FlyBrain

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "data" / "models" / "fly_pilot_production_v1.csv"
GRID = REPO / "data" / "mappings" / "fly_eye_grid_full.csv"
BRAKE = REPO / "data" / "weights" / "fly_brake_weights.csv"
THROTTLE = REPO / "data" / "weights" / "fly_throttle_weights.csv"


def test_production_circuit_one_frame() -> None:
    brain = FlyBrain(WEIGHTS, GRID, BRAKE, THROTTLE)
    right_push, left_push, brake_drive, throttle_drive = brain.process_frame_with_longitudinal(
        np.full((30, 30), 180, dtype=np.uint8)
    )
    assert np.isfinite(right_push) and np.isfinite(left_push)
    assert 0.0 <= brake_drive <= 1.0
    assert 0.0 <= throttle_drive <= 1.0
    assert -1.0 <= get_steer(right_push, left_push) <= 1.0


def test_r0_zeros_straight_bias() -> None:
    w = RIGHT_PUSH_SCALE_W
    right_push = 10.0
    left_push = w * right_push - (w - 1.0) * R0_RIGHT_MEAN
    assert get_steer(right_push, left_push) == 0.0
