"""Instant-cut chase camerawork (no blends) for demo recordings."""

from __future__ import annotations

import carla

import carla_fly_session as cfs


class JumpCutChaseDirector:
    """Hard-cut chase presets on a sim-time grid. Defaults to CHASE_CYCLE_PRESETS."""

    def __init__(
        self,
        presets: tuple[str, ...] = cfs.CHASE_CYCLE_PRESETS,
        *,
        start: str | None = None,
        cut_interval_sec: float = 8.0,
    ) -> None:
        names = [p for p in presets if p in cfs.RECORD_CAMERA_PRESETS]
        if not names:
            names = list(cfs.CHASE_CYCLE_PRESETS)
        start_key = str(start or names[0]).strip().lower()
        if start_key in names:
            idx = names.index(start_key)
            names = names[idx:] + names[:idx]
        self._names = tuple(names)
        self._cut_interval = max(0.5, float(cut_interval_sec))
        self._phase = self._names[0]

    @property
    def phase(self) -> str:
        return str(self._phase)

    def _phase_at(self, sim_t: float) -> str:
        idx = min(int(float(sim_t) // self._cut_interval), len(self._names) - 1)
        return str(self._names[max(0, idx)])

    def advance(self, vehicle_tf: carla.Transform, sim_t: float) -> tuple[carla.Transform, float]:
        del vehicle_tf
        self._phase = self._phase_at(float(sim_t))
        return cfs.record_camera_spec(self._phase)
