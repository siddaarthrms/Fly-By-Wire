#!/usr/bin/env python3
"""Check repo layout and paths used by the demo bats / carla_pilot. Does not start CARLA."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    errors: list[str] = []
    warnings: list[str] = []

    def need_file(p: Path, label: str) -> None:
        if not p.is_file():
            errors.append(f"missing {label}: {p}")

    def need_dir(p: Path, label: str) -> None:
        if not p.is_dir():
            errors.append(f"missing {label}: {p}")

    need_dir(repo, "repo root")
    need_file(repo / "src" / "carla_pilot.py", "carla_pilot.py")
    need_file(repo / "src" / "carla_fly_session.py", "carla_fly_session.py")
    need_file(repo / "data" / "models" / "fly_pilot_production_v1.csv", "default weights CSV")
    need_file(repo / "data" / "mappings" / "fly_eye_grid_full.csv", "default pixel map CSV")
    need_file(repo / "data" / "weights" / "fly_brake_weights.csv", "default brake weights CSV")
    need_file(repo / "data" / "weights" / "fly_throttle_weights.csv", "default throttle weights CSV")
    need_file(repo / "src" / "vision_ingest.py", "vision_ingest")
    need_file(repo / "src" / "demo_telemetry.py", "demo_telemetry.py")
    need_file(repo / "src" / "demo_recorder.py", "demo_recorder.py")
    need_file(repo / "scripts" / "render_circuit_lightup.py", "render_circuit_lightup.py")
    need_file(repo / "demos" / "lib" / "run_demo.bat", "demos/lib/run_demo.bat")
    need_file(repo / "demos" / "Open.bat", "demos/Open.bat")
    need_dir(repo / "demos" / "recordings", "demos/recordings")
    need_dir(repo / "demos" / "telemetry", "demos/telemetry")

    carla_root_env = os.environ.get("CARLA_ROOT", "").strip()
    carla_dir = Path(carla_root_env) if carla_root_env else repo / "CARLA_0.9.16"
    try:
        sys.path.insert(0, str(repo / "src"))
        import carla_fly_session as cfs  # noqa: WPS433

        carla_resolved = cfs.resolve_sim_dir()
    except Exception:
        carla_resolved = None
    if carla_resolved is not None:
        carla_dir = carla_resolved
    carla_exe = carla_dir / "CarlaUE4.exe"
    if not carla_exe.is_file():
        warnings.append(
            f"CARLA install not found (checked resolve_sim_dir + {carla_dir}); "
            "set CARLA_ROOT, demos/carla_root.txt, or see docs/CARLA.md"
        )

    session = (repo / "src" / "carla_fly_session.py").read_text(encoding="utf-8")
    if "def repo_root()" not in session or "Path(__file__).resolve().parent.parent" not in session:
        errors.append("carla_fly_session.py: repo_root() should resolve to project root (parent.parent)")

    pilot = (repo / "src" / "carla_pilot.py").read_text(encoding="utf-8")
    if 'data" / "models" / "fly_pilot_production_v1.csv"' not in pilot:
        errors.append("carla_pilot.py: default weights path should live under data/models/")
    if 'data" / "weights" / "fly_brake_weights.csv"' not in pilot:
        errors.append("carla_pilot.py: default brake weights path should live under data/weights/")
    if 'data" / "weights" / "fly_throttle_weights.csv"' not in pilot:
        errors.append("carla_pilot.py: default throttle weights path should live under data/weights/")
    if 'demos" / "telemetry"' not in pilot:
        errors.append("carla_pilot.py: default telemetry dir should live under demos/telemetry")

    demo_runner = repo / "demos" / "lib" / "run_demo.bat"
    if demo_runner.is_file():
        demo_text = demo_runner.read_text(encoding="utf-8", errors="replace")
        if not re.search(r'cd\s+/d\s+"%~dp0\.\.\\\.\."', demo_text):
            errors.append('demos/lib/run_demo.bat: expected cd /d "%~dp0..\\.." (repo root as cwd)')
        for needle in (
            r"demos\recordings",
            r"demos\telemetry",
            r"--record-video",
            r"python src\carla_pilot.py",
        ):
            if needle not in demo_text:
                errors.append(f"demos/lib/run_demo.bat: expected {needle!r}")

    open_bat = repo / "demos" / "Open.bat"
    if open_bat.is_file():
        open_text = open_bat.read_text(encoding="utf-8", errors="replace")
        for subdir in ("recordings", "telemetry"):
            if subdir not in open_text.replace("/", "\\"):
                errors.append(f"demos/Open.bat: expected to open demos/{subdir}")

    numbered_demos = {
        "demos/01_Cruising_Record.bat": "straight",
        "demos/02_Turning_Record.bat": "left_curve",
        "demos/03_ObstacleAvoidance_Record.bat": "straight_cone",
        "demos/04_RapidBraking_Record.bat": "loom_wall",
        "demos/05_BiologicalBraking_Record.bat": "loom_wall",
    }
    for rel, preset in numbered_demos.items():
        p = repo / rel.replace("/", os.sep)
        need_file(p, rel)
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "lib\\run_demo.bat" not in text and "lib/run_demo.bat" not in text:
            errors.append(f"{rel}: expected call to lib/run_demo.bat")
        if preset not in text:
            errors.append(f"{rel}: expected opendrive preset {preset!r}")

    if errors:
        print("validate_project_paths: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        for w in warnings:
            print(f"  WARN: {w}", file=sys.stderr)
        return 1

    subprocess_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # Import-heavy checks (CARLA egg must be available if PYTHONPATH is set like usual).
    print("Running: python src/carla_pilot.py --help ...")
    r = subprocess.run(
        [sys.executable, str(repo / "src" / "carla_pilot.py"), "--help"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=subprocess_env,
    )
    if r.returncode != 0:
        errors.append(f"carla_pilot.py --help exit {r.returncode}")
        if r.stderr:
            print(r.stderr, file=sys.stderr)
    else:
        out = (r.stdout or "") + (r.stderr or "")
        if "fly_pilot_production_v1.csv" not in out:
            warnings.append("carla_pilot --help output may not list default weights path (check argparse)")

    print("Running: python src/carla_fly_session.py --help ...")
    r2 = subprocess.run(
        [sys.executable, str(repo / "src" / "carla_fly_session.py"), "--help"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=subprocess_env,
    )
    if r2.returncode != 0:
        errors.append(f"carla_fly_session.py --help exit {r2.returncode}")
        if r2.stderr:
            print(r2.stderr, file=sys.stderr)

    if errors:
        print("validate_project_paths: FAILED (runtime checks)", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        for w in warnings:
            print(f"  WARN: {w}", file=sys.stderr)
        return 1

    print("validate_project_paths: OK")
    print(f"  repo root     : {repo}")
    print(f"  default CARLA : {carla_dir}" + (" (from CARLA_ROOT)" if carla_root_env else " (auto-resolved)"))
    print(f"  default weights: {repo / 'data' / 'models' / 'fly_pilot_production_v1.csv'}")
    print(f"  brake weights   : {repo / 'data' / 'weights' / 'fly_brake_weights.csv'}")
    print(f"  throttle weights: {repo / 'data' / 'weights' / 'fly_throttle_weights.csv'}")
    print(f"  demo recordings : {repo / 'demos' / 'recordings'}")
    print(f"  demo telemetry  : {repo / 'demos' / 'telemetry'}")
    for w in warnings:
        print(f"  WARN: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
