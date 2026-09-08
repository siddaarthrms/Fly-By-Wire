#!/usr/bin/env python3
"""
CARLA live pilot: sensor -> 30x30 gray -> fly_brain.process_frame -> carla_bridge -> vehicle.

Uses ``carla_bridge.get_steer`` (mild right bias ``w=1.01``, R0 offset, gain~2), a rolling
average of raw steer over the last N frames (default 25), and
``carla_bridge.get_longitudinal_control`` (center-grid looming → brake/throttle).

Example:
  python carla_pilot.py --launch
  python carla_pilot.py   # sim already running (1000 m straight OpenDRIVE)
  python carla_pilot.py --opendrive-preset left_curve   # arc + cone
  python carla_pilot.py --opendrive-preset straight_cone  # straight + cone dodge
  python carla_pilot.py --opendrive-preset loom_wall    # short straight + wall (loom test)
"""
from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import carla
import numpy as np

import carla_bridge
import carla_fly_session as cfs
from cone_dodge_director import ConeDodgeCameraDirector
from jump_cut_director import JumpCutChaseDirector
from straight_drive_director import StraightDriveCameraDirector
from demo_recorder import ChaseVideoRecorder, default_recording_path
from demo_composite import driving_panel_title
from demo_telemetry import DemoTelemetryRecorder, default_telemetry_run_dir
from fly_brain import FlyBrain
from vision_ingest import RetinotopicIngest

# Repo root when running as ``python src\carla_pilot.py`` from project root (same as cfs.repo_root()).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _camera_only_callback(model_feed: RetinotopicIngest):
    def _on(image: carla.Image) -> None:
        try:
            model_feed.ingest_carla_image(image)
        except Exception:
            pass

    return _on


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("-p", "--port", type=int, default=2000)
    ap.add_argument("--map", default=None, help="Packaged map; omit for built-in OpenDRIVE")
    ap.add_argument("--xodr-file", type=Path, default=None)
    ap.add_argument(
        "--opendrive-preset",
        choices=("straight", "straight_cone", "left_curve", "loom_wall"),
        default="straight",
        help="Built-in OpenDRIVE when --map is omitted (ignored if --xodr-file is set).",
    )
    ap.add_argument(
        "--no-mid-road-obstacle",
        action="store_true",
        help="With curve/loom_wall/straight_cone presets, skip cone / wall spawn.",
    )
    ap.add_argument("--vehicle", default=None, help="CARLA vehicle blueprint id (default: session preference list)")
    ap.add_argument(
        "--vehicle-color",
        default=None,
        help='Paint color as CARLA "R,G,B" string (default: vivid demo palette)',
    )
    ap.add_argument(
        "--weather",
        default="clear",
        choices=("clear", "sunset", "cloudy", "wet"),
        help="Recorded in demo telemetry metadata only; sim world always uses ClearNoon.",
    )
    ap.add_argument(
        "--record-camera",
        default="chase_low",
        choices=tuple(cfs.RECORD_CAMERA_PRESETS.keys()),
        help="Chase-camera rig for MP4 + spectator follow (default: chase_low)",
    )
    ap.add_argument(
        "--chase-dynamic",
        action="store_true",
        help="Cycle chase angles (low / rear 3/4 / side / oncoming) with smooth blends for demo recordings",
    )
    ap.add_argument(
        "--shake-log",
        type=Path,
        default=None,
        metavar="CSV",
        help="Log per-tick vehicle vs chase motion (stale_lag_m diagnoses camera shake)",
    )
    ap.add_argument("--launch", action="store_true")
    ap.add_argument(
        "--sim-dir",
        type=Path,
        default=None,
        help="CARLA install folder (default: $CARLA_ROOT or ./CARLA_0.9.16)",
    )
    ap.add_argument("--quality-level", default="Medium")
    ap.add_argument("--sim-res", default="1920x1080")
    ap.add_argument(
        "--rhi",
        choices=("d3d12", "d3d11", "default"),
        default="d3d12",
        help="Unreal graphics API when using --launch (default d3d12)",
    )
    ap.add_argument("--no-dx11", action="store_true", help="Deprecated: same as --rhi default")
    ap.add_argument(
        "--weights",
        type=Path,
        default=_REPO_ROOT / "data" / "models" / "fly_pilot_production_v1.csv",
        help="Synapse weights CSV (default: data/models/fly_pilot_production_v1.csv)",
    )
    ap.add_argument(
        "--pixel-map",
        type=Path,
        default=_REPO_ROOT / "data" / "mappings" / "fly_eye_grid_full.csv",
        help="neuron grid map CSV (default: data/mappings/fly_eye_grid_full.csv)",
    )
    ap.add_argument(
        "--brake-weights",
        type=Path,
        default=_REPO_ROOT / "data" / "weights" / "fly_brake_weights.csv",
        help="DNg03 upstream synapse CSV (default: data/weights/fly_brake_weights.csv)",
    )
    ap.add_argument(
        "--throttle-weights",
        type=Path,
        default=_REPO_ROOT / "data" / "weights" / "fly_throttle_weights.csv",
        help="DNg01/02 upstream synapse CSV (default: data/weights/fly_throttle_weights.csv)",
    )
    ap.add_argument(
        "--brake-drive-onset",
        type=float,
        default=0.10,
        help="(Legacy absolute) brake-drive onset from neuron signal in [0,1] (default 0.10)",
    )
    ap.add_argument(
        "--brake-drive-full",
        type=float,
        default=0.25,
        help="(Legacy absolute) brake-drive value that maps to full brake (default 0.25)",
    )
    ap.add_argument(
        "--brake-delta-onset",
        type=float,
        default=0.02,
        help="Brake onset from (neuron_drive - startup_baseline) (default 0.02)",
    )
    ap.add_argument(
        "--brake-delta-full",
        type=float,
        default=0.08,
        help="Delta from baseline that maps to full neuron brake (default 0.08)",
    )
    ap.add_argument(
        "--brake-baseline-warmup",
        type=int,
        default=90,
        help="Frames used to estimate neuron baseline before enabling neuron brake (default 90)",
    )
    ap.add_argument(
        "--brake-baseline-alpha",
        type=float,
        default=0.002,
        help="Slow EMA rate for post-warmup brake baseline adaptation (default 0.002)",
    )
    ap.add_argument(
        "--neuron-only-brake",
        action="store_true",
        help=(
            "Longitudinal control from DNg03 path only: brake=neuron (no max with looming); "
            "throttle cut only when neuron brake fires. Looming still runs for logs (b_loom). "
            "Steer hazard damping uses neuron brake so coupling is not rescued by vision."
        ),
    )
    ap.add_argument(
        "--use-neuron-throttle",
        action="store_true",
        help=(
            "Use DNg01/02 pathway as throttle drive and modulate it with global motion flux "
            "(carla_bridge.get_adaptive_throttle)."
        ),
    )
    ap.add_argument(
        "--pure-neural-longitudinal",
        action="store_true",
        help=(
            "Fully neural longitudinal mode: brake=neuron only and "
            "throttle=(neuron_throttle*motion_flux), ignoring looming longitudinal outputs."
        ),
    )
    ap.add_argument(
        "--throttle-drive-onset",
        type=float,
        default=0.30,
        help="Throttle onset from DNg01/02 drive in [0,1] (default 0.30)",
    )
    ap.add_argument(
        "--throttle-drive-full",
        type=float,
        default=0.70,
        help="Throttle full-scale from DNg01/02 drive in [0,1] (default 0.70)",
    )
    ap.add_argument("--seed", type=int, default=7, help="Random seed for reproducible spawn/color choices")
    ap.add_argument("--throttle", type=float, default=0.3, help="Constant throttle (Fly speed)")
    ap.add_argument("--cam-fps", type=float, default=30.0)
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--cam-h", type=int, default=360)
    ap.add_argument(
        "--record-video",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Record chase-camera MP4 while driving. PATH optional (auto -> demos/recordings/<preset>_<time>.mp4). "
            "Use with --record-max-seconds for timed demo runs."
        ),
    )
    ap.add_argument(
        "--record-dir",
        type=Path,
        default=_REPO_ROOT / "demos" / "recordings",
        help="Output folder when --record-video is used without an explicit path",
    )
    ap.add_argument(
        "--record-max-seconds",
        type=float,
        default=0.0,
        help="Auto-stop pilot after N seconds (0 = run until Ctrl+C). Useful for demo .bats.",
    )
    ap.add_argument(
        "--no-bounds-reset",
        action="store_true",
        help="Do not teleport back to spawn when leaving the road (demo recordings of natural failures).",
    )
    ap.add_argument("--record-w", type=int, default=1280, help="Chase camera / MP4 width (default 1280)")
    ap.add_argument("--record-h", type=int, default=720, help="Chase camera / MP4 height (default 720)")
    ap.add_argument(
        "--save-telemetry",
        nargs="?",
        const="auto",
        default=None,
        metavar="DIR",
        help=(
            "Save per-frame circuit telemetry for light-up visuals. DIR optional "
            "(auto -> demos/telemetry/<preset>_<time>/). Enabled automatically with --record-video."
        ),
    )
    ap.add_argument(
        "--telemetry-dir",
        type=Path,
        default=_REPO_ROOT / "demos" / "telemetry",
        help="Parent folder for auto telemetry runs (default: demos/telemetry)",
    )
    ap.add_argument("--print-every", type=int, default=30, help="Print r/l/steer every N applies (0=off)")
    ap.add_argument(
        "--steer-smooth",
        type=int,
        default=None,
        metavar="N",
        help="Rolling average of last N steer samples before apply (default: 6 on curves, 25 straight)",
    )
    ap.add_argument(
        "--r0",
        type=float,
        default=None,
        metavar="MEAN",
        help=(
            "Override carla_bridge.R0_RIGHT_MEAN. For straight driving, use "
            "R0 ~= (w*r_mean - l_mean)/(w-1) from your plateau r_push/l_push logs "
            "(default matches ~14.43/14.51)."
        ),
    )
    ap.add_argument(
        "--steer-deadzone",
        type=float,
        default=None,
        metavar="D",
        help="Override carla_bridge.STEER_DEADZONE (|raw steer| below D -> 0 before clip)",
    )
    args = ap.parse_args()
    random.seed(int(args.seed))

    preset_key = str(args.opendrive_preset).lower()
    if args.steer_smooth is None:
        args.steer_smooth = 6 if preset_key == "left_curve" else 25

    if args.r0 is not None:
        carla_bridge.R0_RIGHT_MEAN = float(args.r0)
    if args.steer_deadzone is not None:
        carla_bridge.STEER_DEADZONE = float(args.steer_deadzone)

    if args.launch:
        sim_dir = cfs.resolve_sim_dir(args.sim_dir)
        print(f"Using CARLA install: {sim_dir.resolve()}", flush=True)
    elif args.sim_dir:
        sim_dir = Path(args.sim_dir)
    else:
        sim_dir = cfs.default_sim_dir()

    if args.launch:
        sw, sh = [int(x) for x in args.sim_res.split("x")]
        rhi = "default" if args.no_dx11 else args.rhi
        reuse = False
        if cfs.server_is_up(args.host, args.port) and cfs.probe_server_world(args.host, args.port):
            print(
                f"CARLA already running on {args.host}:{args.port} — reusing.",
                flush=True,
            )
            reuse = True
        elif cfs.carla_process_running() or cfs.server_is_up(args.host, args.port):
            print("Stopping stuck CARLA process before relaunch…", flush=True)
            cfs.terminate_carla_processes()
            time.sleep(3.0)
        if not reuse:
            print(f"Launching CARLA from {sim_dir.resolve()}...", flush=True)
            launch_quality = str(args.quality_level)
            if args.record_video is not None and launch_quality == "Medium":
                launch_quality = "Epic"
                print("Recording run: using Epic render quality for demo MP4.", flush=True)
            cfs.launch_simulator(
                sim_dir,
                rhi=rhi,
                quality=launch_quality,
                win_w=sw,
                win_h=sh,
            )
            if not cfs.wait_for_server_ready(
                args.host,
                args.port,
                timeout=300.0,
                initial_grace=8.0,
            ):
                print("Timed out waiting for CARLA to finish loading.", flush=True)
                return 1

    print("Connecting...", flush=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)

    try:
        cfs.restore_world_async(client.get_world())
    except Exception:
        pass

    map_label = "unknown"
    use_opendrive = not bool(args.map)
    max_odr_len: float | None = None
    if args.map:
        map_name = cfs.pick_map(client, args.map)
        map_label = map_name
        print(f"Loading map: {map_name}")
        client.load_world(map_name)
        world = client.get_world()
        cfs.restore_world_async(world)
    else:
        if args.xodr_file:
            if not args.xodr_file.is_file():
                print(f"Missing xodr: {args.xodr_file}")
                return 1
            xodr = args.xodr_file.read_text(encoding="utf-8")
            map_label = f"OpenDRIVE:{args.xodr_file.name}"
            max_odr_len = float(cfs.BUILTIN_OPENDRIVE_MAX_LENGTH_M)
        else:
            xodr, map_label, max_odr_len = cfs.builtin_opendrive_for_preset(args.opendrive_preset)
        world = cfs.load_opendrive_string(
            client,
            xodr,
            description=map_label,
            max_road_length=max_odr_len,
        )

    world.set_weather(carla.WeatherParameters.ClearNoon)

    bp = cfs.resolve_vehicle_bp(world, args.vehicle)
    bp.set_attribute("role_name", "hero")
    if bp.has_attribute("color") and bp.get_attribute("color").recommended_values:
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

    vehicle, spawn_ref_wp, attempt_tf = cfs.spawn_hero_vehicle(world, bp, prefer_road_start=use_opendrive)
    if not vehicle:
        print("Spawn failed.")
        return 1
    spawn_attempt_loc = attempt_tf.location if attempt_tf is not None else None

    spawn_transform = cfs.align_vehicle_to_road(
        vehicle,
        world,
        reference_wp=spawn_ref_wp,
        spawn_location=None if spawn_ref_wp is not None else spawn_attempt_loc,
        settle_ticks=4,
    )

    obstacle_actors: list[carla.Actor] = []
    if (
        use_opendrive
        and not args.xodr_file
        and preset_key == "straight_cone"
        and not args.no_mid_road_obstacle
    ):
        obstacle_actors = cfs.spawn_mid_road_obstacle_cluster(
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
        obstacle_actors = cfs.spawn_loom_test_wall(world)

    cfs.print_spawn_probe(vehicle, world, map_label)

    print(f"Loading FlyBrain: {args.weights} + {args.pixel_map}")
    brain = FlyBrain(
        args.weights,
        args.pixel_map,
        brake_weights_path=args.brake_weights,
        throttle_weights_path=args.throttle_weights,
    )
    print(
        f"DNg03 brake pathway: {brain.brake_input_count} mapped upstream input neurons "
        f"from {args.brake_weights}",
        flush=True,
    )
    print(
        f"DNg01/02 throttle pathway: {brain.throttle_input_count} mapped upstream input neurons "
        f"from {args.throttle_weights}",
        flush=True,
    )
    if bool(args.neuron_only_brake):
        print(
            "Longitudinal: neuron-only (--neuron-only-brake); looming still logged as b_loom.",
            flush=True,
        )
    if bool(args.use_neuron_throttle):
        print(
            "Throttle: neuron+flux mode (--use-neuron-throttle) enabled.",
            flush=True,
        )
    if bool(args.pure_neural_longitudinal):
        print(
            "Longitudinal: pure neural mode (--pure-neural-longitudinal) enabled.",
            flush=True,
        )

    cam_fps = float(args.cam_fps)
    cam_tick = 1.0 / cam_fps if cam_fps > 0 else 0.033333
    model_feed = RetinotopicIngest(target_fps=cam_fps, drop_duplicate_frame_id=True)
    rgb_camera = None
    chase_camera = None
    chase_rig: cfs.DynamicChaseCamera | None = None
    cone_director: ConeDodgeCameraDirector | None = None
    straight_director: StraightDriveCameraDirector | None = None
    curve_director: JumpCutChaseDirector | None = None
    video_recorder: ChaseVideoRecorder | None = None
    record_path: Path | None = None
    telemetry: DemoTelemetryRecorder | None = None
    preset_label = str(getattr(args, "opendrive_preset", "pilot") or "pilot")
    record_camera_preset = str(args.record_camera)

    telemetry_requested = args.save_telemetry is not None or args.record_video is not None
    if telemetry_requested:
        if args.save_telemetry is not None and str(args.save_telemetry).strip().lower() not in {"", "auto"}:
            telemetry_dir = Path(args.save_telemetry)
        else:
            telemetry_dir = default_telemetry_run_dir(preset_label, output_dir=args.telemetry_dir)
        telemetry_meta = {
            "label": preset_label,
            "opendrive_preset": preset_label,
            "map": map_label,
            "vehicle": str(args.vehicle),
            "vehicle_color": str(args.vehicle_color or "random_recommended"),
            "weather": str(args.weather),
            "record_camera": record_camera_preset,
            "weights": str(Path(args.weights).resolve()),
            "pixel_map": str(Path(args.pixel_map).resolve()),
            "use_neuron_throttle": bool(args.use_neuron_throttle),
            "pure_neural_longitudinal": bool(args.pure_neural_longitudinal),
            "neuron_only_brake": bool(args.neuron_only_brake),
        }
        telemetry = DemoTelemetryRecorder(telemetry_dir, meta=telemetry_meta)
        telemetry.write_topology(brain)
        print(f"Demo telemetry enabled -> {telemetry_dir.resolve()}", flush=True)

    if args.record_video is not None:
        if str(args.record_video).strip().lower() in {"", "auto"}:
            record_path = default_recording_path(preset_label, output_dir=args.record_dir)
        else:
            record_path = Path(args.record_video)
        if telemetry is not None:
            telemetry.attach_video_path(record_path)
        try:
            video_recorder = ChaseVideoRecorder(
                record_path,
                width=int(args.record_w),
                height=int(args.record_h),
                fps=cam_fps,
                composite=True,
                center_title=driving_panel_title(
                    preset_key,
                    pure_neural=bool(args.pure_neural_longitudinal),
                ),
            )
            print(f"Demo recording enabled -> {record_path.resolve()}", flush=True)
            print(
                "Composite layout: Artificial Retina | Driving video | Neuronal histogram "
                f"({video_recorder.width}x{video_recorder.height})",
                flush=True,
            )
            if telemetry is not None:
                telemetry._meta["composite_layout"] = [
                    "artificial_retina_30x30",
                    "driving_video",
                    "neuronal_histogram",
                ]
                telemetry._meta["composite_center_size"] = {
                    "w": int(args.record_w),
                    "h": int(args.record_h),
                }
        except Exception as ex:
            print(f"Demo recording failed to start: {ex}")
            return 1

    try:
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.cam_w))
        cam_bp.set_attribute("image_size_y", str(args.cam_h))
        cam_bp.set_attribute("fov", "90")
        try:
            cam_bp.set_attribute("sensor_tick", f"{cam_tick:.6f}")
        except Exception:
            pass
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.5))
        rgb_camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        rgb_camera.listen(_camera_only_callback(model_feed))

        if video_recorder is not None:
            rec_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            rec_bp.set_attribute("image_size_x", str(int(args.record_w)))
            rec_bp.set_attribute("image_size_y", str(int(args.record_h)))
            if preset_key == "straight_cone" and obstacle_actors:
                chase_tf, chase_fov = cfs.record_camera_spec("cone_high")
            elif preset_key == "straight" and bool(args.chase_dynamic):
                chase_tf, chase_fov = cfs.record_camera_spec("straight_wide")
            elif preset_key == "left_curve" and bool(args.chase_dynamic):
                chase_tf, chase_fov = cfs.record_camera_spec(record_camera_preset)
            else:
                chase_tf, chase_fov = cfs.record_camera_spec(record_camera_preset)
            rec_bp.set_attribute("fov", str(chase_fov))
            cfs.configure_rgb_camera_bp(
                rec_bp,
                disable_motion_blur=True,
                disable_postprocess=False,
                disable_temporal_aa=False,
            )
            try:
                rec_bp.set_attribute("sensor_tick", f"{cam_tick:.6f}")
            except Exception:
                pass
            try:
                chase_camera = world.spawn_actor(
                    rec_bp,
                    chase_tf,
                    attach_to=vehicle,
                    attachment_type=carla.AttachmentType.Rigid,
                )
            except TypeError:
                chase_camera = world.spawn_actor(rec_bp, chase_tf, attach_to=vehicle)

        if preset_key == "straight_cone" and obstacle_actors:
            cone_director = ConeDodgeCameraDirector(obstacle_actors[0].get_location())
            print(
                "Cone dodge camera: cone_high → [cut] cone_wheel → [prox] cone_bumper → [cut] cone_bird",
                flush=True,
            )
        elif preset_key == "straight" and bool(args.chase_dynamic):
            straight_director = StraightDriveCameraDirector()
            print(
                "Straight drive camera: straight_wide → [cut @7s] straight_profile → [cut @14s] straight_lead",
                flush=True,
            )
        elif preset_key == "left_curve" and bool(args.chase_dynamic):
            curve_director = JumpCutChaseDirector(start=record_camera_preset, cut_interval_sec=8.0)
            print(
                "Curve camera (jump cuts): "
                + " → ".join(curve_director._names)
                + " (cut every 8s sim)",
                flush=True,
            )
        elif bool(args.chase_dynamic):
            chase_rig = cfs.DynamicChaseCamera(start=record_camera_preset)
            print(
                "Dynamic chase camera: "
                + " → ".join(chase_rig._names)
                + f" (hold {chase_rig._hold_sec:.1f}s, blend {chase_rig._blend_sec:.1f}s)",
                flush=True,
            )

        sm = max(1, int(args.steer_smooth))
        print(
            f"Pilot loop: {args.cam_w}x{args.cam_h} @ ~{cam_fps} Hz | throttle={args.throttle} "
            f"| record_camera={record_camera_preset} "
            f"| steer smooth={sm} | bridge w={carla_bridge.RIGHT_PUSH_SCALE_W} "
            f"R0={carla_bridge.R0_RIGHT_MEAN} gain={carla_bridge.GAIN} "
            f"steer_dz={carla_bridge.STEER_DEADZONE} | Ctrl+C stop"
        )
        if float(args.record_max_seconds) > 0.0:
            print(f"Auto-stop after {float(args.record_max_seconds):.0f}s (demo timer).", flush=True)

        sync_dt = cfs.configure_world_sync(world, fps=cam_fps)
        print(
            f"Synchronous sim @ {cam_fps:.0f} Hz (fixed_delta_seconds={sync_dt:.6f})",
            flush=True,
        )
        for _ in range(5):
            vehicle.apply_control(carla.VehicleControl(throttle=float(args.throttle)))
            world.tick()
        if video_recorder is not None and chase_camera is not None:
            chase_camera.listen(video_recorder.on_image)
            video_recorder.begin_recording()
            print(
                f"Offline render capture: {int(args.record_w)}x{int(args.record_h)} PNG/tick "
                f"-> {record_path.resolve()}",
                flush=True,
            )
    except Exception as ex:
        print(f"Camera failed: {ex}")
        try:
            cfs.restore_world_async(world)
        except Exception:
            pass
        for obs in obstacle_actors:
            try:
                if obs.is_alive:
                    obs.destroy()
            except Exception:
                pass
        try:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass
        return 1

    bounds_mode = cfs.bounds_mode_for_args(
        argparse.Namespace(
            map=args.map,
            xodr_file=args.xodr_file,
            opendrive_preset=getattr(args, "opendrive_preset", "straight"),
        )
    )
    spawn_loc = carla.Location(
        spawn_transform.location.x,
        spawn_transform.location.y,
        spawn_transform.location.z,
    )

    frame_apply = 0
    sim_tick = 0
    sync_dt = 1.0 / float(cam_fps) if cam_fps > 0 else 0.033333
    t0 = time.monotonic()
    smooth_n = max(1, int(args.steer_smooth))
    steer_history: deque[float] | None = deque(maxlen=smooth_n) if smooth_n > 1 else None
    looming_ctrl = carla_bridge.LoomingController()
    brake_base_sum = 0.0
    brake_base_count = 0
    brake_base = 0.0
    brake_base_ready = False
    prev_grid: np.ndarray | None = None
    next_ctrl = carla.VehicleControl(throttle=float(args.throttle))
    shake_probe: cfs.TransformShakeProbe | None = (
        cfs.TransformShakeProbe() if args.shake_log is not None else None
    )
    # Sync loop: pose + control BEFORE tick; sensors read AFTER tick (zero-drop PNG capture).
    try:
        while True:
            sim_t = float(sim_tick) * float(sync_dt)

            if chase_camera is not None and chase_camera.is_alive:
                if cone_director is not None:
                    attach_tf, _fov = cone_director.advance(vehicle.get_transform(), sim_t)
                elif straight_director is not None:
                    attach_tf, _fov = straight_director.advance(vehicle.get_transform(), sim_t)
                elif curve_director is not None:
                    attach_tf, _fov = curve_director.advance(vehicle.get_transform(), sim_t)
                elif chase_rig is not None:
                    attach_tf, _fov = chase_rig.advance_attach(sim_t)
                else:
                    attach_tf, _ = cfs.record_camera_spec(record_camera_preset)
                chase_camera.set_transform(attach_tf)
                chase_tf = chase_camera.get_transform()
                world.get_spectator().set_transform(chase_tf)
            else:
                world.get_spectator().set_transform(
                    cfs.spectator_follow_transform(
                        vehicle.get_transform(), record_camera_preset
                    )
                )

            vehicle.apply_control(next_ctrl)
            record_before = video_recorder.frame_count if video_recorder is not None else 0
            chase_seq_before = video_recorder.chase_seq if video_recorder is not None else 0
            world.tick()
            sim_tick += 1

            if (
                video_recorder is not None
                and chase_camera is not None
                and not video_recorder.composite
            ):
                if not video_recorder.await_frame(record_before):
                    print("[record] WARNING: no chase PNG for this sim tick", flush=True)

            if shake_probe is not None and chase_camera is not None and chase_camera.is_alive:
                shake_probe.sample(
                    tick=frame_apply,
                    vehicle_tf=vehicle.get_transform(),
                    chase_tf=chase_camera.get_transform(),
                    attach_tf=attach_tf if chase_camera is not None else None,
                    chase_updated=True,
                    attach_mode="vehicle_rigid",
                )

            grid = model_feed.get_latest_grid_u8()
            if grid is None:
                next_ctrl = carla.VehicleControl(throttle=float(args.throttle))
                continue

            brake_loom, throttle_cmd = looming_ctrl.step(
                grid, cruise_throttle=float(args.throttle)
            )

            right_push, left_push, brake_drive, throttle_drive = brain.process_frame_with_longitudinal(grid)
            if not brake_base_ready:
                brake_base_sum += float(brake_drive)
                brake_base_count += 1
                if brake_base_count >= max(1, int(args.brake_baseline_warmup)):
                    brake_base = float(brake_base_sum / max(1, brake_base_count))
                    brake_base_ready = True
            elif float(brake_loom) < 0.02:
                # Track slow scene/illumination drift so curved-road view changes do not
                # slowly accumulate into false neuron-brake deltas.
                alpha = float(np.clip(float(args.brake_baseline_alpha), 0.0, 1.0))
                brake_base = float((1.0 - alpha) * float(brake_base) + alpha * float(brake_drive))
            brake_delta = float(max(0.0, float(brake_drive) - float(brake_base)))
            onset = float(args.brake_delta_onset)
            full = float(args.brake_delta_full)
            denom = max(1e-6, full - onset)
            brake_neuron = (
                float(np.clip((brake_delta - onset) / denom, 0.0, 1.0))
                if brake_base_ready
                else 0.0
            )
            pure_neural_longitudinal = bool(args.pure_neural_longitudinal)
            if pure_neural_longitudinal or bool(args.neuron_only_brake):
                brake = float(brake_neuron)
                throttle_cmd = 0.0 if brake > 0.02 else float(args.throttle)
                hazard_for_steer = float(brake_neuron)
            else:
                brake = max(float(brake_loom), brake_neuron)
                if brake > 0.02:
                    throttle_cmd = 0.0
                hazard_for_steer = float(looming_ctrl.last_signal)

            t_on = float(args.throttle_drive_onset)
            t_full = float(args.throttle_drive_full)
            t_denom = max(1e-6, t_full - t_on)
            throttle_neuron = float(np.clip((float(throttle_drive) - t_on) / t_denom, 0.0, 1.0))
            throttle_flux = float(carla_bridge.get_adaptive_throttle(grid, prev_grid))
            prev_grid = np.asarray(grid, dtype=np.uint8).copy()
            flux_factor = float(
                np.clip(
                    throttle_flux / max(1e-6, float(carla_bridge.MOTION_BASE_THROTTLE)),
                    0.0,
                    1.0,
                )
            )
            if pure_neural_longitudinal or bool(args.use_neuron_throttle):
                throttle_cmd = float(args.throttle) * throttle_neuron * flux_factor
                if brake > 0.02:
                    throttle_cmd = 0.0
            steer_raw = carla_bridge.get_steer(right_push, left_push)
            steer_cmd = carla_bridge.blend_steer_with_loom(
                steer_raw, brake, hazard_for_steer
            )

            # Smooth **post-loom** steer only. Averaging raw fly steer first kept ~N frames of
            # swerve “in the pipe”, so the car could dodge a full-lane wall with brake=0.
            if steer_history is not None:
                if brake > 0.02:
                    steer_history.clear()
                steer_history.append(float(steer_cmd))
                steer = sum(steer_history) / len(steer_history)
            else:
                steer = float(steer_cmd)

            ctrl = carla.VehicleControl()
            ctrl.steer = float(steer)
            ctrl.throttle = float(throttle_cmd)
            ctrl.brake = float(brake)
            next_ctrl = ctrl

            frame_apply += 1
            brain_diag = brain.frame_diagnostics(grid)
            if telemetry is not None:
                carla_fid, _sim_t, _ingest_n = model_feed.get_meta()
                telemetry.record_frame(
                    frame_idx=frame_apply,
                    t_sec=float(time.monotonic() - t0),
                    grid_u8=grid,
                    brain_diag=brain_diag,
                    scalars={
                        "carla_frame_id": int(carla_fid or 0),
                        "steer_raw": float(steer_raw),
                        "steer": float(steer),
                        "brake": float(brake),
                        "throttle": float(throttle_cmd),
                        "brake_neuron": float(brake_neuron),
                        "brake_loom": float(brake_loom),
                        "throttle_neuron": float(throttle_neuron),
                        "throttle_flux": float(throttle_flux),
                        "loom_signal": float(looming_ctrl.last_signal),
                    },
                )
            if video_recorder is not None and video_recorder.composite:
                if not video_recorder.await_chase(chase_seq_before):
                    print("[record] WARNING: no chase frame for composite tick", flush=True)
                elif not video_recorder.commit_composite(grid, brain_diag["neuron_brightness"]):
                    print("[record] WARNING: composite PNG not written for this sim tick", flush=True)
            if args.print_every and frame_apply % args.print_every == 0:
                dt = time.monotonic() - t0
                carla_fid, _sim_t, ingest_n = model_feed.get_meta()
                if steer_history is not None:
                    extra = f" steer_raw={steer_raw:+.4f}"
                else:
                    extra = ""
                print(
                    f"[pilot] n={frame_apply} t={dt:.1f}s "
                    f"r_push={right_push:.4f} l_push={left_push:.4f} steer={steer:+.4f}{extra} "
                    f"brake={brake:.2f} thr={throttle_cmd:.2f} "
                    f"b_drive={brake_drive:.3f} b_base={brake_base:.3f} b_d={brake_delta:.3f} "
                    f"b_neuron={brake_neuron:.2f} b_loom={brake_loom:.2f} "
                    f"t_drive={throttle_drive:.3f} t_neuron={throttle_neuron:.2f} t_flux={throttle_flux:.2f} "
                    f"loom={looming_ctrl.last_signal:.2f} "
                    f"(c={looming_ctrl.last_center:.2f} a={looming_ctrl.last_approach:.2f} "
                    f"al={looming_ctrl.last_approach_loose:.2f} m={looming_ctrl.last_mean_hazard:.2f} "
                    f"cl={looming_ctrl.last_closure:.2f} dip={looming_ctrl.last_center_dip:.2f}) "
                    f"d_exp={looming_ctrl.last_expansion:+.2f} d_jump={looming_ctrl.last_jump:+.2f} "
                    f"carla_fid={carla_fid} ingests={ingest_n}",
                    flush=True,
                )

            loc = vehicle.get_location()
            if not bool(args.no_bounds_reset) and cfs.is_out_of_bounds(
                loc, spawn_loc, bounds_mode
            ):
                if steer_history is not None:
                    steer_history.clear()
                looming_ctrl.reset()
                cfs.reset_vehicle_to_spawn(vehicle, spawn_transform)
                spawn_transform = cfs.align_vehicle_to_road(
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

            if float(args.record_max_seconds) > 0.0 and sim_t >= float(args.record_max_seconds):
                print(
                    f"\nDemo timer reached ({float(args.record_max_seconds):.0f}s sim time). Stopping.",
                    flush=True,
                )
                break

    except KeyboardInterrupt:
        print("\nPilot stopped.")
    finally:
        if shake_probe is not None and args.shake_log is not None:
            try:
                shake_probe.write_csv(args.shake_log)
                summary = cfs.TransformShakeProbe.summarize_csv(args.shake_log)
                if summary:
                    print(
                        "[shake] summary "
                        f"n={int(summary['n_ticks'])} "
                        f"stale_lag_mean={summary['stale_lag_mean_m']:.4f}m "
                        f"stale_lag_max={summary['stale_lag_max_m']:.4f}m "
                        f"chase_jump_rms={summary['chase_jump_rms_m']:.4f}m "
                        f"-> {args.shake_log}",
                        flush=True,
                    )
            except Exception as ex:
                print(f"[shake] log failed: {ex}", flush=True)
        try:
            cfs.restore_world_async(world)
        except Exception:
            pass
        try:
            if chase_camera is not None and chase_camera.is_alive:
                chase_camera.stop()
                chase_camera.destroy()
        except Exception:
            pass
        if video_recorder is not None:
            try:
                video_recorder.close()
            except Exception:
                pass
        if telemetry is not None:
            try:
                telemetry.close()
            except Exception:
                pass
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
