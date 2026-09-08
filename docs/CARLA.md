# CARLA (Fly-By-Wire)

CARLA is **not bundled** with this repo. Install [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16) separately and point Fly-By-Wire at it with **`CARLA_ROOT`**.

**Related docs:** [architecture.md](architecture.md) · [honest_limits.md](honest_limits.md) · [connectome_pipeline.md](connectome_pipeline.md)

## Install

1. Download and extract CARLA **0.9.16** (Windows: `CARLA_0.9.16.zip`).
2. Set the install path (PowerShell example):

```powershell
$env:CARLA_ROOT = "C:\path\to\CARLA_0.9.16"
pip install "$env:CARLA_ROOT\PythonAPI\carla\dist\carla-*-py3*.whl"
```

3. Optional: add `CARLA_ROOT` to your user environment so demos and `--launch` pick it up without re-exporting each session.

**Resolution order for `--launch` / `--sim-dir`:**

1. Explicit `--sim-dir /path/to/CARLA`
2. Environment variable **`CARLA_ROOT`**
3. Local file **`demos/carla_root.txt`** (one line; copy from `demos/carla_root.example.txt` or run `demos/Set_CARLA_Root.bat`)
4. Fallback: `./CARLA_0.9.16` under the repo root

Demos use `python src/carla_pilot.py --launch`, which reads the same defaults via `carla_fly_session.default_sim_dir()`.

## Default session (minimal OpenDRIVE straight)

- **Script:** `carla_fly_session.py` — by default calls **`client.generate_opendrive_world(xodr)`** with a built-in **1000 m straight**, **one 3.5 m driving lane**, no town geometry (minimal GPU load).
- **Packaged towns:** pass **`--map Town10HD_Opt`** (or any map from `--list-maps`) to use CARLA’s normal maps again.
- **Custom road network:** **`--xodr-file path/to/map.xodr`** instead of the built-in string.
- **Car:** prefers **`vehicle.mercedes.coupe_2020`**, then Tesla / Audi / etc. Override: **`--vehicle vehicle.tesla.model3`**
- **Simulator launch:** `python src/carla_fly_session.py --launch` (or `demos/01_Cruising_Record.bat`) starts **`CarlaUE4.exe`** under your CARLA install with **`-d3d12`**, `-quality-level=Medium`, and `-windowed -ResX=1920 -ResY=1080`. If you hit **“D3D device lost”** on DX11 builds, DX12 is the default; fall back with `--rhi d3d11` or `--rhi default`.

## Commands

CARLA already running:

```bat
python src/carla_fly_session.py
```

Launch simulator then connect:

```bat
python src/carla_fly_session.py --launch
```

Use a full town (requires packaged maps inside your CARLA install):

```bat
python src/carla_fly_session.py --map Town10HD_Opt
```

List packaged maps (server must be running):

```bat
python src/carla_fly_session.py --list-maps
```

Lower resolution / quality if VRAM errors return:

```bat
python src/carla_fly_session.py --launch --quality-level Low --sim-res 1280x720
```

## Artificial retina (30×30 @ 30 Hz)

- **`vision_ingest.py`** — `RetinotopicIngest` converts each CARLA RGB frame to **grayscale 30×30** with **`cv2.INTER_AREA`**, matching **`neural_projector.py`** on training video.
- **`carla_fly_session.py`** wires it in the camera listener: use **`model_feed.get_latest_brightness_01()`** (shape `(30,30)` float64 in `[0,1]`) or **`get_latest_grid_u8()`** from the main loop, or **`add_frame_callback(fn)`** where `fn(grid_u8, frame_id, sim_time)` runs on the sensor thread.
- Layout: **`grid[y, x]`**; flatten with **`np.ravel(order="C")`** for a length-**900** vector aligned with row-major 30×30.

## Camera (pygame RGB)

- Sensor runs at **30 Hz** via blueprint **`sensor_tick`** (1/30 s sim time between frames).
- The pygame loop also ticks at **30 Hz** so you usually get a **new** image each frame instead of duplicate frames at 60 Hz (which felt laggy).
- Motion-blur–style attributes are forced **off** when the blueprint supports them.

### Python vs C++ client

For this stack, **Python + `sensor_tick` + matched loop rate** is the usual fix for “sluggish” video. A **C++ CARLA client** can shave a little IPC overhead, but rendering and the **Unreal server** still dominate latency; moving to C++ is a large separate project (build CARLA from source, C++ API, no pygame). Stay on Python unless you already need a custom C++ pipeline.

## Packaged maps

**`--map Town10HD_Opt`** (and other towns) load worlds from your **external CARLA install**, not from this repository. OpenDRIVE presets (`straight`, `left_curve`, `loom_wall`) generate minimal worlds at runtime and do not need town assets on disk.

## Controls

Focus the **small pygame** window: **W/S** throttle/brake, **A/D** steer, **Space** handbrake, **R** reset + **re-snap** to lane, **ESC** quit. Spawn uses **lane waypoints** when possible; **vibration** applies only while you **hold** **W/S/A/D** (no random steer when going straight on throttle alone; coasting = no shake). Leaving bounds (or falling) **auto-resets** and re-snaps. Watch the **CARLA** window — spectator follows the car.

## Verify spawn in terminal

With CARLA running:

```bat
python src/carla_fly_session.py --probe-spawn
```

Prints world position, rotation, and waypoint (if any), then destroys the test vehicle and exits.

## OpenDRIVE note

Generated worlds often have **no** `get_spawn_points()`; the script tries a grid of transforms along **+X** with lateral offsets. If spawn fails, adjust **`spawn_transforms_for_world()`** in `carla_fly_session.py` for your `.xodr` frame.

## Path validation

```bat
python scripts/validate_project_paths.py
```

Checks repo layout and `--help` for core scripts. CARLA absence is a **warning**, not a failure — set **`CARLA_ROOT`** before running demos with **`--launch`**.
