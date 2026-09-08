# Architecture

Fly-By-Wire maps a proofread *Drosophila* connectome (FlyWire) to live vehicle control in CARLA. Inference is **pure weighted geometric projection** — brightness at retinotopic coordinates × synapse counts — with no neural network training at runtime.

**Thesis:** geometry is the precursor to intelligence.

---

## End-to-end pipeline

```
FlyWire (CAVE) → 30×30 artificial retina → fly_brain → carla_bridge → CARLA vehicle
```

| Stage | Module / artifact | Role |
|-------|-------------------|------|
| Connectome export | `scripts/` | Trace upstream from descending neurons; export synapse weights |
| Retinotopy | `data/mappings/fly_eye_grid_full.csv` | Map ~2,244 neuron 3D positions → 30×30 grid |
| Vision ingest | `src/vision_ingest.py` | CARLA RGB → grayscale 30×30 (`INTER_AREA`) |
| Forward model | `src/fly_brain.py` | Sparse inner products → push / brake / throttle drives |
| Control bridge | `src/carla_bridge.py` | Push → steer; looming + neuron channels → brake/throttle |
| Closed loop | `src/carla_pilot.py` | Camera → grid → FlyBrain → `VehicleControl` |

See [`connectome_pipeline.md`](connectome_pipeline.md) for weight regeneration and [`honest_limits.md`](honest_limits.md) for engineered calibration boundaries.

---

## Phase 0 — FlyWire / CAVE

- **Datastack:** `flywire_fafb_public`
- **Materialization:** v783 (bundled weights were exported at this version)
- **Client:** `CAVEclient` with token at `data/secrets/flywire_token.json` or `FLYWIRE_TOKEN_FILE`
- **Annotations:** `hierarchical_neuron_annotations` (public stack; PNI tables are not materialized here)

Public scripts: `scripts/fetch_power_neurons.py`, `scripts/export_brake_synapses.py`. The bundled steering CSV is the final derived circuit; the full export pipeline is not shipped.

---

## Phase 1 — Topology tracing

Graph-geometry discovery walks **upstream in synapse space** from descending motor neurons:

1. **Anchor** — DNx02 cell type → `pt_root_id` list
2. **Hop 1** — Strong upstream onto DNx02 (synapse count > 5) → supervisors
3. **Hop 2** — Strong upstream onto supervisors → visual layer
4. **Hop 3** — Upstream onto visual (threshold > 3) → motion layer (T4/T5-like population)
5. **Geometry** — Query `proofread_neurons` for `(x, y, z)` of motion-layer IDs

The runtime model uses **where neurons sit in 3D space**, not learned embeddings.

---

## Phase 2 — Retinotopy

```
Min-max normalize x,y from connectome coordinates → [0, 1]
Quantize to 30×30 integer grid (0..29)
Save pt_root_id, grid_x, grid_y
```

**Runtime:** `brightness_at_neuron_i = grid[grid_y[i], grid_x[i]]`

Each mapped neuron sees one pixel of a downsampled world. Live CARLA frames use the same downsampling as offline validation (`neural_projector.py`).

**Shipped artifact:** `data/mappings/fly_eye_grid_full.csv` (~2,244 neurons).

---

## Phase 3 — Synaptic weight export

For all discovered brain IDs, pull `(pre, post, n_syn, NT scores)` and collapse NT argmax → `nt_type`.

**Computation:**

```
motor_signal = Σ pixel_brightness(pre_location) × synapse_count(pre → target)
```

**Shipped artifact:** `data/models/fly_pilot_production_v1.csv` — the only steering weight matrix in the public repo.

---

## Phase 4 — Steering pair selection

Two descending neurons drive lateral control:

| Channel | `pt_root_id` |
|---------|--------------|
| RIGHT steer | `720575940633816986` |
| LEFT steer | `720575940625653029` |

Discovery enumerates DNx02 pairs and scores differential push on straight-road calibration footage. Production fan-in is **extremely sparse**: ~3 mapped synapses per side after retinotopy filtering.

---

## Phase 5 — Calibration (geometry edits)

The bundled weights are not a raw connectome dump. Structural edits on the graph (not gradient descent):

1. Prune synapses into LEFT while straight-driving pixels are active
2. Mirror `grid_x` and duplicate LEFT pathway to RIGHT for binocular symmetry
3. Boost synapses into the command layer for usable drive
4. Apply tiny left-side gain → `fly_pilot_production_v1.csv`

Bridge constants (R₀ offset, gain, deadzone) in `carla_bridge.py` are separate engineered tuning for CARLA — see [`honest_limits.md`](honest_limits.md).

---

## Phase 6 — FlyBrain runtime

**Input:** `(30, 30)` uint8 grayscale grid  
**Module:** `src/fly_brain.py`

### Steering

```
right_push = Σ b[i] × w_r[i]
left_push  = Σ b[i] × w_l[i]
```

Bridge (`src/carla_bridge.py`):

```
bias  = 1.01 × right_push − left_push − 0.01 × R0
steer = clip(bias × 2.0, deadzone, [-1, 1])
```

Default constants: `R0 = 6.6414`, `GAIN = 2.0`, `STEER_DEADZONE = 0.002`.

### Brake (DNg03 upstream)

- Source: `data/weights/fly_brake_weights.csv`
- ~47 mapped inputs in eye grid
- `brake_drive = 1 − weighted_mean_brightness` (dark = loom/brake)

### Throttle (DNg01/02 upstream)

- Source: `data/weights/fly_throttle_weights.csv`
- ~275 mapped inputs in eye grid
- `throttle_drive = weighted_mean_brightness × motion_flux`

---

## Phase 7 — Vision ingest

**Module:** `src/vision_ingest.py` (`RetinotopicIngest`)

```
CARLA BGRA → BGR → grayscale → resize(30, 30, INTER_AREA) → uint8 grid
```

Layout: `grid[y, x]`; flatten with `np.ravel(order="C")` for row-major 900-vector alignment.

---

## Phase 8 — Closed-loop CARLA pilot

**Module:** `src/carla_pilot.py`

```
Camera → 30×30 grid → FlyBrain → steer / brake / throttle → VehicleControl
```

### Longitudinal modes

| Mode | Brake | Throttle |
|------|-------|----------|
| Default | `max(looming, neuron_brake)` | Cruise / cut on brake |
| `--neuron-only-brake` | DNg03 only | Constant unless braking |
| `--use-neuron-throttle` | As above | Base × neuron × motion flux |
| `--pure-neural-longitudinal` | DNg03 only | Neuron throttle × flux only |

### Looming controller

`LoomingController` in `carla_bridge.py` is a **non-connectome geometric fallback** on the same 30×30 grid (center/approach ROI statistics). Distinguish from FlyWire-derived paths when presenting results.

---

## Compute budget

| Metric | Value |
|--------|-------|
| Model FLOPs/frame | ~1,555 |
| Model MFLOPs @ 30 Hz | ~0.047 |

Excludes CARLA rendering, camera ingest, looming controller, and Python overhead.

---

## Three-channel end state

One 30×30 geometric retina feeds three control channels:

| Channel | Source |
|---------|--------|
| Lateral | Connectome steering (RIGHT/LEFT descending targets) |
| Brake | DNg03 upstream geometry (darkness-biased) |
| Throttle | DNg01/02 upstream geometry (brightness-biased) × motion flux |

---

## Related docs

- [`overview.json`](overview.json) — structured project writeup
- [`connectome_pipeline.md`](connectome_pipeline.md) — weight regeneration workflow
- [`honest_limits.md`](honest_limits.md) — credibility boundaries
- [`CARLA.md`](CARLA.md) — simulator setup
- [`../data/models/MANIFEST.md`](../data/models/MANIFEST.md) — shipped circuit provenance
