# Honest limits

Fly-By-Wire is a **geometry-first connectome-to-control demo**, not a claim that a raw FlyWire export drives a car unchanged. This page states what is anatomical, what is engineered, and what is intentionally simplified.

---

## What is real connectome geometry

- **Synapse counts** in bundled CSVs come from FlyWire materialization v783 via CAVE — not learned weights.
- **Retinotopy** binds neuron IDs to grid cells from proofread 3D positions (`fly_eye_grid_full.csv`).
- **Steering, brake, and throttle graphs** are upstream traces from identified descending / power neuron types (DNx02, DNg01/02/03).
- **Runtime inference** is a sparse dot product: brightness × synapse count. No transformer, no RL policy, no backprop at demo time (~1,555 FLOPs/frame).

---

## Sparse steering fan-in

After retinotopic mapping, **six synapses total** (three per side) contribute to RIGHT and LEFT push in the shipped production circuit. The full exported graph is large; most edges do not land on mapped grid cells.

**Implication:** Lateral control is driven by a tiny subset of the traced circuit. Behavior is real geometry on that subset, not a dense population code.

---

## Calibration is structural surgery, not training

The public repo ships **`fly_pilot_production_v1.csv`** — the final circuit after local calibration, not the first raw export. Steps included:

1. **Pruning** — Remove synapses that fire on straight-road pixels (bias reduction).
2. **Mirror-eye** — Duplicate LEFT pathway to RIGHT with mirrored `grid_x` for binocular symmetry.
3. **Gain injection** — Boost command-layer synapses so pushes are usable in CARLA.
4. **Symmetry calibration** — Small left-side gain to zero straight-road steer bias.

These edits modify **which edges exist and their counts** in the exported graph. They are justified as geometry-first tuning, but they are **not** “connectome as frozen at export.”

---

## Engineered bridge constants (R₀, gain, deadzone)

`src/carla_bridge.py` applies CARLA-specific scaling on top of raw pushes:

| Constant | Default | Role |
|----------|---------|------|
| `RIGHT_PUSH_SCALE_W` | 1.01 | Mild right emphasis with straight-road mean preserved |
| `R0_RIGHT_MEAN` | 6.6414 | Offset so straight plateau → zero bias |
| `GAIN` | 2.0 | Maps bias to steer command |
| `STEER_DEADZONE` | 0.002 | Suppresses calibration noise on straight runs |

`R0` was measured on straight-road calibration footage; CARLA live pushes sit at a different scale than offline video validation (`neural_projector.py` uses its own R₀ defaults). Re-tune with `python src/carla_pilot.py --r0 ...` from your logs if you regenerate weights.

---

## Partial eye-map overlap (brake / throttle)

Brake and throttle exports include upstream neurons **beyond** the 2,244-neuron retinotopy map:

| Channel | Mapped inputs in eye grid | Notes |
|---------|---------------------------|-------|
| Brake (DNg03) | ~47 | Many upstream IDs have no grid cell |
| Throttle (DNg01/02) | ~275 | Larger overlap, still incomplete |

Unmapped presynaptic neurons are skipped at runtime. Longitudinal channels therefore use a **partial** view of the exported graph.

---

## Looming heuristics are not FlyWire

Default longitudinal control fuses:

- **Neuron brake/throttle** — connectome-derived weighted brightness
- **Looming controller** — engineered ROI statistics on the 30×30 grid (center dark fraction, approach-zone mean, closure vs peak, buffered hysteresis)

`LoomingController` constants (`LOOM_DARK_THRESHOLD_U8`, approach ROI, steer damping while braking, etc.) are **hand-tuned for CARLA scenes**, not traced from FlyWire.

Use `--pure-neural-longitudinal` (demo 05) to run brake/throttle from connectome channels only and compare behavior.

---

## Steering vs looming interaction

Fly steering can aim a looming obstacle out of a small center ROI. The bridge **damps or zeroes steer** while hazard/brake is high so the vehicle stops instead of dodging. That is intentional safety geometry, not connectome output.

---

## OpenDRIVE worlds vs packaged towns

Demos default to **minimal generated OpenDRIVE** (1000 m straight, curves, loom wall) for low GPU load. Packaged CARLA towns behave differently (lighting, spawn, collision envelopes). A “physics stop” with brake=0 may be a **static prop collider**, not controller failure — see debug notes in [`overview.json`](overview.json).

---

## What we do not claim

- That every behavior channel is pure unedited connectome
- That sparse fan-in represents full fly visuomotor policy
- That CARLA retinotopy matches biological optic lobe geometry
- That bundled weights generalize across FlyWire materialization versions without re-export

---

## FlyWire data terms

This repo ships **derived** connectome weights generated from FlyWire. It does not redistribute raw FlyWire dumps. Regeneration requires your own CAVE credentials — see [`CONTRIBUTING.md`](../CONTRIBUTING.md).
