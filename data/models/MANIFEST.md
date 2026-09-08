# Shipped connectome circuit

This repo ships one production steering weight matrix and its retinotopic grid. Demos load these files by default; no calibration lineage CSVs are bundled.

## Default artifacts

| File | Role |
|------|------|
| `fly_pilot_production_v1.csv` | Steering synapse graph (pre → post, count, neurotransmitter) |
| `../mappings/fly_eye_grid_full.csv` | 2,244-neuron retinotopic map (grid_x, grid_y on 30×30) |
| `../weights/fly_brake_weights.csv` | DNg03 upstream graph for looming brake |
| `../weights/fly_throttle_weights.csv` | DNg01/02 upstream graph for throttle |

## FlyWire provenance

- **Dataset:** FlyWire FAFB public (`flywire_fafb_public`)
- **Materialization version:** v783
- **Export:** Upstream synapse counts from proofread connectome via CAVE (local regen workflow)
- **Steering targets:** Descending neuron pair selected for differential push on straight-road geometry

| Channel | Neuron ID (`pt_root_id`) |
|---------|--------------------------|
| RIGHT steer | `720575940633816986` |
| LEFT steer | `720575940625653029` |

At runtime, `fly_brain.py` computes `right_push` and `left_push` as sparse inner products: brightness at each mapped grid cell × synapse count into the corresponding motor neuron. Six synapses total (three per side) survive retinotopic mapping.

## Calibration story (prose)

The bundled weights are not a raw FlyWire dump. They are the **final circuit** after a geometry-first calibration pipeline:

1. **Connectome export** — Trace upstream from DNx02 descending neurons; export all `(pre, post, synapse_count, nt_type)` edges.
2. **Retinotopy** — Map neuron 3D positions to a 30×30 grid (`fly_eye_grid_full.csv`); live CARLA frames use the same downsampling pipeline.
3. **Steering pair selection** — Enumerate candidate descending pairs; pick the pair with strongest differential activation on straight-road calibration footage.
4. **Geometric surgery** — Structural edits on the graph, not gradient descent: prune straight-road bias, mirror for binocular symmetry, inject command-layer gain, apply tiny left-side symmetry calibration → `fly_pilot_production_v1.csv`.

Bridge constants (R₀ offset, gain, deadzone) in `src/carla_bridge.py` are engineered for CARLA control — see `docs/honest_limits.md`.

## Regeneration

Full steering-weight regeneration is not in this repository. Partial CAVE examples: `scripts/fetch_power_neurons.py`, `scripts/export_brake_synapses.py`. You need your own FlyWire credentials.

## FlyWire terms

This repo ships **derived** connectome weights from FlyWire. It does not redistribute raw FlyWire dumps. Regeneration requires your own CAVE credentials.
