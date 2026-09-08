# Connectome pipeline

How Fly-By-Wire weights are produced from FlyWire, and what this repository ships.

## Overview

```
CAVE auth → graph walk → 3D positions → retinotopy → synapse export → calibration → production CSV
```

| Stage | In this repo |
|-------|----------------|
| Power-neuron query | `scripts/fetch_power_neurons.py` |
| Brake/throttle export | `scripts/export_brake_synapses.py` |
| Retinotopy grid | Shipped: `data/mappings/fly_eye_grid_full.csv` |
| Steering, calibration, pair search | Final CSVs only (`data/models/`, `data/weights/`) |

See [`../data/models/MANIFEST.md`](../data/models/MANIFEST.md).

## Prerequisites

1. **FlyWire CAVE account** — [flywire.ai](https://flywire.ai/)
2. **Auth token** — `data/secrets/flywire_token.json` (gitignored) or `FLYWIRE_TOKEN_FILE`
3. **Python deps** — `pip install -r requirements.txt` (includes `caveclient`)

```powershell
python scripts/fetch_power_neurons.py --verify-tables
```

## Public export scripts

### `scripts/fetch_power_neurons.py`

Queries `hierarchical_neuron_annotations` for GF / DNg01 / DNg02 / DNg03 power neurons on `flywire_fafb_public`. Uses the latest materialization version unless pinned elsewhere.

Token resolution:

1. `FLYWIRE_TOKEN_FILE`
2. `data/secrets/flywire_token.json`
3. Default CAVE auth config

### `scripts/export_brake_synapses.py`

1. Fetch DNg03 (and related power types) post IDs
2. `synapse_query(post_ids=...)` on `synapses_nt_v1`
3. Write `fly_brake_weights.csv` (move to `data/weights/` after export)

Reports overlap between upstream presynaptic IDs and `fly_eye_grid_full.csv`. Throttle weights (`fly_throttle_weights.csv`) follow the same pattern; the bundled file was exported at materialization v783.

## Steering circuit (method)

The bundled steering CSV is the end of a local pipeline that is not shipped here. The steps were:

1. **Export** — Walk upstream from descending motor neurons; fetch 3D coordinates; export connections with neurotransmitter columns. Materialization **783**.
2. **Retinotopy** — Min-max normalize `(x, y)` and quantize to 30×30. Promoted file: `data/mappings/fly_eye_grid_full.csv`.
3. **Steering pair search** — Score differential push on straight-road video; select RIGHT/LEFT target IDs.
4. **Calibration** — Structural graph edits, not gradient descent: prune straight-road bias synapses, mirror for binocular symmetry, inject command-layer gain.

Promoted artifacts:

```
data/models/fly_pilot_production_v1.csv
data/mappings/fly_eye_grid_full.csv
data/weights/fly_brake_weights.csv
data/weights/fly_throttle_weights.csv
```

Update [`data/models/MANIFEST.md`](../data/models/MANIFEST.md) if neuron IDs or materialization version change.

## Offline validation

```powershell
python src/neural_projector.py --video path/to.mp4
```

Uses the same 30×30 `INTER_AREA` downsampling as live `vision_ingest.py`.

## Weight CSV schema

| Column | Meaning |
|--------|---------|
| `pre_pt_root_id` | Presynaptic neuron (FlyWire root ID) |
| `post_pt_root_id` | Postsynaptic neuron |
| `synapse_count` | Anatomical connection strength |
| `nt_type` | Dominant neurotransmitter (steering export) |

Retinotopy map (`fly_eye_grid_full.csv`):

| Column | Meaning |
|--------|---------|
| `pt_root_id` | Neuron ID |
| `grid_x`, `grid_y` | 0..29 cell on 30×30 grid |

## FlyWire terms

This repo ships **derived** connectome weights from FlyWire. It does not redistribute raw FlyWire dumps. Regeneration requires your own CAVE credentials.

See also [`CONTRIBUTING.md`](../CONTRIBUTING.md).
