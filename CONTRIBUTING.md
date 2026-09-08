# Contributing

Thanks for your interest in Fly-By-Wire.

## What you can run without FlyWire

- CARLA demos with the bundled weights (`data/models/`, `data/mappings/`, `data/weights/`)
- Path check: `python scripts/validate_project_paths.py`
- Offline steering check: `python src/neural_projector.py --video path/to.mp4`
- Tests: `pytest tests/ -v`

No CAVE credentials required for the above.

## Regenerating connectome weights

Full steering-weight regeneration is not in this repository. This repo ships the final derived CSVs only.

Partial CAVE exports (power neurons, brake/throttle synapses) are in `scripts/`. You need your own FlyWire credentials:

```
data/secrets/flywire_token.json
```

or `FLYWIRE_TOKEN_FILE`. See [flywire.ai](https://flywire.ai/) for account and token setup.

```powershell
python scripts/fetch_power_neurons.py --verify-tables
python scripts/fetch_power_neurons.py
python scripts/export_brake_synapses.py
```

Move exported CSVs into `data/weights/` as needed. Method notes: [`docs/connectome_pipeline.md`](docs/connectome_pipeline.md).

## FlyWire data terms

This repo ships **derived** connectome weights from FlyWire. It does not redistribute raw FlyWire dumps. Regeneration requires your own CAVE credentials.

Do not commit:

- `data/secrets/*.json`
- Raw FlyWire table dumps
- Intermediate calibration CSVs

## Conventions

- Product name: Fly-By-Wire
- Public drive modes: Cruising, Turning, Obstacle Avoidance, Rapid Braking
- Runtime: `src/fly_brain.py` (30×30 retina → left/right sums)
- Paths: `CARLA_ROOT` and `FLYWIRE_TOKEN_FILE`; no machine-specific paths in committed files

## Documentation

When changing behavior or bundled data:

- [`data/models/MANIFEST.md`](data/models/MANIFEST.md) — circuit provenance
- [`docs/honest_limits.md`](docs/honest_limits.md) — calibration and engineered constants
- [`docs/overview.json`](docs/overview.json) — structured summary

## Pull requests

1. `python scripts/validate_project_paths.py`
2. `pip install -r requirements-dev.txt` then `ruff check src/ scripts/ tests/` and `pytest tests/ -v`
3. Confirm no tokens or personal machine paths in committed files
4. Note whether weights changed and whether FlyWire regeneration was required
