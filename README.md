# Fly-By-Wire

i made a fly drive a car.

Writeup: [siddaarth.com/Fly-By-Wire](https://siddaarth.com/Fly-By-Wire)

A fruit-fly circuit from FlyWire steers a car in CARLA. Not a trained net. Camera frames get crushed down to a 30x30 gray grid, and brightness at each mapped neuron, times that neuron's synapse count, becomes left or right push. Six synapses handle the steering. DNg03 brakes, DNg01/02 throttles. Roughly 1,555 FLOPs per frame.

I started at the descending motor neurons and walked upstream, kept the strong stuff (1,122 cells), mirrored to 2,244, and parked them on the grid. Default car is the Mercedes coupe.

This is not a raw connectome dump. I pruned synapses that kept steering on straight-road video, added a little left bias and 20x gain so the car actually moves, and bolted a looming detector onto the center 10x10 (brakes at ~8% blocked) because otherwise it swerves off the road. [docs/honest_limits.md](docs/honest_limits.md) has the unvarnished version.

## Run it

Python 3.10+, [CARLA 0.9.16](docs/CARLA.md), then:

```powershell
git clone https://github.com/siddaarthrms/Fly-By-Wire
cd Fly-By-Wire
pip install -r requirements.txt
set CARLA_ROOT=C:\path\to\CARLA_0.9.16
pip install %CARLA_ROOT%\PythonAPI\carla\dist\carla-*-py3*.whl
demos\01_Cruising_Record.bat
```

| What the site calls it | Double-click |
|------------------------|--------------|
| Cruising | `demos/01_Cruising_Record.bat` |
| Turning | `demos/02_Turning_Record.bat` |
| Obstacle Avoidance | `demos/03_ObstacleAvoidance_Record.bat` |
| Rapid Braking | `demos/04_RapidBraking_Record.bat` |

`05_BiologicalBraking_Record.bat` is the version without the looming hack. Linux: `demos/run_cruising.sh`.

MP4s land in `demos/recordings/`, telemetry in `demos/telemetry/`. `demos/Open.bat` opens both.

`python scripts/validate_project_paths.py` and `pytest tests/ -v` if you want a sanity check.

Weights are derived from FlyWire. I don't ship raw tables. Rebuilding them needs your own CAVE token, see [CONTRIBUTING.md](CONTRIBUTING.md).

```
src/      carla_pilot, fly_brain, bridge
scripts/  export helpers
tests/    loads the production CSVs, no sim
data/     grid + steering + brake/throttle weights
demos/    record bats
docs/     architecture, limits, CARLA
```

Dorkenwald et al., Nature 2024 (the connectome): https://doi.org/10.1038/s41586-024-07558-y

[CITATION.cff](CITATION.cff). MIT in [LICENSE](LICENSE). FlyWire terms: [flywire.ai](https://flywire.ai/).
