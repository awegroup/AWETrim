# AWETrim
AWETrim provides building blocks to assemble and simulate quasi-steady missions for airborne wind energy systems.  The package wraps the aerodynamic kite model, several tether formulations, wind-field descriptions, and a library of parametrised flight patterns that can be optimised or replayed over a phase grid.

Key modules exposed under `awetrim` include:

- `system`: `SystemModel`, `State`, and component models for the kite, tether, and winch.
- `environment`: wind models, e.g. `Wind` for uniform or logarithmic profiles.
- `kinematics`: parametrised trajectories (`Helix`, `Lissajous`, `FigureEight`) and helper factories.
- `timeseries`: integration helpers such as `PhaseParameterized` with plotting, energy metrics, and optimisation hooks.
- `utils`: plotting palettes, default parameter limits, and reference-frame utilities.

The repository also contains scripts for reproducing optimisation studies (see `scripts/optimize_path/reelout_cst.py`) and notebooks demonstrating typical workflows.

## Installation

```bash
git clone https://github.com/ocayon/quasi-steady-awes.git
cd quasi-steady-awes
python -m venv .venv
.\.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
```

Optional documentation and example dependencies are listed in `requirements.txt`.

## Quick Start

The snippet below loads the LEI-V3 kite aerodynamics, defines a uniform wind, builds the kite+tether system, and marches along a parametrised Lissajous path using the quasi-steady solver.  It mirrors the defaults used in `scripts/optimize_path/reelout_cst.py`.

```python
from pathlib import Path
import json
import numpy as np
from awetrim import SystemModel, State
from awetrim.environment import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase_parametrized import PhaseParameterized

# Aerodynamic input and wind setup
data_path = Path("data/LEI-V3-KITE/v3_aero_input.json")
aero_input = json.loads(data_path.read_text())
wind = Wind(wind_model="uniform", z0=0.1)
wind.speed_wind_ref = 10.0
wind.speed_friction = 0.41 * wind.speed_wind_ref / np.log(100 / wind.z0)

# Kite system
kite = Kite(
    mass_wing=14.2,
    area_wing=19.75,
    aero_input=aero_input,
    mass_kcu=10.0,
    steering_control="asymmetric",
)
tether = RigidLumpedTether(diameter=0.01)
system = SystemModel(dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind)
system.input_depower = 0.0

# Parametrised pattern (same defaults as reelout_cst.py)
pattern_config = {
    "pattern_type": "cst_lissajous",
    "path_parameters": {
        "omega": 1.0,
        "r0": 230.0,
        "az_amp0": 0.4815631965341702,
        "beta_amp0": 0.09875937127714636,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": [0.25485496, -0.99986137, 0.12645635, -0.86821607, 0.35302077],
        "az_coeffs": [0, 0, 0, 0, 0],
        "kbeta": 0.0,
        "beta0": 0.4414535012239937,
        "kappa": 0.0,
    },
    "radial_parameters": {
        "reeling_strategy": "force",
        "force_model": "quadratic",
        "reeling_speed": 0.0,
        "max_tether_force": 2e4,
        "min_tether_force": 2000.0,
        "softplus": True,
        "softplus_beta": 1e-4,
        "softminus": True,
        "softminus_beta": 1e-3,
        "slope": 2716.0,
        "offset": 0.0,
    },
    "start_time": 0.0,
    "end_time": 35.0,
    "start_angle": np.pi / 2,
    "end_angle": 2 * np.pi + np.pi / 2,
    "n_points": 600,
}

start_state = State(
    t=0.0,
    s=np.pi / 2,
    s_dot=2.0,
    length_tether=199.6,
    distance_radial=200.0,
    speed_radial=2.0,
    tension_tether_ground=1e4,
)

phase = PhaseParameterized(
    system,
    quasi_steady=True,
    pattern_config=pattern_config,
    tension_min=3000.0,
    tension_max=25000.0,
)
phase.run_simulation_phase(start_state=start_state)
phase.plot_overview_3d(
    label="Baseline",
    variables=["speed_tangential", "tension_tether_ground", "input_steering", "speed_radial"],
)
```

Call `phase.energy_metrics(reference_phase)` to compare two trajectories.  Additional traces are available through `phase.return_variable("<name>")` and can be plotted with Matplotlib or exported to disk.

## Path Optimisation

- `scripts/optimize_path/reelout_cst.py` demonstrates how to run a force-controlled reel-out study, optimise the Lissajous coefficients, and compare the baseline to the optimised solution using the overview plot and energy metrics.
- The notebook `parametrize_path/reelout_parametrization.ipynb` reproduces the parametrisation workflow interactively, walking through pattern configuration, simulation, and visualisation.

## Project Layout

- `src/awetrim/` – package source: system modelling, kinematics, timeseries utilities, and plotting helpers.
- `scripts/` – ready-to-run studies and validation utilities.
- `data/` – aerodynamic coefficients and reference inputs used by the examples.
- `tests/` – regression tests covering the parametric models and solver utilities.
- `parametrize_path/` – Jupyter notebooks illustrating common parameter studies.

## Contributing

1. Create a feature branch from `main`.
2. Add or update unit tests in `tests/` alongside your changes.
3. Format with `black` and run the test suite (`pytest`).
4. Submit a pull request with a clear description of the problem solved.

## License

This project is distributed under the MIT License (see `LICENSE`).


