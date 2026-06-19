# AWETrim scripts

Runnable entry points that exercise the `awetrim` library. All scripts are
executed **from the project root**, for example:

```bash
python scripts/aerostructural/run_simulation_PSM.py
python scripts/reduced-order-model/optimization/cycle/run_cycle_simulation.py --plot
```

The reference kite throughout is the TU Delft LEI-V3 (`data/LEI-V3-KITE/`).

## Folders

| Folder | What it does |
|--------|--------------|
| [`aerodynamics/`](aerodynamics/) | VSM quasi-steady trim, stability derivatives & flight-dynamic modes, parametric wing/airfoil studies |
| [`aerostructural/`](aerostructural/) | Coupled VSM ↔ structure (PSS or kite_fem) deformed-shape simulations |
| [`reduced-order-model/`](reduced-order-model/) | ROM trajectory simulation/optimisation (reel-out patterns, reel-in, full cycle) and validation against flight data |
| [`identification/`](identification/) | Identify/calibrate ROM aero parameters and the turn-rate law from flight data |
| [`experimental/`](experimental/) | EKF flight-data reconstruction (states, wind, in-flight coefficients) and plotting |

`scripts/personal/` holds unsupported, author-private experiments and is **not**
covered here.

## Conventions

- Aerodynamics scripts write to `results/aerodynamics/<script_name>/` by default
  (override with `--output-dir`) and accept `--vsm-src` to point at a local
  Vortex-Step-Method checkout when it is not installed as a package.
- Aerostructural results follow `results/<kite_name>/aerostructural/<case>/`
  (`sim_output.h5` plus deformed-geometry snapshots).
- Each module's behaviour is governed by the YAML files under `data/<kite>/`
  (`system.yaml`, `aero_geometry.yaml`, `struc_geometry.yaml`, `as_config.yaml`,
  `rom_config.yaml`, `cycle_configs/`). See the top-level `AGENTS.md` for the
  per-kite data layout.
