# AWETrim examples

AWETrim takes a kite from its design to optimised flight trajectories. The
examples follow that workflow, one folder per step. Each step writes files
that the next one reads, so the chain also works for your own kite. Run every
example **from the project root**. They all use the TU Delft LEI-V3 kite in
`data/LEI-V3-KITE/`.

```
 1 aerodynamics ──► 2 aerostructural ──► 3 ROM identification ──► 4 trajectory optimisation
 data/<kite>/         deformed shapes        sweep dataset,             cycle_configs/,
 (your kite)                                 rom_config_*.yaml          optimised paths
                                             (the reduced-order              ▲
                                              model)                         │
                                                  ▲                          │
                                                  │                     server/ (as a service)
                               5 flight data ─────┘
                               EKF-AWE output (.h5): correct and validate the ROM
```

| Step | Folder | What it does | Writes |
|------|--------|--------------|--------|
| 1 | [`1_aerodynamics/`](1_aerodynamics/) | Describe the kite in `data/<kite>/`, compute its lift and drag polars, solve one quasi-steady VSM trim and check its static stability | `results/aerodynamics/` |
| 2 | [`2_aerostructural/`](2_aerostructural/) | Couple VSM and the Billow structure to the deformed shape the kite flies in, at wireframe or full fidelity | `results/<kite>/aerostructural/` |
| 3 | [`3_rom_identification/`](3_rom_identification/) | Sweep the operating range with the coupled model of step 2 (depower and steering chains, frozen-shape polars), then identify the reduced-order model (ROM) from it: lift, drag, bridle pitch and roll gain | `results/<kite>/identification/` (dataset and `rom_config_aerostructural.yaml`) |
| 4 | [`4_trajectory_optimization/`](4_trajectory_optimization/) | Simulate and optimise reel-out patterns and whole pumping cycles with the ROM | `data/<kite>/cycle_configs/`, `results/<kite>/optimization/` |
| 5 | [`5_flight_data/`](5_flight_data/) | Optional, with flight data: correct the ROM's depower pitch and drag on a flight, and validate it against the measurements | `data/<kite>/rom_config_aerostructural_flight_corrected.yaml` |
| | [`server/`](server/) | Step 4 as a REST service, with Python, Julia and MATLAB clients | |

Quick start, steps 1, 2 and 4 on the shipped LEI-V3 ROM:

```bash
python examples/1_aerodynamics/solve_single_state.py
python examples/2_aerostructural/run_simulation_PSM.py
python examples/4_trajectory_optimization/cycle/fit_periodic_cycle_config.py --kite LEI-V3-KITE --auto
python examples/4_trajectory_optimization/cycle/run_full_cycle_opti.py --kite LEI-V3-KITE
```

Each folder's README lists its options and outputs.

## Flight data

Step 5 reads a flight as reconstructed by the extended Kalman filter in
[EKF-AWE](https://github.com/awegroup/EKF-AWE): AWETrim only consumes its
output file. EKF-AWE ships the raw LEI-V3 flight logs; run it on the
2019-10-08 flight and place the output at
`results/LEI-V3-KITE/ekf/LEI-V3 Kite_2019-10-08.h5`.

## Conventions

- Everything about a kite comes from the YAML files in `data/<kite>/`
  (`system.yaml`, `aero_geometry.yaml`, `structure.yml`,
  `as_config.yaml`, `identification.yaml`, the ROM files, `cycle_profile.yaml`,
  `cycle_configs/`).
  `system.yaml` selects which ROM file the trajectory steps use.
- Results go to `results/<kite>/<analysis>/` (step 1: `results/aerodynamics/<script>/`,
  override with `--output-dir`).

The research studies behind the papers and notes in `docs/` are not part of
the examples; their scripts are available on request.
