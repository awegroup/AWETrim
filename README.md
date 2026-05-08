# AWETrim

AWETrim is a Python library for the modelling, trim analysis, aerostructural simulation, and trajectory optimisation of soft-kite Airborne Wind Energy Systems (AWES). It ties together three external solvers — a vortex aerodynamic method, a particle structural model, and a flight-data Kalman filter — with a CasADi-based system model into a single coherent workflow:

```
Flight data  ──► EKF-AWE ──────────────────────────────────► State estimates
                                                                     │
Kite geometry ──► VSM + PSS + AWETrim EoM ──► Aerostructural ──► Aerodynamic
                  (aerostructural coupling)      sweep results      coefficients
                                                      │                 │
                                               Reduced-order model ◄───┘
                                               (CasADi SystemModel)
                                                      │
                                         Trajectory simulation & optimisation
```

The aerostructural coupling iterates between the Vortex Step Method (aerodynamics) and the Particle System Simulator (flexible structure) under AWETrim's equations of motion to produce the loaded wing shape and aerodynamic forces at each flight condition. These results populate a quasi-steady reduced-order model (ROM) that is fast enough for trajectory simulation and path optimisation.

---

## External solvers

AWETrim delegates physics-heavy computations to three purpose-built packages:

### [Vortex Step Method (VSM)](https://github.com/awegroup/Vortex-Step-Method)

An aerodynamic solver based on enhanced lifting-line theory that couples 2D viscous airfoil polars with a three-quarter-chord vortex formulation. It is specifically designed for low-aspect-ratio wings with sweep, dihedral, and anhedral — the geometry typical of LEI kites. VSM provides:
- Quasi-steady trim states (lift, drag, side force, moments) at a given flight condition.
- Automatic stability derivatives and non-dimensional rate derivatives.
- Panel-level force and circulation distributions that the structural solver needs for load transfer.

Within AWETrim, VSM is invoked by the `aerodynamics/` module for standalone aerodynamic analysis and by the `aerostructural/` module inside the coupling loop.

### [Particle System Simulator (PSS)](https://github.com/awegroup/Particle_System_Simulator)

A particle-based structural solver for flexible membrane systems, validated for soft-wing kites and solar sails (Poland & Schmehl, *Energies* 2023). The wing is discretised into mass nodes connected by spring-damper elements; deformation is driven by the aerodynamic loads supplied by VSM. PSS provides:
- Equilibrium node positions under aero and gravity loads.
- Deformed geometry that is fed back to VSM for the next iteration.

The coupling between VSM and PSS is a fixed-point (Aitken-relaxed) loop managed by `aerostructural/coupling.py`.

### [EKF-AWE](https://github.com/ocayon/EKF-AWE)

An Extended Kalman Filter for Airborne Wind Energy flight data. It processes raw on-board sensor logs (position, velocity, tether force, reel-out speed) to estimate kite states, aerodynamic coefficients, and wind velocity. Within AWETrim, the `experimental/` module wraps EKF-AWE with the data layout and preprocessing steps specific to the kite geometries stored in `data/`.

---

## How the components connect

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWETrim workflow                            │
│                                                                     │
│  1. AEROSTRUCTURAL COUPLING                                         │
│  ┌──────────┐    loads     ┌──────────┐   deformed   ┌──────────┐  │
│  │   VSM    │◄────────────►│ AWETrim  │◄────────────►│   PSS    │  │
│  │ (aero)   │   geometry   │  EoM /   │   geometry   │ (struct) │  │
│  └──────────┘              │  coupler │              └──────────┘  │
│                            └──────────┘                             │
│        │                                                            │
│        ▼ sweep over flight conditions                               │
│  2. ROM IDENTIFICATION                                              │
│  Fit quasi-steady CL/CD/CS(α, β, δs, δp) from sweep results        │
│  → aero_coeffs_rom.yaml                                             │
│                                                                     │
│        │                                                            │
│        ▼                                                            │
│  3. TRAJECTORY SIMULATION & OPTIMISATION                            │
│  CasADi SystemModel (kite + tether + winch + wind)                  │
│  PhaseParameterized / Cycle → power-cycle simulation                │
│  CasADi Opti → path-parameter and control optimisation              │
│                                                                     │
│  4. FLIGHT-DATA ANALYSIS (independent)                              │
│  EKF-AWE pipeline → state estimates, coefficient identification     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/awegroup/AWETrim.git
cd AWETrim
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -e .[dev]
```

The external solvers (VSM, PSS, EKF-AWE) are installed automatically from GitHub as part of the `pip install` step; no manual cloning is required.

---

## Package structure

```
src/awetrim/
  system/          CasADi system model: SystemModel, Kite, Tether variants, Wind, Winch
  aerodynamics/    VSM quasi-steady trim wrapper: solve_vsm_quasi_steady_trim, sweeps, derivatives
  aerostructural/  PSS/QSM fixed-point coupling: PssQsmCoupler, mapping, actuation, results I/O
  kinematics/      Course-frame kinematics and parametrised path patterns (B-spline, helix, …)
  timeseries/      Cycle simulation helpers: PhaseParameterized, Reelout, Reelin, Cycle
  environment/     Wind models: uniform, logarithmic, tabulated
  experimental/    EKF flight-data pipeline wrapping EKF-AWE with AWETrim's data layout
  utils/           Reference-frame transforms, plotting palettes, default bounds and limits
```

---

## Scripts

The `scripts/` directory contains ready-to-run examples organised by analysis domain. Each subfolder has a `common.py` that defines shared defaults (kite name, config folder, solver settings) so individual scripts stay concise.

All scripts are run from the **project root**:

```bash
python scripts/<domain>/<script>.py
```

### `scripts/aerodynamics/` — standalone aerodynamic analysis

Use these to explore VSM trim results for a single flight condition or across parameter ranges, without running the full aerostructural coupling.

| Script | What it does |
|--------|-------------|
| `solve_single_state.py` | Single VSM trim solve at a given wind speed and flight condition |
| `run_sweep.py` | Sweep over wind speed or other flight-condition parameters |
| `compute_stability_derivatives.py` | Numerical aerodynamic stability derivatives |
| `calculate_max_roll.py` | Maximum achievable roll angle under trim constraints |
| `plot_polars.py` | Visualise the airfoil polar data used by VSM |

### `scripts/aerostructural/` — coupled structural-aerodynamic simulation

Use these to run the VSM + PSS coupling loop and produce loaded wing shapes, forces, and aerodynamic coefficient tables for ROM identification.

| Script | What it does |
|--------|-------------|
| `run_simulation_level_qsm.py` | Single PSS/QSM aerostructural solve; optional steering sweep |
| `run_sweep_wind_steering.py` | 2-D sweep over wind speed × steering extension |
| `run_sweep_course_steering_depower.py` | 3-D sweep over course angle × steering × depower |

### `scripts/reduced-order-model/` — ROM validation and path optimisation

Use these after ROM identification to validate the quasi-steady model and optimise trajectory parameters.

| Subdirectory | What it does |
|-------------|-------------|
| `optimization/reelout/` | Optimise uploop, downloop, and helix path parameters using CasADi Opti |
| `validation/` | Validate quasi-steady state approximations and B-spline path fits |
| `wind_estimation/` | Inverse wind estimation from trajectory data |

### `scripts/experimental/` — EKF flight-data analysis

Use these to process raw flight logs through the EKF-AWE pipeline.

| Script | What it does |
|--------|-------------|
| `run_analysis_ekf.py` | Interactive pipeline: select a flight log → preprocess → run EKF → save results |
| `plot_analysis_ekf.py` | Visualise saved EKF output |

---

## Reference kite

All scripts and example configurations in this repository are demonstrated with the **TU Delft LEI-V3 kite**, a well-characterised leading-edge inflatable kite used in Airborne Wind Energy research at TU Delft. The authoritative reference for this kite — including literature, wind tunnel data, and field measurements — is collected at:

> [awegroup/TUDELFT_V3_KITE](https://github.com/awegroup/TUDELFT_V3_KITE)

The config files used here (`data/LEI-V3-KITE/`) are AWETrim-specific (aero geometry panels, structural node layout, solver settings). They will eventually be contributed back to the TUDELFT_V3_KITE repository once the format has stabilised.

---

## Data layout

```
data/
  LEI-V3-KITE/                              primary reference kite (see above)
    system.yaml                             hardware config (mass, KCU, tether, winch)
    aero_geometry.yaml                      VSM panel sections and airfoil polars
    struc_geometry.yaml                     PSS node positions and spring connectivity
    as_config.yaml                          aerostructural coupling settings (dt, tolerances, actuation)
    aero_coeffs_rom.yaml                    quasi-steady ROM coefficients (output of aerostructural sweep)
    ekf_config.yaml                         EKF simulation / tuning parameters
    2D_polars_CFD_NF_combined/              section-by-section 2D polar CSVs and comparison PDFs
    cycle_configs/                          optimised path splines: uploop, downloop, helix
    flight_logs/                            raw CSV flight data and pre-processed HDF5 files
    
results/<kite_name>/
  aerostructural/<case_folder>/sim_output.h5   PSS/QSM coupled-solver output
  ekf/<model>_<YYYY>-<MM>-<DD>.h5             EKF analysis output
```

---

## Tests

```bash
pytest                          # all tests
pytest tests/aerostructural/    # one module
pytest -m "not slow"            # skip slow integration tests
pytest --cov=src --cov-report=term-missing
```

---

## Key dependencies

| Package | Role |
|---------|------|
| [Vortex-Step-Method](https://github.com/awegroup/Vortex-Step-Method) | VSM aerodynamic solver |
| [Particle_System_Simulator](https://github.com/awegroup/Particle_System_Simulator) | PSS flexible structural solver |
| [EKF-AWE](https://github.com/ocayon/EKF-AWE) | Extended Kalman Filter for flight-data analysis |
| [CasADi](https://web.casadi.org) | Symbolic computation, NLP solving (system model, optimisation) |
| NumPy / SciPy | Numerical routines |
| pandas / h5py | Data I/O |

---

## Related publications

The methods implemented in this repository are described in the following papers:

- Cayon, O., van Deursen, V., & Schmehl, R. (2026). **Translational dynamics of bridled kites: a reduced-order model in the course reference frame.** *Wind Energy Science*. [https://doi.org/10.5194/wes-11-1097-2026](https://doi.org/10.5194/wes-11-1097-2026)

- Cayon, O., Watson, S., & Schmehl, R. (2025). **Kite as a sensor: wind and state estimation in tethered flying systems.** *Wind Energy Science*. [https://doi.org/10.5194/wes-10-2161-2025](https://doi.org/10.5194/wes-10-2161-2025)

- Cayon, O., Gaunaa, M., & Schmehl, R. (2023). **Fast Aero-Structural Model of a Leading-Edge Inflatable Kite.** *Energies*. [https://doi.org/10.3390/en16073061](https://doi.org/10.3390/en16073061)

---

## License

LGPL-3.0 — see `LICENSE`.

Modifications to AWETrim must be released under the same license. Contributions via pull requests to the [main repository](https://github.com/awegroup/AWETrim) are very welcome.
