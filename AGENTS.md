# AWETrim — Agent Context

## What this repo is

AWETrim is a Python library for the design and optimisation of soft-kite Airborne
Wind Energy Systems (AWES). It models the kite as a point mass in a course-aligned
spherical reference frame, provides CasADi-based trajectory optimisation, and
implements the winch/tether physics for pumping-cycle simulation.

---

## Module status

```
src/awetrim/
  system/            ✅  Wing, Kite, SystemModel, Winch, Tether variants
                         (rigid/flexible link, rigid/flexible lumped,
                          Williams discretised distributed-mass)
  aerodynamics/      ✅  VSM quasi-steady trim — see src/awetrim/aerodynamics/AGENTS.md
  aerostructural/    ✅  Shared interfaces: protocols, mapping, convergence, forces, results, utils
    pss/             ✅  PSS/QSM coupled solver — see src/awetrim/aerostructural/AGENTS.md
    fem/             🟡  FEM coupling implemented; structural solver and
                         chordwise force distribution still need improvement (see note)
  kinematics/        ✅  course-frame kinematics, B-spline path patterns
  timeseries/        ✅  PhaseParameterized, ReeloutSimple, ReelinSimple, Cycle
  environment/       ✅  Wind (logarithmic / uniform / tabulated)
  experimental/      ✅  EKF flight-data analysis pipeline (+ data_preprocessors/)
  plotting/          ✅  shared plotting helpers — see src/awetrim/plotting/AGENTS.md
  utils/             ✅  fitting, defaults, reference frames
  identification/    🟡  started — rigid_body_axes.py + an aero LUT guide
                         (the guide is outdated and needs revising)
```

ROM is **script-based**, not a `src/` module: see `scripts/reduced-order-model/`
(`optimization/`, `validation/`), configured via each kite's `rom_config.yaml`.

**FEM known limitation:** the aero→struc coupling currently spreads each spanwise
VSM force over 10 chordwise nodes using weights from a single-AoA Cp file
(`cp_AOA_8.dat`), with a uniform fallback — see
`aerostructural/fem/aero2struc.py`. A physically correct chordwise force
distribution (from CFD or another source) is still needed, and the FEM structural
solver itself needs further work.

**Read the module's `AGENTS.md` before modifying `aerodynamics/`, `aerostructural/`,
or `plotting/`.
When you add, remove, or rename public functions, dataclasses, config keys, or file layout in any module that has an `AGENTS.md`, update that file in the same commit.**

## Physics references

Every governing equation should trace back to one of:
- **Aerostructural / VSM:** Cayon, Gaunaa, Schmehl (2023) *Energies* 16, 3061
- **Identification / ROM:** Cayon, van Deursen, Schmehl (2026) *WES* 11, 1097
- **Trajectory optimisation:** Cayon & Schmehl (2026) Torque extended abstract
- **Discretised tether (distributed-mass, lumped-element):** Williams (2017)
  *J. Guid. Control Dynam.* 40, 1779–1788, https://doi.org/10.2514/1.G002354
  — basis for `system/williams_tether.py` (`WilliamsTether`)

---

## Symbol ↔ code name table

| Symbol | Code name | Unit |
|--------|-----------|------|
| r | `distance_radial` | m |
| β | `angle_elevation` | rad |
| φ | `angle_azimuth` | rad |
| χ | `angle_course` | rad |
| vτ | `speed_tangential` | m/s |
| vr | `speed_radial` | m/s |
| s | `s` | — |
| ṡ | `s_dot` | 1/s |
| Ft | `tension_tether_ground` | N |
| uₛ | `input_steering` | — |
| uₚ | `input_depower` | — |
| α | `angle_of_attack` | rad |
| θb | `angle_pitch_tether` | rad |
| ΔCD0 | `cd0` | — |

---

## Core conventions

- **CasADi** throughout. All state variables are `ca.MX.sym`. Never replace
  symbolic variables with NumPy scalars inside module code.
- **YAML** configs drive kite parameters and pattern settings.
  See `data/LEI-V3-KITE/` for reference examples.
- **IPOPT** is the default NLP solver: `opti.solver("ipopt", {...})`.
- Optimisation variable bounds live in `utils/defaults.py` (`DEFAULT_OPTI_LIMITS`).
  Add new variables there, not inline in module code.
- In `SystemModel(quasi_steady=True)`, `timeder_speed_tangential = 0` must be enforced.
- Cross-module data uses plain dicts or `dataclass` — keep CasADi symbolics from
  crossing module boundaries unless explicitly decided.
- **Tests:** one file per source module under `tests/`; assert CasADi expression
  structure and symbolic shapes, not numeric solver values; share kite/system
  setups via fixtures in `tests/conftest.py`.

---

## Tooling, Commands and External Dependencies


Common commands

 - **Install (editable, with dev extras):** pip install -e .[dev]
 - **Run all tests:** pytest
 - **Run tests for a module:** pytest tests/aerostructural/
 - **Run a single test:** pytest tests/aerostructural/test_pss.py::test_aerostructural_import_does_not_import_pss
 - **Run with coverage:** pytest --cov=src --cov-report=term-missing

Scripts

 - Scripts live in `scripts/` and are executed from the project root, for example:
   - python scripts/aerostructural/run_simulation_level_qsm.py
   - python scripts/aerodynamics/solve_single_state.py
   - python scripts/experimental/run_analysis_ekf.py

External dependencies referenced by `pyproject.toml` (VCS installs):

| Package | Role | Install note |
|---------|------|-------------|
| `Vortex-Step-Method` | VSM aerodynamic solver | Installed from GitHub `@main` via `pyproject.toml` (https://github.com/awegroup/Vortex-Step-Method) |
| `PSS` | Particle System Simulator (structural solver) | Installed from GitHub via `pyproject.toml` (https://github.com/awegroup/Particle_System_Simulator, pinned v1.0.2) |
| `awes-ekf` | Extended Kalman Filter for flight data | Installed from GitHub; repository used here: https://github.com/ocayon/EKF-AWE |
| `awesIO` | IO helpers used by scripts | Installed from GitHub (https://github.com/awegroup/awesIO) |
| `CasADi` | Symbolic computation in `system/` | Required; used heavily in `src/awetrim/system/` |

Notes

 - The `aerodynamics/` module uses the VSM solver via an adapter; see `src/awetrim/aerodynamics/AGENTS.md` for module-specific guidance.
 - `scripts/aerostructural/common.py` defines `CONFIG_DEFAULTS` used by multiple scripts; prefer importing it for consistent defaults.

## Per-kite data layout

Each kite under `data/<kite_name>/` should include at minimum the following files and folders so scripts and tools can locate inputs automatically:

- `system.yaml` — hardware and system-level configuration (kite mass, KCU, tether properties, winch, mass/inertia). This is the primary source for `SystemModel` properties. **KCU mass is the single source of truth here** (`components.kite.control_system.structure.mass`); both the structural KCU node mass and the QSM `mass_kcu` are resolved from it.
- `struc_geometry.yaml` — structural geometry describing wing nodes, LE/TE positions, bridle nodes and connectivity, spring/rest-length definitions, pulley info. Does **not** carry `kcu_mass` (deprecated; ignored with a warning if present — set it in `system.yaml`).
- `aero_geometry.yaml` — VSM aerodynamic geometry describing wing sections, paneling, and references to airfoil polars; may reference a subfolder with airfoil `.dat` or polar CSVs.
- `as_config.yaml` (or `aerostructural_configs/config.yaml`) — aerostructural solver settings (time-step, tolerances, actuation options, initialisation flags).
- `rom_config.yaml` — reduced-order aerodynamic coefficient definitions (plus ROM tether settings) used by ROM or identification flows.
- `ekf_config/` — EKF configuration files and model-specific tuning parameters used by the `experimental` EKF pipeline.

Optional but recommended:

- `flight_logs/` — raw flight CSVs for EKF and identification.
- `cycle_configs/` — trajectory/pattern YAMLs for timeseries scripts (downloop, uploop, helix, etc.).

Results layout (convention):

- `results/<kite_name>/<analysis_type>/sim_output.h5` — aerostructural coupled-solver outputs.
- `results/<kite_name>/ekf/` — EKF outputs and diagnostics.

Use the canonical filenames above so helper utilities (e.g., `resolve_kite_paths` in `scripts/aerostructural/common.py`) locate files automatically.

