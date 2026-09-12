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
  aerodynamics/      ✅  VSM quasi-steady trim + KCU bluff-body drag
                         (kcu_drag.py, single-sourced, on by default in the
                          aerostructural path via is_with_kcu_drag)
                         + bridle-line SECTION drag (line_drag.py): the
                         steering/depower lines are flat 12 mm tapes, not
                         round cable; declared by a "w" column in
                         struc_geometry bridle_lines and applied as a
                         drag-equivalent diameter (as_config
                         aerodynamic_bridle.tape_roll_model)
                         + apparent_wind.py: THE single source of
                         va(r) = va_free - omega x (r - r0), the relation that
                         puts a load at the right STATION on a rotating kite
                         (bridle segments, the KCU). The tether does NOT use
                         it: williams_tether rotates about the GROUND ANCHOR
                         with per-node wind, and the lumped closed form is an
                         integral, not a point evaluation
                         — see src/awetrim/aerodynamics/AGENTS.md
  aerostructural/    ✅  The coupling: shared interfaces (protocols, mapping,
                         convergence, forces, results, utils) plus one adapter
                         per Billow fidelity. The structural solver itself is
                         Billow, a separate package.
    coupled/         ✅  The coupled solve itself, backend-agnostic.
                         coupled_solver.py dispatches over wireframe / billow,
                         so a new backend is a branch there, not a copy;
                         read_struc_geometry_yaml.py is the ONE geometry reader
                         both consume and aero2struc.py the ONE load transfer.
                         aero2struc.chordwise_distribution offers
                         "moment_matched", which places each panel's load at
                         its own centre of pressure so the LOCAL pitching
                         moment is conserved (default stays "cp_file", one
                         fixed 0.291 c station for every panel, which
                         reproduces stored results)
    wireframe/       ✅  Adapter for Billow's WIREFRAME fidelity (cables,
                         tension-only lines, pulleys) + the PSM production
                         driver coupled_solver_qsm.py and the geometry reader
                         structural_geometry_io.py. Was pss/ until 2026-09-12
    billow/          ✅  Adapter for Billow's FULL fidelity: inflatable
                         Timoshenko tube beams and wrinkling CST membrane
                         canopy on top of the same cables and pulleys. Reads
                         struc_geometry_FEM_full.yaml (the only geometry with
                         tubes and pressure). Converged on the LEI-V3
                         unactuated baseline
  kinematics/        ✅  course-frame kinematics, B-spline path patterns
  timeseries/        ✅  PhaseParameterized, ReeloutSimple, ReelinSimple, Cycle
  environment/       ✅  Wind (uniform / logarithmic / power_law / explog /
                         jet / tabulated). profile_laws.py is the ONLY place the
                         analytic formulas live (pure functions over an ``xp``
                         math namespace: numpy for fits/scripts, casadi inside
                         Wind); Wind stores one amplitude (speed_wind_ref at
                         height_ref, or speed_friction for log-based models) and
                         derives the other lazily. wind_factory.create_wind_model
                         builds fully numeric models; wind_profiles.py maps the
                         co-sim InflowConditions struct (laws 0-6, CUSTOM_* fits)
                         onto create_wind_model kwargs.
  server/            ✅  REST API for reelout trajectory optimization
                         (FastAPI, optional [server] extra; endpoints
                         /init /status /step /trajectory /reset; one
                         warm-started re-solve per /step for co-simulation
                         clients, e.g. an external kite simulator).
                         Co-sim structs: WinchParams {mode, k_v, f_min,
                         f_max, v_max|p_max, optimize_k_v, k_v_bounds}
                         (optimize_k_v -> slope_winch_ro becomes a design
                         variable, bracketed per request around the client's
                         k_v; replies echo the OPTIMIZED k_v and flag
                         optimized_parameters.k_v_at_bound), Trajectory,
                         DepowerParams {mode: fixed|optimize|profile, value},
                         InflowConditions {wind_speed @6 m, wind_direction
                         (compass, FROM), profile_law 0-6, alpha, z0,
                         heights/speeds, turbulence (ignored)},
                         InitParams, StepParams. Replies always echo the
                         depower the returned path assumes — it is not
                         flyable without it. Optional client-supplied
                         min_turn_radius [m] (init/step; echoed) maps to
                         sim_parameters.min_turn_radius, a dense geometric
                         curvature constraint in opti_phase; replies carry
                         metrics.turn_radius_min_m + a turn_radius column.
                         Optional pattern_limits {azimuth_max, elevation_min,
                         elevation_max, azimuth_amplitude_min} [deg]
                         (init/step; echoed) -> opti_limits_override C_phi/
                         C_beta (B-spline hull) + sim_parameters.
                         min_azimuth_amplitude (one smooth row in opti_phase).
                         Launcher: scripts/server/run_reelout_server.py
  experimental/      ✅  EKF flight-data analysis pipeline (+ data_preprocessors/)
  plotting/          ✅  shared plotting helpers — see src/awetrim/plotting/AGENTS.md
  utils/             ✅  fitting, defaults, reference frames
  identification/    🟡  ROM aero-coefficient identification: tidy dataset
                         (aero_dataset.py, AS + EKF sources, shared schema),
                         BIC forward-stepwise polynomial fit with k-fold CV
                         (aero_polynomial.py → rom_config "coeffs"), control
                         conventions (controls.py), rigid_body_axes.py
                         (body axes anchored to the wing's CENTRE PANEL,
                         aircraft FRD sense; never inertia eigenvectors).
                         Regressors: alpha, u_s, u_p, v_a; targets CL/CD/phi_a.
                         (aero LUT guide is outdated and needs revising)
```

ROM is **script-based**, not a `src/` module: see `scripts/reduced-order-model/`
(`optimization/`, `validation/`), configured via each kite's `rom_config.yaml`.

**The structural solver is [Billow](https://github.com/awegroup/Billow)**, a
separate package (`pip install billow`), imported as `billow`. It was
`src/awetrim/structural/` until 2026-09-12 and is now its own repository with
its own documentation, demonstration cases and validation suite -- including
the measured hanging-V3 case, which moved with it.

AWETrim uses BOTH its fidelities, through one adapter each under
`aerostructural/`:

- **wireframe** (`billow.build_line_system`) -- cables, tension-only lines and
  pulleys. The bridle and line system on its own; what the PSS particle system
  used to do.
- **full** -- the above plus inflatable Timoshenko tube beams and wrinkling CST
  membrane canopy.

They are not two force laws. Billow has one cable kernel and both fidelities
call it, so the consistency that used to have to be maintained by hand across
three codes is now structural.

**The pulley rest-length convention is the one thing an adapter must state
explicitly**, because file formats disagree: the PSS reader split `l0` across
the two arms, `kite_fem` stored the total on each, and Billow's `PulleyKernel`
takes the whole rope. `build_line_system` therefore takes `pulley_rest_lengths`
rather than inferring it.

**Chordwise force distribution.** The aero→struc coupling spreads each spanwise
VSM force over 10 chordwise nodes. Where it puts the resultant IS the panel's
local pitching moment, and the wing's trim is a moment balance, so this is a
first-order modelling choice rather than a detail — see
`aerostructural/fem/aero2struc.py`. `chordwise_distribution: moment_matched`
tilts the measured `Delta C_p` shape onto the centre of pressure VSM computes
from each panel's own force and moment, conserving both; the `cp_file` default
applies one fixed shape to every panel (0.291 c on the LEI-V3), which imposes
~917 N m of same-sign spurious moment across the wing. A genuinely resolved
chordwise *shape* (from CFD) would still be an improvement on the tilted prior.

**Still open in the same family:** `map_aero_forces_to_struct_nodes`, the
nearest-node spatial step that follows, loses ~20% of the moment once the kite
has deformed (0.09% on the built shape). `check_moment_preservation` measures
that step alone, not the chordwise placement. The FEM structural solver itself
also needs further work.

**Read the module's `AGENTS.md` before modifying `aerodynamics/`, `aerostructural/`
or `plotting/`; read Billow's own `AGENTS.md` before changing element physics.
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
- **Wind profiles:** never restate a profile law (log/power/explog/jet or the
  `u* = κU/ln(z_ref/z0)` conversion). Build winds with
  `environment.wind_factory.create_wind_model` (or the YAML `wind:` section via
  `system.factory.create_wind_model_from_config`) and, when a NumPy evaluation
  is needed, call the kernels in `environment/profile_laws.py`
  (`log_law`, `friction_velocity`, `speed_from_friction_velocity`, ...).
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
| `billow` | Minimum-energy structural solver, both fidelities | Installed from GitHub `@main` via `pyproject.toml` (https://github.com/awegroup/Billow). For local development: `pip install -e ../Billow` |
| `awes-ekf` | Extended Kalman Filter for flight data | Installed from GitHub; repository used here: https://github.com/ocayon/EKF-AWE |
| `awesIO` | IO helpers used by scripts | Installed from GitHub (https://github.com/awegroup/awesIO) |
| `CasADi` | Symbolic computation in `system/` | Required; used heavily in `src/awetrim/system/` |

Notes

 - The `aerodynamics/` module uses the VSM solver via an adapter; see `src/awetrim/aerodynamics/AGENTS.md` for module-specific guidance.
 - `PSS` (Particle System Simulator) and `kite_fem` were removed on 2026-09-12. Billow replaced both; do not reintroduce either. `tests/aerostructural/test_wireframe_package.py` asserts that importing the package pulls in neither.
 - `scripts/aerostructural/common.py` defines `CONFIG_DEFAULTS` used by multiple scripts; prefer importing it for consistent defaults.

## Per-kite data layout

Each kite under `data/<kite_name>/` should include at minimum the following files and folders so scripts and tools can locate inputs automatically:

- `system.yaml` — hardware and system-level configuration (kite mass, KCU, tether properties, winch, mass/inertia). This is the primary source for `SystemModel` properties. **KCU mass is the single source of truth here** (`components.kite.control_system.structure.mass`); both the structural KCU node mass and the QSM `mass_kcu` are resolved from it. **The winch drive envelope is the single source of truth here too** (`components.ground_station.drums[0]`: `min_tether_speed` / `max_tether_speed` / `max_winch_acceleration` / `max_tether_force`); `factory._extract_hardware_limits` maps it onto the optimizer's `speed_radial` bounds and the `winch_acceleration` slew limit, so cycle configs must not restate it.
- `struc_geometry.yaml` — structural geometry describing wing nodes, LE/TE positions, bridle nodes and connectivity, spring/rest-length definitions, pulley info. A `bridle_lines` row may carry an optional `w` (flat-tape width [m]); `d` then stays the AREA-equivalent diameter used for mass and EA, while the drag uses the projected width (`awetrim.aerodynamics.line_drag`). Does **not** carry `kcu_mass` (deprecated; ignored with a warning if present — set it in `system.yaml`).
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

