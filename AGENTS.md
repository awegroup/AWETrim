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
                         + trim_casadi.py: the quasi-steady trim as ONE CasADi
                         root-finding problem (trim states + circulations +
                         Williams length, exact Jacobian, numeric AIC frozen
                         per wake pass); 20-70x faster than the least-squares
                         trims, same result. panel_kernels.py = the section
                         force laws as xp kernels (numpy/casadi, one formula)
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
                         (+ the slanted up-loop figure eight of the
                         single-period pumping cycle: slanted_eight_angles /
                         slanted_eight_landmarks (NumPy, Gerono lobes scaled to
                         the stated circles, s=0 at the low-lobe bottom),
                         make_slanted_eight_bspline_path_parameters (smallest
                         M under a metre of deviation; returns (dict, info)),
                         count_self_crossings / eight_crossing_s /
                         lobe_signed_areas (by="elevation" low/high or
                         by="azimuth" left/right) as topology + sense checks;
                         symmetrize_periodic_path: half a period of the
                         uniform periodic spline is an index shift by M/2, so
                         a mirrored path is a coefficient-pair condition)
  timeseries/        ✅  PhaseParameterized, ReeloutSimple, ReelinSimple, Cycle
                         (opti_phase sim_parameters.periodic_wrap: the seam
                         interval node N-1 -> node 0 of a one-period periodic
                         spline gets the same continuity / rate / AoA rows,
                         energy, time and regulariser term as the interior
                         ones -- the closed cycle without close_radial_cycle,
                         which it refuses to combine with; NLP byte-identical
                         when off. sim_parameters.mirror_symmetry (+
                         mirror_azimuth, default 0): half-period mirror rows
                         C_phi[k+M/2] = 2 az_c - C_phi[k], C_beta[k+M/2] =
                         C_beta[k] for optimized shape coefficients, and
                         node pairs i / i+N/2 with v_r, u_p equal and u_s
                         opposite; needs periodic_wrap, winch_mode
                         free_speed (the force law makes the v_r rows
                         degenerate), even M and N, and a fixed shape that
                         is already mirrored; off = NLP unchanged. PHYSICAL
                         only about the downwind meridian az = 0 (another
                         mirror_azimuth warns). Known LICQ degeneracy: each
                         active node bound is active at i and i+N/2
                         together, so with the mirror rows those rows are
                         dependent and their multipliers not unique; the
                         clean fix is elimination (declare only the first
                         half as variables) -- open follow-up)
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
  utils/             ✅  fitting, defaults, reference frames, control_metrics
                         (steering-reversal count / pairs outside a deadband,
                         seam pair counted on a periodic history)
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
(`optimization/`, `validation/`), configured via the ROM config the kite's
`system.yaml` selects (`models.reduced_order.aerodynamics`, resolved by
`system.factory.resolve_rom_config_path`; fallback a sibling `rom_config.yaml`).
The LEI-V3 has two: `rom_config_aerostructural.yaml` (identified from the coupled
Billow-wireframe centre-window simulations only, by
`scripts/identification/identify_rom_aerostructural.py`) and
`rom_config_aerostructural_flight_corrected.yaml` (the DEFAULT in every
LEI-V3 system file: the aerostructural one with
theta_b and a single-input dC_D(u_p, u_s^2) fitted to 2019 flight, cycles 60-67
held out: `identify_rom_flight_correction.py` (equation error: theta_b from the
flight LIFT, drag at equal lift) then `refine_rom_flight_output_error.py`
(output error on the quasi-steady tension and v_tau, with the roll gain set by
the turn-rate law -- the solved steering regressing 1:1 on the logged steering
-- which edits the file in place; the equation-error values alone do not transfer to the rigid-tether
quasi-steady solve)). The paper's semi-empirical ROM
(`rom_config_semi_empirical.yaml`) was removed on 2026-10-07 after the
comparison in `docs/identification/`; it lives in git history (last at
commit bad8a9b) and `config_paths.LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG` keeps its
path so the comparison scripts run on a checkout of it.

Full-cycle optimisation scripts (`optimization/cycle/`): `fit_periodic_cycle_config.py`
+ `run_full_cycle_opti.py` (multi-lobe cycle, `close_radial_cycle`). Studies
built on them live in `optimization/studies/` (see its README), e.g.
`run_uploop_eight_opti.py` + `plot_uploop_eight_comparison.py` (ONE slanted up-loop figure eight per period: low lobe
+ crossings reel out, high arc reels in; `periodic_wrap`, `free_speed`, free
period, staged S0 fixed path / S1 shape box / S2 polish; wind from the CLI or
the profile's `uploop_eight.wind`, never the top-level `wind:`; `--shape
symmetric` flies the level eight of `symmetric_eight:` instead, `--symmetric`
adds `mirror_symmetry`; `--baseline-seed` puts the multi-lobe seed on the same
formulation). The
steering-reversal count (`awetrim.utils.control_metrics.count_steering_reversals`,
deadband 0.02 standardised u_s, seam pair counted on a closed cycle) is a
first-class cycle metric next to the mean power in `cycle_comparison_plots`.

**ROM parameter names** (`aerodynamics.params`, read by `system/kite.py`):
`angle_pitch_tether_0` + `slope_angle_pitch_tether_depower` give the bridle
pitch theta_b(u_p) in the PAPER sign, `alpha_w = alpha_b - theta_b`;
`gain_roll_steering` is k_phi,s (`phi_a,w = k u_s`, asymmetric steering); both
angles are taken in the paper's App. D apparent-wind frame, which
`Kite._force_in_wind_frame` builds from the v_a VECTOR (until 2026-10-06 an
Euler-angle frame with a mirrored yaw sign put the bridle pitch 2-3 deg off
in sideslip, zero only at the centre-window trims; every flight calibration
before that date absorbed it);
`aerodynamics.reference_area` [m2] states the area the coefficients are
normalised by and overrides the system wing area (the LEI-V3 ROMs both use
19.75 m2, the EKF area; system.yaml's projected 17.2119 m2 is geometry, not a
ROM reference). `aerodynamics.kcu_drag_in_coefficients` (default true) says
whether C_D already carries the KCU drag (flight-calibrated ROM); false = the
Kite adds it from the system file's KCU length/diameter/turbine through
`aerodynamics/kcu_drag.py`, as a force at the bridle point that also enters the
bridle resultant F_b (so swapping the KCU needs no re-identification). Coefficient
terms may use `alpha`, `u_p`, `u_s`, `v_a` and `stall` (the smooth
attached->separated switch `kite.stall_blend`, needing params
`angle_of_attack_stall`, `width_stall`). Every term is a PLAIN monomial (C_D
included, since 2026-10; before, every C_D term was |monomial|, a kink at 0);
`abs: true` on a term asks for |monomial| and is kept only for the legacy
steering-drag terms. `u_s` is the STANDARDISED steering of
`identification.controls` (1.4 m x u_s = steering-tape half-difference; 2019
V3 u_s = -kcu/200, KCU limit +-0.175) in every V3 file and in
DEFAULT_OPTI_LIMITS.
`aerodynamics.params.angle_of_attack_stall` is the ROM's stall angle; the
optimiser's `sim_parameters.stall_margin_deg` keeps alpha that far below it at
every node and then REPLACES the generic alpha box (no lower bound, no other
upper bound). Quasi-steady roots are selected by being BELOW THE STALL (alpha <
angle_of_attack_stall): the phase seed march prefers pre-stall roots
(`seed_reject_post_stall`) and keeps simulating on the closest post-stall one
when none exists; `setup_qs_solver` exposes `_qs_alpha_function` for other
callers (the flight validator). The speed stability (paper Eq. 36,
`system_model.tangential_speed_stability`, at fixed steering) is a DIAGNOSTIC
(`_qs_stability_function`): every flown reel-in state reads unstable. Legacy keys
`angle_pitch_depower_0` / `delta_pitch_depower` (opposite sign) and a
`CS: u_s` roll gain are still read, with a DeprecationWarning.

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
   - python scripts/aerostructural/run_simulation_PSM.py
   - python scripts/aerodynamics/solve_single_state.py
   - python scripts/experimental/run_analysis_ekf.py

External dependencies referenced by `pyproject.toml` (VCS installs):

| Package | Role | Install note |
|---------|------|-------------|
| `Vortex-Step-Method` | VSM aerodynamic solver | Installed from GitHub, **pinned to `8825a12`** in `pyproject.toml` (https://github.com/awegroup/Vortex-Step-Method). Later VSM changed the panel force interface (`compute_aerodynamic_quantities` returns 3 values, not 4) that `aerodynamics/panel_kernels.py` / `trim_casadi.py` mirror; unpin only together with adapting those (16 CasADi tests fail otherwise). An editable local VSM checkout on another branch silently overrides the pin |
| `billow` | Minimum-energy structural solver, both fidelities | Installed from GitHub `@main` via `pyproject.toml` (https://github.com/awegroup/Billow). For local development: `pip install -e ../Billow` |
| `awes-ekf` | Extended Kalman Filter for flight data | Installed from GitHub; repository used here: https://github.com/ocayon/EKF-AWE |
| `awesIO` | IO helpers used by scripts | Installed from GitHub (https://github.com/awegroup/awesIO) |
| `CasADi` | Symbolic computation in `system/` | Required; used heavily in `src/awetrim/system/` |

Notes

 - The `aerodynamics/` module uses the VSM solver via an adapter; see `src/awetrim/aerodynamics/AGENTS.md` for module-specific guidance.
 - `PSS` (Particle System Simulator) and `kite_fem` were removed on 2026-09-12. Billow replaced both; do not reintroduce either. `tests/aerostructural/test_wireframe_package.py` asserts that importing the package pulls in neither.
 - `awetrim.aerostructural.case` defines `CONFIG_DEFAULTS` and the case helpers used by the aerostructural drivers and scripts (`scripts/aerostructural/common.py` re-exports it); prefer importing it for consistent defaults.
 - `scripts/aerostructural/` is the public demonstrator: `run_simulation_PSM.py` / `run_simulation_BILLOW.py` (edit the Inputs block) and `plot_simulation.py` (trim table + 3-D viewer). Studies cited by `docs/billow/` live in `scripts/aerostructural/studies/`.

## Per-kite data layout

Each kite under `data/<kite_name>/` should include at minimum the following files and folders so scripts and tools can locate inputs automatically:

- `system.yaml` — hardware and system-level configuration (kite mass, KCU, tether properties, winch, mass/inertia). This is the primary source for `SystemModel` properties. **KCU mass is the single source of truth here** (`components.kite.control_system.structure.mass`); both the structural KCU node mass and the QSM `mass_kcu` are resolved from it. **The winch drive envelope is the single source of truth here too** (`components.ground_station.drums[0]`: `min_tether_speed` / `max_tether_speed` / `max_winch_acceleration` / `max_tether_force`); `factory._extract_hardware_limits` maps it onto the optimizer's `speed_radial` bounds and the `winch_acceleration` slew limit, so cycle configs must not restate it.
- `struc_geometry.yaml` — structural geometry describing wing nodes, LE/TE positions, bridle nodes and connectivity, spring/rest-length definitions, pulley info. A `bridle_lines` row may carry an optional `w` (flat-tape width [m]); `d` then stays the AREA-equivalent diameter used for mass and EA, while the drag uses the projected width (`awetrim.aerodynamics.line_drag`). Does **not** carry `kcu_mass` (deprecated; ignored with a warning if present — set it in `system.yaml`).
- `aero_geometry.yaml` — VSM aerodynamic geometry describing wing sections, paneling, and references to airfoil polars; may reference a subfolder with airfoil `.dat` or polar CSVs.
- `as_config.yaml` (or `aerostructural_configs/config.yaml`) — aerostructural solver settings (time-step, tolerances, actuation options, initialisation flags).
- `rom_config*.yaml` — reduced-order aerodynamic coefficient definitions (one file per ROM; `system.yaml` `models.reduced_order.aerodynamics` names the one in use, else a sibling `rom_config.yaml` is used) (plus ROM tether settings) used by ROM or identification flows. A `controls.input_depower: {powered, depowered}` block states the depower band the ROM was identified on, in the ROM's own `u_p` unit (`identification.controls.rom_depower_band`; absent = the V3 power-tape metres 1.7/2.1, but the full-cycle scripts require it). An optional `validity.angle_of_attack_deg: [lo, hi]` states the AoA range the ROM was identified on (the full-cycle optimizer's AoA bound).
- `ekf_config/` — EKF configuration files and model-specific tuning parameters used by the `experimental` EKF pipeline.

Optional but recommended:

- `flight_logs/` — raw flight CSVs for EKF and identification.
- `cycle_configs/` — trajectory/pattern YAMLs for timeseries scripts (downloop, uploop, helix, etc.).
- `cycle_profile.yaml` — inputs of the full-cycle scripts (`scripts/reduced-order-model/optimization/cycle/`, `--kite <folder>`): wind, winch force law and seed size (r0, duration prior, figure-eight/reel-in size), all required; see `cycle_kites.py` and the annotated LEI-V3 file. An optional `uploop_eight:` block (lobe centres/radii in rad, `sense`, depower window, the height band `min_height_m` / `max_height_m` (z = r sin β, the node-wise `height` rows — the binding vertical limit, shared with the `--baseline-seed` cycle; the C_beta hull stays loose), stall margin, turn-radius floor, `r0_max_m` (upper r0 bound; the tether length in system.yaml still caps r) and its OWN `wind`) feeds `run_uploop_eight_opti.py`; an optional `symmetric_eight:` block (az_center, az_offset, el_center, radius, sense; reelin/ramp fractions optional) is the level eight of its `--shape symmetric` (climbing at both sides, two reel-in windows, symmetric C_phi box; `--symmetric` adds the mirror rows). Data folders of kites whose data must stay private (e.g. `data/LEI-V9-KITE/`) are git-ignored; nothing kite-specific lives in code.

Results layout (convention):

- `results/<kite_name>/<analysis_type>/sim_output.h5` — aerostructural coupled-solver outputs.
- `results/<kite_name>/ekf/` — EKF outputs and diagnostics.

Use the canonical filenames above so helper utilities (e.g., `resolve_kite_paths` in `awetrim.aerostructural.case`) locate files automatically.

