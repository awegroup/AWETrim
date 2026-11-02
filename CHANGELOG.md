# Changelog

## Unreleased

### Breaking changes

Legacy readers and unused modules removed. Files in the old formats raise a
`ValueError` naming the replacement; convert them with v1.0.0.

- **System file.** Only awesIO system files (`components`) are read; the flat
  `lei_v3_system_config.yaml` format and the `aero_coeffs_rom.yaml` fallback
  are gone. The pre-awesIO `control_system.turbine {diameter,
  thrust_coefficient}` block is rejected.
- **ROM parameters.** `angle_pitch_depower_0` / `delta_pitch_depower` (state
  `angle_pitch_tether_0` / `slope_angle_pitch_tether_depower`, opposite sign)
  and the `CS: u_s` roll gain (state `params.gain_roll_steering`) are
  rejected. `Kite.k_steering` is removed (use `gain_roll_steering`), and
  `canonical_aero_params` is now `check_aero_params`.
- **Structure files.** `read_struc_geometry_yaml`, `awesio_structure.to_awesio`
  / `convert_file` and its command line, and the retired file names in
  `case.find_struc_geometry` are removed: `load_structure` reads awesIO files
  only (an in-memory PSM wireframe dict is still translated).
- **`rotate_geometry`** no longer takes a single scalar angle about +y.
- **The operating point is a `Case`.** `awetrim.aerostructural.case.Case`
  holds what a run is: flight state, wind spec, atmosphere (`rho`, `mu`,
  gravity) and the tape command, with no defaults on the flight state, wind or
  command. The drivers (`billow.driver.build_case` / `solve_deformation`,
  `wireframe.driver.build_case` / `build_case_reduced_fem` /
  `solve_deformation` / `solve_deformation_reduced_fem`) take
  `(case, overrides)` instead of `config_overrides`; overrides come in
  as_config's subsystem blocks plus `output`. `CONFIG_DEFAULTS` lost its operating-point
  keys, which also lived (with other values) in `as_config.yaml`, the step-2
  scripts, `identification.yaml` and `aerodynamic_vsm`. Stored results keep
  their layout and still load.
- **`as_config.yaml` has one block per subsystem**: `aerodynamics`,
  `structure` (`wireframe`, `billow`), `tether`, `coupling` (`trim`, `loop`,
  `load_transfer`). The flight state, wind, atmosphere, tape command and the
  plot flags left it, and so did the KCU point (`aerodynamic.reference_point`,
  `structural_pss.fixed_point_indices`): the drivers take it from the
  structure. A flat file raises. `identification.yaml`'s
  `sweep.config_overrides` became `sweep.case` + `sweep.overrides`.
- **ROM tether.** A ROM file's `tether` block now selects the tether of a
  ROM run (`system.factory.load_rom_tether`); before, it was written but
  never read and every ROM run flew the rigid lumped tether. The LEI-V3
  ROMs and `identification.yaml` now state `rigid_lumped`, so their runs are
  unchanged; a ROM file that states `williams` now flies it.
- **Wireframe elongation-bound stiffening removed**, with the in-loop
  modulus ramp and `PreparedWireframe.pin_bridle_modulus`
  (`structural_wireframe.adapt_stiffnesses` / `modulus_stiffness_ceiling`).
  They raised element stiffnesses for the PSM file's ten-times-too-soft lines;
  on `structure.yml`'s datasheet moduli the bound never fires (0.13 % largest
  elongation against 1 %, no update). Its as_config keys
  (`update_stiffness`, `max_elongation`, `max_stiffness_factor`, ...) raise;
  a handover's `stiffnesses` / `stiffness_factors` are ignored with a
  warning; the meta keeps the elongation diagnostics and `final_stiffnesses`.
- **Removed modules:** `utils.fitting`, `utils.sparsity_analysis`,
  `utils.talmar_equations`, `timeseries.cycle_phase`,
  `aerostructural.logging_config`.

## v1.0.1 (2026-11-02)

A bug-fix release: the KCU mass is counted once everywhere.

### Fixes

- **Step-1 stability analysis** (`examples/1_aerodynamics/`). The stability
  script counted the KCU mass twice (30.4 kg instead of 22.0 kg on the LEI-V3)
  and both step-1 scripts trimmed about a fixed CG of (0.5, 0, 5) m instead of
  the kite's own. At the default point this turned the yaw verdict divergent;
  on the kite's own CG and mass every channel is restoring.
  `--center-of-gravity` now defaults to the kite's CG (the structure plus KCU
  aggregate). The mass, CG and inertia come from one body, and a mismatch
  (`--mass-wing`, `--no-struc-geometry`, or `system.yaml` and `structure.yml`
  out of sync) raises instead of mixing two.
- **KCU mass in a structure document.** `system.yaml` is the KCU mass's single
  source and the structure readers add it at the KCU point. A KCU mass also
  written into `structure.yml` was counted twice, silently in the coupled
  solvers. The readers now refuse it.
- Documentation: `examples/README.md` names `structure.yml`; the polar CSV
  comment states that alpha may be in degrees or radians.

## v1.0.0 (2026-10-29)

The first stable release. Since v0.1.0, AWETrim has a new structural solver,
a new default aerodynamic model of the LEI-V3 kite, a fast quasi-steady trim,
a full pumping-cycle optimiser and a co-simulation server. Every example runs
end to end on the public LEI-V3 data.

### Breaking changes

- **Structural solver.** The Particle System Simulator (PSS) and `kite_fem`
  are removed. [Billow](https://github.com/awegroup/Billow), a separate
  package, replaces both. Its *wireframe* fidelity covers the bridle and line
  system; its *full* fidelity adds inflatable tube beams and a wrinkling
  membrane canopy.
- **Structural geometry format.** A kite's structure is now one awesIO
  structure document, `structure.yml`. The old `struc_geometry_FEM_full.yaml`
  and PSM-format snapshots still load with a `DeprecationWarning`. Convert them
  once with
  `python -m awetrim.aerostructural.coupled.awesio_structure OLD NEW`.
- **Default LEI-V3 ROM.** Every LEI-V3 system file now uses
  `rom_config_aerostructural.yaml`. It is identified from aerostructural
  simulations only, on the FEM-reduced wireframe with canopy billowing.
  - The paper's semi-empirical ROM is removed; it remains in git history.
  - The flight-corrected ROM is kept, but it is provisional and not yet
    reviewed.
- **ROM parameter names.** The bridle pitch is now `angle_pitch_tether_0` and
  `slope_angle_pitch_tether_depower`, and the roll gain `gain_roll_steering`.
  The legacy keys still load, with a `DeprecationWarning`.
- **Flight depower.** Flight depower is now mapped onto the ROM's depower
  input `u_p` through the match recorded in `identification.yaml`
  (`flight_match`). The fixed 1.7 m / 2.1 m tape lengths are gone: they were
  the previous model's match, not flown values.
- **Flight data.** Raw-log processing and the EKF live in
  [EKF-AWE](https://github.com/ocayon/EKF-AWE). AWETrim only reads its output
  (`awetrim.flight_data.load_flight`).
- **Single source for hardware limits.** The winch envelope, the KCU mass and
  the onboard turbine come from the kite's `system.yaml`. Cycle configs no
  longer restate them.

### Aerodynamics and trim

- The quasi-steady trim can be solved as one CasADi root-finding problem
  (`trim_casadi.py`), 20-70x faster than the least-squares trims with the same
  result. The coupled loop uses it too.
- New drag contributions in the trim:
  - KCU bluff-body drag, including the onboard turbine;
  - flat-tape bridle-line drag, from a `tape_width` column on the structure;
  - an off-radial Williams tether.
- The apparent wind at a point on the kite has a single source
  (`apparent_wind.py`).
- The bridle angles are taken in the apparent-wind frame. Before this fix the
  bridle pitch was 2-3 deg off in sideslip.
- AWETrim builds every VSM object on its own settings (`vsm_settings.py`),
  never on VSM's defaults. The VSM dependency is pinned to a commit.
- Static-stability tools: neutral point and pitch margins, stability about the
  bridle point, and a 12-state block that includes the kite's position.

### Aerostructural coupling

- The coupled solver is backend-agnostic (wireframe or Billow full). One
  geometry reader and one load transfer serve both backends.
- A chordwise load distribution (`moment_matched`) conserves each panel's
  local pitching moment.
- Canopy billowing and an exact strut-lever load transfer are available.
- Convergence uses a tether-normalised residual, and the steering state is
  tracked through depower and steering continuation chains.
- `examples/2_aerostructural/` is the public demonstrator: wireframe and
  Billow on the same geometry.

### ROM identification

- A public step-3 pipeline: `run_identification_sweep.py` builds the dataset,
  then `identify_rom_aerostructural.py` fits the ROM. Both read the kite's
  `identification.yaml`; nothing kite-specific is in the scripts.
- The roll gain can depend on depower,
  `k(u_p) = gain_roll_steering + k1 u_p + k2 u_p^2`.
- A ROM may state its validity: a depower band, a steering limit and a tension
  floor. The optimiser clips its bounds to them.
- The body axes are anchored to the wing's centre panel.
- The 2019 flight-correction scripts are in `examples/5_flight_data/`. The
  correction itself is provisional.

### Trajectory optimisation

- **Full pumping-cycle optimisation.** One periodic B-spline covers the whole
  cycle, with:
  - a free reel speed (`free_speed` winch mode) and a free operating radius;
  - a per-node depower profile;
  - staged solves: close the cycle, then maximise power, then polish.
- **Seeds.** `fit_periodic_cycle_config.py --auto` builds a seed for any kite
  folder; Dubins paths on the sphere can design the reel-in.
- **Roll-gain continuation.** A seed the kite cannot turn through at its own
  roll gain is marched at a scaled gain and continued back to 1 before the
  power stages. Kites with little roll authority start from the reel-in top
  as well.
- **Control smoothness.** A smoothness term, sized relative to the objective
  (`sim_parameters.reg_fraction`), selects the smoothest of the near-equal-power
  cycles.
- **Up-loop figure-eight study** (`optimization/studies/`).
- **Reel-out options:** a minimum turn radius, a stall margin and
  height-dependent air density.
- **Steering metric.** The number of steering reversals per cycle is a
  first-class cycle metric.

### Co-simulation server

- A REST API for reel-out trajectory optimisation (FastAPI, optional `[server]`
  extra), with clients in Julia, MATLAB and Python.
- Each request can set the depower mode, the winch law (including an
  optimisable gain), the inflow profile and air density, a minimum turn radius,
  pattern limits, angle-of-attack limits and the ROM.

### Environment

- Wind profile laws (uniform, log, power, explog, jet, tabulated) live in one
  place, `profile_laws.py`.
- Optional height-dependent air density.

### Dependencies

- Billow, VSM, vplot3d and EKF-AWE are pinned to commits in `pyproject.toml`.

## v0.1.0

First public version.
