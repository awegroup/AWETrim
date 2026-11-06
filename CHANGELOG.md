# Changelog

## Unreleased

### Tether: inertia of the lumped tether (on by default)

- **The lumped tether carries its inertia** (`inertia: false` in a ROM's or
  as_config's `tether` block, or `run_full_cycle_opti.py --no-tether-inertia`, drops
  it): the inertial load of a straight, inextensible tether of mass
  `m_t = rho A r` to the force at the kite, in the course frame:
  `-(m_t/3)(v_tau_dot + 2 v_tau v_r / r)` tangentially, `-(m_t/3) v_tau w_z` in the
  turn and `-m_t (v_r_dot - v_tau^2 / (2 r))` along the tether
  (`Tether.force_inertia_tether_at_kite_for`). Across itself the tether acts as `m_t/3`
  at the kite, along itself as `m_t`. Quasi-steady, its centrifugal, Coriolis and turn
  terms remain; the dynamic march adds the accelerations. On by default, so
  quasi-steady results shift by about 0.1-0.2 % (LEI-V3 at 9 m/s); not available on the
  Williams tether.
- The dynamic-vs-quasi-steady study (`compare_dynamic_quasi_steady_winch.py`, available
  on request) now runs with and without it, with and without winch friction.

### Dynamic march: implicit Euler, with the reel acceleration

- **The dynamic forward march (`quasi_steady=False`) is now implicit (backward) Euler.**
  A dynamic node solves the quasi-steady node's unknowns `[T, u_s, s_dot, v_r]` with
  the inertia of the step ending at it: `s_ddot = (s_dot - s_dot_prev) / dt` and
  `dv_r/dt = (v_r - v_r_prev) / dt`. The old explicit step (`s_ddot` solved, `s_dot`
  stepped forward) oscillated with a growing amplitude at a cycle's node spacing
  (dt about 0.4 s, longer than the kite's speed response) and broke the march; the
  implicit one is stable at any spacing and costs what a quasi-steady node costs.
- **The kite's radial acceleration was zero in dynamic runs**: the kinematics'
  `dot_vr` keeps only `s_ddot * v_r / s_dot` (`dr_ds2` is identically zero). The
  march now uses `dv_r/dt`, the same for the kite and the winch drum.
- **Second-order march: `sim_parameters.time_scheme: bdf2`**, the default for dynamic
  phases (quasi-steady keeps `euler`); the NLP uses the same scheme. The
  step is `h = 2 ds / (s_dot_prev + s_dot)` and the tether length
  `r = r_prev + h (v_r_prev + v_r) / 2`, both solved in the node, and the dynamic
  march's (and the drum's) accelerations are variable-step BDF2 differences, which damp
  the stiff drum mode as backward Euler does. On a 9 m/s LEI-V3 eight the dynamic
  march's power error falls from +2.9 % (euler) to -0.3 % at 25 nodes and its tension
  error from 148 to 44 N at 100 nodes, for 5-10 % more time; the euler "gain" of the
  dynamic model over the quasi-steady one (+3 % at 25 nodes) was numerical, the
  converged difference is -0.15 %.
- **The reel acceleration is its own kinematics symbol** (`timeder_speed_radial`,
  `ParametrizedKinematics.dot_vr`), set by the time scheme in the march and the NLP
  instead of overriding the shared kite model, so quantities that depend on it are
  derived from the recorded states after a dynamic march. The time differences live in
  one module, `awetrim.utils.time_scheme`. The integrator path
  (`run_simulation`, `phase_sim=False`) refuses a dynamic phase.
- **The cycle NLP can be dynamic too** (`quasi_steady=False`, `run_full_cycle_opti.py
  --dynamic`): the node functions take `s_ddot` and `dv_r/dt` as two more inputs, fed
  with the same backward differences as the march (node 0's predecessor is node N-1
  on a periodic wrap). No rows or decisions are added; quasi-steady, the NLP is
  unchanged. The tension bound then holds with the kite's and the drum's inertia in.
- **The march tells a fold from a solver failure.** When every attempt at a node
  fails, a least-squares solve from the guess either finds a root IPOPT missed (taken
  if the branch rules accept it) or stalls where the Jacobian goes singular: a fold,
  reported with the balance that runs out ("the kite cannot follow the path here")
  and kept in `phase.stop_reason`.
- **Dynamic runs reject post-stall roots again.** The node's angle of attack still
  held `s_ddot`, so its function could not be built and the march's stall check was
  silently off in dynamic runs; the implicit `s_ddot` now goes into it as into the
  force balance, and the march warns whenever that function cannot be built.
- **The march checks every node attempt against the steering box**, not only node 0
  and retries: from the previous node's values a first attempt can converge on the
  other (high-steering, low-lift) root, which the dynamic 9 m/s cycle did at its
  reel-out transition (u_s 0.85, the physical root at -0.10). Applies to the
  quasi-steady march too.
- **Node 0 is the quasi-steady solve**, so a dynamic run no longer starts from a
  separate quasi-steady start state, which on a cycle starting in reel-in landed on
  another root.
- `TimeSeries.to_dataframe` derives the `State` fields the march leaves unset
  (`speed_tangential`, the angles) instead of returning them all-NaN.
- A study comparing the full dynamic model with the quasi-steady one, drivetrain
  friction in both (`compare_dynamic_quasi_steady_winch.py`, available on request). A dynamic kite with the drum's inertia cannot follow an eight
  sized at the quasi-steady turn limit (margin 1.2; it flies at 1.6), and on the
  9 m/s cycle optimum it jumps to the low-lift root at the reel-out transition,
  which the march still accepts.

### Winch: drivetrain inertia and friction, simulated and optimised

- **`awetrim.system.winch.Drivetrain`**: the drum between the tether and the motor,
  tether-referred: inertia `m = J (n / r_drum)^2` and friction
  `F_f = (F_c + k T) tanh(v / eps) + c_v v`. The winch force law now sets the MOTOR
  force, as a torque-controlled machine does, so the kite feels
  `T = F_law(v) + F_f(v) + m dv/dt`. The ideal drivetrain (the default) leaves every
  run unchanged.
- **The forward simulation flies it**, without optimising: an implicit Euler step of the
  drum in each node's law row, the drum's previous state passed to the node solver.
- **The cycle NLP uses the same model.** Its objective is the shaft power
  `T v - P_loss - d(m v^2 / 2)`; `force_law` gets the drum in its law row, `free_speed`
  holds the motor force below `max_tether_force`. Drivetrain friction in the objective
  comes from `feat/winch-drivetrain-losses`.
- **LEI-V3 values**: V3Kite.jl's GS1 drivetrain (`sim_settings_reelout.yaml`):
  0.204 kg m2 motor side, i.e. 300.7 kg at the tether, and 122 N + 30.6 N s/m.
  Switched on by `sim_parameters.winch_drivetrain` (or `run_full_cycle_opti.py
  --winch-drivetrain`), or part by part.

### Tether: second-order load placement in `WilliamsTether`

- **Each segment's load now acts at its centre and goes half to each end node**
  (Nguyen & Oland 2026, *Aerospace* 13, 943, App. A), the kite and the anchor taking a
  half each. Williams' (2017) lumping, one segment per interior node and none at the
  kite, is first order: the kite's share of the tether drag is 1/4 (1 - 1/n)^2, against
  1/4 - 1/(8 n^2) now. All of them reach the same limit.
- **`tension_tether_kite`, the direction unknowns and `kite_tension_vector` now state the
  force at the kite**, which differs from the last segment's tension by the kite's half.
  `tether_force_ground` is the anchor's reaction. Row k of `velocities_apparent_wind` and
  `angle_va_tether` is segment k at its centre, with new `*_per_segment` outputs;
  `*_per_node` are the nodes' shares and sum to the whole tether.
- **`n_elements: 10` now carries the right drag.** Stored coupled trims and the ROM
  identified on them used the old loads, which read tension about 9 % and speed about 5 %
  high at 10 elements. They change when the sweep is re-run.
- The frozen CasADi trim reference (`tests/aerodynamics/data/casadi_trim_aoa_false_reference.npz`)
  was regenerated for its Williams-tether entries; the tetherless ones are unchanged.

### Trajectory optimisation: seed fitted to the kite, on every wind

- **`fit_periodic_cycle_config.py --size`** fits the seed path to the turns the
  kite holds at the wind, in about a minute, at the first margin (1.4 to 2.0) whose
  march is complete and inside the steering limit at the kite's own roll gain. It
  needs neither `--auto` nor the roll-gain assist, and `--size` replaces `--auto`
  in the documented commands. LEI-V3 at 8 m/s: 12.92 kW in 375 s, against 12.78 kW
  in 1432 s through `--auto`.
- **New `run_cycle_power_curve.py`**: outward from a start wind, cold until a wind
  stalls, then each wind from the last converged one. LEI-V3 (explog profile, wind
  at 6 m): converges from 4 to 12 m/s, cold up to 8.5 m/s, about 13.3 kW above 9 m/s;
  3.5 m/s does not converge either way. The plateau scatters by about +-0.3 kW:
  these are local optima, and continuing gives 0.15 to 0.2 kW more than a cold solve
  at 8 and 8.5 m/s.
- **The reel-out server retries a first solve that does not converge**, each solve
  capped at 300 iterations: seed at margin 1.4, then a cold solve at a neighbouring
  wind stepped warm to the one asked, then the configured seed. LEI-V3 (log profile,
  z0 0.03 m, 3.5 to 12 m/s): per-node depower converges everywhere and holds the
  rated 29.4 kW from 6.5 m/s up, but for 29.13 kW at 11 m/s; one depower value does
  not converge at 11.5 and 12 m/s, and falls to about 23 kW above 10 m/s.
- **`sim_parameters.slow_path_rate`**: the march of a path parameter spanning a whole
  cycle keeps its warm start. Off for reel-out, whose solves are unchanged.
- **The tracked LEI-V3 cycle seed is regenerated with `--size`.**
- **`helix_pattern.py` and `downloop_pattern.py` size their path the same way**
  (`timeseries.seed_sizing.size_reelout_seed`, shared with the server), and report a
  failed solve. The LEI-V3 helix at 10 m/s now converges in 47 iterations (29.12 kW);
  its fixed path could not be flown on the new ROM.
- **Removed `fit_periodic_cycle_config.py --auto`**: `--size` does its job, in a
  minute instead of 15 or more. It is last at commit 8d88e40.
- **Removed `--shape experimental`**, the seed fitted to one flown cycle: too rough
  to trim, refused by the validator on the current ROM, and in need of a flight file
  that is not in the repository. Last at commit 8d88e40. `cycle_profile.yaml` keeps
  `flight.year`, which picks the depower match; its other `flight` keys are ignored,
  and `m_per_second` now raises like any unknown key.

### LEI-V3 re-identified on the realistic stall branch

- **New `rom_config_aerostructural.yaml` and flight-corrected ROM** (the latter not
  reviewed yet), on the merged sweep and VSM changes and the kcu/100 steering scale.
  Band 1.50-2.60 m, fitted from 1.40 m. Against the shipped ROM's tension at 5 m/s wind:
  +23 % at 1.50 m, +4 % at 1.95 m, +3 % above 2.10 m.
- **The sweep settles every state on the stall branch marched down**
  (`settle_on_down_branch`, on by default) and marches its polars past the stall and back
  (`polar_d_alpha_deg` to +22 deg). Marching up stays attached about 6 deg past it.
- **The stall angle is measured, not fitted**: the median 5 % lift-drop onset of the
  states near their stall (`fit.stall_onset_tapes`), 10.27 deg. The coefficients are fitted
  up to 8 deg past it, on states below `fit.max_stalled_fraction`. Past the stall the ROM
  no longer reproduces the lift drop.
- **`validity.angle_of_attack_deg` now states what may be flown**: its upper end is the
  stall less `fit.stall_margin_deg`, 7.27 deg, not how far the data reach. The optimiser's
  alpha room at the powered end is small: the 1.50 m trim sits near 6.5 deg.
- **Steering chains run past the stall stop**, to 0.50 m or a state that does not converge
  (`steering.max_stalled_fraction`). At 1.50 m they end at a failed trim at 0.45 m.
- **The roll gain depends on the steering**: `gain_roll_steering_squared: [s0, s1, ...]`,
  `k + s(u_p) u_s^2`, one formula in `kite.roll_gain`. Absent = as before.
- **The trim's Newton solve returns its best iterate**, not its last, which ended six chains
  at a residual it had already passed.
- **The flight correction is now the tapes and one gain scale.**
  `identify_rom_flight_correction.py` identifies `flight_match`, 1.95 / 2.48 m, and a scale of
  1.17 on the roll gain, on the ROM's own solve with the logged steering. The fits of
  `theta_b` and `dC_D` and `refine_rom_flight_output_error.py` are gone: reel-in tension on
  the held-out cycles went from +32 % to +16 %, reel-in speed from +7 % to +10 %.
- New `plot_identification.py`. A fit with no polar
  sample left now stops. The geometry-cache move retries when Windows refuses it.
- `docs/identification/rom_aerostructural.tex` rewritten around the method.

- **The 2019 steering scale is `u_s = -kcu/100`, as in 2025** (was `/200`): the
  flown +-35 % is 0.49 m of half-difference, `u_s` +-0.35. With it, the zero offset
  is -0.0329 m of tape (was -0.01645 m, the same in KCU counts), `system.yaml`'s
  steering range +-0.35 and rate +-0.29 /s, and `DEFAULT_OPTI_LIMITS` +-0.35. The
  ROM's `validity.input_steering_abs_max` still clips the optimizer's box. Anything
  fitted to 2019 steering on the old scale is stale:
  `rom_config_aerostructural_flight_corrected.yaml` (roll gain, steering drag) must
  be redone.

- **VSM is pinned to the release `v2.3.0`** instead of the develop commit `ae5515c`:
  `src/VSM/core` and the solver are unchanged between the two.

### Continuation of the wireframe solve

- **`wireframe.driver.solve_step` continues from the solve before it**: the
  prepared structure carries the last converged trim and circulation, and
  `restore` reads them back from the file (`gamma_distribution` is new in
  `sim_output.h5`). Results of a chain change: on `w5.0_depower_2.60` tether
  force 0-2.3 % higher than the cold starts, with 33 % fewer coupled iterations.
  A stated `quasi_steady_initial_guess` still wins; another flight state starts
  cold.
- **One CasADi trim graph per chain**, not two per state: the trim bounds are out
  of the cache key and the system model is reused while the flight state holds.
- **The LEI-V3 `as_config.yaml` sets `coupling.trim.solver: casadi`.** A config
  without the key stays on `least_squares`.
- **A cold CasADi trim no longer diverges**: its true-polar stage starts from the
  attached predictor, a failed Newton solve retries from it, and `max_nfev` no
  longer passes a Newton failure on as a success.
- **A failed CasADi trim now fails**: `run_vsm_package` raises `CasadiTrimFailed`
  instead of taking one `least_squares` trim. A trim just short of its root is
  taken as a step, flagged `trim_inexact`, and can only end the coupled loop below
  `CASADI_FINAL_RESIDUAL`.
- **`run_identification_sweep.py` survives a failed trim**: it ends that chain,
  records the step in `trim_failed.txt`, lists it at the end of the run and goes on
  with the next chain.
- **The LEI-V3 trim attitude box is +-30 deg**, one box with no `pre_loop` block
  (was +-4 deg in the loop, +-15 deg before it): the first trim of a steered state
  can need more, and with its root outside the box Newton has no solution.

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
- **The LEI-V3 wireframe flies the ROM's wing by default.** `as_config.yaml`
  states `aerodynamics.billowing.contraction: 0.05` and
  `load_transfer: strut_lever` (with its 5 panels per FEM section, 135),
  which only `identification.yaml` used to set. The identification sweep
  is unchanged; `run_simulation_PSM.py` drops its 54-panel base-loop
  override and flies `as_config`'s aerodynamics.
- **Step-1 scripts take Inputs blocks, not a CLI.** `solve_single_state.py`,
  `compute_polars.py` and `compute_stability_derivatives.py` read a `Case`,
  a `SHAPE` (`built`, or the `wireframe` / `billow` result of the same case
  from step 2; `DEFORMED_FROM` for any results folder) and their own
  settings. The kite is `examples/1_aerodynamics/common.load_kite`: the
  structure's wing billowed and meshed as `as_config` says
  (`wireframe.driver.wing_edges`, shared with the wireframe), the full
  bridle, the KCU point as the moment reference, mass/CG/inertia from
  `structure.yml` + `system.yaml`, and as_config's KCU drag, tether in the
  trim and `pre_loop` trim bounds. The overrides of physical quantities
  (`--mass-wing`, `--inertia-*`, `--center-of-gravity`, `--tether-diameter`,
  `--n-panels`) and the stability script's `--deformed-case` /
  `--list-cases` / `--rigid-body-result` are gone. Step 1 keeps its VSM
  loop (base, no artificial viscosity): the least_squares trims stop at
  their seed with as_config's AV loop. Default state: the step-2 PSM
  example's case. Step-1 results change (KCU drag, billowed 135-panel wing,
  tether in the trim).
- **`aerodynamic_vsm.trim_bounds` / `trim_settings`**: the trim search box
  and settings of a config, one place for the coupled and step-1 trims.
- **The wing tip is a valid lifting-line bay.** The FEM tip bay (the curled
  leading-edge tube) was meshed with 64-66 deg of panel sweep: VSM found
  several circulations per tip panel and the trims cycled (CasADi trim: 2 of
  12 realistic LEI-V3 states converged, the tips always stalled).
  `aerostructural.wing_tip.correct_tip_bays`, applied by
  `wireframe.driver.fem_wing`, keeps the tube length and slides the tip
  corner until the bay is swept at most 55 deg (`panel_sweep_deg`,
  `MAX_PANEL_SWEEP_DEG`), and warns about any bay of any kite still beyond
  it; aerodynamic mesh only. Billowing (`billowing.billow_wing`) no longer
  rotates the tip bays rigidly with the outer strut: the leading edge and
  the tube end stay, the tip-bay trailing edge takes a fading share of the
  strut's shift. LEI-V3 at the PSM example's case (CasADi trim, AV on):
  converged, no panel past stall; 21.28 instead of 20.86 m/s, C_D -3.4 %,
  tension +3.8 %. Wireframe step-2 and step-1 results change.
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
