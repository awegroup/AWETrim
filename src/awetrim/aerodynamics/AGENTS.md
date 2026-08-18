# AWETrim Aerodynamics Module

## Scope

This module owns aerodynamic analysis and trim functionality that is not part of
the point-mass `system/` equations themselves.

The first accepted interface is the **VSM aerodynamic quasi-steady trim** surface
transferred from `Vortex-Step-Method/src/VSM/quasi_steady_state.py`. It covers:

- rigid aerodynamic VSM trim,
- aerodynamic force and moment residuals,
- aerodynamic stability derivatives around a trim state,
- parameter sweeps and plotting/dataframe helpers.

This module is not the aerostructural coupling module. Do not put PSM, structural
deformation, FSI iteration, or ASKITE coupling code here.

## Boundary

- AWETrim does not vendor or reimplement `VSM.core` solver internals.
- VSM bodies and solvers enter through protocols or optional runtime imports.
- AWETrim `SystemModel` supplies course-frame kinematics, apparent wind,
  inertial force, gravity force, and wind/kite velocity.
- Cross-module data uses dataclasses or plain dictionaries.
- No CasADi symbolic objects cross this module boundary; values are numerical
  at VSM trim solve time.

## Public Source Layout

```text
src/awetrim/aerodynamics/
  __init__.py
  AGENTS.md
  protocols.py
  vsm_quasi_steady.py
  vsm_adapter.py
  parametric_geometry.py
  parametric_airfoil.py
```

`parametric_airfoil.py` owns a dependency-light (numpy only, **no VSM/CasADi**)
parametric **2D airfoil-section** generator — the section-level counterpart to
the 3D-planform `parametric_geometry.py`. It builds a closed LEI kite profile
from cubic Bezier curves controlled by six design parameters (tube size, max
camber position/height, TE reflex, camber tension, LE tension), following the
Masure regression parametrisation. Public API:

- `LEI_airfoil` — low-level constructor returning the full bundle of curves,
  control points and curvature arrays.
- `generate_profile` — high-level wrapper returning `(all_points, profile_name,
  seam_a)` for a closed contour.
- `save_profile_as_dat_file` / `reading_profile_from_airfoil_dat_files` — `.dat`
  write/read round-trip.

Section plotting lives in `awetrim.plotting.plotting.plot_lei_airfoil` (pass
`show=False` for headless runs), keeping this module pure geometry.

`parametric_geometry.py` owns a dependency-light (numpy + yaml, **no VSM/CasADi**)
parametric 3D wing-planform representation. It reads/writes the same
`wing_sections` table that `vsm_adapter.py` consumes, so generated geometries
drop straight into the VSM trim/sweep path. Public API:

- `WingSections` — QC-anchored full-wing planform with `from_aero_geometry` /
  `from_yaml` / `to_aero_geometry` / `to_yaml` and planform metrics
  (`aspect_ratio`, `anhedral_angle_deg`, `taper_ratio`, `tip_twist_deg`, `area`,
  `projected_span`, `flat_span`, `mean_chord`).
- `morph_wing` — direct QC-anchored morph by `span_scale`, `chord_scale`,
  `anhedral_scale`, `taper_ratio` (area-preserving), `twist_deg`.
- `morph_wing_to` — solve scales to hit `target_aspect_ratio` /
  `target_anhedral_deg` (area-preserving by default), plus decoupled
  `taper_ratio` / `twist_deg`. Taper and twist are independent of aspect ratio
  and anhedral.

`parametric_geometry.py` also owns `build_swept_wing` — a scalar-DOF constructor
(span, root_chord, taper, `sweep_deg`, anhedral, twist) that builds a positioned
`WingSections` from scratch (used by the swept-AWES design tool, not by morphing
an existing planform), plus the `sweep_angle_deg` metric and a `sweep_deg` knob
on `morph_wing`. Sweep is quarter-chord sweepback (aft shift ∝ |spanwise offset|)
and is area/AR/anhedral-neutral, like twist and taper.

Area/span conventions are documented in the module docstring (flat/developed
area, flat aspect ratio). Inspired by the QC-anchored shape-variation generator
in `jellepoland/WES_aero_sim_for_kite_design`.

The **swept, KCU-below AWES design tool** that consumed these primitives — the
`SweptWingDesign` design vector, the first-order `weight_model`, `evaluate_design`
(weight → trim → stability → steering), `place_wing_for_target_aoa`, the
stability-derivative / eigen-mode interpreters, `DEFAULT_DESIGN_LIMITS`, and the
design scripts — now lives in the separate **AWEDesign** repo (package
`awedesign`, https://github.com/awegroup/AWEDesign), which imports the AWETrim
primitives below. **Do not re-add the design layer here**; add new *primitives*
here (and re-pin AWEDesign), keep the design/orchestration layer in AWEDesign.

Primitives kept here for AWEDesign (and general use):

- `parametric_geometry.build_swept_wing` — scalar-DOF swept planform constructor
  (documented above), plus the `sweep_deg` / `sweep_angle_deg` additions to
  `morph_wing`.
- `vsm_quasi_steady.solve_vsm_quasi_steady_trim(..., applied_moment_nm=...)` — the
  gravity-free quasi-steady trim, with an optional external moment (e.g. a KCU
  steering roll moment, in the `[course, normal, radial]` basis) added to the CG
  balance so the resulting bank and turn come out as outputs.
- `vsm_quasi_steady.turn_radius_vs_steer_moment` — sweeps an applied roll moment
  and returns bank / aero-roll `phi_a` / turn radius / speed / tether; the
  design-tool analogue of the point-mass `angle_roll_aerodynamic = k_steering · u_s`
  law in `awetrim.system.kite`.
- `vsm_quasi_steady.compute_vsm_trim_stability_derivatives` — the trim
  linearisation AWEDesign interprets into named derivatives and eigen-modes.
  Takes the same optional `gamma_seed` as the Williams trim solver: it seeds
  the BASELINE solve at the trim state (branch selection); the baseline's
  converged circulation then warm-starts every finite-difference solve as
  before. Pass the seed the trim was solved with, so the linearisation sits
  on the same gamma branch as the equilibrium it linearises.
  Inertia enters either as principal scalars `inertia_xx/yy/zz` or as the full
  3x3 CG tensor `inertia_cg` (zero-attitude geometry basis, same convention as
  `solve_vsm_quasi_steady_trim`'s `inertia_cg`), which overrides the scalars
  and keeps the products of inertia. Its result dict always includes
  `nonlinear_rhs`, a callable `f(delta_state) -> xdot` for the nonlinear
  9-state fast subsystem, assembled directly from the governing equations
  (independent of `A_full`): `f(0)` is the trim equilibrium residual and
  central-differencing `f` cross-checks `A_full`.
  Used by `scripts/personal/wes-quasi-steady/verify_linearization.py`.
- `vsm_quasi_steady.corotating_state_transform` — constant unipotent map `T`
  from the linearisation's native FROZEN stability axes to the co-rotating
  course axes (course axes at trim, carried by the body about B — the paper's
  reporting convention; distinct from the principal-body-axes stability-frame
  option). `A_corot = T A inv(T)`, `vec_corot = T vec`; eigenvalues and
  margins invariant, only A-entries and mode participation change. The result
  dict of `compute_vsm_trim_stability_derivatives` carries the assembled
  matrix (`T_corotating_from_frozen`) and its inputs (`v_kite_trim_axes`,
  `omega_c_axes`); `stability_common.linearise_trim` in the wes-quasi-steady
  scripts applies it by default (`state_basis="corotating"`). The map is
  state-name-list agnostic: passing `AUG_STATE_NAMES` extends it with
  identity position rows.
- `vsm_quasi_steady` position-augmented block (`position_states=True`,
  keyword of `compute_vsm_trim_stability_derivatives`): additionally
  linearises the 12-state set `AUG_STATE_NAMES = ALL_STATE_NAMES + (x, y)` —
  tangential kite positions relative to the (rotating + reeling) trim
  reference point, frozen stability axes. New physics in the extra columns:
  the tether at the laterally displaced `r_kite` (Williams fixed-length
  re-solve, now taking a 3-vector offset with the full VSM→course→wind
  chain on input AND wind→course→VSM on output; analytic straight-tether
  tilt `-(T/r)(1 - r_hat r_hat^T)` as the non-Williams fallback) and the
  kite inflow at the displaced height (`wind.speed_wind_at_height` shear;
  the augmented block's own z column carries it too, while the historical
  `J_full`/`A_full` z column stays shear-free). Position kinematics carry
  the frozen-axes transport `delta_r_dot = delta_v - Omega_C x delta_r`;
  gravity/course-orientation columns are zero by construction and `Omega_C`
  stays frozen. Requires `system_model` + positive `distance_radial`
  (`ValueError` otherwise); default `False` leaves every historical output
  byte-identical. Outputs: `J_aug`, `A_aug`, `eig_aug`, `vec_aug`,
  `Tfast_aug`, `stable_aug`, `state_names_aug`,
  `T_corotating_from_frozen_aug`, `tether_position_model_aug`,
  `eps_position_lateral_used`, and the independent verification field
  `nonlinear_rhs_aug` / `nonlinear_rhs_aug_full` (12-state analogue of
  `nonlinear_rhs`, cross-checked by
  `scripts/personal/wes-quasi-steady/verify_linearization.py
  --position-states`). Consumers: `stability_common.linearise_trim`
  (`position_states=True` forwarded; adds `is_lat_aug`,
  `participation_aug`, `alpha_lon_aug`, `alpha_lat_aug`) and
  `run_modal_stability.py` (`--position-states`, default on: 10-vs-12
  comparison table, `modal_stability_position_states.pdf`).

  `tether_lateral_feedback` (default `True`) governs whether a LATERAL offset
  reaches the tether at all. Set `False` and the tether translates WITH the
  kite: only the RADIAL component of the position offset is passed to the
  tether model, so the `x`/`y` columns carry no tether force and need no
  re-solve, while the `z` channel is untouched. Under a uniform wind those
  columns then vanish IDENTICALLY (aerodynamics, gravity and the centrifugal
  term are all blind to a tangential displacement), which removes the
  spherical-pendulum restoring term `~T/r` and with it the position-born
  pendulum modes. Use it when the pendulum is taken to belong to the slow
  trajectory subsystem, which this frozen-slow-state block would otherwise
  double-count. The projection sits in `eval_force_moment`, so the FD columns
  and `nonlinear_rhs_aug` stay consistent by construction. Recorded as
  `tether_lateral_feedback` and as a `_radial_only` suffix on
  `tether_position_model_aug`; `run_modal_stability.py
  --frozen-lateral-tether`.

- `vsm_quasi_steady` course-rate block (`course_rate_state=True`, keyword of
  `compute_vsm_trim_stability_derivatives`): the **index-1 DAE** form of the
  fast subsystem. The course frame is defined by the velocity direction, so the
  kite velocity is `[v_tau, 0, v_r]` by construction and has no normal
  component to integrate; the lateral DOF is the relative turn rate
  `chi_dot_turn`, which is an *algebraic* variable, not a state — `a_n =
  -v_tau chi_dot_turn` determines it and the rigid-body equations produce no
  `chi_ddot_turn`. States are `CHI_STATE_NAMES = ALL_STATE_NAMES` minus `v`
  (9 differential). One extra FD column `J_course_rate = d(F, M) /
  d(chi_dot_turn)` is taken by perturbing the trim course rate, which moves
  `Omega_C`'s RADIAL entry only (it is exactly `-chi_dot_turn`; the normal
  great-circle entry `v_tau/r` is untouched at fixed `v_tau, r, beta, chi`).
  **That perturbation must reach the transport inertial force `-m (Omega_C x
  v)` and NOTHING else** — `Omega_C` is the rate of the course FRAME, and the
  body rate equals it only AT TRIM; `chi_dot_turn` is a trajectory-curvature
  term that changes no apparent wind, so it must not enter the aerodynamic
  body rate, the gyroscopic couple, or the centripetal CG-offset force.
  (Routing it into `omega_total` inflates `G_Omega` by 32-350 % of pure
  aerodynamic garbage and fabricates a spectrum shift.) The constraint is the
  vanishing frame-relative normal acceleration, eliminated as
  `delta_chi_dot_turn = chi_turn_gain_row @ delta_x` (`= -G_Omega^-1 G_x`) and
  substituted back to give `A_chi = F_x - F_Omega G_Omega^-1 G_x`.

  **Two exact results, both verified in the tests.** (1) `G_Omega = -v_tau`
  exactly — pure kinematics, no aerodynamic content; that is the sharpest
  check that the turn rate is not leaking into the body rate. (2) `F_Omega`
  VANISHES on every differential row, so the elimination is a no-op and
  `A_chi` is identical to the 10-state block with `v` pinned to zero. Reason:
  `d(Omega_C x v)/d(chi_dot_turn)` is purely NORMAL, and that force acts at
  the CG, so its right-hand side is the pair `[P, x_cg x P]` which the coupled
  rigid B-point mass matrix maps to `[P/m, 0]` — pure translation, zero
  angular acceleration. Note the raw column's MOMENT rows are *not* zero
  (`J_course_rate = [0, -m v_tau, 0, ...]` with `x_cg x P` moments); they
  cancel against the `-m[c]x` coupling in `M`. Equivalently: about the CG the
  d'Alembert force has zero moment arm. Only `include_added_mass=True` breaks
  the cancellation (M is then not the rigid B-point matrix).

  **Two readings of the same closure, both built.** One normal momentum
  equation, two unknowns `(v_dot, chi_dot_turn)`: either the frame is frozen
  and the equation integrates `v_dot` (the baseline 10-state block, where
  `chi_dot_turn` is only an output — see `chi_turn_gain_row_full`), or the
  frame FOLLOWS the velocity direction and the equation determines
  `chi_dot_turn`. The latter splits again on whether a standing normal offset
  is also forbidden. `A_chi` (9 states) closes on `v = 0`; **`A_chi10`
  (10 states, the default reported by `run_modal_stability
  --course-rate-state`)** closes on `v_dot = 0` only, so the frame carries an
  initial offset along. Its `v` ROW is identically zero ⇒ block upper
  triangular ⇒ `spec(A_chi10) = spec(A_chi)` plus one structural zero (the
  neutral constant-sideslip mode), while the `v` COLUMN survives so a standing
  sideslip still drives the other nine rows aerodynamically — the difference
  is in the eigenVECTORS, not the spectrum. Extra outputs `A_chi10`,
  `eig_chi10`, `vec_chi10`, `Tfast_chi10`, `state_names_chi10`,
  `chi10_gain_row` (== `chi_turn_gain_row_full`). `--course-rate-pin-v`
  reports the 9-state block instead.

  So the block's value is that it is the *correct* 9-state model and that
  `chi_turn_gain_row` is the per-mode turn-rate OUTPUT — not a feedback.
  Contrast with substituting `delta_chi` for `v`: that is an exact change of
  coordinates (`delta_chi = delta_v/u0`, absorbed by yaw, `psi_free =
  psi_frozen - delta_chi`) and leaves every eigenvalue invariant. The tether
  is held at its baseline across the column, which is exact for a straight
  tether: it runs along `e_r` and the perturbation is a rotation ABOUT `e_r`,
  moving no node (only sag responds, at second order). Outputs: `J_course_rate`, `A_chi`, `eig_chi`, `vec_chi`,
  `Tfast_chi`, `stable_chi`, `state_names_chi`, `chi_turn_gain_row`,
  `chi_turn_denominator`, `chi_turn_closure_singular`, `eps_course_rate`
  (default `0.02` rad/s), plus the independent verification fields
  `nonlinear_rhs_chi` / `nonlinear_rhs_chi_full` (the 9-state field that
  Newton-solves the normal equation per evaluation and reports
  `delta_chi_dot_turn` / `normal_residual`). Default `False` leaves every
  historical output byte-identical; `ALL_STATE_NAMES` is unchanged.
  Sign note: `chi_dot_turn` is in the trim vector's `timeder_angle_course`
  convention. The post-processing helper `stability_common.course_vs_yaw_ratio`
  defines `delta_chi = dv/u0`, which is the **opposite** radial sense —
  magnitudes agree, signs do not. Do not mix the two.

- `vsm_quasi_steady` transport-rate convention (`transport_rate_follows_states`,
  keyword of `compute_vsm_trim_stability_derivatives`, **default `True`**):
  `Omega_C = [0, v_tau/r, -chi_dot_turn]` in course components. The **normal**
  entry `v_tau/r` is a KINEMATIC IDENTITY — the rate at which `e_r` tilts as
  the kite flies along the sphere — so it follows the fast states `v_tau`
  (state `u`) and `r` (state `z`) in every column. It used to be frozen at the
  trim value, which set the derivative of the centrifugal term
  `m v_tau^2 / r` w.r.t. those states to exactly ZERO; it is now the correct
  `2 m v_tau / r`. Eigenvalue impact measured at LEI-V3 states A/B/C: 0.1-0.6 %
  on `alpha_lon` / `alpha_lat` / `tau_fast`. Pass `False` (or
  `--frozen-transport-rate` in `run_modal_stability.py`) to reproduce pre-2026-08-13
  results. The **radial** entry is dynamics (the turn rate) and stays frozen
  unless `course_rate_state=True` solves for it.
  Two implementation traps: (1) `_omega_c_for` rebuilds the components directly
  from the trim vector — re-calling `_course_transport_rate_axes` with a
  perturbed speed would ALSO move the radial entry via `chi_dot_gc(v_tau, r)`
  and contaminate the turn rate, so the two entries must be perturbed
  independently; (2) the perturbed `Omega_C` feeds the transport inertial force
  ONLY, never the aerodynamic body rate / gyroscopic couple / centripetal
  CG-offset force. The reduced (radial-only, no-`system_model`) branch kept one
  factor of `f_transport[2] = m v_tau^2 / r` frozen at the trim speed — halving
  that derivative relative to the full branch — and was fixed 2026-08-13; all
  three branches now agree. It is unreachable in production (every script
  passes a `SystemModel`) and was reached only by test mocks.

- `vsm_quasi_steady` turn rate as an OUTPUT of the baseline block
  (`chi_turn_gain_row_full`, **always present**, no keyword, no extra solve).
  `chi_dot_turn` is not a state of the 10-state block, but it is a linear
  functional of one: the same normal equation read on the FREE-`v` block gives
  `delta_chi_dot_turn = -(normal acceleration row) / G_Omega`, i.e. one row of
  `A_full` scaled — equivalently `delta_v_dot / u0`, so on an eigenmode it is
  `lambda * delta_chi`, the mode's course excursion times its own rate. The
  denominator is the FD-measured `G_Omega` when the course-rate block ran and
  the exact kinematic `u0 = -v_tau` otherwise; `chi_turn_denominator_source`
  records which (they agree for the rigid mass matrix but **not** under
  `include_added_mass=True`, where only the measured one is right). Because it
  is a functional and not a state, it slices under a DOF reduction (pinned
  states drop out) and transforms with `inv(T)` under the co-rotating change of
  basis — `stability_common.linearise_trim` does both and adds the per-mode
  magnitude `chi_turn_content_full` (rad/s per unit-norm eigenvector, printed
  by `run_modal_stability.py` and annotated under the participation matrix).
  Restricting the row to `CHI_STATE_NAMES` reproduces `chi_turn_gain_row`
  exactly; the test pins that so the two paths cannot drift. Same sign
  convention caveat as above.

- `cg_eom.py` — the CG-form equations of motion: EoM written at the centre of
  gravity, attitude perturbations rotating **about the CG** so the tether
  attachment B swings and the tether (length measured to B) is re-solved at
  the displaced attachment, supplying the restoring moment through its arm
  `-c_att x F_t`. Mass matrix is block-diagonal `diag(m 1, I_cg)`; the B-form's
  centripetal offset force is absorbed in the transport term via
  `v_cg = v_B + omega x c_att`; gravity/transport carry no moment. A B trim is
  a CG trim identically (`sum F = 0` makes the moment transport exact). The
  evaluator itself is a closure of `compute_vsm_trim_stability_derivatives`
  (result key `"cg_eom_eval"`, lazy — no cost unless called) so it shares the
  trim frame chain, VSM warm start and Williams tether closure with the B
  form; `cg_eom.py` holds the docs plus `verify_cg_trim`, `pitch_sweep`,
  `plot_cg_pitch_forces`, `plot_pitch_moment_sweep`. Driven by
  `run_modal_stability.py --cg-eom-check` (side/front-view force figures +
  pitch/roll moment breakdowns per state; sweeps are channel-generic).

  `cg_eom_eval` also takes `delta_course_rate` (default `0.0`), with the same
  routing rule as the B-form `eval_force_moment`: it perturbs `chi_dot_turn`
  in the **transport inertial force** `-m (Omega_C x v_cg)` only — never the
  aerodynamic body rate, never the gyroscopic couple (see the two
  implementation traps above). Because the CG transport acts on the full
  `v_cg`, its force response differs from the B form's by
  `-m dOmega x (Omega_C x c_att)` (a few percent for LEI-V3). Same
  `timeder_angle_course` sign convention as the course-rate block (see the
  sign note there — do not mix with `delta_chi = dv/u0`). A coordinated turn
  perturbation composes it with `omega_perturb = -delta_course_rate *
  e_radial`. Consumer: `scripts/personal/wes-quasi-steady/
  run_static_stability.py` (six-channel static-stability sweeps).

  `static_slopes_summary(stability)` — the cheap static-stability verdict
  (~20 warm VSM solves via `linearise_cg_eom`, no sweeps): six tangent
  slopes (roll/pitch/yaw stiffness, v_tau speed stability, radial tether
  stiffness, coordinated chi_dot turn damping with its kinematic/body-rate
  decomposition) plus restoring/damping booleans. Restoring iff slope < 0;
  chi_dot damping iff slope > 0 (`timeder_angle_course` sense — a positive
  turn rate rotates about `-e_radial`). Attitude axes default to the course
  frame; pass principal body axes (rows, world components) for the
  body-axes trio TOGETHER WITH `euler_rate_matrix`: the attitude columns
  are per EULER ANGLE of the `R_yaw R_pitch R_roll` composition, so a
  rotation about a non-frame axis `b` has Euler tangents `E^-1 b` (columns
  of E: `[R_yaw R_pitch e_x, R_yaw e_y, e_z]` at trim attitude); raw axis
  components misattribute the pitch column into roll/yaw at nonzero trim
  attitude (35% on the B-form roll at state A). Helpers:
  `attitude_moment_matrix` (K [3x3], world moment per rad about each frame
  axis) and `attitude_slope_from_lin` (one column in moment units; also
  valid on the rate columns, e.g. the yaw-damping reference).
  `pitch_neutral_point(dM_dtheta, dF_dtheta, pitch_axis=, x_axis=,
  cg_offset=, chord=)` — pure moment-transfer helper (numpy only, no VSM):
  from one ANGLE-OF-ATTACK-perturbation response pair (moment about B +
  force, world components per rad of aoa, any contributor subset) it
  returns the neutral point `x_np = S_B/D` along the `x_axis` line through
  B (`S(d) = S_B - d D`, `D = e_p . (x_hat x dF)`) and the pitch stability
  margins of B and of the CG, signed positive = restoring,
  `margin(pivot) = -S(pivot)/|D|` (metres, plus chord fractions). The aoa
  perturbation is an apparent-wind tilt at frozen attitude (`delta_v` =
  `va - R(e_p, -eps) va` through `eval_force_moment` for alpha = +eps),
  NOT a body rotation: a body rotation gives the same pitch-axis moment
  slope (`e_p x M0` is orthogonal to the readout) but adds the rotating
  trim force `e_p x F0` to the force slope, corrupting `D`. Alpha is
  sense-matched to the readout axis (the body-equivalent rotation about
  `pitch_axis`), so restoring iff slope < 0 and all outputs are invariant
  under flipping `pitch_axis`. A pure aoa disturbance moves ONLY the aero
  rows (gravity/tether/gyro/transport are blind to the apparent wind at
  frozen attitude and kite velocity), so the consumer feeds the
  `F_aero`/`M_aero_B` contributor slopes — never the net rows, whose
  `delta_v`-route transport response is a kite-velocity gust artefact. Aero-only inputs give the classic
  aerodynamic neutral point / static margin; net inputs add the transport
  response to the tilted kite velocity (gravity/gyro arms do not respond
  to aoa at frozen attitude). The kite margin is the B one — consumer:
  `scripts/personal/wes-quasi-steady/run_static_stability.py`.
  `eval_force_moment(..., with_contributions=True)` appends a 4th return
  element: the per-contributor B-point breakdown (F_aero/F_tether/
  F_gravity/F_transport/F_centripetal; M_aero_B/M_gravity_B/M_transport_B/
  M_gyro_B — the tether has no moment about B), consumed by the pivot-B
  static sweeps. Consumers:
  `scripts/aerodynamics/compute_stability_derivatives.py` and the
  wes-quasi-steady static pipeline.

  `linearise_cg_eom(stability)` linearises the CG form over `CG_STATE_NAMES`
  = 9 states (no `v` — the normal momentum row is the chi_dot_turn closure,
  `chi_turn_gain_row_cg`; no `x`/`y` — the CG is pinned tangentially, a REAL
  constraint, not a coordinate change). Central differences through
  `cg_eom_eval`, which takes `delta_v_cg` / `omega_perturb` /
  `radial_position_offset` as well as attitude: the state is **v_cg**, so
  rate/attitude columns carry the induced attachment-velocity change
  `delta_v_B = -delta_omega x c` into the apparent wind, and the attitude
  columns carry the swinging-B tether stiffness the B-form ones do not.
  Block-diagonal mass matrix — rows are `F/m` and `I_cg^-1 M_cg`, no coupled
  solve; `F_Omega = 0` holds by construction. Driven by
  `run_modal_stability.py --cg-modes` (per-mode table, CG-vs-B pairing,
  `cg_eom_modes_<state>.pdf`). Measured at state C: longitudinal modes move
  <=1.4% vs B-pinned; the lateral pairs move 21-30% — the unstable pair goes
  +0.425 -> +0.057 1/s (T2x 1.6 s -> 12 s), the tether roll stiffness of the
  swinging attachment being the mechanism.

These use the **VSM axis convention** (matches the LEI-V3 reference): chord along
`+x` (LE forward at smaller x, `+x` is **aft**), `+y` spanwise, `+z` up; anhedral
droops the tips, sweepback shifts tips `+x`. The design-tool conventions
(KCU-pivot reference point, `wing_x_offset` pitch-trim lever, the raised speed
bound, window-centre trim at `az=el=0` with uniform wind) are documented in
AWEDesign's `AGENTS.md`.

If the implementation grows, split internal helpers into:

```text
frames.py
attitude.py
stability_derivatives.py
sweeps.py
```

Keep the top-level public import path stable through `vsm_quasi_steady.py`.

## Public Script Layout

```text
scripts/aerodynamics/vsm_quasi_steady/
  solve_single_state.py
  run_sweep.py
  profile_single_state.py
  compute_stability_derivatives.py
```

Case-specific scripts may live one level deeper, for example:

```text
scripts/aerodynamics/vsm_quasi_steady/tudelft_v3/
```

Parametric shape scripts (built on `parametric_geometry.py` for 3D planforms and
`parametric_airfoil.py` for 2D sections) live in:

```text
scripts/aerodynamics/parametric_shapes/
  generate_shape_variations.py
  optimize_lei_airfoil.py
```

The swept-AWES design-tool scripts (`preview_geometry`, `place_wing`,
`design_single_case`, `design_swept_awes`, `analyze_wing`, `sweep_design`) moved
with the design layer to the **AWEDesign** repo (`scripts/` there). They build on
`build_swept_wing` + `turn_radius_vs_steer_moment` from here plus `awedesign`'s
own `design_stats` / `weight_model`.

`generate_shape_variations.py` sweeps four planform DOFs (aspect ratio,
anhedral, taper, twist) from a baseline `aero_geometry.yaml`, writes one morphed
variant per case, and by default evaluates each with VSM and draws shape + aero
comparison figures coloured by swept parameter. The default sweep is one factor
at a time (OAT); `--factorial` does the full grid; `--no-run-vsm` skips VSM.

`optimize_lei_airfoil.py` optimises the six `parametric_airfoil.py` section DOFs
(differential evolution) to maximise `max_alpha(CL^3 / CD^2)`, evaluating
candidate `.dat` profiles with the VSM `AirfoilAerodynamics` regression model.

Use snake_case Python filenames. Do not use hyphenated script names for new
scripts.

## Naming

Use `vsm_quasi_steady` for the VSM aerodynamic trim adapter. Avoid the generic
name `quasi_steady_state` because AWETrim already has a point-mass
quasi-steady residual solver in `SystemModel`.

Public functions should use these names:

- `solve_vsm_quasi_steady_trim` (accepts an optional `applied_moment_nm` for
  external/steering moments, backward-compatible; also an optional
  `gamma_tolerance` — the inner VSM circulation-loop convergence tolerance,
  default `1e-6`, forwarded to the default solver so sweeps can trade a looser
  tolerance for speed. The coupled variant
  `solve_vsm_qs_trim_with_williams_tether` takes the same `gamma_tolerance`.)
  Both trim solvers accept an optional `prescribed_roll_deg` (geometric bridle
  steering: pass the pre-rotated baseline body and `prescribed_roll_deg=0`):
  the roll unknown and the roll-moment residual are dropped (Williams variant:
  5 unknowns / 5 residuals `[cmy, cmz, ground(3)]`), and `cmx` becomes the
  steering-line reaction reported as `reaction_roll_moment_nm`.
  `solve_vsm_qs_trim_with_williams_tether` is the **consistent (off-radial)
  tether trim**: the tether's own drag + weight enter the kite force balance
  (a large effect for long tethers — tether drag is a dominant AWES loss, so a
  radial-tether assumption is optimistic on crosswind speed/load). Its
  `tether_model` selects `"williams"` (default: full distributed shape, kite-end
  vector baked in as the resultant so only tether length + ground closure are
  solved) or `"rigid_lumped"` (the ROM's `RigidLumpedTether` — lumped off-radial
  drag + half-weight, no ground closure; matches the ROM cycle model but
  overestimates tether drag). Both also take `gamma_loop` (default `"base"`),
  forwarded to
  `_default_vsm_solver` → VSM `Solver(gamma_loop_type=...)`. `"anderson"` selects
  the Anderson-accelerated inner loop (~25× fewer inner iterations in attached
  flow), but **only use it with a tight `gamma_tolerance` (~1e-8)**: these trim
  solvers use a finite-difference outer Jacobian, and Anderson's superlinear
  convergence makes its tolerance-terminated `gamma` non-smooth in `x`, which
  corrupts that Jacobian at loose tolerance and yields wrong trims. `"base"` is
  the safe default and the only correct choice at the sweeps' loose `1e-4`.
  Both trim solvers additionally take `is_with_artificial_viscosity`
  (default `False`) and `artificial_viscosity_factor` (default `0.035`),
  forwarded to `_default_vsm_solver` → VSM `Solver`: the parameter-free
  Li/Gaunaa spanwise artificial viscosity that stabilises the gamma loop
  around stall (same option the aerostructural side exposes via as_config
  `aerodynamic.is_with_artificial_viscosity`). Enable it for trims within
  ~1 deg of the stall margin, where the base loop can oscillate without
  converging and the unconverged gamma corrupts the outer trim residuals.
  Ignored when an explicit `solver` is passed.
  `solve_vsm_qs_trim_with_williams_tether` additionally takes `gamma_seed`
  (optional, one circulation value per panel): an initial guess passed to
  EVERY inner VSM solve, with a cold retry when the seeded loop fails to
  converge. Near stall the circulation is multi-valued and a cold-started
  loop can converge onto a different branch than the one a deformed geometry
  was produced with; seeding each evaluation from the same fixed vector
  (e.g. the coupled solver's converged `gamma_distribution`, which
  `scripts/aerostructural/run_simulation_PSM.py` exports as
  `gamma_distribution.npy` next to the geometry snapshot) selects the
  intended branch while staying deterministic and smooth in the trim
  unknowns — unlike history-dependent warm chaining, which would corrupt the
  FD outer Jacobian the same way Anderson does at loose tolerance.
- `turn_radius_vs_steer_moment` (roll-steering turn map: prescribed KCU roll
  moment → bank, `phi_a`, turn radius, effective `k_steering`)
- `compute_vsm_trim_stability_derivatives`
- `run_vsm_quasi_steady_sweep`
- `vsm_quasi_steady_sweep_to_dataframe`
- `plot_vsm_quasi_steady_sweep`

## Stability Script Configuration

`scripts/aerodynamics/compute_stability_derivatives.py` is the STATIC
stability driver: trim → `compute_vsm_trim_stability_derivatives` (always
course-frame `DEFAULT_AXES`, always `course_rate_state=True`) →
`static_slopes_summary` verdict table + `J_full`/`J_course_rate` diagnostics
and JSON. No modal/eigenmode analysis (the modal version is recoverable from
git history). `--stability-frame body` no longer changes the linearisation
axes — it additionally reports the attitude slopes about the principal body
axes at trim (`rigid_body_axes` cloud — aircraft FRD sense: x forward, y
right, z down — rotated by the trim attitude) via the
`attitude_axes=` argument of `static_slopes_summary`; the old
`stability_config.yaml` (`states`/`coupled`/`frame`) is gone.

## Trim State Convention

The VSM trim unknown vector is ordered as:

```text
[speed_tangential, angle_roll_body_deg, angle_pitch_body_deg,
 angle_yaw_body_deg, timeder_angle_course_body]
```

Do not expose this as `kite_speed` in AWETrim-facing APIs. Use
`speed_tangential` to match the root symbol table.

## Frame Convention

The default course-frame basis is:

```text
course = [1, 0, 0]
normal = [0, 1, 0]
radial = [0, 0, 1]
```

The default transform from AWETrim course-frame values to VSM values is:

```text
[[-1,  0, 0],
 [ 0, -1, 0],
 [ 0,  0, 1]]
```

Any implementation must make this transform configurable through the public
interface.

## Required Developer Checks

Before implementing:

- Read this file and the root `AGENTS.md`.
- Keep the VSM dependency optional or protocol-based at import time.
- Preserve the trim unknown ordering above.
- Preserve warm-start behaviour for sweep cases.
- Keep plotting and dataframe conversion separate from the core solve.
- Add bounds defaults to `awetrim.utils.defaults` if they become package-level
  defaults rather than call arguments.

## Required Tester Checks

Tests for this module should check:

- Public signatures and dataclass fields.
- Shape validation for trim state, bounds, frame transforms, Jacobians, and
  stability outputs.
- That no `VSM.core` import is required merely to import `awetrim`.
- That `SystemModel(quasi_steady=True)` enforces
  `timeder_speed_tangential = 0`.
- Numerical VSM solver tests may be marked `slow` and skipped when VSM is not
  installed.
