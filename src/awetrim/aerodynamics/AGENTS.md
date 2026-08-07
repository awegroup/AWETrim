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
- `turn_radius_vs_steer_moment` (roll-steering turn map: prescribed KCU roll
  moment → bank, `phi_a`, turn radius, effective `k_steering`)
- `compute_vsm_trim_stability_derivatives`
- `run_vsm_quasi_steady_sweep`
- `vsm_quasi_steady_sweep_to_dataframe`
- `plot_vsm_quasi_steady_sweep`

## Stability Script Configuration

`scripts/aerodynamics/compute_stability_derivatives.py` accepts an optional
YAML stability config with:

- `states`: list or comma-separated string of stability state names, or `all`
- `coupled`: boolean selecting coupled vs longitudinal/lateral block assembly
- `frame`: `course` or `body`; course is the default, body requires an
  identified rigid-body result so principal body axes are available

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
