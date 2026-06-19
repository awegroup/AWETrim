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

Area/span conventions are documented in the module docstring (flat/developed
area, flat aspect ratio). Inspired by the QC-anchored shape-variation generator
in `jellepoland/WES_aero_sim_for_kite_design`.

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

- `solve_vsm_quasi_steady_trim`
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
