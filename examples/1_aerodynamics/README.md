# Step 1: aerodynamics

Every step of the workflow reads the kite from one folder, `data/<kite>/`.
To start from your own kite, create that folder; `data/LEI-V3-KITE/` is the
annotated reference.

| File | What it describes |
|------|-------------------|
| `system.yaml` | Masses, inertia, KCU, tether and winch, in the [awesIO](https://github.com/awegroup/awesIO) format; also selects the ROM file the trajectory steps fly |
| `aero_geometry.yaml` | The wing for the Vortex Step Method: sections and airfoils |
| `structure.yml` | The structure, in the [awesIO](https://github.com/awegroup/awesIO) structure format: tubes, canopy, bridle lines and pulleys. The bridle drag of every step reads it; Billow solves it (step 2) |
| `as_config.yaml` | Solver settings of the coupled aerostructural simulation (step 2) |

The examples here analyse the kite's aerodynamics before the heavier steps: one quasi-steady
trim on the undeformed wing, and its static stability. They are built on the
VSM quasi-steady trim layer (`awetrim.aerodynamics.vsm_quasi_steady`). Run
them from the project root.

Shared CLI helpers live in [`common.py`](common.py): the operating-point flags
(`--elevation-deg`, `--azimuth-deg`, `--course-deg`, `--wind-speed`,
`--radial-speed`, `--distance-radial`), `--vsm-src` (point at a local
Vortex-Step-Method checkout when it is not pip-installed), and `--output-dir`.
By default each script writes to `results/aerodynamics/<script_name>/`.

## Scripts

| Script | What it does | Key outputs |
|--------|--------------|-------------|
| [`solve_single_state.py`](solve_single_state.py) | Solves one VSM quasi-steady trim state for the kite at a chosen operating point. Optionally trims on a **deformed** shape from an aerostructural run (`--deformed-from <case_dir>`) and can close the trim with the distributed-mass `WilliamsTether`. | Trim summary (α, pitch, force coefficients), kite+tether 3-D figure, optional `--output-json`. |
| [`compute_polars.py`](compute_polars.py) | Lift and drag polars of the wing on one frozen shape (built, or deformed with `--deformed-from`): VSM solved at each angle of attack for a fixed apparent-wind speed, no sideslip and no re-trim. Coefficients on both the VSM projected area and the ROM's reference area. `--alpha-min/--alpha-max/--alpha-step`, `--apparent-wind-speed`. | `polars.csv` (C_L, C_D on both areas, L/D), `polars.png` (C_L, C_D and L/D against α). |
| [`compute_stability_derivatives.py`](compute_stability_derivatives.py) | Computes the aerodynamic stability derivatives about a trim point by finite-differencing the VSM trim (the raw `J_full` columns), then reports the six-channel **static-stability** verdict via `static_slopes_summary` (roll/pitch/yaw/v_tau/radial stiffness + chi_dot turn damping). `--stability-frame body` additionally reports the attitude slopes about the centre-panel body axes at trim. The previous modal/eigenmode version is recoverable from git history. | Static-verdict table on stdout, stability-derivative JSON (trim + `J_full` + static summary), FD-Jacobian heatmap figure. |

## Example outputs

<img src="../../docs/img/kite_tether_states.png" alt="Single trim state: kite and tether" width="300">

*`solve_single_state.py` — kite and tether at two trim states in the wind frame.*
