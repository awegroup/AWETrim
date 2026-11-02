# Step 1: aerodynamics

Every step of the workflow reads the kite from one folder, `data/<kite>/`.
To start from your own kite, create that folder; `data/LEI-V3-KITE/` is the
annotated reference.

| File | What it describes |
|------|-------------------|
| `system.yaml` | Masses, inertia, KCU, tether and winch, in the [awesIO](https://github.com/awegroup/awesIO) format; also selects the ROM file the trajectory steps fly |
| `aero_geometry.yaml` | The wing for the Vortex Step Method: sections and airfoils |
| `structure.yml` | The structure, in the [awesIO](https://github.com/awegroup/awesIO) structure format: tubes, canopy, bridle lines and pulleys. The bridle drag of every step reads it; Billow solves it (step 2) |
| `as_config.yaml` | How the kite is modelled and solved: aerodynamics (wing billowing, panels, bridle drag), structure, tether, coupling |

The examples here analyse the kite's aerodynamics before the heavier steps: one quasi-steady
trim, the polars of a frozen shape, and the static stability. They are built on the
VSM quasi-steady trim layer (`awetrim.aerodynamics.vsm_quasi_steady`). Edit the
**Inputs** block at the top of a script and run it from the project root:

    python examples/1_aerodynamics/solve_single_state.py

Every script builds the kite through [`common.py`](common.py) (`load_kite`) as the
coupled solver of step 2 does, minus the structural solve. Everything physical comes
from `data/<KITE>/`: the wing (the structure's FEM sections, billowed and meshed as
`as_config.yaml` says), the full bridle, the KCU point (the moment reference), and the
mass, CG and inertia of one body (`structure.yml` + `system.yaml`'s KCU; a mismatch
with the system model's mass raises). The model choices are `as_config.yaml`'s (KCU
drag, the tether in the trim, gravity), except the VSM circulation loop: step 1 trims
with VSM's base loop without artificial viscosity (`common.STEP1_VSM_SETTINGS`).

The inputs:

- `CASE`: what the run is, an `awetrim.aerostructural.case.Case` (flight state, wind,
  tape command). The default is the step-2 PSM example's case.
- `SHAPE`: `"built"` (the structure's wing; it has the built tapes, so a tape command
  needs a deformed shape), or `"wireframe"` / `"billow"`, the deformed shape the step-2
  run of the same `CASE` left in `results/<kite>/aerostructural/<fidelity>/`. A missing
  run names the step-2 script to run first. `DEFORMED_FROM` points at any other results
  case folder.
- `WITH_GRAVITY`, the output folder (default `results/aerodynamics/<script>/`), and each
  script's own settings.

## Scripts

| Script | What it does | Key outputs |
|--------|--------------|-------------|
| [`solve_single_state.py`](solve_single_state.py) | Solves one VSM quasi-steady trim of the kite at `CASE`, with the tether as_config puts in the trim (the distributed-mass `WilliamsTether` for the LEI-V3). | Trim summary (α, attitude, force coefficients), `trim_result.json` (case, kite provenance, trim), residual and kite+tether figures. |
| [`compute_polars.py`](compute_polars.py) | Lift and drag polars of the wing on one frozen shape: VSM solved at each angle of attack (`ALPHA_DEG`) for a fixed apparent-wind speed (`APPARENT_WIND_SPEED`), no sideslip and no re-trim. Coefficients on both the VSM projected area and the ROM's reference area. | `polars.csv` (C_L, C_D on both areas, L/D), `polars.png` (C_L, C_D and L/D against α). |
| [`compute_stability_derivatives.py`](compute_stability_derivatives.py) | Computes the aerodynamic stability derivatives about a trim point by finite-differencing the VSM trim (the raw `J_full` columns, steps `EPS`), then reports the six-channel **static-stability** verdict via `static_slopes_summary` (roll/pitch/yaw/v_tau/radial stiffness + chi_dot turn damping). `STABILITY_FRAME = "body"` additionally reports the attitude slopes about the centre-panel body axes of the kite's own structure at trim. The previous modal/eigenmode version is recoverable from git history. | Static-verdict table on stdout, stability-derivative JSON (trim + `J_full` + static summary), FD-Jacobian heatmap figure. |

## Example outputs

<img src="../../docs/img/kite_tether_states.png" alt="Single trim state: kite and tether" width="300">

*`solve_single_state.py` — kite and tether at two trim states in the wind frame.*
