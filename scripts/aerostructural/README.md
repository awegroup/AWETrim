# Aerostructural scripts

Run a single coupled VSM ↔ structure simulation to obtain the **deformed** kite
shape and the force coefficients on it. The aerodynamic loads (VSM) are iterated
against the deformed wing with an Aitken-relaxed fixed-point loop until the nodal
forces converge. Run from the project root.

Shared defaults and path resolution live in [`common.py`](common.py)
(`CONFIG_DEFAULTS`, `resolve_kite_paths`, `build_system_model`). Solver settings
come from each kite's `as_config.yaml`; geometry from `aero_geometry.yaml` and
`struc_geometry.yaml`.

## Scripts

| Script | Structural solver | What it does |
|--------|-------------------|--------------|
| [`run_simulation_PSM.py`](run_simulation_PSM.py) | PSS (particle-spring) | Couples VSM aerodynamics with the Particle System Simulator. Solves one actuation case (steering / depower), with the actuated KCU and bilinear aero→structure load mapping. |
| [`run_simulation_FEM.py`](run_simulation_FEM.py) | kite_fem (FEM) | Same coupling against the finite-element solver. Reads the FEM full geometry (`struc_geometry_FEM_full.yaml`, with strut and leading-edge tubes); forces `structural_solver: kite_fem`. |

## Outputs

Each run writes a case folder under
`results/<kite_name>/aerostructural/<case>/`:

- `sim_output.h5` — converged nodal positions, forces and tracking history.
- deformed `aero_geometry.yaml` / `struc_geometry.yaml` snapshots and the input
  snapshot, so the deformed shape can be reused (e.g. by
  `scripts/aerodynamics/solve_single_state.py --deformed-from <case_dir>`).

## Notes

- **FEM is a work in progress.** The aero→structure chordwise force distribution
  and the FEM structural solver still need improvement — see the FEM known
  limitation in the project `AGENTS.md` and
  `src/awetrim/aerostructural/AGENTS.md` before changing the coupling.
- KCU mass is taken from `system.yaml` only (single source of truth); the
  structural geometry no longer carries `kcu_mass`.
