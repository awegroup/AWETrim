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

The structural solver is [Billow](https://github.com/awegroup/Billow) in every
case; the scripts differ in which of its two fidelities they drive.

| Script | Billow fidelity | What it does |
|--------|-----------------|--------------|
| [`run_simulation_PSM.py`](run_simulation_PSM.py) | wireframe (cables, pulleys) | Couples VSM aerodynamics with the line system. Solves one actuation case (steering / depower), with the actuated KCU and bilinear aero→structure load mapping. |
| [`run_simulation_BILLOW.py`](run_simulation_BILLOW.py) | full (+ tube beams, membrane canopy) | The same coupling against the whole kite. Reads `struc_geometry_FEM_full.yaml` (the only geometry with strut and leading-edge tubes); forces `structural_solver: billow`. |
| [`run_steering_BILLOW.py`](run_steering_BILLOW.py) | full | Walks the steering tape half-difference to a target, each step from a converged state. |
| [`run_chain_depower_BILLOW.py`](run_chain_depower_BILLOW.py) | full | Depower continuation chain at fixed apparent wind. |

## Example output

<img src="../../docs/img/aerostructural-deformed-shape.png" alt="Converged deformed LEI-V3 shape" width="460">

*`run_simulation_PSM.py` — converged VSM ↔ Billow deformed shape: initial vs. loaded geometry, bridle/tape rest-length change (colour) and external aerodynamic loads (red).*

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
