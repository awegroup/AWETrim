# AWETrim Aerostructural Module

## Status: ✅ Built

## Scope

This module implements the fixed-point PSS/QSM aerostructural coupling: a structural
particle system (PSS) iterated against a VSM quasi-steady aerodynamic trim (QSM) until
the nodal forces converge. It also owns geometry I/O, load mapping, actuation, result
storage, and the sweep orchestration scripts.

This module does **not** own the VSM solver or the point-mass system model — those live
in `aerodynamics/` and `system/` respectively.

Shared plotting utilities live in `src/awetrim/plotting/`.

## Public Layout

```
src/awetrim/aerostructural/
  # ── Solver-agnostic (common to PSS, FEM, and future solvers) ──────────────
  __init__.py                      Re-exports PssKineticDampingSolver, PssQsmCoupler, all protocols
  protocols.py                     All dataclasses and Protocol types
  mapping.py                       LinearStructuralToAeroMapper, BilinearAeroToStructuralLoadMapper
  forces.py                        distribute_total_force_by_particle_mass
  convergence.py                   compute_adaptive_dt, check_convergence
  results.py                       save_sim_output, append_sweep_csv_row, build_sweep_csv_row
  tracking.py                      setup_tracking_arrays, update_tracking_arrays
  utils.py                         rotate_geometry, calculate_cg, calculate_inertia, load_yaml
  logging_config.py                Package-level logging setup
  aerodynamic_vsm.py               VSM body initialisation and run_vsm_package wrapper (shared)
  aerodynamic_bridle_line_drag.py  Bridle line aerodynamic drag (shared)

  # ── PSS-based solver ──────────────────────────────────────────────────────
  pss/
    __init__.py                    PssKineticDampingSolver, PssQsmCoupler
    coupling.py                    PssQsmCoupler (fixed-point loop)
    structural_pss.py              PSS instantiation and kinetic-damping solve
    structural_geometry_io.py      Parse struc_geometry.yaml → StructuralGeometry arrays
    actuation.py                   update_steering_tape_actuation, update_power_tape_actuation
    aerostructural_coupled_solver_qsm.py  Legacy high-level driver (used by production scripts)

  # ── FEM-based solver ──────────────────────────────────────────────────────
  fem/
    __init__.py                    Re-exports all four FEM modules
    aerostructural_coupled_solver.py  FEM/QSM high-level driver
    aero2struc.py                  Aero-to-structural force mapping and moment preservation check
    read_struc_geometry_yaml.py    Parse struc_geometry YAML (strut tubes, LE tubes)
    structural_kite_fem.py         FEM structure instantiation and solve

scripts/aerostructural/
  common.py                        CONFIG_DEFAULTS, build_system_model, shared helpers
  run_simulation_PSM.py            Single-case PSS/QSM (PSM) solve with optional steering sweep
  run_simulation_FEM.py            Single-case FEM (kite_fem) solve
  run_sweep_wind_steering_PSM.py   2-D sweep: wind × steering (PSM)
  run_sweep_course_steering_depower_PSM.py  3-D sweep: course × steering × depower (PSM)
```

## Core Data Flow

```
struc_geometry.yaml
  └─ pss/structural_geometry_io.main() → StructuralGeometry (nodes, connectivity, rest_lengths, …)

aero_geometry.yaml
  └─ pss/aerodynamic_vsm.initialize() → (body_aero, vsm_solver, initial_polar_data)

Fixed-point loop (pss/coupling.PssQsmCoupler.solve  or  pss/aerostructural_coupled_solver_qsm.main):
  1. mapping.LinearStructuralToAeroMapper.map(nodes) → LE/TE points          [common]
  2. body_aero.update_from_points(LE, TE, polar_data)
  3. aerodynamic_vsm.run_vsm_package() → panel forces + trim state           [common]
  4. mapping.BilinearAeroToStructuralLoadMapper.map_loads(panel_forces) → nodal aero forces  [common]
  5. forces.distribute_total_force_by_particle_mass(inertial+gravity) → nodal inertial forces [common]
  6. aerodynamic_bridle_line_drag.main() → nodal bridle drag forces           [common]
  7. pss/structural_pss.run_pss(psystem, total_external_force) → new node positions           [pss]
  8. Aitken relaxation on node displacement
  9. pss/actuation.update_*_tape_actuation() (every N iterations)            [pss]
  10. convergence.check_convergence() → break or continue                    [common]
```

## Key Dataclasses (protocols.py)

All cross-function data uses frozen dataclasses — no raw dicts between module-level functions.

| Dataclass | Role |
|-----------|------|
| `StructuralGeometry` | Full structural model: nodes, masses, connectivity, rest lengths, stiffness, damping, LE/TE indices, pulley dict |
| `QsmCouplingRequest` | All inputs to one coupled solve |
| `QsmCouplingSettings` | Solver numerics: tolerances, relaxation, actuation intervals |
| `TapeActuationState` | Depower/steering tape targets and step sizes |
| `QsmCouplingResult` | Final nodes, residual, iteration records, trim result |
| `QsmIterationRecord` | Per-iteration diagnostics |
| `AerodynamicGeometryUpdate` | LE/TE arrays returned by the structural-to-aero mapper |

## Critical Implementation Notes

### Pulley rest lengths
`structural_pss.instantiate` must set each pulley arm's rest length to its **individual arm length**, not the total rope length stored in the YAML. The individual arm length is at index `[3]` of each entry in `pulley_line_to_other_node_pair_dict`. Using the total length puts both arms in artificial compression and causes catastrophic PSS divergence.

```python
# Correct: PSS expects [idx_p3, idx_p4, rest_length_of_other_arm]
pss_pulley_dict = {key: val[:3] for key, val in pulley_dict.items()}
# Then override each arm's own rest length from val[3]
```

### Frame convention
Panel forces from VSM are in the VSM frame (x and y negated relative to the course frame). The transformation `T_C_from_VSM = [[-1,0,0],[0,-1,0],[0,0,1]]` is applied inside `aerodynamics/vsm_quasi_steady.py` **before** forces reach this module. Structural geometry coordinates are in the course frame throughout.

### PSS convergence
The PSS kinetic-damping convergence check requires `step * dt > 10.0` before it fires. With `n_internal_time_steps = 100` and `dt = 0.005` (total = 0.5 s), the check **never triggers** — the PSS always runs the full step count. Starting from the unloaded YAML geometry (far from loaded equilibrium) with large aero forces will produce large non-physical deformations in the first iteration. Pre-loaded starting geometry (warm-start from a previous result) avoids this.

### Aitken relaxation
Node positions are updated as `nodes += factor * (solved_nodes - nodes)` where `factor` is recalculated by the Aitken method each iteration. The initial factor comes from `QsmCouplingSettings.relaxation_factor`.

## Boundary

- No CasADi symbolics enter this module. All quantities are numeric numpy arrays.
- VSM solver internals (`VSM.core`) are accessed only through `aerodynamic_vsm.py`; the rest of the module is VSM-agnostic.
- `aerodynamic_vsm.py` and `aerodynamic_bridle_line_drag.py` live at the root level and are shared by all solvers. `pss/structural_pss.py` holds the PSS dependency. All other common files (`mapping.py`, `convergence.py`, etc.) depend only on numpy and the module's own protocols.
- `pss/aerostructural_coupled_solver_qsm.py` is a legacy high-level driver retained for production scripts. New protocol-level code should go through `pss/coupling.PssQsmCoupler`.
- When adding a new structural solver, create a new subfolder (e.g., `fem/`) mirroring the `pss/` layout. Common files at the root level are shared by all solvers.

## Config Keys (aerostructural_configs/config.yaml)

All defaults are defined in `scripts/aerostructural/common.CONFIG_DEFAULTS`. Key sections:

```yaml
aerodynamic:
  n_aero_panels_per_struc_section: 3
  spanwise_panel_distribution: uniform
  max_iterations: 1000
  allowed_error: 2.0e-6
  relaxation_factor: 0.05
  reference_point: [0.0, 0.0, 0.0]

structural_pss:
  dt: 0.005
  n_internal_time_steps: 100   # must be >> 2000 for convergence check to fire
  abs_tol: 1.0e-50
  rel_tol: 1.0e-5
  max_iter: 500
  kinetic_energy_tolerance: 1.0e-3
  fixed_point_indices: [0]

aero_structural_solver:
  max_iter: 100
  tol: 5.0
  relaxation_factor: 0.5
  is_with_aitken_relaxation: true
```

## Result Storage

Output goes to `results/aerostructural/<kite_name>/<case_folder>/sim_output.h5` (absolute path from project root, never CWD-relative). Use `results.save_sim_output()` and `results.aerostructural_results_root()`.

## Required Developer Checks

- Read `structural_geometry_io.main()` before changing how struc_geometry.yaml is parsed; the node index ordering (odd = LE, even = TE) and pulley dict format `[cj, ck, l0_cj_ck, l0_ci_cj, ci]` are load-bearing.
- Any change to `PssQsmCoupler` must keep `QsmCouplingRequest` / `QsmCouplingResult` stable; the protocol tests check these fields.
- Scripts in `scripts/aerostructural/` import shared helpers from `common.py` — add new shared defaults to `CONFIG_DEFAULTS` there, not as literals in individual scripts.
