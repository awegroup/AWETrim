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
  convergence.py                   compute_adaptive_dt, check_convergence,
                                   resultant_tether_force, relative_residual_norm,
                                   resolve_residual_tolerances, max_element_elongation
  results.py                       save_sim_output, append_sweep_csv_row, build_sweep_csv_row
  tracking.py                      setup_tracking_arrays, update_tracking_arrays
  utils.py                         rotate_geometry, calculate_cg, calculate_inertia, load_yaml
  logging_config.py                Package-level logging setup
  aerodynamic_vsm.py               VSM body initialisation and run_vsm_package wrapper (shared)
  aerodynamic_bridle_line_drag.py  Bridle line aerodynamic drag (shared)
                                   (the KCU's bluff-body drag is NOT here: it
                                    enters through the TRIM, see below; the
                                    flat-TAPE section model is not here either:
                                    it enters as the drag-equivalent diameter
                                    written into the line system, see below)

  # ── PSS-based solver ──────────────────────────────────────────────────────
  pss/
    __init__.py                    PssKineticDampingSolver, PssQsmCoupler
    coupling.py                    PssQsmCoupler (fixed-point loop)
    structural_pss.py              PSS instantiation, kinetic-damping solve,
                                   adapt_stiffnesses (1% elongation bound)
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
     + aerodynamic_vsm.rebuild_bridle_line_system(body, struc_nodes, specs):
     update_from_points refreshes WINGS only, so the VSM bridle-line drag
     segments are rebuilt from the live struc_nodes per aero call (specs =
     aerodynamic_vsm.parse_bridle_line_specs(struc_geometry), the same parse
     VSM's instantiate(bridle_path=...) does but with node indices). Without
     this the coupled trim flies the INITIAL symmetric bridle for the whole
     loop; on actuated shapes the missing asymmetric bridle drag mis-trims
     roll (measured: cmx ~0.075, phi_a off by ~0.8 deg at 0.2 m steering
     tape). qsm driver passes bridle_line_specs; coupling.py not yet wired.
     The diameters in those specs are DRAG-equivalent, not structural: the
     steering/depower lines are flat 12 mm webbing that the structural
     table stores as an area-equivalent circle (right for mass/EA, wrong
     for drag). aerodynamic_vsm.initialize applies the substitution to the
     freshly instantiated body and parse_bridle_line_specs repeats it for
     the rebuilds, both via awetrim.aerodynamics.line_drag -- so the two
     paths cannot disagree. Selected by as_config
     aerodynamic_bridle.tape_roll_model (default averaged, 2.91x the old
     round drag on LEI-V3; round reproduces pre-2026-08-21 results).
  3. aerodynamic_vsm.run_vsm_package() → panel forces + trim state           [common]
  4. mapping.BilinearAeroToStructuralLoadMapper.map_loads(panel_forces) → nodal aero forces  [common]
  5. forces.distribute_total_force_by_particle_mass(inertial+gravity) → nodal inertial forces [common]
  6. aerodynamic_bridle_line_drag.main() → nodal bridle drag forces           [common]
  7. pss/structural_pss.run_pss(psystem, total_external_force) → new node positions           [pss]
  8. Aitken relaxation on node displacement
  9. pss/actuation.update_*_tape_actuation() (every N iterations)            [pss]
  10. convergence.check_convergence() → break or continue                    [common]
      (relative residual: ||f_int + f_ext|| / |sum f_ext|, see below)
  11. convergence.element_elongations() + pss/structural_pss.adapt_stiffnesses():
      stiffen anything over the 1% elongation bound and keep iterating   [pss]
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

### Convergence criterion (coupled loop)

The coupled solve converges on the GLOBAL NODAL FORCE RESIDUAL
`f_res = f_int + f_ext` (fixed nodes zeroed -- their imbalance is the tether
reaction), measured as

    ||f_res|| / F_tether  <=  residual_tol_relative   (default 1e-4)

`F_tether = |sum f_ext|` is the resultant tether force: the spring forces are
internal and pairwise self-equilibrated, so the whole applied load (aero +
gravity + inertial) is carried by the single fixed bridle/KCU node.
`convergence.resultant_tether_force` is the ONLY place that force is formed and
`convergence.relative_residual_norm` the only place the ratio is; the PSS/QSM
driver, the FEM driver and `pss/coupling.PssQsmCoupler` all call them. The
residual-norm HISTORY (stagnation window, adaptive dt) is relative too -- do not
mix an absolute history with a relative tolerance.

`meta` carries the evidence per solve: `residual_force_n`, `residual_relative`,
`residual_tol_relative`, `tether_force_resultant`, and `max_element_elongation`
(largest `(l - l0)/l0`, pulley arms excluded -- they trade rope length with
their partner, so their elongation is not strain).

Before 2026-08-27 the criterion was an ABSOLUTE `tol: 5` N. On the 2019 LEI-V3
grid that was ~1.3e-3 relative, so results produced under it are NOT converged
to 1e-4; re-run rather than compare across the change.

### Element elongation bound (PSM stiffness update)

The WING element stiffnesses are not known: following Poland & Schmehl they
are chosen so that no element elongates by more than 1%. The PSM loop ENFORCES
that instead of assuming it -- each coupled iteration,
`convergence.element_elongations` measures every element and
`pss/structural_pss.adapt_stiffnesses` multiplies the stiffness of anything over
`max_elongation` by `stiffness_update_factor` (capped at `max_stiffness`), marks
the iteration not-converged, and restarts the stagnation window.

CONTINUATION (`stiffness_ramp_iterations`, default 20 = ON): the particle
system runs at `stiffness_ramp_factor(i) * k_target` while the ramp is active,
and the loop cannot converge until it completes. Needed when the bridle
material modulus is RAISED -- which the shipped geometries now do: since
2026-08-27 `dyneema.youngs_modulus` is SK75 datasheet grade (109 GPa, Avient
technical data sheet) instead of the old 10 GPa. The
bridle rest lengths are photogrammetry-adjusted with ~1-2% line stretch baked
in, so a stiff bridle imposed in one step shortens the effective bridle by
centimetres, pitches the wing up and lands the trim on the stalled branch
(measured: 25/27 panels stalled, residual -> NaN at 55 GPa). Ramping from
`stiffness_ramp_start_factor` (use `k_old / k_new`) keeps the solve on the
attached branch. The elongation update adapts `k_target`, never the ramped
value, so the two compose.

SCOPE (`stiffness_update_scope`, default `wing`): only the wing elements are
adapted. A bridle line's stiffness is NOT a free parameter -- it is `E*A/l0`
from the line material and diameter -- so stiffening it to suppress its stretch
would falsify the bridle. Wing elements are parsed first, so the eligible set is
`range(len(kite_connectivity_arr) - len(bridle_connectivity_arr))`. `meta` keeps
`max_wing_element_elongation` and `max_bridle_element_elongation` apart so the
bridle stretch stays visible.

NOTE on the bridle material: `dyneema.youngs_modulus` is now 109 GPa (SK75
datasheet lower bound) in every shipped `struc_geometry*.yaml`; `density` stays
an EQUIVALENT value (1717 for LEI-V3, not the fibre's 970) because with the
stored line diameters it is the mass per metre that must come out right. The
bridle rest lengths are photogrammetry-adjusted WITH the old 10 GPa stretch
present, so the change re-trims the kite -- measured at u_dp 0.227: bridle
elongation 2.78% -> 0.29%, alpha +0.54 deg, CL 0.684 -> 0.722, tether force
5532 -> 6406 N. Results produced before 2026-08-27 are on the soft bridle.
The FEM path shares the material block but has NO continuation of its own. So a converged
PSM shape satisfies BOTH the force residual and the elongation bound; if
offending elements reach `max_stiffness` nothing is updated any more and the
loop is free to converge (check `max_element_elongation` in `meta`).

A PULLEY's arms share one rope: `element_elongations` gives both arms the
ROPE's strain (`(l_a + l_b - l0_a - l0_b) / (l0_a + l0_b)`), so a pulley that
merely trades length between arms is never stiffened. Pass the driver's
`pulley_line_to_other_node_pair_dict` as `pulley_pairs`.

Damping is deliberately NOT rescaled with `k` (the PSS dissipation is the
kinetic-damping scheme). The SELECTED stiffnesses travel with the deformed
geometry: `results.build_deformed_struc_geometry` writes the changed elements
into an `element_stiffness` table (`headers: [node_i, node_j, k]`) that
`structural_geometry_io` applies on reload, so a snapshot re-solve starts from
the values its solve converged with. Rows are keyed by NODE PAIR, never by name
-- one bridle name covers both sides of the kite and both arms of a pulley, and
the bound can drive those to different stiffnesses. `meta["final_stiffnesses"]`
holds the full array.

Before 2026-08-27 the PSM path had NO stiffness update (only the FEM path did,
via `kite_fem.adapt_stiffnesses`), and converged 2019-grid shapes reached ~1.4%
on a wing element and ~1.9% on the bridle lines.

### PSS convergence
The PSS kinetic-damping convergence check requires `step * dt > 10.0` before it fires. With `n_internal_time_steps = 100` and `dt = 0.005` (total = 0.5 s), the check **never triggers** — the PSS always runs the full step count. Starting from the unloaded YAML geometry (far from loaded equilibrium) with large aero forces will produce large non-physical deformations in the first iteration. Pre-loaded starting geometry (warm-start from a previous result) avoids this.

### Aitken relaxation
Node positions are updated as `nodes += factor * (solved_nodes - nodes)` where `factor` is recalculated by the Aitken method each iteration. The initial factor comes from `QsmCouplingSettings.relaxation_factor`.

## Boundary

- No CasADi symbolics enter this module. All quantities are numeric numpy arrays.
- VSM solver internals (`VSM.core`) are accessed only through `aerodynamic_vsm.py`; the rest of the module is VSM-agnostic.
- `aerodynamic_vsm.py` and `aerodynamic_bridle_line_drag.py` live at the root level and are shared by all solvers. `aerodynamic_vsm.run_vsm_package` also builds the KCU bluff-body drag model (`awetrim.aerodynamics.kcu_drag`, gated by the `is_with_kcu_drag` config key, default true) and hands it to whichever trim it dispatches to. It is deliberately NOT distributed onto structural nodes: the KCU is node 0, a FIXED node, so a force there is absorbed by the constraint and cannot deform anything — the KCU drag reaches the structure only through the trim state the wing is loaded at. `pss/structural_pss.py` holds the PSS dependency. All other common files (`mapping.py`, `convergence.py`, etc.) depend only on numpy and the module's own protocols.
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
  is_with_artificial_viscosity: false  # opt-in Li/Gaunaa spanwise artificial
  artificial_viscosity_factor: 0.035   #   viscosity (TORQUE 2026) in gamma_loop

structural_pss:
  dt: 0.005
  n_internal_time_steps: 100   # must be >> 2000 for convergence check to fire
  abs_tol: 1.0e-50
  rel_tol: 1.0e-5
  max_iter: 500
  kinetic_energy_tolerance: 1.0e-3
  fixed_point_indices: [0]
  update_stiffness: true       # enforce the elongation bound (see below)
  stiffness_update_scope: wing # wing elements only (bridle k is E*A/l0)
  max_elongation: 0.01         # 1%
  stiffness_ramp_iterations: 20 # continuation, ON by default (see below)
  stiffness_ramp_start_factor: 0.1
  stiffness_ramp_scope: bridle
  stiffness_update_factor: 2.0
  max_stiffness: 5.0e+5        # ceiling [N/m]

aero_structural_solver:
  max_iter: 100
  residual_tol_relative: 1.0e-4   # DIMENSIONLESS (see "Convergence criterion")
  stagnation_tol_relative: 4.0e-5 # DIMENSIONLESS
  relaxation_factor: 0.5
  is_with_aitken_relaxation: true
```

The legacy absolute-force keys `tol` [N] and `stagnation_tol` [N] are REJECTED
with a ValueError (`convergence.resolve_residual_tolerances`) rather than
reinterpreted -- a `tol: 5` read as a ratio would "converge" on iteration 1.

## Result Storage

Output goes to `results/aerostructural/<kite_name>/<case_folder>/sim_output.h5` (absolute path from project root, never CWD-relative). Use `results.save_sim_output()` and `results.aerostructural_results_root()`.

## Required Developer Checks

- Read `structural_geometry_io.main()` before changing how struc_geometry.yaml is parsed; the node index ordering (odd = LE, even = TE) and pulley dict format `[cj, ck, l0_cj_ck, l0_ci_cj, ci]` are load-bearing. It is also the single entry point for the point-mass cloud: node ordering, element-to-node mass lumping and the KCU mass (from `system.yaml` via `_resolve_kcu_mass`) all come from here, so read a cloud through it (`identification.rigid_body_axes.load_psm_geometry` for a path, `load_psm_nodes_and_masses` for a dict) rather than re-walking the YAML. `power_tape_index` is `None` when the geometry has no `depower_tape` connection.
- Any change to `PssQsmCoupler` must keep `QsmCouplingRequest` / `QsmCouplingResult` stable; the protocol tests check these fields.
- Scripts in `scripts/aerostructural/` import shared helpers from `common.py` — add new shared defaults to `CONFIG_DEFAULTS` there, not as literals in individual scripts.
