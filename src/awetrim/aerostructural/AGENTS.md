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
  convergence.py                   check_convergence, compute_adaptive_dt,
                                   resolve_residual_tolerances, resultant_tether_force,
                                   relative_residual_norm, element_elongations,
                                   max_element_elongation
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
    structural_pss.py              PSS instantiation and kinetic-damping solve
    structural_nlp.py              Exact minimum-energy inner solve (CasADi/IPOPT):
                                   NlpStructuralSolver (run_pss call contract) +
                                   resolve_structural_solver dispatch. Same spring
                                   physics as PSS (tension-only cutoffs, pulley
                                   pairs), solved to ~1e-8 instead of kinetic
                                   damping's ~1 N leftover. Selected by
                                   structural_pss.solver: nlp (default pss).
    structural_geometry_io.py      Parse struc_geometry.yaml → StructuralGeometry arrays
    actuation.py                   update_steering_tape_actuation, update_power_tape_actuation
    aerostructural_coupled_solver_qsm.py  Legacy high-level driver (used by production scripts)

  # ── FEM-based solver ──────────────────────────────────────────────────────
  fem/
    __init__.py                    Re-exports all four FEM modules
    aerostructural_coupled_solver.py  The coupled QSM driver. Despite living
                                   here it is NOT FEM-only: it dispatches on
                                   config["structural_solver"] over pss /
                                   kite_fem / billow. Add a backend by adding
                                   a branch, not by copying the driver.
                                   Steers only when the caller passes
                                   steering_tape_indices (see "Steering in
                                   the shared driver" below).
    aero2struc.py                  Aero-to-structural force mapping and moment
                                   preservation check. chordwise_distribution
                                   (see below) decides whether the panel's
                                   pitching moment is conserved; load_transfer
                                   decides how the load reaches the nodes
                                   (traction integral for billow, see below).
    read_struc_geometry_yaml.py    Parse struc_geometry YAML (strut tubes, LE tubes)
    structural_kite_fem.py         FEM structure instantiation and solve

  # ── Billow-based solver ───────────────────────────────────────────────────
  billow/
    __init__.py                    Re-exports the structural_billow API
    structural_billow.py           Adapts awetrim.structural (minimum-energy
                                   cables, pulleys, inflatable Timoshenko tube
                                   beams, wrinkling CST membrane canopy) to the
                                   run_kite_fem call contract. Reads the SAME
                                   arrays fem/read_struc_geometry_yaml.main
                                   returns, so one geometry reader serves both.
                                   See "Billow backend" below.

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
  7. structural solve → new node positions
       pss/structural_pss.run_pss                    [pss]
       fem/structural_kite_fem.run_kite_fem          [kite_fem]
       billow/structural_billow.run_billow           [billow]
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

- No CasADi symbolics enter this module. All quantities are numeric numpy
  arrays. (`pss/structural_nlp.py` uses CasADi INTERNALLY to build its solver —
  nothing symbolic crosses its function boundaries.)
- VSM solver internals (`VSM.core`) are accessed only through `aerodynamic_vsm.py`; the rest of the module is VSM-agnostic.
- `aerodynamic_vsm.py` and `aerodynamic_bridle_line_drag.py` live at the root level and are shared by all solvers. `aerodynamic_vsm.run_vsm_package` also builds the KCU bluff-body drag model (`awetrim.aerodynamics.kcu_drag`, gated by the `is_with_kcu_drag` config key, default true) and hands it to whichever trim it dispatches to. It is deliberately NOT distributed onto structural nodes: the KCU is node 0, a FIXED node, so a force there is absorbed by the constraint and cannot deform anything — the KCU drag reaches the structure only through the trim state the wing is loaded at. `pss/structural_pss.py` holds the PSS dependency. All other common files (`mapping.py`, `convergence.py`, etc.) depend only on numpy and the module's own protocols.
- `pss/aerostructural_coupled_solver_qsm.py` is a legacy high-level driver retained for production scripts. New protocol-level code should go through `pss/coupling.PssQsmCoupler`.
- When adding a new structural solver, create a new subfolder (e.g., `fem/`, `billow/`) for the BACKEND, mirroring the `pss/` layout. Common files at the root level are shared by all solvers. The coupled driver in `fem/` is already backend-agnostic — add a branch to its `config["structural_solver"]` dispatch rather than copying its ~1000 lines.

## Billow backend (`billow/`)

`structural_billow.instantiate` consumes the arrays
`fem/read_struc_geometry_yaml.main` already returns and turns them into a
`awetrim.structural` minimum-energy model. The element mapping:

| reader element | Billow element |
|----------------|----------------|
| `inflatable_beam` | `InflatableBeamKernel` tube beams (diameter from `k_arr`, pressure from `c_arr`) |
| `pulley` arm pairs | `PulleyKernel` ropes, rest length = the WHOLE rope |
| bridle `noncompressive` | `CableKernel` tension-only cables |
| canopy grid springs | **dropped**, replaced by wrinkling CST membrane triangles |

On the LEI-V3 `struc_geometry_FEM_full.yaml`: 268 nodes / 1098 DOF, 47 cables,
20 pulleys, 97 tubes, 378 membrane triangles, 693 canopy springs replaced.
Build ~0.9 s, structural solve ~2 s per coupled iteration (the VSM trim is
~25 s, so the structure is not the cost).

Four things the adapter has to do that are not obvious:

- **The bridle is relaxed before the model is built.** The YAMLs store measured
  rest lengths against measured node positions and the two disagree — on the
  LEI-V3 `Br_main_1` is 11.5% long, which at the real `EA/l0` is 88 kN in one
  line. `relax_bridles` settles it with the wing held (so the canopy and tube
  reference configurations are untouched), then re-centres. This is the same
  role `structural_kite_fem.relaxbridles` plays on the FEM path. **The nodes the
  coupled loop starts from are therefore not the raw YAML nodes** — read them
  back from `structure.model.nodes`.
- **Relaxation cannot fix a line that is too LONG**, only one that is too short:
  it moves nodes, and no node motion takes up slack the taut lines have already
  pinned. What survives the relaxation is the real inconsistency in the
  geometry, and on the LEI-V3 it sits on the front tip leg. Against the RAW
  stored nodes, `a5_equiv` is 0.2540 m against a CAD distance of 0.2540 m
  (exact), `br_5_equiv` +30 mm and `Br_5` +42 mm (+0.35%) — but `AIII` is
  **+323 mm (+2.91%)** long. All of it is in that one rope.
  The consequence is visible: the front tip leg carries 0.26 N against 92 N in
  `a4_b4` one station inboard, so the tip is effectively bridle-free and hangs.
  **Unresolved, and do not "fix" it by shortening the legs.** That was tried
  (`a5_equiv` -> 0.0607 m, `br_5_equiv` -> 0.7635 m, the lengths that make the
  measured ropes exactly taut with the sheave hanging in equilibrium from its
  attachment). It bends a length that is already exactly consistent in order to
  absorb an error that lives in `AIII`, and it turns a clean solve into a bad
  one: the baseline goes from 0.04 N in 5 coupled iterations to still bouncing
  at 9-18 N after 30. Reverted. If this is attacked, the candidate is `AIII`'s
  323 mm, and not before the attachment question below is settled, because a
  different attachment changes what `AIII` has to be.
  Related: the tip front leg attaches to WING particles 3/53 — the only two
  bridle connections in the file that do — while bridle particles 84/118,
  declared on the tip closure tubes as its attachment points, have nothing
  connected. They cannot simply take the leg: with the sheave in equilibrium,
  `AIII` would need a leg of -65 mm from node 118, i.e. the sheave above the
  tube, against +61 mm from node 3.
- **Strut padding defaults to `bisect_longest`**, not `legacy`. `legacy` puts
  every padding node in the second-to-last gap AND re-reads the gap ends after
  each insertion, so they land at 0.20/0.52/0.81/0.96 of the gap instead of
  0.2/0.4/0.6/0.8 and pile up against its far end: on the 0.348 m tip closure
  that leaves a 6.7 mm beam beside a 96.8 mm one, a 118x length spread across
  the wing against 46x for `bisect_longest`. The loop contradicts its own
  `ratio = (i+1)/(missing+1)`, so this is a bug, but be honest about its size:
  it is a MESH change, not a physics one. Both paddings give the same 97 tube
  elements, the same tube diameters and the SAME reference curvature field
  (max |omega_0| 10.63 /m, mean 1.20) — padding nodes are projected onto the
  straight line between the strut's end nodes, so a strut is straight whatever
  the spacing and the sliver carries no reference curvature. 96 of 457 built
  nodes move (mean 105 mm) and the canopy triangles reshape; that is all.
  The evidence once cited here for the sliver not hurting conditioning — "the
  legacy baseline converged to 0.04 N in 5 coupled iterations" — is WITHDRAWN:
  that run (2026-09-11) simply landed on the attached branch, and re-run
  unguarded on current code `legacy` is WORSE than `bisect_longest` (101.9 mm
  left-right mismatch at 3.31 N against 310.8 mm at 0.26 N; both fail, and
  neither reproduces). Those numbers are samples of a wandering iteration, not
  mesh properties — see the circulation-loop note below. Compare meshes on a
  base-loop or guarded solve, where the two agree to 1 mm on span (7.9466
  against 7.9453). `legacy` is kept bit-exact to reproduce stored FEM results —
  set `strut_padding: legacy` for that.
  One more trap: `initialize_particles` appends the YAML's OWN `node_indices`
  list to `strut_indices` and the padding loop inserts into it. That aliasing
  is load-bearing — it is how the padding nodes reach
  `initialize_wing_structure`, which re-reads `strut_tubes` to build the strut
  beams. Copy the list and each strut loses its padded segments: 97 beam
  elements become 77 and the unsteered solve stops being mirror-symmetric
  (22.9 mm instead of 2e-5 mm). The cost is that `main()` mutates the dict it
  is given, so load the YAML fresh for every call.

- **With AV on, `anderson` and `base` do not pick the same trim branch from the
  same cold start** — and on the unsteered LEI-V3 the one `anderson` picks is
  spurious. On the built shape it lands stalled-tip at v_w 3.0, 4.2 and 9.0
  (1-2 tip panels 2.0-7.8 deg past their onset) and on a higher-speed attached
  fixed point at 5.5 and 7.0 (v_a 30.27 against 23.37), so its v_a is not even
  monotone in the wind; `base` gives one smooth family at every wind tested
  (tip alpha 4.31-4.32, margin +7.13) at 5.1-7.1 s against 6.3-11.4 s per trim.
  Coupled and unsteered at v_w 4.2: `base` 0.013 N in 7 iterations, 0.025 mm
  mismatch, 94 s; `anderson` 0.26 N at iteration 31, 310.8 mm, 590 s, 355
  bailouts. `run_ab_canopy_pattern.py` therefore defaults to `--gamma-loop
  base` — that script measures mirror symmetry on an unsteered load case, and
  `base` is exactly mirror-symmetric where `anderson` is not.
  **Do not generalise that default to post-stall work.** AV is what regularises
  a stalled circulation, the accelerator is what makes AV-active solves
  tractable, and Anderson's failure mode is a FALLBACK to base ("deep post-stall
  limit cycle; caller falls back to base loop"), not a dead end — so this is a
  branch-selection defect, not a reachability one, and it is no reason to take
  the accelerator off the depower and steering drivers. as_config keeps
  `anderson` + `allowed_error: 1e-8`, a matched pair.
  `base` does NOT suppress a real stall. On the one genuinely stalled Billow
  state measured (`cross_steer_p0150mm_persegment`, u_s 0.15, converged at
  v_a 15.65 and 0.015 N), a COLD re-trim of that converged shape on `base`
  finds the stall and reproduces the run — 1/54 panels 10.8 deg past onset,
  v_a 15.43 — while cold `anderson` jumps to the high-speed branch at v_a 30.12
  and fails to reproduce the run it came from, whose own anderson trims were
  warm-started. So the unreliability is COLD anderson specifically. `base` paid
  2.4x for it there, 19.6 s against 8.0 s: the accelerator does earn its keep
  post-stall, which is the reason not to take it off the steered drivers on the
  strength of an unsteered A/B. One state, one trim — a coupled steered run on
  `base` has not been measured.
  The branch is a SOLVER-PATH fact, not a mesh one: the built shape is
  identical for both strut paddings, because padding nodes are chordwise-
  interior and the aero mesh is built from LE/TE nodes only — they move
  0.000 mm, the mapped LE/TE arrays are bit-identical and the built trims agree
  to 3e-8 deg. The stalled branch is also ill-conditioned, which is why its
  answer wanders: the two paddings agree to 3e-8 deg on the attached branch and
  differ by 1 deg on the stalled one. `aerodynamic.attached_polars` fixes the
  same failure at the same cost (0.018 N, 7 it, 0.026 mm, 97 s) but solves a
  CONTINUED-polar model that must then be validated on the true polars
  (`check_tip_stall.py`, which forces `base` for exactly this reason) — on a
  genuinely stalled wing it would hide the stall, which is why it is not a
  default.

- **Frames are transported over the beam tree, not seeded per chain.** The
  reference curvature is `omega_0 = psi / L0` with `psi` a Rodrigues vector, so
  it grows like `tan(theta/2)` and is singular at `theta = pi`. Seeding each
  chain's roll independently piles an arbitrary roll on top of the physical
  LE/strut joint angle and took the worst element to 173 degrees, i.e.
  `omega_0 = 753 /m` on a 17 mm element. Minimal-rotation transport over the
  beam network plus `junction_frame="strut"` (the joint rotation lands on the
  long LE runs, not the short strut stubs) gives 10.6 /m. The tube section is
  isotropic in roll (`ga_2 == ga_3`, one bending law), so only `d1` is
  physically determined and the roll is free to spend this way -- as a gauge
  along a member, NOT node by node: the element sees the relative rotation of
  its two end frames, so the roll field also has to be mirror-consistent (next
  item).
- **The frames of a mirror-symmetric kite are mirrored, not transported, onto
  the far half** (`symmetric_frames`, config `mirror_frames: true`, default).
  The transport above is not mirror-equivariant on an LEI kite: the LE crosses
  the plane (the reflection reverses its tangent) and the struts do not, so
  every hop through a strut junction rolls the far half the wrong way -- up to
  60 degrees on the LEI-V3, with every node position still symmetric to
  4e-15 m. The tube energy then was not mirror-symmetric (internal moment
  mirror error 40%, a 190 N reaction on a symmetric-constrained solve) and the
  unsteered kite solved to one deterministic asymmetric shape, worst at the
  tips and trailing edge (40-69 mm of shape coarse). Mirrored with one
  director-sign matrix, `R_p = M R diag(-1, 1, 1)`, every element is exactly
  mirror-invariant and the coupled unsteered solve is symmetric to 2e-5 mm.
  Near half untouched; asymmetric geometry left as built; a beam member lying
  IN the plane (a centre strut) raises, and a plane-crossing element joining
  frames > 90 degrees apart is warned about (Cayley singularity). Before
  2026-09-11 every Billow run used transported frames; `mirror_frames: false`
  reproduces them. See `docs/billow/integration.md` §8.6.
- **`run_billow` is a whole solve, not one step.** With a `move_limit` each
  `MinimumEnergySolver.solve` is one trust-region step and `run_billow` walks
  them to force balance, terminating on the residual — IPOPT reports success for
  the *boxed* problem while sitting on the boundary, so its verdict does not
  imply equilibrium.

`build_solver(model, settings, equalities=None)` is the one place the
`structural_billow` numerics become a `MinimumEnergySolver`, so a constrained
solve runs with exactly the numerics of the free one it is compared against;
`symmetric_equalities(structure)` returns the mirror-symmetry equalities over
EVERY node of the built model (bridle knots and quad centres included) and the
partner map. Diagnostics: `scripts/aerostructural/check_symmetric_equilibrium.py`
(free vs symmetric-constrained vs released solves, `Pi`, and the tangent
stiffness on the symmetric/antisymmetric subspaces -- IPOPT checks first-order
optimality only, so a released solve alone would stop on a symmetric saddle),
`check_mirror_asymmetry.py` (global vs rigid-aligned mismatch per section pair
and per chordwise station, LE to TE).

### Steering in the shared driver

`fem/aerostructural_coupled_solver.main` actuates the steering tapes ONLY when
the caller passes `steering_tape_indices` (the reader's two `steering_tape`
elements). Without them `steering_tape_final_extension` is ignored, as it always
was on this driver -- as_config ships 0.3 m of it, and every caller written
before 2026-09-11 would otherwise start steering. With them, the half-difference
(first tape shortened, second lengthened, the `pss/actuation.py` convention)
walks toward `steering_tape_final_extension` in `steering_tape_extension_step`
increments (0 = one step), each taken from a CONVERGED state after the depower
walk has arrived, each followed by a fresh stagnation window and a settle
guard. The guard is a STOPPING RULE, not a fixed hold: after a step, an exit
needs the residual gate AND (a) at least `steering_settle_iterations_after_update`
iterations (the dead band before the course rate moves) AND (b) the trim's
course rate settled -- `remaining_drift` of its history since the step within
max(`steering_settle_course_rate_tol` [rad/s, default 1e-3],
`steering_settle_course_rate_rtol` x |course rate| [default 0]); the exit
estimate is recorded as `meta["course_rate_remaining"]`. `remaining_drift`
extrapolates the geometric tail with the LARGEST of the last three change
ratios, because a fast transient dying onto a slow tail fools a single ratio
(measured: 0.0007 rad/s change with 0.02 rad/s still to come); a growing tail
is never settled. The residual cannot stand in for (b): it is the load change
between iterations, which a slowly relaxing loop keeps below the gate while the
steered state is still growing.

**The attitude split was the bridle-line drag (found and fixed 2026-09-12).**
Until then the structure's equilibrium sat ~0.78 deg rotated about the KCU from
the trimmed attitude on EVERY iteration (unsteered too, in pitch), the trim
rotated it back, and that non-decaying rigid part pinned Aitken at its 0.05
floor: the course rate crawled (0.309 -> 0.322 rad/s over 40 iterations, ratio
~0.97) and with relaxation off (omega 1) drifted AWAY with growing steps
(0.355 rad/s at 26 iterations, residual rising); +-10 cm looked geometric for
~25 iterations then ran away. The cause was not the relaxation and not the
inertias: **the trim carries the bridle-line drag inside its own balance** (the
VSM body is built with `bridle_path`, so `calculate_results` adds each
segment's force and its moment about the reference point), while the driver
handed the structure the same lines re-evaluated at `vel_app` -- the FREESTREAM
vector frozen in `aerodynamic_vsm.initialize` from `wind_speed_wind_ref`, 8 m/s
whatever wind the run asked for. 20.3 N against 76.3 N at v_a 15.9, leaving
-326 N m of pitch about the pinned bridle point; a pin carries force, not
moment, so the structure shed it as a 0.763 deg rigid swing. Both call sites
now go through `_bridle_line_drag`, which takes the trim's own
`results_aero["va_vel_world"]` (the apparent wind in the VSM frame, despite the
name) and honours `is_with_aero_bridle` -- the pre-loop call ignored that gate,
so a bridle-drag-OFF run still got it once, on the state every later iteration
starts from. Measured budget about the bridle point, LEI-V3 cross canopy,
54 panels, v_w 4.2: wing load transfer 0.9 N m (`moment_matched`, built shape),
inertial+gravity 1.1 N m, bridle drag 325.9 -> **15.9 N m**, net swing
0.763 -> **0.038 deg**. Fixing this MOVES every stored Billow/FEM coupled
result.

**And the VSM's bridle now deforms with the kite (2026-09-12).**
`BodyAerodynamics.instantiate(bridle_path=...)` bakes the segments in as
COORDINATES and `update_from_points` refreshes the wings only, so the trim went
on charging bridle drag to the BUILT shape while the structure (and the drag
`_bridle_line_drag` gives it) followed the deformed one -- the 16 N m above,
and 24 N m on a steered state, where the deformed bridle is asymmetric and the
built one cannot be. Every aero call now passes `struc_nodes` +
`bridle_line_specs`, so `run_vsm_package` rebuilds them
(`rebuild_bridle_line_system`), as the PSS driver always did.
`_bridle_line_specs_for_vsm` assembles the rows without a new argument on
`main`: node indices from the reader's `bridle_connectivity_arr` (which owns
the structural array), diameters from the body's own segments (which
`aerodynamic_vsm.initialize` has already replaced with the DRAG-equivalent
ones, a flat tape's projected width). Both lists are the same parse of
`bridle_connections` in the same order -- verified on LEI-V3 FEM_full, all 87
segments agreeing to 7.6e-9 m -- and a count mismatch skips the rebuild with a
warning rather than pairing lines to the wrong nodes. What is left of the
bridle disagreement is the attitude increment (the trim rotates the wing, not
the bridle), which vanishes as the loop converges: 0.2 N m on a converged
steered state.

**And every bridle segment is now charged at its OWN inflow (2026-09-12).**
The relation is single-sourced in `awetrim.aerodynamics.apparent_wind`:

```text
va(r) = va_free - omega x (r - r0)
```

so the rotational term belongs to the station it is evaluated at, and a load
spread over many stations cannot be charged one vector. Since 450d1c3 the
FEM/Billow driver was the only one doing that, so it disagreed with the trim
whose balance it was supposed to be feeding:

| consumer | charged, before | on the LEI-V3 steered case |
|---|---|---|
| VSM `compute_results` (the trim's own balance) | `va_ref_vector` = the freestream | 78.9 N, 419 N m |
| PSS driver, `aerodynamic_bridle_line_drag.main` | `body_aero.va`, the freestream | 78.9 N, 419 N m |
| FEM/Billow driver, `_shared_bridle_line_drag` | per-segment | 72.8 N, 379 N m |

**`va_ref_vector` is NOT the wing's mean inflow** — an easy misreading, made in
450d1c3's own message and again when auditing it. It is
`_compute_reference_velocity_from_distribution(self._va, ...)`, and `self._va`
is the inflow as handed to the VSM's `va` setter, BEFORE `-omega x (r - r0)` is
added to build the panel distribution; the rotational term lives on `panel.va`
only. For the uniform freestream every AWETrim call passes, `va_ref_vector` IS
that freestream. So the bridle never carried a rotational term at all — it was
charged the inflow at the reference point wherever a segment sat, not the
wing's version of it.

All three now evaluate per segment and agree to machine precision (verified on
the 45-segment LEI-V3 bridle, steered and unsteered). Closing the trim's own
row needed the VSM: `compute_results` evaluates each segment at its midpoint
and publishes `bridle_line_forces` / `bridle_line_midpoints` so a driver can
take the load the trim balanced instead of recomputing it, and the `va` setter
now STORES `reference_point` (exposed as `body_aero.reference_point`) — it used
to live on the stack, so `inflow_state_of` and its predecessors could only
default `r0` to the origin and be accidentally right. It also passes `rho`
through, which the bridle call had been dropping in favour of the 1.225
default (latent: every shipped config is 1.225).

**Do not reuse the 8.28 m/s / "roughly doubles" figure** from the first version
of this fix (450d1c3). That is `|omega| * |r|`, the bound the cross product
reaches only with `r` PERPENDICULAR to `omega`; the wing sits **2.6 deg off**
the rotation axis (`omega` is dominantly the radial course rate and the wing is
almost straight out along that same radial), so `omega x r` is near its MINIMUM
there. Measured: area-weighted 1.99 m/s at the wing, 1.26 m/s at the bridle
midpoints, and the spread between inflow choices is -7.6% to +8.3% on the
force, not 2x. Reaching 8.28 m/s would need `sin` of the arm/axis angle to be
1.10. The steered-Billow divergence that commit was chasing is therefore NOT
attributable to it; the mass-split/CG correction bundled into the same commit
(0.64 m, 39 N m of roll) is the far larger effect. All of this vanishes at
`omega = 0`, which is why unsteered baselines are blind to the whole question.

**The inertial and gravity split now weighs nodes exactly as the CG does
(2026-09-12).** `distribute_total_force_by_particle_mass` spreads the trim's
`inertial_force` / `gravity_force` over the nodes, and the trim applied those
resultants as point loads at `calculate_cg`; the two are the same load only if
both weigh every node the same way. This driver kept its own copy of the split
that CLIPPED negative masses away and normalised by the positive sum, while
`calculate_cg` weighs them as they are. The masses do go negative: the reader
hangs `mass_without_bridles - mass_canopy` on the LE/strut nodes, which is
negative whenever the YAML's `canopy_density` over-fills the wing's mass, and
on `struc_geometry_FEM_full.yaml` 96 tube nodes carry about -0.037 kg each
(the reader now warns). The two CGs then sat 0.64 m apart in z (3.24 vs 3.88)
-- a 20% error on every inertial and gravity moment the structure was handed,
worth 39 N m of ROLL on a steered state and invisible at zero course rate. The
duplicate is gone; `..forces` is the one split, shared with the PSS driver,
and `tests/aerostructural/test_forces.py` locks the property that matters (the
split's moment equals `cg x F`, negative masses included).

**The PSS/QSM driver had two of its own (2026-09-12), both now fixed.** Its
in-loop assembly was always consistent -- it is where the three patterns above
came from, and its budget on the PSM geometry (45 panels, 40 nodes, v_w 4.2,
gravity off, tether in trim) is net 1.3 N m and a 0.004 deg swing: wing
transfer 2.0 N m, bridle 5.2 (attitude residue), inertial 0.0, trim residual
1.9 from the live `max_nfev` cap. Its wing transfer beats the FEM/Billow one
(2.0 against 7.9 N m) because it applies each panel load at its OWN cp through
a bilinear corner map -- no chordwise spread, so no placement prior to be wrong
about. But:

- its PRE-LOOP bridle-drag call was commented out (`f_aero_bridle` zeroed)
  while the trim balanced 270.8 N m of bridle moment, so the FIRST structural
  solve was 274.7 N m out about the pinned bridle point and swung 0.886 deg.
  The fixed point does not move, but the loop is seeded off it -- and this map
  has branches to be seeded onto.
- `bridle_node_pairs` came from `build_bridle_node_pairs_from_line_system`,
  which nearest-node matches the VSM's BUILT segment coordinates against
  whatever `struc_nodes` the driver is handed. Clean on the built shape (0 of
  45 collapsed), but a converged shape moves nodes up to 0.571 m and then 2 of
  45 segments snap BOTH ends onto one node -- zero length, **NaN** forces --
  with 8 more re-paired. `run_simulation_PSM`'s `starting_from_sim_subdir`
  continuation starts exactly there. `_bridle_node_pairs` now takes the indices
  from `bridle_line_specs`, which is what they are; the geometric match stays
  as the fallback when a caller passes no specs. Verified identical to the old
  pairing on the built shape, so cold runs keep their behaviour.

Diagnostic: `scripts/aerostructural/check_trim_structure_moment.py` -- one
trim, then the full moment budget of both sides about the bridle point,
term by term, plus the rigid swing that nulls the structure's loads.
`--backend billow|pss` replicates either driver's load assembly (the pss one on
the PSM photogrammetry geometry, with the tether in the trim when the config
says so); `--from-result <folder>` runs it on a stored converged, and for a
steered run asymmetric, shape. It is the only check that sees any of this:
`check_moment_preservation` compares the mapping against the
already-distributed loads and `check_load_transfer.py` compares routes against
each other, so a term the trim balances and the structure never receives -- or
receives at a different wind speed, or at a different CG -- is invisible to
both. On the converged steered +-5 cm shape with all three fixed: bridle drag
0.2 N m, inertial 0.0, **wing transfer 7.9 N m** (chordwise 4.9, spatial 3.7 --
the one term left, and a modelling limit rather than a bug), net 8.0 N m and a
**0.020 deg** swing, from 0.761 deg.

Earlier steered snapshots (taken with `steering_settle_course_rate_rtol: 0.1`,
`run_steering_BILLOW.py --course-rate-rtol 0.1`, reported with
`meta["course_rate_remaining"]`: +-10 cm 0.580 rad/s extrapolated 0.635,
+-15 cm 0.714 / 0.760) all predate the fix and are not comparable to states
solved after it. `update_steering_tape_actuation` reads the half-difference back from the live
rest lengths through `_rest_length` / `_set_rest_length`, the one getter/setter
pair for all three backends. Tracking gains `steering_half_difference` and
`trim_state` (the trim's `[kite_speed, roll, pitch, yaw, course_rate]` per
iteration; the attitude entries are INCREMENTS the geometry is rotated by, so
read the kite's attitude off the positions, not the last row). Script:
`scripts/aerostructural/run_steering_BILLOW.py`.

A gravity-only load case is NOT a valid smoke test: the KCU is pinned below the
wing, so gravity slackens every bridle line and the bridle knots become a
mechanism. Billow reports that faithfully (residual = the free knots' weight);
`kite_fem` hides it because `I_stiffness=25` acts as a ground spring on every
node. Load the wing away from the KCU, as the aero does.

## Chordwise load distribution and the pitching moment

`aero2struc.main`'s `chordwise_distribution` key decides where along each chord
a panel's force is placed, and **that station is the panel's local pitching
moment**:

- `cp_file` (default, reproduces every stored result) applies one measured
  `Delta C_p` shape to every panel. On the LEI-V3 file its centroid is
  `0.291 c`, so every panel gets that station whatever its own `C_m` says.
- `moment_matched` keeps that shape as a prior and tilts it onto the centre of
  pressure VSM already computes from each panel's `F` and `M`
  (`panel_cp_locations`), via `weights_at_centre_of_pressure`. The tilt is
  `w_i ∝ w0_i exp(lambda t_i)` with `lambda` solved from the required centroid:
  strictly positive weights, force preserved exactly, and the
  minimum-relative-entropy correction, so the measured shape is kept wherever
  the moment does not contradict it.

Measured on one LEI-V3 trim (135 panels): the actual centre of pressure runs
`0.303 .. 0.805 c` (the high end is the stalled tips), and the fixed station
imposes **917 N m** of spurious pitching moment, ~-10 N m on nearly every
panel with the SAME sign — a systematic bias, which is exactly what shifts a
moment balance. `moment_matched` takes it to 6e-9 N m.

**`check_moment_preservation` does not measure this.** Both call sites hand it
the already-distributed loads, so it reports the error of the *spatial*
nearest-node step (`map_aero_forces_to_struct_nodes`) alone. That error is
0.09% pre-loop but **22% once the kite has deformed** — a separate, open defect
in the same family.

## Load transfer onto the canopy (`load_transfer`)

Where the chordwise weights put each panel's load is one question; how that
load then reaches the structural NODES is another. `aero2struc.main`'s
`load_transfer` key (used whenever the caller passes `canopy_triangles` and
`canopy_grid`, i.e. the billow backend) picks the route:

- `traction` (default) -- the load as a FIELD over the canopy, integrated over
  every element: `map_aero_traction_to_membrane`. Panel `k` covers a strip of
  the canopy's surface coordinates `(s, xi)` (`canopy_surface_coordinates`: `s`
  the grid row, `xi` the projection onto that row's chord) with traction
  `F_k q_k(xi) / ds_k`, where `q_k` is the piecewise-linear density whose
  consistent nodal loads reproduce the panel's chordwise weights exactly
  (`consistent_chordwise_density`, keeps force AND centre of pressure). Each
  element gets `f_a = int tau N_a dA`, evaluated exactly by clipping against the
  strip/chord-bin boxes. So every canopy node -- quad centres and refined
  interior nodes included -- is loaded by the elements around it, and the total
  force is conserved to roundoff by construction.

  The coordinates are taken from the CURRENT shape, each row then made strictly
  increasing leading to trailing edge. Both halves are measured necessities.
  Current, because under load the canopy slides aft of its built chord
  fractions (x2 cross: 1-3.5% of the chord on average over the aft half), so
  coordinates frozen on the built shape carry every load aft with it -- 17% of
  the moment about the KCU. Repaired, because where slack fabric curls near the
  trailing edge a node projects past its aft neighbour and an element turns
  inside out in `(s, xi)`: that patch covered twice, its load counted twice
  (x2: one column in 3 rows, 0.16-0.25% of the load, a pair of trailing-edge
  nodes unloaded). With every row increasing and each quad centre at the mean
  of its corners no element of any pattern can invert, so a coverage mismatch
  can only be a bug and is raised as one.
- `nearest_element` -- each chordwise point load through its nearest triangle
  (barycentric). Exact force and in-plane moment, but a point reaches three
  nodes: on a x3 canopy 61% of the nodes stay unloaded.
- `sections` -- the historical lattice mapping onto the YAML's chordwise node
  chains; the only route for pss / kite_fem.

`scripts/aerostructural/check_load_transfer.py` compares the three on one aero
state. Measured on the coarse `cross` canopy (413 canopy nodes):

| route | loaded | force rel. error | moment rel. diff. |
|---|---|---|---|
| `sections` | 224 | 1e-15 | 3.9e-3 |
| `nearest_element` | 368 | 1e-15 | 6.8e-7 |
| `traction` | **413** | 2e-15 | 6.3e-4 |

And on a converged, DEFORMED x2 canopy (`--refine 2 --from-result ...`, 1581
canopy nodes), where the coordinates matter:

| route | loaded | force rel. error | moment rel. diff. |
|---|---|---|---|
| `sections` | 689 | 4e-16 | 4.0e-2 |
| `nearest_element` | 914 | 4e-16 | 1.7e-3 |
| `traction` | **1581** | 5e-16 | 8.6e-3 (4 N m) |
| `traction`, coordinates frozen on the built shape | 1581 | 7e-16 | 1.7e-1 |

The relative moment is taken about the KCU, where the resultant nearly passes,
so it is harsh: 4 N m on ~2.2 kN is an effective lever of 2 mm.

The `traction` moment is measured against the loads on the aero CHORD LINE; it
places them on the cambered SURFACE instead, which is where the pressure acts,
so its difference is the camber offset and not a transfer error. On a planar
wing, where the two coincide, the moment is exact
(`tests/aerostructural/test_aero2struc.py`).

## Config Keys (aerostructural_configs/config.yaml)

All defaults are defined in `scripts/aerostructural/common.CONFIG_DEFAULTS`. Key sections:

```yaml
aerodynamic:
  n_aero_panels_per_struc_section: 5   # x 9 sections = 45 panels (2026-08-31)
  spanwise_panel_distribution: uniform
  max_iterations: 1000
  allowed_error: 1.0e-8                # matched pair with gamma_loop_type --
  gamma_loop_type: anderson            #   revert BOTH to base / 2e-6 together
  anderson_max_iterations: 1000        # headroom INSTEAD of the Picard
  anderson_fallback_to_base: false     #   fallback (rescue rate 0.1%)
  relaxation_factor: 0.05
  reference_point: [0.0, 0.0, 0.0]
  is_with_artificial_viscosity: true   # Li/Gaunaa spanwise artificial viscosity
  artificial_viscosity_factor: 0.035   #   (TORQUE 2026); ON = the model default

```

Defaults since 2026-08-31, all measured that day on the LEI-V3 centre-sweep
reference point. `gamma_loop_type: anderson` is Anderson-accelerated: ~1.5x per
coupled point with the same converged state as Picard to <=0.15%, steered
points included — but ONLY at `allowed_error` ~1e-8, because it terminates on
a superlinear, non-smooth residual that corrupts the finite-difference Jacobian
of the QSM trim the coupled solver calls (measured -2.2% v_tau at 2e-6). The
scripts refuse the unsafe half-pairing:
`run_state_aerostructural_stability.py --as-gamma-loop anderson` requires
`--as-gamma-tolerance`. On steered, AV-active flow 1e-8 can be unreachable for
ANY loop — such a point fails and is recorded, accepted by design. Anderson's
base-loop fallback is OFF in as_config since 2026-09-03
(`anderson_fallback_to_base: false` + `anderson_max_iterations: 1000`, both
plumbed through `aerodynamic_vsm.initialize`): measured across the 2019+2025
steering campaigns, the fallback rescued 99 of ~92,400 Anderson failures
(0.1%) while costing up to two 1500-iteration relaxed-Picard crawls per
failure — headroom for the accelerated loop is cheaper in every case. With artificial viscosity ON the circulation
problem is multi-valued near stall and the branch is fixed by the seed, so a
coupled run's one COLD aero solve chooses it and every warm-seeded iteration
after that inherits it; AV-on results (stalled-tip branch, CD +6.7% at the
reference point) are not comparable to the AV-off history. The wes-quasi-steady
trim scripts inherit `is_with_artificial_viscosity` from the kite's as_config
when `--artificial-viscosity` is not given, so the re-solve lens stays on the
same model as the deformation.

```yaml
structural_pss:
  solver: pss                  # 'pss' (kinetic damping, default) | 'nlp'
                               # (exact CasADi/IPOPT minimum-energy solve).
                               # Same spring physics; measured 2026-09-09 on the
                               # actuated reference case: nlp is 100-300x faster
                               # per inner solve (2-3 s -> ~10 ms) and removes
                               # the ~1 N kinetic-damping leftover that floors
                               # the coupled relative residual at ~4e-4 -- the
                               # only way this case converged a 1e-4 gate.
                               # PSS state (psystem) stays the owner of rest
                               # lengths/stiffness, so actuation, the stiffness
                               # ramp and handover are solver-agnostic.
  nlp_tolerance: 1.0e-8        # IPOPT tol (nlp only)
  nlp_max_iterations: 1000     # IPOPT iteration cap (nlp only)
  nlp_anchor_stiffness: 1.0e-3 # [N/m] pins force-free fully-slack nodes (nlp only)
  dt: 0.005
  n_internal_time_steps: 100   # must be >> 2000 for convergence check to fire
  abs_tol: 1.0e-50
  rel_tol: 1.0e-5
  max_iter: 500
  kinetic_energy_tolerance: 1.0e-3
  fixed_point_indices: [0]

aero_structural_solver:
  max_iter: 100
  residual_tol_relative: 1.0e-4   # preferred: dimensionless, see below
  tol: 5.0                        # legacy absolute [N], used only if the above is absent
  relaxation_factor: 0.5
  is_with_aitken_relaxation: true
  # Adaptive Aitken floor (opt-in, EXPERIMENTAL -- measured NOT to help).
  # Far from the fixed point the floor is relaxation_min_far; once the
  # residual first drops below relaxation_release_factor x tolerance it
  # releases (latched) to relaxation_min. Hypothesis was that the small
  # floor then damps the wide-floor limit cycle; measured 2026-09-09 on the
  # actuated case (NLP inner, tight gate) the release instead RE-ENTERS the
  # period-4 limit cycle at LARGER amplitude (4.8/2.9/2.2/6.0 N vs the
  # fixed-0.3 stall at 0.33 N): the cycle is an oscillatory VECTOR mode of
  # the coupled map (attitude rotation vs relaxed deformation) that no
  # scalar omega can damp from that entry point. Kept for experiments;
  # the real fix is vector (Anderson) acceleration of the outer geometry
  # fixed point. Default relaxation_min_far == relaxation_min = off.
  # relaxation_min_far: 0.3
  # relaxation_release_factor: 30.0
  # Anderson-accelerated outer fixed point (opt-in; use with
  # structural_pss.solver: nlp -- PSS's ~1 N inner noise corrupts the
  # residual differences). Replaces the scalar Aitken update with a vector
  # extrapolation over the last anderson_outer_depth iterates -- the tool
  # for the oscillatory period-4 coupling mode no scalar omega can damp.
  # History resets on any map change (tape actuation, stiffness event);
  # steps whose largest node move exceeds anderson_outer_max_step_m fall
  # back to the plain beta-relaxed step.
  # FRAME-CONSISTENT since 2026-09-10: the driver rotates struc_nodes by
  # the solved trim attitude every iteration, so consecutive outer iterates
  # live in different frames; the first Anderson attempt extrapolated
  # across them and its differences were dominated by the attitude
  # increment (100 iters stuck at 2.5e-3--4.6e-3, worse than Aitken). The
  # stored history (positions, residuals, previous increment) is now
  # rotated with exactly the geometry's rotation at each attitude update
  # (_rotate_anderson_history, matrix from utils.rotation_matrix_from_angles),
  # the de-rotated smooth-map formulation. Aitken is deliberately
  # untouched -- its historical behaviour is the campaign baseline.
  # MEASURED on the rotated history (actuated case, NLP inner, full trim):
  # with the noise-robust defaults below the outer iteration contracts
  # MONOTONICALLY ~1e-3 -> 1.2e-4 in ~25 post-actuation iterations on the
  # correct branch, then floors at the map's own evaluation noise
  # (~1.2e-4 relative); beta 0.3 floors at 1.5e-4 in 45 iters; the old
  # aggressive knobs (depth 4, reg 1e-8, beta 0.5) still bounce and beta
  # 1.0 diverges onto a wrong branch. The tight 1e-4 gate therefore still
  # belongs to pinned Aitken 0.05 (91 iters to 5.4e-5, averaging through
  # the noise); identified follow-ups are an Anderson->Aitken handoff near
  # the fixed point or cutting the map noise (tighter trim/gamma
  # tolerances) below 1e-4.
  # outer_acceleration: anderson   # default: aitken
  # anderson_outer_depth: 2
  # anderson_outer_beta: 0.2
  # anderson_outer_reg: 1.0e-4
  # anderson_outer_max_step_m: 0.1
  qs_speed_bound_patience: 3      # runaway stop, see below (0 disables)
  steering_settle_iterations_after_update: 6   # steering settle, see below
```

**Runaway stop (2026-09-01).** `aerodynamic_vsm.run_vsm_package` returns
`results["trim_on_bounds"]` (names of trim unknowns sitting on a search
bound); when `"kite_speed"` is in it for `qs_speed_bound_patience`
consecutive coupled iterations the loop ends immediately with
`converged=False` and `meta["stop_reason"] = "trim_speed_bound"`, and the
plateau fallback can never accept such a run. A bound-pinned trim is a
constrained optimum, not an equilibrium; before this a handover-seeded
deep-depower point burned its whole iteration budget pinned at 40 m/s
before the sweep's cold retry.

**Direct steering preset (2026-09-10).** Opt-in config key
`steering_tape_preset_extension`: sets the asymmetric steering tape rest
lengths (left = initial − δ, right = initial + δ) once, BEFORE the coupled
loop. Exists because the geometry file format carries one symmetric row per
tape, so a snapshot of a steered state cannot store its own actuation and
every restart otherwise re-walks the steering from zero — which re-selects
the solution family instead of resuming the state. Preset == 
`steering_tape_final_extension` → no walk at all; a different final target
walks only the difference. Cold solves must NOT preset (the ramp doubles as
their load continuation); preset only from a deformed snapshot equilibrated
at these lengths.

**Trim-state history / unloaded-branch reference (2026-09-10).** The coupled
solver's meta now carries `opt_x_history` — the trim state
`[kite_speed, roll, pitch, yaw, course_rate]` at every coupled iteration,
shape (n, 5) — and `course_rate_max_settled`, the largest |course rate| on
iterations where the steering actuation was fully applied and settled.
Because the in-loop actuation ramp makes every cold solve its own
continuation in tape length, a final |opt_x[4]| far below
`course_rate_max_settled` marks a solve that fell off the LOADED steering
branch onto the second, UNLOADED attached equilibrium of the tension-only
bridle (steering input absorbed by slack lines — measured 2026-09-10 on the
2019 reel-out steering chains: roll 11.6 -> 5.2 deg, chi_dot -17..-54% at
MORE steering, tips unfolded, zero stalled panels, so the attached/fold
checks alone cannot catch it). Sweep callers use this as the donor-free
branch check; chained rows compare against the donor row's course rate
instead.

**Steering settle (2026-09-01).** `steering_settle_iterations_after_update`
(default 6) blocks every convergence exit for that many coupled iterations
after a steering tape update, the way `depower_settle_iterations_after_update`
does for depower but longer. A tape half-difference first moves the geometry
by millimetres; the steered equilibrium (rolled wing, turning trim) is
reached by the coupled fixed-point iteration amplifying that asymmetry over
several iterations, and until it does the residual sits far below the gate:
without the settle the u_s = 0.025 / 0.05 rows of the 2019 reel-out steering
continuation "converged" in 3 iterations on the still-symmetric state (roll
0.00 deg, chi_dot 0), while the 0.075 row -- whose residual also dipped to
0.37 N at the same point -- went on to 2.1, 17, 21 N before settling rolled.
Rows with no steering update (u_s = 0, or a handover already at the target
u_s) are unaffected.

**VSM requirement for steered coupled solves (2026-09-01).** The structural
-> aero mapper hands `BodyAerodynamics.update_from_points` the sections in
arc order; the VSM must keep that order (`Wing.preserve_section_order`,
set by `update_wing_from_points` in the Vortex-Step-Method checkout since
2026-09-01). Older VSM re-sorts the sections with a nearest-neighbour chain
and, once a steered LEI tip curls inboard past its neighbour (u_s >= ~0.125
m tape), folds the lifting line back over itself: near-coincident control
points, a circulation-map eigenvalue above 1, NaN gamma in both coupled
stages ("Residuals are not finite in the initial point"). The VSM now also
logs "Wing sections double back" for a genuinely folded mesh and stops its
gamma loops on the first non-finite iterate. Reproduce with
`scripts/personal/wes-quasi-steady/probe_vsm_steered_divergence.py` on an
`attached_failed_vw_*/deformation` snapshot.

### Convergence criterion

The coupled loop converges on the global nodal force residual
`f_res = f_int + f_ext` (fixed nodes zeroed), judged one of two ways:

| key | measure | meaning |
|-----|---------|---------|
| `residual_tol_relative` | `\|\|f_res\|\| / F_tether` [-] | **preferred.** `F_tether = \|sum f_ext\|` is the reaction the constrained bridle node carries |
| `tol` | `\|\|f_res\|\|` [N] | legacy; active only when the relative key is absent |

`resolve_residual_tolerances` decides which is live and returns the tolerance
in the active unit, so the convergence test, the stagnation window, the
adaptive dt and the stiffness trigger can never disagree about what
"converged" means. `resultant_tether_force` is the single definition of the
normalising force and `relative_residual_norm` the single place the ratio is
formed. A value >= 1 in the relative key raises rather than being read as
newtons.

**Why the absolute gate is not good enough.** One force tolerance means
different things across a sweep: on LEI-V3 `tol: 5` is 6.8e-4 relative at
7.4 kN of tether load but 1.3e-3 at 3.9 kN. Measured over a 6-row depower x
steering sweep, that was loose enough for the SAME point to land 5-15% apart
in CL, CD and tether force depending only on whether it was reached cold or
from a warm start -- the warm start approaches from closer and so trips the
gate much further from the fixed point. Any state handover, continuation or
warm start is therefore only meaningful under the relative criterion.

### Trim evaluation cap (quasi_steady_trim.max_nfev)

Opt-in key in the ``quasi_steady_trim`` block (read by
``aerodynamic_vsm.run_vsm_package``, forwarded to both trim solvers): caps the
trim's ``least_squares`` at that many residual evaluations PER COUPLED
ITERATION, so trim unknowns and geometry converge together instead of fully
re-converging the trim on every intermediate shape. A capped, still-converging
trim reports ``success=False``; ``run_vsm_package`` keeps that partial result
(marked ``trim_truncated``) instead of taking the direct-solve fallback.
Warm-started trims near the coupled fixed point terminate inside the cap on
their own tolerances, so the accepted final state is a genuinely converged
trim. Measured 2026-09-09 (NLP inner solve, freed relaxation): ``max_nfev: 8``
cut total VSM evaluations ~30% at identical outer iteration count and final
state; 20 was WORSE than 8 — spend little per iteration. Absent key =
historical behaviour.

## Result Storage

Output goes to `results/aerostructural/<kite_name>/<case_folder>/sim_output.h5` (absolute path from project root, never CWD-relative). Use `results.save_sim_output()` and `results.aerostructural_results_root()`.

## Required Developer Checks

- Read `structural_geometry_io.main()` before changing how struc_geometry.yaml is parsed; the node index ordering (odd = LE, even = TE) and pulley dict format `[cj, ck, l0_cj_ck, l0_ci_cj, ci]` are load-bearing.
- Any change to `PssQsmCoupler` must keep `QsmCouplingRequest` / `QsmCouplingResult` stable; the protocol tests check these fields.
- Scripts in `scripts/aerostructural/` import shared helpers from `common.py` — add new shared defaults to `CONFIG_DEFAULTS` there, not as literals in individual scripts.
