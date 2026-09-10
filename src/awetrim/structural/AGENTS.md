# AWETrim Structural Module

## Status: 🟡 Standalone and validated, not yet coupled

A self-contained minimum-energy structural library: cables, frictionless pulleys,
geometrically exact Timoshenko beams and wrinkling membrane fabric, assembled into
one total potential energy and solved for static equilibrium with IPOPT.

**Nothing in `aerostructural/` imports this yet, and this package imports nothing
from AWETrim.** It depends only on NumPy and CasADi. That isolation is deliberate
and is the first rule to preserve: it lets the element physics be validated against
closed-form solutions before any of it touches a coupled run.

## Documentation

`docs/billow/billow.pdf` is the full technical document -- formulation, element
library, validation against external benchmarks, demonstration cases, the
`kite_fem` comparison and the canopy mesh study. Source is `billow.tex`
alongside it; rebuild instructions are in `docs/billow/README.md`.

**Billow** is the intended name for this package once it becomes its own
repository.

## Why it exists

`aerostructural/pss/structural_nlp.py` already solves the *current* bridle by
minimum energy, and it does that well. It does not extend, for two reasons:

1. **Physics.** Its element vocabulary is axial springs. Battens and leading-edge
   tubes need bending and torsion; the canopy needs a membrane that cannot carry
   compression.
2. **Scale.** It builds the objective with a Python loop over elements in MX, so
   the graph grows with the element count. At tens of springs that is free. At
   tens of thousands of fabric triangles the graph build, and especially
   `gradient`/`hessian` on it, becomes the bottleneck long before IPOPT does.

This package keeps the formulation (still an NLP, still IPOPT, still
minimum potential energy) and replaces the assembly.

## Public layout

```
src/awetrim/structural/
  __init__.py     StructuralModel, StructuralState, DofLayout, PotentialEnergy,
                  MinimumEnergySolver, StructuralSolution
  rotations.py    SO(3) kernels over an `xp` namespace (NumPy or CasADi):
                  cayley, cayley_vector, half_vector, skew, axial,
                  orthonormalize, frames_to_flat / flat_to_frames / unflatten_frame
  model.py        DofLayout (gather maps), StructuralState (positions + frames),
                  StructuralModel (elements + reference config + pinned DOF)
  energy.py       PotentialEnergy: mapped assembly, objective, internal_load,
                  tangent_stiffness, parameter packing
  solver.py       MinimumEnergySolver, StructuralSolution
  elements/
    base.py       ElementKernel protocol, ElementSet, local DOF accessors
    cable.py      CableKernel, PulleyKernel + build_cable_elements / build_pulley_elements
    beam.py       TimoshenkoBeamKernel, BeamSection, beam_strains,
                  build_beam_elements, initial_frames_from_polyline
    inflatable.py InflatableTubeLaw, InflatableBeamKernel,
                  build_inflatable_beam_elements, inflatable_beam_state --
                  the ASKITE tube fits integrated into a strain energy
    membrane.py   MembraneKernel, green_strain, membrane_reference,
                  build_membrane_elements, membrane_regimes, SLACK/WRINKLED/TAUT

tests/structural/
  test_rotations.py  Cayley round trips and half-angle identity
  test_cable.py      Hooke parity with PSS, slack cut, pulley tension equalisation
  test_beam.py       objectivity, O(h^2) convergence to Timoshenko, no shear
                     locking, torsion vs GJ, pre-curved members stress-free
  test_membrane.py   fabric on its own -- analytic SVK, the three wrinkling
                     branches, flat-plate inflation, mesh convergence, scale
  test_energy.py     mapped assembly vs a naive per-element sum (the key test),
                     internal_load vs finite differences, DOF layout
  test_benchmarks.py roll-up vs the exact circle (O(h^2) + load-step
                     independence), Bathe and Bolourchi 45 degree bend
  test_inflatable.py energy integrates the fitted moment; pure end-moment
                     solves land on the fitted curve; collapse reporting

scripts/structural/
  run_demo_cases.py           wrinkling / canopy_model / cantilever / sail /
                              scaling figures
  run_validation_benchmarks.py  elastica / rollup / bend45, this model against
                              kite_fem with matched properties
```

## The four design decisions

**1. One mapped kernel per element TYPE, not per element.** Each
`ElementKernel.energy` is compiled once to a small SX `casadi.Function` over a
single element and evaluated across the whole set with `Function.map` on a
gathered index matrix. The objective graph therefore holds one node per element
*type*. This is the entire scalability argument — never reintroduce a Python loop
over elements that builds MX.

Measured (flat clamped canopy, `run_demo_cases.py --case scaling`):

| triangles | DOF  | build | solve | IPOPT iters |
|-----------|------|-------|-------|-------------|
| 128       | 243  | 0.07 s| 0.05 s| 6  |
| 1800      | 2883 | 0.67 s| 1.25 s| 9  |
| 5832      | 9075 | 3.0 s | 6.1 s | 9  |

Iteration count is essentially mesh independent; cost is in the sparse linear
algebra, where it belongs.

**2. Rotations are incremental Rodrigues (Cayley) vectors.** `R = cayley(psi) R_ref`
with `R_ref` stored in the state and the solve always starting from `psi = 0`.
Cayley rather than the exponential map because every resulting formula is
*rational* — no `sin(phi)/phi` removable singularity, so no `if_else` branch and
no kink in the Hessian IPOPT differentiates. Incremental rather than total because
the map is singular at `phi = pi`: the solver absorbs each solved increment into
`R_ref` and starts the next solve from `psi = 0`, so a node can turn arbitrarily
far across a load ramp while every individual solve stays well inside the chart.
An applied moment is work-conjugate to the exponential-map rotation vector, not
to `psi`, so the external work term converts (see `rotations.rotation_vector`);
using `-M . psi` directly made the answer depend on the load stepping.

**3. Rotational DOF only where a kernel asks for them.** Cable, pulley and membrane
nodes stay at three DOF. A canopy of 10 000 fabric nodes costs 30 000 DOF, not
60 000, whatever beams are attached elsewhere.

**4. Everything numeric is an NLP parameter.** Reference frames, every element
parameter column (rest lengths, stiffnesses, section properties), loads, anchor.
One `MinimumEnergySolver` build serves a whole sweep: actuation and stiffness
ramps change parameters, never topology. `StructuralModel.replaced` and
`ElementSet.with_param_column` are the supported way to do that.

## Validation

`scripts/structural/run_validation_benchmarks.py` runs three classical
large-rotation benchmarks in **both** this model and `kite_fem`/`pyfe3d`, with
matched linear section properties. References come from outside both codes.

| benchmark | reference | this model | kite_fem |
|-----------|-----------|-----------|----------|
| Euler elastica, alpha up to 10 | exact BVP solution | 1e-4 to 1.1e-3 L at 40 el | 1.6e-2 to 5e-1 L |
| Roll-up under M = 2 pi EI/L | exact closed circle | 1.13e-1 / 3.16e-2 / 8.14e-3 L at 10/20/40 el, O(h^2) | 5.9e-2 / 1.71e-1 / 2.32e-1 L, diverges with refinement |
| Bathe and Bolourchi 45 degree bend | published tip displacement | within the spread between published values, 0.2% on the magnitude | out-of-plane component 47% low |

Two things that comparison required, both worth knowing:

* `FEM_structure.solve` defaults to `I_stiffness=25`, an **absolute** N/m added
  to every DOF of the tangent. On these benchmark beams `EI/L^3` is 10 N/m, so
  the default is larger than the structure and the solve stalls at a residual of
  20 N. Benchmarks run it at zero. It is a kite-scale tuning constant, not a
  universal one.
* `kite_fem`'s beam matches linear theory to 0.02% below about one degree of
  rotation, so the property matching is right; its error is purely
  rotation-driven and **mesh independent**, which places it in the formulation
  rather than in the discretisation.

### The inflatable tube

`kite_fem` applies the ASKITE fits as *secant* stiffnesses, recomputing `EI` and
`GJ` from the current deflection and twist each iteration. That is a force law,
not a potential: substituting a state-dependent `EI(kappa)` into `1/2 EI kappa^2`
drops the `dEI/dkappa` terms, so the gradient stops being the internal moment.
`elements/inflatable.py` integrates them instead.

The bending fit is published as tip load against normalised tip deflection of a
**one-metre** cantilever. Since `kappa = 3 v / L` and `M = P L` at the
calibration length, it becomes an intrinsic constitutive law:

    M(kappa) = M_max (1 - exp(-EI_0 kappa / M_max)),  EI_0 = N/3,  M_max = D
    W_b(kappa) = M_max [ |kappa| - k0 (1 - exp(-|kappa|/k0)) ],  k0 = M_max/EI_0

quadratic near zero, saturating at `M_max`. Torsion is already intrinsic and
integrates directly to
`W_t = c1 [omega atan(c2 omega) - ln(1 + c2^2 omega^2)/(2 c2)]`, with
`GJ_0 = c1 c2`.

Recasting also removes a length inconsistency: `kite_fem` infers `EI = P/(3v)`,
which is the true `EI` only for a one-metre element because `v` is already
normalised by element length. A moment-curvature law is length-independent.

Verified by pure end-moment solves, where the exact answer is a
constant-curvature arc at whatever curvature the fit prescribes -- exact at any
deflection, unlike the tip-load form, whose own inversion assumes linear
cantilever theory. Worst error 0.06% in bending, 1.5% in torsion
(`run_validation_benchmarks.py --case inflatable`, which completes
`kite_fem/examples/FEM_beam_verification.py`).

**Collapse is reported, not enforced.** A dropping post-collapse moment would
make the energy fall with curvature, i.e. an unbounded mechanism that
minimisation would simply run away from. `inflatable_beam_state` returns
`utilisation` (curvature over collapse curvature) and a `collapsed` flag, so
leaving the calibrated range is visible rather than silent. Axial and shear
stiffness are not covered by the fits and must be supplied from tube geometry.

## Element reference

| kernel | nodes | rot. DOF | key parameters | notes |
|--------|-------|----------|----------------|-------|
| `CableKernel` | 2 | no | `rest_length`, `stiffness` | `tension_only` cuts compression (C1); `slack_smoothing` rounds it |
| `PulleyKernel` | 3 | no | `rest_length`, `stiffness` | `rest_length` is the **whole rope**, both arms |
| `TimoshenkoBeamKernel` | 2 | both | `rest_length`, `ea/ga_2/ga_3/gj/ei_2/ei_3`, `gamma_0`, `omega_0` | 1-point integration; reference strains make curved members stress-free |
| `MembraneKernel` | 3 | no | `area`, `thickness`, `youngs_modulus`, `poisson_ratio`, `d0inv_*` | Pipkin relaxed energy; `slack_stiffness_ratio` is required, see below |

**Fabric needs a residual slack stiffness.** A fully slack region stores exactly
zero energy, so its Hessian block is exactly zero and IPOPT fails with
`Error_In_Step_Computation`. `MembraneKernel.slack_stiffness_ratio` (default
`1e-4`) blends a small fraction of the unrelaxed law back in:
`U = (1-r) U_relaxed + r U_taut`. The taut region is untouched because the two
coincide there. Do not set it to zero on a canopy that can go slack.

**Wrinkling is why energy minimisation is the right formulation here**, not a
workaround. The relaxed functional is the quasiconvex envelope, so the minimiser
lands on the wrinkled state directly instead of chasing the near-singular tangent
a residual-form Newton solve has to fight through. The demo shows the contrast: on
a panel clamped into a frame 6% smaller than itself the relaxed law converges in
31 iterations to a smooth billow, the unrelaxed one takes 211 and buckles into
mesh-scale crumple.

## Relation to the two existing structural paths

Read this before changing element physics — the force laws must stay consistent.

|                    | `aerostructural/pss/structural_nlp.py` | `aerostructural/fem/` (`kite_fem` + `pyfe3d`) | `structural/` (this) |
|--------------------|------------------------|------------------------|----------------------|
| elements           | springs, tension-only, pulleys | springs, pulleys, inflatable Timoshenko beams | cables, pulleys, geometrically exact Timoshenko beams, wrinkling CST membranes |
| DOF                | 3 per node | 6 per node everywhere, masked via `bc` | 3 per node + 3 on beam nodes only |
| formulation        | min potential energy | Newton-Raphson on `fe - fi` | min potential energy |
| assembly           | Python loop building MX | scipy sparse COO per element | mapped SX kernels |
| regularisation     | weak position anchor | identity stiffness `I_stiffness=25`, `step_limit=0.2`, `relax_init=0.5`, optional pseudo-transient mass term | IPOPT inertia correction + membrane slack blend |
| default tolerance  | IPOPT `1e-8` | `1e-2` N residual | IPOPT `1e-8`, accepted on `1e-6` N force balance |

Consistency notes established by reading `kite_fem/SpringElement.py`,
`kite_fem/BeamElement.py` and `kite_fem/FEMStructure.py`:

- **Spring and pulley force laws agree across all three.** `f = k (l - l0)`,
  tension-only cut for `noncompressive` and `pulley`, and a pulley as two arms
  sharing one stretch. `tests/structural/test_cable.py` is the parity check.
- **The pulley rest-length convention does not agree.** The PSS reader splits `l0`
  across the two arms (`structural_nlp.py` sums `L0[e1] + L0[e2]`); `kite_fem`
  stores the total on each arm. This package takes the explicit total. Any future
  geometry adapter must state which it is reading.
- **`kite_fem`'s inflatable-beam law is a secant stiffness, not a potential.**
  `BeamElement.update_inflatable_beam_properties` recomputes `EI` and `GJ` from
  the *current* deflection, twist and inflation pressure using empirical ASKITE
  fits, plus a collapse flag. That is consistent as a force law inside a
  Newton residual, but dropping it into `1/2 EI kappa^2` would give the wrong
  gradient, because the `dEI/dkappa` terms vanish. **To reuse those fits here,
  integrate the moment-curvature relation into `W(kappa) = int M dkappa`** so the
  energy derivative reproduces the calibrated moment. This is the main open item
  before the beams can represent real LEI tubes; `BeamSection.from_tube` is
  linear thin-wall theory and is only a starting point.

## Open items

- The inflatable tube law is ported (see above). Remaining gap: the post-collapse
  branch is reported rather than modelled, and the fits cover only bending and
  torsion -- axial and shear stay linear.
- No coupling adapter yet. When one is written it belongs in `aerostructural/`,
  reading the kite YAML and mapping VSM loads onto nodes; this package must stay
  free of PSS, VSM and schema knowledge.
- Loads are dead (frozen) per solve. That matches the staggered coupling loop, but
  a true follower pressure would not be conservative and could not be posed as a
  potential — worth remembering before anyone tries to fold aero into the same
  minimisation.
- **Applied moments need load stepping, and are exact only about a fixed axis.**
  The rotation DOF are Rodrigues vectors, whose chart is singular at a rotation
  of pi, so a node cannot be asked to turn most of the way round inside a single
  solve; ramp the moment and let the frame updates absorb it (four steps is
  enough for a full roll-up). Separately, a constant "dead" moment in 3-D is
  genuinely non-conservative and so has no potential at all: energy minimisation
  can only represent the fixed-axis case. The work term uses the exponential-map
  rotation vector, not the Rodrigues vector, because a moment is conjugate to
  the former; using the latter made the answer depend on the load stepping.
  Nodal *forces*, which is what the aero coupling applies, carry none of these
  caveats and are exact in one solve.
- Contact and self-contact are not modelled.
- `PotentialEnergy(parallelization=...)` exposes CasADi threaded maps; untested at
  scale, serial is the default.

## Conventions to preserve

- **Isolation.** NumPy and CasADi only. No AWETrim imports.
- **Single source for every formula.** `rotations.py` holds the SO(3) maps and is
  written against an `xp` namespace so NumPy and CasADi share one implementation,
  the same pattern as `environment/profile_laws.py`. `beam_strains` and
  `green_strain` are likewise called by both the kernels and the NumPy
  setup/diagnostic paths — do not restate a strain measure.
- **No CasADi across the module boundary.** Symbolics are created inside
  `energy.py` and stay there; `StructuralModel`, `StructuralState` and
  `StructuralSolution` are plain NumPy dataclasses.
- **Kernels are pure.** No element loops, no global indexing, no closing over
  per-element data; everything element-specific arrives through `params`.
- **`converged` is the physical verdict** (force balance within
  `force_tolerance`), with IPOPT's own verdict preserved as `ipopt_success` and
  `status`. IPOPT routinely reports `Search_Direction_Becomes_Too_Small` on stiff
  structures whose answer is already exact.
- Tests assert against closed-form solutions and convergence orders, not against
  stored solver output.

## Physics references

- **Beam:** Simo & Vu-Quoc (1986) *Comput. Methods Appl. Mech. Engrg.* 58, 79–116.
- **Benchmarks:** Bathe & Bolourchi (1979) *Int. J. Numer. Meth. Engng.* 14,
  961-986; Bisshopp & Drucker (1945) *Q. Appl. Math.* 3, 272-275.
- **Rotation parameterisation:** Ibrahimbegović (1997) *Comput. Methods Appl.
  Mech. Engrg.* 149, 49–71.
- **Membrane wrinkling:** Pipkin (1986) *IMA J. Appl. Math.* 36, 85–99;
  Roddeman et al. (1987) *J. Appl. Mech.* 54, 884–892.
