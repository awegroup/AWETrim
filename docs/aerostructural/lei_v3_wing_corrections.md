# LEI-V3 wing corrections: canopy billowing and the short structural chord

Two corrections come up whenever an aerodynamic mesh is built on the LEI-V3
wireframe. V3Kite.jl ships both of them
(`billowing_percentage: 5`, `aero_scale_chord: 0.129`), and they are easy to
dismiss as tuning factors. They are not. Each one has a measured or
geometric origin, and only one of them is specific to this kite.

| Correction | Value | Origin | Kite-specific? |
|---|---|---|---|
| TE canopy billowing | 5 % | Photogrammetry of the flying V3 (Haanen 2026, Table 9.2) | No. The *mechanism* is generic, only the number is the V3's |
| Wing load / area factor | x1.129 | The wireframe TE nodes are not at the end of the fabric: the PSM wing is 12.9 % short in projected area | **Yes.** It is a property of one structural mesh |

## 1. Canopy billowing (5 %)

**What is measured.** Haanen (MSc thesis, TU Delft, 2026, Sec. 9.2.1) reconstructs the
flying V3 from photogrammetry and measures the distance between the
trailing-edge markers of consecutive struts. Averaged over the span, relative
to the CAD:

| Flight case | Strut-to-strut TE distance wrt CAD |
|---|---|
| Powered, straight | -5.23 % |
| Powered, turn | -5.00 % |
| Depowered, straight | -3.86 % |

The leading edge spacing barely changes (the inflated LE tube holds it).
The canopy between struts bulges into a spanwise arc, so the straight TE
chord between two struts shortens while the fabric itself does not stretch.
Billowing grows with load, but not linearly.

**What the 5 % means in VSM.jl.** `billowing_percentage = p` puts the TE arc
at `1/(1 - p/100)` times the straight strut-to-strut edge, i.e. the straight
edge is `(1 - p/100)` of the fabric length. With p = 5 that is Haanen's
powered -5 %. VSM.jl then gives the arc the shape `-A sin(pi t)` (mid-bay
chords rotated nose-down about the rib-to-rib LE axis).

**Shape: the circular arc.** Cayon (MSc thesis, TU Delft/DTU, 2022, Sec. 5.3)
derives the spanwise shape. A tensioned, inextensible membrane under a
locally constant pressure jump `Delta p` has constant tension `f = Delta p R`.
The shape is therefore a circular arc, and its radius does not depend on the
magnitude of `Delta p`, because scaling `Delta p` by K scales the tension by
K too. The billowing then changes only through the TE length between struts:
a shorter straight TE with the same fabric length gives a deeper arc. That
thesis took the arc angles from the CAD (Table A.3: 1, 10, 15, 18, 20 deg from
the tip to the centre plate). The V3's LE frame is designed with the
orientation of the expected powered billow (B. van Ostheim, Kitepower).

**Billow full fidelity resolves it.** `struc_geometry_FEM_full.yaml` carries
TE canopy nodes between the struts, so the membrane solve shows the
billowing directly. Measured on the stored results (strut TE nodes
2, 8, 14, ..., 56; inner seven bays averaged; date 2026-10-28):

| Case | Wing load (sum f_ext z) | Mean straight TE wrt built |
|---|---|---|
| vw 4.2, v_a 13.5 | 1.5 kN | -6.4 % |
| baseline, v_a 14.9 | 1.9 kN | -5.4 % |
| vw 6, v_a 23.4 | 5.1 kN | -8.3 % |

The TE fabric polyline stays within 0.5 % of its built length, so the
inextensible-canopy assumption holds. The level (-5 to -8 %) and the trend
with load agree with the photogrammetry. One difference: Billow contracts the
outermost bays most (-9 to -15 %), whereas Haanen sees the smallest change
there. The cause is not identified yet; it may be the tip loading in the
simulation or the reconstruction uncertainty at the tips.

**Caveat: the Billow full coupling is not final.** Its leading-edge
aerodynamic coupling still lacks proper handling of the LE shape and of the
moment transfer, so the loads behind these numbers are not yet right. Read
the table as "Billow full reproduces the mechanism and the order of
magnitude", not as a validated billowing law.

**Consequence.** Billowing is physics, not tuning. On a mesh that has no
canopy between the struts (the wireframe), it has to be added to the
aerodynamic mesh. Billow full computes it, within the caveat above.

## 2. The short structural chord (x1.129, LEI-V3 only)

The PSM wireframe (`struc_geometry.yaml`,
`struc_geometry_PSM_reduced_photogrammetry_adjusted.yaml`, removed on
2026-10-29 and kept in v0.1.0) puts the wing
nodes at the bridle attachment points. The TE node is the strut's rear
bridle point, not the end of the fabric (x/c about 0.01 to 0.90). An
aerodynamic mesh built on those nodes (`LinearStructuralToAeroMapper`) flies
a wing that is too short in chord:

| Geometry | Projected area (x-y) | Centre chord |
|---|---|---|
| PSM wireframe nodes | 17.21 m2 | 2.33 m |
| `aero_geometry.yaml` (CAD, full chord) | 19.41 m2 | 2.60 m |
| `struc_geometry_FEM_full.yaml` | 19.56 m2 | 2.63 m |

19.41 / 17.21 = 1.128: this is V3Kite's 1.129 ("12.9 % was the delta in
projected area", V3Kite.jl `data/struc_geometry.yaml`). `system.yaml`'s
`projected_surface_area: 17.2119` is the area of the PSM nodes, not of the
real wing.

**Where AWETrim stands.**
- Billow full and `wireframe.driver.solve_deformation_reduced_fem` fly the
  full-chord edges (`SkinStructuralToAeroMapper` carries the FEM edges on the
  reduced ribs). They need **no** factor.
- `LinearStructuralToAeroMapper` on the PSM wireframe flies the short wing.
  The shipped `rom_config_aerostructural.yaml` was identified on this one
  (`identification.yaml`, fidelity `wireframe`). Its coefficients are
  normalised by 19.75 m2 but come from a 17.2 m2 wing.
- The geometric fix (put the TE at the end of the fabric) is better than the
  force factor. A uniform x1.129 on force and moment keeps CL/CD at equal lift
  and leaves the pitching-moment arm short.

A factor like this belongs to a kite's data folder, never to code or a
generic default: another mesh has another deficit, or none.

## 3. What AWETrim does (since 2026-10-28)

The official identification (`identification.yaml`, fidelity `wireframe`)
flies the wireframe **reduced from the full FEM file**
(`wireframe.driver.build_case_reduced_fem`): one structural file for both
fidelities, the full-chord aerodynamic mesh (no area factor), and:

- **Billowing** (`aerodynamic.billowing.contraction: 0.05`,
  `aerostructural/billowing.py`). Every strut rotates about its leading-edge
  node in the wing surface until each inner bay's straight trailing edge is
  (1 - p) of the CAD one; the canopy between two struts is the circular arc of
  the bay's built fabric length (sin(theta)/theta = c/L_f), each section at its
  own fabric fraction. The two tip bays cannot contract by rotating the 0.35 m
  tip chord; they ride rigidly with the outer strut (Haanen also sees the least
  change there). The reduced ribs move with their struts and the wing members
  rest at the new distances, so the structure holds the contraction the aero
  mesh is drawn with; the bridle keeps its lengths. The arc is built once into
  the reference mesh and carried by the skin mapper (static billowing). At
  p = 5 % the inner bays read 1 - c/L_f = 6.1-8.3 %, theta 35-41 deg, depth
  198-235 mm, and the outer strut rotates 9 deg.
- **Exact load transfer** (`aerodynamic.load_transfer: strut_lever`), see
  `strut_lever_load_transfer.md`.

**Mesh convergence** (rigid VSM, wing only, v_a 20 m/s, VSM develop 1201f6f;
CL and CD on 19.75 m2; change against 12 sections per FEM section = 324 panels):

| sections per FEM section (panels) | CL, p=5 %, 4 deg | CL, p=5 %, 8 deg | CD, p=5 %, 8 deg |
|---|---|---|---|
| 1 (27) | +9.5 % | +5.0 % | +2.7 % |
| 2 (54) | +2.6 % | +1.5 % | +0.5 % |
| 3 (81) | +1.2 % | +0.6 % | +0.7 % |
| **5 (135)** | **+0.4 %** | **+0.2 %** | **+0.1 %** |
| 8 (216) | +0.05 % | +0.07 % | 0.0 % |

The billowed wing needs about twice the resolution of the flat one (the arc
must be resolved). 16 sections (432 panels) agree with 12 to 0.03 %; at 20
(540 panels) VSM's circulation loop no longer converges within its iteration
limit. At 12 deg the values move +-1 % without trend from 3 sections on (panels
near stall changing branch with the mesh), as in the trim panel-count study.
The quarter-chord force directions (`is_aoa_corrected`, off by default) give
the same mesh convergence but change the values: CD -11 to -16 %, CMy -0.04
to -0.08, CL < 0.3 %. Default: **5 sections, 135 panels**.

At fixed angle of attack, 5 % billowing lowers CL by about 20 % (0.82 to 0.66
at 8 deg): the arc pitches the mid-bay sections nose down by about 3 deg on
average. A trim recovers most of it in angle of attack; at equal lift L/D
drops roughly 5 % (interpolated, not yet from a trim).

## History of the confusion

On 2026-10-27/28 both factors were ported as "V3Kite emulation" options on
the local branch `exp/v3kite-emulation` (commit a165225:
`aerodynamic.billowing_percentage`, `aerodynamic.wing_load_scale`) and kept out
of develop as empirical tuning. That reading was wrong on both counts. The 5 %
is measured (Haanen) and reproduced by Billow full. The 12.9 % is the
TE-node area deficit of the PSM mesh. In the emulation gate, those options
closed the tape mismatch with V3Kite but not the ~28 % wing-drag gap, so that
gap has another cause.

## References

- P. J. Haanen, *In-flight experimental measurements of large-scale deformations on a leading-edge inflatable kite using stereoscopic photogrammetry and UWB modules*, MSc thesis, TU Delft, January 2026: LEI-V3 in
  flight, Sec. 9.1-9.2 (span -10.7 %, projected area -12.2 %, TE billowing).
- O. Cayon, *Fast aeroelastic model of a leading-edge inflatable kite for the
  design phase of airborne wind energy systems*, MSc thesis, TU Delft/DTU,
  2022, Sec. 5.3 (canopy billowing) and App. A Table A.3.
- VortexStepMethod.jl 6.1.0, `apply_billowing_to_pair!`.
