# Rigid-body load transfer onto the struts (strut lever)

How AWETrim moves the aerodynamic panel loads of VSM onto the nodes of a
wireframe kite structure so that **both the resultant force and the resultant
moment are conserved exactly**. This is the method of Cayon (2022, MSc thesis,
Sec. 6.3), completed with the one term that thesis implementation dropped.

Code: `awetrim.aerostructural.mapping.StrutLeverAeroToStructuralLoadMapper`,
selected with `aerodynamic.load_transfer: strut_lever` (default `bilinear`).
Tests: `tests/aerostructural/test_mapping.py`.

## 1. Problem

The aerodynamic model (VSM) discretises the wing in $N$ spanwise panels. Panel
$i$ returns a force $\mathbf F_i$ and a moment. The structural model (the
wireframe) has far fewer nodes: one rib per inflatable strut, each rib with a
front node $\mathbf f$ and a rear node $\mathbf r$ (on the reduced LEI-V3
wireframe, the front and rear bridle attachments at $x/c \approx 0.01$ and
$0.875$). Each panel lies in one **bay**, between the struts $A$ and $B$ of two
consecutive ribs, and its load must be replaced by four nodal forces
$\mathbf q_{fA}, \mathbf q_{rA}, \mathbf q_{fB}, \mathbf q_{rB}$ on those
ribs.

The structure only feels the nodal forces, so the transfer must be **statically
equivalent**: about any point $\mathbf o$,

$$
\sum_n \mathbf q_n = \mathbf F_i, \qquad
\sum_n (\mathbf x_n - \mathbf o)\times\mathbf q_n
= (\mathbf x_i - \mathbf o)\times\mathbf F_i + \mathbf m_i ,
$$

where $\mathbf x_i$ is the point the panel force acts at and $\mathbf m_i$ the
panel's own moment about it. Six equations, twelve unknowns: the transfer is a
choice of one statically equivalent distribution. Force conservation alone is
easy; it is the moment that sets the pitch balance of every bay, and with it
the strut twist and the depower response of the kite.

### Panel load as AWETrim defines it

VSM gives each panel's force $\mathbf F_i$, its moment $\mathbf M_i$ about the
solver's reference point $\mathbf r_\mathrm{ref}$, and a per-panel centre of
pressure $\mathbf c_i$, the point on the chord where $\mathbf F_i$ alone would
reproduce the panel's pitching moment. VSM clamps the lever of that point to
$[-0.25c, 0.75c]$, so $\mathbf F_i$ at $\mathbf c_i$ is not always the whole
load. The transfer therefore takes

$$
\mathbf x_i = \mathbf c_i, \qquad
\mathbf m_i = \mathbf M_i - (\mathbf c_i - \mathbf r_\mathrm{ref})\times\mathbf F_i ,
$$

which is exact for any choice of $\mathbf x_i$: $\mathbf m_i$ is whatever
moment the force at $\mathbf x_i$ does not already carry (zero where VSM's
centre of pressure is unclamped).

## 2. The thesis method (Cayon 2022, Sec. 6.3)

The thesis treats the region between the aerodynamic and the structural mesh as
a rigid body and moves each panel load by lever rules
(`functions_PSM.force2nodes`, `moment2nodes`):

1. **Chordwise.** The line through $\mathbf x_i$ along the local chord
   $\hat{\mathbf t}$ meets the front line $\overline{\mathbf f_A\mathbf f_B}$
   at $\mathbf P_1$ and the rear line $\overline{\mathbf r_A\mathbf r_B}$ at
   $\mathbf P_2$. The force is split between them by the lever rule:
   $\mathbf F_{P_2} = \mathbf M_1\times\mathbf d/|\mathbf d|^2$ with
   $\mathbf M_1 = (\mathbf x_i - \mathbf P_1)\times\mathbf F_i$ and
   $\mathbf d = \mathbf P_2 - \mathbf P_1$; $\mathbf F_{P_1} = \mathbf F_i - \mathbf F_{P_2}$.
2. **Spanwise.** Each of $\mathbf F_{P_1}$, $\mathbf F_{P_2}$ is split between
   the two nodes of its line by the same lever rule.
3. **Section moment.** The pitching moment is converted into a couple of two
   opposite forces a short distance apart along the chord, and each is moved
   by steps 1-2.

Every step conserves force. Step 2 also conserves moment exactly, because
$\mathbf P_1$ lies on its line. Step 1 conserves moment **only if the chord line
through $\mathbf x_i$ really meets both lines**: then
$\mathbf M_1 \perp \mathbf d$ and
$\mathbf d\times\mathbf F_{P_2} = \mathbf M_1$. That holds for a flat plate
with the force point in its plane and the nodes at the leading and trailing
edge, which was the thesis geometry.

`line_intersect` returns the *closest point* of the front (rear) line to the
chord line, not an intersection. When the lines are skew, step 1 silently drops
the component of $\mathbf M_1$ along $\mathbf d$ and the moment of the offset
between the chord line and the closest point. On the reduced LEI-V3 wireframe
the chord lines miss the rib lines by up to 241 mm (mean 73 mm unbillowed,
92 mm billowed), because

* the rib nodes are bridle attachments, not the leading and trailing edge;
* each bay is warped by the anhedral: its front and rear lines are skew;
* the billowed panels lie off the plane of their bay.

The result is a 2-5 % loss of the wing's pitching moment (Sec. 5). The thesis
code checked only the force (the moment check is commented out in
`get_aero_forces2nodes`), so the loss was not visible.

## 3. Strut lever: the exact rigid-body transfer

The same rigid-body idea, with the moment of every move carried along instead
of dropped. A **strut** is the straight line from its front to its rear node,
$\mathbf d_A = \mathbf r_A - \mathbf f_A$, $\mathbf d_B = \mathbf r_B - \mathbf f_B$.

**Step 1 - lever rule along the span.** Take the closest points of the two strut
lines to $\mathbf x_i$,

$$
t_k = \frac{(\mathbf x_i - \mathbf f_k)\cdot\mathbf d_k}{\mathbf d_k\cdot\mathbf d_k},\qquad
\mathbf p_k = \mathbf f_k + t_k\,\mathbf d_k ,\qquad k \in \{A, B\},
$$

and split the force between them by the position of $\mathbf x_i$ along
$\mathbf s = \mathbf p_B - \mathbf p_A$:

$$
\sigma = \frac{(\mathbf x_i - \mathbf p_A)\cdot\mathbf s}{\mathbf s\cdot\mathbf s},\qquad
\mathbf F_A = (1-\sigma)\,\mathbf F_i ,\qquad \mathbf F_B = \sigma\,\mathbf F_i .
$$

Force is conserved: $\mathbf F_A + \mathbf F_B = \mathbf F_i$.

**Step 2 - the moment the move costs.** About $\mathbf x_i$, the moved forces
carry $(\mathbf p_A - \mathbf x_i)\times\mathbf F_A + (\mathbf p_B - \mathbf x_i)\times\mathbf F_B$.
The struts still owe

$$
\mathbf R = \mathbf m_i - (\mathbf p_A - \mathbf x_i)\times\mathbf F_A - (\mathbf p_B - \mathbf x_i)\times\mathbf F_B .
$$

When $\mathbf x_i$ lies on the segment $\mathbf p_A\mathbf p_B$ (a flat bay)
the bracket vanishes and $\mathbf R = \mathbf m_i$. Otherwise $\mathbf R$ is
exactly the term the thesis implementation dropped.

**Step 3 - the owed moment onto the struts.** A strut, two nodes on one line,
can carry a moment about any axis normal to itself, as a front/rear force
pair, but none about its own axis. The part of $\mathbf R$ about a strut axis
(a twist about the chord) goes across the span as a couple: $+\mathbf G$ on
strut $B$, $-\mathbf G$ on strut $A$. So

$$
\mathbf R = \mathbf M_A + \mathbf M_B + \mathbf s\times\mathbf G ,
\qquad \mathbf M_A\perp\mathbf d_A,\quad \mathbf M_B\perp\mathbf d_B,\quad \mathbf G\perp\mathbf s .
$$

Each unknown has two components (six in all) for three equations. AWETrim takes
the minimum-norm solution, with $\mathbf G$ scaled by the span $|\mathbf s|$ so
that every unknown is a moment of comparable size. The system is singular only
if a strut runs along the span, which an LEI wing does not do.

**Step 4 - each strut onto its two nodes.** The force on strut $k$,
$\mathbf F_k \mp \mathbf G$ at $\mathbf p_k$, splits by the lever rule along
the strut; the couple $\mathbf M_k$ becomes equal and opposite node forces:

$$
\mathbf q_{fk} = (1 - t_k)(\mathbf F_k \mp \mathbf G) - \frac{\mathbf M_k\times\mathbf d_k}{|\mathbf d_k|^2},\qquad
\mathbf q_{rk} = t_k(\mathbf F_k \mp \mathbf G) + \frac{\mathbf M_k\times\mathbf d_k}{|\mathbf d_k|^2},
$$

with the upper sign for $A$. The lever split has zero moment about
$\mathbf p_k$ because $\mathbf p_k$ is on the strut, and the couple's moment is
$\mathbf d_k\times(\mathbf M_k\times\mathbf d_k)/|\mathbf d_k|^2 = \mathbf M_k$
because $\mathbf M_k\perp\mathbf d_k$.

**Exactness.** Steps 1, 2 and 4 are identities in force and moment, and step 3
solves for the remainder exactly, so the four nodal forces reproduce the panel
force and its moment about any point, on warped bays and for points off the
bay.

**Reduction to the thesis.** On a flat rectangle with $\mathbf x_i$ inside it,
$\mathbf p_A$, $\mathbf p_B$ and $\mathbf x_i$ are collinear, $\mathbf R = \mathbf m_i$
and the split is the lever rule along the chord and along the span: the bilinear
split, and the thesis'.

**Which bay.** Each panel is assigned to its bay by construction (the
aerodynamic mesh is built section by section between the struts,
`billowing.panel_bays`), not by sorting on the span coordinate $y$. At the
curled LEI-V3 tip $y$ is not monotonic, and a $y$ search gave the tip panels a
degenerate bay (one front node twice), which the bilinear mapper accepted
silently and on which the thesis code divides by zero.

## 4. Relation to other exact choices

Any statically equivalent distribution is exact; they differ only in how the
load is shared among the four nodes (the six-dimensional null space). The plain
minimum-norm solution of the six equations is also exact and differs from the
strut lever by less than 10 N per node on the LEI-V3 wing at $\alpha = 8°$
(of up to 400 N). The strut lever is kept because it is the thesis' lever-rule
physics wherever the geometry allows it, and solves only for the remainder.

## 5. Verification

**Synthetic** (`tests/aerostructural/test_mapping.py`): 2000 random warped bays
with points off the bay and random forces and moments conserve force and moment
to $10^{-13}$; a flat rectangle reproduces the bilinear split; a pure twist about
the chord goes across the span as a couple; zero load gives zero; mirrored bay
and load give mirrored forces.

**LEI-V3 full-FEM wing**, rigid VSM (develop 1201f6f, 3/4-chord force
directions), 135 panels, loads onto the reduced wireframe's 10 ribs, pitching
moment about the centre section's quarter chord:

| Mesh | $\alpha$ | true $M_y$ (N m) | bilinear | thesis | strut lever |
|---|---|---|---|---|---|
| full chord | 4° / 8° / 12° | -826 / -1076 / -1274 | -1.0 / -4.5 / -6.0 % | -2.9 / -4.2 / -4.4 % | 0 |
| billowed, $p = 5\,\%$ | 4° / 8° / 12° | -533 / -873 / -1032 | +3.5 / -0.4 / -3.6 % | -2.0 / -3.4 / -4.9 % | 0 |

All three conserve force. With the billowed wing the bilinear error changes sign
with $\alpha$, so it is not an offset a calibration could absorb.

Nodal forces at $\alpha = 8°$, billowed (|q| in N, thesis / strut lever):

| rib | front | rear |
|---|---|---|
| centre | 394 / 400 | 223 / 224 |
| 2 | 253 / 264 | 155 / 158 |
| tip | 46 / 25 | 5 / 9 |

The largest changes are at the tip ribs, where the chord lines miss the rib
lines the most.

**Coupled.** One state of the identification setup (wind 5 m/s, built tape, gravity
off) converges in 8 coupling iterations with the strut lever; the in-loop
moment check reads $|\Delta\mathbf M| \approx 10^{-12}$ N m.

**Cost.** Batched over all panels, 0.34 ms per coupling iteration at 135 panels,
against 0.5-1 s for one VSM solve.

## References

- O. Cayon, *Fast aeroelastic model of a leading-edge inflatable kite for the
  design phase of airborne wind energy systems*, MSc thesis, TU Delft / DTU,
  2022, Sec. 6.3 (force coupling).
- Verification scripts (2026-10-28): `load_transfer_check.py`,
  `thesis_transfer_check.py`, `test_strut_lever.py` (session scratchpad;
  the tests above are their permanent form).
