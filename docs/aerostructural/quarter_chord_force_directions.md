# VSM force directions from the quarter chord

Since 2026-11-03 AWETrim runs the Vortex Step Method with its force
directions taken from the flow at the quarter chord
(`is_aoa_corrected: true`; `aerodynamics/vsm_settings.py`,
`data/LEI-V3-KITE/as_config.yaml`). Before, it took them from the flow at the
3/4-chord control point (`is_aoa_corrected: false`, the LL-3/4 model of
Gaunaa, Li & Pirrung). This note says what the switch changes, why the
quarter-chord model is the physical one, and what it does to the LEI-V3.
(Earlier AWETrim comments called the `false` setting "freestream
directions"; they are the 3/4-chord ones.)

## What the option does

The VSM solves the circulation at the 3/4-chord control point: the polar
gives the magnitudes C_l, C_d at the angle of the flow there (thin-airfoil
theory, TAT2), and the circulation loop is the same with either setting. The
setting changes only the direction the lift and drag act in:

- `false`: perpendicular and parallel to the flow at the 3/4-chord control
  point, the angle the polar was read at;
- `true`: perpendicular and parallel to the flow at the quarter chord, where
  the bound vortex sits, with the velocity induced there by the other panels'
  bound and trailed vortices (TAT3; the LL-Gaunaa model, Gaunaa, Li &
  Pirrung, TORQUE 2026).

The tilt of the lift by the induced angle IS the induced drag. The 3/4-chord
point sees a larger induced angle than the bound vortex does, so tilting the
force by it overestimates the induced drag; the quarter-chord flow is the
flow the bound vortex actually feels (Kutta-Joukowski).

## Verification in VSM

The check that does not depend on where forces are taken on the blade is the
Trefftz-plane induced drag of the trailed wake (VSM `a404e10`,
`BodyAerodynamics.compute_trefftz_plane_induced_drag`). On inviscid wings:

| | quarter-chord directions | 3/4-chord directions |
|---|---|---|
| straight wing | equal to Trefftz (four digits) | |
| elliptic wing | within 2 % of L^2 / (q pi b^2) | 20-40 % over |

VSM commit `62c53af` (2026-09-17) made the quarter-chord directions VSM's
default on that basis (`docs/figures/03_trefftz_check.png` in VSM). On a
curved or swept bound vortex the quarter-chord velocity holds a logarithmic
self-induction singularity; VSM PR #165 (`feature/curved-bound-vortex-regularisation`,
merged as `1201f6f`, the commit AWETrim pins) regularises it by spreading the
bound vorticity along the chord (`f43bcdb`) and verifies the result
(`a4ca2ca`, `docs/validation/curved_bound_vortex_report.pdf`). On-blade over
Trefftz induced drag, inviscid:

| wing | ratio |
|---|---|
| straight | 1.000 |
| circular arc, R = 4c (40-160 panels) | 0.999 |
| 30 deg swept | 0.99 |
| swept-anhedral elliptic wing (Gaunaa et al.) | 1.01 |
| LEI-V3 | 1.007-1.010 |

The 3/4-chord directions were AWETrim's model only because they had been
VSM's default; their higher drag happened to sit closer to the V3 wind-tunnel
drag, for reasons not understood.

## Effect on the LEI-V3

Built-shape polar of the 135-panel billowed wing
([lei_v3_wing_corrections.md](lei_v3_wing_corrections.md)): CD -11 to -16 %,
CMy -0.04 to -0.08, CL within 0.3 %. The mesh convergence is the same.

Coupled wireframe, the identification sweep (`run_identification_sweep.py`,
depower chains at 4-6 m/s and steering chains at 6 m/s, on a coarse 0.10 m /
0.05 m grid), `true` against `false`:

- C_L(alpha) unchanged (lift slope within 1 %, same C_L,max); C_D 13-15 %
  lower at alpha 10 deg;
- the trims settle 0.1-0.5 deg lower in alpha; on average C_L -4 %, C_D -7 %,
  L/D +3.6 %, tether force -3 %;
- turn rate per unit v_a u_s at equal power tape (6 m/s): 0.72 x at 1.40 m,
  0.78 x at 1.70 m, 0.83 x at 2.10 m. With less drag the steered wing turns
  less; the reduction is largest powered.

## Limits

- The CasADi trim runs both laws through VSM's own force-law kernels
  (`VSM.core.panel_kernels` and
  `BodyAerodynamics.compute_quarter_chord_AIC_matrices`, VSM PR #166); with
  `true` it reproduces the NumPy trims, and `false` trims are unchanged.
- Stored results are not recomputed: the default ROM
  (`rom_config_aerostructural.yaml`) and its flight match were identified with
  `false`, and the snapshots under `data/LEI-V3-KITE/deformed_results/`
  record `false` in their configs. The ROM is re-identified separately.
- `is_aoa_corrected: false` in a case's `aerodynamic` block reproduces the
  earlier model.
