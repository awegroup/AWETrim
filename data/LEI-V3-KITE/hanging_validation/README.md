# Hanging V3 kite — measured shape validation

Measured deformed-shape data for the TU Delft V3 kite suspended from its bridle
point, together with the reference model that was solved against it.

## Provenance

Vendored from [`awegroup/kite_fem`](https://github.com/awegroup/kite_fem)
(MIT, © 2025 Patrick Roeleveld), `main` at commit `630c8c0` — recorded in
`.source_commit`. Files are byte-for-byte copies except that two were renamed
for clarity:

| here | upstream |
|---|---|
| `hanging_test_initial.npz` | `data/TUDELFT_V3_KITE/hanging_test_initial.npz` |
| `mass_hanging_test.csv` | `data/TUDELFT_V3_KITE/mass_hanging_test.csv` |
| `struc_geometry_hanging_test.yaml` | `data/TUDELFT_V3_KITE/struc_geometry_hanging_test.yaml` |
| `measured_lengths.csv` | `examples/validation/validation_data.csv` |
| `kite_fem_results.csv` | `examples/validation/model_results.csv` |
| `kite_fem_states/load_case_*.npz` | `examples/validation/results/load_case_*.npz` |

**On the measurement method:** upstream describes `validation_data.csv` only as
"experimental validation reference data". Roeleveld's thesis (Section 6.3)
identifies it as **stereoscopic camera measurement** of markers on the
inflatable structure — not photogrammetry, despite the marker-to-marker form of
the data. The rows are perfectly symmetric left-to-right to four decimals
(`La` ≡ `Li`, `LcsTL` ≡ `LcsTR`), so they have been mirrored or
port/starboard-averaged.

`kite_fem_states/` holds its ten solved configurations. They are vendored so
the comparison figures can show its deformed geometry, and so
`tests/structural/test_hanging_kite.py` can assert that our length extractor
reproduces `kite_fem_results.csv` from them exactly rather than taking that on
trust.

## The experiment

The kite hangs upside down in a hangar from a **bar**, devoid of aerodynamic
loading, and the deformed shape is measured. Under self-weight alone the span
falls from **8.305 m as built to 4.867 m** — a 41% closure, with every
trailing-edge segment shortening 26–43%. That closure is the deformation this
dataset is really testing.

**The suspension is a bar, not a point.** Thesis Appendix B, Table B.1 marks
`A_main`, `A_I`, `M`, `PL` and `SL` as N/A for the hanging test and adds a
`bar` of 1650 mm; Section 6.3 explains the bridle was adapted to fit the hangar
height. The same table records the one documented shortening, `B_r,5` from
12168 to 9425 mm — already applied in the stored model as `brmain 3` = 4.7125 m
(half of 9425).

**The stored model has no load path to its anchor.** Node 0 is flagged fixed
but appears in no spring, pulley or beam, and seven further bridle nodes are
orphaned with it (all of exactly zero mass). Anything solving this file must
supply a suspension; `scripts/structural/hanging_kite.py` holds the two main
bridle convergence points (nodes 82 and 116) at the bar half-span.

Ten load cases: two inflation pressures × five loadings.

| case | pressure (bar) | tip load (kg) | point load (kg) |
|---|---|---|---|
| 1 | 0.15 | — | — |
| 2 | 0.15 | — | 9.7 |
| 3 | 0.15 | — | 25.2 |
| 4 | 0.15 | 2 | — |
| 5 | 0.15 | 5 | — |
| 6–10 | 0.25 | *(same five)* | |

The tip load is applied `+y` at node 2 and `−y` at node 58 — a symmetric
tip-*spreading* load that pulls the kite back open (measured span returns to
7.8–8.3 m). The point load is `+z` at node 29, the mid-span leading edge.

## Measured quantities

Twelve lengths per case, columns in the order the upstream extractor emits them:

- `La`…`Li` — nine trailing-edge segment lengths between consecutive struts
- `b` — tip-to-tip span, TE node 58 to TE node 2
- `LcsTL`, `LcsTR` — tip-to-centre-LE diagonals

`kite_fem_results.csv` carries two extra trailing columns, `Tolerance` and
`max strain`, which are solver diagnostics rather than lengths. Cases 3 and 8
did not reach the 0.01 convergence tolerance (0.229 and 0.076); case 5 is
marginal.

## Model contents

`hanging_test_initial.npz` is the fully-assembled reference model — 276 nodes,
1656 DOF, 767 noncompressive springs, 14 pulleys, 98 Timoshenko beams tuned to
inflatable-tube properties. The initial configuration is stress-free: canopy
strain is exactly zero and no bridle carries tension.

The canopy is a structured **29 × 8 grid** (29 chordwise sections, 8 nodes
each), stored only as a flat spring list. `scripts/structural/hanging_kite.py`
recovers the grid geometrically and verifies the recovery: all 721 canopy
springs must come out as grid edges, and no spring joining two grid nodes may
be anything else.

## Use

```
python scripts/structural/run_hanging_validation.py
```

builds the same model in `awetrim.structural` (Billow), replacing only the
canopy — 721 springs become 392 wrinkling membrane triangles on the identical
grid — and scores both against `measured_lengths.csv`.
