# Identification scripts

Identify and calibrate the parameters used by the ROM — the quasi-steady aero
parameters and the turn-rate (steering) law — from reconstructed flight data,
plus a helper to inspect the rigid-body axes of a deformed shape. Run from the
project root.

The turn-rate law and the calibration setup follow Cayon & Schmehl, *Quasi-Steady
Mechanics of Tethered Flight*, and mirror the ROM validators in
[`../reduced-order-model/validation/`](../reduced-order-model/validation/).

## Scripts

| Script | What it does |
|--------|--------------|
| [`identify_rom_aerostructural.py`](identify_rom_aerostructural.py) | Identify the LEI-V3 **aerostructural ROM** (`data/LEI-V3-KITE/rom_config_aerostructural.yaml`) from coupled simulations only: theta_b(u_p) from the centre-window trims, C_L/C_D polynomials in (alpha, u_p, u_s, v_a) from the frozen-shape alpha polars (`scripts/personal/wes-quasi-steady/run_center_sweep_alpha_polar.py`), selected on anchor-grouped cross-validation, and the steering roll gain. Writes the config plus the tidy datasets under `results/LEI-V3-KITE/identification/rom_aerostructural/`. |
| [`validate_rom_aerostructural.py`](validate_rom_aerostructural.py) | Re-solve every coupled trim of that dataset with the ROM's own quasi-steady solver (gravity off, tetherless, same wind/u_p/u_s) for both LEI-V3 ROMs and compare v_tau, tether force, alpha_w and the turn rate. |
| [`plot_rom_comparison.py`](plot_rom_comparison.py) | Compare the two LEI-V3 ROMs with the aerostructural data: C_L(alpha), C_D(alpha) and C_L-C_D at powered/depowered (KCU drag included in every curve), and theta_b against l_dp in both F_b conventions. |
| [`plot_rom_flight_validation.py`](plot_rom_flight_validation.py) | Flight validation of the LEI-V3 ROMs: per-phase error table over the cycles in the per-ROM CSVs written by `validate_quasi_steady_state_v3.py --rom`, and a one-cycle time series of tension, v_tau and steering against the EKF reconstruction. |
| [`identify_rom_flight_correction.py`](identify_rom_flight_correction.py) | Steps 1-2 of the flight correction of the aerostructural ROM (equation error on the 2019 EKF coefficients): theta_b(u_p) from the flight lift, a single-input drag correction at equal lift, a roll-gain starting value; writes `rom_config_aerostructural_flight_corrected.yaml` and the correction report/figure. Cycles 60-67 held out. |
| [`refine_rom_flight_output_error.py`](refine_rom_flight_output_error.py) | Step 3 (the identification proper): output error on the quasi-steady tension and v_tau of training cycles (validator `--out` CSVs), nested Newton roots with exact parameter sensitivities, the roll gain set by the turn-rate law (solved steering 1:1 on the logged steering), steering drag bounded >= 0. `--rom aerostructural_flight` (default) or `--rom semi_empirical` (recalibrates the paper's ROM in place, comments kept). |
| [`plot_rom_turn_rate_law.py`](plot_rom_turn_rate_law.py) | The check of the roll gain: each ROM's turn rate with the logged steering in against the measured one, and the turn-rate law chi_dot = K v_a u_s + c_g g-term for the flight and the ROMs. |
| [`identify_aero_parameters_turn_law.py`](identify_aero_parameters_turn_law.py) | Identify the turn-rate law from flight data in three formulations (simple, two-term, full rational) by least-squares / nonlinear fit, per flight phase. Produces the fitted gains and per-phase fit plots. |
| [`plot_body_axes.py`](plot_body_axes.py) | 3-D visualisation of the centre-panel body axes for a deformed aerostructural result: deformed nodes (sized by nodal mass), CG, the body triad (anchored to the wing's centre panel, which is drawn) and the global frame. Locates the struc geometry from the result path (override with `--struc`); `--save` to write a PNG. |

## Aerostructural ROM: how the model is built

`identify_rom_aerostructural.py` identifies every parameter of the ROM of
Cayon, van Deursen & Schmehl (2026, *WES* 11, 1097) from coupled VSM +
Billow-wireframe solutions at the centre of the wind window (gravity off). The
constants named below are at the top of the script.

### Data

- **Anchors**: the converged `_billow9` continuation chains (2019 hardware):
  depower chains (u_p sweep, u_s = 0) at v_a 13–25 m s⁻¹ and steering chains
  (reel-out and reel-in) — 341 coupled trims.
- **Frozen-shape alpha polars** around each anchor
  (`scripts/personal/wes-quasi-steady/run_center_sweep_alpha_polar.py`):
  −6…+8° about the trim (`alpha_polar.json`) and +9…+22° into stall
  (`alpha_polar_stall.json`). A trimmed chain ties alpha to u_p almost one to
  one (alpha moves ~1° across the whole v_a range), so the trims alone cannot
  separate the polar slope from the depower effect; the polars can.
- **Lens check**: a polar is used only if its frozen-shape re-trim reproduces
  the coupled trim (C_L within `LENS_CL_TOLERANCE` = 1 % at the coupled alpha).
  Steered snapshots with steering tape ≥ 0.125 m re-trim to a different state,
  so their polars are dropped and their coupled trims are used as samples
  instead.
- **Coefficient scope**: wing + bridle forces on `S_REF` = 19.75 m². The KCU
  drag is *not* in C_D (`kcu_drag_in_coefficients: false`): the ROM adds it from
  the system file's KCU hardware. Fitted band `UP_BAND` = 1.6–2.2 m.

### Candidate terms

Monomials in alpha, u_p, u_s, v_a and the stall switch, each with a physical
reason:

| Group | Terms | Why |
|---|---|---|
| attached polar | α, α² | lift slope and curvature |
| depower | u_p, u_p², α·u_p, α²·u_p | shape change with the power tape |
| steering | u_s², α·u_s², u_p·u_s² | lift and drag are symmetric in left/right steering, so only even powers |
| load (aeroelastic) | v_a, α·v_a, u_p·v_a | the deformed shape depends on dynamic pressure |
| stall | σ, σ·α, σ·α², σ·u_p, σ·u_s² | act only past the stall |

- **Stall switch** σ = ½[1 + tanh((α − α_stall)/w)] (`awetrim.system.kite.stall_blend`,
  the single source used by the ROM and by the identification), so
  C = C_attached + σ·(C_separated − C_attached) stays linear in its
  coefficients. α_stall and w are chosen first, on a grid
  (`STALL_ANGLE_GRID_DEG`, `STALL_WIDTH_GRID_DEG`), by the cross-validation
  error of the full C_L library.
- **No α³**: the ROM's solvers do not bound alpha, so the model must behave
  outside the data. A cubic C_D went negative beyond the data, which produced
  a spurious 178 m s⁻¹ reel-in root in the flight validation. A cubic C_L
  cannot reproduce the sharp stall either.
- All terms, C_D included, are plain smooth monomials (the ROM's law since
  2026-10; a ROM term may ask for `abs: true`, which puts a kink at zero, so
  the identification never offers it).

### Selection

`aero_polynomial.select_model(criterion="cv", backward=True)`:

1. **Score = cross-validation RMSE with folds by anchor** (`CV_FOLDS` = 5). All
   samples of one frozen shape fall in the same fold. A random split would
   score interpolation inside a polar, not prediction at an unseen state.
   BIC is not used: it treats ~10⁴ strongly correlated polar samples as
   independent evidence and keeps accepting terms worth a fraction of a
   percent.
2. **Forward stepwise**: from the intercept, add the term that lowers the CV
   RMSE most. Stop when the best term improves it by less than
   `MIN_RELATIVE_IMPROVEMENT` = 1 %.
3. **Backward elimination**: drop any term whose removal costs less than the
   same 1 %. A term picked early can become redundant once later terms enter,
   and forward selection never revisits.
4. **Extrapolation check** (`check_extrapolation`): C_D must stay positive
   over ±30° alpha across the fitted u_p, u_s and v_a ranges, otherwise the
   script fails. It also prints where C_L peaks.

The selection path (each term added or removed, and the CV RMSE after the
step) is written to
`results/LEI-V3-KITE/identification/rom_aerostructural/identification_report.json`.

### The other parameters

- **θ_b(u_p)** (`angle_pitch_tether_0`, `slope_angle_pitch_tether_depower`):
  linear fit over the unsteered trims in the band. α_b is the paper Eq. D11
  angle of the bridle resultant F_b the ROM builds (tether plus the KCU's
  weight, inertia and drag); α_w is the centre-panel angle.
- **Roll gain** (`gain_roll_steering`): φ_a,w = k·u_s through the origin, on
  the steering trims. φ_a,w is the roll of the lift about v_a relative to F_b.

### Checks

- `validate_rom_aerostructural.py` re-solves every coupled trim with the ROM's
  own quasi-steady solver.
- `plot_rom_comparison.py` compares the polars and θ_b with the
  semi-empirical ROM (removed 2026-10-07, archived in git history at bad8a9b).
- Flight validation: `../reduced-order-model/validation/validate_quasi_steady_state_v3.py --rom aerostructural --cycles 60-67 --no-show`,
  then `plot_rom_flight_validation.py`.

## Notes

- The flight-data scripts take EKF-reconstructed flight data (HDF5) produced by the
  [`../experimental/`](../experimental/) pipeline.
- The identification module is still maturing (`src/awetrim/identification/`); the
  aero-LUT guide there is outdated — see the project `AGENTS.md`.
