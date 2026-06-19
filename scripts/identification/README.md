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
| [`calibrate_cd0_depower_qs.py`](calibrate_cd0_depower_qs.py) | Calibrate the ROM aero parameters `CD0`, `angle_pitch_depower_0` and `delta_pitch_depower` against the quasi-steady validation: fit them so the **predicted** tether force and tangential speed match the **measured** ones, split per powered/depowered phase (this jointly breaks the CD0 ↔ depower-pitch degeneracy). |
| [`identify_aero_parameters_turn_law.py`](identify_aero_parameters_turn_law.py) | Identify the turn-rate law from flight data in three formulations (simple, two-term, full rational) by least-squares / nonlinear fit, per flight phase. Produces the fitted gains and per-phase fit plots. |
| [`plot_body_axes.py`](plot_body_axes.py) | 3-D visualisation of the rigid-body principal axes for a deformed aerostructural result: deformed nodes (sized by nodal mass), CG, principal body axes and the global frame. Locates the struc geometry from the result path (override with `--struc`); `--save` to write a PNG. |

## Notes

- Inputs come from EKF-reconstructed flight data (HDF5) produced by the
  [`../experimental/`](../experimental/) pipeline.
- The identification module is still maturing (`src/awetrim/identification/`); the
  aero-LUT guide there is outdated — see the project `AGENTS.md`.
