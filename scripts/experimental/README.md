# Experimental (EKF) scripts

Reconstruct real flight behaviour from logged flight-test data with the
EKF-AWE Extended Kalman Filter, then plot and post-process the results. This is
the bridge between measured flights and model validation/tuning. Run from the
project root.

Requires the `awes-ekf` package (installed from GitHub via `pyproject.toml`) and
an EKF configuration under `data/<kite>/ekf_config/`. Minimum filter inputs are
kite position, velocity, tether force and tether length.

## Scripts

| Script | What it does |
|--------|--------------|
| [`run_analysis_ekf.py`](run_analysis_ekf.py) | Run the EKF over a chosen flight log: estimate kite states, in-flight aerodynamic coefficients and the wind vector. Lists available flights and writes the reconstruction to HDF5. |
| [`plot_analysis_ekf.py`](plot_analysis_ekf.py) | Interactive plotting of a saved EKF result by category — aerodynamics, kinematics, tether, wind velocity, system performance, EKF performance, and turn-rate-law identification. |
| [`inverse_wind_estimation.py`](inverse_wind_estimation.py) | Inverse solve: given a prescribed tether force, solve the system model for the wind speed (measured wind as the initial guess). |

## Outputs

EKF results and diagnostics are written under `results/<kite_name>/ekf/`
(HDF5 + plots). These reconstructed states/wind/coefficients feed the
[`../identification/`](../identification/) calibration and the ROM validators.

## Notes

- Reference system is the TU Delft V3; open flight-data sets (V3 and Kitepower V9,
  several with lidar wind) are listed on the project website and in the top-level
  data references.
- Physics/method: Cayon, Watson & Schmehl (2025) *WES* — "Kite as a sensor".
