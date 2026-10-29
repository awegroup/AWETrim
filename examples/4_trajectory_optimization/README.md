# Step 4: trajectory optimisation

The reduced-order model (ROM) is the fast, CasADi quasi-steady kite model
identified in step 3 (coefficients in the ROM file the kite's `system.yaml`
selects). These examples use it to **simulate and optimise trajectories**:
single reel-out patterns and whole pumping cycles. Run everything from the
project root.

System properties come from `data/<kite>/system.yaml`; trajectory/pattern
parameters from `data/<kite>/cycle_configs/*.yaml`. The path patterns themselves
(B-splines, uploop/downloop/helix) live in
`awetrim.kinematics.parametrized_patterns`.

## `reelout/` — single reel-out (production) patterns

| Script | What it does |
|--------|--------------|
| [`downloop_pattern.py`](reelout/downloop_pattern.py) | Simulate a periodic-spline **downloop** production loop from `cycle_configs/downloop_spline.yaml`. |
| [`helix_pattern.py`](reelout/helix_pattern.py) | Same for a **helix** pattern. |
| [`generate_spline_config.py`](reelout/generate_spline_config.py) | Generate a reel-out periodic-B-spline cycle config from a simple named initial curve (writes `cycle_configs/generated_spline.yaml`, or `--output`; does not simulate). |

Each pattern script builds a `Phase`, simulates the loop, saves the timeseries
(JSON) and produces trajectory/power plots.

## `cycle/` — full pumping cycle

| Script | What it does |
|--------|--------------|
| [`fit_periodic_cycle_config.py`](cycle/fit_periodic_cycle_config.py) | Generate a trim-feasible **seed** for the full-cycle optimisation: one periodic B-spline over the whole cycle (figure-eights + reel-in lobe), a synthetic depower profile and a flight-regressed winch law with a depower-dependent offset. `--kite K` picks a kite folder under `data/` (or any path to one); see *Running a full cycle for your own kite* below. `--loops N` sets the visible half-figures, `--auto` tunes the shape until every node trims inside the optimizer's bounds at the profile's wind (the seed need not close radially: the optimizer closes it). A kite whose seed does not trim at its own roll gain (the bare LEI-V3 ROM needs 1.75x at 6 m/s) is marched with the gain scaled up, and `run_full_cycle_opti.py` continues it back to 1 before optimising power. `--reelin dubins --loops N` designs the reel-in instead of fading the figures out: bounded-curvature Dubins paths on the sphere from the peel-off to a level apex at `beta_reelin_peak` and down to a tangential landing (arcs 60 m, tightening to the profile's `min_turn_radius` only where that saves a detour loop). |
| [`run_full_cycle_opti.py`](cycle/run_full_cycle_opti.py) | Optimise the whole cycle as ONE periodic phase (path + per-node depower profile + free reel speed within the drum envelope) for cycle-average power, staged: a feasibility pass closes and trims the seed with the smallest move, then the power pass and a polish, each with a proximal trust region; `--kite` as above, `--no-optimize` only forward-simulates the seed (the flown-settings baseline). Writes seed-vs-optimised figures and a metrics table via [`cycle_comparison_plots.py`](cycle/cycle_comparison_plots.py). |
### Running a full cycle for your own kite

The full-cycle scripts read everything about a kite from one folder
(`data/<kite>/` or any path given to `--kite`); nothing is kite-specific in
code, so the folder can stay outside the repository.

| File | What the cycle needs from it |
|------|------------------------------|
| `system.yaml` | Mass, tether, the drum envelope (`drums[0]`: reel speeds, acceleration, `max_tether_force`) and the KCU actuator ranges/rates. |
| ROM config (`system.yaml` `models.reduced_order.aerodynamics`, else `rom_config.yaml`) | The ROM coefficients plus `controls.input_depower: {powered, depowered}` (**required**: the depower band in the ROM's own `u_p` unit) and, optionally, `validity.angle_of_attack_deg: [lo, hi]` (the AoA range the ROM was identified on; replaces the default AoA bound). |
| `cycle_profile.yaml` | **Required** `wind`, `winch_law` (the force law the seed's forward simulation marches with) and `seed` (`r0`, the cycle-duration prior and the pattern size: `reelout_fraction`, `beta0`, `beta_amp0`, `az_amp0`, `beta_reelin_peak`); optional `m_per_second`, `min_turn_radius` and a reference `flight`. [`data/LEI-V3-KITE/cycle_profile.yaml`](../../data/LEI-V3-KITE/cycle_profile.yaml) is the annotated example. |

Missing required entries and unknown keys raise instead of falling back on
the V3's numbers. Then:

```bash
python examples/4_trajectory_optimization/cycle/fit_periodic_cycle_config.py --kite path/to/MY-KITE --auto
python examples/4_trajectory_optimization/cycle/run_full_cycle_opti.py --kite path/to/MY-KITE
```

The seed is written to `<kite>/cycle_configs/`, results to
`results/<kite folder name>/optimization/full_cycle/`.

## Example outputs

<img src="../../docs/img/pumping-cycle-trajectory.png" alt="Simulated pumping cycle" width="380"> <img src="../../docs/img/b-spline.png" alt="B-spline trajectory parametrisation" width="380">

*Left: a simulated pumping cycle — downloop reel-out (blue), reel-in (orange) and transition (green). Right: `reelout/generate_spline_config.py` — periodic B-spline pattern in the (φ, β) plane.*

## Notes

- ROM aero parameters are identified in [step 3](../3_rom_identification/)
  and, with flight data, corrected in [step 5](../5_flight_data/); the coefficient definitions
  live in the ROM config the kite's `system.yaml` selects (LEI-V3:
  `rom_config_aerostructural.yaml` by default; its 2019 flight correction is
  `rom_config_aerostructural_flight_corrected.yaml`; the paper's semi-empirical ROM was removed
  2026-10-07 and is archived in git history). The depower input the ROM was identified on is a
  per-kite convention (V3: power-tape length in metres; a kite logging a
  normalised signal uses that) declared in that file's `controls.input_depower` block and read by
  `awetrim.identification.controls.rom_depower_band`; the cycle scripts express
  every depower profile in the kite's own unit.
- Optimisation-variable bounds are centralised in
  `src/awetrim/utils/defaults.py` (`DEFAULT_OPTI_LIMITS`) — add new variables
  there, not inline. Physics traces to Cayon, van Deursen & Schmehl (2026) *WES*
  and the trajectory-optimisation abstract; see the project `AGENTS.md`.
