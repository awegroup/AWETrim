# Reduced-order model (ROM) scripts

The ROM is the fast, CasADi quasi-steady kite model fitted from aerostructural
sweeps (coefficients defined in each kite's `rom_config.yaml`). These scripts use
it to **simulate and optimise trajectories** and to **validate** the model
against flight data. Run everything from the project root.

System properties come from `data/<kite>/system.yaml`; trajectory/pattern
parameters from `data/<kite>/cycle_configs/*.yaml`. The path patterns themselves
(B-splines, uploop/downloop/helix) live in
`awetrim.kinematics.parametrized_patterns`.

## `optimization/`

### `optimization/reelout/` — single reel-out (production) patterns

| Script | What it does |
|--------|--------------|
| [`downloop_pattern.py`](optimization/reelout/downloop_pattern.py) | Simulate a periodic-spline **downloop** production loop from `cycle_configs/downloop_spline.yaml`. |
| [`uploop_pattern.py`](optimization/reelout/uploop_pattern.py) | Same for an **uploop** pattern. |
| [`helix_pattern.py`](optimization/reelout/helix_pattern.py) | Same for a **helix** pattern. |
| [`generate_spline_config.py`](optimization/reelout/generate_spline_config.py) | Generate a reel-out periodic-B-spline cycle config from a simple named initial curve (writes the YAML; does not simulate). |

Each pattern script builds a `Phase`, simulates the loop, saves the timeseries
(JSON) and produces trajectory/power plots.

### `optimization/reelin/` — reel-in

| Script | What it does |
|--------|--------------|
| [`simple_reelin.py`](optimization/reelin/simple_reelin.py) | Build a `ReelinSimple` (pure reel-in followed by the transition back to reel-out), simulate it, then optimise a small parameter set (e.g. transition start elevation) with the end radius constrained to its target. `--plot` to show figures. |

### `optimization/cycle/` — full pumping cycle

| Script | What it does |
|--------|--------------|
| [`run_cycle_simulation.py`](optimization/cycle/run_cycle_simulation.py) | Stitch a reel-out `Phase` + `ReelinSimple` into a full `CycleSimple` pumping cycle. Flags: `--shape {downloop,uploop,helix}`, `--plot`, `--figures N`, and `--optimize` with `--method {alternating,monolithic}` to maximise cycle power over the path and control parameters (CasADi Opti / IPOPT). |

## `validation/` — against flight data

Both validators reproduce a specific flight, so they use the **as-flown** KCU
mass (`LEI_V3_SYSTEM_FLOWN_CONFIG`).

| Script | What it does |
|--------|--------------|
| [`validate_quasi_steady_state_v3.py`](validation/validate_quasi_steady_state_v3.py) | Take measured kinematics + wind as given and solve the quasi-steady force balance for tether force, steering input and tangential speed; compare predicted vs. measured. Can run rigid / flexible-lumped / `WilliamsTether` variants. |
| [`validate_spline_v3.py`](validation/validate_spline_v3.py) | Fit B-spline patterns to measured trajectories cycle-by-cycle, simulate them with the ROM and compare against the flight data. |

## Notes

- ROM aero parameters are calibrated/identified by the
  [`identification/`](../identification/) scripts; the coefficient definitions
  live in `rom_config.yaml`.
- Optimisation-variable bounds are centralised in
  `src/awetrim/utils/defaults.py` (`DEFAULT_OPTI_LIMITS`) — add new variables
  there, not inline. Physics traces to Cayon, van Deursen & Schmehl (2026) *WES*
  and the trajectory-optimisation abstract; see the project `AGENTS.md`.
