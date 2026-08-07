# Reelout trajectory optimization — API guide

AWETrim can compute the **power-optimal reelout flight path** for a kite, given a
wind profile. This server exposes that computation as a small JSON API, so it can
be used from **any language** (Julia, MATLAB, ...) without installing Python,
CasADi, or IPOPT.

The intended use is a simulation loop in another program:

```
your simulator                          this server
--------------                          -----------
                    ── POST /init ──►   build the model (once, no solve)
                    ── POST /step ──►   optimize in the background
fly the kite along  ── GET /status ─►   "solving" ... "converged"
the trajectory      ── GET /trajectory► the optimal path as a table
   ...
tether got longer,  ── POST /step ──►   re-optimize (warm start, faster)
wind changed        ── GET /trajectory► refreshed path
```

While a re-optimization runs, `GET /trajectory` keeps returning the **previous**
path — the simulator is never left without a curve to follow. If a solve fails,
the previous path is also kept.

## One-time setup (on whichever machine runs the server)

Prerequisites: [Python 3.10+](https://www.python.org/downloads/) and
[git](https://git-scm.com/downloads) (used to fetch dependencies). No Python
knowledge is needed beyond running these commands once:

```
git clone https://github.com/awegroup/AWETrim.git
cd AWETrim
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -e .[server]       # ~5-10 min, downloads the solver stack
```

## Starting the server (every session: one command)

From the AWETrim folder, with the venv active:

```
python scripts/server/run_reelout_server.py --port 8000
```

Leave the terminal open. Your own program (Julia/MATLAB/...) talks to
`http://127.0.0.1:8000`. Interactive documentation of every message:
http://127.0.0.1:8000/docs

## Inputs and outputs

All quantities: **angles rad, rates rad/s, lengths m, speeds m/s, forces N, time s.**
Frame: spherical, centered at the winch. `azimuth = 0` points straight downwind,
`elevation` is measured up from the ground plane, `distance_radial` (r) is the
tether-sphere radius.

### 1. `POST /init` — set up (once)

Input — the wind profile is the only required field; everything else has a
default:

```json
{
  "wind": {"model_type": "logarithmic", "U_ref": 8.0, "z_ref": 100.0, "z0": 0.03},
  "distance_radial": 200.0,
  "initial_guess": {"curve_type": "lissajous", "az_amp0": 0.3, "beta0": 0.35,
                    "beta_amp0": 0.12, "downloops": true, "M": 10},
  "n_points": 100,
  "sim_parameters": {"input_depower": 1.6, "reg_weight": 1.0, "detect_simple_bounds": true}
}
```

| field | default | meaning |
|---|---|---|
| `wind` | required | see below |
| `distance_radial` | 200 | initial tether length / pattern sphere radius r0 [m] |
| `initial_guess.curve_type` | `lissajous` | starting shape: `lissajous`/`lemniscate` = figure-eight, `helix` = circular loops |
| `initial_guess.az_amp0` | 0.3 | azimuth half-width of the figure [rad] |
| `initial_guess.beta0` | 0.35 | mean elevation above the horizon [rad] |
| `initial_guess.beta_amp0` | 0.12 | elevation half-height of the figure [rad] |
| `initial_guess.downloops` | true | kite turns downward (true) or upward (false) through the loops |
| `initial_guess.M` | 10 | B-spline control points — the optimizer's shape freedom |
| `n_points` | 100 | optimization grid nodes (also the reply-table resolution unless you send a trajectory — then replies match your resolution) |
| `depower` | `{"mode":"optimize"}` | how the depower input l_dp is handled, see below |
| `optimization_params` | `["C_phi","C_beta","input_depower"]` | what the optimizer is allowed to change |
| `sim_parameters` | `{}` | solver knobs, see below |

Solver knobs worth knowing (`sim_parameters`):

| key | default | meaning |
|---|---|---|
| `winch_mode` | `"force_law"` | `"force_law"`: reel speed follows the winch tension curve (the WinchParams law). `"free_speed"`: reel speed is a free, rate-limited control — the optimizer picks it |
| `reg_weight` | config | smoothness regularization |
| `detect_simple_bounds` | config | IPOPT speed-up |

### Depower — read this before flying a returned path

Depower (l_dp) sets the kite's angle of attack, so **an optimized path is only
flyable at the depower it was optimized for**. Choose the mode with the
`depower` field:

```json
{"depower": {"mode": "optimize", "value": 1.6}}
```

| mode | meaning | measured on the LEI-V3 reference case |
|---|---|---|
| `fixed` | l_dp pinned at `value` — your setting is honoured, the optimizer only reshapes the path | 4741 W |
| `optimize` (default) | one scalar l_dp optimized, seeded at `value` | 5167 W (**+9%**) |
| `profile` | l_dp optimized per node — a depower schedule along the pattern | 5733 W (**+21%**), needs an actuator that can follow it |

`value` is the fixed setting in `fixed` mode and the starting value otherwise;
it defaults to the kite cycle config's value.

**Both replies carry the depower back**, next to the trajectory:

```json
{"depower": {"mode": "optimize", "value": 1.5173, "profile": null}}
```

In `profile` mode `profile` holds the per-node values, index-aligned with the
reply trajectory (`profile[i]` belongs to `azimuth[i]`/`elevation[i]`), and
`value` is their mean. The same information is in the `/trajectory` table's
`input_depower` column, which is always present.

Fly the trajectory **and** its depower. On the reference case, flying an
optimized path at the client's own l_dp = 1.6 instead gives 4640 W against a
reported 5167 W — 10% below the number, and worse than simply asking for
`mode: "fixed"`. If your simulator cannot command depower, use `fixed`.

`POST /step` accepts the same `depower` field to change mode or value
mid-session. The older combination of `optimization_params` +
`sim_parameters.input_depower` / `optimize_depower_profile` still works
unchanged when `depower` is omitted.

The initial guess is only a starting shape — the optimizer reshapes the curve
freely; a rough guess is fine. Wind can alternatively be a measured/forecast
profile:

```json
{"model_type": "tabulated", "heights": [10, 100, 300], "speeds": [5.5, 8.0, 9.3]}
```

Output: `{"state": "ready", ...}`. The same field descriptions appear in the
interactive docs (`/docs`, expand the POST /init schema).

### 2. `POST /step` — request an optimization

Input — empty `{}` for the first solve; on refreshes send what changed:

```json
{
  "wind": {"model_type": "tabulated", "heights": [10, 100, 300], "speeds": [5.5, 8.0, 9.3]},
  "distance_radial": 220.0,
  "winch_params": {"mode": "reelout", "k_v": 0.02, "v_max": 8.0, "f_min": 1000, "f_max": 8400},
  "trajectory": {"azimuth": ["..."], "elevation": ["..."]}
}
```

`distance_radial` is the **current tether length** from your simulator; the
refreshed pattern is re-anchored there. All fields are optional.

**Blocking by default:** the call returns when the solve finishes (~10-20 s)
and the reply contains the StepParams struct — `winch_params`, the
**optimized** `trajectory` (azimuth/elevation in degrees, closed, same number
of points you sent) and the `depower` that path assumes — plus `metrics`.
An infeasible request returns **422**
with the solver message and the previous trajectory stays available.
Add `"wait": false` to get the old asynchronous behavior instead
(`202 {"state": "solving"}`, then poll `/status`).

### 2b. Co-simulation structs (Julia/host-side contract)

`/init` and `/step` accept and return the shared structs:

- `WinchParams {mode, k_v, v_max, f_min, f_max}` — the ground-station winch
  law `v_set = k_v*sqrt(force)`. It is mapped onto the optimizer's quadratic
  radial force model (`F = v²/k_v²`, clamped to `[f_min, f_max]`, `v_max`
  bounding the reel speed), so the optimized path assumes exactly the
  client's winch behavior. `mode: "reelin"` → 400 (not supported yet).
- `Trajectory {azimuth[], elevation[]}` — flight path in **degrees**,
  periodic (closing point optional on input, always present in replies).
  On input it is fitted to the pattern B-spline as the starting guess;
  on output it is the optimized path sampled at your resolution.
- `DepowerParams {mode, value}` on requests, `{mode, value, profile}` on
  replies — the depower the path is optimized for. See the depower section
  above; the returned trajectory is only flyable together with this value.
- `InitParams {name, max_time, winch_params, trajectory, depower}` — the
  `/init` request and reply carry these fields (reply trajectory = fitted
  starting path); `wind` (and optionally `distance_radial`) travel as sibling
  JSON fields, since the optimizer cannot work without wind.
- `StepParams {winch_params, trajectory, depower}` — the `/step` request and
  reply.

See [`client_example.jl`](client_example.jl) for the exact structs in Julia.

### 3. `GET /status` — poll until done

```json
{
  "state": "converged",
  "metrics": {"energy_J": 85636.5, "total_time_s": 14.82, "avg_power_W": 5776.7},
  "last_error": null
}
```

`state` is one of `uninitialized | ready | solving | converged | failed`.
A cold solve takes ~10 s, warm refreshes similar or faster.

### 4. `GET /trajectory` — the result

The path as a table of 100 points (`table.<column>[i]` = node i), real output:

| column | first values | meaning |
|---|---|---|
| `t` | 0.0, 0.275, 0.569, ... | time along the path, restarts at 0 each refresh |
| `s` | 0.0, 0.01, 0.02, ... | path parameter, one figure = `spline.s_init` → `spline.s_final` (0 → 1 with the shipped config; periodic) |
| `azimuth` | -0.097, -0.074, ... | kite azimuth [rad] |
| `elevation` | 0.377, 0.386, ... | kite elevation [rad] |
| `azimuth_dot` | 0.082, 0.080, ... | azimuth rate [rad/s] (feedforward) |
| `elevation_dot` | 0.032, 0.034, ... | elevation rate [rad/s] |
| `distance_radial` | 220.0, 220.3, ... | tether length r [m] |
| `speed_radial` | 1.24, 1.25, ... | reel-out speed r_dot [m/s] |
| `tension_tether_ground` | 3715, 3779, ... | ground tether tension [N] |
| `input_depower` | 1.5173, 1.5173, ... | depower the path assumes — constant in `fixed`/`optimize` mode, per-node in `profile` mode |
| `s_dot`, `input_steering` | ... | path speed, steering input |

Plus `spline` (the B-spline coefficients defining the same curve continuously),
`metrics`, and `optimized_parameters`. Cartesian position if needed:
`x = r·cos(el)·cos(az)`, `y = r·cos(el)·sin(az)`, `z = r·sin(el)`.

**Important:** when swapping in a refreshed trajectory mid-flight, align on `s`
(or on azimuth/elevation), **not** on `t` — `t` restarts at zero and the pattern
is periodic in `s`.

## Ready-made clients

- **Julia**: [`client_example.jl`](client_example.jl) — the shared structs
  (InitParams/StepParams) with HTTP.jl + JSON3; blocking init/step flow.
- **Python**: [`client_example.py`](client_example.py) — the same structs and
  flow as the Julia example (dataclasses + httpx, self-contained);
  `python scripts/server/client_example.py` runs the demo.
- **MATLAB**: [`client_example.m`](client_example.m) — built-in webread/webwrite
  only; includes a 3D plot of the optimized pattern.
- **Python (rich API)**: [`client.py`](client.py) — `ReeloutClient` class using
  the asynchronous step/status/trajectory flow and the full guidance table;
  `python scripts/server/client.py` runs a full demo.

## Errors

| situation | response |
|---|---|
| `/step` while a solve is running | `409` — wait for `/status` to finish |
| `/step` before `/init` | `409` |
| `/trajectory` before any success | `404` |
| solve failed | `/status` → `"state": "failed"` + `last_error`; previous trajectory still served |
