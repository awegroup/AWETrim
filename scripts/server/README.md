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

On Linux/macOS, from the AWETrim folder — the script picks up `venv/` itself, so
no `activate` needed:

```
./bin/run_server                      # http://127.0.0.1:8000
./bin/run_server --port 9000          # different port
./bin/run_server --host 0.0.0.0       # reachable from other machines
```

It checks that the venv exists and that the server extra is installed, and tells
you what to run if not. On Windows (or to bypass the script), with the venv
active:

```
python scripts/server/run_reelout_server.py --port 8000
```

Leave the terminal open. Your own program (Julia/MATLAB/...) talks to
`http://127.0.0.1:8000`. Interactive documentation of every message:
http://127.0.0.1:8000/docs

## Inputs and outputs

Lengths m, speeds m/s, forces N, time s.

**Angles are in DEGREES in the co-simulation structs** — `trajectory.azimuth`,
`trajectory.elevation` (`/init` and `/step`, request *and* reply) and
`inflow_conditions.wind_direction`. **Radians** are used only by the low-level
parts: the guidance table of `GET /trajectory` (`azimuth`, `elevation` [rad],
`azimuth_dot`, `elevation_dot` [rad/s]) and the `initial_guess` amplitudes.

Frame: spherical, centered at the winch. `azimuth = 0` points straight downwind,
`elevation` is measured up from the ground plane, `distance_radial` (r) is the
tether-sphere radius.

### 1. `POST /init` — set up (once)

Input — the `InitParams` struct. `inflow_conditions` is the only required
field; everything else has a default:

```json
{
  "name": "uwe-sim-1",
  "max_time": 600.0,
  "length": 200.0,
  "winch_params": {"mode": "reelout", "k_v": 0.11, "f_min": 1000, "f_max": 8400},
  "inflow_conditions": {"wind_speed": 5.2, "wind_direction": 270.0, "profile_law": 2, "z0": 0.03},
  "trajectory": {"azimuth": ["..."], "elevation": ["..."]},
  "input_depower": 1.6, "reg_weight": 1.0, "detect_simple_bounds": true
}
```

| field | default | meaning |
|---|---|---|
| `inflow_conditions` | required | the wind seen by the kite, see below |
| `name` | `reelout-optimization` | name of the simulation, echoed in the reply |
| `max_time` | — | your simulation horizon [s]; stored and echoed, not used by the optimizer |
| `length` | from config (200) | initial tether length / pattern sphere radius r0 [m] |
| `winch_params` | from config | the ground-station winch law, see [2b](#2b-co-simulation-structs-juliahost-side-contract) |
| `trajectory` | built-in figure-eight | starting flight path in degrees; fitted to the pattern B-spline. Also sets the reply resolution |
| `input_depower` | from config | depower setting; the FIXED value when depower is not optimized |
| `reg_weight` | from config | smoothness regularization |
| `detect_simple_bounds` | from config | IPOPT speed-up |

The last three are the complete set of solver knobs `InitParams` exposes; there
is no generic overrides dict. Everything else comes from the cycle config the
server loads (the LEI-V3 downloop spline config by default) — see
[Low-level API](#low-level-api) for the few extra request fields that reach
into it.

The trajectory you send is only a starting shape — the optimizer reshapes the
curve freely; a rough guess is fine.

#### `inflow_conditions` — the wind

The shared `InflowConditions` struct. `wind_speed` is the speed at **6 m**
height; the profile law extends it upwards:

| field | default | meaning |
|---|---|---|
| `wind_speed` | required | wind speed [m/s] at 6 m |
| `wind_direction` | 0 | direction the wind comes FROM [deg], 0 = North, 90 = East |
| `profile_law` | required | `0`=CONST, `1`=EXP, `2`=LOG, `3`=EXPLOG, `4`=CUSTOM_LOG, `5`=CUSTOM_EXP, `6`=CUSTOM_JET |
| `alpha` | 0.08163 | power-law exponent (EXP, EXPLOG) |
| `z0` | 0.0002 | surface roughness [m] (LOG, EXPLOG) |
| `turbulence` | 0.0 | accepted and echoed, but **not used**: the optimizer is deterministic and works on the mean profile |
| `heights` | `[6.0]` | sample heights [m], for the `CUSTOM_*` laws |
| `speeds` | `[wind_speed]` | wind speeds [m/s] at `heights` |

The laws are those of AtmosphericModels.jl, normalized to `wind_speed` at 6 m:
`EXP` = `(z/6)^alpha`, `LOG` = `ln(z/z0)/ln(6/z0)`, `EXPLOG` = `2*LOG - EXP`.

The `CUSTOM_*` laws describe a **measured or forecast profile**: they are
least-squares fits of the `heights`/`speeds` table and ignore `wind_speed`,
`alpha` and `z0`. `CUSTOM_LOG` fits a log law (≥2 samples), `CUSTOM_EXP` a
power law (≥2 samples), `CUSTOM_JET` a log law plus a Gaussian low-level jet
`u(z) = u_bg(z) + U_J*exp(-(z - z_c)²/(2σ²))` (≥5 samples):

```json
{"wind_speed": 8.4, "wind_direction": 265.0, "profile_law": 6,
 "heights": [10, 50, 100, 200, 300], "speeds": [5.5, 7.4, 8.0, 9.3, 8.6]}
```

A request the fit cannot handle (too few samples, a profile decreasing with
height for `CUSTOM_LOG`, ...) is rejected with **422** before any solve.
`wind_direction` only orients the result in the world frame — the path is
optimized in the wind-aligned frame, where azimuth 0 is downwind by
definition.

`inflow_conditions` is the only way to describe the wind: the low-level
`wind` field (an AWETrim `Wind` model sent verbatim) is gone, and requests
carrying it are rejected with **422**. `GET /status` still reports the
resolved `Wind` model under `wind`, read-only, as diagnostics.

Output: `{"state": "ready", ...}`. The same field descriptions appear in the
interactive docs (`/docs`, expand the POST /init schema).

### 2. `POST /step` — request an optimization

Input — empty `{}` for the first solve; on refreshes send what changed:

```json
{
  "inflow_conditions": {"wind_speed": 8.4, "wind_direction": 265.0, "profile_law": 4,
                        "heights": [10, 100, 300], "speeds": [5.5, 8.0, 9.3]},
  "length": 220.0,
  "winch_params": {"mode": "reelout", "k_v": 0.02, "f_min": 1000, "f_max": 8400},
  "trajectory": {"azimuth": ["..."], "elevation": ["..."]}
}
```

`length` is the **current tether length** from your simulator; the
refreshed pattern is re-anchored there. All fields are optional.

**Blocking by default:** the call returns when the solve finishes (~10-20 s)
and the reply contains the StepParams struct — `winch_params` plus the
**optimized** `trajectory` (azimuth/elevation in degrees, closed, same number
of points you sent) — plus `metrics`. An infeasible request returns **422**
with the solver message and the previous trajectory stays available.
Add `"wait": false` to get the old asynchronous behavior instead
(`202 {"state": "solving"}`, then poll `/status`).

### 2b. Co-simulation structs (Julia/host-side contract)

`/init` and `/step` accept and return the shared structs:

- `WinchParams {mode, k_v, f_min, f_max}` — the ground-station winch
  law `v_set = k_v*sqrt(force)`. It is mapped onto the optimizer's quadratic
  radial force model (`F = v²/k_v²`, clamped to `[f_min, f_max]`), so the
  optimized path assumes exactly the client's winch behavior. `mode: "reelin"` → 400 (not supported yet).
- `Trajectory {azimuth[], elevation[]}` — flight path in **degrees**,
  periodic (closing point optional on input, always present in replies).
  On input it is fitted to the pattern B-spline as the starting guess;
  on output it is the optimized path sampled at your resolution.
- `InflowConditions {wind_speed, wind_direction, profile_law, ...}` — the wind
  seen by the kite, see the table above. Required on `/init`; optional on
  `/step`, where it replaces the profile for the re-optimization.
- `InitParams {name, max_time, length, winch_params, inflow_conditions,
  trajectory, input_depower, reg_weight, detect_simple_bounds}` — the `/init`
  request and reply carry these fields (reply trajectory = fitted starting
  path, `inflow_conditions` echoed with the defaults filled in). `length` is
  the initial tether length; the last three are the solver knobs and default
  to `1.6`, `1.0` and `true`.
- `StepParams {length, winch_params, trajectory}` — the `/step` request and
  reply, `length` being the current tether length.
- The struct fields go on the wire as they are — the clients do no mapping.
  The three solver knobs are sent flat; the server merges them into the cycle
  config it solves with. Both replies echo `length`, `/init` also echoes the knobs.
  (`distance_radial` stays the name of the tether length in the guidance
  *table* below, where it is the physical radius r along the path.)

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
| `s` | 0.0, 0.063, 0.126, ... | path parameter, one figure = 0 → 2π (periodic) |
| `azimuth` | -0.097, -0.074, ... | kite azimuth [rad] |
| `elevation` | 0.377, 0.386, ... | kite elevation [rad] |
| `azimuth_dot` | 0.082, 0.080, ... | azimuth rate [rad/s] (feedforward) |
| `elevation_dot` | 0.032, 0.034, ... | elevation rate [rad/s] |
| `distance_radial` | 220.0, 220.3, ... | tether length r [m] |
| `speed_radial` | 1.24, 1.25, ... | reel-out speed r_dot [m/s] |
| `tension_tether_ground` | 3715, 3779, ... | ground tether tension [N] |
| `s_dot`, `input_steering` | ... | path speed, steering input |

Plus `spline` (the B-spline coefficients defining the same curve continuously),
`metrics`, and `optimized_parameters`. Cartesian position if needed:
`x = r·cos(el)·cos(az)`, `y = r·cos(el)·sin(az)`, `z = r·sin(el)`.

**Important:** when swapping in a refreshed trajectory mid-flight, align on `s`
(or on azimuth/elevation), **not** on `t` — `t` restarts at zero and the pattern
is periodic in `s`.

## Ready-made clients

- **Python**: [`client_example.py`](client_example.py) — the shared structs
  (InflowConditions/WinchParams/InitParams/StepParams) as dataclasses + httpx,
  self-contained; `python scripts/server/client_example.py` runs the demo.
- **Julia**: [`client_example.jl`](client_example.jl) — the same structs with
  HTTP.jl + JSON3; blocking init/step flow.
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

## Low-level API

Everything above is the co-simulation contract: `InitParams` in, `StepParams`
out. `POST /init` additionally accepts the fields below. The ready-made clients
never send them and a co-simulation does not need them — they are for
exploring from `/docs`, for scripted parameter studies, and for pointing the
server at a different kite:

| field | default | meaning |
|---|---|---|
| `initial_guess.curve_type` | `lissajous` | starting shape: `lissajous`/`lemniscate` = figure-eight, `helix` = circular loops |
| `initial_guess.az_amp0` | 0.3 | azimuth half-width of the figure [rad] |
| `initial_guess.beta0` | 0.35 | mean elevation above the horizon [rad] |
| `initial_guess.beta_amp0` | 0.12 | elevation half-height of the figure [rad] |
| `initial_guess.downloops` | true | kite turns downward (true) or upward (false) through the loops |
| `initial_guess.M` | 10 | B-spline control points — the optimizer's shape freedom |
| `n_points` | 100 | optimization grid nodes (also the reply-table resolution unless you send a `trajectory` — then replies match your resolution) |
| `optimization_params` | `["C_phi","C_beta","input_depower"]` | what the optimizer is allowed to change |
| `target` | `power` | `power` = maximize average power (energy/time), `energy` = maximize energy per pattern |
| `system_config_path` | built-in LEI-V3 | path on the server to a different system (kite) config |
| `cycle_config_path` | built-in downloop spline | path on the server to a different cycle config |

`initial_guess` is the parametric alternative to `trajectory`, for clients that
have no flight path yet. If both are sent, `trajectory` wins.

Unknown keys are rejected with **422** rather than silently ignored, so a
client still sending a removed field name notices immediately.

### Cycle-config solver knobs

Everything not listed above comes from the cycle config the server loads, under
its `reelout.sim_parameters:` section. Two entries there change the
optimization noticeably:

| key | default | meaning |
|---|---|---|
| `winch_mode` | `"force_law"` | `"force_law"`: reel speed follows the winch tension curve (the WinchParams law). `"free_speed"`: reel speed is a free, rate-limited control — the optimizer picks it |
| `optimize_depower_profile` | `false` | `false`: depower optimized as ONE scalar (if `input_depower` is in `optimization_params`). `true`: optimized PER NODE (a depower time-profile) |

Depower cheat-sheet: **fixed** = remove `"input_depower"` from
`optimization_params` and set the `input_depower` field; **one optimized
value** = keep it in `optimization_params` (default); **optimized per point** =
additionally set `optimize_depower_profile: true` in the cycle config.
