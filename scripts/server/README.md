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

If you are using [SimpleKiteControllers.jl](https://github.com/OpenSourceAWE/SimpleKiteControllers.jl),
just run `bin/install_awetrim` from that repository instead — it clones this repo as a
sibling checkout and sets up the venv for you.

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

The same contract is checked in as an OpenAPI 3.1 document,
[`openapi.yaml`](openapi.yaml) — readable without a running server and usable
for client-code generation. Regenerate it after changing `app.py` or
`schemas.py`:

```
python scripts/server/generate_openapi.py
```

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
| `length` | from config (200) | initial tether length / pattern sphere radius r0 [m] |
| `winch_params` | from config | the ground-station winch law, see [2b](#2b-co-simulation-structs-juliahost-side-contract) |
| `trajectory` | built-in figure-eight | starting flight path in degrees; fitted to the pattern B-spline. Also sets the reply resolution |
| `input_depower` | from config | depower setting l_dp: the starting value when depower is optimized (the default), the fixed value otherwise |
| `reg_weight` | from config | smoothness regularization |
| `detect_simple_bounds` | from config | IPOPT speed-up |
| `depower` | `{"mode":"optimize"}` | whether depower is optimized and how, see [Depower](#depower--read-this-before-flying-a-returned-path) |
| `min_turn_radius` | none | minimum physical turn radius [m] the path must respect, see [Minimum turn radius](#minimum-turn-radius--if-your-kite-cannot-turn-as-tightly-as-the-optimizers) |
| `pattern_limits` | none | box on where the path may go, in degrees: `{azimuth_max, elevation_min, elevation_max, azimuth_amplitude_min}`, see [Pattern limits](#pattern-limits--keeping-the-optimizer-out-of-shapes-you-do-not-want) |

`input_depower`, `reg_weight` and `detect_simple_bounds` are the complete set
of solver knobs `InitParams` exposes; there is no generic overrides dict.
Everything else comes from the cycle config the server loads (the LEI-V3
downloop spline config by default) — see [Low-level API](#low-level-api) for
the few extra request fields that reach into it.

### Depower — read this before flying a returned path

Depower (l_dp) sets the kite's angle of attack, so **an optimized path is only
flyable at the depower it was optimized for**. By default the optimizer
optimizes one scalar depower, so the value it returns differs from the
`input_depower` you sent (1.6 → ~1.44 on the reference case). Say explicitly
what you want with the `depower` struct:

```json
{"depower": {"mode": "optimize", "value": 1.6}}
```

| mode | meaning |
|---|---|
| `fixed` | l_dp pinned at `value` — your setting is honoured, the optimizer only reshapes the path |
| `optimize` (default) | one scalar l_dp optimized, seeded at `value` |
| `profile` | l_dp optimized per node — a depower schedule along the pattern; needs an actuator that can follow it |

`value` defaults to the `input_depower` knob, else the cycle config's value.
When the struct is omitted the mode is derived the old way (`input_depower`
listed in `optimization_params` = optimize), so existing clients keep their
behaviour.

**Both replies carry the depower back**, next to the trajectory:

```json
{"depower": {"mode": "optimize", "value": 1.4362, "profile": null}}
```

In `profile` mode `profile` holds the per-node values, index-aligned with the
reply trajectory (`profile[i]` belongs to `azimuth[i]`/`elevation[i]`), and
`value` is their mean. The same information is in the `/trajectory` table's
`input_depower` column (always present) and in `optimized_parameters`.

Fly the trajectory **and** its depower — flying an optimized path at the
starting depower loses ~10 % of the reported power on the reference case. If
your simulator cannot command depower, use `mode: "fixed"`. `POST /step`
accepts the same `depower` field to change mode or value mid-session.

### Minimum turn radius — if your kite cannot turn as tightly as the optimizer's

The optimizer's kite model already limits how tightly the path can turn (its
steering saturates at the actuator range in `system.yaml`), but that limit is
the point-mass model's, not your kite's. If your controller or simulator has
its own limit — e.g. `1/(c1*u_s_max)` for a kite steered by
`psi_dot = c1*v_a*u_s` — send it and the optimizer will not return anything
tighter:

```json
{"min_turn_radius": 11.35}
```

What is enforced: the **physical geodesic turn radius on the tether sphere**,
`r/|kappa|`, at the tether length `r` at which each part of the path is flown
(the lap reels out tens of metres, so a loop near the end of the lap is flown
at a larger radius than `length` — convert with the `distance_radial` column of
`GET /trajectory`, not with the initial length, when you check the path
yourself). It is enforced densely along the path (4 samples per node interval,
together with a floor on the spline's parametric speed so the path cannot hide
a millimetre-long kink in a hesitation point of the parametrization), which
also rules out the cusps and near-degenerate tiny loops that cold solves
occasionally converge to. Purely geometric — it does not change the kite model.

Both replies echo the limit (`min_turn_radius`, `null` when unconstrained), and
`metrics.turn_radius_min_m` reports the tightest radius the returned path
actually has (from the optimizer's own curvature expression, sampled densely);
the per-node values are the `turn_radius` column of `GET /trajectory`. Those
two numbers are exact for the returned spline, so a client-side curvature gate
is no longer needed. `POST /step` accepts `min_turn_radius` too (omit = keep,
`0` = remove the constraint).

Measured on the LEI-V3 reference winch/wind: on the normal branch the limit is
inactive (the model's own optimum has ~12.3–13 m loops); it removes the
sporadic tight/cusped basins of cold solves. Because a cold *constrained* solve
finds the best branch less often than an unconstrained one, the first solve of
a session with a limit is staged — cold without the limit, then a warm re-solve
with it (fallback: cold with it) — so expect the first `/step` to take about
twice as long; later steps solve once. Cold solves remain path-sensitive:
prefer stepping sequentially (see below).

### Pattern limits — keeping the optimizer out of shapes you do not want

By default the optimizer may put the figure anywhere in |azimuth| ≤ 45.8°,
0.6° ≤ elevation ≤ 51.6° (the bounds on the pattern's B-spline coefficients),
and a cold start occasionally converges to a degenerate shape there: a figure
run away to the top of that elevation range where the tension collapses, or a
zero-width "figure" (`C_phi → 0`) on which the dynamics are infeasible. If your
controller has its own envelope, send it and the optimizer stays inside it:

```json
{"pattern_limits": {"azimuth_max": 35, "elevation_min": 10, "elevation_max": 45,
                    "azimuth_amplitude_min": 5}}
```

All fields in **degrees**, all optional (an omitted field keeps the default).
`azimuth_max` / `elevation_min` / `elevation_max` bound the spline's control
coefficients, and a B-spline never leaves the hull of its coefficients, so they
hold for the whole continuous path, not just at the nodes. `azimuth_amplitude_min`
is a floor on the figure's azimuth half-width (one smooth constraint:
mean over the path of azimuth² ≥ value²/2, which a figure-eight or helix of
half-width A satisfies with A ≥ value) — the guard against the zero-width
collapse. Both replies echo the limits in force (`pattern_limits`, `null` =
defaults). `POST /step` accepts the struct too: omit = keep, a struct replaces
the limits as a whole, `{}` clears them. Purely shape limits — the kite model
is unchanged — and the normal optimum (figures of ~25–35° half-width at 20–35°
elevation) is well inside the defaults, so they only act on the bad basins.

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
refreshed pattern is re-anchored there. All fields are optional (`depower`,
`min_turn_radius` and `pattern_limits` may also be updated here).

**Step sequentially.** Each `/step` warm-starts from the previous optimum
(node-wise), so a reel-out driven as one session — `/init` once, then `/step`
with the current `length` (and the previous reply's `trajectory`, or no
trajectory at all) — follows one smooth branch. Solving cold at each length
(a fresh `/init` + `/step` from a rough guess) is what produces the occasional
422 at isolated lengths and the odd tight-looped or cusped path: those are
different local optima of the cold start, not properties of the length.

**Blocking by default:** the call returns when the solve finishes (~10-20 s)
and the reply contains the StepParams struct — `winch_params` plus the
**optimized** `trajectory` (azimuth/elevation in degrees, closed, same number
of points you sent) and the `depower` that path assumes — plus `metrics`
(including `turn_radius_min_m`). An infeasible request returns **422**
with the solver message and the previous trajectory stays available.
Add `"wait": false` to get the old asynchronous behavior instead
(`202 {"state": "solving"}`, then poll `/status`).

### 2b. Co-simulation structs (Julia/host-side contract)

`/init` and `/step` accept and return the shared structs:

- `WinchParams {mode, k_v, f_min, f_max, v_max | p_max}` — the ground-station
  winch law `v_set = k_v*sqrt(force)`. It is mapped onto the optimizer's
  quadratic radial force model (`F = v²/k_v²`, clamped to `[f_min, f_max]`),
  so the optimized path assumes exactly the client's winch behavior. Past
  `f_max` the controller holds the force while the reel speed keeps rising up
  to the winch's power limit, so give the speed cap as **either** `v_max`
  [m/s] **or** `p_max` [W] (`v_max = p_max/f_max`); both optional, both at
  once is rejected; omitted = the optimizer's default bound (10 m/s).
  `mode: "reelin"` → 400 (not supported yet).
- `Trajectory {azimuth[], elevation[]}` — flight path in **degrees**,
  periodic (closing point optional on input, always present in replies).
  On input it is fitted to the pattern B-spline as the starting guess;
  on output it is the optimized path sampled at your resolution.
- `InflowConditions {wind_speed, wind_direction, profile_law, ...}` — the wind
  seen by the kite, see the table above. Required on `/init`; optional on
  `/step`, where it replaces the profile for the re-optimization.
- `DepowerParams {mode, value}` on requests, `{mode, value, profile}` on
  replies — whether/how depower is optimized and the value the returned path
  assumes; see the depower section above.
- `PatternLimits {azimuth_max, elevation_min, elevation_max,
  azimuth_amplitude_min}` [deg], all optional — a box on where the path may
  go plus an azimuth-amplitude floor; see the pattern-limits section above.
- `InitParams {name, length, winch_params, inflow_conditions, trajectory,
  input_depower, reg_weight, detect_simple_bounds, depower, min_turn_radius,
  pattern_limits}` — the `/init` request and reply carry these fields (reply
  trajectory = fitted starting path, `inflow_conditions` echoed with the
  defaults filled in, `depower` echoed as mode + starting value). `length` is
  the initial tether length; the three solver knobs default to `1.6`, `1.0`
  and `true`; `depower`, `min_turn_radius` and `pattern_limits` are optional.
- `StepParams {length, winch_params, trajectory, depower, min_turn_radius,
  pattern_limits}` — the `/step` request and reply, `length` being the
  current tether length; on the reply `depower` is the setting the optimized
  path assumes, `min_turn_radius` the limit and `pattern_limits` the box it
  was optimized under.
- The struct fields go on the wire as they are — the clients do no mapping.
  The three solver knobs are sent flat; the server merges them into the cycle
  config it solves with. Both replies echo `length`, `/init` also echoes the knobs.
  (`distance_radial` stays the name of the tether length in the guidance
  *table* below, where it is the physical radius r along the path.)
- The structs carry the physical state only. `wait` and `max_iter` control how
  the call is made, not what is optimized, so they are wire fields of `/step`
  and keyword arguments of the clients' `step()` — not `StepParams` fields.
  There is no `wait` on `/init`: it only assembles the session and fits the
  starting path, it never solves.

See [`client_example.jl`](client_example.jl) for the exact structs in Julia.

### 3. `GET /status` — poll until done

```json
{
  "state": "converged",
  "metrics": {"energy_J": 85636.5, "total_time_s": 14.82, "avg_power_W": 5776.7,
              "turn_radius_min_m": 12.6},
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
| `input_depower` | 1.4362, 1.4362, ... | depower the path assumes — constant in `fixed`/`optimize` mode, per-node in `profile` mode |
| `turn_radius` | 46.2, 38.9, ... | physical turn radius of the path at the node [m] (geodesic, at that node's `distance_radial`); the tightest value along the whole path is `metrics.turn_radius_min_m` |
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
  HTTP.jl + JSON3; blocking init/step flow, `step(params; wait = false)` for
  the asynchronous one.
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
