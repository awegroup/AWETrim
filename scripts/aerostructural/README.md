# Aerostructural simulation

Simulate the kite as a flexible structure in flight: the aerodynamic loads from
the Vortex Step Method (VSM) and the deformation from the structural solver
[Billow](https://github.com/awegroup/Billow) are iterated until they agree. The
quasi-steady trim is solved at every iteration, so the result is a deformed shape
that flies in equilibrium at the flight state you set.

Run every script from the project root.

## 1. Run a simulation

Open one of the two scripts, edit the **Inputs** block at the top, and run it.

| Script | Structural model | Time |
|--------|------------------|------|
| [`run_simulation_PSM.py`](run_simulation_PSM.py) | **Wireframe**: one rib per strut, the bridle with its pulleys, tubes as stiff axial springs, canopy as tension-only springs | under a minute |
| [`run_simulation_BILLOW.py`](run_simulation_BILLOW.py) | **Full**: inflatable tube beams (leading edge, struts), a wrinkling membrane canopy and the complete bridle | about 1 min |

Both read the same geometry file, `struc_geometry_FEM_full.yaml`, and fly the
same aerodynamic mesh with the same trim and tether model. With the same inputs
(the two Inputs blocks are identical) they differ only in the structural model.
The wireframe is reduced from the full geometry on the fly (one rib per strut,
each strut's bridle fan collapsed to one equivalent line, tube and canopy
stiffness from the Billow materials) and saved in the case folder as
`struc_geometry_wireframe.yaml`. It is not the photogrammetry-corrected PSM
geometry of the wes-quasi-steady paper; `awetrim.aerostructural.wireframe.driver.solve_deformation`
still solves that one.

```bash
python scripts/aerostructural/run_simulation_PSM.py
python scripts/aerostructural/run_simulation_BILLOW.py
```

The inputs:

| Input | Meaning |
|-------|---------|
| `KITE`, `SYSTEM_FILE` | Kite folder under `data/` and the system file in it (masses, KCU, tether) |
| `ELEVATION_DEG`, `AZIMUTH_DEG` | Tether direction: elevation above the ground, azimuth from downwind |
| `COURSE_DEG` | Flight direction on the sphere; 90° is flying across the wind window |
| `TETHER_LENGTH_M`, `REEL_OUT_SPEED_MS` | Radial distance and reel-out speed |
| `WIND_SPEED_MS`, `WITH_GRAVITY` | Wind at the wind model's reference height; gravity on or off |
| `DEPOWER_TAPE_EXTENSION_M`, `STEERING_TAPE_EXTENSION_M` | Actuation relative to the tape lengths in the geometry, stepped in from there |

Each run writes its own case folder, named after its inputs, under
`results/<kite>/aerostructural/wireframe/` or `.../billow/`:

- `sim_output.h5`: node positions, forces and the solver history
- `case.json`: what to draw (wing, tubes, bridle lines with names and tensions) and the flight state
- `config.yaml`, plus the deformed `aero_geometry.yaml` / `struc_geometry.yaml`, which other
  scripts can reuse (e.g. `scripts/aerodynamics/solve_single_state.py --deformed-from <case>`)

Solver settings (tolerances, panel counts, the material model) come from the
kite's `as_config.yaml`.

## 2. Look at the result

```bash
python scripts/aerostructural/plot_simulation.py              # newest case
python scripts/aerostructural/plot_simulation.py <case folder>
```

The run scripts do this automatically when they finish (`SHOW_RESULT = True`).
It prints the trim characteristics and writes `kite_view.html` into the case
folder, a self-contained page that opens in the browser:

- the kite in 3-D (drag to orbit, scroll to zoom), converged or initial shape, or both overlaid
- bridle lines shaded by tension (hover a line for its name and load)
- the trim table: flight state, apparent wind, angle of attack, sideslip, tether force,
  lift and drag, CL and CD with their reference areas, masses, line tensions,
  deformation and convergence

Lift, drag and L/D in the table are those of the VSM force (the wing and its
bridle lines). The KCU drag is listed separately, and the tether is in neither, so
this L/D is higher than the kite's glide ratio.

## Example output

<img src="../../docs/img/aerostructural-deformed-shape.png" alt="Converged deformed LEI-V3 shape" width="460">

## Under the hood

The scripts only set inputs. The pipeline is in the package:

- `awetrim.aerostructural.wireframe.driver.solve_deformation`: wireframe case
- `awetrim.aerostructural.billow.driver.solve_deformation`: Billow full-model case
- `awetrim.aerostructural.case`: kite paths, config defaults, system model
  (`common.py` here re-exports it for older scripts)
- `awetrim.aerostructural.case_view`: `case.json`, and the trim table

## Studies

[`studies/`](studies/) holds the Billow analyses behind `docs/billow/`:
depower and steering continuation chains, the flight-matched sweep and its
interactive page, symmetry and load-transfer checks. See its
[README](studies/README.md).
