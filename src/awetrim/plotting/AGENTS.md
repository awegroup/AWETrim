# AWETrim Plotting Module

## Status: ✅ Built

## Scope

Shared plotting utilities and reference-frame conventions used across modules.
Module-specific plots should remain in their owning module unless they are reused.

## Public Layout

```
src/awetrim/plotting/
  __init__.py
  plotting.py
  kite_tether.py
```

`kite_tether.py` holds the shared 3D kite + Williams-tether drawing
(`draw_kite_tether`, `draw_wind_axes`, `tether_info_str`), reused by
`scripts/aerodynamics/solve_single_state.py` and the personal kite/tether state
scripts. It depends on `awetrim.aerodynamics` (lazy import inside the draw call,
to keep `import awetrim.plotting` free of casadi/aerodynamics), so it is **not**
re-exported from `__init__.py` — import it directly:
`from awetrim.plotting.kite_tether import draw_kite_tether`.

## Frame and Label Conventions

- Default coordinates are in the course frame unless noted otherwise.
- Axes labels use X/Y/Z without units; units should be in titles or legends.
- Use `set_plot_style()` for consistent fonts and color palettes.

## Units in axis labels (Copernicus style — repo standard)

Figures follow the Copernicus journal style, since the paper figures are
submitted as drawn:

- Units in **parentheses**, never square brackets: `$v_\mathrm{k}$ (m s$^{-1}$)`.
- **Negative exponents**, not slashes: `m s$^{-1}$`, `rad s$^{-1}$`, never `m/s`.
- Unit symbols upright (roman), quantity symbols italic (math mode).
- **Dimensionless quantities carry no unit marker at all**: `$C_L$`, not
  `$C_L$ [-]` or `$C_L$ (-)`.
- Angles in labels use `($^\circ$)`.

Console/CSV headers may keep the compact `[unit]` form; the rule is for
rendered figures.

## Reference Frames for Structural and Aerodynamic Plots

Two reference frames should be drawn when visualising the kite structure:

### Course Frame (C)

The course frame is the primary AWETrim / Casadi-model frame.
Its basis vectors in 3-D structural plots are:

```
X_C  — tangential  (direction of kite motion along the wind-sphere surface, i.e. forward)
Y_C  — normal      (perpendicular to motion, pointing laterally on the sphere surface)
Z_C  — radial      (along the tether, positive outward from the ground station)
```

Defined by `transformation_C_from_W(azimuth, elevation, course)` in
`awetrim/utils/reference_frames.py`.

### Body Frame (K)

The body frame is fixed to the kite and is reached from the course frame by applying
Euler angles (roll φ, pitch θ, yaw ψ) via `transformation_C_from_K(pitch, roll, yaw)`
(order: Yaw → Pitch → Roll):

```
X_K  — longitudinal body axis (approximately aligned with X_C at zero angles)
Y_K  — lateral body axis
Z_K  — normal body axis
```

### Structural-Model / VSM Frame

The structural model and VSM use a convention where X and Y are negated relative to
the course frame:

```
T_structural_from_C = [[-1, 0, 0],
                        [ 0,-1, 0],
                        [ 0, 0, 1]]
```

This transform is applied inside `aerodynamics/vsm_quasi_steady.py` before forces reach
the aerostructural module. Structural geometry coordinates exposed to the rest of AWETrim
are always in the course frame.
