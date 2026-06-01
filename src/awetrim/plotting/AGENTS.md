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
```

## Frame and Label Conventions

- Default coordinates are in the course frame unless noted otherwise.
- Axes labels use X/Y/Z without units; units should be in titles or legends.
- Use `set_plot_style()` for consistent fonts and color palettes.

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
