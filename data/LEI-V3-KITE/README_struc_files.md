# LEI-V3 structural geometry

`structure.yml` is the kite's ONE structural geometry, an
[awesIO](https://github.com/awegroup/awesIO) structure document
(`structure_schema.yml`). Read it with
`awetrim.aerostructural.coupled.awesio_structure.load_structure`. Both
structural fidelities start from it:

- **Billow full** (`awetrim.aerostructural.billow.driver`) reads it as it is:
  inflatable strut and leading-edge tubes (`tubes`, on one-point `bodies`,
  Breukels law at 0.3 bar), the membrane canopy between them (28
  `spanwise_stations`, 189 `canopy_faces`), and the complete bridle down to
  the individual a/b/c/d lines that fan onto each strut (`segments`, with each
  pulley's two arms in `pulleys`).
- **Wireframe** (`awetrim.aerostructural.wireframe.driver`) reduces it on the
  fly (`wireframe.reduce_fem.reduce_fem_geometry`). The reduction has one rib
  per strut with two wing nodes, where the front and rear bridle attach, and
  each per-strut line fan collapsed to one equivalent line. It flies the full
  geometry's aerodynamic mesh. The reduction is written next to each case as
  `structure_wireframe.yml` (awesIO too).

`awetrim.aerostructural.case.DEFAULT_STRUC_GEOMETRY_FILENAME` names the file.
The step-1 aerodynamics (bridle drag, mass and inertia) read it too.

## What the file holds

The model is written FULLY RESOLVED, as awesIO requires: every node, including
the strut padding and the canopy chord nodes the old reader used to generate,
is a row of `points`. Positions are the kite's KA frame (x leading to trailing
edge, y to the left tip, z up) with the KCU at the origin.

- **Masses** follow awesIO's rule: a reader derives the lines (half of each
  segment to its ends) and the fabric (an equal share of each face), and the
  file states the rest as extra mass -- the pulleys on their points, the tubes
  on their bodies (11 kg wing: canopy 170 g/m2, the rest on the tubes by
  length). **The KCU mass is not here**: `system.yaml` is its single source.
- **Columns of AWETrim's own** on `segments`: `line` (the bridle line a
  segment belongs to; `depower_tape` and `steering_tape` are the actuated
  ones), `linktype` (`noncompressive` or `pulley`) and `tape_width` (the drag
  width of a flat tape; `diameter` stays the area-equivalent one). The
  top-level `awetrim` block names the KCU point and the pulley mass.
- **Provenance** is in `metadata.note`: the photogrammetry-adjusted bridle
  rest lengths, the Dyneema SK75 modulus (109 GPa, the datasheet's lower
  bound) and equivalent density, the 12 x 1.5 mm tapes, the equivalent legs
  `a5_equiv` / `br_5_equiv`.

To edit a rest length, edit the segment's `rest_length`; a pulley's rope is
the sum of its two arms. After editing node or connectivity rows, rewrite the
file through `Structure.write` so `metadata.connectivity_sha` is refreshed.

A deformed results folder holds a snapshot of the solved shape under the same
name, `structure.yml` (a wireframe: its wing is spring members with
`stiffness`, `damping` and `mass` columns of AWETrim's own).
`case.find_struc_geometry` picks it up there.

## Retired formats

Until 2026-10-29 the geometry was `struc_geometry_FEM_full.yaml` (an AWETrim
authoring format the reader expanded on every run) and the snapshots were
PSM-format `struc_geometry.yaml`. Both are in git history.
`load_structure` still reads them, converting on the fly with a
DeprecationWarning; convert a file once with

    python -m awetrim.aerostructural.coupled.awesio_structure OLD.yaml structure.yml

The PSM wireframe files removed earlier the same day
(`struc_geometry_PSM_reduced*.yaml`) are in the v0.1.0 release. They
reproduce the wes-quasi-steady paper; see
`docs/aerostructural/lei_v3_wing_corrections.md`.

The tape lengths that match the 2019 flight are in `identification.yaml`
(`fit.fidelities.wireframe.flight_match`).
