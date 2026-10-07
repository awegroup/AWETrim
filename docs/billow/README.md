# Billow integration documentation

Billow itself lives in its own repository: <https://github.com/awegroup/Billow>.
Its formulation, element library, validation against external benchmarks,
demonstration cases and the measured-kite study are documented there
(`docs/billow.pdf` and <https://awegroup.github.io/Billow/>).

What is left here is the **coupling**: how AWETrim drives Billow against the VSM
quasi-steady trim, and what had to be true before the coupled answer was
physical.

- `integration.md` / `integration.tex` / `integration.pdf` — the coupling
  document: architecture, element mapping, chordwise load placement, the load
  transfer onto the canopy, mirror symmetry, and the open items.
- `matched_sweep.html` — the interactive matched-depower study on Billow's full
  model (TU Delft V3, centre of the wind window): sliders over depower, steering
  and apparent speed, the sweeps charted against the paper's wireframe chains
  and the 2019 flight, every solved row in a table. One self-contained file;
  served at <https://awegroup.github.io/AWETrim/billow/matched_sweep.html> once
  pushed. Rebuilt by `scripts/aerostructural/studies/export_matched_sweep_data.py` and
  `build_matched_sweep_page.py`.
- `figures/` — its figures, produced by the scripts in
  `scripts/aerostructural/studies/` (`plot_billow_geometry.py`,
  `plot_billow_strut_bending.py`, `plot_chordwise_moment_matching.py`,
  `plot_billow_depower_chains.py`).

Rebuild:

```sh
cd docs/billow && pdflatex integration.tex && pdflatex integration.tex
```

Figure paths are resolved via `\graphicspath` relative to this directory.
