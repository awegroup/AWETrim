# Billow documentation

`billow.tex` is the full technical document: formulation, element library,
validation against external benchmarks, demonstration cases, the `kite_fem`
comparison, and the canopy mesh study.

Rebuild (needs the figures under `results/structural/`, regenerate them with the
scripts listed in the document's final section):

```sh
python scripts/structural/run_demo_cases.py
python scripts/structural/run_validation_benchmarks.py
python scripts/structural/run_mesh_requirement.py
cd docs/billow && pdflatex billow.tex && pdflatex billow.tex
```

Figure paths are resolved via `\graphicspath` relative to this directory.
