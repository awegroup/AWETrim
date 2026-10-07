# Billow studies

The analyses behind [`docs/billow/`](../../../docs/billow/), all on Billow's full
model (`struc_geometry_FEM_full.yaml`). For a single simulation, use
[`../run_simulation_BILLOW.py`](../run_simulation_BILLOW.py) instead. Run from the
project root, e.g. `python scripts/aerostructural/studies/run_chain_depower_BILLOW.py`.
Scripts in this folder import helpers from each other (`build_once`, `rebuild`,
`decompose`, ...), so keep them together.

## Continuation runs

| Script | What it does |
|--------|--------------|
| [`run_chain_depower_BILLOW.py`](run_chain_depower_BILLOW.py) | Depower continuation chains at the centre of the wind window, one per wind speed. Also the shared model builder (`build_once`, `prepare_structure`, `run_chain`) |
| [`run_sweep_depower_BILLOW.py`](run_sweep_depower_BILLOW.py) | Depower sweep over several apparent speeds |
| [`run_steering_BILLOW.py`](run_steering_BILLOW.py) | Walks the steering half-difference to a target, each step from a converged state |
| [`run_matched_sweep_BILLOW.py`](run_matched_sweep_BILLOW.py) | Depower and steering chains at the depower attributed from flight, one chain per target apparent speed; resumable |

## Figures and pages

| Script | What it does |
|--------|--------------|
| [`plot_billow_geometry.py`](plot_billow_geometry.py) | Converged shape, canopy shaded by wrinkling regime. Also `rebuild`, the model rebuilt from the YAML |
| [`plot_billow_depower_chains.py`](plot_billow_depower_chains.py) | How the trimmed shape moves with depower and apparent speed |
| [`plot_billow_strut_bending.py`](plot_billow_strut_bending.py) | Strut bending and span change across configurations |
| [`plot_chordwise_moment_matching.py`](plot_chordwise_moment_matching.py) | The aero→structure load map with and without conserving the panel pitching moment |
| [`plot_matched_sweep_BILLOW.py`](plot_matched_sweep_BILLOW.py) | Shapes and curves of the matched sweep against flight |
| [`export_billow_viewer.py`](export_billow_viewer.py) | Interactive viewer of several converged states ([`billow_viewer_template.html`](billow_viewer_template.html)) |
| [`export_matched_sweep_data.py`](export_matched_sweep_data.py) → [`build_matched_sweep_page.py`](build_matched_sweep_page.py) | One JSON payload of every matched-sweep row, then the self-contained interactive page from [`matched_sweep_page_template.html`](matched_sweep_page_template.html) and [`matched_sweep_page_findings.json`](matched_sweep_page_findings.json) |

## Checks

| Script | Question it answers |
|--------|---------------------|
| [`check_symmetric_equilibrium.py`](check_symmetric_equilibrium.py) | Is the unsteered kite's asymmetry a solver artefact or a genuine bifurcation? |
| [`check_structural_symmetry.py`](check_structural_symmetry.py) | Does the structure alone stay symmetric under an exactly symmetric load? |
| [`check_mirror_asymmetry.py`](check_mirror_asymmetry.py) | Left–right asymmetry of a converged shape, rigid rotation vs shape |
| [`check_load_transfer.py`](check_load_transfer.py) | Is the aero-to-structure load transfer conservative? |
| [`run_ab_canopy_pattern.py`](run_ab_canopy_pattern.py) | Does the canopy triangulation keep the unsteered kite symmetric? |
