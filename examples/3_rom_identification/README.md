# Step 3: ROM identification

Identify the reduced-order model (ROM) of the kite from aerostructural
simulations only, no flight data. The ROM is what makes trajectory
optimisation (step 4) fast: a quasi-steady kite whose aerodynamics are a few
fitted functions instead of a VSM solve.

Two scripts, run in order. Both read the kite's
[`identification.yaml`](../../data/LEI-V3-KITE/identification.yaml): its
`sweep:` block sets the states to solve, its `fit:` block the kite constants
(reference area, steering scale, depower band) the ROM is written with.

The sweep runs at either structural fidelity of step 2 (`fidelity:` in the
file, or `--fidelity` on both scripts):

- `billow`: the full model, with inflatable tube beams, a wrinkling membrane
  canopy and the bridle, built from the kite's awesIO `structure.yml`.
- `wireframe`: the bridle line system on the wireframe reduced from the same
  `structure.yml` (written per case as `structure_wireframe.yml`), the model
  the shipped LEI-V3 ROM was identified on. Its bridle runs at a fixed 9x
  modulus, tapes included, with no in-loop stiffening, so every state in the
  sweep is the same kite.

Each fidelity has its own block in the file (geometry, tape range, fit band),
because the tape coordinate is the model's own: the full model flies a given
tape length at a higher angle of attack than the wireframe. The tape ranges
and the fit band are modelling choices: for a new kite, set them where its
model flies the loads of interest (the LEI-V3 values are the current
estimates, commented in the file).

| Script | What it does | Writes |
|--------|--------------|--------|
| [`run_identification_sweep.py`](run_identification_sweep.py) | Solves coupled VSM + Billow states at the centre of the wind window, gravity off, along continuation chains in depower and steering. Each chain warm-starts every step from the previous one. On every converged state it also solves a frozen-shape angle-of-attack polar. About a minute per state. Resumable: a wireframe chain goes on after its last saved state, a Billow chain starts over. A state that does not converge ends its chain. Runs on disjoint `--chains` can go in parallel on one output folder. With `start_from_converged` (wireframe) every chain starts from a converged, unsteered state at its wind instead of the unloaded build (solve them first with `--base-only` before parallel runs). | `results/<kite>/identification/<fidelity>/sweep/` (`trims.csv`, `polars.csv`, one `sim_output.h5` per state) |
| [`identify_rom_aerostructural.py`](identify_rom_aerostructural.py) | Fits every ROM parameter on that dataset: the bridle pitch `theta_b(u_p)` from the coupled trims, the lift and drag polynomials in `(alpha, u_p, u_s, v_a)` from the trims and polars (terms chosen on cross-validation, a smooth stall switch located on the data), and the steering roll gain. | `results/<kite>/identification/<fidelity>/rom_aerostructural/` (`rom_config_aerostructural.yaml`, the tidy samples, a report) |

```bash
python examples/3_rom_identification/run_identification_sweep.py --list   # the chains
python examples/3_rom_identification/run_identification_sweep.py
python examples/3_rom_identification/identify_rom_aerostructural.py
python examples/3_rom_identification/run_identification_sweep.py --fidelity wireframe
python examples/3_rom_identification/identify_rom_aerostructural.py --fidelity wireframe
```

`--chains <name> ...` solves only some chains. `--no-write` fits and reports
without writing the ROM file, and `--out <path>` writes it elsewhere, e.g.
into `data/<kite>/`.

What goes in and comes out:

- **The dataset** holds dimensional quantities only: forces, apparent wind,
  tape lengths, masses. The reference area and the control scaling are choices
  made in the fit. The format is documented in
  [`awetrim.identification.aerostructural_dataset`](../../src/awetrim/identification/aerostructural_dataset.py).
- **Why polars**: along a trimmed chain the angle of attack follows the
  depower almost one to one, so the trims alone cannot separate the polar
  slope from the depower effect. The polar turns the inflow about each frozen
  deformed shape; its zero step re-solves the trim itself.
- **The ROM file** is in the format `awetrim.system.kite` reads. Name it in
  the kite's `system.yaml` (`models.reduced_order.aerodynamics`) to fly it in
  step 4.
- The drag coefficient is the wing's, bridle-line drag included. The KCU drag
  is not in it: the ROM adds it from the system file's KCU hardware
  (`awetrim.aerodynamics.kcu_drag`), so a KCU change needs no
  re-identification.

The LEI-V3 ROM shipped in `data/LEI-V3-KITE/rom_config_aerostructural.yaml`
was identified the same way with the wireframe model, on a larger campaign
(243 states, run at target apparent speeds). The `wireframe` sweep is the
smaller version of it; the `billow` sweep gives a different ROM.

With flight data, [step 5](../5_flight_data/) corrects the ROM's depower
pitch and drag on a flown cycle.
