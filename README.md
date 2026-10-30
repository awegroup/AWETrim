<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/awetrim-lockup-dark.svg">
    <img src="docs/logo/awetrim-lockup.svg" alt="AWETrim" height="96">
  </picture>
</h1>

[![Interactive framework](https://img.shields.io/badge/website-AWETrim%20interactive%20framework-2563eb?style=flat-square&logo=githubpages&logoColor=white)](https://awegroup.github.io/AWETrim/)

AWETrim is a Python library that takes a soft kite for Airborne Wind Energy from its design to optimised flight trajectories. The workflow:

1. **Aerodynamics**: describe the kite (geometry, structure, masses, tether, winch) in one folder, then trim the rigid wing with VSM and check its static stability.
2. **Aerostructural deformation**: couple a vortex aerodynamic method ([VSM](https://github.com/awegroup/Vortex-Step-Method)) with a minimum-energy structural solver ([Billow](https://github.com/awegroup/Billow)) to get the deformed shape the kite flies in.
3. **ROM identification**: reduce those simulations to a fast quasi-steady reduced-order model (ROM).
4. **Trajectory optimisation**: simulate and optimise reel-out patterns and whole pumping cycles with the ROM, in a CasADi-based system model.
5. **Flight data** (optional): with a flight reconstructed by [EKF-AWE](https://github.com/awegroup/EKF-AWE), correct the ROM and validate it against the measurements.

> **Explore the [interactive framework diagram](https://awegroup.github.io/AWETrim/)** — click any block (inputs, experimental reconstruction, the multi-fidelity core, outputs & applications) to see what it does, the figures it produces, and the code and papers behind it. The site also lists the related repositories, open flight-data sets and publications.

---

## Installation

```bash
git clone https://github.com/awegroup/AWETrim.git
cd AWETrim
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -e .[dev]
```

The external solvers (VSM, Billow, EKF-AWE) are installed automatically from GitHub by the `pip install` step — no manual cloning required.

To develop against a local checkout of the structural solver, install it in
editable mode afterwards:

```bash
git clone https://github.com/awegroup/Billow.git ../Billow
pip install -e ../Billow
```

---

## Quickstart

All examples run from the **project root** and use the reference **TU Delft LEI-V3** kite in `data/LEI-V3-KITE/` (kite reference data: [awegroup/TUDELFT_V3_KITE](https://github.com/awegroup/TUDELFT_V3_KITE)):

```bash
# One VSM quasi-steady trim state
python examples/1_aerodynamics/solve_single_state.py

# One coupled VSM ↔ structure deformed-shape simulation (wireframe fidelity)
python examples/2_aerostructural/run_simulation_PSM.py

# The same, at full fidelity: inflatable tube beams + wrinkling membrane canopy
python examples/2_aerostructural/run_simulation_BILLOW.py

# Optimise a full pumping cycle with the reduced-order model (seed first, then optimise)
python examples/4_trajectory_optimization/cycle/fit_periodic_cycle_config.py --kite LEI-V3-KITE --auto
python examples/4_trajectory_optimization/cycle/run_full_cycle_opti.py --kite LEI-V3-KITE
```

[`examples/`](examples/README.md) has one folder per workflow step, `1_aerodynamics/` to `5_flight_data/`, each with its own README.

---

## Repository layout

```
src/awetrim/
  system/          CasADi system model: SystemModel, Kite, Wind, Winch, Tether variants
  aerodynamics/    VSM quasi-steady trim + parametric wing/airfoil geometry
  aerostructural/  VSM ↔ Billow coupling — coupled/ (the driver, shared),
                   wireframe/ and billow/ (one adapter per Billow fidelity)
  kinematics/      Course-frame kinematics and parametrised path patterns
  timeseries/      Phase / Reel-in / Cycle simulation
  environment/     Wind models (uniform, logarithmic, tabulated)
  flight_data/     Reader for flights reconstructed by EKF-AWE (workflow step 5)
  identification/  ROM aero-parameter and turn-rate-law identification
  plotting/        Shared kite + tether plotting helpers
  utils/           Reference frames, default bounds and limits

data/<kite>/       Per-kite configs (system, geometry, aerostructural, ROM, EKF)
examples/          Runnable examples, one folder per workflow step (see examples/README.md)
results/<kite>/    Solver outputs (aerostructural/, ekf/)
```

Several modules carry an `AGENTS.md` documenting their internals and conventions; see the top-level [`AGENTS.md`](AGENTS.md) for the full module map and the per-kite data layout.

---

## Tests

```bash
pytest                          # all tests
pytest tests/aerostructural/    # one module
pytest -m "not slow"            # skip slow integration tests
pytest --cov=src --cov-report=term-missing
```

---

## External solvers & key dependencies

AWETrim delegates physics-heavy computations to purpose-built packages (installed automatically):

| Package | Role |
|---------|------|
| [Vortex-Step-Method](https://github.com/awegroup/Vortex-Step-Method) | VSM aerodynamic solver (enhanced lifting line) |
| [Billow](https://github.com/awegroup/Billow) | Minimum-energy structural solver — cables and pulleys (wireframe fidelity), plus inflatable tube beams and a wrinkling membrane canopy (full fidelity) |
| [EKF-AWE](https://github.com/ocayon/EKF-AWE) | Extended Kalman Filter for flight-data analysis |
| [CasADi](https://web.casadi.org) | Symbolic computation and NLP solving (system model, optimisation) |
| NumPy / SciPy · pandas / h5py | Numerical routines and data I/O |

---

## Citing AWETrim

The methods are described across several publications — the full list (with datasets and related code) is on the [website](https://awegroup.github.io/AWETrim/). Core references:

- Cayon, O., van Deursen, V., & Schmehl, R. (2026). **Translational dynamics of bridled kites: a reduced-order model in the course reference frame.** *Wind Energy Science.* [doi:10.5194/wes-11-1097-2026](https://doi.org/10.5194/wes-11-1097-2026)
- Cayon, O., Watson, S., & Schmehl, R. (2025). **Kite as a sensor: wind and state estimation in tethered flying systems.** *Wind Energy Science.* [doi:10.5194/wes-10-2161-2025](https://doi.org/10.5194/wes-10-2161-2025)
- Cayon, O., Gaunaa, M., & Schmehl, R. (2023). **Fast aero-structural model of a leading-edge inflatable kite.** *Energies.* [doi:10.3390/en16073061](https://doi.org/10.3390/en16073061)

The distributed-mass tether model (`WilliamsTether`) follows Williams, P. (2017), *J. Guid. Control Dyn.* 40(7), [doi:10.2514/1.G002354](https://doi.org/10.2514/1.G002354).

---

## Acknowledgements

<p>
  <a href="https://meridional.eu/"><img src="docs/img/Meridional_logo.png" alt="MERIDIONAL project logo" height="38"></a>
  &nbsp;&nbsp;
  <img src="docs/img/eu-funded.svg" alt="Funded by the European Union" height="38">
</p>

AWETrim is developed within the [MERIDIONAL](https://meridional.eu/) project, which has received funding from the European Union's Horizon Europe research and innovation programme under Grant Agreement No. 101084216. Views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

---

## License

Technische Universiteit Delft hereby disclaims all copyright interest in the program “AWETrim” (a Python library for the design and optimisation of soft-kite Airborne Wind Energy Systems) written by the Author(s).

Prof. H.G.C. (Henri) Werij, Dean of Aerospace Engineering

Copyright © 2023–2026 Oriol Cayon, Delft University of Technology.

Licensed under the **Apache License, Version 2.0 (Apache-2.0)**.

The Apache License 2.0 is a permissive open-source licence approved by the Open Source Initiative. In short:

- You may use, study, modify, and redistribute this package, including for commercial purposes and in proprietary derivative works.
- If you redistribute the package or a derivative work, you must retain the copyright, patent, trademark, and attribution notices, include a copy of the licence, and state any significant changes you made. If a `NOTICE` file is present, its attribution notices must be preserved.
- The licence includes an express grant of patent rights from contributors and provides the software "as is", without warranty.

We encourage users who improve this package to contribute their changes back to the upstream repository.

See the [`LICENSE`](LICENSE) file for the full, legally binding text.

### Third-party code

Portions of the aerostructural coupling (under `src/awetrim/aerostructural/`) were
adapted from [ASKITE](https://github.com/awegroup/ASKITE), which is distributed under
the MIT License (Copyright © 2024 Jelle Poland, Patrick Roeleveld, TU Delft). The
required copyright and permission notices are reproduced in the [`NOTICE`](NOTICE) file.
