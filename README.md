# AWETrim

[![Interactive framework](https://img.shields.io/badge/website-AWETrim%20interactive%20framework-2563eb?style=flat-square&logo=githubpages&logoColor=white)](https://awegroup.github.io/AWETrim/)

AWETrim is a Python library for the modelling, trim analysis, aerostructural simulation, and trajectory optimisation of soft-kite Airborne Wind Energy Systems (AWES). It couples a vortex aerodynamic method (VSM), a flexible structural model (PSS or kite_fem) and a flight-data Kalman filter (EKF-AWE) to a CasADi-based system model: the aerostructural coupling produces loaded wing shapes and force coefficients, these are reduced to a fast quasi-steady reduced-order model (ROM), and the ROM drives power-cycle simulation and path optimisation.

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

The external solvers (VSM, PSS, EKF-AWE) are installed automatically from GitHub by the `pip install` step — no manual cloning required.

---

## Quickstart

All scripts run from the **project root** and use the reference **TU Delft LEI-V3** kite in `data/LEI-V3-KITE/` (kite reference data: [awegroup/TUDELFT_V3_KITE](https://github.com/awegroup/TUDELFT_V3_KITE)):

```bash
# One VSM quasi-steady trim state
python scripts/aerodynamics/solve_single_state.py

# One coupled VSM ↔ structure (PSS) deformed-shape simulation
python scripts/aerostructural/run_simulation_PSM.py

# Simulate a full pumping cycle with the ROM (add --optimize to maximise cycle power)
python scripts/reduced-order-model/optimization/cycle/run_cycle_simulation.py --plot
```

Runnable examples are organised by domain under `scripts/`, each folder with its own README — start at [`scripts/README.md`](scripts/README.md).

---

## Repository layout

```
src/awetrim/
  system/          CasADi system model: SystemModel, Kite, Wind, Winch, Tether variants
  aerodynamics/    VSM quasi-steady trim + parametric wing/airfoil geometry
  aerostructural/  VSM ↔ structure coupling — pss/ (PSS) and fem/ (kite_fem)
  kinematics/      Course-frame kinematics and parametrised path patterns
  timeseries/      Phase / Reel-in / Cycle simulation
  environment/     Wind models (uniform, logarithmic, tabulated)
  experimental/    EKF flight-data pipeline (EKF-AWE wrapper)
  identification/  ROM aero-parameter and turn-rate-law identification
  plotting/        Shared kite + tether plotting helpers
  utils/           Reference frames, fitting, default bounds and limits

data/<kite>/       Per-kite configs (system, geometry, aerostructural, ROM, EKF)
scripts/           Runnable examples by domain (see scripts/README.md)
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
| [Particle_System_Simulator](https://github.com/awegroup/Particle_System_Simulator) | PSS flexible structural solver |
| [kite_fem](https://github.com/awegroup/kite_fem) | FEM structural solver (alternative to PSS in the coupling) |
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
