# Winch drivetrain losses

`drivetrain_physics.pdf` explains the physics of each mechanical element between the tether and the motor shaft of the TU Delft GS1 ground station (2010-2011): the three pulleys, the force sensor on P3, the drum, the belt, the gearbox and the motor's bearings and fan. It gives a bottom-up loss estimate on optimised LEI-V3 pumping cycles, describes the cold-day tow test behind the load-independent friction (122 N + 30.7 N s/m), and uses the GS1 energy balance from Fechner's AWEC 2011 talk to show where a cycle's energy goes and which cheap fixes would recover most.

- Tables, macros and all `drivetrain_*` figures: `python scripts/identification/estimate_drivetrain_losses.py` writes them to `generated/` and `figures/`. `figures/ground_station_layout.png` is a drawing of GS1.
- Friction-aware optimisation: `run_uploop_eight_opti.py --winch-friction` (in `scripts/reduced-order-model/optimization/studies/`).
- Rebuild the PDF: `latexmk -pdf drivetrain_physics.tex` in this folder.
