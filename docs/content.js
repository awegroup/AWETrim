// ---------------------------------------------------------------------------
// Shared links (papers and repositories) reused across several blocks.
// Edit a URL here once and it updates everywhere it is referenced.
// ---------------------------------------------------------------------------
const PAPER_ROM = {
  label: "Paper · Translational dynamics (WES 2026)",
  url: "https://doi.org/10.5194/wes-11-1097-2026"
};
const PAPER_EKF = {
  label: "Paper · Kite as a sensor (WES 2025)",
  url: "https://doi.org/10.5194/wes-10-2161-2025"
};
const PAPER_AERO = {
  label: "Paper · Fast aero-structural model (Energies 2023)",
  url: "https://doi.org/10.3390/en16073061"
};
const PAPER_VSM = {
  label: "Paper · Computational aerodynamics for soft-wing kite design (WES 2026)",
  url: "https://doi.org/10.5194/wes-2026-46"
};
const PAPER_OPT = {
  label: "Paper · Optimal reel-out trajectories (Torque 2026)",
  url: "https://www.researchgate.net/publication/403912785_Optimal_Reel-Out_Trajectories_for_Soft_Kites_under_Varying_Wind_Conditions"
};

const REPO_AWETRIM = { label: "AWETrim repository", url: "https://github.com/awegroup/AWETrim" };
const REPO_VSM = { label: "Vortex Step Method", url: "https://github.com/awegroup/Vortex-Step-Method" };
const REPO_PSS = { label: "Particle System Simulator", url: "https://github.com/awegroup/Particle_System_Simulator" };
const REPO_FEM = { label: "kite_fem (FEM structure)", url: "https://github.com/awegroup/kite_fem" };
const REPO_EKF = { label: "EKF-AWE repository", url: "https://github.com/ocayon/EKF-AWE" };
const REPO_KITE = { label: "TU Delft LEI-V3 kite", url: "https://github.com/awegroup/TUDELFT_V3_KITE" };
const REPO_ML = { label: "LEI airfoil ML models (Zenodo)", url: "https://doi.org/10.5281/zenodo.16925759" };

// Open flight-data sets (newest first).
const DATA_20251009 = { label: "Dataset · Flight test 9 Oct 2025", url: "https://github.com/awegroup/Flightdata09102025" };
const DATA_20240605 = { label: "Dataset · Flight test 5 Jun 2024", url: "https://github.com/awegroup/Flightdata05062024" };
const DATA_20231127 = { label: "Dataset · Flight test 27 Nov 2023", url: "https://github.com/awegroup/Flightdata27112023" };
const DATA_20230512 = { label: "Dataset · Flight test 12 May 2023", url: "https://github.com/awegroup/Flightdata12052023" };
const DATA_20191008 = { label: "Dataset · Flight test 8 Oct 2019", url: "https://github.com/awegroup/Flightdata08102019" };

const CONTENT = {
  "awetrim": {
    title: "AWETrim",
    text: "AWETrim is a Python library for the modelling, trim analysis, aerostructural simulation and trajectory optimisation of soft-kite Airborne Wind Energy Systems (AWES). It ties external solvers — a vortex aerodynamic method (VSM), a structural model (PSS particle system or kite_fem FEM) and a flight-data Kalman filter (EKF-AWE) — to a CasADi-based system model in one quasi-steady, multi-fidelity workflow.",
    bullets: [
      "CasADi symbolic system model (kite + tether + winch + wind)",
      "Couples VSM aerodynamics with a PSS/FEM structure, reduced to a fast ROM",
      "Fast enough for power-cycle simulation and path optimisation"
    ],
    image: "assets/computational_framework.png",
    caption: "The AWETrim computational framework: inputs, experimental reconstruction, the multi-fidelity core, and outputs.",
    links: [REPO_AWETRIM, PAPER_ROM, PAPER_AERO, PAPER_EKF]
  },
  "experimental-flight-data": {
    title: "Experimental Flight Data",
    text: "Measured flight-test data is the starting point for reconstruction. For wind and state reconstruction with the EKF, at least the kite position, velocity, tether force and tether length must be provided. The reference kite is the TU Delft V3; the open data sets include two Kitepower V9 flights with lidar wind measurements, and the most recent of the V3 flights also has lidar.",
    bullets: [
      "Minimum EKF inputs: position, velocity, tether force, tether length",
      "Reference system: TU Delft V3 kite",
      "Two Kitepower V9 flights with lidar; latest V3 flight also has lidar"
    ],
    image: "img/flight-setup.JPG",
    caption: "Pre-flight ground setup: the kite, bridle lines and control unit laid out before a flight test.",
    links: [REPO_KITE, DATA_20251009, DATA_20240605, DATA_20231127, DATA_20230512, DATA_20191008]
  },
  "ekf-awe": {
    title: "EKF-AWE Experimental Reconstruction",
    text: "EKF-AWE processes flight logs with an Extended Kalman Filter to estimate kite states, aerodynamic coefficients and wind velocity. In AWETrim it is the bridge between real flight data and model validation or tuning, wrapped by the experimental/ module with the data layout used in data/.",
    bullets: ["State reconstruction from noisy logs", "Wind-vector estimation", "In-flight aerodynamic coefficient identification"],
    image: "img/placeholder.svg",
    caption: "Add a reconstructed trajectory, wind-speed estimate or EKF validation plot.",
    links: [REPO_EKF, PAPER_EKF]
  },
  "wind-state-estimation": {
    title: "Wind and State Estimation",
    text: "The reconstructed states and wind estimates describe the real flight conditions and let experimental behaviour be compared against model predictions. They feed model validation and tuning, and can drive the wind models used in simulation.",
    bullets: ["Estimated position and velocity", "Reconstructed wind vector", "Inputs for validation and tuning"],
    image: "img/placeholder.svg",
    caption: "Add a time series of estimated wind or kite states.",
    links: [PAPER_EKF, REPO_EKF]
  },
  "system-kite": {
    title: "System / Kite Characteristics",
    text: "The system definition holds the geometry and hardware properties of the kite, tether, KCU and winch. The examples use the TU Delft LEI-V3 leading-edge inflatable kite, described by system.yaml, aero_geometry.yaml and struc_geometry.yaml under data/LEI-V3-KITE/.",
    bullets: ["Mass, inertia and geometry", "VSM aero and PSS structural configuration files", "Tether, KCU and winch parameters"],
    image: "img/placeholder.svg",
    caption: "Add a kite geometry rendering or a parameter table.",
    links: [REPO_KITE, REPO_AWETRIM]
  },
  "environmental-conditions": {
    title: "Environmental Conditions",
    text: "Environmental inputs define the wind field and atmosphere used by the simulations. AWETrim ships uniform, logarithmic-shear and tabulated wind models, and can also use wind profiles reconstructed from flight data.",
    bullets: ["Uniform wind", "Logarithmic shear", "Tabulated or reconstructed inflow"],
    image: "img/placeholder.svg",
    caption: "Add a wind profile or turbulence plot.",
    links: [PAPER_EKF]
  },
  "operational-constraints": {
    title: "Operational Constraints",
    text: "Operational limits define the feasible flight envelope during simulation and optimisation: tether-force limits, reel-speed bounds, steering and depower limits, and path constraints. Default bounds live in utils/defaults.py (DEFAULT_OPTI_LIMITS).",
    bullets: ["Tether force and reel-speed limits", "Steering and depower control bounds", "Flight-envelope and path constraints"],
    image: "img/placeholder.svg",
    caption: "Add a constraint envelope or optimisation-bound figure.",
    links: [PAPER_OPT]
  },
  "shared-kinematics": {
    title: "Shared Kinematics",
    text: "The kinematic layer provides the common course/wind reference frame and state definitions so every fidelity level uses consistent motion variables. The kite is modelled as a point mass in a course-aligned spherical frame, following the translational-dynamics reduced-order model.",
    bullets: ["Course-frame (course-aligned spherical) kinematics", "Shared reference-frame transforms", "Consistent state definitions across fidelities"],
    image: "img/placeholder.svg",
    caption: "Add a coordinate-frame diagram.",
    links: [PAPER_ROM, REPO_AWETRIM]
  },
  "aero-structural": {
    title: "Aero-Structural Kite Model",
    text: "The high-fidelity model couples VSM aerodynamics with a flexible structural model. An Aitken-relaxed fixed-point loop iterates the aerodynamic loads against the deformed wing shape until the nodal forces converge, giving the loaded geometry and force coefficients across flight conditions — the fast aero-structural model of an LEI kite. Two interchangeable structural solvers are supported: the Particle System Simulator (PSS, particle-spring) and a finite-element model (kite_fem).",
    bullets: ["VSM aerodynamic loads (enhanced lifting line)", "Structural solver: PSS particle-spring or kite_fem FEM", "Aitken-relaxed fixed-point coupling to convergence"],
    image: "img/placeholder.svg",
    caption: "Add a loaded wing shape or pressure / load distribution plot.",
    links: [PAPER_AERO, PAPER_VSM, REPO_VSM, REPO_ML, REPO_PSS, REPO_FEM]
  },
  "model-reduction": {
    title: "Model Reduction & Aero Identification",
    text: "The reduced-order model is not assumed — it is identified from the high-fidelity aero-structural model. AWETrim sweeps the coupled VSM–structural model across flight conditions and control inputs, then fits compact quasi-steady aerodynamic coefficient relations to those sweep results. This model-reduction / system-identification step is what turns the expensive aero-structural model (left) into the fast ROM (right) used for trajectory simulation and optimisation.",
    bullets: [
      "Aerostructural sweeps over angle of attack, sideslip and control inputs",
      "Fits quasi-steady CL / CD / CS coefficient relations (rom_config.yaml)",
      "Bridges the high-fidelity model and the fast CasADi ROM"
    ],
    image: "img/placeholder.svg",
    caption: "Add a coefficient-fit or aerostructural-vs-ROM comparison plot.",
    links: [PAPER_ROM, REPO_AWETRIM]
  },
  "rom": {
    title: "Reduced-Order Kite Model",
    text: "The reduced-order model is fitted from aerostructural sweep results. It gives quasi-steady aerodynamic coefficients as functions of flight variables and control inputs (rom_config.yaml), enabling efficient trajectory simulation and optimisation inside the CasADi system model.",
    bullets: ["Coefficient fitting from aerostructural sweeps", "CasADi quasi-steady SystemModel", "Fast trajectory simulation and optimisation"],
    image: "img/placeholder.svg",
    caption: "Add fitted CL/CD/CS surfaces or ROM validation plots.",
    links: [PAPER_ROM, REPO_AWETRIM]
  },
  "tether-models": {
    title: "Tether Models",
    text: "Tether models represent the force and drag contribution of the tether and its coupling to the kite dynamics. They follow the course-frame formulation of the translational-dynamics model and are part of the CasADi system model.",
    bullets: ["Tether tension at the ground station", "Distributed tether drag", "Coupling to point-mass kite dynamics"],
    image: "img/placeholder.svg",
    caption: "Add a tether-force or tether-drag illustration.",
    links: [PAPER_ROM]
  },
  "winch-models": {
    title: "Winch Models",
    text: "Winch models describe reel-in and reel-out operation, coupling the flight trajectory to ground-station power production and to operational limits such as maximum reel speed and tether force.",
    bullets: ["Reel-in / reel-out speed", "Tether-length evolution", "Ground-station power-cycle coupling"],
    image: "img/placeholder.svg",
    caption: "Add a reel-out speed or power-cycle plot.",
    links: [PAPER_OPT]
  },
  "wind-models": {
    title: "Wind Models",
    text: "Wind models provide the inflow used for simulation, validation and optimisation. AWETrim can use idealised uniform or logarithmic profiles, tabulated fields, or wind estimates reconstructed from experimental data.",
    bullets: ["Uniform profiles", "Logarithmic-shear profiles", "Tabulated / reconstructed wind fields"],
    image: "img/placeholder.svg",
    caption: "Add a wind field or vertical wind-profile image.",
    links: [PAPER_EKF]
  },
  "trajectory-parametrization": {
    title: "Trajectory Parametrization",
    text: "Trajectory parametrisation defines path patterns — B-spline curves, uploops, downloops and helices. Their parameters become the optimisation variables for power-cycle analysis, expressed in the shared course frame.",
    bullets: ["B-spline path patterns", "Uploop, downloop and helix trajectories", "Path and control parameters as optimisation variables"],
    image: "img/placeholder.svg",
    caption: "Add a 3D trajectory or path-parameter figure.",
    links: [PAPER_ROM, PAPER_OPT]
  },
  "operational-optimization": {
    title: "Operational Optimization",
    text: "Public ROM scripts simulate and optimise full pumping cycles. run_cycle_simulation.py stitches a reel-out production loop, a reel-in phase and the transition into one CycleSimple, and with --optimize searches the path and control parameters that maximise cycle power (CasADi Opti / IPOPT) subject to the operational limits. Reel-out patterns — downloop, uploop and helix — and a standalone reel-in optimisation are available as separate entry points.",
    bullets: [
      "Full pumping cycle: reel-out → reel-in → transition (run_cycle_simulation.py)",
      "Reel-out patterns (downloop / uploop / helix) and standalone reel-in",
      "Cycle-power maximisation over path & control parameters (CasADi Opti / IPOPT)"
    ],
    image: "img/pumping-cycle-trajectory.png",
    caption: "Simulated LEI-V3 pumping cycle: downloop reel-out (blue), reel-in (orange) and transition (green), produced by run_cycle_simulation.py.",
    links: [PAPER_OPT, REPO_AWETRIM]
  },
  "performance-assessment": {
    title: "Performance Assessment",
    text: "The same cycle and pattern scripts report the per-phase energy balance and net cycle power, and let you study how power and loads vary with wind speed and configuration across the wind window. The ROM validators close the loop against measurements: validate_quasi_steady_state_v3.py compares the quasi-steady force balance to reconstructed flight data, and validate_spline_v3.py fits B-spline patterns to measured trajectories and replays them with the ROM.",
    bullets: [
      "Per-phase energy balance and net cycle power",
      "ROM validation vs flight data (validate_quasi_steady_state_v3.py, validate_spline_v3.py)",
      "Sensitivity of power and loads to wind and configuration"
    ],
    image: "img/cycle-power-breakdown.png",
    caption: "Pumping-cycle energy balance and net power for the LEI-V3 (10 m/s at 100 m), from the ROM cycle simulation.",
    links: [PAPER_OPT, PAPER_ROM, REPO_AWETRIM]
  },
  "design-analysis": {
    title: "Design and Model Analysis",
    text: "Public aerodynamics scripts turn the framework into a design tool. parametric_shapes/generate_shape_variations.py morphs the wing planform — aspect ratio, anhedral, taper, twist — and re-evaluates each variant with VSM, while optimize_lei_airfoil.py tunes the LEI airfoil with the ML regression model. compute_stability_derivatives.py finite-differences the VSM trim to obtain the aerodynamic stability derivatives, then extracts and animates the longitudinal and lateral flight-dynamic eigenmodes.",
    bullets: [
      "Parametric wing-planform & LEI-airfoil studies (scripts/aerodynamics/parametric_shapes/)",
      "Aerodynamic stability derivatives and animated flight-dynamic eigenmodes",
      "Configuration comparison and model validation against flight data"
    ],
    image: "img/3d-wing-design.png",
    caption: "Parametric wing-planform variations (aspect ratio and anhedral) from generate_shape_variations.py.",
    links: [PAPER_AERO, PAPER_VSM, REPO_AWETRIM]
  }
};
