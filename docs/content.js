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
    text: "Raw measured data from flight tests is the starting point for reconstruction. Typical signals include position, velocity, tether force, reel-out speed and onboard sensor measurements. The reference data set in this repository comes from the TU Delft LEI-V3 kite.",
    bullets: ["Raw CSV logs and pre-processed HDF5 files", "Position, velocity, tether force, reel-out speed", "Onboard IMU / sensor measurements"],
    image: "img/placeholder.svg",
    caption: "Add a photo of the experimental setup or a raw-data time series.",
    links: [REPO_KITE]
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
    links: [PAPER_AERO, REPO_VSM, REPO_PSS, REPO_FEM]
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
    text: "Operational optimisation uses the reduced-order model and CasADi Opti (IPOPT) to search for the path and control parameters that maximise performance while satisfying constraints — for example optimal reel-out trajectories for soft kites under varying wind conditions.",
    bullets: ["CasADi Opti / IPOPT NLP solver", "Path-parameter and control optimisation", "Power-cycle and envelope constraints"],
    image: "img/placeholder.svg",
    caption: "Add an optimisation convergence or optimal-trajectory plot.",
    links: [PAPER_OPT, REPO_AWETRIM]
  },
  "performance-assessment": {
    title: "Performance Assessment",
    text: "Performance assessment evaluates power production, aerodynamic efficiency and tether loads, and how sensitive these are to model choices and operating conditions across the whole wind window.",
    bullets: ["Cycle and instantaneous power", "Loads and aerodynamic efficiency", "Sensitivity to wind and configuration"],
    image: "img/placeholder.svg",
    caption: "Add power, load, or efficiency plots.",
    links: [PAPER_OPT]
  },
  "design-analysis": {
    title: "Design and Model Analysis",
    text: "Design and model analysis uses the framework to compare configurations, investigate sensitivities, and evaluate how modelling assumptions affect kite performance and stability — including the flight-dynamic modes derived from the aerodynamic stability derivatives.",
    bullets: ["Configuration comparison", "Model validation against flight data", "Stability and eigen-mode analysis"],
    image: "img/placeholder.svg",
    caption: "Add a design comparison or stability plot.",
    links: [PAPER_AERO, REPO_AWETRIM]
  }
};
