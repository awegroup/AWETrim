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
const PAPER_WILLIAMS = {
  label: "Paper · Lumped-element tether dynamics (Williams, JGCD 2017)",
  url: "https://doi.org/10.2514/1.G002354"
};

const REPO_AWETRIM = { label: "AWETrim repository", url: "https://github.com/awegroup/AWETrim" };
const REPO_VSM = { label: "Vortex Step Method", url: "https://github.com/awegroup/Vortex-Step-Method" };
const REPO_BILLOW = { label: "Billow · structural solver", url: "https://github.com/awegroup/Billow" };
const DOCS_BILLOW = { label: "Billow documentation", url: "https://awegroup.github.io/Billow/" };
const REPO_EKF = { label: "EKF-AWE repository", url: "https://github.com/ocayon/EKF-AWE" };
const REPO_KITE = { label: "TU Delft LEI-V3 kite", url: "https://github.com/awegroup/TUDELFT_V3_KITE" };
const REPO_ML = { label: "LEI airfoil ML models (Zenodo)", url: "https://doi.org/10.5281/zenodo.16925759" };
const REPO_AWESIO = { label: "awesIO standard", url: "https://github.com/awegroup/awesIO" };
const REPO_AWERA = { label: "AWERA · wind-resource analysis", url: "https://github.com/awegroup/AWERA" };

// Open flight-data sets (newest first).
const DATA_20251009 = { label: "Dataset · Flight test 9 Oct 2025", url: "https://github.com/awegroup/Flightdata09102025" };
const DATA_20240605 = { label: "Dataset · Flight test 5 Jun 2024", url: "https://github.com/awegroup/Flightdata05062024" };
const DATA_20231127 = { label: "Dataset · Flight test 27 Nov 2023", url: "https://github.com/awegroup/Flightdata27112023" };
const DATA_20230512 = { label: "Dataset · Flight test 12 May 2023", url: "https://github.com/awegroup/Flightdata12052023" };
const DATA_20191008 = { label: "Dataset · Flight test 8 Oct 2019", url: "https://github.com/awegroup/Flightdata08102019" };

const CONTENT = {
  "awetrim": {
    title: "AWETrim",
    text: "AWETrim is a Python library for the modelling, aerostructural simulation, trim analysis and trajectory optimisation of soft-kite Airborne Wind Energy Systems (AWES). It ties external solvers — a vortex aerodynamic method (VSM), a minimum-energy structural solver (Billow) and a flight-data Kalman filter (EKF-AWE) — to a CasADi-based system model in one quasi-steady, multi-fidelity workflow.",
    bullets: [
      "CasADi symbolic system model (kite + tether + winch + wind)",
      "Couples VSM aerodynamics with the Billow structural solver, reduced to a fast ROM",
      "Fast enough for power-cycle simulation and path optimisation"
    ],
    image: "img/awetrim-overview.png",
    caption: "AWETrim in one picture: VSM and Billow are coupled into the aero-structural model, whose identified aerodynamics give the reduced-order model that trajectory optimisation runs on. Flight data validates the chain and identifies the model.",
    links: [REPO_AWETRIM, REPO_BILLOW, PAPER_ROM, PAPER_AERO, PAPER_EKF]
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
    text: "EKF-AWE processes flight logs with an Extended Kalman Filter to estimate kite states, aerodynamic coefficients and wind velocity. Its output file per flight is the bridge between real flight data and model validation or tuning: AWETrim reads it with the flight_data/ module, and EKF-AWE reads the as-flown hardware files AWETrim keeps in data/.",
    bullets: ["State reconstruction from noisy logs", "Wind-vector estimation", "In-flight aerodynamic coefficient identification"],
    image: "img/kite-trajectory.png",
    caption: "Reconstructed kite flight trajectory from the EKF-AWE pipeline.",
    links: [REPO_EKF, PAPER_EKF]
  },
  "wind-state-estimation": {
    title: "Wind and State Estimation",
    text: "The reconstructed states and wind estimates describe the real flight conditions and let experimental behaviour be compared against model predictions. They feed model validation and tuning, and can drive the wind models used in simulation.",
    bullets: ["Estimated position and velocity", "Reconstructed wind vector", "Inputs for validation and tuning"],
    image: "img/identified-wind.png",
    caption: "Wind velocity identified from flight data by the EKF.",
    links: [PAPER_EKF, REPO_EKF]
  },
  "system-kite": {
    title: "System / Kite Characteristics",
    text: "The system definition holds the geometry and hardware properties of the kite, tether, KCU and winch, following the open awesIO standard schema so a single system.yaml describes the whole system in an interoperable, validated format. The examples use the TU Delft LEI-V3 leading-edge inflatable kite, described by system.yaml (awesIO format), aero_geometry.yaml and struc_geometry.yaml under data/LEI-V3-KITE/.",
    bullets: ["Defined with the awesIO standard schema (system.yaml)", "Mass, inertia and geometry", "VSM aero and structural configuration files", "Tether, KCU and winch parameters"],
    image: "img/awesio-logo.svg",
    caption: "System and kite characteristics follow the awesIO open standard.",
    links: [REPO_KITE, REPO_AWESIO, REPO_AWETRIM]
  },
  "environmental-conditions": {
    title: "Environmental Conditions",
    text: "Environmental inputs define the wind field and atmosphere used by the simulations. You choose the wind profile: a uniform or logarithmic-shear model, or a tabulated profile (height–speed samples) — which can come from flight-data reconstruction or from external wind-resource analysis. A direct coupling with AWERA, which clusters reanalysis data into representative vertical wind profiles, is planned to feed site-specific tabulated inflow.",
    bullets: ["Choose uniform, logarithmic-shear or tabulated profile", "Tabulated inflow from flight data or wind-resource analysis", "Planned coupling with AWERA for site-specific profiles"],
    image: "img/wind-profiles-input.png",
    caption: "Clustered, normalised vertical wind profiles at Bangor Erris with a power-law fit (α = 0.189) — representative inflow of the kind AWERA produces, usable directly as tabulated input.",
    links: [PAPER_EKF, REPO_AWERA]
  },
  "operational-constraints": {
    title: "Operational Constraints",
    text: "Operational constraints bound the feasible flight envelope during simulation and optimisation, and come from two sources. Hardware limits live in the system configuration (system.yaml): maximum tether force, operational tether length, and the winch drum/gearbox properties. Site- and trajectory-related limits constrain the path itself — minimum flight height above ground, elevation and azimuth bounds, and other geometric path constraints. Default optimisation-variable bounds (including reel-speed limits) live in utils/defaults.py (DEFAULT_OPTI_LIMITS).",
    bullets: ["Hardware limits in system.yaml: max tether force, tether length, winch/drum properties", "Site/trajectory limits: minimum flight height, elevation/azimuth and path bounds", "Default optimisation bounds (incl. reel speed) in utils/defaults.py (DEFAULT_OPTI_LIMITS)"],
    links: [PAPER_OPT]
  },
  "shared-kinematics": {
    title: "Shared Kinematics",
    text: "The kinematic layer defines the shared course reference frame and the state variables — position and velocity in a course-aligned spherical frame — used at every fidelity level, so the aero-structural model and the ROM describe motion in a consistent way. It provides the course-aligned spherical frame and the transforms between the course, wind and ground frames introduced by Cayon, van Deursen & Schmehl (WES 2026), independently of any mass or force model.",
    bullets: ["Course reference frame (course-aligned spherical) kinematics", "Shared reference-frame transforms", "Consistent state definitions across fidelities"],
    image: "img/shared-kinematics.png",
    caption: "The course reference frame: the kite position on the sphere is set by radial distance, elevation β and azimuth φ, with the course-aligned unit vectors (e_χ, e_β, e_φ) oriented by the flight-path direction χ.",
    links: [PAPER_ROM, REPO_AWETRIM]
  },
  "aero-structural": {
    title: "Aero-Structural Kite Model",
    text: "The high-fidelity model couples two external solvers: aerodynamic loads and the deformed wing shape are iterated against each other until consistent, giving the loaded geometry and force coefficients across flight conditions. Aerodynamics use the Vortex Step Method (VSM); the structure is solved by Billow, which minimises the total potential energy. AWETrim owns the coupling — the load transfer, the trim, the fixed-point loop — and neither solver knows about the other. Which structural fidelity is used is not a detail of accuracy: the trimmed angle of attack is where the moment balance lands, so the structure's stiffness sets it, and the same depower setting means a different flight state on each model. This is the fast aero-structural model of an LEI kite (Cayon, Gaunaa & Schmehl, Energies 2023), built on the computational-aerodynamics approach for soft-wing kite design (WES 2026).",
    bullets: ["Aerodynamics: Vortex Step Method (VSM)", "Structure: Billow, at either fidelity", "Fixed-point loop on the nodal forces, Aitken-relaxed", "Returns the loaded geometry and force coefficients", "Actuation sweeps: depower and steering, matched against flight"],
    image: "img/matched-sweep-curves.png",
    caption: "The coupled model against the 2019 flight. Left: tether force versus apparent speed, each model drawn at the depower where it matches the flown force — the full model at u_dp 0.315, the wireframe at 0.242 — with the near-straight flight medians they are matched to. Right: the turn law under steering, against the flown cloud. Solved at the centre of the wind window, one chain per target apparent speed.",
    links: [PAPER_AERO, PAPER_VSM, REPO_VSM, REPO_BILLOW, REPO_ML]
  },
  "vsm": {
    title: "VSM · Aerodynamics",
    text: "The Vortex Step Method resolves the spanwise circulation over the deformed wing and returns the panel forces the structure is loaded with. It is a separate package: AWETrim reaches it through one adapter, and the rest of the coupling is VSM-agnostic. The same solver supplies the quasi-steady trim, including the bridle-line and KCU drag that the force balance has to carry.",
    bullets: [
      "Spanwise circulation over the deformed wing, panel by panel",
      "Li/Gaunaa artificial viscosity for post-stall regularisation",
      "Anderson-accelerated circulation loop",
      "Supplies the quasi-steady trim the structure is loaded at",
      "Canopy billowing between the struts as a circular arc of fixed fabric length"
    ],
    image: "img/vsm-billowed-forces.svg",
    caption: "The LEI-V3 VSM mesh with the canopy billowed between the struts, at a converged coupled state: the total aerodynamic force on each panel (blue) and the aerodynamic force on each bridle-line segment (orange), drawn to separate scales.",
    links: [REPO_VSM, PAPER_VSM, REPO_ML]
  },
  "billow": {
    title: "Billow · Structural Solver",
    text: "Billow solves for the static equilibrium of the soft structure — a bridle of tension-only lines, an inflatable tube frame, a canopy of fabric that wrinkles rather than buckles — by minimising one total potential energy with IPOPT. That is a physics choice rather than a solver preference: a slack line and a wrinkled panel are exactly the states where a residual-form Newton solve has to fight a near-singular tangent, and posed over the relaxed (quasiconvex) energy they are simply where the minimum is. It is a standalone package that imports only NumPy and CasADi — it knows nothing about aerodynamics or any kite file format, which is what lets its element physics be validated against closed-form solutions before any of it touches a coupled run.",
    bullets: [
      "Wireframe fidelity: cables, tension-only lines, frictionless pulleys",
      "Full fidelity: adds inflatable tube beams and a wrinkling membrane canopy",
      "One mapped kernel per element type — 9k DOF solves in ~6 s, mesh-independent iteration count",
      "Validated against the Euler elastica, roll-up, Bathe & Bolourchi, and a measured hanging kite"
    ],
    image: "img/billow-displacement.svg",
    caption: "A converged coupled state of the LEI-V3 on Billow's wireframe fidelity, reduced from the full FEM geometry and flying the billowed canopy: the loaded shape coloured by the displacement from the built shape, with the rigid motion removed. The dashed lines are the built leading and trailing edges. Billow's own demonstration and validation figures live in its documentation.",
    links: [REPO_BILLOW, DOCS_BILLOW, PAPER_AERO]
  },
  "model-reduction": {
    title: "Model Reduction & Aero Identification",
    text: "The reduced-order model is not assumed — it is identified from the high-fidelity aero-structural model. AWETrim sweeps the coupled VSM–Billow model over angle of attack and control inputs, then fits compact quasi-steady aerodynamic coefficient relations to those sweep results. The same fit can be run on EKF flight-data reconstructions, so an aerostructural-identified ROM can be compared directly against an experiment-identified one. This model-reduction / system-identification step is what turns the expensive aero-structural model (above) into the fast ROM (below) used for trajectory simulation and optimisation.",
    bullets: [
      "Aerostructural (or EKF flight-data) sweeps over angle of attack, control inputs and airspeed",
      "Fits quasi-steady CL / CD / CS coefficient relations (rom_config.yaml)",
      "Bridges the high-fidelity model and the fast CasADi ROM"
    ],
    image: "img/placeholder.svg",
    caption: "Add a coefficient-fit or aerostructural-vs-ROM comparison plot.",
    links: [PAPER_ROM, REPO_AWETRIM]
  },
  "rom": {
    title: "Reduced-Order Kite Model",
    text: "The reduced-order kite model is the point-mass translational-dynamics model in the course reference frame (Cayon, van Deursen & Schmehl, WES 2026): the kite is reduced to a point mass whose motion follows from the aerodynamic, tether and gravity forces, using quasi-steady aerodynamic coefficients fitted from the aerostructural sweeps (rom_config.yaml). This is the fast CasADi model used for power-cycle simulation and trajectory optimisation.",
    bullets: ["Point-mass translational dynamics in the course reference frame", "Quasi-steady aero coefficients fitted from aerostructural sweeps (rom_config.yaml)", "Fast CasADi model for power-cycle simulation and optimisation"],
    image: "img/rom-kite.png",
    caption: "Quasi-steady force balance of the point-mass kite at two depower angles (θ_d = 0° and 10°): lift L, drag D and aerodynamic force F_a balanced against the bridle/tether force F_b.",
    links: [PAPER_ROM, REPO_AWETRIM]
  },
  "tether-models": {
    title: "Tether Models",
    text: "Tether models represent the force and drag contribution of the tether and its coupling to the kite dynamics, all within the course-frame CasADi system model. AWETrim offers three quasi-static fidelity levels: a rigid link (massless, dragless straight line, pure radial tension on the kite), a rigid lumped model that keeps the straight geometry but transfers the tether weight and a lumped cross-flow drag to the kite (the translational-dynamics formulation), and a discretised lumped-element model (Williams) whose mass, gravity and per-segment aerodynamics let the tether actually sag and bow.",
    bullets: ["Rigid link: straight, massless/dragless, radial tension only", "Rigid lumped: straight geometry, tether weight and drag lumped onto the kite", "Discretised (Williams): distributed mass and drag — sagging shape, force balance per segment"],
    image: "img/tether-models-comparison.png",
    caption: "Quasi-static tether models compared: (a) the discretised Williams tether bows several metres off the straight rigid line, and (b) that mass and drag tilt the kite-end force off the radial direction — captured closely by the cheaper rigid lumped model and ignored by the rigid link.",
    links: [PAPER_ROM, PAPER_WILLIAMS, REPO_AWETRIM]
  },
  "winch-models": {
    title: "Winch Models",
    text: "The winch couples the flight trajectory to ground-station power and to operational limits (maximum reel speed and tether force). It is modelled as an ideal actuator that follows its control command exactly, with no drivetrain dynamics. Three command types are available: a force–speed curve that sets the ground tether force as a saturated function of reeling speed (the law used in the LEI-V3 cycle configs), constant speed, and constant force. The reel-out / reel-in power follows directly as P = T·v_r.",
    bullets: ["Force–speed curve: tether force as a saturated function of reeling speed", "Constant-speed and constant-force commands", "Reel-out generation / reel-in recovery and reel-speed & force limits"],
    image: "img/winch-models-comparison.png",
    caption: "Idealised winch control laws: (a) the force–speed curve, constant-force and constant-speed commands in the reeling-speed / tether-force plane within the winch limits, and (b) the resulting mechanical power P = T·v_r — generation on reel-out, small recovery on reel-in.",
    links: [PAPER_OPT, REPO_AWETRIM]
  },
  "wind-models": {
    title: "Wind Models",
    text: "Wind models provide the inflow used for simulation, validation and optimisation. A single Wind model selects between an idealised uniform or logarithmic-shear profile and a tabulated profile interpolated from height–speed samples — the latter taking wind estimates reconstructed from experimental data or, in the planned AWERA coupling, representative profiles from wind-resource analysis.",
    bullets: ["Uniform profiles", "Logarithmic-shear profiles", "Tabulated profiles (flight data or AWERA wind-resource analysis)"],
    image: "img/wind-profiles-input.png",
    caption: "Vertical wind profiles interpolated by the tabulated model: clustered, normalised profiles at Bangor Erris with a power-law fit (α = 0.189).",
    links: [PAPER_EKF, REPO_AWERA]
  },
  "trajectory-parametrization": {
    title: "Trajectory Optimisation",
    text: "Trajectory optimisation runs on the reduced-order model. The parametrisation defines the flight path in the shared course frame using B-spline path patterns, whose control points and parameters become the optimisation variables for power-cycle analysis (Cayon, van Deursen & Schmehl, WES 2026; Cayon & Schmehl, Torque 2026). A full pumping cycle can be represented in two ways: as a single periodic B-spline closing the whole cycle, or by splitting it into a parametrised reel-out production phase — downloop, uploop or helix — followed by a simplified reel-in phase. Both representations are implemented.",
    bullets: ["B-spline path patterns in the course frame as optimisation variables", "Whole cycle as a single periodic B-spline (run_full_cycle_opti.py)", "Or split into a parametrised reel-out (downloop/uploop/helix) + simplified reel-in"],
    image: "img/three-cycle-trajectories.svg",
    caption: "Optimised LEI-V3 pumping cycles, each closed by one periodic B-spline, at the same wind and under the same limits, coloured by reel-out speed. Left: multi-lobe cycle. Middle: slanted up-loop figure eight. Right: symmetric figure eight. The dashed line is the initial guess.",
    links: [PAPER_ROM, PAPER_OPT, REPO_AWETRIM]
  },
  "flight-data-processing": {
    title: "Flight-Data Processing",
    text: "Alongside the model chain, AWETrim uses measured flights. EKF-AWE preprocesses the raw logs and runs the reconstruction; AWETrim reads its output with the flight_data/ module, and the identification/ module turns the reconstructed states into a tidy aerodynamic dataset with the same schema as the aero-structural sweeps. That shared schema is what lets the reduced-order model be identified from flight data and from the VSM–Billow model with the same fit, and compared one against the other.",
    bullets: [
      "Flight reconstructions read from EKF-AWE output (flight_data/)",
      "EKF reconstruction of states, wind and aerodynamic coefficients",
      "One aerodynamic dataset schema for flight data and aero-structural sweeps (identification/)"
    ],
    image: "img/kite-trajectory.png",
    caption: "Reconstructed kite flight trajectory from the EKF-AWE pipeline.",
    links: [REPO_EKF, PAPER_EKF, PAPER_ROM, REPO_AWETRIM]
  },
  "operational-optimization": {
    title: "Operating Strategy",
    text: "Optimising the full pumping cycle at every wind speed gives more than a power number: it says how the system should be flown. Read across the wind range, the optimal cycles show the operating strategy — how fast to reel out and in, which tether force to pull, how much to depower, how high to fly and over which tether lengths — and which operating limit takes over as the wind grows. At low wind the kite flies fully powered and the tether force rises with the wind; once the force reaches its limit the reel-out speed takes up the extra power, and once that is limited too the kite sheds power by depowering and climbing higher in the wind window.",
    bullets: [
      "Optimal control and flight settings over the wind speed, phase by phase",
      "Which operating limit binds in each wind range",
      "Derived from the same optimised cycles as the power curve"
    ],
    image: "img/operating-strategy.svg",
    caption: "LEI-V3 operating strategy from the optimised pumping cycles over the wind speed: reel-out and reel-in means of the reeling speed, tether force, depower and elevation angle, the tether-length range flown and the cycle duration. Dashed lines are the operating limits.",
    links: [PAPER_OPT, REPO_AWETRIM]
  },
  "co-simulation": {
    title: "Co-simulation",
    text: "AWETrim can run as a small JSON server next to another simulator. The client describes the kite system, wind and limits once; at every step it reports the current tether length and wind, and the server re-optimises the power-optimal reel-out path from the previous solution and returns it with the reel-out speed and depower it assumes. The simulator keeps flying the previous path while a re-solve runs. Any language can be the client: examples are provided for Julia, MATLAB and Python, and SimpleKiteControllers.jl installs the server directly.",
    bullets: [
      "JSON API: /init, /step, /status, /trajectory, /reset (examples/server/)",
      "Warm-started re-optimisation of the reel-out path at every step",
      "Client examples in Julia, MATLAB and Python"
    ],
    image: "img/co-simulation.svg",
    caption: "The co-simulation loop between an external kite simulator and the AWETrim server.",
    links: [REPO_AWETRIM]
  },
  "performance-assessment": {
    title: "Performance Assessment",
    text: "Performance is assessed at two levels. At system level, the full-cycle optimisation is run along the wind speed, each wind continued from the optimum of its neighbour, to give the power curve: the best mean cycle power over the number of figure-eight lobes. At kite level, the coupled VSM–Billow solve gives the loaded, deformed kite, with the aerodynamic load on the canopy and the tension in every bridle line. The ROM validators close the loop against measurements: validate_quasi_steady_state_v3.py compares the quasi-steady force balance to reconstructed flight data, and validate_spline_v3.py replays measured trajectories with the ROM.",
    bullets: [
      "Power curve from optimised full pumping cycles, continued along the wind speed",
      "Loaded kite: canopy load and bridle-line tensions from the coupled VSM–Billow solve",
      "ROM validation vs flight data (validate_quasi_steady_state_v3.py, validate_spline_v3.py)"
    ],
    image: "img/performance-assessment.svg",
    caption: "Left: LEI-V3 power curve, the mean power of optimised pumping cycles for different numbers of figure-eight lobes and the best of them, over the wind speed. Right: the LEI-V3 under load at the centre of the wind window, canopy shaded by the aerodynamic load and bridle lines by tension.",
    links: [PAPER_OPT, PAPER_ROM, REPO_AWETRIM, REPO_BILLOW]
  },
  "design-analysis": {
    title: "Kite Design",
    text: "The parametric geometry turns the framework into a design tool for different kite types. A leading-edge-inflatable kite such as the LEI-V3 is morphed from its baseline (aspect ratio, anhedral, taper, twist, sweep), a ram-air kite is built from its ribs, and rigid swept or three-panel wings are generated from a handful of scalar parameters. Each candidate is trimmed with VSM, and compute_stability_derivatives.py finite-differences that trim to obtain the aerodynamic stability derivatives and the longitudinal and lateral flight-dynamic eigenmodes.",
    bullets: [
      "Kite types: leading-edge inflatable, ram-air, swept and three-panel rigid wings",
      "Parametric morphing of the planform and LEI-airfoil studies",
      "Aerodynamic stability derivatives and flight-dynamic eigenmodes of each design"
    ],
    image: "img/kite-design.svg",
    caption: "Four kite types from the parametric geometry: the LEI-V3, the SK160 ram-air wing (Thedens 2022), and two aircraft-like rigid wings, one swept and one cranked three-panel.",
    links: [PAPER_AERO, PAPER_VSM, REPO_AWETRIM]
  }
};
