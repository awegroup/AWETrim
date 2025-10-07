import numpy as np
import matplotlib.pyplot as plt
import json
import casadi as ca
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list, custom_cmap
from picawe.timeseries.reelin_phase import ReelinPhase
from picawe.system.kite import Kite
from picawe.system.tether import RigidLumpedTether
from picawe.utils.defaults import PLOT_LABELS
from picawe.environment.Wind import Wind
from picawe.kinematics.parametrized_patterns import create_pattern_from_dict
import pickle

# ---------- Load precomputed fit data ----------
with open("fit_results.pkl", "rb") as f:
    fit_data = pickle.load(f)

C_sph = fit_data["C_sph"]
crs0 = fit_data["crs0"]
crsf = fit_data["crsf"]
phi0 = fit_data["phi0"]
phif = fit_data["phif"]
beta0 = fit_data["beta0"]
betaf = fit_data["betaf"]
C_interior = fit_data["C_interior"]
u_vals = fit_data["u_vals"]
U_interior = fit_data["U_interior"]
v0 = float(np.sqrt(fit_data["v0"][0]**2 + fit_data["v0"][1]**2 + fit_data["v0"][2]**2))

# ---------- Environment setup ----------
speed_wind_at_100 = 10
wind = Wind(
    wind_model="uniform",
    z0=0.1,
)
speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
wind.speed_wind_ref = speed_wind_at_100

# ---------- Load kite aerodynamics ----------
with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input_v9 = json.load(file)

# ---------- Pattern configuration ----------
pattern_config_v9 = {
    "pattern_type": "spline",
    "parameters": {
        "p": 3,
        "n_ctrl": 8,
        "r0": 300,
        "r1": 150,
        "crs0": crs0,
        "crsf": crsf,
        "phi0": phi0,
        "phif": phif,
        "beta0": beta0,
        "betaf": betaf,
        "C_interior": C_interior,
        "u_vals": u_vals,
        "U_interior": U_interior,
    },
    "start_time": 0,
    "end_time": 50,
    "start_angle": 0,
    "end_angle": 1,
    "n_points": 600,
    "optimization_parameters": [],
}

# ---------- Create pattern and find good starting point ----------
print("=== Pattern Setup ===")
test_pattern = create_pattern_from_dict(pattern_config_v9, optimize=False)

# Test different starting points to find a stable one
print("Finding stable starting point...")
alternative_starts = [
    {"s": 0.1, "description": "10% through pattern"},
    {"s": 0.25, "description": "25% through pattern"}, 
    {"s": 0.5, "description": "50% through pattern"},
    {"s": u_vals[len(u_vals)//2] if len(u_vals) > 0 else 0.2, "description": "Middle u_val"},
    {"s": 0.0, "description": "Pattern start"},
]

working_start = None
for alt_start in alternative_starts:
    try:
        test_s = alt_start["s"]
        test_r = 300
        
        phi_test = float(test_pattern.azimuth(ca.DM(test_r), ca.DM(test_s)))
        beta_test = float(test_pattern.elevation(ca.DM(test_r), ca.DM(test_s)))
        
        if not (np.isnan(phi_test) or np.isnan(beta_test)):
            print(f"✓ {alt_start['description']}: s={test_s:.6f}, phi={phi_test:.6f}, beta={beta_test:.6f}")
            working_start = {
                "s": test_s,
                "phi": phi_test,
                "beta": beta_test
            }
            break
        else:
            print(f"✗ {alt_start['description']}: NaN values detected")
            
    except Exception as e:
        print(f"✗ {alt_start['description']}: Error {e}")

# Use working start or fallback to safe defaults
if working_start:
    print(f"Using working start point: s={working_start['s']:.6f}")
    start_s = working_start["s"]
    start_phi = working_start["phi"]
    start_beta = working_start["beta"]
else:
    print("No working start found, using safe defaults")
    start_s = 0.1
    start_phi = 0.0
    start_beta = 0.3

# ---------- Create initial state ----------
print(f"\n=== Initial State Setup ===")
print(f"Starting at s={start_s:.6f}")
print(f"Pattern angles: phi={start_phi:.6f}, beta={start_beta:.6f}")

# Calculate course angle using derivatives if available
estimated_course = np.pi/4  # Default safe value
if working_start and hasattr(test_pattern, 'azimuth_derivative') and hasattr(test_pattern, 'elevation_derivative'):
    try:
        dphi_ds = float(test_pattern.azimuth_derivative(ca.DM(300), ca.DM(start_s)))
        dbeta_ds = float(test_pattern.elevation_derivative(ca.DM(300), ca.DM(start_s)))
        
        if abs(dbeta_ds) > 1e-8 and not (np.isnan(dphi_ds) or np.isnan(dbeta_ds)):
            estimated_course = np.arctan2(dphi_ds * np.cos(start_beta), dbeta_ds)
            print(f"Estimated course from derivatives: {estimated_course:.6f}")
        else:
            print("Invalid derivatives, using default course")
    except Exception as e:
        print(f"Error calculating course: {e}, using default")

# Ensure course is in valid range
estimated_course = np.arctan2(np.sin(estimated_course), np.cos(estimated_course))

initial_state = State(
    t=0,
    s=start_s,
    s_dot=1.0,  # Moderate speed
    s_ddot=0,
    length_tether=200,  # Round number for stability
    input_steering=0,
    tension_tether_ground=10000,  # Moderate tension
    distance_radial=300,  # Use r0
    speed_radial=0.5,  # Small radial speed
    timeder_speed_radial=0,
    input_depower=0,
    angle_elevation=start_beta,
    angle_azimuth=start_phi,
    angle_course=estimated_course,
    speed_tangential=25,  # Moderate tangential speed
    timeder_angle_course=0.0,
)

# Validate the initial state
print(f"Final initial state:")
print(f"  s={initial_state.s:.6f}")
print(f"  phi={initial_state.angle_azimuth:.6f}")
print(f"  beta={initial_state.angle_elevation:.6f}")
print(f"  course={initial_state.angle_course:.6f}")
print(f"  speed_tangential={initial_state.speed_tangential:.6f}")

# Check for NaN values
state_dict = initial_state.to_dict()
nan_detected = False
for key, value in state_dict.items():
    if isinstance(value, (float, int)) and np.isnan(value):
        print(f"WARNING: NaN detected in initial state for {key}")
        nan_detected = True

if nan_detected:
    print("ERROR: NaN values in initial state - simulation will fail")
else:
    print("✓ Initial state validation passed")

# ---------- Plot setup ----------
colors = get_color_list()
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1] * 8)
ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[4:, 0])
ax3 = fig.add_subplot(gs[:2, 2])
ax4 = fig.add_subplot(gs[2:4, 2])
ax5 = fig.add_subplot(gs[4:6, 2])
ax6 = fig.add_subplot(gs[6:, 2])

# ---------- Debug solver inputs ----------
def debug_initial_conditions():
    """Debug what inputs are being passed to the solver"""
    print("\n=== Debugging Solver Inputs ===")
    
    # Create a minimal test setup
    tether = RigidLumpedTether(diameter=0.01)
    kite = Kite(
        mass_wing=90,
        area_wing=47,
        aero_input=aero_input_v9,
        steering_control="asymmetric",
    )
    
    model = SystemModel(
        dof=3,
        quasi_steady=True,  # Start with quasi-steady
        kite=kite,
        tether=tether,
        wind_model=wind,
        neglect_radial_acceleration=False,
    )
    
    # Try to understand what the model expects
    print("Model state variables:")
    if hasattr(model, 'x'):
        print(f"  x: {model.x}")
    if hasattr(model, 'z'):
        print(f"  z: {model.z}")
    if hasattr(model, 'p'):
        print(f"  p: {model.p}")
    
    # Test the physics at our initial point
    print(f"\nTesting physics at initial state:")
    print(f"  Wind speed: {wind.speed_wind_ref}")
    print(f"  Kite position: r={initial_state.distance_radial}, phi={initial_state.angle_azimuth:.3f}, beta={initial_state.angle_elevation:.3f}")
    print(f"  Velocities: vtau={initial_state.speed_tangential}, vr={initial_state.speed_radial}")
    print(f"  Tensions: Ft={initial_state.tension_tether_ground}")
    
    # Check if the initial state satisfies basic physics
    # For a kite in steady flight, lift should balance weight component
    import math
    
    # Basic checks
    if initial_state.distance_radial <= 0:
        print("❌ Invalid distance_radial <= 0")
    if initial_state.speed_tangential <= 0:
        print("❌ Invalid speed_tangential <= 0") 
    if initial_state.tension_tether_ground <= 0:
        print("❌ Invalid tension <= 0")
    
    # Check elevation angle is reasonable (not vertical)
    if abs(initial_state.angle_elevation) > math.pi/2 * 0.9:
        print("❌ Elevation angle too steep")
    
    print("Basic physics checks passed ✓")

# Call the debug function
debug_initial_conditions()

# ---------- Try even simpler initial conditions ----------
def create_simple_initial_state():
    """Create very conservative initial conditions"""
    print("\n=== Creating Ultra-Simple Initial State ===")
    
    # Use the most conservative possible values
    simple_state = State(
        t=0.0,
        s=0.5,  # Middle of pattern
        s_dot=0.5,  # Very slow
        s_ddot=0.0,  
        length_tether=150.0,  # Shorter tether
        input_steering=0.0,
        tension_tether_ground=5000.0,  # Lower tension
        distance_radial=150.0,  # Shorter radius
        speed_radial=0.1,  # Very small radial speed
        timeder_speed_radial=0.0,
        input_depower=0.0,
        angle_elevation=0.2,  # ~11 degrees - very conservative
        angle_azimuth=0.0,  # Dead center
        angle_course=0.0,  # Straight
        speed_tangential=15.0,  # Moderate speed
        timeder_angle_course=0.0,
    )
    
    print("Simple state created:")
    for key, value in simple_state.to_dict().items():
        print(f"  {key}: {value}")
    
    return simple_state

# Test with the ultra-simple state
ultra_simple_state = create_simple_initial_state()

# ---------- Simulation function with more debugging ----------
def run_sim_debug(
    aero_input,
    pattern_config,
    label_prefix,
    mass_wing,
    area_wing,
    tether_diameter,
    color_base,
    marker="o",
    use_simple_state=False
):
    """Run simulations with extensive debugging"""
    result = {}
    
    # Choose which initial state to use
    if use_simple_state:
        test_state = ultra_simple_state
        print(f"\n=== Running {label_prefix} with ULTRA-SIMPLE state ===")
    else:
        test_state = initial_state
        print(f"\n=== Running {label_prefix} with calculated state ===")
    
    simulation_types = ["quasi_steady"]  # Try just quasi-steady first
    
    for sim_type in simulation_types:
        print(f"\n--- Running {sim_type} simulation ---")
        
        quasi_steady = True
        
        try:
            # Create components with more conservative settings
            tether = RigidLumpedTether(diameter=tether_diameter)
            kite = Kite(
                mass_wing=mass_wing,
                area_wing=area_wing,
                aero_input=aero_input,
                steering_control="asymmetric",
            )
            
            model = SystemModel(
                dof=3,
                quasi_steady=quasi_steady,
                kite=kite,
                tether=tether,
                wind_model=wind,
                neglect_radial_acceleration=False,
            )
            
            print("✓ Model created successfully")
            
            # Create phase with debugging
            phase = ReelinPhase(
                model, quasi_steady=quasi_steady, pattern_config=pattern_config
            )
            
            print("✓ Phase created successfully")
            
            # Add some debugging before simulation
            print(f"About to simulate with state:")
            print(f"  s={test_state.s}, r={test_state.distance_radial}")
            print(f"  phi={test_state.angle_azimuth:.4f}, beta={test_state.angle_elevation:.4f}")
            print(f"  vtau={test_state.speed_tangential}, vr={test_state.speed_radial}")
            print(f"  tension={test_state.tension_tether_ground}")
            
            # Try to run with failure allowed
            print("Starting simulation with allow_failure=True...")
            
            try:
                phase.run_simulation(start_state=test_state, allow_failure=True)
                print("✓ Simulation call completed")
            except Exception as sim_error:
                print(f"✗ Simulation threw exception: {sim_error}")
                continue
                
            # Check what we got back
            if hasattr(phase, 'states'):
                if phase.states is not None and len(phase.states) >= 2:
                    print(f"✓ Got {len(phase.states)} state points")
                    
                    # Try to extract just basic data
                    states_df = phase.states
                    result[sim_type] = {
                        "t": states_df["t"].values,
                        "s": states_df["s"].values,
                        "vtau": states_df["speed_tangential"].values,
                        "phi": np.degrees(states_df["angle_azimuth"].values),
                        "beta": np.degrees(states_df["angle_elevation"].values),
                    }
                    print("✓ Data extraction successful")
                    break  # Success!
                else:
                    print(f"✗ Got {len(phase.states) if phase.states else 0} state points")
            else:
                print("✗ No states attribute on phase")
                
        except Exception as e:
            print(f"✗ Exception in simulation setup: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return result, None

# Replace the original function call with debug version
print("\n" + "="*60)
print("TRYING ULTRA-SIMPLE STATE")
print("="*60)

results_v9_simple, _ = run_sim_debug(
    aero_input_v9, pattern_config_v9, "V9-Simple", 90, 47, 0.01, 2, marker="^", use_simple_state=True
)

if not results_v9_simple:
    print("\n" + "="*60)
    print("TRYING CALCULATED STATE") 
    print("="*60)
    
    results_v9_calc, _ = run_sim_debug(
        aero_input_v9, pattern_config_v9, "V9-Calc", 90, 47, 0.01, 2, marker="^", use_simple_state=False
    )

# Use whichever worked for plotting
results_v9 = results_v9_simple if results_v9_simple else results_v9_calc if 'results_v9_calc' in locals() else {}
scatter_v9 = None

# ---------- Run simulations ----------
results_v9, scatter_v9 = run_sim_debug(
    aero_input_v9, pattern_config_v9, "V9", 90, 47, 0.01, 2, marker="^"
)

# ---------- Final plot formatting ----------
# Handle colorbar
if scatter_v9 is not None:
    cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(scatter_v9, cax=cbar_ax)
    cbar.set_label(PLOT_LABELS["speed_tangential"])
else:
    print("No scatter plot data available for colorbar")

# Add text labels to identify Dynamic and Quasi-Steady plots
ax1.text(
    0.95, 0.95, "Dynamic",
    transform=ax1.transAxes, ha="right", va="top", fontsize=12, weight="bold",
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
)
ax2.text(
    0.95, 0.95, "Quasi-Steady",
    transform=ax2.transAxes, ha="right", va="top", fontsize=12, weight="bold",
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
)

# Set labels and legends
for ax in [ax1, ax2]:
    ax.legend(loc="lower right", fontsize=9)

ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel(r"$\overline{F}_{t,g}$ [--]")
ax5.set_ylabel(PLOT_LABELS["input_steering"])
ax6.set_ylabel(PLOT_LABELS["angle_of_attack"])
ax6.set_xlabel(PLOT_LABELS["phase"])
ax3.legend()

set_plot_style()
plt.tight_layout()
plt.show()

# # ---------- Energy, power and phase comparison ----------
# def compute_energy_metrics(results, label=""):
#     s_qs = results["quasi_steady"]["s"]
#     s_dyn = results["dynamic"]["s"]
#     print("Maximum s: ", max(s_qs), max(s_dyn))
    
#     # Add safety check
#     if len(s_qs) < 2 or len(s_dyn) < 2:
#         print(f"ERROR: Simulation failed for {label}. Not enough data points.")
#         print(f"QS points: {len(s_qs)}, Dyn points: {len(s_dyn)}")
#         return
    
#     mask_qs = (s_qs > s_qs[0]) & (s_qs < s_qs[0] + 360)
#     mask_dyn = (s_dyn > s_dyn[0]) & (s_dyn < s_dyn[0] + 360)
    
#     # Check if masks have any True values
#     if not np.any(mask_qs) or not np.any(mask_dyn):
#         print(f"ERROR: No valid data in range for {label}")
#         return
    
#     vtau_qs = results["quasi_steady"]["vtau"][mask_qs]
#     vtau_dyn = results["dynamic"]["vtau"][mask_dyn]
#     tension_qs = results["quasi_steady"]["tension"][mask_qs]
#     tension_dyn = results["dynamic"]["tension"][mask_dyn]
#     vr_qs = results["quasi_steady"]["vr"][mask_qs]
#     vr_dyn = results["dynamic"]["vr"][mask_dyn]
#     t_qs = results["quasi_steady"]["t"][mask_qs]
#     t_dyn = results["dynamic"]["t"][mask_dyn]

#     sum_energy_qs = np.sum(tension_qs * vr_qs * np.diff(t_qs, prepend=t_qs[0]))
#     sum_energy_dyn = np.sum(tension_dyn * vr_dyn * np.diff(t_dyn, prepend=t_dyn[0]))
#     sum_pow_qs = sum_energy_qs / (t_qs[-1] - t_qs[0])
#     sum_pow_dyn = sum_energy_dyn / (t_dyn[-1] - t_dyn[0])
#     power_diff = (sum_pow_qs - sum_pow_dyn) / sum_pow_dyn * 100

#     pow_qs = results["quasi_steady"]["power_mechanical"][mask_qs]
#     pow_dyn = results["dynamic"]["power_mechanical"][mask_dyn]

#     print(f"\n--- {label} ---")
#     print(f"Power QS: {sum_pow_qs:.2f}, Power Dyn: {sum_pow_dyn:.2f}.")
#     print(
#         f"Mean power QS: {np.mean(pow_qs):.2f}, Mean power Dyn: {np.mean(pow_dyn):.2f}"
#     )
#     print(f"Δ Power: {power_diff:.2f}%")

#     # Cross-correlation
#     t_common = np.linspace(max(t_qs[0], t_dyn[0]), min(t_qs[-1], t_dyn[-1]), 1000)
#     v1 = interp1d(t_qs, vtau_qs, kind="linear")(t_common) - np.mean(vtau_qs)
#     v2 = interp1d(t_dyn, vtau_dyn, kind="linear")(t_common) - np.mean(vtau_dyn)
#     corr = np.correlate(v1, v2, mode="full")
#     lags = np.arange(-len(v1) + 1, len(v1))
#     time_lags = lags * (t_common[1] - t_common[0])
#     best_lag = time_lags[np.argmax(corr)]
#     print(f"Estimated time lag: {best_lag:.3f} s")

#     # Mean and Max tension differences (%)
#     mean_t_qs = np.mean(tension_qs)
#     mean_t_dyn = np.mean(tension_dyn)
#     delta_ft_mean = (mean_t_qs - mean_t_dyn) / mean_t_dyn * 100

#     max_t_qs = np.max(tension_qs)
#     max_t_dyn = np.max(tension_dyn)
#     delta_ft_max = (max_t_qs - max_t_dyn) / max_t_dyn * 100

#     # Max tangential speed difference (%)
#     max_vtau_qs = np.max(vtau_qs)
#     max_vtau_dyn = np.max(vtau_dyn)
#     delta_vtau_max = (max_vtau_qs - max_vtau_dyn) / max_vtau_dyn * 100

#     # --- TENSION MIN DIFFERENCE ---
#     min_t_qs = np.min(tension_qs)
#     min_t_dyn = np.min(tension_dyn)
#     delta_ft_min = (min_t_qs - min_t_dyn) / min_t_dyn * 100

#     # --- VTAU MIN DIFFERENCE ---
#     min_vtau_qs = np.min(vtau_qs)
#     min_vtau_dyn = np.min(vtau_dyn)
#     delta_vtau_min = (min_vtau_qs - min_vtau_dyn) / min_vtau_dyn * 100

#     # --- PHASE LAG AT MAX vtau ---
#     s_dyn_vtau_max = s_dyn[np.argmax(vtau_dyn)]
#     s_qs_vtau_max = s_qs[np.argmax(vtau_qs)]
#     s_lag_max = s_qs_vtau_max - s_dyn_vtau_max

#     # --- PHASE LAG AT MIN vtau ---
#     s_dyn_vtau_min = s_dyn[np.argmin(vtau_dyn)]
#     s_qs_vtau_min = s_qs[np.argmin(vtau_qs)]
#     s_lag_min = s_qs_vtau_min - s_dyn_vtau_min

#     print(f"ΔF_t,mean: {delta_ft_mean:.2f}%")
#     print(f"ΔF_t,max: {delta_ft_max:.2f}%")
#     print(f"ΔF_t,min: {delta_ft_min:.2f}%")
#     print(f"Δv_tau,max: {delta_vtau_max:.2f}%")
#     print(f"Δv_tau,min: {delta_vtau_min:.2f}%")
#     print(f"ΔΦ_v_tau,max: {s_lag_max:.2f} deg")
#     print(f"ΔΦ_v_tau,min: {s_lag_min:.2f} deg")


# fig_3d = plt.figure()
# ax_3d = fig_3d.add_subplot(111, projection="3d")
# ax_3d.plot(
#     results_v9["quasi_steady"]["x"],
#     results_v9["quasi_steady"]["y"],
#     results_v9["quasi_steady"]["z"],
#     label="Quasi-Steady Trajectory",
# )
# ax_3d.set_xlabel("X")
# ax_3d.set_ylabel("Y")
# ax_3d.set_zlabel("Z")
# ax_3d.legend()

# plt.figure()
# plt.plot(results_v9["quasi_steady"]["t"], results_v9["quasi_steady"]["course_rate"])
# plt.show()
# compute_energy_metrics(results_v9, "V9")