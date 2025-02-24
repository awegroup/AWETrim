from picawe.system.williams_tether import WilliamsTether
from picawe.system.tether import RigidLumpedTether, DistributedDragTether
import numpy as np
from picawe.system.system_model import SystemModel
from picawe.utils.reference_frames import transformation_C_from_W
import copy

azimuth = 0
elevation = np.radians(45)
distance_radial = 200
speed_tangential = 40
diameter_tether = 0.01
density_tether = 970
uf = 0.6

position = [distance_radial*np.cos(azimuth)*np.cos(elevation), distance_radial*np.sin(azimuth)*np.cos(elevation), distance_radial*np.sin(elevation)]


# Define the tether model
tether1 = WilliamsTether(diameter=diameter_tether)
tether1.speed_friction = uf
tether1.velocity_rotation_course_frame = [0,0,2/4]
tether1.tension_tether_ground = 1e5
tether1.z0 = 0.01
tether1.rho = 1.225
tether1.g = 9.81
tether1.kappa = 0.4
tether1.mass_wing = 20
tether1.area_wing = 20


print("Williams Tether Model")
print("------------------------------")
print(tether1.force_tether_at_kite)
print(tether1.solve_tether_shape(position))

solver,names = tether1.solve_tether_shape(position)
# Bounds for the constraints
lbx = [-np.pi/2, -np.pi/2, 100]
ubx = [np.pi/2, np.pi/2, 300]
lbg = [0] * 3
ubg = [0] * 3
sol = solver(x0 = [0.01,0.01,200], lbg = lbg, ubg = ubg)
print(sol['x'])
print(sol['g'])


print(tether1.force_tether_at_kite)
tether2 = DistributedDragTether(diameter=diameter_tether)

system2 = SystemModel(dof=3, quasi_steady=True, wind_model="logarithmic", tether=tether2)
import copy
# system2 = copy.deepcopy(system2)
system2.distance_radial = 200
system2.angle_elevation = elevation
system2.angle_azimuth = azimuth
system2.angle_course = np.pi/2
system2.speed_radial = 0
system2.speed_tangential = 50
system2.tension_tether_ground = 1e5
system2.wind.speed_friction = uf
system2.input_depower = 0



tether3 = RigidLumpedTether(diameter=diameter_tether)
system3 = SystemModel(dof=3, quasi_steady=True, wind_model="logarithmic", tether=tether3)
system3.distance_radial = 200
system3.angle_elevation = elevation
system3.angle_azimuth = azimuth
system3.angle_course = np.pi/2
system3.speed_radial = 0
system3.speed_tangential = 50
system3.tension_tether_ground = 1e5
system3.wind.speed_friction = uf
system3.input_depower = 0



print(system3.tether.force_tether_at_kite(system3))


print(system2.tether.force_tether_at_kite(system2))

force1 = tether1.force_tether_at_kite



T_C_from_W = transformation_C_from_W(azimuth, elevation, np.pi/2)
print(T_C_from_W@force1(sol["x"][0], sol["x"][1], sol["x"][2]))





import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from picawe import SystemModel
import json
from picawe.system.kite import Kite

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
tether = RigidLumpedTether()
kite = Kite(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=25, steering_control="asymmetric")
kite_model = SystemModel(
    dof=3,
    quasi_steady=True,
    wind_model="logarithmic",
    kite=kite,
    tether=tether,
)

# Set constant parameters
kite_model.wind.speed_friction = uf
kite_model.input_depower = 0.0
kite_model.timeder_angle_course = np.radians(-2)

# Extract the tension tether function
aoa_func = kite_model.extract_function("angle_of_attack")
omega_func = kite_model.extract_function("velocity_rotation_course_frame")
position_W_func = kite_model.extract_function("position_W")
velocity_W_func = kite_model.extract_function("velocity_kite_W")

# -----------------------------------------------
# Define simulation parameters and initial state
# -----------------------------------------------
unknown_vars = ["tension_tether_ground", "input_steering", "speed_tangential"]
current_state = {
    "distance_radial": 200,
    "angle_elevation": np.radians(5),
    "angle_azimuth": 0,
    "angle_course": np.pi/4,
    "speed_radial": 0,
    "speed_tangential": 10,
}
solver_options = {
    "ipopt": {"print_level": 0, "sb": "yes"},
    "print_time": False,
}
time_step = 0.1
time = np.arange(0, 50, time_step)
qs_guess = [200, 0, 40]
states = []
import time as timet
start_time = timet.time()
solve_qs, inputs_name = kite_model.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )
# Solve quasi-steady state
p = [current_state[name] for name in inputs_name]

lbx,ubx,lbg,ubg = kite_model.get_boundaries(unknown_vars)
sol = solve_qs(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
kite_model.establish_ode()

# -----------------------------------------------
# Time integration loop
# -----------------------------------------------
for t in time:
    # Solve quasi-steady state
    p = [current_state[name] for name in inputs_name]

    lbx,ubx,lbg,ubg = kite_model.get_boundaries(unknown_vars)
    sol = solve_qs(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    qs_guess = sol["x"]
    qs_state = {name: float(qs_guess[i]) for i, name in enumerate(unknown_vars)}

    # Construct initial conditions for integration
    x0 = [
        current_state["distance_radial"],
        current_state["angle_elevation"],
        current_state["angle_azimuth"],
        current_state["angle_course"],
        current_state["speed_radial"],
        float(sol['x'][2]),  # speed_tangential
    ]

    # Integrate the dynamics
    xf = kite_model.integrate(x0, t, time_step)

    # Update the current state
    current_state = {name: float(xf[i]) for i, name in enumerate(current_state.keys())}

    full_state = {**current_state, 
                         'tension_tether_ground': float(sol['x'][0]),
    }

    # Evaluate tension tether

    aoa = aoa_func(
        *[full_state[name] for name in aoa_func.name_in()]
    )
    omega = omega_func(
        *[full_state[name] for name in omega_func.name_in()]
    )
    position_W = position_W_func(
        *[full_state[name] for name in position_W_func.name_in()]
    )
    velocity_W = velocity_W_func(
        *[full_state[name] for name in velocity_W_func.name_in()]
    )

    states.append({**full_state, "aoa": float(aoa), "omega": omega, "position_W": position_W, "velocity_W": velocity_W})

    # Stop if the system reaches critical limits
    if current_state["angle_elevation"] < 0 or current_state["distance_radial"] < 20:
        break

print("Elapsed time: ", timet.time() - start_time)
print("Simulated time: ", time[-1])

# -----------------------------------------------
# Process and visualize results
# -----------------------------------------------
solution_df = pd.DataFrame(states)


# Plot speed
plt.figure()
plt.plot(solution_df["speed_tangential"], label="Speed Tangential")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.legend()

# Plot tether tension
plt.figure()
plt.plot(solution_df["tension_tether_ground"], label="Tether Tension")
plt.xlabel("Time [s]")
plt.ylabel("Tether Tension [N]")
plt.legend()

#Plot angle of attack
plt.figure()
plt.plot(solution_df["aoa"]*180/np.pi, label="Angle of Attack")
plt.xlabel("Time [s]")
plt.ylabel("Angle of Attack [deg]")
plt.legend()

# Extract spherical coordinates
r = solution_df["distance_radial"]
theta = solution_df["angle_azimuth"]
phi = solution_df["angle_elevation"]

# Convert to Cartesian coordinates
x = r * np.cos(phi) * np.cos(theta)
y = r * np.cos(phi) * np.sin(theta)
z = r * np.sin(phi)

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, label="Trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_xlim(0, 200)
ax.set_ylim(-100, 100)
ax.set_zlim(0, 200)
ax.legend()

plt.show()

# -----------------------------------------------
# Print final results
# -----------------------------------------------
print("Reel-in elevation angle: ", np.degrees(states[-1]["angle_elevation"]))
print("Reel-in tether force: ", states[-1]["tension_tether_ground"])

force1 = [] 
force2 = []
force3 = []
for i in range(150):
    data = solution_df.iloc[i]
    system2.distance_radial = data["distance_radial"]
    system2.angle_elevation = data["angle_elevation"]
    system2.angle_azimuth = data["angle_azimuth"]
    system2.angle_course = data["angle_course"]
    system2.speed_radial = data["speed_radial"]
    system2.speed_tangential = data["speed_tangential"]
    system2.tension_tether_ground = data["tension_tether_ground"]
    system2.wind.speed_friction = uf
    system2.input_depower = 0
    force2.append(np.array(system2.tether.force_tether_at_kite(system2)).reshape(-1))   

    system3.distance_radial = data["distance_radial"]
    system3.angle_elevation = data["angle_elevation"]
    system3.angle_azimuth = data["angle_azimuth"]
    system3.angle_course = data["angle_course"]
    system3.speed_radial = data["speed_radial"]
    system3.speed_tangential = data["speed_tangential"]
    system3.tension_tether_ground = data["tension_tether_ground"]
    system3.wind.speed_friction = uf
    system3.input_depower = 0
    force3.append(np.array(system3.tether.force_tether_at_kite(system3)).reshape(-1))


    tether1.speed_friction = uf
    T_C_from_W = transformation_C_from_W(data["angle_azimuth"], data["angle_elevation"], data["angle_course"]) 
    tether1.velocity_rotation_course_frame =  -data["omega"]
    tether1.tension_tether_ground = data["tension_tether_ground"]
    solver,names = tether1.solve_tether_shape(data["position_W"])
    
    # Bounds for the constraints
    lbx = [-np.pi/2, -np.pi/2, 50]
    ubx = [np.pi/2, np.pi/2, 300]
    lbg = [0] * 3
    ubg = [0] * 3
    x0 = [data["angle_elevation"], data["angle_azimuth"], data["distance_radial"]]
    sol = solver(x0 = x0, lbg = lbg, ubg = ubg)    
    # print(sol['g'])
    force1.append(np.array(T_C_from_W@tether1.force_tether_at_kite(sol["x"][0], sol["x"][1], sol["x"][2])).reshape(-1))


figure,axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(np.array(force1)[:,0], label="Williams")
axs[0].plot(np.array(force2)[:,0], label="Distributed")
axs[0].plot(np.array(force3)[:,0], label="Lumped")
axs[0].set_title("Force x")
axs[0].legend()

axs[1].plot(np.array(force1)[:,1], label="Williams")
axs[1].plot(np.array(force2)[:,1], label="Distributed")
axs[1].plot(np.array(force3)[:,1], label="Lumped")
axs[1].set_title("Force y")
axs[1].legend()

axs[2].plot(np.array(force1)[:,2], label="Williams")
axs[2].plot(np.array(force2)[:,2], label="Distributed")
axs[2].plot(np.array(force3)[:,2], label="Lumped")
axs[2].set_title("Force z")
axs[2].legend()
plt.show()


    
