import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.Kinematics import ParametrizedKinematics
from picawe import State
import casadi as ca
import time as timet
import json
import os


# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
#Load initial state

file_path = "./results/impact_inertial_forces/"
file_name = "helix_quasi_steady.csv"
results = pd.read_csv(os.path.join(file_path, file_name))


# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------

omega = -1
x0 = 200
rh = ca.SX.sym("rh")
vr = ca.SX.sym("vr")
beta = np.radians(30)
ry = 120
rz = 40
helix = Helix(omega, x0, rh, vr, beta)
lissajous = Lissajous(omega, x0, ry, rz, vr, beta)
figure_eight = FigureEight(omega, x0, 80, 80, vr, beta)

pattern = helix
kinematics = ParametrizedKinematics(pattern)
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=25, dof=3, quasi_steady=True)

# Substitute the numeric values into the symbolic expressions using CasADi functions
chi_func = ca.Function(
    "chi", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr], [kinematics.chi]
)

vk_func = ca.Function(
    "vk", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr], [kinematics.vk]
)

vr_func = ca.Function("vr", [kinematics.t, vr], [kinematics.vr])
dot_chi_func = ca.Function(
    "dot_chi", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr], [kinematics.dot_chi]
)
vtau_func = ca.Function(
    "vtau", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr], [kinematics.vtau]
)


state.speed_wind = 10
state.input_depower = 0

solver_options = {
    "ipopt": {
        "print_level": 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        "sb": "yes",  # Suppresses more detailed solver information
        
    },
    "print_time": False,  # Disables CasADi's internal timing output
    # "allow_free": True,  # Allows free variables
}
s = np.linspace(0, 2*np.pi, 100) + np.pi/2

states = []
unknown_vars = ["length_tether", "input_steering", "s_dot"]

s_dot_sym = ca.SX.sym("s_dot")
s_sym = ca.SX.sym("s")
time_sym = ca.SX.sym("time")
state.s_dot = s_dot_sym
start_time = timet.time()
opti = ca.Opti()

state.timeder_angle_course =  dot_chi_func(time_sym, s_sym, s_dot_sym, rh, vr)
state.speed_tangential = vtau_func(time_sym, s_sym, s_dot_sym, rh, vr)
state.angle_course = chi_func(time_sym, s_sym, s_dot_sym, rh, vr)


tension_tether_func = state.extract_function("tension_tether")
print(tension_tether_func)
solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )

sf = ca.SX.sym("sf")
si = ca.SX.sym("si")
t = ca.SX.sym("t")
ts = (sf-si)/s_dot_sym
timestep_func = ca.Function("t_func", [si,sf,s_dot_sym], [ts])

rh_var = opti.variable()
vr_var = opti.variable()
time_var = opti.variable(len(s))
s_dot_var = opti.variable(len(s))
input_steering_var = opti.variable(len(s))
length_tether_var = opti.variable(len(s))


state.distance_radial = pattern.r(time_sym)
distance_radial = ca.Function("distance_radial", [time_sym,vr], [state.distance_radial])
state.angle_elevation = pattern.elevation(time_sym, s_sym)
angle_elevation = ca.Function("angle_elevation", [time_sym,s_sym, rh, vr], [state.angle_elevation])
state.angle_azimuth = pattern.azimuth(time_sym, s_sym)
state.speed_radial = vr_func(time_sym, vr)
state.establish_residual()
tension_tether = 0
power = 0
z = ca.SX.zeros(len(s))
residual = ca.Function("residual", [time_sym,s_sym,s_dot_sym,rh, vr, state.input_steering, state.length_tether], [state.residual])
for i in range(len(s)-1):
    opti.subject_to(residual(time_var[i],s[i],s_dot_var[i],rh_var, vr_var, input_steering_var[i],length_tether_var[i]) == 0)
    time_step = timestep_func(s[i],s[i+1],s_dot_var[i])
    opti.subject_to(time_var[i+1] == time_var[i] + time_step)
    tension_tether = tension_tether_func(distance_radial(time_var[i], vr_var),length_tether_var[i])
    power += tension_tether*time_step*vr_var
    # print(distance_radial(time_var[i], vr_var)*ca.sin(angle_elevation(time_var[i],s[i],rh_var,vr_var)))
    # z[i] = distance_radial(time_var[i], vr_var)*ca.sin(angle_elevation(time_var[i],s[i],rh_var,vr_var))

power = power/time_var[-1]
opti.subject_to(5 >= (input_steering_var[:] >= -5))
opti.subject_to(300 >= (length_tether_var[:] >= 195))
# opti.subject_to(0 <= (s_var[:] <= 8*np.pi))
opti.subject_to(40 <= (rh_var <= 80))
opti.subject_to(1 <= (vr_var <= 5))
opti.subject_to(0 <= (s_dot_var[:] <= 5))
opti.subject_to(0 <= (time_var[:] <= 100))
# opti.subject_to(100 <= (z[:] <= 600))

opti.set_initial(rh_var, 62)
opti.set_initial(vr_var, 2.43)
opti.set_initial(time_var, results['time'])
opti.set_initial(s_dot_var, results['s_dot'])
opti.set_initial(input_steering_var, results['input_steering'])
opti.set_initial(length_tether_var, results['length_tether'])

opti.minimize(-power)

opti.solver("ipopt")

try:
    sol = opti.solve()
    print("Solution found!")
    print("rh_var:", sol.value(rh_var))
    print("power:", sol.value(power))
    print("time:", sol.value(time_var[-1]))
    print("vr:", sol.value(vr_var))
except RuntimeError as e:
    print("Solver failed with error:", e)
    
    # Debugging variables
    print("Debugging variable values:")
    print("rh_var:", opti.debug.value(rh_var))
    print("power:", opti.debug.value(power))
    print("time:", opti.debug.value(time_var[-1]))
    print("vr:", opti.debug.value(vr_var))


