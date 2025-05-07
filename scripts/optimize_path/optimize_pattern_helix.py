import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.kinematics.Kinematics import ParametrizedKinematics
from picawe import SystemModel
import casadi as ca
import time as timet
import json
from picawe.system.tether import FlexibleLinkTether
from picawe.system.kite import Kite
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
beta = ca.SX.sym("beta")
ry = 120
rz = 40
helix = Helix(omega, x0, rh, vr, beta)
lissajous = Lissajous(omega, x0, ry, rz, vr, beta)
figure_eight = FigureEight(omega, x0, 80, 80, vr, beta)

pattern = helix
kinematics = ParametrizedKinematics(pattern)
tether = FlexibleLinkTether()
kite = Kite(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=28, steering_control="asymmetric")
kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model="uniform")

# Substitute the numeric values into the symbolic expressions using CasADi functions
chi_func = ca.Function(
    "chi", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr, beta], [kinematics.chi]
)

vk_func = ca.Function(
    "vk", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr, beta], [kinematics.vk]
)

vr_func = ca.Function("vr", [kinematics.t, vr], [kinematics.vr])
dot_chi_func = ca.Function(
    "dot_chi", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr, beta], [kinematics.dot_chi]
)
vtau_func = ca.Function(
    "vtau", [kinematics.t, kinematics.s, kinematics.s_dot, rh, vr, beta], [kinematics.vtau]
)


kite_model.wind.speed_wind_ref = 10
kite_model.input_depower = 0

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
kite_model.s_dot = s_dot_sym
start_time = timet.time()
opti = ca.Opti()

kite_model.timeder_angle_course =  dot_chi_func(time_sym, s_sym, s_dot_sym, rh, vr, beta)
kite_model.speed_tangential = vtau_func(time_sym, s_sym, s_dot_sym, rh, vr, beta)
kite_model.angle_course = chi_func(time_sym, s_sym, s_dot_sym, rh, vr, beta)


solve_func, inputs_name = kite_model.setup_qs_solver(
        unknown_vars, solver_options=solver_options
    )

sf = ca.SX.sym("sf")
si = ca.SX.sym("si")
t = ca.SX.sym("t")
ts = (sf-si)/s_dot_sym
timestep_func = ca.Function("t_func", [si,sf,s_dot_sym], [ts])

rh_var = opti.variable()
vr_var = opti.variable()
beta_var = opti.variable()
time_var = opti.variable(len(s))
s_dot_var = opti.variable(len(s))
input_steering_var = opti.variable(len(s))
tension_tether_var = opti.variable(len(s))


kite_model.distance_radial = pattern.r(time_sym)
distance_radial = ca.Function("distance_radial", [time_sym,vr], [kite_model.distance_radial])
kite_model.angle_elevation = pattern.elevation(time_sym, s_sym)
angle_elevation_fun = ca.Function("angle_elevation", [time_sym,s_sym, rh, vr, beta], [kite_model.angle_elevation])
kite_model.angle_azimuth = pattern.azimuth(time_sym, s_sym)
kite_model.speed_radial = vr_func(time_sym, vr)
kite_model.establish_residual()
tension_tether = 0
power = 0
angle_elevation = ca.MX.zeros(len(s))
residual = ca.Function("residual", [time_sym,s_sym,s_dot_sym,rh, vr, beta, kite_model.input_steering, kite_model.length_tether], [kite_model.residual])
time_step = timestep_func(s[0],s[1],s_dot_var[0])
for i in range(len(s)):
    opti.subject_to(residual(time_var[i],s[i],s_dot_var[i],rh_var, vr_var, beta_var, input_steering_var[i],tension_tether_var[i]) == 0)
    power += tension_tether_var[i]*time_step*vr_var
    if i < len(s)-1:
        time_step = timestep_func(s[i],s[i+1],s_dot_var[i])
        opti.subject_to(time_var[i+1] == time_var[i] + time_step)
    
    
    angle_elevation[i] = angle_elevation_fun(time_var[i], s[i], rh_var, vr_var ,beta_var)
    # print(distance_radial(time_var[i], vr_var)*ca.sin(angle_elevation(time_var[i],s[i],rh_var,vr_var)))
    # z[i] = distance_radial(time_var[i], vr_var)*ca.sin(angle_elevation(time_var[i],s[i],rh_var,vr_var))

power = power/time_var[-1]
opti.subject_to(1 >= (input_steering_var[:] >= -1))
opti.subject_to(1e5 >= (tension_tether_var[:] >= 300))
# opti.subject_to(0 <= (s_var[:] <= 8*np.pi))
opti.subject_to(40 <= (rh_var <= 80))
opti.subject_to(1 <= (vr_var <= 6))
opti.subject_to(0 <= (s_dot_var[:] <= 30))
opti.subject_to(0 <= (time_var[:] <= 200))
opti.subject_to(25*np.pi/180 <= (beta_var <= 45*np.pi/180))
opti.subject_to(0*np.pi/180 <= (angle_elevation[:] <= 80*np.pi/180))

opti.set_initial(rh_var, 56.5)
opti.set_initial(beta_var,25/180*np.pi)
opti.set_initial(vr_var, 2.5)
opti.set_initial(time_var, results['time'])
opti.set_initial(s_dot_var, results['s_dot'])
opti.set_initial(input_steering_var, results['input_steering'])
opti.set_initial(tension_tether_var, results['tension_tether_ground'])

opti.minimize(-power/10000)

solver_options = {
    "ipopt": {
        "print_level": 5,  # Verbose output
        "tol": 1e-4,  # Relaxed tolerance
        "acceptable_tol": 1e-2,
        "acceptable_constr_viol_tol": 1e-2,
        "acceptable_iter": 10,
        "max_iter": 5000,
        "mu_strategy": "monotone",
        "nlp_scaling_method": "gradient-based",
        # "barrier_tol": 1e-6,  # Adjust as necessary
    },
    "print_time": True,
}

opti.solver("ipopt")
# opti.solver("sqpmethod")

try:
    sol = opti.solve()
    print("Solution found!")
    print("rh_var:", sol.value(rh_var))
    print("power:", sol.value(power))
    print("time:", sol.value(time_var[-1]))
    print("vr:", sol.value(vr_var))
    print("beta:", sol.value(beta_var)*180/np.pi)
except RuntimeError as e:
    print("Solver failed with error:", e)
    
    # Debugging variables
    print("Debugging variable values:")
    print("rh_var:", opti.debug.value(rh_var))
    print("power:", opti.debug.value(power))
    print("time:", opti.debug.value(time_var[-1]))
    print("vr:", opti.debug.value(vr_var))


