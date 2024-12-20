
import numpy as np
from picawe import KiteSystem, Environment, Control
import pandas as pd
import matplotlib.pyplot as plt
import time
import casadi as ca


kcu_pos = np.array([0, 0, 0])
wing_pos = np.array([0, 0, 10])
kcu_mass = 30
wing_mass = 15

kite_cg = (kcu_mass * kcu_pos + wing_mass * wing_pos) / (kcu_mass + wing_mass)
print(kite_cg)


# LLoyd

# This is in the kite frame
T = 5000
aero_dir_angle = np.radians(5.07)
forces = np.array([0, 0, T])
aero_force = np.array([T*np.sin(aero_dir_angle), 0, -T*np.cos(aero_dir_angle)])

weight_kcu = np.array([-(kcu_mass*9.81),0,0])
weight_wing = np.array([-(wing_mass*9.81),0,0])

weight_moment = np.cross(wing_pos - kite_cg, weight_wing) + np.cross(kcu_pos - kite_cg, weight_kcu)
T_moment = np.cross(wing_pos - kite_cg, forces)
aero_moment = np.cross(wing_pos - kite_cg, aero_force)

total_moment = weight_moment + T_moment + aero_moment
total_force = forces + weight_kcu + weight_wing + aero_force
print('Total force:', total_force)
print('Total moment:', total_moment)



# Rotate kite around x to get moment equal to zero
roll = np.radians(8.5)
pitch = np.radians(aero_dir_angle)
Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
aero_force = np.dot(Ry, aero_force)
cg = np.dot(Ry, kite_cg)
wing_pos = np.dot(Ry, wing_pos)

# calculate moment
aero_moment = np.cross(wing_pos - cg, aero_force)
weight_moment = np.cross(wing_pos - cg, weight_wing) + np.cross(kcu_pos - cg, weight_kcu)
T_moment = np.cross(wing_pos - cg, forces)
total_moment = weight_moment + T_moment + aero_moment
print(total_moment)

print(T*np.sin(aero_dir_angle)- kcu_mass*9.81*np.cos(aero_dir_angle)- wing_mass*9.81*np.cos(aero_dir_angle))
