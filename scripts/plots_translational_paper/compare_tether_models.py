from picawe.system.williams_tether import WilliamsTether
from picawe.system.tether import RigidLumpedTether
import numpy as np
from picawe.system.system_model import SystemModel
from picawe.utils.reference_frames import transformation_C_from_W



azimuth = 0
elevation = np.radians(45)
distance_radial = 200
speed_tangential = 40


position = [distance_radial*np.cos(azimuth)*np.cos(elevation), distance_radial*np.sin(azimuth)*np.cos(elevation), distance_radial*np.sin(elevation)]


# Define the tether model
tether1 = WilliamsTether()
tether1.speed_friction = 2
tether1.position = position
tether1.velocity_kite = [0, 50, 0]
tether1.velocity_wind = [10, 0, 0]
tether1.velocity_rotation_course_frame = [0,0,-1/4]
tether1.z0 = 0.01
tether1.rho = 1.225
tether1.g = 9.81
tether1.kappa = 0.4
tether1.mass_wing = 20
tether1.area_wing = 20
tether1.tension_tether_ground = 1e5

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

tether1.elevation_first_element = sol['x'][0]
tether1.azimuth_first_element = sol['x'][1]
tether1.tether_length = sol['x'][2]

print(tether1.force_tether_at_kite)

system2 = SystemModel(dof=3, quasi_steady=True, wind_model="uniform")
system2.distance_radial = 200
system2.angle_elevation = elevation
system2.angle_azimuth = azimuth
system2.angle_course = np.pi/2
system2.speed_radial = 0
system2.speed_tangential = 50
system2.tension_tether_ground = 1e5
system2.speed_wind_ref = 10

print("Rigid Lumped Tether Model")
print("------------------------------")
print(system2.force_tether_at_kite)

T_C_from_W = transformation_C_from_W(azimuth, elevation, np.pi/2)
print(T_C_from_W@-tether1.force_tether_at_kite)