import numpy as np
from QSM.solver import find_zero_crossings
from QSM.reference_frames import transformation_W2C, transformation_A2C
import matplotlib.pyplot as plt

def CL(alpha):
    return CL0 + CLa * np.sin(alpha)

def CD(alpha):
    return CD0 + (CL(alpha) ** 2) * k_cd

def tangential_force(vtau, wind_speed, course, elevation, azimuth, relout_speed=0):
    T_W2C = transformation_W2C(azimuth, elevation, course)
    vw_c = T_W2C@np.array([wind_speed, 0, 0])
    va = np.array([-vtau, 0, 0]) + vw_c + np.array([0, 0, -relout_speed])

    chi_a = np.arctan(va[1]/ va[0])
    theta_a = np.arctan(va[2]/np.sqrt(va[0]**2 + va[1]**2))

    alpha = theta_a - alpha_d

    lift = 0.5 * rho * area * (np.linalg.norm(va)**2) * CL(alpha)
    drag = 0.5 * rho * area * (np.linalg.norm(va)**2) * CD(alpha)

    f_tan = lift * np.sin(theta_a)*np.cos(chi_a) - drag * np.cos(theta_a)*np.cos(chi_a) - mass * 9.81 * np.cos(course)*np.cos(elevation)
    return f_tan    

def find_va_trim(wind_speed, course, elevation, azimuth, relout_speed=0):
    va_range = np.linspace(-10, 50, 100)
    f_tan = []
    for vtau in va_range:
        f_tan.append(tangential_force(vtau, wind_speed, course, elevation, azimuth, relout_speed))
    f_tan = np.array(f_tan)
    a_tan = f_tan / mass
    zero_crossings = find_zero_crossings(va_range, a_tan)
    va_trim = zero_crossings[0] if len(zero_crossings) > 0 else np.nan
    return va_trim




mass = 15
area = 19.75
CL0 = 0.0
CLa = 2*np.pi
CD0 = 0.1
k_cd = 0.1

rho = 1.225
alpha_d = np.radians(2)

course = np.radians(90)
wind_speed = 8

elevation_angle = np.linspace(0, 90, 91) * np.pi / 180
azimuth_angle = np.linspace(-90, 90, 181) * np.pi / 180

# Create meshgrid for elevation and azimuth angles
azimuth_grid, elevation_grid = np.meshgrid(azimuth_angle,elevation_angle)


# Compute va_trim for each combination of elevation and azimuth angles
va_trim_grid = np.zeros_like(azimuth_grid)

for i in range(elevation_grid.shape[0]):
    for j in range(azimuth_grid.shape[1]):
        course = np.radians(90 if azimuth_grid[i, j] >= 0 else -90)
        va_trim_grid[i, j] = find_va_trim(wind_speed, course, elevation_grid[i, j], azimuth_grid[i, j])

# Plot the wind window
plt.figure(figsize=(10, 6))
plt.contourf(np.degrees(azimuth_grid), np.degrees(elevation_grid), va_trim_grid, levels=100, cmap='viridis')
plt.colorbar(label='va_trim')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Elevation Angle (degrees)')
plt.title('Wind Window with va_trim Values')
plt.savefig("wind_window.pdf")
plt.show()
# Convert spherical coordinates to Cartesian coordinates for 3D plotting
X = 100 * np.cos(elevation_grid) * np.cos(azimuth_grid)
Y = 100 * np.cos(elevation_grid) * np.sin(azimuth_grid)
Z = 100 * np.sin(elevation_grid)

# Plot the wind window in 3D with color representing va_trim
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(va_trim_grid / np.nanmax(va_trim_grid)), edgecolor='none')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(va_trim_grid)
fig.colorbar(mappable, ax=ax, label='va_trim')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Wind Window with va_trim Values')
ax.set_ylim(-100, 100)
ax.set_xlim(-100, 100)
ax.set_zlim(0, 100)
plt.savefig("wind_window_3d.pdf")
plt.show()