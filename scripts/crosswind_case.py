import numpy as np
from QSM.solver import find_zero_crossings

mass = 100
area = 19.75
CL0 = 0.0
CLa = 2*np.pi
CD0 = 0.1
k_cd = 0.1

rho = 1.225
alpha_d = np.radians(5)

course = np.radians(90)
wind_speed = 8

def CL(alpha):
    return CL0 + CLa * np.sin(alpha)

def CD(alpha):
    return CD0 + (CL(alpha) ** 2) * k_cd

def tangential_force(alpha, wind_speed, course):
    return 0.5 * rho * area* (wind_speed ** 2)*(1+1/np.tan(alpha+alpha_d)**2) * (CL(alpha) * np.sin(alpha +alpha_d)-CD(alpha) * np.cos(alpha + alpha_d))-mass * 9.81 * np.cos(course)

va_range = np.linspace(5, 80, 100)
alpha_t = np.arctan2(wind_speed, va_range)
alpha_range = alpha_t - alpha_d


f_tan = tangential_force(alpha_range, wind_speed, course)
a_tan = f_tan / mass
alpha_trim = find_zero_crossings(alpha_range, a_tan)[0]
va_trim = find_zero_crossings(va_range, a_tan)[0]

f_tan_up = tangential_force(alpha_range, wind_speed, course - np.radians(90))
a_tan_up = f_tan_up / mass
alpha_trim_up = find_zero_crossings(alpha_range, a_tan_up)[0]
va_trim_up = find_zero_crossings(va_range, a_tan_up)[0]

f_tan_down = tangential_force(alpha_range, wind_speed, course + np.radians(90))
a_tan_down = f_tan_down / mass
alpha_trim_down = find_zero_crossings(alpha_range, a_tan_down)[0]
va_trim_down = find_zero_crossings(va_range, a_tan_down)[0]


import matplotlib.pyplot as plt

# Plot alpha_range
plt.figure()
plt.plot(np.degrees(alpha_range), a_tan, label='horizontal')
plt.plot(np.degrees(alpha_range), a_tan_up, label='up')
plt.plot(np.degrees(alpha_range), a_tan_down, label='down')
plt.axvline(np.degrees(alpha_trim), linestyle='--', label=r'$\alpha_{eq}$')
plt.axvline(np.degrees(alpha_trim_up), color='orange', linestyle='--')
plt.axvline(np.degrees(alpha_trim_down), color='green', linestyle='--')
plt.legend()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'tangent acceleration [m/s$^2$]')
plt.grid()
plt.xlim(0, 15)
plt.ylim(-100, 100)
plt.savefig("tangential_force_alpha.pgf")
plt.close()

# Plot va_range
plt.figure()
plt.plot(va_range, a_tan, label='horizontal')
plt.plot(va_range, a_tan_up, label='up')
plt.plot(va_range, a_tan_down, label='down')
plt.axvline(va_trim, linestyle='--', label=r'$v_{a,eq}$')
plt.axvline(va_trim_up, color='orange', linestyle='--')
plt.axvline(va_trim_down, color='green', linestyle='--')
plt.legend()
plt.xlabel(r'$v_a$ [m/s]')
plt.ylabel(r'tangent acceleration [m/s$^2$]')
plt.grid()
plt.xlim(0, 80)
plt.ylim(-100, 100)
plt.savefig("tangential_force_va.pgf")
plt.close()

plt.show()


