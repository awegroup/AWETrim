import numpy as np
from QSM.velocities import calculate_apparent_speed, calculate_radial_speed, calculate_tangential_speed, calculate_side_speed
from QSM.constants import rho

def calculate_tangential_force(aoa,CL,CD,azimuth, elevation, reelout_speed, wind_speed, wing_area,CD_tether = 0.0, mass = 15, heading = 90 ):
    """
    Calculate the tangential force acting on the kite.
    """
    g = 9.81
    lambda0 = np.radians(5)
    return 0.5 * rho * wing_area * (wind_speed * np.cos(azimuth) * np.cos(elevation) - reelout_speed)**2 * \
            (1 + 1 / np.arctan(aoa+lambda0)**2) *  (CL * np.sin(aoa+lambda0) - CD * np.cos(aoa+lambda0)-CD_tether)-mass*g*np.cos(elevation)*np.cos(heading)

def calculate_tether_force(aoa_trim,CL_trim,CD_trim,azimuth, elevation, reelout_speed, wind_speed,kite_speed, wing_area, mass = 15, heading = 0 ):
    
    g  = 9.81
    Umag = calculate_apparent_speed(azimuth, elevation, reelout_speed, wind_speed, kite_speed, heading)
    F_tether = 0.5*rho *Umag**2* wing_area*(CL_trim * np.cos(aoa_trim) + CD_trim * np.sin(aoa_trim)) - mass*g*np.cos(elevation)*np.sin(heading)

    return F_tether




     