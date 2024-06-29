import numpy as np

def calculate_radial_speed(azimuth, elevation, reelout_speed, wind_speed):
    """
    Calculate the radial speed of the kite.
    """

    return wind_speed * np.cos(azimuth) * np.cos(elevation) - reelout_speed

def calculate_tangential_speed(azimuth, wind_speed, kite_speed, heading = 0):
    """
    Calculate the tangential speed of the kite.
    """
    return kite_speed - wind_speed*np.sin(azimuth)*np.cos(heading)

def calculate_side_speed(azimuth, elevation, wind_speed):
    """
    Calculate the side speed of the kite.
    """
    return wind_speed*np.cos(azimuth)*np.sin(elevation)


def calculate_apparent_speed(azimuth, elevation, reelout_speed, wind_speed, kite_speed, heading = 0):
    """
    Calculate the apparent speed of the kite.
    """

    #TODO: TRIPLE CHECK!!!!!!!!!!!!!!!!!!!!!!!
    radial_speed = calculate_radial_speed(azimuth, elevation, reelout_speed, wind_speed)
    tangential_speed = calculate_tangential_speed(azimuth, wind_speed, kite_speed, heading)
    side_speed = calculate_side_speed(azimuth, elevation, wind_speed)

    apparent_speed = np.sqrt(radial_speed**2 + tangential_speed**2)

    return apparent_speed

def calculate_kite_speed(aoa_trim,azimuth, elevation, reelout_speed, wind_speed, heading =0):
    """
    Calculate the kite speed.
    """
    radial_speed = calculate_radial_speed(azimuth, elevation, reelout_speed, wind_speed)
    kite_speed = radial_speed/ np.tan(aoa_trim) + wind_speed *( np.sin(azimuth)*np.cos(heading)+np.sin(elevation)*np.sin(heading))
    return kite_speed


