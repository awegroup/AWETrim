from QSM.forces import calculate_tangential_force
from QSM.velocities import calculate_radial_speed
import numpy as np

def find_trim_angle(aoa_curve,CL_curve,CD_curve, wind_speed=10,azimuth = 0, elevation = 0, reelout_speed =0, wing_area=20, mass = 15, heading = 0):
    """
    Find the trim angle of attack.
    """
    tangential_force = calculate_tangential_force(aoa_curve,CL_curve,CD_curve,azimuth, elevation, reelout_speed, wind_speed, wing_area, mass = mass, heading = heading)

    aoa_trim = find_zero_crossings(aoa_curve, tangential_force)

    return aoa_trim



def find_zero_crossings(x, y):
    # Ensure x is sorted in ascending order (important if the data is not sorted)
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Find where the sign of y changes (i.e., product of consecutive elements is negative)
    sign_changes = np.where(np.diff(np.sign(y)))[0]

    # Calculate zero crossings with linear interpolation
    zero_crossings = []
    for index in sign_changes:
        if y[index] * y[index + 1] < 0:  # Ensure that there's an actual sign change
            # Linear interpolation formula: x_cross = x1 - y1 * ((x2 - x1) / (y2 - y1))
            x1, x2 = x[index], x[index + 1]
            y1, y2 = y[index], y[index + 1]
            x_cross = x1 - (y1 * ((x2 - x1) / (y2 - y1)))
            if x_cross > 0:  # Check if the crossing is in positive x
                zero_crossings.append(x_cross)
    return zero_crossings