from awetrim.kinematics.parametrized_patterns import create_pattern_from_dict
import numpy as np
from scipy.optimize import least_squares

def find_Lissajous_RO_start_end_angles(pattern_type, parameters):
    s_start_init = 1.36 * np.pi
    range_init = 1.45 * np.pi
    cycles = 1

    trial_pattern = create_pattern_from_dict(pattern_type, parameters)

    az0 = trial_pattern.azimuth(1, s_start_init)
    el0 = trial_pattern.elevation(1, s_start_init)

    azf = trial_pattern.azimuth(1, s_start_init + range_init + cycles * (2*np.pi))
    elf = trial_pattern.elevation(1, s_start_init + range_init + cycles * (2*np.pi))

    def residuals(params):
        s_start, range = params
        az_start = trial_pattern.azimuth(1, s_start)
        el_start = trial_pattern.elevation(1, s_start)
        az_end = trial_pattern.azimuth(1, s_start + range + cycles * (2*np.pi))
        el_end = trial_pattern.elevation(1, s_start + range + cycles * (2*np.pi))

        res = np.array([
            float(az_start - az0),
            float(el_start - el0),
            float(az_end - azf),
            float(el_end - elf),
        ])

        return res

    initial_guess = [s_start_init, range_init]
    result = least_squares(residuals, initial_guess)
    s_start_opt, range_opt = result.x

    print(f"Optimized start angle: {s_start_opt}")
    print(f"Optimized range: {range_opt}")

    return s_start_opt, range_opt, cycles