import numpy as np


def compute_Rmin(m, rho, S, CL):
    """
    Compute minimal radius Rmin.

    Parameters:
        m  : mass (kg)
        rho: air density (kg/m^3)
        S  : wing area (m^2)
        CL : lift coefficient

    Returns:
        Rmin: minimal flight radius (m)
    """
    return (2 * m) / (rho * S * CL)


def compute_power(m, rho, S, CL, E, R, vw):
    """
    Compute normalized power ratio (P / P_wsopt).

    Parameters:
        R    : tether radius (m)
        Rmin : minimal radius (m)
        CL   : lift coefficient
        E    : aerodynamic efficiency

    Returns:
        P_ratio: normalized power output
    """
    Rmin = compute_Rmin(m, rho, S, CL)
    term = 1 - (Rmin / R) ** 2
    P = (4 / 27) * CL * E**2 * term ** (3 / 2) * 0.5 * rho * S * vw**3
    return P


def compute_power_analytical_talmar(gamma, rho, S, CL, E, vw, lambd):
    """
    Compute power P according to the given formula.

    Parameters:
        fx     : force fraction
        rho    : air density
        S      : wing area
        CL     : lift coefficient
        E      : aerodynamic efficiency (CL/CD)
        vw     : wind speed
        lambd  : lambda parameter (tip-speed ratio or equivalent)

    Returns:
        P      : power output
    """

    fx = f / np.cos(gamma)

    prefactor = fx / (1 - fx) * (rho * S * CL) / (2 * E)
    bracket_term = (lambd**-2 + (1 - fx) ** 2) ** (1.5)
    P = prefactor * vw**3 * bracket_term
    return P
