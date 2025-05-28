import casadi as ca
from abc import ABC, abstractmethod

class ParametrizedPatterns(ABC):

    def __init__(self, **kwargs):
        self.optimization_vars = {}  # Dictionary to store symbolic optimization variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.SX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value

    def x(self, t, s):
        return self.xd(t, s) * ca.cos(self.beta(t)) - self.zd(t, s) * ca.sin(self.beta(t))

    def z(self, t, s):
        return self.xd(t, s) * ca.sin(self.beta(t)) + self.zd(t, s) * ca.cos(self.beta(t))

    def y(self, t, s):
        return self.yd(t, s)

    def azimuth(self, t, s):
        return ca.atan2(self.y(t, s), self.x(t, s))

    def elevation(self, t, s):
        return ca.atan2(self.z(t, s), ca.sqrt(self.x(t, s) ** 2 + self.y(t, s) ** 2))


class Helix(ParametrizedPatterns):

    def __init__(self, omega, r0, d0, vr, beta0, kappa=1, kbeta=0):
        super().__init__(omega=omega, r0=r0, d0=d0, vr=vr, beta0=beta0, kappa=kappa, kbeta=kbeta)

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0/self.r(t) - 1))
    
    def d(self, t):
        return self.d0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def yd(self, t, s):
        return self.d(t) / 2 * ca.sin(self.omega * s)

    def zd(self, t, s):
        return self.d(t) / 2 * ca.cos(self.omega * s)

    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(t, s)
        zd = self.zd(t, s)
        return ca.sqrt(r ** 2 - yd ** 2 - zd ** 2)


class Lissajous(ParametrizedPatterns):

    def __init__(self, omega, r0, a0, h0, vr, beta0, kappa=0):
        super().__init__(omega=omega, r0=r0, a0=a0, h0=h0, vr=vr, beta0=beta0, kappa=kappa)

    def a(self, t):
        return self.a0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def h(self, t):
        return self.h0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def yd(self, t, s):
        return self.a(t) * ca.cos(self.omega * s)

    def zd(self, t, s):
        return self.h(t) * ca.sin(2 * self.omega * s)

    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(t, s)
        zd = self.zd(t, s)
        return ca.sqrt(r ** 2 - yd ** 2 - zd ** 2)


class FigureEight(ParametrizedPatterns):

    def __init__(self, omega, r0, ry, rz, vr, beta0, ky=1, kz=1, kappa=0, kbeta=0):
        super().__init__(omega=omega, r0=r0, ry0=ry, rz0=rz, vr=vr, ky=ky, kz=kz, kappa=kappa, beta0=beta0, kbeta=kbeta)

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0/self.r(t) - 1))
    
    def r(self, t):
        return self.r0 + self.vr * t

    def ry(self, t):
        return self.ry0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def rz(self, t):
        return self.rz0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def yd(self, t, s):
        return self.ry(t) * ca.cos(self.omega * s) / (1 + self.ky * ca.sin(self.omega * s) ** 2)

    def zd(self, t, s):
        return self.rz(t) * ca.sin(self.omega * s) * ca.cos(self.omega * s) / (1 + self.kz * ca.sin(self.omega * s) ** 2)

    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(t, s)
        zd = self.zd(t, s)
        return ca.sqrt(r ** 2 - yd ** 2 - zd ** 2)
    
class ParametrizedPatternsAngles(ParametrizedPatterns):
    def __init__(self, **kwargs):
        self.optimization_vars = {}  # Dictionary to store symbolic optimization variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.SX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value
    def x(self, t, s):
        return self.r(t) * ca.cos(self.azimuth(t, s)) * ca.cos(self.elevation(t, s))
    def y(self, t, s):
        return self.r(t) * ca.sin(self.azimuth(t, s)) * ca.cos(self.elevation(t, s))
    def z(self, t, s):
        return self.r(t) * ca.sin(self.elevation(t, s))
        
class FigureEightAngles(ParametrizedPatternsAngles):

    def __init__(self, omega, r0, az_amp0, beta_amp0, vr, beta0, ky=1, kz=1, kappa=0, kbeta=0):
        super().__init__(omega=omega, r0=r0, az_amp0=az_amp0, beta_amp0=beta_amp0, vr=vr, ky=ky, kz=kz, kappa=kappa, beta0=beta0, kbeta=kbeta)

    def beta(self, t):
        return self.beta0 * (1 + self.kbeta * (self.r0/self.r(t) - 1))

    def r(self, t):
        return self.r0 + self.vr * t

    def az_amp(self, t):
        return self.az_amp0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def beta_amp(self, t):
        return self.beta_amp0 * (1 + self.kappa * (self.r(t) / self.r0 - 1))

    def azimuth(self, t, s):
        return self.az_amp(t) * ca.cos(self.omega * s) / (1 + self.ky * ca.sin(self.omega * s) ** 2)

    def elevation(self, t, s):
        return self.beta_amp(t) * ca.sin(self.omega * s) * ca.cos(self.omega * s) / (1 + self.kz * ca.sin(self.omega * s) ** 2)+ self.beta(t)

def create_pattern_from_dict(config: dict, optimize: bool = False) -> ParametrizedPatterns:
    pattern_type = config.get("pattern_type").lower()
    params = config.get("parameters", {})
    optimization_params = config.get("optimization_parameters", {})

    print(params)

    required_params = {
        "helix": ["omega", "r0", "d0", "vr", "beta0", "kappa"],
        "lissajous": ["omega", "r0", "a0", "h0", "vr", "beta", "kappa"],
        "figure_eight": ["omega", "r0", "ry", "rz", "vr", "beta0", "ky", "kz", "kappa"],
        "figure_eight_angles": ["omega", "r0", "az_amp0", "beta_amp0", "vr", "beta0", "ky", "kz", "kappa"]
    }

    if pattern_type not in required_params:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    missing_params = [param for param in required_params[pattern_type] if param not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters in 'initial_parameters' for '{pattern_type}': {', '.join(missing_params)}")

    # Replace optimized parameters with symbolic variables
    final_params = params.copy()
    if optimize:
        for param in optimization_params:
            if param in required_params[pattern_type]:
                final_params[param] = ca.SX.sym(param)

    # Instantiate the appropriate pattern class
    pattern_classes = {
        "helix": Helix,
        "lissajous": Lissajous,
        "figure_eight": FigureEight,
        "figure_eight_angles": FigureEightAngles

    }

    return pattern_classes[pattern_type](**final_params)
