import casadi as ca
from abc import ABC, abstractmethod

class ParametrizedPatterns(ABC):

    def __init__(self):
        pass

    def x(self, t, s):
        return self.xd(t, s)*ca.cos(self.beta) - self.zd(s)*ca.sin(self.beta)

    def z(self, t, s):
        return self.xd(t, s)*ca.sin(self.beta) + self.zd(s)*ca.cos(self.beta)
    
    def y(self, s):
        return self.yd(s)
    
    def azimuth(self, t, s):
        return ca.atan2(self.y(s), self.xd(t, s))
    
    def elevation(self, t, s):
        return ca.atan2(self.z(t,s), ca.sqrt(self.xd(t, s)**2 + self.y(s)**2))



class Helix(ParametrizedPatterns):

    def __init__(self, omega, r0, rh, vr, beta):
        self.omega = omega
        self.r0 = r0
        self.rh = rh
        self.vr = vr
        self.beta = beta

     
    
    def r(self, t):
        return self.r0 + self.vr*t
    
    def yd(self, s):
        return self.rh*ca.sin(self.omega * s)
    
    def zd(self, s):
        return self.rh*ca.cos(self.omega * s)
    
    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(s)
        zd = self.zd(s)
        return ca.sqrt(r**2 - yd**2 - zd**2)


class Lissajous(ParametrizedPatterns):

    def __init__(self, omega, r0, ry, rz, vr, beta):
        self.omega = omega
        self.r0 = r0
        self.ry = ry
        self.rz = rz
        self.vr = vr
        self.beta = beta

     
    
    def r(self, t):
        return self.r0 + self.vr*t
    
    def yd(self, s):
        return self.ry*ca.cos(self.omega * s)
    
    def zd(self, s):
        return self.rz*ca.sin(2*self.omega * s)
    
    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(s)
        zd = self.zd(s)
        return ca.sqrt(r**2 - yd**2 - zd**2)
    

class FigureEight(ParametrizedPatterns):

    def __init__(self, omega, r0, ry, rz, vr, beta,ky = 1, kz = 1):
        self.omega = omega
        self.r0 = r0
        self.ry = ry
        self.rz = rz
        self.vr = vr
        self.beta = beta
        self.ky = ky
        self.kz = kz

     
    
    def r(self, t):
        return self.r0 + self.vr*t
    
    def yd(self, s):
        return self.ry*ca.cos(self.omega * s)/(1 + self.ky*ca.sin(self.omega * s)**2)
    
    def zd(self, s):
        return self.rz*ca.sin(self.omega * s)*ca.cos(self.omega * s)/(1 + self.kz*ca.sin(self.omega * s)**2)
    
    def xd(self, t, s):
        r = self.r(t)
        yd = self.yd(s)
        zd = self.zd(s)
        return ca.sqrt(r**2 - yd**2 - zd**2)
    
    # def azimuth(self, t, s):
    #     return self.ry*ca.cos(self.omega * s)/(1 + self.ky*ca.sin(self.omega * s)**2)
    
    # def elevation(self, t, s):
    #     return self.rz*ca.sin(self.omega * s)*ca.cos(self.omega * s)/(1 + self.kz*ca.sin(self.omega * s)**2) + self.beta