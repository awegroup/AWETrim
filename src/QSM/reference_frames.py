import numpy as np

def transformation_A2C(va_c):
    theta_a = va_c[2]/np.sqrt(va_c[0]**2 + va_c[1]**2)
    chi_a = -np.arctan2(va_c[1], va_c[0]) #TODO: check sign 
    T = np.array([[np.cos(theta_a) * np.cos(chi_a), -np.sin(chi_a), np.sin(theta_a) * np.cos(chi_a)],
                  [np.cos(theta_a) * np.sin(chi_a), np.cos(chi_a), np.sin(theta_a) * np.sin(chi_a)],
                  [-np.sin(theta_a), 0, np.cos(theta_a)]])
    return T

def transformation_C2A(va_c):
    return transformation_A2C(va_c).T


def transformation_W2AZR(phi,beta):
    T = np.array([[-np.sin(phi), np.cos(phi), 0],
                    [-np.sin(beta) * np.cos(phi), -np.sin(beta) * np.sin(phi), np.cos(beta)],
                    [np.cos(beta) * np.cos(phi), np.cos(beta) * np.sin(phi), np.sin(beta)]])
    return T

def transformation_AZR2W(phi,beta):
    return transformation_W2AZR(phi,beta).T

def transformation_AZR2C(chi):
    T  = np.array([[np.sin(chi), np.cos(chi), 0],
                    [-np.cos(chi), np.sin(chi), 0],
                    [0, 0, 1]])
    return T

def transformation_C2AZR(chi):
    return transformation_AZR2C(chi).T

def transformation_W2C(phi,beta,chi):
    return transformation_AZR2C(chi) @ transformation_W2AZR(phi,beta)

