import numpy as np
import matplotlib.pyplot as plt
from picawe.kinematics.parametrized_patterns import (
    Helix,
    Lissajous,
    FigureEight,
    CST_Lissajous,
)
from picawe.utils.color_palette import (
    get_color_list,
    set_plot_style,
    set_plot_style_no_latex,
)
import itertools

colors = get_color_list()
set_plot_style_no_latex()
save_folder = "./results/figures/translational_paper/"


def W_from_AZR(phi, beta, r):
    return np.array(
        [
            r * np.cos(phi) * np.cos(beta),
            r * np.sin(phi) * np.cos(beta),
            r * np.sin(beta),
        ]
    )


# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
omega = 1
r0 = 200
vr = 1
beta_p = np.radians(30)

s = np.linspace(0, 26 * np.pi, 2000)
r = np.linspace(200, 300, 2000)

fig, ax = plt.subplots(figsize=(10, 6))


figure = CST_Lissajous(
    omega=omega,
    r0=r0,
    vr=vr,
    beta0=beta_p,
    az_amp0=np.radians(40),
    beta_amp0=np.radians(20),
    beta_coeffs=[0, 0, 0],
    az_coeffs=[0, 0, 0],
    kappa=0,
    kbeta=0,
)
beta = figure.elevation(r, s)
azimuth = figure.azimuth(r, s)
ax.plot(
    np.degrees(azimuth),
    np.degrees(beta),
)
radius_curvature = figure.radius_curvature(r, s)
plt.figure()
plt.plot(
    np.degrees(s),
    radius_curvature,
)

plt.show()
