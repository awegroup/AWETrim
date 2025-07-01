import numpy as np
import matplotlib.pyplot as plt
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
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
omega = -1
r0 = 200
vr = 0
beta_p = np.radians(30)

# Parameter ranges
ry_vals = [100]
rz_vals = [40, 80, 120]
ky_vals = [0, 0.5, 1]
kz_vals = [0, 0.5, 1]

s = np.linspace(0, 2 * np.pi, 200)
t = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(10, 6))

for i, (ry, rz, ky, kz) in enumerate(
    itertools.product(ry_vals, rz_vals, ky_vals, kz_vals)
):
    figure = FigureEight(omega, r0, ry, rz, vr, beta_p, ky, kz)
    beta = figure.elevation(t, s)
    azimuth = figure.azimuth(t, s)
    ax.plot(
        np.degrees(azimuth),
        np.degrees(beta),
        label=f"ry={ry}, rz={rz}, ky={ky}, kz={kz}",
        color=colors[i % len(colors)],
    )
    # plt.show()


ax.set_xlabel("Azimuth [deg]")
ax.set_ylabel("Elevation [deg]")
# ax.legend()
plt.tight_layout()
# plt.savefig(save_folder + "figure8_param_combinations.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for i, (ry, rz, ky, kz) in enumerate(
    itertools.product(ry_vals, rz_vals, ky_vals, kz_vals)
):
    figure = FigureEight(omega, r0, ry, rz, vr, beta_p, ky, kz)
    y = figure.y(t, s)
    z = figure.z(t, s)
    ax.plot(
        y, z, label=f"ry={ry}, rz={rz}, ky={ky}, kz={kz}", color=colors[i % len(colors)]
    )
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
# ax.legend()

plt.tight_layout()
# plt.savefig(save_folder + "figure8_param_combinations_yz.pdf")
plt.show()
