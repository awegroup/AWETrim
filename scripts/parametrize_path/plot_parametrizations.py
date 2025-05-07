import numpy as np
import matplotlib.pyplot as plt
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.utils.color_palette import get_color_list, set_plot_style, set_plot_style_no_latex


colors = get_color_list()
set_plot_style()
save_folder = './results/figures/translational_paper/'
def W_from_AZR(phi,beta,r):
    return np.array([r*np.cos(phi)*np.cos(beta), r*np.sin(phi)*np.cos(beta), r*np.sin(beta)])

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
omega = -2*np.pi
r0 = 100
d0 = 60
vr = 2.414
beta_p = np.radians(30)
a0 = 30
h0 = 20
helix = Helix(omega, r0, d0, vr, beta_p)
lissajous = Lissajous(omega, r0, a0, h0, vr, beta_p)

s = np.linspace(0, 4, 1000)
t = np.linspace(0, 30, 1000)



# beta = np.radians([0, 30, 60 ,90])
r = np.linspace(0, 200, 1000)

fig, axs = plt.subplots(2, 1, figsize=(5,4), sharex=True)
arg_min = np.argmin(helix.elevation(t,s))
arg_max = np.argmax(helix.elevation(t,s))
beta = [helix.elevation(t,s)[arg_min], beta_p, helix.elevation(t,s)[arg_max]]
az = [helix.azimuth(t,s)[arg_min], 0, helix.azimuth(t,s)[arg_max]]
for b,a in zip(beta, az):
    pos = W_from_AZR(a, b, r)
    axs[0].plot(pos[0], pos[2], linestyle="--", color = colors[0])

axs[0].plot(helix.x(t,s), helix.z(t,s), label="$\kappa_e = 1$", color = colors[1])
helix.kappa = 0
axs[0].plot(helix.x(t,s), helix.z(t,s), label="$\kappa_e = 0$", color = colors[2])
axs[0].set_ylabel("z (m)")
axs[0].text(165, 100, r"$\beta_p$", ha="center", va="bottom")
axs[0].set_xlim(0,175)
axs[0].set_ylim(0,150)
axs[0].legend()

helix.kappa = 1
arg_min = np.argmin(helix.azimuth(t,s))
arg_max = np.argmax(helix.azimuth(t,s))
beta = [helix.elevation(t,s)[arg_min], beta_p, helix.elevation(t,s)[arg_max]]
az = [helix.azimuth(t,s)[arg_min], 0, helix.azimuth(t,s)[arg_max]]
for a, b in zip(az, beta):
    pos = W_from_AZR(a, b, r)
    axs[1].plot(pos[0], pos[1], linestyle="--", color = colors[0])
axs[1].plot(helix.x(t,s), helix.y(t,s), label="$\kappa_e = 1$", color = colors[1])
helix.kappa = 0
axs[1].plot(helix.x(t,s), helix.y(t,s), label="$\kappa_e = 0$", color = colors[2])
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")
axs[1].text(165, 2, r"$\phi_p$", ha="center", va="bottom")
axs[1].set_xlim(0,175)
axs[1].set_ylim(-60,60)
# Save figure as pdf
plt.savefig(save_folder+"helix_parametrization_3d.pdf", bbox_inches='tight')
# plt.show()


fig, axs = plt.subplots(2, 1, figsize=(5,4), sharex=True)
lissajous.kappa = 1
arg_min = np.argmin(lissajous.elevation(t,s))
arg_max = np.argmax(lissajous.elevation(t,s))
beta = [lissajous.elevation(t,s)[arg_min], beta_p, lissajous.elevation(t,s)[arg_max]]
az = [lissajous.azimuth(t,s)[arg_min], 0, lissajous.azimuth(t,s)[arg_max]]
for b,a in zip(beta, az):
    pos = W_from_AZR(a, b, r)
    axs[0].plot(pos[0], pos[2], linestyle="--", color = colors[0])

axs[0].plot(lissajous.x(t,s), lissajous.z(t,s), label="$\kappa_e = 1$", color = colors[1])
lissajous.kappa = 0
axs[0].plot(lissajous.x(t,s), lissajous.z(t,s), label="$\kappa_e = 0$", color = colors[2])

axs[0].set_ylabel("z (m)")
axs[0].text(165, 100, r"$\beta_p$", ha="center", va="bottom")
axs[0].set_xlim(0,175)
axs[0].set_ylim(0,150)
axs[0].legend()

lissajous.kappa = 1
arg_min = np.argmin(lissajous.azimuth(t,s))
arg_max = np.argmax(lissajous.azimuth(t,s))
beta = [lissajous.elevation(t,s)[arg_min], beta_p, lissajous.elevation(t,s)[arg_max]]
az = [lissajous.azimuth(t,s)[arg_min], 0, lissajous.azimuth(t,s)[arg_max]]
for a, b in zip(az, beta):
    pos = W_from_AZR(a, b, r)
    axs[1].plot(pos[0], pos[1], linestyle="--", color = colors[0])
axs[1].plot(lissajous.x(t,s), lissajous.y(t,s), label="$\kappa_e = 1$", color = colors[1])
lissajous.kappa = 0
axs[1].plot(lissajous.x(t,s), lissajous.y(t,s), label="$\kappa_e = 0$", color = colors[2])
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")
axs[1].text(165, 2, r"$\phi_p$", ha="center", va="bottom")
axs[1].set_xlim(0,175)
axs[1].set_ylim(-60,60)
# Save figure as pdf
plt.savefig(save_folder+"lissajous_parametrization_3d.pdf", bbox_inches='tight')

# plt.show()

helix.r0 = 100
helix.vr = 0
helix.beta = 0
lissajous.r0 = 100
lissajous.vr = 0
lissajous.beta = 0
lissajous.a0 = 30
lissajous.h0 = 10
s = np.linspace(0, 1, 1000)

fig, axs = plt.subplots(1, 2, figsize=(10, 3))

# Plot helix trajectory
axs[0].plot(helix.y(t, s), helix.z(t, s), color=colors[0])

# Remove axis and grid, and set equal aspect ratio
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_frame_on(False)
axs[0].set_aspect('equal')

# Plot Lissajous trajectory
axs[1].plot(lissajous.y(t, s), lissajous.z(t, s), color=colors[0])

# Remove axis and grid, and set equal aspect ratio
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_frame_on(False)
axs[1].set_aspect('equal')
# Save as svg
plt.savefig(save_folder+"helix_lissajous_trajectory.svg", bbox_inches='tight')
plt.show()