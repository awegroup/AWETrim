import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from awetrim.utils.color_palette import set_plot_style_no_latex, get_color_list
from awetrim.utils.fitting import fit_and_evaluate_model

set_plot_style_no_latex()
colors = get_color_list()
save_folder = "./results/figures/translational_paper/"


CL_mean_powered = 0.95
CD_mean_powered = 0.16
CL_mean_powered_steer = 0.63
CD_mean_powered_steer = 0.18
CL_mean_depowered = 0.35
CD_mean_depowered = 0.09


# --- Load experimental data ---
polars_path = (
    r"c:\Users\ocayon\Repositories\quasi-steady-awes\data\LEI-V9-KITE\polars_VSM.csv"
)
df = pd.read_csv(polars_path, comment="/")  # skip comment lines if any

alpha_VSM = np.radians(df["angle_of_attack_deg"])
CL_VSM = df["CL"]
CD_VSM = df["CD"]
LD_max_VSM = np.max(CL_VSM / CD_VSM)
print(f"Max L/D VSM: {LD_max_VSM}")

dependencies = [
    "np.ones(len(alpha))",
    "alpha",
    "alpha**2",
]
# Fit lift coeffcients
fit_cl = fit_and_evaluate_model(
    CL_VSM,
    dependencies=dependencies,
    alpha=alpha_VSM,
)
print("Fit CL VSM:")
print(fit_cl["coeffs"])
# Fit drag coeffcients
fit_cd = fit_and_evaluate_model(
    CD_VSM,
    dependencies=dependencies,
    alpha=alpha_VSM,
)
print("Fit CD VSM:")
print(fit_cd["coeffs"])

c0, c1, c2 = fit_cd["coeffs"]

c0 = c0 + 0.105
corr_factor = 1.6994209396860733
print(f"Correction factor: {corr_factor}")
coeffs_VSM_corrected = np.array([c0, c1, c2])
print("Corrected VSM CD coeffs:")
print(coeffs_VSM_corrected)

# --- Define alpha range for model curves ---
alpha_model = np.linspace(-np.radians(5), np.max(alpha_VSM), 200)


def compute_coeffs(alpha, coeffs):
    CL = coeffs[0] + coeffs[1] * alpha + coeffs[2] * alpha**2
    return CL


# --- Compute model curves ---
CL_VSM_fit = compute_coeffs(alpha_model, fit_cl["coeffs"])
CD_VSM_fit = compute_coeffs(alpha_model, fit_cd["coeffs"])
CD_VSM_fit_corrected = compute_coeffs(alpha_model, coeffs_VSM_corrected)


# --- Plot ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(np.degrees(alpha_VSM), CL_VSM, "o", label="VSM CL")
plt.plot(np.degrees(alpha_model), CL_VSM_fit, "-", label="Corrected Powered CL")


plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL")
plt.title("Lift Coefficient")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.degrees(alpha_VSM), CD_VSM, "o", label="VSM CD")
plt.plot(np.degrees(alpha_model), CD_VSM_fit, "-", label="Corrected Powered CD")
plt.plot(np.degrees(alpha_model), CD_VSM_fit_corrected, "-.", label="Corrected VSM CD")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CD")
plt.title("Drag Coefficient")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()

# --- Plot CL vs CD with mean markers ---
plt.figure(figsize=(6, 4))
plt.plot(CD_VSM, CL_VSM, "o", label="VSM", color=colors[0])
plt.plot(CD_VSM_fit, CL_VSM_fit, "-", label="VSM_fit", color=colors[2])
plt.plot(CD_VSM_fit_corrected, CL_VSM_fit, "--", label="Corrected VSM", color=colors[1])

# Mark mean powered and depowered points
plt.plot(
    CD_mean_powered,
    CL_mean_powered,
    "s",
    color=colors[2],
    label="Mean Powered Exp.",
    markersize=8,
)
# plt.plot(
#     CD_mean_powered_steer,
#     CL_mean_powered_steer,
#     "s",
#     color=colors[5],
#     label="Mean Powered Steered Exp.",
#     markersize=8,
# )
plt.plot(
    CD_mean_depowered,
    CL_mean_depowered,
    "s",
    label="Mean Depowered Exp.",
    markersize=8,
    color=colors[3],
)

plt.xlabel("$C_D$")
plt.ylabel("$C_L$")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure as pdf
plt.savefig(save_folder + "polars_comparison.pdf", bbox_inches="tight")


# Plot CL**3/CD**2 and CL/CD vs angle of attack
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(
    np.degrees(alpha_VSM),
    (CL_VSM**3) / (CD_VSM**2),
    "o",
    label="VSM CL^3/CD^2",
)
plt.plot(
    np.degrees(alpha_model),
    CL_VSM_fit**3 / CD_VSM_fit**2,
    "-",
    label="Corrected Powered CL^3/CD^2",
)


plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL^3 / CD^2")
plt.title("CL^3 / CD^2 vs Angle of Attack")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(
    np.degrees(alpha_VSM),
    CL_VSM / CD_VSM,
    "o",
    label="VSM CL/CD",
)
plt.plot(
    np.degrees(alpha_model),
    CL_VSM_fit / CD_VSM_fit,
    "-",
    label="Fit VSM",
)

plt.plot(
    np.degrees(alpha_model),
    CL_VSM_fit / CD_VSM_fit_corrected,
    "x",
    label="Corrected VSM",
)
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL / CD")
plt.title("CL / CD vs Angle of Attack")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
