import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


CL_mean_powered = 0.94
CD_mean_powered = 0.18
CL_mean_powered_steer = 0.78
CD_mean_powered_steer = 0.17
CL_mean_depowered = 0.35
CD_mean_depowered = 0.08


# --- Load experimental data ---
polars_path = (
    r"c:\Users\ocayon\Repositories\quasi-steady-awes\data\LEI-V9-KITE\polars.csv"
)
df = pd.read_csv(polars_path, comment="/")  # skip comment lines if any

alpha_exp = np.radians(df["angle_of_attack_deg"])
CL_exp = df["CL"]
CD_exp = df["CD"]

# --- Load first JSON model ---
with open(
    r"c:\Users\ocayon\Repositories\quasi-steady-awes\data\LEI-V9-KITE\v9_aero_input.json",
    "r",
) as f:
    aero1 = json.load(f)

params1 = aero1["params"]
coeffs1 = aero1["coefficients"]

# --- (Optional) Load second JSON model ---
with open(
    r"c:\Users\ocayon\Repositories\quasi-steady-awes\data\LEI-V9-KITE\v9_VSM_aero_input.json",
    "r",
) as f:
    aero2 = json.load(f)
params2 = aero2["params"]
coeffs2 = aero2["coefficients"]

# --- Define alpha range for model curves ---
alpha_model = np.linspace(np.min(alpha_exp), np.max(alpha_exp), 200)


def compute_coeffs(alpha, params, coeffs, u_s=0, u_p=0):
    CL = params.get("CL0", 0)
    CD = params.get("CD0", 0)
    for term in coeffs["CL"]:
        if term["var"] == "alpha":
            CL += term["coef"] * alpha ** term["power"]
        elif term["var"] == "u_s":
            CL += term["coef"] * u_s ** term["power"]
        elif term["var"] == "u_p":
            CL += term["coef"] * u_p ** term["power"]
    for term in coeffs["CD"]:
        if term["var"] == "alpha":
            CD += term["coef"] * alpha ** term["power"]
        elif term["var"] == "u_s":
            CD += term["coef"] * u_s ** term["power"]
        elif term["var"] == "u_p":
            CD += term["coef"] * u_p ** term["power"]
    return CL, CD


# --- Compute model curves ---
CL1, CD1 = compute_coeffs(alpha_model, params1, coeffs1)
CL2, CD2 = compute_coeffs(alpha_model, params2, coeffs2)
CL1_depowered, CD1_depowered = compute_coeffs(
    alpha_model, params1, coeffs1, u_p=1
)  # up=1 (fully depowered)
CL1_steered, CD1_steered = compute_coeffs(
    alpha_model, params1, coeffs1, u_s=1, u_p=0
)  # us=1 (fully steered)

# --- Plot ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(np.degrees(alpha_exp), CL_exp, "o", label="VSM CL")
plt.plot(np.degrees(alpha_model), CL1, "-", label="Corrected Powered CL")
plt.plot(np.degrees(alpha_model), CL2, "--", label="Fit VSM CL")
plt.plot(np.degrees(alpha_model), CL1_depowered, "-.", label="Corrected Depowered CL")
plt.plot(np.degrees(alpha_model), CL1_steered, ":", label="Corrected Steered CL")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL")
plt.title("Lift Coefficient")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.degrees(alpha_exp), CD_exp, "o", label="VSM CD")
plt.plot(np.degrees(alpha_model), CD1, "-", label="Corrected Powered CD")
plt.plot(np.degrees(alpha_model), CD2, "--", label="Fit VSM CD")
plt.plot(np.degrees(alpha_model), CD1_depowered, "-.", label="Corrected Depowered CD")
plt.plot(np.degrees(alpha_model), CD1_steered, ":", label="Corrected Steered CD")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CD")
plt.title("Drag Coefficient")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()

# --- Plot CL vs CD with mean markers ---
plt.figure(figsize=(7, 6))
plt.plot(CD_exp, CL_exp, "o", label="VSM CL vs CD")
plt.plot(CD1, CL1, "-", label="Corrected Powered CL vs CD")
plt.plot(CD2, CL2, "--", label="Fit VSM CL vs CD")
plt.plot(CD1_depowered, CL1_depowered, "-.", label="Corrected Depowered CL vs CD")
plt.plot(CD1_steered, CL1_steered, ":", label="Corrected Steered CL vs CD")

# Mark mean powered and depowered points
plt.plot(CD_mean_powered, CL_mean_powered, "rs", label="Mean Powered", markersize=8)
plt.plot(
    CD_mean_powered_steer,
    CL_mean_powered_steer,
    "go",
    label="Mean Powered Steered",
    markersize=8,
)
plt.plot(
    CD_mean_depowered, CL_mean_depowered, "bs", label="Mean Depowered", markersize=8
)

plt.xlabel("CD")
plt.ylabel("CL")
plt.title("CL vs CD Polar")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Plot CL**3/CD**2 and CL/CD vs angle of attack
plt.figure(figsize=(10, 5))
magnitude = np.sqrt(CL1**2 / CD1**2 + 1) * (CL1**2 / CD1 + CD1)
plt.subplot(1, 2, 1)
plt.plot(
    np.degrees(alpha_exp),
    (CL_exp**3) / (CD_exp**2),
    "o",
    label="VSM CL^3/CD^2",
)
plt.plot(
    np.degrees(alpha_model),
    magnitude,
    "-",
    label="Corrected Powered CL^3/CD^2",
)
plt.plot(
    np.degrees(alpha_model),
    (CL2**3) / (CD2**2),
    "--",
    label="Fit VSM CL^3/CD^2",
)
plt.plot(
    np.degrees(alpha_model),
    (CL1_depowered**3) / (CD1_depowered**2),
    "-.",
    label="Corrected Depowered CL^3/CD^2",
)
plt.plot(
    np.degrees(alpha_model),
    (CL1_steered**3) / (CD1_steered**2),
    ":",
    label="Corrected Steered CL^3/CD^2",
)
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL^3 / CD^2")
plt.title("CL^3 / CD^2 vs Angle of Attack")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(
    np.degrees(alpha_exp),
    CL_exp / CD_exp,
    "o",
    label="VSM CL/CD",
)
plt.plot(
    np.degrees(alpha_model),
    CL1 / CD1,
    "-",
    label="Corrected Powered CL/CD",
)
plt.plot(
    np.degrees(alpha_model),
    CL2 / CD2,
    "--",
    label="Fit VSM CL/CD",
)
plt.plot(
    np.degrees(alpha_model),
    CL1_depowered / CD1_depowered,
    "-.",
    label="Corrected Depowered CL/CD",
)
plt.plot(
    np.degrees(alpha_model),
    CL1_steered / CD1_steered,
    ":",
    label="Corrected Steered CL/CD",
)
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL / CD")
plt.title("CL / CD vs Angle of Attack")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
