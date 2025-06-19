from picawe.utils.utils import read_ekf_results
import numpy as np
import matplotlib.pyplot as plt

results, flight_data, config_data = read_ekf_results(
    "2023", "11", "27", "v9", addition=""
)

# Extract wind speed columns and heights
wind_cols = [
    col for col in flight_data.columns if "_Wind_Speed_m_s" in col and col[0].isdigit()
]
heights = [float(col.split("m")[0]) for col in wind_cols]

# Sort by height
sorted_indices = np.argsort(heights)
heights = np.array(heights)[sorted_indices]
wind_cols = np.array(wind_cols)[sorted_indices]

# Compute mean wind speed at each height
mean_wind = np.array([flight_data[col].mean() for col in wind_cols])

# Logarithmic profile using 100m as reference
z0 = 0.1  # roughness length in meters
kappa = 0.4  # von Karman constant (not needed for ratio)
u_100 = flight_data["200m_Wind_Speed_m_s"].mean()
log_profile = u_100 * np.log(heights / z0) / np.log(200 / z0)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(mean_wind, heights, "o-", label="Measured mean wind profile")
plt.plot(log_profile, heights, "--", label="Log profile (ref: 100m)")
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Height [m]")
plt.title("Wind profile")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
