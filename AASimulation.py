import numpy as np
import matplotlib.pyplot as plt

# 1. Define given physical constraints
dx_sensitivity = 1.25  # Misalignment offset in micrometers
db_drop = 0.1          # Power drop in dB at that offset
center_position = 0.0  # Assume perfectly aligned center is at 0 um
max_power_linear = 1.0 # Normalized maximum power (equivalent to 0 dB loss)

# 2. Convert dB drop to a linear transmission ratio (T)
# Formula: dB = 10 * log10(T) -> T = 10^(dB/10)
# Since it's a loss, we use a negative value for the dB drop
T = 10**(-db_drop / 10) 

# 3. Algebraically solve for the Gaussian width parameter 'c' (Sigma)
# T = exp(-dx^2 / (2 * c^2)) -> c = sqrt(-dx^2 / (2 * ln(T)))
c = np.sqrt(-(dx_sensitivity**2) / (2 * np.log(T)))
print(f"Calculated Gaussian standard deviation (c): {c:.4f} µm")

# 4. Generate an array of scanning positions (x-axis)
# A window of -25 to +25 microns captures the entire curve beautifully
x = np.linspace(-25, 25, 500)

# 5. Calculate the Gaussian peak curves
# Linear scale profile (0.0 to 1.0)
y_linear = max_power_linear * np.exp(-((x - center_position)**2) / (2 * c**2))

# Logarithmic scale profile (dB relative to peak)
y_db = 10 * np.log10(y_linear)

# 6. Plot the results
plt.figure(figsize=(10, 5))

# Plot Linear Power Curve
plt.subplot(1, 2, 1)
plt.plot(x, y_linear, 'b-', label='Coupling Profile')
plt.axvline(dx_sensitivity, color='r', linestyle='--', label=f'Sensitivity Point ({dx_sensitivity}µm)')
plt.title('Linear Power Scale')
plt.xlabel('Displacement Offset (µm)')
plt.ylabel('Transmission Ratio')
plt.grid(True)
plt.legend()

# Plot Decibel (dB) Loss Curve
plt.subplot(1, 2, 2)
plt.plot(x, y_db, 'g-', label='Loss Profile')
plt.axhline(-db_drop, color='r', linestyle='--', label=f'-{db_drop} dB drop')
plt.axvline(dx_sensitivity, color='r', linestyle='--')
plt.title('Decibel (dB) Scale')
plt.xlabel('Displacement Offset (µm)')
plt.ylabel('Relative Power (dB)')
plt.ylim(-15, 1)  # Focus on the top 15 dB of the peak
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
