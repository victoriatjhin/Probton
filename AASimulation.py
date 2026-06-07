import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# USER-DEFINED BOUNDS (INPUT CONFIGURATION)
# ==========================================
initial_sensor_offset = -12.0  # Initial tracking center of the MEMS (µm)

step_size_bounds     = (3.0, 6.0)   # Min and Max possible step size (µm)
vibration_bounds     = (1.0, 2.0)   # Min and Max base vibration window (µm)
asymmetry_factor_max = 0.25         # Max skew percentage 

# ==========================================
# RANDOMIZED HARDWARE BEHAVIOUR GENERATION
# ==========================================
random_step_size    = np.random.uniform(*step_size_bounds)
base_vibration_span = np.random.uniform(*vibration_bounds)

# Generate hardware asymmetry
skew_percent    = np.random.uniform(-asymmetry_factor_max, asymmetry_factor_max)
left_amplitude  = (base_vibration_span / 2) * (1.0 + skew_percent)
right_amplitude = (base_vibration_span / 2) * (1.0 - skew_percent)

# Human-readable asymmetry metrics
total_actual_span = left_amplitude + right_amplitude
left_ratio_pct    = (left_amplitude / total_actual_span) * 100
right_ratio_pct   = (right_amplitude / total_actual_span) * 100

# Tracking centers
origin_centroid = 0.0  
initial_center  = initial_sensor_offset
corrected_center = initial_center + random_step_size if initial_center < 0 else initial_center - random_step_size

# ==========================================
# MATHEMATICAL PROPAGATION CONSTANTS
# ==========================================
dx_sensitivity = 1.25  
db_drop = 0.1          
linear_transmission = 10**(-db_drop / 10) 
sigma = np.sqrt(-(dx_sensitivity**2) / (2 * np.log(linear_transmission)))  # σ ≈ 5.82 µm

# ==========================================
# CAPTURING THE ASYMMETRIC SWEPT WINDOWS
# ==========================================
sample_resolution = 300

init_lower_bound = initial_center - left_amplitude
init_upper_bound = initial_center + right_amplitude
x_initial_sweep  = np.linspace(init_lower_bound, init_upper_bound, sample_resolution)

corr_lower_bound = corrected_center - left_amplitude
corr_upper_bound = corrected_center + right_amplitude
x_corrected_sweep = np.linspace(corr_lower_bound, corr_upper_bound, sample_resolution)

# ==========================================
# SIGNAL RESPONSE FIELD EXTRACTION
# ==========================================
y_initial_signal   = np.exp(-((x_initial_sweep - origin_centroid)**2) / (2 * sigma**2))
y_corrected_signal = np.exp(-((x_corrected_sweep - origin_centroid)**2) / (2 * sigma**2))

# ==========================================
# METRICS COMPILATION EXTRACTOR
# ==========================================
max_init, min_init = np.max(y_initial_signal), np.min(y_initial_signal)
max_corr, min_corr = np.max(y_corrected_signal), np.min(y_corrected_signal)

# Gradient
grad_amp_init = max_init - min_init
grad_amp_corr = max_corr - min_corr

try:
    area_init = np.trapezoid(y_initial_signal, x_initial_sweep)
    area_corr = np.trapezoid(y_corrected_signal, x_corrected_sweep)
except AttributeError:
    area_init = np.trapz(y_initial_signal, x_initial_sweep)
    area_corr = np.trapz(y_corrected_signal, x_corrected_sweep)

# First-Order Delta Shifts
delta_max      = max_corr - max_init
delta_min      = min_corr - min_init
delta_area     = area_corr - area_init
delta_grad_amp = grad_amp_corr - grad_amp_init

# ==========================================
# PLOTTING UNIFIED GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7.5))

# 1. Plot the underlying fixed continuous optical field
x_profile = np.linspace(-25, 25, 1000)
y_profile = np.exp(-((x_profile - origin_centroid)**2) / (2 * sigma**2))
ax.plot(x_profile, y_profile, color='#d3d3d3', linestyle=':', label='Continuous Gaussian Beam Profile')

# 2. Plot Initial Swept Profile (With Shading)
ax.plot(x_initial_sweep, y_initial_signal, color='#e60000', linewidth=2.5, label='Initial Sweep Profile')
ax.fill_between(x_initial_sweep, y_initial_signal, color='#e60000', alpha=0.12)

# 3. Plot Corrected Swept Profile (With Shading)
ax.plot(x_corrected_sweep, y_corrected_signal, color='#0066cc', linewidth=2.5, label='Next Step Profile')
ax.fill_between(x_corrected_sweep, y_corrected_signal, color='#0066cc', alpha=0.12)

# 4. Draw Mechanical Hardware Bars directly onto the baseline floor of the same plot
# Offset lines down slightly below 0 to cleanly isolate mechanical tracking ranges
ax.errorbar(initial_center, -0.08, xerr=[[left_amplitude], [right_amplitude]], fmt='ro', capsize=4, linewidth=2)
ax.errorbar(corrected_center, -0.14, xerr=[[left_amplitude], [right_amplitude]], fmt='bo', capsize=4, linewidth=2)

# Mechanical Labels floating next to the hardware pins
ax.text(initial_center, -0.06, f"{left_ratio_pct:.1f}% L ◄► {right_ratio_pct:.1f}% R", color='#e60000', fontsize=8.5, ha='center')
ax.text(corrected_center, -0.22, f"{left_ratio_pct:.1f}% L ◄► {right_ratio_pct:.1f}% R", color='#0066cc', fontsize=8.5, ha='center')

# Target indicator line
ax.axvline(origin_centroid, color='#404040', linestyle='--', alpha=0.6, label='Target Center (0µm)')

# --- FLOATING CALLOUT LABELS NEAR CROPPED GAUSSIAN PEAKS ---
text_init_label = (f"Initial Step Metrics:\n"
                   f"• Max: {max_init:.4f}\n"
                   f"• Min: {min_init:.4f}\n"
                   f"• Delta: {grad_amp_init:.4f}\n"
                   f"• Area: {area_init:.4f}")

text_corr_label = (f"Next Step Metrics:\n"
                   f"• Max: {max_corr:.4f}\n"
                   f"• Min: {min_corr:.4f}\n"
                   f"• Delta: {grad_amp_corr:.4f}\n"
                   f"• Area: {area_corr:.4f}")

ax.text(initial_center, max_init + 0.08, text_init_label, color='#b30000', fontsize=9,
         ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff7f7', edgecolor='#e60000', alpha=0.9))

ax.text(corrected_center, max_corr + 0.08, text_corr_label, color='#004499', fontsize=9,
         ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7faff', edgecolor='#0066cc', alpha=0.9))

# --- SYSTEM METRIC PERFORMANCE BOX ---
text_panel_deltas = (f"Step Performance Changes:\n"
                     f"• Δ Max : {delta_max:+.4f}\n"
                     f"• Δ Min : {delta_min:+.4f}\n"
                     f"• Δ Delta: {delta_grad_amp:+.4f}\n"
                     f"• Δ Area: {delta_area:+.4f}\n")

props_deltas = dict(boxstyle='round,pad=0.5', facecolor='#f7fff2', edgecolor='#4da61a', alpha=0.95)
ax.text(0.02, 0.95, text_panel_deltas, transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=props_deltas)

# Unified Graph Presentation Settings
ax.set_title('Gaussian Alignment tracking', fontsize=11, fontweight='bold')
ax.set_xlabel('Spatial Coordinate Positioning (µm)')
ax.set_ylabel('Coupled Intensity Efficiency')
ax.set_xlim(-25, 25)
ax.set_ylim(-0.25, 1.35)  # Axis limits adjusted to include floor space for mechanical bars and ceiling for text boxes
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()
plt.show()
