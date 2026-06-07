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

# Hardware execution of Step 2 (Moving toward zero by the random step size)
corrected_center = initial_center + random_step_size if initial_center < 0 else initial_center - random_step_size

# ==========================================
# MATHEMATICAL PROPAGATION CONSTANTS
# ==========================================
dx_sensitivity = 1.25  
db_drop = 0.1          
linear_transmission = 10**(-db_drop / 10) 
sigma = np.sqrt(-(dx_sensitivity**2) / (2 * np.log(linear_transmission)))  # σ ≈ 5.82 µm

# ==========================================
# SAMPLE STEP 1 & STEP 2 profiles
# ==========================================
sample_resolution = 300

# Step 1
init_lower_bound  = initial_center - left_amplitude
init_upper_bound  = initial_center + right_amplitude
x_initial_sweep   = np.linspace(init_lower_bound, init_upper_bound, sample_resolution)
y_initial_signal  = np.exp(-((x_initial_sweep - origin_centroid)**2) / (2 * sigma**2))

# Step 2
corr_lower_bound  = corrected_center - left_amplitude
corr_upper_bound  = corrected_center + right_amplitude
x_corrected_sweep = np.linspace(corr_lower_bound, corr_upper_bound, sample_resolution)
y_corrected_signal = np.exp(-((x_corrected_sweep - origin_centroid)**2) / (2 * sigma**2))

# Extract metrics
max_init, min_init = np.max(y_initial_signal), np.min(y_initial_signal)
max_corr, min_corr = np.max(y_corrected_signal), np.min(y_corrected_signal)

grad_amp_init = max_init - min_init
grad_amp_corr = max_corr - min_corr

try:
    area_init = np.trapezoid(y_initial_signal, x_initial_sweep)
    area_corr = np.trapezoid(y_corrected_signal, x_corrected_sweep)
except AttributeError:
    area_init = np.trapz(y_initial_signal, x_initial_sweep)
    area_corr = np.trapz(y_corrected_signal, x_corrected_sweep)

# ==========================================
# EXACT LOG-DERIVATIVE ESTIMATION ALGORITHM
# ==========================================
# Calculate the log-slope across the Step 2 window width
log_slope_corr = (np.log(max_corr) - np.log(min_corr)) / base_vibration_span

# Predict exactly how far away the center axis is from Step 2's location
predicted_offset_distance = abs(log_slope_corr * (sigma**2))

# Determine alignment path direction toward zero
direction = 1.0 if corrected_center < 0 else -1.0
estimated_center_target = corrected_center + (direction * predicted_offset_distance)

# ==========================================
# STEP 3 EXECUTION: MOVE STRAIGHT TO TARGET
# ==========================================
step3_lower_bound  = estimated_center_target - left_amplitude
step3_upper_bound  = estimated_center_target + right_amplitude
x_step3_sweep      = np.linspace(step3_lower_bound, step3_upper_bound, sample_resolution)
y_step3_signal     = np.exp(-((x_step3_sweep - origin_centroid)**2) / (2 * sigma**2))

max_step3, min_step3 = np.max(y_step3_signal), np.min(y_step3_signal)
grad_amp_step3       = max_step3 - min_step3
try:
    area_step3 = np.trapezoid(y_step3_signal, x_step3_sweep)
except AttributeError:
    area_step3 = np.trapz(y_step3_signal, x_step3_sweep)

# First-Order Delta Shifts (Step 1 to Step 2)
delta_max      = max_corr - max_init
delta_min      = min_corr - min_init
delta_area     = area_corr - area_init
delta_grad_amp = grad_amp_corr - grad_amp_init

# ==========================================
# PLOTTING UNIFIED GRAPH WITH GREEN STEP 3
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# Continuous Hidden Profile
x_profile = np.linspace(-25, 25, 1000)
y_profile = np.exp(-((x_profile - origin_centroid)**2) / (2 * sigma**2))
ax.plot(x_profile, y_profile, color='#d3d3d3', linestyle=':', label='Continuous Gaussian Beam Profile')

# Plot Step 1 (Red)
ax.plot(x_initial_sweep, y_initial_signal, color='#e60000', linewidth=2, label='Initial Sweep Profile')
ax.fill_between(x_initial_sweep, y_initial_signal, color='#e60000', alpha=0.08)

# Plot Step 2 (Blue)
ax.plot(x_corrected_sweep, y_corrected_signal, color='#0066cc', linewidth=2, label='Next Step Profile')
ax.fill_between(x_corrected_sweep, y_corrected_signal, color='#0066cc', alpha=0.08)

# Plot Step 3 (Green) - Algorithmic Target Jump
ax.plot(x_step3_sweep, y_step3_signal, color='#2eb82e', linewidth=2.5, label='Estimated Step 3 Target')
ax.fill_between(x_step3_sweep, y_step3_signal, color='#2eb82e', alpha=0.15)

# Hardware mechanical tracking pins on floor
ax.errorbar(initial_center, -0.08, xerr=[[left_amplitude], [right_amplitude]], fmt='ro', capsize=4, linewidth=1.5)
ax.errorbar(corrected_center, -0.14, xerr=[[left_amplitude], [right_amplitude]], fmt='bo', capsize=4, linewidth=1.5)
ax.errorbar(estimated_center_target, -0.20, xerr=[[left_amplitude], [right_amplitude]], fmt='go', capsize=4, linewidth=2)

ax.text(initial_center, -0.06, f"{left_ratio_pct:.1f}% L ◄► {right_ratio_pct:.1f}% R", color='#e60000', fontsize=8, ha='center')
ax.text(corrected_center, -0.12, f"{left_ratio_pct:.1f}% L ◄► {right_ratio_pct:.1f}% R", color='#0066cc', fontsize=8, ha='center')
ax.text(estimated_center_target, -0.18, f"{left_ratio_pct:.1f}% L ◄► {right_ratio_pct:.1f}% R", color='#2eb82e', fontsize=8, ha='center')

ax.axvline(origin_centroid, color='#404040', linestyle='--', alpha=0.6, label='Target Center (0µm)')

# --- FLOATING CALLOUT LABELS NEAR ALL THREE CROPPED PEAKS ---
text_init_label = f"Initial Step:\n• Max: {max_init:.4f}\n• Min: {min_init:.4f}\n• Delta: {grad_amp_init:.4f}\n• Area: {area_init:.4f}"
text_corr_label = f"Next Step:\n• Max: {max_corr:.4f}\n• Min: {min_corr:.4f}\n• Delta: {grad_amp_corr:.4f}\n• Area: {area_corr:.4f}"
text_step3_label = f"Step 3 Target:\n• Max: {max_step3:.4f}\n• Min: {min_step3:.4f}\n• Delta: {grad_amp_step3:.4f}\n• Area: {area_step3:.4f}"

ax.text(initial_center, max_init + 0.08, text_init_label, color='#b30000', fontsize=8.5, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7f7', edgecolor='#e60000', alpha=0.85))
ax.text(corrected_center, max_corr + 0.08, text_corr_label, color='#004499', fontsize=8.5, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7faff', edgecolor='#0066cc', alpha=0.85))
ax.text(estimated_center_target, max_step3 + 0.08, text_step3_label, color='#1f7a1f', fontsize=8.5, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2fff2', edgecolor='#2eb82e', alpha=0.85))

# --- SYSTEM METRIC PERFORMANCE BOX ---
text_panel_deltas = (f"Step Performance Changes (1->2):\n"
                     f"• Δ Max : {delta_max:+.4f}\n"
                     f"• Δ Min : {delta_min:+.4f}\n"
                     f"• Δ Delta: {delta_grad_amp:+.4f}\n"
                     f"• Δ Area: {delta_area:+.4f}\n\n"
                     f"Algorithmic Predictor Vector:\n"
                     f"• Step 2 Center Pos: {corrected_center:.3f} µm\n"
                     f"• Calculated Offset: {predicted_offset_distance:.3f} µm\n"
                     f"• Estimated Center : {estimated_center_target:.3f} µm")

props_deltas = dict(boxstyle='round,pad=0.5', facecolor='#f7fff2', edgecolor='#4da61a', alpha=0.95)
ax.text(0.02, 0.95, text_panel_deltas, transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=props_deltas)

ax.set_title('Active Alignment Tracking for Gaussian Peak', fontsize=11, fontweight='bold')
ax.set_xlabel('Spatial Coordinate Positioning (µm)')
ax.set_ylabel('Coupled Intensity Efficiency')
ax.set_xlim(-25, 25)
ax.set_ylim(-0.28, 1.45)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()
plt.show()
