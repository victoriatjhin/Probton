import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SYSTEM PARAMETERS & CONFIGURATION
# ==========================================
INITIAL_OFFSET_BOUNDS    = (-12.0, 12.0) # Random initialization boundary (µm)
STEP_1_TO_2_SIZE_BOUNDS  = (3.0, 6.0)    # Blind first mechanical move range (µm)
STEP_2_TO_2B_SIZE_BOUNDS = (3.0, 6.0)    # Second mechanical move range for area track (µm)
VIBRATION_SPAN_BOUNDS    = (1.0, 2.0)    # Base MEMS sweep window size (µm)
MAX_ASYMMETRIC_SKEW      = 0.25          # Max percentage imbalance allowed between wings

# ==========================================
# RANDOMIZED HARDWARE COMPONENT SELECTION
# ==========================================
step_dist_1_to_2  = np.random.uniform(*STEP_1_TO_2_SIZE_BOUNDS)
step_dist_2_to_2b = np.random.uniform(*STEP_2_TO_2B_SIZE_BOUNDS)
vibration_span    = np.random.uniform(*VIBRATION_SPAN_BOUNDS)

# Generate MEMS physical sweep asymmetry parameters
skew_factor = np.random.uniform(-MAX_ASYMMETRIC_SKEW, MAX_ASYMMETRIC_SKEW)
left_wing   = (vibration_span / 2) * (1.0 + skew_factor)
right_wing  = (vibration_span / 2) * (1.0 - skew_factor)

# Convert asymmetry metrics to percentages for layout presentation
total_mems_span = left_wing + right_wing
left_pct        = (left_wing / total_mems_span) * 100
right_pct       = (right_wing / total_mems_span) * 100

# ==========================================
# PHYSICAL CONSTANTS (GAUSSIAN FIELD)
# ==========================================
dx_sensitivity      = 1.25  
db_drop             = 0.1          
linear_transmission = 10**(-db_drop / 10) 
sigma               = np.sqrt(-(dx_sensitivity**2) / (2 * np.log(linear_transmission)))  # σ ≈ 5.82 µm

# Helper utility to collect sensor profile traces across an asymmetric window
def sweep_sensor(center_position):
    lower_bound = center_position - left_wing
    upper_bound = center_position + right_wing
    x_trace = np.linspace(lower_bound, upper_bound, 300)
    y_trace = np.exp(-(x_trace**2) / (2 * sigma**2)) # Fixed physical core lands exactly at 0.000
    return x_trace, y_trace

# ==========================================
# PHYSICAL STEP 1: INITIAL POSITION CAPTURE
# ==========================================
X1 = np.random.uniform(*INITIAL_OFFSET_BOUNDS)
X1_trace, Y1_trace = sweep_sensor(X1)

# Extract Step 1 Hardware Feedback Data
max_Y1 = np.max(Y1_trace)
min_Y1 = np.min(Y1_trace)
delta_Y1 = max_Y1 - min_Y1
try:
    area_A1 = np.trapezoid(Y1_trace, X1_trace)
except AttributeError:
    area_A1 = np.trapz(Y1_trace, X1_trace)

# ==========================================
# PHYSICAL STEP 2: FIRST MOTION TOWARD CENTER
# ==========================================
initial_direction = 1.0 if X1 < 0 else -1.0
X2 = X1 + (initial_direction * step_dist_1_to_2)
X2_trace, Y2_trace = sweep_sensor(X2)

# Extract Step 2 Hardware Feedback Data
max_Y2 = np.max(Y2_trace)
min_Y2 = np.min(Y2_trace)
delta_Y2 = max_Y2 - min_Y2
try:
    area_A2 = np.trapezoid(Y2_trace, X2_trace)
except AttributeError:
    area_A2 = np.trapz(Y2_trace, X2_trace)

# ==========================================
# ALGORITHM TRACK 1: 2-STEP LOG-DELTA FINAL PREDICTOR (PARAMETER-FREE & CROSSOVER-SAFE)
# ==========================================
# Extract directional slope signs directly from the physical sweep array boundaries
slope_sign_step1 = 1.0 if Y1_trace[-1] >= Y1_trace[0] else -1.0
slope_sign_step2 = 1.0 if Y2_trace[-1] >= Y2_trace[0] else -1.0

# Compute signed log-space gradients (natively registers peak crossover boundaries)
G1 = (np.log(max_Y1) - np.log(min_Y1)) * slope_sign_step1
G2 = (np.log(max_Y2) - np.log(min_Y2)) * slope_sign_step2

denominator_delta = G2 - G1

if abs(denominator_delta) > 1e-9:
    # Exact linear intercept solver. Sign-flips are naturally resolved, and sigma cancels out perfectly.
    final_target_delta = X2 - (G2 * (X2 - X1)) / denominator_delta
else:
    final_target_delta = X2

final_delta_x_trace, final_delta_y_trace = sweep_sensor(final_target_delta)
final_delta_max = np.max(final_delta_y_trace)
final_delta_min = np.min(final_delta_y_trace)
final_delta_val = final_delta_max - final_delta_min
try:
    final_delta_area = np.trapezoid(final_delta_y_trace, final_delta_x_trace)
except AttributeError:
    final_delta_area = np.trapz(final_delta_y_trace, final_delta_x_trace)

# ==========================================
# ALGORITHM TRACK 2: 3-POINT LOG-AREA SWEEP STEP
# ==========================================
area_heading = initial_direction if (area_A2 >= area_A1) else -initial_direction
X2b = X2 + (area_heading * step_dist_2_to_2b)  # Cleaned name target variable
X2b_trace, Y2b_trace = sweep_sensor(X2b)

max_Y2b = np.max(Y2b_trace)
min_Y2b = np.min(Y2b_trace)
delta_Y2b = max_Y2b - min_Y2b
try:
    area_A2b = np.trapezoid(Y2b_trace, X2b_trace)
except AttributeError:
    area_A2b = np.trapz(Y2b_trace, X2b_trace)

# Process natural log metrics across the 3-point alignment curve to resolve final destination
ln_A1  = np.log(area_A1)
ln_A2  = np.log(area_A2)
ln_A2b = np.log(area_A2b)

# Exact second-order curvature denominator for non-equidistant grids
h1 = X2 - X1
h2 = X2b - X2
denominator_area = (ln_A2b - ln_A2) / h2 - (ln_A2 - ln_A1) / h1

if abs(denominator_area) > 1e-9:
    final_target_area = (X1 + X2) / 2.0 - ((ln_A2 - ln_A1) / (2.0 * h1 * denominator_area)) * (h1 + h2)
else:
    final_target_area = X2

# Evaluate final target profile convergence accuracy from the target positioning
final_area_x_trace, final_area_y_trace = sweep_sensor(final_target_area)
final_area_max = np.max(final_area_y_trace)
final_area_min = np.min(final_area_y_trace)
final_area_delta = final_area_max - final_area_min
try:
    final_area_val = np.trapezoid(final_area_y_trace, final_area_x_trace)
except AttributeError:
    final_area_val = np.trapz(final_area_y_trace, final_area_x_trace)

# Calculated system performance updates
delta12_max   = max_Y2 - max_Y1
delta12_min   = min_Y2 - min_Y1
delta12_delta = delta_Y2 - delta_Y1
delta12_area  = area_A2 - area_A1

delta2b2_max   = max_Y2b - max_Y2
delta2b2_min   = min_Y2b - min_Y2
delta2b2_delta = delta_Y2b - delta_Y2
delta2b2_area  = area_A2b - area_A2

# ==========================================
# PLOTTING UNIFIED ALIGNMENT TELEMETRY
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# Continuous Hidden Reference Profile
x_profile = np.linspace(-25, 25, 1000)
y_profile = np.exp(-(x_profile**2) / (2 * sigma**2))
ax.plot(x_profile, y_profile, color='#d3d3d3', linestyle=':', label='Continuous Gaussian Beam Profile')

# Plot Step 1 (Red) and Step 2 (Blue)
ax.plot(X1_trace, Y1_trace, color='#e60000', linewidth=2, label='Initial Step 1 Profile')
ax.fill_between(X1_trace, Y1_trace, color='#e60000', alpha=0.08)
ax.plot(X2_trace, Y2_trace, color='#0066cc', linewidth=2, label='Next Step 2 Profile')
ax.fill_between(X2_trace, Y2_trace, color='#0066cc', alpha=0.08)

# Plot Final Calculated Dest Profile Outputs
ax.plot(final_delta_x_trace, final_delta_y_trace, color='#2eb82e', linewidth=2.5, label='Log-Delta Calculated Final Target')
ax.fill_between(final_delta_x_trace, final_delta_y_trace, color='#2eb82e', alpha=0.08)

# Plot Intermediate Physical Area Sweep Profile (Orange Dash)
ax.plot(X2b_trace, Y2b_trace, color='#ff9900', linewidth=2, linestyle='--', label='Step 2b Log-Area Sweep Profile')
ax.fill_between(X2b_trace, Y2b_trace, color='#ff9900', alpha=0.08)

ax.plot(final_area_x_trace, final_area_y_trace, color='#8a2be2', linewidth=2.5, linestyle='-.', label='Log-Area Calculated Final Target')
ax.fill_between(final_area_x_trace, final_area_y_trace, color='#8a2be2', alpha=0.08)

# Hardware mechanical tracking pins on floor
ax.errorbar(X1, -0.08, xerr=[[left_wing], [right_wing]], fmt='ro', capsize=4, linewidth=1.5)
ax.errorbar(X2, -0.14, xerr=[[left_wing], [right_wing]], fmt='bo', capsize=4, linewidth=1.5)
ax.errorbar(final_target_delta, -0.20, xerr=[[left_wing], [right_wing]], fmt='go', capsize=4, linewidth=2)
ax.errorbar(X2b, -0.26, xerr=[[left_wing], [right_wing]], fmt='yo', color='#ff9900', capsize=4, linewidth=1.5)
ax.errorbar(final_target_area, -0.32, xerr=[[left_wing], [right_wing]], fmt='mo', capsize=4, linewidth=2)

# Footprint text timeline anchors on the layout floor
ax.text(X1, -0.06, f"{left_pct:.1f}% L ◄► {right_pct:.1f}% R", color='#e60000', fontsize=8, ha='center')
ax.text(X2, -0.12, f"{left_pct:.1f}% L ◄► {right_pct:.1f}% R", color='#0066cc', fontsize=8, ha='center')
ax.text(final_target_delta, -0.18, f"{left_pct:.1f}% L ◄► {right_pct:.1f}% R", color='#2eb82e', fontsize=8, ha='center')
ax.text(X2b, -0.24, f"{left_pct:.1f}% L ◄► {right_pct:.1f}% R", color='#ff9900', fontsize=8, ha='center')
ax.text(final_target_area, -0.30, f"{left_pct:.1f}% L ◄► {right_pct:.1f}% R", color='#8a2be2', fontsize=8, ha='center')

ax.axvline(0.0, color='#404040', linestyle='--', alpha=0.6, label='Target Center (0µm)')

# --- FLOATING CALLOUT LABELS POSITIONED CLEANLY UNDERNEATH THE GAUSSIAN CURVE SLOPES ---
text_stage1      = f"Initial Step 1:\n• Max: {max_Y1:.4f}\n• Min: {min_Y1:.4f}\n• Delta: {delta_Y1:.4f}\n• Area: {area_A1:.4f}"
text_stage2      = f"Next Step 2:\n• Max: {max_Y2:.4f}\n• Min: {min_Y2:.4f}\n• Delta: {delta_Y2:.4f}\n• Area: {area_A2:.4f}"
text_delta_tar   = f"Log-Delta Final:\n• Max: {final_delta_max:.4f}\n• Min: {final_delta_min:.4f}\n• Delta: {final_delta_val:.4f}\n• Area: {final_delta_area:.4f}"
text_area3_meas  = f"Log-Area Step 2b:\n• Max: {max_Y2b:.4f}\n• Min: {min_Y2b:.4f}\n• Delta: {delta_Y2b:.4f}\n• Area: {area_A2b:.4f}"
text_area_tar    = f"Log-Area Final:\n• Max: {final_area_max:.4f}\n• Min: {final_area_min:.4f}\n• Delta: {final_area_delta:.4f}\n• Area: {final_area_val:.4f}"

# Historical baseline position readouts nested low on outer wings
ax.text(X1 - 4.5, 0.15, text_stage1, color='#b30000', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7f7', edgecolor='#e60000', alpha=0.9))
ax.text(X2 + 4.5, 0.15, text_stage2, color='#004499', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7faff', edgecolor='#0066cc', alpha=0.9))

# Active processing profiles positioned cleanly below the steep mid-slopes
ax.text(final_target_delta - 5.0, 0.35, text_delta_tar, color='#1f7a1f', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2fff2', edgecolor='#2eb82e', alpha=0.9))

# ==========================================
# PLOTTING UNIFIED ALIGNMENT TELEMETRY
# ==========================================
plt.close('all')

# Create a clean canvas layout
fig, ax = plt.subplots(figsize=(13, 8))

# Continuous Hidden Reference Profile
x_profile = np.linspace(-25, 25, 1000)
y_profile = np.exp(-((x_profile - 0)**2) / (2 * sigma**2))
ax.plot(x_profile, y_profile, color='#d3d3d3', linestyle=':', label='Continuous Gaussian Beam Profile')

# Plot Step 1 (Red)
ax.plot(X1_trace, Y1_trace, color='#e60000', linewidth=2, label='Step 1 Profile')
ax.fill_between(X1_trace, Y1_trace, color='#e60000', alpha=0.08)

# Plot Step 2 (Blue)
ax.plot(X2_trace, Y2_trace, color='#0066cc', linewidth=2, label='Step 2 Profile')
ax.fill_between(X2_trace, Y2_trace, color='#0066cc', alpha=0.08)

# Plot Step 2b (Orange)
ax.plot(X2b_trace, Y2b_trace, color='#ff9900', linewidth=2, label='Step 2b Profile')
ax.fill_between(X2b_trace, Y2b_trace, color='#ff9900', alpha=0.08)

# --- VISUAL OVERLAP REMEDY: SEPARATE HATCH PATTERNS ---
# Log-Delta uses forward hatches (/) and Log-Area uses backward hatches (\)
# Where they merge over the peak, it naturally forms a clean cross-hatched grid (X)
ax.plot(final_delta_x_trace, final_delta_y_trace, color='#2eb82e', linewidth=3.0, label='Log-Delta Calculated Final Target')
ax.fill_between(final_delta_x_trace, final_delta_y_trace, facecolor='none', edgecolor='#2eb82e', hatch='//', alpha=0.25)

ax.plot(final_area_x_trace, final_area_y_trace, color='#8a2be2', linewidth=2.0, linestyle='-.', label='Log-Area Calculated Final Target')
ax.fill_between(final_area_x_trace, final_area_y_trace, facecolor='none', edgecolor='#8a2be2', hatch='\\\\', alpha=0.25)

# Target Core Centroid Indicator Axis (0 µm)
ax.axvline(0, color='#404040', linestyle='--', alpha=0.6)

# --- SEQUENTIAL LEFT-TO-RIGHT METRIC PANEL PLACEMENT ---
text_step1_label = f"Step 1 Metrics:\n• Max: {max_Y1:.4f}\n• Min: {min_Y1:.4f}\n• Delta: {delta_Y1:.4f}\n• Area: {area_A1:.4f}"
text_step2_label = f"Step 2 Metrics:\n• Max: {max_Y2:.4f}\n• Min: {min_Y2:.4f}\n• Delta: {delta_Y2:.4f}\n• Area: {area_A2:.4f}"
text_step2b_meas = f"Step 2b Metrics:\n• Max: {max_Y2b:.4f}\n• Min: {min_Y2b:.4f}\n• Delta: {delta_Y2b:.4f}\n• Area: {area_A2b:.4f}"
text_delta_final = f"Log-Delta Final:\n• Max: {final_delta_max:.4f}\n• Min: {final_delta_min:.4f}\n• Delta: {final_delta_val:.4f}\n• Area: {final_delta_area:.4f}"
text_area_final  = f"Log-Area Final:\n• Max: {final_area_max:.4f}\n• Min: {final_area_min:.4f}\n• Delta: {final_area_delta:.4f}\n• Area: {final_area_val:.4f}"

# Step performance
ax.text(-35.0, 0.40, text_step1_label, color='#b30000', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7f7', edgecolor='#e60000', alpha=0.9))
ax.text(-25.0, 0.40, text_step2_label, color='#004499', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7faff', edgecolor='#0066cc', alpha=0.9))
ax.text(-15.0, 0.40, text_step2b_meas, color='#b36b00', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffaf2', edgecolor='#ff9900', alpha=0.9))
# Calculated optimization targets
ax.text(-30, 0.20, text_delta_final, color='#1f7a1f', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2fff2', edgecolor='#2eb82e', alpha=0.9))
ax.text(-20, 0.20, text_area_final, color='#5c1da3', fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fbf7ff', edgecolor='#8a2be2', alpha=0.9))

# --- VERTICAL CENTERLINES TRUNCATED CLEANLY UNDER RESPECTIVE PEAKS ---
ax.vlines(X1, ymin=0, ymax=max_Y1, color='#b30000', linestyle='--', alpha=0.4)
ax.vlines(X2, ymin=0, ymax=max_Y2, color='#004499', linestyle='--', alpha=0.4)
ax.vlines(final_target_delta, ymin=0, ymax=final_delta_max, color='#1f7a1f', linestyle='--', alpha=0.5)
ax.vlines(X2b, ymin=0, ymax=max_Y2b, color='#ff9900', linestyle='--', alpha=0.4)
ax.vlines(final_target_area, ymin=0, ymax=final_area_max, color='#5c1da3', linestyle='--', alpha=0.5)

# --- SYSTEM METRIC PERFORMANCE BOX ---
text_panel_deltas = (f"Multi-Stage Delta Analysis:\n"
                     f"                 [Stage 1➔2]  [Stage 2➔2b]\n"
                     f"• Δ Max Value  :  {delta12_max:+.4f}      {delta2b2_max:+.4f}\n"
                     f"• Δ Min Value  :  {delta12_min:+.4f}      {delta2b2_min:+.4f}\n"
                     f"• Δ Amplitude  :  {delta12_delta:+.4f}      {delta2b2_delta:+.4f}\n"
                     f"• Δ Sweep Area :  {delta12_area:+.4f}      {delta2b2_area:+.4f}\n\n"
                     f"Hardware Profile Coordinates:\n"
                     f"• MEMs Vibration Asymmetry : {left_pct:.1f}% Left ◄► {right_pct:.1f}% Right\n"
                     f"• Step 1 Position Vector   : {X1:.3f} µm\n"
                     f"• Step 2 Position Vector   : {X2:.3f} µm (Move Dist: {step_dist_1_to_2:.3f}µm)\n"
                     f"• Step 2b Position Vector  : {X2b:.3f} µm (Move Dist: {step_dist_2_to_2b:.3f}µm)\n\n"
                     f"Ultimate Optimization Targets:\n"
                     f"• Log-Delta Final Destination : {final_target_delta:.3f} µm\n"
                     f"• Log-Area Final Destination  : {final_target_area:.3f} µm")

props_deltas = dict(boxstyle='round,pad=0.5', facecolor='#f7fff2', edgecolor='#4da61a', alpha=0.95)
# FIXED: Relocated main analytics summary panel back to the upper left (x=0.02, ha='left')
ax.text(0.02, 0.95, text_panel_deltas, transform=ax.transAxes, fontsize=9.5, verticalalignment='top', horizontalalignment='left', bbox=props_deltas, fontfamily='monospace')

# Interface Formatting Configuration
ax.set_title('Active Alignment Tracking for Gaussian Peak', fontsize=11, fontweight='bold')
ax.set_xlabel('Spatial Coordinate Positioning (µm)')
ax.set_ylabel('Coupled Intensity Efficiency')
ax.set_xlim(-40, 25)

# Enforce strict 0.0 to 1.2 viewport framing on the data face
ax.set_ylim(0.0, 1.2)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
