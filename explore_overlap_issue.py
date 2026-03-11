"""
Explore potential overlap issues at HRRR cell boundaries
"""
import numpy as np
from generate_adaptive_grid import DataLoader
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

loader = DataLoader()
loader.load_all()

lats = loader.hrrr_grid['lats']
lons = loader.hrrr_grid['lons']

# Create projection
lat_ref = (lats.min() + lats.max()) / 2
lon_ref = (lons.min() + lons.max()) / 2
proj = Basemap(
    projection='lcc',
    lat_0=lat_ref,
    lon_0=lon_ref,
    lat_1=lat_ref - 5,
    lat_2=lat_ref + 5,
    llcrnrlat=lats.min(),
    urcrnrlat=lats.max(),
    llcrnrlon=lons.min(),
    urcrnrlon=lons.max(),
    resolution=None
)

x, y = proj(lons, lats)

# Analyze two adjacent cells
i, j = 413, 256

print("="*70)
print("ANALYZING ADJACENT CELLS FOR OVERLAP")
print("="*70)

# Cell A: (i, j)
# Cell B: (i, j+1)

# Get vectors and distances
dx_east_A = x[i, j+1] - x[i, j]
dy_east_A = y[i, j+1] - y[i, j]
dist_A = np.sqrt(dx_east_A**2 + dy_east_A**2)

dx_east_B = x[i, j+2] - x[i, j+1]
dy_east_B = y[i, j+2] - y[i, j+1]
dist_B = np.sqrt(dx_east_B**2 + dy_east_B**2)

print(f"\nCell A at ({i},{j}):")
print(f"  Center: x={x[i,j]:.2f}, y={y[i,j]:.2f}")
print(f"  Distance to eastern neighbor: {dist_A:.2f}m")

print(f"\nCell B at ({i},{j+1}):")
print(f"  Center: x={x[i,j+1]:.2f}, y={y[i,j+1]:.2f}")
print(f"  Distance to eastern neighbor: {dist_B:.2f}m")

# Using CURRENT implementation (aligned with projection axes)
n_fine = 32
print(f"\n\nCURRENT IMPLEMENTATION (projection-aligned):")
print(f"="*70)

# Cell A fine grid
x_spacing_A = dist_A / n_fine
x_offsets_A = np.arange(n_fine) * x_spacing_A - dist_A / 2

print(f"\nCell A fine grid (X-direction only, aligned with X-axis):")
print(f"  First point offset: {x_offsets_A[0]:.2f}m")
print(f"  Last point offset: {x_offsets_A[-1]:.2f}m")
print(f"  Span: {x_offsets_A[-1] - x_offsets_A[0]:.2f}m")

# Absolute positions of Cell A's fine grid
x_positions_A = x[i,j] + x_offsets_A
last_x_A = x_positions_A[-1]

print(f"  Last point absolute X: {last_x_A:.2f}m")

# Cell B fine grid
x_spacing_B = dist_B / n_fine
x_offsets_B = np.arange(n_fine) * x_spacing_B - dist_B / 2
x_positions_B = x[i,j+1] + x_offsets_B
first_x_B = x_positions_B[0]

print(f"\nCell B fine grid (X-direction only, aligned with X-axis):")
print(f"  First point offset: {x_offsets_B[0]:.2f}m")
print(f"  First point absolute X: {first_x_B:.2f}m")

gap_current = first_x_B - last_x_A
print(f"\nGap between cells (Cell B first - Cell A last):")
print(f"  {gap_current:.2f}m")

if gap_current < 0:
    print(f"  ⚠ OVERLAP of {abs(gap_current):.2f}m!")
elif gap_current < 50:
    print(f"  ⚠ Very small gap - potential for rounding errors")
else:
    print(f"  ✓ Adequate separation")

# But wait - cells are ROTATED!
print(f"\n\nISSUE: ROTATION COMPLICATES BOUNDARY CHECK")
print(f"="*70)

east_unit_x_A = dx_east_A / dist_A
east_unit_y_A = dy_east_A / dist_A

print(f"\nCell A is rotated {np.degrees(np.arctan2(dy_east_A, dx_east_A)):.3f}° from X-axis")
print(f"  East unit vector: ({east_unit_x_A:.6f}, {east_unit_y_A:.6f})")

# The "last point" of Cell A in the X-direction is NOT the actual easternmost point
# because the cell is rotated. The actual easternmost point is at a corner.

# Let's compute the actual boundary positions
print(f"\n\nACTUAL BOUNDARY POSITIONS (accounting for rotation):")
print(f"="*70)

# For Cell A, the eastern edge runs from southwest to northwest of eastern neighbor
# Southwest corner of eastern edge
sw_corner_A_x = x[i,j] + (n_fine-1) * x_spacing_A * east_unit_x_A / 2
sw_corner_A_y = y[i,j] + (n_fine-1) * x_spacing_A * east_unit_y_A / 2 - dist_A / 2

# Actually, let me think about this differently...
# The fine grid points at the eastern edge of Cell A are at indices (i_fine=anything, j_fine=31)
# Their positions are determined by BOTH the x and y offsets

print("\nThis is getting complex. The key issues are:")
print("1. Cells are rotated relative to projection axes")
print("2. Fine grid is created aligned with axes, not cell edges")
print("3. This creates misalignment at boundaries")
print()
print("Let me visualize a specific boundary...")

# Create a detailed plot of the boundary region
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Simulate fine grid for both cells
i_indices = np.arange(n_fine)
j_indices = np.arange(n_fine)
ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')

# Cell A
x_off_A = ii * x_spacing_A - dist_A / 2
y_spacing_A = dist_A / n_fine  # Assuming square cells
y_off_A = jj * y_spacing_A - dist_A / 2
x_fine_A = x[i,j] + x_off_A
y_fine_A = y[i,j] + y_off_A

# Cell B
x_off_B = ii * x_spacing_B - dist_B / 2
y_spacing_B = dist_B / n_fine
y_off_B = jj * y_spacing_B - dist_B / 2
x_fine_B = x[i,j+1] + x_off_B
y_fine_B = y[i,j+1] + y_off_B

# Plot - zoomed to boundary
axes[0].scatter(x_fine_A.ravel(), y_fine_A.ravel(), s=10, c='blue', alpha=0.5, label='Cell A')
axes[0].scatter(x_fine_B.ravel(), y_fine_B.ravel(), s=10, c='red', alpha=0.5, label='Cell B')
axes[0].scatter([x[i,j], x[i,j+1]], [y[i,j], y[i,j+1]], s=100, c='black', marker='+', linewidths=2, label='Centers')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].set_title('Full cells (projection-aligned grid)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Zoom to boundary region
boundary_x = (x[i,j] + x[i,j+1]) / 2
x_zoom_half = 200  # meters
y_zoom_half = 200
axes[1].scatter(x_fine_A.ravel(), y_fine_A.ravel(), s=50, c='blue', alpha=0.6, label='Cell A', marker='o')
axes[1].scatter(x_fine_B.ravel(), y_fine_B.ravel(), s=50, c='red', alpha=0.6, label='Cell B', marker='s')
axes[1].axvline(boundary_x, color='green', linestyle='--', linewidth=2, label='Midpoint', alpha=0.7)
axes[1].set_xlim(boundary_x - x_zoom_half, boundary_x + x_zoom_half)
axes[1].set_ylim(y[i,j] - y_zoom_half, y[i,j] + y_zoom_half)
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Y (m)')
axes[1].set_title('Boundary region (zoomed)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('output/overlap_exploration.png', dpi=150)
print(f"\n✓ Visualization saved: output/overlap_exploration.png")
print("\nThis shows how projection-aligned fine grids create patterns at boundaries")
print("when the HRRR cells themselves are rotated.")
