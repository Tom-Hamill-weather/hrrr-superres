"""
Test creating fine grid aligned with HRRR cell's actual orientation
"""
import numpy as np
from generate_adaptive_grid import DataLoader
from mpl_toolkits.basemap import Basemap

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

# Project to x, y
x, y = proj(lons, lats)

# Focus on one cell
i, j = 413, 256

print(f"HRRR cell ({i}, {j}):")
print(f"  Center: ({lats[i,j]:.6f}, {lons[i,j]:.6f})")
print(f"  Projected: ({x[i,j]:.1f}, {y[i,j]:.1f})")

# Get the four corners by looking at neighbors
# Vector to east neighbor
dx_east = x[i, j+1] - x[i, j]
dy_east = y[i, j+1] - y[i, j]
dist_east = np.sqrt(dx_east**2 + dy_east**2)

# Vector to north neighbor
dx_north = x[i+1, j] - x[i, j]
dy_north = y[i+1, j] - y[i, j]
dist_north = np.sqrt(dx_north**2 + dy_north**2)

print(f"\n  Vector to eastern neighbor:")
print(f"    dx={dx_east:.2f}m, dy={dy_east:.2f}m")
print(f"    distance={dist_east:.2f}m")
print(f"    angle from X-axis: {np.degrees(np.arctan2(dy_east, dx_east)):.3f}°")

print(f"\n  Vector to northern neighbor:")
print(f"    dx={dx_north:.2f}m, dy={dy_north:.2f}m")
print(f"    distance={dist_north:.2f}m")
print(f"    angle from X-axis: {np.degrees(np.arctan2(dy_north, dx_north)):.3f}°")

# Check orthogonality
dot_product = dx_east * dx_north + dy_east * dy_north
angle_between = np.degrees(np.arccos(dot_product / (dist_east * dist_north)))
print(f"\n  Angle between vectors: {angle_between:.3f}° (should be 90° if orthogonal)")

# OLD approach: offsets aligned with projection axes
n_fine = 32
old_x_offsets = np.arange(n_fine) * (dist_east / n_fine) - dist_east / 2
old_y_offsets = np.arange(n_fine) * (dist_north / n_fine) - dist_north / 2

print(f"\n\nOLD APPROACH (aligned with projection axes):")
print(f"  Creates offsets in X and Y directions independently")
print(f"  Assumes HRRR cell edges are parallel to X and Y axes")
print(f"  But actual HRRR cell is rotated {np.degrees(np.arctan2(dy_east, dx_east)):.3f}° from X-axis")

# NEW approach: offsets aligned with HRRR cell orientation
# Create unit vectors along cell edges
east_unit_x = dx_east / dist_east
east_unit_y = dy_east / dist_east
north_unit_x = dx_north / dist_north
north_unit_y = dy_north / dist_north

print(f"\n\nNEW APPROACH (aligned with HRRR cell orientation):")
print(f"  East unit vector: ({east_unit_x:.6f}, {east_unit_y:.6f})")
print(f"  North unit vector: ({north_unit_x:.6f}, {north_unit_y:.6f})")

# For a fine grid point at index (i_fine, j_fine):
# offset_x = (i_fine - n_fine/2) * (dist_east/n_fine) * east_unit_x + (j_fine - n_fine/2) * (dist_north/n_fine) * north_unit_x
# offset_y = (i_fine - n_fine/2) * (dist_east/n_fine) * east_unit_y + (j_fine - n_fine/2) * (dist_north/n_fine) * north_unit_y

# Example: corner point (0, 0) - southwest corner
i_fine, j_fine = 0, 0
offset_x_new = (i_fine * (dist_east/n_fine) - dist_east/2) * east_unit_x + \
               (j_fine * (dist_north/n_fine) - dist_north/2) * north_unit_x
offset_y_new = (i_fine * (dist_east/n_fine) - dist_east/2) * east_unit_y + \
               (j_fine * (dist_north/n_fine) - dist_north/2) * north_unit_y

offset_x_old = -dist_east/2
offset_y_old = -dist_north/2

print(f"\n\nSouthwest corner (0,0) offsets:")
print(f"  Old: ({offset_x_old:.2f}, {offset_y_old:.2f})")
print(f"  New: ({offset_x_new:.2f}, {offset_y_new:.2f})")
print(f"  Difference: ({abs(offset_x_new-offset_x_old):.2f}, {abs(offset_y_new-offset_y_old):.2f})")

# The key insight:
print(f"\n\n{'='*70}")
print("KEY INSIGHT:")
print("="*70)
print("The OLD approach creates a fine grid aligned with the projection's")
print("X and Y axes, which are NOT aligned with the HRRR grid orientation.")
print()
print("The NEW approach creates a fine grid aligned with the HRRR cell's")
print("actual edges (directions to east and north neighbors).")
print()
print("This ensures adjacent HRRR cells have fine grids that align perfectly")
print("at their shared boundaries, eliminating the rotation artifacts.")
