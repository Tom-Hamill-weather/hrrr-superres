"""
Check if HRRR cell center spacing matches the computed edge-to-edge spacing
"""
import numpy as np
from generate_adaptive_grid import DataLoader

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in meters"""
    R = 6371000
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

print("Loading HRRR grid...")
loader = DataLoader()
loader.load_all()

lats = loader.hrrr_grid['lats']
lons = loader.hrrr_grid['lons']

# Focus on Catalina region
i, j = 413, 256

print(f"\nHRRR cell at ({i}, {j}):")
print(f"  Center: ({lats[i,j]:.6f}, {lons[i,j]:.6f})")

# Compute spacing to neighbors (as done in grid generation)
dx_to_east = haversine_distance(lats[i,j], lons[i,j], lats[i,j+1], lons[i,j+1])
dy_to_north = haversine_distance(lats[i,j], lons[i,j], lats[i+1,j], lons[i+1,j])

print(f"  Computed spacing (edge-to-edge):")
print(f"    To east: {dx_to_east:.2f}m")
print(f"    To north: {dy_to_north:.2f}m")

# Actual center-to-center distance
center_to_center_x = haversine_distance(lats[i,j], lons[i,j], lats[i,j+1], lons[i,j+1])
center_to_center_y = haversine_distance(lats[i,j], lons[i,j], lats[i+1,j], lons[i+1,j])

print(f"  Actual center-to-center:")
print(f"    To east: {center_to_center_x:.2f}m")
print(f"    To north: {center_to_center_y:.2f}m")

# They should be the same!
print(f"\n  Match: {abs(dx_to_east - center_to_center_x) < 0.01}")

# Now check the implication for fine grid
n_fine = 32
print(f"\nFine grid (n_fine={n_fine}):")
print(f"  Expected spacing at tier 0: {dx_to_east / (n_fine-1):.2f}m")

# My implementation creates offsets from -dx/2 to +dx/2
offsets_x = np.linspace(-dx_to_east/2, dx_to_east/2, n_fine)
print(f"  Offsets span: {offsets_x[0]:.2f}m to {offsets_x[-1]:.2f}m")
print(f"  Spacing: {offsets_x[1] - offsets_x[0]:.2f}m")

# For cell at (i,j), fine points in X direction are at:
cell_center_x = lons[i,j]
fine_points_x_cell_ij = cell_center_x + (offsets_x / (111000 * np.cos(np.radians(lats[i,j]))))

# For cell at (i, j+1), fine points are:
offsets_x_next = np.linspace(-center_to_center_x/2, center_to_center_x/2, n_fine)
cell_center_x_next = lons[i,j+1]
fine_points_x_cell_ij1 = cell_center_x_next + (offsets_x_next / (111000 * np.cos(np.radians(lats[i,j+1]))))

# Check boundary alignment
last_point_ij = fine_points_x_cell_ij[-1]
first_point_ij1 = fine_points_x_cell_ij1[0]

gap_deg = first_point_ij1 - last_point_ij
gap_m = gap_deg * 111000 * np.cos(np.radians(lats[i,j]))

print(f"\nBoundary check between cells ({i},{j}) and ({i},{j+1}):")
print(f"  Last point of cell ({i},{j}): {last_point_ij:.8f}")
print(f"  First point of cell ({i},{j+1}): {first_point_ij1:.8f}")
print(f"  Gap: {gap_m:.2f}m")

if abs(gap_m) < 1:
    print("  ✓ Boundary alignment is good!")
elif gap_m > 0:
    print(f"  ⚠ GAP of {gap_m:.2f}m between cells")
else:
    print(f"  ⚠ OVERLAP of {abs(gap_m):.2f}m between cells")
