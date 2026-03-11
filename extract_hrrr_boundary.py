"""
Extract HRRR domain boundary from a GRIB2 file.
This creates boundary files used by generate_adaptive_grid_SUPERFAST.py
to filter out points outside the HRRR Lambert Conformal projection domain.

Usage:
    python extract_hrrr_boundary.py [path_to_hrrr_grib2_file]

If no file is specified, uses HRRR_IC2025092912_APCP_lead6h.grib2 in current directory.
"""

import sys
import numpy as np
import pygrib

def extract_hrrr_boundary(grib_file):
    """Extract boundary coordinates from HRRR GRIB2 file"""
    print(f"Reading HRRR grid from: {grib_file}")

    grbs = pygrib.open(grib_file)
    grb = grbs[1]
    lats, lons = grb.latlons()

    print(f"  Grid shape: {lats.shape}")
    print(f"  Lat range: {lats.min():.6f} to {lats.max():.6f}")
    print(f"  Lon range: {lons.min():.6f} to {lons.max():.6f}")

    # Extract boundary coordinates (perimeter of the grid)
    # Walk around the edges: top -> right -> bottom -> left
    ny, nx = lats.shape

    boundary_lats = np.concatenate([
        lats[0, :],        # top edge (left to right)
        lats[1:-1, -1],    # right edge (top to bottom, excluding corners)
        lats[-1, ::-1],    # bottom edge (right to left)
        lats[-2:0:-1, 0]   # left edge (bottom to top, excluding corners)
    ])

    boundary_lons = np.concatenate([
        lons[0, :],        # top edge
        lons[1:-1, -1],    # right edge
        lons[-1, ::-1],    # bottom edge
        lons[-2:0:-1, 0]   # left edge
    ])

    # Save boundary files
    np.save('hrrr_boundary_lats.npy', boundary_lats)
    np.save('hrrr_boundary_lons.npy', boundary_lons)

    print(f"\n✓ Saved boundary with {len(boundary_lats)} points:")
    print(f"  hrrr_boundary_lats.npy")
    print(f"  hrrr_boundary_lons.npy")

    grbs.close()
    return boundary_lats, boundary_lons


if __name__ == '__main__':
    # Get GRIB file from command line or use default
    if len(sys.argv) > 1:
        grib_file = sys.argv[1]
    else:
        grib_file = 'HRRR_IC2025092912_APCP_lead6h.grib2'

    try:
        extract_hrrr_boundary(grib_file)
    except FileNotFoundError:
        print(f"\nError: File not found: {grib_file}")
        print("Please provide a valid HRRR GRIB2 file path.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
