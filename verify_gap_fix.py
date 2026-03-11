"""
Verify that gaps between HRRR cells have been eliminated
by examining point positions along transects
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import config
import os

def load_points_in_region(nc_file, lat_center, lon_center, box_size_km=10):
    """Load points in a box around center"""
    lat_half = (box_size_km / 2.0) / 111.0
    lon_half = (box_size_km / 2.0) / (111.0 * np.cos(np.radians(lat_center)))

    lat_min = lat_center - lat_half
    lat_max = lat_center + lat_half
    lon_min = lon_center - lon_half
    lon_max = lon_center + lon_half

    with nc.Dataset(nc_file, 'r') as ds:
        all_lats = ds.variables['latitude'][:]
        all_lons = ds.variables['longitude'][:]
        all_tiers = ds.variables['tier'][:]

        in_box = (
            (all_lats >= lat_min) & (all_lats <= lat_max) &
            (all_lons >= lon_min) & (all_lons <= lon_max)
        )

        lats = all_lats[in_box]
        lons = all_lons[in_box]
        tiers = all_tiers[in_box]

    return lats, lons, tiers


def analyze_point_spacing(lats, lons, tiers, lat_center, lon_center):
    """Analyze spacing between adjacent points"""

    print(f"\n{'='*70}")
    print(f" POINT SPACING ANALYSIS")
    print(f"{'='*70}")
    print(f"\nRegion: ({lat_center:.4f}°N, {lon_center:.4f}°W)")
    print(f"Total points: {len(lats):,}")

    # Focus on Tier 0 points (300m resolution)
    tier0_mask = (tiers == 0)
    tier0_lats = lats[tier0_mask]
    tier0_lons = lons[tier0_mask]

    print(f"\nTier 0 points (300m resolution): {len(tier0_lats):,}")

    if len(tier0_lats) < 10:
        print("Not enough Tier 0 points for analysis")
        return

    # Select a narrow latitude band to analyze longitude spacing
    lat_band = 0.01  # ~1.1km band
    in_band = np.abs(tier0_lats - lat_center) < lat_band
    band_lons = np.sort(tier0_lons[in_band])

    print(f"\nPoints in narrow latitude band (±{lat_band*111:.1f}km): {len(band_lons)}")

    if len(band_lons) < 10:
        print("Not enough points in band for spacing analysis")
        return

    # Compute spacing between adjacent points
    spacings_deg = np.diff(band_lons)
    spacings_km = spacings_deg * 111 * np.cos(np.radians(lat_center))
    spacings_m = spacings_km * 1000

    print(f"\nSpacing statistics (meters):")
    print(f"  Min:    {np.min(spacings_m):.1f}m")
    print(f"  Max:    {np.max(spacings_m):.1f}m")
    print(f"  Mean:   {np.mean(spacings_m):.1f}m")
    print(f"  Median: {np.median(spacings_m):.1f}m")
    print(f"  Std:    {np.std(spacings_m):.1f}m")

    # Check for large gaps
    gap_threshold_m = 350  # More than 1.15× the 300m spacing
    large_gaps = spacings_m > gap_threshold_m

    print(f"\nLarge gaps (>{gap_threshold_m}m):")
    print(f"  Count: {np.sum(large_gaps)}")
    if np.any(large_gaps):
        print(f"  Sizes: {spacings_m[large_gaps]}")

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of all spacings
    ax1.hist(spacings_m, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(300, color='red', linestyle='--', linewidth=2, label='Target: 300m')
    ax1.set_xlabel('Spacing (meters)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Point Spacing Distribution\n(Tier 0, longitude transect)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Zoomed histogram focusing on 0-600m range
    ax2.hist(spacings_m[spacings_m < 600], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(300, color='red', linestyle='--', linewidth=2, label='Target: 300m')
    ax2.set_xlabel('Spacing (meters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Point Spacing Distribution (zoomed)\n(0-600m range)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(config.OUTPUT_DIR, 'point_spacing_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Spacing histogram saved: {output_file}")

    # Print sample of spacings
    print(f"\nFirst 20 spacings (meters):")
    for i, spacing in enumerate(spacings_m[:20]):
        print(f"  {i:3d}: {spacing:6.1f}m", end='')
        if spacing > gap_threshold_m:
            print(" ← LARGE GAP!", end='')
        print()

    # Check for systematic patterns
    print(f"\nLooking for systematic patterns...")
    # Group spacings by approximate value
    bins = np.arange(0, np.max(spacings_m) + 50, 50)
    hist, bin_edges = np.histogram(spacings_m, bins=bins)

    print(f"\nSpacing distribution by 50m bins:")
    for i, (count, bin_start) in enumerate(zip(hist, bin_edges[:-1])):
        if count > 0:
            pct = 100 * count / len(spacings_m)
            print(f"  {bin_start:5.0f}-{bin_start+50:5.0f}m: {count:4d} gaps ({pct:5.1f}%)")


if __name__ == '__main__':
    nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_GEN2.nc')

    # Analyze Southern California coast (where gaps were visible)
    lat_center = 33.4
    lon_center = -118.4

    print(f"\n{'='*70}")
    print(f" VERIFYING GAP FIX")
    print(f"{'='*70}")
    print(f"\nLocation: Southern California coast")
    print(f"  Latitude: {lat_center}°N")
    print(f"  Longitude: {lon_center}°W")

    # Load points in 10km box
    lats, lons, tiers = load_points_in_region(nc_file, lat_center, lon_center, box_size_km=10)

    # Analyze spacing
    analyze_point_spacing(lats, lons, tiers, lat_center, lon_center)

    print(f"\n{'='*70}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
