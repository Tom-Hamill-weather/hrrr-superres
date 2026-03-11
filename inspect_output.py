"""
Utility script to inspect and summarize the adaptive grid output
"""
import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import config

def inspect_netcdf(filename):
    """Inspect the adaptive grid netCDF file"""
    filepath = os.path.join(config.OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        print("  Run generate_adaptive_grid.py first")
        return False

    print("="*70)
    print(" ADAPTIVE GRID OUTPUT INSPECTION")
    print("="*70)
    print(f"\nFile: {filepath}")
    print(f"Size: {os.path.getsize(filepath) / (1024**3):.2f} GB")

    with nc.Dataset(filepath, 'r') as ds:
        # Global attributes
        print("\n" + "-"*70)
        print(" GLOBAL ATTRIBUTES")
        print("-"*70)
        for attr in ds.ncattrs():
            print(f"  {attr}: {getattr(ds, attr)}")

        # Dimensions
        print("\n" + "-"*70)
        print(" DIMENSIONS")
        print("-"*70)
        for dim in ds.dimensions.values():
            print(f"  {dim.name}: {len(dim)}")

        # Variables
        print("\n" + "-"*70)
        print(" VARIABLES")
        print("-"*70)
        for var_name, var in ds.variables.items():
            print(f"\n  {var_name}:")
            print(f"    Shape: {var.shape}")
            print(f"    Type: {var.dtype}")
            if hasattr(var, 'units'):
                print(f"    Units: {var.units}")
            if hasattr(var, 'long_name'):
                print(f"    Long name: {var.long_name}")

        # Load data for analysis
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]

        # Check if tier variable exists
        has_tiers = 'tier' in ds.variables
        if has_tiers:
            tiers = ds.variables['tier'][:]
        else:
            print("\n⚠ Warning: No tier information in file")
            tiers = None

        # Check if metadata variable exists (older files may have it)
        has_metadata = 'metadata' in ds.variables
        if has_metadata:
            metadata = ds.variables['metadata'][:]

        # Summary statistics
        print("\n" + "-"*70)
        print(" SUMMARY STATISTICS")
        print("-"*70)

        print(f"\nTotal points: {len(lats):,}")

        print("\nGeographic extent:")
        print(f"  Latitude:  {lats.min():.2f}° to {lats.max():.2f}°")
        print(f"  Longitude: {lons.min():.2f}° to {lons.max():.2f}°")

        if has_tiers:
            print("\nTier distribution:")
            for tier in [0, 1, 2, 3, 4, 5]:
                count = (tiers == tier).sum()
                if count > 0:
                    pct = 100 * count / len(tiers)
                    print(f"  Tier {tier}: {count:,} points ({pct:.1f}%)")

        if has_metadata:
            print("\nMetadata criteria met:")
            criteria = {
                0: 'Urban',
                1: 'Suburban',
                2: 'Coastline',
                3: 'Lake',
                4: 'Ski Resort',
                5: 'Golf Course',
                6: 'Park',
                7: 'Highway',
                8: 'High Terrain Var'
            }

            for bit, name in criteria.items():
                count = ((metadata & (1 << bit)) > 0).sum()
                pct = 100 * count / len(metadata)
                print(f"  {name}: {count:,} points ({pct:.1f}%)")

        # Tier-specific statistics
        if has_tiers:
            print("\n" + "-"*70)
            print(" TIER-SPECIFIC STATISTICS")
            print("-"*70)

            for tier in [0, 1, 2, 3, 4, 5]:
                tier_mask = (tiers == tier)

                if tier_mask.sum() == 0:
                    continue

                print(f"\nTier {tier} ({config.TIER_RESOLUTIONS[tier]}m resolution):")
                print(f"  Total points: {tier_mask.sum():,}")

                if has_metadata:
                    tier_metadata = metadata[tier_mask]

                    # Count criteria for this tier
                    tier_criteria_counts = {}
                    for bit, name in criteria.items():
                        count = ((tier_metadata & (1 << bit)) > 0).sum()
                        if count > 0:
                            tier_criteria_counts[name] = count

                    if tier_criteria_counts:
                        print("  Top criteria:")
                        for name, count in sorted(tier_criteria_counts.items(),
                                                 key=lambda x: x[1], reverse=True)[:5]:
                            pct = 100 * count / tier_mask.sum()
                            print(f"    - {name}: {count:,} ({pct:.1f}%)")

        # Geographic distribution
        print("\n" + "-"*70)
        print(" GEOGRAPHIC DISTRIBUTION")
        print("-"*70)

        # Divide CONUS into regions
        regions = {
            'Northeast': (37, 48, -80, -67),
            'Southeast': (25, 37, -92, -75),
            'Midwest': (37, 49, -104, -80),
            'Southwest': (25, 42, -125, -104),
            'West': (32, 49, -125, -104),
        }

        for region_name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
            mask = ((lats >= lat_min) & (lats <= lat_max) &
                   (lons >= lon_min) & (lons <= lon_max))
            count = mask.sum()
            if count > 0:
                pct = 100 * count / len(lats)
                print(f"  {region_name}: {count:,} points ({pct:.1f}%)")

    print("\n" + "="*70)
    print(" INSPECTION COMPLETE")
    print("="*70)

    return True

def plot_tier_distribution(filename):
    """Create a simple plot showing tier distribution"""
    filepath = os.path.join(config.OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False

    print("\nCreating tier distribution plot...")

    with nc.Dataset(filepath, 'r') as ds:
        if 'tier' not in ds.variables:
            print("  ⚠ No tier information in file - skipping tier distribution plot")
            return False
        tiers = ds.variables['tier'][:]

    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    tier_counts = [((tiers == t).sum()) for t in [0, 1, 2, 3, 4, 5]]
    tier_labels = [f'Tier {t}\n({config.TIER_RESOLUTIONS[t]}m)'
                   for t in [0, 1, 2, 3, 4, 5]]

    colors = ['#c0392b', '#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']

    ax1.pie(tier_counts, labels=tier_labels, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Tier Distribution by Percentage', fontsize=12, fontweight='bold')

    # Bar chart
    ax2.bar(tier_labels, tier_counts, color=colors)
    ax2.set_ylabel('Number of Points', fontsize=11)
    ax2.set_title('Tier Distribution by Count', fontsize=12, fontweight='bold')
    ax2.ticklabel_format(axis='y', style='plain')
    for i, (label, count) in enumerate(zip(tier_labels, tier_counts)):
        ax2.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Adaptive Grid Tier Distribution\nTotal Points: {len(tiers):,}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(config.OUTPUT_DIR, 'tier_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Tier distribution plot saved: {output_path}")

    return True

def export_sample_points(filename, n_samples=1000):
    """Export a random sample of points to CSV for inspection"""
    filepath = os.path.join(config.OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False

    print(f"\nExporting {n_samples:,} random sample points to CSV...")

    with nc.Dataset(filepath, 'r') as ds:
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]

        has_tiers = 'tier' in ds.variables
        if has_tiers:
            tiers = ds.variables['tier'][:]
        else:
            tiers = None

        has_metadata = 'metadata' in ds.variables
        if has_metadata:
            metadata = ds.variables['metadata'][:]

    # Random sample
    total_points = len(lats)
    if n_samples > total_points:
        n_samples = total_points

    indices = np.random.choice(total_points, n_samples, replace=False)

    # Export to CSV
    output_path = os.path.join(config.OUTPUT_DIR, 'sample_points.csv')
    with open(output_path, 'w') as f:
        if has_metadata:
            f.write('latitude,longitude,tier,metadata,urban,suburban,coastline,lake,ski,golf,park,highway,terrain_var\n')
        elif has_tiers:
            f.write('latitude,longitude,tier\n')
        else:
            f.write('latitude,longitude\n')

        for idx in indices:
            lat = lats[idx]
            lon = lons[idx]
            tier = tiers[idx] if has_tiers else None

            if has_metadata:
                meta = metadata[idx]
                # Decode metadata bits
                bits = [(meta & (1 << i)) > 0 for i in range(9)]
                f.write(f'{lat:.6f},{lon:.6f},{tier},{meta},'
                       f'{int(bits[0])},{int(bits[1])},{int(bits[2])},'
                       f'{int(bits[3])},{int(bits[4])},{int(bits[5])},'
                       f'{int(bits[6])},{int(bits[7])},{int(bits[8])}\n')
            elif has_tiers:
                f.write(f'{lat:.6f},{lon:.6f},{tier}\n')
            else:
                f.write(f'{lat:.6f},{lon:.6f}\n')

    print(f"✓ Sample points exported: {output_path}")
    print(f"  {n_samples:,} points sampled from {total_points:,} total")

    return True

def main():
    """Main inspection routine"""
    nc_file = 'adaptive_grid_SPARSE.nc'

    print("\n" + "="*70)
    print(" ADAPTIVE GRID OUTPUT INSPECTOR")
    print("="*70)

    # Inspect netCDF
    success = inspect_netcdf(nc_file)

    if success:
        # Create additional visualizations
        plot_tier_distribution(nc_file)

        # Export sample
        export_sample_points(nc_file, n_samples=10000)

        print("\n" + "="*70)
        print(" ALL INSPECTIONS COMPLETE")
        print("="*70)
        print(f"\nOutputs in: {config.OUTPUT_DIR}")
        print("  - adaptive_grid_SPARSE.nc (main output)")
        print("  - adaptive_grid_density_discrete.png (density map)")
        print("  - tier_distribution.png (tier statistics)")
        print("  - sample_points.csv (random sample for inspection)")
        print("="*70)

        return 0
    else:
        print("\n✗ Inspection failed - check that output file exists")
        return 1

if __name__ == '__main__':
    sys.exit(main())
