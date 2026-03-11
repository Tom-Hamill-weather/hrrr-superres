"""
Inspect metadata from adaptive grid NetCDF file
Shows which features caused each point to have its resolution
"""
import numpy as np
import netCDF4 as nc
import sys

# Metadata bit flags (must match generate_adaptive_grid_SPARSE_v2.py)
METADATA_FLAGS = {
    0: 'Coastline',
    1: 'Lake',
    2: 'Ski resort',
    3: 'High-density urban',
    4: 'Suburban',
    5: 'Small town',
    6: 'Major road',
    7: 'National park',
    8: 'National forest (deprecated)',
    9: 'Extreme terrain',
    10: 'High terrain',
    11: 'Moderate terrain',
    12: 'Background',
    13: 'Hiking trail',
}

def decode_metadata(metadata_value):
    """Decode metadata bit flags"""
    features = []
    for bit, name in METADATA_FLAGS.items():
        if metadata_value & (1 << bit):
            features.append(name)
    return features

def inspect_metadata(nc_file):
    """Inspect metadata in NetCDF file"""
    print("="*70)
    print(" ADAPTIVE GRID METADATA INSPECTION")
    print("="*70)
    print(f"\nFile: {nc_file}\n")

    with nc.Dataset(nc_file, 'r') as ds:
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]
        tiers = ds.variables['tier'][:]
        metadata = ds.variables['metadata'][:]

        print(f"Total points: {len(lats):,}")
        print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}")

        # Tier distribution
        print("\nTier distribution:")
        for tier in range(6):
            count = (tiers == tier).sum()
            if count > 0:
                pct = 100 * count / len(tiers)
                print(f"  Tier {tier}: {count:,} points ({pct:.1f}%)")

        # Metadata statistics
        print("\nFeature statistics (multiple features can apply to one point):")
        for bit, name in METADATA_FLAGS.items():
            count = ((metadata & (1 << bit)) > 0).sum()
            if count > 0:
                pct = 100 * count / len(metadata)
                print(f"  {name}: {count:,} points ({pct:.1f}%)")

        # Sample points from each tier
        print("\n" + "="*70)
        print(" SAMPLE POINTS BY TIER")
        print("="*70)
        for tier in range(6):
            tier_mask = (tiers == tier)
            tier_count = tier_mask.sum()
            if tier_count == 0:
                continue

            print(f"\nTier {tier} ({tier_count:,} points) - Sample of 5:")
            tier_indices = np.where(tier_mask)[0]
            sample_indices = np.random.choice(tier_indices, min(5, len(tier_indices)), replace=False)

            for idx in sample_indices:
                lat, lon = lats[idx], lons[idx]
                meta = metadata[idx]
                features = decode_metadata(meta)
                print(f"  lat={lat:.4f}, lon={lon:.4f}: {', '.join(features)}")

        # Find points with multiple features
        print("\n" + "="*70)
        print(" POINTS WITH MULTIPLE FEATURES")
        print("="*70)
        multi_feature_counts = []
        for meta in metadata:
            count = bin(meta).count('1')  # Count set bits
            multi_feature_counts.append(count)

        multi_feature_counts = np.array(multi_feature_counts)
        print(f"\nPoints with multiple features:")
        for n_features in range(1, 6):
            count = (multi_feature_counts == n_features).sum()
            if count > 0:
                pct = 100 * count / len(metadata)
                print(f"  {n_features} features: {count:,} points ({pct:.1f}%)")

        # Show examples of points with 3+ features
        multi_indices = np.where(multi_feature_counts >= 3)[0]
        if len(multi_indices) > 0:
            print(f"\nExample points with 3+ features (showing first 5):")
            for idx in multi_indices[:5]:
                lat, lon = lats[idx], lons[idx]
                tier = tiers[idx]
                meta = metadata[idx]
                features = decode_metadata(meta)
                print(f"  Tier {tier}, lat={lat:.4f}, lon={lon:.4f}: {', '.join(features)}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        nc_file = sys.argv[1]
    else:
        nc_file = 'output/adaptive_grid_SPARSE_west_wa.nc'

    inspect_metadata(nc_file)
