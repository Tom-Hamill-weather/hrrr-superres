"""
Quick test to verify trail filtering excludes Alaska and Hawaii
"""
import geopandas as gpd
import os
import config

print("="*70)
print(" TESTING TRAIL FILTERING")
print("="*70)

# CONUS bounds - same as in load_trails()
conus_lat_min, conus_lat_max = 24.0, 49.5
conus_lon_min, conus_lon_max = -125.0, -67.0

def is_in_conus(geom):
    """Check if geometry is within CONUS bounds"""
    bounds = geom.bounds  # (minx, miny, maxx, maxy)
    # Check if geometry center is in CONUS
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    return (conus_lon_min <= center_lon <= conus_lon_max and
            conus_lat_min <= center_lat <= conus_lat_max)

# Load trail data
nrt_file = os.path.join(config.DATA_DIR, 'trails', 'national_recreation_trails.shp')

print(f"\nLoading: {nrt_file}")
nrt_gdf = gpd.read_file(nrt_file)
print(f"  Total trails before filtering: {len(nrt_gdf)}")

# Apply filtering
nrt_filtered = nrt_gdf[nrt_gdf.geometry.apply(is_in_conus)].copy()
print(f"  Total trails after filtering: {len(nrt_filtered)}")
print(f"  Filtered out: {len(nrt_gdf) - len(nrt_filtered)}")

# Check for Alaska and Hawaii trails
print("\nChecking for problematic trails:")
for idx, row in nrt_gdf.iterrows():
    bounds = row.geometry.bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # Check for Alaska (> 50°N or < -130°W)
    if center_lat > 50 or center_lon < -130:
        trail_name = row.get('TRLNAME', row.get('NAME', 'Unknown'))
        in_filtered = idx in nrt_filtered.index
        status = "✗ KEPT (BAD)" if in_filtered else "✓ FILTERED"
        print(f"  {status}: {trail_name} at ({center_lat:.2f}°N, {center_lon:.2f}°W)")

    # Check for Hawaii (< 25°N and > -160°W)
    if center_lat < 25 and center_lon > -160:
        trail_name = row.get('TRLNAME', row.get('NAME', 'Unknown'))
        in_filtered = idx in nrt_filtered.index
        status = "✗ KEPT (BAD)" if in_filtered else "✓ FILTERED"
        print(f"  {status}: {trail_name} at ({center_lat:.2f}°N, {center_lon:.2f}°W)")

# Show bounds of filtered data
if len(nrt_filtered) > 0:
    all_bounds = nrt_filtered.geometry.bounds
    print(f"\nFiltered data bounds:")
    print(f"  Latitude: {all_bounds['miny'].min():.2f} to {all_bounds['maxy'].max():.2f}")
    print(f"  Longitude: {all_bounds['minx'].min():.2f} to {all_bounds['maxx'].max():.2f}")

print("\n" + "="*70)
print(" TEST COMPLETE")
print("="*70)
