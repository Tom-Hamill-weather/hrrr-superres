"""
Quick sample of OSM trails from a few representative states
to visualize trail coverage before committing to full CONUS download

Samples from:
- California (West, mountainous)
- Colorado (Mountain West)
- North Carolina (East, Appalachian)
- Washington (Pacific Northwest)
- Arizona (Southwest)
"""

import os
import subprocess
import geopandas as gpd
import pandas as pd
from datetime import datetime
import config

# Representative states to sample
STATES = {
    'california': 'https://download.geofabrik.de/north-america/us/california-latest.osm.pbf',
    'colorado': 'https://download.geofabrik.de/north-america/us/colorado-latest.osm.pbf',
    'north-carolina': 'https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf',
    'washington': 'https://download.geofabrik.de/north-america/us/washington-latest.osm.pbf',
    'arizona': 'https://download.geofabrik.de/north-america/us/arizona-latest.osm.pbf',
}


def download_state_osm(state_name, url):
    """Download OSM extract for a single state"""
    trails_dir = os.path.join(config.DATA_DIR, 'trails', 'osm_samples')
    os.makedirs(trails_dir, exist_ok=True)

    osm_file = os.path.join(trails_dir, f'{state_name}.osm.pbf')

    if os.path.exists(osm_file):
        print(f"  ✓ Already downloaded: {state_name}")
        return osm_file

    print(f"  Downloading {state_name}... ", end='', flush=True)

    try:
        # Use curl with progress suppressed for cleaner output
        subprocess.run(
            ['curl', '-o', osm_file, '-L', '-s', url],
            check=True,
            timeout=600
        )
        size_mb = os.path.getsize(osm_file) / (1024**2)
        print(f"Done ({size_mb:.0f} MB)")
        return osm_file

    except Exception as e:
        print(f"Failed: {e}")
        return None


def extract_trails_osmium(osm_file, state_name):
    """Extract trails using osmium-tool"""
    trails_dir = os.path.dirname(osm_file)

    # Filter for named trails
    trails_osm = os.path.join(trails_dir, f'{state_name}_trails.osm.pbf')

    # Osmium filter for named paths/footways
    cmd = [
        'osmium', 'tags-filter', osm_file,
        'w/highway=path,name',
        'w/highway=footway,name',
        '-o', trails_osm,
        '--overwrite'
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # Convert to GeoJSON
    trails_geojson = os.path.join(trails_dir, f'{state_name}_trails.geojson')

    cmd = ['osmium', 'export', trails_osm, '-o', trails_geojson, '--overwrite']
    subprocess.run(cmd, check=True, capture_output=True)

    # Load and return
    gdf = gpd.read_file(trails_geojson)

    # Clean up intermediate files
    os.remove(trails_osm)
    os.remove(trails_geojson)

    return gdf


def extract_trails_pyrosm(osm_file, state_name):
    """Extract trails using pyrosm"""
    from pyrosm import OSM

    osm = OSM(osm_file)

    # Get walking network
    gdf = osm.get_network(network_type='walking')

    # Filter to named trails
    if '@name' in gdf.columns:
        gdf = gdf[gdf['@name'].notna() & (gdf['@name'] != '')].copy()
    elif 'name' in gdf.columns:
        gdf = gdf[gdf['name'].notna() & (gdf['name'] != '')].copy()

    return gdf


def main():
    """Sample OSM trails from representative states"""
    print("="*70)
    print(" SAMPLING OSM TRAILS FROM REPRESENTATIVE STATES")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check for required tool
    has_osmium = False
    has_pyrosm = False

    try:
        subprocess.run(['osmium', '--version'], check=True, capture_output=True)
        has_osmium = True
        print("✓ Found osmium-tool")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        import pyrosm
        has_pyrosm = True
        print("✓ Found pyrosm")
    except ImportError:
        pass

    if not has_osmium and not has_pyrosm:
        print("✗ Error: Neither osmium-tool nor pyrosm available")
        print("\nInstall one of:")
        print("  brew install osmium-tool  (recommended)")
        print("  pip install pyrosm")
        return 1

    print(f"\nDownloading OSM data for {len(STATES)} states...")
    print("(Total ~1-2 GB, may take a few minutes)\n")

    all_trails = []

    for state_name, url in STATES.items():
        # Download
        osm_file = download_state_osm(state_name, url)
        if osm_file is None:
            continue

        # Extract trails
        print(f"  Extracting trails from {state_name}... ", end='', flush=True)

        try:
            if has_osmium:
                gdf = extract_trails_osmium(osm_file, state_name)
            else:
                gdf = extract_trails_pyrosm(osm_file, state_name)

            print(f"Done ({len(gdf):,} segments)")

            gdf['state'] = state_name
            all_trails.append(gdf)

        except Exception as e:
            print(f"Failed: {e}")

    if not all_trails:
        print("\n✗ No trails extracted")
        return 1

    # Combine all trails
    print("\nCombining trail data...")
    combined_gdf = pd.concat(all_trails, ignore_index=True)

    # Ensure CRS
    if combined_gdf.crs is None:
        combined_gdf.set_crs('EPSG:4326', inplace=True)

    # Keep only essential columns to avoid field type issues
    essential_cols = ['geometry', 'state']

    # Add name column if it exists (trying both formats)
    if '@name' in combined_gdf.columns:
        combined_gdf['name'] = combined_gdf['@name']
        essential_cols.append('name')
    elif 'name' in combined_gdf.columns:
        essential_cols.append('name')

    # Add highway type if it exists
    if '@highway' in combined_gdf.columns:
        combined_gdf['highway'] = combined_gdf['@highway']
        essential_cols.append('highway')
    elif 'highway' in combined_gdf.columns:
        essential_cols.append('highway')

    # Keep only these columns
    combined_gdf = combined_gdf[essential_cols].copy()

    print(f"✓ Total: {len(combined_gdf):,} trail segments")

    # Calculate stats
    total_km = combined_gdf.to_crs('EPSG:3857').length.sum() / 1000
    print(f"  Total length: {total_km:,.0f} km")

    print("\n  Trails by state:")
    for state in STATES.keys():
        state_trails = combined_gdf[combined_gdf['state'] == state]
        if len(state_trails) > 0:
            state_km = state_trails.to_crs('EPSG:3857').length.sum() / 1000
            print(f"    {state:15s}: {len(state_trails):6,} segments, {state_km:8,.0f} km")

    # Save as GeoPackage (better format, no field limitations)
    output_file = os.path.join(config.DATA_DIR, 'trails', 'osm_trails_sample.gpkg')
    combined_gdf.to_file(output_file, driver='GPKG')

    print(f"\n✓ Saved: {output_file}")

    print("\n" + "="*70)
    print(" ✓ SAMPLING COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
