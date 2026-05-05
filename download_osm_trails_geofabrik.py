"""
Download and process OSM trail data from Geofabrik extracts

Strategy: Download OSM extracts by region, extract trails, merge

Geofabrik provides regional OSM extracts that are manageable in size.
For CONUS, we can download the US extract and filter to trails.
"""

import os
import sys
import subprocess
import geopandas as gpd
import pandas as pd
from datetime import datetime
import config

def download_osm_us_extract():
    """
    Download US OSM extract from Geofabrik

    This is a large file (~10 GB compressed), but more reliable than Overpass API
    """
    print("="*70)
    print(" DOWNLOADING OSM US EXTRACT FROM GEOFABRIK")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # URL for US extract
    url = "https://download.geofabrik.de/north-america/us-latest.osm.pbf"

    # Output file
    trails_dir = os.path.join(config.DATA_DIR, 'trails')
    os.makedirs(trails_dir, exist_ok=True)

    osm_file = os.path.join(trails_dir, 'us-latest.osm.pbf')

    # Check if already downloaded
    if os.path.exists(osm_file):
        file_size_gb = os.path.getsize(osm_file) / (1024**3)
        print(f"✓ OSM file already exists: {osm_file}")
        print(f"  Size: {file_size_gb:.1f} GB")

        response = input("\nRe-download? (y/N): ").strip().lower()
        if response != 'y':
            return osm_file

    print(f"Downloading from: {url}")
    print("⚠ This is a large file (~10 GB) and will take some time...")
    print()

    # Use wget or curl to download with progress
    try:
        # Try wget first (shows progress)
        subprocess.run(['wget', '-O', osm_file, url], check=True)
        print(f"\n✓ Downloaded: {osm_file}")
        return osm_file

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to curl
        try:
            subprocess.run(['curl', '-o', osm_file, '-L', url], check=True)
            print(f"\n✓ Downloaded: {osm_file}")
            return osm_file
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            return None


def extract_trails_from_osm(osm_file):
    """
    Extract trail data from OSM PBF file using osmium/pyrosm

    Requires: osmium-tool or pyrosm library
    """
    print("\n" + "="*70)
    print(" EXTRACTING TRAILS FROM OSM DATA")
    print("="*70)

    # Check if osmium is installed
    try:
        subprocess.run(['osmium', '--version'], check=True, capture_output=True)
        has_osmium = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_osmium = False

    trails_dir = os.path.dirname(osm_file)

    if has_osmium:
        print("\nUsing osmium-tool to filter trails...")

        # Output file for filtered data
        trails_osm = os.path.join(trails_dir, 'trails.osm.pbf')

        # Filter for trails using osmium tags-filter
        # highway=path, footway, track with trail-related tags
        cmd = [
            'osmium', 'tags-filter', osm_file,
            'w/highway=path',
            'w/highway=footway',
            'w/highway=track',
            '-o', trails_osm,
            '--overwrite'
        ]

        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        print(f"✓ Filtered OSM data: {trails_osm}")

        # Convert to GeoJSON (more manageable)
        trails_geojson = os.path.join(trails_dir, 'trails_raw.geojson')

        cmd = ['osmium', 'export', trails_osm, '-o', trails_geojson, '--overwrite']
        print(f"\n  Converting to GeoJSON...")
        subprocess.run(cmd, check=True)

        print(f"✓ Converted to GeoJSON: {trails_geojson}")

        # Load with geopandas and filter
        print("\n  Loading with GeoPandas...")
        gdf = gpd.read_file(trails_geojson)

    else:
        print("⚠ osmium-tool not found. Trying pyrosm...")

        try:
            from pyrosm import OSM

            print("\nUsing pyrosm to extract trails...")
            osm = OSM(osm_file)

            # Get all ways with trail-related tags
            gdf = osm.get_network(network_type='walking')

            print(f"✓ Loaded {len(gdf)} walking paths")

        except ImportError:
            print("✗ Error: Neither osmium-tool nor pyrosm available")
            print("\nInstall one of:")
            print("  brew install osmium-tool")
            print("  pip install pyrosm")
            return None

    # Filter to CONUS bounds
    print("\n  Filtering to CONUS bounds...")
    gdf = gdf.cx[-135:-60, 21:53].copy()

    # Filter to named trails only (major trails)
    print("  Filtering to named trails...")
    if '@name' in gdf.columns:
        gdf = gdf[gdf['@name'].notna() & (gdf['@name'] != '')].copy()
    elif 'name' in gdf.columns:
        gdf = gdf[gdf['name'].notna() & (gdf['name'] != '')].copy()

    print(f"✓ Filtered to {len(gdf)} named trail segments in CONUS")

    # Calculate total length
    if len(gdf) > 0:
        total_km = gdf.to_crs('EPSG:3857').length.sum() / 1000
        print(f"  Total trail length: {total_km:,.0f} km")

    # Save to shapefile
    output_file = os.path.join(trails_dir, 'osm_trails_conus.shp')
    gdf.to_file(output_file)
    print(f"\n✓ Saved: {output_file}")

    return gdf


def main():
    """Main execution"""

    # Check for required tools
    print("Checking for required tools...")
    print("  This script requires either:")
    print("    - osmium-tool: brew install osmium-tool")
    print("    - pyrosm: pip install pyrosm")
    print()

    # Download OSM extract
    osm_file = download_osm_us_extract()

    if osm_file is None:
        print("\n✗ Failed to download OSM data")
        return 1

    # Extract trails
    trails_gdf = extract_trails_from_osm(osm_file)

    if trails_gdf is None or len(trails_gdf) == 0:
        print("\n✗ Failed to extract trails")
        return 1

    print("\n" + "="*70)
    print(" ✓ TRAIL EXTRACTION COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
