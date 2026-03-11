"""
Extract state parks from PAD-US database
Filters for state-managed parks only to reduce file size
"""
import os
import geopandas as gpd
import zipfile
from pathlib import Path

def extract_state_parks():
    """Extract and filter state parks from PAD-US GeoPackage"""

    data_dir = '/Users/tom.hamill@weather.com/python/superres/data'
    padus_dir = os.path.join(data_dir, 'padus')

    # Input files
    zip_file = os.path.join(padus_dir, 'PADUS3_0.zip')
    state_parks_file = os.path.join(padus_dir, 'state_parks.geojson')

    # Check if already done
    if os.path.exists(state_parks_file):
        print(f"✓ State parks already extracted: {state_parks_file}")
        parks = gpd.read_file(state_parks_file)
        print(f"  {len(parks)} state park units")
        return state_parks_file

    if not os.path.exists(zip_file):
        print(f"✗ PAD-US zip file not found: {zip_file}")
        print("  Run the download script first")
        return None

    print("="*70)
    print(" EXTRACTING STATE PARKS FROM PAD-US")
    print("="*70)

    # Extract zip file
    print(f"\nExtracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # List contents
        file_list = zip_ref.namelist()
        print(f"Archive contains {len(file_list)} files")

        # Find the GeoPackage file
        gpkg_files = [f for f in file_list if f.endswith('.gpkg')]
        if not gpkg_files:
            print("✗ No GeoPackage file found in archive")
            return None

        gpkg_file = gpkg_files[0]
        print(f"Found GeoPackage: {gpkg_file}")

        # Extract just the GeoPackage
        print("Extracting GeoPackage (this may take a minute)...")
        zip_ref.extract(gpkg_file, padus_dir)

    gpkg_path = os.path.join(padus_dir, gpkg_file)
    print(f"✓ Extracted to: {gpkg_path}")
    print(f"  File size: {os.path.getsize(gpkg_path) / 1e9:.2f} GB")

    # Read and filter
    print("\nReading PAD-US GeoPackage...")
    print("(This will take 2-5 minutes for the full dataset)")

    # List available layers
    import fiona
    layers = fiona.listlayers(gpkg_path)
    print(f"Available layers: {layers}")

    # Read the main layer (usually PADUS3_0Combined or similar)
    main_layer = [l for l in layers if 'Combined' in l or 'Fee' in l][0]
    print(f"\nReading layer: {main_layer}")

    padus_gdf = gpd.read_file(gpkg_path, layer=main_layer)
    print(f"✓ Loaded {len(padus_gdf):,} protected areas")

    # Filter for CONUS
    print("\nFiltering for CONUS...")
    conus_gdf = padus_gdf.cx[-135:-60, 21:53].copy()
    print(f"  CONUS areas: {len(conus_gdf):,}")

    # Filter for state parks
    print("\nFiltering for state parks...")
    print("  Looking for:")
    print("    - Mang_Type == 'STAT' (state-managed)")
    print("    - Des_Tp contains 'Park', 'Recreation', 'Natural', 'Wilderness'")

    # Check what columns exist
    print(f"\nAvailable columns: {list(conus_gdf.columns)}")

    # Filter for state-managed
    if 'Mang_Type' in conus_gdf.columns:
        state_managed = conus_gdf[conus_gdf['Mang_Type'] == 'STAT'].copy()
        print(f"  State-managed areas: {len(state_managed):,}")
    else:
        print("  ⚠ Mang_Type column not found, using alternative filter")
        state_managed = conus_gdf.copy()

    # Further filter for parks
    if 'Des_Tp' in state_managed.columns:
        park_keywords = ['Park', 'Recreation', 'Natural', 'Wilderness', 'Historic', 'Scenic']
        state_parks = state_managed[
            state_managed['Des_Tp'].str.contains('|'.join(park_keywords), case=False, na=False)
        ].copy()
        print(f"  State parks (filtered): {len(state_parks):,}")
    else:
        print("  ⚠ Des_Tp column not found, keeping all state-managed areas")
        state_parks = state_managed.copy()

    # Save filtered result
    print(f"\nSaving state parks to: {state_parks_file}")
    state_parks.to_file(state_parks_file, driver='GeoJSON')

    file_size_mb = os.path.getsize(state_parks_file) / 1e6
    print(f"✓ Saved {len(state_parks):,} state parks ({file_size_mb:.1f} MB)")

    # Clean up GeoPackage to save space (optional)
    print(f"\nCleaning up extracted GeoPackage...")
    os.remove(gpkg_path)
    print(f"✓ Removed {gpkg_path}")

    return state_parks_file

if __name__ == '__main__':
    extract_state_parks()
