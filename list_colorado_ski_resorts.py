"""
List ski resorts in Colorado that will receive 90m resolution
"""
import os
import geopandas as gpd
import config

def list_colorado_ski_resorts():
    """List all ski resorts in Colorado"""
    print("\n" + "="*70)
    print(" SKI RESORTS IN COLORADO (90m Resolution)")
    print("="*70)

    # Load ski resort data
    ski_file = os.path.join(config.DATA_DIR, 'ski_resorts', 'colorado_ski_resorts.geojson')

    if not os.path.exists(ski_file):
        print(f"\n✗ Ski resort file not found: {ski_file}")
        return

    print(f"\nLoading Colorado ski resort data from {ski_file}...")
    ski_gdf = gpd.read_file(ski_file)

    print(f"Total Colorado ski resorts: {len(ski_gdf)}")

    # Get coordinates from Point geometries
    ski_gdf['lat'] = ski_gdf.geometry.y
    ski_gdf['lon'] = ski_gdf.geometry.x

    co_ski = ski_gdf.copy()

    print(f"\nAll {len(co_ski)} resorts will receive 90m resolution (Tier 1)")
    print("\n" + "-"*70)
    print(f"{'Resort Name':<40} {'Latitude':>10} {'Longitude':>10}")
    print("-"*70)

    # Sort by name
    co_ski_sorted = co_ski.sort_values('name' if 'name' in co_ski.columns else co_ski.columns[0])

    for idx, row in co_ski_sorted.iterrows():
        # Try different possible name columns
        name = None
        for col in ['name', 'Name', 'ski_area', 'resort']:
            if col in row and row[col]:
                name = row[col]
                break

        if name is None:
            # Use first string column as name
            for col in row.index:
                if isinstance(row[col], str) and row[col]:
                    name = row[col]
                    break

        if name is None:
            name = f"Resort {idx}"

        lat = row['lat']
        lon = row['lon']

        print(f"{name:<40} {lat:>10.4f} {lon:>10.4f}")

    print("-"*70)
    print(f"\nAll {len(co_ski)} Colorado ski resorts will receive:")
    print(f"  - Tier 1 classification (90m resolution)")
    print(f"  - 5km buffer around resort location")
    print(f"  - ~1,100 points per 3km HRRR grid cell")
    print("="*70)

    # Show major resorts specifically
    major_resorts = ['Vail', 'Copper', 'Winter Park', 'Breckenridge',
                     'Keystone', 'Aspen', 'Steamboat', 'Telluride']

    print("\nChecking for major resorts:")
    for resort_name in major_resorts:
        found = False
        for idx, row in co_ski_sorted.iterrows():
            name = None
            for col in ['name', 'Name', 'ski_area', 'resort']:
                if col in row and row[col]:
                    name = str(row[col])
                    break

            if name and resort_name.lower() in name.lower():
                print(f"  ✓ {name}")
                found = True
                break

        if not found:
            print(f"  ? {resort_name} (not found - may be named differently)")

    print("\n")

if __name__ == '__main__':
    list_colorado_ski_resorts()
