"""
List all US ski resorts that will receive 90m resolution
"""
import os
import geopandas as gpd
import config

def list_all_us_ski_resorts():
    """List all US ski resorts"""
    print("\n" + "="*70)
    print(" ALL US SKI RESORTS (90m Resolution)")
    print("="*70)

    # Load ski resort data
    ski_file = os.path.join(config.DATA_DIR, 'ski_resorts', 'us_ski_resorts.geojson')

    if not os.path.exists(ski_file):
        print(f"\n✗ Ski resort file not found: {ski_file}")
        return

    print(f"\nLoading US ski resort data from {ski_file}...")
    ski_gdf = gpd.read_file(ski_file)

    print(f"Total US ski resorts: {len(ski_gdf)}")

    # Group by state
    states = ski_gdf['state'].unique()
    states = sorted(states)

    print(f"\nCoverage: {len(states)} states")
    print("\n" + "="*70)

    for state in states:
        state_resorts = ski_gdf[ski_gdf['state'] == state].sort_values('name')
        print(f"\n{state} ({len(state_resorts)} resorts):")
        print("-" * 70)
        for idx, row in state_resorts.iterrows():
            name = row['name']
            lat = row.geometry.y
            lon = row.geometry.x
            print(f"  • {name:<35} ({lat:>7.4f}, {lon:>8.4f})")

    print("\n" + "="*70)
    print(f"TOTAL: {len(ski_gdf)} ski resorts across {len(states)} states")
    print("="*70)
    print(f"\nAll resorts will receive:")
    print(f"  - Tier 1 classification (90m resolution)")
    print(f"  - 4km buffer around resort location (covers base to summit)")
    print(f"  - ~1,100 points per 3km HRRR grid cell")
    print("="*70 + "\n")

if __name__ == '__main__':
    list_all_us_ski_resorts()
