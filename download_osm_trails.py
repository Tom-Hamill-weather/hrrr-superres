"""
Download hiking trail data from OpenStreetMap for CONUS
Uses Overpass API to query for trails with classification metadata
"""

import requests
import json
import geopandas as gpd
from shapely.geometry import LineString
import os
import config
from datetime import datetime

def download_osm_trails_conus():
    """
    Download trail data from OSM for CONUS domain

    OSM Trail Classification:
    - highway=path: General paths (most hiking trails)
    - highway=footway: Pedestrian ways
    - highway=track: Vehicle-accessible tracks (forestry roads, often used for hiking)

    Additional tags for filtering:
    - sac_scale: Hiking difficulty (hiking, mountain_hiking, demanding_mountain_hiking, etc.)
    - trail_visibility: excellent, good, intermediate, bad, horrible, no
    - name: Trail name (major trails usually have names)
    """

    print("="*70)
    print(" DOWNLOADING OSM TRAIL DATA FOR CONUS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # CONUS bounding box
    bbox = "21.0,-135.0,53.0,-60.0"  # south, west, north, east

    # Overpass QL query
    # Focus on designated hiking trails with names (major trails)
    overpass_query = f"""
    [out:json][timeout:300][bbox:{bbox}];
    (
      way["highway"="path"]["name"]["sac_scale"];
      way["highway"="footway"]["name"]["sac_scale"];
      way["highway"="path"]["name"]["trail_visibility"];
      way["highway"="footway"]["name"]["trail_visibility"];
    );
    out geom;
    """

    print("\nQuery details:")
    print(f"  Domain: {bbox}")
    print(f"  Filters: Named trails with difficulty/visibility ratings")
    print(f"  Timeout: 300 seconds")

    # Overpass API endpoint
    overpass_url = "https://overpass-api.de/api/interpreter"

    print("\nSending request to Overpass API...")
    print("  (This may take several minutes for CONUS-scale query)")

    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=600)
        response.raise_for_status()

        print(f"✓ Response received: {len(response.content)/1024/1024:.1f} MB")

        data = response.json()
        elements = data.get('elements', [])

        print(f"✓ Retrieved {len(elements)} trail segments")

        if len(elements) == 0:
            print("⚠ Warning: No trails found. Query may be too restrictive.")
            return None

        # Convert to GeoDataFrame
        print("\nConverting to GeoDataFrame...")
        features = []

        for element in elements:
            if element['type'] != 'way':
                continue

            # Extract geometry
            coords = [(node['lon'], node['lat']) for node in element.get('geometry', [])]
            if len(coords) < 2:
                continue

            # Extract tags
            tags = element.get('tags', {})

            features.append({
                'geometry': LineString(coords),
                'osm_id': element.get('id'),
                'name': tags.get('name', ''),
                'highway': tags.get('highway', ''),
                'sac_scale': tags.get('sac_scale', ''),
                'trail_visibility': tags.get('trail_visibility', ''),
                'surface': tags.get('surface', ''),
            })

        gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')

        # Calculate statistics
        print(f"\n✓ Created GeoDataFrame with {len(gdf)} trail segments")
        print(f"  Unique trail names: {gdf['name'].nunique()}")
        print(f"  Total trail length: {gdf.to_crs('EPSG:3857').length.sum()/1000:.0f} km")

        # Show distribution by difficulty
        if 'sac_scale' in gdf.columns:
            print("\n  Trail difficulty distribution (sac_scale):")
            for scale, count in gdf['sac_scale'].value_counts().head(10).items():
                if scale:
                    print(f"    {scale}: {count:,} segments")

        # Save to file
        output_file = os.path.join(config.DATA_DIR, 'trails', 'osm_trails_conus.shp')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        gdf.to_file(output_file)
        print(f"\n✓ Saved to: {output_file}")

        print("\n" + "="*70)
        print(" DOWNLOAD COMPLETE")
        print("="*70)

        return gdf

    except requests.exceptions.Timeout:
        print("✗ Error: Request timed out")
        print("  CONUS is a very large area. Consider:")
        print("  1. Breaking into regions and downloading separately")
        print("  2. Using a different data source")
        print("  3. Using cached OSM extracts (e.g., Geofabrik)")
        return None

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


if __name__ == '__main__':
    download_osm_trails_conus()
