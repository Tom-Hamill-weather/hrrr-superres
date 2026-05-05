#!/usr/bin/env python3
"""
Download golf courses from OpenStreetMap and save to local GeoJSON file.
This preprocessing step avoids repeated API calls during grid generation.

Usage:
    python preprocess_golf_courses.py
"""

import os
import json
import time
import requests
from shapely.geometry import Polygon, MultiPolygon, Point, mapping
from shapely.ops import unary_union
import geopandas as gpd

# CONUS bounds
MIN_LAT, MAX_LAT = 24.0, 49.0
MIN_LON, MAX_LON = -125.0, -67.0

# Output file
OUTPUT_DIR = 'data/golf'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'us_golf_courses_osm.geojson')

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def download_golf_courses():
    """Download golf courses from OSM Overpass API."""

    print("Downloading golf courses from OpenStreetMap...")
    print(f"CONUS bounds: {MIN_LAT}-{MAX_LAT}°N, {MIN_LON}-{MAX_LON}°W")

    # Overpass QL query for golf courses
    overpass_query = f"""
    [out:json][timeout:300];
    (
      way["leisure"="golf_course"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
      relation["leisure"="golf_course"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
    );
    out body;
    >;
    out skel qt;
    """

    try:
        print("Querying Overpass API... (this may take several minutes)")
        response = requests.post(
            OVERPASS_URL,
            data={'data': overpass_query},
            timeout=600  # 10 minute timeout
        )
        response.raise_for_status()

        data = response.json()
        print(f"Received {len(data.get('elements', []))} elements from OSM")

        return data

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. The Overpass API may be overloaded.")
        print("Try again later or use a smaller bounding box.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download data from Overpass API: {e}")
        return None


def parse_osm_to_geojson(osm_data):
    """Parse OSM JSON data to extract golf course geometries."""

    if not osm_data or 'elements' not in osm_data:
        print("ERROR: No valid OSM data to parse")
        return None

    print("\nParsing OSM data...")

    # Build lookup for nodes (coordinates)
    nodes = {}
    for element in osm_data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = (element['lon'], element['lat'])

    print(f"Found {len(nodes)} nodes")

    # Extract golf courses
    golf_courses = []

    for element in osm_data['elements']:
        if element['type'] == 'way' and element.get('tags', {}).get('leisure') == 'golf_course':
            # Get way nodes
            way_nodes = element.get('nodes', [])

            if len(way_nodes) < 3:
                continue  # Need at least 3 nodes for a polygon

            # Build coordinate list
            coords = []
            for node_id in way_nodes:
                if node_id in nodes:
                    coords.append(nodes[node_id])

            if len(coords) < 3:
                continue

            # Close polygon if needed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                polygon = Polygon(coords)

                # Basic validation
                if not polygon.is_valid or polygon.area == 0:
                    continue

                # Extract metadata
                tags = element.get('tags', {})
                name = tags.get('name', f"Golf Course {element['id']}")

                golf_courses.append({
                    'geometry': polygon,
                    'name': name,
                    'osm_id': element['id'],
                    'area_sqkm': polygon.area * 111 * 111,  # Rough approximation
                })

            except Exception as e:
                print(f"Warning: Failed to create polygon for way {element['id']}: {e}")
                continue

        elif element['type'] == 'relation' and element.get('tags', {}).get('leisure') == 'golf_course':
            # Handle multipolygon relations
            outer_ways = []
            inner_ways = []

            for member in element.get('members', []):
                if member['type'] == 'way':
                    # Build coordinate list for this way
                    way_id = member['ref']
                    # Find the way in elements
                    way_nodes = None
                    for elem in osm_data['elements']:
                        if elem['type'] == 'way' and elem['id'] == way_id:
                            way_nodes = elem.get('nodes', [])
                            break

                    if not way_nodes:
                        continue

                    coords = []
                    for node_id in way_nodes:
                        if node_id in nodes:
                            coords.append(nodes[node_id])

                    if len(coords) < 3:
                        continue

                    # Close polygon if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])

                    if member.get('role') == 'outer':
                        outer_ways.append(coords)
                    elif member.get('role') == 'inner':
                        inner_ways.append(coords)

            # Create multipolygon
            try:
                if outer_ways:
                    outer_polys = [Polygon(coords) for coords in outer_ways]
                    # Union outer polygons
                    outer = unary_union(outer_polys)

                    # Subtract inner holes
                    if inner_ways:
                        inner_polys = [Polygon(coords) for coords in inner_ways]
                        for inner_poly in inner_polys:
                            outer = outer.difference(inner_poly)

                    if outer.is_valid and outer.area > 0:
                        tags = element.get('tags', {})
                        name = tags.get('name', f"Golf Course Relation {element['id']}")

                        golf_courses.append({
                            'geometry': outer,
                            'name': name,
                            'osm_id': element['id'],
                            'area_sqkm': outer.area * 111 * 111,  # Rough approximation
                        })

            except Exception as e:
                print(f"Warning: Failed to create polygon for relation {element['id']}: {e}")
                continue

    print(f"\nExtracted {len(golf_courses)} golf courses")

    if not golf_courses:
        print("ERROR: No golf courses found in the data")
        return None

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(golf_courses, crs='EPSG:4326')

    # Filter by area (remove very small courses - likely errors or driving ranges)
    print("\nFiltering golf courses...")
    print(f"Area range: {gdf['area_sqkm'].min():.4f} - {gdf['area_sqkm'].max():.2f} km²")

    # Keep courses > 0.05 km² (50,000 m² ≈ 12 acres)
    MIN_AREA_KM2 = 0.05
    gdf_filtered = gdf[gdf['area_sqkm'] > MIN_AREA_KM2].copy()

    print(f"Filtered to {len(gdf_filtered)} courses (removed {len(gdf) - len(gdf_filtered)} small courses)")
    print(f"Filtered area range: {gdf_filtered['area_sqkm'].min():.4f} - {gdf_filtered['area_sqkm'].max():.2f} km²")

    return gdf_filtered


def save_geojson(gdf, output_path):
    """Save GeoDataFrame to GeoJSON file."""

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nSaving to {output_path}...")

    try:
        gdf.to_file(output_path, driver='GeoJSON')

        # Print summary
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Saved {len(gdf)} golf courses ({file_size_mb:.2f} MB)")

        # Print some statistics
        print("\nSummary:")
        print(f"  Total courses: {len(gdf)}")
        print(f"  Named courses: {gdf['name'].notna().sum()}")
        print(f"  Total area: {gdf['area_sqkm'].sum():.2f} km²")
        print(f"  Mean area: {gdf['area_sqkm'].mean():.4f} km²")
        print(f"  Median area: {gdf['area_sqkm'].median():.4f} km²")

        return True

    except Exception as e:
        print(f"ERROR: Failed to save GeoJSON: {e}")
        return False


def main():
    """Main preprocessing workflow."""

    print("=" * 60)
    print("Golf Course Preprocessing Script")
    print("=" * 60)
    print()

    # Check if file already exists
    if os.path.exists(OUTPUT_FILE):
        response = input(f"File {OUTPUT_FILE} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Download from OSM
    osm_data = download_golf_courses()

    if not osm_data:
        print("\nFailed to download data from OSM.")
        return

    # Parse to GeoJSON
    gdf = parse_osm_to_geojson(osm_data)

    if gdf is None or len(gdf) == 0:
        print("\nFailed to parse golf course data.")
        return

    # Save to file
    success = save_geojson(gdf, OUTPUT_FILE)

    if success:
        print("\n" + "=" * 60)
        print("✓ Preprocessing complete!")
        print("=" * 60)
        print(f"\nGolf course data saved to: {OUTPUT_FILE}")
        print("You can now run generate_adaptive_grid.py")
    else:
        print("\nPreprocessing failed.")


if __name__ == '__main__':
    main()
