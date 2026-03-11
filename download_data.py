"""
Data acquisition script for adaptive grid generation
Downloads and preprocesses all necessary geospatial datasets
"""
import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
import pygrib
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import requests
from tqdm import tqdm
import config

def download_file(url, output_path, description=""):
    """Download a file with progress bar"""
    if os.path.exists(output_path):
        print(f"✓ {description} already exists: {output_path}")
        return output_path

    print(f"Downloading {description}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ Downloaded: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Error downloading {description}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

def extract_archive(archive_path, extract_to):
    """Extract zip or gzip archive"""
    print(f"Extracting {archive_path}...")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.gz'):
        output_path = archive_path[:-3]
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return output_path

    print(f"✓ Extracted to: {extract_to}")

def download_hrrr_grid_and_terrain():
    """Download HRRR grid definition and terrain from AWS"""
    print("\n" + "="*60)
    print("DOWNLOADING HRRR GRID AND TERRAIN")
    print("="*60)

    # Use a recent HRRR file to extract grid and terrain
    # HRRR files are in GRIB2 format
    hrrr_dir = os.path.join(config.DATA_DIR, 'hrrr')
    os.makedirs(hrrr_dir, exist_ok=True)

    # Download a sample HRRR file (surface level)
    # We'll use the 00z analysis from a recent date
    sample_date = '20240101'  # Adjust to a known available date
    hrrr_file_url = f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{sample_date}/conus/hrrr.t00z.wrfsfcf00.grib2'
    hrrr_file = os.path.join(hrrr_dir, 'hrrr_sample.grib2')

    print("Downloading HRRR sample file for grid extraction...")
    print(f"URL: {hrrr_file_url}")
    print("Note: This is a large file (~500MB). This may take several minutes.")

    downloaded = download_file(hrrr_file_url, hrrr_file, "HRRR GRIB2")

    if downloaded:
        print("\nExtracting grid and terrain information from HRRR file...")
        try:
            grbs = pygrib.open(hrrr_file)

            # Get the first message to extract grid
            grb = grbs.message(1)
            lats, lons = grb.latlons()

            # Save lat/lon grid
            np.save(os.path.join(hrrr_dir, 'hrrr_lats.npy'), lats)
            np.save(os.path.join(hrrr_dir, 'hrrr_lons.npy'), lons)
            print(f"✓ Saved HRRR grid: {lats.shape}")

            # Try to extract terrain height (HGT surface)
            terrain = None
            for grb in grbs:
                if grb.name == 'Orography' or grb.shortName == 'orog':
                    terrain = grb.values
                    break

            if terrain is not None:
                np.save(os.path.join(hrrr_dir, 'hrrr_terrain.npy'), terrain)
                print(f"✓ Saved HRRR terrain: {terrain.shape}")
            else:
                print("⚠ Terrain not found in this HRRR file. Will need to download separately.")

            grbs.close()

        except Exception as e:
            print(f"✗ Error processing HRRR file: {e}")
            print("You may need to manually provide HRRR grid files.")

    return hrrr_dir

def download_natural_earth_data():
    """Download coastline and lakes from Natural Earth"""
    print("\n" + "="*60)
    print("DOWNLOADING NATURAL EARTH COASTLINE AND LAKES")
    print("="*60)

    ne_dir = os.path.join(config.DATA_DIR, 'natural_earth')
    os.makedirs(ne_dir, exist_ok=True)

    datasets = {
        'coastline': 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip',
        'lakes': 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip',
        'ocean': 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip'
    }

    for name, url in datasets.items():
        zip_file = os.path.join(ne_dir, f'{name}.zip')
        downloaded = download_file(url, zip_file, f"Natural Earth {name}")

        if downloaded:
            extract_archive(zip_file, ne_dir)

    return ne_dir

def download_protected_areas():
    """Download PAD-US protected areas database"""
    print("\n" + "="*60)
    print("DOWNLOADING PROTECTED AREAS (PAD-US)")
    print("="*60)

    padus_dir = os.path.join(config.DATA_DIR, 'padus')
    os.makedirs(padus_dir, exist_ok=True)

    # PAD-US 3.0 is available as a large geodatabase
    # For this application, we'll use a simplified approach or point to manual download
    print("PAD-US data is very large (several GB).")
    print("Please download manually from:")
    print("https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download")
    print(f"Save to: {padus_dir}")
    print("\nAlternatively, we can use state park data from individual states.")

    # Option: Download from ScienceBase
    padus_url = "https://www.sciencebase.gov/catalog/item/602370c4d34eb12031172aa6"
    print(f"\nDirect link: {padus_url}")

    return padus_dir

def download_urban_areas():
    """Download Census Urban Areas"""
    print("\n" + "="*60)
    print("DOWNLOADING CENSUS URBAN AREAS")
    print("="*60)

    urban_dir = os.path.join(config.DATA_DIR, 'urban')
    os.makedirs(urban_dir, exist_ok=True)

    # Census Urban Areas shapefile
    urban_url = 'https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac10.zip'
    zip_file = os.path.join(urban_dir, 'urban_areas.zip')

    downloaded = download_file(urban_url, zip_file, "Census Urban Areas")
    if downloaded:
        extract_archive(zip_file, urban_dir)

    return urban_dir

def download_primary_roads():
    """Download TIGER Primary Roads"""
    print("\n" + "="*60)
    print("DOWNLOADING PRIMARY ROADS")
    print("="*60)

    roads_dir = os.path.join(config.DATA_DIR, 'roads')
    os.makedirs(roads_dir, exist_ok=True)

    roads_url = 'https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/tl_2023_us_primaryroads.zip'
    zip_file = os.path.join(roads_dir, 'primary_roads.zip')

    downloaded = download_file(roads_url, zip_file, "Primary Roads")
    if downloaded:
        extract_archive(zip_file, roads_dir)

    return roads_dir

def download_ski_resorts():
    """Download ski resort locations"""
    print("\n" + "="*60)
    print("DOWNLOADING SKI RESORT DATA")
    print("="*60)

    ski_dir = os.path.join(config.DATA_DIR, 'ski_resorts')
    os.makedirs(ski_dir, exist_ok=True)

    # Try multiple potential ski resort data sources
    ski_urls = [
        'https://raw.githubusercontent.com/openskistats/openskistats.github.io/main/data/ski_areas.geojson',
        'https://raw.githubusercontent.com/openskistats/openskistats.github.io/gh-pages/data/ski_areas.geojson',
    ]

    ski_file = os.path.join(ski_dir, 'ski_areas.geojson')
    downloaded = None

    for ski_url in ski_urls:
        downloaded = download_file(ski_url, ski_file, "Ski Resorts")
        if downloaded:
            break

    if not downloaded:
        print("⚠ Ski resort data not available from OpenSkiStats")
        print("  Creating placeholder - you can add ski resort data manually later")
        # Create empty GeoJSON as placeholder
        import json
        with open(ski_file, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)

    return ski_dir

def download_national_parks():
    """Download National Park Service boundaries"""
    print("\n" + "="*60)
    print("DOWNLOADING NATIONAL PARK SERVICE BOUNDARIES")
    print("="*60)

    parks_dir = os.path.join(config.DATA_DIR, 'parks')
    os.makedirs(parks_dir, exist_ok=True)

    parks_file = os.path.join(parks_dir, 'nps_boundaries.geojson')

    # Skip if already exists
    if os.path.exists(parks_file):
        print(f"✓ NPS boundaries already exist: {parks_file}")
        return parks_dir

    # NPS Park Boundaries from ArcGIS REST API
    nps_url = "https://services1.arcgis.com/fBc8EJBxQRMcHlei/arcgis/rest/services/NPS_Land_Resources_Division_Boundary_and_Tract_Data_Service/FeatureServer/2/query?where=1%3D1&outFields=*&f=geojson"

    print("Downloading NPS boundaries from ArcGIS REST API...")
    downloaded = download_file(nps_url, parks_file, "NPS Park Boundaries")

    if downloaded:
        # Verify it's valid GeoJSON
        try:
            import json
            with open(parks_file, 'r') as f:
                data = json.load(f)
            print(f"✓ Downloaded {len(data.get('features', []))} NPS units")
        except Exception as e:
            print(f"⚠ Warning: Could not parse GeoJSON: {e}")

    return parks_dir

def download_national_forests():
    """Download National Forest System boundaries"""
    print("\n" + "="*60)
    print("DOWNLOADING NATIONAL FOREST BOUNDARIES")
    print("="*60)

    forests_dir = os.path.join(config.DATA_DIR, 'forests')
    os.makedirs(forests_dir, exist_ok=True)

    # Check if already exists
    forests_shp = os.path.join(forests_dir, 'S_USA.AdministrativeForest.shp')
    if os.path.exists(forests_shp):
        print(f"✓ National Forest boundaries already exist: {forests_shp}")
        return forests_dir

    # USFS Administrative Forest boundaries
    usfs_url = "https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.AdministrativeForest.zip"
    zip_file = os.path.join(forests_dir, 'national_forests.zip')

    print("Downloading USFS National Forest boundaries...")
    downloaded = download_file(usfs_url, zip_file, "National Forest Boundaries")

    if downloaded:
        extract_archive(zip_file, forests_dir)
        print(f"✓ National Forest boundaries extracted to: {forests_dir}")

    return forests_dir

def download_golf_courses():
    """Download golf course locations from OpenStreetMap via Overpass API"""
    print("\n" + "="*60)
    print("DOWNLOADING GOLF COURSE DATA")
    print("="*60)

    golf_dir = os.path.join(config.DATA_DIR, 'golf')
    os.makedirs(golf_dir, exist_ok=True)

    golf_file = os.path.join(golf_dir, 'us_golf_courses.geojson')

    # Skip if already exists
    if os.path.exists(golf_file):
        print(f"✓ Golf course data already exists: {golf_file}")
        return golf_dir

    print("Downloading golf courses from OpenStreetMap Overpass API...")
    print("This may take 5-10 minutes for CONUS coverage...")

    try:
        import overpy
        import json
        import time

        api = overpy.Overpass()

        # CONUS bounding box: [south, west, north, east]
        bbox_str = "24.396308,-125.0,49.384358,-66.93457"

        # Overpass query for golf courses in CONUS
        query = f"""
        [bbox:{bbox_str}][timeout:300];
        (
          way["leisure"="golf_course"];
          relation["leisure"="golf_course"];
        );
        out center;
        """

        print("  Querying Overpass API (this may take several minutes)...")
        result = api.query(query)

        # Convert to GeoJSON
        features = []

        # Process ways
        for way in result.ways:
            if hasattr(way, 'center_lat') and hasattr(way, 'center_lon'):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(way.center_lon), float(way.center_lat)]
                    },
                    "properties": {
                        "name": way.tags.get("name", "Unnamed Golf Course"),
                        "osm_id": way.id,
                        "osm_type": "way"
                    }
                }
                features.append(feature)

        # Process relations (for large golf course complexes)
        for relation in result.relations:
            if hasattr(relation, 'center_lat') and hasattr(relation, 'center_lon'):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(relation.center_lon), float(relation.center_lat)]
                    },
                    "properties": {
                        "name": relation.tags.get("name", "Unnamed Golf Course"),
                        "osm_id": relation.id,
                        "osm_type": "relation"
                    }
                }
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # Save to file
        with open(golf_file, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"✓ Downloaded {len(features)} golf courses")
        print(f"  Saved to: {golf_file}")

    except ImportError:
        print("✗ Error: overpy library not found")
        print("  Install with: pip install overpy")
        print("\n  Alternative: Download OSM data manually from:")
        print("    https://download.geofabrik.de/north-america.html")
        print(f"  Extract golf courses and save to: {golf_file}")
    except Exception as e:
        print(f"✗ Error downloading golf courses: {e}")
        print("\n  Alternative methods:")
        print("    1. Download OSM extract from Geofabrik")
        print("    2. Use osmium or ogr2ogr to filter leisure=golf_course")
        print(f"    3. Save result to: {golf_file}")

    return golf_dir

def main():
    """Main data download orchestration"""
    print("\n" + "="*70)
    print(" ADAPTIVE GRID DATA ACQUISITION")
    print("="*70)
    print(f"\nData will be saved to: {config.DATA_DIR}\n")

    # Download all datasets
    datasets = {
        'HRRR Grid & Terrain': download_hrrr_grid_and_terrain,
        'Natural Earth (Coastline/Lakes)': download_natural_earth_data,
        'National Parks (NPS)': download_national_parks,
        'National Forests (USFS)': download_national_forests,
        'Protected Areas (PAD-US)': download_protected_areas,
        'Urban Areas': download_urban_areas,
        'Primary Roads': download_primary_roads,
        'Ski Resorts': download_ski_resorts,
        'Golf Courses': download_golf_courses
    }

    results = {}
    for name, func in datasets.items():
        try:
            results[name] = func()
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
            results[name] = None

    # Summary
    print("\n" + "="*70)
    print(" DOWNLOAD SUMMARY")
    print("="*70)
    for name, path in results.items():
        status = "✓" if path and os.path.exists(path) else "✗"
        print(f"{status} {name}: {path}")

    print("\n" + "="*70)
    print("Data acquisition complete!")
    print("Large datasets (PAD-US, golf courses) may require manual download.")
    print("="*70)

if __name__ == '__main__':
    main()
