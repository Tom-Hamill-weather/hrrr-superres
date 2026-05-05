"""
Preprocessing: Create and cache unified feature buffers
This saves 5-10 minutes per grid generation run by pre-computing the expensive buffer operations

CURRENT CONFIGURATION:
- Ski resorts: config.SKI_RESORT_BUFFER_KM buffer at Tier 0 (93.75m)
- Coastlines: handled at runtime by WrfTerrainAnalyzer via WRF land_sea_mask distance transform
- Great Lakes (5): config.GREAT_LAKES_BUFFER_KM shoreline buffer at Tier 1 (187.5m)
- Specific water bodies (Puget Sound, SF Bay etc.): Tier 1 (187.5m)
- Inland lakes: config.INLAND_LAKES_BUFFER_KM buffer at Tier config.INLAND_LAKES_TIER
- Golf courses: excluded from cache (polygon-interior points generated in generate_points())
- Urban areas (Urbanized Areas ≥50k pop): config.URBAN_BUFFER_KM buffer at Tier 3
- Suburban areas (Urban Clusters >60th pct area): config.SUBURBAN_BUFFER_KM at Tier config.SUBURBAN_TIER
- Primary roads: config.HIGHWAY_BUFFER_KM buffer at Tier config.HIGHWAY_TIER
- National forests: Tier config.NATIONAL_FOREST_TIER (with NPS exclusion)

Run once after downloading/updating feature data:
    python3 preprocess_feature_buffers.py

Output: data/preprocessed/tier_buffers.gpkg (contains unified buffers for all tiers)
"""
import os
from collections import defaultdict
import geopandas as gpd
from shapely.ops import unary_union
import time
from mpl_toolkits.basemap import Basemap
from generate_adaptive_grid import DataLoader
import config

def preprocess_buffers():
    """Preprocess and cache all feature buffers"""
    print("="*70)
    print(" PREPROCESSING FEATURE BUFFERS")
    print("="*70)

    start_time = time.time()

    # Load all features
    print("\nLoading all features...")
    loader = DataLoader()
    loader.load_all()

    # Setup HRRR projection
    lats_hrrr = loader.hrrr_grid['lats']
    lons_hrrr = loader.hrrr_grid['lons']
    lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
    lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2

    proj = Basemap(
        projection='lcc',
        lat_0=lat_ref,
        lon_0=lon_ref,
        lat_1=lat_ref - 5,
        lat_2=lat_ref + 5,
        llcrnrlat=lats_hrrr.min(),
        urcrnrlat=lats_hrrr.max(),
        llcrnrlon=lons_hrrr.min(),
        urcrnrlon=lons_hrrr.max(),
        resolution='l'
    )

    hrrr_crs = proj.proj4string

    print("\n" + "="*70)
    print(" CREATING UNIFIED BUFFERS")
    print("="*70)

    all_geometries = defaultdict(list)

    # 1. OCEAN COASTLINES — handled at runtime by WrfTerrainAnalyzer.
    # The WRF land_sea_mask + distance_transform_edt computes coastal bands
    # per WRF pixel (~800m resolution), not cached as vector polygons here.
    print("\n1. Ocean coastlines: handled at runtime by WrfTerrainAnalyzer (not cached)")

    # 2. GREAT LAKES - Tier 1 (0.5km shoreline buffer)
    print("\n2. Processing Great Lakes...")
    great_lakes_gdf = loader.data.get('great_lakes')
    if great_lakes_gdf is not None and len(great_lakes_gdf) > 0:
        print(f"   {len(great_lakes_gdf)} Great Lakes, {great_lakes_gdf['coastline_length_km'].sum():.0f}km shoreline")
        gl_proj = great_lakes_gdf.to_crs(hrrr_crs)
        gl_buffered = gl_proj.geometry.buffer(config.GREAT_LAKES_BUFFER_KM * 1000)
        all_geometries[1].extend(gl_buffered.tolist())

    # 3. SKI RESORTS - Tier 0 (93.75m, 4km buffer)
    print("\n3. Processing ski resorts...")
    ski_gdf = loader.data.get('ski_resorts')
    if ski_gdf is not None and len(ski_gdf) > 0:
        print(f"   {len(ski_gdf)} locations ({config.SKI_RESORT_BUFFER_KM}km buffer)")
        ski_proj = ski_gdf.to_crs(hrrr_crs)
        ski_buffered = ski_proj.geometry.buffer(config.SKI_RESORT_BUFFER_KM * 1000)
        all_geometries[0].extend(ski_buffered.tolist())

    # 4. SPECIFIC WATER BODIES - Tier 1 (187.5m)
    print("\n4. Processing specific water bodies...")
    water_bodies_gdf = loader.data.get('water_bodies')
    if water_bodies_gdf is not None and len(water_bodies_gdf) > 0:
        print(f"   {len(water_bodies_gdf)} specific water bodies")
        wb_proj = water_bodies_gdf.to_crs(hrrr_crs)
        all_geometries[1].extend(wb_proj.geometry.tolist())

    # 5. INLAND LAKES - Tier 1 (0.5km buffer)
    print("\n5. Processing inland lakes...")
    inland_lakes_gdf = loader.data.get('inland_lakes')
    if inland_lakes_gdf is not None and len(inland_lakes_gdf) > 0:
        print(f"   {len(inland_lakes_gdf)} inland lakes, {inland_lakes_gdf['coastline_length_km'].sum():.0f}km shoreline")
        il_proj = inland_lakes_gdf.to_crs(hrrr_crs)
        # Buffer by 0.5km for Tier 1
        il_buffered = il_proj.geometry.buffer(config.INLAND_LAKES_BUFFER_KM * 1000)
        all_geometries[config.INLAND_LAKES_TIER].extend(il_buffered.tolist())

    # 6. GOLF COURSES — excluded from cached buffer.
    # Golf points are generated polygon-interior-only in generate_points() using a
    # globally-aligned 375m LCC grid.  Baking a 2.5 km cell buffer into the cache
    # causes entire HRRR cells near each course to be upgraded to Tier 2, which
    # produces rectangular gray blocks of 375m sub-grid points around every course.
    print("\n6. Golf courses: skipped (handled by polygon-based generation in generate_points)")

    # 7. URBAN AREAS - Tier 3 (Urbanized Areas ≥50k population)
    # Buffer by URBAN_BUFFER_KM (default 1.5 km = half HRRR cell).  Census UA
    # polygons are fragmented parcel-level geometries with many internal gaps;
    # without a buffer most 3 km HRRR cell centres land in the gaps and miss
    # the urban classification entirely.
    print("\n7. Processing urban areas (Tier 3)...")
    urban_gdf = loader.data.get('high_density_urban')
    if urban_gdf is not None and len(urban_gdf) > 0:
        print(f"   {len(urban_gdf)} urbanized areas (≥50k pop, "
              f"{config.URBAN_BUFFER_KM}km buffer)")
        urban_proj = urban_gdf.to_crs(hrrr_crs)
        urban_buffered = urban_proj.geometry.buffer(config.URBAN_BUFFER_KM * 1000)
        all_geometries[3].extend(urban_buffered.tolist())

    # 7b. SUBURBAN AREAS - Tier config.SUBURBAN_TIER (Urban Clusters, larger ones)
    print(f"\n7b. Processing suburban areas (Tier {config.SUBURBAN_TIER})...")
    suburban_gdf = loader.data.get('suburban')
    if suburban_gdf is not None and len(suburban_gdf) > 0:
        print(f"   {len(suburban_gdf)} suburban clusters "
              f"({config.SUBURBAN_BUFFER_KM}km buffer)")
        suburban_proj = suburban_gdf.to_crs(hrrr_crs)
        suburban_buffered = suburban_proj.geometry.buffer(config.SUBURBAN_BUFFER_KM * 1000)
        all_geometries[config.SUBURBAN_TIER].extend(suburban_buffered.tolist())

    # 8. PRIMARY ROADS - Tier config.HIGHWAY_TIER
    print(f"\n8. Processing roads (Tier {config.HIGHWAY_TIER})...")
    if config.INCLUDE_HIGHWAYS:
        roads_gdf = loader.data.get('roads')
        if roads_gdf is not None and len(roads_gdf) > 0:
            print(f"   {len(roads_gdf)} segments ({config.HIGHWAY_BUFFER_KM}km buffer)")
            roads_proj = roads_gdf.to_crs(hrrr_crs)
            roads_buffered = roads_proj.geometry.buffer(config.HIGHWAY_BUFFER_KM * 1000)
            all_geometries[config.HIGHWAY_TIER].extend(roads_buffered.tolist())

    # 9. NATIONAL FORESTS - Tier config.NATIONAL_FOREST_TIER
    print(f"\n9. Processing national forests (Tier {config.NATIONAL_FOREST_TIER})...")
    if config.INCLUDE_NATIONAL_FORESTS:
        forests_gdf = loader.data.get('national_forests')
        if forests_gdf is not None and len(forests_gdf) > 0:
            print(f"   {len(forests_gdf)} forest units")
            forests_proj = forests_gdf.to_crs(hrrr_crs)
            # Subtract NPS boundaries if configured
            if config.EXCLUDE_PARKS_FROM_FORESTS and hasattr(loader, 'nps_boundaries'):
                print("     Excluding NPS boundaries from forests...")
                parks_proj = loader.nps_boundaries.to_crs(hrrr_crs)
                parks_union = unary_union(parks_proj.geometry)
                forests_proj_geom = gpd.GeoSeries(forests_gdf.geometry.tolist(), crs=forests_gdf.crs).to_crs(hrrr_crs)
                forests_diff = forests_proj_geom.difference(parks_union)
                forests_diff = forests_diff[~forests_diff.is_empty]
                all_geometries[config.NATIONAL_FOREST_TIER].extend(forests_diff.tolist())
            else:
                all_geometries[config.NATIONAL_FOREST_TIER].extend(forests_proj.geometry.tolist())

    # Create unified buffers
    print("\n" + "="*70)
    print(" UNIFYING BUFFERS (this may take 5-10 minutes)")
    print("="*70)

    buffers = {}

    for tier in sorted(all_geometries.keys()):
        if not all_geometries[tier]:
            continue
        n = len(all_geometries[tier])
        print(f"\nUnifying Tier {tier} features ({n} geometries)...")
        if n > 500:
            print("  (Large geometry count — this may take several minutes)")
        geom_gdf = gpd.GeoSeries(all_geometries[tier], crs=hrrr_crs)
        buffers[tier] = geom_gdf.union_all()
        print(f"  ✓ Tier {tier} unified")

    # Simplify buffers
    print("\nSimplifying buffers (1m tolerance)...")
    for tier in buffers:
        buffers[tier] = buffers[tier].simplify(tolerance=1.0, preserve_topology=True)

    # Save to GeoPackage
    output_dir = os.path.join(config.DATA_DIR, 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'tier_buffers.gpkg')

    print("\n" + "="*70)
    print(" SAVING PREPROCESSED BUFFERS")
    print("="*70)

    # Create GeoDataFrame with all tier buffers
    buffer_data = []
    for tier, geom in buffers.items():
        buffer_data.append({
            'tier': tier,
            'geometry': geom
        })

    buffers_gdf = gpd.GeoDataFrame(buffer_data, crs=hrrr_crs)

    print(f"\nSaving to: {output_file}")
    buffers_gdf.to_file(output_file, driver='GPKG', layer='tier_buffers')

    # Coastal band ring geometries are no longer cached here.  Coastal classification
    # is handled at runtime by WrfTerrainAnalyzer using the WRF land_sea_mask and
    # scipy distance_transform_edt — no polygon buffers required.

    # Write config fingerprint so generate_adaptive_grid.py can detect a stale cache
    from shapely.geometry import Point as _Point
    fp_gdf = gpd.GeoDataFrame(
        [{'hash': config.cache_config_hash(), 'geometry': _Point(0, 0)}],
        crs='EPSG:4326'
    )
    fp_gdf.to_file(output_file, driver='GPKG', layer='config_fingerprint')
    print(f"  ✓ Config fingerprint stored: {config.cache_config_hash()}")

    file_size_mb = os.path.getsize(output_file) / 1e6
    elapsed = time.time() - start_time

    print(f"✓ Saved {len(buffers)} tier buffers ({file_size_mb:.1f} MB)")
    print(f"✓ Preprocessing complete in {elapsed/60:.1f} minutes")

    print("\n" + "="*70)
    print(" PREPROCESSING COMPLETE")
    print("="*70)
    print("\nCached buffers saved and ready to use!")
    print("\nConfiguration used:")
    print(f"  - Ski resorts buffer: {config.SKI_RESORT_BUFFER_KM}km (Tier 0, 93.75m)")
    print(f"  - Coastline: handled at runtime by WrfTerrainAnalyzer (WRF land_sea_mask)")
    print(f"  - Golf courses: excluded from cache (polygon-interior in generate_points)")
    print(f"  - Great Lakes buffer: {config.GREAT_LAKES_BUFFER_KM}km (Tier 1, 187.5m)")
    print(f"  - Specific water bodies: Tier 1 (187.5m)")
    print(f"  - Inland lakes buffer: {config.INLAND_LAKES_BUFFER_KM}km (Tier {config.INLAND_LAKES_TIER})")
    print(f"  - Urban areas (≥50k): {config.URBAN_BUFFER_KM}km buffer (Tier 3)")
    print(f"  - Suburban clusters:  {config.SUBURBAN_BUFFER_KM}km buffer (Tier {config.SUBURBAN_TIER})")
    print(f"  - Primary roads:      {config.HIGHWAY_BUFFER_KM}km buffer (Tier {config.HIGHWAY_TIER})")
    print(f"  - National forests:   Tier {config.NATIONAL_FOREST_TIER}")
    print("\nFuture grid generations can load these cached buffers")
    print("instead of recomputing them (saves 5-10 minutes per run)")

    return output_file

if __name__ == '__main__':
    preprocess_buffers()
