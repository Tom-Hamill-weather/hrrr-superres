"""
Preprocessing: Create and cache unified feature buffers
This saves 5-10 minutes per grid generation run by pre-computing the expensive buffer operations

Run once after downloading/updating feature data:
    python3 preprocess_feature_buffers.py

Output: data/preprocessed/tier_buffers.gpkg (contains unified buffers for all tiers)
"""
import os
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

    all_geometries = {1: [], 2: [], 3: []}

    # 1. OCEAN COASTLINES - Tier 1
    print("\n1. Processing ocean coastlines...")
    coastline_gdf = loader.data.get('ocean_coastline')
    if coastline_gdf is not None:
        significant_coastlines = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
        print(f"   {len(significant_coastlines)} segments (>50km), {significant_coastlines['length_km'].sum():.0f}km")
        coast_proj = significant_coastlines.to_crs(hrrr_crs)
        all_geometries[1].extend(coast_proj.geometry.tolist())

    # 2. LAKES - Tier 1
    print("\n2. Processing lakes...")
    lakes_gdf = loader.data.get('lakes')
    if lakes_gdf is not None and len(lakes_gdf) > 0:
        print(f"   {len(lakes_gdf)} features, {lakes_gdf['coastline_length_km'].sum():.0f}km")
        lakes_proj = lakes_gdf.to_crs(hrrr_crs)
        all_geometries[1].extend(lakes_proj.geometry.boundary.tolist())

    # 3. URBAN AREAS - Tier 2
    print("\n3. Processing urban areas...")
    urban_high = loader.data.get('high_density_urban')
    urban_low = loader.data.get('urban')
    urban_count = 0
    if urban_high is not None and len(urban_high) > 0:
        urban_proj = urban_high.to_crs(hrrr_crs)
        all_geometries[2].extend(urban_proj.geometry.boundary.tolist())
        urban_count += len(urban_high)
    if urban_low is not None and len(urban_low) > 0:
        cluster_proj = urban_low.to_crs(hrrr_crs)
        all_geometries[2].extend(cluster_proj.geometry.boundary.tolist())
        urban_count += len(urban_low)
    if urban_count > 0:
        print(f"   {urban_count} areas")

    # 4. SKI RESORTS - Tier 2
    print("\n4. Processing ski resorts...")
    ski_gdf = loader.data.get('ski_resorts')
    if ski_gdf is not None and len(ski_gdf) > 0:
        print(f"   {len(ski_gdf)} locations (2km buffer)")
        ski_proj = ski_gdf.to_crs(hrrr_crs)
        ski_buffered = ski_proj.geometry.buffer(2000)
        all_geometries[2].extend(ski_buffered.tolist())

    # 5. PRIMARY ROADS - Tier 3
    print("\n5. Processing roads...")
    roads_gdf = loader.data.get('roads')
    if roads_gdf is not None and len(roads_gdf) > 0:
        print(f"   {len(roads_gdf)} segments (500m buffer)")
        roads_proj = roads_gdf.to_crs(hrrr_crs)
        all_geometries[3].extend(roads_proj.geometry.tolist())

    # 6. NATIONAL PARKS - Tier 3
    print("\n6. Processing national parks...")
    parks_gdf = loader.data.get('national_parks')
    if parks_gdf is not None and len(parks_gdf) > 0:
        print(f"   {len(parks_gdf)} NPS units")
        parks_proj = parks_gdf.to_crs(hrrr_crs)
        all_geometries[3].extend(parks_proj.geometry.tolist())

    # 7. NATIONAL FORESTS - Tier 3
    print("\n7. Processing national forests...")
    forests_gdf = loader.data.get('national_forests')
    if forests_gdf is not None and len(forests_gdf) > 0:
        print(f"   {len(forests_gdf)} forest units")
        forests_proj = forests_gdf.to_crs(hrrr_crs)
        all_geometries[3].extend(forests_proj.geometry.tolist())

    # Create unified buffers
    print("\n" + "="*70)
    print(" UNIFYING BUFFERS (this may take 5-10 minutes)")
    print("="*70)

    buffers = {}

    # Tier 1 features: create distance-based buffers
    if all_geometries[1]:
        print("\nCreating tier 1-5 distance buffers from coastlines/lakes...")
        geom_gdf = gpd.GeoSeries(all_geometries[1], crs=hrrr_crs)

        print("  750m buffer (tier 1)...")
        buffers[1] = geom_gdf.buffer(750).union_all()

        print("  1500m buffer (tier 2)...")
        buffers[2] = geom_gdf.buffer(1500).union_all()

        print("  3km buffer (tier 3)...")
        buffers[3] = geom_gdf.buffer(3000).union_all()

        print("  6km buffer (tier 4)...")
        buffers[4] = geom_gdf.buffer(6000).union_all()

        print("  12km buffer (tier 5)...")
        buffers[5] = geom_gdf.buffer(12000).union_all()

    # Merge tier 2 features (urban + ski)
    if all_geometries[2]:
        print("\nMerging tier 2 features (urban + ski)...")
        geom_gdf = gpd.GeoSeries(all_geometries[2], crs=hrrr_crs)
        urban_ski_geom = geom_gdf.union_all()
        if 2 in buffers:
            buffers[2] = buffers[2].union(urban_ski_geom)
        else:
            buffers[2] = urban_ski_geom

    # Merge tier 3 features (roads + parks + forests)
    if all_geometries[3]:
        print("\nMerging tier 3 features (roads + parks + forests)...")
        print("  (This step takes longest due to road buffering)")
        geom_gdf = gpd.GeoSeries(all_geometries[3], crs=hrrr_crs)
        tier3_features = geom_gdf.buffer(500).union_all()
        if 3 in buffers:
            buffers[3] = buffers[3].union(tier3_features)
        else:
            buffers[3] = tier3_features

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

    file_size_mb = os.path.getsize(output_file) / 1e6
    elapsed = time.time() - start_time

    print(f"✓ Saved {len(buffers)} tier buffers ({file_size_mb:.1f} MB)")
    print(f"✓ Preprocessing complete in {elapsed/60:.1f} minutes")

    print("\n" + "="*70)
    print(" PREPROCESSING COMPLETE")
    print("="*70)
    print("\nFuture grid generations will load these cached buffers")
    print("instead of recomputing them (saves 5-10 minutes per run)")
    print("\nTo use cached buffers, update generate_adaptive_grid_BINARY_FINAL.py")
    print("to load from:", output_file)

    return output_file

if __name__ == '__main__':
    preprocess_buffers()
