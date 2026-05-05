"""
Preprocess features for the hi-res binary-mask point generator.

Outputs to data/hires_preprocessed/:
  terrain_tier.npy   — WRF-resolution terrain tier (uint8; 1/2/3/5)
  coastal_tier.npy   — WRF-resolution coastal tier (uint8; 1/2/3/5/99)
  wrf_index.npz      — WRF 1-D LCC x/y arrays for per-patch searchsorted lookup
  features_lcc.gpkg  — Vector features buffered and projected to HRRR LCC
  hires_params.json  — Hi-res grid geometry parameters
  config_hash.txt    — Cache fingerprint for invalidation

Run once, then again whenever config.py or source data change:
    python3 preprocess_hires_features.py
"""

import os
import json
import time

import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from pyproj import Proj

import config
from generate_adaptive_grid import DataLoader, WrfTerrainAnalyzer

HRRR_PROJ_STR = ('+proj=lcc +lat_0=38.5 +lon_0=-97.5 '
                 '+lat_1=38.5 +lat_2=38.5 +units=m +R=6371229 +no_defs')


# ---------------------------------------------------------------------------
# Grid parameter computation
# ---------------------------------------------------------------------------

def compute_hires_params(lats_hrrr, lons_hrrr):
    """Return a dict of hi-res grid parameters anchored at HRRR cell [0,0]."""
    hrrr_proj = Proj(proj='lcc', lat_0=38.5, lon_0=-97.5,
                     lat_1=38.5, lat_2=38.5, R=6371229)
    x00, y00 = hrrr_proj(float(lons_hrrr[0, 0]), float(lats_hrrr[0, 0]))

    N_HIRES  = 32                     # sub-pixels per HRRR cell side
    HIRES_DX = 3000.0 / N_HIRES      # 93.75 m

    HRRR_NY, HRRR_NX = lats_hrrr.shape
    HIRES_NY = HRRR_NY * N_HIRES     # full-domain hi-res row count
    HIRES_NX = HRRR_NX * N_HIRES     # full-domain hi-res col count

    # Origin = center of hi-res pixel [0, 0] (southwest corner of HRRR domain)
    hires_x0 = x00 - (N_HIRES / 2.0 - 0.5) * HIRES_DX  # = x00 - 15.5 * 93.75
    hires_y0 = y00 - (N_HIRES / 2.0 - 0.5) * HIRES_DX

    return dict(
        hrrr_proj_str=HRRR_PROJ_STR,
        x00=float(x00),
        y00=float(y00),
        N_HIRES=N_HIRES,
        HIRES_DX=HIRES_DX,
        HRRR_NY=HRRR_NY,
        HRRR_NX=HRRR_NX,
        HIRES_NY=HIRES_NY,
        HIRES_NX=HIRES_NX,
        hires_x0=float(hires_x0),
        hires_y0=float(hires_y0),
    )


# ---------------------------------------------------------------------------
# WRF tier maps and index arrays
# ---------------------------------------------------------------------------

def compute_wrf_outputs(out_dir):
    """Run WrfTerrainAnalyzer and save terrain_tier, coastal_tier, wrf_index."""
    wrf = WrfTerrainAnalyzer().load_and_classify()

    np.save(os.path.join(out_dir, 'terrain_tier.npy'),
            wrf.terrain_tier.astype(np.uint8))
    np.save(os.path.join(out_dir, 'coastal_tier.npy'),
            wrf.coastal_tier.astype(np.uint8))

    print(f'  ✓ terrain_tier.npy  {wrf.terrain_tier.shape}')
    print(f'  ✓ coastal_tier.npy  {wrf.coastal_tier.shape}')

    # Project WRF lats/lons to HRRR LCC so that generate_hires_points.py
    # can do fast nearest-neighbour lookup via searchsorted.
    print('  Projecting WRF grid to HRRR LCC...')
    hrrr_proj = Proj(proj='lcc', lat_0=38.5, lon_0=-97.5,
                     lat_1=38.5, lat_2=38.5, R=6371229)
    wrf_x, wrf_y = hrrr_proj(wrf.lons, wrf.lats)   # (ny_wrf, nx_wrf)
    ny_wrf, nx_wrf = wrf_x.shape

    # Use the central row/column as 1-D lookup arrays (monotonically increasing
    # in the HRRR LCC projection for the CONUS domain).
    wrf_x_row = wrf_x[ny_wrf // 2, :]    # (nx_wrf,)  west → east
    wrf_y_col = wrf_y[:, nx_wrf // 2]    # (ny_wrf,)  south → north

    np.savez(os.path.join(out_dir, 'wrf_index.npz'),
             wrf_x_row=wrf_x_row,
             wrf_y_col=wrf_y_col,
             ny_wrf=np.array(ny_wrf),
             nx_wrf=np.array(nx_wrf))
    print(f'  ✓ wrf_index.npz     WRF grid {ny_wrf} × {nx_wrf}')

    return wrf


# ---------------------------------------------------------------------------
# Vector feature processing
# ---------------------------------------------------------------------------

def save_layer(gdf, layer_name, gpkg_path, buffer_m=0.0):
    """Project GDF to HRRR LCC, optionally buffer, write to GPKG layer."""
    if gdf is None or len(gdf) == 0:
        print(f'  ⚠  {layer_name}: no data, skipping')
        return

    t0 = time.time()
    projected = gdf.to_crs(HRRR_PROJ_STR)
    if buffer_m > 0:
        projected = projected.copy()
        projected['geometry'] = projected.geometry.buffer(buffer_m)
        projected = projected[~projected.is_empty].reset_index(drop=True)

    projected.to_file(gpkg_path, driver='GPKG', layer=layer_name)
    label = f'(buffered {buffer_m/1000:.2f} km)' if buffer_m > 0 else ''
    print(f'  ✓ {layer_name}: {len(projected)} features {label}  '
          f'({time.time()-t0:.1f}s)')


def compute_coastal_vector_buffers(out_dir, loader):
    """Create coast_t1/t2/t3 GPKG layers from GSHHG coastline buffers.

    coast_t1: 0–750 m from shoreline  → Tier 1 (stride 2, 188m spacing)
    coast_t2: 0–1500 m from shoreline → Tier 2 (stride 4, 375m spacing)
    coast_t3: 0–2250 m from shoreline → Tier 3 (stride 8, 750m spacing)

    Cumulative buffers work correctly with OR-accumulation in the point
    generator: the finer criterion fires first and coarser layers can only
    add pixels in the ring between their boundary and the finer boundary.
    """
    gpkg_path = os.path.join(out_dir, 'features_lcc.gpkg')

    coast_gdf = loader.data.get('ocean_coastline')
    if coast_gdf is None or len(coast_gdf) == 0:
        print('  ⚠  ocean_coastline: no data — coastal buffers skipped')
        return

    t0 = time.time()
    print(f'  Projecting coastline ({len(coast_gdf)} segments) to HRRR LCC...')
    coast_lcc = coast_gdf.to_crs(HRRR_PROJ_STR)
    # Simplify to remove redundant vertices (50 m tolerance in LCC metres)
    coast_lcc = coast_lcc.copy()
    coast_lcc['geometry'] = coast_lcc.geometry.simplify(50)
    coast_lcc = coast_lcc[~coast_lcc.is_empty].reset_index(drop=True)
    print(f'  Simplification done ({time.time()-t0:.1f}s)')

    # Cumulative buffers from the shoreline
    for tier, buf_m in [(1, 750), (2, 1500), (3, 2250)]:
        layer = f'coast_t{tier}'
        t1 = time.time()
        buffered = coast_lcc.copy()
        buffered['geometry'] = coast_lcc.geometry.buffer(buf_m)
        buffered = buffered[~buffered.is_empty].reset_index(drop=True)
        buffered.to_file(gpkg_path, driver='GPKG', layer=layer)
        print(f'  ✓ {layer}: {len(buffered)} polygons '
              f'(±{buf_m/1000:.2f}km buffer)  ({time.time()-t1:.1f}s)')


def compute_vector_features(out_dir, loader):
    """Load all vector datasets, project + buffer, write to features_lcc.gpkg."""
    gpkg_path = os.path.join(out_dir, 'features_lcc.gpkg')

    save_layer(loader.data.get('golf_courses'),      'golf_courses', gpkg_path)
    save_layer(loader.data.get('ski_resorts'),        'ski_resorts',  gpkg_path,
               buffer_m=config.SKI_RESORT_BUFFER_KM * 1000)
    save_layer(loader.data.get('great_lakes'),        'great_lakes',  gpkg_path,
               buffer_m=config.GREAT_LAKES_BUFFER_KM * 1000)
    save_layer(loader.data.get('water_bodies'),       'water_bodies', gpkg_path)
    save_layer(loader.data.get('inland_lakes'),       'inland_lakes', gpkg_path,
               buffer_m=config.INLAND_LAKES_BUFFER_KM * 1000)
    save_layer(loader.data.get('high_density_urban'), 'urban',        gpkg_path,
               buffer_m=config.URBAN_BUFFER_KM * 1000)
    save_layer(loader.data.get('suburban'),           'suburban',     gpkg_path,
               buffer_m=config.SUBURBAN_BUFFER_KM * 1000)

    if config.INCLUDE_HIGHWAYS:
        save_layer(loader.data.get('roads'), 'roads', gpkg_path,
                   buffer_m=config.HIGHWAY_BUFFER_KM * 1000)

    if config.INCLUDE_NATIONAL_FORESTS:
        forests = loader.data.get('national_forests')
        if forests is not None and len(forests) > 0:
            t0 = time.time()
            forests_proj = forests.to_crs(HRRR_PROJ_STR)
            if config.EXCLUDE_PARKS_FROM_FORESTS and hasattr(loader, 'nps_boundaries'):
                print('    Excluding NPS boundaries from forests...')
                parks_proj = loader.nps_boundaries.to_crs(HRRR_PROJ_STR)
                parks_union = unary_union(parks_proj.geometry)
                forests_proj = forests_proj.copy()
                forests_proj['geometry'] = forests_proj.geometry.difference(parks_union)
                forests_proj = forests_proj[~forests_proj.is_empty].reset_index(drop=True)
            forests_proj.to_file(gpkg_path, driver='GPKG', layer='forests')
            print(f'  ✓ forests: {len(forests_proj)} features  '
                  f'({time.time()-t0:.1f}s)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    out_dir = os.path.join(config.DATA_DIR, 'hires_preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print(' PREPROCESSING HI-RES FEATURES')
    print('=' * 70)

    # ── 1. HRRR grid ─────────────────────────────────────────────────────────
    print('\n[1/5] Loading HRRR grid...')
    hrrr_dir = os.path.join(config.DATA_DIR, 'hrrr')
    lats_hrrr = np.load(os.path.join(hrrr_dir, 'hrrr_lats.npy'))
    lons_hrrr = np.load(os.path.join(hrrr_dir, 'hrrr_lons.npy'))
    print(f'  HRRR grid: {lats_hrrr.shape[0]} × {lats_hrrr.shape[1]}')

    params = compute_hires_params(lats_hrrr, lons_hrrr)
    with open(os.path.join(out_dir, 'hires_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    n_pixels = params['HIRES_NY'] * params['HIRES_NX']
    print(f'  Hi-res grid: {params["HIRES_NY"]} × {params["HIRES_NX"]} '
          f'= {n_pixels / 1e9:.2f}B pixels at {params["HIRES_DX"]}m')
    print(f'  ✓ hires_params.json saved')

    # ── 2. WRF terrain & coastal tiers ──────────────────────────────────────
    print('\n[2/5] Computing WRF terrain and coastal tiers...')
    compute_wrf_outputs(out_dir)

    # ── 3. Vector features ───────────────────────────────────────────────────
    print('\n[3/5] Loading all geospatial features...')
    loader = DataLoader()
    loader.load_all()

    print('\n[4/6] Projecting and buffering vector features → features_lcc.gpkg...')
    compute_vector_features(out_dir, loader)

    # ── 5. Coastal vector buffers ─────────────────────────────────────────────
    print('\n[5/6] Computing vector coastline buffers → features_lcc.gpkg...')
    compute_coastal_vector_buffers(out_dir, loader)

    # ── 6. Config fingerprint ─────────────────────────────────────────────────
    print('\n[6/6] Writing config fingerprint...')
    h = config.cache_config_hash()
    with open(os.path.join(out_dir, 'config_hash.txt'), 'w') as f:
        f.write(h)
    print(f'  ✓ Hash: {h}')

    elapsed = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f' PREPROCESSING COMPLETE in {elapsed / 60:.1f} min')
    print(f'{"=" * 70}')
    print(f'  Output: {out_dir}')
    print()
    print('Next steps:')
    print('  python3 generate_hires_points.py --test-patches 4 --patch-offset 50  # timing test')
    print('  python3 generate_hires_points.py                                      # full domain')


if __name__ == '__main__':
    main()
