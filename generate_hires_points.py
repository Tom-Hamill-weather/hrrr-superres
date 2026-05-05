"""
Generate hi-res adaptive grid points via binary-mask patch processing.

Architecture:
  - The full CONUS domain is divided into patches of HIRES_PATCH_SIZE HRRR
    cells × 32 hi-res pixels each (default 64 cells → 2048 × 2048 hi-res px).
  - For each patch, a binary output mask (0/1) is built by looping over
    criteria in finest-first order and OR-ing in stride-selected pixels:
        output |= criterion_mask & stride_mask(tier)
    where stride = 2^tier (Tier 0→1, Tier 1→2, Tier 2→4, Tier 3→8, Tier 5→32).
  - A 1 in the output mask is never overwritten to 0.
  - Selected pixels are inverse-projected to lat/lon and written to NetCDF.

Criteria handled:
  Tier 0 (stride 1 ): golf courses, ski resort buffers
  Tier 1 (stride 2 ): near coast (0–0.75 km GSHHG vector buffer), Great Lakes
                       shores, specific water bodies, extreme terrain (σ>400 m)
  Tier 2 (stride 4 ): mid coast (0–1.5 km buffer), inland lakes,
                       suburban areas, rugged terrain (σ 250–400 m)
  Tier 3 (stride 8 ): outer coast (0–2.25 km buffer), urban areas, highways,
                       national forests, moderate terrain (σ 100–250 m)
  Tier 4 (stride 16): background (everywhere not already covered, 1.5 km spacing)

Prerequisites:
    python3 preprocess_hires_features.py

Usage:
    python3 generate_hires_points.py                    # full CONUS domain
    python3 generate_hires_points.py --test-patches 3   # timing estimate only
    python3 generate_hires_points.py --patch-size 32    # smaller patches (debug)
    python3 generate_hires_points.py --output my_out.nc

Output:
    output/hires_points.nc   (lat/lon float32 with zlib compression)
"""

import os
import sys
import json
import time
import math
import itertools
import argparse

import numpy as np
import geopandas as gpd
import netCDF4 as nc
from pyproj import Proj
from shapely.geometry import box as shp_box
from rasterio.transform import from_origin
from rasterio.features import rasterize as rio_rasterize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree

import config

HRRR_PROJ_STR = ('+proj=lcc +lat_0=38.5 +lon_0=-97.5 '
                 '+lat_1=38.5 +lat_2=38.5 +units=m +R=6371229 +no_defs')

# stride = 2^tier for tiers 0–3; tier 4→16 (1.5km), tier 5→32 (3km)
TIER_STRIDE = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32}

# Processing order: finest tier first so that coarser criteria never
# overwrite already-activated fine pixels (the |= ensures this anyway,
# but finest-first avoids redundant computation on already-lit pixels).
CRITERIA = [
    ('golf_courses', 0),
    ('ski_resorts',  0),
    ('coast_t1',     1),
    ('great_lakes',  1),
    ('water_bodies', 1),
    ('terrain_t1',   1),
    ('coast_t2',     2),
    ('inland_lakes', 2),
    ('suburban',     2),
    ('terrain_t2',   2),
    ('coast_t3',     3),
    ('urban',        3),
    ('roads',        3),
    ('forests',      3),
    ('terrain_t3',   3),
    ('background',   4),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preprocessed(preproc_dir):
    """Load all preprocessed arrays and GeoDataFrames from preproc_dir."""
    params_path = os.path.join(preproc_dir, 'hires_params.json')
    if not os.path.exists(params_path):
        sys.exit(f'ERROR: {params_path} not found.\n'
                 f'Run:  python3 preprocess_hires_features.py')

    with open(params_path) as f:
        params = json.load(f)

    terrain_tier = np.load(os.path.join(preproc_dir, 'terrain_tier.npy'))

    idx = np.load(os.path.join(preproc_dir, 'wrf_index.npz'))
    wrf_x_row = idx['wrf_x_row']   # (nx_wrf,) WRF x in HRRR LCC, west→east
    wrf_y_col = idx['wrf_y_col']   # (ny_wrf,) WRF y in HRRR LCC, south→north

    gpkg_path = os.path.join(preproc_dir, 'features_lcc.gpkg')
    if not os.path.exists(gpkg_path):
        sys.exit(f'ERROR: {gpkg_path} not found.\n'
                 f'Run:  python3 preprocess_hires_features.py')

    import fiona
    available = set(fiona.listlayers(gpkg_path))
    gdfs = {}
    for layer in available:
        gdf = gpd.read_file(gpkg_path, layer=layer)
        gdfs[layer] = gdf
        print(f'  ✓ {layer}: {len(gdf)} features')

    return params, terrain_tier, wrf_x_row, wrf_y_col, gdfs


# ---------------------------------------------------------------------------
# Per-patch helpers
# ---------------------------------------------------------------------------

def rasterize_layer(gdf, px, py, dx):
    """Rasterize a GeoDataFrame layer onto the patch hi-res grid.

    px : 1-D float array of patch x-coords (HRRR LCC metres), west→east, (W,)
    py : 1-D float array of patch y-coords (HRRR LCC metres), south→north, (H,)
    dx : pixel size in metres (= 93.75)

    Returns a bool mask of shape (H, W) with row 0 = southernmost row,
    or None if no features intersect the patch.
    """
    if gdf is None or len(gdf) == 0:
        return None

    H, W = len(py), len(px)
    patch_bounds = (px[0] - dx/2, py[0] - dx/2, px[-1] + dx/2, py[-1] + dx/2)

    # Fast spatial filter via R-tree index
    hits = list(gdf.sindex.intersection(patch_bounds))
    if not hits:
        return None

    candidates = gdf.iloc[hits]
    patch_box  = shp_box(*patch_bounds)
    exact = candidates[candidates.intersects(patch_box)]
    if len(exact) == 0:
        return None

    # rasterio affine transform: row 0 = north, y decreases downward
    transform = from_origin(px[0] - dx/2, py[-1] + dx/2, dx, dx)
    shapes = [
        (geom.__geo_interface__, 1)
        for geom in exact.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return None

    # Rasterize with default all_touched=False (suitable for polygon features)
    mask_north_first = rio_rasterize(
        shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8
    )
    # Flip rows so that row 0 = southernmost (matches ascending py ordering)
    return mask_north_first[::-1, :].astype(bool)


def make_stride_mask(H, W, r0, c0, stride):
    """Return bool (H, W) selecting globally-aligned rows and columns.

    A pixel at patch-local (r, c) is selected iff
      (r0 + r) % stride == 0  AND  (c0 + c) % stride == 0.
    """
    R = ((r0 + np.arange(H)) % stride == 0)[:, np.newaxis]   # (H, 1)
    C = ((c0 + np.arange(W)) % stride == 0)[np.newaxis, :]   # (1, W)
    return R & C   # broadcast → (H, W)


# ---------------------------------------------------------------------------
# Output NetCDF
# ---------------------------------------------------------------------------

def open_output_nc(path):
    ds = nc.Dataset(path, 'w', format='NETCDF4')
    ds.createDimension('points', None)  # unlimited
    lat_v = ds.createVariable('latitude',  'f4', ('points',),
                              zlib=True, complevel=4, chunksizes=(65536,))
    lon_v = ds.createVariable('longitude', 'f4', ('points',),
                              zlib=True, complevel=4, chunksizes=(65536,))
    tier_v = ds.createVariable('tier', 'i1', ('points',),
                               zlib=True, complevel=4, chunksizes=(65536,))
    lat_v.units = 'degrees_north'
    lon_v.units = 'degrees_east'
    tier_v.long_name = 'Grid Tier (lower = finer resolution)'
    tier_v.description = ('Tier 0: 93.75m (golf/ski), '
                          'Tier 1: 187.5m (coast/lakes/extreme terrain), '
                          'Tier 2: 375m (mid-coast/suburban/rugged terrain), '
                          'Tier 3: 750m (outer-coast/urban/roads/forests/moderate terrain), '
                          'Tier 4: 1500m (background)')
    ds.description = 'Hi-res adaptive grid points'
    ds.resolution_m = 93.75
    ds.generator = 'generate_hires_points.py'
    return ds


def append_points(ds, lats, lons, tiers):
    """Append lat/lon/tier arrays to the unlimited 'points' dimension."""
    n = ds.dimensions['points'].size
    m = len(lats)
    ds.variables['latitude'][n:n + m]  = lats.astype(np.float32)
    ds.variables['longitude'][n:n + m] = lons.astype(np.float32)
    ds.variables['tier'][n:n + m]      = tiers.astype(np.int8)


# ---------------------------------------------------------------------------
# Main patch loop
# ---------------------------------------------------------------------------

def process_patches(params, terrain_tier,
                    wrf_x_row, wrf_y_col, gdfs, out_ds,
                    patch_hrrr_size, n_patches_limit=None, patch_offset=0):
    """Iterate over all patches and write selected points to out_ds.

    Returns (total_points, list_of_per-patch_elapsed_seconds, total_patches).
    """
    HIRES_DX = params['HIRES_DX']
    HIRES_NY = params['HIRES_NY']
    HIRES_NX = params['HIRES_NX']
    hires_x0 = params['hires_x0']
    hires_y0 = params['hires_y0']

    hires_x = hires_x0 + np.arange(HIRES_NX) * HIRES_DX   # (HIRES_NX,)
    hires_y = hires_y0 + np.arange(HIRES_NY) * HIRES_DX   # (HIRES_NY,)

    PATCH_PIX  = patch_hrrr_size * 32   # hi-res pixels per patch side
    n_pr = math.ceil(HIRES_NY / PATCH_PIX)
    n_pc = math.ceil(HIRES_NX / PATCH_PIX)
    total_patches = n_pr * n_pc

    hrrr_proj = Proj(proj='lcc', lat_0=38.5, lon_0=-97.5,
                     lat_1=38.5, lat_2=38.5, R=6371229)

    total_points = 0
    patch_times  = []

    # Per-criterion timing accumulators (for the verbose breakdown)
    crit_times = {name: 0.0 for name, _ in CRITERIA}

    patches_run = 0
    for pi, (pr, pc) in enumerate(itertools.product(range(n_pr), range(n_pc))):
        if pi < patch_offset:
            continue
        if n_patches_limit is not None and patches_run >= n_patches_limit:
            break
        patches_run += 1

        t_patch = time.time()

        # Hi-res index bounds for this patch
        r0 = pr * PATCH_PIX
        r1 = min(r0 + PATCH_PIX, HIRES_NY)
        c0 = pc * PATCH_PIX
        c1 = min(c0 + PATCH_PIX, HIRES_NX)
        H, W = r1 - r0, c1 - c0

        px = hires_x[c0:c1]   # (W,) west→east
        py = hires_y[r0:r1]   # (H,) south→north

        # ── WRF nearest-neighbour index ─────────────────────────────────────
        # searchsorted on the monotone 1-D middle-row/column arrays.
        wrf_ci = np.searchsorted(wrf_x_row, px).clip(0, len(wrf_x_row) - 1)
        wrf_ri = np.searchsorted(wrf_y_col, py).clip(0, len(wrf_y_col) - 1)
        WRF_R, WRF_C = np.ix_(wrf_ri, wrf_ci)          # (H,1) × (1,W)
        terrain_patch = terrain_tier[WRF_R, WRF_C]      # (H, W) uint8

        # ── Per-pixel tier map (255 = unset; finest-first processing ensures
        #    the first write is always the finest applicable tier) ────────────
        tier_out = np.full((H, W), 255, dtype=np.uint8)

        for name, tier in CRITERIA:
            stride = TIER_STRIDE[tier]
            tc = time.time()

            # --- compute binary mask for this criterion ---
            if name == 'golf_courses':
                mask = rasterize_layer(gdfs.get('golf_courses'), px, py, HIRES_DX)
            elif name == 'ski_resorts':
                mask = rasterize_layer(gdfs.get('ski_resorts'), px, py, HIRES_DX)
            elif name == 'coast_t1':
                mask = rasterize_layer(gdfs.get('coast_t1'), px, py, HIRES_DX)
            elif name == 'great_lakes':
                mask = rasterize_layer(gdfs.get('great_lakes'), px, py, HIRES_DX)
            elif name == 'water_bodies':
                mask = rasterize_layer(gdfs.get('water_bodies'), px, py, HIRES_DX)
            elif name == 'terrain_t1':
                mask = (terrain_patch <= 1)
            elif name == 'coast_t2':
                mask = rasterize_layer(gdfs.get('coast_t2'), px, py, HIRES_DX)
            elif name == 'inland_lakes':
                mask = rasterize_layer(gdfs.get('inland_lakes'), px, py, HIRES_DX)
            elif name == 'suburban':
                mask = rasterize_layer(gdfs.get('suburban'), px, py, HIRES_DX)
            elif name == 'terrain_t2':
                mask = (terrain_patch == 2)
            elif name == 'coast_t3':
                mask = rasterize_layer(gdfs.get('coast_t3'), px, py, HIRES_DX)
            elif name == 'urban':
                mask = rasterize_layer(gdfs.get('urban'), px, py, HIRES_DX)
            elif name == 'roads':
                mask = rasterize_layer(gdfs.get('roads'), px, py, HIRES_DX)
            elif name == 'forests':
                mask = rasterize_layer(gdfs.get('forests'), px, py, HIRES_DX)
            elif name == 'terrain_t3':
                mask = (terrain_patch == 3)
            elif name == 'background':
                # All pixels get stride-16 coverage
                mask = np.ones((H, W), dtype=bool)
            else:
                continue

            crit_times[name] += time.time() - tc

            if mask is None or not mask.any():
                continue

            # Apply globally-aligned stride; only record tier for pixels not
            # yet activated (finest-first order guarantees finest tier wins).
            sm = make_stride_mask(H, W, r0, c0, stride)
            new_px = mask & sm & (tier_out == 255)
            tier_out[new_px] = tier

        # ── Collect selected pixels and inverse-project ─────────────────────
        sel_r, sel_c = np.where(tier_out != 255)
        n_sel = len(sel_r)
        if n_sel > 0:
            sel_lons, sel_lats = hrrr_proj(px[sel_c], py[sel_r], inverse=True)
            append_points(out_ds, np.asarray(sel_lats), np.asarray(sel_lons),
                          tier_out[sel_r, sel_c])
            total_points += n_sel

        elapsed = time.time() - t_patch
        patch_times.append(elapsed)

        if patches_run <= 5 or patches_run % 100 == 0:
            print(f'  [{pi + 1:4d}/{total_patches}] ({pr:3d},{pc:3d})  '
                  f'{H}×{W} px → {n_sel:7,} pts  {elapsed:.2f}s  '
                  f'running total: {total_points:,}')

    return total_points, patch_times, total_patches, crit_times


# ---------------------------------------------------------------------------
# Density visualization
# ---------------------------------------------------------------------------

def create_density_visualization(nc_path):
    """Create a point-density PNG in HRRR LCC projection.

    Reads the written NetCDF back, assigns each point to its nearest HRRR cell
    via KDTree, then renders a pcolormesh with discrete colour bins.

    The PNG is saved alongside the NetCDF with '_density.png' suffix.
    """
    print('\n' + '=' * 70)
    print(' CREATING DENSITY VISUALIZATION')
    print('=' * 70)

    # ── Load HRRR grid ───────────────────────────────────────────────────────
    hrrr_dir = os.path.join(config.DATA_DIR, 'hrrr')
    lats_hrrr = np.load(os.path.join(hrrr_dir, 'hrrr_lats.npy'))
    lons_hrrr = np.load(os.path.join(hrrr_dir, 'hrrr_lons.npy'))
    shape = lats_hrrr.shape
    print(f'\n  HRRR grid: {shape[0]} × {shape[1]} = {shape[0]*shape[1]:,} cells')

    # ── Read points from NetCDF ──────────────────────────────────────────────
    print(f'  Reading points from {nc_path}...')
    with nc.Dataset(nc_path, 'r') as ds:
        pt_lats = ds.variables['latitude'][:]
        pt_lons = ds.variables['longitude'][:]
    total_pts = len(pt_lats)
    print(f'  {total_pts:,} points loaded')

    # ── Project to HRRR LCC ─────────────────────────────────────────────────
    print('\n[1/4] Building HRRR LCC projection...')
    lat_0, lon_0, lat_1, lat_2 = 38.5, -97.5, 38.5, 38.5

    temp_proj = Basemap(
        projection='lcc', lat_0=lat_0, lon_0=lon_0,
        lat_1=lat_1, lat_2=lat_2,
        llcrnrlat=21, urcrnrlat=53, llcrnrlon=-135, urcrnrlon=-60,
        resolution=None
    )
    x_hrrr, y_hrrr = temp_proj(lons_hrrr, lats_hrrr)
    width  = x_hrrr.max() - x_hrrr.min()
    height = y_hrrr.max() - y_hrrr.min()

    m = Basemap(
        projection='lcc', lat_0=lat_0, lon_0=lon_0,
        lat_1=lat_1, lat_2=lat_2,
        width=width, height=height, resolution='l', area_thresh=1000
    )

    # ── KDTree density count ─────────────────────────────────────────────────
    print('[2/4] Computing point density per HRRR cell...')
    hrrr_pts_proj = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])
    tree = cKDTree(hrrr_pts_proj)

    x_pts, y_pts = temp_proj(pt_lons, pt_lats)
    _, indices = tree.query(np.column_stack([x_pts, y_pts]), k=1)

    point_density = np.bincount(indices, minlength=hrrr_pts_proj.shape[0]).reshape(shape)
    print(f'    Max points per cell:  {point_density.max():,}')
    print(f'    Mean points per cell: {point_density.mean():.1f}')

    # ── Discrete colour bins ─────────────────────────────────────────────────
    bins   = [0, 5, 10, 50, 100, 200, 400, 800, 1000, 1100]
    labels = ['≤4', '5-9', '10-49', '50-99', '100-199',
              '200-399', '400-799', '800-999', '1000+']
    colors = [
        'White',   '#C4E8FF', '#8FB3FF', '#42F742',
        'Yellow',  'Gold',    'Orange',  '#F6A3AE', 'Orchid',
    ]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)

    # ── Plot ─────────────────────────────────────────────────────────────────
    print('[3/4] Drawing map...')
    fig = plt.figure(figsize=(18, 11))
    ax  = fig.add_axes([0.05, 0.05, 0.85, 0.90])

    m.drawcoastlines(linewidth=0.5, color='black', zorder=5)
    m.drawcountries(linewidth=0.5, color='black', zorder=5)
    m.drawstates(linewidth=0.3,   color='gray',  zorder=5)
    m.drawparallels(np.arange(20, 55, 5),    labels=[1, 0, 0, 0],
                    fontsize=13.5, linewidth=0.3, color='gray')
    m.drawmeridians(np.arange(-130, -60, 10), labels=[0, 0, 0, 1],
                    fontsize=13.5, linewidth=0.3, color='gray')

    x_map, y_map = m(lons_hrrr, lats_hrrr)
    density_masked = np.ma.masked_where(point_density == 0, point_density)

    pcm = m.pcolormesh(
        x_map, y_map, density_masked,
        cmap=cmap, norm=norm, shading='auto', zorder=1, rasterized=True
    )

    cax  = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    cbar = plt.colorbar(pcm, cax=cax, spacing='uniform', extend='max')
    tick_pos = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(labels))]
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels(labels, fontsize=13)
    cbar.set_label('Points per HRRR Cell', fontsize=14)

    n_cells = int((point_density > 0).sum())
    ax.set_title(
        f'Hi-Res Adaptive Grid Point Density\n'
        f'Total Points: {total_pts:,} | '
        f'HRRR Cells with Points: {n_cells:,} / {point_density.size:,} '
        f'({100 * n_cells / point_density.size:.1f}%)',
        fontsize=20, pad=10
    )

    png_path = os.path.splitext(nc_path)[0] + '_density.png'
    print(f'[4/4] Saving {png_path}...')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    file_size_mb = os.path.getsize(png_path) / (1024 ** 2)
    print(f'\n✓ Density visualization saved: {png_path}')
    print(f'  File size: {file_size_mb:.1f} MB')
    return png_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate hi-res adaptive grid points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--test-patches', type=int, default=None, metavar='N',
                        help='Process only N patches and estimate timing')
    parser.add_argument('--patch-offset', type=int, default=0, metavar='K',
                        help='Skip the first K patches (useful for sampling interior regions)')
    parser.add_argument('--patch-size', type=int, default=config.HIRES_PATCH_SIZE,
                        metavar='HRRR_CELLS',
                        help=f'Patch size in HRRR cells (default: {config.HIRES_PATCH_SIZE})')
    parser.add_argument('--output', default=None,
                        help='Output NetCDF filename or path '
                             f'(default: output/{config.HIRES_OUTPUT_FILE})')
    args = parser.parse_args()

    patch_hrrr = args.patch_size
    output_path = args.output or os.path.join(config.OUTPUT_DIR,
                                              config.HIRES_OUTPUT_FILE)

    preproc_dir = os.path.join(config.DATA_DIR, 'hires_preprocessed')

    print('=' * 70)
    print(' GENERATING HI-RES ADAPTIVE GRID POINTS')
    print('=' * 70)

    # ── 1. Load preprocessed data ────────────────────────────────────────────
    print('\n[1/3] Loading preprocessed data...')
    params, terrain_tier, wrf_x_row, wrf_y_col, gdfs = \
        load_preprocessed(preproc_dir)

    PATCH_PIX = patch_hrrr * 32
    n_pr = math.ceil(params['HIRES_NY'] / PATCH_PIX)
    n_pc = math.ceil(params['HIRES_NX'] / PATCH_PIX)
    total_patches = n_pr * n_pc

    print(f'\n  Hi-res grid : {params["HIRES_NY"]} × {params["HIRES_NX"]} '
          f'({params["HIRES_NY"]*params["HIRES_NX"]/1e9:.2f} B pixels)')
    print(f'  Resolution  : {params["HIRES_DX"]}m')
    print(f'  Patch size  : {patch_hrrr} HRRR cells = {PATCH_PIX} hi-res pixels')
    print(f'  Total patches: {n_pr} × {n_pc} = {total_patches}')

    if args.test_patches:
        print(f'\n  *** TEST MODE: processing {args.test_patches} of '
              f'{total_patches} patches ***')

    # ── 2. Open output ───────────────────────────────────────────────────────
    print(f'\n[2/3] Creating output: {output_path}')
    out_ds = open_output_nc(output_path)

    # ── 3. Process patches ───────────────────────────────────────────────────
    print(f'\n[3/3] Processing patches...\n')
    t_run = time.time()

    total_points, patch_times, _, crit_times = process_patches(
        params, terrain_tier,
        wrf_x_row, wrf_y_col, gdfs, out_ds,
        patch_hrrr_size=patch_hrrr,
        n_patches_limit=args.test_patches,
        patch_offset=args.patch_offset,
    )
    elapsed = time.time() - t_run

    out_ds.close()

    # ── Density visualization ────────────────────────────────────────────────
    if not args.test_patches:
        try:
            create_density_visualization(output_path)
        except Exception as exc:
            print(f'\n⚠  Density visualization failed: {exc}')

    # ── Summary ──────────────────────────────────────────────────────────────
    n_done = len(patch_times)
    print(f'\n{"=" * 70}')
    print(f' RESULTS')
    print(f'{"=" * 70}')
    print(f'  Points generated : {total_points:,}')
    print(f'  Patches done     : {n_done}')
    print(f'  Wall time        : {elapsed/60:.1f} min')

    if n_done > 0:
        mean_t = float(np.mean(patch_times))
        med_t  = float(np.median(patch_times))
        print(f'\n  Per-patch timing (mean {mean_t:.2f}s  median {med_t:.2f}s):')

        # Criterion breakdown (sum across all patches processed)
        total_crit = sum(crit_times.values())
        if total_crit > 0:
            print(f'  Criterion breakdown (total across {n_done} patches):')
            for name, t in sorted(crit_times.items(), key=lambda x: -x[1]):
                if t > 0.01:
                    print(f'    {name:<20s} {t:6.2f}s  ({100*t/total_crit:.0f}%)')

        if args.test_patches and total_patches > n_done:
            est_total  = total_patches * mean_t
            est_remain = (total_patches - n_done) * mean_t
            pts_per_patch = total_points / n_done
            est_total_pts = pts_per_patch * total_patches
            print(f'\n  *** FULL-DOMAIN ESTIMATE (based on {n_done} patches) ***')
            print(f'  Total patches       : {total_patches}')
            print(f'  Estimated run time  : {est_total/60:.0f} min '
                  f'({est_total/3600:.1f} hr)')
            print(f'  Estimated remaining : {est_remain/60:.0f} min')
            print(f'  Estimated output pts: ~{est_total_pts/1e6:.0f}M')

    print(f'\n  Output: {output_path}')
    print()
    print('Visualize:')
    print(f'  python3 visualize_hires_grid.py hires_points.nc <lat> <lon>')


if __name__ == '__main__':
    main()
