"""
Two-panel map of HRRR terrain elevation and terrain roughness.

Elevation is taken from the HRRR terrain array (hrrr_terrain.npy).
Roughness is the local std dev of elevation computed with
scipy.ndimage.generic_filter over a 3×3 pixel window on the 3 km HRRR
grid (~9 km effective scale) — the same method used by
generate_adaptive_grid.py for tier classification.

Percentiles of roughness over the full CONUS HRRR domain are printed to
stdout before the plot is made.

Usage:
    python visualize_terrain.py <lat> <lon>

Examples:
    python visualize_terrain.py 39.7 -105.0    # Colorado Rockies
    python visualize_terrain.py 47.6 -122.3    # Seattle / Cascades
    python visualize_terrain.py 35.0  -83.5    # Southern Appalachians

Output: output/terrain_<lat>_<lon>.png

The 1.6-degree plot domain matches the default box used by
visualize_point_grid_binary.py.  Colour bounds are set to the 2nd-98th
percentile of the visible data so the scale is never anchored at 0 when
the region is well above sea level.

Tier threshold lines are annotated on the roughness colorbar.
"""
import argparse
import os
import sys
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import config

HALF_DEG  = 0.8   # ±0.8° → 1.6° total domain (matches visualize_point_grid_binary.py)
HRRR_DIR  = os.path.join(config.DATA_DIR, 'hrrr')


# ── Terrain colormap from visualize_point_grid_binary.py ────────────────────
def _topo_cmap():
    return LinearSegmentedColormap.from_list(
        'topo_terrain',
        [(0.00, 'white'),       # flat / low
         (0.18, '#f5e8a8'),     # pale straw
         (0.38, '#c8d870'),     # yellow-green
         (0.58, '#64a840'),     # medium green
         (0.78, '#2d6828'),     # forest green
         (1.00, '#7a4218')]     # warm brown (peaks / high roughness)
    )


# ── Roughness computation (matches generate_adaptive_grid.py) ────────────────
def _compute_roughness(terrain):
    """Std dev in a sliding window on the HRRR grid — same as TerrainAnalyzer."""
    window_pixels = max(1, int(config.TERRAIN_WINDOW_SIZE / 3000))
    return generic_filter(terrain, np.std, size=window_pixels,
                          mode='constant', cval=0)


# ── Domain-wide percentiles (printed, not plotted) ────────────────────────────
def print_roughness_percentiles():
    """Compute roughness over the full HRRR domain and print key percentiles."""
    print('Loading full HRRR terrain for roughness percentiles...')
    terrain = np.load(os.path.join(HRRR_DIR, 'hrrr_terrain.npy'))
    roughness = _compute_roughness(terrain)
    valid = roughness[np.isfinite(roughness)].ravel()
    window_pixels = max(1, int(config.TERRAIN_WINDOW_SIZE / 3000))
    print(f'  Roughness (σ, {window_pixels}×{window_pixels} px window ≈ '
          f'{window_pixels * 3} km)  N={len(valid):,}  '
          f'min={valid.min():.1f}  max={valid.max():.1f}  mean={valid.mean():.1f} m')
    print('  Percentiles of terrain roughness over full CONUS HRRR domain:')
    for p in [90, 92, 95, 97, 99]:
        print(f'    {p:2d}th: {np.percentile(valid, p):.1f} m')
    print()


# ── Subset loader ─────────────────────────────────────────────────────────────
def load_hrrr_subset(lat_c, lon_c):
    """Return (lats, lons, elev, roughness) 2-D arrays for the plot domain.

    Loads the full HRRR arrays (fast .npy read, ~1.9 M cells), computes
    roughness on the full domain to avoid edge effects, then subsets to the
    1.6° plot box plus a small buffer.
    """
    print(f'Loading HRRR terrain for ({lat_c:.3f}°N, {lon_c:.3f}°)...')
    lats    = np.load(os.path.join(HRRR_DIR, 'hrrr_lats.npy'))
    lons    = np.load(os.path.join(HRRR_DIR, 'hrrr_lons.npy'))
    terrain = np.load(os.path.join(HRRR_DIR, 'hrrr_terrain.npy'))

    print('  Computing terrain roughness...')
    roughness = _compute_roughness(terrain)

    buf = 0.05
    lat_min = lat_c - HALF_DEG - buf
    lat_max = lat_c + HALF_DEG + buf
    lon_min = lon_c - HALF_DEG - buf
    lon_max = lon_c + HALF_DEG + buf

    mask = (
        (lats >= lat_min) & (lats <= lat_max) &
        (lons >= lon_min) & (lons <= lon_max)
    )
    if not mask.any():
        sys.exit(f'ERROR: no HRRR data found near ({lat_c}, {lon_c})')

    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    return (lats[r0:r1, c0:c1], lons[r0:r1, c0:c1],
            terrain[r0:r1, c0:c1], roughness[r0:r1, c0:c1])


# ── Colour-scale helpers ──────────────────────────────────────────────────────
def pct_bounds(arr, lo=2, hi=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


# ── Single-panel renderer ─────────────────────────────────────────────────────
def draw_panel(ax, fig, m, lons2d, lats2d, data2d,
               cmap, vmin, vmax, title, cbar_label,
               lat_c, lon_c, threshold_lines=None):
    """Render pcolormesh + geographic lines + centre marker on one axes."""
    x, y = m(lons2d, lats2d)

    outside = (
        (x < m.llcrnrx) | (x > m.urcrnrx) |
        (y < m.llcrnry) | (y > m.urcrnry)
    )
    plot_data = data2d.copy()
    plot_data[outside] = np.nan

    im = ax.pcolormesh(x, y, plot_data,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='nearest', zorder=1)
    ax.set_xlim(m.llcrnrx, m.urcrnrx)
    ax.set_ylim(m.llcrnry, m.urcrnry)

    m.drawcoastlines(linewidth=1.2, color='black',   zorder=5)
    m.drawstates    (linewidth=0.8, color='black',   zorder=5)
    m.drawcounties  (linewidth=0.4, color='dimgray', zorder=5)

    xc, yc = m(lon_c, lat_c)
    ax.plot(xc, yc, 'k+', markersize=14, markeredgewidth=2.0, zorder=10)

    ax.set_title(title, fontsize=13, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Mark tier thresholds on the roughness colorbar
    if threshold_lines:
        extra_ticks  = [v   for v, _ in threshold_lines if vmin <= v <= vmax]
        extra_labels = [lbl for v, lbl in threshold_lines if vmin <= v <= vmax]
        if extra_ticks:
            existing = list(cbar.get_ticks())
            all_ticks = sorted(set(existing + extra_ticks))
            tick_labels = []
            for t in all_ticks:
                match = [lbl for v, lbl in threshold_lines if abs(v - t) < 0.5]
                tick_labels.append(match[0] if match else f'{t:.0f}')
            cbar.set_ticks(all_ticks)
            cbar.set_ticklabels(tick_labels, fontsize=8)
            for v in extra_ticks:
                norm_pos = (v - vmin) / (vmax - vmin)
                cbar.ax.axhline(norm_pos, color='crimson',
                                linewidth=1.5, linestyle='--')

    return im


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Two-panel WRF terrain elevation + roughness map',
        epilog='Example: python visualize_terrain.py 39.7 -105.0'
    )
    parser.add_argument('lat', type=float, help='Centre latitude  (degrees N)')
    parser.add_argument('lon', type=float, help='Centre longitude (degrees E; negative = west)')
    args = parser.parse_args()
    lat_c, lon_c = args.lat, args.lon

    # ── Percentiles over full domain (printed, not plotted) ──────────────────
    print_roughness_percentiles()

    # ── Load subset for the 1.6° plot domain ─────────────────────────────────
    lats, lons, elev, roughness = load_hrrr_subset(lat_c, lon_c)
    print(f'  Subset: {elev.shape[0]} × {elev.shape[1]} cells  '
          f'elev {np.nanmin(elev):.0f}–{np.nanmax(elev):.0f} m  '
          f'σ {np.nanmin(roughness):.1f}–{np.nanmax(roughness):.1f} m')

    # ── Colour bounds using only cells inside the plot box ───────────────────
    lat_min = lat_c - HALF_DEG;  lat_max = lat_c + HALF_DEG
    lon_min = lon_c - HALF_DEG;  lon_max = lon_c + HALF_DEG
    in_box = (
        (lats >= lat_min) & (lats <= lat_max) &
        (lons >= lon_min) & (lons <= lon_max)
    )
    elev_vmin,  elev_vmax  = pct_bounds(elev[in_box])
    rough_vmin, rough_vmax = pct_bounds(roughness[in_box])
    print(f'  Elevation colour range  (2–98th pct): {elev_vmin:.0f}–{elev_vmax:.0f} m')
    print(f'  Roughness colour range  (2–98th pct): {rough_vmin:.0f}–{rough_vmax:.0f} m')

    tier_thresholds = [
        (config.TERRAIN_TIER1_THRESHOLD, f'T0≥{config.TERRAIN_TIER1_THRESHOLD}m'),
        (config.TERRAIN_TIER2_THRESHOLD, f'T1≥{config.TERRAIN_TIER2_THRESHOLD}m'),
        (config.TERRAIN_TIER3_THRESHOLD, f'T2≥{config.TERRAIN_TIER3_THRESHOLD}m'),
    ]

    # ── Figure ───────────────────────────────────────────────────────────────
    cmap = _topo_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    maps = []
    for ax in axes:
        m = Basemap(
            projection='merc',
            llcrnrlat=lat_min, urcrnrlat=lat_max,
            llcrnrlon=lon_min, urcrnrlon=lon_max,
            resolution='i', ax=ax,
        )
        maps.append(m)

    lat_str = f"{abs(lat_c):.2f}{'N' if lat_c >= 0 else 'S'}"
    lon_str = f"{abs(lon_c):.2f}{'E' if lon_c >= 0 else 'W'}"

    window_km = max(1, int(config.TERRAIN_WINDOW_SIZE / 3000)) * 3

    draw_panel(
        axes[0], fig, maps[0], lons, lats, elev,
        cmap=cmap, vmin=elev_vmin, vmax=elev_vmax,
        title=(f'Terrain Elevation  (HRRR)\n'
               f'{lat_str} {lon_str}  |  scale {elev_vmin:.0f}–{elev_vmax:.0f} m'),
        cbar_label='Elevation (m)',
        lat_c=lat_c, lon_c=lon_c,
    )

    draw_panel(
        axes[1], fig, maps[1], lons, lats, roughness,
        cmap=cmap, vmin=rough_vmin, vmax=rough_vmax,
        title=(f'Terrain Roughness  (HRRR σ, {window_km} km window)\n'
               f'{lat_str} {lon_str}  |  scale {rough_vmin:.0f}–{rough_vmax:.0f} m'),
        cbar_label='Elevation std dev (m)',
        lat_c=lat_c, lon_c=lon_c,
        threshold_lines=tier_thresholds,
    )

    out_path = os.path.join(config.OUTPUT_DIR,
                            f'terrain_{lat_str}_{lon_str}.png')
    plt.savefig(out_path, dpi=150)
    print(f'\n✓ Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
