"""
STAGE 2: Generate Final Grid from Masks
========================================

Load precomputed boolean masks and:
1. Compute tier assignments using union logic
2. Apply stride patterns for each tier
3. Output final adaptive grid

This stage is FAST - can iterate on tier assignments quickly.
"""

import numpy as np
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import time
import os
import config
from generate_adaptive_grid import DataLoader

class GridGenerator:
    def __init__(self, mask_dir='output/masks'):
        self.mask_dir = mask_dir
        self.loader = DataLoader()

    def generate_grid(self):
        """Generate final grid from precomputed masks"""
        print("="*70)
        print(" STAGE 2: GENERATING ADAPTIVE GRID FROM MASKS")
        print("="*70)

        start_time = time.time()

        # Load HRRR grid for coordinate generation
        print("\nLoading HRRR grid...")
        self.loader.load_hrrr_grid()

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']
        shape = lats_hrrr.shape

        # Create projection
        lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
        lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2
        self.proj = Basemap(
            projection='lcc',
            lat_0=lat_ref, lon_0=lon_ref,
            lat_1=lat_ref - 5, lat_2=lat_ref + 5,
            llcrnrlat=lats_hrrr.min(), urcrnrlat=lats_hrrr.max(),
            llcrnrlon=lons_hrrr.min(), urcrnrlon=lons_hrrr.max(),
            resolution=None
        )

        # Convert to projected coordinates
        x_hrrr, y_hrrr = self.proj(lons_hrrr, lats_hrrr)

        # Precompute cell vectors
        print("Precomputing cell geometry...")
        dx_east, dy_east, dx_north, dy_north = self._compute_cell_vectors(x_hrrr, y_hrrr)

        # Generate fine grid coordinates
        print("Generating fine grid coordinates...")
        n_fine = 32
        lats_fine, lons_fine = self._generate_fine_grid_coords(
            lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
            dx_east, dy_east, dx_north, dy_north, shape, n_fine
        )

        print(f"Fine grid shape: {lats_fine.shape}")

        # Load all masks
        print("\nLoading precomputed masks...")
        masks = self._load_all_masks()

        # Compute tier assignments
        print("\nComputing tier assignments...")
        binary_grids = self._compute_tier_assignments(masks)

        # Apply strides and collect points
        print("\nApplying stride patterns and collecting points...")
        lats_out, lons_out = self._apply_strides(lats_fine, lons_fine, binary_grids)

        print(f"\nTotal points: {len(lats_out):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_out) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # Write output
        print("\nWriting output...")
        output_path = 'output/adaptive_grid_STAGED.nc'
        self._write_output(lats_out, lons_out, output_path)

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ Complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Output: {output_path}")
        print(f"{'='*70}")

        return lats_out, lons_out

    def _compute_cell_vectors(self, x_hrrr, y_hrrr):
        """Compute orientation vectors for HRRR cells"""
        dx_east = np.zeros_like(x_hrrr)
        dy_east = np.zeros_like(x_hrrr)
        dx_north = np.zeros_like(x_hrrr)
        dy_north = np.zeros_like(x_hrrr)

        dx_east[:, :-1] = x_hrrr[:, 1:] - x_hrrr[:, :-1]
        dy_east[:, :-1] = y_hrrr[:, 1:] - y_hrrr[:, :-1]
        dx_east[:, -1] = dx_east[:, -2]
        dy_east[:, -1] = dy_east[:, -2]

        dx_north[:-1, :] = x_hrrr[1:, :] - x_hrrr[:-1, :]
        dy_north[:-1, :] = y_hrrr[1:, :] - y_hrrr[:-1, :]
        dx_north[-1, :] = dx_north[-2, :]
        dy_north[-1, :] = dy_north[-2, :]

        return dx_east, dy_east, dx_north, dy_north

    def _generate_fine_grid_coords(self, lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                                   dx_east, dy_east, dx_north, dy_north, shape, n_fine):
        """Generate fine grid lat/lon coordinates"""
        # Repeat to fine grid
        x_repeated = np.repeat(np.repeat(x_hrrr, n_fine, axis=0), n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_hrrr, n_fine, axis=0), n_fine, axis=1)

        dx_east_repeated = np.repeat(np.repeat(dx_east, n_fine, axis=0), n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east, n_fine, axis=0), n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north, n_fine, axis=0), n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north, n_fine, axis=0), n_fine, axis=1)

        n_fine_i = shape[0] * n_fine
        n_fine_j = shape[1] * n_fine

        # Fine grid indices
        i_fine_idx = np.arange(n_fine_i) % n_fine
        j_fine_idx = np.arange(n_fine_j) % n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        # Fractional offsets
        frac_east = jj_fine_idx / n_fine - 0.5
        frac_north = ii_fine_idx / n_fine - 0.5

        # Apply offsets
        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        # Convert back to lat/lon
        lons_fine, lats_fine = self.proj(x_fine, y_fine, inverse=True)

        return lats_fine, lons_fine

    def _load_all_masks(self):
        """Load all precomputed masks"""
        masks = {}

        # Terrain masks
        for thresh in [800, 600, 300, 150]:
            filename = f'{self.mask_dir}/terrain_gt{thresh}m.npz'
            if os.path.exists(filename):
                data = np.load(filename)
                masks[f'terrain_gt{thresh}'] = data['mask']
                print(f"  Loaded: terrain >{thresh}m ({masks[f'terrain_gt{thresh}'].sum():,} points)")

        # Coastline masks
        for dist in [750, 1500, 3000, 6000, 12000]:
            filename = f'{self.mask_dir}/coastline_{dist}m.npz'
            if os.path.exists(filename):
                data = np.load(filename)
                masks[f'coastline_{dist}m'] = data['mask']
                print(f"  Loaded: coastline {dist}m ({masks[f'coastline_{dist}m'].sum():,} points)")

        # Lake masks
        for dist in [750, 1500, 3000, 6000, 12000]:
            filename = f'{self.mask_dir}/lakes_{dist}m.npz'
            if os.path.exists(filename):
                data = np.load(filename)
                masks[f'lakes_{dist}m'] = data['mask']
                print(f"  Loaded: lakes {dist}m ({masks[f'lakes_{dist}m'].sum():,} points)")

        # Ski resorts
        filename = f'{self.mask_dir}/ski_resorts_2km.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            masks['ski_resorts'] = data['mask']
            print(f"  Loaded: ski resorts ({masks['ski_resorts'].sum():,} points)")

        # Urban
        filename = f'{self.mask_dir}/urban_high_density.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            masks['urban_high'] = data['mask']
            print(f"  Loaded: high-density urban ({masks['urban_high'].sum():,} points)")

        filename = f'{self.mask_dir}/urban_suburban.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            masks['urban_suburban'] = data['mask']
            print(f"  Loaded: suburban ({masks['urban_suburban'].sum():,} points)")

        # Roads
        filename = f'{self.mask_dir}/roads_500m.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            masks['roads'] = data['mask']
            print(f"  Loaded: roads ({masks['roads'].sum():,} points)")

        # Parks
        filename = f'{self.mask_dir}/national_parks.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            masks['parks'] = data['mask']
            print(f"  Loaded: national parks ({masks['parks'].sum():,} points)")

        # Forests
        filename = f'{self.mask_dir}/national_forests.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            masks['forests'] = data['mask']
            print(f"  Loaded: national forests ({masks['forests'].sum():,} points)")

        return masks

    def _compute_tier_assignments(self, masks):
        """
        Compute tier assignments using union logic with priority hierarchy

        Tier 0 (93.75m): Ski resorts (HIGHEST RESOLUTION)
        Tier 1 (187.5m): Coastlines/lakes (750m) OR extreme terrain (>800m)
        Tier 2 (375m): Coastlines/lakes (1500m) OR high-density urban OR very rugged (>600m)
        Tier 3 (750m): Coastlines/lakes (3km) OR suburban OR roads OR parks OR forests OR rugged (>300m)
        Tier 4 (1.5km): Coastlines/lakes (6km) OR moderate terrain (>150m)
        Tier 5 (3km): Everything else (background)
        """
        shape = list(masks.values())[0].shape
        binary_grids = [np.zeros(shape, dtype=bool) for _ in range(6)]

        # Tier 0: Ski resorts (HIGHEST PRIORITY)
        if 'ski_resorts' in masks:
            tier0_full = masks['ski_resorts'].copy()
        else:
            tier0_full = np.zeros(shape, dtype=bool)

        # Tier 1: 187.5m resolution
        tier1_full = np.zeros(shape, dtype=bool)
        if 'coastline_750m' in masks:
            tier1_full |= masks['coastline_750m']
        if 'lakes_750m' in masks:
            tier1_full |= masks['lakes_750m']
        if 'terrain_gt800' in masks:
            tier1_full |= masks['terrain_gt800']

        # Tier 2: 375m resolution
        tier2_full = np.zeros(shape, dtype=bool)
        if 'coastline_1500m' in masks:
            tier2_full |= masks['coastline_1500m']
        if 'lakes_1500m' in masks:
            tier2_full |= masks['lakes_1500m']
        if 'urban_high' in masks:
            tier2_full |= masks['urban_high']
        if 'terrain_gt600' in masks:
            tier2_full |= masks['terrain_gt600']

        # Tier 3: 750m resolution
        tier3_full = np.zeros(shape, dtype=bool)
        if 'coastline_3000m' in masks:
            tier3_full |= masks['coastline_3000m']
        if 'lakes_3000m' in masks:
            tier3_full |= masks['lakes_3000m']
        if 'urban_suburban' in masks:
            tier3_full |= masks['urban_suburban']
        if 'roads' in masks:
            tier3_full |= masks['roads']
        if 'parks' in masks:
            tier3_full |= masks['parks']
        if 'forests' in masks:
            tier3_full |= masks['forests']
        if 'terrain_gt300' in masks:
            tier3_full |= masks['terrain_gt300']

        # Tier 4: 1.5km resolution
        tier4_full = np.zeros(shape, dtype=bool)
        if 'coastline_6000m' in masks:
            tier4_full |= masks['coastline_6000m']
        if 'lakes_6000m' in masks:
            tier4_full |= masks['lakes_6000m']
        if 'terrain_gt150' in masks:
            tier4_full |= masks['terrain_gt150']

        # Tier 5: Background (everything)
        tier5_full = np.ones(shape, dtype=bool)

        # Apply priority hierarchy (finest resolution wins)
        binary_grids[0] = tier0_full
        binary_grids[1] = tier1_full & ~tier0_full
        binary_grids[2] = tier2_full & ~tier1_full & ~tier0_full
        binary_grids[3] = tier3_full & ~tier2_full & ~tier1_full & ~tier0_full
        binary_grids[4] = tier4_full & ~tier3_full & ~tier2_full & ~tier1_full & ~tier0_full
        binary_grids[5] = tier5_full & ~tier4_full & ~tier3_full & ~tier2_full & ~tier1_full & ~tier0_full

        # Print tier statistics
        print("\nTier assignments:")
        resolutions = ['93.75m', '187.5m', '375m', '750m', '1.5km', '3km']
        for tier in range(6):
            count = binary_grids[tier].sum()
            pct = 100 * count / binary_grids[tier].size
            print(f"  Tier {tier} ({resolutions[tier]}): {count:,} points ({pct:.2f}%)")

        return binary_grids

    def _apply_strides(self, lats_fine, lons_fine, binary_grids):
        """Apply stride decimation and collect unique points"""
        points_set = set()

        for tier in range(6):
            stride = 2 ** tier
            binary = binary_grids[tier]

            # Apply stride
            for i in range(0, binary.shape[0], stride):
                for j in range(0, binary.shape[1], stride):
                    if binary[i, j]:
                        lat = round(float(lats_fine[i, j]), 6)
                        lon = round(float(lons_fine[i, j]), 6)
                        points_set.add((lat, lon))

        # Convert to arrays
        if len(points_set) > 0:
            points = list(points_set)
            lats = np.array([p[0] for p in points])
            lons = np.array([p[1] for p in points])
            return lats, lons
        else:
            return np.array([]), np.array([])

    def _write_output(self, lats, lons, output_path):
        """Write output NetCDF file"""
        ds = nc.Dataset(output_path, 'w')
        ds.createDimension('points', len(lats))

        lat_var = ds.createVariable('latitude', 'f4', ('points',))
        lon_var = ds.createVariable('longitude', 'f4', ('points',))

        lat_var[:] = lats
        lon_var[:] = lons

        ds.close()


if __name__ == '__main__':
    generator = GridGenerator()
    lats, lons = generator.generate_grid()
