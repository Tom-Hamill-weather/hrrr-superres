"""
Raster-based adaptive grid generation
Uses binary masks at finest resolution (93.75m) with decimation patterns for coarser tiers
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

class RasterGridGenerator:
    def __init__(self):
        self.loader = DataLoader()
        self.terrain_analyzer = TerrainAnalyzer(self.loader)

    def generate(self):
        print("=" * 70)
        print(" RASTER-BASED ADAPTIVE GRID GENERATION")
        print("=" * 70)
        print("\nConfiguration:")
        print(f"  Finest resolution: {config.TIER_RESOLUTIONS[0]}m")
        print(f"  Target points: {config.TARGET_TOTAL_POINTS:,}")

        start_time = time.time()

        # Load data
        print("\n" + "=" * 70)
        print(" STEP 1: LOADING DATA")
        print("=" * 70)
        self.loader.load_all()
        self.terrain_analyzer.compute_terrain_variability()

        # Get HRRR grid
        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']

        # Create projection
        lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
        lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2
        self.proj = Basemap(
            projection='lcc',
            lat_0=lat_ref,
            lon_0=lon_ref,
            lat_1=lat_ref - 5,
            lat_2=lat_ref + 5,
            llcrnrlat=lats_hrrr.min(),
            urcrnrlat=lats_hrrr.max(),
            llcrnrlon=lons_hrrr.min(),
            urcrnrlon=lons_hrrr.max(),
            resolution=None
        )

        # Define fine grid at 93.75m resolution
        print("\n" + "=" * 70)
        print(" STEP 2: CREATE FINE RASTER GRID")
        print("=" * 70)

        finest_res = config.TIER_RESOLUTIONS[0]  # 93.75m

        # Get domain bounds in projected coordinates
        corners_lon = [lons_hrrr.min(), lons_hrrr.max(), lons_hrrr.max(), lons_hrrr.min()]
        corners_lat = [lats_hrrr.min(), lats_hrrr.min(), lats_hrrr.max(), lats_hrrr.max()]
        corners_x, corners_y = self.proj(corners_lon, corners_lat)

        x_min, x_max = min(corners_x), max(corners_x)
        y_min, y_max = min(corners_y), max(corners_y)

        # Create fine grid
        n_x = int((x_max - x_min) / finest_res) + 1
        n_y = int((y_max - y_min) / finest_res) + 1

        print(f"\nFine grid dimensions: {n_y} × {n_x}")
        print(f"Total fine grid points: {n_y * n_x:,}")
        print(f"Memory required: ~{n_y * n_x * 6 / 1e9:.1f} GB for 6 tier masks")

        # Create coordinates
        x_coords = np.linspace(x_min, x_max, n_x)
        y_coords = np.linspace(y_min, y_max, n_y)

        # Initialize tier masks (one per tier)
        print("\nInitializing tier masks...")
        tier_masks = {}
        for tier in range(6):
            tier_masks[tier] = np.zeros((n_y, n_x), dtype=np.uint8)

        print("\n" + "=" * 70)
        print(" STEP 3: POPULATE TIER MASKS")
        print("=" * 70)

        # Process coastlines
        print("\n[Tier 0] Coastlines...")
        self._rasterize_coastlines(tier_masks[0], x_coords, y_coords, finest_res)

        # Process lakes
        print("\n[Tier 0] Lakes...")
        self._rasterize_lakes(tier_masks[0], x_coords, y_coords, finest_res)

        # Process terrain for multiple tiers
        print("\n[Tiers 1-3] Terrain...")
        self._rasterize_terrain(tier_masks, x_coords, y_coords, lats_hrrr, lons_hrrr)

        # Process urban for Tier 2
        print("\n[Tier 2] Urban areas...")
        self._rasterize_urban(tier_masks[2], x_coords, y_coords, finest_res)

        # Apply decimation patterns
        print("\n" + "=" * 70)
        print(" STEP 4: APPLY DECIMATION PATTERNS")
        print("=" * 70)

        for tier in range(1, 6):
            decimation = 2 ** tier  # Tier 1: /2, Tier 2: /4, Tier 3: /8, etc.
            print(f"\nTier {tier}: Decimating by {decimation}x (every {decimation}th point)")

            # Create staggered sampling mask
            sample_mask = np.zeros((n_y, n_x), dtype=np.uint8)
            sample_mask[::decimation, ::decimation] = 1

            # Apply: only keep points where both tier criteria met AND in sample pattern
            tier_masks[tier] = tier_masks[tier] * sample_mask

            print(f"  Points in Tier {tier}: {tier_masks[tier].sum():,}")

        print(f"\nTier 0 points: {tier_masks[0].sum():,}")

        # Combine: max across all tiers
        print("\n" + "=" * 70)
        print(" STEP 5: COMBINE TIER MASKS")
        print("=" * 70)

        final_mask = np.maximum.reduce([tier_masks[t] for t in range(6)])
        total_points = final_mask.sum()

        print(f"\nTotal points in final mask: {total_points:,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        print(f"Difference: {100*(total_points - config.TARGET_TOTAL_POINTS)/config.TARGET_TOTAL_POINTS:+.1f}%")

        # Extract points
        print("\n" + "=" * 70)
        print(" STEP 6: EXTRACT POINT COORDINATES")
        print("=" * 70)

        y_indices, x_indices = np.where(final_mask == 1)
        x_pts = x_coords[x_indices]
        y_pts = y_coords[y_indices]

        # Convert back to lat/lon
        lons_pts, lats_pts = self.proj(x_pts, y_pts, inverse=True)

        print(f"\nExtracted {len(lats_pts):,} points")

        # Write output
        print("\n" + "=" * 70)
        print(" STEP 7: WRITE OUTPUT")
        print("=" * 70)

        output_path = 'output/adaptive_grid_RASTER.nc'
        ds = nc.Dataset(output_path, 'w')
        ds.createDimension('points', len(lats_pts))

        lat_var = ds.createVariable('latitude', 'f4', ('points',))
        lon_var = ds.createVariable('longitude', 'f4', ('points',))

        lat_var[:] = lats_pts
        lon_var[:] = lons_pts

        ds.close()

        elapsed = time.time() - start_time
        print(f"\n✓ Complete in {elapsed/60:.1f} minutes")
        print(f"  Output: {output_path}")
        print(f"  Total points: {len(lats_pts):,}")

        return lats_pts, lons_pts

    def _rasterize_coastlines(self, mask, x_coords, y_coords, resolution):
        """Rasterize coastlines onto binary mask"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None:
            return

        # Reproject and buffer
        coast_proj = coastline_gdf.to_crs(self.proj.proj4string)
        coast_proj['geometry'] = coast_proj.geometry.simplify(100)
        coast_buffered = coast_proj.buffer(config.COASTLINE_BUFFER_KM * 1000)

        # Rasterize
        from rasterio import features
        from affine import Affine

        # Create affine transform
        transform = Affine.translation(x_coords[0], y_coords[0]) * Affine.scale(resolution, resolution)

        # Rasterize geometries
        shapes = [(geom, 1) for geom in coast_buffered]
        rasterized = features.rasterize(
            shapes,
            out_shape=mask.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        mask[:] = np.maximum(mask, rasterized)
        print(f"  Rasterized {len(coastline_gdf)} coastline features")
        print(f"  Points in mask: {mask.sum():,}")

    def _rasterize_lakes(self, mask, x_coords, y_coords, resolution):
        """Rasterize lake shorelines onto binary mask"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None:
            return

        # Reproject, extract boundaries, buffer
        lakes_proj = lakes_gdf.to_crs(self.proj.proj4string)
        lakes_boundaries = lakes_proj.boundary
        lakes_buffered = lakes_boundaries.buffer(config.LAKE_BUFFER_KM * 1000)

        from rasterio import features
        from affine import Affine

        transform = Affine.translation(x_coords[0], y_coords[0]) * Affine.scale(resolution, resolution)
        shapes = [(geom, 1) for geom in lakes_buffered]

        rasterized = features.rasterize(
            shapes,
            out_shape=mask.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        mask[:] = np.maximum(mask, rasterized)
        print(f"  Rasterized {len(lakes_gdf)} lake features")
        print(f"  Points in mask: {mask.sum():,}")

    def _rasterize_terrain(self, tier_masks, x_coords, y_coords, lats_hrrr, lons_hrrr):
        """Rasterize terrain classifications"""
        terrain_var = self.terrain_analyzer.terrain_var

        # Get HRRR cell coordinates in projection
        lons_flat = lons_hrrr.ravel()
        lats_flat = lats_hrrr.ravel()
        x_hrrr, y_hrrr = self.proj(lons_flat, lats_flat)

        # For each HRRR cell, find corresponding region in fine grid
        # This is approximate but fast
        print("  Mapping terrain to fine grid...")
        # TODO: Implement efficient terrain mapping
        print("  (Terrain mapping not yet implemented)")

    def _rasterize_urban(self, mask, x_coords, y_coords, resolution):
        """Rasterize urban areas"""
        urban_gdf = self.loader.data.get('high_density_urban')
        if urban_gdf is None:
            return

        urban_proj = urban_gdf.to_crs(self.proj.proj4string)

        from rasterio import features
        from affine import Affine

        transform = Affine.translation(x_coords[0], y_coords[0]) * Affine.scale(resolution, resolution)
        shapes = [(geom, 1) for geom in urban_proj.geometry]

        rasterized = features.rasterize(
            shapes,
            out_shape=mask.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        mask[:] = np.maximum(mask, rasterized)
        print(f"  Rasterized {len(urban_gdf)} urban features")
        print(f"  Points in mask: {mask.sum():,}")

if __name__ == '__main__':
    generator = RasterGridGenerator()
    lats, lons = generator.generate()
