"""
Visualize Catalina Island coverage to verify complete coastline resolution
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.basemap import Basemap

# Load the generated grid
ds = nc.Dataset('output/adaptive_grid_RASTER.nc', 'r')
lats_all = ds.variables['latitude'][:]
lons_all = ds.variables['longitude'][:]
ds.close()

print(f"Total points loaded: {len(lats_all):,}")

# Define Catalina Island zoom region
cat_lat_min, cat_lat_max = 33.2, 33.6
cat_lon_min, cat_lon_max = -118.6, -118.2

# Filter points in Catalina region
mask = (lats_all >= cat_lat_min) & (lats_all <= cat_lat_max) & \
       (lons_all >= cat_lon_min) & (lons_all <= cat_lon_max)

lats_cat = lats_all[mask]
lons_cat = lons_all[mask]

print(f"Points in Catalina region: {len(lats_cat):,}")

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))

# Create basemap for Catalina region
m = Basemap(
    projection='merc',
    llcrnrlat=cat_lat_min,
    urcrnrlat=cat_lat_max,
    llcrnrlon=cat_lon_min,
    urcrnrlon=cat_lon_max,
    resolution='h',  # High resolution
    ax=ax
)

# Draw coastlines
m.drawcoastlines(linewidth=1.5, color='red', zorder=5)
m.fillcontinents(color='lightgray', lake_color='aqua', zorder=1)

# Plot grid points
x, y = m(lons_cat, lats_cat)
m.scatter(x, y, s=0.5, c='black', alpha=0.8, zorder=3)

ax.set_title('Catalina Island - Adaptive Grid Coverage\\n(Red = coastline, Black dots = grid points)',
             fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('output/catalina_coverage_RASTER.png', dpi=200, bbox_inches='tight')
print(f"\\n✓ Visualization saved: output/catalina_coverage_RASTER.png")

plt.close()

# Create a higher zoom to see point spacing
fig, ax = plt.subplots(figsize=(16, 12))

# Zoom to western tip of Catalina
zoom_lat_min, zoom_lat_max = 33.35, 33.50
zoom_lon_min, zoom_lon_max = -118.55, -118.40

m2 = Basemap(
    projection='merc',
    llcrnrlat=zoom_lat_min,
    urcrnrlat=zoom_lat_max,
    llcrnrlon=zoom_lon_min,
    urcrnrlon=zoom_lon_max,
    resolution='h',
    ax=ax
)

# Filter for zoom region
mask_zoom = (lats_all >= zoom_lat_min) & (lats_all <= zoom_lat_max) & \
            (lons_all >= zoom_lon_min) & (lons_all <= zoom_lon_max)

lats_zoom = lats_all[mask_zoom]
lons_zoom = lons_all[mask_zoom]

print(f"Points in zoom region: {len(lats_zoom):,}")

m2.drawcoastlines(linewidth=2, color='red', zorder=5)
m2.fillcontinents(color='lightgray', lake_color='aqua', zorder=1)

x2, y2 = m2(lons_zoom, lats_zoom)
m2.scatter(x2, y2, s=2, c='black', alpha=0.8, zorder=3)

ax.set_title('Catalina Island (West Tip) - High Resolution Points\\n(Each black dot = one grid point at 93.75m or finer)',
             fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('output/catalina_zoom_RASTER.png', dpi=200, bbox_inches='tight')
print(f"✓ Zoom visualization saved: output/catalina_zoom_RASTER.png")

print(f"\\n✓ Visualizations complete")
