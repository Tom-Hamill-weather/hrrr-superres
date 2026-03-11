# python gin_cosine.py

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap, interp
import pygrib
from netCDF4 import Dataset
from scipy.interpolate import griddata

infile = 'HRRR_IC2025092912_APCP_lead6h.grib2'
fcstfile = pygrib.open(infile)
grb = fcstfile.select()[0]
hrrr_lats, hrrr_lons = grb.latlons()
fcstfile.close()

lats_regular = np.arange(20,60,0.5)
lons_regular = np.arange(-135,-60,0.5)
lons_2d, lats_2d = np.meshgrid(lons_regular, lats_regular)

lons_2d_01 = (lons_2d + 60.)/(-135 + 60.)
print ('min, max lons_2d_01 = ', np.min(lons_2d_01), np.max(lons_2d_01))
synthetic_temp = np.cos(4.*math.pi*lons_2d_01) + (60 - lats_2d)

print ('reading adaptive points...')
infile = 'output/adaptive_grid_points.nc'
nc = Dataset(infile, 'r')
lats_adaptive = nc.variables['latitude'][:]
lons_adaptive = nc.variables['longitude'][:]
nc.close()
lons_adaptive_01 = (lons_adaptive + 60.)/(-135 + 60.)
synthetic_temp_adaptive = \
    np.cos(4.*math.pi*lons_adaptive_01) + (60 - lats_adaptive)

# --- save synthetic_temp_adaptive to netCDF file
print ('saving adaptive grid data to netCDF file...')
outfile = 'output/synthetic_temp_adaptive.nc'
ncout = Dataset(outfile, 'w', format='NETCDF4')

# Create dimension
npoints = len(lats_adaptive)
ncout.createDimension('npoints', npoints)

# Create variables
lat_var = ncout.createVariable('latitude', 'f8', ('npoints',))
lon_var = ncout.createVariable('longitude', 'f8', ('npoints',))
temp_var = ncout.createVariable('synthetic_temp', 'f8', ('npoints',))

# Add attributes
lat_var.units = 'degrees_north'
lat_var.long_name = 'latitude'
lon_var.units = 'degrees_east'
lon_var.long_name = 'longitude'
temp_var.units = 'degrees_C'
temp_var.long_name = 'synthetic temperature'

# Write data
lat_var[:] = lats_adaptive
lon_var[:] = lons_adaptive
temp_var[:] = synthetic_temp_adaptive

# Add global attributes
ncout.description = 'Synthetic temperature data on adaptive grid points'
ncout.history = 'Created by gin_cosine.py'

ncout.close()
print ('saved to ', outfile)


plotit = False
if plotit == True:
    
    print ('interpolating back to regular grid')
    synthetic_temp_adaptive_gridded = griddata((lons_adaptive, lats_adaptive), \
        synthetic_temp_adaptive, (lons_2d, lats_2d), method='linear')
    
    
    # HRRR projection parameters (standard CONUS HRRR)
    lat_0 = 38.5
    lon_0 = -97.5
    lat_1 = 38.5
    lat_2 = 38.5

    m = Basemap(
        rsphere=(6378137.00, 6356752.3142),
        resolution='l',
        area_thresh=1000.,
        projection='lcc',
        lat_1=lat_1,
        lat_2=lat_2,
        lat_0=lat_0,
        lon_0=lon_0,
        llcrnrlon=hrrr_lons[0, 0],
        llcrnrlat=hrrr_lats[0, 0],
        urcrnrlon=hrrr_lons[-1, -1],
        urcrnrlat=hrrr_lats[-1, -1])

    x, y = m(lons_2d, lats_2d)
    colorst = ['White','#C4E8FF','#8FB3FF',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange','#FCD5D9','#F6A3AE',\
        '#FA5257','Orchid','#AD8ADB','#A449FF','LightGray']
    clevs = [0,10,15,20,25,30,35,40]

    fig = plt.figure(figsize=(8,5.5))
    axloc = [0.01,0.11,0.98,0.8]
    ax1 = fig.add_axes(axloc)
    ax1.set_title('Synthetic temperature',fontsize=13,color='Black')
    CS2 = m.contourf(x, y, synthetic_temp, clevs, \
        cmap=None, colors=colorst, extend='both')
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')

    cax = fig.add_axes([0.01,0.07,0.98,0.02])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs, format='%g')
    cb.ax.tick_params(labelsize=9)
    cb.set_label('Temperature (deg C)', fontsize=12)

    # ---- set plot title

    plot_title = 'synthetic_temp_regular_grid.png'
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)
    print ('Done!')

    # =========== 2nd plot with adaptive points

    colorst = ['White','#C4E8FF','#8FB3FF',\
        '#A6ECA6','#42F742','Yellow','Gold','Orange','#FCD5D9','#F6A3AE',\
        '#FA5257','Orchid','#AD8ADB','#A449FF','LightGray']
    clevs = [0,10,15,20,25,30,35,40]

    fig = plt.figure(figsize=(8,5.5))
    axloc = [0.01,0.11,0.98,0.8]
    ax1 = fig.add_axes(axloc)
    ax1.set_title('Synthetic temperature (adaptive)',fontsize=13,color='Black')
    CS2 = m.contourf(x, y, synthetic_temp_adaptive_gridded, clevs, \
        cmap=None, colors=colorst, extend='both',tri=True)
    
    m.drawcoastlines(linewidth=0.6,color='Gray')
    m.drawcountries(linewidth=0.4,color='Gray')
    m.drawstates(linewidth=0.2,color='Gray')

    cax = fig.add_axes([0.01,0.07,0.98,0.02])
    cb = plt.colorbar(CS2, orientation='horizontal', cax=cax,\
        drawedges=True, ticks=clevs, format='%g')
    cb.ax.tick_params(labelsize=9)
    cb.set_label('Temperature (deg C)', fontsize=12)

    # ---- set plot title

    plot_title = 'synthetic_temp_regular_adaptive.png'
    fig.savefig(plot_title, dpi=200, bbox_inches='tight')
    print ('saving plot to file = ',plot_title)
    print ('Done!')





