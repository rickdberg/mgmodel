# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:51:06 2017

@author: rickdberg
"""

import numpy as np
import pandas as pd
import netCDF4 as ncdf
from sqlalchemy import create_engine
from osgeo import gdal
import rasterio

engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
gdal.UseExceptions()

# Load metadata
sql = "SELECT * FROM metadata_mg_flux;"
metadata = pd.read_sql(sql, engine)

# Load site data
sql = "SELECT * FROM site_info;"
sitedata = pd.read_sql(sql, engine)

# Load hole data
sql = "SELECT * FROM summary_all;"
holedata = pd.read_sql(sql, engine)
# Group and average hole data for sites
hole_grouped = holedata.loc[:,('site_key', 'lat','lon','water_depth','total_penetration')].groupby("site_key").mean().reset_index()

# Combine all tables
site_meta_data = pd.merge(metadata, sitedata, how='outer', on=('site_key', 'leg', 'site'))
data = pd.merge(site_meta_data, hole_grouped, how='outer', on=('site_key')).fillna(np.nan)
data = data.dropna(subset = ['interface_flux']).reset_index(drop=True)

# Get site lat and lon values
site_lat = data['lat']
site_lon = data['lon']

# Load porosity data - not filled
f = ncdf.Dataset(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
, "r")
lat = f.variables['y'][:]
lon = f.variables['x'][:]
z = f.variables['z'][:]
f.close()




# Using scipy
coords = np.array([]).reshape(0,2)
for n in np.arange(len(lat)):
    stack = np.column_stack((np.ones(len(lon))*lat[n], lon))
    coords = np.vstack((coords, stack))
values = np.array([]).reshape(0,1)
for n in np.arange(len(lat)):
    values = np.concatenate((values, z[n,:]), axis=1)

zz = z[0,:]


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
points = np.random.rand(1000, 2)
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
values = func(points[:,0], points[:,1])
porosities_filled = scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)






# Using GDAL
porosities = gdal.Open(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
)
porosity_data = porosities.GetRasterBand(1)
porosities_filled = gdal.FillNodata(porosity_data, maskBand = None, maxSearchDist=250, smoothingIterations=0)
