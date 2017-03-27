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

engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
gdal.UseExceptions()


# Using scipy
# Load porosity data - not filled
f = ncdf.Dataset(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
, "r")
y = f.variables['y'][:]  # lat
x = f.variables['x'][:]  # lon
z = f.variables['z'][:]
f.close()

# Get into Pandas
porosities  = pd.DataFrame(z.data, columns=x.data, index=y.data)
lat = pd.Series(y)
lon = pd.Series(x)

# Fill unknown porosity values with NaN
porosities = porosities.fillna(np.nan)

# Restructure lat lon data for iterpolation function - not working yet - very slow
coords = pd.DataFrame()
for n in np.arange(len(lat)):
    stack = pd.concat((pd.Series(np.ones(len(lon))*lat[n]), lon), axis=1)
    coords = pd.concat((coords, stack), axis=0)
values = pd.Series()
for n in np.arange(len(lat)):
    values = pd.concat((values, z.iloc[n,:]), axis=0)






# This is just an example of how this function works
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
