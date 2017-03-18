# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:42:04 2017

@author: rickdberg

Data explorer

"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from scipy import stats

engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")

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

data = data[data['leg'] != '161']  # Mediterranean
data = data[data['leg'] != '160']  # Mediterranean
data = data[data['site'] != '768']  # Sulu sea
data = data[data['site'] != '769']  # Sulu Sea
data = data[data['water_depth'].notnull()]
data = data[data['bottom_water_temp'].notnull()]
deep_data = data[data['water_depth'] > 1500]
shallow_data = data[data['water_depth'] <= 1500]

# Plot deep data
plt.scatter(deep_data['bottom_water_temp'], deep_data['water_depth'], s=abs(deep_data['lat']), c=deep_data['bottom_water_temp'])

deep_mean = np.mean(deep_data['bottom_water_temp'])
deep_med = np.median(deep_data['bottom_water_temp'])
deep_stdev = np.std(deep_data['bottom_water_temp'])

# Plot shallow data and fit linear curve
plt.scatter(shallow_data['bottom_water_temp'], shallow_data['water_depth'], s=abs(shallow_data['lat']), c=shallow_data['bottom_water_temp'])
[slope, intercept, r, p, std] = stats.linregress(shallow_data['water_depth'], shallow_data['bottom_water_temp'])
y = np.linspace(0, 1500)
plt.plot(slope*y+intercept, y, 'k-')

# Error calculated as relative root mean squared error of curve fit to reported values
def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())
shallow_rmse = rmse((slope*shallow_data['water_depth']+intercept), shallow_data['bottom_water_temp'])
# shallow_rmse = pd.Series.std((slope*shallow_data['water_depth']+intercept) - shallow_data['bottom_water_temp'])

# Plot all data
plt.scatter(data['bottom_water_temp'], data['water_depth'], s=abs(data['lat']), c=data['bottom_water_temp'])
plt.plot(slope*y+intercept, y, 'k-')

# Plot histogram of results
plt.hist(deep_data['bottom_water_temp'], normed=1, bins=30, facecolor='orange')


# eof
