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
holedata = holedata[holedata['leg'] != '161']
holedata = holedata[holedata['leg'] != '160']

# Group and average hole data for sites
hole_grouped = holedata.loc[:,('site_key', 'lat','lon','water_depth','total_penetration')].groupby("site_key").mean().reset_index()

# Combine all tables
site_meta_data = pd.merge(metadata, sitedata, how='outer', on=('site_key', 'leg', 'site'))
data = pd.merge(site_meta_data, hole_grouped, how='outer', on=('site_key')).fillna(np.nan)

# Play w data
plt.scatter(data['lon'], data['lat'], c=data['bottom_temp'], s=data['water_depth'].astype(float)/10)
plt.xlim((-180,180))
plt.ylim((-90,90))

plt.scatter(data['bottom_water_temp'], data['water_depth'], s=abs(data['lat']))
# plt.xlim((0,0.05))
plt.show()

plt.scatter(data['bottom_water_temp'], data['water_depth']*abs(data['lat']), s=abs(data['lat']))
# plt.xlim((0,0.05))
plt.show()



# eof
