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

# Group and average hole data for sites
hole_grouped = holedata.loc[:,('site_key', 'lat','lon','water_depth','total_penetration')].groupby("site_key").mean().reset_index()

# Combine all tables
site_meta_data = pd.merge(metadata, sitedata, how='outer', on=('site_key', 'leg', 'site'))
data = pd.merge(site_meta_data, hole_grouped, how='outer', on=('site_key')).fillna(np.nan)

# Play w data
plt.scatter(data['lon'], data['lat'], c=data['interface_flux'], s=data['water_depth'].astype(float)/10)
plt.xlim((-180,180))
plt.ylim((0,90))

plt.plot(data['interface_flux'], data['gradient'], 'o')
plt.xlim((0,0.05))

# eof
