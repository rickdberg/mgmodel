# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:58:42 2017

@author: rickdberg
Histograms
"""
import numpy as np
import datetime
from pylab import savefig
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from site_metadata_compiler_completed import comp

# Connect to database
database = "mysql://root:neogene227@localhost/iodp_compiled"
portable = 'mad_all'
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"


# Load and prepare all input data
site_metadata = comp(database, metadata, site_info, hole_info)
site_metadata = site_metadata[site_metadata['advection'].astype(float)>=0]

# Histogram
d = site_metadata['interface_flux'].astype(float)*1000
ax = d.hist(bins=50)
ax.set_xlabel("$ Interface\ flux\ (mmol\ m^{-2}\ y^{-1})$", fontsize=20)
ax.set_ylabel("$n$", fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

r = site_metadata['r_squared'].astype(float)
ax = r.hist(bins=50)
ax.set_xlabel("$ R^2\ values$", fontsize=20)
ax.set_ylabel("$n$", fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


std = site_metadata['stdev_flux'].astype(float)*1000
ax = std.hist(bins=5000)
ax.set_xlabel("$ Standard\ deviations$", fontsize=20)
ax.set_ylabel("$n$", fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

np.nanmedian(site_metadata['stdev_flux'].astype(float)/site_metadata['interface_flux'].astype(float)*100)



dp = site_metadata['datapoints'].astype(float)
plt.scatter(d, std)







