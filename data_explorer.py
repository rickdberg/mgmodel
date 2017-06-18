# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:42:04 2017

@author: rickdberg

Data explorer

['etopo1_depth', 'surface_porosity', 'sed_thickness', 'crustal_age',
                'coast_distance', 'ridge_distance', 'seamount', 'surface_productivity',
                'toc', 'opal', 'caco3', 'woa_temp', 'woa_salinity', 'woa_o2',
                'lith1','lith2','lith3','lith4','lith5','lith6','lith7','lith8',
                'lith9','lith10','lith11','lith12','lith13']

"""
import numpy as np
import matplotlib.pyplot as plt
from site_metadata_compiler_completed import comp

database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load site data
data = comp(database, metadata, site_info, hole_info)


# Geographic patterns
plt.scatter(data['lon'], data['lat'],
            c=data['burial_flux'], s=data['sed_rate'].astype(float)*1000000)
plt.xlim((-180,180))
plt.ylim((-90,90))

# Direct comparison
plt.scatter(data['sed_rate_combined'].astype(float),
            data['sed_rate'].astype(float), s= 100,
            c='b')
# plt.xlim((-10,0))
# plt.ylim((0,1))
plt.show()

# Histogram
d = data['interface_flux'].astype(float)
h = d.hist(bins=50)
h.set_ylabel('$Count$', fontsize=20)
h.set_xlabel('$Mg\ Flux\ (mol\ m^{-2}\ y^{-1})$', fontsize=20)

median_stdev = np.nanmedian(data['stdev_flux'].astype(float)/data['interface_flux'].astype(float)*100)
# eof
