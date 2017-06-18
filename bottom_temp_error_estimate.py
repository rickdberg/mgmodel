# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:42:04 2017

@author: rickdberg

Evaluate error associated with bottom-water temperature gridded data

"""
import numpy as np
import matplotlib.pyplot as plt
from site_metadata_compiler_completed import comp


database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

data = comp(database, metadata, site_info, hole_info)
data = data[data['leg'] != '166']  # Bahamas Bank, not used due to flushed zone
data = data[data['water_depth'].notnull()]
data = data[data['bottom_water_temp'].notnull()]
deep_data = data[data['water_depth'] > 1500]
shallow_data = data[data['water_depth'] <= 1500]

# Error calculated as relative root mean squared error of curve fit to reported values
def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())
total_rmse = rmse(data['woa_bottom_temp'], data['bottom_water_temp'])
shallow_rmse = rmse(shallow_data['woa_bottom_temp'], shallow_data['bottom_water_temp'])
deep_rmse = rmse(deep_data['woa_bottom_temp'], deep_data['bottom_water_temp'])

# Plot histogram of errors
# plt.hist(data['bottom_water_temp']-data['woa_bottom_temp'], normed=1, bins=30)

# Plot modeled vs. measured
plt.scatter(deep_data['woa_bottom_temp'], deep_data['bottom_water_temp'], c='blue', s=40, label='deep sites >1500 mbsl')
plt.scatter(shallow_data['woa_bottom_temp'], shallow_data['bottom_water_temp'], c='yellow', s=40, label='shallow sites <1500 mbsl')

plt.xlim((-2, 20))
plt.ylim((-2,20))
plt.xlabel("Estimated temperature (\u00b0C)", fontsize=20)
plt.ylabel("Measured temperature (\u00b0C)", fontsize=20)
plt.plot(np.linspace(-2, 20, num=50), np.linspace(-2, 20, num=50), 'k--')
plt.legend(loc='best')
plt.tick_params(labelsize=16)

# plt.colorbar()
# Plot temp vs water depth
#plt.scatter(data['bottom_water_temp'], data['water_depth'], s=abs(data['lat']), c=data['bottom_water_temp'])
#plt.scatter(data['woa_bottom_temp'], data['water_depth'], s=abs(data['lat']), c=data['bottom_water_temp'])
plt.show()
# eof
