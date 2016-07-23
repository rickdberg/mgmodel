# -*- coding: utf-8 -*-
"""
@author: Rick Berg, University of Washington, School of Oceanography

Monte Carlo Simulation for central difference model
Must run central difference model before running this script
"""
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage


# Simulation parameters
cycles = 100
relativeerror = precision

'''
# Random profile using full gaussian probability
i=0
offsets = []
while i < cycles:
    offset = np.random.normal(scale=relativeerror)*np.ones(len(concunique[:,1]))    
    offsets.append(offset)
    i = len(offsets)
'''

# Truncate error at 1-sigma of gaussian distribution
i=0
offsets = []
while i < cycles:
    offset = []
    j=0
    while j < len(concunique[:,1]):
        errors = np.random.normal(scale=relativeerror)
        if abs(errors) <= relativeerror:
            offset.append(errors)
            j = len(offset)
    offsets.append(offset)
    i = len(offsets)

# Calculate integrated rates for random profiles after smoothing randomized 
# errored data
integratedrates_avg = []
integratedrates_modern = []

# Smooth concentrations using 5-point gaussian using reflect at edges
conc = np.sum((concunique[:,1], np.multiply(offsets, concunique[:,1])))
concsmooth = ndimage.filters.gaussian_filter1d(conc, 1, axis=1, mode='reflect')

# Interpolate smoothed profile
concinterp = interp1d(concunique[:, 0], concsmooth, kind='linear') 
concvalues = concinterp(intervaldepths)


for n in range(cycles):
    '''
    # Concentration smoothing 5-pt gaussian (approximate), reflected at adges
    prepad = [(conc[0]+(conc[0]-conc[2])), (conc[0]+(conc[0]-conc[1]))]
    postpad = [(conc[-1]+(conc[-1]-conc[-2])), (conc[-1]+(conc[-1]-conc[-3]))]
    concpad = np.append(np.append(prepad, conc[:]), postpad)

    conplen = concpad.size
    befconc2 = concpad[0:conplen-4]
    befconc1 = concpad[1:conplen-3]
    aftconc1 = concpad[3:conplen-1]
    aftconc2 = concpad[4:conplen]
    concsmooth = befconc2*0.06+befconc1*0.24+conc[:]*0.4+aftconc1*0.24+aftconc2*0.06
    '''
    # Run model (Check formulation)
    edgevalues = concvalues[n]
    modelrates = []
    for i in range(timesteps):
        flux = porosity[1:-1]*Dsed[1:-1]*(edgevalues[2:] - 2*edgevalues[1:-1]
        + edgevalues[:-2])/(intervalthickness**2) - (pwburialflux[i] 
        + porosity[0]*advection)*(edgevalues[2:] - edgevalues[:-2]) / (2*intervalthickness)
        next_edgevalues = edgevalues[1:-1] + flux*dt
        concdiff = concvalues[n][1:-1] - next_edgevalues
        modelrate = concdiff/dt
        edgevalues = np.append(np.append([ct[i]], next_edgevalues + modelrate*dt), [cb])
        modelrates.append(modelrate)

    modelratefinal_avg = np.mean(modelrates, axis=0)
    integratedrate_avg = sum(modelratefinal_avg * intervalvector)
    integratedrates_avg.append(integratedrate_avg)
    
    modelratefinal_modern = modelrates[-1]
    integratedrate_modern = sum(modelratefinal_modern * intervalvector)
    integratedrates_modern.append(integratedrate_modern)

# Distribution statistics
modelmean_avg = np.mean(integratedrates_avg)
modelmean_modern = np.mean(integratedrates_modern)
print('Modern Mean:', modelmean_modern)

modelmedian_avg = np.median(integratedrates_avg)
modelmedian_modern = np.median(integratedrates_modern)
print('Modern Median:', modelmedian_modern)

stdev_avg = np.std(integratedrates_avg)
stdev_modern = np.std(integratedrates_modern)
print('Modern Std Dev:', stdev_modern)


pl.hist(integratedrates_modern, bins=50)
pl.xlabel("Integrated Rate")
pl.ylabel("N")
pl.savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\montecarlo_{}_{}.png".format(Leg, Site))

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cur = con.cursor()

# Save metadata in database
cur.execute("""select site_key from site_summary where leg = '{}' and site = '{}' ;""".format(Leg, Site))
site_key = cur.fetchone()[0]
cur.execute("""insert into model_metadata_{} (site_key, leg, site, hole, solute, 
mean_integrated_rate_avg, median_integrated_rate_avg, model_std_deviation_avg, mean_integrated_rate_modern, median_integrated_rate_modern, model_std_deviation_modern, r_squared, 
timesteps, bottom_age, number_of_intervals, datapoints, smoothing_window, measurement_precision, 
monte_carlo_cycles, grid_peclet, courant, ds, ds_reference_temp, script, run_date) 
VALUES ({}, {}, {}, '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 
{}, '{}', '{}')  ON DUPLICATE KEY UPDATE hole='{}', solute='{}', mean_integrated_rate_avg={}, 
median_integrated_rate_avg={}, model_std_deviation_avg={}, mean_integrated_rate_modern={}, median_integrated_rate_modern={},model_std_deviation_modern={}, r_squared={}, timesteps={}, bottom_age={}, 
number_of_intervals={}, datapoints={}, smoothing_window={}, measurement_precision={}, 
monte_carlo_cycles={}, grid_peclet={}, courant={}, ds={}, ds_reference_temp={}, 
script='{}', run_date='{}' ;""".format(Solute, site_key, Leg, Site, Hole, Solute, modelmean_avg, 
modelmedian_avg, stdev_avg, modelmean_modern, modelmedian_modern, stdev_modern, rsquared, timesteps, bottomage, intervals, datapoints, smoothing, precision, 
cycles, gpeclet, courant, Ds, TempD, Script, Date, Hole, Solute, modelmean_avg, modelmedian_avg, 
stdev_avg, modelmean_modern, modelmedian_modern, stdev_modern, rsquared, timesteps, bottomage, intervals, datapoints, smoothing, precision, cycles, gpeclet, 
courant, Ds, TempD, Script, Date))
con.commit()

# eof
