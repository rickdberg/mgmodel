# -*- coding: utf-8 -*-
"""
@author: Rick Berg, University of Washington, School of Oceanography

Monte Carlo Simulation for interface flux model
Must run interface_flux model before running this script


Coonect to database, what to do with stats?, plot non-normalized distributions
"""
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize, integrate, stats
from matplotlib import mlab
from interface_flux import Precision, concunique, porvalues, pordepth, rsq, porcurve, Solute, dp, porfit, seddepths, sedtimes, Dstp, TempD, bottom_temp, z, advection, Leg, Site
import MySQLdb

# Simulation parameters
cycles = 5000

# Concentration offsets - using full gaussian probability
relativeerror = Precision
conc_offsets = np.random.normal(scale=relativeerror, size=(cycles, len(concunique[:dp,1])))  

# Porosity offsets - using full gaussian probability
# Error calculated as relative root mean squared error of curve fit to reported values
def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())
por_error = rmse(porcurve(pordepth, porfit), porvalues)
por_offsets = np.random.normal(scale=por_error, size=(cycles, len(porvalues)))

'''
# Concentration offsets - truncate error at 1-sigma of gaussian distribution
i=0
offsets = []
while i < cycles:
    offset = []
    j=0
    while j < len(concunique[:dp,1]):
        errors = np.random.normal(scale=relativeerror)
        if abs(errors) <= relativeerror:
            offset.append(errors)
            j = len(offset)
    offsets.append(offset)
    i = len(offsets)
'''
###############################################################################
# Calculate fluxes for random profiles

# Get randomized concentration matrix (within realistic ranges)
conc_rand = np.add(concunique[:dp,1], np.multiply(conc_offsets, concunique[:dp,1]))
conc_rand[conc_rand < 0] = 0

por_rand = np.add(porvalues, por_offsets)
por_rand[por_rand > 0.90] = 0.90
for n in range(cycles):
    por_rand[por_rand < 0.3] = 0.3

portop = np.max(porvalues[:3])
portop_rand = np.add(portop, por_offsets[:,0])
portop_rand[portop_rand > 0.90] = 0.90
for n in range(cycles):
    portop_rand[portop_rand < por_rand[n,-1]] = por_rand[n,-1]

def conc_curve_mc(z, a):
    return (conc_rand[n,0]-conc_rand[n,-1]) * np.exp(np.multiply(np.multiply(-1, a), z)) + conc_rand[n,-1]

def por_curve_mc(z, a):
    portop = np.max(por_rand[n,:3])  # Greatest of top 3 porosity measurements for upper porosity boundary
    porbottom = por_rand[n,-1]  # Takes lowest porosity measurement as the lower boundary
    return (portop-porbottom) * np.exp(np.multiply(np.multiply(-1, a), z)) + porbottom

def solid_curve_mc(z, a):
    portop = np.max(por_rand[n,:3])  # Greatest of top 3 porosity measurements for upper porosity boundary
    porbottom = por_rand[n,-1]  # Takes lowest porosity measurement as the lower boundary
    return np.subtract(1, ((portop-porbottom) * np.exp(np.multiply(np.multiply(-1, a), z)) + porbottom))

# Calculate flux
interface_fluxes = []
conc_fits = []
por_fits = []
sectionmasses = []
for n in range(cycles):
    # Fit exponential curve to each randomized concentration profile
    conc_fit, conc_cov = optimize.curve_fit(conc_curve_mc, concunique[:dp,0], conc_rand[n], p0=0.1)
    conc_fit = conc_fit[0]
    conc_fits.append(conc_fit)
    
    # R-squared of each randomized fit
    r_squared = rsq(conc_curve_mc(concunique[:dp,0], conc_fit), conc_rand[n])

    # Fit exponential curve to each randomized porosity profile
    por_fit, por_cov = optimize.curve_fit(por_curve_mc, pordepth, por_rand[n], p0=0.01)
    por_fit = por_fit[0]
    por_fits.append(por_fit)
    # Pore water burial mass flux
    # Sediment mass (1-dimensional volume of solids) accumulation rates for each age-depth section
    # Assumes constant sediment mass (really volume of solids) accumulation rates between age-depth measurements
    sectionmass = (integrate.quad(solid_curve_mc, seddepths[0], seddepths[1], args=(por_fit)))[0]
    sectionmasses.append(sectionmass)

# Pore water burial flux
sedmassrate = sectionmasses/np.diff(sedtimes)[0]
deeppor = por_rand[:,-1]
deepsolid = 1 - por_rand[:,-1]
pwburialflux = deeppor*sedmassrate/deepsolid

# Calculate random porosities

# por_fit_rand = por_curve_mc(z, por_fits)
tortuosity_rand = 1-np.log(portop_rand**2)
Dsed_rand = Dstp(TempD, bottom_temp)/tortuosity_rand


# Plot all the monte carlo runs
# conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fits)
# por_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fits)

# Calculate fluxes
a = conc_fits
gradient = (conc_rand[:,0] - conc_rand[:, -1]) * -1 * a * np.exp(np.multiply(np.multiply(-1, a), z))  # Derivative of conc_curve @ z
interface_fluxes = portop_rand * Dsed_rand * gradient + (portop_rand * advection + pwburialflux) * conc_curve_mc(z, conc_fits)

###############################################################################
# Distribution statistics

# Stats on normal distribution
mean_flux = np.mean(interface_fluxes)
print('Mean Flux:', mean_flux)

median_flux = np.median(interface_fluxes)
print('Median Flux:', median_flux)

stdev_flux = np.std(interface_fluxes)
print('Std Dev Flux:', stdev_flux)

skewness = stats.skew(abs(interface_fluxes))
print('skewness:', skewness)
z_score, p_value = stats.skewtest(abs(interface_fluxes))
print('z-score:', z_score)

# Stats on lognormal distribution
interface_fluxes_log = -np.log(abs(interface_fluxes))

mean_flux_log = np.mean(interface_fluxes_log)
print('Mean ln(Flux):', mean_flux_log)

median_flux_log = np.median(interface_fluxes_log)
print('Median ln(Flux):', median_flux_log)

stdev_flux_log = np.std(interface_fluxes_log)
print('Std Dev ln(Flux):', stdev_flux_log)

skewness_log = stats.skew(interface_fluxes_log)
print('skewness (ln):', skewness_log)
z_score_log, p_value_log = stats.skewtest(interface_fluxes_log)
print('z-score (ln):', z_score_log)

stdev_flux_lower = np.exp(-(median_flux_log-stdev_flux_log))
print('Std Dev Lower: ', stdev_flux_lower-median_flux)
stdev_flux_upper = np.exp(-(median_flux_log+stdev_flux_log))
print('Std Dev Upper: ', stdev_flux_upper-median_flux)

###############################################################################
# Plot distributions

figure_2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot histogram of results
n_1, bins_1, patches_1 = ax1.hist(interface_fluxes, normed=1, bins=30, facecolor='orange')

# Best fit normal distribution line to results
bf_line_1 = mlab.normpdf(bins_1, median_flux, stdev_flux)
ax1.plot(bins_1, bf_line_1, 'k--', linewidth=2)
ax1.set_xlabel("Interface Flux")

[left_raw, right_raw] = ax1.get_xlim()
[bottom_raw, top_raw] = ax1.get_ylim()
ax1.text((left_raw+(right_raw-left_raw)/20), (top_raw-(top_raw-bottom_raw)/20), 'sk = {}'.format(np.round(skewness, 2)))
ax1.text((left_raw+(right_raw-left_raw)/20), (top_raw-(top_raw-bottom_raw)/10), "z' = {}".format(np.round(z_score, 2)))

# Plot histogram of ln(results)
n_2, bins_2, patches_2 = ax2.hist(interface_fluxes_log, normed=1, bins=30, facecolor='g')

# Best fit normal distribution line to ln(results)
bf_line_2 = mlab.normpdf(bins_2, median_flux_log, stdev_flux_log)
ax2.plot(bins_2, bf_line_2, 'k--', linewidth=2)
ax2.set_xlabel("ln(abs(Interface Flux)")

[left_log, right_log] = ax2.get_xlim()
[bottom_log, top_log] = ax2.get_ylim()
ax2.text((left_log+(right_log-left_log)/20), (top_log-(top_log-bottom_log)/20), 'sk = {}'.format(np.round(skewness_log, 2)))
ax2.text((left_log+(right_log-left_log)/20), (top_log-(top_log-bottom_log)/10), "z' = {}".format(np.round(z_score_log, 2)))

plt.show()

###############################################################################
# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cursor = con.cursor()

# Send metadata to database
cursor.execute("""select site_key from site_info where leg = '{}' and site = '{}' ;""".format(Leg, Site))
site_key = cursor.fetchone()[0]
cursor.execute("""insert into metadata_{}_flux (site_key, mc_cycles, porosity_error, mean_flux, 
median_flux, stdev_flux, skewness, z_score, mean_flux_log, median_flux_log, stdev_flux_log, 
stdev_flux_lower, stdev_flux_upper, skewness_log, z_score_log) 
VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}) 
ON DUPLICATE KEY UPDATE mc_cycles={}, porosity_error={}, mean_flux={}, median_flux={}, stdev_flux={}, 
skewness={}, z_score={}, mean_flux_log={}, median_flux_log={}, stdev_flux_log={}, stdev_flux_lower={}, 
stdev_flux_upper={}, skewness_log={}, z_score_log={}
;""".format(Solute, site_key, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, z_score, 
            mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, 
            skewness_log, z_score_log, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, 
            z_score, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, 
            skewness_log, z_score_log))
con.commit()


# Save figure and fluxes from each run
pl.savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\montecarlo_{}_{}.png".format(Leg, Site))
np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\monte carlo_{}_{}.csv".format(Leg, Site), interface_fluxes, delimiter=",")

# eof
