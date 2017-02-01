# -*- coding: utf-8 -*-
"""
@author: Rick Berg, University of Washington, School of Oceanography

Monte Carlo Simulation for interface flux model
Must run interface_flux model before running this script
"""
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage


# Simulation parameters
cycles = 1000
relativeerror = Precision

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
    while j < len(concunique[:dp,1]):
        errors = np.random.normal(scale=relativeerror)
        if abs(errors) <= relativeerror:
            offset.append(errors)
            j = len(offset)
    offsets.append(offset)
    i = len(offsets)

###############################################################################
# Calculate fluxes for random profiles

conc = np.sum((concunique[:dp,1], np.multiply(offsets, concunique[:dp,1])))


# Calculate flux
interface_fluxes = []
a = conc_fit
for n in range(cycles):
    # Fit exponential curve to rendomized profile
    conc_fit, conc_cov = optimize.curve_fit(conc_curve, concunique[:dp,0], conc[n])
    conc_fit = conc_fit[0]
    conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fit)

    # R-squared of randomized fit
    r_squared = rsq(conc_curve(concunique[:dp,0], conc_fit), conc)

    conc_rand = conc[n]
    gradient = (conc_rand[0] - conc_rand[dp-1]) * -a * np.exp(-a * z)  # Derivative of conc_curve @ z
    flux = porosity * Dsed * gradient + (porosity * advection + pwburialflux) * conc_curve(z, conc_fit)
    interface_fluxes.append(flux)

# Distribution statistics
mean_flux = np.mean(interface_fluxes)
print('Mean Flux:', mean_flux)

median_flux = np.median(interface_fluxes)
print('Median Flux:', median_flux)

stdev_flux = np.std(interface_fluxes)
print('Std Dev Flux:', stdev_flux)


pl.hist(interface_fluxes, bins=30)
pl.xlabel("Interface Flux")
pl.ylabel("N")


# pl.savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\montecarlo_{}_{}.png".format(Leg, Site))



# eof
