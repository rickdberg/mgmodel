# -*- coding: utf-8 -*-
"""
@author: Rick Berg, University of Washington, School of Oceanography

Monte Carlo Simulation for central difference model
Must run central difference model before running this script
"""
import numpy as np
import pylab as pl

# Simulation parameters
cycles = 1000
relativeerror = 0.02

'''
# Random profile using full gaussian probability
i=0
offsets = []
while i < cycles:
    offset = np.random.normal(scale=relativeerror)*np.ones(len(concunique[:,1]))    
    offsets.append(offset)
    i = len(offsets)
'''

# Constrain error to 1-sigma of gaussian distribution
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

# Calculate integrated rates for random profiles
integratedrates = []
for n in range(cycles):
    conc = concunique[:,1] + concunique[:,1]*offsets[n]
    
    # Concentration smoothing 5-pt gaussian (approximate)
    prepad = [(conc[0]+(conc[0]-conc[2])), (conc[0]+(conc[0]-conc[1]))]
    postpad = [(conc[-1]+(conc[-1]-conc[-2])), (conc[-1]+(conc[-1]-conc[-3]))]
    concpad = np.append(np.append(prepad, conc[:]), postpad)

    conplen = concpad.size
    befconc2 = concpad[0:conplen-4]
    befconc1 = concpad[1:conplen-3]
    aftconc1 = concpad[3:conplen-1]
    aftconc2 = concpad[4:conplen]
    concsmooth = befconc2*0.06+befconc1*0.24+conc[:]*0.4+aftconc1*0.24+aftconc2*0.06
    
    # Interpolate smoothed profile
    concinterp = interp1d(concdata[:, 0], concsmooth[:], kind='linear')
    concvalues = concinterp(intervaldepths)
    
    # Run model
    edgevalues = concvalues
    concprofile = []
    modelrates = []
    for i in np.arange(timesteps):
        flux = porosity[1:-1]*Dsed[1:-1]*(edgevalues[2:] - 2*edgevalues[1:-1] + edgevalues[:-2])/(intervalthickness**2) - (pwburialflux[i] + porosity[0]*advection)*(edgevalues[2:] - edgevalues[:-2])/(2*intervalthickness)      
        edgevalues = edgevalues[1:-1] + flux*dt
        concdiff = concvalues[1:-1] - edgevalues
        modelrate = concdiff/dt
        edgevalues = np.append(np.append([ct[i]], edgevalues + modelrate*dt), [cb])
        modelrates.append(modelrate)


    modelratefinal = np.mean(modelrates, axis=0)
    integratedrate = sum(modelratefinal*intervalvector)
    integratedrates.append(integratedrate)


# Distribution statistics
modelmean = np.mean(integratedrates)
print('Mean:', modelmean)

modelmedian = np.median(integratedrates)
print('Median:', modelmedian)

stdev = np.std(integratedrates)
print('Std Dev:', stdev)


pl.hist(integratedrates, bins=50)
pl.xlabel("Integrated Rate")
pl.ylabel("N")
pl.savefig("montecarlo.png")



# eof
