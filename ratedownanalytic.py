# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:33 2015

@author: Rick Berg, University of Washington, School of Oceanography

!!!!!!!!!!Need to add in way to deal with upper boundary condition, saving output,
input process. Is this following the sediment package rather than depth?

Script for modeling reaction rates in marine sediments.

Starts with existing sediment column, models sediment column build upward as a downward flow of porewater.

Bottom boundary condition set as constant concentration.

Accounts for external flow and sediment/porewater burial and compaction.

Rates are kept constant over the period of sediment column build and are 
kept constant at specific depths.(Do not follow the sediment packages)

Start with the current profile, after each interation, calculate the
rate needed to keep the profile there.

Final rate output is in mol m-3(bulk sediment) y-1
(Same as Wang model)

Units used: meters, years, mol m**-3 (aka mM), Celsius. 


[Column, Row]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from scipy.interpolate import interp1d
from scipy import optimize, integrate
from collections import defaultdict
import matplotlib.gridspec as gridspec


# Site ID
Leg = 'Analytical Ratedown'
Site = 'NA'
Solute = 'Mg'


# Model parameters
timesteps = 30000  # Number of timesteps
intervals = 100  # Number of intervals
iterations = 20  # Number of iterations for rate fitting

#Site parameters
advection = 0  # external advection rate (at seafloor) (m/y) Negative is upward
bottomtemp = 18  # bottom water temperature (C)
tempgradient = 0  # geothermal gradient (C/m)

# Species parameters
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C
TempD = 18  # Temperature at which diffusion coefficient is known

###############################################################################
# Load files as pandas dataframes

# Current concentration profile (mM or mol m**-3)
# Note: Input data with no headers
concdata = pd.read_csv(r'C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Working\conc.txt', sep="\t", header=None, skiprows=None)
concdata = concdata.as_matrix()
ct0 = [concdata[0, 1]]  # mol per m^3 in modern average seawater

# Porosity profile
# Note: Input data with no headers
pordata = pd.read_csv(r'C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Working\por.txt', sep="\t", header=None, skiprows=None)
pordata = pordata.as_matrix()

# Temperature profile (degrees C)
def sedtemp(z, bottomtemp):
    return bottomtemp + np.multiply(z, tempgradient)

# Sedimentation rate profile (m/y)
# Note: Input data with no headers
# Note: Input data must have time at depth=0
sed = pd.read_csv(r'C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Working\sedrate.txt', sep="\t", header=None, skiprows=None)
sed = sed.as_matrix()

###############################################################################
###############################################################################
###############################################################################
# Average duplicates in concentration and porosity datasets

# Averaging function (from http://stackoverflow.com/questions/4022465/average-the-duplicated-values-from-two-paired-lists-in-python)
def averages(names, values):
    # Group the items by name
    value_lists = defaultdict(list)
    for name, value in zip(names, values):
        value_lists[name].append(value)

    # Take the average of each list
    result = {}
    for name, values in value_lists.items():
        result[name] = sum(values) / float(len(values))
    
    # Make it a Numpy array and pull out values
    resultkeys = np.array(list(result.keys()))
    resultvalues = np.array(list(result.values()))
    sorted = np.column_stack((resultkeys[np.argsort(resultkeys)], resultvalues[np.argsort(resultkeys)]))
    return sorted

# Concentration vector
conc = averages(concdata[:, 0], concdata[:, 1])

# Porosity vector
por = averages(pordata[:, 0], pordata[:, 1])

###############################################################################
# Concentration data preparation
'''
# Smooth and interpolate the concentration data using 3 point gaussian

# Concentration smoothing 3-pt gaussian (approximate)

prepad = [(conc[0, 1]+(conc[0, 1]-conc[2, 1])), (conc[0, 1]+(conc[0, 1]-conc[1, 1]))]
postpad = [(conc[-1, 1]+(conc[-1, 1]-conc[-2, 1])), (conc[-1, 1]+(conc[-1, 1]-conc[-3, 1]))]
concpad = np.append(np.append(prepad, conc[:, 1]), postpad)

conplen = concpad.size
befconc2 = concpad[0:conplen-4]
befconc1 = concpad[1:conplen-3]
aftconc1 = concpad[3:conplen-1]
aftconc2 = concpad[4:conplen]

concsmooth = np.column_stack((conc[:,0], (befconc2*0.06+befconc1*0.24+conc[:,1]*0.4+aftconc1*0.24+aftconc2*0.06)))
conc = concsmooth
'''
# Add in the modern seawater concentration at surface and make interpolation function
concfull = np.concatenate((np.array(([0],ct0)).T, conc), axis=0)

concinterp = interp1d(concfull[:, 0], concfull[:, 1], kind='linear')

###############################################################################
# Boundary conditions

# Bottom water concentration curve vs time (upper boundary condition)
ct = ct0*np.ones(timesteps)
ct = ct.tolist()

# Lower boundary condition (constant set at dirichlet)
cb = conc[-1, 1]

###############################################################################
# Porosity data preparation

porvalues = por[:, 1]
pordepth = por[:, 0]

'''
# Porosity smoothing 3-pt gaussian (approximate)
porpad = pd.concat([Series(por.iloc[0, 1]+(por.iloc[0, 1]-por.iloc[1, 1])), porvalues, Series(por.iloc[-1, 1]+(por.iloc[-1, 1]-por.iloc[-2, 1]))], axis=0, ignore_index=True)
conplen = porpad.size
befpor = Series(porpad.iloc[0:conplen-2])
aftpor = Series(porpad.iloc[2:conplen+1])
aftpor = aftpor.reset_index(drop=True)
porsmooth = pd.concat([por.iloc[:, 0], Series(befpor*0.25+por.iloc[:, 1]*0.5+aftpor*0.25)], axis=1, ignore_index=True)

# Add in the surface porosity
surfacepor = DataFrame([0, porsmooth.iloc[0, 1]])
surfacepor = surfacepor.T
porfull = pd.concat([surfacepor, porsmooth], axis=0, ignore_index=True)


# Interpolate every 1 meter to depth of concentration profile
porinterp = interp1d(porfull.iloc[:, 0], porfull.iloc[:, 1], kind='linear')
porfinaldepth = np.arange(1, max(concfull.iloc[:, 0]))
porfinal = pd.concat([Series(porfinaldepth), Series(porinterp(porfinaldepth))], axis=1)
porfinal = pd.concat([surfacepor, porfinal], axis=0, ignore_index=True)
'''

# Porosity curve fit
def porcurve(z, a):
    portop = por[0, 1]
    porbottom = por[-1, 1]
    return (portop-porbottom) * np.exp(-a*z) + porbottom

porfit, porcov = optimize.curve_fit(porcurve, pordepth, porvalues)

# Solids curve fit (based on porosity)
def solidcurve(z, a):
    portop = por[0, 1]
    porbottom = por[-1, 1]
    return 1-((portop-porbottom) * np.exp(-a*z) + porbottom)

###############################################################################
# Diffusion coefficient function

# Viscosity from Mostafa H. Sharqawy 12-18-2009, MIT (mhamed@mit.edu) Sharqawy M. H., Lienhard J. H., and Zubair, S. M., Desalination and Water Treatment, 2009
# Viscosity at diffusion coefficient reference temperature
# Td is the reference temperature (TempD), T is the in situ temperature
def Dst(Td, T):
    # Viscosity at reference temperature 
    muwd = 4.2844324477E-05 + 1/(1.5700386464E-01*(Td+6.4992620050E+01)**2+-9.1296496657E+01)
    A = 1.5409136040E+00 + 1.9981117208E-02 * Td + -9.5203865864E-05 * Td**2
    B = 7.9739318223E+00 + -7.5614568881E-02 * Td + 4.7237011074E-04 * Td**2
    visd = muwd*(1 + A*0.035 + B*0.035**2)

    # Viscosity vector
    muw = 4.2844324477E-05 + 1/(1.5700386464E-01*(T+6.4992620050E+01)**2+-9.1296496657E+01)
    C = 1.5409136040E+00 + 1.9981117208E-02 * T + -9.5203865864E-05 * T**2
    D = 7.9739318223E+00 + -7.5614568881E-02 * T + 4.7237011074E-04 * T**2
    vis = muw*(1 + C*0.035 + D*0.035**2)
    T = T+273.15
    Td = Td+273.15
    return T/vis*visd*Ds/Td  # Stokes-Einstein equation

###############################################################################
# Calculate ages at depth intervals

sedtimes = sed[:, 1]
seddepths = sed[:, 0]
sedrates = np.diff(seddepths, axis=0)/np.diff(sedtimes, axis=0)  # m/y

# Function for appending float to array and trimming the series to that value
def insertandcut(value, array):
    findindex = array.searchsorted(value)
    sortedarray = np.insert(array, findindex, value)[:findindex+1]
    return sortedarray
maxconcdepth = np.max(concdata[:,0])
sedconcdepths = insertandcut(maxconcdepth, seddepths)

# Split sediment column into intervals (intervals of differing mass but same thickness)
intervalthickness = maxconcdepth/intervals
intervalradius = intervalthickness/2
intervaldepths = (np.arange(intervals+1)*intervalthickness)
midpoints = np.add(intervaldepths[:-1], np.diff(intervaldepths)/2)

# Mass of each interval based on porosity curve
runningmass = []
for i in intervaldepths:
    runningmass.append(integrate.quad(solidcurve, 0, i, args=(porfit))[0])
intervalmass = np.diff(runningmass)
columnmass = np.cumsum(intervalmass)

# Sediment mass accumulation rates for each age-depth section
sectionmasses = [0]
sedmassrates = np.zeros(len(sedrates))  # unique avg sed mass accumulation rates to bottom of conc profile
for i in np.arange(len(sedrates)):
    sectionmass = (integrate.quad(solidcurve, seddepths[i], seddepths[i+1], args=(porfit)))[0]
    sedmassrate = (sectionmass/np.diff(sedtimes)[i])
    sedmassrates[i] = sedmassrate
    sectionmasses.append(sectionmass)
sectionmasses = np.array(sectionmasses)

# Interval ages and bottom age (at bottom of concentration profile)

midpointages = []
intervalages = [] 
for i in np.arange(len(seddepths)-1):
    for n in np.arange(len(intervaldepths)):
        if intervaldepths[n] >= seddepths[i] and intervaldepths[n] < seddepths[i+1]:
            intervalage = sedtimes[i]+(integrate.quad(solidcurve, sedconcdepths[i], intervaldepths[n], args=(porfit)))[0]/sedmassrates[i]
            intervalages.append(intervalage)
    for m in np.arange(len(midpoints)):
        if midpoints[m] >= seddepths[i] and midpoints[m] < seddepths[i+1]:
            midpointage = sedtimes[i]+(integrate.quad(solidcurve, sedconcdepths[i], midpoints[m], args=(porfit)))[0]/sedmassrates[i]
            midpointages.append(midpointage)

for n in np.arange(len(intervaldepths)):
    if intervaldepths[n] >= seddepths[-1]:
        intervalage = sedtimes[-1]+(integrate.quad(solidcurve, seddepths[-1], intervaldepths[n], args=(porfit)))[0]/sedmassrates[-1]
        intervalages.append(intervalage)
for m in np.arange(len(midpoints)):    
    if midpoints[m] >= seddepths[-1]:
        midpointage = sedtimes[-1]+(integrate.quad(solidcurve, seddepths[-1], midpoints[m], args=(porfit)))[0]/sedmassrates[-1]
        midpointages.append(midpointage)

midpointages = np.array(midpointages)
intervalages = np.array(intervalages[1:])
bottomage = intervalages[-1]

# intervaldiff = np.diff(midpointages)

  
dt = bottomage/timesteps # time step for each interval
sedconctimes = np.append(sedtimes[0:len(sedrates)], bottomage)

###############################################################################
# Plot data for inspection

previewfigs, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 6))

gs = gridspec.GridSpec(1, 5)
gs.update(wspace=0.3)
ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4])

ax1.plot(concdata[:, 1], concdata[:, 0], '--go')
ax2.plot(por[:, 1], por[:, 0], 'mo', label='Measured')
ax2.plot(porcurve(pordepth, porfit), pordepth, 'k-', label='Curve fit', linewidth=3)
ax2.legend(loc='lower right', fontsize='small')
ax3.plot(sedtemp(np.arange(maxconcdepth), bottomtemp), np.arange(maxconcdepth), 'k-', linewidth=3)
ax1.set_ylabel('Depth (mbsf)')
ax1.set_xlabel('Concentration (mM)')
ax2.set_xlabel('Porosity')
ax3.set_xlabel('Temperature (\u00b0C)')
ax1.locator_params(axis='x', nbins=4)
ax2.locator_params(axis='x', nbins=4)
ax3.locator_params(axis='x', nbins=4)
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
plt.waitforbuttonpress()
plt.close()

###############################################################################
#############
#########
###########
############
#########
# fix
# Sedimentation rate at each time step
iterationtimes = ((np.arange(timesteps+1)*dt)[1:]).tolist()
sedmassratetime = []
for i in np.arange(len(sedconctimes)-1):
    if i == 0:
        sedmassratet = np.ones(iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i+1]))) + 1 - iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i])))) * sedmassrates[i]
    else:
        sedmassratet = np.ones(iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i+1])))-iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i])))) * sedmassrates[i]
    sedmassratetime = sedmassratetime + sedmassratet.tolist()
sedmassratetime = np.array(sedmassratetime)

###############################################################################
# Reactive transport model

#Sediment properties
porosity = porcurve(intervaldepths, porfit)
tortuosity = 1-np.log(porosity**2)  # Tortuosity
Dsed = Dst(TempD, sedtemp(intervaldepths, bottomtemp))/tortuosity  # Effective diffusion coefficient

# Pore water burial flux at each time step
deeppor = porcurve(pordepth[-1], porfit)
deepsolid = solidcurve(pordepth[-1], porfit)
pwburialflux = np.zeros([timesteps,1])
for i in np.arange(len(sedmassratetime)):
    pwburial = deeppor*sedmassratetime[i]/deepsolid
    pwburialflux[i] = np.flipud(pwburial)

# Can change this to make all parameters at midpoints instead of edges
# Concentration with no reaction ("no reaction" model run)
concvalues = concinterp(midpoints).tolist() # initial conditions
midconc = concinterp(midpoints).tolist()
concprofile = []
modelrates = []
timevector = np.append(np.append([0.5*dt], np.ones(intervals-2)*dt), [0.5*dt])
for i in np.arange(timesteps):
    if (pwburialflux[i] + porosity[0]*advection) >= 0:
        fluxtop = (porosity[0]*Dsed[0]*(concvalues[0]-ct[i])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*ct[i])    
        fluxbottom = (porosity[-1]*Dsed[-1]*(cb-concvalues[-1])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*concvalues[-1])
        flux = (porosity[1:-1]*Dsed[1:-1]*np.diff(concvalues)/intervalthickness - (pwburialflux[i] + porosity[0]*advection)*concvalues[:-1])
    else:
        fluxtop = (porosity[0]*Dsed[0]*(concvalues[0]-ct[i])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*concvalues[0])    
        fluxbottom = (porosity[-1]*Dsed[-1]*(cb-concvalues[-1])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*cb)
        flux = (porosity[1:-1]*Dsed[1:-1]*np.diff(concvalues)/intervalthickness - (pwburialflux[i] + porosity[0]*advection)*concvalues[1:])
    fluxtotal = np.append(np.insert(flux,0,fluxtop), fluxbottom)
    concvalues = concvalues + np.diff(fluxtotal)/intervalthickness*dt
    concdiff = midconc - concvalues
    modelrate = 0.5*concdiff/timevector
    concvalues = concvalues + modelrate*timevector
    modelrates.append(modelrate)
    concprofile.append(concvalues)
'''
# Plot to show progression of model run
for i in np.arange(11):
    plt.plot(concprofile[i*900], midpoints)
plt.gca().invert_yaxis()
plt.show()
'''
# Grid peclet and courant numbers
gpeclet = np.abs((advection*porosity[0]/porosity[-1]+pwburialflux[0]/porosity[-1])*intervalthickness/np.min(Dsed))
print('Grid Peclet (less than 1):', gpeclet)
courant = np.abs((advection*porosity[0]/porosity[-1]+pwburialflux[0]/porosity[-1])*dt/intervalthickness)
print('Courant (less than 0.5):', courant)  
neumann = intervalthickness**2/(3*np.max(Dsed))
print('Von Neumann (greater than dt):', neumann, 'dt:', dt)
'''
# Iterate until good fit with profile, output integrated rate (mol m^-2(bulk) y^-1)
concvalues = concinterp(midpoints).tolist()
modelprofile = []
modelrates = []
for n in np.arange(iterations):
    for i in np.arange(timesteps):
        if (pwburialflux[i] + porosity[0]*advection) >= 0:        
            fluxtop = (porosity[0]*(Dsed[0]*(concvalues[0]-ct[i])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*ct[i]))    
            fluxbottom = (porosity[-1]*Dsed[-1]*(cb-concvalues[-1])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*concvalues[-1])
            flux = (porosity[1:-1]*Dsed[1:-1]*np.diff(concvalues)/intervalthickness - (pwburialflux[i] + porosity[0]*advection)*concvalues[:-1])
        else:
            fluxtop = (porosity[0]*(Dsed[0]*(concvalues[0]-ct[i])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*concvalues[0]))    
            fluxbottom = (porosity[-1]*Dsed[-1]*(cb-concvalues[-1])/intervalradius - (pwburialflux[i] + porosity[0]*advection)*cb)
            flux = (porosity[1:-1]*Dsed[1:-1]*np.diff(concvalues)/intervalthickness - (pwburialflux[i] + porosity[0]*advection)*concvalues[1:])
        fluxtotal = np.append(np.insert(flux,0,fluxtop), fluxbottom)
        concfinal = concvalues + np.diff(fluxtotal)/intervalthickness*dt + modelrate*dt
        concvalues = concfinal
    concdiff = midconc - concvalues
    modelrates.append(modelrate)
    modelrate = modelrate + 0.5*concdiff/intervalages
    modelprofile.append(concvalues)
'''
modelratefinal = np.mean(modelrates, axis=0)
integratedrate = sum(modelratefinal)*intervalthickness
print('Integrated rate:', integratedrate)

'''
# modelratesed = porcurve(midpoints, porfit)*concdiff/midpointages
modelratefinal = modelrates[-1]
integratedrate = sum(modelratefinal)*intervalthickness
print('Integrated rate:', integratedrate)
'''
'''
# Plot to show progression of model iterations
for i in np.arange(iterations):
    plt.plot(modelprofile[i], midpoints)
    plt.plot(concdata[:, 1], concdata[:, 0], '--go')
plt.gca().invert_yaxis()
plt.show()
'''

# R-squared function
def rsq(modeled, measured):
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured)-1)*np.var(measured)
    return 1-sse/sstotal

# Calculate r-squared
rsquared = rsq(concprofile[-1], midconc)
print('r-squared:', rsquared)

###############################################################################
# Plot final data (original concentrations, modeled concentrations, and reaction rates)

previewfigs2, (ax4, ax5) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))

gs2 = gridspec.GridSpec(1, 4)
gs2.update(wspace=0.3)
ax4 = plt.subplot(gs2[0, :2])
ax5 = plt.subplot(gs2[0, 2:4])
ax5.grid()

# ax4.plot(noreaction, midpoints)
ax4.plot(concprofile[-1], midpoints, label='Model')
ax4.plot(concdata[:,1], concdata[:,0], 'go', label='Measured', mfc='None')
ax4.plot(midconc, midpoints, 'r+', label='Interpolated', mfc='None')
ax5.plot(modelratefinal*1000000, midpoints)
ax4.legend(loc='lower left', fontsize='small')
ax4.set_ylabel('Depth (mbsf)')
ax4.set_xlabel('Concentration (mM)')
ax5.set_xlabel('Reaction Rate x 10^-6')
ax4.locator_params(axis='x', nbins=4)
ax5.locator_params(axis='x', nbins=4)
ax4.invert_yaxis()
ax5.invert_yaxis()




###### Save figure, data, script, and stats in files

# eof
