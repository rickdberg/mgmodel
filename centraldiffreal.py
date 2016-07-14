# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:33 2015

@author: Rick Berg, University of Washington, School of Oceanography

Script for modeling reaction rates in marine sediments.

Starts with existing sediment column, 
sediment column building upward modeled as a downward flow of porewater.

Bottom boundary condition set as constant concentration of lowest measured value.

Accounts for external flow and sediment/porewater burial and compaction.

Rates are kept constant over the period of sediment column build and are 
kept constant at specific depths.(Do not follow the sediment packages)

Start with the current profile, after each iteration, calculate the
rate needed to keep the profile there. Then averages those rates and runs the
model forward to get fit.

Final rate output is in mol m-3 (bulk sediment) y-1
(Same as Wang model)

Units used: meters, years, mol m**-3 (aka mM), Celsius. 


[Column, Row]
"""
from pylab import savefig
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from scipy.interpolate import interp1d
from scipy import optimize, integrate
from collections import defaultdict
import matplotlib.gridspec as gridspec
import MySQLdb
import csv
from pandas import DataFrame


#Script = os.path.basename(__file__)
Date = datetime.date.today()

# Site ID
Leg = "'311'"
Site = "'1325'"
Hole = "('C')"
Solute = 'Mg'

# Model parameters
timesteps = 1000  # Number of timesteps
intervals = 51  # Number of intervals
smoothing = 1  # Data points to use for smoothing modelrate profile

# Species parameters
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C
TempD = 18  # Temperature at which diffusion coefficient is known
precision = 0.02  # measurement precision

###############################################################################
# Load files as pandas dataframes

# Import data from database, create files for loading
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp'
conctable = 'iw_100_312'
portable = 'mad_100_312'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)

####### Import data from database #######
# Pore water chemistry data
sql = "SELECT sample_depth, {} FROM {} where leg = {} and site = {} and hole in {} and {} is not null; ".format(Solute, conctable, Leg, Site, Hole, Solute)
concdata = pd.read_sql(sql, con)
concdata = concdata.as_matrix()
ct0 = [concdata[0, 1]]  # mol per m^3 in modern average seawater

# Porosity data
sql = "SELECT sample_depth, porosity FROM {} where leg = {} and site = {} and hole in {} and method like('%C') and {} is not null ;".format(portable, Leg, Site, Hole, 'porosity')
pordata = pd.read_sql(sql, con)
pordata = pordata.as_matrix()

# Sea level data for salinity
sql = "SELECT age, sealevel FROM sealevel"
salinity = pd.read_sql(sql, con)
salinity = salinity.as_matrix()
salinityval = (salinity[:,1]+3900)/3900*34.7
salinity = np.column_stack((salinity[:,0], salinityval))

# Temperature gradient
sql = "SELECT temp_gradient FROM summary_all where leg = {} and site = {} and hole in {} ;".format(Leg, Site, Hole)
temp_gradient = pd.read_sql(sql, con)
temp_gradient = temp_gradient.iloc[0,0]


#temp_gradient = temp_gradient.as_matrix()

# Bottom water temp
sql = "SELECT bottom_water_temp FROM summary_all where leg = {} and site = {} and hole in {} ;".format(Leg, Site, Hole)
bottom_temp = pd.read_sql(sql, con)
bottom_temp = bottom_temp.iloc[0,0]

# Temperature profile (degrees C)
def sedtemp(z, bottom_temp):
    return bottom_temp + np.multiply(z, temp_gradient)

# Advection rate
sql = "SELECT advection_rate FROM summary_all where leg = {} and site = {} and hole in {} ;".format(Leg, Site, Hole)
advection_rate = pd.read_sql(sql, con)

####### Load Files ########

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
concunique = averages(concdata[:, 0], concdata[:, 1])
##### Remove for analytical tests######
concunique = np.concatenate((np.array(([0],ct0)).T, concunique), axis=0)

concprocessed = concunique

# Porosity vector
por = averages(pordata[:, 0], pordata[:, 1])

# Function for running mean for variable window size (window must be odd number)
def runningmean(arr, window):
    prepad = []
    postpad = []
    windowhalf = int((window-1)/2)
    for i in np.arange(windowhalf)+1:
        pre = arr[0] + (arr[0] - arr[i])
        post = arr[-1] + (arr[-1] - arr[-(i+1)])
        prepad.append(pre)
        postpad.append(post)
    prepad = np.flipud(prepad)
    padded = np.append(np.append(prepad, arr), postpad)
    cumsum = np.cumsum(np.insert(padded, 0, 0))
    return (cumsum[window:] - cumsum[:-window])/window

#Function for running mean, taking nearest of window-1 and averaging (window must be odd number)
def nearestmean(arr, window):
    prepad = np.ones((window-1)/2)*np.average(arr[0:window])
    postpad = np.ones((window-1)/2)*np.average(arr[-window:])
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window])/window
    return np.append(np.append(prepad, smoothed), postpad)

###############################################################################
# Concentration data preparation

# Smooth and interpolate the concentration data using 5 point gaussian

# Concentration smoothing 5-pt gaussian (approximate)

prepad = [(concunique[0, 1]+(concunique[0, 1]-concunique[2, 1])), (concunique[0, 1]+(concunique[0, 1]-concunique[1, 1]))]
postpad = [(concunique[-1, 1]+(concunique[-1, 1]-concunique[-2, 1])), (concunique[-1, 1]+(concunique[-1, 1]-concunique[-3, 1]))]
concpad = np.append(np.append(prepad, concunique[:, 1]), postpad)

conplen = concpad.size
befconc2 = concpad[0:conplen-4]
befconc1 = concpad[1:conplen-3]
aftconc1 = concpad[3:conplen-1]
aftconc2 = concpad[4:conplen]

concsmooth = np.column_stack((concunique[:,0], (befconc2*0.06+befconc1*0.24+concunique[:,1]*0.4+aftconc1*0.24+aftconc2*0.06)))
concprocessed = concsmooth

# Make interpolation function
concinterp = interp1d(concprocessed[:, 0], concprocessed[:, 1], kind='linear')
# concfit = np.polyfit(conc[:,0], conc[:,1], 4)
# conclinefit = np.poly1d(concfit)


###############################################################################
# Lower boundary condition (constant set at dirichlet)
cb = concprocessed[-1, 1]

###############################################################################
# Porosity data preparation

porvalues = por[:, 1]
pordepth = por[:, 0]

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

dt = bottomage/timesteps # time step for each interval
sedconctimes = np.append(sedtimes[0:len(sedrates)], bottomage)


# concline = conclinefit(intervaldepths)
###############################################################################
# Plot data for inspection

previewfigs, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 6))

gs = gridspec.GridSpec(1, 5)
gs.update(wspace=0.3)
ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4])

ax1.plot(concdata[:, 1], concdata[:, 0], '--go', label='Measured')
ax2.plot(por[:, 1], por[:, 0], 'mo', label='Measured')
ax2.plot(porcurve(pordepth, porfit), pordepth, 'k-', label='Curve fit', linewidth=3)
ax2.legend(loc='lower right', fontsize='small')
ax1.legend(loc='lower right', fontsize='small')
ax3.plot(sedtemp(np.arange(maxconcdepth), bottom_temp), np.arange(maxconcdepth), 'k-', linewidth=3)
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
savefig("dataprofiles.png")
#plt.waitforbuttonpress()
#plt.close()

###############################################################################
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
# # Bottom water concentration curve vs time at each timestep (upper boundary condition)
ct = np.interp(iterationtimes, salinity[:,0]*10**6, salinity[:,1])/34.7*ct0

###############################################################################

# Reactive transport model

#Sediment properties
porosity = porcurve(intervaldepths, porfit)
tortuosity = 1-np.log(porosity**2)
Dsed = Dst(TempD, sedtemp(intervaldepths, bottom_temp))/tortuosity  # Effective diffusion coefficient

# Pore water burial flux at each time step
deeppor = porcurve(pordepth[-1], porfit)
deepsolid = solidcurve(pordepth[-1], porfit)
pwburialflux = np.zeros([timesteps,1])
for i in np.arange(len(sedmassratetime)):
    pwburial = deeppor*sedmassratetime[i]/deepsolid
    pwburialflux[i] = np.flipud(pwburial)

# Reactive-transport engine
concvalues = concinterp(intervaldepths) # initial conditions (this is where to put approximate profile also)
edgedepths = intervaldepths[1:-1]
edgevalues = concinterp(intervaldepths)
concprofile = []
modelrates = []
intervalvector = np.append(np.append([1.5*intervalthickness], np.ones(intervals-3)*intervalthickness), [1.5*intervalthickness])
for i in np.arange(timesteps):
    flux = porosity[1:-1]*Dsed[1:-1]*(edgevalues[2:] - 2*edgevalues[1:-1] + edgevalues[:-2])/(intervalthickness**2) - (pwburialflux[i] + porosity[0]*advection)*(edgevalues[2:] - edgevalues[:-2])/(2*intervalthickness)      
    edgevalues = edgevalues[1:-1] + flux*dt
    concdiff = concvalues[1:-1] - edgevalues
    modelrate = concdiff/dt
    edgevalues = np.append(np.append([ct[i]], edgevalues + modelrate*dt), [cb])
    modelrates.append(modelrate)
    concprofile.append(edgevalues[1:-1])

modelrate = np.mean(modelrates, axis=0)
modelratefinal = np.mean(modelrates, axis=0)


'''
# Plot to show progression of model run
for i in np.arange(11):
plt.plot(modelrate, intervaldepths[1:-1])
plt.gca().invert_yaxis()
plt.show()
'''
# Grid peclet and courant numbers
gpeclet = np.abs((advection*porosity[0]/porosity[-1]+pwburialflux[0]/porosity[-1])*intervalthickness/np.min(Dsed))
print('Grid Peclet (less than 2):', gpeclet)
courant = np.abs((advection*porosity[0]/porosity[-1]+pwburialflux[0]/porosity[-1])*dt/intervalthickness)
print('Courant (less than 1):', courant)  
#neumann = intervalthickness**2/(3*np.max(Dsed))
#print('Von Neumann (greater than dt):', neumann, 'dt:', dt)

# Run average model rates over every time step
concprofile = []
for i in np.arange(timesteps):
    flux =  porosity[1:-1]*Dsed[1:-1]*(concvalues[2:] - 2*concvalues[1:-1] + concvalues[:-2])/(intervalthickness**2) - (pwburialflux[i] + porosity[0]*advection)*(concvalues[2:] - concvalues[:-2])/(2*intervalthickness)     
    avgvalues = concvalues[1:-1] + flux*dt + modelratefinal*dt
concprofile.append(avgvalues)


modelratesmooth = nearestmean(modelratefinal, smoothing)
integratedrate = sum(modelratesmooth*intervalvector)
print('Integrated rate:', integratedrate)

# R-squared function
def rsq(modeled, measured):
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured)-1)*np.var(measured)
    return 1-sse/sstotal

# Calculate r-squared
rsquared = rsq(concprofile[-1], concvalues[1:-1])
print('r-squared:', rsquared)

###############################################################################
# Plot final data (original concentrations, modeled concentrations, and reaction rates)

previewfigs2, (ax4, ax5) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))

gs2 = gridspec.GridSpec(1, 4)
gs2.update(wspace=0.3)
ax4 = plt.subplot(gs2[0, :2])
ax5 = plt.subplot(gs2[0, 2:4], sharey=ax4)
ax5.grid()

# ax4.plot(noreaction, midpoints)
ax4.plot(concprofile[-1], intervaldepths[1:-1], 'r+', label='Model')
ax4.plot(concdata[:,1], concdata[:,0], 'go', label='Measured', mfc='None')
ax4.plot(concvalues, intervaldepths, '-', label='Line fit', mfc='None')
ax5.plot(modelratefinal*1000000, intervaldepths[1:-1])
ax5.plot(modelratesmooth*1000000, intervaldepths[1:-1])
ax4.legend(loc='lower right', fontsize='small')
ax4.set_ylabel('Depth (mbsf)')
ax4.set_xlabel('Concentration (mM)')
ax5.set_xlabel('Reaction Rate x 10^-6')
ax4.locator_params(axis='x', nbins=4)
ax5.locator_params(axis='x', nbins=4)
ax4.invert_yaxis()
savefig("rateprofile.png")


# Save data in csv files
metadata = {'Precision': precision, 'Grid Peclet #': gpeclet, 'Courant #': courant, 'Leg': Leg, 'Site': Site, 'Solute': Solute, 'Timesteps': timesteps, 'Number of Intervals': intervals, 'Smoothing Window': smoothing, 'Ds': Ds, 'Ds reference temp': TempD, 'Integrated Rate': integratedrate, 'R-squared': rsquared, 'Script': Script, 'Date': Date}
metadatadf = Series(metadata)
metadatadf.to_csv("metadata.csv")

modelrxnrates = np.column_stack((modelratesmooth, intervaldepths[1:-1]))
modelporosity = np.column_stack((intervaldepths, porosity))
np.savetxt("modelporosity.csv", modelporosity, delimiter="\t")
np.savetxt("modelrates.csv", modelrxnrates, delimiter="\t")

# eof
