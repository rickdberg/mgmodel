# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:31:33 2015

@author: Rick Berg, University of Washington, School of Oceanography

Script for modeling reaction rates in marine sediments.

Starts with existing sediment column, sediment column building upward is 
modeled as a downward flow of porewater.

Bottom boundary condition set as constant concentration of lowest measured 
value.
Upper boundary condition set as average seawater value adjusted for 
changing salinity based on sealevel records.

Accounts for external flow and sediment/porewater burial and compaction.

Rates are kept constant over the period of sediment column build and are 
kept constant at specific depths relative to seafloor.(Do not follow the 
sediment packages)

Start with the current profile, after each iteration, calculate the
rate needed to keep the profile there. It then averages those rates and runs 
the model forward to get the fit.

Final rate output is in mol m-3 (bulk sediment) y-1
(Same as Wang model)

Units: meters, years, mol m**-3 (aka mM), Celsius.
Positive values of reaction rate indicate uptake into sediment.

[Column, Row]


run model on each isotope

save rxn rate profile

calculate epsilon values 26/24

save epsilon profiles


Change 25/24std ratio in calculation


"""

import numpy as np
import pandas as pd
import scipy as sp
import MySQLdb
import datetime
import os
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
from scipy import optimize, integrate
from collections import defaultdict
import matplotlib.gridspec as gridspec
from pylab import savefig

Script = os.path.basename(__file__)
Date = datetime.datetime.now()

# Site ID
Leg = '315'
Site = 'C0002'
Holes = "('B', 'D', 'H', 'J', 'K', 'L', 'M', 'P')"
Bottom_boundary = 'none' # 'none', or a depth
Hole = ''.join(filter(str.isalpha, Holes))

# Model parameters
timesteps = 1000  # Number of timesteps
smoothing = 1  # Window to use for smoothing modelrate profile (Running mean - must be odd number)

# Species parameters
Solute = 'Mg'
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)
TempD = 18  # Temperature at which diffusion coefficient is known
precision = 0.02  # measurement precision

###############################################################################
###############################################################################
###############################################################################
# Load data from database

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
conctable = 'iw_all'
portable = 'mad_all'
isotopetable = 'mg_isotopes'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)

# Pore water chemistry data
sql = """SELECT sample_depth, {} FROM {} where leg = '{}' and site = '{}' and hole in {} and {} is not null and hydrate_affected is null; """.format(Solute, conctable, Leg, Site, Holes, Solute)
concdata = pd.read_sql(sql, con)
concdata = concdata.sort_values(by='sample_depth')
concdata = concdata.as_matrix()
if Bottom_boundary == 'none':
    concdata = concdata
else:
    deepest_iw_idx = np.searchsorted(concdata[:,0], Bottom_boundary)
    concdata = concdata[:deepest_iw_idx, :]

ct0 = [54.0]  # [concdata[0, 1]]  # mol per m^3 in modern average seawater at specific site

# Porosity data
sql = """SELECT sample_depth, porosity FROM {} where leg = '{}' and site = '{}' and hole in {} and coalesce(method,'C') like '%C%'  and {} is not null ;""".format(portable, Leg, Site, Holes, 'porosity')
pordata = pd.read_sql(sql, con)
pordata = pordata.as_matrix()

# Sea level data for salinity
sql = "SELECT age, sealevel FROM sealevel"
salinity = pd.read_sql(sql, con)
salinity = salinity.as_matrix()
salinityval = (salinity[:,1]+3900)/3900*34.7 # Modern day avg salinity 34.7 from (ref), depth of ocean from (ref)
salinity = np.column_stack((salinity[:,0], salinityval)) # Avg salinity vs time

# Temperature gradient (degrees C/m)
sql = """SELECT temp_gradient FROM site_info where leg = '{}' and site = '{}';""".format(Leg, Site)
temp_gradient = pd.read_sql(sql, con)
temp_gradient = temp_gradient.iloc[0,0]

# Bottom water temp (degrees C)
sql = """SELECT bottom_water_temp FROM site_info where leg = '{}' and site = '{}';""".format(Leg, Site)
bottom_temp = pd.read_sql(sql, con)
bottom_temp = bottom_temp.iloc[0,0]

# Temperature profile (degrees C)
def sedtemp(z, bottom_temp):
    return bottom_temp + np.multiply(z, temp_gradient)

# Advection rate (m/y)
sql = """SELECT advection_rate FROM site_info where leg = '{}' and site = '{}';""".format(Leg, Site)
advection = pd.read_sql(sql, con)
advection = advection.iloc[0,0]

# Sedimentation rate profile (m/y)
# Get sedimentation rates from database
sql = """SELECT sedrate_ages, sedrate_depths FROM metadata_sed_rate where leg = '{}' and site = '{}' ; """.format(Leg, Site)
sedratedata = pd.read_sql(sql, con)
sedratedata = sedratedata.sort_values(by='sedrate_depths')

sedtimes = np.asarray(sedratedata.iloc[:,0][0][1:-1].split(","))
seddepths = np.asarray(sedratedata.iloc[:,1][0][1:-1].split(","))
sedtimes = sedtimes.astype(np.float)
seddepths = seddepths.astype(np.float)
sedrates = np.diff(seddepths, axis=0)/np.diff(sedtimes, axis=0)  # m/y

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

# Concentration vector after averaging duplicates
concunique = averages(concdata[:, 0], concdata[:, 1])
##### Remove for analytical tests ###### Adds value ct0 at seafloor
concunique = np.concatenate((np.array(([0],ct0)).T, concunique), axis=0)

# Porosity vector
por = averages(pordata[:, 0], pordata[:, 1])

# Function for running mean for variable window size 
# Window must be odd number
# Negates furthest n points while window hangs off edge by n points
def runningmean(arr, window):
    prepad = []
    postpad = []
    windowhalf = int((window-1)/2)
    for i in np.arange(windowhalf)+1:
        pre = arr[0] + (arr[0] - arr[i])
        post = arr[-1] + (arr[-1] - arr[-(i+1)])
        prepad.append(pre)
        postpad.append(post)
    postpad = np.flipud(postpad)
    padded = np.append(np.append(prepad, arr), postpad)
    cumsum = np.cumsum(np.insert(padded, 0, 0))
    return (cumsum[window:] - cumsum[:-window])/window
    
###############################################################################
# Mg isotope data, combined with corresponding Mg concentrations

sql = """select mg_isotopes.sample_depth, mg_isotopes.d26mg, mg_isotopes.d25mg
from mg_isotopes
where mg_isotopes.leg = '{}' and mg_isotopes.site = '{}' and mg_isotopes.hole in {} and d26mg is not NULL;""".format(Leg, Site, Holes)
isotopedata = pd.read_sql(sql, con)
isotopedata = isotopedata.sort_values(by='sample_depth')
isotopedata = pd.merge(isotopedata.iloc[:,0:3], pd.DataFrame(concunique, columns = ['sample_depth', 'mg_conc']), how = 'inner', on = 'sample_depth')
isotopedata = isotopedata.as_matrix()

d26sw = [-0.82]  # d26Mg in modern average seawater
d25sw = [-0.39]  # d25Mg in modern average seawater

# Add seawater values (upper boundary condition, dirichlet) to profiles
isotopedata = np.concatenate((np.array(([0],d26sw, d25sw, ct0)).T, isotopedata), axis=0)

datapoints = len(isotopedata) # Used to insert into metadata and set number of intervals
def round_down_to_even(f):
     return math.floor(f / 2.) * 2
intervals = round_down_to_even(datapoints)  # Number of intervals

# Calculate Mg isotope concentrations
# Source for 26/24std: Isotope Geochemistry, William White, pp.365

mg26_24 = ((isotopedata[:,1]/1000)+1)*0.13979
mg25_24 = ((isotopedata[:,2]/1000)+1)*0.126598
mg24conc = isotopedata[:,3]/(mg26_24+mg25_24+1)
mg25conc = mg24conc*mg25_24
mg26conc = mg24conc*mg26_24

###############################################################################
# Isotope data preparation


# Add lower boundary condition (constant, set at lowest measurement, dirichlet)
cb_26 = isotopedata[-1, 1]
cb_25 = isotopedata[-1, 2]

# Make interpolation function for individual isotope concentrations
concinterp_d26 = interp1d(isotopedata[:,0], mg26conc, kind='linear')
concinterp_d25 = interp1d(isotopedata[:,0], mg25conc, kind='linear')

###############################################################################
# Concentration data preparation
'''
# Smooth and interpolate the concentration data using 5 point gaussian

# Concentration smoothing 5-pt gaussian (approximate), reflected at edges

prepad = [(concunique[0, 1]+(concunique[0, 1]-concunique[2, 1])), (concunique[0, 1]+(concunique[0, 1]-concunique[1, 1]))]
postpad = [(concunique[-1, 1]+(concunique[-1, 1]-concunique[-2, 1])), (concunique[-1, 1]+(concunique[-1, 1]-concunique[-3, 1]))]
concpad = np.append(np.append(prepad, concunique[:, 1]), postpad)

conplen = concpad.size
befconc2 = concpad[0:conplen-4]
befconc1 = concpad[1:conplen-3]
aftconc1 = concpad[3:conplen-1]
aftconc2 = concpad[4:conplen]

concsmooth = np.column_stack((concunique[:,0], (befconc2*0.06+befconc1*0.24+concunique[:,1]*0.4+aftconc1*0.24+aftconc2*0.06)))
'''
# Smooth concentrations using 1-std-deviation gaussian using reflect at edges
concsmooth = np.column_stack((concunique[:,0], (ndimage.filters.gaussian_filter1d(concunique[:,1], 1, axis=0, mode='reflect'))))

concprocessed = concsmooth  #Seawater value added, duplicates averaged, smoothed data

# Make interpolation function for concentrations
concinterp = interp1d(concprocessed[:, 0], concprocessed[:, 1], kind='linear')
# concfit = np.polyfit(conc[:,0], conc[:,1], 4)
# conclinefit = np.poly1d(concfit)


###############################################################################
# Lower boundary condition (constant, set at dirichlet)
cb = concprocessed[-1, 1]  # Same as concunique[-1, 1]

###############################################################################
# Porosity data preparation

porvalues = por[:, 1]
pordepth = por[:, 0]

# Porosity curve fit (ref?) (Makes porosity at sed surface equal to first measurement)
def porcurve(z, a):
    portop = np.max(por[:3, 1])
    porbottom = por[-1, 1]
    return (portop-porbottom) * np.exp(-a*z) + porbottom

porfit, porcov = optimize.curve_fit(porcurve, pordepth, porvalues)

# Solids curve fit (based on porosity curve fit function)
def solidcurve(z, a):
    portop = por[0, 1]
    porbottom = por[-1, 1]
    return 1-((portop-porbottom) * np.exp(-a*z) + porbottom)

###############################################################################
# Diffusion coefficient function

# Calculates viscosity from Mostafa H. Sharqawy 12-18-2009, MIT (mhamed@mit.edu) Sharqawy M. H., Lienhard J. H., and Zubair, S. M., Desalination and Water Treatment, 2009
# Viscosity used as input into Stokes-Einstein equation
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
# Calculate ages at depth intervals and get age at lowest pore water measurement


# Function for appending float to array and trimming the sorted array to have 
# that value at end of the array
def insertandcut(value, array):
    findindex = array.searchsorted(value)
    sortedarray = np.insert(array, findindex, value)[:findindex+1]
    return sortedarray

# Make bottom depth of deepest pore water concentration measurement the end 
# of age-depth array
maxconcdepth = np.max(concdata[:,0])
sedconcdepths = insertandcut(maxconcdepth, seddepths)

# Split sediment column into intervals of differing mass (due to porosity 
# change) but same thickness
intervalthickness = maxconcdepth/intervals
intervaldepths = (np.arange(intervals+1)*intervalthickness)  # Depths at bottom of intervals
midpoints = np.add(intervaldepths[:-1], np.diff(intervaldepths)/2)

# Mass (1-dimensional volume of solids) of each model interval based on porosity curve
runningmass = []
for i in intervaldepths:
    runningmass.append(integrate.quad(solidcurve, 0, i, args=(porfit))[0])
intervalmass = np.diff(runningmass)
columnmass = np.cumsum(intervalmass)

# Sediment mass (1-dimensional volume of solids) accumulation rates for each age-depth section
# Assumes constant sediment mass (really volume of solids) accumulation rates between age-depth measurements
sectionmasses = [0]
sedmassrates = np.zeros(len(sedrates))  # unique avg sed mass accumulation rates to bottom of conc profile
for i in range(len(sedrates)):
    sectionmass = (integrate.quad(solidcurve, seddepths[i], seddepths[i+1], args=(porfit)))[0]
    sedmassrate = (sectionmass/np.diff(sedtimes)[i])
    sedmassrates[i] = sedmassrate
    sectionmasses.append(sectionmass)
sectionmasses = np.array(sectionmasses)

# Interval ages and bottom age (at bottom of concentration profile)
# Can be slightly off where model sediment interval straddles age-depth datapoint depth, uses deeper (older) sed rate in that case
####### Could be minimized by using midpoint as marker for sed rate call #######
midpointages = []
intervalages = [] 
for i in range(len(seddepths)-1):
    for n in np.arange(len(intervaldepths)):
        if intervaldepths[n] >= seddepths[i] and intervaldepths[n] < seddepths[i+1]:
            intervalage = sedtimes[i]+(integrate.quad(solidcurve, sedconcdepths[i], intervaldepths[n], args=(porfit)))[0]/sedmassrates[i]
            intervalages.append(intervalage)
    for m in range(len(midpoints)):
        if midpoints[m] >= seddepths[i] and midpoints[m] < seddepths[i+1]:
            midpointage = sedtimes[i]+(integrate.quad(solidcurve, sedconcdepths[i], midpoints[m], args=(porfit)))[0]/sedmassrates[i]
            midpointages.append(midpointage)

# In case deepest pore water measurement is deeper than deepest age-depth 
# measurement, uses deepest age-depth measurement for rest of depth
for n in range(len(intervaldepths)):
    if intervaldepths[n] >= seddepths[-1]:
        intervalage = sedtimes[-1]+(integrate.quad(solidcurve, seddepths[-1], intervaldepths[n], args=(porfit)))[0]/sedmassrates[-1]
        intervalages.append(intervalage)
for m in range(len(midpoints)):    
    if midpoints[m] >= seddepths[-1]:
        midpointage = sedtimes[-1]+(integrate.quad(solidcurve, seddepths[-1], midpoints[m], args=(porfit)))[0]/sedmassrates[-1]
        midpointages.append(midpointage)

midpointages = np.array(midpointages)
intervalages = np.array(intervalages[1:])
bottomage = intervalages[-1]

dt = bottomage/timesteps # time step for each interval
sedconctimes = np.append(sedtimes[0:len(sedrates)], bottomage)

###############################################################################
# Plot data for inspection

previewfigs, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 6))

gs = gridspec.GridSpec(1, 7)
gs.update(wspace=0.5)
ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4])
ax4 = plt.subplot(gs[0, 5:7])

ax1.plot(concdata[:, 1], concdata[:, 0], '--go', label='Measured')
ax2.plot(por[:, 1], por[:, 0], 'mo', label='Measured')
ax2.plot(porcurve(pordepth, porfit), pordepth, 'k-', label='Curve fit', linewidth=3)
ax2.legend(loc='best', fontsize='small')
ax1.legend(loc='best', fontsize='small')
ax3.plot(sedtemp(np.arange(maxconcdepth), bottom_temp), np.arange(maxconcdepth), 'k-', linewidth=3)
ax4.plot(sed[:,1]/1000000, sed[:,0], 'ko', label='Picks')
ax4.plot(sedtimes/1000000, seddepths, 'b-', label='Curve fit', linewidth=2)
ax4.legend(loc='best', fontsize='small')
ax1.set_ylabel('Depth (mbsf)')
ax1.set_xlabel('Concentration (mM)')
ax2.set_xlabel('Porosity')
ax3.set_xlabel('Temperature (\u00b0C)')
ax4.set_xlabel('Age (Ma)')
ax1.locator_params(axis='x', nbins=4)
ax2.locator_params(axis='x', nbins=4)
ax3.locator_params(axis='x', nbins=4)
ax4.locator_params(axis='x', nbins=4)
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()
savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output data figures\dataprofiles_{}_{}.png".format(Leg, Site))

###############################################################################
# Sedimentation rate at each time step
# Uses sed rate at end of timestep in cases where timestep straddles age-depth measurement
#### Can write function to fix this
iterationtimes = ((np.arange(timesteps+1)*dt)[1:]).tolist()
sedmassratetime = []
for i in range(len(sedconctimes)-1):
    if i == 0:
        sedmassratet = np.ones(iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i+1]))) + 1 - iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i])))) * sedmassrates[i]
    else:
        sedmassratet = np.ones(iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i+1])))-iterationtimes.index(min(iterationtimes, key=lambda x:abs(x-sedconctimes[i])))) * sedmassrates[i]
    sedmassratetime = sedmassratetime + sedmassratet.tolist()
sedmassratetime = np.array(sedmassratetime)

###############################################################################
# Bottom water concentration curve vs time at each timestep 
# (upper boundary condition)
ct = np.interp(iterationtimes, salinity[:,0]*10**6, salinity[:,1])/34.7*ct0
ct = np.flipud(ct)

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

# Reactive-transport engine (central difference formula)
concvalues = concinterp(intervaldepths) # initial conditions
edgevalues = concvalues
concprofile = []
modelrates = []
intervalvector = np.append(np.append([1.5*intervalthickness], np.ones(intervals-3) * intervalthickness), [1.5*intervalthickness])
for i in range(timesteps):
    flux = porosity[1:-1] * Dsed[1:-1] * (edgevalues[2:] - 2*edgevalues[1:-1] 
    + edgevalues[:-2]) / (intervalthickness**2) - (pwburialflux[i] 
    + porosity[0] * advection) * (edgevalues[2:] - edgevalues[:-2]) / (2*intervalthickness)      
    next_edgevalues = edgevalues[1:-1] + flux*dt
    concdiff = concvalues[1:-1] - next_edgevalues
    modelrate = concdiff/dt
    edgevalues = np.append(np.append([ct[i]], next_edgevalues + modelrate*dt), [cb])
    modelrates.append(modelrate)
    concprofile.append(edgevalues[1:-1])

modelrate_avg = np.mean(modelrates, axis=0)
modelrate_modern = modelrates[-1]

'''
# Plot to show progression of model run
for i in np.arange(11):
plt.plot(modelrate, intervaldepths[1:-1])
plt.gca().invert_yaxis()
plt.show()
'''
# Grid peclet and Courant numbers
gpeclet = np.abs((advection*porosity[0]/porosity[-1]+pwburialflux[0]/porosity[-1])*intervalthickness/np.min(Dsed))[0]
print('Grid Peclet (less than 2):', gpeclet)
courant = np.abs((advection*porosity[0]/porosity[-1]+pwburialflux[0]/porosity[-1])*dt/intervalthickness)[0]
print('Courant (less than 1):', courant)  
#neumann = intervalthickness**2/(3*np.max(Dsed))
#print('Von Neumann (greater than dt):', neumann, 'dt:', dt)

'''# Run average model rates forward over every time step
concprofile = []
avgvalues = concvalues
for i in np.arange(timesteps):
    flux =  porosity[1:-1]*Dsed[1:-1]*(avgvalues[2:] - 2*avgvalues[1:-1] 
    + avgvalues[:-2])/(intervalthickness**2) - (pwburialflux[i] + porosity[0]
    *advection)*(avgvalues[2:] - avgvalues[:-2])/(2*intervalthickness)     
    
    next_avgvalues = avgvalues[1:-1] + flux*dt + modelrate_avg*dt
    avgvalues = np.append(np.append([ct[i]], next_avgvalues), [cb])
    concprofile.append(avgvalues)
'''

#### Smoothing set to one right now, not used, available if needed
modelratesmooth = runningmean(modelrate_modern, smoothing)
integratedrate = sum(modelratesmooth*intervalvector)
print('Integrated rate:', integratedrate)

# R-squared function
def rsq(modeled, measured):
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured)-1) * np.var(measured)
    return 1-sse/sstotal

# Calculate r-squared between original data and modeled data
rsquared = rsq(concprocessed[:,1], concunique[:,1])
print('r-squared:', rsquared)

###############################################################################
# Plot final data (original concentrations, modeled concentrations, and reaction rates)

previewfigs2, (ax4, ax5) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))

gs2 = gridspec.GridSpec(1, 4)
gs2.update(wspace=0.3)
ax4 = plt.subplot(gs2[0, :2])
ax4.grid()
ax5 = plt.subplot(gs2[0, 2:4], sharey=ax4)
ax5.grid()

ax4.plot(concprofile[-1], intervaldepths[1:-1], '-', label='Model fit')
ax4.plot(concdata[:,1], concdata[:,0], 'go', label='Measured', mfc='None')
ax5.plot(modelrate_avg*1000000, intervaldepths[1:-1], 'k-', label='Average', mfc='None')
# ax5.plot(modelratesmooth*1000000, intervaldepths[1:-1])
ax5.plot(modelrate_modern*1000000, intervaldepths[1:-1], 'b-', label='Modern', mfc='None')
ax4.legend(loc='best', fontsize='small')
ax5.legend(loc='best', fontsize='small')
ax4.set_ylabel('Depth (mbsf)')
ax4.set_xlabel('Concentration (mM)')
ax5.set_xlabel('Reaction Rate x 10^-6')
ax4.locator_params(axis='x', nbins=4)
ax5.locator_params(axis='x', nbins=4)
ax4.invert_yaxis()
savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output rate figures\rateprofile_{}_{}.png".format(Leg, Site))


# Save reaction rate and porosity data in csv files
modelrate_modern_isotopes = np.column_stack((intervaldepths[1:-1], modelrate_modern_24, modelrate_modern_25, modelrate_modern_26))
full_modelrates_isotopes = modelrates
modelporosity = np.column_stack((intervaldepths, porosity))
np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output porosity data\modelporosity_{}_{}.csv".format(Leg, Site), modelporosity, delimiter="\t")
np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output rate data\avg_modelrates_{}_{}.csv".format(Leg, Site), avg_modelrates_isotopes, delimiter="\t")
np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output rate data\full_modelrates_{}_{}.csv".format(Leg, Site), full_modelrates_isotopes, delimiter="\t")

# eof
