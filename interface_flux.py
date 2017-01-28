# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017

@author: rickdberg

Script to calculate sediment-water interface fluxes of a dissolved constituent

Uses central difference formula of a spline fit to the upper sediment section

Must run age-depth.py for site first


How to deal with tortuosity and Dsed change w depth?
Add in plotting of input data, processed data, and results


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MySQLdb
import os
import datetime
from scipy import interpolate
from collections import defaultdict
from scipy import optimize, integrate
import matplotlib.gridspec as gridspec



Script = os.path.basename(__file__)
Date = datetime.datetime.now()


# Site ID
Leg = '315'
Site = 'C0002'
Holes = "('B', 'D', 'H', 'J', 'K', 'L', 'M', 'P')"
Advection = 0  # External (hydrothermal) advection at surface
Hole = ''.join(filter(str.isalpha, Holes))  # Formatting for saving in metadata

# Species parameters
Solute = 'Mg'
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)
TempD = 18  # Temperature at which diffusion coefficient is known
Precision = 0.02  # measurement precision
Ocean = 54.0  # Concentration in modern ocean (mM)

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

ct0 = [Ocean]  # mol per m^3 in modern average seawater at specific site

# Porosity data
sql = """SELECT sample_depth, porosity FROM {} where leg = '{}' and site = '{}' and hole in {} and coalesce(method,'C') like '%C%'  and {} is not null ;""".format(portable, Leg, Site, Holes, 'porosity')
pordata = pd.read_sql(sql, con)
pordata = pordata.as_matrix()

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

# Sedimentation rate profile (m/y) (Calculated from age_depth.py)
sql = """SELECT sedrate_ages, sedrate_depths FROM metadata_sed_rate where leg = '{}' and site = '{}' ; """.format(Leg, Site)
sedratedata = pd.read_sql(sql, con)
sedratedata = sedratedata.sort_values(by='sedrate_depths')

sedtimes = np.asarray(sedratedata.iloc[:,0][0][1:-1].split(","))
seddepths = np.asarray(sedratedata.iloc[:,1][0][1:-1].split(","))
sedtimes = sedtimes.astype(np.float)
seddepths = seddepths.astype(np.float)
sedrates = np.diff(seddepths, axis=0)/np.diff(sedtimes, axis=0)  # m/y
sedrate = sedrates[0]

# Load age-depth data for plots  (From age_depth.py)
sql = """SELECT depth, age FROM age_depth where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
picks = pd.read_sql(sql, con)
picks = picks.as_matrix()
picks = picks[np.argsort(picks[:,0])]

###############################################################################
# Average duplicates in concentration dataset, add seawater value, and make spline fit to first three values

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
concunique = np.concatenate((np.array(([0],ct0)).T, concunique), axis=0)

# Spline fit to first 4 unique concentration values
spline_function = interpolate.splrep(concunique[:4,0], concunique[:4,1], k=2)
intervalthickness = 1  # meters
conc_interp_depths = np.arange(0,3,intervalthickness)  # Zero to 2 meters
conc_interp = interpolate.splev(conc_interp_depths, spline_function)
conc_interp_plot = interpolate.splev(np.linspace(concunique[0,0], concunique[3,0], num=50), spline_function)

###############################################################################
# Porosity and solids fraction functions and data preparation

# Porosity vectors
por = averages(pordata[:, 0], pordata[:, 1])  # Average duplicates
porvalues = por[:, 1]
pordepth = por[:, 0]

# Porosity curve fit (ref?) (Makes porosity at sed surface equal to average of first 3 measurements)
def porcurve(z, a):
    portop = np.max(por[:3, 1])  # Averages top 3 porosity measurements for upper porosity boundary
    porbottom = por[-1, 1]  # Takes lowest porosity measurement as the lower boundary
    return (portop-porbottom) * np.exp(-a*z) + porbottom

porfit, porcov = optimize.curve_fit(porcurve, pordepth, porvalues)

#Sediment properties
porosity = porcurve(conc_interp_depths, porfit)
tortuosity = 1-np.log(porosity**2)

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

Dsed = Dst(TempD, bottom_temp)/tortuosity[0]  # Effective diffusion coefficient

###############################################################################
# Pore water burial mass flux

# Sediment mass (1-dimensional volume of solids) accumulation rates for each age-depth section
# Assumes constant sediment mass (really volume of solids) accumulation rates between age-depth measurements
sectionmass = (integrate.quad(solidcurve, seddepths[0], seddepths[1], args=(porfit)))[0]
sedmassrate = (sectionmass/np.diff(sedtimes)[0])

# Pore water burial flux calculation (ref?)
deeppor = porcurve(pordepth[-1], porfit)
deepsolid = solidcurve(pordepth[-1], porfit)
pwburialflux = deeppor*sedmassrate/deepsolid

###############################################################################
# Flux model # Needs work

gradient = (-3*conc_interp[0] + 4*conc_interp[1] - conc_interp[2])/(2*intervalthickness)

flux = porcurve(pordepth[0], porfit) * Dsed * gradient + (porcurve(pordepth[0], porfit) * Advection + pwburialflux) * conc_interp[0]
print('Flux (mol/m^2 y^-1):', flux)

figure_1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 8, figsize = (15, 8))
grid = gridspec.GridSpec(3, 8, wspace=0.7, hspace=0.5)

ax1 = plt.subplot(grid[1:3, :2])
ax1.grid()
ax2 = plt.subplot(grid[1:3, 2:4], sharey=ax1)
ax2.grid()
ax3 = plt.subplot(grid[1:3, 4:6], sharey=ax1)
ax3.grid()
ax4 = plt.subplot(grid[1:3, 6:8], sharey=ax1)
ax4.grid()
ax5 = plt.subplot(grid[0, 0:2])
ax5.grid()
ax6 = plt.subplot(grid[0, 2:4], sharey=ax5)
ax6.grid()
ax7 = plt.subplot(grid[0, 4:6], sharey=ax5)
ax7.grid()
ax8 = plt.subplot(grid[0, 6:8], sharey=ax5)
ax8.grid()


ax1.plot(concunique[:,1], concunique[:,0], 'go', label='Measured')
ax1.plot(concunique[0:4,1], concunique[0:4,0], 'bo', label="Fit points")
ax2.plot(por[:, 1], por[:, 0], 'mo', label='Measured')
ax2.plot(porcurve(pordepth, porfit), pordepth, 'k-', label='Curve fit', linewidth=3)
ax3.plot(sedtemp(np.arange(concunique[-1,0]), bottom_temp), np.arange(concunique[-1,0]), 'k-', linewidth=3)
ax4.plot(picks[:,1]/1000000, picks[:,0], 'ro', label='Picks')
ax4.plot(sedtimes/1000000, seddepths, 'k-', label='Curve fit', linewidth=2)
ax5.plot(concunique[:7,1], concunique[:7,0], 'go', label='Measured')
ax5.plot(concunique[:4,1], concunique[:4,0], 'bo', label='Measured')
ax5.plot(conc_interp_plot, np.linspace(concunique[0,0], concunique[3,0], num=50), 'k-', label='Interpolated')
ax6.plot(por[:len(por[:, 0].clip([0,concunique[6,0]])), 1], por[:, 0].clip([0,concunique[6,0]]), 'mo', label='Measured')
ax6.plot(porcurve(pordepth, porfit), pordepth.clip([0,concunique[6,0]]), 'k-', label='Curve fit', linewidth=3)
ax7.plot(sedtemp(np.arange(concunique[-1,0]), bottom_temp), np.arange(concunique[6,0]), 'k-', linewidth=3)
ax8.plot(picks[:,1]/1000000, picks[:,0].clip([0,concunique[6,0]]), 'ro', label='Picks')
ax8.plot(sedtimes/1000000, seddepths.clip([0,concunique[6,0]]), 'k-', label='Curve fit', linewidth=2)

ax1.legend(loc='best', fontsize='small')
ax2.legend(loc='best', fontsize='small')
ax4.legend(loc='best', fontsize='small')
ax1.set_ylabel('Depth (mbsf)')
ax1.set_xlabel('Concentration (mM)')
ax2.set_xlabel('Porosity')
ax3.set_xlabel('Temperature (\u00b0C)')
ax4.set_xlabel('Age (Ma)')
ax5.set_ylabel('Depth (mbsf)')
ax1.locator_params(axis='x', nbins=4)
ax2.locator_params(axis='x', nbins=4)
ax3.locator_params(axis='x', nbins=4)
ax4.locator_params(axis='x', nbins=4)
ax5.locator_params(axis='x', nbins=4)
ax6.locator_params(axis='x', nbins=4)
ax7.locator_params(axis='x', nbins=4)
ax8.locator_params(axis='x', nbins=4)
ax1.invert_yaxis()
ax5.set_ylim([0,concunique[6,0]])
ax5.invert_yaxis()





# eof
