# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017

@author: rickdberg

Module to calculate sediment-water interface fluxes of a dissolved constituent
Uses an exponential fit to the upper sediment section for concentration gradient
Positive flux is downward (into the sediment)

"""
import numpy as np
import pandas as pd
from scipy import optimize, integrate
import MySQLdb
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pylab import savefig
from collections import defaultdict
from site_metadata_compiler import metadata_compiler
import seawater


Script = os.path.basename(__file__)
Date = datetime.datetime.now()

# Site Information
Leg = '172'
Site = '1062'
Holes = "('A','B') or hole is null"
Hole = ''.join(filter(str.isupper, filter(str.isalpha, Holes)))  # Formatting for saving in metadata
Comments = ''
Complete = ''

# Species parameters
Solute = 'Mg'  # Change to Mg_ic if needed, based on what's available in database
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)
TempD = 18  # Temperature at which diffusion coefficient is known
Precision = 0.02  # measurement precision
Ocean = 54  # Concentration in modern ocean (mM)
Solute_db = 'Mg' # Label for the database loading

# Model parameters
dp = 20 # Number of concentration datapoints to use for exponential curve fit
z = 0  # Depth (meters) at which to calculate flux

###############################################################################
###############################################################################
###############################################################################
# Load data from database

# Connect to database
database = "mysql://root:neogene227@localhost/iodp_compiled"
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
conctable = 'iw_all'
portable = 'mad_all'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cursor = con.cursor()
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Get site metadata
site_metadata = metadata_compiler(database, metadata, site_info, hole_info, Leg, Site)

# Temperature gradient (degrees C/m)
temp_gradient = site_metadata['temp_gradient'][0]

# Water depth for bottom water estimation
water_depth = site_metadata['water_depth'][0]

# Bottom water temp (degrees C)
if site_metadata['bottom_water_temp'].isnull().all():
    bottom_temp = site_metadata['woa_bottom_temp'][0]
    if water_depth >= 1500:
        bottom_temp_est = 'deep'
    else:
        bottom_temp_est = 'shallow'
else:
    bottom_temp = site_metadata['bottom_water_temp'][0]
    bottom_temp_est = 'na'

# Temperature profile (degrees C)
def sedtemp(z, bottom_temp):
    if z.all() == 0:
        return bottom_temp*np.ones(len(z))
    else:
        return bottom_temp + np.multiply(z, temp_gradient)

# Advection rate (m/y)
advection = float(site_metadata['advection'][0])
if pd.isnull(advection):
    advection = 0


# Pore water chemistry data
sql = """SELECT sample_depth, {}
FROM {}
where leg = '{}' and site = '{}' and (hole in {})
and {} is not null and hydrate_affected is null; """.format(Solute, conctable, Leg, Site, Holes, Solute)
concdata = pd.read_sql(sql, con)
concdata = concdata.sort_values(by='sample_depth')
concdata = concdata.as_matrix()

# Bottom water concentration of Mg based on WOA bottom salinities.
woa_salinity = site_metadata['woa_bottom_salinity'][0]
density = seawater.eos80.dens0(woa_salinity, bottom_temp)
def sal_to_cl(salinity, density):
    return (1000*(salinity-0.03)*density/1000)/(1.805*35.45)
woa_cl = sal_to_cl(woa_salinity, density)
bottom_conc=woa_cl/558*Ocean
ct0 = [bottom_conc]  # mol per m^3 in modern seawater at specific site

# Porosity data
sql = """SELECT sample_depth, porosity
FROM {}
where leg = '{}' and site = '{}' and (hole in {})
and coalesce(method,'C') like '%C%'
and {} is not null and {} > 0 and {} < 1 AND sample_depth is not null;""".format(portable, Leg, Site, Holes, 'porosity', 'porosity', 'porosity')
pordata = pd.read_sql(sql, con)
if pordata.iloc[:,1].count() < 40:  # If not enough data from method C, take all data
    sql = """SELECT sample_depth, porosity
    FROM {}
    where leg = '{}' and site = '{}' and (hole in {})
    and {} is not null and {} > 0 AND {} < 1 and sample_depth is not null;""".format(portable, Leg, Site, Holes, 'porosity', 'porosity', 'porosity')
    pordata_b = pd.read_sql(sql, con)
    pordata = pd.concat((pordata, pordata_b), axis=0)
pordata = pordata.as_matrix()


# Sedimentation rate profile (m/y) (Calculated in age_depth.py)
sql = """SELECT sedrate_ages, sedrate_depths
FROM metadata_sed_rate
where leg = '{}' and site = '{}' ; """.format(Leg, Site)
sedratedata = pd.read_sql(sql, con)
sedratedata = sedratedata.sort_values(by='sedrate_depths')

sedtimes = np.asarray(sedratedata.iloc[:,0][0][1:-1].split(","))
seddepths = np.asarray(sedratedata.iloc[:,1][0][1:-1].split(","))
sedtimes = sedtimes.astype(np.float)
seddepths = seddepths.astype(np.float)
sedrates = np.diff(seddepths, axis=0)/np.diff(sedtimes, axis=0)  # m/y
sedrate = sedrates[0]  # Using modern sedimentation rate
print('Modern sed rate (cm/ky):', np.round(sedrate*100000, decimals=3))

# Load age-depth data for plots
sql = """SELECT depth, age
FROM age_depth
where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
picks = pd.read_sql(sql, con)
picks = picks.as_matrix()
picks = picks[np.argsort(picks[:,0])]

# Age-Depth boundaries from database used in this run
sql = """SELECT age_depth_boundaries
FROM metadata_sed_rate
where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
age_depth_boundaries = pd.read_sql(sql, con).iloc[0,0] # Indices when sorted by age

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
if concunique[0,0] > 0.05:
    concunique = np.concatenate((np.array(([0],ct0)).T, concunique), axis=0)  # Add in seawater value (upper limit)

# Fit exponential curve to concentration datapoints (specified as "dp")
def conc_curve(z, a):
    return concunique[0,1] * np.exp(np.multiply(a, z))  # a = (v - sqrt(v**2 * 4Dk))/2D


'''def conc_curve(z, a, b, k):
    return a*np.sin(np.sqrt(k)*z)+b*np.cos(np.sqrt(k)*z)
'''
'''def conc_curve(z, a):
    return (concunique[0,1]-concunique[dp-1,1]) * np.exp(np.multiply(np.multiply(-1, a), z)) + concunique[dp-1,1]
'''
conc_fit, conc_cov = optimize.curve_fit(conc_curve, concunique[:dp,0], concunique[:dp,1], p0=[1])
conc_fit = conc_fit[0]
# conc_interp_depths = np.arange(0,3,intervalthickness)  # Three equally-spaced points
# conc_interp_fit = conc_curve(conc_interp_depths, conc_fit)  # To be used if Boudreau method for conc gradient is used
conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fit)

# R-squared function
def rsq(modeled, measured):
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured)-1)*np.var(measured)
    return 1-sse/sstotal

r_squared = rsq(conc_curve(concunique[:dp,0], conc_fit), concunique[:dp,1])

###############################################################################
# Porosity and solids fraction functions and data preparation

# Porosity vectors
por = averages(pordata[:, 0], pordata[:, 1])  # Average duplicates
porvalues = por[:, 1]
pordepth = por[:, 0]

# Porosity curve fit (Modified Athy's Law, ) (Makes porosity at sed surface equal to greatest of first 3 measurements)
def porcurve(z, a):
    portop = np.max(porvalues[:3])  # Greatest of top 3 porosity measurements for upper porosity boundary
    porbottom = np.mean(porvalues[-3:])  # Takes lowest porosity measurement as the lower boundary
    return (portop-porbottom) * np.exp(np.multiply(np.multiply(-1, a), z)) + porbottom

porfit, porcov = optimize.curve_fit(porcurve, pordepth, porvalues, p0=0.01)
porfit = porfit[0]

# Sediment properties at flux depth
porosity = porcurve(z, porfit)
tortuosity = 1-np.log(porosity**2)


###############################################################################
# Calculates viscosity from Mostafa H. Sharqawy 12-18-2009, MIT (mhamed@mit.edu) Sharqawy M. H., Lienhard J. H., and Zubair, S. M., Desalination and Water Treatment, 2009
# Viscosity used as input into Stokes-Einstein equation
# Td is the reference temperature (TempD), T is the in situ temperature
def Dstp(Td, T):
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

# Diffusion coefficient
D_in_situ = Dstp(TempD, bottom_temp)
Dsed = D_in_situ/tortuosity  # Effective diffusion coefficient

###############################################################################
# Pore water burial mass flux

# Solids curve fit (based on porosity curve fit function)
def solidcurve(z, a):
    portop = np.max(porvalues[:3])  # Greatest of top 3 porosity measurements for upper porosity boundary
    porbottom = np.mean(porvalues[-3:])  # Takes lowest porosity measurement as the lower boundary
    return 1-((portop-porbottom) * np.exp(np.multiply(np.multiply(-1, a), z)) + porbottom)

# Sediment mass (1-dimensional volume of solids) accumulation rates for each age-depth section
# Assumes constant sediment mass (really volume of solids) accumulation rates between age-depth measurements
sectionmass = (integrate.quad(solidcurve, seddepths[0], seddepths[1], args=(porfit)))[0]
sedmassrate = (sectionmass/np.diff(sedtimes)[0])

# Pore water burial flux calculation (ref?)
deeppor = porcurve(pordepth[-1], porfit)
deepsolid = solidcurve(pordepth[-1], porfit)
pwburialflux = deeppor*sedmassrate/deepsolid

###############################################################################
# Flux model

# gradient = (-3*conc_interp_fit[0] + 4*conc_interp_fit[1] - conc_interp_fit[2])/(2*intervalthickness)  # Approximation according to Boudreau 1997 Diagenetic Models and Their Implementation. Matches well with derivative method
a = conc_fit
gradient = concunique[0, 1] * a * np.exp(np.multiply(a, z))
# gradient = (concunique[0, 1] - concunique[dp-1, 1]) * -a * np.exp(-a * z)  # Derivative of conc_curve @ z
burial_flux = pwburialflux * conc_curve(z, conc_fit)
flux = porosity * Dsed * -gradient + (porosity * advection + pwburialflux) * conc_curve(z, conc_fit)
print('Flux (mol/m^2 y^-1):', flux)

###############################################################################
# Plotting

# Set up axes and subplot grid
figure_1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 7))
grid = gridspec.GridSpec(3, 8, wspace=0.7)
ax1 = plt.subplot(grid[0:3, :2])
ax1.grid()
ax2 = plt.subplot(grid[0:3, 2:4], sharey=ax1)
ax2.grid()
ax3 = plt.subplot(grid[0:3, 4:6], sharey=ax1)
ax3.grid()
ax4 = plt.subplot(grid[0:3, 6:8], sharey=ax1)
ax4.grid()

# Figure title
figure_1.suptitle(r"$Expedition\ {},\ Site\ {}\ \ \ \ \ \ \ \ \ \ \ \ \ \ {}\ flux={}\ mol/m^2y$".format(Leg, Site, Solute_db, round(flux,4)), fontsize=20)

# Plot input data
ax1.plot(concunique[:,1], concunique[:,0], 'go')
ax1.plot(concunique[0:dp,1], concunique[0:dp,0], 'bo', label="Used for curve fit")
ax1.plot(conc_interp_fit_plot, np.linspace(concunique[0,0], concunique[dp-1,0], num=50), 'k-')
ax2.plot(por[:, 1], por[:, 0], 'mo', label='Measured')
ax2.plot(porcurve(pordepth, porfit), pordepth, 'k-', label='Curve fit', linewidth=3)
ax3.plot(sedtemp(np.arange(concunique[-1,0]), bottom_temp), np.arange(concunique[-1,0]), 'k-', linewidth=3)
ax4.plot(picks[:,1]/1000000, picks[:,0], 'ro', label='Picks')
ax4.plot(sedtimes/1000000, seddepths, 'k-', label='Curve fit', linewidth=2)

# Inset in concentration plot
y2 = np.ceil(concunique[dp-1,0])
x2 = max(concunique[:dp,1])+2
x1 = min(concunique[:dp,1])-2
axins1 = inset_axes(ax1, width="50%", height="30%", loc=5)
axins1.plot(concunique[:,1], concunique[:,0], 'go')
axins1.plot(concunique[0:dp,1], concunique[0:dp,0], 'bo', label="Used for curve fit")
axins1.plot(conc_interp_fit_plot, np.linspace(concunique[0,0], concunique[dp-1,0], num=50), 'k-')
axins1.set_xlim(x1-1, x2+1)
axins1.set_ylim(0, y2)
mark_inset(ax1, axins1, loc1=1, loc2=2, fc="none", ec="0.5")

# Additional formatting
ax1.legend(loc='best', fontsize='small')
ax2.legend(loc='best', fontsize='small')
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
axins1.locator_params(axis='x', nbins=3)
ax1.invert_yaxis()
axins1.invert_yaxis()
figure_1.show()

# Save Figure
savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\interface_flux_{}_{}.png".format(Leg, Site))

###############################################################################
# Send metadata to database

cursor.execute("""select site_key from site_info where leg = '{}' and site = '{}' ;""".format(Leg, Site))
site_key = cursor.fetchone()[0]
cursor.execute("""insert into metadata_{}_flux (site_key, leg, site, hole, solute,
interface_flux, burial_flux, gradient, top_por, flux_depth, datapoints, bottom_conc, r_squared, age_depth_boundaries, sed_rate,
advection, measurement_precision, ds, ds_reference_temp, bottom_temp, bottom_temp_est, script, run_date, comments, complete)
VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, '{}', {}, {}, {}, {}, {}, {}, '{}', '{}', '{}', '{}', '{}')
ON DUPLICATE KEY UPDATE hole='{}', solute='{}', interface_flux={}, burial_flux={}, gradient={}, top_por={}, flux_depth={},
datapoints={}, bottom_conc={}, r_squared={}, age_depth_boundaries='{}', sed_rate={}, advection={},
measurement_precision={}, ds={}, ds_reference_temp={}, bottom_temp={}, bottom_temp_est='{}', script='{}', run_date='{}', comments='{}', complete='{}'
;""".format(Solute_db, site_key, Leg, Site, Hole, Solute, flux, burial_flux, gradient, porosity, z, dp, bottom_conc, r_squared,
age_depth_boundaries, sedrate, advection, Precision, Ds, TempD, bottom_temp, bottom_temp_est, Script, Date, Comments, Complete,
Hole, Solute, flux, burial_flux, gradient, porosity, z, dp, bottom_conc, r_squared, age_depth_boundaries, sedrate, advection,
Precision, Ds, TempD, bottom_temp, bottom_temp_est, Script, Date, Comments, Complete))
con.commit()

# eof
