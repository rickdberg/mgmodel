# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017

@author: rickdberg

Script to calculate sediment-water interface fluxes of a dissolved constituent

Uses central difference formula of a spline fit to the upper sediment section

Add in pw burial flux
Add in capability to handle advection at surface
How to deal with tortuosity change w depth?
Add in appropriate bottom boundary
Add in concentration spline fit
Add in interpolated depths/thicknesses
Add in plotting of input data, processed data, and results


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MySQLdb
import os
import datetime

Script = os.path.basename(__file__)
Date = datetime.datetime.now()


# Site ID
Leg = '315'
Site = 'C0002'
Holes = "('B', 'D', 'H', 'J', 'K', 'L', 'M', 'P')"
Bottom_boundary = 'none' # 'none', or a depth
Hole = ''.join(filter(str.isalpha, Holes))  # Formatting for saving in metadata

# Species parameters
Solute = 'Mg'
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)
TempD = 18  # Temperature at which diffusion coefficient is known
Precision = 0.02  # measurement precision

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
sedrate = sedrates[0]

# Load age-depth data for plots
sql = """SELECT depth, age FROM age_depth where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
picks = pd.read_sql(sql, con)
picks = picks.as_matrix()
picks = picks[np.argsort(picks[:,0])]
if Bottom_boundary == 'none':
    picks = picks
else:
    deepest_pick_idx = np.searchsorted(picks[:,0], Bottom_boundary)
    picks = picks[:deepest_pick_idx, :]
picks = picks[np.argsort(picks[:,1])]

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
# Flux model # Needs work

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

gradient = (-3*conc_interp[0] + 4*conc_interp[1] - conc_interp[2])/(2*intervalthickness)

flux = Dsed * gradient + (porcurve(pordepth[0], porfit) * advection + pwburialflux) * conc_interp[0]




# eof
