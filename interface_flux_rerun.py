# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:15:18 2017

@author: rickdberg

Script to re-run sites through interface_flux.py
and interface_flux_montecarlo.py using information in database metadata


"""
import numpy as np
import pandas as pd
from scipy import optimize, integrate, stats
import sqlalchemy as sa
import MySQLdb
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pylab import savefig
from collections import defaultdict
from matplotlib import mlab
plt.ioff()

Script = os.path.basename(__file__)
Date = datetime.datetime.now()

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
conctable = 'iw_all'
portable = 'mad_all'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cursor = con.cursor()
Complete = 'no'

# Species parameters
Ocean = 54  # Concentration in modern ocean (mM)
Solute_db = 'Mg' # Label for the database loading

# Simulation parameters
cycles = 5000

# Load metadata from database
engine = sa.create_engine("mysql://root:neogene227@localhost/iodp_compiled")
sql = "SELECT * FROM metadata_mg_flux WHERE complete = 'yes';"
metadata = pd.read_sql(sql, con=engine)

for i in np.arange(np.size(metadata, axis=0))+21:
    Comments = metadata.comments[i]
    Leg = metadata.leg[i]
    Site = metadata.site[i]
    Holes = "('{}')".format('\',\''.join(filter(str.isalpha, metadata.hole[i])))
    Hole = metadata.hole[i]
    Solute = metadata.solute[i]
    Ds = float(metadata.ds[i])
    TempD = float(metadata.ds_reference_temp[i])
    Precision = float(metadata.measurement_precision[i])
    dp = int(metadata.datapoints[i])
    z = float(metadata.flux_depth[i])
    ###############################################################################
    ###############################################################################
    ###############################################################################
    # Load data from database


    # Pore water chemistry data
    sql = """SELECT sample_depth, {} FROM {} where leg = '{}' and site = '{}' and hole in {} and {} is not null and hydrate_affected is null; """.format(Solute, conctable, Leg, Site, Holes, Solute)
    concdata = pd.read_sql(sql, con)
    concdata = concdata.sort_values(by='sample_depth')
    concdata = concdata.as_matrix()

    # Bottom water concentration
    sql = """SELECT sample_depth, Cl, Cl_ic FROM {} where leg = '{}' and site = '{}' and hole in {} and hydrate_affected is null; """.format(conctable, Leg, Site, Holes)
    cl_data = pd.read_sql(sql, con)
    cl_data = cl_data.fillna(np.nan).sort_values(by='sample_depth')
    if cl_data.iloc[:3,1].isnull().all():
        cl_bottom_water = stats.nanmean(cl_data.iloc[:3,2])
    else:
        cl_bottom_water = stats.nanmean(cl_data.iloc[:3,1])
    bottom_conc = Ocean/558*cl_bottom_water  # Solute normalized to Cl in top three iw measurements from site
    ct0 = [bottom_conc]  # mol per m^3 in modern seawater at specific site

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

    # Sedimentation rate profile (m/y) (Calculated in age_depth.py)
    sql = """SELECT sedrate_ages, sedrate_depths FROM metadata_sed_rate where leg = '{}' and site = '{}' ; """.format(Leg, Site)
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
    sql = """SELECT depth, age FROM age_depth where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
    picks = pd.read_sql(sql, con)
    picks = picks.as_matrix()
    picks = picks[np.argsort(picks[:,0])]

    # Age-Depth boundaries from database used in this run
    sql = """SELECT age_depth_boundaries FROM metadata_sed_rate where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
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
        return (concunique[0,1]-concunique[dp-1,1]) * np.exp(np.multiply(np.multiply(-1, a), z)) + concunique[dp-1,1]

    conc_fit, conc_cov = optimize.curve_fit(conc_curve, concunique[:dp,0], concunique[:dp,1], p0=0.01)
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
        porbottom = porvalues[-1]  # Takes lowest porosity measurement as the lower boundary
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
        porbottom = porvalues[-1]  # Takes lowest porosity measurement as the lower boundary
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

    #  gradient = (-3*conc_interp_fit[0] + 4*conc_interp_fit[1] - conc_interp_fit[2])/(2*intervalthickness)  # Approximation according to Boudreau 1997 Diagenetic Models and Their Implementation. Matches well with derivative method
    a = conc_fit
    gradient = (concunique[0, 1] - concunique[dp - 1, 1]) * -a * np.exp(-a * z)  # Derivative of conc_curve @ z
    burial_flux = pwburialflux * conc_curve(z, conc_fit)
    flux = porosity * Dsed * -gradient + (porosity * advection + pwburialflux) * conc_curve(z, conc_fit)

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
    y2 = np.ceil(concunique[dp+1,0])
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

    # Save Figure
    savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\interface_flux_{}_{}.png".format(Leg, Site))

    ###############################################################################
    # Send metadata to database

    cursor.execute("""select site_key from site_info where leg = '{}' and site = '{}' ;""".format(Leg, Site))
    site_key = cursor.fetchone()[0]
    cursor.execute("""insert into metadata_{}_flux (site_key, leg, site, hole, solute,
    interface_flux, burial_flux, flux_depth, datapoints, bottom_conc, r_squared, age_depth_boundaries, sed_rate,
    advection, measurement_precision, ds, ds_reference_temp, bottom_temp, script, run_date, comments, complete)
    VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, '{}', {}, {}, {}, {}, {}, {}, '{}', '{}', '{}', '{}')
    ON DUPLICATE KEY UPDATE hole='{}', solute='{}', interface_flux={}, burial_flux={}, flux_depth={},
    datapoints={}, bottom_conc={}, r_squared={}, age_depth_boundaries='{}', sed_rate={}, advection={},
    measurement_precision={}, ds={}, ds_reference_temp={}, bottom_temp={}, script='{}', run_date='{}', comments='{}', complete='{}'
    ;""".format(Solute_db, site_key, Leg, Site, Hole, Solute, flux, burial_flux, z, dp, bottom_conc, r_squared,
    age_depth_boundaries, sedrate, advection, Precision, Ds, TempD, bottom_temp, Script, Date, Comments, Complete,
    Hole, Solute, flux, burial_flux, z, dp, bottom_conc, r_squared, age_depth_boundaries, sedrate, advection,
    Precision, Ds, TempD, bottom_temp, Script, Date, Comments, Complete))
    con.commit()

    ###############################################################################
    ###############################################################################
    ###############################################################################
    # Monte Carlo simulation

    # Concentration offsets - using full gaussian probability
    relativeerror = Precision
    conc_offsets = np.random.normal(scale=relativeerror, size=(cycles, len(concunique[:dp,1])))

    # Error calculated as relative root mean squared error of curve fit to reported values
    def rmse(model_values, measured_values):
        return np.sqrt(((model_values-measured_values)**2).mean())

    # Porosity offsets - using full gaussian probability
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

    # Get randomized porosity matrix (within realistic ranges between 30% and 90%)
    por_rand = np.add(porcurve(pordepth, porfit), por_offsets)
    por_rand[por_rand > 0.90] = 0.90
    por_rand[por_rand < 0.30] = 0.30

    portop = np.max(porvalues[:3])
    portop_rand = np.add(portop, por_offsets[:,0])
    portop_rand[portop_rand > 0.90] = 0.90
    for n in range(cycles):
        portop_rand[portop_rand < por_rand[n,-1]] = por_rand[n,-1]

    # Define curve fit functions for Monte Carlo Method
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
    conc_fits = []
    por_fits = []
    sectionmasses = []
    for n in range(cycles):
        # Fit exponential curve to each randomized concentration profile
        conc_fit, conc_cov = optimize.curve_fit(conc_curve_mc, concunique[:dp,0], conc_rand[n], p0=0.1)
        conc_fit = conc_fit[0]
        conc_fits.append(conc_fit)

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


    tortuosity_rand = 1-np.log(portop_rand**2)
    Dsed_rand = Dstp(TempD, bottom_temp)/tortuosity_rand


    # Plot all the monte carlo runs
    # conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fits)
    # por_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fits)

    # Calculate fluxes
    a = conc_fits
    gradient = (conc_rand[:,0] - conc_rand[:, -1]) * -1 * a * np.exp(np.multiply(np.multiply(-1, a), z))  # Derivative of conc_curve @ z
    interface_fluxes = portop_rand * Dsed_rand * -gradient + (portop_rand * advection + pwburialflux) * conc_curve_mc(z, conc_fits)

    ###############################################################################
    # Distribution statistics

    # Stats on normal distribution
    mean_flux = np.mean(interface_fluxes)
    # print('Mean Flux:', mean_flux)

    median_flux = np.median(interface_fluxes)
    # print('Median Flux:', median_flux)

    stdev_flux = np.std(interface_fluxes)
    # print('Std Dev Flux:', stdev_flux)

    skewness = stats.skew(abs(interface_fluxes))
    # print('skewness:', skewness)
    z_score, p_value = stats.skewtest(abs(interface_fluxes))
    # print('z-score:', z_score)

    # Stats on lognormal distribution
    interface_fluxes_log = -np.log(abs(interface_fluxes))

    mean_flux_log = np.mean(interface_fluxes_log)
    # print('Mean ln(Flux):', mean_flux_log)

    median_flux_log = np.median(interface_fluxes_log)
    # print('Median ln(Flux):', median_flux_log)

    stdev_flux_log = np.std(interface_fluxes_log)
    # print('Std Dev ln(Flux):', stdev_flux_log)

    skewness_log = stats.skew(interface_fluxes_log)
    # print('skewness (ln):', skewness_log)
    z_score_log, p_value_log = stats.skewtest(interface_fluxes_log)
    # print('z-score (ln):', z_score_log)

    stdev_flux_lower = np.exp(-(median_flux_log-stdev_flux_log))
    # print('Std Dev Lower: ', stdev_flux_lower-median_flux)
    stdev_flux_upper = np.exp(-(median_flux_log+stdev_flux_log))
    # print('Std Dev Upper: ', stdev_flux_upper-median_flux)

    ###############################################################################
    # Plot distributions

    figure_2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram of results
    n_1, bins_1, patches_1 = ax5.hist(interface_fluxes, normed=1, bins=30, facecolor='orange')

    # Best fit normal distribution line to results
    bf_line_1 = mlab.normpdf(bins_1, median_flux, stdev_flux)
    ax5.plot(bins_1, bf_line_1, 'k--', linewidth=2)
    ax5.set_xlabel("Interface Flux")

    [left_raw, right_raw] = ax5.get_xlim()
    [bottom_raw, top_raw] = ax5.get_ylim()
    ax5.text((left_raw+(right_raw-left_raw)/20), (top_raw-(top_raw-bottom_raw)/20), 'sk = {}'.format(np.round(skewness, 2)))
    ax5.text((left_raw+(right_raw-left_raw)/20), (top_raw-(top_raw-bottom_raw)/10), "z' = {}".format(np.round(z_score, 2)))

    # Plot histogram of ln(results)
    n_2, bins_2, patches_2 = ax6.hist(interface_fluxes_log, normed=1, bins=30, facecolor='g')

    # Best fit normal distribution line to ln(results)
    bf_line_2 = mlab.normpdf(bins_2, median_flux_log, stdev_flux_log)
    ax6.plot(bins_2, bf_line_2, 'k--', linewidth=2)
    ax6.set_xlabel("ln(abs(Interface Flux)")

    [left_log, right_log] = ax6.get_xlim()
    [bottom_log, top_log] = ax6.get_ylim()
    ax6.text((left_log+(right_log-left_log)/20), (top_log-(top_log-bottom_log)/20), 'sk = {}'.format(np.round(skewness_log, 2)))
    ax6.text((left_log+(right_log-left_log)/20), (top_log-(top_log-bottom_log)/10), "z' = {}".format(np.round(z_score_log, 2)))

    # figure_2.show()

    ###############################################################################
    Complete='yes'

    # Send metadata to database
    cursor.execute("""select site_key from site_info where leg = '{}' and site = '{}' ;""".format(Leg, Site))
    site_key = cursor.fetchone()[0]
    cursor.execute("""insert into metadata_{}_flux (site_key, mc_cycles, porosity_error, mean_flux,
    median_flux, stdev_flux, skewness, z_score, mean_flux_log, median_flux_log, stdev_flux_log,
    stdev_flux_lower, stdev_flux_upper, skewness_log, z_score_log, complete)
    VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, '{}')
    ON DUPLICATE KEY UPDATE mc_cycles={}, porosity_error={}, mean_flux={}, median_flux={}, stdev_flux={},
    skewness={}, z_score={}, mean_flux_log={}, median_flux_log={}, stdev_flux_log={}, stdev_flux_lower={},
    stdev_flux_upper={}, skewness_log={}, z_score_log={}, complete='{}'
    ;""".format(Solute_db, site_key, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, z_score,
                mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper,
                skewness_log, z_score_log, Complete, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness,
                z_score, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper,
                skewness_log, z_score_log, Complete))
    con.commit()

    # Save figure and fluxes from each run
    savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\montecarlo_{}_{}.png".format(Leg, Site))
    np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\monte carlo_{}_{}.csv".format(Leg, Site), interface_fluxes, delimiter=",")
    plt.close("all")
