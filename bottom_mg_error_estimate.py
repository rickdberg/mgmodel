# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:55:05 2017

@author: rickdberg

Calculate the rmse of modeled bottom water Mg concentration vs actual measurements

"""
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plot
import seawater

engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")

# Cl titration data
sql = """
SELECT hole_key, leg, site, hole, sample_depth, Cl
FROM iw_all
where hydrate_affected is null and Cl is not null
and sample_depth > 0.05 and sample_depth < 10
;
"""
cl_data = pd.read_sql(sql, engine)
cl_data = cl_data.fillna(np.nan).sort_values(['hole_key', 'sample_depth'])
cl_cut_data = cl_data.groupby(by='hole_key').head(1)
# cl_avgs = cl_cut_data.groupby(by='hole_key')['Cl'].mean().reset_index()
# cl_avgd = cl_avgs.merge(cl_data['hole_key'], how='inner', on=('hole_key')

# Cl ic data
sql = """
SELECT hole_key, leg, site, hole, sample_depth, Cl_ic
FROM iw_all
where hydrate_affected is null and Cl_ic is not null
and sample_depth > 0.05 and sample_depth < 10
;
"""
cl_ic_data = pd.read_sql(sql, engine)
cl_ic_data = cl_ic_data.fillna(np.nan).sort_values(['hole_key', 'sample_depth'])
cl_ic_cut_data = cl_ic_data.groupby(by='hole_key').head(1)
# cl_avgs = cl_cut_data.groupby(by='hole_key')['Cl'].mean().reset_index()
# cl_avgd = cl_avgs.merge(cl_data, how='inner', on='hole_key')

# Mg measurement directly from bottom water
sql = """
SELECT hole_key, leg, site, hole, sample_depth, Mg, Mg_ic
FROM iw_all
where hydrate_affected is null and Mg is not null
and sample_depth < 0.05
;
"""
mg_bw_data = pd.read_sql(sql, engine)

# Mg calculated from WOA salinity data
sql = """
SELECT hole_key, leg, site, hole, woa_bottom_salinity,
water_depth, woa_bottom_temp, lat, lon
FROM summary_all
;"""
woa_salinity = pd.read_sql(sql, engine)
density = seawater.eos80.dens0(woa_salinity['woa_bottom_salinity'], woa_salinity['woa_bottom_temp'])
def sal_to_cl(salinity, density):
    return (1000*(salinity-0.03)*density/1000)/(1.805*35.45)
woa_cl = sal_to_cl(woa_salinity['woa_bottom_salinity'].rename('woa_mg'), density)
woa_mg=woa_cl/558*54

woa = pd.concat((woa_salinity, woa_mg), axis=1)

all_data = cl_cut_data.merge(cl_ic_cut_data, how='outer', on=(
'hole_key', 'leg', 'site', 'hole','sample_depth'))
all_data = all_data.merge(mg_bw_data.loc[:,(
'hole_key','leg', 'site', 'hole', 'Mg', 'Mg_ic')], how='inner', on=(
'hole_key','leg', 'site', 'hole'))
all_data = all_data.merge(woa.loc[:,(
'hole_key','leg', 'site', 'hole', 'woa_bottom_salinity', 'woa_mg',
'water_depth', 'woa_bottom_temp', 'lat','lon')], how='inner', on=(
'hole_key','leg', 'site', 'hole'))
# all_data = cl_avgs.merge(mg_bw_data, how='inner', on=('hole_key'))

stacked_data = pd.concat([
all_data.loc[:,('hole_key','leg', 'site', 'hole',
'sample_depth', 'Cl', 'Mg', 'Mg_ic')],
all_data.loc[:,('hole_key','leg', 'site', 'hole',
'sample_depth', 'Cl_ic', 'Mg', 'Mg_ic')].rename(columns={'Cl_ic':'Cl'})])

plot.plot(54/558*all_data['Cl'], all_data['Mg'], 'go')
# plot.plot(54/558*all_data['Cl_ic'], all_data['Mg'], 'ro')
# plot.plot(all_data['woa_mg'], all_data['Mg'], 'go')
plot.plot(np.linspace(20, 60, num=50), np.linspace(20, 60, num=50), 'k--')
plot.xlabel('Estimated Mg concentration (mM)', fontsize=20)
plot.ylabel('Measured Mg concentration (mM)', fontsize=20)
plt.tick_params(labelsize=16)
plot.show()
def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())

error = rmse(54/558*all_data['Cl'], all_data['Mg'])/54
error_all = rmse(54/558*stacked_data['Cl'], stacked_data['Mg'])/54
error_woa = rmse(all_data['woa_mg'], all_data['Mg'])/54

# all_err = 54/558*all_data['Cl'] - all_data['Mg']
# plot.hist(all_err[all_err.notnull()], bins=50)
# plot.hist(all_data['woa_mg']- all_data['Mg'], bins=50)

# eof
