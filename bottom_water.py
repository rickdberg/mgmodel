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

all_data = cl_cut_data.merge(cl_ic_cut_data, how='outer', on=('hole_key', 'leg', 'site', 'hole','sample_depth'))
all_data = all_data.merge(mg_bw_data.loc[:,('hole_key','leg', 'site', 'hole', 'Mg', 'Mg_ic')], how='inner', on=('hole_key','leg', 'site', 'hole'))
# all_data = cl_avgs.merge(mg_bw_data, how='inner', on=('hole_key'))

stacked_data = pd.concat([
all_data.loc[:,('hole_key','leg', 'site', 'hole',
'sample_depth', 'Cl', 'Mg', 'Mg_ic')],
all_data.loc[:,('hole_key','leg', 'site', 'hole',
'sample_depth', 'Cl_ic', 'Mg', 'Mg_ic')].rename(columns={'Cl_ic':'Cl'})])

plot.plot(54/558*all_data['Cl'], all_data['Mg'], 'bo')
plot.plot(54/558*all_data['Cl_ic'], all_data['Mg'], 'ro')
plot.plot(np.linspace(20, 60, num=50), np.linspace(20, 60, num=50), 'k-')

def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())

error = rmse(54/558*all_data['Cl'], all_data['Mg'])/54
error_all = rmse(54/558*stacked_data['Cl'], stacked_data['Mg'])/54

# eof
