# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:49:51 2016

@author: rickdberg
Script for building age-depth profile from biostratigraphy or
magnetostratigraphy data.

Boundaries between different sedimentation rates are manually input based on
visual inspection of data.

Inputs: Leg, Site, Hole(s), bottom_boundary, age_depth_boundaries
"""

import numpy as np
import pandas as pd
import MySQLdb
import matplotlib.pyplot as plt
from scipy import optimize, integrate
import os
import datetime

Script = os.path.basename(__file__)
Date = datetime.datetime.now()

# Site ID
Leg = '315'
Site = 'C0002'
Holes = "('D', 'B')"
Bottom_boundary = 'none' # 'none', or an integer depth
age_depth_boundaries = [0, 4, 16, 35] # Index when sorted by age

###############################################################################
###############################################################################
###############################################################################
# Load data from database

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cur = con.cursor()

# Sedimentation rate profile (m/y)
# Note: Input data must have time at depth=0
sql = """SELECT depth, age FROM age_depth where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site)
sed = pd.read_sql(sql, con)
sed = sed.as_matrix()
sed = sed[np.argsort(sed[:,0])]
if Bottom_boundary == 'none':
    sed = sed
else:
    deepest_sed_idx = np.searchsorted(sed[:,0], Bottom_boundary)
    sed = sed[:deepest_sed_idx, :]

# Sort by age
sed = sed[np.argsort(sed[:,1])]
datapoints = len(sed)

# Put in for loop to run linear regressions for as many sections as each site has

def age_curve(z, p):
    return p*z

plt.figure()
plt.plot(sed[:,1], sed[:,0], "o")

cut_depths = sed[age_depth_boundaries, 0]
last_depth = 0
last_age = 0
sedrate_ages = [0]
sedrate_depths = [0]
for n in np.arange(len(cut_depths)-1):
    next_depth = cut_depths[n+1]
    sed_alt = np.column_stack((sed[:,0]-last_depth, sed[:,1]-last_age))
    p , e = optimize.curve_fit(age_curve, sed_alt[age_depth_boundaries[n]:age_depth_boundaries[n+1],0], sed_alt[age_depth_boundaries[n]:age_depth_boundaries[n+1],1])
    z = np.linspace(0, next_depth-last_depth, 100)
    age = age_curve(z, p)+last_age
    plt.plot(age, z+last_depth, '-x')
    last_age = age_curve(z[-1], *p)+last_age
    last_depth = cut_depths[n+1]
    sedrate_ages.append(last_age)
    sedrate_depths.append(last_depth)
plt.show()

# Formatting for input into SQL database metadata table
Hole = ''.join(filter(str.isalpha, Holes))

# Save metadata in database
cur.execute("""select site_key from site_info where leg = '{}' and site = '{}' ;""".format(Leg, Site))
site_key = cur.fetchone()[0]
cur.execute("""insert into metadata_sed_rate (site_key, leg, site, hole, 
bottom_boundary, age_depth_boundaries, sedrate_ages, sedrate_depths, datapoints, script, run_date) 
VALUES ({}, '{}', '{}', '{}', '{}', '{}', '{}', '{}', {}, '{}', '{}')  ON DUPLICATE KEY UPDATE hole='{}', 
bottom_boundary='{}', age_depth_boundaries='{}', sedrate_ages='{}', sedrate_depths='{}', datapoints={},
script='{}', run_date='{}' ;""".format(site_key, Leg, Site, Hole, Bottom_boundary, age_depth_boundaries, sedrate_ages, sedrate_depths, datapoints, Script, Date, 
Hole, Bottom_boundary, age_depth_boundaries, sedrate_ages, sedrate_depths, datapoints, Script, Date))
con.commit()

# eof
