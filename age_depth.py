# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:49:51 2016

@author: rickdberg
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

# Site ID
Leg = '315'
Site = 'C0001'
Holes = "('E','F','H','B')"
Bottom_boundary = 'none' # 'none', or a depth
age_depth_boundaries = [0, 7, 15, 23, 29] # Index when sorted by age
Hole = ''.join(filter(str.isalpha, Holes))

###############################################################################
###############################################################################
###############################################################################
# Load data from database

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp'
conctable = 'iw_chikyu'
portable = 'mad_all'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)

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
# sedfit = np.polyfit(sed[:,0], sed[:,1], 4)

# Averaging function (from http://stackoverflow.com/questions/4022465/average-the-duplicated-values-from-two-paired-lists-in-python)
# Altered to sort by age, not depth
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
    sorted = np.column_stack((resultkeys[np.argsort(resultvalues)], resultvalues[np.argsort(resultvalues)]))
    return sorted

# Age-depth data after averaging and do piece-wise linear regression on age-depth data
sed = averages(sed[:,0], sed[:,1])


# Put in for loop to run linear regressions for as many sections as each site has

def age_curve(z, p):
    return p*z

plt.figure()
plt.plot(sed[:,1], sed[:,0], "o")

cut_depths = sed[age_depth_boundaries, 0]
cut_ages = sed[age_depth_boundaries, 1]
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








