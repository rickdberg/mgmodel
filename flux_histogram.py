# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:16:08 2016

@author: rickdberg

Histogram for modeling results

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
import plotly.plotly as py



# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)


# Rate data
sql = """SELECT mean_integrated_rate_modern FROM model_metadata_mg; """
ratedata = pd.read_sql(sql, con)
ratedata = pd.Series(ratedata.iloc[:,0]).astype(float)
#ratedata = ratedata.tolist()
#ratedata = ratedata[0]


plt.xlim([min(ratedata)-0.01, max(ratedata)+0.01])

plt.hist(ratedata, bins=50)
plt.title("Gaussian Histogram")
plt.xlabel("Flux (mol m-2 y-1)")
plt.ylabel("Frequency")

fig = plt.gcf()


