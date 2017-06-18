# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017

@author: rickdberg

Combine calculated metadata, site_info, and summary_all tables for sites with
calculated fluxes
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def metadata_compiler(database, metadata_table, site_info, hole_info, leg, site):
    engine = create_engine(database)

    # Load metadata
    sql = "SELECT * FROM {} where leg = '{}' and site = '{}';".format(metadata_table, leg, site)
    metadata = pd.read_sql(sql, engine)

    # Load site data
    sql = "SELECT * FROM {} where leg = '{}' and site = '{}';".format(site_info, leg, site)
    sitedata = pd.read_sql(sql, engine)

    # Load hole data
    sql = "SELECT * FROM {} where leg = '{}' and site = '{}';".format(hole_info, leg, site)
    holedata = pd.read_sql(sql, engine)
    # Group and average hole data for sites
    hole_grouped = holedata.loc[:,('site_key', 'lat','lon','water_depth',
                                   'total_penetration', 'woa_bottom_temp',
                                   'woa_bottom_salinity', 'woa_bottom_o2'
                                   )].groupby("site_key").mean().reset_index()

    # Combine all tables
    site_meta_data = pd.merge(metadata, sitedata, how='outer', on=('site_key', 'leg', 'site'))
    data = pd.merge(site_meta_data, hole_grouped, how='outer', on=('site_key')).fillna(np.nan)
    site_metadata = data.dropna(subset = ['interface_flux']).reset_index(drop=True)
    return site_metadata

# eof
