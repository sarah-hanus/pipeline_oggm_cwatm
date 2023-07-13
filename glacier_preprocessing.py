import numpy as np
import glacier_pre_post
import glob
import geopandas as gpd
import xarray as xr
import pandas as pd
import pickle as pkl
import warnings
import time
# enter path to git repository
path_general = 'C:/Users/shanus/Data/Glaciers_new/'
#location of shapefiles of rgis
path_rgi_files = path_general + '/rgi60/00_rgi60/'
#directory to glacier statistics containing center and terminus location of grid cells
glacier_csv = glob.glob(path_general + '/rgi60/Glacier_Statistics/glacier_statistics_*.csv')

#which resolution to use: 5min or 30min
resolution = "30min"
path_output = path_general + '/glaciers_preprocessed/{}/'.format(resolution)

#rgi regions to use
rgi_regions = ['01','02', '03', '04', '05','06', '07','08', '09','10','11','12','13','14','15','16','17','18']
grid = gpd.read_file(path_general + "/Grid_shp/grid_coordinates_{}.shp".format(resolution))

# ------------------------------ PREPROCESSING ----------------------------
'''preprocessing is independent of OGGM results and relates glacier IDs to grid cells and glacier area at RGI date (shapefiles) to grid cells'''

# ------------------------------ STEP 1 -------------------------------------
# 1) generate dictionaries of glacier ids as keys and coordinates of gridcell in which glaciers have terminus as values
# (the coordinates are the center of each grid cell, so depends on the resolution)
#glacier_pre_post.make_glacier_outlet_dict(glacier_csv, path_output, "glaciers", resolution, rgi_ids=None)
# ------------------------------------------- STEP 2 ----------------------------------
# 2) generate csv file with info about glacier area in each grid cell
# glacier_pre_post.df_glacier_grid_area(grid, rgi_regions, path_output, "glaciers", path_rgi_files, "glacier_area_df", resolution)

