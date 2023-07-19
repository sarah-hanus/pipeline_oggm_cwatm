import numpy as np
import glacier_pre_post_functions
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
resolution = "5min"
path_output = path_general + '/glaciers_preprocessed/{}/'.format(resolution)

#rgi regions to use
rgi_regions = ['01','02', '03', '04', '05','06', '07','08', '09','10','11','12','13','14','15','16','17','18']

path_grid = path_general + "/Grid_shp/grid_coordinates_{}_rgi_regions_ex19_4326.shp".format(resolution)
# ------------------------------ PREPROCESSING ----------------------------
'''preprocessing is independent of OGGM results and relates glacier IDs to grid cells and glacier area at RGI date (shapefiles) to grid cells'''

# ------------------------------ STEP 1 -------------------------------------
# 1) generate dictionaries of glacier ids as keys and coordinates of gridcell in which glaciers have terminus as values
# (the coordinates are the center of each grid cell, so depends on the resolution)
#glacier_pre_post_functions.make_glacier_outlet_dict(glacier_csv, path_output, "glaciers", resolution, rgi_ids=None)
# ------------------------------------------- STEP 2 ----------------------------------
# overlay grid with glacier shape files to get
path_glacier_shape =  r'C:\Users\shanus\Data\Glaciers_new\rgi60\00_rgi60/rgi_60_all_rgi_regions.shp'
glacier_pre_post_functions.overlay_area_grid(path_glacier_shape, path_grid,  path_output + '/rgi_regions_all_{}.shp'.format(resolution))
# ------------------------------------------ STEP 3 -----------------------------------------
# 3) generate csv file with info about glacier area in each grid cell
#glacier_pre_post_functions.df_glacier_grid_area(path_glacier_info + '/rgi_regions_all_{}.shp'.format(resolution), path_output, "glaciers", path_rgi_files, "glacier_area_df", resolution)


def overlay_area_grid(file_path_glacier_shape, file_path_grid, path_save):
    '''overlays the glacier geometries with the grid to get the glacier area per grid cell'''
    rgi = gpd.read_file(file_path_glacier_shape)
    grid = gpd.read_file(file_path_grid)
    # results from overlay functions
    overlay_result = gpd.overlay(rgi, grid, how='intersection')
    # #get area of each part
    #transform it to CRS in meter
    overlay_54012 = overlay_result.to_crs('esri:54012')
    overlay_54012["Area_Cell"] = overlay_54012.area
    overlay_54012.to_file(path_save)


from line_profiler import LineProfiler
lp2 = LineProfiler()
lp2.add_function(glacier_pre_post_functions.transform_to_df) # add additional function to profile
lp2_wrapper = lp2(glacier_pre_post_functions.df_glacier_grid_area)
lp2_wrapper(rgi_regions, path_output, "glaciers", path_rgi_files, "glacier_area_df_new", resolution)
file2 = open(path_output + 'preprocessing_area_new_{}.txt'.format(resolution),"w+")
lp2.print_stats(stream= file2)

