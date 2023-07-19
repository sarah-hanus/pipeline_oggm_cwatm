import numpy as np
import glacier_pre_post_functions
import glob
import geopandas as gpd
import xarray as xr
import pandas as pd
import pickle as pkl
import warnings
import time

# # ------------------ POSTPROCESSING FOR WHOLE WORLD
# ------------------- ENTER YOUR PATHS ---------------
# enter path to data folder, this can be different than git folder, because data does not fit on github
path_general = r'C:\Users\shanus\Processing\pipeline_oggm_cwatm'
#resolution (5min or 30min)
resolution = '5min'
# put in output directory
path_output_results = 'C:/Users/shanus/Data/Glaciers_new/try/'

#precipitation factor
pf = '3.0'
path_oggm_results = r'D:\Results\OGGM\Global\past/fixed/'
all_regions = glob.glob(path_oggm_results + 'historical_run_output_*_1990_2019_mb_real_daily_cte_pf_{}*'.format(str(pf)))
oggm_results = []
for region in all_regions:
    oggm_results.append(xr.open_dataset(region))

# -----------------------------------------------------

path_cellarea = path_general + '/cellarea/cellarea_{}.nc'.format(resolution)
cellarea = xr.open_dataset(path_cellarea)
path_preprocessed = path_general + '/glaciers_preprocessed/{}/'.format(resolution)

if resolution == '30min':
    glacier_area_csv = pd.read_csv(path_preprocessed + 'glacier_area_df_{}.csv'.format(resolution))
elif resolution == '5min':
    glacier_area_csv = pd.read_csv(path_preprocessed + 'glacier_area_df_{}.csv'.format(resolution))

example_nc = path_cellarea
glacier_outlet = pkl.load(open(path_preprocessed +'/glaciers_key_coordinates_terminus_{}.pkl'.format(resolution), "rb"))

# ------------- GENERATE GLACIER AREA INPUT FOR CWATM -------------------
#TODO this is slow for 30arcmin global and will be very slow for 5arcmin global

# start_time = time.time()
# glacier_pre_post_functions.oggm_area_to_cwatm_input_world(glacier_area_csv, all_regions, cellarea, 1990, 2019, path_output_results + '5arcmin_area/', "all_regions_pf{}_{}".format(str(pf), resolution), example_nc,resolution, fraction=True, fixed_year=None, include_off_area = False)
# end_time = time.time()
# print("\ntime to run whole function " + str(end_time - start_time))

from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(glacier_pre_post_functions.change_area_world) # add additional function to profile
lp_wrapper = lp(glacier_pre_post_functions.oggm_area_to_cwatm_input_world)
lp_wrapper(glacier_area_csv, all_regions, cellarea, 1990, 2019, path_output_results + '5arcmin_area/', "all_regions_pf{}_{}".format(str(pf), resolution), example_nc,resolution, fraction=True, fixed_year=None, include_off_area = False)
file2 = open(path_output_results + 'output_area_csv_ready_{}_all.txt'.format(resolution),"w+")
lp.print_stats(stream= file2)

#check if sum of area in netcdf files is similar to sum of area from csv files
area = xr.open_dataset(r'C:\Users\shanus\Data\Glaciers_new\try\5arcmin_area/on_area_fraction_all_regions_pf3.0_5min.nc')
area_glaciers = area.on_area[0].values * cellarea.cellarea.values
diff = (area_glaciers.sum() - glacier_area_csv.Area.sum()) / glacier_area_csv.Area.sum()
# differences is around 6%, so the area in netcdf files is around 6% lower than area in glacier csv
#might be because area already lower than in RGI date for some regions?
# --------------- GENERATE GLACIER MELT INPUT FOR CWATM -----------------
#TODO this is very slow

# start_time = time.time()
# glacier_pre_post_functions.oggm_output_to_cwatm_input_world(glacier_outlet, all_regions[5:6], float(pf), 1990, 2019, path_output_results, "all_regions_mask_pf{}_{}".format(str(pf), resolution), example_nc, resolution, include_off_area = False)
# end_time = time.time()
# print("\ntime to run whole function" + str(end_time - start_time))
#glacier_pre_post.oggm_output_to_cwatm_input_world(glacier_outlet_30min, all_regions[5:7], pf, 1990, 2019, path_output_results, " test_all_regions_mask_pf{}_{}".format(str(pf), resolution), example_nc_30arcmin, resolution, include_off_area = False)

# lp2 = LineProfiler()
# lp2.add_function(glacier_pre_post_functions.change_format_oggm_output_world) # add additional function to profile
# lp2_wrapper = lp2(glacier_pre_post_functions.oggm_output_to_cwatm_input_world)
# lp2_wrapper(glacier_outlet, all_regions, float(pf), 1990, 2019, path_output_results, "all_regions_pf{}_{}".format(str(pf), resolution), example_nc, resolution, include_off_area = False)
# file2 = open(path_output_results + 'output_melt_{}_nosave.txt'.format(resolution),"w+")
# lp2.print_stats(stream= file2)