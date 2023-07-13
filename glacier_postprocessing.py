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
resolution = '30min'
# put in output directory
path_output_results = 'C:/Users/shanus/Data/Glaciers_new/try/'

#precipitation factor
pf = '3.0'
path_oggm_results = r'E:\Results\OGGM\Global\past/fixed/'
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
    glacier_area_csv = pd.read_csv(path_preprocessed + 'all_glaciers_area_5min_new.csv')

example_nc = path_cellarea
glacier_outlet = pkl.load(open(path_preprocessed +'/glaciers_key_coordinates_terminus_{}.pkl'.format(resolution, resolution), "rb"))

# ------------- GENERATE GLACIER AREA INPUT FOR CWATM -------------------
#TODO this is slow for 30arcmin global and will be very slow for 5arcmin global
start_time = time.time()
glacier_pre_post_functions.oggm_area_to_cwatm_input_world(glacier_area_csv, all_regions[10:13], cellarea, 1990, 2019, path_output_results, "all_regions_mask_new_pf{}_{}".format(str(pf), resolution), example_nc,resolution, fraction=True, fixed_year=None, include_off_area = False)
end_time = time.time()
print("\ntime to run whole function " + str(end_time - start_time))

# --------------- GENERATE GLACIER MELT INPUT FOR CWATM -----------------
#TODO this is very slow
start_time = time.time()
glacier_pre_post_functions.oggm_output_to_cwatm_input_world(glacier_outlet, all_regions, pf, 1990, 2019, path_output_results, "all_regions_mask_pf{}_{}".format(str(pf), resolution), example_nc, resolution, include_off_area = False)
end_time = time.time()
print("\ntime to run whole function" + str(end_time - start_time))
#glacier_pre_post.oggm_output_to_cwatm_input_world(glacier_outlet_30min, all_regions[5:7], pf, 1990, 2019, path_output_results, " test_all_regions_mask_pf{}_{}".format(str(pf), resolution), example_nc_30arcmin, resolution, include_off_area = False)
