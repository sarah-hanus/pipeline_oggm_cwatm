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

# ---------------- POST PROCESSING WITH OGGM RESULTS ----------------------

# -------------------------- BASIN -----------------------------

pf = 2.0
CatchmentName = 'rhone_t_dw'
Catchment = 'rhone'
rgi_region = ['11']
region = '11'
#example_nc = "C:/Users/shanus/Data/Climate_Data/ISIMIP3a/Rhone/Tavg_daily.nc"
example_nc_30arcmin = [-180, 90, 720, 360, 1/2]
#cell_area_5min = xr.open_dataset("C:/Users/shanus/Data/CWatM_Input/cwatm_input5min/landsurface/topo/cellarea.nc")

glacier_area_csv = pd.read_csv(path_output + 'glacier_area_df_{}.csv'.format(resolution))
# oggm_results = xr.open_dataset(r"C:\Users\shanus\Documents\Results\WP1\OGGM\{}/historical_run_output_{}_1990_2019_mb_real_daily_cte_pf_{}.nc".format(CatchmentName, CatchmentName, str(pf)))
#

# 1) calculate area of oggm results
glacier_pre_post.oggm_area_to_cwatm_input(glacier_area_csv, oggm_results, cellarea_30min, path_output_results, "{}_pf{}_{}".format(Catchment, str(pf), resolution), example_nc_30arcmin,resolution, fraction=True, fixed_year=None, include_off_area = False)

#TODO: whats going on with this? why is it incorporated into the oggm_output function? how was it generated?
#glacier_outlet_5min_world= pickle.load(open(r'C:\Users\shanus\Data\Glaciers\IDs_basins/glacier_coordinate_id_terminus_world2.pkl'.format(Catchment), "rb"))
glacier_outlet_30min= pkl.load(open(r'C:\Users\shanus\Data\Glaciers_new\glacier_files_output/glaciers_key_coordinates_terminus_{}.pkl'.format(resolution), "rb"))
#2) calculate melt of oggm results
#glacier_pre_post.oggm_output_to_cwatm_input(glacier_outlet_30min, [oggm_results_11, oggm_results_8], pf, 1990, 2019, path_output_results, "mask_{}_pf".format(region) + str(pf), example_nc_30arcmin, resolution, include_off_area = False, melt_or_prcp='prcp')
#glacier_pre_post.oggm_output_to_cwatm_input(glacier_outlet_30min, oggm_results, pf, 1990, 2019, path_output_results, "mask_{}_pf".format(region) + str(pf), example_nc_30arcmin, resolution, include_off_area = False, melt_or_prcp='melt')

# enter path to git repository
path_general = r'C:\Users\shanus\Processing\pipeline_oggm_cwatm'
#resolution (5min or 30min)
resolution = '30min'
# put in output directory
path_output_results = 'C:/Users/shanus/Data/Glaciers_new/try/'

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