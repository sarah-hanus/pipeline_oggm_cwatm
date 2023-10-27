import numpy as np
import glacier_postprocessing_functions
import glob
import geopandas as gpd
import xarray as xr
import pandas as pd
import pickle as pkl
import os
import warnings
import time
# enter path to git repository

# ---------------- POST PROCESSING WITH OGGM RESULTS ----------------------

# -------------------------- BASIN -----------------------------
resolution = '5min'
pf = 3.0
CatchmentName = 'gloma_t_dw'
Catchment = 'gloma'
# give the rgi_region the basin is located in
rgi_region = ['08']
region = '08'

path_general = r'C:\Users\shanus\Processing\pipeline_oggm_cwatm/'
path_preprocessed = path_general + '/glaciers_preprocessed/{}/'.format(resolution)

# define output directory
path_output_results = 'C:/Users/shanus/Data/Glaciers_new/new_try/'

#make path for output directory if it does not exists
if not os.path.exists(path_output_results):
    os.makedirs(path_output_results)

#TODO provide example of example nc
example_nc = "C:/Users/shanus/Data/Climate_Data/ISIMIP3a/Gloma/Tavg_daily.nc"
#example_nc_30arcmin = [-180, 90, 720, 360, 1/2]

# open the csv files which contains information about which glacier covers which grid cells and which grid cells contain the center and terminus of glacier
glacier_area_csv = pd.read_csv(path_preprocessed + 'glacier_area_df_{}.csv'.format(resolution))

# path to netcdf files of cell area
path_cellarea = path_general + '/Data/cellarea/cellarea_{}.nc'.format(resolution)
cellarea = xr.open_dataset(path_cellarea)
# dictionary with infor about which glacier has terminus in which grid cell
glacier_outlet = pkl.load(open(path_preprocessed +'/glaciers_key_coordinates_terminus_{}.pkl'.format(resolution), "rb"))

#ATTENTION: an example OGGM output is given on github, more OGGM outputs can be accessed via Zenodo
path_oggm_results = path_general + '/Data/oggm_results/historical_run_output_{}_1990_2019_mb_real_daily_cte_pf_{}.nc'.format(CatchmentName, str(pf))
oggm_results = xr.open_dataset(path_oggm_results)

# what to process
process_runoff = True
process_area = False

# ------------- GENERATE GLACIER AREA INPUT FOR CWATM -------------------
if process_area:
    # 1) calculate area of oggm results
    glacier_postprocessing_functions.oggm_area_to_cwatm_input(glacier_area_csv, [path_oggm_results], cellarea, 1990, 2019,
                                                              path_output_results + '{}_area/'.format(resolution),"mask_{}_pf{}_{}".format(Catchment, str(pf), resolution),
                                                              example_nc, resolution, fraction=True,
                                                              fixed_year=None, include_off_area=False)

# --------------- GENERATE GLACIER MELT INPUT FOR CWATM -----------------
if process_runoff:
    #2) calculate melt of oggm results
    glacier_postprocessing_functions.oggm_output_to_cwatm_input(glacier_outlet, [path_oggm_results], float(pf), 1990,
                                                                2019, path_output_results,
                                                                      "mask_{}_pf{}_{}".format(Catchment, str(pf), resolution),
                                                                example_nc, resolution, include_off_area=False)


