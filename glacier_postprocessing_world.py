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
from line_profiler import LineProfiler

# # ------------------ POSTPROCESSING FOR WHOLE WORLD
# ------------------- ENTER YOUR PATHS ---------------
# enter path to data folder, this can be different than git folder, because data does not fit on github
path_general = r'C:\Users\shanus\Processing\pipeline_oggm_cwatm/'
#resolution (5min or 30min)
resolution = '30min'
# put in output directory
path_output_results = 'C:/Users/shanus/Data/Glaciers_new/new_try/'

#make path for output directory if it does not exists
if not os.path.exists(path_output_results):
    os.makedirs(path_output_results)

#precipitation factor
pf = '3.0'
# path to OGGM results
# ATTENTION: OGGM outputs for the whole globe can be accessed via zenodo
path_oggm_results = r'D:\Results\OGGM\Global\past/fixed/'

# list all paths of OGGM results that should be processed to CWatM input, e.g. paths of different RGI region results
#TODO this looks different for a single basin
all_regions = glob.glob(path_oggm_results + 'historical_run_output_*_1990_2019_mb_real_daily_cte_pf_{}*'.format(str(pf)))
# make a list of the OGGM results files
oggm_results = []
for region in all_regions:
    oggm_results.append(xr.open_dataset(region))

# -----------------------------------------------------
# path to netcdf files of cell area
path_cellarea = path_general + '/cellarea/cellarea_{}.nc'.format(resolution)
cellarea = xr.open_dataset(path_cellarea)
# path to directory with preprocessed files generated with glacier_preprocessing.py
path_preprocessed = path_general + '/glaciers_preprocessed/{}/'.format(resolution)

# open the csv files which contains information about which glacier covers which grid cells and which grid cells contain the center and terminus of glacier
glacier_area_csv = pd.read_csv(path_preprocessed + 'glacier_area_df_{}.csv'.format(resolution))

# to generate the netcdf files for cwatm input, the cellarea file is used as example file for the structure
example_nc = path_cellarea
# dictionary with infor about which glacier has terminus in which grid cell
glacier_outlet = pkl.load(open(path_preprocessed +'/glaciers_key_coordinates_terminus_{}.pkl'.format(resolution), "rb"))

# what to process
process_runoff = False
process_area = False

# ------------- GENERATE GLACIER AREA INPUT FOR CWATM -------------------
# ATTENTION: if you postprocess the whole world this can be relatively slow, especially for 5arcmin
if process_area:
    start_time = time.time()
    glacier_postprocessing_functions.oggm_area_to_cwatm_input_world(glacier_area_csv, all_regions, cellarea, 1990, 2019, path_output_results + '{}_area/'.format(resolution), "all_regions_test_pf{}_{}".format(str(pf), resolution), example_nc, resolution, fraction=True, fixed_year=None, include_off_area = False)
    end_time = time.time()
    print("\ntime to run whole function " + str(end_time - start_time))

    #if you want to check how long the postprocessing takes

    # lp = LineProfiler()
    # lp.add_function(glacier_postprocessing_functions.change_area_world) # add additional function to profile
    # lp_wrapper = lp(glacier_postprocessing_functions.oggm_area_to_cwatm_input_world)
    # lp_wrapper(glacier_area_csv, all_regions, cellarea, 1990, 2019, path_output_results + '{}_area/'.format(resolution), "all_regions_pf{}_{}".format(str(pf), resolution), example_nc,resolution, fraction=True, fixed_year=None, include_off_area = False)
    # file2 = open(path_output_results + 'output_area_csv_ready_{}_all.txt'.format(resolution),"w+")
    # lp.print_stats(stream= file2)

# --------------- GENERATE GLACIER MELT INPUT FOR CWATM -----------------
if process_runoff:
    start_time = time.time()
    glacier_postprocessing_functions.oggm_output_to_cwatm_input_world(glacier_outlet, all_regions, float(pf), 1990, 1991, path_output_results, "rgi6_pf{}_{}".format(str(pf), resolution), example_nc, resolution, include_off_area = False)
    end_time = time.time()
    print("\ntime to run whole function" + str(end_time - start_time))

    #if you want to check how long the postprocessing takes
    # lp2 = LineProfiler()
    # lp2.add_function(glacier_postprocessing_functions.change_format_oggm_output_world) # add additional function to profile
    # lp2_wrapper = lp2(glacier_postprocessing_functions.oggm_output_to_cwatm_input_world)
    # lp2_wrapper(glacier_outlet, all_regions, float(pf), 1990, 2019, path_output_results, "all_regions_new_pf{}_{}".format(str(pf), resolution), example_nc, resolution, include_off_area = False)
    # file2 = open(path_output_results + 'output_liq_prcp_{}_all_regions_new2.txt'.format(resolution),"w+")
    # lp2.print_stats(stream= file2)