import glacier_preprocessing_functions
import glob


# ------------------------------ PREPROCESSING ----------------------------
'''
preprocessing is independent of OGGM results and relates glacier IDs to grid cells and glacier area at RGI date (shapefiles) to grid cells.
This information is later used for transforming OGGM resutls to CWatM input.
'''

# enter path to data repo
path_general = 'C:/Users/shanus/Data/Glaciers_new/'

path_general = 'C:\Users\shanus\Processing\pipeline_oggm_cwatm/'
#location of glacier statistics
path_glacier_stats = path_general + 'Data/glacier_statistics/'
#directory to glacier statistics containing center and terminus location of grid cells
glacier_csv = glob.glob(path_glacier_stats + '/glacier_statistics_??.csv')

#which resolution to use: 5min or 30min
resolution = "30min"
path_output = path_general + '/glaciers_preprocessed/{}/'.format(resolution)

#rgi regions to use
# RGI region 19 is not used because these are only glaciers in Antarctica
rgi_regions = ['01','02', '03', '04', '05','06', '07','08', '09','10','11','12','13','14','15','16','17','18']

# path to shapefiles of 5arcmin or 30arcmin grid
if resolution == '5min':
    # ATTENTION THIS FILE IS AVAILABLE VIA ZENODO but not on github repo because of its size
    path_grid = path_general + "Data/grid_shp/grid_coordinates_5min_rgi_regions_ex19_4326.shp"
elif resolution == '30min':
    path_grid = path_general + "Data/grid_shp/grid_coordinates_30min.shp"

# ATTENTION THE GLACIER SHAPE FILES MUST BE DOWNLOADED FROM https://doi.org/10.7265/N5-RGI-60
# the shapefiles of various rgi regions can be merged
path_glacier_shape =  ''
#path_glacier_shape =  r'C:\Users\shanus\Data\Glaciers_new\rgi60\00_rgi60/rgi_60_all_rgi_regions.shp'

preprocessing = False

if preprocessing:
    # ------------------------------ STEP 1 -------------------------------------
    # 1) generate dictionaries of glacier ids as keys and th glacier terminus grid cell's coordinates as values
    # (the coordinates are the center of each grid cell, so depends on the resolution)
    glacier_preprocessing_functions.make_glacier_outlet_dict(glacier_csv, path_output, "glaciers", resolution, rgi_ids=None)
    # ------------------------------------------- STEP 2 ----------------------------------
    # overlay grid with glacier shape files to get the glacier area within each grid cell
    glacier_preprocessing_functions.overlay_area_grid(path_glacier_shape, path_grid,  path_output + '/rgi_regions_all_{}.shp'.format(resolution))
    # ------------------------------------------ STEP 3 -----------------------------------------
    # 3) generate csv file with info about glacier area in each grid cell
    glacier_preprocessing_functions.df_glacier_grid_area(path_output + '/rgi_regions_all_{}.shp'.format(resolution), path_output, "glaciers", "glacier_area_df", resolution)


