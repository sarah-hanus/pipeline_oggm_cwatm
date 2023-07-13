import pandas as pd
import pickle
import xarray as xr
import numpy as np
import netCDF4 as nc
import os
import math
import warnings
import glob
import geopandas as gpd
import time
from memory_profiler import profile
import datetime as dt
# -------------------------- FUNCTIONS ------------------------------------


#compare glaciers that are in area with glaciers_geodetic und benutze die schnittmenge

#function that gives the areal extent of glaciers at the rgi date in each grid cell
'''maybe this can be preprocessed for the whoel world and then the function just gets the values of the current glaciers'''

def transform_to_df(glacier_center, glacier_terminus, overlay_result, resolution):
    '''
    :param glacier_center: dictonary containing the center coordinates of the grid cell that contains center point of glacier
    :param glacier_terminus: dictonary containing the center coordinates of the grid cell that contains terminus point of glacier
    :param overlay_result: Dataframe resulting from overlaying the grid shapefile with the glacier shapfile;
                           contains information about Area of Glacier in each grid cell that is partly covered by the glacier
    :param resolution: "5min" or "30min"
    :return: Dataframe with RGIId, gridcell lat and lon, area of glacier in this girdcell and whether this gridcell contains terminus/center of the gridcell
             this dataframe can be used to calculate total glacier area per grid cell and can be updated when glacier areas change in OGGM
    '''
    df = pd.DataFrame()
    #rgiid
    df["RGIId"] = overlay_result.RGIId
    #gridcell
    df["Nr_Gridcell"] = overlay_result.FID
    df["Latitude"] = overlay_result.lat
    df["Longitude"] = overlay_result.lon
    if resolution == "5min":
        df["Latitude"] = np.round(overlay_result.lat, decimals=5)
        df["Longitude"] = np.round(overlay_result.lon, decimals=5)
    else:
        df["Latitude"] = overlay_result.lat
        df["Longitude"] = overlay_result.lon
    #area
    df["Area"] = overlay_result.Area_Cell
    # recorc in which grid cell center and terminus of glacier is located
    center = np.zeros(len(overlay_result))
    terminus = np.zeros(len(overlay_result))
    for i in range(0,len(overlay_result)):
        if resolution == "5min":
            overlay_lat = np.round(overlay_result.lat[i], decimals=5)
            overlay_lon = np.round(overlay_result.lon[i], decimals=5)
        else:
            overlay_lat = overlay_result.lat[i]
            overlay_lon = overlay_result.lon[i]
        try:
            # print(overlay_lat, glacier_center.get(overlay_result.RGIId[i])[0])
            # print("lon ", overlay_lon, glacier_center.get(overlay_result.RGIId[i])[1])
            if overlay_lat == glacier_center.get(overlay_result.RGIId[i])[0] and overlay_lon == glacier_center.get(overlay_result.RGIId[i])[1]:
                center[i] = 1
            if overlay_lat == glacier_terminus.get(overlay_result.RGIId[i])[0] and overlay_lon == glacier_terminus.get(overlay_result.RGIId[i])[1]:
                terminus[i] = 1
        except:
            print("An exception occurred for ", overlay_result.RGIId[i])
    #terminus yes, no?
    df["terminus"] = terminus
    df["center"] = center
    return df

def df_glacier_grid_area(grid, rgi_regions, path_glacier_info, name_glacier_info, path_rgi_files, name_output, resolution):
    '''
    makes one csv file containing all glaciers of the world and the corresponding gridcells etc,
    uses transform_df function
    :param grid: shapefile of grid
    :param rgi_regions: list of rgi region codes to be used
    :param path_glacier_info: path to dictionary with glacier info???
    :param name_glacier_info: name of dictionary ???
    :param path_rgi_files:
    :param name_output: output name of csv
    :param resolution: "5min" or "30min"
    :return:
    '''
    #path_glacier_info = "C:/Users/shanus/Data/Glaciers/glacier_id_dict/"
    glacier_center = pickle.load(open(path_glacier_info + name_glacier_info + "_key_id_center_{}.pkl".format(resolution), "rb"))
    glacier_terminus = pickle.load(open(path_glacier_info + name_glacier_info + "_key_id_terminus_{}.pkl".format(resolution), "rb"))
    for i, current_rgi in enumerate(rgi_regions):
        #print(current_rgi)
        overlay_54012_filename = path_glacier_info + '/rgi_region_{}_{}.shp'.format(current_rgi,resolution)
        if os.path.isfile(overlay_54012_filename):
            overlay_54012 = gpd.read_file(overlay_54012_filename)
        else:
            rgi_file = glob.glob(path_rgi_files + current_rgi + '*/*.shp')[0]
            rgi = gpd.read_file(rgi_file)

            # results from overlay functions
            overlay_result = gpd.overlay(rgi, grid, how='intersection')
            # #get area of each part
            overlay_54012 = overlay_result.to_crs('esri:54012')
            overlay_54012["Area_Cell"] = overlay_54012.area
            overlay_54012.to_file(
                path_glacier_info + '/rgi_region_{}_{}.shp'.format(current_rgi,
                                                                                                    resolution))
        if i == 0:
            df_all = transform_to_df(glacier_center, glacier_terminus, overlay_54012, resolution)
        else:
            df_all = pd.concat([df_all, transform_to_df(glacier_center, glacier_terminus, overlay_54012, "5min")])
        df_all.to_csv(path_glacier_info+name_output+'_{}.csv'.format(resolution))

#generate pkl files of glacier id and coordinates of terminus/center of glaciers
def make_glacier_outlet_dict(list_path_glacierstats, outpath, out_name_dict, resolution, rgi_ids=None):
    '''generates the glacier outlet dictionary with the glacier_outlet_dict function
    list_path_glacierstats: list of paths with rgi glacier statistics
    outpath:
    out_name:
    resolution: '5min' or '30min'
    rgi_ids: if rgi ids are given only generates it the dictionary for the given rgi_ids
    '''
    glacier_terminus_lon = []
    glacier_terminus_lat = []
    glacier_id = []
    glacier_rgi_area = []
    glacier_cen_lon = []
    glacier_cen_lat = []
    glacier_elv_terminus = []

    # loop through all RGI regions glacier statistics
    for i, path_rgi_reg in enumerate(list_path_glacierstats):
        glaciers = pd.read_csv(path_rgi_reg, low_memory=False)

        if rgi_ids:
            glacier_stats_basin = glaciers[np.isin(glaciers.rgi_id, rgi_ids)]
        else:
            glacier_stats_basin = glaciers
        glacier_terminus_lon.extend(glacier_stats_basin["terminus_lon"].values.tolist())
        glacier_terminus_lat.extend(glacier_stats_basin["terminus_lat"].values.tolist())
        glacier_id.extend(glacier_stats_basin["rgi_id"].values.tolist())
        glacier_rgi_area.extend(glacier_stats_basin["rgi_area_km2"].values.tolist())
        glacier_cen_lon.extend(glacier_stats_basin["cenlon"].values.tolist())
        glacier_cen_lat.extend(glacier_stats_basin["cenlat"].values.tolist())
        glacier_elv_terminus.extend(glacier_stats_basin["dem_min_elev"].values.tolist())

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    #generate a dictionary of rgi ids and coordinates
    glacier_outlet_dict(glacier_id, glacier_terminus_lat, glacier_terminus_lon, glacier_cen_lat, glacier_cen_lon, outpath, out_name_dict, resolution, 'id')
    glacier_outlet_dict(glacier_id, glacier_terminus_lat, glacier_terminus_lon, glacier_cen_lat, glacier_cen_lon,
                        outpath, out_name_dict, resolution, 'coordinates')
    # df_glacier_grid_area(grid, rgi_regions, outpath, out_name_dict, path_rgi_files, out_name_csv,
    #                      resolution)
# glacier_outlet_dict used by make_glacier_outlet_dict function
def glacier_outlet_dict(glacier_id, glacier_lat, glacier_lon, glacier_cen_lat, glacier_cen_lon, outpath, out_name, resolution, keys):
    '''makes a dictionary with keys (lat,lon) of gridcell with glaciers and
    items: RGIIDs of glaciers that have terminus in that grid cell
    Input:  - glacier_id: list of glacier ids
            - glacier_lon/glacier_lat: list of postion of terminus of glaciers corresponding to glacier ids
            - glacier_cen_lon/glacier_cen_lat: list of postion of center of glaciers corresponding to glacier ids
            - out_name: name of new dictionary
            - resolution of grid ("5min", "30min")
    '''
    if keys not in ["coordinates", "id"]:
        raise ValueError("keys should be coordinates or id")
    if resolution not in ["30min", "5min"]:
        raise ValueError("keys should be 30min or 5min")
    if resolution == "30min":
        # this function rounds the location of terminus of glaciers to the corresponding center of the grid cell
        def round_to_grid(x):
            return np.round(math.floor(x * 2) / 2 + 0.25, decimals=3)

    elif resolution == "5min":
        # this function rounds the location of terminus of glaciers to the corresponding center of the grid cell
        def round_to_grid(x):
            return np.round(math.floor(x * 12) / 12 + 1 / 24, decimals=3)
    #make a dictonary
    glaciers_dict = dict()
    # loop through the number of glaciers
    for i in range(len(glacier_lat)):
        # if there is value for the location of the terminus of the glacier
        if math.isnan(glacier_lat[i]) == False:
            # get grid cell of terminus of glacier
            gridcell_terminus = (round_to_grid(glacier_lat[i]), round_to_grid(glacier_lon[i]))

            if keys == "coordinates":
                if gridcell_terminus not in glaciers_dict:
                    glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
                else:
                    glaciers_dict[gridcell_terminus].append(glacier_id[i])
            elif keys == "id":
                glaciers_dict.update({glacier_id[i]: gridcell_terminus})

            # if gridcell_terminus not in glaciers_dict:
            #     glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
            # else:
            #     glaciers_dict[gridcell_terminus].append(glacier_id[i])
        else:
            if math.isnan(glacier_cen_lat[i]) == False:
                warnings.warn("No terminus location known for glacier {}. Center location used ".format(glacier_id[i]))
                gridcell_terminus = (round_to_grid(glacier_cen_lat[i]), round_to_grid(glacier_cen_lon[i]))
                if keys == "coordinates":
                    if gridcell_terminus not in glaciers_dict:
                        glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
                    else:
                        glaciers_dict[gridcell_terminus].append(glacier_id[i])
                elif keys == "id":
                    glaciers_dict.update({glacier_id[i]: gridcell_terminus})
                # if gridcell_terminus not in glaciers_dict:
                #     glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
                # else:
                #     glaciers_dict[gridcell_terminus].append(glacier_id[i])
            else:
                warnings.warn("No terminus or center location known for glacier {}. Glacier is disregarded".format(glacier_id[i]))
    glacier_ids_file = open(outpath+out_name+'_key_{}_terminus_{}.pkl'.format(keys, resolution), "wb")
    pickle.dump(glaciers_dict, glacier_ids_file)
    glacier_ids_file.close()

    glaciers_dict = dict()
    #center
    for i in range(len(glacier_cen_lat)):
        # if there is value for the location of the terminus of the glacier
        if math.isnan(glacier_cen_lat[i]) == False:
            # get grid cell of terminus of glacier
            gridcell_terminus = (round_to_grid(glacier_cen_lat[i]), round_to_grid(glacier_cen_lon[i]))
            if keys == "coordinates":
                if gridcell_terminus not in glaciers_dict:
                    glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
                else:
                    glaciers_dict[gridcell_terminus].append(glacier_id[i])
            elif keys == "id":
                glaciers_dict.update({glacier_id[i]: gridcell_terminus})
            # if gridcell_terminus not in glaciers_dict:
            #     glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
            # else:
            #     glaciers_dict[gridcell_terminus].append(glacier_id[i])
        else:
            if math.isnan(glacier_lat[i]) == False:
                warnings.warn("No center location known for glacier {}. Terminus location used ".format(glacier_id[i]))
                gridcell_terminus = (round_to_grid(glacier_lat[i]), round_to_grid(glacier_lon[i]))
                if keys == "coordinates":
                    if gridcell_terminus not in glaciers_dict:
                        glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
                    else:
                        glaciers_dict[gridcell_terminus].append(glacier_id[i])
                elif keys == "id":
                    glaciers_dict.update({glacier_id[i]: gridcell_terminus})

                # if gridcell_terminus not in glaciers_dict:
                #     glaciers_dict.update({gridcell_terminus: [glacier_id[i]]})
                # else:
                #     glaciers_dict[gridcell_terminus].append(glacier_id[i])
            else:
                warnings.warn("No terminus or center location known for glacier {}. Glacier is disregarded".format(glacier_id[i]))
    glacier_ids_file = open(outpath+out_name+'_key_{}_center_{}.pkl'.format(keys, resolution), "wb")
    pickle.dump(glaciers_dict, glacier_ids_file)
    glacier_ids_file.close()


#-------- POSTPROCESSING --------------------

def change_format_oggm_output(glacier_run_results_list, variable):
    '''Function changes the format of oggm output from 2d array (days,years) to a 1d timeseries'''
    #get start date and end date of OGGM runs
    for idx, glacier_run_results in enumerate(glacier_run_results_list):
        startyear = glacier_run_results.calendar_year.values[0]
        startmonth = glacier_run_results.calendar_month.values[0]
        endyear = glacier_run_results.calendar_year.values[-1]
        endmonth = glacier_run_results.calendar_month.values[-1]
        #construct a string of start time and endtime
        starttime = str(startyear) + '-' + str(startmonth) + '-01'
        endtime = str(endyear) + '-' + str(endmonth) + '-01'
        #construct a timeseries with daily timesteps
        #last day is not used because the data of 2020 is not there
        timeseries = pd.date_range(starttime, endtime, freq="D")[:-1]
        #concatenate the arrays
        #concatenate current year with next year
        # the last year in OGGM results is not used because it does not give proper results
        #TODO understand why this is
        for i in range(0,len(glacier_run_results.calendar_year)-2):
            if i == 0:
                all_months = np.concatenate((glacier_run_results[variable][i].values, glacier_run_results[variable][i+1].values),axis=0)
            else:
                all_months = np.concatenate((all_months, glacier_run_results[variable][i+1].values), axis=0)

        if idx == 0:
            all_months_all = all_months
        else:
            # TODO all months next has to be concatenated
            all_months_all = np.concatenate((all_months_all,all_months),axis=1)

            #delete the nan values on the 29th of february
    all_months_all = all_months_all[~np.isnan(all_months_all).any(axis=1), :]
    #assert that the timeseries and new dara have same length
    assert len(timeseries) == np.shape(all_months_all)[0]
    return all_months_all, timeseries

@profile
def change_format_oggm_output_world(oggm_results_list, startyear_df, endyear_df, variable):
    '''Function changes the format of oggm output from 2d array (days,years) to a 1d timeseries
    :param oggm_results_list: list of paths with oggm results'''
    # TODO can these lops be somehow represented in map functions instead??
    for idx, oggm_results_path in enumerate(oggm_results_list):
        print(oggm_results_path)

        oggm_results = xr.open_dataset(oggm_results_path)
        nok = oggm_results.volume.isel(time=0).isnull()
        # drop rgi_ids with nan values, so RGI IDs that were not correctly modelled in OGGM
        # the threshold was set to 10 days because end of modelling period it is always 0
        # check if OGGM results contains glaciers with missing results
        if np.count_nonzero(nok) > 0: #len(oggm_results.rgi_id) != len(oggm_results_new.rgi_id):
            oggm_results_new = oggm_results.dropna('rgi_id', thresh=10)
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results.rgi_id.values)) - set(list(oggm_results_new.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results.rgi_id) - len(oggm_results_new.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
            oggm_results.close()
            del oggm_results
            oggm_results = oggm_results_new
            del oggm_results_new

        startyear = oggm_results.calendar_year.values[0]
        startmonth = oggm_results.calendar_month.values[0]
        endyear = oggm_results.calendar_year.values[-1]
        endmonth = oggm_results.calendar_month.values[-1]
        #construct a string of start time and endtime
        starttime = str(startyear) + '-' + str(startmonth) + '-01'
        endtime = str(endyear) + '-' + str(endmonth) + '-01'
        #construct a timeseries with daily timesteps
        #last day is not used because the data of 2020 is not there
        timeseries = pd.date_range(starttime, endtime, freq="D")[:-1]
        #concatenate the arrays
        #concatenate current year with next year
        # the last year in OGGM results is not used because it does not give proper results
        diff_start = startyear_df - startyear
        diff_end = endyear - endyear_df
        starttime = str(startyear_df) + '-01-01'
        endtime = str(endyear_df) + '-12-31'
        timeseries = pd.date_range(starttime, endtime, freq="D")
        melt_below_zero = 0
        #TODO understand why this is
        for i in range(diff_start, diff_start + endyear_df - startyear_df + 1):
        #for i in range(0,len(oggm_results.calendar_year)-1):
            melt_year = oggm_results['melt{}'.format(variable)][i].values
            if np.min(melt_year) < 0:
                melt_below_zero = 1
            # CLLIP TO ZERO TO AVOID SMALL NEGATIVE VALUES
            #these negative values are due to mass conservation in oggm
            melt_year = melt_year.clip(min=0)
            rain_year = oggm_results['liq_prcp{}'.format(variable)][i].values
            if i == diff_start:
                all_months_melt = melt_year #np.concatenate((melt_year, melt_year),axis=0)
                all_months_rain = rain_year #np.concatenate((rain_year, rain_year), axis=0)
            else:
                all_months_melt = np.concatenate((all_months_melt, melt_year), axis=0)
                all_months_rain = np.concatenate((all_months_rain, rain_year), axis=0)

        if melt_below_zero == 1:
            msg = 'Original OGGM results has some days with melt below 0, this is due to mass conservation scheme in OGGM. melt was clipped to minimum 0'
            warnings.warn(msg)

        if idx == 0:
            glacier_ids = list(oggm_results.rgi_id.values)
        else:
            glacier_ids += list(oggm_results.rgi_id.values)
        oggm_results.close()
        del oggm_results

        if idx == 0:
            all_months_melt_all = all_months_melt
            all_months_rain_all = all_months_rain
        else:
            # TODO all months next has to be concatenated
            all_months_melt_all = np.concatenate((all_months_melt_all,all_months_melt),axis=1)
            all_months_rain_all = np.concatenate((all_months_rain_all, all_months_rain), axis=1)

            #delete the nan values on the 29th of february
    all_months_melt_all = all_months_melt_all[~np.isnan(all_months_melt_all).any(axis=1), :]
    all_months_rain_all = all_months_rain_all[~np.isnan(all_months_rain_all).any(axis=1), :]
    #assert that the timeseries and new dara have same length
    assert len(timeseries) == np.shape(all_months_melt_all)[0]
    assert len(timeseries) == np.shape(all_months_rain_all)[0]
    return all_months_melt_all, all_months_rain_all, glacier_ids, timeseries


def oggm_output_to_cwatm_input(glacier_outlet, oggm_results_org, pf, startyear, endyear, outpath, out_name, example_netcdf, resolution, include_off_area = False, melt_or_prcp = 'melt'):
    '''
    Function generates a netcdf of daily glacier melt using OGGM outputs. The outlet point of each glacier is fixed to the gridcell of the terminus as in RGI
        and will not change even if glacier retreats, as the effect is assumed to be minimal.

        glacier_outlet: a dictionary of all glaciers worldwide with keys coordinates of gridcells and values a list of glacier ids which drain into the gridcell
        (CAREFUL: you might want to adapt it to fit to the basin)
        oggm_results: netcdf file of daily results of an OGGM run
        pf: precipitation factor which was used to generate OGGM results
        startyear: startyear for which to create netcdf
        endyear: endyear for which to create netcdf
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :example_netcdf: example netcdf with same extent that you want to generate

        returns: metcdf with daily glacier melt (m3/d)
    '''
    if melt_or_prcp not in ["melt", "prcp"]:
        raise ValueError("melt_or_prcp should be melt or prcp")
    if melt_or_prcp == 'melt':
        var_name = 'melt'
        pf = 1
    elif melt_or_prcp == 'prcp':
        var_name = 'liq_prcp'
        pf = pf

    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    # drop rgi_ids with nan values, so RGI IDs that were not correctly modelled in OGGM
    # the threshold was set to 10 days because end of modelling period it is always 0
    #TODO this has to work for list of oggm resuults
    if isinstance(oggm_results_org, list):
        oggm_results = []
        for i, oggm_results_org_part in enumerate(oggm_results_org):
            # drop rgi_ids with nan values
            oggm_results_part = oggm_results_org_part.dropna('rgi_id', thresh=10)
            # check if OGGM results contains glaciers with missing results
            if len(oggm_results_org_part.rgi_id) != len(oggm_results_part.rgi_id):
                # get RGI IDs of missing values
                rgi_ids_nan = list(
                    set(list(oggm_results_org_part.rgi_id.values)) - set(list(oggm_results_part.rgi_id.values)))
                # get number of glaciers with nan values
                missing_rgis = len(oggm_results_org_part.rgi_id) - len(oggm_results_part.rgi_id)
                # print warning that netcdf will not be generated for the glaciers with missing OGGM results
                msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                    missing_rgis, rgi_ids_nan)
                warnings.warn(msg)
            oggm_results.append(oggm_results_part)
            if i == 0:
                glacier_ids = list(oggm_results_part.rgi_id.values)
            else:
                glacier_ids += list(oggm_results_part.rgi_id.values)
    else:
        # drop rgi_ids with nan values
        oggm_results = oggm_results_org.dropna('rgi_id', thresh=10)
        # check if OGGM results contains glaciers with missing results
        if len(oggm_results_org.rgi_id) != len(oggm_results.rgi_id):
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results_org.rgi_id.values)) - set(list(oggm_results.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results_org.rgi_id) - len(oggm_results.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
        #get ids of glaciers of oggm run output
        glacier_ids = list(oggm_results.rgi_id.values)
        oggm_results = [oggm_results]

    # define extent for which glacier maps should be created
    if type(example_netcdf) == str:
        example_nc = xr.open_dataset(example_netcdf)
        lat = np.round(example_nc.lat.values,
                       decimals=5)  # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = np.round(example_nc.lon.values,
                       decimals=5) # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth = np.round((lat[0] - lat[-1]) / (len(lat)-1), decimals = 5) #lat[0] -lat[1]
        cellnr_lat = len(lat)
        cellnr_lon = len(lon)
    #if you do not have an example netcdf file you need the cellsize, the number of cells and the coordinates in upper left corner
    elif len(example_netcdf) == 5:
        lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth = example_netcdf
        lon_max = lon_min + (cellnr_lon * cellwidth)
        lat_min = lat_max - (cellnr_lat * cellwidth)
        lon = np.round(np.arange(lon_min + cellwidth/2, lon_max, cellwidth), decimals=5)
        # #start latitude from -60 because there is only ocean below anyways
        lat = np.round(np.arange(lat_min + cellwidth/2, lat_max, cellwidth), decimals=5)
        cellnr_lon = np.round(cellnr_lon, decimals=5)
        cellnr_lat = np.round(cellnr_lat, decimals=5)
        cellwidth = np.round(cellwidth, decimals=5)

    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1/12
    np.testing.assert_almost_equal(cellwidth, cellwidth_res, decimal=4,
                                   err_msg='example_nc and resolution need to have same resolution', verbose=True)


    #change output of OGGM from 2darray (years, daysofyear, glaciers) to continous timeseries
    flux_on_glacier, timeseries_glacier = change_format_oggm_output(oggm_results, '{}_on_glacier_daily'.format(var_name))
    if include_off_area:
        flux_off_glacier, _ = change_format_oggm_output(oggm_results, '{}_off_glacier_daily'.format(var_name))

    #get start and end date corresponding to the inputs of the function
    start_index = np.where(timeseries_glacier.year == startyear)[0][0]
    end_index = np.where(timeseries_glacier.year == endyear)[0][-1]
    timeseries = pd.date_range(timeseries_glacier[start_index].strftime('%Y-%m'), timeseries_glacier[end_index],freq='D')

    #create netcdf file for each variable (maybe only do this for melt_on, liq_prcp_on because we do not need the others)
    name = '{}_on'.format(var_name)
    if include_off_area:
        ds = nc.Dataset(outpath +name[:-2]  + 'total_'+ out_name +'.nc', 'w', format='NETCDF4')
    else:
        ds = nc.Dataset(outpath + name+ '_' + out_name + '.nc', 'w', format='NETCDF4')
    # add dimenstions, specify how long they are
    # use 0.5° grid
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', None)
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"
    #create variables glacier melt on off, liquid precipitation on off
    if include_off_area:
        var_nc = ds.createVariable(name[:-2] + 'total', 'f4', ('time', 'lat', 'lon',),zlib=True, least_significant_digit=1)
    else:
        var_nc = ds.createVariable(name, 'f4', ('time', 'lat', 'lon',),zlib=True, least_significant_digit=1)
    var_nc.units = "m3/d"
    #use the extent of the example netcdf
    #TODO maybe no example netcdf needed but it can be done from scratch
    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)
    timeseries_str = str(timeseries)
    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    #now total melt input for timeseries
    #output is in kg/day -> transform to m3/day by dividing by 1000
    glacier_flux_on = flux_on_glacier[start_index:end_index+1]
    if include_off_area:
        glacier_flux_off = flux_off_glacier[start_index:end_index + 1]
        glacier_flux_on = glacier_flux_on / 1000 / pf + glacier_flux_off / 1000 / pf
    else:
        glacier_flux_on = glacier_flux_on / 1000 / pf

    # get gridcells of glaciers in right format
    gridcell_glaciers = list(glacier_outlet.keys())
    #get latitude and longitude of grid cells into which glaciers drain
    grid_lat = [a_tuple[0] for a_tuple in gridcell_glaciers]
    grid_lon = [a_tuple[1] for a_tuple in gridcell_glaciers]

    #define dataframe to store results, index are the days in timeseries
    df_flux_on = pd.DataFrame(index=timeseries)

    # TODO: see if it makes sense to mask outlet by correct basin outline
    x = np.zeros((cellnr_lat, cellnr_lon))
    x = x.astype(int)
    glacier_flux_on_array = x
    glacier_flux_on_array = glacier_flux_on_array.flatten()
    assert len(timeseries) == np.shape(df_flux_on)[0]


    #loop through all gridcells with glaciers in basin and sum up the melt of all glaciers in each grid cell
    #if glacier_outlet for the whole world, first constrain it to basin
    keys_gridcells = list(glacier_outlet)
    list_rgi_ids_world = list(glacier_outlet.values())
    count_gl_not_oggm_results = 0
    for i, rgi_ids_world in enumerate(list_rgi_ids_world):
        if np.isin(rgi_ids_world, glacier_ids).any():
            grid_lat = keys_gridcells[i][0]
            grid_lon = keys_gridcells[i][1]
            ids_gridcell = glacier_outlet[keys_gridcells[i]]


    # for gridcell in range(len(grid_lat)):
            daily_flux_on = 0
        #loop through all glaciers in the gridcell by getting the items in the dict of this grid cell
            for id in ids_gridcell: #list(glacier_outlet.items())[gridcell][1]:
                #assert that there are no nan values
                #assert np.sum(np.nonzero(np.isnan(glacier_melt[:, glacier_ids.index(id)]))) == 0
                #if there are no nan values in timeseries
                # if id o

                if id not in glacier_ids:
                    #if glacier id of glacier that drains into the basin is not modelled in OGGM, raise ERROR
                    # THIS CAN BE TURNED OFF, IF YOU ONLY WANT TO MODEL SOME GLACIERS OR SO
                    #TODO make this better
                    msg = 'Glacier {} was not found in OGGM results'.format(id)
                    #raise ValueError(msg)
                    warnings.warn(msg)
                    count_gl_not_oggm_results += 1
                    #if no nan values exist, then sum up the variables of all glaciers in the gridcell
                elif np.sum(np.nonzero(np.isnan(glacier_flux_on[:, glacier_ids.index(id)]))) == 0:
                    # sum up timeseries of all glaciers in gridcell
                    daily_flux_on += glacier_flux_on[:, glacier_ids.index(id)]
                else:
                    raise ValueError('Nan values encountered in timeseries of {} for variable {}'.format(id, glacier_flux_on))

            #daily melt volumes have to be stored in a datafram with column names being the gridcell lat, lon
            round_lat = np.round(grid_lat, decimals=3)
            round_lon = np.round(grid_lon, decimals=3)

            cell_lat = (np.max(lat) - round_lat) / cellwidth
            cell_lon = (round_lon - np.min(lon)) / cellwidth
            if cell_lat % 1 <  0.1 or cell_lat % 1 >  0.9:
                cell_lat = int(np.round(cell_lat))
            else:
                print(cell_lat)
                raise ValueError
            if cell_lon % 1 < 0.1  or cell_lon % 1 >  0.9:
                cell_lon = int(np.round(cell_lon))
            else:
                print(cell_lon)
                raise ValueError

            ind_cell = (cell_lat) * cellnr_lon + cell_lon
            if ind_cell <= len(glacier_flux_on_array):
                df_flux_on[ind_cell] = daily_flux_on
            else:
                warnings.warn("The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")
    glacier_gridcell_index = df_flux_on.columns
    if count_gl_not_oggm_results != 0:
        warnings.warn(
            "{} glaciers were not found in OGGM results but are in a grid cell for which other glaciers were modelled in OGGM. Check carefully".format(count_gl_not_oggm_results))
    for i in range(len(timeseries)):
        glacier_flux_on_array[glacier_gridcell_index] = df_flux_on.iloc[i, :].values
        # glacier_flux_on[glacier_ids] = np.ones(len(glacier_ids))
        glacier_flux_on_2d = np.reshape(glacier_flux_on_array, (cellnr_lat, cellnr_lon))
        var_nc[i, :, :] = glacier_flux_on_2d
    ds.close()

@profile
def oggm_output_to_cwatm_input_world(glacier_outlet, oggm_results_path, pf_sim, startyear, endyear, outpath, out_name, example_netcdf, resolution, include_off_area = False):
    '''
    Function generates a netcdf of daily glacier melt using OGGM outputs. The outlet point of each glacier is fixed to the gridcell of the terminus as in RGI
        and will not change even if glacier retreats, as the effect is assumed to be minimal.

        glacier_outlet: a dictionary of all glaciers worldwide with keys coordinates of gridcells and values a list of glacier ids which drain into the gridcell
        (CAREFUL: you might want to adapt it to fit to the basin)
        oggm_results: netcdf file of daily results of an OGGM run
        pf: precipitation factor which was used to generate OGGM results
        startyear: startyear for which to create netcdf
        endyear: endyear for which to create netcdf
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :example_netcdf: example netcdf with same extent that you want to generate

        returns: metcdf with daily glacier melt (m3/d)
    '''

    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    # define extent for which glacier maps should be created
    if type(example_netcdf) == str:
        example_nc = xr.open_dataset(example_netcdf)
        lat = np.round(example_nc.lat.values,
                       decimals=5)  # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = np.round(example_nc.lon.values,
                       decimals=5) # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth = np.round((lat[0] - lat[-1]) / (len(lat)-1), decimals = 5) #lat[0] -lat[1]
        cellnr_lat = len(lat)
        cellnr_lon = len(lon)
    #if you do not have an example netcdf file you need the cellsize, the number of cells and the coordinates in upper left corner
    elif len(example_netcdf) == 5:
        lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth = example_netcdf
        lon_max = lon_min + (cellnr_lon * cellwidth)
        lat_min = lat_max - (cellnr_lat * cellwidth)
        lon = np.round(np.arange(lon_min + cellwidth/2, lon_max, cellwidth), decimals=5)
        # #start latitude from -60 because there is only ocean below anyways
        lat = np.round(np.arange(lat_min + cellwidth/2, lat_max, cellwidth), decimals=5)
        cellnr_lon = np.round(cellnr_lon, decimals=5)
        cellnr_lat = np.round(cellnr_lat, decimals=5)
        cellwidth = np.round(cellwidth, decimals=5)

    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1/12
    np.testing.assert_almost_equal(cellwidth, cellwidth_res, decimal=4,
                                   err_msg='example_nc and resolution need to have same resolution', verbose=True)


    #change output of OGGM from 2darray (years, daysofyear, glaciers) to continous timeseries
    melt_on_glacier, rain_on_glacier, glacier_ids, timeseries_glacier = change_format_oggm_output_world(oggm_results_path, startyear, endyear, '_on_glacier_daily')
    if include_off_area:
        melt_off_glacier, rain_off_glacier, glacier_ids, _ = change_format_oggm_output_world(oggm_results_path, startyear, endyear, '_off_glacier_daily')

    #get start and end date corresponding to the inputs of the function
    start_index = np.where(timeseries_glacier.year == startyear)[0][0]
    end_index = np.where(timeseries_glacier.year == endyear)[0][-1]
    timeseries = pd.date_range(timeseries_glacier[start_index].strftime('%Y-%m'), timeseries_glacier[end_index],freq='D')
    vars = ['melt', 'liq_prcp']
    pfs = [1, pf_sim]
    for k, flux_on_glacier in enumerate([melt_on_glacier, rain_on_glacier]):
        var_name = vars[k]
        pf = pfs[k]
        #create netcdf file for each variable (maybe only do this for melt_on, liq_prcp_on because we do not need the others)
        name = '{}_on'.format(var_name)
        if include_off_area:
            ds = nc.Dataset(outpath +name[:-2]  + 'total_'+ out_name +'.nc', 'w', format='NETCDF4')
        else:
            ds = nc.Dataset(outpath + name+ '_' + out_name + '.nc', 'w', format='NETCDF4')
        # add dimenstions, specify how long they are
        # use 0.5° grid
        lat_dim = ds.createDimension('lat', len(lat))
        lon_dim = ds.createDimension('lon', len(lon))
        time_dim = ds.createDimension('time', None)
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        time_date = ds.createVariable('time', 'f4', ('time',))
        time_date.units = "days since 1961-01-01"
        time_date.calendar = "standard"
        #create variables glacier melt on off, liquid precipitation on off
        if include_off_area:
            var_nc = ds.createVariable(name[:-2] + 'total', 'f4', ('time', 'lat', 'lon',),zlib=True, least_significant_digit=1)
        else:
            var_nc = ds.createVariable(name, 'f4', ('time', 'lat', 'lon',),zlib=True, least_significant_digit=1)
        var_nc.units = "m3/d"
        #use the extent of the example netcdf
        #TODO maybe no example netcdf needed but it can be done from scratch
        lats[:] = np.sort(lat)[::-1]
        lons[:] = np.sort(lon)
        timeseries_str = str(timeseries)
        time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

        #now total melt input for timeseries
        #output is in kg/day -> transform to m3/day by dividing by 1000
        #TODO: why is there a divison by pf??? this should only be thee case for liquid precipitation??
        glacier_flux_on = flux_on_glacier[start_index:end_index+1]
        if include_off_area:
            glacier_flux_off = [melt_off_glacier, rain_off_glacier][k][start_index:end_index + 1]
            glacier_flux_on = glacier_flux_on / 1000 / pf + glacier_flux_off / 1000 / pf
        else:
            glacier_flux_on = glacier_flux_on / 1000 / pf

        # get gridcells of glaciers in right format
        gridcell_glaciers = list(glacier_outlet.keys())
        #get latitude and longitude of grid cells into which glaciers drain
        grid_lat = [a_tuple[0] for a_tuple in gridcell_glaciers]
        grid_lon = [a_tuple[1] for a_tuple in gridcell_glaciers]

        #define dataframe to store results, index are the days in timeseries
        df_flux_on = pd.DataFrame(index=timeseries)

        # TODO: see if it makes sense to mask outlet by correct basin outline
        x = np.zeros((cellnr_lat, cellnr_lon))
        x = x.astype(int)
        glacier_flux_on_array = x
        glacier_flux_on_array = glacier_flux_on_array.flatten()
        assert len(timeseries) == np.shape(df_flux_on)[0]


        #loop through all gridcells with glaciers in basin and sum up the melt of all glaciers in each grid cell
        #if glacier_outlet for the whole world, first constrain it to basin
        keys_gridcells = list(glacier_outlet)
        list_rgi_ids_world = list(glacier_outlet.values())
        count_gl_not_oggm_results = 0
        #TODO can these for loops be put into map functions instead??
        #loop through all gridcells of world

        #less_than_zero = list(filter(lambda x: np.isin(x, glacier_ids).any(), list_rgi_ids_world))

        # loop through grid cells with glaciers and check if glaciers are in current results
        cell_lat_all_gridcells = []
        cell_lon_all_gridcells = []
        for i, rgi_ids_world in enumerate(list_rgi_ids_world):
            #check if any glacier in these grid cell is in glacier_ids
            if np.isin(rgi_ids_world, glacier_ids).any():
                grid_lat = keys_gridcells[i][0]
                grid_lon = keys_gridcells[i][1]
                ids_gridcell = glacier_outlet[keys_gridcells[i]]


        # for gridcell in range(len(grid_lat)):
                daily_flux_on = 0
            #loop through all glaciers in the gridcell by getting the items in the dict of this grid cell
                for id in ids_gridcell: #list(glacier_outlet.items())[gridcell][1]:
                    #assert that there are no nan values
                    #assert np.sum(np.nonzero(np.isnan(glacier_melt[:, glacier_ids.index(id)]))) == 0
                    #if there are no nan values in timeseries
                    # if id o

                    if id not in glacier_ids:
                        #if glacier id of glacier that drains into the basin is not modelled in OGGM, raise ERROR
                        # THIS CAN BE TURNED OFF, IF YOU ONLY WANT TO MODEL SOME GLACIERS OR SO
                        #TODO make this better
                        msg = 'Glacier {} was not found in OGGM results'.format(id)
                        #raise ValueError(msg)
                        warnings.warn(msg)
                        count_gl_not_oggm_results += 1
                        #if no nan values exist, then sum up the variables of all glaciers in the gridcell
                    elif np.sum(np.nonzero(np.isnan(glacier_flux_on[:, glacier_ids.index(id)]))) == 0:
                        # sum up timeseries of all glaciers in gridcell
                        daily_flux_on += glacier_flux_on[:, glacier_ids.index(id)]
                    else:
                        raise ValueError('Nan values encountered in timeseries of {} for variable {}'.format(id, glacier_flux_on))

                #daily melt volumes have to be stored in a datafram with column names being the gridcell lat, lon
                round_lat = np.round(grid_lat, decimals=3)
                round_lon = np.round(grid_lon, decimals=3)

                cell_lat = (np.max(lat) - round_lat) / cellwidth
                cell_lon = (round_lon - np.min(lon)) / cellwidth

                if cell_lat % 1 <  0.1 or cell_lat % 1 >  0.9:
                    cell_lat = int(np.round(cell_lat))
                else:
                    print(cell_lat)
                    raise ValueError
                if cell_lon % 1 < 0.1  or cell_lon % 1 >  0.9:
                    cell_lon = int(np.round(cell_lon))
                else:
                    print(cell_lon)
                    raise ValueError

                cell_lat_all_gridcells.append(cell_lat)
                cell_lon_all_gridcells.append(cell_lon)

                ind_cell = (cell_lat) * cellnr_lon + cell_lon
                if ind_cell <= len(glacier_flux_on_array):
                    #df_flux_on[ind_cell] = daily_flux_on #makes new column for grid cell
                    df_flux_on = pd.concat([df_flux_on, pd.DataFrame(daily_flux_on, columns=[ind_cell], index=timeseries)], axis=1)
                else:
                    warnings.warn("The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")

        glacier_gridcell_index = df_flux_on.columns
        if count_gl_not_oggm_results != 0:
            warnings.warn(
                "{} glaciers were not found in OGGM results but are in a grid cell for which other glaciers were modelled in OGGM. Check carefully".format(count_gl_not_oggm_results))
        print("start putting results into netcdf")
        #TODO this is what takes longest, so maybe try to work on this with map?

        # decide whether to loop through time or number of grid cells depending on length of timeseries vs number of glaciers

        # if df_flux_on.shape[1] < len(timeseries):
        #     for i in range(df_flux_on.shape[1]):
        #         #does it make sense to directly write it to the outpu netcdf??
        #         var_nc[:, cell_lat_all_gridcells[i], cell_lon_all_gridcells[i]] = df_flux_on.iloc[:, i].values
        # else:
        for i in range(len(timeseries)):
            #array with all gridcells of the world, the index is used to add glacier melt to correct grid cell
            glacier_flux_on_array[glacier_gridcell_index] = df_flux_on.iloc[i, :].values
            # glacier_flux_on[glacier_ids] = np.ones(len(glacier_ids))
            glacier_flux_on_2d = np.reshape(glacier_flux_on_array, (cellnr_lat, cellnr_lon))
            #for each time step produce the 2d array of glacier flux
            var_nc[i, :, :] = glacier_flux_on_2d
        ds.close()
        del ds


def oggm_area_to_cwatm_input(glacier_area_csv, oggm_results_org, cell_area, outpath, out_name, example_netcdf, resolution, fraction=True, fixed_year=None, include_off_area = False):
    '''
    This function generates a netcdf file of the area (area fraction) covered by glacier in each gridcell.
    Note that only the glaciers which were run by OGGM will be taken into account, which are normally the glaciers that drain into the basin at the model resolution.
    Note that only the area of the glacier within the gridded basin outline will be used, so that the total area is likely lower than the total area in OGGM results.

        :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
        :param oggm_results: results from oggm run, can either be
        :param cell_area: netcdf with global cell_area at output resolution
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :param example_netcdf: example netcdf with same extent that you want to generate
        :param fraction: if netcdf should contain area fraction of gridcell covered by glacier
        :param fixed_year: if also netcdf file for a fixed year should be generated
        :param include_off_area: whether to use constant area in OGGM or variable area

        returns: netcdf file of glacier areas per year
    '''
    #if outpath does not exist make it
    if not os.path.exists(outpath):
        os.makedirs(outpath)




    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    if isinstance(oggm_results_org, list):
        oggm_results = []
        for oggm_results_org_part in oggm_results_org:
            #drop rgi_ids with nan values
            oggm_results_part = oggm_results_org_part.dropna('rgi_id', thresh=10)
            #check if OGGM results contains glaciers with missing results
            if len(oggm_results_org_part.rgi_id) != len(oggm_results_part.rgi_id):
                #get RGI IDs of missing values
                rgi_ids_nan = list(set(list(oggm_results_org_part.rgi_id.values)) - set(list(oggm_results_part.rgi_id.values)))
                #get number of glaciers with nan values
                missing_rgis = len(oggm_results_org_part.rgi_id) - len(oggm_results_part.rgi_id)
                #print warning that netcdf will not be generated for the glaciers with missing OGGM results
                msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(missing_rgis, rgi_ids_nan)
                warnings.warn(msg)
            oggm_results.append(oggm_results_part)

    else:
        # drop rgi_ids with nan values
        oggm_results = oggm_results_org.dropna('rgi_id', thresh=10)
        # check if OGGM results contains glaciers with missing results
        if len(oggm_results_org.rgi_id) != len(oggm_results.rgi_id):
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results_org.rgi_id.values)) - set(list(oggm_results.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results_org.rgi_id) - len(oggm_results.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
        oggm_results = [oggm_results]

    # define extent for which glacier maps should be created
    #if an example netcdf is given
    if type(example_netcdf) == str:
        # open it
        example_nc = xr.open_dataset(example_netcdf)
        #  get the coordinates, the cell width and the number of cells
        lat = np.round(example_nc.lat.values,
                       decimals=5)  # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = np.round(example_nc.lon.values,
                       decimals=5) # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth = np.round((lat[0] - lat[-1]) / (len(lat)-1), decimals = 5) #lat[0] -lat[1]
        cellnr_lat = len(lat)
        cellnr_lon = len(lon)
    #if you do not have an example netcdf file you need the cellsize, the number of cells and the coordinates in upper left corner
    elif len(example_netcdf) == 5:
        lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth = example_netcdf
        lon_max = lon_min + (cellnr_lon * cellwidth)
        lat_min = lat_max - (cellnr_lat * cellwidth)
        lon = np.round(np.arange(lon_min + cellwidth/2, lon_max, cellwidth), decimals=5)
        # #start latitude from -60 because there is only ocean below anyways
        lat = np.round(np.arange(lat_min + cellwidth/2, lat_max, cellwidth), decimals=5)
        cellnr_lon = np.round(cellnr_lon, decimals=5)
        cellnr_lat = np.round(cellnr_lat, decimals=5)
        cellwidth = np.round(cellwidth, decimals=5)
    # if neither of it is giving, the function does not work
    else:
        msg = 'The input {} is not a valid input. EIther give path to an example netcdf file or provide ' \
              'lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth of the upper left corner of model domain'.format(example_nc)
        raise ValueError(msg)

    x = np.zeros((cellnr_lat, cellnr_lon))
    glacier_on_area_array = x
    glacier_on_area_array = glacier_on_area_array.flatten()

    #define mask attributes from example_netcdf
    mask_attributes = [np.min(lon), np.max(lat), cellwidth, cellnr_lon, cellnr_lat]

    #get area of each gridcell covered by glacier for the timeperiod of OGGM run
    # check if cell_area and example_nc have the same cellwidth (same resolution)
    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1/12
    np.testing.assert_almost_equal(cellwidth, cellwidth_res, decimal=4,
                                   err_msg='example_nc and resolution need to have same resolution', verbose=True)
    if fraction == True:
        #check if cell_area and example_nc have the same cellwidth (same resolution)
        np.testing.assert_almost_equal(cellwidth, abs(cell_area.lon[1].values -cell_area.lon[0].values), decimal=4, err_msg='example_nc and cell_area need to have same resolution', verbose=True)
        area_gl_gridcell = change_area(glacier_area_csv, oggm_results, mask_attributes, include_off_area = include_off_area, cell_area=cell_area)
    else:
        area_gl_gridcell = change_area(glacier_area_csv, oggm_results, mask_attributes, include_off_area = include_off_area)

    timeseries = pd.date_range(str(area_gl_gridcell.columns.values[0]), str(area_gl_gridcell.columns.values[-1]),freq='AS')

    assert len(timeseries) == np.shape(area_gl_gridcell)[1]


    #create netcdf file for variable
    if include_off_area:
        label = "total_area"
    else:
        label = "on_area"
    if fraction == True:
        ds = nc.Dataset(outpath + label + '_fraction_' + out_name +'.nc', 'w', format='NETCDF4')
    else:
        ds = nc.Dataset(outpath + label + '_' + out_name + '.nc', 'w', format='NETCDF4')
    # add dimenstions, specify how long they are
    # use 0.5° grid
    #dimensions are created based on example netcdf
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', None)
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"
    #create variables glacier area
    var_nc = ds.createVariable(label, 'f4', ('time', 'lat', 'lon',),zlib=True)
    if fraction:
        var_nc.units = "fraction of cell_area"
    else:
        var_nc.units = "m2" #units is m2

    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)
    timeseries_str = str(timeseries)
    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    glacier_ids = area_gl_gridcell.index

    for i in range(len(timeseries)):
        glacier_on_area_array[glacier_ids] = area_gl_gridcell.iloc[:,i].values
        # glacier_on_area[glacier_ids] = np.ones(len(glacier_ids))
        glacier_on_area_2d = np.reshape(glacier_on_area_array, (cellnr_lat, cellnr_lon))
        var_nc[i, :, :] = glacier_on_area_2d
    ds.close()

    if fixed_year:
        if fraction == True:
            ds_fixed = nc.Dataset(outpath + label + '_fraction_' + out_name + '_constant_' + str(fixed_year) + '.nc',
                                  'w', format='NETCDF4')
        else:
            ds_fixed = nc.Dataset(outpath + label + '_' + out_name + '_constant_' + str(fixed_year) + '.nc', 'w',
                                  format='NETCDF4')

        lat_dim_fixed = ds_fixed.createDimension('lat', len(lat))
        lon_dim_fixed = ds_fixed.createDimension('lon', len(lon))
        lats_fixed = ds_fixed.createVariable('lat', 'f4', ('lat',))
        lons_fixed = ds_fixed.createVariable('lon', 'f4', ('lon',))
        # create variables glacier melt on off, liquid precipitation on off
        var_fixed_nc = ds_fixed.createVariable(label, 'f4', ('lat', 'lon',),zlib=True)
        if fraction:
            var_fixed_nc.units = "fraction of cell_area"
        else:
            var_fixed_nc.units = "m2"  # units is m2

        lats_fixed[:] = np.sort(lat)[::-1]
        lons_fixed[:] = np.sort(lon)

        x = np.zeros((cellnr_lat, cellnr_lon))
        glacier_on_area = x
        glacier_on_area = glacier_on_area.flatten()
        glacier_ids = area_gl_gridcell.index

        glacier_on_area[glacier_ids] = area_gl_gridcell.iloc[:,np.argwhere(area_gl_gridcell.columns == fixed_year)[0][0]].values
        glacier_on_area_2d = np.reshape(glacier_on_area, (cellnr_lat, cellnr_lon))
        var_fixed_nc[:, :] = glacier_on_area_2d

        ds_fixed.close()

@profile
def oggm_area_to_cwatm_input_world(glacier_area_csv, oggm_results_path, cell_area, startyear, endyear, outpath, out_name, example_netcdf,
                             resolution, fraction=True, fixed_year=None, include_off_area=False):
    '''
    This function generates a netcdf file of the area (area fraction) covered by glacier in each gridcell.
    Note that only the glaciers which were run by OGGM will be taken into account, which are normally the glaciers that drain into the basin at the model resolution.
    Note that only the area of the glacier within the gridded basin outline will be used, so that the total area is likely lower than the total area in OGGM results.

        :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
        :param oggm_results: results from oggm run, can either be
        :param cell_area: netcdf with global cell_area at output resolution
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :param example_netcdf: example netcdf with same extent that you want to generate
        :param fraction: if netcdf should contain area fraction of gridcell covered by glacier
        :param fixed_year: if also netcdf file for a fixed year should be generated
        :param include_off_area: whether to use constant area in OGGM or variable area

        returns: netcdf file of glacier areas per year
    '''
    # if outpath does not exist make it
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    # define extent for which glacier maps should be created
    # if an example netcdf is given
    if type(example_netcdf) == str:
        # open it
        example_nc = xr.open_dataset(example_netcdf)
        #  get the coordinates, the cell width and the number of cells
        lat = np.round(example_nc.lat.values,
                       decimals=5)  # np.round(np.arange(end_lat - 1 / 24, start_lat - 1 / 24, -1 / 12), decimals=3)
        lon = np.round(example_nc.lon.values,
                       decimals=5)  # np.round(np.arange(start_lon + 1 / 24, end_lon - 1 / 24, 1 / 12), decimals=3)
        cellwidth = np.round((lat[0] - lat[-1]) / (len(lat) - 1), decimals=5)  # lat[0] -lat[1]
        cellnr_lat = len(lat)
        cellnr_lon = len(lon)
    # if you do not have an example netcdf file you need the cellsize, the number of cells and the coordinates in upper left corner
    elif len(example_netcdf) == 5:
        lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth = example_netcdf
        lon_max = lon_min + (cellnr_lon * cellwidth)
        lat_min = lat_max - (cellnr_lat * cellwidth)
        lon = np.round(np.arange(lon_min + cellwidth / 2, lon_max, cellwidth), decimals=5)
        # #start latitude from -60 because there is only ocean below anyways
        lat = np.round(np.arange(lat_min + cellwidth / 2, lat_max, cellwidth), decimals=5)
        cellnr_lon = np.round(cellnr_lon, decimals=5)
        cellnr_lat = np.round(cellnr_lat, decimals=5)
        cellwidth = np.round(cellwidth, decimals=5)
    # if neither of it is giving, the function does not work
    else:
        msg = 'The input {} is not a valid input. Either give path to an example netcdf file or provide ' \
              'lon_min, lat_max, cellnr_lon, cellnr_lat, cellwidth of the upper left corner of model domain'.format(example_netcdf)
        raise ValueError(msg)

    x = np.zeros((cellnr_lat, cellnr_lon))
    glacier_on_area_array = x
    glacier_on_area_array = glacier_on_area_array.flatten()

    # define mask attributes from example_netcdf
    mask_attributes = [np.min(lon), np.max(lat), cellwidth, cellnr_lon, cellnr_lat]

    # get area of each gridcell covered by glacier for the timeperiod of OGGM run
    # check if cell_area and example_nc have the same cellwidth (same resolution)
    if resolution == '30min':
        cellwidth_res = 0.5
    elif resolution == '5min':
        cellwidth_res = 1 / 12
    np.testing.assert_almost_equal(cellwidth, cellwidth_res, decimal=4,
                                   err_msg='example_nc and resolution need to have same resolution', verbose=True)
    if fraction == True:
        # check if cell_area and example_nc have the same cellwidth (same resolution)
        np.testing.assert_almost_equal(cellwidth, abs(cell_area.lon[1].values - cell_area.lon[0].values), decimal=4,
                                       err_msg='example_nc and cell_area need to have same resolution', verbose=True)
        area_gl_gridcell = change_area_world(glacier_area_csv, oggm_results_path, mask_attributes, startyear, endyear, outpath + out_name,
                                       include_off_area=include_off_area, cell_area=cell_area)
    #TODO also for fraction = False cell area has to be given
    else:
        area_gl_gridcell = change_area_world(glacier_area_csv, oggm_results_path, mask_attributes, startyear, endyear, outpath,
                                       include_off_area=include_off_area)

    timeseries = pd.date_range(str(area_gl_gridcell.columns.values[0]), str(area_gl_gridcell.columns.values[-1]),
                               freq='AS')

    assert len(timeseries) == np.shape(area_gl_gridcell)[1]

    # create netcdf file for variable
    if include_off_area:
        label = "total_area"
    else:
        label = "on_area"
    if fraction == True:
        ds = nc.Dataset(outpath + label + '_fraction_' + out_name + '.nc', 'w', format='NETCDF4')
    else:
        ds = nc.Dataset(outpath + label + '_' + out_name + '.nc', 'w', format='NETCDF4')
    # add dimenstions, specify how long they are
    # use 0.5° grid
    # dimensions are created based on example netcdf
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', None)
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"
    # create variables glacier area
    var_nc = ds.createVariable(label, 'f4', ('time', 'lat', 'lon',), zlib=True)
    if fraction:
        var_nc.units = "fraction of cell_area"
    else:
        var_nc.units = "m2"  # units is m2

    lats[:] = np.sort(lat)[::-1]
    lons[:] = np.sort(lon)
    timeseries_str = str(timeseries)
    time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

    glacier_ids = area_gl_gridcell.index

    for i in range(len(timeseries)):
        glacier_on_area_array[glacier_ids] = area_gl_gridcell.iloc[:, i].values
        # glacier_on_area[glacier_ids] = np.ones(len(glacier_ids))
        glacier_on_area_2d = np.reshape(glacier_on_area_array, (cellnr_lat, cellnr_lon))
        var_nc[i, :, :] = glacier_on_area_2d
    ds.close()
    del ds

    if fixed_year:
        if fraction == True:
            ds_fixed = nc.Dataset(outpath + label + '_fraction_' + out_name + '_constant_' + str(fixed_year) + '.nc',
                                  'w', format='NETCDF4')
        else:
            ds_fixed = nc.Dataset(outpath + label + '_' + out_name + '_constant_' + str(fixed_year) + '.nc', 'w',
                                  format='NETCDF4')

        lat_dim_fixed = ds_fixed.createDimension('lat', len(lat))
        lon_dim_fixed = ds_fixed.createDimension('lon', len(lon))
        lats_fixed = ds_fixed.createVariable('lat', 'f4', ('lat',))
        lons_fixed = ds_fixed.createVariable('lon', 'f4', ('lon',))
        # create variables glacier melt on off, liquid precipitation on off
        var_fixed_nc = ds_fixed.createVariable(label, 'f4', ('lat', 'lon',), zlib=True)
        if fraction:
            var_fixed_nc.units = "fraction of cell_area"
        else:
            var_fixed_nc.units = "m2"  # units is m2

        lats_fixed[:] = np.sort(lat)[::-1]
        lons_fixed[:] = np.sort(lon)

        x = np.zeros((cellnr_lat, cellnr_lon))
        glacier_on_area = x
        glacier_on_area = glacier_on_area.flatten()
        glacier_ids = area_gl_gridcell.index

        glacier_on_area[glacier_ids] = area_gl_gridcell.iloc[:,
                                       np.argwhere(area_gl_gridcell.columns == fixed_year)[0][0]].values
        glacier_on_area_2d = np.reshape(glacier_on_area, (cellnr_lat, cellnr_lon))
        var_fixed_nc[:, :] = glacier_on_area_2d

        ds_fixed.close()
        del ds_fixed


def change_area(glacier_area_csv, oggm_results_list, mask_attributes, include_off_area=False, cell_area = None):
    '''
    Update glacier area in each gridcell by substracting the decreased area from all gridcells the glacier is covering, relative to the percentage coverage
    function works for area reduction and area growth

    :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
    :param oggm_results: results from oggm run
    :return: a pandas data frame of the glacier area in each grid cell partially covered by glaciers
            with index tuple of lat, lon of gridcell and
            with columns all years in oggm_results and their corresponding area
    '''
    min_lon, max_lat, cellwidth, cellnr_lon, cellnr_lat = mask_attributes

    for i, oggm_results in enumerate(oggm_results_list):

        #TODO: is there a faster option than using pandas?
        #get IDs of glaciers modelled by OGGM
        rgi_ids = oggm_results.rgi_id.values
        #only look at glaciers which were modelled by OGGM
        #TODO: should this be for all OGGM_results together?
        glacier_area_basin = glacier_area_csv[np.isin(glacier_area_csv.RGIId, rgi_ids)]
        glacier_area_basin = glacier_area_basin.reset_index(drop=True)
        #make a new array that contains latitudes longitudes, Gridcell Nr and years of data corresponding to length of OGGM results
        # + 2 because we need lat, lon, Nr Gridcells but we do not need last year
        array_area = np.zeros((np.shape(glacier_area_basin)[0], np.shape(oggm_results.time)[0] + 2))
        array_area[:, 0] = glacier_area_basin.Nr_Gridcell.values
        array_area[:,1] = glacier_area_basin.Latitude.values
        array_area[:,2] = glacier_area_basin.Longitude.values

        x = np.zeros((cellnr_lat, cellnr_lon))
        x = x.astype(int)
        glacier_on_area_array = x
        glacier_on_area_array = glacier_on_area_array.flatten()

        #TODO loop through rgi results if necessary
        #loop through all glaciers to reduce area of each glacier in the gridcells that are covered by the glacier
        for rgi_id in rgi_ids:
            #get all rows of df of current glacier
            current_glacier = glacier_area_basin[glacier_area_basin.RGIId == rgi_id]
            #area of RGI is start area, because for this area we have the outlines
            area_start = np.sum(glacier_area_basin[glacier_area_basin.RGIId == rgi_id].Area)
            #area of all years from OGGM results
            if include_off_area:
                area_oggm = oggm_results.off_area.loc[:, rgi_id].values + oggm_results.on_area.loc[:, rgi_id].values
            else:
                area_oggm = oggm_results.on_area.loc[:, rgi_id].values
            #area reduction is relative to area at RGI date
            #TODO: area reduction should be relative to area at start date
            area_reduction = area_start - area_oggm
            #get relative area reduction compared to glacier area at RGI date
            rel_reduction = area_reduction / area_start
            #multiply the relative area that remains with the glacier area in the grid cell
            area_glacier = np.outer((1 - rel_reduction), current_glacier.Area)
            #area that remains should be the same as glacier area in oggm
            #if area close to zero, relative differences can be large, therefore atol = 0.001

            np.testing.assert_allclose(np.sum(area_glacier, axis=1), area_oggm, rtol=1e-3, atol = 0.1)# for past: rtol=1e-5, atol=0.001)
            # everything below a an area of 1 should be neglected to avoid ridiculously small areas
            area_glacier = np.where(area_glacier < 1, 0, area_glacier)

            #put result in array to generate dataframe
            assert np.all(current_glacier.Latitude.values == array_area[current_glacier.index.values, 1])
            assert np.all(current_glacier.Longitude.values == array_area[current_glacier.index.values, 2])
            array_area[current_glacier.index.values, 3:] = area_glacier[:-1, :].T

        #generate a dataaframe
        if i == 0:
            #TODO: here it should be appended
            df = pd.DataFrame(array_area,
                              columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(oggm_results.time.values.astype('str')[:-1]))
        else:
            df = pd.concat([df, pd.DataFrame(array_area,
                              columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(oggm_results.time.values.astype('str')[:-1]))])

        print(np.shape(df))
    #TODO: this assertion does not always work only if RGI date it 2000?
    #for small glacier area rel_dif can be large
    #for large glacier area abs_diff can be large
    # if oggm_results.time.values[0] < 2000:
    #     np.testing.assert_allclose(df.iloc[:, 18], glacier_area_basin.Area, rtol=0.02, atol = 50000)
    #sum area across glaciers for same gridcells by using the Nr Gridcell as indicator
    dfinal = pd.DataFrame(df.groupby(by='Nr_Gridcell').sum().iloc[:, 2:].values,
        columns=list(oggm_results.time.values.astype('int')[:-1]))
    #get the latitude, longitude of these gridcells by taking mean ofver Nr of Gridcells
    round_lat = np.round(df.groupby(by='Nr_Gridcell').mean().Latitude, decimals=3)
    round_lon = np.round(df.groupby(by='Nr_Gridcell').mean().Longitude, decimals=3)

    cell_lat = (max_lat - np.array(round_lat)) / cellwidth
    cell_lon = (np.array(round_lon) - min_lon) / cellwidth
    #make sure that cell_lat and cell_lon are indices (should be integer)
    if cell_lat.all() % 1 < 0.1 or cell_lat.all() % 1 > 0.9:
        cell_lat = np.round(cell_lat).astype('int')
    else:
        print(cell_lat)
        raise ValueError
    if cell_lon.all() % 1 < 0.1 or cell_lon.all() % 1 > 0.9:
        cell_lon = np.round(cell_lon).astype('int')
    else:
        print(cell_lon)
        raise ValueError

    #celllat and cell_lon should be within the bounds of the example_nc
    #crop all entries that are not within the mask attributes
    #arg_out = np.argwhere((cell_lat < 0) | (cell_lat >= cellnr_lat) | (cell_lon < 0) | (cell_lon >= cellnr_lon)).flatten()
    arg_in =np.argwhere((cell_lat >= 0) & (cell_lat < cellnr_lat) & (cell_lon >= 0) & (cell_lon < cellnr_lon)).flatten()
    cell_lat = cell_lat[arg_in]
    cell_lon = cell_lon[arg_in]
    #crop values where index is larger than index
    if len(arg_in) < len(cell_lat):
        warnings.warn(
            "The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")
    dfinal_array = np.array(dfinal)

    #set the index of the dataframe as a tuple of latitude, longitude
    #TODO: in case you want to have fraction of area instead of total area divide area by area fraction
    if cell_area:
        ind_lat_area = [np.argmin(abs(cell_area["lat"].values - x)) for x in round_lat] #np.argmin(abs(cell_area["lat"].values - round_lat))
        ind_lon_area = np.array([np.argmin(abs(cell_area["lon"].values - x)) for x in round_lon]) #np.argmin(abs(cell_area["lon"].values - round_lon))
        assert len(ind_lon_area) == len(ind_lat_area)
        cell_area_gl =[]
        for k in range(len(ind_lon_area)):
            cell_area_gl.append(cell_area[list(cell_area.keys())[0]].values[ind_lat_area[k]][ind_lon_area[k]])
        # only add area if grid cell is on land
        #TODO only add area if grid cells are on land
        #dfinal = dfinal.iloc[:, :].values / np.array(cell_area_gl)[:, None]
        dfinal_array = np.divide(dfinal.iloc[:, :].values, np.array(cell_area_gl)[:, None], where=np.array(cell_area_gl)[:, None] != 0,
                  out=np.zeros(np.shape(dfinal.iloc[:, :].values)))

    dfinal_array = dfinal_array[arg_in, :]

    dfinal = pd.DataFrame(dfinal_array,
                          columns=list(oggm_results.time.values.astype('int')[:-1]))
    #dfinal.index = list(zip(round_lat, round_lon))

    ind_cell = (cell_lat) * cellnr_lon + cell_lon
    dfinal.index = ind_cell

    return dfinal
@profile
def change_area_world(glacier_area_csv, oggm_results_list, mask_attributes, startyear, endyear, outpath, include_off_area=False, cell_area = None):
    '''
    Update glacier area in each gridcell by substracting the decreased area from all gridcells the glacier is covering, relative to the percentage coverage
    function works for area reduction and area growth

    :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
    :param oggm_results_list: list of paths with oggm results
    :return: a pandas data frame of the glacier area in each grid cell partially covered by glaciers
            with index tuple of lat, lon of gridcell and
            with columns all years in oggm_results and their corresponding area
    '''
    min_lon, max_lat, cellwidth, cellnr_lon, cellnr_lat = mask_attributes
    print("start processing")
    #if not os.path.exists(outpath + 'df_area.csv'):
    for i, oggm_results_path in enumerate(oggm_results_list):
        print(oggm_results_path)

        oggm_results = xr.open_dataset(oggm_results_path)
        #delete nan values
        nok = oggm_results.volume.isel(time=0).isnull()
        # check if OGGM results contains glaciers with missing results
        if np.count_nonzero(nok) > 0: #len(oggm_results_org.rgi_id) != len(oggm_results.rgi_id):
            oggm_results_new = oggm_results.dropna('rgi_id', thresh=10)
            # get RGI IDs of missing values
            rgi_ids_nan = list(set(list(oggm_results.rgi_id.values)) - set(list(oggm_results_new.rgi_id.values)))
            # get number of glaciers with nan values
            missing_rgis = len(oggm_results.rgi_id) - len(oggm_results_new.rgi_id)
            # print warning that netcdf will not be generated for the glaciers with missing OGGM results
            msg = 'Original OGGM results has {} glaciers with missing values ({}). These are not used for NetCDF generation'.format(
                missing_rgis, rgi_ids_nan)
            warnings.warn(msg)
            oggm_results.close()
            del oggm_results
            oggm_results = oggm_results_new
            del oggm_results_new


        years_oggm = list(oggm_results.time.values.astype('int')[:-1])
        years = list(np.arange(startyear, endyear + 1))
        assert all(item in years_oggm for item in years), 'Years are out of range of years of OGGM results. Change startyear and endyear.'

        #get IDs of glaciers modelled by OGGM
        rgi_ids = oggm_results.rgi_id.values
        #only look at glaciers which were modelled by OGGM
        #TODO: should this be for all OGGM_results together?
        glacier_area_basin = glacier_area_csv[np.isin(glacier_area_csv.RGIId, rgi_ids)]
        glacier_area_basin = glacier_area_basin.reset_index(drop=True)
        #make a new array that contains latitudes longitudes, Gridcell Nr and years of data corresponding to length of OGGM results
        # + 2 because we need lat, lon, Nr Gridcells but we do not need last year
        array_area = np.zeros((np.shape(glacier_area_basin)[0], len(years) + 3))
        array_area[:, 0] = glacier_area_basin.Nr_Gridcell.values
        array_area[:,1] = glacier_area_basin.Latitude.values
        array_area[:,2] = glacier_area_basin.Longitude.values

        x = np.zeros((cellnr_lat, cellnr_lon))
        x = x.astype(int)
        #make a array in which data can be sroted
        glacier_on_area_array = x
        glacier_on_area_array = glacier_on_area_array.flatten()

        # ----------- with map function -----------------------------

        # current_glacier = list(map(lambda rgi_id: glacier_area_basin[glacier_area_basin.RGIId == rgi_id], rgi_ids))
        # area_start = list(map(lambda rgi_id: np.sum(glacier_area_basin[glacier_area_basin.RGIId == rgi_id].Area), rgi_ids))
        # if include_off_area:
        #     area_oggm = list(map(lambda rgi_id: oggm_results.off_area.loc[:, rgi_id].values + oggm_results.on_area.loc[:, rgi_id].values, rgi_ids))
        # else:
        #     area_oggm = list(map(lambda rgi_id: oggm_results.on_area.loc[:, rgi_id].values, rgi_ids))

        # ------------ with list comprehension

        #get info about current glacier
        current_glacier = [glacier_area_basin[glacier_area_basin.RGIId == rgi_id] for rgi_id in rgi_ids]
        #get the area at RGI date
        area_start = [np.sum(glacier_area_basin[glacier_area_basin.RGIId == rgi_id].Area) for rgi_id in rgi_ids]

        #get glacier areas as modeled by OGGM (for x years)
        if include_off_area:
            area_oggm = [list(oggm_results.off_area.loc[years].values + oggm_results.on_area.loc[years].values) for rgi_id in rgi_ids]
        else:
            area_oggm = [oggm_results.on_area.loc[years, rgi_id].values for rgi_id in rgi_ids]

        # -------------

        #calculate area change (can be reduction or growth
        area_change = list(map(lambda x, y: x-y, area_start, area_oggm))
        rel_change = list(map(lambda x, y: y / x, area_start, area_change))
        # calculate area of glacier using area change
        area_glacier = list(map(lambda x, y: (np.outer((1 - x), y.Area)), rel_change, current_glacier)) #.flatten()

        list(map(lambda x, y: np.testing.assert_allclose(np.sum(x, axis=1), y, rtol=1e-3, atol=0.1), area_glacier, area_oggm))
        # everything below a an area of 1 should be neglected to avoid ridiculously small areas
        area_glacier = list(map(lambda x: np.where(x < 1, 0, x), area_glacier))

        #assert that coordinate values area correct
        assert all(x.Latitude.values[0] == array_area[x.index.values[0], 1] for x in current_glacier)
        assert all(x.Longitude.values[0] == array_area[x.index.values[0], 2] for x in current_glacier)

        for k in range(len(area_glacier)):
            array_area[current_glacier[k].index.values, 3:] = area_glacier[k][:, :].T


        #loop through all glaciers to reduce area of each glacier in the gridcells that are covered by the glacier
        # for rgi_id in rgi_ids:
        #     #get all rows of df of current glacier
        #     current_glacier = glacier_area_basin[glacier_area_basin.RGIId == rgi_id]
        #     #area of RGI is start area, because for this area we have the outlines
        #     area_start = np.sum(glacier_area_basin[glacier_area_basin.RGIId == rgi_id].Area)
        #     #area of all years from OGGM results
        #     if include_off_area:
        #         area_oggm = oggm_results.off_area.loc[:, rgi_id].values + oggm_results.on_area.loc[:, rgi_id].values
        #     else:
        #         area_oggm = oggm_results.on_area.loc[:, rgi_id].values
        #     #area reduction is relative to area at RGI date
        #     #area reduction is negative if the glacier is growing
        #     area_change = area_start - area_oggm
        #     #get relative area reduction compared to glacier area at RGI date
        #     #relative reduction is negative if glacier is growing
        #     rel_change = area_change / area_start
        #     #multiply the relative area that remains with the glacier area in the grid cell
        #     area_glacier = np.outer((1 - rel_change), current_glacier.Area)
        #     #area that remains should be the same as glacier area in oggm
        #     #if area close to zero, relative differences can be large, therefore atol = 0.001
        #
        #     np.testing.assert_allclose(np.sum(area_glacier, axis=1), area_oggm, rtol=1e-3, atol = 0.1)# for past: rtol=1e-5, atol=0.001)
        #     # everything below a an area of 1 should be neglected to avoid ridiculously small areas
        #     area_glacier = np.where(area_glacier < 1, 0, area_glacier)
        #
        #     #put result in array to generate dataframe
        #     # assert np.all(current_glacier.Latitude.values == array_area[current_glacier.index.values, 1])
        #     # assert np.all(current_glacier.Longitude.values == array_area[current_glacier.index.values, 2])
        #     array_area[current_glacier.index.values, 3:] = area_glacier[:-1, :].T

        # delete current oggm result from workspace
        oggm_results.close()
        del oggm_results

        #generate a dataaframe
        # if i == 0:
        df = pd.DataFrame(array_area,
                          columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(years))
        # else:
        #     df = pd.concat([df, pd.DataFrame(array_area,
        #                       columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(years))])

        #save it just as backup
        df.to_csv(outpath + '_df_area_rgi{}.csv'.format(oggm_results_path.split('run_output_')[1].split('_1990')[0]))
    # else:
    #     df = pd.read_csv(outpath + 'df_area.csv', index_col=0)
    #     years = list(df.columns[3:].astype('int'))#r'C:\Users\shanus\Data\Glaciers_new\results_oggm\world_30min\input_cwatm/all_regions_new_pf3.0_30mindf_area.csv', index_col=0)
    #TODO: this assertion does not always work only if RGI date it 2000?
    #for small glacier area rel_dif can be large
    #for large glacier area abs_diff can be large
    # if oggm_results.time.values[0] < 2000:
    #     np.testing.assert_allclose(df.iloc[:, 18], glacier_area_basin.Area, rtol=0.02, atol = 50000)
    #sum area across glaciers for same gridcells by using the Nr Gridcell as indicator
    dfinal = pd.DataFrame(df.groupby(by='Nr_Gridcell').sum().iloc[:, 2:].values,
        columns=years)
    #get the latitude, longitude of these gridcells by taking mean ofver Nr of Gridcells
    round_lat = np.round(df.groupby(by='Nr_Gridcell').mean().Latitude, decimals=3)
    round_lon = np.round(df.groupby(by='Nr_Gridcell').mean().Longitude, decimals=3)

    cell_lat = (max_lat - np.array(round_lat)) / cellwidth
    cell_lon = (np.array(round_lon) - min_lon) / cellwidth
    #make sure that cell_lat and cell_lon are indices (should be integer)
    if cell_lat.all() % 1 < 0.1 or cell_lat.all() % 1 > 0.9:
        cell_lat = np.round(cell_lat).astype('int')
    else:
        print(cell_lat)
        raise ValueError
    if cell_lon.all() % 1 < 0.1 or cell_lon.all() % 1 > 0.9:
        cell_lon = np.round(cell_lon).astype('int')
    else:
        print(cell_lon)
        raise ValueError

    #celllat and cell_lon should be within the bounds of the example_nc
    #crop all entries that are not within the mask attributes
    #arg_out = np.argwhere((cell_lat < 0) | (cell_lat >= cellnr_lat) | (cell_lon < 0) | (cell_lon >= cellnr_lon)).flatten()
    arg_in =np.argwhere((cell_lat >= 0) & (cell_lat < cellnr_lat) & (cell_lon >= 0) & (cell_lon < cellnr_lon)).flatten()
    cell_lat = cell_lat[arg_in]
    cell_lon = cell_lon[arg_in]
    #crop values where index is larger than index
    if len(arg_in) < len(cell_lat):
        warnings.warn(
            "The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")
    dfinal_array = np.array(dfinal)

    # get cell area of grid cells with glaciers
    ind_lat_area = [np.argmin(abs(cell_area["lat"].values - x)) for x in round_lat] #np.argmin(abs(cell_area["lat"].values - round_lat))
    ind_lon_area = np.array([np.argmin(abs(cell_area["lon"].values - x)) for x in round_lon]) #np.argmin(abs(cell_area["lon"].values - round_lon))
    assert len(ind_lon_area) == len(ind_lat_area)
    cell_area_gl =[]
    for k in range(len(ind_lon_area)):
        cell_area_gl.append(cell_area[list(cell_area.keys())[0]].values[ind_lat_area[k]][ind_lon_area[k]])
    #check whether the glacier area in grid cell is larger than grid cell
    diff_rel = (np.array(cell_area_gl)[:, None] - dfinal_array) / np.array(cell_area_gl)[:, None]
    diff = np.array(cell_area_gl)[:, None] - dfinal_array
    # while there is too much glacier area in one grid cell, spead it to other grid cells
    count_rounds = 0
    while np.count_nonzero(diff_rel <-0.001) > 0:
        print('round {} Nr gridcells with glaciers {}'.format(count_rounds, len(cell_lat)))
        # if any of the years is negative in one grid cell, get the adjacent girdcells and distribute it
        # get index of gridcells with too much area in some year
        id_gridcell = np.unique(np.where(diff_rel <-0.001)[0])
        too_much = np.where(diff[id_gridcell, :] >= 0, 0, np.abs(diff[id_gridcell, :]))
        #indexes of grid cells with negative difference (glacier area > 1)
        idx_diff = (diff_rel < -0.001).max(axis=1)
        cell_lat_big = cell_lat[1 == idx_diff]
        cell_lon_big = cell_lon[1 == idx_diff]

        # loop over grid cells with too much area and distribute it equally to the four adjacent grid cells
        for i in range(len(id_gridcell)):
            #get grid cell of current glacier
            curr_lat = cell_lat_big[i]
            curr_lon = cell_lon_big[i]
            # add new grid cells
            cell_lat = np.concatenate((cell_lat, [curr_lat + 1, curr_lat, curr_lat -1 , curr_lat]))
            cell_lon = np.concatenate((cell_lon, [curr_lon, curr_lon -1 , curr_lon, curr_lon + 1]))
            # delete too much volume from gridcell itself
            dfinal.iloc[id_gridcell[i], :] = dfinal.iloc[id_gridcell[i], :] - too_much[i]
            # add four gridcells with same glacier volume
            dfinal = pd.concat([dfinal, pd.DataFrame(np.tile(too_much[i] / 4,(4, 1)), columns=years)])

            new_gl_cell_area = list(map(lambda x, y: cell_area[list(cell_area.keys())[0]].values[x][y], [curr_lat + 1, curr_lat, curr_lat -1 , curr_lat], [curr_lon, curr_lon -1 , curr_lon, curr_lon + 1]))
            cell_area_gl = cell_area_gl + new_gl_cell_area
        # check new diff
        # there can be cells duplicated
        ind_cell = (cell_lat) * cellnr_lon + cell_lon
        dfinal.index = ind_cell
        #sum up duplicates
        dfinal["cell_area"] = np.array(cell_area_gl)
        dfinal["cell_lat"] = np.array(cell_lat)
        dfinal["cell_lon"] = np.array(cell_lon)
        dfinal_cell = dfinal.groupby(by=dfinal.index).mean().iloc[:, -3:]
        #TODO id_gridcell does not work here anymore
        dfinal = dfinal.groupby(by=dfinal.index).sum().iloc[:, :-3]

        cell_area_gl = dfinal_cell["cell_area"].values
        cell_lat = dfinal_cell["cell_lat"].values.astype('int')
        cell_lon = dfinal_cell["cell_lon"].values.astype('int')
        # get new arg_in to only get grid cells in example_nc
        # so it can happen that area is redistributed to grid cell which area not in modelled area, but also without redistribution not all glacier parts are in model domain
        arg_in = np.argwhere(
            (cell_lat >= 0) & (cell_lat < cellnr_lat) & (cell_lon >= 0) & (cell_lon < cellnr_lon)).flatten()

        #if cell_area = 0, set diff to 0, this is then because of land module in CWatM
        idx_water = np.where(cell_area_gl== 0)
        diff = (cell_area_gl[:, None] - dfinal.values[:,:])
        diff_rel = (cell_area_gl[:, None] - dfinal.values[:,:]) / cell_area_gl[:, None]
        diff[idx_water, :] = 0
        cell_area_gl = list(cell_area_gl)
        count_rounds += 1

        print('Some grid cells where overfilled with glaciers, the area was redistributed to neighbouring grid cells.')

        #or maybe also because gridcells are "overfilled" with glaciers??
        #dfinal = dfinal.iloc[:, :].values / np.array(cell_area_gl)[:, None]
    if cell_area:
        dfinal_array = np.divide(dfinal.iloc[:, :].values, np.array(cell_area_gl)[:, None], where=np.array(cell_area_gl)[:, None] != 0,
                  out=np.zeros(np.shape(dfinal.iloc[:, :].values)))
        #clip values to a maximum of 1
        if np.max(dfinal_array) > 1:
            dfinal_array = np.clip(dfinal_array, 0, 1)
            if np.max(dfinal_array) > 1.001:
                msg = 'Calculated area fraction for some pixels was larger 1, due to cell area data of CWatM only containing land cell area. This is clipped to 1'
                warnings.warn(msg)

    dfinal_array = dfinal_array[arg_in, :]

    dfinal = pd.DataFrame(dfinal_array,
                          columns=years)
    ind_cell = (cell_lat) * cellnr_lon + cell_lon
    dfinal.index = ind_cell.astype('int')

    return dfinal
