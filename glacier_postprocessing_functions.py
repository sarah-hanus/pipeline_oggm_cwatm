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
#-------- POSTPROCESSING --------------------

#@profile
def change_format_oggm_output(oggm_results_list, startyear_df, endyear_df, variable):
    '''Function changes the format of oggm output from 2d array (days,years) to a 1d timeseries
    :param oggm_results_list: list of paths with oggm results
    :param startyear_df: first year of period for which output should be generated
    :param endyear_df: last year of period for which output should be generated
    :param variable: '_on_glacier_daily' or '_off_glacier_daily'
    :param outpath:
    :return:
    '''
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

        glacier_ids_current = list(oggm_results.rgi_id.values)
        oggm_results.close()
        del oggm_results

        if idx == 0:
            all_months_melt_all = all_months_melt
            all_months_rain_all = all_months_rain
            glacier_ids = glacier_ids_current
        else:
            # TODO all months next has to be concatenated
            all_months_melt_all = np.concatenate((all_months_melt_all,all_months_melt),axis=1)
            all_months_rain_all = np.concatenate((all_months_rain_all, all_months_rain), axis=1)
            glacier_ids += glacier_ids_current


    #delete the nan values on the 29th of february
    # OGGM was run such that the year always has 366 days and if the year is not a leap year a nan value is inserted
    all_months_melt_all = all_months_melt_all[~np.isnan(all_months_melt_all).any(axis=1), :]
    all_months_rain_all = all_months_rain_all[~np.isnan(all_months_rain_all).any(axis=1), :]
    #assert that the timeseries and new dara have same length
    assert len(timeseries) == np.shape(all_months_melt_all)[0]
    assert len(timeseries) == np.shape(all_months_rain_all)[0]
    return all_months_melt_all, all_months_rain_all, glacier_ids, timeseries

#@profile
def oggm_output_to_cwatm_input(glacier_outlet, oggm_results_path, pf_sim, startyear, endyear, outpath, out_name, example_netcdf, resolution, include_off_area = False):
    '''
    Function generates a netcdf of daily glacier melt using OGGM outputs. The outlet point of each glacier is fixed to the gridcell of the terminus as at RGI date
        and will not change even if glacier retreats, as the effect is assumed to be minimal on a larger modelling domain.

        glacier_outlet: a dictionary of all glaciers worldwide with keys coordinates of gridcells and values a list of glacier ids which drain into the gridcell
        (CAREFUL: you might want to adapt it to fit to the basin)

        :param glacier_outlet: a dictionary of all glaciers worldwide with keys coordinates of gridcells and values a list of glacier ids which drain into the gridcell
        :param oggm_results_path: list of paths to netcdf files of daily results of an OGGM run
        :param pf_sim: precipitation factor which was used to generate OGGM results
        :param startyear: first year of period for which output should be generated
        :param endyear: last year of period for which output should be generated
        :param outpath: path where result will be stored
        :param out_name: name of output file
        :param example_netcdf: example netcdf with same extent that you want to generate
        :param resolution: "30min" or "5min"
        :param include_off_area: whether to use constant area in OGGM or variable area

        returns: metcdf with daily glacier melt (m3/d)
    '''

    if resolution not in ["30min", "5min"]:
        raise ValueError("resolution should be 30min or 5min")

    # define extent for which glacier maps should be created
    if type(example_netcdf) == str:
        example_nc = xr.open_dataset(example_netcdf)
        lat = np.round(example_nc.lat.values,
                       decimals=5)
        lon = np.round(example_nc.lon.values,
                       decimals=5)
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
    melt_on_glacier, rain_on_glacier, glacier_ids, timeseries_glacier = change_format_oggm_output(oggm_results_path, startyear, endyear, '_on_glacier_daily')
    if include_off_area:
        melt_off_glacier, rain_off_glacier, glacier_ids, _ = change_format_oggm_output(oggm_results_path, startyear, endyear, '_off_glacier_daily')

    #get start and end date corresponding to the inputs of the function
    start_index = np.where(timeseries_glacier.year == startyear)[0][0]
    end_index = np.where(timeseries_glacier.year == endyear)[0][-1]
    timeseries = pd.date_range(timeseries_glacier[start_index].strftime('%Y-%m'), timeseries_glacier[end_index],freq='D')

    #define factor (pf) by which OGGM results are divided
    #pf is 1 for melt because for snow accumulation and melt precipitation factor is used
    # for liquid prcp we do not want to use pf of OGGM to make results more similar to CWatM, therefore results are divided by pf
    pfs = [1, pf_sim]
    vars = ['melt', 'liq_prcp']
    # now create a separate netcdf fil for rain on glaciers and melt on glaciers
    for k, flux_on_glacier in enumerate([melt_on_glacier, rain_on_glacier]):
        var_name = vars[k]
        pf = pfs[k]
        #create netcdf file for each variable
        name = '{}_on'.format(var_name)
        if include_off_area:
            ds = nc.Dataset(outpath +name[:-2]  + 'total_'+ out_name +'.nc', 'w', format='NETCDF4')
        else:
            ds = nc.Dataset(outpath + name+ '_' + out_name + '.nc', 'w', format='NETCDF4')
        # add dimenstions, specify how long they are
        lat_dim = ds.createDimension('lat', len(lat))
        lon_dim = ds.createDimension('lon', len(lon))
        time_dim = ds.createDimension('time', len(timeseries))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        time_date = ds.createVariable('time', 'f4', ('time',))
        time_date.units = "days since 1961-01-01"
        time_date.calendar = "standard"
        #create variables glacier melt on off, liquid precipitation on off
        if include_off_area:
            var_nc = ds.createVariable(name[:-2] + 'total', 'i4', ('time', 'lat', 'lon',), chunksizes=(1,len(lat),len(lon)) ,zlib=True)
        else:
            var_nc = ds.createVariable(name, 'i4', ('time', 'lat', 'lon',), chunksizes=(1,len(lat),len(lon)), zlib=True)
        var_nc.units = "m3/d"
        #use the extent of the example netcdf
        lats[:] = np.sort(lat)[::-1]
        lons[:] = np.sort(lon)
        timeseries_str = str(timeseries)
        time_date[:] = nc.date2num(timeseries.to_pydatetime(), units=time_date.units)

        #now total melt input for timeseries
        #output is in kg/day -> transform to m3/day by dividing by 1000
        glacier_flux_on = flux_on_glacier[start_index:end_index+1]
        if include_off_area:
            glacier_flux_off = [melt_off_glacier, rain_off_glacier][k][start_index:end_index + 1]
            glacier_flux_on = glacier_flux_on / 1000 / pf + glacier_flux_off / 1000 / pf
        else:
            glacier_flux_on = glacier_flux_on / 1000 / pf

        #define dataframe to store results, index are the days in timeseries
        df_flux_on = pd.DataFrame(index=timeseries)

        x = np.zeros((cellnr_lat, cellnr_lon))
        x = x.astype(int)
        glacier_flux_on_array = x
        glacier_flux_on_array = glacier_flux_on_array.flatten()
        assert len(timeseries) == np.shape(df_flux_on)[0]

        #loop through all gridcells with glaciers in basin and sum up the melt of all glaciers in each grid cell
        keys_gridcells = list(glacier_outlet)
        list_rgi_ids_world = list(glacier_outlet.values())
        count_gl_not_oggm_results = 0
        # loop through grid cells with glaciers and check if glaciers are in current results
        cell_lat_all_gridcells = []
        cell_lon_all_gridcells = []
        # loop through list of grid cells in world where glaciers are located
        for i, rgi_ids_world in enumerate(list_rgi_ids_world):
            #check if any glacier in these grid cell is in glacier_ids
            if np.isin(rgi_ids_world, glacier_ids).any():
                grid_lat = keys_gridcells[i][0]
                grid_lon = keys_gridcells[i][1]
                ids_gridcell = glacier_outlet[keys_gridcells[i]]

                #daily melt volumes have to be stored in a datafram with column names being the gridcell lat, lon
                round_lat = np.round(grid_lat, decimals=3)
                round_lon = np.round(grid_lon, decimals=3)

                #the cell number,e.g as index
                cell_lat = (np.max(lat) - round_lat) / cellwidth
                cell_lon = (round_lon - np.min(lon)) / cellwidth

                if cell_lat % 1 <  0.1 or cell_lat % 1 >  0.9:
                    cell_lat = int(np.round(cell_lat))
                else:
                    print(cell_lat)
                    raise ValueError
                # for lon and 5 arcmin there are rounding errors that is why I use 0.2 as threshold
                if cell_lon % 1 < 0.2  or cell_lon % 1 > 0.9:
                    cell_lon = int(np.round(cell_lon))
                else:
                    print(cell_lon)
                    raise ValueError

                cell_lat_all_gridcells.append(cell_lat)
                cell_lon_all_gridcells.append(cell_lon)
                ind_cell = (cell_lat) * cellnr_lon + cell_lon

                daily_flux_on = 0
                #loop through all glaciers in the gridcell by getting the items in the dict of this grid cell
                for id in ids_gridcell:
                    if id not in glacier_ids: #17% of time for all rgis 30arcmin
                        #if glacier id of glacier that drains into the basin is not modelled in OGGM, raise warning, but you coula also raise an error
                        msg = 'Glacier {} was not found in OGGM results'.format(id)
                        warnings.warn(msg)
                        count_gl_not_oggm_results += 1
                        #if no nan values exist, then sum up the variables of all glaciers in the gridcell
                    elif np.sum(np.nonzero(np.isnan(glacier_flux_on[:, glacier_ids.index(id)]))) == 0:
                        # sum up timeseries of all glaciers in gridcell
                        daily_flux_on += glacier_flux_on[:, glacier_ids.index(id)] #23% of time for all rgis 30arcmin
                    else:
                        raise ValueError('Nan values encountered in timeseries of {} for variable {}'.format(id, glacier_flux_on))

                if ind_cell <= len(glacier_flux_on_array):
                    df_flux_on = pd.concat([df_flux_on, pd.DataFrame(daily_flux_on, columns=[ind_cell], index=timeseries)], axis=1)
                else:
                    warnings.warn("The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")

        glacier_gridcell_index = df_flux_on.columns
        if count_gl_not_oggm_results != 0:
            warnings.warn(
                "{} glaciers were not found in OGGM results but are in a grid cell for which other glaciers were modelled in OGGM. Check carefully".format(count_gl_not_oggm_results))
        print("start putting results into netcdf")
        for i in range(len(timeseries)):
            #array with all gridcells of the world, the index is used to add glacier melt to correct grid cell
            glacier_flux_on_array[glacier_gridcell_index] = df_flux_on.iloc[i, :].values
            glacier_flux_on_array[glacier_flux_on_array < 1e-9] = 0.0
            glacier_flux_on_2d = np.reshape(glacier_flux_on_array, (cellnr_lat, cellnr_lon))
            #for each time step produce the 2d array of glacier flux
            #for single rgi regions a lot of time is spend on this> especially for 5arcmin this already take 60min for smallest rgi region (rgi region 6)
            var_nc[i, :, :] = glacier_flux_on_2d
        ds.close()
        del ds


#@profile
def oggm_area_to_cwatm_input(glacier_area_csv, oggm_results_path, cell_area, startyear, endyear, outpath, out_name, example_netcdf,
                             resolution, fraction=True, fixed_year=None, include_off_area=False):
    '''
    This function generates a netcdf file of the area (area fraction) covered by glacier in each gridcell.
    Note that only the glaciers which were run by OGGM will be taken into account, which should be the glaciers that drain into the basin at the model resolution.
    Note that only the area of the glacier within the gridded basin outline will be used, so that the total area is likely lower than the total area in OGGM results.

        :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
        :param oggm_results_path: results from oggm run, must be a list
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
                       decimals=5)
        lon = np.round(example_nc.lon.values,
                       decimals=5)
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
        # update the glacier area in each grid cell
        area_gl_gridcell = change_area(glacier_area_csv, oggm_results_path, mask_attributes, startyear, endyear, outpath + out_name,
                                       cell_area, include_off_area=include_off_area, area_fraction=True)
    else:
        area_gl_gridcell = change_area(glacier_area_csv, oggm_results_path, mask_attributes, startyear, endyear, outpath + out_name,
                                       cell_area, include_off_area=include_off_area, area_fraction=False)

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
    # dimensions are created based on example netcdf
    lat_dim = ds.createDimension('lat', len(lat))
    lon_dim = ds.createDimension('lon', len(lon))
    time_dim = ds.createDimension('time', len(timeseries))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    time_date = ds.createVariable('time', 'f4', ('time',))
    time_date.units = "days since 1961-01-01"
    time_date.calendar = "standard"
    # create variables glacier area
    var_nc = ds.createVariable(label, 'f4', ('time', 'lat', 'lon',), chunksizes=(1,len(lat),len(lon)), zlib=True)
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
        var_fixed_nc = ds_fixed.createVariable(label, 'f4', ('lat', 'lon',), chunksizes=(1,len(lat),len(lon)), zlib=True)
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

#@profile
def change_area(glacier_area_csv, oggm_results_list, mask_attributes, startyear, endyear, outpath, cell_area, include_off_area=False, area_fraction = True):
    '''
    Update glacier area in each gridcell by substracting the decreased area from all gridcells the glacier is covering, relative to the percentage coverage
    function works for area reduction and area growth
    :param glacier_area_csv: csv file with information about area of each glacier in each grid cell, generated with df_glacier_grid_area function, valid for the RGI date
    :param oggm_results_list: list of paths with oggm results
    :param mask_attributes: the attributes of the desired output netcdf (longitue, latitude of upper left coorner of output domain, the number of cells in longitude, latitude and the cell width
    :param startyear: first year of period for which output should be generated
    :param endyear: last year of period for which output should be generated
    :param outpath: path where to save output
    :param cell_area: cellarea
    :param include_off_area: whether to include area that is not covered by glacier in current year but has been covered by glacier before
    :param area_fraction: whether the araea fraction of glaciers should be calculated or the total area
    :return: a pandas data frame of the glacier area in each grid cell partially covered by glaciers
            with index tuple of lat, lon of gridcell and
            with columns all years in oggm_results and their corresponding area
    '''
    min_lon, max_lat, cellwidth, cellnr_lon, cellnr_lat = mask_attributes
    print("start processing")
    for i, oggm_results_path in enumerate(oggm_results_list):
        print(oggm_results_path)
        years = list(np.arange(startyear, endyear + 1))
        #check if file already exists, if yea, then skip this step
        if not os.path.isfile(
            outpath + '_df_area_rgi{}.csv'.format(oggm_results_path.split('run_output_')[1].split('_1990')[0])):

            oggm_results = xr.open_dataset(oggm_results_path)
            #delete nan values
            nok = oggm_results.volume.isel(time=0).isnull()
            # check if OGGM results contains glaciers with missing results
            if np.count_nonzero(nok) > 0:
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
            assert all(item in years_oggm for item in years), 'Years are out of range of years of OGGM results. Change startyear and endyear.'

            #get IDs of glaciers modelled by OGGM
            rgi_ids = oggm_results.rgi_id.values
            #only look at glaciers which were modelled by OGGM
            glacier_area_basin = glacier_area_csv[np.isin(glacier_area_csv.RGIId, rgi_ids)]
            glacier_area_basin = glacier_area_basin.reset_index(drop=True)
            #make a new array that contains latitudes longitudes, Gridcell Nr and years of data corresponding to length of OGGM results
            # + 2 because we need lat, lon, Nr Gridcells but we do not need last year
            array_area = np.zeros((np.shape(glacier_area_basin)[0], len(years) + 3))
            array_area[:, 0] = glacier_area_basin.Nr_Gridcell.values
            array_area[:,1] = glacier_area_basin.Latitude.values
            array_area[:,2] = glacier_area_basin.Longitude.values

            # ------------ with list comprehension
            #check that all glaciers that where modelled are also in csv, otherwise raise warning
            if len(glacier_area_basin.RGIId.unique()) < len(rgi_ids):
                warnings.warn('{} glaciers were modelled but information on these glaciers is missing in csv file. These glaciers are not used'.format(len(rgi_ids) - len(glacier_area_basin.RGIId.unique())))
                rgi_ids = [x for x in rgi_ids if x in glacier_area_basin.RGIId.unique()]

            #get info about current glacier
            current_glacier = [glacier_area_basin[glacier_area_basin.RGIId == rgi_id] for rgi_id in rgi_ids]

            #get the area at RGI date
            area_start = [np.sum(glacier_area_basin[glacier_area_basin.RGIId == rgi_id].Area) for rgi_id in rgi_ids]

            #get glacier areas as modeled by OGGM (for x years)
            if include_off_area:
                area_oggm = [list(oggm_results.off_area.loc[years, rgi_id].values + oggm_results.on_area.loc[years, rgi_id].values) for rgi_id in rgi_ids]
            else:
                area_oggm = [oggm_results.on_area.loc[years, rgi_id].values for rgi_id in rgi_ids]

            # -------------

            #calculate area change (can be reduction or growth)
            # use the area at RGI date as baseline
            area_change = list(map(lambda x, y: x-y, area_start, area_oggm))
            #get the relative change in area
            rel_change = list(map(lambda x, y: y / x, area_start, area_change))
            # calculate area of glacier part in each grid cell using area of glacier in each grid cell and relative area change
            area_glacier = list(map(lambda x, y: (np.outer((1 - x), y.Area)), rel_change, current_glacier)) #.flatten()

            #assert that the total area of glacier over all grid cell is the same as the area simulated in OGGM
            list(map(lambda x, y: np.testing.assert_allclose(np.sum(x, axis=1), y, rtol=1e-3, atol=0.1), area_glacier, area_oggm))
            # everything below a an area of 1 should be neglected to avoid ridiculously small areas
            area_glacier = list(map(lambda x: np.where(x < 1, 0, x), area_glacier))

            #assert that coordinate values area correct
            assert all(x.Latitude.values[0] == array_area[x.index.values[0], 1] for x in current_glacier)
            assert all(x.Longitude.values[0] == array_area[x.index.values[0], 2] for x in current_glacier)

            for k in range(len(area_glacier)):
                array_area[current_glacier[k].index.values, 3:] = area_glacier[k][:, :].T

            # delete current oggm result from workspace
            oggm_results.close()
            del oggm_results

            #generate a dataaframe
            if i == 0:
                df = pd.DataFrame(array_area,
                                  columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(years))
                # save it just as backup
                df.to_csv(
                    outpath + '_df_area_rgi{}.csv'.format(oggm_results_path.split('run_output_')[1].split('_1990')[0]))
            else:
                df_new = pd.DataFrame(array_area,
                                  columns=['Nr_Gridcell', 'Latitude', 'Longitude'] + list(years))
                # save it just as backup
                df_new.to_csv(
                    outpath + '_df_area_rgi{}.csv'.format(oggm_results_path.split('run_output_')[1].split('_1990')[0]))
                df = pd.concat([df, df_new])
        else:
            if i == 0:
                df = pd.read_csv(outpath + '_df_area_rgi{}.csv'.format(oggm_results_path.split('run_output_')[1].split('_1990')[0]), index_col=0)
            else:
                df_new = pd.read_csv(outpath + '_df_area_rgi{}.csv'.format(oggm_results_path.split('run_output_')[1].split('_1990')[0]), index_col=0)
                df = pd.concat([df, df_new])

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
    arg_in =np.argwhere((cell_lat >= 0) & (cell_lat < cellnr_lat) & (cell_lon >= 0) & (cell_lon < cellnr_lon)).flatten()
    cell_lat = cell_lat[arg_in]
    cell_lon = cell_lon[arg_in]
    #crop values where index is larger than index
    if len(arg_in) < len(cell_lat):
        warnings.warn(
            "The extent for which glacier output should be generated is smaller than the extent run by OGGM. Check carefully")
    dfinal_array = np.array(dfinal)

    # get cell area of grid cells with glaciers
    ind_lat_area = [np.argmin(abs(cell_area["lat"].values - x)) for x in round_lat]
    ind_lon_area = np.array([np.argmin(abs(cell_area["lon"].values - x)) for x in round_lon])
    assert len(ind_lon_area) == len(ind_lat_area)
    cell_area_gl =[]
    for k in range(len(ind_lon_area)):
        cell_area_gl.append(cell_area[list(cell_area.keys())[0]].values[ind_lat_area[k]][ind_lon_area[k]])
    #check whether the glacier area in grid cell is larger than grid cell
    diff_rel = (np.array(cell_area_gl)[:, None] - dfinal_array) / np.array(cell_area_gl)[:, None]
    diff = np.array(cell_area_gl)[:, None] - dfinal_array
    # while there is too much glacier area in one grid cell, spread it to other grid cells
    #this can take several iterations
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

    if area_fraction:
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
