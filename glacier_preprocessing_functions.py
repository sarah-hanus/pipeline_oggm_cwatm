import geopandas as gpd
import pandas as pd
import os
import math
import warnings
import pickle
import numpy as np

'''
functions used for preprocessing, which is independent of OGGM results.
It relates glacier IDs to grid cells and glacier area at RGI date (shapefiles) to grid cells.
This information is later used for transforming OGGM resutls to CWatM input.
'''

# glacier_outlet_dict used by make_glacier_outlet_dict function
def glacier_outlet_dict(glacier_id, glacier_lat, glacier_lon, glacier_cen_lat, glacier_cen_lon, outpath, out_name, resolution, keys):
    '''makes a dictionary with keys (lat,lon) of gridcell with glaciers and items: RGIIDs of glaciers that have terminus in that grid cell
        or vice versa
    Input:  - glacier_id: list of glacier ids
            - glacier_lon/glacier_lat: list of postion of terminus of glaciers corresponding to glacier ids
            - glacier_cen_lon/glacier_cen_lat: list of postion of center of glaciers corresponding to glacier ids
            - out_name: name of new dictionary
            - resolution of grid ("5min", "30min")
            :param glacier_id: list of glacier ids
            :param glacier_lat:  list of latitude postion of terminus of glaciers corresponding to glacier ids
            :param glacier_lon: list of longitude postion of terminus of glaciers corresponding to glacier ids
            :param glacier_cen_lat: list of latitude postion of center of glaciers corresponding to glacier ids
            :param glacier_cen_lon: list of longitude postion of center of glaciers corresponding to glacier ids
            :param outpath: path to save the output dictionaries
            :param out_name: name of output file
            :param resolution: '5min' or '30min'
            :param keys: 'id' or 'coordinates' specifies what the key of the resulting dictionary should be
            :return: dictionary with gridcell coordinates and corresponding RGIIds of glaciers
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

            else:
                warnings.warn("No terminus or center location known for glacier {}. Glacier is disregarded".format(glacier_id[i]))
    glacier_ids_file = open(outpath+out_name+'_key_{}_center_{}.pkl'.format(keys, resolution), "wb")
    pickle.dump(glaciers_dict, glacier_ids_file)
    glacier_ids_file.close()

#generate pkl files of glacier id and coordinates of terminus/center of glaciers
def make_glacier_outlet_dict(list_path_glacierstats, outpath, out_name_dict, resolution, rgi_ids=None):
    '''generates the glacier outlet dictionary with the glacier_outlet_dict function

    :param list_path_glacierstats: list of paths of rgi glacier statistics, which must include coordinates of terminus and center of grid cells
    :param outpath: path to save the output dictionaries
    :param out_name_dict: name of output file
    :param resolution: '5min' or '30min'
    :param rgi_ids:  if list of rgi ids is provided only generates it the dictionary for the given rgi_ids
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

def overlay_area_grid(file_path_glacier_shape, file_path_grid, path_save):
    '''overlays the glacier geometries with the grid to get the glacier area per grid cell
    :param file_path_glacier_shape: path to glacier geometry shapefile
    :param file_path_grid: path to shapfile of grid used
    :param path_save: path to save the output
    :return shapefile as result of overlaying grid with glacier shapes
    '''
    rgi = gpd.read_file(file_path_glacier_shape)
    grid = gpd.read_file(file_path_grid)
    # results from overlay functions
    overlay_result = gpd.overlay(rgi, grid, how='intersection')
    # #get area of each part
    #transform it to CRS in meter
    overlay_54012 = overlay_result.to_crs('esri:54012')
    overlay_54012["Area_Cell"] = overlay_54012.area
    overlay_54012.to_file(path_save)


def transform_to_df(glacier_center, glacier_terminus, overlay_result, resolution):
    '''
    :param glacier_center: dictonary containing the center coordinates of the grid cell that contains center point of glacier
    :param glacier_terminus: dictonary containing the center coordinates of the grid cell that contains terminus point of glacier
    :param overlay_result: Geopandas dataframe resulting from overlaying the grid shapefile with the glacier shapfile;
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
            if math.isclose(overlay_lat, glacier_center.get(overlay_result.RGIId[i])[0], abs_tol=1e-3) and math.isclose(overlay_lon, glacier_center.get(overlay_result.RGIId[i])[1], abs_tol=1e-3):
                center[i] = 1
            #if overlay_lat == glacier_terminus.get(overlay_result.RGIId[i])[0] and overlay_lon == glacier_terminus.get(overlay_result.RGIId[i])[1]:
            if math.isclose(overlay_lat, glacier_terminus.get(overlay_result.RGIId[i])[0],abs_tol=1e-3) and math.isclose(overlay_lon, glacier_terminus.get(overlay_result.RGIId[i])[1], abs_tol=1e-3):
                terminus[i] = 1
        except:
            print("An exception occurred for ", overlay_result.RGIId[i])
    #terminus yes, no?
    df["terminus"] = terminus
    df["center"] = center
    return df


def df_glacier_grid_area(overlay_54012_filename, path_glacier_info, name_glacier_info, name_output, resolution):
    '''
    makes one csv file containing all glaciers of the world and the corresponding gridcells and information about terminus and center location of gridcell
    uses transform_df function
    :param overlay_54012_filename: shapefile resulting from overlay of grid with glacier shapes
    :param path_glacier_info: path to preprocessed dictionary which relates glacier ids to coordinates (generated with make_glacier_outlet_dict)
    :param name_glacier_info: name prefix of this dictionary
    :param name_output: output name of csv
    :param resolution: "5min" or "30min"
    :return:
    '''
    #path_glacier_info = "C:/Users/shanus/Data/Glaciers/glacier_id_dict/"
    glacier_center = pickle.load(open(path_glacier_info + name_glacier_info + "_key_id_center_{}.pkl".format(resolution), "rb"))
    glacier_terminus = pickle.load(open(path_glacier_info + name_glacier_info + "_key_id_terminus_{}.pkl".format(resolution), "rb"))

    overlay_54012 = gpd.read_file(overlay_54012_filename)
    df_all = transform_to_df(glacier_center, glacier_terminus, overlay_54012, resolution)
    df_all.to_csv(path_glacier_info+name_output+'_{}.csv'.format(resolution))

