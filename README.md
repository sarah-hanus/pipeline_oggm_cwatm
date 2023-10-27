**GENERAL INFORMATION**

This repository contains a set of functions and scripts that transform OGGM simulation results to netcdf files which can be used as input to the hydrological model CWatM.

This pipeline is part of the model development described in the paper: "Coupling a large-scale glacier and hydrological model (OGGM v1.5.3 and CWatM V1.08) – Towards an improved representation of mountain water resources in global assessments"

When using this repository, please refer to the original publication in addition to this Zenodo repository.

**DATA & FILE OVERVIEW**

The data is structured in thre directories
- Data: contains data necessary for processing
  - glacier_statistics: csv files of glacier statistics based on RGIv6.0 generated with OGGM (https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/no_match/RGI62/b_010/L3/summary/)
  - cellarea: netcdf files of cell areas which are used as CWatM input files
  - grid_shp: shapefiles of global grid at 30arcmin resolution
  - oggm_results: example OGGM result for the GLomma basin (additional OGGM output data can be accessed via https://doi.org/10.5281/zenodo.10046823)
  - gloma_discharge_totalavg.nc CWatM output file for example, which is used as template to create netcdf
- glaciers_preprocessed: results of glacier_preprocessing.py, independent of OGGM results, does only have to be repeated if different glacier shapefiles are used, these files are used to transfer OGGM output to CWatM input
  - glacier_area_df: csv file with information about which glacier covers which grid cells and which grid cells contain the center and terminus of glacier
  - glaciers_key_coordinates_terminus.pkl: dictionary with info about which glacier has terminus in which grid cell
- glacier_postprocessing_functions.py: Functions to translate OGGM output to netcdf files for CWatM input
- glacier_postprocessing_world.py: script to process global OGGM results to global CWatM input files (ATTENTION: paths must be adapted)
- glacier_postprocessing_basins.py: script to process OGGM results of one basin to CWatM input files for this basin  (ATTENTION: paths must be adapted)

**Don't hesitate to contact us in case of any questions (sarah.hanus@geo.uzh.ch)**