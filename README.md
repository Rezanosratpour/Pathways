# Extreme-Value Dependence Detection for Hyper-Dimensional Datasets

This document presents the pathway-detection modelling workflow for multi-dimensional datasets. The proposed pipeline uses an empirical multivariate tail-dependence coefficient to identify extreme-value relationships and generate feature datasets for forecasting applications.

Pathways are detected across the study area using predefined time lags. The workflow also identifies homogeneous regions that share similar sources of extreme-value dependence. The analysis can be extended further in the spatial domain by applying consecutive time lags, allowing the roots and propagation structure of each extreme-value pattern to be detected.

The framework is flexible and can also be applied using a station-based approach. In this mode, the algorithm searches the spatial domain of the dataset using specified time lags for a selected station, with the aim of identifying extreme-value pathways within the provided data extent.

The input dataset is a time series of images with the following shape:

[(T, H, W)]

where:

[T = \text{number of time steps}]

[H = \text{height of the spatial grid}]

[W = \text{width of the spatial grid}]

The time lags are defined as a sequence of integer values:

[\text{lags} = [l_0, l_1, \ldots, l_d]]
 
## Overview

The pipeline is implemented primarily in:

- `CHI` : The class object and graph construction methods
- `project3/Pazzeling_layers.py` — reshaping and dataset creation for lagged CHI outputs
- `project3/Tasker.ipynb` — orchestration examples for model building and evaluation

The workflow supports two main stages:

1. Build CHI models and spatial lag graphs from raw or real IMERG data.
2. Create `pazzel` datasets by merging lagged CHI outputs and saving the resulting `X`, `Y`, and `date` arrays to HDF.

## Data inputs

The current code expects IMERG-style datasets in HDF5 or HDF format.

For `type='real'` datasets, the expected HDF file contains at least:

- `Data` — 3D array `(time, row, col)`
- `lat` — latitude coordinates
- `lon` — longitude coordinates
- `date` — date strings

For `type='raw'` datasets, the reader will look for:

- `root/data`
- `root/lat`
- `root/lon`
- `root/time`

## Key components

### `project3/CHI.py`

The `CHI` class implements the core CHI model and graph construction.

Important methods:

- `CHI(Data, lat, lon, date, Start_date, End_date, land_indices)`
  - Initializes the model and computes empirical CDF transforms for each location.
- `build_Graph_V2_parallel(lag, out_folder, percentile, n_jobs=-1, chunk_size=16)`
  - Builds a lagged CHI graph in parallel and saves output TIFFs.
- `generate_max_path2(prcntile_threshold, out_folder)`
  - Extracts the strongest CHI links and saves spatial vectors and maximum chi images.

The model stores:

- `cdf_data` — transformed probability data for each grid point
- `array_chi` — matrix of chi values between source and target locations
- `highest_chi_idx` — top high-CHI neighbors for each grid cell
- `uv` and `max_chi` — output spatial images saved as georeferenced TIFFs

### `project3/Pazzeling_layers.py`

This module reshapes CHI outputs into a dataset suitable for training a model.

Key functions:

- `X_construct(chi_model, Lags, i, I_start, I_end)`
  - Builds a lagged feature block for one land index using stored `H_memory`.
- `X_construct_reshape(chi_model, Lags, land_indices, I_start=None, I_end=None)`
  - Reconstructs full image tensors from lagged CHI points and returns aligned `X`, `Y`, and `date`.
- `load_and_merge_chi_models(model_paths)`
  - Merges `H_memory` from multiple saved CHI models into a single CHI object.
- `create_multi_lag_dataset(...)`
  - Builds a common dataset across multiple lag combinations and saves a complete HDF file.

## Example workflow in `Tasker.ipynb`

### Building models

The function `build_models(fp_hdf, dir, lag_memory, type)` performs:

- Data loading from HDF/HDF5
- Creation of a `CHI` model for the full spatial domain
- Saving the model to disk as `IMERG_chi_model_lag_<lag>.pkl`
- Parallel graph construction with `build_Graph_V2_parallel`
- Extraction of maximum chi paths with `generate_max_path2`
- Exploration of high-CHI station points via `explore_for_station_optimized2`

Example:

```python
from CHI import CHI

chi_model = CHI(Data, lat, lon , date,
                 Start_date  = date[0]  , End_date  = date[-1], 
                 land_indices= land_indices   )

chi_model.build_Graph_V2_parallel( lag = lag_0 , out_folder  = out_folder , percentile = 0.95 ) 

chi_model.generate_max_path2( prcntile_threshold = 0.95 ,out_folder  = out_folder )

chi_model.find_optimum_k(k_range= , out_folder= )
chi_model.opt_k = 
chi_model.cluster(out_folder)
chi_model.explore_hyper_dim(D = 3 , lags = lags , sampling_rates= , Percentile_sampling= , delta= )

```

```python
chi_model.explore_for_station_optimized( lat_st = -37.8136  , 
                              lon_st =  144.9631  , 
                              D = 3 , 
                              lags  = [1 , 1 , 1 ]  ,
                              Sampling_rates  = [1 , 0.01, 0.01  ] , 
                              Percentile_sampling =0.99 , 
                              out_folder  = out_folder , 
                              delta = 0.01 , 
                              tolerance = 0.001 ,  
                              name = 'Melbourne')

```

### Creating a puzzled dataset

The function `solve_pazzels(fp_models, lag_memory, of)`:

- Loads one model and merges additional CHI `H_memory`
- Builds lagged datasets for each lag combination
- use graph constrcuction for the 2-Dimentional tail depedence 
- Finds common dates across all lag views
- Concatenates channel features for each date
- Saves the result as an HDF file with `X`, `Y`, and `date`

Example:

```python
out = solve_pazzels(
    fp_models=[
        '/path/to/IMERG_chi_model_lag_2.pkl',
        '/path/to/IMERG_chi_model_lag_1.pkl'
    ],
    lag_memory=[[1,1],[1,2],[2,1]],
    of='/path/to/output/Pazz.h5'
)
```

## Outputs

The pipeline generates:

- `outputs/chi_Lag-<lag>.tiff` — CHI matrix raster for each lag
- `outputs/max_geo_Lag-<lag>.tiff` — strongest CHI path map
- `outputs/uv_geo_Lag-<lag>.tiff` — flow vector image derived from CHI extremes
- `outputs/dict_highest_idx_Lag-<lag>.hdf` — mapping of highest CHI neighbors
- `Pazz.h5` — final data file containing `X`, `Y`, and `date`

## Recommended usage

1. Place IMERG data in `project3/Datasets` or another accessible folder.
2. Run `build_models(...)` to generate CHI models for the selected lags.
3. Call `solve_pazzels(...)` to build the lagged feature dataset.
4. Use the resulting HDF for model training or evaluation.

## Dependencies

The codebase depends on:

- Python 3
- `numpy`
- `h5py`
- `tifffile`
- `joblib`
- `tqdm`
- `geopandas`
- `shapely`
- `torch`
- `tensorflow` (used elsewhere in notebook evaluation)

## Notes
- The pipeline is tuned for spatial extreme-event analysis using empirical chi statistics.
- The CHI graph uses lagged time relationships between grid cells.
- The `Pazz` dataset is designed for spatiotemporal model input with multiple lag channels.
