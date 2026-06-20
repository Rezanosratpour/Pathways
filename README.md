# Extreme Value Pathways in Multi-Dimentional Space

This document describes the Pathways detection modelling workflow used in `project3` for Multi-Dimentional datasets. The pipeline uses empirical multivariate tail dependence coeffient to identify extreme value relationships and build feature datasets for forecasting.

## Overview

The pipeline is implemented primarily in:

- `CHI` " core model class and graph construction methods
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
fp_models = build_models(
    fp_hdf='/path/to/IMERG_1deg_1day_Data_moved.hdf5',
    dir='/path/to/output/folder',
    lag_memory=[[1,1],[1,2],[2,1]],
    type='real'
)
```

### Creating a puzzled dataset

The function `solve_pazzels(fp_models, lag_memory, of)`:

- Loads one model and merges additional CHI `H_memory`
- Builds lagged datasets for each lag combination
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
