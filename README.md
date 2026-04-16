# Pathways Detection Repository

A Python repository for detecting and analysing **extreme precipitation pathways** using **tail dependence measures**, **graph-based representations**, and **clustering methods**. The framework is designed for spatiotemporal precipitation data and can be used to identify regions with strong upper-tail dependence, infer propagation pathways of extreme rainfall, and support downstream prediction and hydroclimatic analysis.

## Overview

Extreme precipitation events often exhibit complex spatial organisation and temporal propagation that are not fully captured by conventional correlation-based methods. This repository provides a workflow for:

- constructing a spatial dependency structure from precipitation data,
- estimating **upper-tail dependence** between locations,
- identifying **homogeneous regions** using clustering,
- extracting likely **pathways of extreme precipitation propagation**, and
- analysing the results across different temporal lags.

The methodology is particularly suitable for satellite- and gauge-based precipitation products over large domains such as Australia.

## Main Features

- Tail dependence analysis focused on extremes rather than average behaviour
- Support for **multi-lag** spatiotemporal dependency analysis
- Graph-based representation of grid cells or stations
- Clustering of locations into homogeneous regions
- Detection of directional or lag-dependent extreme precipitation pathways
- Flexible workflow for large gridded precipitation datasets
- Compatible with Python-based scientific and geospatial tools

## Methodological Summary

The general workflow of the repository is as follows:

1. **Input precipitation data** are loaded as gridded or station-based time series.
2. A set of **extreme thresholds** is defined, typically based on high quantiles such as the 95th or 99th percentile.
3. For each pair of locations, the **upper-tail dependence coefficient** is estimated to measure the probability of joint extreme behaviour.
4. The study domain is represented as a **graph**, where:
   - nodes correspond to grid cells or stations,
   - edges represent pairwise dependence, and
   - edge weights are derived from upper-tail dependence.
5. The resulting dependence structure is used to form a similarity representation for **clustering**.
6. **K-means++** or related clustering methods are applied to identify homogeneous regions.
7. By analysing dependence across multiple time lags, the framework infers likely **pathways of extreme precipitation propagation**.
8. Outputs are summarised as maps, clusters, adjacency structures, and lag-dependent pathway diagnostics.

## Typical Applications

- Extreme rainfall pathway detection
- Hydroclimatic regionalisation
- Event propagation analysis
- Identification of precursor regions for heavy rainfall
- Support for forecasting and early warning studies
- Teleconnection and circulation-related precipitation analysis

## Repository Structure

A typical project structure may look like this:

```text
pathways-detection/
├── data/
│   ├── raw/                # Raw precipitation data
│   ├── processed/          # Preprocessed inputs
│   └── external/           # Ancillary datasets
├── notebooks/              # Exploratory notebooks
├── src/
│   ├── io/                 # Data loading and saving
│   ├── preprocessing/      # Resampling, masking, cleaning
│   ├── extremes/           # Thresholding and tail dependence
│   ├── graph/              # Graph construction and weighting
│   ├── clustering/         # K-means++ and region detection
│   ├── pathways/           # Pathway extraction and lag analysis
│   └── utils/              # Helper functions
├── results/
│   ├── figures/            # Maps and visualisations
│   ├── tables/             # Summary tables
│   └── models/             # Saved outputs and intermediate files
├── config/                 # Configuration files
├── requirements.txt
└── README.md
```

Adjust this structure as needed to match the actual repository layout.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pathways-detection.git
cd pathways-detection
```

Create and activate a Python environment:

```bash
conda create -n pathways python=3.11
conda activate pathways
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Suggested Dependencies

Depending on the implementation, the repository may use:

```text
numpy
pandas
scipy
scikit-learn
xarray
netCDF4
h5py
matplotlib
geopandas
rasterio
joblib
tqdm
networkx
```

Additional domain-specific packages can be included depending on the data format and analysis workflow.

## Input Data

The repository is intended for precipitation datasets such as:

- satellite precipitation products,
- gauge observations,
- reanalysis-derived rainfall fields, or
- merged precipitation datasets.

Typical input format:

- **3D gridded array**: `(time, lat, lon)`
- **2D station matrix**: `(time, station)`

Recommended preprocessing steps:

- temporal alignment,
- missing-value handling,
- spatial masking,
- regridding or resampling,
- threshold calculation for extremes.

## Example Workflow

### 1. Preprocess the precipitation dataset

```python
from src.preprocessing import prepare_data

data = prepare_data("data/raw/precipitation.nc")
```

### 2. Estimate upper-tail dependence

```python
from src.extremes import compute_tail_dependence

chi = compute_tail_dependence(data, q=0.95)
```

### 3. Construct the dependence graph

```python
from src.graph import build_graph

G = build_graph(chi)
```

### 4. Cluster the domain into homogeneous regions

```python
from src.clustering import cluster_regions

labels = cluster_regions(chi, n_clusters=4)
```

### 5. Detect lag-based pathways

```python
from src.pathways import detect_pathways

pathways = detect_pathways(data, labels, lags=[1, 2, 3])
```

> The function names above are placeholders and should be replaced with the actual function names used in the repository.

## Outputs

Typical outputs may include:

- pairwise upper-tail dependence matrices,
- cluster labels for each location,
- lag-based pathway maps,
- pathway strength summaries,
- adjacency or transition tables,
- diagnostic plots and evaluation figures.

Examples:

- `results/figures/cluster_map.png`
- `results/figures/pathways_lag1.png`
- `results/tables/tail_dependence_summary.csv`
- `results/models/pathway_graph.pkl`

## Configuration

A configuration file can be used to define analysis settings such as:

```yaml
data_path: data/raw/precipitation.nc
quantile: 0.95
lags: [1, 2, 3]
n_clusters: 4
min_samples: 30
output_dir: results/
```

This improves reproducibility and makes experiments easier to track.

## Scientific Rationale

Unlike standard correlation measures, **tail dependence** focuses specifically on the co-occurrence of extreme values. This is important for precipitation extremes because two locations may show weak average correlation while still exhibiting strong dependence during severe rainfall events.

By combining tail dependence with clustering and lag analysis, the framework can reveal:

- regions that behave similarly during extremes,
- likely source-to-target relationships,
- directional propagation patterns, and
- spatial structures relevant to forecasting and hazard assessment.

## Performance Considerations

For large spatiotemporal domains, pairwise dependence estimation can be computationally demanding. To improve performance, consider:

- parallel processing with `joblib`,
- chunked computation,
- storing intermediate outputs in HDF5 or NetCDF,
- restricting computations to land cells or valid masks,
- heuristic sampling for higher-dimensional dependence analysis.

## Reproducibility

To ensure reproducible results:

- fix random seeds where relevant,
- keep preprocessing settings consistent,
- save intermediate matrices and labels,
- document thresholds, lags, and clustering parameters.

## Citation

If you use this repository in academic work, please cite the associated paper or manuscript. A sample BibTeX entry can be added here:

```bibtex
@article{yourpaper2026,
  title={Extreme Precipitation Pathway Detection Using Tail Dependence and Clustering},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2026}
}
```

## Contributing

Contributions are welcome. Suggested contributions include:

- improving computational efficiency,
- extending the workflow to higher-dimensional extremes,
- adding new clustering or graph-based methods,
- improving documentation and examples,
- validating the framework on additional datasets and regions.

## License

Add the project license here, for example:

```text
MIT License
```

## Contact

For questions, collaborations, or bug reports, please open an issue or contact the repository maintainer.

---

This README is a general template for the **Pathways Detection Repository** and can be customised to match the exact code structure, filenames, and scientific scope of the project.
