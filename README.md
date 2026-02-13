# Radar & Hyperspectral Vegetation Classification

XGBoost-based land cover classification and regression for Arctic/tundra vegetation mapping using multi-sensor remote sensing data.

## Overview

This repository contains a Jupyter notebook that performs supervised land cover classification (or continuous-value regression) by fusing multiple remote sensing datasets. The workflow is designed for Arctic vegetation mapping as part of a master's thesis, using vegetation field reference and remote sensing data from the Inuvik-Tuktoyaktuk region of Canada's Northwest Territories.

The pipeline is modular: any combination of input datasets can be enabled -- hyperspectral imagery, polarimetric SAR, Sentinel-2 multispectral, and/or foundation model embeddings (e.g., AlphaEarth) -- and the notebook will automatically stack the selected bands and optional spectral indices into a unified feature set.

## Remote Sensing Data

### Hyperspectral Imagery
**Wyvern Dragonette** satellite hyperspectral data with 31 spectral bands. The imagery has been denoised using a Minimum Noise Fraction (MNF) transformation in ENVI, retaining the three bands that explain the most spectral variance. Hyperspectral data captures the spectral reflectance signatures of vegetation, enabling differentiation between species based on their unique absorption and reflectance patterns across visible to near-infrared wavelengths.

### Polarimetric SAR
**UAVSAR** (Uninhabited Aerial Vehicle Synthetic Aperture Radar) L-band data processed as a **Pauli decomposition** RGB composite. Polarimetric SAR provides structural information about vegetation canopy architecture:
- **Surface scattering** (single-bounce): Dominant in bare ground, water, and sparse vegetation
- **Volume scattering** (multiple bounces): Indicates vegetation canopy density and biomass
- **Double-bounce scattering**: Associated with vertical structures like shrub stems

### Sentinel-2 Multispectral
**Sentinel-2** L2A imagery providing 12 spectral bands from visible to shortwave infrared. Can be used both as raw band features and as a source for spectral indices (NDVI, NDWI, BSI, etc.).

### Foundation Model Embeddings
**AlphaEarth** (or similar) foundation model embeddings -- high-dimensional feature vectors (e.g., 63 bands) derived from pre-trained remote sensing models. These encode learned representations of land surface characteristics without requiring hand-crafted feature engineering.

The fusion of spectral, structural, and learned features improves classification accuracy by capturing complementary vegetation characteristics that no single sensor can fully resolve alone.

## Classification Approach

The workflow uses **XGBoost (Extreme Gradient Boosting)**, a tree-based ensemble machine learning algorithm, for supervised land cover classification or continuous-value regression.

### Training Data

Training labels can be provided as either:
- **Raster mode:** A classified raster (e.g., `LandcoverRef.tif`) where each pixel has a class label
- **Vector mode:** A shapefile or GeoPackage with point or polygon features, each attributed with a class label or continuous target value

Training references are derived from **field plots and drone interpretation**: (1) **1x1 m vegetation plots** described in the field and assigned to land cover classes, and (2) a complementary set of **training pixels** manually interpreted from **4 cm drone imagery**.

### Model Features

- **Random undersampling** to balance training samples across minority and majority classes (configurable multiplier)
- **5-fold stratified cross-validation** for unbiased accuracy estimation
- **Early stopping** to prevent overfitting by monitoring validation loss
- **Randomized hyperparameter search** (optional) for automated tuning
- **Chunked prediction** to handle large images without running out of memory
- **Time series prediction** to apply a trained model across multiple dates

## Workflow

1. **Configure** -- Set prediction mode, training data source, input datasets, and model parameters
2. **Load Training Data** -- Stack multi-sensor bands into a unified feature set; compute optional spectral indices (NDVI, NDWI, BSI, etc.)
3. **Class Balancing** -- Apply random undersampling, capping majority classes at a configurable multiplier of the minority class size
4. **Model Training** -- Train XGBoost with cross-validation, early stopping, and optional hyperparameter search
5. **Prediction** -- Classify/regress all valid pixels in configurable chunks to avoid OOM errors
6. **Visualization** -- Generate interactive maps with satellite basemap overlay using hvplot/holoviews
7. **Export** -- Save predictions as GeoTIFF with embedded colormap and a detailed run log
8. **Time Series** (optional) -- Apply the trained model to a folder of multi-date images

## Land Cover Classes

| Code | Class | Description |
|------|-------|-------------|
| 1 | BAR | Barren Ground |
| 2 | BRN | Burned Areas |
| 3 | DST | Dwarf Shrub Tundra |
| 4 | LTDST | Low to Tall Deciduous Shrub Tundra |
| 5 | OST | Open Shrub Tundra |
| 6 | PBHV | Partially Barren Herbaceous Vegetation |
| 7 | TST | Tussock Tundra |
| 8 | Waterbody | Water |
| 9 | Wet | Wetland |

*Note: Class names and count are auto-generated from vector training data when `VECTOR_CLASS_MAPPING=None`. The table above reflects the current default configuration.*

## Input Data Structure

```
input/
├── LandcoverRef.tif                              # Raster training labels (optional)
├── LandcoverClassification.shp                    # Vector training labels (optional)
├── SAR Pauli/                                     # UAVSAR Pauli decomposition RGB
│   └── PauliRGB.tif
├── Wyvern Tiles Nohistmatch FMNF/                 # Wyvern Dragonette MNF (3 bands)
│   └── mosaic_nomatch_nofeather_quac_fmnf.dat
├── Wyvern Tiles MNF3C/                            # Hyperspectral source for spectral indices
│   └── mosaic_nomatch_nofeather_quac_imnf.dat
├── S2/                                            # Sentinel-2 multispectral
│   └── *.tif
└── AlphaEarth/                                    # Foundation model embeddings (63 bands)
    └── *.tiff
```

## Output

Each run creates a timestamped folder in `output/` containing:
- `*.tif` -- Classified raster (uint8 with embedded colormap) or regression raster (float32)
- `*.clr` -- ArcGIS color file for symbology (classification only)
- `*.txt` -- Run log with parameters, class distributions, and accuracy metrics
- `*_learning_curves.png` -- Training/validation loss curves and confusion matrix

Time series runs additionally create a `timeseries/` subfolder with per-date GeoTIFFs.

## Configuration

Key parameters in the notebook's Configuration section (Cell 2):

| Parameter | Description |
|-----------|-------------|
| `PREDICTION_MODE` | 'classification' or 'regression' |
| `TRAINING_DATA_MODE` | 'raster' or 'vector' training labels |
| `INPUT_DATASETS` | List of input rasters with band selection and optional spectral indices |
| `XGBOOST_PARAMS` | XGBoost hyperparameters (n_estimators, max_depth, learning_rate, etc.) |
| `BALANCE_MULTIPLIER` | Max ratio of majority to minority class samples |
| `PREDICTION_CHUNK_SIZE` | Pixels per prediction batch (reduce if OOM) |

Advanced parameters in Cell 4:

| Parameter | Description |
|-----------|-------------|
| `REGRESSION_CONFIG` | Output dtype, NoData, clipping, and outlier filtering for regression |
| `EARLY_STOPPING` | Enable/disable and set patience rounds |
| `RANDOM_SEARCH_ENABLED` | Toggle automated hyperparameter tuning |
| `CLASS_NAMES` | Land cover class name mapping (auto-generated or manual) |

## Requirements

- Python 3.8+
- xarray, rioxarray, rasterio
- xgboost (with CUDA support recommended)
- scikit-learn, imbalanced-learn
- hvplot, holoviews, geoviews, bokeh
- matplotlib, numpy
- scipy (for hyperparameter search distributions)

Conda environment name: `DaskXArray`

## Usage

1. Place input data in the `input/` folder following the structure above
2. Open `XGBoost Vegetation Classification Balanced LCClasses.ipynb`
3. In Section 1, configure prediction mode, training source, and enable desired datasets
4. Adjust hyperparameters and balancing as needed
5. Run all cells

## License

See [LICENSE](LICENSE) for details.
