# Radar & Hyperspectral Vegetation Classification

XGBoost-based land cover classification for Arctic/tundra vegetation mapping using multi-sensor remote sensing data.

## Overview

This repository contains a Jupyter notebook that performs supervised land cover classification by fusing hyperspectral imagery with polarimetric Synthetic Aperture Radar (polSAR) data. The workflow is designed for Arctic vegetation mapping where class imbalance is common.

## Remote Sensing Data

### Hyperspectral Imagery
**Wyvern Dragonette** satellite hyperspectral data with 31 spectral bands. The imagery has been denoised using a Minimum Noise Fraction (MNF) transformation in ENVI, retaining the three bands that explain the most spectral variance. Hyperspectral data captures the spectral reflectance signatures of vegetation, enabling differentiation between species based on their unique absorption and reflectance patterns across visible to near-infrared wavelengths.

### Polarimetric SAR
**UAVSAR** (Uninhabited Aerial Vehicle Synthetic Aperture Radar) L-band data processed as a **Pauli decomposition** RGB composite. Polarimetric SAR provides structural information about vegetation canopy architecture:
- **Surface scattering** (single-bounce): Dominant in bare ground, water, and sparse vegetation
- **Volume scattering** (multiple bounces): Indicates vegetation canopy density and biomass
- **Double-bounce scattering**: Associated with vertical structures like shrub stems

The fusion of spectral (hyperspectral) and structural (polSAR) information improves classification accuracy by capturing complementary vegetation characteristics that neither sensor can fully resolve alone.

## Classification Approach

The workflow uses **XGBoost (Extreme Gradient Boosting)**, a tree-based ensemble machine learning algorithm, for supervised land cover classification. Remote sensing band values serve as the **explanatory variables (features)** for predicting land cover class membership:

- **Hyperspectral MNF bands (3)**: Capture spectral variance related to vegetation biochemistry and species composition
- **Pauli decomposition RGB bands (3)**: Encode structural scattering properties related to canopy architecture and biomass

XGBoost iteratively builds decision trees that correct errors from previous trees, making it robust to class imbalance and capable of handling non-linear relationships between spectral/radar features and vegetation types. The model includes:

- **Random undersampling** to balance training samples across minority and majority classes
- **5-fold stratified cross-validation** for unbiased accuracy estimation
- **Early stopping** to prevent overfitting by monitoring validation loss
- **Hyperparameter tuning** options via randomized search

## Workflow

1. **Load Training Data** - Imports hyperspectral MNF (Minimum Noise Fraction) transformed imagery, SAR Pauli decomposition RGB, and reference land cover points
2. **Feature Stacking** - Combines multi-sensor bands into a unified feature stack, with optional spectral indices (NDVI, NDWI, BSI)
3. **Class Balancing** - Applies random undersampling to handle class imbalance, capping majority classes at a configurable multiplier of the minority class size
4. **Model Training** - Trains an XGBoost classifier with 5-fold stratified cross-validation and early stopping
5. **Prediction** - Applies the trained model to classify all valid pixels in the imagery
6. **Visualization** - Generates interactive maps with satellite basemap overlay using hvplot/holoviews
7. **Export** - Saves predictions as GeoTIFF with embedded colormap for ArcGIS compatibility

## Land Cover Classes

| Code | Class | Description |
|------|-------|-------------|
| 1 | DST | Deciduous Shrub Tundra |
| 2 | TST | Tussock Tundra |
| 3 | LTDST | Low to Tall Deciduous Shrub Tundra |
| 4 | OST | Open Shrub Tundra |
| 5 | Wet | Wetland |
| 6 | Waterbody | Water |
| 7 | PBHV | Partially Barren Herbaceous Vegetation |
| 8 | Barren | Barren Ground |
| 9 | SASH | Sandy Shore |
| 10 | BRN | Burned Areas |

## Input Data Structure

```
input/
├── LandcoverRef.tif                  # Reference land cover raster (training labels)
├── SAR Pauli/                        # UAVSAR Pauli decomposition RGB
│   └── PauliRGB.tif
├── Wyvern Tiles Nohistmatch FMNF/    # Wyvern Dragonette MNF-transformed (3 bands)
│   └── mosaic_nomatch_nofeather_quac_fmnf.dat
└── Wyvern Tiles MNF3C/               # Alternative hyperspectral source for indices
    └── mosaic_nomatch_nofeather_quac_imnf.dat
```

## Output

Each run creates a timestamped folder in `output/` containing:
- `*.tif` - Classified land cover raster (uint8 with embedded colormap)
- `*.clr` - ArcGIS color file for symbology
- `*.txt` - Run log with parameters, class distributions, and accuracy metrics
- `*_learning_curves.png` - Training/validation loss curves and confusion matrix

## Configuration

Key parameters in the notebook's Configuration section:

| Parameter | Description |
|-----------|-------------|
| `INPUT_DATASETS` | List of input rasters with band selection |
| `SPECTRAL_INDICES` | Optional vegetation indices (NDVI, NDWI, BSI) |
| `BALANCE_MULTIPLIER` | Max ratio of majority to minority class samples |
| `XGBOOST_PARAMS` | XGBoost hyperparameters |
| `EARLY_STOPPING` | Stop training when validation loss plateaus |

## Requirements

- Python 3.8+
- xarray, rioxarray, rasterio
- xgboost (with CUDA support recommended)
- scikit-learn, imbalanced-learn
- hvplot, holoviews, geoviews
- matplotlib

## Usage

1. Place input data in the `input/` folder following the structure above
2. Open `XGBoost Vegetation Classification Balanced LCClasses.ipynb`
3. Adjust configuration parameters as needed
4. Run all cells

## License

See [LICENSE](LICENSE) for details.
