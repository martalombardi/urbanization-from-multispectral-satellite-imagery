# Learning Urbanization Patterns from Multispectral Satellite Data

Representation learning for urbanization analysis from Sentinel-2 multispectral imagery using GHSL-based weak supervision and metric learning.

**Big Data for Official Statistics — A.Y. 2025/2026**  
*Sapienza University of Rome*

**Marta Lombardi — 2156537**

## Overview

This project investigates whether urbanization patterns can be learned from **multispectral satellite imagery** by combining local Sentinel-2 observations with large-scale built-up information from the **Global Human Settlement Layer (GHSL)**.

The analysis uses **6,857 georeferenced EuroSAT patches** from the Highway, Industrial and Residential classes, associated with continuous GHSL built-up values.

The workflow consists of two main stages:

1. **Exploratory analysis and regression baselines** using spectral, geographic and land-use predictors.
2. **Representation learning** using a Siamese network with a pretrained ResNet-18 encoder and an adaptive contrastive loss.

GHSL is used as a weak supervisory signal to learn an embedding structured according to urbanization intensity.

## Methodology

Initial non-linear regression experiments with Random Forest, Extra Trees and XGBoost reach approximately **R² ≈ 0.65**, highlighting the limitations of spectral and geographic predictors alone in capturing urban morphology.

GHSL values are subsequently organized into three ordered urbanization strata — **Low, Medium and High** — and used to construct image pairs for metric learning.

A **Siamese ResNet-18** projects satellite patches into a **128-dimensional L2-normalized embedding space**, where the adaptive contrastive objective encourages larger separation between increasingly different urbanization levels.

## Results

The learned representation shows a strong ordering with continuous GHSL built-up intensity:

- **Spearman ρ = 0.88** on the training set
- **Spearman ρ = 0.79** on the test set
- **≈ 90–91% accuracy** in downstream Low vs. Medium/High classification

Grad-CAM analysis further indicates that the model focuses on increasingly coherent built-up structures as urbanization intensity increases.

Overall, the results show that metric learning can recover meaningful spatial structure from multispectral satellite imagery using coarse built-up information as weak supervision.

## Repository Structure

```text
urbanization-from-multispectral-satellite-imagery/
├── urbanization_from_multispectral_satellite_imagery.ipynb # complete end-to-end implementation and experimental analysis
├── urbanization_from_multispectral_satellite_imagery_presentation.pdf # presentation summarizing the motivation, methodology and main results.
├── requirements.txt # required Python packages.
└── README.md # project documentation
```

## Installation

The required Python packages are listed in `requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

## Reproducibility

The project was developed and executed in **Google Colab**.

The notebook automatically downloads EuroSAT through TorchGeo and retrieves the external geographic resources required by the analysis.

Random seeds are fixed to `42` for Python, NumPy and PyTorch.

The notebook automatically selects CUDA when a compatible GPU is available and otherwise runs on CPU.

## Data Sources

- [EuroSAT](https://github.com/phelber/eurosat)
- [Global Human Settlement Layer (GHSL) — European Commission JRC](https://human-settlement.emergency.copernicus.eu/download.php)
