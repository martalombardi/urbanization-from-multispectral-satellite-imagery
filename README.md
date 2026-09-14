# Learning Urbanization Patterns from Multispectral Satellite Data

Representation learning for urbanization analysis from Sentinel-2 multispectral imagery using GHSL-based weak supervision and metric learning.

**Big Data for Official Statistics — A.Y. 2025/2026**  
*Sapienza University of Rome*

**Marta Lombardi — 2156537**

## Overview

This project investigates whether urbanization patterns can be learned from **multispectral satellite imagery** by combining local Sentinel-2 observations with large-scale built-up information from the **Global Human Settlement Layer (GHSL)**.

The main challenge is the **spatial-scale mismatch between the two data sources**: EuroSAT provides local 64×64 Sentinel-2 image patches, while GHSL represents built-up surface at a substantially coarser spatial resolution. GHSL is therefore used as a **weak supervisory signal**, rather than as an exact pixel-level ground truth.

The analysis uses **6,857 georeferenced EuroSAT patches** from the Highway, Industrial and Residential classes, associated with continuous GHSL built-up values.

The workflow consists of two main stages:

1. **Exploratory analysis and regression baselines** using spectral, geographic and land-use predictors.
2. **Representation learning** using a Siamese network with a pretrained ResNet-18 encoder and an adaptive contrastive loss.

The objective is to learn an embedding space in which satellite observations are organized according to their underlying urbanization intensity.

## Methodology

Initial non-linear regression experiments with Random Forest, Extra Trees and XGBoost reach approximately **R² ≈ 0.65**, highlighting the limitations of spectral and geographic predictors alone in capturing urban morphology.

To derive a discrete supervisory structure from the continuous GHSL signal, **clustering is applied to GHSL built-up values to identify three data-driven urbanization strata — Low, Medium and High**. These ordered strata are then used to construct positive and negative image pairs for metric learning.

A **Siamese ResNet-18** projects satellite patches into a **128-dimensional L2-normalized embedding space**. Images belonging to the same urbanization stratum form positive pairs, while images from different strata form negative pairs.

Training uses an **adaptive contrastive loss**, with larger separation margins assigned to pairs belonging to more distant urbanization strata. The resulting embedding is therefore encouraged to preserve not only similarity within each group, but also the **ordinal structure of urbanization intensity**.

## Results

The learned representation shows a strong ordering with continuous GHSL built-up intensity:

- **Spearman ρ = 0.88** on the training set
- **Spearman ρ = 0.79** on the test set
- **≈ 90–91% accuracy** in downstream Low vs. Medium/High classification

Grad-CAM analysis further indicates that the model focuses on increasingly coherent built-up structures as urbanization intensity increases.

Overall, the results show that metric learning can recover spatial structure associated with urbanization from local multispectral satellite observations, despite supervision being available only at a coarser spatial scale.


### Notebook

[View the complete notebook with nbviewer]([https://nbviewer.org/github/martalombardi/urbanization-from-multispectral-satellite-imagery/blob/main/urbanization_from_multispectral_satellite_imagery.ipynb](https://nbviewer.org/github/martalombardi/urbanization-from-multispectral-satellite-imagery/blob/main/urbanization_from_multispectral_satellite_imagery.ipynb)

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
