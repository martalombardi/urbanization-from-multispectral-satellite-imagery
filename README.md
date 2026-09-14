# Learning Urbanization Patterns from Multispectral Satellite Data

Representation learning for urbanization analysis from Sentinel-2 multispectral imagery using GHSL-based weak supervision and metric learning.

**Big Data for Official Statistics — A.Y. 2025/2026**  
*Sapienza University of Rome*

**Marta Lombardi — 2156537**

## Overview

This project investigates whether urbanization patterns can be learned directly from multispectral satellite imagery.

The analysis combines **EuroSAT Sentinel-2 image patches** with continuous built-up intensity values derived from the **Global Human Settlement Layer (GHSL)**. The two data sources operate at different spatial scales: EuroSAT provides local 64×64 image patches, while GHSL represents built-up surface at a substantially coarser resolution.

The workflow first evaluates how much urbanization information can be explained by spectral, geographic and land-use predictors through exploratory analysis and non-linear regression baselines. It then moves to image-based representation learning using a **Siamese neural network with a pretrained ResNet-18 encoder** and an adaptive contrastive objective.

The learned embedding space is evaluated through urbanization ranking, downstream classification and Grad-CAM interpretability analysis.

## Data

The project combines two Earth Observation datasets:

- **EuroSAT** — georeferenced 64×64 multispectral Sentinel-2 image patches.
- **Global Human Settlement Layer (GHSL)** — built-up surface estimates used as a continuous proxy for urbanization intensity.

The analysis focuses on three EuroSAT classes directly informative of built-up environments:

- Highway
- Industrial
- Residential

The resulting dataset contains **6,857 satellite patches** associated with GHSL built-up values.

## Exploratory Analysis

The exploratory analysis investigates:

- the distribution of GHSL built-up intensity;
- differences and overlap across land-use classes;
- geographic variability across Europe;
- relationships between Sentinel-2 spectral bands and GHSL;
- spectral redundancy and inter-band correlations;
- feature relevance using **Spearman correlation** and **mutual information**.

Non-linear regression baselines are then trained using selected spectral bands together with geographic and land-use information. The evaluated models include:

- Random Forest;
- Extra Trees;
- XGBoost.

Their performance stabilizes around **R² ≈ 0.65**, suggesting that tabular spectral and geographic predictors alone do not fully capture the spatial and morphological information associated with urbanization.

## Representation Learning

To move from tabular predictors to image structure, GHSL built-up values are converted into three ordered urbanization strata:

- Low
- Medium
- High

The strata are obtained through data-driven clustering while preserving their ordinal relationship.

A **Siamese neural network** is then trained on pairs of satellite images:

- images from the same urbanization stratum form positive pairs;
- images from different strata form negative pairs.

The encoder is based on a pretrained **ResNet-18**, followed by global average pooling and projection into a **128-dimensional L2-normalized embedding space**.

Training uses an **adaptive contrastive loss**, assigning larger separation margins to pairs belonging to more distant urbanization levels.

## Results

The learned representation exhibits a strong global ordering with respect to continuous GHSL built-up intensity.

Spearman correlation between the embedding-derived urbanization score and GHSL is approximately:

- **Train:** ρ = 0.88
- **Test:** ρ = 0.79

The embeddings are also evaluated through a downstream binary classification task separating low built-up areas from medium/high built-up areas.

Several standard classifiers achieve consistent test performance around **90–91% accuracy**, indicating that the learned representation already contains substantial information about built-up intensity.

Grad-CAM analysis further shows systematic differences across urbanization strata:

- low-urbanization samples tend to produce weaker and more diffuse activations;
- intermediate samples show more localized but heterogeneous responses;
- high-urbanization samples produce stronger and more spatially coherent activations over dense built-up structures.

These results indicate that the learned embedding captures spatial structure aligned with the GHSL supervisory signal. Since GHSL is also used to construct the weak supervision, the embedding–GHSL correlation should be interpreted as evidence of successful representation learning with respect to this proxy rather than as an independent validation of urbanization itself.

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
