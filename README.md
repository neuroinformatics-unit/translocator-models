# translocator-models

## Introduction

This repository contains scripts and modules used for modeling and analyzing fold changes in population firing rates in the visual cortex under various stimulus conditions. The scripts implement rate-based models, fitting procedures, and data visualization tools used in Velez-Fort 2025. If you have any questions, please open an issue.

## Usage

Dependencies are declared in the pyproject.toml file. To create a virtual environment and install the dependencies, run the following commands:

```bash
conda create -n translocator-models python=3.11
conda activate translocator-models
pip install .
```

To run the script that fits the model to the data and visualizes the results, use the following command:

```bash
python translcator-models/fit_the_model.py
```

## Data Description

### Datasets

- **`visual_flow`**: Mean fold changes for visual flow stimuli with 39 clusters.
- **`passive_same_luminance`**: Mean fold changes for passive same luminance stimuli with 151 clusters.
- **`matched_dataset`**: Re-balanced fold changes in a matched dataset, including conditions simulating slip.

### Conditions

- **V**: Visual flow only.
- **VT**: Visual flow with passive translation.
- **RV**: Visual flow with static running.
- **RVT**: Locomotion with visual flow.
- **T**: Passive translation with visual static stimulus.
- **RV_slip**: Running with increased visual flow (simulating slip).
- **RVT_slip**: Locomotion with visual flow and translation slip.

## Model Description

### Rate-Based Model

The rate-based model computes the fold change in population firing rate based on stimuli for visual flow (`VF`), translation (`T`), and running (`R`). It uses adaptive responses and combines them with weights and thresholds to model the neural population response.

The model equations:

$$v = w_1(VF > 0) + w_2 ((T-R) > 0)(T-R) + w_3R$$
$$\hat fc(v) = \alpha v + c$$


where:
- $v$ is the neuron rate response above baseline in arbitrary units,
- $w_1$, $w_2$, and $w_3$ are weights applied to each component of the response,
- $\alpha$ is a scaling factor, and $c$ is a constant offset.

In this formulation:
- The expressions $(VF > 0)$ and $((T - R) > 0)$ act as adaptive thresholds, activating their respective terms only if the stimulus exceeds a certain threshold.
- The resulting value, $v$, represents the weighted, thresholded combination of these stimuli, and the final fold change $\hat fc(v)$ scales and offsets this response.

In the article we consider the following weights:
- $w_1 = 1$
- $w_2 = 0.6$
- $w_3 = 1$
- $\alpha = 0.8$
- $c = 1$
which simplifies the model equations to:

$$v = (VF > 0) + 0.6((T-R) > 0)(T-R) + R$$
$$\hat fc(v) = 0.8v + 1$$

These weights can also be optimized to fit the model to the data, leading to a similar result.

### Arithmetic Sum Model

The arithmetic sum model predicts the fold change by combining empirical fold changes using the formula:

$$\hat fc(VF + T) = \beta_0 + \beta_1 fc(T_{VS}) + \beta_2 fc(VF)$$

- $fc(T_{VS})$ is the fold change from translation with constant luminance.
- $fc(VF)$ is the fold change from visual flow.
- $\beta_0$, $\beta_1$, and $\beta_2$ are weights and an intercept for the model that have been previously fitted to the data.

## Functions Overview

- **Model Functions (`models.py`):**
  - `adaptive_response`: Processes stimuli based on thresholds.
  - `rate_based_model`: Computes fold changes using the rate-based model.
  - `arithmetic_sum_model`: Predicts fold changes using the arithmetic sum model.

- **Fitting Functions (`fitting_methods.py`):**
  - `fit_fold_changes_to_data`: Fits the rate-based model to experimental data.
  - `get_predicted_fold_changes_*`: Compute predicted fold changes for different datasets.

- **Main Script Functions (`fit_the_model.py`):**
  - `fit_and_print_results`: Fits the model to data, prints results, and computes arithmetic sum predictions if applicable.
