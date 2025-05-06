# Case-Level MIL for CRC Lymph Node Metastasis Prediction - Analysis Code

This repository contains Python scripts used for data processing, feature extraction, and evaluation metric calculation related to the research project: **"Case-Level Multiple Instance Learning for Lymph Node Metastasis Prediction in Colorectal Cancer: A Pilot Study"**.

## Project Overview

Accurate prediction of lymph node metastasis (LNM) is crucial for managing locally advanced (T3/T4) colorectal cancer (CRC). This pilot study explored a case-level Multiple Instance Learning (MIL) framework using pre-trained models (CONCH v1.5, UNI2-h) on whole-slide images (WSIs) from 130 T3/T4 CRC patients. Features were extracted from H&E slides, clinical data was integrated, and model performance was evaluated using AUC and other metrics.

Key findings highlighted:
*   Case-level training significantly improved LNM prediction compared to slide-level training.
*   The CONCH v1.5 model generally outperformed UNI2-h in this cohort.
*   Integrating pathology features with clinical data substantially enhanced prediction accuracy.
*   Model-identified high-attention regions corresponded to known prognostic histopathological features validated by pathologists.

This repository provides the scripts used for **feature extraction from image tiles**, **calculation of evaluation metrics** (both slide-level and patient-level, with and without confidence intervals) from model outputs, and **univariate analysis of clinical data**.

*(Note: The core MIL model training scripts (e.g., using CLAM) are not included here, but this repository contains scripts to process inputs for such models and analyze their outputs.)*

## Repository Contents

This repository includes the following key Python scripts:

1.  **`extract_image_features.py`**:
    *   Extracts deep learning features from a directory of PNG image tiles using a specified pre-trained encoder model (e.g., `conch_v1_5`, `UNI2-h`).
    *   Requires a `models.py` script (or equivalent import) containing the `get_encoder` function.
    *   Saves the extracted features along with filenames into a CSV file, suitable for input into MIL frameworks like CLAM.

2.  **`univariate_logistic_regression.py`**:
    *   Performs univariate logistic regression analysis on clinical data (provided in a CSV file).
    *   Calculates Odds Ratios (OR), 95% Confidence Intervals (CI), and P-values for specified predictors against a binary outcome (e.g., 'label' for LNM status).
    *   Includes specific handling for categorical variables like 'Age'.
    *   Saves the regression results to a summary CSV file.

3.  **`calculate_clam_metrics_on_slide_level.py`**:
    *   Calculates standard evaluation metrics (AUC, Accuracy, Specificity, Sensitivity, NPV, PPV) directly at the **slide level**.
    *   Reads results from CLAM evaluation CSV files (e.g., `fold_0.csv`, `fold_1.csv`, ...). Assumes columns like `Y`, `Y_hat`, `p_1`.
    *   Summarizes metrics across all folds into a single output CSV file.

4.  **`calculate_clam_slide_metrics_ci.py`**:
    *   Similar to `calculate_clam_metrics_on_slide_level.py`, but additionally calculates **bootstrapped 95% Confidence Intervals (CIs)** for each slide-level metric.
    *   Provides a more robust assessment of metric uncertainty.
    *   Saves metrics and their CIs across all folds to a summary CSV file.

5.  **`calculate_clam_metrics_on_patient_level.py`**:
    *   Calculates evaluation metrics (AUC, Accuracy, Specificity, Sensitivity, NPV, PPV) aggregated to the **patient level**.
    *   Reads results from CLAM evaluation CSV files (`fold_*.csv`).
    *   Aggregates slide predictions/labels to the patient level (e.g., using `max` for binary labels, `mean` for probabilities based on `slide_id` prefixes). *Ensure the patient ID extraction logic matches your `slide_id` format.*
    *   Summarizes patient-level metrics across all folds into a single output CSV file.

6.  **`calculate_clam_patient_metrics_ci.py`**:
    *   Similar to `calculate_clam_metrics_on_patient_level.py`, but additionally calculates **bootstrapped 95% Confidence Intervals (CIs)** for each patient-level metric.
    *   Aggregates slide results to the patient level before bootstrapping.
    *   Saves patient-level metrics and their CIs across all folds to a summary CSV file.

## Workflow Context

These scripts fit into the research workflow as follows:

1.  **WSI Preprocessing:** WSIs are tiled into smaller PNG images (external process).
2.  **Feature Extraction:** `extract_image_features.py` is used on the PNG tiles to generate feature vectors (e.g., using CONCH).
3.  **MIL Model Training:** The extracted features are used to train an MIL model (e.g., CLAM). This step typically generates evaluation output files (e.g., `fold_0.csv`, `fold_1.csv`, ...) containing slide-level predictions, true labels, and probabilities. (Training script not included).
4.  **Evaluation Metric Calculation:**
    *   `calculate_clam_metrics_on_slide_level.py` / `_ci.py` analyze the `fold_*.csv` files directly for slide-level performance.
    *   `calculate_clam_metrics_on_patient_level.py` / `_ci.py` analyze the `fold_*.csv` files, aggregate results per patient, and then calculate patient-level performance.
5.  **Clinical Data Analysis:** `univariate_logistic_regression.py` analyzes separate clinical data CSV files to assess the individual contribution of clinical factors to LNM prediction.

## Prerequisites

*   Python 3.x
*   Required Python packages:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `statsmodels` (for `univariate_logistic_regression.py`)
    *   `torch` (for `extract_image_features.py`)
    *   `Pillow` (PIL) (for `extract_image_features.py`)
    *   `tqdm` (for progress bars)
*   Access to a `models.py` script or library providing the `get_encoder` function used in `extract_image_features.py`.
*   (Optional but recommended) A virtual environment manager like `conda` or `venv`.

You can typically install the required packages using pip:
```bash
pip install pandas numpy scikit-learn statsmodels torch torchvision Pillow tqdm
# Note: Ensure you install the correct PyTorch version for your system (CPU/GPU)
# See: https://pytorch.org/get-started/locally/
