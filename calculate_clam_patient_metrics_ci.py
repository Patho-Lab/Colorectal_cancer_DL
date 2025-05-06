#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates and summarizes patient-level evaluation metrics (AUC, Accuracy, 
Specificity, Sensitivity, NPV, PPV) with bootstrapped confidence intervals
from CLAM evaluation result CSV files (fold_*.csv).

Reads CSV files for each fold, performs patient-level aggregation, calculates 
metrics and their 95% CIs using bootstrapping, and saves a summary CSV file.
"""

import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import warnings
from tqdm import tqdm # Added for progress bar during bootstrapping

# --- Configuration ---
INPUT_FOLD_DIR = ''  # <--- **REPLACE WITH YOUR INPUT DIRECTORY**
OUTPUT_SUMMARY_FILENAME = '' # Filename for the summary CSV
NUM_FOLDS = 5  # Number of folds (e.g., fold_0.csv to fold_4.csv)
N_BOOTSTRAP = 1000 # Number of bootstrap samples for CI calculation
CONFIDENCE_INTERVAL = 0.95 # Confidence interval level
# --- End Configuration ---

# --- Helper Functions for Metric Calculation with Bootstrapping ---

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, confidence_level=0.95, **metric_kwargs):
    """
    Calculates a metric and its confidence interval using bootstrapping.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predictions (probabilities or binary).
        metric_func (callable): The scikit-learn metric function 
                                (e.g., roc_auc_score, accuracy_score).
        n_bootstrap (int): Number of bootstrap samples.
        confidence_level (float): The confidence level for the interval.
        **metric_kwargs: Additional keyword arguments for the metric function.

    Returns:
        tuple: (point_estimate, (lower_ci, upper_ci))
               Returns (np.nan, (np.nan, np.nan)) if calculation fails.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bootstrap_scores = []
    
    # Calculate the point estimate on the original data
    try:
        point_estimate = metric_func(y_true, y_pred, **metric_kwargs)
    except ValueError:
         # Handle cases where metric cannot be calculated on original data (e.g., one class)
         point_estimate = np.nan 

    rng = np.random.default_rng() # Modern way to generate random numbers
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(indices) == 0: # Skip if indices array is empty (shouldn't happen with replace=True)
            continue 
            
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        
        # Check if bootstrap sample is valid for the metric
        if len(np.unique(y_true_bootstrap)) < 2 and metric_func == roc_auc_score:
            bootstrap_scores.append(np.nan) # AUC undefined for single class
            continue
        # Add other checks if needed for different metrics

        try:
            score = metric_func(y_true_bootstrap, y_pred_bootstrap, **metric_kwargs)
            bootstrap_scores.append(score)
        except ValueError:
            bootstrap_scores.append(np.nan) # Append NaN if metric calculation fails on bootstrap sample

    # Calculate confidence interval
    bootstrap_scores = np.array(bootstrap_scores)
    bootstrap_scores = bootstrap_scores[~np.isnan(bootstrap_scores)] # Remove NaNs

    if len(bootstrap_scores) < 2: # Need at least 2 valid scores for percentile calculation
        warnings.warn(f"Could not calculate CIs for {metric_func.__name__}: Insufficient valid bootstrap scores ({len(bootstrap_scores)}).")
        return point_estimate, (np.nan, np.nan)

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * alpha / 2.0
    upper_percentile = 100 * (1.0 - alpha / 2.0)
    
    lower_ci, upper_ci = np.percentile(bootstrap_scores, [lower_percentile, upper_percentile])

    return point_estimate, (lower_ci, upper_ci)


def bootstrap_confusion_metrics(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculates Specificity, Sensitivity, NPV, PPV and their CIs using bootstrapping.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bootstrap_specificities = []
    bootstrap_sensitivities = []
    bootstrap_npvs = []
    bootstrap_ppvs = []

    rng = np.random.default_rng()
    
    # Calculate point estimates on original data
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec_point = tn / (tn + fp) if (tn + fp) != 0 else np.nan
        sens_point = tp / (tp + fn) if (tp + fn) != 0 else np.nan
        npv_point = tn / (tn + fn) if (tn + fn) != 0 else np.nan
        ppv_point = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    except ValueError: # Handle cases where confusion matrix cannot be calculated
        spec_point, sens_point, npv_point, ppv_point = np.nan, np.nan, np.nan, np.nan

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(indices) == 0:
             continue
             
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]

        try:
            # Use labels=[0, 1] to ensure 2x2 matrix
            tn, fp, fn, tp = confusion_matrix(y_true_bootstrap, y_pred_bootstrap, labels=[0, 1]).ravel()

            specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) != 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) != 0 else np.nan

            bootstrap_specificities.append(specificity)
            bootstrap_sensitivities.append(sensitivity)
            bootstrap_npvs.append(npv)
            bootstrap_ppvs.append(ppv)
        except ValueError:
             # Append NaNs if confusion matrix failed for the bootstrap sample
             bootstrap_specificities.append(np.nan)
             bootstrap_sensitivities.append(np.nan)
             bootstrap_npvs.append(np.nan)
             bootstrap_ppvs.append(np.nan)


    def calculate_ci_from_list(metric_list, point_estimate):
        metric_list = np.array(metric_list)
        metric_list = metric_list[~np.isnan(metric_list)] # Remove NaNs
        if len(metric_list) < 2:
            return point_estimate, (np.nan, np.nan) # Return original point estimate if CI fails
        
        alpha = 1.0 - confidence_level
        lower_percentile = 100 * alpha / 2.0
        upper_percentile = 100 * (1.0 - alpha / 2.0)
        lower_ci, upper_ci = np.percentile(metric_list, [lower_percentile, upper_percentile])
        return point_estimate, (lower_ci, upper_ci)

    spec_mean_ci = calculate_ci_from_list(bootstrap_specificities, spec_point)
    sens_mean_ci = calculate_ci_from_list(bootstrap_sensitivities, sens_point)
    npv_mean_ci = calculate_ci_from_list(bootstrap_npvs, npv_point)
    ppv_mean_ci = calculate_ci_from_list(bootstrap_ppvs, ppv_point)

    return spec_mean_ci, sens_mean_ci, npv_mean_ci, ppv_mean_ci

# --- Main Function ---

def main():
    """Main function to process folds and generate summaries with CIs."""

    # Define output path
    output_summary_path = os.path.join(INPUT_FOLD_DIR, OUTPUT_SUMMARY_FILENAME)

    # Ensure output directory exists
    try:
        abs_output_dir = os.path.dirname(os.path.abspath(output_summary_path))
        os.makedirs(abs_output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {abs_output_dir}")
    except OSError as e:
        print(f"Error creating output directory {os.path.dirname(output_summary_path)}: {e}")
        return

    # Initialize dictionary to store results (easier to manage columns)
    results_data = {
        'fold': [],
        'patient_auc': [], 'patient_auc_lower_ci': [], 'patient_auc_upper_ci': [],
        'patient_accuracy': [], 'patient_accuracy_lower_ci': [], 'patient_accuracy_upper_ci': [],
        'patient_specificity': [], 'patient_specificity_lower_ci': [], 'patient_specificity_upper_ci': [],
        'patient_sensitivity': [], 'patient_sensitivity_lower_ci': [], 'patient_sensitivity_upper_ci': [],
        'patient_npv': [], 'patient_npv_lower_ci': [], 'patient_npv_upper_ci': [],
        'patient_ppv': [], 'patient_ppv_lower_ci': [], 'patient_ppv_upper_ci': [],
    }

    print(f"--- Starting Patient-Level Evaluation Metric Calculation (with {N_BOOTSTRAP} bootstraps) ---")

    # Loop through fold files
    for fold_num in range(NUM_FOLDS):
        input_csv_path = os.path.join(INPUT_FOLD_DIR, f'fold_{fold_num}.csv')
        print(f"\n--- Processing Fold {fold_num} ({input_csv_path}) ---")

        # --- 1. Load data ---
        try:
            df_fold = pd.read_csv(input_csv_path)
        except FileNotFoundError:
            print(f"Error: File not found: {input_csv_path}. Skipping fold {fold_num}.")
            continue
        except pd.errors.EmptyDataError:
             print(f"Warning: File {input_csv_path} is empty. Skipping fold {fold_num}.")
             continue
        except Exception as e:
            print(f"Error loading file {input_csv_path}: {e}. Skipping fold {fold_num}.")
            continue

        # Check for necessary columns
        required_cols = ['slide_id', 'Y', 'Y_hat', 'p_1']
        missing_cols = [col for col in required_cols if col not in df_fold.columns]
        if missing_cols:
            print(f"Error: Missing required columns in {input_csv_path}: {missing_cols}. Skipping.")
            continue
        if df_fold.empty:
             print(f"Warning: DataFrame loaded from {input_csv_path} is empty. Skipping.")
             continue

        # --- 2. Extract Patient ID ---
        try:
            df_fold['patient_id'] = df_fold['slide_id'].astype(str).str.split('_').str[0].str.split('-').str[0]
        except Exception as e:
             print(f"Error extracting patient_id from slide_id: {e}. Skipping fold {fold_num}.")
             continue

        # --- 3. Aggregate to Patient Level ---
        try:
            patient_results = df_fold.groupby('patient_id').agg(
                Y_true=('Y', 'max'),         # Use 'max' for aggregation
                Y_pred_binary=('Y_hat', 'max'),
                Y_pred_prob=('p_1', 'mean') # Use 'mean' for probability score
            ).reset_index()
        except Exception as e:
             print(f"Error during patient aggregation: {e}. Skipping fold {fold_num}.")
             continue

        if patient_results.empty:
             print("Warning: No patient results after aggregation. Skipping fold {fold_num}.")
             continue

        y_true_patient = patient_results['Y_true'].astype(int)
        y_pred_binary_patient = patient_results['Y_pred_binary'].astype(int)
        y_pred_prob_patient = patient_results['Y_pred_prob'].astype(float)

        # Basic sanity check for binary labels
        if not (y_true_patient.isin([0, 1]).all() and y_pred_binary_patient.isin([0, 1]).all()):
             print(f"Warning: Non-binary values found in patient-level 'Y_true' or 'Y_pred_binary' for fold {fold_num}. Check aggregation.")

        results_data['fold'].append(fold_num)

        # --- 4. Calculate Metrics with CIs ---
        print(f"Calculating metrics with CIs (n_bootstrap={N_BOOTSTRAP})...")
        
        # AUC
        print("  Calculating AUC CI...")
        auc, auc_ci = bootstrap_metric(y_true_patient, y_pred_prob_patient, roc_auc_score, N_BOOTSTRAP, CONFIDENCE_INTERVAL)
        results_data['patient_auc'].append(auc)
        results_data['patient_auc_lower_ci'].append(auc_ci[0])
        results_data['patient_auc_upper_ci'].append(auc_ci[1])

        # Accuracy
        print("  Calculating Accuracy CI...")
        acc, acc_ci = bootstrap_metric(y_true_patient, y_pred_binary_patient, accuracy_score, N_BOOTSTRAP, CONFIDENCE_INTERVAL)
        results_data['patient_accuracy'].append(acc)
        results_data['patient_accuracy_lower_ci'].append(acc_ci[0])
        results_data['patient_accuracy_upper_ci'].append(acc_ci[1])

        # Confusion Matrix Metrics
        print("  Calculating Confusion Matrix Metrics CIs...")
        spec_ci, sens_ci, npv_ci, ppv_ci = bootstrap_confusion_metrics(y_true_patient, y_pred_binary_patient, N_BOOTSTRAP, CONFIDENCE_INTERVAL)

        results_data['patient_specificity'].append(spec_ci[0])
        results_data['patient_specificity_lower_ci'].append(spec_ci[1][0])
        results_data['patient_specificity_upper_ci'].append(spec_ci[1][1])

        results_data['patient_sensitivity'].append(sens_ci[0])
        results_data['patient_sensitivity_lower_ci'].append(sens_ci[1][0])
        results_data['patient_sensitivity_upper_ci'].append(sens_ci[1][1])

        results_data['patient_npv'].append(npv_ci[0])
        results_data['patient_npv_lower_ci'].append(npv_ci[1][0])
        results_data['patient_npv_upper_ci'].append(npv_ci[1][1])

        results_data['patient_ppv'].append(ppv_ci[0])
        results_data['patient_ppv_lower_ci'].append(ppv_ci[1][0])
        results_data['patient_ppv_upper_ci'].append(ppv_ci[1][1])

        # --- Print fold results ---
        ci_label = f"{int(CONFIDENCE_INTERVAL*100)}%CI"
        print(f"Fold {fold_num} Results:")
        print(f"  AUC:       {auc:.4f} ({ci_label}: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
        print(f"  Accuracy:  {acc:.4f} ({ci_label}: {acc_ci[0]:.4f}-{acc_ci[1]:.4f})")
        print(f"  Specificty:{spec_ci[0]:.4f} ({ci_label}: {spec_ci[1][0]:.4f}-{spec_ci[1][1]:.4f})")
        print(f"  Sensitivity:{sens_ci[0]:.4f} ({ci_label}: {sens_ci[1][0]:.4f}-{sens_ci[1][1]:.4f})")
        print(f"  NPV:       {npv_ci[0]:.4f} ({ci_label}: {npv_ci[1][0]:.4f}-{npv_ci[1][1]:.4f})")
        print(f"  PPV:       {ppv_ci[0]:.4f} ({ci_label}: {ppv_ci[1][0]:.4f}-{ppv_ci[1][1]:.4f})")
        print("-" * 40)


    # --- Create and Save Summary DataFrame ---
    if not results_data['fold']:
         print("\nNo folds were processed successfully. Exiting without creating summary file.")
         return

    print("\n--- Generating Summary ---")
    summary_df = pd.DataFrame(results_data)
    
    # Define precise column order
    summary_cols = [
        'fold',
        'patient_auc', 'patient_auc_lower_ci', 'patient_auc_upper_ci',
        'patient_accuracy', 'patient_accuracy_lower_ci', 'patient_accuracy_upper_ci',
        'patient_specificity', 'patient_specificity_lower_ci', 'patient_specificity_upper_ci',
        'patient_sensitivity', 'patient_sensitivity_lower_ci', 'patient_sensitivity_upper_ci',
        'patient_npv', 'patient_npv_lower_ci', 'patient_npv_upper_ci',
        'patient_ppv', 'patient_ppv_lower_ci', 'patient_ppv_upper_ci',
    ]
    summary_df = summary_df[summary_cols] # Reorder columns


    # Display summary
    print("Summary across folds:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(summary_df.round(4))

    # Save Summary DataFrame to CSV
    try:
        # Use a specific float format for saving
        summary_df.to_csv(output_summary_path, index=False, float_format='%.5f') 
        print(f"\nPatient-level summary with {int(CONFIDENCE_INTERVAL*100)}% CIs saved to: {output_summary_path}")
    except Exception as e:
        print(f"\nError saving summary file to {output_summary_path}: {e}")

    print("\nProcessing complete.")

# --- Execution Guard ---
if __name__ == "__main__":
    # Optional: Suppress warnings if they become excessive and are understood
    # warnings.simplefilter(action='ignore', category=RuntimeWarning) # e.g., for division by zero
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
