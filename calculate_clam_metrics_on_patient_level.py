#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates and summarizes patient-level evaluation metrics (AUC, Accuracy, 
Specificity, Sensitivity, NPV, PPV) from CLAM evaluation result CSV files 
(fold_*.csv).

Reads CSV files for each fold, performs patient-level aggregation based on 
slide IDs (using max for labels/predictions and mean for probabilities), 
calculates the metrics, and saves a summary CSV file containing 
the patient-level results across all folds.
"""

import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import warnings

# --- Configuration ---
INPUT_FOLD_DIR = ''  # <--- **REPLACE WITH YOUR INPUT DIRECTORY**
OUTPUT_SUMMARY_FILENAME = '' # Filename for the summary CSV
NUM_FOLDS = 5  # Number of folds (e.g., fold_0.csv to fold_4.csv)
# --- End Configuration ---


def calculate_patient_metrics(y_true, y_pred_prob, y_pred_binary):
    """
    Calculates AUC, Accuracy, Specificity, Sensitivity, NPV, PPV for patient level.

    Args:
        y_true (array-like): Aggregated true labels for patients.
        y_pred_prob (array-like): Aggregated predicted probabilities for patients.
        y_pred_binary (array-like): Aggregated binary predictions for patients.

    Returns:
        dict: A dictionary containing the calculated metrics. Returns NaN for
              metrics that could not be calculated.
    """
    metrics = {}
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    y_pred_binary = np.asarray(y_pred_binary)

    # AUC
    try:
        # Check if only one class is present in true labels
        if len(np.unique(y_true)) < 2:
             print(f"  Warning: Only one class present in true labels ({np.unique(y_true)}). AUC is undefined.")
             metrics['auc'] = float('nan')
        else:
             metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e: # Catch other potential value errors
        print(f"  Warning: Could not calculate AUC. Error: {e}. Setting AUC to NaN.")
        metrics['auc'] = float('nan')

    # Accuracy
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    except ValueError as e:
        print(f"  Warning: Could not calculate Accuracy. Error: {e}. Setting Accuracy to NaN.")
        metrics['accuracy'] = float('nan')


    # Confusion Matrix based metrics
    try:
        # Use labels=[0, 1] to ensure matrix is always 2x2 if possible, handles cases where only one class is predicted/present
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics, checking for division by zero
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) != 0 else np.nan
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
        # Correct definition for NPV: TN / (TN + FN)
        metrics['npv'] = tn / (tn + fn) if (tn + fn) != 0 else np.nan
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) != 0 else np.nan

    except ValueError as e: # Catch errors during confusion_matrix calculation itself
        print(f"  Warning: Could not calculate confusion matrix or derived metrics. Error: {e}. Setting Spec/Sens/NPV/PPV to NaN.")
        # Ensure keys exist even if calculation fails
        metrics['specificity'] = np.nan
        metrics['sensitivity'] = np.nan
        metrics['npv'] = np.nan
        metrics['ppv'] = np.nan

    return metrics


def main():
    """Main function to process folds and generate the patient-level summary."""

    # Define output path
    output_summary_path = os.path.join(INPUT_FOLD_DIR, OUTPUT_SUMMARY_FILENAME)

    # Ensure output directory exists
    try:
        # Use os.path.abspath to handle potential relative paths in INPUT_FOLD_DIR
        abs_output_dir = os.path.dirname(os.path.abspath(output_summary_path))
        os.makedirs(abs_output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {abs_output_dir}")
    except OSError as e:
        print(f"Error creating output directory {os.path.dirname(output_summary_path)}: {e}")
        return # Exit if directory cannot be created

    # Initialize dictionary to store results
    results_data = {
        'fold': [],
        'patient_auc': [],
        'patient_accuracy': [],
        'patient_specificity': [],
        'patient_sensitivity': [],
        'patient_npv': [],
        'patient_ppv': [],
    }

    print("--- Starting Patient-Level Evaluation Metric Calculation ---")

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

        # Check for necessary columns for aggregation
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
            # Attempt to extract patient ID based on common patterns (e.g., TCGA-XX-YYYY...)
            # Split by '_' first, take the first part, then split by '-' and take the first part.
            # Adjust this logic if your patient ID format is different.
            df_fold['patient_id'] = df_fold['slide_id'].astype(str).str.split('_').str[0].str.split('-').str[0]
            print(f"  Extracted {df_fold['patient_id'].nunique()} unique patient IDs.")
            if df_fold['patient_id'].isnull().any():
                 print("  Warning: Some patient IDs could not be extracted (resulting in NaN).")
                 # Option: df_fold = df_fold.dropna(subset=['patient_id']) # Drop rows with missing patient IDs

        except Exception as e:
             print(f"Error extracting patient_id from slide_id: {e}. Skipping fold {fold_num}.")
             continue

        # --- 3. Aggregate to Patient Level ---
        try:
            patient_results = df_fold.groupby('patient_id').agg(
                # Patient is positive (Y=1) if ANY slide is positive
                Y_true=('Y', 'max'),
                # Patient is predicted positive (Y_hat=1) if ANY slide is predicted positive
                Y_pred_binary=('Y_hat', 'max'),
                # Use the MEAN probability of the positive class across slides for patient-level AUC
                Y_pred_prob=('p_1', 'mean')
            ).reset_index()
            print(f"  Aggregated results for {len(patient_results)} patients.")
        except Exception as e:
             print(f"Error during patient aggregation: {e}. Skipping fold {fold_num}.")
             continue

        if patient_results.empty:
             print(f"Warning: No patient results after aggregation for fold {fold_num}. Skipping.")
             continue

        # --- 4. Extract aggregated data ---
        try:
            y_true_patient = patient_results['Y_true'].astype(int)
            y_pred_prob_patient = patient_results['Y_pred_prob'].astype(float)
            y_pred_binary_patient = patient_results['Y_pred_binary'].astype(int)
        except KeyError as e:
            print(f"Error: Missing column {e} after aggregation. Skipping fold {fold_num}.")
            continue
        except Exception as e:
            print(f"Error processing aggregated data: {e}. Skipping fold {fold_num}.")
            continue

        # Basic sanity check for binary labels after aggregation
        if not (y_true_patient.isin([0, 1]).all()):
             print(f"  Warning: Non-binary values found in aggregated 'Y_true' for fold {fold_num}. Check 'max' aggregation logic if source 'Y' is not binary.")
        if not (y_pred_binary_patient.isin([0, 1]).all()):
             print(f"  Warning: Non-binary values found in aggregated 'Y_pred_binary' for fold {fold_num}. Check 'max' aggregation logic if source 'Y_hat' is not binary.")

        # --- 5. Calculate Patient-Level Metrics ---
        print("Calculating Patient-Level Metrics...")
        patient_metrics = calculate_patient_metrics(
            y_true_patient,
            y_pred_prob_patient,
            y_pred_binary_patient
        )

        # --- 6. Store metrics for summary ---
        results_data['fold'].append(fold_num)
        results_data['patient_auc'].append(patient_metrics.get('auc')) # Use .get() to handle potential missing keys if calculation failed
        results_data['patient_accuracy'].append(patient_metrics.get('accuracy'))
        results_data['patient_specificity'].append(patient_metrics.get('specificity'))
        results_data['patient_sensitivity'].append(patient_metrics.get('sensitivity'))
        results_data['patient_npv'].append(patient_metrics.get('npv'))
        results_data['patient_ppv'].append(patient_metrics.get('ppv'))

        # --- 7. Print metrics for the current fold ---
        print(f"Fold {fold_num} Results (Patient-Level):")
        # Use f-string formatting with :.4f for floats, handle None/NaN gracefully
        print(f"  AUC:       {patient_metrics.get('auc', 'N/A'):.4f}" if pd.notna(patient_metrics.get('auc')) else "  AUC:       N/A")
        print(f"  Accuracy:  {patient_metrics.get('accuracy', 'N/A'):.4f}" if pd.notna(patient_metrics.get('accuracy')) else "  Accuracy:  N/A")
        print(f"  Specificty:{patient_metrics.get('specificity', 'N/A'):.4f}" if pd.notna(patient_metrics.get('specificity')) else "  Specificty:N/A")
        print(f"  Sensitivity:{patient_metrics.get('sensitivity', 'N/A'):.4f}" if pd.notna(patient_metrics.get('sensitivity')) else "  Sensitivity:N/A")
        print(f"  NPV:       {patient_metrics.get('npv', 'N/A'):.4f}" if pd.notna(patient_metrics.get('npv')) else "  NPV:       N/A")
        print(f"  PPV:       {patient_metrics.get('ppv', 'N/A'):.4f}" if pd.notna(patient_metrics.get('ppv')) else "  PPV:       N/A")
        print("-" * 40)


    # --- Create and Save Summary DataFrame ---
    if not results_data['fold']:
         print("\nNo folds were processed successfully. Exiting without creating summary file.")
         return

    print("\n--- Generating Summary ---")
    summary_df = pd.DataFrame(results_data)

    # Define expected column order
    summary_cols = [
        'fold', 'patient_auc', 'patient_accuracy', 'patient_specificity',
        'patient_sensitivity', 'patient_npv', 'patient_ppv'
    ]
    # Ensure all expected columns exist before reordering
    summary_df = summary_df.reindex(columns=summary_cols)

    # Display summary
    print("Summary across folds (Patient-Level):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 100):
        print(summary_df.round(4))

    # Save Summary DataFrame to CSV
    try:
        # Use specific float format for saving
        summary_df.to_csv(output_summary_path, index=False, float_format='%.5f')
        print(f"\nPatient-level summary saved to: {output_summary_path}")
    except Exception as e:
        print(f"\nError saving summary file to {output_summary_path}: {e}")

    print("\nProcessing complete.")

# --- Execution Guard ---
if __name__ == "__main__":
    # Optional: Suppress specific warnings if they are known and acceptable
    # warnings.simplefilter(action='ignore', category=UserWarning)
    # warnings.simplefilter(action='ignore', category=RuntimeWarning) # e.g., for division by zero if NaNs expected
    main()
