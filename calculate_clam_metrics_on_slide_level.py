#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates and summarizes slide-level evaluation metrics (AUC, Accuracy, 
Specificity, Sensitivity, NPV, PPV) from CLAM evaluation result CSV files 
(fold_*.csv).

Reads CSV files for each fold, calculates metrics directly from the slide data,
and saves a summary CSV file containing the slide-level results across folds.
"""

import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import warnings

# --- Configuration ---
INPUT_FOLD_DIR = '/home/chan87/deeplearning/pythonProject/CLAM/CLAM/eval_results/EVAL_whole/'  # <--- **REPLACE WITH YOUR INPUT DIRECTORY**
OUTPUT_SUMMARY_FILENAME = 'slide_summary.csv' # Filename for the summary CSV
NUM_FOLDS = 5  # Number of folds (e.g., fold_0.csv to fold_4.csv)
# --- End Configuration ---

def calculate_slide_metrics(y_true, y_pred_prob, y_pred_binary):
    """Calculates AUC, Accuracy, Specificity, Sensitivity, NPV, PPV for slide level."""
    metrics = {}

    # AUC
    try:
        # Check if only one class is present in true labels
        if len(np.unique(y_true)) < 2:
             print(f"Warning: Only one class present in true labels ({np.unique(y_true)}). AUC is undefined.")
             metrics['auc'] = float('nan')
        else:
             metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e: # Catch other potential value errors
        print(f"Warning: Could not calculate AUC. Error: {e}. Setting AUC to NaN.")
        metrics['auc'] = float('nan')

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

    # Confusion Matrix based metrics
    try:
        # Ensure there are predictions for both classes or handle appropriately
        # Use labels=[0, 1] to ensure matrix is always 2x2 if possible
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics, checking for division by zero
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) != 0 else np.nan
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
        # Correct definition for NPV: TN / (TN + FN)
        metrics['npv'] = tn / (tn + fn) if (tn + fn) != 0 else np.nan
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) != 0 else np.nan

    except ValueError as e: # Catch errors during confusion_matrix calculation itself
        print(f"Warning: Could not calculate confusion matrix or derived metrics. Error: {e}. Setting Spec/Sens/NPV/PPV to NaN.")
        metrics['specificity'] = np.nan
        metrics['sensitivity'] = np.nan
        metrics['npv'] = np.nan
        metrics['ppv'] = np.nan

    return metrics

def main():
    """Main function to process folds and generate the slide-level summary."""

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

    # Initialize lists to store results for the summary DataFrame
    fold_numbers_list = []
    slide_metrics_list = [] # List to store dictionaries of slide metrics per fold

    print("--- Starting Slide-Level Evaluation Metric Calculation ---")

    # Loop through fold files
    for fold_num in range(NUM_FOLDS):
        input_csv_path = os.path.join(INPUT_FOLD_DIR, f'fold_{fold_num}.csv')
        print(f"\n--- Processing Fold {fold_num} ({input_csv_path}) ---")

        # --- 1. Load data from fold CSV file ---
        try:
            df_fold = pd.read_csv(input_csv_path)
        except FileNotFoundError:
            print(f"Error: File not found: {input_csv_path}. Skipping fold {fold_num}.")
            continue # Skip to the next fold
        except pd.errors.EmptyDataError:
             print(f"Warning: File {input_csv_path} is empty. Skipping fold {fold_num}.")
             continue
        except Exception as e:
            print(f"Error loading file {input_csv_path}: {e}. Skipping fold {fold_num}.")
            continue

        # Check for necessary columns
        required_cols = ['Y', 'Y_hat', 'p_1']
        missing_cols = [col for col in required_cols if col not in df_fold.columns]
        if missing_cols:
            print(f"Error: Missing required columns in {input_csv_path}. Required: {required_cols}. Found: {df_fold.columns.tolist()}. Skipping fold {fold_num}.")
            continue

        # --- 2. Extract data for slide-level metrics ---
        try:
            y_true_slide = df_fold['Y'].astype(int) # Ensure integer type for labels
            y_pred_slide_binary = df_fold['Y_hat'].astype(int) # Ensure integer type
            y_pred_slide_prob = df_fold['p_1'].astype(float) # Ensure float type

            # Basic sanity check
            if not (y_true_slide.isin([0, 1]).all() and y_pred_slide_binary.isin([0, 1]).all()):
                 print(f"Warning: Non-binary values found in 'Y' or 'Y_hat' columns in fold {fold_num}. Check data integrity.")
                 # Calculation will proceed, but warning is noted.

            # --- 3. Calculate Slide-Level Metrics ---
            print("Calculating Slide-Level Metrics...")
            slide_metrics = calculate_slide_metrics(
                y_true_slide,
                y_pred_slide_prob,
                y_pred_slide_binary
            )

            # --- 4. Store metrics for summary ---
            # Append metrics only if calculation was successful (or returned NaNs gracefully)
            fold_numbers_list.append(fold_num) # Add fold number only if processing reaches this point
            slide_metrics_list.append(slide_metrics)

            # --- 5. Print metrics for the current fold ---
            # Use .get() with default 'N/A' in case a metric failed to calculate
            print(f"Slide-Level Metrics -> AUC: {slide_metrics.get('auc', 'N/A'):.4f}, "
                  f"Acc: {slide_metrics.get('accuracy', 'N/A'):.4f}, "
                  f"Spec: {slide_metrics.get('specificity', 'N/A'):.4f}, "
                  f"Sens: {slide_metrics.get('sensitivity', 'N/A'):.4f}, "
                  f"NPV: {slide_metrics.get('npv', 'N/A'):.4f}, "
                  f"PPV: {slide_metrics.get('ppv', 'N/A'):.4f}")

        except KeyError as e:
            print(f"Error: Column {e} not found during data extraction for fold {fold_num}. Skipping metric calculation for this fold.")
            # Do not append fold_num or metrics list if essential data is missing
        except Exception as e:
            print(f"An unexpected error occurred during metric calculation for fold {fold_num}: {e}")
            # Do not append fold_num or metrics list if calculation fails unexpectedly

        print("-" * 40)

    # --- Create and Save Summary DataFrame ---
    if not fold_numbers_list:
         print("\nNo folds were processed successfully. Exiting without creating summary file.")
         return

    # Ensure lists have the same length before creating DataFrame (should be guaranteed by current logic)
    if len(fold_numbers_list) != len(slide_metrics_list):
        print("\nError: Mismatch between number of processed folds and collected metrics. Cannot create summary.")
        # Add details for debugging if this occurs
        print(f"Fold numbers collected: {len(fold_numbers_list)}")
        print(f"Metrics collected: {len(slide_metrics_list)}")
        return

    print("\n--- Generating Summary ---")

    summary_df = pd.DataFrame(slide_metrics_list)
    summary_df.insert(0, 'fold', fold_numbers_list) # Add fold numbers as the first column
    summary_df.rename(columns={
        'auc': 'slide_auc', 'accuracy': 'slide_accuracy',
        'specificity': 'slide_specificity', 'sensitivity': 'slide_sensitivity',
        'npv': 'slide_npv', 'ppv': 'slide_ppv'
    }, inplace=True)

    # Display summary
    print("Summary across folds:")
    # Use display options for better formatting in console
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(summary_df.round(4))


    # Save Summary DataFrame to CSV
    try:
        summary_df.to_csv(output_summary_path, index=False, float_format='%.5f')
        print(f"\nSlide-level summary saved to: {output_summary_path}")
    except Exception as e:
        print(f"\nError saving slide summary file to {output_summary_path}: {e}")

    print("\nProcessing complete.")

# --- Execution Guard ---
if __name__ == "__main__":
    # Optional: Suppress specific warnings if they are known and acceptable
    # warnings.simplefilter(action='ignore', category=RuntimeWarning) # e.g., for division by zero if NaNs are expected
    main()
