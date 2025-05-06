#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs univariate logistic regression analysis on clinical data.

Calculates Odds Ratios (OR), 95% Confidence Intervals (CI), and P-values
for specified predictors against a binary outcome ('label').
Handles 'Age' as a categorical variable (0, 1, 2, 3) with level 0 as reference.
Makes assumptions about reference/compared categories for other binary predictors.
Saves the results to a CSV file.
"""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import warnings

# --- Configuration ---
INPUT_CSV_PATH = '/home/chan87/deeplearning/pythonProject/colorectal_adenocarcinoma_node_metastases/crc_clinic_0.csv'
OUTPUT_CSV_PATH = '/home/chan87/deeplearning/pythonProject/colorectal_adenocarcinoma_node_metastases/logistic_regression_univariate_results.csv'
OUTCOME_VAR = 'label'
PREDICTORS = ['Sideness', 'Sex', 'CRC_location', 'Age', 'CEA', 'T_Stage']
# --- End Configuration ---

def run_univariate_logistic_regression(data, outcome, predictors):
    """
    Runs univariate logistic regression for each predictor against the outcome.

    Args:
        data (pd.DataFrame): DataFrame containing the outcome and predictor variables.
        outcome (str): Name of the outcome variable column.
        predictors (list): List of predictor variable column names.

    Returns:
        pd.DataFrame: DataFrame containing the regression results (OR, CI, P-value).
    """
    results_list = []
    df = data.copy() # Work on a copy to avoid modifying the original DataFrame

    print("--- Starting Univariate Logistic Regression Analysis ---")

    for predictor in predictors:
        print(f"Processing predictor: {predictor}")

        # --- Special Handling for 'Age' ---
        if predictor == 'Age':
            # Ensure Age is treated as categorical by statsmodels
            # Creating a temporary column is one way, C() in formula is another.
            # We'll use a temporary column to match the original logic.
            temp_cat_col = f"{predictor}_Cat"
            try:
                df[temp_cat_col] = pd.Categorical(df[predictor])
                formula = f'{outcome} ~ {temp_cat_col}'
                model = smf.logit(formula=formula, data=df)
                result = model.fit(disp=0) # disp=0 suppresses optimization output

                # Determine reference level (lowest category by default)
                reference_level = df[temp_cat_col].cat.categories[0]
                reference_category_label = f'Level {reference_level}'
                print(f"  Reference category for {predictor}: {reference_category_label}")

                # Iterate through non-reference levels
                for level in df[temp_cat_col].cat.categories[1:]:
                    param_name = f"{temp_cat_col}[T.{level}]" # Name statsmodels uses
                    compared_category_label = f'Level {level}'
                    try:
                        log_odds = result.params[param_name]
                        p_value = result.pvalues[param_name]
                        conf_int_log_odds = result.conf_int().loc[param_name]

                        or_value = np.exp(log_odds)
                        lower_ci = np.exp(conf_int_log_odds[0])
                        upper_ci = np.exp(conf_int_log_odds[1])

                        results_list.append({
                            'Predictor': predictor,
                            'Category Comparison': f'{compared_category_label} vs {reference_category_label}',
                            'Reference Category': reference_category_label,
                            'Compared Category': compared_category_label,
                            'OR': or_value,
                            'Lower CI (95%)': lower_ci,
                            'Upper CI (95%)': upper_ci,
                            'P-value': p_value
                        })
                        print(f"    - {compared_category_label} vs {reference_category_label}: OR={or_value:.3f}, P={p_value:.3g}")

                    except KeyError:
                        print(f"    - Warning: Parameter '{param_name}' not found in results for {predictor}. Level might be missing or collinear.")
                        results_list.append({
                            'Predictor': predictor,
                            'Category Comparison': f'{compared_category_label} vs {reference_category_label}',
                            'Reference Category': reference_category_label,
                            'Compared Category': compared_category_label,
                            'OR': np.nan,
                            'Lower CI (95%)': np.nan,
                            'Upper CI (95%)': np.nan,
                            'P-value': np.nan
                        })

            except Exception as e:
                 print(f"  Error processing predictor '{predictor}': {e}")
                 # Append NaNs for all levels if the initial model fails
                 if temp_cat_col in df.columns:
                     all_levels = df[temp_cat_col].cat.categories
                     ref_level = all_levels[0]
                     for level in all_levels[1:]:
                         results_list.append({
                            'Predictor': predictor,
                            'Category Comparison': f'Level {level} vs Level {ref_level}',
                            'Reference Category': f'Level {ref_level}',
                            'Compared Category': f'Level {level}',
                            'OR': np.nan, 'Lower CI (95%)': np.nan, 'Upper CI (95%)': np.nan, 'P-value': np.nan
                         })
            finally:
                # Clean up temporary column
                if temp_cat_col in df.columns:
                    df.drop(columns=[temp_cat_col], inplace=True)

        # --- Handling for Other Predictors ---
        else:
            # Check if predictor is likely categorical (binary assumed here based on original code)
            is_categorical_like = df[predictor].nunique() <= 5 or df[predictor].dtype == 'object' # Heuristic

            try:
                # Use C() in formula to explicitly treat as categorical if needed
                # This lets statsmodels handle dummy coding and reference selection
                formula = f'{outcome} ~ C({predictor})' if is_categorical_like else f'{outcome} ~ {predictor}'
                model = smf.logit(formula=formula, data=df)
                result = model.fit(disp=0)

                # Identify the parameter name(s) associated with the predictor
                predictor_params = [p for p in result.params.index if p.startswith(f"C({predictor})[T.") or p == predictor]

                if not predictor_params:
                     # This might happen if the variable was dropped due to collinearity or zero variance
                     print(f"  Warning: No parameters found for predictor '{predictor}' in the model results. Skipping.")
                     reference_category_label = 'N/A (Not Found)'
                     compared_category_label = 'N/A (Not Found)'
                     or_value, lower_ci, upper_ci, p_value = np.nan, np.nan, np.nan, np.nan
                     category_comparison = 'N/A (Parameter Not Found)'
                elif len(predictor_params) == 1:
                    # This usually means it's treated as continuous or it's binary categorical
                    param_name = predictor_params[0]
                    log_odds = result.params[param_name]
                    p_value = result.pvalues[param_name]
                    conf_int_log_odds = result.conf_int().loc[param_name]

                    or_value = np.exp(log_odds)
                    lower_ci = np.exp(conf_int_log_odds[0])
                    upper_ci = np.exp(conf_int_log_odds[1])

                    # --- Infer Reference/Compared (Best Effort) ---
                    reference_category_label = 'N/A'
                    compared_category_label = 'N/A'
                    category_comparison = 'Overall' # Default for continuous or simple binary

                    if param_name.startswith(f"C({predictor})[T."):
                         # Infer from statsmodels parameter name C(Pred)[T.ComparedLevel]
                         compared_value = param_name.split('T.')[1].split(']')[0]
                         all_values = sorted(df[predictor].unique())
                         reference_value = all_values[0] # statsmodels default reference
                         reference_category_label = f'Level {reference_value}'
                         compared_category_label = f'Level {compared_value}'
                         category_comparison = f'{compared_category_label} vs {reference_category_label}'
                         print(f"  Reference: {reference_category_label}, Compared: {compared_category_label}")
                    elif df[predictor].dtype in [np.int64, np.float64] and not is_categorical_like:
                         category_comparison = 'Per Unit Increase'
                         print(f"  Interpreting as continuous: OR per unit increase")
                    # --- End Infer Reference/Compared ---

                    print(f"    - {category_comparison}: OR={or_value:.3f}, P={p_value:.3g}")

                else: # Multiple parameters means more than 2 categories (handled like 'Age' now)
                    print(f"  Predictor '{predictor}' has multiple categories. Handling similar to 'Age'.")
                    # Determine reference level (lowest category by default)
                    reference_level = sorted(df[predictor].unique())[0]
                    reference_category_label = f'Level {reference_level}'
                    print(f"  Reference category for {predictor}: {reference_category_label}")

                    for param_name in predictor_params: # Should be C(Pred)[T.Level1], C(Pred)[T.Level2] etc.
                        if param_name.startswith(f"C({predictor})[T."):
                            level = param_name.split('T.')[1].split(']')[0]
                            compared_category_label = f'Level {level}'
                            try:
                                log_odds = result.params[param_name]
                                p_value = result.pvalues[param_name]
                                conf_int_log_odds = result.conf_int().loc[param_name]

                                or_value = np.exp(log_odds)
                                lower_ci = np.exp(conf_int_log_odds[0])
                                upper_ci = np.exp(conf_int_log_odds[1])

                                results_list.append({
                                    'Predictor': predictor,
                                    'Category Comparison': f'{compared_category_label} vs {reference_category_label}',
                                    'Reference Category': reference_category_label,
                                    'Compared Category': compared_category_label,
                                    'OR': or_value,
                                    'Lower CI (95%)': lower_ci,
                                    'Upper CI (95%)': upper_ci,
                                    'P-value': p_value
                                })
                                print(f"    - {compared_category_label} vs {reference_category_label}: OR={or_value:.3f}, P={p_value:.3g}")
                            except KeyError:
                                print(f"    - Warning: Parameter '{param_name}' not found in results for {predictor}.")
                                # Add NaN row
                                results_list.append({
                                    'Predictor': predictor, 'Category Comparison': f'{compared_category_label} vs {reference_category_label}',
                                    'Reference Category': reference_category_label, 'Compared Category': compared_category_label,
                                    'OR': np.nan, 'Lower CI (95%)': np.nan, 'Upper CI (95%)': np.nan, 'P-value': np.nan
                                })
                    # Skip appending the single row below if handled here
                    continue # Go to next predictor

                # Append the result (only if it wasn't multi-category handled above)
                results_list.append({
                    'Predictor': predictor,
                    'Category Comparison': category_comparison,
                    'Reference Category': reference_category_label,
                    'Compared Category': compared_category_label,
                    'OR': or_value,
                    'Lower CI (95%)': lower_ci,
                    'Upper CI (95%)': upper_ci,
                    'P-value': p_value
                })

            except Exception as e:
                print(f"  Error processing predictor '{predictor}': {e}")
                # Append a row with NaNs if the model fails
                results_list.append({
                    'Predictor': predictor,
                    'Category Comparison': 'Error During Processing',
                    'Reference Category': 'N/A',
                    'Compared Category': 'N/A',
                    'OR': np.nan,
                    'Lower CI (95%)': np.nan,
                    'Upper CI (95%)': np.nan,
                    'P-value': np.nan
                })

    print("--- Analysis Complete ---")
    # Create DataFrame from the results list
    results_df = pd.DataFrame(results_list)
    # Reorder columns for clarity
    cols_order = ['Predictor', 'Category Comparison', 'Reference Category', 'Compared Category',
                  'OR', 'Lower CI (95%)', 'Upper CI (95%)', 'P-value']
    # Ensure all expected columns exist before reordering
    results_df = results_df.reindex(columns=cols_order)

    return results_df

def main():
    """Loads data, runs analysis, prints, and saves results."""
    # Ignore common statsmodels warnings during fitting
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning) # Can ignore specific statsmodels warnings too if needed

    # Load the data
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded data from: {INPUT_CSV_PATH}")
        # Basic check for required columns
        required_cols = [OUTCOME_VAR] + PREDICTORS
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in input CSV: {missing_cols}")
            return # Exit if columns are missing
        if OUTCOME_VAR not in df.columns:
             print(f"Error: Outcome variable '{OUTCOME_VAR}' not found in the CSV.")
             return
        if not PREDICTORS:
             print("Error: No predictors specified in the PREDICTORS list.")
             return

    except FileNotFoundError:
        print(f"Error: Input file not found at: {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Run the analysis
    results_df = run_univariate_logistic_regression(df, OUTCOME_VAR, PREDICTORS)

    # Format and display results
    pd.set_option('display.float_format', lambda x: f'{x:.3f}' if pd.notna(x) else 'NaN')
    pd.set_option('display.max_rows', None) # Show all rows
    pd.set_option('display.width', 1000) # Adjust display width
    print("\n--- Univariate Logistic Regression Results ---")
    print(results_df)

    # Save results to CSV
    try:
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.5f')
        print(f"\nResults successfully saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

# --- Execution Guard ---
if __name__ == "__main__":
    main()
