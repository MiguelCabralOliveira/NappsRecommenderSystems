# preprocessing/preprocess_person_data.py
import argparse
import os
import pandas as pd
import traceback

# Use relative imports for the processing functions within the person_processor sub-package
try:
    from .person_processor.process_direct_features import process_direct_person_columns
    from .person_processor.process_properties_features import process_person_properties
except ImportError:
    # Fallback for potential direct execution testing (less recommended)
    print("Warning: Could not use relative imports. Attempting direct import (may fail depending on execution context).")
    try:
        from person_processor.process_direct_features import process_direct_person_columns
        from person_processor.process_properties_features import process_person_properties
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import person_processor modules.")
         print("Ensure you are running from the project root 'NappsRecommenderSystems/' using")
         print("`python -m preprocessing.run_preprocessing ...` or `python -m preprocessing.preprocess_person_data ...`")
         raise # Re-raise the error to stop execution


def get_final_person_columns():
    """Returns the list of expected final columns for person data."""
    # Define the final column order and selection
    # REMOVED: 'shopify_state', 'shopify_verified_email', 'city', 'days_since_last_session'
    return [
        'person_id_hex',
        # Direct Features
        'is_identified',
        'days_since_creation',
        'days_since_last_update',
        # Properties Features (Identified)
        # 'shopify_state', # <- REMOVED
        'shopify_currency',
        # 'shopify_verified_email', # <- REMOVED
        'shopify_account_age_days',
         # Properties Features (Anonymous / General)
        # 'city', # <- REMOVED
        'country_iso',
        'in_eu',
        'timezone',
        # 'days_since_last_session', # <- REMOVED
        # Add any other derived features here if created during merge/finalization
    ]

def run_person_preprocessing(input_path: str, output_path: str):
    """
    Orchestrates the preprocessing of person data by calling processors and merging.

    Args:
        input_path: Path to the raw person data CSV (e.g., person_data_loaded.csv).
        output_path: Path to save the processed person features CSV.
    """
    print(f"\n--- Starting Person Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # --- Load Raw Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        person_df_raw = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {len(person_df_raw)} raw person rows from {input_path}.")

        if person_df_raw.empty:
            print("Input DataFrame is empty. Creating empty output file.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(columns=get_final_person_columns()).to_csv(output_path, index=False) # Use updated columns
            print(f"Empty output file created at: {output_path}")
            return
        if 'person_id_hex' not in person_df_raw.columns:
            raise ValueError("Input person data must contain 'person_id_hex' column.")
        required_processing_cols = ['is_identified', 'created_at', 'updated_at', 'properties']
        missing_essential = [col for col in required_processing_cols if col not in person_df_raw.columns]
        if missing_essential:
             print(f"Warning: Input data missing expected columns for processing: {missing_essential}. Proceeding, but some features might be missing/defaulted.")
             for col in missing_essential:
                 person_df_raw[col] = None

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error loading or validating input CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Process Data ---
    try:
        # Define necessary columns for processors (even if output is dropped later)
        direct_cols = ['person_id_hex', 'is_identified', 'created_at', 'updated_at']
        props_cols = ['person_id_hex', 'properties', 'is_identified'] # process_properties still needs these

        direct_cols_present = [col for col in direct_cols if col in person_df_raw.columns]
        props_cols_present = [col for col in props_cols if col in person_df_raw.columns]

        print("\nProcessing direct columns...")
        if all(col in direct_cols_present for col in ['person_id_hex', 'is_identified', 'created_at', 'updated_at']):
             df_direct = process_direct_person_columns(person_df_raw[direct_cols_present].copy())
             print(f"Direct features processed, shape: {df_direct.shape}")
        else:
             print("Skipping direct columns processing due to missing required input columns.")
             df_direct = person_df_raw[['person_id_hex']].copy()
             df_direct['is_identified'] = 0
             df_direct['days_since_creation'] = 0
             df_direct['days_since_last_update'] = 0


        print("\nProcessing properties columns...")
        # We still run this processor, it might create columns needed by other processors or intermediate steps
        # We just won't select the unwanted ones at the end.
        if all(col in props_cols_present for col in ['person_id_hex', 'properties', 'is_identified']):
             df_props = process_person_properties(person_df_raw[props_cols_present].copy())
             print(f"Properties features processed, shape: {df_props.shape}")
        else:
             print("Skipping properties columns processing due to missing required input columns.")
             df_props = person_df_raw[['person_id_hex']].copy()
             # Add default values for columns that might be expected (based on the *original* get_final_person_columns)
             # It's less critical now, as we select only desired columns later
             original_prop_cols = ['shopify_state', 'shopify_currency', 'shopify_verified_email',
                                   'shopify_account_age_days', 'city', 'country_iso',
                                   'in_eu', 'timezone', 'days_since_last_session']
             for col in original_prop_cols:
                  if 'days' in col or 'age' in col or 'verified' in col or 'in_eu' in col: df_props[col] = 0
                  else: df_props[col] = 'Unknown'


    except Exception as e:
        print(f"Error during feature processing step: {e}")
        traceback.print_exc()
        raise

    # --- Merge Results ---
    print("\n--- Merging Processed Features ---")
    try:
        if 'person_id_hex' not in df_direct.columns or 'person_id_hex' not in df_props.columns:
             raise ValueError("Critical error: 'person_id_hex' missing from one of the processed dataframes before merge.")

        final_df = pd.merge(df_direct, df_props, on='person_id_hex', how='left')
        print(f"Merged DataFrame shape: {final_df.shape}")

        if final_df.empty and not df_direct.empty:
            print("Warning: Merged DataFrame is empty, check merge keys and logic.")
        if final_df.columns.duplicated().any():
            dups = final_df.columns[final_df.columns.duplicated()].tolist()
            print(f"Warning: Duplicate columns found after merge: {dups}. Keeping first instance.")
            final_df = final_df.loc[:,~final_df.columns.duplicated()]

    except Exception as e:
        print(f"Error during merging processed features: {e}")
        traceback.print_exc()
        raise

    # --- Finalize Columns and Save ---
    # Get the desired final columns from the updated function
    FINAL_COLUMNS = get_final_person_columns()

    print("\nFinalizing columns...")
    missing_final_cols = []
    available_cols_before_select = final_df.columns.tolist() # Columns available after merge

    # Check which of the *desired* final columns are actually available after processing/merge
    for col in FINAL_COLUMNS:
        if col not in available_cols_before_select:
            missing_final_cols.append(col)
            # Add appropriate default based on expected type convention
            if 'days' in col or 'age' in col: final_df[col] = 0
            elif 'identified' in col or 'in_eu' in col : final_df[col] = 0 # Removed verified_email
            elif 'shopify_currency' in col or 'country_iso' in col or 'timezone' in col: final_df[col] = 'Unknown' # Removed state, city
            else: final_df[col] = None
            print(f"Warning: Expected final column '{col}' was missing after merge. Added with default value.")

    if missing_final_cols:
        print(f"Columns added with defaults: {missing_final_cols}")

    # Select and reorder columns *only* to the desired FINAL_COLUMNS
    try:
        # This step effectively drops the unwanted columns ('city', 'days_since_last_session', etc.)
        # because they are not in the FINAL_COLUMNS list.
        final_df = final_df[FINAL_COLUMNS]
        print(f"Selected final columns defined by get_final_person_columns(). Shape: {final_df.shape}")
    except KeyError as e:
        available_cols = final_df.columns.tolist()
        required_but_missing = [c for c in FINAL_COLUMNS if c not in available_cols]
        print(f"Error selecting final columns: {e}. Required but missing from DataFrame: {required_but_missing}. Available columns: {available_cols}")
        raise

    # Final checks and type enforcement on the *selected* columns
    print("Performing final checks and type conversions...")
    # Update these lists to match the columns defined in get_final_person_columns()
    num_cols = ['days_since_creation', 'days_since_last_update', 'shopify_account_age_days'] # REMOVED 'days_since_last_session'
    cat_cols = ['shopify_currency', 'country_iso', 'timezone'] # REMOVED 'shopify_state', 'city'
    bool_cols = ['is_identified', 'in_eu']

    for col in num_cols:
        if col in final_df.columns: final_df[col] = pd.to_numeric(final_df[col].fillna(0), errors='coerce').fillna(0).astype(int)
    for col in cat_cols:
        if col in final_df.columns: final_df[col] = final_df[col].fillna('Unknown').astype(str)
    for col in bool_cols:
        if col in final_df.columns: final_df[col] = final_df[col].fillna(0).astype(int)

    if 'person_id_hex' in final_df.columns: final_df['person_id_hex'] = final_df['person_id_hex'].astype(str)

    print(f"\nFinal Processed DataFrame shape: {final_df.shape}")
    print(f"Final columns: {final_df.columns.tolist()}") # Should now match get_final_person_columns()

    # --- Save Output ---
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

        final_df.to_csv(output_path, index=False)
        print(f"Successfully saved processed person data to: {output_path}")
        print(f"--- Finished Person Data Preprocessing Orchestration ---")
    except Exception as e:
        print(f"Error saving output CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

# --- Command Line Execution Logic (for standalone testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess person data (Orchestrator). Requires person_data_loaded.csv as input.')
    parser.add_argument('--input', required=True, help='Path to the input person_data_loaded.csv file.')
    parser.add_argument('--output', required=True, help='Path to save the final person_features_processed.csv file.')
    args = parser.parse_args()

    try:
        run_person_preprocessing(args.input, args.output)
        print("\nPerson preprocessing finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error:
         print(f"\nError: {val_error}")
         exit(1)
    except Exception as e:
        print(f"\n--- Person preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)