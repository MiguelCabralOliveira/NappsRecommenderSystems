# preprocessing/preprocess_person_data.py
import argparse
import os
import pandas as pd
import traceback

# Use relative imports for the processing functions
from .process_direct_features import process_direct_person_columns
from .process_properties_features import process_person_properties

def get_final_person_columns():
    """Returns the list of expected final columns for person data."""
    return [
        'person_id_hex', 'is_identified', 'days_since_creation',
        'days_since_last_update', 'shopify_state', 'shopify_currency',
        'shopify_verified_email', 'shopify_account_age_days', 'city',
        'country_iso', 'in_eu', 'timezone', 'days_since_last_session',
    ]

def run_person_preprocessing(input_path: str, output_path: str):
    """
    Orchestrates the preprocessing of person data by calling processors and merging.
    """
    print(f"\n--- Starting Person Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # --- Load Raw Data ---
    try:
        person_df_raw = pd.read_csv(input_path)
        print(f"Loaded {len(person_df_raw)} raw person rows.")
        if person_df_raw.empty:
            print("Input DataFrame is empty. Creating empty output file.")
            pd.DataFrame(columns=get_final_person_columns()).to_csv(output_path, index=False)
            return # Successfully handled empty input
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error loading input CSV: {e}")
        traceback.print_exc()
        raise

    # --- Process Data ---
    try:
        # Process direct columns
        df_direct = process_direct_person_columns(person_df_raw)

        # Process properties columns
        df_props = process_person_properties(person_df_raw)

    except Exception as e:
        print(f"Error during feature processing: {e}")
        traceback.print_exc()
        raise

    # --- Merge Results ---
    print("\n--- Merging Processed Features ---")
    try:
        # Ensure both dataframes have the ID column
        if 'person_id_hex' not in df_direct.columns:
             raise ValueError("Direct features DataFrame missing 'person_id_hex'")
        if 'person_id_hex' not in df_props.columns:
             raise ValueError("Properties features DataFrame missing 'person_id_hex'")

        final_df = pd.merge(df_direct, df_props, on='person_id_hex', how='left')
        print(f"Merged DataFrame shape: {final_df.shape}")

    except Exception as e:
        print(f"Error during merging: {e}")
        traceback.print_exc()
        # Debugging merge issues:
        # print("Direct DF Info:", df_direct.info())
        # print("Props DF Info:", df_props.info())
        raise

    # --- Finalize Columns and Save ---
    FINAL_COLUMNS = get_final_person_columns()

    # Ensure all desired columns exist, add missing ones with default values
    for col in FINAL_COLUMNS:
        if col not in final_df.columns:
            print(f"Warning: Expected final column '{col}' not found after merge. Adding default.")
            # Add appropriate default based on expected type
            if 'days' in col or 'age' in col or 'identified' in col or 'in_eu' in col or 'verified' in col:
                 final_df[col] = 0 # Default numerical/binary
            else:
                 final_df[col] = 'Unknown' # Default categorical

    # Select and reorder columns
    try:
        final_df = final_df[FINAL_COLUMNS]
    except KeyError as e:
        print(f"Error selecting final columns: {e}. Available columns: {final_df.columns.tolist()}")
        raise

    # Optional: Final NaN handling (should be minimal now)
    num_cols = ['days_since_creation', 'days_since_last_update', 'shopify_account_age_days', 'days_since_last_session']
    cat_cols = ['shopify_state', 'shopify_currency', 'city', 'country_iso', 'timezone']
    for col in num_cols:
        if col in final_df.columns: final_df[col] = final_df[col].fillna(0).astype(int)
    for col in cat_cols:
        if col in final_df.columns: final_df[col] = final_df[col].fillna('Unknown').astype(str)

    print(f"\nFinal Processed DataFrame shape: {final_df.shape}")
    print(f"Final columns: {final_df.columns.tolist()}")

    # Save Output
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Successfully saved processed person data to: {output_path}")
        print(f"--- Finished Person Data Preprocessing Orchestration ---")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        traceback.print_exc()
        raise

# --- Command Line Execution Logic (Optional) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess person data (Orchestrator).')
    parser.add_argument('--input', required=True, help='Path to the input person_data_loaded.csv file.')
    parser.add_argument('--output', required=True, help='Path to save the final processed person_features.csv file.')
    args = parser.parse_args()

    try:
        run_person_preprocessing(args.input, args.output)
        print("\nPreprocessing finished successfully via command line.")
    except Exception as e:
        print(f"\n--- Preprocessing pipeline failed: {e} ---")
        exit(1) # Exit with error code

    # Example Usage:
    # python -m preprocessing.preprocess_person_data --input results/sjstarcorp/person_data_loaded.csv --output results/sjstarcorp/person_features_processed.csv