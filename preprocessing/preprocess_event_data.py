# preprocessing/preprocess_event_data.py
import argparse
import os
import pandas as pd
import traceback
import re
import sys

# --- Import the core processing function ---
try:
    # Relative import from the sub-package
    from .event_processor.process_event_interactions import extract_interactions_from_events
    # Import the final columns definition as well
    from .event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS
except ImportError:
    print("Warning: Could not use relative imports for event processor. Attempting direct import.")
    try:
        from event_processor.process_event_interactions import extract_interactions_from_events
        from event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import event_processor module.")
         print("Ensure structure preprocessing/event_processor/ exists with __init__.py.")
         sys.exit(1)

# --- Helper Function (Optional, if needed for context) ---
def get_shop_id_from_path(path: str) -> str | None:
    """Helper to extract shop_id, assuming path structure like '.../results/{shop_id}/(raw|preprocessed)/file.csv'"""
    match = re.search(r'[\\/]results[\\/]([^\\/]+)[\\/](?:raw|preprocessed)[\\/]', path)
    if match: return match.group(1)
    match = re.search(r'[\\/]results[\\/]([^\\/]+)[\\/]', path)
    if match: return match.group(1)
    print(f"Warning: Could not automatically determine shop_id from path: {path}")
    return None

# --- Main Orchestration Function ---
def run_event_preprocessing(input_path: str, output_path: str):
    """
    Orchestrates the preprocessing of event data to extract interactions.

    Args:
        input_path: Path to the filtered event data CSV (e.g., .../raw/shop_person_event_data_filtered.csv).
        output_path: Path to save the processed event interactions CSV (e.g., .../preprocessed/shop_event_interactions_processed.csv).
    """
    print(f"\n--- Starting Event Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # --- Load Raw Event Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        event_df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {len(event_df)} filtered event rows from {input_path}.")

        if event_df.empty:
            print("Input event DataFrame is empty. Creating empty output interactions file.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS).to_csv(output_path, index=False) # Use final cols for empty file
            print(f"Empty output file created at: {output_path}")
            return # Successfully handled empty input

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error loading input event CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Process Events to Extract Interactions ---
    try:
        print("\nExtracting and processing interactions...")
        # This function now includes the channel refinement logic, deduplication, and selects final columns
        interactions_df = extract_interactions_from_events(event_df)
        if not interactions_df.empty:
             print(f"Processed interactions DataFrame shape (ready to save): {interactions_df.shape}")
             print(f"Processed interactions columns (ready to save): {interactions_df.columns.tolist()}")
        else:
             print("Processing resulted in an empty DataFrame.")

    except Exception as e:
        print(f"Error during event interaction extraction: {e}")
        traceback.print_exc()
        raise # Re-raise error to stop the pipeline

    # --- START: Verification Step for Duplicates ---
    if not interactions_df.empty and 'checkout_id' in interactions_df.columns and 'product_id' in interactions_df.columns:
        print("\n--- Verifying Duplicates (checkout_id, product_id) ---")
        # Consider only rows where checkout_id is not null/NA
        relevant_df = interactions_df[interactions_df['checkout_id'].notna()].copy()

        if not relevant_df.empty:
            # Find all rows that are part of a duplicate pair based on checkout_id and product_id
            duplicate_mask = relevant_df.duplicated(subset=['checkout_id', 'product_id'], keep=False)
            n_duplicates_found = duplicate_mask.sum()

            if n_duplicates_found > 0:
                print(f"WARNING: Found {n_duplicates_found} rows involved in duplicate (checkout_id, product_id) pairs AFTER processing.")
                # Get the actual duplicate rows
                duplicate_rows = relevant_df[duplicate_mask]
                # Sort to see them together
                duplicate_rows_sorted = duplicate_rows.sort_values(by=['checkout_id', 'product_id', 'timestamp'])
                print("Interaction types of duplicate rows:")
                print(duplicate_rows_sorted['interaction_type'].value_counts())
                print("\nSample of duplicate rows found:")
                # Show relevant columns for easy comparison
                print(duplicate_rows_sorted[['person_id_hex', 'checkout_id', 'product_id', 'timestamp', 'interaction_type']].head(10))
            else:
                print("OK: No duplicate (checkout_id, product_id) pairs found after processing.")
        else:
            print("Skipping duplicate check: No rows with non-null checkout_id found.")
    else:
        print("\nSkipping duplicate check: DataFrame is empty or missing required columns ('checkout_id', 'product_id').")
    # --- END: Verification Step ---

    # --- Save Processed Interactions ---
    try:
        # Check if DataFrame is empty after processing
        if interactions_df.empty:
             print("\nInteraction DataFrame is empty. Saving empty file.")
             output_dir = os.path.dirname(output_path)
             if output_dir: os.makedirs(output_dir, exist_ok=True)
             pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS).to_csv(output_path, index=False) # Use final cols for empty file
        else:
            # DataFrame should already contain only the desired columns from extract_interactions_from_events
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nEnsured output directory exists: {output_dir}")
            interactions_df.to_csv(output_path, index=False)

        print(f"Successfully saved processed event interactions to: {output_path}")
        print(f"--- Finished Event Data Preprocessing Orchestration ---")

    except Exception as e:
        print(f"Error saving output event interactions CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

# --- Command Line Execution Logic (for standalone testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess event data to extract interactions.')
    parser.add_argument('--input', required=True, help='Path to the input person_event_data_filtered.csv file.')
    parser.add_argument('--output', required=True, help='Path to save the final event_interactions_processed.csv file.')
    args = parser.parse_args()

    try:
        run_event_preprocessing(args.input, args.output)
        print("\nEvent preprocessing finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error: # Catch potential errors from extract function
         print(f"\nError: {val_error}")
         exit(1)
    except Exception as e:
        print(f"\n--- Event preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)