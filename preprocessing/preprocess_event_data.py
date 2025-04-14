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
except ImportError:
    print("Warning: Could not use relative imports for event processor. Attempting direct import.")
    try:
        from event_processor.process_event_interactions import extract_interactions_from_events
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
            # Define expected columns for empty output based on extract_interactions_from_events
            output_columns = [
                'person_id_hex', 'product_id', 'interaction_type', 'timestamp',
                'event_converted_price', 'event_converted_currency', 'event_quantity',
                'order_id', 'checkout_id', 'channel', 'session_id', 'event_id'
            ]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
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
        print("\nExtracting interactions from events...")
        interactions_df = extract_interactions_from_events(event_df)
        print(f"Extracted {len(interactions_df)} interactions.")

    except Exception as e:
        print(f"Error during event interaction extraction: {e}")
        traceback.print_exc()
        raise

    # --- Save Processed Interactions ---
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

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

    # Example Usage (from project root 'NappsRecommenderSystems/'):
    # python -m preprocessing.preprocess_event_data --input results/missusstore/raw/missusstore_person_event_data_filtered.csv --output results/missusstore/preprocessed/missusstore_event_interactions_processed.csv