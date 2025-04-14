# preprocessing/run_preprocessing.py
import argparse
import os
import traceback
import sys

# --- Imports ---
try:
    from .preprocess_person_data import run_person_preprocessing
    from .preprocess_product_data import run_product_preprocessing
    from .preprocess_event_data import run_event_preprocessing
except ImportError as e:
    print(f"Import Error: {e}. Failed to import submodules from within 'preprocessing'.")
    print("Check __init__.py files and execution context (run from project root with -m).")
    sys.exit(1)
# --- End Imports ---

# --- Constants ---
SM_INPUT_CHANNEL = 'raw_data'
SM_OUTPUT_CHANNEL = 'processed_data'
# --- End Constants ---

# --- Environment Detection ---
def is_sagemaker_environment():
    """Checks if the script is running in a SageMaker Processing Job environment."""
    return 'SM_CURRENT_HOST' in os.environ or os.path.exists('/opt/ml/processing/')
# --- End Environment Detection ---

# --- Path Management ---
def get_data_paths(shop_id: str) -> dict:
    """
    Determines input/output paths based on environment and desired structure.
    Inputs from: .../{shop_id}/raw/
    Outputs to: .../{shop_id}/preprocessed/

    Args:
        shop_id: The identifier for the specific shop.

    Returns:
        A dictionary containing the absolute paths for input/output files.
    """
    paths = {}
    is_sm = is_sagemaker_environment()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if is_sm:
        print("Detected SageMaker environment.")
        base_input_dir = f"/opt/ml/processing/{SM_INPUT_CHANNEL}/{shop_id}/raw"
        base_output_dir = f"/opt/ml/processing/{SM_OUTPUT_CHANNEL}/{shop_id}/preprocessed"
        print(f"Using SageMaker base input: {base_input_dir}")
        print(f"Using SageMaker base output: {base_output_dir}")
    else:
        print("Detected local environment.")
        base_input_dir = os.path.join(project_root, "results", shop_id, "raw")
        base_output_dir = os.path.join(project_root, "results", shop_id, "preprocessed")
        print(f"Using local base input directory: {base_input_dir}")
        print(f"Using local base output directory: {base_output_dir}")

    # --- Define standard filenames ---
    # Verify these match your actual filenames
    person_input_filename = f'{shop_id}_person_data_loaded.csv'
    product_input_filename = f'{shop_id}_shopify_products_raw.csv'
    # --- ADDED: Event filenames ---
    event_input_filename = f'{shop_id}_person_event_data_filtered.csv' # From data_loading DB output
    event_output_filename = f'{shop_id}_event_interactions_processed.csv' # New output name
    # --- End ADDED ---
    person_output_filename = f'{shop_id}_person_features_processed.csv'
    product_output_filename = f'{shop_id}_products_final_all_features.csv'
    # --- End Define standard filenames ---

    # Construct full paths
    paths['person_input'] = os.path.join(base_input_dir, person_input_filename)
    paths['product_input'] = os.path.join(base_input_dir, product_input_filename)
    # --- ADDED: Event paths ---
    paths['event_input'] = os.path.join(base_input_dir, event_input_filename)
    paths['event_output'] = os.path.join(base_output_dir, event_output_filename)
    # --- End ADDED ---
    paths['person_output'] = os.path.join(base_output_dir, person_output_filename)
    paths['product_output'] = os.path.join(base_output_dir, product_output_filename)

    print("\nDetermined data paths:")
    for key, value in paths.items():
        print(f"- {key}: {value}")

    # Create the main output directory ('.../preprocessed/') if it doesn't exist
    output_dir_to_create = base_output_dir
    if output_dir_to_create:
        os.makedirs(output_dir_to_create, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir_to_create}")
        # Also create the 'models' subdirectory within the output path ('.../preprocessed/models/')
        os.makedirs(os.path.join(output_dir_to_create, "models"), exist_ok=True)
        print(f"Ensured models subdirectory exists: {os.path.join(output_dir_to_create, 'models')}")

    return paths
# --- End Path Management ---

# --- Main Execution Logic ---
def main():
    """Main function to parse arguments and orchestrate preprocessing."""
    parser = argparse.ArgumentParser(description='Run the preprocessing pipeline for person, product, and/or event data.')
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop to process.')
    # --- ADDED: 'events' mode ---
    parser.add_argument('--mode', required=True, choices=['person', 'product', 'events', 'all'],
                        help='Which preprocessing pipeline(s) to run.')
    # --- END ADDED ---
    args = parser.parse_args()
    shop_id = args.shop_id
    mode = args.mode

    print(f"\n=== Running Preprocessing Pipeline ===")
    print(f"Shop ID: {shop_id}")
    print(f"Mode: {mode}")

    try:
        # 1. Determine paths
        paths = get_data_paths(shop_id)

        # 2. Execute Person Preprocessing if requested
        if mode in ['person', 'all']:
            print("\n--- Starting Person Preprocessing ---")
            if not os.path.exists(paths['person_input']):
                raise FileNotFoundError(f"Person input file not found: {paths['person_input']}")
            run_person_preprocessing(paths['person_input'], paths['person_output'])
            print("--- Person Preprocessing Completed Successfully ---")

        # 3. Execute Product Preprocessing if requested
        if mode in ['product', 'all']:
            print("\n--- Starting Product Preprocessing ---")
            if not os.path.exists(paths['product_input']):
                raise FileNotFoundError(f"Product input file not found: {paths['product_input']}")
            run_product_preprocessing(paths['product_input'], paths['product_output'])
            print("--- Product Preprocessing Completed Successfully ---")

        # --- ADDED: Event Preprocessing ---
        # 4. Execute Event Preprocessing if requested
        if mode in ['events', 'all']:
            print("\n--- Starting Event Interaction Processing ---")
            if not os.path.exists(paths['event_input']):
                raise FileNotFoundError(f"Event input file not found: {paths['event_input']}")
            run_event_preprocessing(paths['event_input'], paths['event_output'])
            print("--- Event Interaction Processing Completed Successfully ---")
        # --- END ADDED ---

        # 5. Final Success Message
        print(f"\n=== Preprocessing Pipeline Finished Successfully for Shop '{shop_id}' (Mode: {mode}) ===")

    # --- Error Handling (includes FileNotFoundError and ValueError) ---
    except FileNotFoundError as e:
        print(f"\nPipeline Aborted: Required file not found.")
        print(f"Error details: {e}")
        sys.exit(2)
    except ValueError as e:
         print(f"\nPipeline Aborted: Configuration or Data Error.")
         print(f"Error details: {e}")
         traceback.print_exc()
         sys.exit(3)
    except Exception as e:
        print("\n--- Preprocessing Pipeline FAILED Overall ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
    # --- End Error Handling ---

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
# --- End Script Entry Point ---

# Example Local Usage (from project root NappsRecommenderSystems/):
# python -m preprocessing.run_preprocessing --shop-id missusstore --mode all
# python -m preprocessing.run_preprocessing --shop-id missusstore --mode events