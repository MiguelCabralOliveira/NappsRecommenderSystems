# preprocessing/run_preprocessing.py
import argparse
import os
import traceback
import sys

# Ensure the project root is in the Python path for relative imports
# (Potentially needed depending on execution context)
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# if project_root not in sys.path:
#     sys.path.append(project_root)


# Import the specific preprocessing runners
try:
    # Use relative imports if running as a module (e.g., python -m preprocessing.run_preprocessing)
    from .preprocess_person_data import run_person_preprocessing
    from .preprocess_product_data import run_product_preprocessing
except ImportError:
     # Fallback for direct execution (e.g., python preprocessing/run_preprocessing.py)
     # Requires project root to be in PYTHONPATH or careful execution path.
    try:
        from preprocess_person_data import run_person_preprocessing
        from preprocess_product_data import run_product_preprocessing
    except ModuleNotFoundError as e:
        print(f"Import Error: {e}. Could not import specific preprocessing modules.")
        print("Ensure you are running from the project root 'NappsRecommender/' using")
        print("`python -m preprocessing.run_preprocessing ...` or that the")
        print("project structure is correctly added to your PYTHONPATH.")
        sys.exit(1)


# Define expected SageMaker input/output channel names
SM_INPUT_CHANNEL = 'raw_data'  # Assumes input data is under this channel
SM_OUTPUT_CHANNEL = 'processed_data' # Output data will be saved here

def is_sagemaker_environment():
    """Checks if the script is running in a SageMaker Processing Job environment."""
    # Check for common SageMaker environment variables or paths
    return 'SM_CURRENT_HOST' in os.environ or os.path.exists('/opt/ml/processing/')

def get_data_paths(shop_id: str) -> dict:
    """
    Determines the input and output paths based on the environment (local vs SageMaker).

    Args:
        shop_id: The identifier for the specific shop.

    Returns:
        A dictionary containing the paths for person and product input/output files.
    """
    paths = {}
    is_sm = is_sagemaker_environment()

    if is_sm:
        print("Detected SageMaker environment.")
        # SageMaker paths: /opt/ml/processing/<channel_name>/<optional_subdir>/<filename>
        # Assuming data for the shop is directly under the channel directory OR in a shop_id subdir
        # Option 1: Files directly under channel (e.g., /opt/ml/processing/raw_data/person_data_loaded.csv)
        # base_input_dir = f"/opt/ml/processing/{SM_INPUT_CHANNEL}"
        # base_output_dir = f"/opt/ml/processing/{SM_OUTPUT_CHANNEL}"
        # Option 2: Files under a shop_id subdir (e.g., /opt/ml/processing/raw_data/shop123/person_data_loaded.csv)
        # This seems more likely if multiple shops might be processed or data is organized this way in S3
        base_input_dir = f"/opt/ml/processing/{SM_INPUT_CHANNEL}/{shop_id}"
        base_output_dir = f"/opt/ml/processing/{SM_OUTPUT_CHANNEL}/{shop_id}"
        print(f"Using SageMaker base input: {base_input_dir}")
        print(f"Using SageMaker base output: {base_output_dir}")

    else:
        print("Detected local environment.")
        # Local paths relative to the assumed project root execution
        # Assumes 'results/<shop_id>/' structure exists based on project structure
        base_dir = os.path.join("results", shop_id)
        base_input_dir = base_dir # Inputs are in the shop directory
        base_output_dir = base_dir # Outputs go to the same shop directory
        print(f"Using local base directory: {base_dir}")


    # Define standard filenames (matching results/README.md conventions)
    paths['person_input'] = os.path.join(base_input_dir, 'person_data_loaded.csv')
    paths['person_output'] = os.path.join(base_output_dir, 'person_features_processed.csv')
    paths['product_input'] = os.path.join(base_input_dir, 'products_raw.csv')
    paths['product_output'] = os.path.join(base_output_dir, 'products_final_all_features.csv')

    print("\nDetermined data paths:")
    for key, value in paths.items():
        print(f"- {key}: {value}")

    # Create the output directory if it doesn't exist (important locally and sometimes in SM)
    # Use the directory of the first output file to determine the base output dir to create
    output_dir_to_create = os.path.dirname(paths['person_output']) # Or product_output
    if output_dir_to_create: # Ensure it's not empty
        os.makedirs(output_dir_to_create, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir_to_create}")


    return paths

def main():
    """Main function to parse arguments and orchestrate preprocessing."""
    parser = argparse.ArgumentParser(description='Run the preprocessing pipeline for person and/or product data.')
    # Add shop-id as the primary argument
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop to process.')
    # Add mode argument
    parser.add_argument('--mode', required=True, choices=['person', 'product', 'all'],
                        help='Which preprocessing pipeline to run: "person", "product", or "all".')

    args = parser.parse_args()
    shop_id = args.shop_id
    mode = args.mode

    print(f"\n=== Running Preprocessing Pipeline ===")
    print(f"Shop ID: {shop_id}")
    print(f"Mode: {mode}")

    try:
        # Determine the correct paths based on environment and shop_id
        paths = get_data_paths(shop_id)

        # --- Execute Person Preprocessing ---
        if mode in ['person', 'all']:
            print("\n--- Starting Person Preprocessing ---")
            # Check if input file exists before running
            if not os.path.exists(paths['person_input']):
                print(f"Error: Person input file not found at designated path: {paths['person_input']}")
                raise FileNotFoundError(f"Person input file not found: {paths['person_input']}")

            try:
                # Call the person preprocessing orchestrator
                run_person_preprocessing(paths['person_input'], paths['person_output'])
                print("--- Person Preprocessing Completed Successfully ---")
            except FileNotFoundError as fnf_error:
                 print(f"Error during person preprocessing step: {fnf_error}")
                 raise # Stop the pipeline
            except Exception as e:
                print(f"--- Person Preprocessing FAILED ---")
                print(f"Error details: {e}")
                traceback.print_exc()
                raise # Stop execution

        # --- Execute Product Preprocessing ---
        if mode in ['product', 'all']:
            print("\n--- Starting Product Preprocessing ---")
            # Check if input file exists before running
            if not os.path.exists(paths['product_input']):
                print(f"Error: Product input file not found at designated path: {paths['product_input']}")
                raise FileNotFoundError(f"Product input file not found: {paths['product_input']}")

            try:
                # Call the product preprocessing orchestrator
                run_product_preprocessing(paths['product_input'], paths['product_output'])
                print("--- Product Preprocessing Completed Successfully ---")
            except FileNotFoundError as fnf_error:
                 print(f"Error during product preprocessing step: {fnf_error}")
                 raise # Stop the pipeline
            except Exception as e:
                print(f"--- Product Preprocessing FAILED ---")
                print(f"Error details: {e}")
                traceback.print_exc()
                raise # Stop execution

        print(f"\n=== Preprocessing Pipeline Finished Successfully for Shop '{shop_id}' (Mode: {mode}) ===")

    except FileNotFoundError as e:
        print(f"\nPipeline Aborted: Required file not found.")
        print(f"Error details: {e}")
        sys.exit(2) # Specific exit code for file not found
    except ValueError as e:
         print(f"\nPipeline Aborted: Configuration or Data Error.")
         print(f"Error details: {e}")
         traceback.print_exc()
         sys.exit(3) # Specific exit code for value errors
    except Exception as e:
        print("\n--- Preprocessing Pipeline FAILED Overall ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1) # Generic error code

if __name__ == "__main__":
    main()

# Example Local Usage (from project root NappsRecommender/):
# python -m preprocessing.run_preprocessing --shop-id test_shop --mode all
# python -m preprocessing.run_preprocessing --shop-id another_shop --mode product
#
# Example SageMaker Command (in Processing Job definition):
# command: ["python", "/opt/ml/processing/code/preprocessing/run_preprocessing.py", "--shop-id", "arg_shop_id", "--mode", "all"]
# (Assuming your code is copied to /opt/ml/processing/code/ and shop_id is passed, e.g. via hyperparameters or args)