# preprocessing/preprocess_product_data.py
import argparse
import os
import pandas as pd
import traceback
import re
import numpy as np # Import numpy for numeric checks

# --- Import all necessary processor functions ---
try:
    from .product_processor.availableForSale_processor import process_avaiable_for_sale
    from .product_processor.collectionHandle_processor import process_collections
    from .product_processor.createdAt_processor import process_created_at
    from .product_processor.handle_processor import process_handles_tfidf
    from .product_processor.isGiftCard_processor import process_gift_card
    from .product_processor.metafields_processor import process_metafields
    from .product_processor.product_type_processor import process_product_type
    from .product_processor.tags_processor import process_tags
    from .product_processor.title_processor import process_titles_tfidf
    from .product_processor.variants_processor import process_variants
    from .product_processor.vendor_processor import process_vendors
    # --- ADDED IMPORT for description processor ---
    from .product_processor.tfidf_processor import process_descriptions_tfidf
    # Assuming utils.py is one level up (though not strictly needed here)
    # from .utils import safe_json_parse
except ImportError as e:
    print(f"Error importing product processors: {e}. Check paths and __init__.py files.")
    # Add fallback imports if necessary for local testing, but prefer fixing the structure/execution
    # from product_processor.availableForSale_processor import process_avaiable_for_sale
    # ... etc ...
    raise # Stop execution if imports fail critically

# --- Helper Function (if needed, otherwise remove) ---
def get_shop_id_from_path(path: str) -> str | None:
    """Helper to extract shop_id, assuming path structure like '.../results/{shop_id}/(raw|preprocessed)/file.csv'"""
    # Look for '/results/SHOP_ID/' pattern
    match = re.search(r'[\\/]results[\\/]([^\\/]+)[\\/](?:raw|preprocessed)[\\/]', path)
    if match:
        return match.group(1)
    # Fallback: Try just '/results/SHOP_ID/' if subfolders aren't guaranteed
    match = re.search(r'[\\/]results[\\/]([^\\/]+)[\\/]', path)
    if match:
         return match.group(1)
    print(f"Warning: Could not automatically determine shop_id from path: {path}")
    return None

# --- Main Orchestration Function ---
def run_product_preprocessing(input_path: str, output_path: str):
    """
    Orchestrates the preprocessing of product data by calling individual processors.

    Args:
        input_path: Path to the raw product data CSV (e.g., .../raw/shop_products_raw.csv).
        output_path: Path to save the processed product features CSV (e.g., .../preprocessed/shop_products_final.csv).
    """
    print(f"\n--- Starting Product Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # Determine shop_id and output directory for auxiliary files/models
    shop_id = get_shop_id_from_path(output_path) # Use output path as it defines the target shop dir structure
    if not shop_id:
         raise ValueError("Could not determine shop_id from output path. Needed for saving models/auxiliary files.")
    print(f"Determined Shop ID: {shop_id}")

    # Base output directory is the one containing the final output file (.../preprocessed/)
    output_dir = os.path.dirname(output_path)
    # Models will go into a subdirectory within this base output directory
    models_output_dir = os.path.join(output_dir, "models")
    # Ensure directories exist (redundant if run_preprocessing creates them, but safe)
    os.makedirs(models_output_dir, exist_ok=True)
    print(f"Main output directory: {output_dir}")
    print(f"Models output directory: {models_output_dir}")


    # --- Load Raw Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {len(df)} raw product rows from {input_path}.")
        if df.empty:
            print("Input DataFrame is empty. Creating empty output file.")
            pd.DataFrame(columns=['product_id']).to_csv(output_path, index=False) # Ensure output dir exists via get_data_paths
            print(f"Empty output file created at: {output_path}")
            return
        if 'product_id' not in df.columns:
            raise ValueError("Input product data must contain a 'product_id' column.")
        df['product_id'] = df['product_id'].astype(str) # Ensure consistent string type

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error loading input product CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Apply Processors Sequentially ---
    processed_df = df.copy() # Start with a copy

    try:
        # 1. Filtering (remove unwanted products first)
        processed_df = process_gift_card(processed_df)
        if processed_df.empty: raise ValueError("DataFrame empty after gift card filter.")
        processed_df = process_avaiable_for_sale(processed_df)
        if processed_df.empty: raise ValueError("DataFrame empty after available_for_sale filter.")

        # 2. Basic Column Transformations & Dummies
        processed_df = process_created_at(processed_df)
        processed_df = process_product_type(processed_df)
        processed_df = process_vendors(processed_df)

        # 3. List/Dict Column Processing & Dummies
        processed_df = process_tags(processed_df)
        processed_df = process_collections(processed_df) # Corrected processor

        # 4. Variant Processing (Extracts features, options, normalizes prices)
        processed_df = process_variants(processed_df)

        # 5. Text Processing (TF-IDF) - Title, Handle, Description
        # Pass models_output_dir for saving vectorizers
        processed_df = process_titles_tfidf(processed_df, models_output_dir, shop_id)
        processed_df = process_handles_tfidf(processed_df, models_output_dir, shop_id)
        
        # Pass output_dir for similarity report, function puts model in output_dir/models/
        processed_df = process_descriptions_tfidf(processed_df, output_dir, shop_id)
        

        # 6. Metafields Processing (Complex - may add more features, saves aux files)
        # This function needs the base output_dir to save refs/similarity correctly
        processed_df, references_df, color_similarity_df = process_metafields(processed_df, output_dir, shop_id)
        # references_df and color_similarity_df are saved by the function, not used further here

        # --- Final Checks ---
        print("\nPerforming final checks on processed product data...")
        if 'product_id' not in processed_df.columns:
            raise ValueError("'product_id' column lost during processing!")
        processed_df['product_id'] = processed_df['product_id'].astype(str)

        # Optional: Fill any remaining NaNs in numeric columns with 0
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if processed_df[col].isnull().any():
                print(f"Filling remaining NaNs in numeric column '{col}' with 0.")
                processed_df[col] = processed_df[col].fillna(0)

        # Ensure product_id is the first column
        cols = ['product_id'] + [col for col in processed_df.columns if col != 'product_id']
        final_df = processed_df[cols]

        print(f"\nFinal Processed Product DataFrame shape: {final_df.shape}")
        print(f"Final columns ({len(final_df.columns)}): {final_df.columns.tolist()}")

    except ValueError as ve: # Catch value errors from empty DFs during processing
         print(f"\n--- Processing stopped due to empty data after a step: {ve} ---")
         # Decide if to save empty or raise error. Let's raise.
         raise
    except Exception as e:
        print(f"\n--- Error during product processing sequence: {e} ---")
        traceback.print_exc()
        raise # Propagate the error

    # --- Save Final Processed Data ---
    try:
        # Ensure output directory exists (should already exist)
        os.makedirs(output_dir, exist_ok=True)

        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved processed product data to: {output_path}")
        print(f"--- Finished Product Data Preprocessing Orchestration ---")

    except Exception as e:
        print(f"Error saving final processed product CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

# --- Command Line Execution Logic (for standalone testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess product data (Orchestrator).')
    parser.add_argument('--input', required=True, help='Path to the input products_raw.csv file.')
    parser.add_argument('--output', required=True, help='Path to save the final products_final_all_features.csv file.')
    args = parser.parse_args()

    try:
        run_product_preprocessing(args.input, args.output)
        print("\nProduct preprocessing finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error: # Catch empty DataFrame errors etc.
        print(f"\nError: {val_error}")
        exit(1)
    except Exception as e:
        print(f"\n--- Product preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)

    # Example Usage (from project root 'NappsRecommenderSystems/'):
    # python -m preprocessing.preprocess_product_data --input results/missusstore/raw/missusstore_shopify_products_raw.csv --output results/missusstore/preprocessed/missusstore_products_final_all_features.csv