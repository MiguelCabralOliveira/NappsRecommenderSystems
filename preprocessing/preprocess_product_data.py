# preprocessing/preprocess_product_data.py
import argparse
import os
import numpy as np
import pandas as pd
import traceback
import re

# Import all necessary processor functions using relative imports
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
    # Assuming utils.py is one level up
    from .utils import safe_json_parse
except ImportError:
    print("Error importing product processors. Check paths and __init__.py files.")
    # Fallback for potential local testing issues (less ideal)
    from product_processor.availableForSale_processor import process_avaiable_for_sale
    from product_processor.collectionHandle_processor import process_collections
    from product_processor.createdAt_processor import process_created_at
    from product_processor.handle_processor import process_handles_tfidf
    from product_processor.isGiftCard_processor import process_gift_card
    from product_processor.metafields_processor import process_metafields
    from product_processor.product_type_processor import process_product_type
    from product_processor.tags_processor import process_tags
    from product_processor.title_processor import process_titles_tfidf
    from product_processor.variants_processor import process_variants
    from product_processor.vendor_processor import process_vendors
    from utils import safe_json_parse


def get_shop_id_from_path(path: str) -> str | None:
    """Helper to extract shop_id, assuming path structure like '.../results/shop_id/file.csv'"""
    # Match '/results/SHOP_ID/' part
    match = re.search(r'[\\/]results[\\/]([^\\/]+)[\\/]', path)
    if match:
        return match.group(1)
    # Fallback: maybe shop_id is in filename like 'shop_id_products_raw.csv' in 'results/'?
    match = re.search(r'[\\/]results[\\/]([^\\/_]+)_[^\\/]+\.csv$', path)
    if match:
        return match.group(1)
    print(f"Warning: Could not automatically determine shop_id from path: {path}")
    return None


def run_product_preprocessing(input_path: str, output_path: str):
    """
    Orchestrates the preprocessing of product data by calling individual processors.

    Args:
        input_path: Path to the raw product data CSV (e.g., products_raw.csv).
        output_path: Path to save the processed product features CSV.
    """
    print(f"\n--- Starting Product Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # Determine shop_id and output directory for auxiliary files/models
    shop_id = get_shop_id_from_path(input_path)
    if not shop_id:
        # Fallback: try output path
        shop_id = get_shop_id_from_path(output_path)
    if not shop_id:
         # If still not found, raise error or use a default? Let's raise.
         raise ValueError("Could not determine shop_id from input/output paths. Needed for saving models/auxiliary files.")
    print(f"Determined Shop ID: {shop_id}")

    # Output directory for models, references, similarity etc.
    # Should be the directory containing the main output file.
    output_dir = os.path.dirname(output_path)
    models_output_dir = os.path.join(output_dir, "models") # Specific subdir for models
    os.makedirs(models_output_dir, exist_ok=True)
    print(f"Auxiliary output directory: {output_dir}")
    print(f"Models output directory: {models_output_dir}")


    # --- Load Raw Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {len(df)} raw product rows from {input_path}.")
        if df.empty:
            print("Input DataFrame is empty. Creating empty output file.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Create empty file with at least product_id if possible, otherwise empty
            pd.DataFrame(columns=['product_id']).to_csv(output_path, index=False)
            print(f"Empty output file created at: {output_path}")
            return
        # Ensure product_id exists early
        if 'product_id' not in df.columns:
            raise ValueError("Input product data must contain a 'product_id' column.")
        # Convert product_id to string to handle potential numeric GIDs
        df['product_id'] = df['product_id'].astype(str)

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
        if processed_df.empty: print("DataFrame empty after gift card filter."); raise ValueError("No products left after filtering.")
        processed_df = process_avaiable_for_sale(processed_df)
        if processed_df.empty: print("DataFrame empty after available_for_sale filter."); raise ValueError("No products left after filtering.")

        # 2. Basic Column Transformations & Dummies
        processed_df = process_created_at(processed_df)
        processed_df = process_product_type(processed_df)
        processed_df = process_vendors(processed_df)

        # 3. List/Dict Column Processing & Dummies
        processed_df = process_tags(processed_df)
        processed_df = process_collections(processed_df) # Before metafields potentially?

        # 4. Variant Processing (Extracts features, options, normalizes prices)
        processed_df = process_variants(processed_df)

        # 5. Text Processing (TF-IDF on title and handle) - Needs output dir for models
        processed_df = process_titles_tfidf(processed_df, models_output_dir, shop_id)
        processed_df = process_handles_tfidf(processed_df, models_output_dir, shop_id)

        # 6. Metafields Processing (Complex - extracts features, vectorizes text, saves refs/similarity)
        # This function saves auxiliary files itself, given the base output_dir
        processed_df, references_df, color_similarity_df = process_metafields(processed_df, output_dir, shop_id)
        # We don't strictly *need* references_df and color_similarity_df here,
        # as they are saved by the function, but good to have them if needed later.

        # --- Final Checks ---
        print("\nPerforming final checks on processed product data...")
        # Ensure product_id is still present and string
        if 'product_id' not in processed_df.columns:
            raise ValueError("'product_id' column lost during processing!")
        processed_df['product_id'] = processed_df['product_id'].astype(str)

        # Optional: Fill any remaining NaNs in numeric columns with 0
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if processed_df[col].isnull().any():
                print(f"Filling remaining NaNs in numeric column '{col}' with 0.")
                processed_df[col] = processed_df[col].fillna(0)

        # Make product_id the first column for readability
        cols = ['product_id'] + [col for col in processed_df.columns if col != 'product_id']
        final_df = processed_df[cols]

        print(f"\nFinal Processed Product DataFrame shape: {final_df.shape}")
        print(f"Final columns ({len(final_df.columns)}): {final_df.columns.tolist()}")
        # print("\nSample of final product data:")
        # print(final_df.head())
        # print("\nData types of final product data:")
        # print(final_df.dtypes)

    except Exception as e:
        print(f"\n--- Error during product processing sequence: {e} ---")
        traceback.print_exc()
        raise # Propagate the error to the main runner

    # --- Save Final Processed Data ---
    try:
        # Ensure output directory exists (should already exist from model saving)
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
    except Exception as e:
        print(f"\n--- Product preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)

    # Example Usage (from project root):
    # python -m preprocessing.preprocess_product_data --input results/test_shop/products_raw.csv --output results/test_shop/products_final_all_features.csv