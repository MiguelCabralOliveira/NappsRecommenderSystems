# training/similarity/train_knn_model.py

import argparse
import os
import sys
import traceback
import pandas as pd
import numpy as np


# --- Ajustar o sys.path para permitir imports relativos corretos ---
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(training_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
if training_dir not in sys.path: sys.path.insert(0, training_dir)

# --- Imports do Módulo KNN ---
try:
    # Use import relativo dentro do pacote similarity
    from .knn.knn_pipeline import run_knn_training
    from .utils import load_dataframe, load_json_config, ensure_dir
except ImportError as e:
    print(f"Error importing KNN modules (relative): {e}")
    try: # Fallback absoluto
         print("Attempting absolute import fallback...")
         from training.similarity.knn.knn_pipeline import run_knn_training
         from .utils import load_dataframe, load_json_config, ensure_dir
    except ImportError:
         print("Absolute import fallback also failed. Exiting.")
         sys.exit(1)

def get_default_paths(shop_id: str, base_project_root: str) -> dict:
    """
    Determines the default input and output paths based on shop_id and project root.
    Catalog path now points to the 'raw' directory.
    """
    results_base = os.path.join(base_project_root, "results", shop_id)
    preprocessed_dir = os.path.join(results_base, "preprocessed")
    raw_dir = os.path.join(results_base, "raw") # <-- Path para a pasta raw
    training_output_dir = os.path.join(results_base, "training_knn")

    if not os.path.isdir(os.path.join(base_project_root, "results")):
        print(f"Error: Base 'results' directory not found at {os.path.join(base_project_root, 'results')}")
        return None

    paths = {
        "input_features": os.path.join(preprocessed_dir, f"{shop_id}_products_final_all_features.csv"),
        "similar_products_csv": os.path.join(preprocessed_dir, f"{shop_id}_description_similarity_report.csv"),
        # *** CORRIGIDO: Aponta para a pasta 'raw' para o catálogo ***
        "catalog_csv": os.path.join(raw_dir, f"{shop_id}_shopify_products_raw.csv"), # Ajusta nome do ficheiro se necessário
        "output_dir": training_output_dir
    }
    return paths


def main():
    """
    Main script to run the KNN training pipeline from the command line.
    Derives input/output paths based on shop_id.
    """
    parser = argparse.ArgumentParser(
        description='Train a Weighted KNN model for product recommendations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input/Output Arguments ---
    parser.add_argument('--shop-id', required=True,
                        help='Shop identifier. Used to determine input/output paths.')
    # Optional overrides
    parser.add_argument('--input-features-override', type=str, default=None,
                        help='Optional: Explicitly override path for input features CSV.')
    parser.add_argument('--output-dir-override', type=str, default=None,
                        help='Optional: Explicitly override path for the main output directory.')
    parser.add_argument('--similar-products-csv-override', type=str, default=None,
                        help='Optional: Explicitly override path for similar products CSV.')
    parser.add_argument('--catalog-csv-override', type=str, default=None,
                        help='Optional: Explicitly override path for the product catalog CSV (used for evaluation). Default assumes <shop_id>_shopify_products_raw.csv in raw dir.')


    # --- KNN Core Parameters ---
    parser.add_argument('--neighbors', type=int, default=20,
                        help='Number of neighbors (recommendations).')

    # --- Weighting Configuration ---
    parser.add_argument('--weights-config', type=str, default=None,
                        help='Optional: Path to JSON file defining base weights.')
    parser.add_argument('--optimize-weights', action='store_true',
                        help='Run feature weight optimization.')
    parser.add_argument('--optimization-method', type=str, default='evolutionary',
                        choices=['grid', 'random', 'evolutionary'],
                        help='Method for weight optimization.')
    parser.add_argument('--optimization-config', type=str, default=None,
                        help='Optional: Path to JSON file with optimization parameters.')

    # --- Filtering ---
    parser.add_argument('--filter-variants', action='store_true',
                        help='Filter recommendations identified as similar variants.')

    # --- Execution Control ---
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip the evaluation step.')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save the trained KNN model.')
    parser.add_argument('--no-save-model', action='store_false', dest='save_model',
                        help='Do not save the trained KNN model.')

    # --- Parsing ---
    args = parser.parse_args()

    # --- Determine Paths ---
    print(f"Determining paths for shop_id: {args.shop_id}")
    default_paths = get_default_paths(args.shop_id, project_root)
    if default_paths is None: sys.exit(1)

    input_features_path = args.input_features_override or default_paths['input_features']
    output_dir_path = args.output_dir_override or default_paths['output_dir']
    similar_products_csv_path = args.similar_products_csv_override or default_paths['similar_products_csv']
    catalog_csv_path = args.catalog_csv_override or default_paths['catalog_csv'] # Determine catalog path

    print(f"Using Input Features Path: {input_features_path}")
    print(f"Using Output Directory: {output_dir_path}")
    # Check existence before printing the path to be used
    catalog_path_to_print = "Not Found / Not Used"
    if os.path.exists(catalog_csv_path):
         catalog_path_to_print = catalog_csv_path
    print(f"Using Catalog Path (for Eval): {catalog_path_to_print}")
    if args.filter_variants:
        similar_path_to_print = "Not Found / Not Used"
        if os.path.exists(similar_products_csv_path):
             similar_path_to_print = similar_products_csv_path
        print(f"Using Similar Products Path (for filtering): {similar_path_to_print}")

    # --- Input Validation ---
    if not os.path.exists(input_features_path):
        print(f"Error: Input features file not found: {input_features_path}"); sys.exit(1)

    effective_similar_products_path = None
    if args.filter_variants:
        if not os.path.exists(similar_products_csv_path):
            print(f"Error: Similar products file not found at '{similar_products_csv_path}', but --filter-variants was specified."); sys.exit(1)
        effective_similar_products_path = similar_products_csv_path

    # Validate catalog path existence *if* evaluation is NOT skipped
    if not args.skip_evaluation and not os.path.exists(catalog_csv_path):
         print(f"Warning: Catalog file not found at '{catalog_csv_path}'. Coverage metric will be skipped during evaluation.")
         catalog_csv_path = None # Set to None so evaluation handles it gracefully
    elif args.skip_evaluation:
        catalog_csv_path = None # Don't need it if skipping eval

    # --- Setup Output Directory ---
    ensure_dir(output_dir_path)
    print(f"Output directory ensured: {output_dir_path}")

    # --- Load Data ---
    print("\n--- Loading Data ---")
    features_df_full = load_dataframe(input_features_path, required_cols=['product_id'])
    if features_df_full.empty: print("Failed to load features DataFrame. Exiting."); sys.exit(1)

    product_ids = features_df_full['product_id'].copy()
    # Extract feature columns robustly - select only numeric types AFTER loading
    feature_cols_df = features_df_full.select_dtypes(include=np.number).copy()
    if 'product_id' in feature_cols_df.columns: # Drop ID if it was numeric somehow
        feature_cols_df = feature_cols_df.drop(columns=['product_id'])
    print(f"Extracted {len(feature_cols_df.columns)} potential numeric feature columns.")

    # Keep original df for titles if available
    products_df_for_titles = None
    if 'product_title' in features_df_full.columns and 'product_id' in features_df_full.columns:
        products_df_for_titles = features_df_full[['product_id', 'product_title']].copy()

    # Load optional data
    similar_products_df = load_dataframe(effective_similar_products_path) if args.filter_variants and effective_similar_products_path else None
    base_weights = load_json_config(args.weights_config) if args.weights_config else None
    optimization_params = load_json_config(args.optimization_config) if args.optimization_config else None

    # --- Run KNN Training Pipeline ---
    print("\n--- Starting KNN Training Pipeline ---")
    final_recommendations = None
    final_model = None
    try:
        final_recommendations, final_model = run_knn_training(
            features_df=feature_cols_df,
            product_ids=product_ids,
            output_dir=output_dir_path,
            n_neighbors=args.neighbors,
            base_weights=base_weights,
            optimize_weights=args.optimize_weights,
            optimization_method=args.optimization_method,
            optimization_params=optimization_params,
            filter_variants=args.filter_variants,
            similar_products_df=similar_products_df,
            products_df_for_titles=products_df_for_titles,
            catalog_csv_path=catalog_csv_path, 
            skip_evaluation=args.skip_evaluation,
            save_model=args.save_model
        )

        if final_model is not None:
            print("\n--- KNN Training Pipeline Completed Successfully ---")
            if final_recommendations is not None and not final_recommendations.empty:
                 print(f"Final recommendations generated ({len(final_recommendations)} pairs). Check '{output_dir_path}' for details.")
            else:
                 print("Pipeline completed, but no final recommendations were generated (check filters or logs).")
        else:
            print("\n--- KNN Training Pipeline Finished with Errors (No model returned) ---")
            sys.exit(1)

    except Exception as e:
        print(f"\n--- Critical Error during KNN pipeline execution: {type(e).__name__} - {str(e)} ---")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
