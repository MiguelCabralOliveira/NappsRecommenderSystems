# training/similarity/train_knn_model.py

import argparse
import os
import sys
import traceback
import pandas as pd

# --- Ajustar o sys.path para permitir imports relativos corretos ---
# Adiciona o diretório pai (training) ao sys.path para encontrar 'knn'
# Isso é necessário quando executamos como script principal
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(current_dir) # Vai para training/
project_root = os.path.dirname(training_dir) # Vai para NappsRecommender/ (assumindo estrutura)

# Adicionar project_root ao path se não estiver lá pode ajudar com outros imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Adicionar training_dir ao path para encontrar o módulo 'similarity'
if training_dir not in sys.path:
     sys.path.insert(0, training_dir)

# --- Imports do Módulo KNN ---
try:
    from training.similarity.knn.knn_pipeline import run_knn_training
    from .utils import load_dataframe, load_json_config, ensure_dir
except ImportError as e:
    print(f"Error importing KNN modules: {e}")
    print("Ensure the script is run correctly relative to the project structure,")
    print("or check the __init__.py files in training/similarity/ and training/similarity/knn/")
    # Tenta um import absoluto como fallback (pode funcionar se o PYTHONPATH estiver configurado)
    try:
         print("Attempting absolute import fallback...")
         from training.similarity.knn.knn_pipeline import run_knn_training
         from utils import load_dataframe, load_json_config, ensure_dir
    except ImportError:
         print("Absolute import fallback also failed. Exiting.")
         sys.exit(1)

def main():
    """
    Main script to run the KNN training pipeline from the command line.
    Parses arguments and calls the core training function.
    """
    parser = argparse.ArgumentParser(
        description='Train a Weighted KNN model for product recommendations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Melhora a ajuda
    )

    # --- Input/Output Arguments ---
    parser.add_argument('--input-features', required=True,
                        help='Path to the input CSV file containing processed product features (e.g., products_final_all_features.csv).')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save all training outputs (model, recommendations, reports, plots). Example: results/<shop_id>/training_knn/')
    parser.add_argument('--shop-id', required=True,
                        help='Shop identifier (used for context/logging).')

    # --- KNN Core Parameters ---
    parser.add_argument('--neighbors', type=int, default=20,
                        help='Number of neighbors (recommendations) to generate per product.')

    # --- Weighting Configuration ---
    parser.add_argument('--weights-config', type=str, default=None,
                        help='Optional path to a JSON file defining base weights for feature groups.')
    parser.add_argument('--optimize-weights', action='store_true',
                        help='If set, run feature weight optimization.')
    parser.add_argument('--optimization-method', type=str, default='evolutionary',
                        choices=['grid', 'random', 'evolutionary'],
                        help='Method for weight optimization (if --optimize-weights is set).')
    parser.add_argument('--optimization-config', type=str, default=None,
                        help='Optional path to a JSON file with parameters for the chosen optimization method.')

    # --- Filtering ---
    parser.add_argument('--filter-variants', action='store_true',
                        help='If set, filter out recommendations identified as similar variants.')
    parser.add_argument('--similar-products-csv', type=str, default=None,
                        help='Path to the CSV file containing pairs of similar products (e.g., from description TF-IDF). Required if --filter-variants is set.')

    # --- Execution Control ---
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='If set, skip the evaluation step after generating recommendations.')
    parser.add_argument('--save-model', action='store_true', default=True, # Default to saving
                        help='Save the trained KNN model components.')
    parser.add_argument('--no-save-model', action='store_false', dest='save_model',
                        help='Do not save the trained KNN model components.')

    # --- Parsing ---
    args = parser.parse_args()

    # --- Basic Input Validation ---
    if not os.path.exists(args.input_features):
        print(f"Error: Input features file not found: {args.input_features}")
        sys.exit(1)

    if args.filter_variants and not args.similar_products_csv:
        print("Error: --filter-variants requires --similar-products-csv to be specified.")
        sys.exit(1)
    if args.filter_variants and args.similar_products_csv and not os.path.exists(args.similar_products_csv):
        print(f"Error: Similar products file not found: {args.similar_products_csv}")
        sys.exit(1)

    # --- Setup Output Directory ---
    ensure_dir(args.output_dir)
    print(f"Output directory set to: {args.output_dir}")

    # --- Load Data ---
    print("\n--- Loading Data ---")
    features_df = load_dataframe(args.input_features, required_cols=['product_id']) # Ensure product_id exists
    if features_df.empty:
        print("Failed to load or validate features DataFrame. Exiting.")
        sys.exit(1)

    # Separate IDs and features
    product_ids = features_df['product_id'].copy()
    # Assume all other columns *should* be numeric features after preprocessing
    # Let the core KNN class handle detailed numeric checks if needed
    feature_cols_df = features_df.drop(columns=['product_id'], errors='ignore')
    # Keep original df for titles if available
    products_df_for_titles = features_df[['product_id', 'product_title']].copy() if 'product_title' in features_df.columns else None

    # Load optional data
    similar_products_df = load_dataframe(args.similar_products_csv) if args.filter_variants else None
    base_weights = load_json_config(args.weights_config) if args.weights_config else None
    optimization_params = load_json_config(args.optimization_config) if args.optimization_config else None

    # --- Run KNN Training Pipeline ---
    print("\n--- Starting KNN Training Pipeline ---")
    final_recommendations = None
    final_model = None
    try:
        final_recommendations, final_model = run_knn_training(
            features_df=feature_cols_df, # Pass only feature columns
            product_ids=product_ids,     # Pass product IDs separately
            output_dir=args.output_dir,
            n_neighbors=args.neighbors,
            base_weights=base_weights,
            optimize_weights=args.optimize_weights,
            optimization_method=args.optimization_method,
            optimization_params=optimization_params,
            filter_variants=args.filter_variants,
            similar_products_df=similar_products_df,
            products_df_for_titles=products_df_for_titles, # Pass for adding titles in export
            skip_evaluation=args.skip_evaluation,
            save_model=args.save_model
        )

        if final_model is not None:
            print("\n--- KNN Training Pipeline Completed Successfully ---")
            if final_recommendations is not None and not final_recommendations.empty:
                 print(f"Final recommendations generated ({len(final_recommendations)} pairs). Check '{args.output_dir}' for details.")
            else:
                 print("Pipeline completed, but no final recommendations were generated (check filters or logs).")
        else:
            print("\n--- KNN Training Pipeline Finished with Errors (No model returned) ---")
            sys.exit(1) # Exit with error if model failed

    except Exception as e:
        print(f"\n--- Critical Error during KNN pipeline execution: {type(e).__name__} - {str(e)} ---")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()