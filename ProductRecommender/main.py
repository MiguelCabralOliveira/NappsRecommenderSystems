# main.py
# ... (imports e código anterior permanecem os mesmos) ...
import pandas as pd
import argparse
import os
import traceback # Import traceback for better error reporting

# --- Local Imports ---
# (Assume all previous imports are correct)
from shopifyInfo.shopify_queries import get_all_products
from shopifyInfo.database_query import fetch_recent_order_items
from preprocessing.metafields_processor import process_metafields, apply_tfidf_processing, save_color_similarity_data, remove_color_columns, save_product_references
from preprocessing.isGiftCard_processor import process_gif_card
from preprocessing.availableForSale_processor import process_avaiable_for_sale
from preprocessing.product_type_processor import process_product_type
from preprocessing.tags_processor import process_tags
from preprocessing.collectionHandle_processor import process_collections
from preprocessing.vendor_processor import process_vendors
from preprocessing.variants_processor import process_variants
from preprocessing.createdAt_processor import process_created_at
from preprocessing.title_processor import process_titles_tfidf
from preprocessing.handle_processor import process_handles_tfidf
from preprocessing.tfidf_processor import process_descriptions_tfidf
from training.weighted_knn import run_knn
from training.recommendation_combiner import generate_hybrid_recommendations_adaptative, save_hybrid_recommendations
from training.evaluation import run_evaluation

# --- Helper Functions (export_sample, document_feature_sources) ---
# (Keep these functions as they were)
def export_sample(df, step_name, output_dir="results"):
    """Export the first row of the DataFrame to a CSV for debugging"""
    if df is not None and not df.empty: # Check if df is not None and not empty
        sample_df = df.head(1)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"sample_after_{step_name}.csv") # Use os.path.join
        try:
            sample_df.to_csv(filename, index=False)
            # print(f"Exported sample to {filename}") # Optional: reduce verbosity
        except Exception as e:
            print(f"Error exporting sample for {step_name}: {e}")

def document_feature_sources(df, output_dir="results", output_file="feature_sources.csv"):
    """
    Document the sources of all features in the processed DataFrame
    """
    if df is None or df.empty:
        print("Cannot document feature sources: DataFrame is empty or None.")
        return None
    feature_sources = []
    for column in df.columns: # (Logic for source mapping remains the same)
        if column.startswith('description_'): source = 'TF-IDF from product descriptions'
        elif column.startswith('title_'): source = 'TF-IDF from product titles'
        elif column.startswith('handle_'): source = 'TF-IDF from product handles'
        elif column.startswith('metafield_'): source = 'TF-IDF/Data from product metafields'
        elif column.startswith('tag_'): source = 'Tag dummy variable'
        elif column.startswith('product_type_'): source = 'Product type dummy variable'
        elif column.startswith('collections_'): source = 'Collection dummy variable'
        elif column.startswith('vendor_'): source = 'Vendor dummy variable'
        elif column.startswith('variant_'): source = 'Variant option dummy variable'
        elif column.startswith('price_') or column in ['min_price', 'max_price', 'has_discount']: source = 'Price-related feature from variants'
        elif column == 'product_id' or column == 'product_title' or column == 'handle': source = 'Original product information'
        elif column.startswith('created_') or column in ['days_since_first_product', 'is_recent_product']: source = 'Time-based features from createdAt timestamp'
        elif column.startswith('release_quarter_'): source = 'Release quarter features from createdAt timestamp'
        elif column == 'is_gift_card': source = 'Gift Card flag'
        elif column == 'available_for_sale': source = 'Availability flag'
        elif pd.api.types.is_numeric_dtype(df[column]): source = 'Other Numeric/Processed Feature'
        else: source = 'Other/Categorical Feature'
        feature_sources.append({'feature_name': column, 'source': source})
    sources_df = pd.DataFrame(feature_sources)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_file)
    try:
        sources_df.to_csv(full_path, index=False); print(f"Feature sources documented in {full_path}")
    except Exception as e: print(f"Error saving feature sources to {full_path}: {e}"); return None
    return sources_df


def main():
    """Main function to process product data and generate recommendations"""
    # --- Argument Parser Setup ---
    # (Parser setup remains the same - includes KNN, Optimization, Hybrid args)
    parser = argparse.ArgumentParser(description='Process product data and generate recommendations')
    parser.add_argument('--neighbors', type=int, default=20, help='Number of neighbors for KNN and K for evaluation (default: 20)') # Default K=20
    parser.add_argument('--skip-recommendations', action='store_true', help='Skip recommendation generation and evaluation')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation step')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results (default: results)')
    parser.add_argument('--optimize-weights', action='store_true', help='Optimize feature weights')
    parser.add_argument('--optimization-method', type=str, default='evolutionary', choices=['grid', 'random', 'evolutionary'], help='Method for weight optimization (default: evolutionary)')
    parser.add_argument('--opt-iterations', type=int, default=50, help='Iterations for random search optimization (default: 50)')
    parser.add_argument('--population-size', type=int, default=30, help='Population size for evolutionary optimization (default: 30)')
    parser.add_argument('--generations', type=int, default=15, help='Generations for evolutionary optimization (default: 15)')
    parser.add_argument('--strong-sim', type=float, default=0.3, help='KNN similarity threshold for strong signal (default: 0.3)')
    parser.add_argument('--high-sim', type=float, default=0.95, help='KNN similarity threshold to filter out very high similarity (default: 0.95)')
    parser.add_argument('--min-sim', type=float, default=0.01, help='Minimum KNN similarity to include a recommendation (default: 0.01)')
    parser.add_argument('--pop-boost', type=float, default=0.05, help='Boost factor for popularity in hybrid score (default: 0.05)')
    parser.add_argument('--group-boost', type=float, default=0.03, help='Boost factor for group frequency in hybrid score (default: 0.03)')
    args = parser.parse_args()

    # --- Setup Output Directory ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Get Shop ID ---
    shop_id = input("Shop ID: ")

    # --- Fetch Order Data (Popularity, Groups) ---
    print("\n=== Fetching recent order data from database ===")
    try:
        order_items_df, popular_products_df, product_groups_df = fetch_recent_order_items(shop_id)
    except Exception as e:
        print(f"Error fetching order data: {e}")
        traceback.print_exc()
        print("Warning: Continuing without order data. Hybrid/Popularity metrics may be affected.")
        order_items_df, popular_products_df, product_groups_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Save fetched order data
    try:
        pop_output_path = os.path.join(output_dir, "popular_products.csv")
        groups_output_path = os.path.join(output_dir, "product_groups.csv")
        if order_items_df is not None and not order_items_df.empty: order_items_df.to_csv(os.path.join(output_dir, "recent_order_items.csv"), index=False); print(f"Recent order items saved.")
        if popular_products_df is not None and not popular_products_df.empty: popular_products_df.to_csv(pop_output_path, index=False); print(f"Popular products saved.")
        if product_groups_df is not None and not product_groups_df.empty: product_groups_df.to_csv(groups_output_path, index=False); print(f"Product groups saved.")
    except Exception as e: print(f"Error saving fetched order data: {e}")


    # --- Fetch Product Data (Shopify) ---
    print("\n=== Fetching product data from Shopify ===")
    try: raw_data = get_all_products(shop_id)
    except Exception as e: print(f"Critical Error fetching product data: {e}"); traceback.print_exc(); raw_data = None
    if raw_data is None or raw_data.empty: print("Failed to retrieve data from Shopify. Cannot proceed."); return
    raw_data_path = os.path.join(output_dir, "products_raw.csv")
    try: raw_data.to_csv(raw_data_path, index=False); print(f"Raw product data saved to {raw_data_path}")
    except Exception as e: print(f"Error saving raw product data: {e}")
    export_sample(raw_data, "raw_data", output_dir)


    # --- Data Processing Pipeline ---
    print("\n=== Starting data processing pipeline ===")
    df = raw_data.copy()
    all_steps_successful = True
    df_final_features = None
    similar_products = pd.DataFrame()
    color_similarity_df = pd.DataFrame()
    tfidf_output_prefix = os.path.join(output_dir, "products") # Define prefix once
    try:
        # Step 1: Metafields
        print("Processing metafields..."); all_processed_data = []; all_product_references = {}
        if 'metafields_config' not in df.columns: df['metafields_config'] = [[] for _ in range(len(df))]
        for index, row in df.iterrows():
            try:
                processed_data, product_refs = process_metafields(row, row.get('metafields_config', []))
                all_processed_data.append(processed_data)
                product_id = row.get('product_id', f'index_{index}')
                if product_id and product_refs:
                    clean_product_id = str(product_id).split('/')[-1] if isinstance(product_id, str) and 'gid://shopify/Product/' in product_id else product_id
                    for key, value in product_refs.items(): all_product_references[f"{clean_product_id}_{key}"] = value
            except Exception as e: print(f"Error processing metafields row {index}: {e}"); all_processed_data.append({})
        metafields_processed = pd.DataFrame(all_processed_data, index=df.index)
        if all_product_references: save_product_references(all_product_references, output_dir, "product_references.csv"); print(f"Product references saved.")
        print("Calculating color similarities..."); color_similarity_df = save_color_similarity_data(output_dir=output_dir)
        print("Removing color columns..."); metafields_processed = remove_color_columns(metafields_processed)
        print("Applying TF-IDF to metafields..."); metafields_processed = apply_tfidf_processing(metafields_processed)
        if 'metafields' in df.columns: df = df.drop('metafields', axis=1)
        if 'metafields_config' in df.columns: df = df.drop('metafields_config', axis=1)
        if not df.index.equals(metafields_processed.index):
             metafields_processed = metafields_processed.reset_index(drop=True); df = df.reset_index(drop=True)
        df = pd.concat([df, metafields_processed], axis=1)
        export_sample(df, "metafields", output_dir)

        # Steps 2-9: Other processors
        print("Processing gift_card..."); df = process_gif_card(df); export_sample(df, "gift_card", output_dir)
        print("Processing available_for_sale..."); df = process_avaiable_for_sale(df); export_sample(df, "available_for_sale", output_dir)
        print("Processing product types..."); df = process_product_type(df); export_sample(df, "product_type", output_dir)
        print("Processing tags..."); df = process_tags(df); export_sample(df, "tags", output_dir)
        print("Processing collections..."); df = process_collections(df); export_sample(df, "collections", output_dir)
        print("Processing vendors..."); df = process_vendors(df); export_sample(df, "vendors", output_dir)
        print("Processing variants..."); df = process_variants(df); export_sample(df, "variants", output_dir)
        print("Processing createdAt..."); df = process_created_at(df); export_sample(df, "created_at", output_dir)

        # Step 10: Title TF-IDF
        print("Processing product titles with TF-IDF..."); df, _ = process_titles_tfidf(df, output_prefix=tfidf_output_prefix); export_sample(df, "title_tfidf", output_dir)
        # Step 11: Handle TF-IDF
        print("Processing product handles with TF-IDF..."); df, _ = process_handles_tfidf(df, output_prefix=tfidf_output_prefix); export_sample(df, "handle_tfidf", output_dir)

        # Save intermediate state
        intermediate_file = os.path.join(output_dir, "products_features_before_desc_tfidf.csv")
        try: df.to_csv(intermediate_file, index=False); print(f"Intermediate features saved.")
        except Exception as e: print(f"Error saving intermediate features: {e}")

        # Step 12: Description TF-IDF (Final Feature Step)
        print("Processing product descriptions with TF-IDF...")
        df_final_features, _, _, similar_products = process_descriptions_tfidf(df, output_prefix=tfidf_output_prefix)

        # Validation & Final Saves
        if df_final_features is None or df_final_features.empty: print("Error: Description TF-IDF failed."); all_steps_successful = False
        else:
            print("Description TF-IDF processing complete.")
            export_sample(df_final_features, "final_features_incl_desc", output_dir)
            final_features_file = os.path.join(output_dir, "products_final_all_features.csv")
            try: df_final_features.to_csv(final_features_file, index=False); print(f"Final features DataFrame saved.")
            except Exception as e: print(f"Error saving final features: {e}")
            document_feature_sources(df_final_features, output_dir, "feature_sources.csv")
        if similar_products is None or similar_products.empty: print("Warning: No similar products generated from descriptions."); similar_products = pd.DataFrame()
        else:
             # Save similar products if generated
             similar_products_path = os.path.join(output_dir, "similar_products.csv")
             try: similar_products.to_csv(similar_products_path, index=False); print(f"Similar products saved.")
             except Exception as e: print(f"Error saving similar products: {e}")


    except Exception as e:
        print(f"\n--- Critical Error during data processing pipeline: {e} ---")
        traceback.print_exc(); all_steps_successful = False

    # --- Post-Processing Summary ---
    if all_steps_successful and df_final_features is not None and not df_final_features.empty:
        print("\n=== Data processing pipeline completed successfully! ===")
        print(f"Key files saved in '{output_dir}':")
        print(f"  - products_raw.csv")
        print(f"  - products_final_all_features.csv")
        print(f"  - feature_sources.csv")
        if similar_products is not None and not similar_products.empty: print(f"  - similar_products.csv")
        if popular_products_df is not None and not popular_products_df.empty: print(f"  - popular_products.csv")
        if product_groups_df is not None and not product_groups_df.empty: print(f"  - product_groups.csv")
    else:
        print("\n=== Data processing pipeline finished with errors or no final features. ===")
        print("Skipping recommendation generation and evaluation due to processing issues.")
        return

    # --- Recommendation Generation (KNN) ---
    knn_recommendations_df = None # Initialize
    hybrid_recommendations = None # Initialize hybrid results
    final_knn_model = None
    # Define output file paths consistently
    knn_output_file = os.path.join(output_dir, "all_recommendations.csv")
    hybrid_output_file = os.path.join(output_dir, "hybrid_recommendations.csv")
    knn_pre_filter_output_file = os.path.join(output_dir, "knn_recommendations_pre_variant_filter.csv")
    
    final_recommendations_path = None


    if not args.skip_recommendations:
        print("\n" + "="*80); print("Starting Weighted KNN recommendation generation..."); print("="*80)
        optimization_params = None
        if args.optimize_weights:
            print(f"Weight optimization enabled ({args.optimization_method})...")
            if args.optimization_method == 'random': optimization_params = {'n_iterations': args.opt_iterations, 'weight_range': (0.1, 5.0)}
            elif args.optimization_method == 'evolutionary': optimization_params = {'population_size': args.population_size, 'generations': args.generations, 'mutation_rate': 0.2, 'weight_range': (0.1, 5.0)}
            elif args.optimization_method == 'grid': optimization_params = {}; print("Using default grid search params.")

        if df_final_features is None or df_final_features.empty: print("Error: Cannot run KNN, final features missing.")
        else:
            try:
                
                knn_recommendations_df, final_knn_model = run_knn( 
                    df=df_final_features.copy(), 
                    output_dir=output_dir,
                    n_neighbors=args.neighbors,
                    save_model=True,
                    similar_products_df=similar_products if similar_products is not None and not similar_products.empty else None,
                    optimize_weights=args.optimize_weights,
                    optimization_method=args.optimization_method,
                    optimization_params=optimization_params
                )
                
                if knn_recommendations_df is not None and not knn_recommendations_df.empty:
                    print(f"\nKNN generation complete. Filtered results saved to {knn_output_file}")
                    final_recommendations_path = knn_output_file 
                    
                elif knn_recommendations_df is not None: print("\nKNN process completed, but no filtered recommendations generated.")
                else: print("\nKNN process failed.")
            except Exception as e: print(f"\n--- Critical Error during KNN: {e} ---"); traceback.print_exc(); knn_recommendations_df = None; final_knn_model = None

        
        if final_knn_model is not None and os.path.exists(knn_pre_filter_output_file):
            print("\n" + "="*80); print("Combining KNN recommendations with Popularity/Group data (Hybrid)..."); print("="*80)
            try:
                
                knn_input_for_hybrid = pd.read_csv(knn_pre_filter_output_file)

                pop_df_path = os.path.join(output_dir, "popular_products.csv")
                groups_df_path = os.path.join(output_dir, "product_groups.csv")
                pop_df_input = pd.read_csv(pop_df_path) if os.path.exists(pop_df_path) else None
                groups_df_input = pd.read_csv(groups_df_path) if os.path.exists(groups_df_path) else None

                hybrid_recommendations = generate_hybrid_recommendations_adaptative(
                    all_knn_recommendations=knn_input_for_hybrid, 
                    products_df=df_final_features,
                    popular_products_df=pop_df_input,
                    product_groups_df=groups_df_input,
                    total_recommendations=args.neighbors,
                    strong_similarity_threshold=args.strong_sim, high_similarity_threshold=args.high_sim,
                    min_similarity_threshold=args.min_sim, pop_boost_factor=args.pop_boost,
                    group_boost_factor=args.group_boost
                )
                if hybrid_recommendations is not None and not hybrid_recommendations.empty:
                    save_hybrid_recommendations(hybrid_recommendations, hybrid_output_file)
                    print(f"Hybrid combination complete. Results saved to {hybrid_output_file}")
                    final_recommendations_path = hybrid_output_file 
                elif hybrid_recommendations is not None: print("Hybrid combination completed, but no final recommendations generated.")
                else: print("Hybrid combination process failed.")
            except Exception as e: print(f"\n--- Critical Error during Hybrid combination: {e} ---"); traceback.print_exc()
        elif final_knn_model is None:
             print("\nSkipping Hybrid combination because KNN model fitting failed.")
        else: # KNN model fitted, but pre-filter file missing (shouldn't happen normally)
             print("\nSkipping Hybrid combination: Pre-filtered KNN file is missing.")


    else: 
         print("\nSkipping Recommendation Generation as requested by --skip-recommendations flag.")


    # --- Evaluation Step ---
    if not args.skip_recommendations and not args.skip_evaluation:
        print("\n" + "="*80); print("Starting Model Evaluation"); print("="*80)

        
        catalog_file = raw_data_path
        pop_file = os.path.join(output_dir, "popular_products.csv")
        
        
        knn_sims_file_for_ils = knn_pre_filter_output_file 


        pop_file_path = pop_file if os.path.exists(pop_file) else None
        knn_sims_file_path_for_ils = knn_sims_file_for_ils if os.path.exists(knn_sims_file_for_ils) else None

        # Evaluation output directory
        eval_output_dir = os.path.join(output_dir, "evaluation_metrics")
        os.makedirs(eval_output_dir, exist_ok=True)

        # --- Evaluate KNN PRE-FILTER Recommendations --- # 
        if os.path.exists(knn_pre_filter_output_file):
            print(f"\n--- Evaluating KNN PRE-FILTER Recommendations ({os.path.basename(knn_pre_filter_output_file)}) ---")
            try:
                run_evaluation(
                    recs_path=knn_pre_filter_output_file, # Evaluate the pre-filter file
                    catalog_path=catalog_file,
                    k=args.neighbors,
                    output_dir=eval_output_dir,
                    pop_path=pop_file_path,
                    knn_sims_path=knn_sims_file_path_for_ils, # Use pre-filter sims for ILS calculation
                    results_filename=f"evaluation_k{args.neighbors}_knn_pre_filter.json" # Descriptive filename
                )
            except Exception as e:
                print(f"\n--- Error during KNN PRE-FILTER Evaluation: {e} ---")
                traceback.print_exc()
        else:
            print("\nSkipping KNN PRE-FILTER Evaluation: File not found.")
        # --- END NEW EVALUATION BLOCK ---

        # --- Evaluate FINAL Recommendations (Hybrid or Filtered KNN if Hybrid failed/skipped) ---
        if final_recommendations_path and os.path.exists(final_recommendations_path):
            # Check if the final file is different from the pre-filter file we just evaluated
            if final_recommendations_path != knn_pre_filter_output_file:
                print(f"\n--- Evaluating FINAL Recommendations ({os.path.basename(final_recommendations_path)}) ---")
                try:
                    run_evaluation(
                        recs_path=final_recommendations_path,
                        catalog_path=catalog_file,
                        k=args.neighbors, # Use the neighbors value for K
                        output_dir=eval_output_dir,
                        pop_path=pop_file_path,
                        knn_sims_path=knn_sims_file_path_for_ils, 
                        results_filename=f"evaluation_k{args.neighbors}_final_{os.path.basename(final_recommendations_path).replace('.csv','')}.json" 
                    )
                except Exception as e:
                    print(f"\n--- Error during FINAL Evaluation: {e} ---")
                    traceback.print_exc()
            else:
                 print(f"\nSkipping FINAL evaluation as it's the same as the pre-filter file ({os.path.basename(final_recommendations_path)}).")

        else:
            print("\nSkipping FINAL Evaluation: No valid final recommendation file found or defined.")

        # --- Evaluate FILTERED KNN Recommendations (if different from final AND pre-filter) ---
        # This might be redundant if hybrid failed and final_recommendations_path is already knn_output_file
        if os.path.exists(knn_output_file) \
           and knn_output_file != final_recommendations_path \
           and knn_output_file != knn_pre_filter_output_file:
            print(f"\n--- Evaluating FILTERED KNN Recommendations ({os.path.basename(knn_output_file)}) ---")
            try:
                run_evaluation(
                    recs_path=knn_output_file, # Evaluate the filtered KNN file
                    catalog_path=catalog_file,
                    k=args.neighbors,
                    output_dir=eval_output_dir,
                    pop_path=pop_file_path,
                    knn_sims_path=knn_sims_file_path_for_ils, # STILL use pre-filter KNN sims for ILS
                    results_filename=f"evaluation_k{args.neighbors}_knn_filtered.json" # Specific filename
                )
            except Exception as e:
                print(f"\n--- Error during FILTERED KNN Evaluation: {e} ---")
                traceback.print_exc()
        elif os.path.exists(knn_output_file):
            if knn_output_file == final_recommendations_path:
                 print(f"\nSkipping separate FILTERED KNN evaluation as it's the same as the final evaluated file ({os.path.basename(knn_output_file)}).")
            elif knn_output_file == knn_pre_filter_output_file:
                 print(f"\nSkipping separate FILTERED KNN evaluation as it's the same as the pre-filter evaluated file ({os.path.basename(knn_output_file)}).")

        elif not os.path.exists(knn_output_file):
             print("\nSkipping FILTERED KNN evaluation as the file was not found.")


    elif args.skip_evaluation:
         print("\nSkipping Evaluation as requested by --skip-evaluation flag.")
    elif args.skip_recommendations:
         print("\nSkipping Evaluation because recommendations were skipped.")


    print("\n=== Full Pipeline Finished ===")


if __name__ == "__main__":
    main()