# main.py

# ... (imports) ...
import pandas as pd
import argparse
import os
# ... (outros imports) ...
from shopifyInfo.shopify_queries import get_all_products
from shopifyInfo.database_query import fetch_recent_order_items
from preprocessing.tfidf_processor import process_descriptions_tfidf
from preprocessing.collectionHandle_processor import process_collections
from preprocessing.tags_processor import process_tags
from preprocessing.variants_processor import process_variants
from preprocessing.vendor_processor import process_vendors
from preprocessing.metafields_processor import process_metafields, apply_tfidf_processing, save_color_similarity_data, remove_color_columns
from preprocessing.isGiftCard_processor import process_gif_card
from preprocessing.product_type_processor import process_product_type
from preprocessing.createdAt_processor import process_created_at
from preprocessing.availableForSale_processor import process_avaiable_for_sale
from training.weighted_knn import run_knn
from training.recommendation_combiner import generate_hybrid_recommendations_adaptative, save_hybrid_recommendations
import traceback # Import traceback for better error reporting

# ... (funções export_sample, document_feature_sources) ...
def export_sample(df, step_name, output_dir="results"):
    """Export the first row of the DataFrame to a CSV for debugging"""
    if df is not None and len(df) > 0: # Adicionada verificação df is not None
        sample_df = df.head(1)
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/sample_after_{step_name}.csv"
        try:
            sample_df.to_csv(filename, index=False)
            print(f"Exported sample to {filename}")
        except Exception as e:
            print(f"Error exporting sample for {step_name}: {e}")

def document_feature_sources(df, output_dir="results", output_file="feature_sources.csv"):
    """
    Document the sources of all features in the processed DataFrame
    """
    if df is None or df.empty:
        print("Cannot document feature sources: DataFrame is empty or None.")
        return None # Retornar None se o DataFrame for inválido

    feature_sources = []

    # Categorize columns by their prefixes
    for column in df.columns:
        if column.startswith('description_'):
            source = 'TF-IDF from product descriptions'
        elif column.startswith('metafield_'):
            source = 'TF-IDF from product metafields'
        elif column.startswith('tag_'):
            source = 'Tag dummy variable'
        elif column.startswith('product_type_'):
            source = 'Product type dummy variable'
        elif column.startswith('collections_'):
            source = 'Collection dummy variable'
        elif column.startswith('vendor_'):
            source = 'Vendor dummy variable'
        elif column.startswith('variant_'):
            source = 'Variant option dummy variable'
        elif column.startswith('price_') or column in ['min_price', 'max_price', 'has_discount']: # Adicionado price_ e outros comuns
            source = 'Price-related feature from variants'
        elif column == 'product_id' or column == 'product_title' or column == 'handle':
            source = 'Original product information'
        elif column.startswith('created_') or column == 'days_since_first_product' or column == 'is_recent_product':
            source = 'Time-based features from createdAt timestamp'
        elif column.startswith('release_quarter_'):
            source = 'Release quarter features from createdAt timestamp'
        elif column == 'is_gift_card':
             source = 'Gift Card flag'
        elif column == 'available_for_sale':
             source = 'Availability flag'
        else:
            source = 'Other/Processed Feature' # Mais genérico

        feature_sources.append({
            'feature_name': column,
            'source': source
        })

    # Create DataFrame and save to CSV
    sources_df = pd.DataFrame(feature_sources)
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_file) # Usar os.path.join
    try:
        sources_df.to_csv(full_path, index=False)
        print(f"Feature sources documented in {full_path}")
    except Exception as e:
        print(f"Error saving feature sources to {full_path}: {e}")
        return None # Retornar None em caso de erro

    return sources_df


def main():
    """Main function to process product data from Shopify store"""
    # --- Bloco argparse ---
    parser = argparse.ArgumentParser(description='Process product data and generate recommendations')
    parser.add_argument('--neighbors', type=int, default=12,
                      help='Number of neighbors for KNN recommendations (default: 12)')
    parser.add_argument('--skip-recommendations', action='store_true',
                      help='Skip the recommendation generation step')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save recommendation results (default: results)')
    parser.add_argument('--optimize-weights', action='store_true',
                      help='Optimize feature weights for better recommendations')
    parser.add_argument('--optimization-method', type=str, default='grid', choices=['grid', 'random', 'evolutionary'],
                      help='Method to use for weight optimization (default: grid)')
    parser.add_argument('--opt-iterations', type=int, default=50,
                      help='Number of iterations for random search optimization (default: 50)')
    parser.add_argument('--population-size', type=int, default=10,
                      help='Population size for evolutionary optimization (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                      help='Number of generations for evolutionary optimization (default: 5)')

    # --- Argumentos para recommendation_combiner ---
    parser.add_argument('--strong-sim', type=float, default=0.3,
                        help='KNN similarity threshold to define strong signal (default: 0.3)')
    parser.add_argument('--high-sim', type=float, default=0.9,
                        help='KNN similarity threshold to filter out very high similarity (default: 0.9)')
    parser.add_argument('--min-sim', type=float, default=0.05, # <<< DEFINIDO AQUI
                        help='Minimum KNN similarity to include a recommendation (default: 0.05)')
    parser.add_argument('--pop-boost', type=float, default=0.05, # <<< DEFINIDO AQUI
                        help='Boost factor for popularity in hybrid score (default: 0.05)')
    parser.add_argument('--group-boost', type=float, default=0.03, # <<< DEFINIDO AQUI
                        help='Boost factor for group frequency in hybrid score (default: 0.03)')
    # --- FIM DOS NOVOS ARGUMENTOS ---

    args = parser.parse_args() # Parse DEPOIS de definir TODOS os argumentos

    # Create output directory if it doesn't exist
    output_dir = args.output_dir # Usar variável para consistência
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get shop ID from user input
    shop_id = input("Shop ID: ")


    # Fetch order data from database
    print("\n=== Fetching recent order data from database ===")
    order_items_df, popular_products_df, product_groups_df = fetch_recent_order_items(shop_id)


    # Save order data to CSV files
    try:
        if order_items_df is not None and not order_items_df.empty:
            order_items_df.to_csv(os.path.join(output_dir, "recent_order_items.csv"), index=False)
            print(f"Recent order items saved to {output_dir}/recent_order_items.csv")

        if popular_products_df is not None and not popular_products_df.empty:
            popular_products_df.to_csv(os.path.join(output_dir, "popular_products.csv"), index=False)
            print(f"Popular products saved to {output_dir}/popular_products.csv")

        if product_groups_df is not None and not product_groups_df.empty:
            product_groups_df.to_csv(os.path.join(output_dir, "product_groups.csv"), index=False)
            print(f"Product groups saved to {output_dir}/product_groups.csv")
    except Exception as e:
        print(f"Error saving order data: {e}")

    print("\n=== Fetching product data from Shopify ===")
    raw_data = get_all_products(shop_id)

    if raw_data is None or raw_data.empty: # Adicionada verificação de vazio
        print("Failed to retrieve data from Shopify or data is empty.")
        return # Terminar se não houver dados

    # Save raw data immediately after importing
    try:
        raw_data.to_csv(os.path.join(output_dir, "products.csv"), index=False)
        print(f"Raw data saved to {output_dir}/products.csv")
    except Exception as e:
         print(f"Error saving raw data: {e}")

    # Export raw data sample
    export_sample(raw_data, "raw_data", output_dir)

    # Start with the raw data for processing
    df = raw_data.copy()

    # --- Data processing pipeline ---
    print("\n=== Starting data processing pipeline ===")
    all_steps_successful = True
    df_final_features = None # Initialize
    similar_products = pd.DataFrame() # Initialize as empty

    try:
        # ... (Passos 1 a 8 do pipeline - metafields, gift_card, etc.) ...
        # 1. Process metafields
        print("Processing metafields...")
        all_processed_data = []
        all_product_references = {}
        if 'metafields_config' not in df.columns: df['metafields_config'] = [[] for _ in range(len(df))]
        for index, row in df.iterrows():
            try:
                processed_data, product_refs = process_metafields(row, row.get('metafields_config', []))
                all_processed_data.append(processed_data)
                product_id = row.get('product_id', f'index_{index}')
                if product_id:
                    for key, value in product_refs.items(): all_product_references[f"{product_id}_{key}"] = value
            except Exception as e:
                print(f"Error processing metafields for product at index {index}: {e}")
                all_processed_data.append({})
        metafields_processed = pd.DataFrame(all_processed_data, index=df.index)
        if all_product_references:
            from preprocessing.metafields_processor import save_product_references
            save_product_references(all_product_references, output_dir, "product_references.csv")
            print(f"Product references saved to {output_dir}/product_references.csv")
        print("Calculating color similarities...")
        color_similarity_df = save_color_similarity_data(output_dir) # Assuming this function handles None/empty case
        print("Removing color columns after similarity calculation...")
        metafields_processed = remove_color_columns(metafields_processed)
        metafields_processed = apply_tfidf_processing(metafields_processed)
        if 'metafields' in df.columns: df = df.drop('metafields', axis=1)
        if 'metafields_config' in df.columns: df = df.drop('metafields_config', axis=1)
        df = pd.concat([df, metafields_processed.set_index(df.index)], axis=1)
        export_sample(df, "metafields", output_dir)

        # 2. Process is_gift_card
        print("Processing is_gift_card...")
        df = process_gif_card(df)
        export_sample(df, "gift_card", output_dir)
        # 3. Process available_for_sale
        print("Processing available_for_sale...")
        df = process_avaiable_for_sale(df)
        export_sample(df, "available_for_sale", output_dir)
        # Product Types
        print("Processing product types...")
        df = process_product_type(df)
        export_sample(df, "product_type", output_dir)
        # 4. Process tags
        print("Processing tags...")
        df = process_tags(df)
        export_sample(df, "tags", output_dir)
        # 5. Process collections
        print("Processing collections...")
        df = process_collections(df)
        export_sample(df, "collections", output_dir)
        # 6. Process vendors
        print("Processing vendors...")
        df = process_vendors(df)
        export_sample(df, "vendors", output_dir)
        # 7. Process variants
        print("Processing variants...")
        df = process_variants(df)
        export_sample(df, "variants", output_dir)
        # 8. Process createdAt timestamps
        print("Processing createdAt timestamps...")
        df = process_created_at(df)
        export_sample(df, "created_at", output_dir)
        # --- Fim dos passos 1-8 ---


        # 9. Process text with TF-IDF (Using descriptions)
        print("Processing product descriptions with TF-IDF...")
        df_final_features, recommendation_input_df, vectorizer, similar_products = process_descriptions_tfidf(
             df, output_prefix=os.path.join(output_dir, "products")
        )

        # Validar resultados do TF-IDF
        if df_final_features is None or df_final_features.empty:
             print("Error: TF-IDF processing did not return a valid feature DataFrame.")
             all_steps_successful = False
        else:
            export_sample(df_final_features, "final_features", output_dir)

        if similar_products is None:
            print("Warning: TF-IDF processing did not return similar products data.")
            similar_products = pd.DataFrame() # Garantir que é um DataFrame vazio

        # Guardar ficheiros intermédios/finais
        if all_steps_successful:
            try:
                 df.to_csv(os.path.join(output_dir, "products_features_before_desc_tfidf.csv"), index=False)
                 df_final_features.to_csv(os.path.join(output_dir, "products_final_all_features.csv"), index=False)
                 if recommendation_input_df is not None and not recommendation_input_df.empty:
                      recommendation_input_df.to_csv(os.path.join(output_dir, "products_recommendation_input.csv"), index=False)
            except Exception as e:
                 print(f"Error saving intermediate/final processed data: {e}")

            # Documentar fontes
            document_feature_sources(df_final_features, output_dir, "feature_sources.csv")

    except Exception as e:
        print(f"\n--- Critical Error during data processing pipeline: {e} ---")
        traceback.print_exc()
        all_steps_successful = False


    # --- Output Pós-Processamento ---
    if all_steps_successful:
        print("\n=== Data processing pipeline completed successfully! ===")
        # ... (mensagens de output dos ficheiros salvos) ...
        print("\nKey processed files saved:")
        print(f"- {output_dir}/products.csv (raw data)")
        print(f"- {output_dir}/products_features_before_desc_tfidf.csv (features before description TF-IDF)")
        print(f"- {output_dir}/products_final_all_features.csv (complete features including description TF-IDF)")
        # print(f"- {output_dir}/products_recommendation_input.csv (data potentially used for KNN input)")
        print(f"- {output_dir}/products_tfidf_matrix.npy (TF-IDF vectors for descriptions)")
        print(f"- {output_dir}/products_tfidf_features.csv (TF-IDF feature names for descriptions)")
        print(f"- {output_dir}/feature_sources.csv (documentation of feature origins)")
        if os.path.exists(os.path.join(output_dir, "recent_order_items.csv")): print(f"- {output_dir}/recent_order_items.csv")
        if os.path.exists(os.path.join(output_dir, "popular_products.csv")): print(f"- {output_dir}/popular_products.csv")
        if os.path.exists(os.path.join(output_dir, "product_groups.csv")): print(f"- {output_dir}/product_groups.csv")
        if color_similarity_df is not None and not color_similarity_df.empty: print(f"- {output_dir}/products_color_similarity.csv")
        if similar_products is not None and not similar_products.empty: print(f"- {output_dir}/products_similar_products.csv")

    else:
        print("\n=== Data processing pipeline finished with errors. Please check logs. ===")
        print("Skipping recommendation generation due to processing errors.")
        return # Terminar a execução


    # --- Recommendation Generation ---
    knn_recommendations_df = None
    if not args.skip_recommendations:
        print("\n" + "="*80)
        print("Starting KNN recommendation generation...")
        print("="*80)

        optimization_params = None
        if args.optimize_weights:
            # ... (lógica para definir optimization_params como antes) ...
            print(f"Weight optimization enabled using method: {args.optimization_method}")
            if args.optimization_method == 'random':
                optimization_params = {'n_iterations': args.opt_iterations, 'weight_range': (0.1, 10.0)}
            elif args.optimization_method == 'evolutionary':
                optimization_params = {'population_size': args.population_size, 'generations': args.generations, 'mutation_rate': 0.2, 'weight_range': (0.1, 10.0)}
            elif args.optimization_method == 'grid':
                 optimization_params = {}
                 print("Using default grid search parameters unless configured inside run_knn/WeightOptimizer.")


        if df_final_features is None or df_final_features.empty:
             print("Error: Cannot run KNN, the final feature DataFrame is missing or empty.")
        else:
            try:
                # Run the KNN function
                knn_recommendations_df = run_knn(
                    df=df_final_features, # Usar o DataFrame com todas as features
                    output_dir=output_dir,
                    n_neighbors=args.neighbors,
                    save_model=True,
                    similar_products_df=similar_products if similar_products is not None and not similar_products.empty else None, # Passar similar_products
                    optimize_weights=args.optimize_weights,
                    optimization_method=args.optimization_method,
                    optimization_params=optimization_params
                )

                if knn_recommendations_df is not None and not knn_recommendations_df.empty:
                    print("\nKNN recommendation generation complete!")
                    # ... (mensagens sobre ficheiros KNN salvos) ...
                    if os.path.exists(os.path.join(output_dir, 'all_recommendations.csv')): print(f"- {output_dir}/all_recommendations.csv")
                    if os.path.exists(os.path.join(output_dir, 'similar_products_example.csv')): print(f"- {output_dir}/similar_products_example.csv")
                    if args.optimize_weights and os.path.exists(os.path.join(output_dir, "weight_optimization", "optimization_results.csv")): print(f"- {output_dir}/weight_optimization/")
                elif knn_recommendations_df is not None and knn_recommendations_df.empty:
                     print("KNN process completed, but no recommendations were generated.")
                else:
                     print("KNN process failed to return recommendations.")

            except Exception as e:
                print(f"\n--- Critical Error during KNN recommendation generation: {e} ---")
                traceback.print_exc()
                knn_recommendations_df = None


    # --- Hybrid Recommendation Combination ---
    if not args.skip_recommendations and knn_recommendations_df is not None and not knn_recommendations_df.empty:
        print("\n" + "="*80)
        print("Combining KNN recommendations with other sources (Hybrid)...")
        print("="*80)

        try:
            # Generate hybrid recommendations - CHAMADA ATUALIZADA
            hybrid_recommendations = generate_hybrid_recommendations_adaptative(
                all_knn_recommendations=knn_recommendations_df,
                products_df=df_final_features,
                popular_products_df=popular_products_df, # Passa None se não carregado
                product_groups_df=product_groups_df,   # Passa None se não carregado
                total_recommendations=args.neighbors,
                # Passar argumentos da linha de comando
                strong_similarity_threshold=args.strong_sim,
                high_similarity_threshold=args.high_sim,
                min_similarity_threshold=args.min_sim,
                pop_boost_factor=args.pop_boost,
                group_boost_factor=args.group_boost
            )

            # Salvar recomendações híbridas
            if hybrid_recommendations is not None and not hybrid_recommendations.empty:
                save_hybrid_recommendations(hybrid_recommendations, os.path.join(output_dir, "hybrid_recommendations.csv"))
            elif hybrid_recommendations is not None and hybrid_recommendations.empty:
                 print("Hybrid combination process completed, but no final recommendations were generated.")
            else:
                 print("Hybrid combination process failed to return recommendations.")

        except Exception as e:
            print(f"\n--- Critical Error during Hybrid recommendation combination: {e} ---")
            traceback.print_exc()

    elif not args.skip_recommendations:
        print("\nSkipping Hybrid recommendation combination because KNN recommendations are missing or empty.")

    print("\n=== Full Pipeline Finished ===")


if __name__ == "__main__":
    main()