from shopifyInfo.shopify_queries import get_all_products
from shopifyInfo.database_query import fetch_recent_order_items
from preprocessing.tfidf_processor import process_descriptions_tfidf  # Changed from Word2Vec to TF-IDF
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
import pandas as pd
import argparse
import os


def export_sample(df, step_name, output_dir="results"):
    """Export the first row of the DataFrame to a CSV for debugging"""
    if len(df) > 0:
        sample_df = df.head(1)
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/sample_after_{step_name}.csv"
        sample_df.to_csv(filename, index=False)
        print(f"Exported sample to {filename}")


def document_feature_sources(df, output_dir="results", output_file="feature_sources.csv"):
    """
    Document the sources of all features in the processed DataFrame
    """
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
        elif column.startswith('price_'):
            source = 'Price-related feature from variants'
        elif column == 'product_id' or column == 'product_title' or column == 'handle':
            source = 'Original product information'
        elif column.startswith('created_') or column == 'days_since_first_product' or column == 'is_recent_product':
            source = 'Time-based features from createdAt timestamp'
        elif column.startswith('release_quarter_'):
            source = 'Release quarter features from createdAt timestamp'
        else:
            source = 'Other/Unknown'
        
        feature_sources.append({
            'feature_name': column,
            'source': source
        })
    
    # Create DataFrame and save to CSV
    sources_df = pd.DataFrame(feature_sources)
    os.makedirs(output_dir, exist_ok=True)
    full_path = f"{output_dir}/{output_file}"
    sources_df.to_csv(full_path, index=False)
    print(f"Feature sources documented in {full_path}")
    
    return sources_df


def main():
    """Main function to process product data from Shopify store"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process product data and generate recommendations')
    parser.add_argument('--neighbors', type=int, default=12,
                      help='Number of neighbors for KNN recommendations (default: 12)')
    parser.add_argument('--skip-recommendations', action='store_true',
                      help='Skip the recommendation generation step')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save recommendation results (default: results)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get shop ID from user input
    shop_id = input("Shop ID: ")
    
   
    # Fetch order data from database
    print("\n=== Fetching recent order data from database ===")
    order_items_df, popular_products_df, product_groups_df = fetch_recent_order_items(shop_id)
    
    
    # Save order data to CSV files
    if not order_items_df.empty:
        order_items_df.to_csv(f"{args.output_dir}/recent_order_items.csv", index=False)
        print(f"Recent order items saved to {args.output_dir}/recent_order_items.csv")
    
    if not popular_products_df.empty:
        popular_products_df.to_csv(f"{args.output_dir}/popular_products.csv", index=False)
        print(f"Popular products saved to {args.output_dir}/popular_products.csv")
    
    if not product_groups_df.empty:
        product_groups_df.to_csv(f"{args.output_dir}/product_groups.csv", index=False)
        print(f"Product groups saved to {args.output_dir}/product_groups.csv")
    
    print("\n=== Fetching product data from Shopify ===")
    raw_data = get_all_products(shop_id)
    
    if raw_data is None:
        print("Failed to retrieve data from Shopify.")
        return
    
    # Save raw data immediately after importing
    raw_data.to_csv(f"{args.output_dir}/products.csv", index=False)
    print(f"Raw data saved to {args.output_dir}/products.csv")
    
    # Export raw data sample
    export_sample(raw_data, "raw_data", args.output_dir)
    
    # Start with the raw data for processing
    df = raw_data.copy()
    
    # Process the data through the pipeline
    print("\n=== Starting data processing pipeline ===")
    
    # 1. Process metafields
    print("Processing metafields...")
    all_processed_data = []
    all_product_references = {}

    for _, row in df.iterrows():
        processed_data, product_refs = process_metafields(row, row.get('metafields_config', []))
        all_processed_data.append(processed_data)
        
        # Collect product references
        product_id = row.get('product_id', '')
        for key, value in product_refs.items():
            all_product_references[f"{product_id}_{key}"] = value

    # Create DataFrame from processed metafields
    metafields_processed = pd.DataFrame(all_processed_data, index=df.index)

    # Save product references if any were found
    if all_product_references:
        from preprocessing.metafields_processor import save_product_references
        save_product_references(all_product_references, args.output_dir, "product_references.csv")
        print(f"Product references saved to {args.output_dir}/product_references.csv")
    
    # Calculate and save color similarity data before removing color columns
    print("Calculating color similarities...")
    color_similarity_df = save_color_similarity_data(args.output_dir)
    if not color_similarity_df.empty:
        print(f"Color similarity data saved for {color_similarity_df['source_product_id'].nunique()} products")
    
    # Remove color columns after similarity calculation
    print("Removing color columns after similarity calculation...")
    metafields_processed = remove_color_columns(metafields_processed)

    # Apply TF-IDF processing to the metafields data - this will also remove original text columns
    metafields_processed = apply_tfidf_processing(metafields_processed)

    # Add processed metafields to the dataframe
    if 'metafields' in df.columns:
        df = df.drop('metafields', axis=1)
    if 'metafields_config' in df.columns:
        df = df.drop('metafields_config', axis=1)

    # Combine the processed metafields with the original dataframe
    df = pd.concat([df, metafields_processed], axis=1)
    export_sample(df, "metafields", args.output_dir)
    
    # 2. Process is_gift_card
    print("Processing is_gift_card...")
    df = process_gif_card(df)
    export_sample(df, "gift_card", args.output_dir)

    # 3. Process available_for_sale
    print("Processing available_for_sale...")
    df = process_avaiable_for_sale(df)
    export_sample(df, "available_for_sale", args.output_dir)

    print("Processing product types...")
    df = process_product_type(df)
    export_sample(df, "product_type", args.output_dir)
    
    # 4. Process tags
    print("Processing tags...")
    df = process_tags(df)
    export_sample(df, "tags", args.output_dir)
    
    # 5. Process collections
    print("Processing collections...")
    df = process_collections(df)
    export_sample(df, "collections", args.output_dir)

    # 6. Process vendors
    print("Processing vendors...")
    df = process_vendors(df)
    export_sample(df, "vendors", args.output_dir)
    
    # 7. Process variants
    print("Processing variants...")
    df = process_variants(df)
    export_sample(df, "variants", args.output_dir)

    # 8. Process createdAt timestamps
    print("Processing createdAt timestamps...")
    df = process_created_at(df)
    export_sample(df, "created_at", args.output_dir)
    
    # 9. Process text with TF-IDF (Changed from Word2Vec to TF-IDF)
    print("Processing product descriptions with TF-IDF...")
    tfidf_df, recommendation_df, vectorizer, similar_products = process_descriptions_tfidf(df, args.output_dir)
    export_sample(tfidf_df, "tfidf", args.output_dir)
    export_sample(recommendation_df, "recommendation", args.output_dir)
    
    # Save the processed data
    df.to_csv(f"{args.output_dir}/products_with_variants.csv", index=False)
    tfidf_df.to_csv(f"{args.output_dir}/products_with_tfidf.csv", index=False)
    recommendation_df.to_csv(f"{args.output_dir}/products_recommendation.csv", index=False)
    
    # Document feature sources
    document_feature_sources(tfidf_df, args.output_dir, "feature_sources.csv")
    
    # Output completion message for data processing
    print("\n=== Data processing complete! ===")
    print("\nProcessed files saved:")
    print(f"- {args.output_dir}/products.csv (raw data)")
    print(f"- {args.output_dir}/products_with_variants.csv (features before TF-IDF)")
    print(f"- {args.output_dir}/products_with_tfidf.csv (complete features)")
    print(f"- {args.output_dir}/products_recommendation.csv (ready for recommendation model)")
    print(f"- {args.output_dir}/products_tfidf_matrix.npy (TF-IDF vector representations)")
    print(f"- {args.output_dir}/products_tfidf_features.csv (TF-IDF feature names)")
    print(f"- {args.output_dir}/feature_sources.csv (documentation of feature sources)")
    print(f"- {args.output_dir}/recent_order_items.csv (all recent order items)")
    print(f"- {args.output_dir}/popular_products.csv (products ordered by popularity)")
    print(f"- {args.output_dir}/product_groups.csv (products grouped by checkout)")
    
    if not color_similarity_df.empty:
        print(f"- {args.output_dir}/products_color_similarity.csv (similar products by color)")
    
    # Run recommendation generation if not skipped
    if not args.skip_recommendations:
        print("\n" + "="*80)
        print("Starting recommendation generation...")
        print("="*80)
        
        # Run the KNN function to generate recommendations
        recommendations_df = run_knn(
            df = df,
            output_dir=args.output_dir,
            n_neighbors=args.neighbors,
            save_model=True,
            similar_products_df=similar_products if not similar_products.empty else None,
        )
        
        print("\nRecommendation generation complete!")
        print(f"Product recommendations and visualizations saved to {args.output_dir}")
        print(f"- {args.output_dir}/all_recommendations.csv (comprehensive recommendations for all products)")
        print(f"- {args.output_dir}/similar_products_example.csv (example recommendations for a sample product)")

        # After running the KNN function in main.py, add the following code:
    if not args.skip_recommendations and 'recommendations_df' in locals() and not recommendations_df.empty:
        print("\n" + "="*80)
        print("Combining recommendations with popularity and co-purchase data...")
        print("="*80)
        
        from training.recommendation_combiner import generate_hybrid_recommendations_adaptative
        
        # Generate hybrid recommendations
        hybrid_recommendations = generate_hybrid_recommendations_adaptative(
            all_knn_recommendations=recommendations_df,
            products_df=df,
            popular_products_df=popular_products_df if not popular_products_df.empty else None,
            product_groups_df=product_groups_df if not product_groups_df.empty else None,
            total_recommendations=args.neighbors,
            strong_similarity_threshold=0.3,
            high_similarity_threshold=0.9,
            min_similarity_threshold=0.05
        )
        
        # Save the hybrid recommendations
        if not hybrid_recommendations.empty:
            hybrid_output_file = f"{args.output_dir}/hybrid_recommendations.csv"
            hybrid_recommendations.to_csv(hybrid_output_file, index=False)
            print(f"Hybrid recommendations saved to {hybrid_output_file}")
            
            # Calculate source distribution
            if 'source' in hybrid_recommendations.columns:
                source_counts = hybrid_recommendations['source'].value_counts()
                print("\nRecommendation sources distribution:")
                for source, count in source_counts.items():
                    percentage = 100 * count / len(hybrid_recommendations)
                    print(f"  - {source}: {count} ({percentage:.1f}%)")
        else:
            print("No hybrid recommendations generated.")

if __name__ == "__main__":
    main()