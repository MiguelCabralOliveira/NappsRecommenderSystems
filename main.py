from shopifyInfo.shopify_queries import get_all_products
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
from training.weighted_kmeans import run_clustering
import pandas as pd
import argparse


def export_sample(df, step_name):
    """Export the first row of the DataFrame to a CSV for debugging"""
    if len(df) > 0:
        sample_df = df.head(1)
        filename = f"sample_after_{step_name}.csv"
        sample_df.to_csv(filename, index=False)
        print(f"Exported sample to {filename}")


def document_feature_sources(df, output_file="feature_sources.csv"):
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
        else:
            source = 'Other/Unknown'
        
        feature_sources.append({
            'feature_name': column,
            'source': source
        })
    
    # Create DataFrame and save to CSV
    sources_df = pd.DataFrame(feature_sources)
    sources_df.to_csv(output_file, index=False)
    print(f"Feature sources documented in {output_file}")
    
    return sources_df


def main():
    """Main function to process product data from Shopify store"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process product data and perform clustering')
    parser.add_argument('--clusters', type=int, default=10,
                      help='Number of clusters for weighted KMeans (default: 10)')
    parser.add_argument('--find-optimal-k', action='store_true',
                      help='Find optimal number of clusters')
    parser.add_argument('--skip-clustering', action='store_true',
                      help='Skip the clustering step')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save clustering results (default: results)')
    args = parser.parse_args()
    
    # Get shop ID from user input
    shop_id = input("Shop ID: ")
    
    print("Fetching data from Shopify...")
    raw_data = get_all_products(shop_id)
    
    if raw_data is None:
        print("Failed to retrieve data from Shopify.")
        return
    
    # Save raw data immediately after importing
    raw_data.to_csv("products.csv", index=False)
    print("Raw data saved to products.csv")
    
    # Export raw data sample
    export_sample(raw_data, "raw_data")
    
    # Start with the raw data for processing
    df = raw_data.copy()
    
    # Process the data through the pipeline
    print("\nStarting data processing pipeline...")
    
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
        save_product_references(all_product_references, "product_references.csv")
        print("Product references saved to product_references.csv")
    
    # Calculate and save color similarity data before removing color columns
    print("Calculating color similarities...")
    color_similarity_df = save_color_similarity_data()
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
    export_sample(df, "metafields")
    
    # 2. Process is_gift_card
    print("Processing is_gift_card...")
    df = process_gif_card(df)
    export_sample(df, "gift_card")

    # 3. Process available_for_sale
    print("Processing available_for_sale...")
    df = process_avaiable_for_sale(df)
    export_sample(df, "available_for_sale")

    print("Processing product types...")
    df = process_product_type(df)
    export_sample(df, "product_type")
    
    # 4. Process tags
    print("Processing tags...")
    df = process_tags(df)
    export_sample(df, "tags")
    
    # 5. Process collections
    print("Processing collections...")
    df = process_collections(df)
    export_sample(df, "collections")

    # 6. Process vendors
    print("Processing vendors...")
    df = process_vendors(df)
    export_sample(df, "vendors")
    
    # 7. Process variants
    print("Processing variants...")
    df = process_variants(df)
    export_sample(df, "variants")

    # 8. Process createdAt timestamps
    print("Processing createdAt timestamps...")
    df = process_created_at(df)
    export_sample(df, "created_at")
    
    # 9. Process text with TF-IDF
    print("Processing product descriptions with TF-IDF...")
    tfidf_df, recommendation_df, _, similar_products = process_descriptions_tfidf(df)
    export_sample(tfidf_df, "tfidf")
    export_sample(recommendation_df, "recommendation")


    
    # Save the processed data
    df.to_csv("products_with_variants.csv", index=False)
    tfidf_df.to_csv("products_with_tfidf.csv", index=False)
    recommendation_df.to_csv("products_recommendation.csv", index=False)
    
    # Document feature sources
    document_feature_sources(tfidf_df, "feature_sources.csv")
    
    # Output completion message for data processing
    print("\nData processing complete!")
    print("\nProcessed files saved:")
    print("- products.csv (raw data)")
    print("- products_with_variants.csv (features before TF-IDF)")
    print("- products_with_tfidf.csv (complete features)")
    print("- products_recommendation.csv (ready for recommendation model)")
    print("- products_tfidf_matrix.npy (TF-IDF vector representations)")
    print("- products_tfidf_features.csv (TF-IDF feature names)")
    print("- feature_sources.csv (documentation of feature sources)")
    
    if not color_similarity_df.empty:
        print("- products_color_similarity.csv (similar products by color)")
    
    # Run clustering analysis if not skipped
    if not args.skip_clustering:
        print("\n" + "="*50)
        print("Starting clustering analysis...")
        print("="*50)
        
        # Run the clustering function
        clustered_df = run_clustering(
        input_file="products_with_tfidf.csv",
        output_dir=args.output_dir,
        n_clusters=args.clusters,
        find_optimal_k=True,  
        min_k=10,             
        max_k=50,             
        save_model=True
)
        
        print("\nClustering analysis complete!")
        print(f"Cluster assignments and visualizations saved to {args.output_dir}")
        print("- products_with_clusters.csv (products with cluster assignments)")
    else:
        print("\nClustering analysis skipped. Use --skip-clustering=False to run clustering.")


if __name__ == "__main__":
    main()