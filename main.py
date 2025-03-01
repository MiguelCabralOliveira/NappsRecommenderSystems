from shopifyInfo.shopify_queries import get_all_products
from preprocessing.tfidf_processor import process_descriptions_tfidf
from preprocessing.collectionHandle_processor import process_collections
from preprocessing.tags_processor import process_tags
from preprocessing.variants_processor import process_variants
from preprocessing.related_products_processor import process_related_products
from preprocessing.vendor_processor import process_vendors
from preprocessing.metafields_processor import process_metafields, apply_tfidf_processing
from preprocessing.isGiftCard_processor import process_gif_card
from preprocessing.availableForSale_processor import process_avaiable_for_sale
import pandas as pd


def main():
    """Main function to process product data from Shopify store"""
    # Get shop ID from user input
    shop_id = input("Shop ID: ")
    
    print("Fetching data from Shopify...")
    collections_data = get_all_products(shop_id)
    
    if collections_data is None:
        print("Failed to retrieve data from Shopify.")
        return
        
    # Process the data through the pipeline
    print("\nStarting data processing pipeline...")
    
    # 1. Process metafields
    print("Processing metafields...")
    metafields_processed = pd.DataFrame([
        process_metafields(row, row.get('metafields_config', [])) 
        for _, row in collections_data.iterrows()
    ], index=collections_data.index)

    # Apply TF-IDF processing to the metafields data
    metafields_processed = apply_tfidf_processing(metafields_processed)

    # Add processed metafields to the dataframe
    if 'metafields' in collections_data.columns:
        collections_data = collections_data.drop('metafields', axis=1)
    if 'metafields_config' in collections_data.columns:
        collections_data = collections_data.drop('metafields_config', axis=1)

    # Combine the processed metafields with the original dataframe
    collections_data = pd.concat([collections_data, metafields_processed], axis=1)
    
    # Save raw data
    collections_data.to_csv("products.csv", index=False)


    # Process is_gift_card
    print("Processing is_gift_card...")
    df = process_gif_card(collections_data)


    # Process available_for_sale
    print("Processing avaiable_for_sale...")
    df = process_avaiable_for_sale(df)
    
    # 3. Process tags
    print("Processing tags...")
    df = process_tags(df)
    
    # 2. Process collections
    print("Processing collections...")
    df = process_collections(collections_data)
    
    # 5. Process vendors
    print("Processing vendors...")
    df = process_vendors(df)
    
    # 6. Process variants
    print("Processing variants...")
    df = process_variants(df)
    
    # 7. Process text with TF-IDF
    print("Processing product descriptions with TF-IDF...")
    tfidf_df, recommendation_df, _, similar_products = process_descriptions_tfidf(df)
    
    # Save the processed data
    df.to_csv("products_with_variants.csv", index=False)
    tfidf_df.to_csv("products_with_tfidf.csv", index=False)
    recommendation_df.to_csv("products_recommendation.csv", index=False)
    
    # Output completion message
    print("\nProcessing complete!")
    print("\nProcessed files saved:")
    print("- products.csv (raw data)")
    print("- products_with_variants.csv (features before TF-IDF)")
    print("- products_with_tfidf.csv (complete features)")
    print("- products_recommendation.csv (ready for recommendation model)")
    print("- products_tfidf_matrix.npy (TF-IDF vector representations)")
    print("- products_tfidf_features.csv (TF-IDF feature names)")
    
    # 8. Train product recommender model
    print("\nTraining product recommendation model...")
    try:
        from training.ProductRecommender import ProductRecommender
        
        # Create and train the recommender
        recommender = ProductRecommender()
        recommender.fit(tfidf_df, find_optimal=True)
        
        # Save the trained model
        recommender.save_model("product_recommender_model.pkl")
        
        # Generate sample recommendations
        if len(tfidf_df) > 0:
            sample_product_id = tfidf_df.iloc[0]['product_id']
            sample_product_title = tfidf_df.iloc[0]['product_title']
            
            print(f"\nSample recommendations for product: {sample_product_title}")
            
            # Get hybrid recommendations (best overall approach)
            recommendations = recommender.get_recommendations(
                sample_product_id, n_recommendations=5, strategy='hybrid'
            )
            
            # Print recommendations
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['product_title']}")
            else:
                print("No recommendations found for this product.")
            
            # 9. Generate recommendations for all products
            print("\nGenerating recommendations for all products...")
            all_recommendations_df = recommender.generate_all_recommendations(tfidf_df)
            
            # Save all recommendations
            all_recommendations_df.to_csv("all_product_recommendations.csv", index=False)
            
            print(f"\nAll recommendations generated successfully!")
            print(f"Total products processed: {len(tfidf_df)}")
            print(f"Total products with recommendations: {all_recommendations_df['source_product_id'].nunique()}")
            print(f"Total recommendations generated: {len(all_recommendations_df)}")
            print("Recommendations saved to all_product_recommendations.csv")
                
            print("\nRecommendation model training completed successfully!")
            print("\nOutputs:")
            print("- product_recommender_model.pkl (trained model)")
            print("- product_clusters.png (visualization of product clusters)")
            print("- optimal_clusters.png (analysis of optimal cluster count)")
            print("- all_product_recommendations.csv (recommendations for all products)")
            
        else:
            print("No products available to generate sample recommendations.")
            
    except Exception as e:
        print(f"\nError training recommendation model: {e}")
        print("Recommendation model training failed, but data processing was completed successfully.")


if __name__ == "__main__":
    main()