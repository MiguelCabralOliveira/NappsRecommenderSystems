from shopifyInfo.shop_settings import get_shop_settings
from shopifyInfo.shopify_queries import get_collections_products
from preprocessing.tfidf_processor import process_descriptions_tfidf
from preprocessing.collectionHandle_processor import process_collections
from preprocessing.tags_processor import process_tags
from preprocessing.variants_processor import process_variants

def main():
    if (result := get_shop_settings(input("Shop ID: "))):
        store_url, token = result
        collections_data = get_collections_products(store_url, token)
        
        if collections_data is not None:
            # Save raw data
            collections_data.to_csv("products.csv", index=False)
            
            # Process collections
            print("\nProcessing collections...")
            collections_processed_df = process_collections(collections_data)
            
            # Process tags
            print("\nProcessing tags...")
            tags_processed_df = process_tags(collections_processed_df)
            
            # Process variants
            print("\nProcessing variants...")
            variants_processed_df = process_variants(tags_processed_df)
            
            # Process with TF-IDF
            print("\nStarting TF-IDF processing...")
            tfidf_df, _, similar_products = process_descriptions_tfidf(variants_processed_df)
            
            print("\nProcessing complete!")
            
            # Save intermediate processed data
            variants_processed_df.to_csv("products_with_variants.csv", index=False)
            
            # Save final processed data
            print("\nProcessed features saved to:")
            print("- products_with_variants.csv")
            print("- products_with_tfidf.csv")
            print("- products_tfidf_matrix.npy")
            print("- products_tfidf_features.csv")

if __name__ == "__main__":
    main()