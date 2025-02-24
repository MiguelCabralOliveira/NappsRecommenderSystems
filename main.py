from shopifyInfo.shop_settings import get_shop_settings
from shopifyInfo.shopify_queries import get_collections_products
from preprocessing.tfidf_processor import process_descriptions_tfidf
from preprocessing.collectionHandle_processor import process_collections
from preprocessing.tags_processor import process_tags
from preprocessing.variants_processor import process_variants
from preprocessing.metafields_processor import get_metafield_keys, process_metafields

def main():
    if (result := get_shop_settings(input("Shop ID: "))):
        store_url, token = result
        
        # Get metafields configuration
        metafields = get_metafield_keys()["metafields"]
        
        # Get raw data with metafields
        collections_data = get_collections_products(store_url, token, metafields)
        
        if collections_data is not None:
            # Process metafields
            print("\nProcessing metafields...")
            metafields_processed = collections_data.apply(
                lambda row: process_metafields(
                    {'metafields': row['metafields']}, 
                    metafields
                ), 
                axis=1
            )
            collections_data = collections_data.drop('metafields', axis=1)
            for key in metafields_processed.iloc[0].keys():
                collections_data[key] = metafields_processed.apply(lambda x: x[key])
            
            # Save raw data
            collections_data.to_csv("products.csv", index=False)
            
            # Rest of the processing pipeline remains the same
            print("\nProcessing collections...")
            collections_processed_df = process_collections(collections_data)
            
            print("\nProcessing tags...")
            tags_processed_df = process_tags(collections_processed_df)
            
            print("\nProcessing variants...")
            variants_processed_df = process_variants(tags_processed_df)
            
            print("\nStarting TF-IDF processing...")
            tfidf_df, recommendation_df, _, similar_products = process_descriptions_tfidf(variants_processed_df)
            
            print("\nProcessing complete!")
            
            # Save outputs
            variants_processed_df.to_csv("products_with_variants.csv", index=False)
            
            print("\nProcessed features saved to:")
            print("- products_with_variants.csv")
            print("- products_with_tfidf.csv")
            print("- products_recommendation.csv")
            print("- products_tfidf_matrix.npy")
            print("- products_tfidf_features.csv")

if __name__ == "__main__":
    main()