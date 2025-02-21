from shopifyInfo.shop_settings import get_shop_settings
from shopifyInfo.shopify_queries import get_collections_products
from preprocessing.tfidf_processor import process_descriptions_tfidf

def main():
    # Fetch data from Shopify
    if (result := get_shop_settings(input("Shop ID: "))):
        store_url, token = result
        collections_data = get_collections_products(store_url, token)
        
        if collections_data is not None:
            # Save raw data
            collections_data.to_csv("products.csv", index=False)
            print("\nRaw data saved to products.csv")
            
            # Process with TF-IDF
            print("\nStarting TF-IDF processing...")
            tfidf_df, _ = process_descriptions_tfidf()
            
            print("\nProcessing complete! Final output:")
            print(tfidf_df[['product_title', 'clean_text']].head())
            print("\nTF-IDF features saved to:")
            print("- products_with_tfidf.csv")
            print("- products_tfidf_matrix.npy")
            print("- products_tfidf_features.csv")

if __name__ == "__main__":
    main()