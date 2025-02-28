import os
import sys
import requests
import pandas as pd
import json
from dotenv import load_dotenv

# Erro com os imports 
try:
    from .metafields_fetcher import fetch_metafields
    from .shop_settings import get_shop_settings
except ImportError:
    
    from metafields_fetcher import fetch_metafields
    from shop_settings import get_shop_settings

# Load environment variables from .env file
load_dotenv()

def get_shopify_products(store_url: str, access_token: str, metafields_config: list, products_limit: int = 1, cursor: str = None) -> tuple[pd.DataFrame, bool, str] | tuple[None, bool, str]:
    """
    Fetch products data from Shopify API with pagination support
    Args:
        store_url: Shopify store URL
        access_token: Shopify access token
        metafields_config: List of metafield configurations
        products_limit: Maximum number of products to fetch per request
        cursor: Pagination cursor for fetching next page of results
    Returns:
        Tuple containing:
            - DataFrame: Raw products data or None if error occurs
            - bool: Whether there are more pages to fetch
            - str: End cursor for pagination
    """
    url = f"{store_url}/api/2025-01/graphql.json"
    
    headers = {
        "X-Shopify-Storefront-Access-Token": access_token,
        "Content-Type": "application/json",
    }
    
    metafields_identifiers = ",".join([
        f'{{ key: "{field["key"]}", namespace: "{field["namespace"]}" }}'
        for field in metafields_config
    ])
    
    # Build the after parameter for pagination if cursor is provided
    after_param = f'after: "{cursor}"' if cursor else ''
    
    query = f"""
    query {{
      products(first: {products_limit} {after_param}) {{
        nodes {{
          availableForSale
          id
          vendor
          title
          tags
          handle
          metafields(identifiers: [{metafields_identifiers}]) {{
                id
                value
                key
                namespace
                type
                references(first: 10) {{
                    nodes{{
                        ... on Metaobject{{
                        id
                        fields{{
                        value
                        key
                        type
                        }}
                        type
                        
                        }}

                    }}
                
                }}
              }}
          isGiftCard
          productType
          description
          collections(first: 250) {{
            nodes {{
              id
              handle
              title
            }}
          }}
          variants(first: 250) {{
            nodes {{
              id
              title
              currentlyNotInStock
              availableForSale
              price {{
                amount
              }}
              compareAtPrice {{
                amount
              }}
              weight
              quantityAvailable
              taxable
              selectedOptions {{
                name
              }}
              unitPrice {{
                amount
              }}
            }}
          }}
        }}
        pageInfo {{
          hasNextPage
          endCursor
        }}
      }}
    }}
    """
    
    # Print the query for debugging
    print("\n--- DEBUG: GraphQL Query ---")
    print(query)
    print("--- END DEBUG ---\n")
    
    try:
        print(f"Making request to: {url}")
        
        response = requests.post(url, json={"query": query}, headers=headers)
        print(f"Response Status Code: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        # Check if there are errors in the response
        if 'errors' in data:
            print(f"GraphQL Errors: {data['errors']}")
            return None, False, ""
        
        # Extract pagination info
        page_info = data['data']['products']['pageInfo']
        has_next_page = page_info['hasNextPage']
        end_cursor = page_info['endCursor']
        
        # Print pagination info for debugging
        print(f"Pagination Info - Has Next Page: {has_next_page}, End Cursor: {end_cursor}")
        
        flattened_data = []
        for product in data['data']['products']['nodes']:
            # Extract collections data
            collections = []
            if 'collections' in product and 'nodes' in product['collections']:
                collections = [
                    {
                        'id': collection['id'],
                        'handle': collection['handle'],
                        'title': collection.get('title', '')
                    } 
                    for collection in product['collections']['nodes']
                ]
            
            product_data = {
                'product_title': product['title'],
                'product_id': product['id'],
                'vendor': product['vendor'],
                'description': product.get('description', ''),
                'handle': product['handle'],
                'product_type': product['productType'],
                'is_gift_card': product.get('isGiftCard', False),
                'available_for_sale': product['availableForSale'],
                'tags': product['tags'],
                'collections': collections,
                'variants': [variant for variant in product['variants']['nodes']],
                'metafields': [], 
                'metafields_config': metafields_config  
            }
            
            # Extract metafields from the product level
            if 'metafields' in product:
                # Filter out None/null values and keep only valid metafield objects
                valid_metafields = [m for m in product['metafields'] if m is not None]
                product_data['metafields'] = valid_metafields
            
            flattened_data.append(product_data)
        
        print(f"Successfully retrieved {len(flattened_data)} products")
        
        df = pd.DataFrame(flattened_data)
        return df, has_next_page, end_cursor
        
    except Exception as e:
        print(f"GraphQL Error: {e}")
        if 'response' in locals():
            try:
                error_details = response.json()
                print(f"Response Error Details: {error_details}")
            except:
                print(f"Response Content: {response.text[:500]}...")
        
        return None, False, ""

def get_all_products(shop_id: str) -> pd.DataFrame | None:
    """
    Fetch all products using pagination
    Args:
        shop_id: Shop ID to fetch data for
    Returns:
        DataFrame: Combined data from all pages or None if error occurs
    """
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    
    if not email or not password:
        print("EMAIL and PASSWORD environment variables are required in .env file")
        return None
    
    print(f"Fetching metafields for shop ID: {shop_id}")
    metafields_config, fetched_store_url, fetched_token = fetch_metafields(email, password, shop_id)
    
    if not metafields_config:
        print("Failed to fetch metafields configuration")
        return None
    
    if not fetched_store_url or not fetched_token:
        print("Fetching shop settings as fallback...")
        shop_settings = get_shop_settings(shop_id)
        
        if not shop_settings:
            print("Failed to get shop settings")
            return None
        
        store_url, access_token = shop_settings
    else:
        store_url, access_token = fetched_store_url, fetched_token
    
    print(f"Using store URL: {store_url}")
    print(f"Using metafields config: {len(metafields_config)} items")
    
    all_products = []
    has_next_page = True
    cursor = None
    page_count = 0
    
    while has_next_page:
        page_count += 1
        print(f"\n--- Fetching page {page_count} ---")
        
        products_df, has_next_page, cursor = get_shopify_products(
            store_url, access_token, metafields_config, cursor=cursor
        )
        
        if products_df is None:
            print(f"Failed to retrieve data on page {page_count}")
            return None
        
        all_products.append(products_df)
        print(f"Fetched {len(products_df)} products. More pages: {has_next_page}")
        
        # Limit to prevent infinite loops during testing
        if page_count >= 10:
            print("Reached maximum page count (10). Stopping pagination.")
            break
    
    if all_products:
        combined_df = pd.concat(all_products, ignore_index=True)
        print(f"Total products retrieved: {len(combined_df)}")
        return combined_df
    
    print("No products retrieved")
    return None

if __name__ == "__main__":
   
    shop_id = input("Enter shop ID: ")
    result = get_all_products(shop_id)
    
    if result is not None:
        print("\nSuccessfully retrieved data!")
        print(f"\nTotal products found: {len(result)}")
        print("\nSample of the data:")
        print(result[['product_title', 'vendor', 'product_type']].head())
        
        result.to_csv("shopify_products_output.csv", index=False)
        print("\nFull data saved to 'shopify_products_output.csv'")
    else:
        print("\nFailed to retrieve data.")