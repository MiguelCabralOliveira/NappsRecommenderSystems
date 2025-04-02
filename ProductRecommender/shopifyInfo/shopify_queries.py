import os
import sys
import requests
import pandas as pd
import json
from dotenv import load_dotenv
from collections.abc import Mapping # Keep this import for safety checks

# Erro com os imports
try:
    from .metafields_fetcher import fetch_metafields
    from .shop_settings import get_shop_settings
except ImportError:
    from metafields_fetcher import fetch_metafields
    from shop_settings import get_shop_settings

# Load environment variables from .env file
load_dotenv()

# KEEP THE ORIGINAL FUNCTION SIGNATURE (no new country/language args)
def get_shopify_products(store_url: str, access_token: str, metafields_config: list, products_limit: int = 250, cursor: str = None) -> tuple[pd.DataFrame, bool, str] | tuple[None, bool, str]:
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
    url = f"{store_url}/api/2025-01/graphql.json" # Use consistent API version

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

    # --- MODIFIED: Added GraphQL variables and @inContext directive ---
    query = f"""
    query ($country: CountryCode, $language: LanguageCode) @inContext(country: $country, language: $language) {{
      products(first: {products_limit} {after_param}) {{
        nodes {{
          # --- KEEP ORIGINAL FIELDS ---
          availableForSale
          id
          vendor
          title
          createdAt
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

 
    payload = {
        "query": query,
        "variables": {
            "country": "US",
            "language": "EN"
        }
    }

    # Print the query and variables for debugging (optional but helpful)
    # print("\n--- DEBUG: GraphQL Query ---")
    # print(query)
    # print("--- DEBUG: GraphQL Variables ---")
    # print(json.dumps(payload["variables"], indent=2))
    # print("--- END DEBUG ---\n")

    try:
        print(f"Making request to: {url} with EN/EN context") # Added context note

        # --- MODIFIED: Send payload instead of just query ---
        response = requests.post(url, json=payload, headers=headers)
        print(f"Response Status Code: {response.status_code}")

        response.raise_for_status()
        data = response.json()

       
        if 'errors' in data:
            print(f"GraphQL Errors: {json.dumps(data['errors'], indent=2)}") 
            return None, False, ""

       
        if 'data' not in data or not isinstance(data.get('data'), Mapping) or 'products' not in data['data'] or not isinstance(data['data'].get('products'), Mapping):
             print(f"GraphQL Error: Unexpected data structure in response.")
             print(f"Response Data: {json.dumps(data.get('data', {}), indent=2)}")
             return None, False, ""

        page_info = data['data']['products'].get('pageInfo', {})
        has_next_page = page_info.get('hasNextPage', False)
        end_cursor = page_info.get('endCursor')

        print(f"Pagination Info - Has Next Page: {has_next_page}, End Cursor: {end_cursor}")

        product_nodes = data['data']['products'].get('nodes', [])
        if not isinstance(product_nodes, list):
             print(f"GraphQL Error: 'products.nodes' is not a list.")
             print(f"Nodes received: {product_nodes}")
             product_nodes = []

        flattened_data = []
        for product in product_nodes:
             if not isinstance(product, Mapping):
                 print(f"Warning: Skipping invalid product node: {product}")
                 continue

             collections_data = product.get('collections', {})
             collection_nodes = collections_data.get('nodes', []) if isinstance(collections_data, Mapping) else []
             collections = [
                 {
                     'id': collection.get('id'),
                     'handle': collection.get('handle'),
                     'title': collection.get('title', '')
                 }
                 for collection in collection_nodes if isinstance(collection, Mapping) and collection.get('id')
             ]

             variants_data = product.get('variants', {})
             variant_nodes = variants_data.get('nodes', []) if isinstance(variants_data, Mapping) else []
             variants = [variant for variant in variant_nodes if isinstance(variant, Mapping)]

             metafield_nodes = product.get('metafields', [])
             valid_metafields = [m for m in metafield_nodes if isinstance(m, Mapping)]

             product_data = {
                'product_title': product.get('title'),
                'product_id': product.get('id'),
                'vendor': product.get('vendor'),
                'description': product.get('description', ''),
                'handle': product.get('handle'),
                'product_type': product.get('productType'),
                'is_gift_card': product.get('isGiftCard', False),
                'available_for_sale': product.get('availableForSale'),
                'tags': product.get('tags', []),
                'collections': collections,
                'variants': variants,
                'metafields': valid_metafields,
                'metafields_config': metafields_config,
                'createdAt': product.get('createdAt', '')
            }

             if product_data['product_id']:
                 flattened_data.append(product_data)
             else:
                  print(f"Warning: Skipping product node due to missing 'id': {product}")

        print(f"Successfully processed {len(flattened_data)} products from this page.")

        if not flattened_data:
             print("No valid product data processed on this page.")
             return pd.DataFrame(), has_next_page, end_cursor

        df = pd.DataFrame(flattened_data)
        return df, has_next_page, end_cursor

    except requests.exceptions.RequestException as e:
        print(f"Network Error connecting to Shopify API: {e}")
        return None, False, ""
    except Exception as e:
        print(f"Error processing Shopify Products: {e}")
        if 'response' in locals() and response is not None:
            try:
                error_details = response.json()
                print(f"Response Error Details: {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response Content (non-JSON): {response.text[:500]}...")
            except Exception as json_e:
                 print(f"Could not parse error response: {json_e}")
        import traceback
        traceback.print_exc()
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
        print("Warning: Failed to fetch metafields configuration. Metafields will not be included.")
        metafields_config = []

    if not fetched_store_url or not fetched_token:
        print("Fetching shop settings as fallback for URL/token...")
        shop_settings = get_shop_settings(shop_id)

        if not shop_settings:
            print("Failed to get shop settings (URL/token). Cannot proceed.")
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
    max_pages = 200 

    while has_next_page:
        page_count += 1
        print(f"\n--- Fetching page {page_count} ---")

       
        
        products_df, has_next_page, cursor = get_shopify_products(
            store_url=store_url,
            access_token=access_token,
            metafields_config=metafields_config,
            cursor=cursor
            
        )

        if products_df is None:
            print(f"Critical Error: Failed to retrieve data on page {page_count}. Aborting fetch.")
            if all_products:
                 print("Returning partially fetched data due to error.")
                 break
            else:
                 return None

        if not products_df.empty:
            all_products.append(products_df)
            print(f"Processed {len(products_df)} products from page {page_count}. More pages: {has_next_page}")
        else:
            print(f"Page {page_count} returned no valid product data. More pages according to API: {has_next_page}")
            if cursor is None and has_next_page:
                 print("Warning: API indicates more pages but returned no endCursor. Stopping pagination.")
                 has_next_page = False

        if page_count >= max_pages:
            print(f"Reached maximum page count ({max_pages}). Stopping pagination.")
            break

        if not has_next_page:
             print("API indicates no more pages. Stopping pagination.")
             break

    if all_products:
        combined_df = pd.concat(all_products, ignore_index=True)
        print(f"\nTotal valid products retrieved across all pages: {len(combined_df)}")
        if 'createdAt' in combined_df.columns:
            print("'createdAt' column is present in the final combined dataframe.")
        else:
            print("WARNING: 'createdAt' column is MISSING from the final combined dataframe.")
        return combined_df

    print("\nNo products retrieved or processed successfully.")
    return None



if __name__ == "__main__":
    shop_id = input("Enter shop ID: ")
    result = get_all_products(shop_id)

    if result is not None and not result.empty:
        print("\nSuccessfully retrieved data!")
        print(f"\nTotal products found: {len(result)}")
        print("\nSample of the data:")
        print(result[['product_title', 'vendor', 'product_type', 'product_id']].head())

        output_filename = "shopify_products_output.csv"
        try:
            result.to_csv(output_filename, index=False)
            print(f"\nFull data saved to '{output_filename}'")
        except Exception as e:
            print(f"\nError saving data to CSV: {e}")
    elif result is not None and result.empty:
         print("\nSuccessfully connected and fetched, but no valid product data was returned or processed.")
    else:
        print("\nFailed to retrieve data.")