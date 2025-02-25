import requests
import pandas as pd
from preprocessing.metafields_processor import get_metafield_keys, process_metafields

def get_collections_products(store_url: str, access_token: str, metafields: list, collections_limit: int = 100, products_limit: int = 100) -> pd.DataFrame | None:
    """
    Fetch collections and products data from Shopify API
    Args:
        store_url: Shopify store URL
        access_token: Shopify access token
        metafields: List of metafield configurations
        collections_limit: Maximum number of collections to fetch
        products_limit: Maximum number of products per collection
    Returns:
        DataFrame: Raw products data or None if error occurs
    """
    url = f"{store_url}/api/2024-01/graphql.json"
    
    headers = {
        "X-Shopify-Storefront-Access-Token": access_token,
        "Content-Type": "application/json",
    }
    
    metafields_identifiers = ",".join([
        f'{{ key: "{field["key"]}", namespace: "{field["namespace"]}" }}'
        for field in metafields
    ])
    
    query = f"""
    query {{
        collections(first: $collections) {{
            edges {{
                node {{
                    id
                    title
                    products(first: $products) {{
                        edges {{
                            node {{
                                id
                                description
                                handle
                                title
                                productType
                                tags
                                vendor
                                metafields(identifiers: [{metafields_identifiers}]) {{
                                    value
                                }}
                                variants(first: 10) {{
                                    edges {{
                                        node {{
                                            barcode
                                            availableForSale
                                            compareAtPrice {{
                                                amount
                                            }}
                                            price {{
                                                amount
                                            }}
                                            quantityAvailable
                                            selectedOptions {{
                                                name
                                            }}
                                            title
                                            unitPrice {{
                                                amount
                                            }}
                                            taxable
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    """.replace("$collections", str(collections_limit)).replace("$products", str(products_limit))
    
    try:
        response = requests.post(url, json={"query": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        flattened_data = []
        for collection_edge in data['data']['collections']['edges']:
            collection = collection_edge['node']
            collection_title = collection['title']
            for product_edge in collection['products']['edges']:
                product = product_edge['node']
                
                product_data = {
                    'product_title': product['title'],
                    'collection_title': collection_title,
                    'product_id': product['id'],
                    'vendor': product['vendor'],
                    'description': product['description'],
                    'handle': product['handle'],
                    'product_type': product['productType'],
                    'tags': product['tags'],
                    'variants': [variant['node'] for variant in product['variants']['edges']],
                    'metafields': product['metafields']
                }
                flattened_data.append(product_data)
        
        df = pd.DataFrame(flattened_data)
        return df
    
    except Exception as e:
        print(f"GraphQL Error: {e}")
        return None

if __name__ == "__main__":
    store_url = "https://missusstore.myshopify.com"
    access_token = "b7ac831b000aacb148837ca314aa1e3e"
    collections_limit = 100
    products_limit = 100
    
    result = get_collections_products(store_url, access_token, collections_limit, products_limit)
    
    if result is not None:
        print("\nSuccessfully retrieved data!")
        print(f"\nTotal products found: {len(result)}")
        print("\nSample of the data:")
        print(result[['product_title', 'collection_title', 'product_type']].head())
        
        result.to_csv("test_products_output.csv", index=False)
        print("\nFull data saved to 'test_products_output.csv'")
    else:
        print("\nFailed to retrieve data.")