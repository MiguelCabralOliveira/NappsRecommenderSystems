import requests
import pandas as pd

# Fetch all collections and products using shopify storefront api
# Currently fetching : Collections, Products, Variants, Options, Tags
def get_collections_products(store_url: str, access_token: str, collections_limit: int = 10, products_limit: int = 100) -> pd.DataFrame | None:
    url = f"{store_url}/api/2024-01/graphql.json"
    
    headers = {
        "X-Shopify-Storefront-Access-Token": access_token,
        "Content-Type": "application/json",
    }
    
    query = """
    query {
        collections(first: $collections) {
            edges {
                node {
                    id
                    title
                    products(first: $products) {
                        edges {
                            node {
                                id
                                description
                                handle
                                title
                                productType
                                tags
                                variants(first: 10) {
                                    edges {
                                    node {
                                        barcode
                                        availableForSale
                                        compareAtPrice {
                                        amount
                                        }
                                        price {
                                        amount
                                        }
                                        quantityAvailable
                                        selectedOptions {
                                        name
                                        }
                                        title
                                        unitPrice {
                                        amount
                                        }
                                        taxable
                                    }
                                    }
                                }
                                options(first: 10) {
                                    name
                                    values
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """.replace("$collections", str(collections_limit)).replace("$products", str(products_limit))
    
    try:
        response = requests.post(url, json={"query": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Flatten the data
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
                    'description': product['description'],
                    'handle': product['handle'],
                    'product_type': product['productType'],
                    'tags': product['tags'],
                    'variants': [variant['node'] for variant in product['variants']['edges']],
                    'options': product['options']
                }
                flattened_data.append(product_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_data)
        return df
    
    except Exception as e:
        print(f"GraphQL Error: {e}")
        return None