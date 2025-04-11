# data_loading/shopify/response_parser.py
import pandas as pd
import json
import traceback
from collections.abc import Mapping # Use Mapping for broader dict-like check

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

def _parse_single_product(product_node: dict, metafields_config: list) -> dict | None:
    """Parses a single product node from the GraphQL response."""
    if not isinstance(product_node, Mapping) or not product_node.get('id'):
        print(f"Warning: Skipping invalid or ID-missing product node: {product_node}")
        return None

    # Helper to safely get nested dicts/lists
    def safe_get(data, keys, default=None):
        _data = data
        for key in keys:
            if isinstance(_data, Mapping) and key in _data:
                _data = _data[key]
            else:
                return default
        return _data

    product_data = {
        'product_id': product_node.get('id'), # Keep the GID format initially
        'product_title': product_node.get('title'),
        'vendor': product_node.get('vendor'),
        'description': product_node.get('description'),
        'handle': product_node.get('handle'),
        'product_type': product_node.get('productType'),
        'is_gift_card': product_node.get('isGiftCard', False),
        'available_for_sale': product_node.get('availableForSale'),
        'tags': product_node.get('tags', []), # Already a list
        'createdAt': product_node.get('createdAt'),
    }

    # --- Collections ---
    collection_nodes = safe_get(product_node, ['collections', 'nodes'], [])
    product_data['collections'] = [
        { # Store structured collection info
            'id': coll.get('id'),
            'handle': coll.get('handle'),
            'title': coll.get('title')
        } for coll in collection_nodes if isinstance(coll, Mapping) and coll.get('id')
    ]

    # --- Variants ---
    variant_nodes = safe_get(product_node, ['variants', 'nodes'], [])
    product_data['variants'] = [ # Store structured variant info
        {
            'id': var.get('id'),
            'title': var.get('title'),
            'sku': var.get('sku'), # Ensure SKU is captured if available
            'currentlyNotInStock': var.get('currentlyNotInStock'),
            'availableForSale': var.get('availableForSale'),
            'price': var.get('price'), # Keep nested structure {amount, currencyCode}
            'compareAtPrice': var.get('compareAtPrice'), # Keep nested structure
            'weight': var.get('weight'),
            'weightUnit': var.get('weightUnit'), # Ensure weightUnit captured if available
            'quantityAvailable': var.get('quantityAvailable'),
            'taxable': var.get('taxable'),
            'selectedOptions': var.get('selectedOptions'), # List of {name, value}
            'unitPrice': var.get('unitPrice') # Keep nested structure
        } for var in variant_nodes if isinstance(var, Mapping) and var.get('id')
    ]

    # --- Metafields ---
    # Store metafields raw for now. Further processing might happen later.
    metafield_nodes = product_node.get('metafields', [])
    # Ensure it's a list of dictionaries before assigning
    product_data['metafields'] = [mf for mf in metafield_nodes if isinstance(mf, dict)] if isinstance(metafield_nodes, list) else []

    # --- Add Metafield Config (Optional, for context) ---
    # product_data['metafields_config_used'] = metafields_config # Might be large/redundant

    return product_data


def parse_product_page(response_data: dict, metafields_config: list) -> tuple[pd.DataFrame | None, bool, str | None]:
    """
    Parses one page of product data from the GraphQL JSON response.

    Args:
        response_data: The raw JSON dictionary for one page from the GraphQL client.
        metafields_config: The metafield configuration used for the query (for context).

    Returns:
        Tuple containing:
            - DataFrame: Parsed product data for the page (or None if parsing fails critically).
                         Returns empty DataFrame if page has no valid products.
            - bool: Whether there is a next page.
            - str | None: The end cursor for the next page (or None).
    """
    if not isinstance(response_data, Mapping) or 'data' not in response_data:
        print("Error: Invalid response data structure passed to parser (missing 'data').")
        return None, False, None # Critical failure

    # Helper to safely get nested dicts/lists
    def safe_get(data, keys, default=None):
        _data = data
        for key in keys:
            if isinstance(_data, Mapping) and key in _data:
                _data = _data[key]
            else:
                return default
        return _data

    products_root = safe_get(response_data, ['data', 'products'])
    if not isinstance(products_root, Mapping):
        print("Error: Invalid response data structure (missing 'data.products' or not a dict).")
        print(f"Data received under 'data': {response_data.get('data')}")
        return None, False, None # Critical failure

    page_info = products_root.get('pageInfo', {})
    if not isinstance(page_info, Mapping): # Ensure pageInfo is a dict
        print("Warning: 'products.pageInfo' is missing or not a dictionary. Assuming no next page.")
        page_info = {} # Default to empty dict

    has_next_page = page_info.get('hasNextPage', False)
    end_cursor = page_info.get('endCursor') # Can be None legitimately

    product_nodes = products_root.get('nodes', [])
    if not isinstance(product_nodes, list):
        print(f"Warning: 'products.nodes' is not a list. Found type: {type(product_nodes)}. Treating as empty.")
        product_nodes = [] # Treat as empty list if structure is wrong

    print(f"Parsing page with {len(product_nodes)} product nodes. Next page: {has_next_page}. Cursor: {end_cursor}")

    parsed_products = []
    for node in product_nodes:
        parsed = _parse_single_product(node, metafields_config)
        if parsed:
            parsed_products.append(parsed)

    if not parsed_products:
        print("No valid products parsed from this page.")
        # Return empty DF but still provide page info
        return pd.DataFrame(), has_next_page, end_cursor

    try:
        # Convert list of dicts to DataFrame
        df_page = pd.DataFrame(parsed_products)

        # Basic check: ensure essential columns exist after creating DataFrame
        if 'product_id' not in df_page.columns:
             print("Critical Error: 'product_id' column missing after creating DataFrame from parsed data.")
             # This shouldn't happen if _parse_single_product always adds 'product_id' for valid products
             return None, False, None # Indicate critical failure
        return df_page, has_next_page, end_cursor
    except Exception as e:
        print(f"Error creating DataFrame from parsed products: {e}")
        traceback.print_exc()
        return None, False, None # Indicate critical failure during DataFrame creation