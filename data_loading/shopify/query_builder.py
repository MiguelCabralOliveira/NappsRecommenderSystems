# data_loading/shopify/query_builder.py
import json

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

# --- GraphQL Query Template ---
# This template is based *exactly* on the query structure provided by the user
# from their working `get_shopify_products` function.
# It uses GraphQL variables ($limit, $cursor) instead of f-string interpolation for those.
# It retains the .format() placeholder ONLY for {metafields_identifiers}.
_PRODUCT_QUERY_TEMPLATE = """
query ($country: CountryCode, $language: LanguageCode, $limit: Int!, $cursor: String)
@inContext(country: $country, language: $language) {{
  products(first: $limit, after: $cursor) {{
    nodes {{
      # --- Fields from User's Working Query ---
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
          nodes {{
            ... on Metaobject {{
              id
              type
              # handle # Optional: Add back if needed by parser
              fields {{
                key
                type
                value
                # Optional: Add back reference handling if needed by parser
                # reference {{
                #   ... on MediaImage {{
                #     id
                #     image {{ url }}
                #   }}
                # }}
              }}
            }}
          }}
        }}
      }}
      isGiftCard
      productType
      description
      collections(first: 250) {{ # Limit from user query
        nodes {{
          id
          handle
          title
        }}
      }}
      variants(first: 250) {{ # Limit from user query
        nodes {{
          id
          title
          currentlyNotInStock
          availableForSale
          price {{
            amount
            # currencyCode # Optional: Add if needed by parser
          }}
          compareAtPrice {{
            amount
            # currencyCode # Optional: Add if needed by parser
          }}
          weight
          # weightUnit # Optional: Add if needed by parser
          quantityAvailable
          taxable
          sku # Included SKU
          selectedOptions {{
            name
            value # User's original code parsing relies on name+value
          }}
          unitPrice {{
            amount
            # currencyCode # Optional: Add if needed by parser
          }}
        }}
      }}
      # --- End of Fields ---
    }}
    pageInfo {{
      hasNextPage
      endCursor
    }}
  }}
}}
"""
# --- End Query Template ---


def build_product_query(metafields_config: list, limit: int = 50, cursor: str = None) -> tuple[str, dict]:
    """
    Builds the GraphQL query string and variables dictionary for fetching products,
    using the template derived from the user's working query.

    Args:
        metafields_config: List of dicts [{'namespace': 'ns', 'key': 'k'}, ...]
                           to identify which metafields to fetch. Can be empty or None.
        limit: Number of products per page.
        cursor: Pagination cursor for the 'after' argument.

    Returns:
        tuple: (query_string, variables_dict)
    """
    # Format metafield identifiers for the query
    metafields_identifiers_str = ""
    if metafields_config: # Check if list is not None and not empty
        identifiers = []
        for mf in metafields_config:
            # Ensure mf is a dict and has the required keys with non-empty values
            if isinstance(mf, dict) and mf.get('namespace') and mf.get('key'):
                 # Basic sanitation for namespace/key if they contain quotes
                 ns = str(mf['namespace']).replace('"', '\\"')
                 key = str(mf['key']).replace('"', '\\"')
                 identifiers.append(f'{{ namespace: "{ns}", key: "{key}" }}')
            else:
                 print(f"Skipping invalid metafield config item: {mf}")
        metafields_identifiers_str = ",".join(identifiers)
    else:
        print("No valid metafield configuration provided, fetching products without specific metafields.")
        # If metafields_config is empty/None, ensure the identifiers list is empty in the query
        # The template might need adjustment if metafields() with empty identifiers is invalid GraphQL
        # Assuming empty identifiers list `metafields(identifiers: [])` is valid:
        pass # metafields_identifiers_str remains "" which results in `identifiers: []`

    # Inject ONLY the formatted metafield identifiers into the template using .format()
    try:
        # Ensure the placeholder exists exactly once
        if "{metafields_identifiers}" not in _PRODUCT_QUERY_TEMPLATE:
             raise ValueError("Placeholder '{metafields_identifiers}' not found in _PRODUCT_QUERY_TEMPLATE.")
        if _PRODUCT_QUERY_TEMPLATE.count("{metafields_identifiers}") > 1:
             raise ValueError("Placeholder '{metafields_identifiers}' found multiple times in _PRODUCT_QUERY_TEMPLATE.")

        query = _PRODUCT_QUERY_TEMPLATE.format(metafields_identifiers=metafields_identifiers_str)

    except (KeyError, ValueError) as e:
         print(f"ERROR: Query template formatting failed: {e}")
         print("Please double-check the _PRODUCT_QUERY_TEMPLATE string.")
         raise # Re-raise the error to stop execution

    # Define query variables for $limit and $cursor (and context)
    variables = {
        "limit": limit, # Use the passed limit argument
        # "cursor" will be added only if not None
        "country": "US", # Default context from user's code - MAKE CONFIGURABLE if needed
        "language": "EN"  # Default context from user's code - MAKE CONFIGURABLE if needed
    }
    # Add cursor variable only if it's not None
    if cursor:
        variables["cursor"] = cursor

    return query, variables