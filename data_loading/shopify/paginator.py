# data_loading/shopify/paginator.py
import pandas as pd
import time
import traceback

# Imports RELATIVOS dentro do pacote data_loading
from .client import execute_graphql_query # No mesmo dir (shopify)
from .query_builder import build_product_query # No mesmo dir
from .response_parser import parse_product_page # No mesmo dir

# --- Configuration ---
DEFAULT_PRODUCTS_PER_PAGE = 50  # Adjust based on API limits and performance
MAX_PAGES_DEFAULT = 200         # Safety limit to prevent infinite loops
DELAY_BETWEEN_REQUESTS = 0.5    # Optional delay to be kind to the API


def fetch_all_product_pages(
    store_url: str,
    access_token: str,
    metafields_config: list,
    products_limit: int = DEFAULT_PRODUCTS_PER_PAGE,
    max_pages: int = MAX_PAGES_DEFAULT,
    api_version: str = "2025-01"
) -> pd.DataFrame | None:
    """
    Fetches all pages of product data using the Shopify GraphQL API.

    Args:
        store_url: The Shopify store URL.
        access_token: Storefront access token.
        metafields_config: List of metafields to query.
        products_limit: Number of products per GraphQL request page.
        max_pages: Maximum number of pages to fetch (safety).
        api_version: Shopify API version.

    Returns:
        A combined DataFrame of all products, or None if a critical error occurs during fetch/parse.
        Returns an empty DataFrame if no products are found but process completes.
    """
    all_products_dfs = []
    has_next_page = True
    cursor = None
    page_count = 0

    print(f"\n--- Starting Shopify Product Pagination ---")
    print(f"Fetching up to {max_pages} pages, {products_limit} products per page.")

    while has_next_page:
        page_count += 1
        if page_count > max_pages:
            print(f"Reached maximum page limit ({max_pages}). Stopping pagination.")
            break

        print(f"\nFetching Page {page_count} (Cursor: {cursor})...")

        try:
            # 1. Build Query for the current page
            query, variables = build_product_query(
                metafields_config=metafields_config,
                limit=products_limit,
                cursor=cursor
            )

            # 2. Execute Query
            response_data = execute_graphql_query(
                store_url=store_url,
                access_token=access_token,
                query=query,
                variables=variables,
                api_version=api_version
            )

            # If execute_graphql_query returns None, it's a fetch failure
            if response_data is None:
                print(f"Error: Failed to fetch data for page {page_count}. Aborting pagination.")
                # Return None to signal critical failure to the caller (loader)
                return None

            # 3. Parse Response
            df_page, has_next_page, cursor = parse_product_page(response_data, metafields_config)

            # If parse_product_page returns None for df_page, it's a parsing failure
            if df_page is None:
                 print(f"Error: Failed to parse data for page {page_count}. Aborting pagination.")
                 # Return None to signal critical failure to the caller (loader)
                 return None

            # 4. Store page results (only if DataFrame is not None and not empty)
            if not df_page.empty:
                all_products_dfs.append(df_page)
                print(f"Successfully processed {len(df_page)} products from page {page_count}.")
            else:
                print(f"Page {page_count} returned or resulted in no valid product data.")

            # Safety check: If API says hasNextPage but returns no cursor, stop.
            if has_next_page and cursor is None:
                print("Warning: API indicates more pages but returned no endCursor. Stopping pagination.")
                has_next_page = False # Prevent infinite loop

            # 5. Check for loop termination
            if not has_next_page:
                print("API indicates no more pages or pagination stopped. Finishing.")
                break

            # Optional delay between requests
            if DELAY_BETWEEN_REQUESTS > 0:
                # print(f"Waiting {DELAY_BETWEEN_REQUESTS}s before next request...")
                time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"An unexpected error occurred during pagination loop (Page {page_count}): {e}")
            traceback.print_exc()
            # Return None to signal critical failure to the caller (loader)
            return None

    # --- Pagination Finished ---
    if not all_products_dfs:
        print("\nPagination complete. No product dataframes were collected.")
        return pd.DataFrame() # Return empty dataframe if nothing was fetched successfully

    print(f"\nConcatenating {len(all_products_dfs)} pages of product data...")
    try:
        combined_df = pd.concat(all_products_dfs, ignore_index=True)
        print(f"Total valid products retrieved: {len(combined_df)}")
        # Optional final validation check
        if 'product_id' not in combined_df.columns or combined_df['product_id'].isnull().any():
             print("Warning: Final combined DataFrame missing 'product_id' or contains nulls.")
        return combined_df
    except Exception as e:
        print(f"Error concatenating DataFrames: {e}")
        traceback.print_exc()
        # Return None to signal critical failure during final combination
        return None