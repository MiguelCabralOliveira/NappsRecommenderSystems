# data_loading/shopify/loader.py
import pandas as pd
import traceback
import os # Needed for join

# Imports RELATIVOS dentro do pacote data_loading
from ..utils import save_dataframe # Sobe para data_loading, encontra utils
from .auth import perform_napps_authentication # No mesmo dir (shopify), encontra auth
from .settings import get_shop_settings # No mesmo dir, encontra settings
from .paginator import fetch_all_product_pages # No mesmo dir, encontra paginator


# Modified function signature and logic
def load_all_shopify_products(shop_id: str, email: str, password: str, output_base_path: str) -> bool:
    """
    Orchestrates the Shopify product loading and saves the result to local/S3 path.

    Args:
        shop_id: The Shopify Shop ID.
        email: Admin email for Napps authentication.
        password: Admin password for Napps authentication.
        output_base_path: The base directory (local) or S3 URI prefix (s3://bucket/prefix).
                          Filename will be appended to this path.

    Returns:
        True if successful, False otherwise.
    """
    print(f"--- Starting Shopify Product Load for Shop ID: {shop_id} ---")
    success = False # Default to failure

    try:
        # 1. Napps Authentication & Metafield Config Fetching
        napps_access_token, metafields_config = perform_napps_authentication(email, password, shop_id)
        if not napps_access_token:
            print("Critical Error: Failed Napps Authentication. Cannot proceed.")
            return False
        # Handle case where metafields_config is None explicitly
        if metafields_config is None:
            print("Warning: Metafield configuration fetching failed or returned None. Proceeding without metafields.")
            metafields_config = []

        # 2. Get Shopify Store Settings
        store_url, storefront_access_token = get_shop_settings(shop_id)
        if not store_url or not storefront_access_token:
             print("Critical Error: Failed to get Shopify Store URL/Token. Cannot proceed.")
             return False

        # 3. Fetch All Product Pages
        combined_df = fetch_all_product_pages(
            store_url=store_url,
            access_token=storefront_access_token,
            metafields_config=metafields_config # Pass the potentially empty list
        )

        # Check if fetching failed critically
        if combined_df is None:
            print("Shopify product fetching failed critically during pagination.")
            return False # Indicate failure if pagination returned None

        # --- Save Data ---
        # Construct full path (works for both local and S3 URIs)
        products_filename = f"{shop_id}_shopify_products_raw.csv"
        products_path = os.path.join(output_base_path, products_filename).replace("\\", "/")

        print(f"Attempting to save products data to: {products_path}")
        save_dataframe(combined_df, products_path) # Use the utility function
        # save_dataframe handles empty df case, so we just need to check if fetch was ok
        success = True # Mark success if fetch process completed and didn't return None

        # --- Final Logging ---
        if combined_df.empty:
             print("Shopify product fetching completed, but no products were returned/processed.")
        else:
             print(f"Shopify product fetching complete. Total products: {len(combined_df)}")

        return success

    except Exception as e:
        print(f"\n--- An unexpected error occurred in Shopify Loader: {e} ---")
        traceback.print_exc()
        return False # Indicate failure