# data_loading/shopify/settings.py
import requests
import traceback
import os

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

# URL constant (consider moving to config or env var)
NAPPS_SETTINGS_URL_TPL = os.getenv("NAPPS_SETTINGS_URL_TPL", "https://master.napps-solutions.com/shop/v2/{shop_id}/settings")

def get_shop_settings(shop_id: str) -> tuple[str | None, str | None]:
    """
    Get shop settings (store URL and storefront access token) from Napps backend.
    Returns:
        tuple: (store_url, storefront_access_token) or (None, None) on failure.
    """
    url = NAPPS_SETTINGS_URL_TPL.format(shop_id=shop_id)
    headers = {
        'maintenance_override': '1' # Seems important based on original code
    }
    print(f"Fetching shop settings from: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()

        # Explicitly check for keys and non-empty values
        store_url = data.get('storeUrl')
        storefront_token = data.get('storeFrontAccessToken')

        if store_url and storefront_token:
            print("Successfully retrieved shop settings.")
            # Basic validation
            if not store_url.startswith("https://"):
                 print(f"Warning: Store URL '{store_url}' does not start with https://.")
                 # Decide if this is fatal - for now, allow it but warn.
            return store_url, storefront_token
        else:
            missing_info = []
            if not store_url: missing_info.append("'storeUrl'")
            if not storefront_token: missing_info.append("'storeFrontAccessToken'")
            print(f"Error: Missing or empty required fields in settings response: {', '.join(missing_info)}.")
            print(f"Received data: {data}")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Error getting shop settings (Network/Request Error): {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}, text: {e.response.text[:200]}...")
        # traceback.print_exc()
        return None, None
    except Exception as e:
        print(f"Error getting shop settings (Other Error): {e}")
        # traceback.print_exc()
        return None, None