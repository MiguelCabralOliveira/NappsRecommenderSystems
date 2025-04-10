# data_loading/shopify/auth.py
import requests
import json
import traceback
import os

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

# URL constants (consider moving to a config file or env variables)
NAPPS_AUTH_ADMIN_URL = os.getenv("NAPPS_AUTH_ADMIN_URL", "https://master.napps-solutions.com/auth/admin")
NAPPS_DASH_URL_TPL = os.getenv("NAPPS_DASH_URL_TPL", "https://master.napps-solutions.com/client/{shop_id}/admin/dash")
NAPPS_AUTH_URL = os.getenv("NAPPS_AUTH_URL", "https://master.napps-solutions.com/auth")
NAPPS_METAFIELDS_URL = os.getenv("NAPPS_METAFIELDS_URL", "https://master.napps-solutions.com/v1/product/metafields?namespace=")


def authenticate_admin(email, password):
    """Step 1: Authenticate as admin to get admin access token."""
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"email": email, "password": password}
    print("Step 1: Authenticating as admin...")
    try:
        response = requests.post(NAPPS_AUTH_ADMIN_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        print("Admin authentication successful!")
        return data.get("accessToken")
    except requests.exceptions.RequestException as e:
        print(f"Admin authentication failed (Network/Request Error): {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}, text: {e.response.text[:200]}...")
        # traceback.print_exc() # Optional: less verbose for auth failure
        return None
    except Exception as e:
        print(f"Admin authentication failed (Other Error): {e}")
        # traceback.print_exc() # Optional: less verbose
        return None


def get_shop_refresh_token_cookie(admin_access_token, shop_id):
    """Step 2: Get shop refresh token cookie using admin access token."""
    url = NAPPS_DASH_URL_TPL.format(shop_id=shop_id)
    headers = {"accept": "application/json", "Authorization": f"Bearer {admin_access_token}"} # Use Bearer token standard
    print(f"Step 2: Getting refresh token cookie for shop {shop_id}...")
    try:
        response = requests.post(url, headers=headers, timeout=30)
        response.raise_for_status()
        cookies = response.cookies
        refresh_token_cookie = cookies.get("refresh_token")
        if refresh_token_cookie:
            print("Successfully obtained shop refresh token cookie!")
            return refresh_token_cookie, cookies # Return value and full cookie jar
        else:
            print("Failed to find refresh_token in response cookies.")
            print(f"Cookies received: {cookies}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Failed to get shop refresh token (Network/Request Error): {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}, text: {e.response.text[:200]}...")
        # traceback.print_exc()
        return None, None
    except Exception as e:
        print(f"Failed to get shop refresh token (Other Error): {e}")
        # traceback.print_exc()
        return None, None


def get_shop_access_token(shop_cookies):
    """Step 3: Use the shop refresh token (cookie jar) to get an access token."""
    headers = {"accept": "application/json"}
    print("Step 3: Getting shop access token...")
    try:
        # Pass the whole cookie jar from step 2
        response = requests.get(NAPPS_AUTH_URL, headers=headers, cookies=shop_cookies, timeout=30)
        response.raise_for_status()
        data = response.json()
        access_token = data.get("accessToken")
        if access_token:
            print("Successfully obtained shop access token!")
            return access_token
        else:
            print("Failed to find accessToken in response.")
            print(f"Data received: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to get shop access token (Network/Request Error): {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}, text: {e.response.text[:200]}...")
        # traceback.print_exc()
        return None
    except Exception as e:
        print(f"Failed to get shop access token (Other Error): {e}")
        # traceback.print_exc()
        return None


def get_product_metafields_config(shop_access_token, shop_id, refresh_token_cookie_value):
    """Step 4: Get product metafield definitions."""
    headers = {
        "accept": "application/json",
        "appid": shop_id,
        "authorization": f"Bearer {shop_access_token}", # Use Bearer token standard
        "maintenance_override": "1",
    }
    cookies = {"refresh_token": refresh_token_cookie_value} if refresh_token_cookie_value else {}

    print("Step 4: Getting product metafield definitions...")
    try:
        # Ensure NAPPS_METAFIELDS_URL is correctly formed (ends with namespace=)
        if not NAPPS_METAFIELDS_URL.endswith("namespace="):
             print(f"Warning: NAPPS_METAFIELDS_URL ({NAPPS_METAFIELDS_URL}) doesn't end with 'namespace='. Appending it.")
             target_url = NAPPS_METAFIELDS_URL + "?namespace=" if '?' not in NAPPS_METAFIELDS_URL else NAPPS_METAFIELDS_URL + "&namespace="
        else:
             target_url = NAPPS_METAFIELDS_URL

        response = requests.get(target_url, headers=headers, cookies=cookies, timeout=45)
        response.raise_for_status()
        metafields_data = response.json()

        if not isinstance(metafields_data, list):
             print(f"Warning: Expected list from metafields endpoint, got {type(metafields_data)}")
             print(f"Data received: {metafields_data}")
             return [] # Return empty list if not a list

        print(f"Successfully retrieved {len(metafields_data)} potential metafield definitions.")
        # Extract key, namespace, type
        metafield_config = []
        valid_count = 0
        for metafield in metafields_data:
            if isinstance(metafield, dict):
                key = metafield.get("key")
                namespace = metafield.get("namespace")
                # Type can be complex, just ensure it exists for now
                type_info = metafield.get("type", {})
                type_name = type_info.get("name") if isinstance(type_info, dict) else None

                # Basic validation: require key and namespace
                if key and namespace:
                     metafield_config.append({
                         "key": key,
                         "namespace": namespace,
                         # Store the raw type object or name for flexibility later
                         "type": type_name if type_name else type_info # Store name if available, else raw dict/value
                     })
                     valid_count += 1
                else:
                    print(f"Skipping metafield definition due to missing key or namespace: {metafield}")

        print(f"Extracted {valid_count} valid metafield configurations.")
        return metafield_config

    except requests.exceptions.RequestException as e:
        print(f"Failed to get product metafields (Network/Request Error): {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}, text: {e.response.text[:200]}...")
        # traceback.print_exc()
        return [] # Return empty list on error
    except Exception as e:
        print(f"Failed to get product metafields (Other Error): {e}")
        # traceback.print_exc()
        return []


def perform_napps_authentication(email: str, password: str, shop_id: str) -> tuple[str | None, list | None]:
    """
    Orchestrates the Napps authentication flow to get shop access token and metafield config.

    Returns:
        tuple: (shop_access_token, metafield_config_list) or (None, None) if critical auth fails.
               Metafield config can be None or [] if that specific step fails but auth succeeds.
    """
    print("\n--- Starting Napps Authentication Flow ---")
    admin_token = authenticate_admin(email, password)
    if not admin_token:
        print("Napps Auth Flow Failed at Step 1 (Admin Auth).")
        return None, None

    refresh_cookie_val, shop_cookies = get_shop_refresh_token_cookie(admin_token, shop_id)
    if not refresh_cookie_val or not shop_cookies:
        print("Napps Auth Flow Failed at Step 2 (Refresh Token).")
        return None, None

    shop_access_token = get_shop_access_token(shop_cookies)
    if not shop_access_token:
        print("Napps Auth Flow Failed at Step 3 (Shop Access Token).")
        return None, None

    # Metafield config is non-critical for returning the auth token
    metafield_config = get_product_metafields_config(shop_access_token, shop_id, refresh_cookie_val)
    if metafield_config is None: # Check specifically for None vs empty list
        print("Warning: Could not retrieve metafield configurations (Error occurred).")
        metafield_config = [] # Default to empty list if fetch failed
    elif not metafield_config:
        print("Info: No metafield configurations found or extracted.")

    print("--- Napps Authentication Flow Completed Successfully (Token Obtained) ---")
    return shop_access_token, metafield_config