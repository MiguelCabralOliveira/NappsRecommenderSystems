import requests
import json

def authenticate_admin(email, password):
    """
    Step 1: Authenticate as admin to get admin access token
    """
    url = "https://master.napps-solutions.com/auth/admin"
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "email": email,
        "password": password
    }
    
    print("Step 1: Authenticating as admin...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("Admin authentication successful!")
        return data
    else:
        print(f"Admin authentication failed. Status code: {response.status_code}")
        print(response.text)
        return None

def get_shop_refresh_token(admin_access_token, shop_id):
    """
    Step 2: Get shop refresh token using admin access token and shop ID
    """
    url = f"https://master.napps-solutions.com/client/{shop_id}/admin/dash"
    
    headers = {
        "accept": "application/json",
        "Authorization": admin_access_token
    }
    
    print(f"Step 2: Getting refresh token for shop {shop_id}...")
    response = requests.post(url, headers=headers)
    
    if response.status_code == 200:
        # The cookie from this response will contain the refresh token
        cookies = response.cookies
        print("Successfully obtained shop refresh token!")
        return cookies
    else:
        print(f"Failed to get shop refresh token. Status code: {response.status_code}")
        print(response.text)
        return None

def get_shop_access_token(shop_cookies):
    """
    Step 3: Use the shop refresh token (cookie) to get an access token
    """
    url = "https://master.napps-solutions.com/auth"
    
    headers = {
        "accept": "application/json"
    }
    
    print("Step 3: Getting shop access token...")
    response = requests.get(url, headers=headers, cookies=shop_cookies)
    
    if response.status_code == 200:
        data = response.json()
        print("Successfully obtained shop access token!")
        return data
    else:
        print(f"Failed to get shop access token. Status code: {response.status_code}")
        print(response.text)
        return None

def get_product_metafields(shop_access_token, shop_id, refresh_token_cookie=None):
    """
    Step 4: Get product metafields using the shop access token and all required headers
    
    Args:
        shop_access_token (str): The shop access token
        shop_id (str): The shop ID
        refresh_token_cookie (str, optional): The refresh token cookie value
    """
    # Manually construct the URL with query parameters
    url = "https://master.napps-solutions.com/v1/product/metafields?namespace="
    
    # Construct headers based on the example HTTP request
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en",
        "appid": shop_id,  # Use shop ID as appid
        "authorization": shop_access_token,
        "content-type": "application/json",
        "maintenance_override": "1",
        "priority": "u=1, i",
        "user-agent": "Python/3.x Requests Library",
        "x-requested-with": "",
    }
    
    # Set up cookies if provided
    cookies = {}
    if refresh_token_cookie:
        cookies["refresh_token"] = refresh_token_cookie
    
    print("Step 4: Getting product metafields with complete headers...")
    print(f"Request URL: {url}")
    print(f"Using AppID: {shop_id}")
    
    response = requests.get(url, headers=headers, cookies=cookies)
    
    if response.status_code == 200:
        data = response.json()
        print("Successfully retrieved product metafields!")
        return data
    else:
        print(f"Failed to get product metafields. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Try with just the essential headers if the first attempt fails
        print("Trying with essential headers only...")
        essential_headers = {
            "accept": "application/json",
            "appid": shop_id,
            "authorization": shop_access_token,
            "maintenance_override": "1",
        }
        
        alt_response = requests.get(url, headers=essential_headers, cookies=cookies)
        
        if alt_response.status_code == 200:
            alt_data = alt_response.json()
            print("Successfully retrieved product metafields with essential headers!")
            return alt_data
        else:
            print(f"Essential headers approach also failed. Status code: {alt_response.status_code}")
            print(f"Response: {alt_response.text}")
            return None

def extract_metafield_config(metafields_data):
    """
    Extract key, namespace, and type from metafields data
    """
    metafield_config = []
    
    for metafield in metafields_data:
        # Extract the relevant fields
        key = metafield.get("key", "")
        namespace = metafield.get("namespace", "")
        type_info = metafield.get("type", {})
        type_name = type_info.get("name", "")
        
        # Create a configuration entry
        config_entry = {
            "key": key,
            "namespace": namespace,
            "type": type_name
        }
        
        metafield_config.append(config_entry)
    
    print(f"Extracted {len(metafield_config)} metafield configurations")
    return metafield_config

def get_shop_settings(shop_id):
    """
    Get shop settings (store URL and storefront access token)
    """
    try:
        response = requests.get(f"https://master.napps-solutions.com/shop/v2/{shop_id}/settings")
        data = response.json()
        return data['storeUrl'], data['storeFrontAccessToken']
    except Exception as e:
        print(f"Error getting shop settings: {e}")
        return None, None

def fetch_metafields(email, password, shop_id):
    """
    Main function to fetch metafields
    """
    # Step 1: Authenticate as admin
    admin_auth_data = authenticate_admin(email, password)
    if not admin_auth_data:
        print("Authentication workflow failed at step 1")
        return None, None, None
    
    admin_access_token = admin_auth_data["accessToken"]
    print(f"Admin access token obtained: {admin_access_token[:20]}...")
    
    # Step 2: Get shop refresh token
    shop_cookies = get_shop_refresh_token(admin_access_token, shop_id)
    if not shop_cookies:
        print("Authentication workflow failed at step 2")
        return None, None, None
    
    # Get the refresh token cookie
    refresh_token_cookie = None
    for cookie in shop_cookies:
        if cookie.name == "refresh_token":
            refresh_token_cookie = cookie.value
            print(f"Refresh token cookie obtained: {refresh_token_cookie[:20]}...")
    
    # Step 3: Get shop access token
    shop_auth_data = get_shop_access_token(shop_cookies)
    if not shop_auth_data:
        print("Authentication workflow failed at step 3")
        return None, None, None
    
    # Success! Get the final shop access token
    shop_access_token = shop_auth_data.get("accessToken")
    print(f"Shop access token: {shop_access_token[:20]}...")
    
    # Step 4: Get product metafields with complete headers
    metafields_data = get_product_metafields(shop_access_token, shop_id, refresh_token_cookie)
    
    if not metafields_data:
        print("Failed to retrieve product metafields after all attempts")
        return None, None, None
    
    print(f"Found {len(metafields_data) if isinstance(metafields_data, list) else 'N/A'} metafields")
    
    # Extract metafield configurations
    metafields_config = extract_metafield_config(metafields_data)
    
    # Get store URL and access token
    store_url, storefront_token = get_shop_settings(shop_id)
    
    if not store_url or not storefront_token:
        print("Failed to retrieve shop settings")
        return metafields_config, None, None
    
    print(f"Store URL: {store_url}")
    print(f"Storefront Access Token: {storefront_token[:10]}...")
    
    return metafields_config, store_url, storefront_token