import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def authenticate_admin():
    """
    Step 1: Authenticate as admin to get admin access token
    """
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    
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

def main():
    # Check if required environment variables are set
    if not os.getenv("EMAIL") or not os.getenv("PASSWORD"):
        print("Error: EMAIL and PASSWORD must be set in .env file")
        return
    
    # Get shop ID from environment or prompt
    shop_id = os.getenv("SHOP_ID")
    if not shop_id:
        shop_id = input("Enter the Shop ID: ")
    
    # Step 1: Authenticate as admin
    admin_auth_data = authenticate_admin()
    if not admin_auth_data:
        print("Authentication workflow failed at step 1")
        return
    
    admin_access_token = admin_auth_data["accessToken"]
    print(f"Admin access token obtained: {admin_access_token[:20]}...")
    
    # Step 2: Get shop refresh token
    shop_cookies = get_shop_refresh_token(admin_access_token, shop_id)
    if not shop_cookies:
        print("Authentication workflow failed at step 2")
        return
    
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
        return
    
    # Success! Get the final shop access token
    shop_access_token = shop_auth_data.get("accessToken")
    print(f"Shop access token: {shop_access_token[:20]}...")
    
    # Step 4: Get product metafields with complete headers
    metafields = get_product_metafields(shop_access_token, shop_id, refresh_token_cookie)
    
    if not metafields:
        print("Failed to retrieve product metafields after all attempts")
        return
    
    # Save the metafields to a file
    with open("product_metafields.json", "w") as f:
        json.dump(metafields, f, indent=2)
    
    print("\n===== Workflow Completed Successfully =====")
    print(f"Found {len(metafields) if isinstance(metafields, list) else 'N/A'} metafields")
    print("Product metafields saved to product_metafields.json")
    
    # Also save the tokens to a file for later use
    output = {
        "admin": admin_auth_data,
        "shop": shop_auth_data
    }
    
    with open("auth_tokens.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Authentication tokens saved to auth_tokens.json")

if __name__ == "__main__":
    main()