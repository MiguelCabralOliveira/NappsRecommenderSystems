import requests

def get_shop_settings(shop_id: str) -> tuple[str, str] | None:
    try:
        response = requests.get(f"https://master.napps-solutions.com/shop/v2/{shop_id}/settings")
        data = response.json()
        return data['storeUrl'], data['storeFrontAccessToken']
    except Exception as e:
        print(f"Error: {e}")
        return None