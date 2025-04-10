# data_loading/shopify/client.py
import requests
import json
import traceback
import time

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

def execute_graphql_query(store_url: str, access_token: str, query: str, variables: dict = None, api_version: str = "2025-01", retries=3, delay=5) -> dict | None:
    """
    Executes a GraphQL query against the Shopify Storefront API.

    Args:
        store_url: The base URL of the Shopify store (e.g., https://your-shop.myshopify.com).
        access_token: The Storefront API access token.
        query: The GraphQL query string.
        variables: A dictionary of variables for the query (optional).
        api_version: The Shopify API version to target.
        retries: Number of times to retry on potentially transient errors (like 5xx).
        delay: Seconds to wait between retries.

    Returns:
        The JSON response data as a dictionary, or None if a persistent error occurs.
    """
    if not store_url or not store_url.startswith("https://"):
        print(f"Error: Invalid store_url provided: {store_url}")
        return None
    endpoint = f"{store_url.rstrip('/')}/api/{api_version}/graphql.json"
    headers = {
        "X-Shopify-Storefront-Access-Token": access_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    last_exception = None
    for attempt in range(retries):
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=60) # Add timeout

            # Check for common Shopify rate limit response (though storefront is less prone)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", delay))
                print(f"Rate limit hit (429). Retrying after {retry_after}s... (Attempt {attempt + 1}/{retries})")
                time.sleep(retry_after)
                continue # Skip increasing delay manually if header provides value

            # Check for server errors that might be transient
            if response.status_code >= 500:
                 print(f"Server error ({response.status_code}). Retrying after {delay}s... (Attempt {attempt + 1}/{retries})")
                 print(f"Response: {response.text[:200]}...")
                 time.sleep(delay)
                 delay *= 2 # Exponential backoff for server errors
                 continue

            response.raise_for_status() # Raise HTTPError for other bad responses (4xx excluding 429 handled above)

            data = response.json()

            # Check for GraphQL-level errors
            if 'errors' in data:
                print(f"GraphQL API returned errors:")
                try:
                    print(json.dumps(data['errors'], indent=2))
                except Exception:
                    print(data['errors'])
                # Consider specific errors non-fatal if needed, for now treat all as failure
                return None # Indicate failure due to GraphQL errors

            # Check if 'data' key exists - vital for successful response
            if 'data' not in data:
                 print("Error: GraphQL response missing 'data' key.")
                 print(f"Response: {json.dumps(data, indent=2)}")
                 return None # Indicate unexpected response structure

            return data # Success

        except requests.exceptions.Timeout as e:
            print(f"Request timed out. Retrying after {delay}s... (Attempt {attempt + 1}/{retries})")
            last_exception = e
            time.sleep(delay)
            delay *= 2
            continue
        except requests.exceptions.RequestException as e:
            # Network errors, connection errors etc. - potentially retryable
            print(f"Network error during GraphQL request: {e}. Retrying after {delay}s... (Attempt {attempt + 1}/{retries})")
            last_exception = e
            time.sleep(delay)
            delay *= 2
            continue
        except json.JSONDecodeError as e:
             print(f"Error decoding JSON response from GraphQL endpoint: {e}")
             print(f"Response status: {response.status_code}, text: {response.text[:200]}...")
             last_exception = e
             break # JSON decode errors are usually not retryable
        except Exception as e:
            print(f"An unexpected error occurred during GraphQL request: {e}")
            traceback.print_exc()
            last_exception = e
            break # Break on other unexpected errors

    print(f"GraphQL request failed after {retries} attempts.")
    if last_exception:
        print(f"Last error: {last_exception}")
    return None