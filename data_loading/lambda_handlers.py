# data_loading/lambda_handlers.py
import json
import os
import traceback

# Imports ABSOLUTOS a partir da raiz do Lambda (/var/task)
# Estes caminhos funcionam porque copiamos 'data_loading' para /var/task/data_loading
# e o PYTHONPATH do Lambda inclui /var/task
try:
    from data_loading.database.loader import load_all_db_data
    from data_loading.shopify.loader import load_all_shopify_products
except ImportError as e:
    # Adiciona logging extra se a importação falhar - útil para debug no Lambda
    print(f"ERROR: Failed to import loader functions. Check structure and paths in Lambda environment. Error: {e}")
    # Define stubs para permitir que o resto do ficheiro seja analisado, mas falharão se chamados
    def load_all_db_data(*args, **kwargs):
        raise ImportError("DB Loader function not available due to import error.")
    def load_all_shopify_products(*args, **kwargs):
        raise ImportError("Shopify Loader function not available due to import error.")


# --- Configuration ---
# GET S3 BUCKET NAME FROM LAMBDA ENVIRONMENT VARIABLE
S3_BUCKET_NAME = os.environ.get("OUTPUT_S3_BUCKET")
if not S3_BUCKET_NAME:
    print("CRITICAL WARNING: OUTPUT_S3_BUCKET environment variable not set. Using default 'napps-recommender'.")
    S3_BUCKET_NAME = "napps-recommender" # Use o nome correto como default

DEFAULT_DAYS_LIMIT = 90
# --- End Configuration ---

# --- Unified Handler ---
def unified_data_loader_handler(event, context):
    """
    Lambda handler that loads data based on the 'source' parameter ('database', 'shopify', 'all').
    Reads credentials from environment variables and saves output to s3://{BUCKET}/{shop_id}/raw/.
    """
    print("--- Unified Data Loader Lambda Handler Started ---")
    print(f"Event received: {json.dumps(event)}") # Dump event for better CloudWatch visibility

    # --- Extract Parameters ---
    source = event.get('source')
    shop_id = event.get('shop_id')
    days_limit = int(event.get('days_limit', DEFAULT_DAYS_LIMIT)) # Default if not provided

    # --- Input Validation ---
    if not source or source not in ['database', 'shopify', 'all']:
        print(f"ERROR: Missing or invalid required parameter 'source'. Received: '{source}'. Must be 'database', 'shopify', or 'all'.")
        return {'statusCode': 400, 'body': json.dumps("Bad Request: Missing or invalid 'source' parameter. Must be 'database', 'shopify', or 'all'.")}
    if not shop_id:
        print("ERROR: Missing required parameter 'shop_id'.")
        return {'statusCode': 400, 'body': json.dumps("Bad Request: Missing required 'shop_id' parameter.")}

    # --- Setup ---
    # --- MODIFIED: Define S3 prefix to include /raw/ ---
    # Assumes S3 bucket 'contains' the conceptual 'results' level
    output_s3_prefix = f"s3://{S3_BUCKET_NAME}/{shop_id}/raw/" # Added /raw/
    # --- END MODIFICATION ---
    print(f"Processing request for Shop ID: {shop_id}")
    print(f"Data Source(s): {source}")
    print(f"Target S3 Prefix: {output_s3_prefix}")
    if source in ['database', 'all']:
        print(f"Days Limit (for DB): {days_limit}")

    # Tracking results
    db_success = True # Assume success unless run and failed
    shopify_success = True # Assume success unless run and failed
    ran_db = False
    ran_shopify = False
    error_messages = []

    try:
        # --- Database Loading ---
        if source in ['database', 'all']:
            ran_db = True
            print("\n--- Initiating Database Loading ---")
            print("Reading DB credentials from environment variables...")
            db_config = {
                'host': os.environ.get('DB_HOST'),
                'user': os.environ.get('DB_USER'),
                'password': os.environ.get('DB_PASSWORD'),
                'name': os.environ.get('DB_NAME'),
                'port': os.environ.get('DB_PORT')
            }
            missing_db_vars = [k for k, v in db_config.items() if v is None] # Check for None
            if missing_db_vars:
                 raise ValueError(f"Missing required DB environment variables: {missing_db_vars}")
            print("DB credentials retrieved from environment.")

            # Execute DB loading function, passing the prefix ending in /raw/
            db_success = load_all_db_data(
                shop_id=shop_id,
                days_limit=days_limit,
                db_config=db_config,
                output_base_path=output_s3_prefix
            )
            print(f"--- Database Loading Finished (Success: {db_success}) ---")
            if not db_success:
                error_messages.append("Database loading step reported errors.")

        # --- Shopify Loading ---
        if source in ['shopify', 'all']:
            ran_shopify = True
            print("\n--- Initiating Shopify Product Loading ---")
            print("!!! WARNING: Shopify scraping can take significant time and MAY EXCEED the 15-minute Lambda timeout !!!")
            print("Reading Shopify credentials from environment variables...")
            email = os.environ.get('EMAIL')
            password = os.environ.get('PASSWORD')
            missing_shopify_vars = []
            if not email: missing_shopify_vars.append('EMAIL')
            if not password: missing_shopify_vars.append('PASSWORD')
            if missing_shopify_vars:
                 raise ValueError(f"Missing required Shopify environment variables: {missing_shopify_vars}")
            print("Shopify credentials retrieved from environment.")

            # Execute Shopify loading function, passing the prefix ending in /raw/
            shopify_success = load_all_shopify_products(
                shop_id=shop_id,
                email=email,
                password=password,
                output_base_path=output_s3_prefix
            )
            print(f"--- Shopify Product Loading Finished (Success: {shopify_success}) ---")
            if not shopify_success:
                error_messages.append("Shopify loading step reported errors.")

        # --- Determine Overall Status ---
        final_success = (db_success if ran_db else True) and (shopify_success if ran_shopify else True)

        if final_success:
            print("\n--- Unified Data Loader Lambda Handler Finished Successfully ---")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Data loading process completed successfully for shop {shop_id} (Source: {source}). Output at {output_s3_prefix}')
            }
        else:
            print("\n--- Unified Data Loader Lambda Handler Finished With Errors ---")
            error_summary = "; ".join(error_messages)
            return {
                'statusCode': 500, # Indicate partial or full failure
                'body': json.dumps(f'Data loading process for shop {shop_id} (Source: {source}) finished with errors: [{error_summary}]. Check logs at {output_s3_prefix} and CloudWatch.')
            }

    except ValueError as ve:
         # Catches missing env vars or bad input parameters detected before loading starts
         print(f"--- Lambda Handler Configuration or Input Error ---")
         print(f"Error: {ve}")
         traceback.print_exc() # Log traceback for config errors too
         return {'statusCode': 400, 'body': json.dumps(f'Configuration or Bad Request Error: {ve}')}
    except ImportError as imp_err:
         # Catches the import error from the top try/except block if stubs were called
         print(f"--- Lambda Handler Critical Import Error ---")
         print(f"Error: {imp_err}")
         traceback.print_exc()
         return {'statusCode': 500, 'body': json.dumps(f'Critical error: Could not import necessary loader functions. Check deployment package. Error: {imp_err}')}
    except Exception as e:
        # Catches unexpected errors during the loading process itself
        print(f"--- Unified Data Loader Lambda Handler Failed Critically ---")
        print(f"Unexpected Error: {e}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps(f'Critical unexpected error during data loading for shop {shop_id} (Source: {source}): {e}. Check CloudWatch logs.')
        }

# --- Deprecated Handlers (Optional: Keep or Remove) ---
def db_loader_handler(event, context):
     print("WARNING: db_loader_handler is deprecated. Use unified_data_loader_handler.")
     return {'statusCode': 410, 'body': json.dumps("This handler is deprecated. Use unified_data_loader_handler with source='database'.")}

def shopify_trigger_handler(event, context):
     print("WARNING: shopify_trigger_handler is deprecated. Use unified_data_loader_handler.")
     return {'statusCode': 410, 'body': json.dumps("This handler is deprecated. Use unified_data_loader_handler with source='shopify'.")}