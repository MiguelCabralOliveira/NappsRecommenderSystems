# data_loading/lambda_handlers.py
import json
import os
import traceback

# --- Use RELATIVE imports within the data_loading package ---
# The '.' refers to the current package level (data_loading)

from .database.loader import load_all_db_data
# If you uncomment and use the Shopify functionality in a Lambda, use relative import:
# from .shopify.loader import load_all_shopify_products
# If you uncomment and use utils directly here, use relative import:
# from .utils import save_dataframe


# --- Configuration ---
# GET S3 BUCKET NAME FROM LAMBDA ENVIRONMENT VARIABLE
S3_BUCKET_NAME = os.environ.get("OUTPUT_S3_BUCKET")
if not S3_BUCKET_NAME:
    print("WARNING: OUTPUT_S3_BUCKET environment variable not set. Using default 'nappsrecommender'.")
    S3_BUCKET_NAME = "napps-recommender" # Make sure this default is correct for your setup

DEFAULT_DAYS_LIMIT = 90


# --- Handler for Database Loading ---
def db_loader_handler(event, context):
    """
    Lambda handler specifically for loading database data.
    Reads credentials directly from Lambda environment variables.
    Uses relative imports for internal modules.
    """
    print("--- DB Loader Lambda Handler Started ---")
    print(f"Event received: {json.dumps(event)}") # Log the event safely

    try:
        # Extract parameters from event
        shop_id = event.get('shop_id')
        days_limit = int(event.get('days_limit', DEFAULT_DAYS_LIMIT))

        if not shop_id:
            raise ValueError("Missing required parameter 'shop_id'.")

        # Construct the S3 output path prefix (using forward slashes is safer for S3)
        # Ensure no double slashes if S3_BUCKET_NAME has a trailing slash
        output_s3_prefix = f"s3://{S3_BUCKET_NAME.rstrip('/')}/{shop_id}/"
        print(f"Shop ID: {shop_id}")
        print(f"Target S3 Prefix: {output_s3_prefix}")
        print(f"Days Limit: {days_limit}")

        # --- Database Loading ---
        print("Reading DB credentials from environment variables...")
        db_config = {
            'host': os.environ.get('DB_HOST'),
            'user': os.environ.get('DB_USER'),
            'password': os.environ.get('DB_PASSWORD'),
            'name': os.environ.get('DB_NAME'),
            'port': os.environ.get('DB_PORT')
        }
        # Validate that required variables are present
        missing_db_vars = [k for k, v in db_config.items() if v is None]
        if missing_db_vars:
             raise ValueError(f"Missing required DB environment variables: {missing_db_vars}")

        print("DB credentials read from environment.")

        # Call the imported function (using the corrected relative import)
        db_success = load_all_db_data(
            shop_id=shop_id,
            days_limit=days_limit,
            db_config=db_config,
            output_base_path=output_s3_prefix # Pass the S3 URI prefix
        )

        # --- Determine Status ---
        if db_success:
            print("--- DB Loader Lambda Handler Finished Successfully ---")
            return {'statusCode': 200, 'body': json.dumps(f'Database loading complete for shop {shop_id}. Output at {output_s3_prefix}')}
        else:
            # The function load_all_db_data should log its own errors, so we just report failure here.
            print("--- DB Loader Lambda Handler Finished With Errors (check function logs) ---")
            return {'statusCode': 500, 'body': json.dumps(f'Database loading for shop {shop_id} finished with errors. Check detailed logs.')}

    except ValueError as ve:
         # Catch specific configuration/input errors
         print(f"--- Lambda Handler Input/Config Error ---")
         print(f"Error: {ve}")
         traceback.print_exc() # Log stack trace for debugging
         return {'statusCode': 400, 'body': json.dumps(f'Bad Request or Configuration Error: {ve}')}
    except Exception as e:
        # Catch any other unexpected errors during handler execution
        print(f"--- DB Loader Lambda Handler Failed Critically ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return {'statusCode': 500, 'body': json.dumps(f'Critical error during DB data loading: {e}')}


# --- Handler for Shopify Loading Trigger ---
def shopify_trigger_handler(event, context):
    """
    Lambda handler to INITIATE Shopify loading (placeholder).
    Reads credentials directly from Lambda environment variables.
    Uses relative imports if internal Shopify modules were called.
    """
    print("--- Shopify Trigger Lambda Handler Started ---")
    print(f"Event received: {json.dumps(event)}") # Log the event safely

    try:
        shop_id = event.get('shop_id')
        if not shop_id:
            raise ValueError("Missing required parameter 'shop_id'.")

        # Construct the S3 output path prefix (using forward slashes is safer for S3)
        output_s3_prefix = f"s3://{S3_BUCKET_NAME.rstrip('/')}/{shop_id}/"
        print(f"Shop ID: {shop_id}")
        print(f"Target S3 Prefix for external process: {output_s3_prefix}")

        # --- Initiate Shopify Loading ---
        print("Reading Shopify credentials from environment variables...")
        email = os.environ.get('EMAIL')
        password = os.environ.get('PASSWORD')
        # Validate that required variables are present
        missing_shopify_vars = []
        if not email: missing_shopify_vars.append('EMAIL')
        if not password: missing_shopify_vars.append('PASSWORD')
        if missing_shopify_vars:
             raise ValueError(f"Missing required Shopify environment variables: {missing_shopify_vars}")

        print("Shopify credentials read from environment.")

        # --- Placeholder: Trigger External Process ---
        # This is where you would add code (e.g., using boto3) to start
        # a Step Function, ECS Task, Batch Job, or another Lambda
        # to perform the actual Shopify data loading.
        # Pass the necessary parameters securely.
        print("ACTION REQUIRED: Implement logic to trigger external process (e.g., Step Function, Batch Job) for Shopify product fetch.")
        print("Parameters to pass:")
        print(f"  shop_id: {shop_id}")
        print(f"  email: {email}") # Consider if email needs to be passed or if target process can get it
        print(f"  password: [REDACTED - Pass securely or have target fetch it]")
        print(f"  output_s3_uri: {output_s3_prefix}")

        # Example: If you were calling load_all_shopify_products directly (NOT recommended for long tasks):
        # shopify_success = load_all_shopify_products(
        #     shop_id=shop_id,
        #     email=email,
        #     password=password, # Pass password securely if needed
        #     output_base_path=output_s3_prefix
        # )
        # triggered_successfully = shopify_success # Or based on trigger call status

        # Assuming the trigger call itself was successful (replace with actual status check)
        triggered_successfully = True

        # --- Determine Status based on Trigger call ---
        if triggered_successfully:
            print("--- Shopify Trigger Lambda Handler Finished Successfully (Process Initiated) ---")
            return {'statusCode': 200, 'body': json.dumps(f'Shopify load initiation request processed for shop {shop_id}. Monitor external process. Target output: {output_s3_prefix}')}
        else:
            print("--- Shopify Trigger Lambda Handler Finished But Trigger Failed ---")
            return {'statusCode': 500, 'body': json.dumps(f'Shopify load initiation failed for shop {shop_id}. Check trigger logic logs.')}

    except ValueError as ve:
         # Catch specific configuration/input errors
         print(f"--- Lambda Handler Input/Config Error ---")
         print(f"Error: {ve}")
         traceback.print_exc() # Log stack trace for debugging
         return {'statusCode': 400, 'body': json.dumps(f'Bad Request or Configuration Error: {ve}')}
    except Exception as e:
        # Catch any other unexpected errors during handler execution
        print(f"--- Shopify Trigger Lambda Handler Failed Critically ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return {'statusCode': 500, 'body': json.dumps(f'Critical error initiating Shopify load: {e}')}