# data_loading/lambda_handlers.py
import json
import os
import traceback

# Imports ABSOLUTOS a partir da raiz do Lambda (/var/task)
# O Lambda encontrará 'database/loader.py' em /var/task/database/loader.py
from database.loader import load_all_db_data
# Se precisasse de shopify: from shopify.loader import load_all_shopify_products
# Se precisasse de utils: from utils import save_dataframe


# --- Configuration ---
# GET S3 BUCKET NAME FROM LAMBDA ENVIRONMENT VARIABLE
S3_BUCKET_NAME = os.environ.get("OUTPUT_S3_BUCKET")
if not S3_BUCKET_NAME:
    print("WARNING: OUTPUT_S3_BUCKET environment variable not set. Using default 'nappsrecommender'.")
    S3_BUCKET_NAME = "napps-recommender" # Use o nome correto como default

DEFAULT_DAYS_LIMIT = 90

# --- Handler for Database Loading ---
def db_loader_handler(event, context):
    """
    Lambda handler specifically for loading database data.
    Reads credentials directly from Lambda environment variables.
    """
    print("--- DB Loader Lambda Handler Started ---")
    print(f"Event received: {event}")

    try:
        # Extract parameters from event
        shop_id = event.get('shop_id')
        days_limit = int(event.get('days_limit', DEFAULT_DAYS_LIMIT))

        if not shop_id:
            raise ValueError("Missing required parameter 'shop_id'.")

        # Construct the S3 output path prefix
        output_s3_prefix = f"s3://{S3_BUCKET_NAME}/{shop_id}/"
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
        missing_db_vars = [k for k, v in db_config.items() if v is None]
        if missing_db_vars:
             raise ValueError(f"Missing required DB environment variables: {missing_db_vars}")

        print("DB credentials read from environment.")

        # Chama a função importada com caminho absoluto (relativo a /var/task)
        db_success = load_all_db_data(
            shop_id=shop_id,
            days_limit=days_limit,
            db_config=db_config,
            output_base_path=output_s3_prefix
        )

        # --- Determine Status ---
        if db_success:
            print("--- DB Loader Lambda Handler Finished Successfully ---")
            return {'statusCode': 200, 'body': json.dumps(f'Database loading complete for shop {shop_id}. Output at {output_s3_prefix}')}
        else:
            print("--- DB Loader Lambda Handler Finished With Errors ---")
            return {'statusCode': 500, 'body': json.dumps(f'Database loading for shop {shop_id} finished with errors. Check logs.')}

    except ValueError as ve:
         print(f"--- Lambda Handler Input/Config Error ---")
         print(f"Error: {ve}")
         return {'statusCode': 400, 'body': json.dumps(f'Bad Request or Configuration Error: {ve}')}
    except Exception as e:
        print(f"--- DB Loader Lambda Handler Failed Critically ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return {'statusCode': 500, 'body': json.dumps(f'Critical error during DB data loading: {e}')}


# --- Handler for Shopify Loading Trigger ---
def shopify_trigger_handler(event, context):
    """
    Lambda handler to INITIATE Shopify loading.
    Reads credentials directly from Lambda environment variables.
    """
    print("--- Shopify Trigger Lambda Handler Started ---")
    print(f"Event received: {event}")

    try:
        shop_id = event.get('shop_id')
        if not shop_id:
            raise ValueError("Missing required parameter 'shop_id'.")

        output_s3_prefix = f"s3://{S3_BUCKET_NAME}/{shop_id}/"
        print(f"Shop ID: {shop_id}")
        print(f"Target S3 Prefix for external process: {output_s3_prefix}")

        # --- Initiate Shopify Loading ---
        print("Reading Shopify credentials from environment variables...")
        email = os.environ.get('EMAIL')
        password = os.environ.get('PASSWORD')
        missing_shopify_vars = []
        if not email: missing_shopify_vars.append('EMAIL')
        if not password: missing_shopify_vars.append('PASSWORD')
        if missing_shopify_vars:
             raise ValueError(f"Missing required Shopify environment variables: {missing_shopify_vars}")

        print("Shopify credentials read from environment.")

        # --- Placeholder: Trigger External Process ---
        print("ACTION REQUIRED: Trigger external process (e.g., Step Function, Batch Job) for Shopify product fetch.")
        print("Parameters to pass:")
        print(f"  shop_id: {shop_id}")
        print(f"  email: {email}")
        print(f"  password: [REDACTED - Pass securely or have target fetch it]")
        print(f"  output_s3_uri: {output_s3_prefix}")
        # Add logic here to actually trigger the process (e.g., boto3 call)
        shopify_triggered = True # Assume trigger logic is added here

        # --- Determine Status ---
        if shopify_triggered:
            print("--- Shopify Trigger Lambda Handler Finished Successfully ---")
            return {'statusCode': 200, 'body': json.dumps(f'Shopify load initiation request processed for shop {shop_id}. Check external process status. Target output: {output_s3_prefix}')}
        else:
            print("--- Shopify Trigger Lambda Handler Finished But Trigger Failed ---")
            return {'statusCode': 500, 'body': json.dumps(f'Shopify load initiation failed for shop {shop_id}. Check logs.')}

    except ValueError as ve:
         print(f"--- Lambda Handler Input/Config Error ---")
         print(f"Error: {ve}")
         return {'statusCode': 400, 'body': json.dumps(f'Bad Request or Configuration Error: {ve}')}
    except Exception as e:
        print(f"--- Shopify Trigger Lambda Handler Failed Critically ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return {'statusCode': 500, 'body': json.dumps(f'Critical error initiating Shopify load: {e}')}