# scripts/run_full_pipeline.py

import argparse
import boto3
import json
import os
import sys
import time
import traceback

# --- AJUSTE sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FIM AJUSTE sys.path ---

# Importa a função para lançar o job de pré-processamento
try:
    # Agora vamos precisar que sagemaker_deploy aceite parâmetros
    from preprocessing.sagemaker_deploy import run_sagemaker_preprocessing_job
except ImportError:
    print("Error: Could not import 'run_sagemaker_preprocessing_job' from 'preprocessing.sagemaker_deploy'.")
    # ... (mensagens de erro iguais) ...
    sys.exit(1)

# --- Configurações Gerais (Podem ser defaults ou overrides) ---
DEFAULT_LAMBDA_FUNCTION_NAME = "napps-recommender-data-loading"
DEFAULT_AWS_REGION = "eu-west-3"
DEFAULT_IMAGE_URI = "694795525682.dkr.ecr.eu-west-3.amazonaws.com/napps-recommender/preprocessing:latest"
DEFAULT_S3_BASE_URI = "s3://napps-recommender/"
DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_VOLUME_SIZE_GB = 50
DEFAULT_ROLE_ARN = "arn:aws:iam::694795525682:role/service-role/AmazonSageMaker-ExecutionRole-20250422T110596" # Mantém como default aqui
DEFAULT_WAIT = True
DEFAULT_LOGS = True
# --- Fim Configurações ---


def invoke_lambda(function_name: str, payload: dict, region: str) -> bool:
    """
    Invokes an AWS Lambda function synchronously and checks for success.
    (Função invoke_lambda permanece igual à versão anterior)
    """
    print(f"\n--- Invoking Lambda function: {function_name} in {region} ---")
    lambda_client = boto3.client('lambda', region_name=region)
    success = False
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            LogType='Tail',
            Payload=json.dumps(payload)
        )
        status_code = response.get('StatusCode')
        print(f"Lambda invocation completed with status code: {status_code}")

        if 'FunctionError' in response:
            print(f"Lambda function execution failed with error: {response['FunctionError']}")
            log_result_b64 = response.get('LogResult')
            if log_result_b64:
                import base64
                log_result = base64.b64decode(log_result_b64).decode('utf-8')
                print("--- Lambda Execution Log (Tail) ---")
                print(log_result)
                print("--- End Lambda Log ---")
            payload_bytes = response.get('Payload')
            if payload_bytes:
                 try:
                     error_payload = json.loads(payload_bytes.read().decode('utf-8'))
                     print("--- Lambda Error Payload ---")
                     print(json.dumps(error_payload, indent=2))
                     print("--- End Error Payload ---")
                 except Exception as parse_err:
                      print(f"Could not parse error payload: {parse_err}")
        elif status_code == 200:
            print("Lambda function executed successfully.")
            payload_bytes = response.get('Payload')
            if payload_bytes:
                 try:
                     response_payload = json.loads(payload_bytes.read().decode('utf-8'))
                     print("--- Lambda Response Payload ---")
                     print(json.dumps(response_payload, indent=2))
                     print("--- End Response Payload ---")
                     if isinstance(response_payload, dict) and 'statusCode' in response_payload:
                         if 200 <= response_payload['statusCode'] < 300:
                             print("Lambda internal status indicates success.")
                             success = True
                         else:
                             print(f"Warning: Lambda internal statusCode indicates failure: {response_payload['statusCode']}")
                             if 'body' in response_payload: print(f"Lambda body: {response_payload['body']}")
                     else:
                         print("Warning: Lambda response payload structure unknown or missing 'statusCode'. Assuming success based on invocation status 200.")
                         success = True
                 except Exception as parse_err:
                      print(f"Could not parse success payload: {parse_err}")
                      success = True
            else:
                 success = True
        else:
            print(f"Lambda invocation failed with unexpected status code: {status_code}")

    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Error: Lambda function '{function_name}' not found in region {region}.")
        traceback.print_exc()
    except Exception as e:
        print(f"Error invoking Lambda function: {e}")
        traceback.print_exc()

    return success


def main():
    parser = argparse.ArgumentParser(description="Run Lambda data loading and then SageMaker Preprocessing Job, with configurable parameters.")

    # --- Argumentos Comuns ---
    parser.add_argument('--shop-id', required=True, help='Shop identifier for Lambda and SageMaker.')
    parser.add_argument('--region', default=DEFAULT_AWS_REGION, help='AWS Region for Lambda and SageMaker.')

    # --- Argumentos para Lambda ---
    parser.add_argument('--lambda-name', default=DEFAULT_LAMBDA_FUNCTION_NAME, help='Name of the Data Loading Lambda function.')
    parser.add_argument('--lambda-source', default='all', choices=['database', 'shopify', 'all'], help='Source parameter for the Data Loading Lambda.')
    # Adicionar outros argumentos específicos da Lambda se necessário
    # parser.add_argument('--lambda-some-param', help='...')

    # --- Argumentos para SageMaker Processing Job ---
    parser.add_argument('--processing-mode', default='all', choices=['person', 'product', 'events', 'all'], help='Preprocessing mode for SageMaker Job.')
    parser.add_argument('--image-uri', default=DEFAULT_IMAGE_URI, help='ECR Image URI for SageMaker Preprocessing Job.')
    parser.add_argument('--role-arn', default=DEFAULT_ROLE_ARN, help='IAM Execution Role ARN for SageMaker Job.')
    parser.add_argument('--s3-base-uri', default=DEFAULT_S3_BASE_URI, help='Base S3 URI for input/output (e.g., s3://bucket/).')
    parser.add_argument('--instance-type', default=DEFAULT_INSTANCE_TYPE, help='Instance type for SageMaker Job.')
    parser.add_argument('--instance-count', type=int, default=DEFAULT_INSTANCE_COUNT, help='Instance count for SageMaker Job.')
    parser.add_argument('--volume-size', type=int, default=DEFAULT_VOLUME_SIZE_GB, help='EBS Volume size (GB) for SageMaker Job.')
    parser.add_argument('--no-wait', action='store_false', dest='wait', default=DEFAULT_WAIT, help='Do not wait for SageMaker Job completion.')
    parser.add_argument('--no-logs', action='store_false', dest='logs', default=DEFAULT_LOGS, help='Do not show logs while waiting for SageMaker Job.')

    args = parser.parse_args()

    print(f"=== Starting Full Pipeline for Shop ID: {args.shop_id} ===")
    print(f"Target Region: {args.region}")

    # --- Step 1: Invoke Lambda ---
    lambda_payload = {
        "shop_id": args.shop_id,
        "source": args.lambda_source # Usa o argumento da linha de comando
        # Adiciona outros args.lambda_... aqui se definidos no parser
    }

    lambda_success = invoke_lambda(args.lambda_name, lambda_payload, args.region)

    # --- Step 2: Run SageMaker Processing Job (if Lambda succeeded) ---
    if lambda_success:
        print(f"\n--- Lambda completed successfully. Starting SageMaker Processing Job for shop: {args.shop_id} ---")
        try:
            # Chama a função importada, passando TODOS os argumentos relevantes
            run_sagemaker_preprocessing_job(
                shop_id=args.shop_id,
                mode=args.processing_mode,
                image_uri=args.image_uri,
                role_arn=args.role_arn, # Passa a role explicitamente ou None se não fornecida
                s3_base_data_uri=args.s3_base_uri,
                instance_type=args.instance_type,
                instance_count=args.instance_count,
                volume_size_gb=args.volume_size,
                wait=args.wait,
                logs=args.logs
                # A função em sagemaker_deploy.py precisa ser ajustada para aceitar estes
            )
            print(f"\n--- SageMaker Processing Job submitted (or completed if wait={args.wait}) for shop: {args.shop_id} ---")
            print("=== Full Pipeline Finished Successfully ===")

        except Exception as e:
            print(f"\n--- Error running SageMaker Processing Job for shop: {args.shop_id} ---")
            sys.exit(1)
    else:
        print("\n--- Lambda function failed. Skipping SageMaker Processing Job. ---")
        print("=== Full Pipeline Failed ===")
        sys.exit(1)


if __name__ == "__main__":
    main()