# preprocessing/sagemaker_deploy.py

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import boto3
import time
import argparse
import os
import sys
import traceback
from urllib.parse import urlparse 

# Adiciona a raiz do projeto ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Extrai o bucket e a key (prefixo) de um S3 URI."""
    parsed = urlparse(s3_uri, allow_fragments=False)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI scheme: {parsed.scheme}")
    return parsed.netloc, parsed.path.lstrip('/') # bucket, key

def run_sagemaker_preprocessing_job(
    shop_id: str,
    mode: str,
    image_uri: str,
    role_arn: str,
    s3_base_data_uri: str, # Novo: Ex: s3://napps-recommender/results/
    instance_type: str = 'ml.m5.xlarge',
    instance_count: int = 1,
    volume_size_gb: int = 50,
    input_channel_name: str = 'raw_data',
    output_channel_name: str = 'processed_data',
    wait: bool = True,
    logs: bool = True
):
    """
    Configures and runs a SageMaker Processing Job for the preprocessing pipeline.
    Derives S3 input/output paths based on s3_base_data_uri and shop_id.
    """
    if not s3_base_data_uri:
        raise ValueError("s3_base_data_uri must be provided (e.g., 's3://your-bucket/results/')")

    # Garante que o URI base termina com '/'
    if not s3_base_data_uri.endswith('/'):
        s3_base_data_uri += '/'

    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name

    # Define S3 paths relativos ao base URI e shop_id
    s3_input_path = f"{s3_base_data_uri}{shop_id}/raw"
    s3_output_path = f"{s3_base_data_uri}{shop_id}/preprocessed"

    print(f"SageMaker Region: {region}")
    print(f"Using ECR Image URI: {image_uri}")
    print(f"Using IAM Role ARN: {role_arn}")
    print(f"Using Instance Type: {instance_type}")
    print(f"Base S3 Data URI: {s3_base_data_uri}")
    print(f"Derived S3 Input Path: {s3_input_path}")
    print(f"Derived S3 Output Path: {s3_output_path}")
    print(f"Shop ID: {shop_id}")
    print(f"Processing Mode: {mode}")

    script_processor = ScriptProcessor(
        command=['python', '-m', 'preprocessing.run_preprocessing'],
        image_uri=image_uri,
        role=role_arn,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=volume_size_gb,
        sagemaker_session=sagemaker_session,
        env={'PYTHONUNBUFFERED': '1'}
    )

    processing_input_local_path = f'/opt/ml/processing/{input_channel_name}'
    processing_output_local_path = f'/opt/ml/processing/{output_channel_name}'

    inputs = [
        ProcessingInput(
            source=s3_input_path,
            destination=processing_input_local_path,
            input_name=input_channel_name,
            s3_data_type='S3Prefix',
            s3_input_mode='File',
            s3_data_distribution_type='FullyReplicated'
        )
    ]

    outputs = [
        ProcessingOutput(
            source=processing_output_local_path,
            destination=s3_output_path,
            output_name=output_channel_name,
            s3_upload_mode='EndOfJob'
        )
    ]

    job_arguments = [
        '--shop-id', shop_id,
        '--mode', mode
    ]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"preprocessing-{shop_id}-{mode}-{timestamp}"
    print(f"\nSubmitting SageMaker Processing Job: {job_name}")

    try:
        script_processor.run(
            inputs=inputs,
            outputs=outputs,
            arguments=job_arguments,
            job_name=job_name,
            wait=wait,
            logs=logs
        )
        print(f"\nProcessing job {job_name} submitted.")
        if wait:
            print("Job complete.")
            job_description = script_processor.describe()
            output_config = job_description.get('ProcessingOutputConfig', {}).get('Outputs', [])
            for output in output_config:
                if output.get('OutputName') == output_channel_name:
                     s3_out = output.get('S3Output', {}).get('S3Uri', 'N/A')
                     print(f"Output data saved to: {s3_out}")
    except Exception as e:
         print(f"Error submitting/running SageMaker job {job_name}: {e}")
         traceback.print_exc()
         raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SageMaker Processing Job for Preprocessing")

    parser.add_argument('--shop-id', required=True, help='Shop identifier.')
    parser.add_argument('--mode', required=True, choices=['person', 'product', 'events', 'all'], help='Preprocessing mode.')
    parser.add_argument('--image-uri', required=True, help='ECR URI of the preprocessing Docker image (must exist in ECR).')
    parser.add_argument('--role-arn', required=False, help='SageMaker execution role ARN. If not provided, attempts to get default.')
    # *** Argumento Alterado ***
    parser.add_argument('--s3-base-uri', required=True, help="Base S3 URI where 'results/<shop_id>/raw/' exists and 'results/<shop_id>/preprocessed/' will be created (e.g., s3://your-bucket/results/).")
    parser.add_argument('--instance-type', default='ml.m5.xlarge', help='EC2 instance type for the job.')
    parser.add_argument('--instance-count', type=int, default=1, help='Number of instances.')
    parser.add_argument('--volume-size', type=int, default=50, help='EBS volume size in GB.')
    parser.add_argument('--no-wait', action='store_false', dest='wait', help='Submit the job and exit without waiting.')
    parser.add_argument('--no-logs', action='store_false', dest='logs', help='Do not display logs while waiting.')

    args = parser.parse_args()

    # Get default role if not specified
    role = args.role_arn
    if not role:
        try: role = get_execution_role(); print(f"Using default SageMaker execution role: {role}")
        except ValueError: print("Error: Could not automatically determine SageMaker role. Use --role-arn."); sys.exit(1)
        except Exception as e: print(f"Error getting execution role: {e}"); sys.exit(1) # Catch other potential errors
    else: print(f"Using provided Role ARN: {role}")

    try:
        run_sagemaker_preprocessing_job(
            shop_id=args.shop_id,
            mode=args.mode,
            image_uri=args.image_uri,
            role_arn=role,
            s3_base_data_uri=args.s3_base_uri, # Passar o novo argumento
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            volume_size_gb=args.volume_size,
            wait=args.wait,
            logs=args.logs
        )
    except Exception as e:
        print(f"\n--- Error running SageMaker Processing Job ---")
        # A função run_... já deve ter impresso o traceback
        sys.exit(1)