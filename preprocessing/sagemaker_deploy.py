# preprocessing/sagemaker_deploy.py

import sagemaker
# IMPORTA Processor em vez de ScriptProcessor
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
import boto3
import time
import argparse
import os
import sys
import traceback
from typing import Tuple
from urllib.parse import urlparse

# Adiciona a raiz do projeto ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Valores Fixos ---
DEFAULT_IMAGE_URI = "694795525682.dkr.ecr.eu-west-3.amazonaws.com/napps-recommender/preprocessing:latest"
DEFAULT_S3_BASE_URI = "s3://napps-recommender/" # Garante a barra no final
DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_VOLUME_SIZE_GB = 50
# ATUALIZA AQUI: Substitui None pelo ARN real da tua role se não funcionar automaticamente
DEFAULT_ROLE_ARN = "arn:aws:iam::694795525682:role/service-role/AmazonSageMaker-ExecutionRole-20250422T110596"
DEFAULT_WAIT = True
DEFAULT_LOGS = True
# --- Fim Valores Fixos ---

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Extrai o bucket e a key (prefixo) de um S3 URI."""
    parsed = urlparse(s3_uri, allow_fragments=False)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI scheme: {parsed.scheme}")
    return parsed.netloc, parsed.path.lstrip('/') # bucket, key

def run_sagemaker_preprocessing_job(
    shop_id: str,
    mode: str,
    image_uri: str = DEFAULT_IMAGE_URI, # Usa valor fixo como default
    role_arn: str = DEFAULT_ROLE_ARN, # Usa valor fixo como default
    s3_base_data_uri: str = DEFAULT_S3_BASE_URI, # Usa valor fixo como default
    instance_type: str = DEFAULT_INSTANCE_TYPE, # Usa valor fixo como default
    instance_count: int = DEFAULT_INSTANCE_COUNT, # Usa valor fixo como default
    volume_size_gb: int = DEFAULT_VOLUME_SIZE_GB, # Usa valor fixo como default
    input_channel_name: str = 'raw_data',
    output_channel_name: str = 'processed_data',
    wait: bool = DEFAULT_WAIT, # Usa valor fixo como default
    logs: bool = DEFAULT_LOGS # Usa valor fixo como default
):
    """
    Configures and runs a SageMaker Processing Job for the preprocessing pipeline.
    Derives S3 input/output paths based on s3_base_data_uri and shop_id.
    Uses predefined defaults for most parameters.
    """
    if not s3_base_data_uri:
        raise ValueError("s3_base_data_uri must be provided (e.g., 's3://your-bucket/')")

    if not s3_base_data_uri.endswith('/'):
         s3_base_data_uri += '/' # Garante a barra

    # --- CORREÇÃO INÍCIO ---
    # Primeiro, inicializa a sessão e obtém a região
    try:
        sagemaker_session = sagemaker.Session()
        region = sagemaker_session.boto_region_name
        if not region:
             raise ValueError("SageMaker session could not determine AWS region.")
    except Exception as e:
         print(f"Error getting AWS region from SageMaker session: {e}")
         print("Ensure your local AWS configuration (e.g., via 'aws configure') includes a default region.")
         sys.exit(1)

    # Depois, determina a role a usar
    final_role_arn = role_arn # Começa com o argumento passado (que pode ser o default da constante)
    if not final_role_arn:
        try:
            # Tenta obter role; se falhar, usa a constante definida (se existir)
            final_role_arn = get_execution_role(sagemaker_session=sagemaker_session) # Passa a sessão
            print(f"Using default SageMaker execution role: {final_role_arn}")
        except ValueError:
             if not DEFAULT_ROLE_ARN:
                print("ERROR: Could not automatically determine SageMaker role. Provide it in DEFAULT_ROLE_ARN or ensure script runs in an environment with a default role.")
                sys.exit(1)
             else: # Usa a role definida como constante se get_execution_role falhar
                 final_role_arn = DEFAULT_ROLE_ARN
                 print(f"Could not automatically determine SageMaker role. Using defined Role ARN: {final_role_arn}")
        except Exception as e:
            print(f"Error getting execution role: {e}")
            sys.exit(1)
    else:
         # Usa a role do argumento/constante se não for vazia
         print(f"Using provided Role ARN: {final_role_arn}")
    # --- CORREÇÃO FIM ---


    s3_input_path = f"{s3_base_data_uri.rstrip('/')}/{shop_id}/raw" # rstrip para garantir que não haja '//'
    s3_output_path = f"{s3_base_data_uri.rstrip('/')}/{shop_id}/preprocessed"

    print(f"SageMaker Region: {region}")
    print(f"Using ECR Image URI: {image_uri}")
    print(f"Using IAM Role ARN: {final_role_arn}")
    print(f"Using Instance Type: {instance_type}")
    print(f"Base S3 Data URI: {s3_base_data_uri}")
    print(f"Derived S3 Input Path: {s3_input_path}")
    print(f"Derived S3 Output Path: {s3_output_path}")
    print(f"Shop ID: {shop_id}")
    print(f"Processing Mode: {mode}")

    # --- USA Processor ---
    processor = Processor(
        image_uri=image_uri,
        role=final_role_arn,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=volume_size_gb,
        sagemaker_session=sagemaker_session,
        env={'PYTHONUNBUFFERED': '1'}
        # base_job_name=f"preprocessing-{shop_id}-{mode}" # Opcional
    )
    # --- FIM DA ALTERAÇÃO ---

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

    # Argumentos passados para o script DENTRO do container
    job_arguments = [
        '--shop-id', shop_id,
        '--mode', mode
    ]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Se não definires base_job_name no Processor, podes continuar a gerar o nome aqui
    job_name = f"preprocessing-{shop_id}-{mode}-{timestamp}"
    print(f"\nSubmitting SageMaker Processing Job: {job_name}")

    try:
        # A chamada .run() para Processor não precisa do argumento 'code'
        processor.run(
            inputs=inputs,
            outputs=outputs,
            arguments=job_arguments,
            job_name=job_name, # Passa o nome do job aqui se não foi definido via base_job_name
            wait=wait,
            logs=logs
        )
        print(f"\nProcessing job {job_name} submitted.")
        if wait:
            print("Job complete.")
            # Tenta obter a descrição do job para mostrar o output path
            try:
                job_description = sagemaker_session.describe_processing_job(job_name)
                output_config = job_description.get('ProcessingOutputConfig', {}).get('Outputs', [])
                for output in output_config:
                    if output.get('OutputName') == output_channel_name:
                         s3_out = output.get('S3Output', {}).get('S3Uri', 'N/A')
                         print(f"Output data saved to: {s3_out}")
            except Exception as desc_err:
                 print(f"Could not describe job {job_name} after completion: {desc_err}")

    except Exception as e:
         print(f"Error submitting/running SageMaker job {job_name}: {e}")
         traceback.print_exc() # Imprime o traceback detalhado do erro
         raise # Re-lança a exceção para que o bloco __main__ a apanhe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SageMaker Processing Job for Preprocessing")

    # --- Argumentos Essenciais ---
    parser.add_argument('--shop-id', required=True, help='Shop identifier.')
    parser.add_argument('--mode', required=True, choices=['person', 'product', 'events', 'all'], help='Preprocessing mode.')
    # --- Fim Argumentos Essenciais ---

    # --- Argumentos Opcionais (para override dos defaults, se necessário) ---
    # Poderias adicionar overrides aqui se quisesses, por exemplo:
    # parser.add_argument('--instance-type', default=DEFAULT_INSTANCE_TYPE, help=f'Override instance type (default: {DEFAULT_INSTANCE_TYPE})')
    # Mas como pediste SÓ shop-id e mode, deixamos sem overrides por agora.
    # --- Fim Argumentos Opcionais ---

    args = parser.parse_args()

    try:
        # Chama a função principal passando apenas os args necessários
        # Os outros argumentos virão dos defaults definidos na função
        run_sagemaker_preprocessing_job(
            shop_id=args.shop_id,
            mode=args.mode
        )
    except Exception as e:
        # O erro já deve ter sido impresso com traceback pela função run_...
        print(f"\n--- SageMaker Processing Job submission/execution failed ---")
        sys.exit(1)