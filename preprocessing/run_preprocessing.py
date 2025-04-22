# preprocessing/run_preprocessing.py
import argparse
import os
import traceback
import sys
import time # Added time for consistency

# --- Imports ---
try:
    # Import the stage orchestrator functions
    from .preprocess_person_data import run_person_preprocessing
    from .preprocess_product_data import run_product_preprocessing
    from .preprocess_event_data import run_event_preprocessing
except ImportError as e:
    print(f"Import Error: {e}. Failed to import submodules from within 'preprocessing'.")
    print("Check __init__.py files and execution context (run from project root with -m).")
    sys.exit(1)
# --- End Imports ---

# --- Constants ---
SM_INPUT_CHANNEL = 'raw_data'      # Corresponde ao 'input_name' no ProcessingInput
SM_OUTPUT_CHANNEL = 'processed_data' # Corresponde ao 'output_name' no ProcessingOutput
# --- End Constants ---

# --- Environment Detection ---
def is_sagemaker_environment():
    """Checks if the script is running in a SageMaker Processing Job environment."""
    return os.path.exists('/opt/ml/processing/')
# --- End Environment Detection ---

# --- Path Management ---
def get_data_paths(shop_id: str) -> dict:
    """
    Determines input/output paths based on environment and desired structure.
    SageMaker: Uses paths provided by ProcessingInput/ProcessingOutput channels.
    Local: Constructs paths based on project structure relative to this script.

    Args:
        shop_id: The identifier for the specific shop.

    Returns:
        A dictionary containing the absolute paths for input/output files.
    """
    paths = {}
    is_sm = is_sagemaker_environment()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if is_sm:
        print("Detected SageMaker environment.")
        base_input_dir = f"/opt/ml/processing/{SM_INPUT_CHANNEL}"
        base_output_dir = f"/opt/ml/processing/{SM_OUTPUT_CHANNEL}"
        print(f"Using SageMaker base input directory: {base_input_dir}")
        print(f"Using SageMaker base output directory: {base_output_dir}")
    else:
        print("Detected local environment.")
        base_input_dir = os.path.join(project_root, "results", shop_id, "raw")
        base_output_dir = os.path.join(project_root, "results", shop_id, "preprocessed")
        print(f"Using local base input directory: {base_input_dir}")
        print(f"Using local base output directory: {base_output_dir}")

    # --- Define standard filenames ---
    person_input_filename = f'{shop_id}_person_data_loaded.csv'
    product_input_filename = f'{shop_id}_shopify_products_raw.csv'
    event_input_filename = f'{shop_id}_person_event_data_filtered.csv'

    event_output_filename = f'{shop_id}_event_interactions_processed.csv'
    person_output_filename = f'{shop_id}_person_features_processed.csv'
    product_output_filename = f'{shop_id}_products_final_all_features.csv'
    # --- End Define standard filenames ---

    # --- Constrói os caminhos completos ---
    paths['person_input'] = os.path.join(base_input_dir, person_input_filename)
    paths['product_input'] = os.path.join(base_input_dir, product_input_filename)
    paths['event_input'] = os.path.join(base_input_dir, event_input_filename)

    paths['event_output'] = os.path.join(base_output_dir, event_output_filename)
    paths['person_output'] = os.path.join(base_output_dir, person_output_filename)
    paths['product_output'] = os.path.join(base_output_dir, product_output_filename)
    # --- Fim Construção ---

    print("\nDetermined data paths:")
    for key, value in paths.items():
        print(f"- {key}: {value}")

    # Criação dos diretórios de output
    if base_output_dir:
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {base_output_dir}")
        models_dir = os.path.join(base_output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        print(f"Ensured models subdirectory exists: {models_dir}")

    return paths
# --- End Path Management ---

# --- Main Execution Logic ---
def main():
    """Main function to parse arguments and orchestrate preprocessing."""
    parser = argparse.ArgumentParser(description='Run the preprocessing pipeline for person, product, and/or event data.')
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop to process.')
    parser.add_argument('--mode', required=True, choices=['person', 'product', 'events', 'all'],
                        help='Which preprocessing pipeline(s) to run.')
    args = parser.parse_args()
    shop_id = args.shop_id # Obtém shop_id dos argumentos
    mode = args.mode

    # --- DEBUGGING: LISTAR FICHEIROS (Pode remover após confirmar que funciona) ---
    print("\n--- DEBUG: Listing files in /opt/ml/processing/ ---")
    processing_base = "/opt/ml/processing"
    if os.path.exists(processing_base) and is_sagemaker_environment(): # Só lista se estiver em SM
        for root, dirs, files in os.walk(processing_base):
            # Calcula indentação para melhor visualização
            level = root.replace(processing_base, '').count(os.sep)
            indent = '---' * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = '---' * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
    elif is_sagemaker_environment():
         print(f"Directory not found for debug listing: {processing_base}")
    else:
        print("Skipping debug listing (not in SageMaker environment).")
    print("--- END DEBUG ---\n")
    # --- FIM DEBUGGING ---

    print(f"\n=== Running Preprocessing Pipeline ===")
    print(f"Shop ID: {shop_id}")
    print(f"Mode: {mode}")
    start_time = time.time() # Start timer
    success = True # Track overall success

    try:
        # 1. Determine paths
        paths = get_data_paths(shop_id)

        # --- Execute Stages in Product -> Event -> Person order ---

        # 2. Execute Product Preprocessing if requested
        if mode in ['product', 'all']:
            print("\n" + "="*10 + " STEP 1: Starting Product Processing " + "="*10)
            step_success = False
            try:
                if not os.path.exists(paths['product_input']):
                    raise FileNotFoundError(f"Product input file not found: {paths['product_input']}")
                # --- ALTERAÇÃO AQUI: Passa shop_id ---
                run_product_preprocessing(paths['product_input'], paths['product_output'], shop_id)
                # --- FIM DA ALTERAÇÃO ---
                print("--- Product Preprocessing Completed Successfully ---")
                step_success = True
            except FileNotFoundError as e:
                 print(f"\nProduct Processing Aborted: Required file not found.")
                 print(f"Error details: {e}")
                 step_success = False # Marca falha
            except Exception as e:
                 print("\n--- Product Processing FAILED ---")
                 print(f"An unexpected error occurred: {e}")
                 traceback.print_exc()
                 step_success = False # Marca falha
            success &= step_success # Atualiza sucesso geral
            print("="*10 + f" STEP 1: Finished Product Processing (Success: {step_success}) " + "="*10)
            if not step_success and mode != 'all':
                 print("Exiting due to failure in 'product' mode.")
                 sys.exit(1) # Sai se só este modo foi pedido e falhou

        # 3. Execute Event Preprocessing if requested
        # Só executa se o modo for 'events' ou ('all' E o passo anterior teve sucesso)
        if mode in ['events', 'all'] and (mode == 'events' or success):
            print("\n" + "="*10 + " STEP 2: Starting Event Interaction Processing " + "="*10)
            step_success = False
            try:
                if not os.path.exists(paths['event_input']):
                    raise FileNotFoundError(f"Event input file not found: {paths['event_input']}")
                # Chama com 2 argumentos
                run_event_preprocessing(paths['event_input'], paths['event_output'])
                print("--- Event Interaction Processing Completed Successfully ---")
                step_success = True
            except FileNotFoundError as e:
                 print(f"\nEvent Processing Aborted: Required file not found.")
                 print(f"Error details: {e}")
                 step_success = False # Marca falha
            except Exception as e:
                 print("\n--- Event Processing FAILED ---")
                 print(f"An unexpected error occurred: {e}")
                 traceback.print_exc()
                 step_success = False # Marca falha
            success &= step_success # Atualiza sucesso geral
            print("="*10 + f" STEP 2: Finished Event Processing (Success: {step_success}) " + "="*10)
            if not step_success and mode != 'all':
                 print("Exiting due to failure in 'events' mode.")
                 sys.exit(1) # Sai se só este modo foi pedido e falhou
        elif mode in ['events', 'all'] and not success:
             print("\n" + "="*10 + " STEP 2: Skipping Event Interaction Processing (Previous Step Failed) " + "="*10)


        # 4. Execute Person Preprocessing if requested
        # Só executa se o modo for 'person' ou ('all' E os passos anteriores tiveram sucesso)
        if mode in ['person', 'all'] and (mode == 'person' or success):
            print("\n" + "="*10 + " STEP 3: Starting Person Preprocessing " + "="*10)
            step_success = False
            event_output_file = paths['event_output']
            # Verifica se o ficheiro de eventos existe (necessário)
            if not os.path.exists(event_output_file):
                 print(f"\nError: Cannot run person processing because the required event interactions file is missing:")
                 print(f"  {event_output_file}")
                 print(f"Ensure 'events' mode or 'all' mode ran successfully first, or the file exists if running 'person' mode alone.")
                 success = False # Marca falha geral
                 step_success = False # Marca falha neste passo
                 # Não precisa sair aqui necessariamente se for modo 'all', mas marca falha
            else:
                # Ficheiro de eventos existe, tenta processar pessoas
                try:
                    if not os.path.exists(paths['person_input']):
                        raise FileNotFoundError(f"Person input file not found: {paths['person_input']}")

                    # Chama a função de processamento de pessoas
                    run_person_preprocessing(paths['person_input'], paths['event_output'], paths['person_output'])

                    print("--- Person Preprocessing Completed Successfully ---")
                    step_success = True
                except FileNotFoundError as e:
                     print(f"\nPerson Processing Aborted: Required file not found.")
                     print(f"Error details: {e}")
                     step_success = False # Marca falha
                except Exception as e:
                     print("\n--- Person Processing FAILED ---")
                     print(f"An unexpected error occurred: {e}")
                     traceback.print_exc()
                     step_success = False # Marca falha

            success &= step_success # Atualiza sucesso geral
            print("="*10 + f" STEP 3: Finished Person Processing (Success: {step_success}) " + "="*10)
            if not step_success and mode != 'all':
                 print("Exiting due to failure in 'person' mode.")
                 sys.exit(1) # Sai se só este modo foi pedido e falhou
        elif mode in ['person', 'all'] and not success:
             print("\n" + "="*10 + " STEP 3: Skipping Person Preprocessing (Previous Step Failed) " + "="*10)


        # 5. Final Status Message
        end_time = time.time()
        total_seconds = end_time - start_time
        print("\n" + "="*20 + " Processing Pipeline Finished " + "="*20)
        print(f"Shop ID: {shop_id}")
        print(f"Mode executed: {mode}")
        print(f"Overall Success: {success}")
        print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_seconds))} ({total_seconds:.2f} seconds)")

        if not success:
            print("\nOne or more processing stages failed.")
            sys.exit(1) # Sai com código de erro
        else:
            print("\nAll requested processing stages completed successfully.")
            # Código de saída 0 é o default em caso de sucesso

    # --- Overall Error Handling ---
    except Exception as e:
        print("\n--- Preprocessing Pipeline FAILED Overall (Unexpected Error) ---")
        print(f"An unexpected error occurred outside the specific steps: {e}")
        traceback.print_exc()
        sys.exit(1) # Sai com código de erro
    # --- End Overall Error Handling ---

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
# --- End Script Entry Point ---