# preprocessing/run_preprocessing.py
import argparse
import os
import traceback
import sys
import time
import json # Import json
import pandas as pd # Import pandas for Stage 4

# --- Imports ---
try:
    from .preprocess_person_data import run_person_preprocessing
    from .preprocess_product_data import run_product_preprocessing
    from .preprocess_event_data import run_event_preprocessing
    # Importar a função auxiliar de preparação de arestas para STAGE 4
    from .preprocess_event_data import _prepare_gnn_edge_data
except ImportError as e:
    print(f"Import Error: {e}. Failed to import submodules.")
    sys.exit(1)
# --- End Imports ---

# --- Constants ---
SM_INPUT_CHANNEL = 'raw_data'
SM_OUTPUT_CHANNEL = 'processed_data'
GNN_DATA_DIR_NAME = 'gnn_data'
PERSON_NODE_DIR = 'person_nodes'
PRODUCT_NODE_DIR = 'product_nodes'
EDGE_DATA_DIR = 'edges'
# --- Fim Constantes ---

def is_sagemaker_environment():
    return os.path.exists('/opt/ml/processing/')

# --- Path Management (sem alterações desde a última versão) ---
def get_data_paths(shop_id: str) -> dict:
    """Determines input/output paths for raw, intermediate, and GNN data."""
    paths = {}
    is_sm = is_sagemaker_environment()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if is_sm:
        base_input_dir = f"/opt/ml/processing/{SM_INPUT_CHANNEL}"
        base_output_dir_intermediate = f"/opt/ml/processing/{SM_OUTPUT_CHANNEL}"
        base_output_dir_gnn = os.path.join(base_output_dir_intermediate, GNN_DATA_DIR_NAME)
    else:
        base_input_dir = os.path.join(project_root, "results", shop_id, "raw")
        base_output_dir_intermediate = os.path.join(project_root, "results", shop_id, "preprocessed")
        base_output_dir_gnn = os.path.join(project_root, "results", shop_id, GNN_DATA_DIR_NAME)

    # --- Raw Inputs ---
    paths['person_raw_input'] = os.path.join(base_input_dir, f'{shop_id}_person_data_loaded.csv')
    paths['product_raw_input'] = os.path.join(base_input_dir, f'{shop_id}_shopify_products_raw.csv')
    paths['event_raw_input'] = os.path.join(base_input_dir, f'{shop_id}_person_event_data_filtered.csv')

    # --- Intermediate Outputs (CSVs) ---
    paths['event_processed_csv'] = os.path.join(base_output_dir_intermediate, f'{shop_id}_event_interactions_processed.csv')
    paths['person_processed_csv'] = os.path.join(base_output_dir_intermediate, f'{shop_id}_person_features_processed.csv')
    paths['product_processed_csv'] = os.path.join(base_output_dir_intermediate, f'{shop_id}_products_final_all_features.csv')

    # --- GNN Output Directories ---
    paths['person_gnn_output_dir'] = os.path.join(base_output_dir_gnn, PERSON_NODE_DIR)
    paths['product_gnn_output_dir'] = os.path.join(base_output_dir_gnn, PRODUCT_NODE_DIR)
    paths['edge_gnn_output_dir'] = os.path.join(base_output_dir_gnn, EDGE_DATA_DIR)

    # --- GNN ID Map Paths (Inputs for Edge Prep) ---
    paths['person_id_map_path'] = os.path.join(paths['person_gnn_output_dir'], 'person_id_to_index_map.json')
    paths['product_id_map_path'] = os.path.join(paths['product_gnn_output_dir'], 'product_id_to_index_map.json')

    print("\nDetermined data paths:")
    for key, value in paths.items(): print(f"- {key}: {value}")

    # Criação dos diretórios de output
    os.makedirs(base_output_dir_intermediate, exist_ok=True)
    os.makedirs(os.path.join(base_output_dir_intermediate, "models"), exist_ok=True)
    os.makedirs(paths['person_gnn_output_dir'], exist_ok=True)
    os.makedirs(paths['product_gnn_output_dir'], exist_ok=True)
    os.makedirs(paths['edge_gnn_output_dir'], exist_ok=True)
    print(f"Ensured all output directories exist.")

    return paths
# --- End Path Management ---

# --- Main Execution Logic com Ordem Corrigida ---
def main():
    parser = argparse.ArgumentParser(description='Run the full preprocessing pipeline including GNN data preparation.')
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop.')
    parser.add_argument('--mode', required=True,
                        choices=['product', 'person', 'events', 'all', 'full_pipeline'],
                        help='Which pipeline stage(s) to run. '
                             '`all` runs product, events, person sequentially, then finalizes GNN edges. '
                             '`full_pipeline` is equivalent to `all`.')
    parser.add_argument('--include-edge-features', action='store_true',
                        help='(Used with events/all/full_pipeline modes) Include edge features in GNN edge data.')
    args = parser.parse_args()
    shop_id = args.shop_id
    mode = args.mode
    include_edge_features = args.include_edge_features

    # --- DEBUGGING ---
    print("\n--- DEBUG: Listing files in /opt/ml/processing/ ---")
    processing_base = "/opt/ml/processing"
    if os.path.exists(processing_base) and is_sagemaker_environment(): # Só lista se estiver em SM
        for root, dirs, files in os.walk(processing_base):
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

    print(f"\n=== Running Preprocessing Pipeline (Integrated GNN Prep) ===")
    print(f"Shop ID: {shop_id}")
    print(f"Mode: {mode}")
    start_time = time.time()
    overall_success = True

    try:
        paths = get_data_paths(shop_id)
        product_success = True
        person_success = True
        event_csv_success = True # Sucesso da geração do CSV de eventos
        event_gnn_prep_success = True # Sucesso da preparação GNN de eventos (final)

        # Definir quais etapas executar
        run_product = mode in ['product', 'all', 'full_pipeline']
        run_events = mode in ['events', 'all', 'full_pipeline'] # Eventos (CSV) ANTES de pessoa
        run_person = mode in ['person', 'all', 'full_pipeline']

        # --- Executar na ordem correta: Product -> Events (CSV) -> Person -> Events (GNN Prep) ---

        # STAGE 1: Product
        if run_product:
            print("\n" + "="*10 + " STAGE 1: Product Preprocessing & GNN Prep " + "="*10)
            step_success = False
            try:
                if not os.path.exists(paths['product_raw_input']): raise FileNotFoundError(f"Product raw input not found: {paths['product_raw_input']}")
                run_product_preprocessing(
                    paths['product_raw_input'],
                    paths['product_processed_csv'],
                    shop_id,
                    paths['product_gnn_output_dir']
                )
                print("--- Product Stage Completed Successfully ---")
                step_success = True
            except Exception as e:
                print(f"--- Product Stage FAILED: {e} ---"); traceback.print_exc()
            product_success = step_success
            print("="*10 + f" STAGE 1: Finished Product (Success: {product_success}) " + "="*10)
            if not product_success and mode == 'product': sys.exit(1)

        # STAGE 2: Events (CSV Extraction Only initially)
        if run_events:
            if not product_success: # Precisa do sucesso do produto para continuar (embora não use o mapa ainda)
                 print("\n" + "="*10 + " STAGE 2: Skipping Event CSV Extraction (Product Stage Failed) " + "="*10)
            else:
                print("\n" + "="*10 + " STAGE 2: Event Interaction Extraction (CSV) " + "="*10)
                step_success = False
                try:
                    if not os.path.exists(paths['event_raw_input']): raise FileNotFoundError(f"Event raw input not found: {paths['event_raw_input']}")

                    # Chamar run_event_preprocessing, mas a parte GNN falhará graciosamente sem o mapa de pessoa
                    # O importante é gerar o event_processed_csv
                    run_event_preprocessing(
                        paths['event_raw_input'],
                        paths['event_processed_csv'], # Salva o CSV intermediário
                        paths['person_id_map_path'], # Passa o caminho, mesmo que ainda não exista
                        paths['product_id_map_path'], # Passa o caminho (deve existir)
                        paths['edge_gnn_output_dir'], # Diretório de saída GNN
                        include_edge_features # Flag
                    )
                    # Verificar se o CSV foi realmente criado
                    if not os.path.exists(paths['event_processed_csv']):
                        raise RuntimeError("Event interaction CSV was not created successfully.")
                    print("--- Event Stage (Interaction Extraction to CSV) Completed Successfully ---")
                    step_success = True
                except Exception as e:
                    print(f"--- Event Stage (Interaction Extraction) FAILED: {e} ---"); traceback.print_exc()
                event_csv_success = step_success # Rastreia o sucesso da criação do CSV
                print("="*10 + f" STAGE 2: Finished Event CSV Extraction (Success: {event_csv_success}) " + "="*10)
                if not event_csv_success and mode == 'events': sys.exit(1) # Falha se SÓ eventos foi pedido

        # STAGE 3: Person (Depende do CSV de Eventos)
        if run_person:
            if not event_csv_success: # Precisa que o CSV de eventos exista
                 print("\n" + "="*10 + " STAGE 3: Skipping Person (Event CSV Extraction Failed) " + "="*10)
            else:
                print("\n" + "="*10 + " STAGE 3: Person Preprocessing & GNN Prep " + "="*10)
                step_success = False
                try:
                    if not os.path.exists(paths['person_raw_input']): raise FileNotFoundError(f"Person raw input not found: {paths['person_raw_input']}")
                    if not os.path.exists(paths['event_processed_csv']):
                         raise FileNotFoundError(f"Event interactions CSV ({paths['event_processed_csv']}) is required for person metrics and was not found.")

                    run_person_preprocessing(
                        paths['person_raw_input'],
                        paths['event_processed_csv'], # Input para métricas (agora existe)
                        paths['person_processed_csv'], # Saída CSV intermediária
                        paths['person_gnn_output_dir'] # Saída GNN (gera o person_id_map.json)
                    )
                    # Verificar se o mapa de pessoas foi criado
                    if not os.path.exists(paths['person_id_map_path']):
                        raise RuntimeError("Person ID map was not created successfully.")
                    print("--- Person Stage Completed Successfully ---")
                    step_success = True
                except Exception as e:
                    print(f"--- Person Stage FAILED: {e} ---"); traceback.print_exc()
                person_success = step_success
                print("="*10 + f" STAGE 3: Finished Person (Success: {person_success}) " + "="*10)
                if not person_success and mode == 'person': sys.exit(1)

        # STAGE 4: Finalizar Preparação GNN de Eventos (Depende de Person e Product maps)
        # Só executa se as etapas anteriores foram bem-sucedidas E se eventos era parte do modo.
        if run_events and product_success and person_success and event_csv_success:
            print("\n" + "="*10 + " STAGE 4: Finalizing GNN Edge Preparation " + "="*10)
            step_success = False
            try:
                # Verificar se os mapas de ID e o CSV de eventos existem
                if not os.path.exists(paths['person_id_map_path']): raise FileNotFoundError(f"Person ID map needed for event GNN prep not found: {paths['person_id_map_path']}")
                if not os.path.exists(paths['product_id_map_path']): raise FileNotFoundError(f"Product ID map needed for event GNN prep not found: {paths['product_id_map_path']}")
                if not os.path.exists(paths['event_processed_csv']): raise FileNotFoundError(f"Event interactions CSV needed for event GNN prep not found: {paths['event_processed_csv']}")

                # Carregar o DF de interações já processado
                print(f"Loading interactions from {paths['event_processed_csv']} for final GNN edge prep...")
                interactions_df_final = pd.read_csv(paths['event_processed_csv'], low_memory=False)

                # Chamar a função interna de preparação GNN diretamente
                print("Calling _prepare_gnn_edge_data...")
                _prepare_gnn_edge_data(
                    interactions_df_final,
                    paths['person_id_map_path'],
                    paths['product_id_map_path'],
                    paths['edge_gnn_output_dir'],
                    include_edge_features
                )
                print("--- GNN Edge Data Preparation Completed Successfully ---")
                step_success = True
            except Exception as e:
                print(f"--- GNN Edge Preparation (Stage 4) FAILED: {e} ---"); traceback.print_exc()
            event_gnn_prep_success = step_success # Rastreia o sucesso desta etapa final
            print("="*10 + f" STAGE 4: Finished GNN Edge Prep (Success: {event_gnn_prep_success}) " + "="*10)
            # O sucesso geral da etapa 'events' agora depende desta etapa final
            if not event_gnn_prep_success and mode == 'events': sys.exit(1)

        # Final Status
        end_time = time.time()
        total_seconds = end_time - start_time
        # Sucesso geral depende do modo e do sucesso das etapas *relevantes* e suas dependências
        if mode == 'product': overall_success = product_success
        elif mode == 'events': overall_success = product_success and event_csv_success and event_gnn_prep_success # Precisa de produto e ambas as partes de eventos
        elif mode == 'person': overall_success = product_success and event_csv_success and person_success # Precisa de produto e CSV de eventos
        elif mode in ['all', 'full_pipeline']: overall_success = product_success and event_csv_success and person_success and event_gnn_prep_success # Precisa de tudo
        else: overall_success = False

        print("\n" + "="*20 + " Processing Pipeline Finished " + "="*20)
        print(f"Shop ID: {shop_id}")
        print(f"Mode executed: {mode}")
        print(f"Overall Success for requested mode(s): {overall_success}")
        print(f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(total_seconds))} ({total_seconds:.2f} seconds)")

        if not overall_success:
            print("\nOne or more requested processing stages failed.")
            sys.exit(1)
        else:
            print("\nAll requested processing stages completed successfully.")

    except Exception as e:
        print("\n--- Preprocessing Pipeline FAILED Overall (Unexpected Error) ---")
        print(f"An unexpected error occurred outside the specific steps: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()