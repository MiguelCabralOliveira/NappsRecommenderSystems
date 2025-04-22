# preprocessing/run_preprocessing.py
import argparse
import os
import traceback
import sys
import time

# --- Imports ---
try:
    from .preprocess_person_data import run_person_preprocessing
    from .preprocess_product_data import run_product_preprocessing
    from .preprocess_event_data import run_event_preprocessing
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

# --- Path Management Atualizado ---
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

# --- Main Execution Logic Atualizado ---
def main():
    parser = argparse.ArgumentParser(description='Run the full preprocessing pipeline including GNN data preparation.')
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop.')
    # --- MODO SIMPLIFICADO ---
    parser.add_argument('--mode', required=True,
                        choices=['product', 'person', 'events', 'all', 'full_pipeline'],
                        help='Which pipeline stage(s) to run. '
                             '`all` runs product, person, events sequentially. '
                             '`full_pipeline` is equivalent to `all`.')
    parser.add_argument('--include-edge-features', action='store_true',
                        help='(Used with events/all/full_pipeline modes) Include edge features in GNN edge data.')
    args = parser.parse_args()
    shop_id = args.shop_id
    mode = args.mode
    include_edge_features = args.include_edge_features

    # --- DEBUGGING ---
    # ... (código de listagem de arquivos) ...
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
        event_success = True

        # Definir quais etapas executar
        run_product = mode in ['product', 'all', 'full_pipeline']
        run_person = mode in ['person', 'all', 'full_pipeline']
        run_events = mode in ['events', 'all', 'full_pipeline']

        # --- Executar na ordem correta: Product -> Person -> Events ---

        if run_product:
            print("\n" + "="*10 + " STAGE 1: Product Preprocessing & GNN Prep " + "="*10)
            try:
                if not os.path.exists(paths['product_raw_input']): raise FileNotFoundError(f"Product raw input not found: {paths['product_raw_input']}")
                run_product_preprocessing(
                    paths['product_raw_input'],
                    paths['product_processed_csv'], # Saída CSV intermediária
                    shop_id,
                    paths['product_gnn_output_dir'] # Saída GNN
                )
                print("--- Product Stage Completed Successfully ---")
            except Exception as e:
                print(f"--- Product Stage FAILED: {e} ---"); traceback.print_exc()
                product_success = False
            print("="*10 + f" STAGE 1: Finished Product (Success: {product_success}) " + "="*10)
            if not product_success and mode == 'product': sys.exit(1)

        if run_person and product_success: # Person não depende diretamente de produto, mas a ordem é boa
            print("\n" + "="*10 + " STAGE 2: Person Preprocessing & GNN Prep " + "="*10)
            try:
                if not os.path.exists(paths['person_raw_input']): raise FileNotFoundError(f"Person raw input not found: {paths['person_raw_input']}")
                # Person precisa do CSV de eventos *intermediário*
                if not os.path.exists(paths['event_processed_csv']):
                     print(f"Warning: Event interactions CSV ({paths['event_processed_csv']}) not found/generated yet. Person event metrics might be missing or default.")
                     # Tentar criar um arquivo vazio para evitar falha total? Ou deixar falhar se métricas são cruciais?
                     # Por enquanto, vamos permitir continuar, mas as métricas serão zeradas/default.
                     # Alternativa: Falhar se o modo for 'person', 'all', ou 'full_pipeline' e o arquivo de eventos não existir após a etapa de eventos (se executada).
                     # Vamos assumir que calculate_person_event_metrics lida com arquivo ausente.

                run_person_preprocessing(
                    paths['person_raw_input'],
                    paths['event_processed_csv'], # Input para métricas
                    paths['person_processed_csv'], # Saída CSV intermediária
                    paths['person_gnn_output_dir'] # Saída GNN
                )
                print("--- Person Stage Completed Successfully ---")
            except Exception as e:
                print(f"--- Person Stage FAILED: {e} ---"); traceback.print_exc()
                person_success = False
            print("="*10 + f" STAGE 2: Finished Person (Success: {person_success}) " + "="*10)
            if not person_success and mode == 'person': sys.exit(1)
        elif run_person and not product_success:
             print("\n" + "="*10 + " STAGE 2: Skipping Person (Product Stage Failed) " + "="*10) # Ou permitir rodar? Decidir dependência.

        if run_events and product_success and person_success: # Event GNN Prep depende dos mapas de ID de produto e pessoa
            print("\n" + "="*10 + " STAGE 3: Event Preprocessing & GNN Prep " + "="*10)
            try:
                if not os.path.exists(paths['event_raw_input']): raise FileNotFoundError(f"Event raw input not found: {paths['event_raw_input']}")
                # Verificar se os mapas de ID existem *antes* de chamar
                if not os.path.exists(paths['person_id_map_path']): raise FileNotFoundError(f"Person ID map needed for event GNN prep not found: {paths['person_id_map_path']}")
                if not os.path.exists(paths['product_id_map_path']): raise FileNotFoundError(f"Product ID map needed for event GNN prep not found: {paths['product_id_map_path']}")

                run_event_preprocessing(
                    paths['event_raw_input'],
                    paths['event_processed_csv'], # Saída CSV intermediária
                    paths['person_id_map_path'], # Input GNN
                    paths['product_id_map_path'], # Input GNN
                    paths['edge_gnn_output_dir'], # Saída GNN
                    include_edge_features # Flag
                )
                print("--- Event Stage Completed Successfully ---")
            except Exception as e:
                print(f"--- Event Stage FAILED: {e} ---"); traceback.print_exc()
                event_success = False
            print("="*10 + f" STAGE 3: Finished Event (Success: {event_success}) " + "="*10)
            if not event_success and mode == 'events': sys.exit(1)
        elif run_events and (not product_success or not person_success):
             print("\n" + "="*10 + " STAGE 3: Skipping Event (Previous Stage Failed) " + "="*10)


        # Final Status
        end_time = time.time()
        total_seconds = end_time - start_time
        # Sucesso geral depende do modo e do sucesso das etapas executadas
        if mode == 'product': overall_success = product_success
        elif mode == 'person': overall_success = person_success
        elif mode == 'events': overall_success = event_success
        elif mode in ['all', 'full_pipeline']: overall_success = product_success and person_success and event_success
        else: overall_success = False # Should not happen

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