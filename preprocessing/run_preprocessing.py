# preprocessing/run_preprocessing.py
import argparse
import os
import traceback
import sys
import time
import json
import pandas as pd

# --- Imports ---
try:
    from .preprocess_person_data import run_person_preprocessing
    from .preprocess_product_data import run_product_preprocessing
    from .preprocess_event_data import run_event_preprocessing
    # --- NOVO: Importar prepare_edge_data diretamente ---
    from .event_processor.prepare_edge_data import prepare_edge_data
except ImportError as e:
    print(f"Import Error: {e}. Failed to import submodules.")
    print("Please ensure that __init__.py files exist in subdirectories and paths are correct.")
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

# --- Path Management ---
def get_data_paths(shop_id: str) -> dict:
    """Determines input/output paths for raw, intermediate, and GNN data."""
    paths = {}
    is_sm = is_sagemaker_environment()
    # Ajustar project_root se a estrutura do seu projeto for diferente
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if not is_sm else '/opt/ml/processing'


    if is_sm:
        base_input_dir = os.path.join(project_root, SM_INPUT_CHANNEL) # /opt/ml/processing/raw_data
        base_output_dir_intermediate = os.path.join(project_root, SM_OUTPUT_CHANNEL) # /opt/ml/processing/processed_data
        base_output_dir_gnn = os.path.join(base_output_dir_intermediate, GNN_DATA_DIR_NAME)
    else:
        base_input_dir = os.path.join(project_root, "results", shop_id, "raw")
        base_output_dir_intermediate = os.path.join(project_root, "results", shop_id, "preprocessed")
        base_output_dir_gnn = os.path.join(project_root, "results", shop_id, GNN_DATA_DIR_NAME)

    paths['person_raw_input'] = os.path.join(base_input_dir, f'{shop_id}_person_data_loaded.csv')
    paths['product_raw_input'] = os.path.join(base_input_dir, f'{shop_id}_shopify_products_raw.csv')
    paths['event_raw_input'] = os.path.join(base_input_dir, f'{shop_id}_person_event_data_filtered.csv')

    paths['event_processed_csv'] = os.path.join(base_output_dir_intermediate, f'{shop_id}_event_interactions_processed.csv')
    paths['person_processed_csv'] = os.path.join(base_output_dir_intermediate, f'{shop_id}_person_features_processed.csv')
    paths['product_processed_csv'] = os.path.join(base_output_dir_intermediate, f'{shop_id}_products_final_all_features.csv')

    paths['person_gnn_output_dir'] = os.path.join(base_output_dir_gnn, PERSON_NODE_DIR)
    paths['product_gnn_output_dir'] = os.path.join(base_output_dir_gnn, PRODUCT_NODE_DIR)
    paths['edge_gnn_output_dir'] = os.path.join(base_output_dir_gnn, EDGE_DATA_DIR) # Diretório base para edges

    paths['person_id_map_path'] = os.path.join(paths['person_gnn_output_dir'], 'person_id_to_index_map.json')
    paths['product_id_map_path'] = os.path.join(paths['product_gnn_output_dir'], 'product_id_to_index_map.json')
    # --- NOVO: Caminho para o ficheiro de cutoff ---
    paths['train_cutoff_timestamp_file'] = os.path.join(paths['edge_gnn_output_dir'], 'train_cutoff_timestamp.txt')


    print("\nDetermined data paths:")
    for key, value in paths.items(): print(f"- {key}: {value}")

    os.makedirs(base_output_dir_intermediate, exist_ok=True)
    # Diretório para modelos TF-IDF, etc., dentro de 'preprocessed'
    os.makedirs(os.path.join(base_output_dir_intermediate, "models"), exist_ok=True)
    os.makedirs(paths['person_gnn_output_dir'], exist_ok=True)
    os.makedirs(paths['product_gnn_output_dir'], exist_ok=True)
    os.makedirs(paths['edge_gnn_output_dir'], exist_ok=True) # Garante que o diretório de edges existe
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
                             '`all` or `full_pipeline` runs product, events (CSV), person (direct/props), GNN edges (determines cutoff), person (event metrics with cutoff).')
    parser.add_argument('--include-edge-features', action='store_true',
                        help='(Used with events/all/full_pipeline modes) Include edge features in GNN edge data.')
    args = parser.parse_args()
    shop_id = args.shop_id
    mode = args.mode
    include_edge_features = args.include_edge_features

    # ... (DEBUG listing - pode manter se útil) ...
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


    print(f"\n=== Running Preprocessing Pipeline (Integrated GNN Prep - Corrected Order) ===")
    print(f"Shop ID: {shop_id}")
    print(f"Mode: {mode}")
    start_time = time.time()
    overall_success = True

    try:
        paths = get_data_paths(shop_id)
        product_success = False
        event_csv_success = False
        person_direct_props_success = False
        gnn_edge_prep_success = False # Inclui geração do cutoff
        person_final_success = False
        train_cutoff_timestamp_value = None # Para guardar o timestamp

        run_product = mode in ['product', 'all', 'full_pipeline']
        run_events_csv = mode in ['events', 'all', 'full_pipeline'] # Extração CSV
        run_person_direct_props = mode in ['person', 'all', 'full_pipeline'] # Features diretas/props + ID map
        run_gnn_edges = mode in ['events', 'all', 'full_pipeline'] # Prepara arestas GNN e determina cutoff
        run_person_final = mode in ['person', 'all', 'full_pipeline'] # Métricas de evento com cutoff + GNN features

        # --- NOVA ORDEM DE EXECUÇÃO ---

        # STAGE 1: Product Preprocessing & GNN Node Features
        if run_product:
            print("\n" + "="*10 + " STAGE 1: Product Preprocessing & GNN Node Features " + "="*10)
            try:
                if not os.path.exists(paths['product_raw_input']): raise FileNotFoundError(f"Product raw input not found: {paths['product_raw_input']}")
                run_product_preprocessing(
                    paths['product_raw_input'],
                    paths['product_processed_csv'],
                    shop_id,
                    paths['product_gnn_output_dir'] # Gera product_id_map.json
                )
                if not os.path.exists(paths['product_id_map_path']):
                    raise RuntimeError("Product ID map was not created successfully.")
                print("--- Product Stage Completed Successfully ---")
                product_success = True
            except Exception as e:
                print(f"--- Product Stage FAILED: {e} ---"); traceback.print_exc()
            print("="*10 + f" STAGE 1: Finished Product (Success: {product_success}) " + "="*10)
            if not product_success and mode == 'product': sys.exit(1)

        # STAGE 2: Event Interaction Extraction (CSV only)
        if run_events_csv:
            print("\n" + "="*10 + " STAGE 2: Event Interaction Extraction (to CSV) " + "="*10)
            try:
                if not os.path.exists(paths['event_raw_input']): raise FileNotFoundError(f"Event raw input not found: {paths['event_raw_input']}")
                run_event_preprocessing( # Esta função agora só faz a extração para CSV
                    paths['event_raw_input'],
                    paths['event_processed_csv']
                )
                if not os.path.exists(paths['event_processed_csv']):
                    raise RuntimeError("Event interaction CSV was not created successfully.")
                print("--- Event CSV Extraction Stage Completed Successfully ---")
                event_csv_success = True
            except Exception as e:
                print(f"--- Event CSV Extraction Stage FAILED: {e} ---"); traceback.print_exc()
            print("="*10 + f" STAGE 2: Finished Event CSV Extraction (Success: {event_csv_success}) " + "="*10)
            if not event_csv_success and mode == 'events': sys.exit(1) # Se só eventos for pedido, falha aqui

        # STAGE 3: Person Direct/Properties Features & GNN ID Map
        # Esta etapa calcula features que NÃO dependem do cutoff de eventos.
        # E gera o person_id_map.json necessário para a preparação de arestas.
        if run_person_direct_props:
            if not event_csv_success: # Depende do CSV de eventos para a estrutura, mesmo que não use cutoff ainda
                print("\n" + "="*10 + " STAGE 3: Skipping Person Direct/Properties (Event CSV Failed) " + "="*10)
            else:
                print("\n" + "="*10 + " STAGE 3: Person Direct/Properties Features & GNN ID Map " + "="*10)
                try:
                    if not os.path.exists(paths['person_raw_input']): raise FileNotFoundError(f"Person raw input not found: {paths['person_raw_input']}")
                    # Chamar run_person_preprocessing SEM o cutoff_timestamp aqui.
                    # A função irá calcular métricas com todos os eventos, mas o importante é gerar o ID map
                    # e o person_processed_csv base que será *sobrescrito* depois.
                    run_person_preprocessing(
                        paths['person_raw_input'],
                        paths['event_processed_csv'],
                        paths['person_processed_csv'], # Cria um CSV base
                        paths['person_gnn_output_dir'], # Gera person_id_map.json
                        train_cutoff_timestamp=None # SEM CUTOFF NESTA FASE
                    )
                    if not os.path.exists(paths['person_id_map_path']):
                        raise RuntimeError("Person ID map was not created successfully.")
                    print("--- Person Direct/Properties & ID Map Stage Completed Successfully ---")
                    person_direct_props_success = True
                except Exception as e:
                    print(f"--- Person Direct/Properties & ID Map Stage FAILED: {e} ---"); traceback.print_exc()
                print("="*10 + f" STAGE 3: Finished Person Direct/Properties & ID Map (Success: {person_direct_props_success}) " + "="*10)
                # Não sair aqui se falhar, pois o modo 'person' completo inclui a etapa final

        # STAGE 4: GNN Edge Data Preparation (determines train_cutoff_timestamp)
        if run_gnn_edges:
            if not (product_success and event_csv_success and person_direct_props_success):
                print("\n" + "="*10 + " STAGE 4: Skipping GNN Edge Preparation (Previous Stage(s) Failed) " + "="*10)
            else:
                print("\n" + "="*10 + " STAGE 4: GNN Edge Data Preparation " + "="*10)
                try:
                    if not os.path.exists(paths['person_id_map_path']): raise FileNotFoundError(f"Person ID map not found: {paths['person_id_map_path']}")
                    if not os.path.exists(paths['product_id_map_path']): raise FileNotFoundError(f"Product ID map not found: {paths['product_id_map_path']}")
                    if not os.path.exists(paths['event_processed_csv']): raise FileNotFoundError(f"Event interactions CSV not found: {paths['event_processed_csv']}")

                    # Chamar prepare_edge_data do módulo event_processor
                    # Esta função retorna o train_cutoff_timestamp e salva-o num ficheiro.
                    train_cutoff_timestamp_value = prepare_edge_data(
                        paths['event_processed_csv'],
                        paths['person_id_map_path'],
                        paths['product_id_map_path'],
                        paths['edge_gnn_output_dir'], # Diretório onde o .txt será salvo
                        include_edge_features
                    )
                    if train_cutoff_timestamp_value is None:
                        raise RuntimeError("Failed to determine train_cutoff_timestamp during GNN edge preparation.")
                    if not os.path.exists(paths['train_cutoff_timestamp_file']):
                         # Verificar se o ficheiro foi salvo como esperado
                         print(f"Warning: train_cutoff_timestamp.txt was expected but not found at {paths['train_cutoff_timestamp_file']}. Using returned value.")

                    print(f"--- GNN Edge Data Preparation Completed. Train Cutoff: {train_cutoff_timestamp_value} ---")
                    gnn_edge_prep_success = True
                except Exception as e:
                    print(f"--- GNN Edge Data Preparation FAILED: {e} ---"); traceback.print_exc()
                print("="*10 + f" STAGE 4: Finished GNN Edge Prep (Success: {gnn_edge_prep_success}) " + "="*10)
                if not gnn_edge_prep_success and mode == 'events': sys.exit(1) # Se só eventos for pedido, falha aqui

        # STAGE 5: Person Final Features (Event Metrics with Cutoff & GNN Node Features)
        if run_person_final:
            if not (person_direct_props_success and gnn_edge_prep_success and train_cutoff_timestamp_value is not None):
                print("\n" + "="*10 + " STAGE 5: Skipping Final Person Feature Processing (Previous Stage(s) or Cutoff Failed) " + "="*10)
            else:
                print("\n" + "="*10 + " STAGE 5: Person Final Features (Event Metrics with Cutoff & GNN Nodes) " + "="*10)
                try:
                    # Chamar run_person_preprocessing novamente, desta vez COM o cutoff.
                    # Isto irá sobrescrever person_processed_csv e os GNN node features de pessoa.
                    run_person_preprocessing(
                        paths['person_raw_input'],
                        paths['event_processed_csv'],
                        paths['person_processed_csv'], # Sobrescreve
                        paths['person_gnn_output_dir'], # Sobrescreve
                        train_cutoff_timestamp=train_cutoff_timestamp_value # Passa o cutoff
                    )
                    print("--- Person Final Feature Processing Stage Completed Successfully ---")
                    person_final_success = True
                except Exception as e:
                    print(f"--- Person Final Feature Processing Stage FAILED: {e} ---"); traceback.print_exc()
                print("="*10 + f" STAGE 5: Finished Person Final Features (Success: {person_final_success}) " + "="*10)
                if not person_final_success and mode == 'person': sys.exit(1)


        # Final Status
        end_time = time.time()
        total_seconds = end_time - start_time

        if mode == 'product': overall_success = product_success
        elif mode == 'events': overall_success = product_success and event_csv_success and person_direct_props_success and gnn_edge_prep_success
        elif mode == 'person': overall_success = product_success and event_csv_success and person_direct_props_success and gnn_edge_prep_success and person_final_success
        elif mode in ['all', 'full_pipeline']:
            overall_success = (product_success and
                               event_csv_success and
                               person_direct_props_success and
                               gnn_edge_prep_success and
                               person_final_success)
        else:
            overall_success = False # Should not happen due to choices in argparse

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