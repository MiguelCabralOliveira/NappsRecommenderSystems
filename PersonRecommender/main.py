# main.py
import argparse
import os
import pandas as pd
import json
import traceback
import shutil # Importar shutil (embora não usado ativamente agora, pode ser útil)

# --- Local Imports ---
from data_pipeline.data_loader import load_all_data
from preprocessing.preprocess_person_data import run_person_preprocessing
from preprocessing.process_event_interactions import extract_interactions_from_events
# --- Adicionar import para a criação de features de produto ---
from preprocessing.create_product_features import calculate_product_features

# Placeholders for future steps
# from graph_builder import build_graph_data
# from model_trainer import train_model
# from embedding_generator import generate_embeddings
# from similarity_calculator import calculate_similarities

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Load, preprocess, and prepare user interaction data.')
    parser.add_argument('--shop-id', required=True, help='The tenant_id for the shop.')
    parser.add_argument('--output-dir', default='results', help='Base directory to save results.')
    parser.add_argument('--days', type=int, default=90, help='Number of past days of event data to load.')
    args = parser.parse_args()

    run_output_dir = os.path.join(args.output_dir, args.shop_id)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output directory: {run_output_dir}")

    print(f"\nStarting pipeline for shop_id: {args.shop_id}")

    # Define file paths consistently
    person_loaded_path = os.path.join(run_output_dir, "person_data_loaded.csv")
    event_loaded_path = os.path.join(run_output_dir, "person_event_data_filtered.csv")
    events_discovered_path = os.path.join(run_output_dir, "discovered_event_types.json")
    person_processed_path = os.path.join(run_output_dir, "person_features_processed.csv")
    interactions_processed_path = os.path.join(run_output_dir, "event_interactions_processed.csv")
    # --- Definir path para as features de produto ---
    product_features_path = os.path.join(run_output_dir, "product_features_aggregated.csv")
    # Manter o placeholder para event_processed_path se ainda for útil para algo mais
    # event_processed_path = os.path.join(run_output_dir, "event_data_processed.csv") # Comentado por agora

    # === Step 1: Load Data ===
    print("\n=== Step 1: Loading Data ===")
    person_df_loaded = None
    event_df_loaded = None
    distinct_event_types = []
    try:
        person_df_loaded, event_df_loaded, distinct_event_types = load_all_data(args.shop_id, args.days)

        # Validação após carregamento
        loading_successful = True
        if person_df_loaded is None or person_df_loaded.empty:
            print("Warning: Person data loading resulted in None or empty DataFrame.")
            # loading_successful = False # Decidir se crítico

        if event_df_loaded is None or event_df_loaded.empty:
            print("Warning: Filtered Event data loading resulted in None or empty DataFrame. Subsequent steps might fail.")
            loading_successful = False # Event data é essencial para interações

        if not loading_successful:
            print("Stopping pipeline due to missing essential data after loading.")
            return

        # --- Save Loaded Data ---
        print("\n--- Saving Loaded Data ---")
        if person_df_loaded is not None: person_df_loaded.to_csv(person_loaded_path, index=False)
        print(f"Loaded person data saved to {person_loaded_path}")
        if event_df_loaded is not None: event_df_loaded.to_csv(event_loaded_path, index=False)
        print(f"Filtered person event data saved to {event_loaded_path}")
        if distinct_event_types:
            with open(events_discovered_path, 'w') as f: json.dump(distinct_event_types, f, indent=4)
            print(f"Discovered event types saved to {events_discovered_path}")

    except Exception as e:
        print(f"\n--- Critical Error during Step 1 (Data Loading): {e} ---")
        traceback.print_exc()
        return

    # === Step 2: Preprocessing Person Data ===
    print("\n=== Step 2: Preprocessing Person Data ===")
    try:
        if not os.path.exists(person_loaded_path):
             raise FileNotFoundError(f"Person loaded file not found for preprocessing: {person_loaded_path}")
        run_person_preprocessing(
            input_path=person_loaded_path,
            output_path=person_processed_path,
        )
        print("Person data preprocessing step completed.")
        if not os.path.exists(person_processed_path):
            raise RuntimeError("Processed person file missing after preprocessing step.")
    except Exception as e:
        print(f"\n--- Critical Error during Step 2 (Person Preprocessing): {e} ---")
        traceback.print_exc()
        return

    # === Step 2b: Preprocessing Event Data -> Extract Interactions ===
    print("\n=== Step 2b: Processing Event Data to Extract Interactions ===")
    interactions_df = None # Inicializar para verificar se foi criado
    try:
        if not os.path.exists(event_loaded_path):
             print(f"Warning: Event loaded file not found at {event_loaded_path}, skipping interaction extraction.")
        # Tentar usar o event_df_loaded se existir
        elif 'event_df_loaded' in locals() and event_df_loaded is not None and not event_df_loaded.empty:
            print("Using loaded event data DataFrame for interaction extraction.")
            interactions_df = extract_interactions_from_events(event_df_loaded)
        else:
             print(f"Warning: Event data DataFrame not available, skipping interaction extraction.")

        # Guardar o resultado se foi criado
        if interactions_df is not None and not interactions_df.empty:
            interactions_df.to_csv(interactions_processed_path, index=False)
            print(f"Processed interactions saved to {interactions_processed_path}")
        elif os.path.exists(event_loaded_path): # Só avisar se o input existia mas não gerou output
             print("No interactions were extracted or the resulting DataFrame is empty.")

        # Verificar se o ficheiro de output existe (importante para o próximo passo)
        if not os.path.exists(interactions_processed_path) and os.path.exists(event_loaded_path):
             print(f"Warning: Processed interactions file was not created at {interactions_processed_path}. Subsequent steps might fail.")
             # Considerar parar aqui se as interações são essenciais
             # return

    except Exception as e:
        print(f"\n--- Critical Error during Step 2b (Event Interaction Extraction): {e} ---")
        traceback.print_exc()
        return # Parar se a extração de interações falhar

    # === Step 2c: Create Aggregated Product Features ===
    print("\n=== Step 2c: Creating Aggregated Product Features ===")
    product_features_df = None # Inicializar
    try:
        # Tenta usar o DataFrame de interações se ainda estiver em memória
        if 'interactions_df' in locals() and interactions_df is not None and not interactions_df.empty:
             print("Using interactions DataFrame from memory for product feature calculation.")
             product_features_df = calculate_product_features(interactions_df)
        # Senão, tenta ler do ficheiro criado no passo anterior
        elif os.path.exists(interactions_processed_path):
             print(f"Reading interactions from file: {interactions_processed_path}")
             interactions_df_loaded = pd.read_csv(interactions_processed_path, low_memory=False)
             if not interactions_df_loaded.empty:
                 product_features_df = calculate_product_features(interactions_df_loaded)
             else:
                 print(f"Interactions file {interactions_processed_path} is empty.")
        else:
             print(f"Warning: Processed interactions file not found at {interactions_processed_path}. Skipping product feature creation.")

        # Guardar o resultado se foi calculado
        if product_features_df is not None and not product_features_df.empty:
            product_features_df.to_csv(product_features_path, index=False)
            print(f"Processed product features saved to {product_features_path}")
        elif os.path.exists(interactions_processed_path): # Só avisar se o input existia mas não gerou output
             print("No product features were calculated or the resulting DataFrame is empty.")

    except Exception as e:
        print(f"\n--- Error during Step 2c (Product Feature Creation): {e} ---")
        traceback.print_exc()
        # Decidir se é crítico parar o pipeline (provavelmente sim para GraphSAGE)
        # return

    # === Step 3: Graph Data Preparation (Placeholder) ===
    print("\n=== Step 3: Graph Data Preparation (Placeholder) ===")
    # Este passo agora usaria os 3 ficheiros principais:
    print(f"Input for this step would conceptually be:")
    print(f"  Person Nodes Features: {person_processed_path}")
    print(f"  Product Nodes Features: {product_features_path}") # Adicionado
    print(f"  Interaction Edges: {interactions_processed_path}")
    # Exemplo: graph_data = build_graph_data(person_processed_path, product_features_path, interactions_processed_path)
    # ...
    print("Skipping Graph Data Preparation.")


    # === Step 4: Model Training (Placeholder) ===
    print("\n=== Step 4: Model Training (Placeholder) ===")
    # Uses artifacts from Step 3
    print("Skipping Model Training.")


    # === Step 5: Embedding Generation (Placeholder) ===
    print("\n=== Step 5: Embedding Generation (Placeholder) ===")
    # Uses artifacts from Step 3 & 4
    print("Skipping Embedding Generation.")


    # === Step 6: Similarity Calculation (Placeholder) ===
    print("\n=== Step 6: Similarity Calculation (Placeholder) ===")
    # Uses embeddings from Step 5
    print("Skipping Similarity Calculation.")


    print("\n=== Pipeline Finished ===")


if __name__ == "__main__":
    main()