# preprocessing/preprocess_event_data.py
import argparse
import os
import pandas as pd
import numpy as np
import traceback
import re
import sys
import json
from sklearn.preprocessing import RobustScaler
import joblib

# --- Import the core processing function ---
try:
    # Relative import from the sub-package
    from .event_processor.process_event_interactions import extract_interactions_from_events
    # Import the final columns definition as well
    from .event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS as FINAL_INTERACTION_COLUMNS_CSV
except ImportError:
    # Fallback
    print("Warning: Could not use relative imports for event processor. Attempting direct import.")
    try:
        from event_processor.process_event_interactions import extract_interactions_from_events
        from event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS as FINAL_INTERACTION_COLUMNS_CSV
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import event_processor module.")
         sys.exit(1)

# --- Configurações GNN (Movidas para cá) ---
INTERACTION_TYPE_MAP_GNN = {
    "Product Viewed": "viewed", "Product Wishlisted": "wishlisted", "Product Shared": "shared",
    "Variant Meta Added": "variant_meta_added", "Shopify Checkout Created": "checkout_created",
    "Shopify Checkout Updated": "checkout_updated", "Shopify Order Created": "purchased",
    "Shopify Order Updated": "order_updated",
}
EDGE_NUMERIC_FEATURES_GNN = ['event_converted_price']
EDGE_CATEGORICAL_FEATURES_GNN = ['channel']
TRAIN_PERC_GNN = 0.8
VALID_PERC_GNN = 0.1
ID_COL_PERSON_GNN = 'person_id_hex'
ID_COL_PRODUCT_GNN = 'product_id'
INTERACTION_COL_GNN = 'interaction_type'
TIMESTAMP_COL_GNN = 'timestamp'
# --- Fim Configurações GNN ---


# --- Função Auxiliar para Preparação GNN (Movida para dentro e MODIFICADA com DEBUG) ---
def _prepare_gnn_edge_data(
    interactions_df: pd.DataFrame, # Recebe o DF já processado
    person_id_map_path: str,
    product_id_map_path: str,
    output_dir_gnn: str,
    include_edge_features: bool
):
    """
    Internal function to prepare and save GNN-ready edge data from the processed interactions df.
    Includes debugging step to save removed interactions.
    """
    print(f"\n--- Preparing Edge Data for GNN ---")
    print(f"Person ID map: {person_id_map_path}")
    print(f"Product ID map: {product_id_map_path}")
    print(f"Output directory for GNN edge data: {output_dir_gnn}")
    print(f"Include edge features: {include_edge_features}")

    os.makedirs(output_dir_gnn, exist_ok=True)

    try:
        # Carregar Mapeamentos
        print("Loading ID maps...")
        if not os.path.exists(person_id_map_path): raise FileNotFoundError(f"Person ID map not found: {person_id_map_path}")
        if not os.path.exists(product_id_map_path): raise FileNotFoundError(f"Product ID map not found: {product_id_map_path}")
        with open(person_id_map_path, 'r') as f: person_id_map = json.load(f)
        with open(product_id_map_path, 'r') as f: product_id_map = json.load(f)
        print(f"Loaded {len(person_id_map)} person ID mappings.")
        print(f"Loaded {len(product_id_map)} product ID mappings.")

        # Carregar Interações (Assumindo que 'interactions_df' foi passado como argumento para esta função)
        print(f"Input interactions count: {len(interactions_df)}")
        if interactions_df.empty:
            print("Input interactions DataFrame for GNN prep is empty. Skipping GNN edge generation.")
            return

        df_gnn = interactions_df.copy() # Usar uma cópia para modificações
        initial_count = len(df_gnn)

        # Garantir que colunas essenciais existem e converter IDs para STRING
        required_cols = [ID_COL_PERSON_GNN, ID_COL_PRODUCT_GNN, INTERACTION_COL_GNN, TIMESTAMP_COL_GNN]
        for col in required_cols:
            if col not in df_gnn.columns:
                raise KeyError(f"Essential column '{col}' not found in input DataFrame for GNN prep.")
            if col in [ID_COL_PERSON_GNN, ID_COL_PRODUCT_GNN]:
                 # Converter para string ANTES de qualquer filtro .isin
                 df_gnn[col] = df_gnn[col].astype(str)
                 print(f"DEBUG: Converted '{col}' column to string type.")

        # --- DEBUG: Identificar e Salvar Interações Removidas ---
        print("Identifying interactions to be removed...")

        # Condições de filtro separadas para clareza
        person_match = df_gnn[ID_COL_PERSON_GNN].isin(person_id_map)
        product_match = df_gnn[ID_COL_PRODUCT_GNN].isin(product_id_map)
        type_match = df_gnn[INTERACTION_COL_GNN].isin(INTERACTION_TYPE_MAP_GNN)

        # Máscara para linhas que seriam MANTIDAS
        keep_mask = person_match & product_match & type_match
        # Máscara para linhas que seriam REMOVIDAS (inverso de keep_mask)
        remove_mask = ~keep_mask

        removed_interactions_df = df_gnn[remove_mask].copy()
        num_removed = len(removed_interactions_df)
        print(f"Identified {num_removed} interactions that would be removed by filters.")

        # Adicionar colunas de debug para indicar PORQUÊ foram removidas
        if not removed_interactions_df.empty:
            removed_interactions_df['debug_person_match'] = person_match[remove_mask]
            removed_interactions_df['debug_product_match'] = product_match[remove_mask]
            removed_interactions_df['debug_type_match'] = type_match[remove_mask]

            # Salvar uma amostra (e.g., as primeiras 1000) num ficheiro CSV
            sample_size = min(1000, num_removed) # Pega até 1000 linhas
            removed_sample_df = removed_interactions_df.head(sample_size)
            # Usa o output_dir_gnn (passado para a função _prepare_gnn...) para guardar o ficheiro de debug
            debug_output_path = os.path.join(output_dir_gnn, "debug_removed_interactions_sample.csv")
            try:
                removed_sample_df.to_csv(debug_output_path, index=False)
                print(f"Saved a sample of {len(removed_sample_df)} removed interactions to: {debug_output_path}")
            except Exception as e_save:
                 print(f"Error saving debug CSV for removed interactions: {e_save}")
        # --- FIM DEBUG ---

        # Filtrar e Mapear IDs (Aplicar a máscara 'keep_mask' calculada acima)
        print("Filtering interactions by ID maps and mapping IDs...")
        df_gnn = df_gnn[keep_mask].copy() # Manter apenas as linhas válidas
        filtered_count = len(df_gnn)
        removed_count_actual = initial_count - filtered_count # Recalcular só para confirmar
        print(f"Actually removed {removed_count_actual} interactions (should match identified count).")
        print(f"Remaining interactions after filtering: {filtered_count}")

        if df_gnn.empty:
            print("No valid interactions remaining after map filtering. Skipping GNN edge generation.")
            return

        # Mapear IDs para índices inteiros (isto só acontece nas linhas mantidas)
        print("Mapping IDs to indices...")
        df_gnn['person_idx'] = df_gnn[ID_COL_PERSON_GNN].map(person_id_map)
        df_gnn['product_idx'] = df_gnn[ID_COL_PRODUCT_GNN].map(product_id_map)
        df_gnn['relation'] = df_gnn[INTERACTION_COL_GNN].map(INTERACTION_TYPE_MAP_GNN)

        # Garantir Timestamp e Ordenar
        print("Ensuring timestamp format and sorting...")
        if not pd.api.types.is_datetime64_any_dtype(df_gnn[TIMESTAMP_COL_GNN]):
             df_gnn[TIMESTAMP_COL_GNN] = pd.to_datetime(df_gnn[TIMESTAMP_COL_GNN], errors='coerce', utc=True)
        elif df_gnn[TIMESTAMP_COL_GNN].dt.tz is None:
             df_gnn[TIMESTAMP_COL_GNN] = df_gnn[TIMESTAMP_COL_GNN].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT') # Lidar com NaT
        else:
             df_gnn[TIMESTAMP_COL_GNN] = df_gnn[TIMESTAMP_COL_GNN].dt.tz_convert('UTC')

        initial_rows_before_ts_dropna = len(df_gnn)
        df_gnn.dropna(subset=[TIMESTAMP_COL_GNN], inplace=True)
        rows_dropped_ts = initial_rows_before_ts_dropna - len(df_gnn)
        if rows_dropped_ts > 0:
             print(f"Dropped {rows_dropped_ts} interactions due to NaT timestamps after filtering.")

        if df_gnn.empty:
             print("No valid interactions remaining after timestamp check. Skipping GNN edge generation.")
             return

        df_gnn = df_gnn.sort_values(by=TIMESTAMP_COL_GNN)

        # Divisão Temporal
        print("Performing temporal split...")
        n_interactions = len(df_gnn)
        n_train = int(n_interactions * TRAIN_PERC_GNN)
        n_valid = int(n_interactions * VALID_PERC_GNN)
        train_data = df_gnn.iloc[:n_train]
        valid_data = df_gnn.iloc[n_train : n_train + n_valid]
        test_data = df_gnn.iloc[n_train + n_valid :]
        print(f"Split sizes: Train={len(train_data)}, Validation={len(valid_data)}, Test={len(test_data)}")
        datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
        edge_features_data = {}

        # Preparar Features de Aresta (Opcional)
        edge_feature_scalers = {}
        edge_feature_vocabs = {}
        edge_numeric_cols_order = []
        edge_categorical_cols_order = []

        if include_edge_features:
            print("Preparing edge features for GNN...")
            # --- Features Numéricas ---
            numeric_edge_final = [col for col in EDGE_NUMERIC_FEATURES_GNN if col in train_data.columns]
            if numeric_edge_final:
                print(f"  Processing numeric edge features: {numeric_edge_final}")
                scaler = RobustScaler()
                train_numeric = train_data[numeric_edge_final].fillna(0)
                scaler.fit(train_numeric)
                edge_feature_scalers['numeric'] = scaler
                edge_numeric_cols_order = numeric_edge_final
                for split, df_split in datasets.items():
                    if not df_split.empty:
                        numeric_data = df_split[numeric_edge_final].fillna(0)
                        scaled_data = scaler.transform(numeric_data)
                        edge_features_data.setdefault(split, {})['numeric'] = scaled_data
                print(f"    Scaled numeric edge features using RobustScaler (fitted on train).")
                scaler_path = os.path.join(output_dir_gnn, 'edge_numeric_scaler.joblib')
                joblib.dump(scaler, scaler_path)
                print(f"    Saved edge numeric scaler to {scaler_path}")
                numeric_order_path = os.path.join(output_dir_gnn, 'edge_numeric_cols_order.json')
                with open(numeric_order_path, 'w') as f: json.dump(edge_numeric_cols_order, f)
                print(f"    Saved edge numeric column order to {numeric_order_path}")

            # --- Features Categóricas ---
            categorical_edge_final = [col for col in EDGE_CATEGORICAL_FEATURES_GNN if col in train_data.columns]
            if categorical_edge_final:
                print(f"  Processing categorical edge features: {categorical_edge_final}")
                edge_categorical_cols_order = categorical_edge_final
                for col in categorical_edge_final:
                    train_cat = train_data[col].fillna('__UNKNOWN__').astype(str)
                    unique_values = sorted([v for v in train_cat.unique() if v != '__UNKNOWN__'])
                    vocab = {val: i+1 for i, val in enumerate(unique_values)}
                    vocab['__UNKNOWN__'] = 0
                    edge_feature_vocabs[col] = vocab
                    print(f"    Vocabulary size for {col}: {len(vocab)}")
                    for split, df_split in datasets.items():
                        if not df_split.empty:
                            encoded_col = df_split[col].fillna('__UNKNOWN__').astype(str).map(vocab)
                            encoded_col = encoded_col.fillna(vocab['__UNKNOWN__']).astype(int)
                            edge_features_data.setdefault(split, {}).setdefault('categorical', {})[col] = encoded_col.values
                vocabs_path = os.path.join(output_dir_gnn, 'edge_categorical_vocabs.json')
                with open(vocabs_path, 'w') as f: json.dump(edge_feature_vocabs, f, indent=4)
                print(f"    Saved edge categorical vocabularies to {vocabs_path}")
                cat_order_path = os.path.join(output_dir_gnn, 'edge_categorical_cols_order.json')
                with open(cat_order_path, 'w') as f: json.dump(edge_categorical_cols_order, f)
                print(f"    Saved edge categorical column order to {cat_order_path}")

        # Salvar Dados das Arestas por Split
        print("Saving processed GNN edge data...")
        for split, df_split in datasets.items():
            split_output_dir = os.path.join(output_dir_gnn, split)
            os.makedirs(split_output_dir, exist_ok=True)
            print(f"  Saving {split} data to {split_output_dir}...")
            if df_split.empty:
                print(f"    Split '{split}' is empty, skipping save.")
                continue

            grouped = df_split.groupby('relation')
            edge_index_dict = {}
            edge_features_split_dict = {}

            for relation, group in grouped:
                edge_type = ('Person', relation, 'Product')
                src = group['person_idx'].values
                dst = group['product_idx'].values
                edge_index_dict[str(edge_type)] = np.stack([src, dst], axis=0)

                if include_edge_features and split in edge_features_data:
                    edge_features_relation = {}
                    group_indices = group.index
                    original_indices_in_split = df_split.index.get_indexer(group_indices) # Get positions in the split df

                    if 'numeric' in edge_features_data[split]:
                         numeric_feats_all = edge_features_data[split]['numeric']
                         edge_features_relation['numeric'] = numeric_feats_all[original_indices_in_split]

                    if 'categorical' in edge_features_data[split]:
                         cat_feats_dict = edge_features_data[split]['categorical']
                         categorical_relation_data = {}
                         for cat_col, encoded_vals_all in cat_feats_dict.items():
                              categorical_relation_data[cat_col] = encoded_vals_all[original_indices_in_split]
                         if categorical_relation_data:
                             edge_features_relation['categorical'] = np.stack(
                                 [categorical_relation_data[col] for col in edge_categorical_cols_order],
                                 axis=1
                             )
                    if edge_features_relation:
                         edge_features_split_dict[str(edge_type)] = edge_features_relation

            edge_index_path = os.path.join(split_output_dir, 'edge_index.npz')
            np.savez(edge_index_path, **edge_index_dict)
            print(f"    Saved edge indices for {len(edge_index_dict)} relation types to {edge_index_path}")

            if include_edge_features and edge_features_split_dict:
                edge_features_path = os.path.join(split_output_dir, 'edge_features.npz')
                save_dict = {}
                for edge_type_str, features in edge_features_split_dict.items():
                    if 'numeric' in features: save_dict[f"{edge_type_str}_numeric"] = features['numeric']
                    if 'categorical' in features: save_dict[f"{edge_type_str}_categorical"] = features['categorical']
                if save_dict:
                    np.savez(edge_features_path, **save_dict)
                    print(f"    Saved edge features to {edge_features_path}")

        print("--- GNN Edge Data Preparation Finished ---")

    except FileNotFoundError as e:
        print(f"Error preparing GNN edge data: Input file not found: {e}")
        traceback.print_exc()
    except KeyError as e:
        print(f"Error preparing GNN edge data: Column not found: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during GNN edge data preparation: {e}")
        traceback.print_exc()
# --- Fim Função Auxiliar GNN ---


# --- Função Principal Atualizada ---
def run_event_preprocessing(
    input_path: str,
    output_path: str, # Caminho para o CSV intermediário
    # --- Novos argumentos para GNN ---
    person_id_map_path: str,
    product_id_map_path: str,
    edge_gnn_output_dir: str,
    include_edge_features: bool
):
    """
    Orchestrates the preprocessing of event data to extract interactions,
    saves the intermediate CSV, AND prepares/saves GNN-ready edge data.

    Args:
        input_path: Path to the filtered event data CSV.
        output_path: Path to save the intermediate processed event interactions CSV.
        person_id_map_path: Path to the person ID -> index map JSON file.
        product_id_map_path: Path to the product ID -> index map JSON file.
        edge_gnn_output_dir: Directory to save GNN-ready edge data (train/valid/test splits).
        include_edge_features: Whether to prepare and save edge features.
    """
    print(f"\n--- Starting Event Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Intermediate Output CSV: {output_path}")
    print(f"Person ID Map: {person_id_map_path}")
    print(f"Product ID Map: {product_id_map_path}")
    print(f"GNN Edges Output Dir: {edge_gnn_output_dir}")
    print(f"Include Edge Features: {include_edge_features}")

    interactions_df = pd.DataFrame() # Inicializar

    # --- Load Raw Event Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        event_df = pd.read_csv(input_path, low_memory=False)
        if event_df.empty:
            print("Input event DataFrame is empty. Cannot proceed.")
            return
    except Exception as e:
        print(f"Error loading input event CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Process Events to Extract Interactions ---
    try:
        print("\nExtracting and processing interactions...")
        interactions_df = extract_interactions_from_events(event_df)
        # Não fazer mais verificações de duplicatas aqui, deixa para a fase GNN
    except Exception as e:
        print(f"Error during event interaction extraction: {e}")
        traceback.print_exc()
        raise # Parar se a extração falhar

    # --- Save Intermediate Processed Interactions CSV ---
    try:
        if interactions_df.empty:
             print("\nInteraction DataFrame is empty after processing. Saving empty intermediate file.")
             output_dir_csv = os.path.dirname(output_path)
             if output_dir_csv: os.makedirs(output_dir_csv, exist_ok=True)
             pd.DataFrame(columns=FINAL_INTERACTION_COLUMNS_CSV).to_csv(output_path, index=False, na_rep='') # Adiciona na_rep
        else:
            output_dir_csv = os.path.dirname(output_path)
            os.makedirs(output_dir_csv, exist_ok=True)
            print(f"\nEnsured intermediate output directory exists: {output_dir_csv}")
            # Verifica tipos antes de salvar
            print("\nData types in final DataFrame before save:")
            print(interactions_df.info())
            interactions_df.to_csv(output_path, index=False, na_rep='') # Adiciona na_rep
            print(f"Successfully saved intermediate processed event interactions to: {output_path}")

    except Exception as e:
        print(f"Error saving intermediate event interactions CSV to '{output_path}': {e}")
        traceback.print_exc()
        # Considerar parar aqui
        # raise

    # --- Preparar e Salvar Dados de Aresta para GNN ---
    # Usa o 'interactions_df' que foi salvo no CSV intermediário
    # Este passo só é chamado explicitamente no Estágio 4 do run_preprocessing.py
    # A chamada aqui dentro pode ser considerada redundante no fluxo 'full_pipeline',
    # mas permite correr este script isoladamente para gerar CSV + GNN (se os mapas existirem).
    try:
        # Verifica se os mapas de ID existem ANTES de chamar a função
        # Adiciona verificação se interactions_df não está vazio ANTES de chamar
        if not interactions_df.empty:
            if not os.path.exists(person_id_map_path):
                 # Esta mensagem é normal no Estágio 2, mas um erro se este script for chamado isoladamente depois do Estágio 3
                 print(f"Warning/Error: Cannot prepare GNN edges YET. Person ID map not found at: {person_id_map_path}")
            elif not os.path.exists(product_id_map_path):
                 print(f"Error: Cannot prepare GNN edges. Product ID map not found at: {product_id_map_path}")
            else:
                 # Chama a função de preparação GNN (com a lógica de debug incluída)
                 _prepare_gnn_edge_data(
                     interactions_df, # Passa o DF extraído
                     person_id_map_path,
                     product_id_map_path,
                     edge_gnn_output_dir,
                     include_edge_features
                 )
        else:
            print("\nSkipping GNN edge data preparation because the interactions DataFrame is empty.")

    except Exception as e:
        print(f"Error during GNN edge data preparation step (within run_event_preprocessing): {e}")
        traceback.print_exc()
        print("Warning: GNN edge data preparation failed, but intermediate CSV might be saved.")


    print(f"--- Finished Event Data Preprocessing Orchestration (including GNN prep attempt) ---")


# --- Command Line Execution Logic (Atualizado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess event data, extract interactions, and prepare GNN edge data.')
    parser.add_argument('--input', required=True, help='Path to the input person_event_data_filtered.csv file.')
    parser.add_argument('--output-csv', required=True, help='Path to save the intermediate event_interactions_processed.csv file.')
    # --- Novos argumentos ---
    parser.add_argument('--person-map', required=True, help='Path to the person_id_to_index_map.json file.')
    parser.add_argument('--product-map', required=True, help='Path to the product_id_to_index_map.json file.')
    parser.add_argument('--output-gnn-dir', required=True, help='Directory to save GNN-ready edge data (train/valid/test splits).')
    parser.add_argument('--include-edge-features', action='store_true', help='Include edge features (price, channel) in the GNN output.')

    args = parser.parse_args()

    try:
        run_event_preprocessing(
            args.input,
            args.output_csv,
            args.person_map,
            args.product_map,
            args.output_gnn_dir,
            args.include_edge_features
        )
        print("\nEvent preprocessing & GNN prep finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error:
         print(f"\nError: {val_error}")
         exit(1)
    except Exception as e:
        print(f"\n--- Event preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)