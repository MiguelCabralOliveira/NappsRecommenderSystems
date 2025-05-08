# preprocessing/event_processor/prepare_edge_data.py
import pandas as pd
import numpy as np
import json
import os
import argparse
import traceback
from sklearn.preprocessing import RobustScaler # Para features de aresta opcionais
import joblib # Para salvar scaler de aresta opcional

# --- Configurações ---
# Mapeamento de interaction_type para nomes de relação canônicos (simplificados)
INTERACTION_TYPE_MAP = {
    "Product Viewed": "viewed",
    "Product Wishlisted": "wishlisted",
    "Product Shared": "shared",
    "Variant Meta Added": "variant_meta_added", # Ou talvez 'interacted_with'?
    "Shopify Checkout Created": "checkout_created",
    "Shopify Checkout Updated": "checkout_updated",
    "Shopify Order Created": "purchased",
    "Shopify Order Updated": "order_updated",
    # Adicione outros tipos se necessário, ou mapeie para 'interacted_with'
}

# Definir quais colunas usar como features de aresta (opcional)
EDGE_NUMERIC_FEATURES = ['event_converted_price']
EDGE_CATEGORICAL_FEATURES = ['channel']

# Percentagens para divisão temporal
TRAIN_PERC = 0.8
VALID_PERC = 0.1
# TEST_PERC é o restante (0.1)

ID_COL_PERSON = 'person_id_hex'
ID_COL_PRODUCT = 'product_id'
INTERACTION_COL = 'interaction_type'
TIMESTAMP_COL = 'timestamp'
# --- Fim Configurações ---


def prepare_edge_data(
    interactions_path: str,
    person_id_map_path: str,
    product_id_map_path: str,
    output_dir: str,
    include_edge_features: bool = False # Flag para incluir features de aresta
) -> pd.Timestamp | None: # --- ALTERAÇÃO: Definir tipo de retorno ---
    """
    Loads interactions, maps IDs, splits temporally, saves edge data for GNN,
    and saves the cutoff timestamp (max timestamp of train data).
    Optionally prepares and saves edge features.

    Returns:
        pd.Timestamp | None: The calculated cutoff timestamp (max timestamp from
                             the training split) or None if it couldn't be determined.
    """
    print(f"--- Starting Edge Data Preparation for GNN ---")
    print(f"Interactions file: {interactions_path}")
    print(f"Person ID map: {person_id_map_path}")
    print(f"Product ID map: {product_id_map_path}")
    print(f"Output directory: {output_dir}")
    print(f"Include edge features: {include_edge_features}")

    os.makedirs(output_dir, exist_ok=True)
    train_cutoff_timestamp = None # Inicializar variável de retorno

    try:
        # 1. Carregar Dados e Mapeamentos
        print("Loading interactions and ID maps...")
        # --- ALTERAÇÃO: Ler timestamp como objeto inicialmente ---
        interactions_df = pd.read_csv(interactions_path, low_memory=False) # Não fazer parse_dates aqui
        if interactions_df.empty:
            print("Interactions DataFrame is empty. Exiting.")
            return None # Retornar None se vazio

        # --- ALTERAÇÃO: Converter timestamp após leitura ---
        if TIMESTAMP_COL not in interactions_df.columns:
             raise KeyError(f"Timestamp column '{TIMESTAMP_COL}' not found in interactions file.")
        print("Converting timestamp column...")
        # Usar conversão robusta (two-pass) como em process_event_interactions
        timestamps_obj = interactions_df[TIMESTAMP_COL].astype(str).str.strip()
        format_pass1 = '%Y-%m-%d %H:%M:%S.%f%z'
        format_pass2 = '%Y-%m-%d %H:%M:%S%z'
        converted_pass1 = pd.to_datetime(timestamps_obj, format=format_pass1, errors='coerce', utc=True)
        failures_pass1_mask = converted_pass1.isna()
        converted_pass2 = converted_pass1.copy()
        if failures_pass1_mask.any():
            converted_pass2[failures_pass1_mask] = pd.to_datetime(
                timestamps_obj[failures_pass1_mask], format=format_pass2, errors='coerce', utc=True
            )
        interactions_df[TIMESTAMP_COL] = converted_pass2
        initial_rows = len(interactions_df)
        interactions_df.dropna(subset=[TIMESTAMP_COL], inplace=True) # Remover falhas de conversão
        if len(interactions_df) < initial_rows:
             print(f"Warning: Dropped {initial_rows - len(interactions_df)} rows due to invalid timestamps during conversion.")
        if interactions_df.empty:
            print("No valid interactions remaining after timestamp conversion. Exiting.")
            return None
        # --- FIM ALTERAÇÃO ---


        with open(person_id_map_path, 'r') as f:
            person_id_map = json.load(f)
        with open(product_id_map_path, 'r') as f:
            product_id_map = json.load(f)

        print(f"Loaded {len(interactions_df)} valid interactions.")
        print(f"Loaded {len(person_id_map)} person ID mappings.")
        print(f"Loaded {len(product_id_map)} product ID mappings.")

        # 2. Filtrar Interações Válidas e Mapear IDs
        print("Filtering interactions and mapping IDs...")
        initial_count = len(interactions_df)
        interactions_df = interactions_df[
            interactions_df[ID_COL_PERSON].isin(person_id_map) &
            interactions_df[ID_COL_PRODUCT].isin(product_id_map) &
            interactions_df[INTERACTION_COL].isin(INTERACTION_TYPE_MAP)
        ].copy() # Usar .copy() para evitar SettingWithCopyWarning
        filtered_count = len(interactions_df)
        print(f"Removed {initial_count - filtered_count} interactions due to missing IDs or unmapped interaction types.")

        if interactions_df.empty:
            print("No valid interactions remaining after filtering. Exiting.")
            return None

        interactions_df['person_idx'] = interactions_df[ID_COL_PERSON].map(person_id_map)
        interactions_df['product_idx'] = interactions_df[ID_COL_PRODUCT].map(product_id_map)
        interactions_df['relation'] = interactions_df[INTERACTION_COL].map(INTERACTION_TYPE_MAP)

        # Garantir que índices são inteiros (após map podem ser float se houver NaNs, embora filtrados)
        interactions_df['person_idx'] = interactions_df['person_idx'].astype(int)
        interactions_df['product_idx'] = interactions_df['product_idx'].astype(int)


        # 3. Divisão Temporal
        print("Performing temporal split (Train/Validation/Test)...")
        interactions_df = interactions_df.sort_values(by=TIMESTAMP_COL)

        n_interactions = len(interactions_df)
        n_train = int(n_interactions * TRAIN_PERC)
        n_valid = int(n_interactions * VALID_PERC)

        train_data = interactions_df.iloc[:n_train]
        valid_data = interactions_df.iloc[n_train : n_train + n_valid]
        test_data = interactions_df.iloc[n_train + n_valid :]

        print(f"Split sizes: Train={len(train_data)}, Validation={len(valid_data)}, Test={len(test_data)}")

        # --- ALTERAÇÃO: Determinar e Salvar Cutoff Timestamp ---
        if not train_data.empty:
            train_cutoff_timestamp = train_data[TIMESTAMP_COL].max()
            if pd.notna(train_cutoff_timestamp):
                cutoff_file_path = os.path.join(output_dir, 'train_cutoff_timestamp.txt')
                try:
                    with open(cutoff_file_path, 'w') as f:
                        f.write(train_cutoff_timestamp.isoformat())
                    print(f"Saved train cutoff timestamp ({train_cutoff_timestamp}) to {cutoff_file_path}")
                except Exception as e:
                    print(f"Error saving cutoff timestamp: {e}")
                    train_cutoff_timestamp = None # Resetar se falhar ao salvar
            else:
                print("Warning: Could not determine a valid max timestamp from train_data.")
                train_cutoff_timestamp = None
        else:
            print("Warning: train_data is empty, cannot determine cutoff timestamp.")
            train_cutoff_timestamp = None
        # --- FIM ALTERAÇÃO ---

        datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
        edge_features_data = {}

        # 4. (Opcional) Preparar Features de Aresta
        # ... (código para preparar features de aresta permanece o mesmo) ...
        edge_feature_scalers = {}
        edge_feature_vocabs = {}
        edge_numeric_cols_order = []
        edge_categorical_cols_order = []

        if include_edge_features:
            print("Preparing edge features...")
            # --- Features Numéricas ---
            numeric_edge_final = [col for col in EDGE_NUMERIC_FEATURES if col in train_data.columns]
            if numeric_edge_final:
                print(f"  Processing numeric edge features: {numeric_edge_final}")
                scaler = RobustScaler()
                train_numeric = train_data[numeric_edge_final].fillna(0)
                # Verificar se há dados para treinar o scaler
                if not train_numeric.empty:
                    scaler.fit(train_numeric)
                    edge_feature_scalers['numeric'] = scaler
                    edge_numeric_cols_order = numeric_edge_final

                    for split, df_split in datasets.items():
                        if not df_split.empty:
                            numeric_data = df_split[numeric_edge_final].fillna(0)
                            scaled_data = scaler.transform(numeric_data)
                            edge_features_data.setdefault(split, {})['numeric'] = scaled_data
                    print(f"    Scaled numeric edge features using RobustScaler (fitted on train).")
                    scaler_path = os.path.join(output_dir, 'edge_numeric_scaler.joblib')
                    joblib.dump(scaler, scaler_path)
                    print(f"    Saved edge numeric scaler to {scaler_path}")
                    numeric_order_path = os.path.join(output_dir, 'edge_numeric_cols_order.json')
                    with open(numeric_order_path, 'w') as f: json.dump(edge_numeric_cols_order, f)
                    print(f"    Saved edge numeric column order to {numeric_order_path}")
                else:
                    print("    Skipping numeric edge feature scaling: No training data.")


            # --- Features Categóricas ---
            categorical_edge_final = [col for col in EDGE_CATEGORICAL_FEATURES if col in train_data.columns]
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

                vocabs_path = os.path.join(output_dir, 'edge_categorical_vocabs.json')
                with open(vocabs_path, 'w') as f: json.dump(edge_feature_vocabs, f, indent=4)
                print(f"    Saved edge categorical vocabularies to {vocabs_path}")
                cat_order_path = os.path.join(output_dir, 'edge_categorical_cols_order.json')
                with open(cat_order_path, 'w') as f: json.dump(edge_categorical_cols_order, f)
                print(f"    Saved edge categorical column order to {cat_order_path}")


        # 5. Salvar Dados das Arestas por Split
        print("Saving processed edge data...")
        for split, df_split in datasets.items():
            split_output_dir = os.path.join(output_dir, split)
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

                    if 'numeric' in edge_features_data[split]:
                         numeric_feats_all = edge_features_data[split]['numeric']
                         original_indices_in_split = df_split.index.get_indexer_for(group_indices) # Usar get_indexer_for
                         edge_features_relation['numeric'] = numeric_feats_all[original_indices_in_split]

                    if 'categorical' in edge_features_data[split]:
                         cat_feats_dict = edge_features_data[split]['categorical']
                         categorical_relation_data = {}
                         original_indices_in_split = df_split.index.get_indexer_for(group_indices) # Usar get_indexer_for
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
                    if 'numeric' in features:
                        save_dict[f"{edge_type_str}_numeric"] = features['numeric']
                    if 'categorical' in features:
                         save_dict[f"{edge_type_str}_categorical"] = features['categorical']
                if save_dict:
                    np.savez(edge_features_path, **save_dict)
                    print(f"    Saved edge features to {edge_features_path}")

        print("--- Edge Data Preparation Completed Successfully ---")
        # --- ALTERAÇÃO: Retornar o timestamp de corte ---
        return train_cutoff_timestamp
        # --- FIM ALTERAÇÃO ---

    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        traceback.print_exc()
    except KeyError as e:
        print(f"Error: Column not found during processing: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    # --- ALTERAÇÃO: Retornar None em caso de erro ---
    return None
    # --- FIM ALTERAÇÃO ---


# --- Bloco para Execução Standalone ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Edge data for GNN training.")
    parser.add_argument('--interactions', required=True, help='Path to the input event_interactions_processed.csv file.')
    parser.add_argument('--person-map', required=True, help='Path to the person_id_to_index_map.json file.')
    parser.add_argument('--product-map', required=True, help='Path to the product_id_to_index_map.json file.')
    parser.add_argument('--output-dir', required=True, help='Directory to save the processed train/valid/test edge data and cutoff timestamp.') # Descrição atualizada
    parser.add_argument('--include-edge-features', action='store_true', help='Include edge features (price, channel) in the output.')

    args = parser.parse_args()

    if not os.path.exists(args.product_map):
         print(f"Warning: Product ID map not found at {args.product_map}. Edge preparation might fail or be incomplete.")

    # A função agora retorna o timestamp, mas não precisamos dele aqui na execução standalone
    prepare_edge_data(
        args.interactions,
        args.person_map,
        args.product_map,
        args.output_dir,
        args.include_edge_features
    )