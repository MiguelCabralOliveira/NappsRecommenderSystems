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
):
    """
    Loads interactions, maps IDs, splits temporally, and saves edge data for GNN.
    Optionally prepares and saves edge features.
    """
    print(f"--- Starting Edge Data Preparation for GNN ---")
    print(f"Interactions file: {interactions_path}")
    print(f"Person ID map: {person_id_map_path}")
    print(f"Product ID map: {product_id_map_path}")
    print(f"Output directory: {output_dir}")
    print(f"Include edge features: {include_edge_features}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Carregar Dados e Mapeamentos
        print("Loading interactions and ID maps...")
        interactions_df = pd.read_csv(interactions_path, low_memory=False, parse_dates=[TIMESTAMP_COL])
        if interactions_df.empty:
            print("Interactions DataFrame is empty. Exiting.")
            return

        with open(person_id_map_path, 'r') as f:
            person_id_map = json.load(f)
        with open(product_id_map_path, 'r') as f:
            product_id_map = json.load(f)

        print(f"Loaded {len(interactions_df)} interactions.")
        print(f"Loaded {len(person_id_map)} person ID mappings.")
        print(f"Loaded {len(product_id_map)} product ID mappings.")

        # 2. Filtrar Interações Válidas e Mapear IDs
        print("Filtering interactions and mapping IDs...")
        initial_count = len(interactions_df)
        # Manter apenas interações onde ambos os IDs existem nos mapas
        interactions_df = interactions_df[
            interactions_df[ID_COL_PERSON].isin(person_id_map) &
            interactions_df[ID_COL_PRODUCT].isin(product_id_map)
        ]
        # Manter apenas interações cujo tipo está no mapa de relações
        interactions_df = interactions_df[interactions_df[INTERACTION_COL].isin(INTERACTION_TYPE_MAP)]
        filtered_count = len(interactions_df)
        print(f"Removed {initial_count - filtered_count} interactions due to missing IDs or unmapped interaction types.")

        if interactions_df.empty:
            print("No valid interactions remaining after filtering. Exiting.")
            return

        # Mapear IDs para índices inteiros
        interactions_df['person_idx'] = interactions_df[ID_COL_PERSON].map(person_id_map)
        interactions_df['product_idx'] = interactions_df[ID_COL_PRODUCT].map(product_id_map)

        # Mapear interaction_type para nome da relação
        interactions_df['relation'] = interactions_df[INTERACTION_COL].map(INTERACTION_TYPE_MAP)

        # Garantir que timestamp é datetime UTC
        if not pd.api.types.is_datetime64_any_dtype(interactions_df[TIMESTAMP_COL]):
             interactions_df[TIMESTAMP_COL] = pd.to_datetime(interactions_df[TIMESTAMP_COL], errors='coerce', utc=True)
        elif interactions_df[TIMESTAMP_COL].dt.tz is None:
             interactions_df[TIMESTAMP_COL] = interactions_df[TIMESTAMP_COL].dt.tz_localize('UTC')
        else:
             interactions_df[TIMESTAMP_COL] = interactions_df[TIMESTAMP_COL].dt.tz_convert('UTC')
        interactions_df.dropna(subset=[TIMESTAMP_COL], inplace=True) # Remover falhas de conversão

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

        datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
        edge_features_data = {} # Para guardar features processadas

        # 4. (Opcional) Preparar Features de Aresta
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
                # Fit no treino, transforma todos
                train_numeric = train_data[numeric_edge_final].fillna(0) # Imputar NaNs
                scaler.fit(train_numeric)
                edge_feature_scalers['numeric'] = scaler
                edge_numeric_cols_order = numeric_edge_final # Guardar ordem

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

            # --- Features Categóricas ---
            categorical_edge_final = [col for col in EDGE_CATEGORICAL_FEATURES if col in train_data.columns]
            if categorical_edge_final:
                print(f"  Processing categorical edge features: {categorical_edge_final}")
                edge_categorical_cols_order = categorical_edge_final # Guardar ordem
                for col in categorical_edge_final:
                    # Criar vocabulário a partir do treino
                    train_cat = train_data[col].fillna('__UNKNOWN__').astype(str)
                    unique_values = sorted([v for v in train_cat.unique() if v != '__UNKNOWN__'])
                    vocab = {val: i+1 for i, val in enumerate(unique_values)}
                    vocab['__UNKNOWN__'] = 0
                    edge_feature_vocabs[col] = vocab
                    print(f"    Vocabulary size for {col}: {len(vocab)}")

                    # Codificar todos os splits usando o vocabulário do treino
                    for split, df_split in datasets.items():
                        if not df_split.empty:
                            encoded_col = df_split[col].fillna('__UNKNOWN__').astype(str).map(vocab)
                            # Lidar com valores no valid/test que não estavam no treino
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

            # Agrupar por tipo de relação para salvar separadamente (formato comum para PyG/DGL)
            grouped = df_split.groupby('relation')
            edge_index_dict = {}
            edge_features_split_dict = {} # Para features desta divisão

            for relation, group in grouped:
                # Definir tipo de aresta canônico (origem, relação, destino)
                edge_type = ('Person', relation, 'Product')
                # Obter índices de origem e destino
                src = group['person_idx'].values
                dst = group['product_idx'].values
                # Armazenar como array [2, num_edges] - formato comum PyG
                edge_index_dict[str(edge_type)] = np.stack([src, dst], axis=0)

                # Salvar features de aresta para esta relação, se existirem
                if include_edge_features and split in edge_features_data:
                    edge_features_relation = {}
                    group_indices = group.index # Índices originais no df_split

                    # Numéricas
                    if 'numeric' in edge_features_data[split]:
                         # Selecionar as linhas correspondentes do array de features numéricas
                         numeric_feats_all = edge_features_data[split]['numeric']
                         # Precisamos mapear os índices do grupo para as posições corretas no array original do split
                         original_indices_in_split = df_split.index.get_indexer(group_indices)
                         edge_features_relation['numeric'] = numeric_feats_all[original_indices_in_split]


                    # Categóricas
                    if 'categorical' in edge_features_data[split]:
                         cat_feats_dict = edge_features_data[split]['categorical']
                         categorical_relation_data = {}
                         original_indices_in_split = df_split.index.get_indexer(group_indices)
                         for cat_col, encoded_vals_all in cat_feats_dict.items():
                              categorical_relation_data[cat_col] = encoded_vals_all[original_indices_in_split]
                         # Empilhar features categóricas para esta relação
                         if categorical_relation_data:
                             edge_features_relation['categorical'] = np.stack(
                                 [categorical_relation_data[col] for col in edge_categorical_cols_order], # Usar ordem salva
                                 axis=1
                             )

                    if edge_features_relation:
                         edge_features_split_dict[str(edge_type)] = edge_features_relation


            # Salvar edge_index (estrutura do grafo)
            edge_index_path = os.path.join(split_output_dir, 'edge_index.npz')
            np.savez(edge_index_path, **edge_index_dict)
            print(f"    Saved edge indices for {len(edge_index_dict)} relation types to {edge_index_path}")

            # Salvar features de aresta (se preparadas)
            if include_edge_features and edge_features_split_dict:
                edge_features_path = os.path.join(split_output_dir, 'edge_features.npz')
                # Precisamos salvar arrays numpy dentro do npz
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

    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        traceback.print_exc()
    except KeyError as e:
        print(f"Error: Column not found during processing: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


# --- Bloco para Execução Standalone ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Edge data for GNN training.")
    parser.add_argument('--interactions', required=True, help='Path to the input event_interactions_processed.csv file.')
    parser.add_argument('--person-map', required=True, help='Path to the person_id_to_index_map.json file.')
    parser.add_argument('--product-map', required=True, help='Path to the product_id_to_index_map.json file.')
    parser.add_argument('--output-dir', required=True, help='Directory to save the processed train/valid/test edge data.')
    parser.add_argument('--include-edge-features', action='store_true', help='Include edge features (price, channel) in the output.')

    args = parser.parse_args()

    # Assume product map is generated elsewhere (needs a similar script for product features)
    # For testing, you might need to create a dummy product map if the product feature prep script isn't ready
    if not os.path.exists(args.product_map):
         print(f"Warning: Product ID map not found at {args.product_map}. Edge preparation might fail or be incomplete.")
         # Example: Create a dummy map if needed for testing structure
         # dummy_prod_map = {"product1": 0, "product2": 1}
         # with open(args.product_map, 'w') as f: json.dump(dummy_prod_map, f)

    prepare_edge_data(
        args.interactions,
        args.person_map,
        args.product_map,
        args.output_dir,
        args.include_edge_features
    )