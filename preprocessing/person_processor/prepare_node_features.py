# preprocessing/person_processor/prepare_node_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import json
import os
import argparse
import traceback

# --- Colunas Definidas na Análise Atual ---
NUMERIC_COLS_TO_SCALE = [
    'is_identified',
    'days_since_creation',
    'days_since_last_update',
    'shopify_account_age_days',
    'in_eu',
    'unique_products_interacted',
    'view_count',
    'purchase_count',
    'wishlist_count',
    'share_count',
    'checkout_start_count',
    'distinct_channels_count',
    'variant_meta_added_count',
    'checkout_update_count',
    'order_update_count',
    'days_since_last_interaction',
    'days_since_first_interaction',
    'avg_purchase_value'
]

CATEGORICAL_COLS_TO_EMBED = [
    'shopify_currency',
    'country_iso',
    'timezone',
    'most_frequent_channel'
]

# Colunas a serem explicitamente descartadas do input GNN
COLS_TO_DISCARD_FROM_GNN = [
    'total_interactions',
    'is_buyer'
]

ID_COL = 'person_id_hex'
# --- Fim Definição Colunas ---

def prepare_person_features(input_csv_path: str, output_dir: str):
    """
    Loads final processed person features, selects for GNN, scales numeric,
    encodes categorical, and saves the processed features, scalers, vocabs, and ID map.
    """
    print(f"--- Starting Person Node Feature Preparation for GNN ---")
    print(f"Input file: {input_csv_path}")
    print(f"Output directory: {output_dir}")

    # Garante que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Carregar Dados
        print("Loading data...")
        df = pd.read_csv(input_csv_path, low_memory=False)
        if df.empty:
            print("Input DataFrame is empty. Exiting.")
            return
        print(f"Loaded {len(df)} rows.")

        # Verificar ID Col
        if ID_COL not in df.columns:
             raise ValueError(f"Critical Error: ID column '{ID_COL}' not found in input CSV.")

        # Garantir que colunas a processar existem, ajustar listas se necessário
        available_cols = df.columns.tolist()
        numeric_final = [col for col in NUMERIC_COLS_TO_SCALE if col in available_cols]
        categorical_final = [col for col in CATEGORICAL_COLS_TO_EMBED if col in available_cols]
        missing_numeric = [col for col in NUMERIC_COLS_TO_SCALE if col not in available_cols]
        missing_categorical = [col for col in CATEGORICAL_COLS_TO_EMBED if col not in available_cols]
        if missing_numeric: print(f"Warning: Missing expected numeric columns: {missing_numeric}")
        if missing_categorical: print(f"Warning: Missing expected categorical columns: {missing_categorical}")

        # 2. Seleção de Features para GNN
        print("Selecting final features for GNN...")
        cols_to_keep_for_gnn = [ID_COL] + numeric_final + categorical_final
        df_processed = df[cols_to_keep_for_gnn].copy()
        print(f"Selected {len(df_processed.columns) - 1} features for GNN processing.")

        # 3. Mapeamento ID -> Índice (antes de qualquer reordenação)
        print("Creating ID to index mapping...")
        # Ordenar por ID para garantir consistência se a ordem do CSV mudar
        df_processed = df_processed.sort_values(by=ID_COL).reset_index(drop=True)
        person_id_to_index = {person_id: index for index, person_id in enumerate(df_processed[ID_COL])}
        id_map_path = os.path.join(output_dir, 'person_id_to_index_map.json')
        with open(id_map_path, 'w') as f:
            json.dump(person_id_to_index, f)
        print(f"Saved ID mapping to {id_map_path} (Sorted by ID)")

        # Guardar o ID para referência, mas removê-lo das features a processar
        # person_ids = df_processed[ID_COL] # Não precisamos mais guardar separadamente
        df_features = df_processed.drop(columns=[ID_COL])

        # 4. Separar e Tratar Features Numéricas
        print("Processing numeric features...")
        if not numeric_final:
            print("No numeric features to process.")
            # Criar um array vazio para consistência, se necessário
            scaled_numeric_features = np.zeros((len(df_features), 0))
        else:
            df_numeric = df_features[numeric_final].copy()
            initial_nans = df_numeric.isnull().sum().sum()
            if initial_nans > 0:
                print(f"Imputing {initial_nans} NaN values in numeric features with 0...")
                df_numeric.fillna(0, inplace=True)

            scaler = RobustScaler()
            print("Applying RobustScaler...")
            scaled_numeric_features = scaler.fit_transform(df_numeric)
            print("Numeric features scaled.")

            scaler_path = os.path.join(output_dir, 'person_numeric_scaler.joblib')
            joblib.dump(scaler, scaler_path)
            print(f"Saved numeric scaler to {scaler_path}")

        numeric_features_path = os.path.join(output_dir, 'person_features_numeric_scaled.npy')
        np.save(numeric_features_path, scaled_numeric_features)
        print(f"Saved scaled numeric features to {numeric_features_path} (Shape: {scaled_numeric_features.shape})")
        # Salvar nomes das colunas numéricas na ordem correta
        numeric_cols_order_path = os.path.join(output_dir, 'person_numeric_cols_order.json')
        with open(numeric_cols_order_path, 'w') as f:
            json.dump(numeric_final, f)
        print(f"Saved numeric column order to {numeric_cols_order_path}")


        # 5. Separar e Tratar Features Categóricas
        print("Processing categorical features...")
        if not categorical_final:
            print("No categorical features to process.")
            encoded_categorical_array = np.zeros((len(df_features), 0))
            vocabularies = {}
            categorical_cols_order = []
        else:
            df_categorical = df_features[categorical_final].copy()
            vocabularies = {}
            encoded_categorical_data = {}
            categorical_cols_order = [] # Manter a ordem

            for col in categorical_final:
                print(f"  Encoding column: {col}")
                df_categorical[col] = df_categorical[col].fillna('__UNKNOWN__').astype(str)
                unique_values = df_categorical[col].unique()
                # Adicionar explicitamente __UNKNOWN__ se não estiver presente mas foi usado
                if '__UNKNOWN__' not in unique_values and (df_categorical[col] == '__UNKNOWN__').any():
                     unique_values = np.append(unique_values, '__UNKNOWN__')
                # Ordenar valores para vocabulário consistente (exceto __UNKNOWN__ no início)
                unique_values = sorted([v for v in unique_values if v != '__UNKNOWN__'])
                vocab = {val: i+1 for i, val in enumerate(unique_values)} # Começa índice em 1
                vocab['__UNKNOWN__'] = 0 # Índice 0 para desconhecido/padding
                vocabularies[col] = vocab
                print(f"    Vocabulary size for {col}: {len(vocab)}")

                encoded_categorical_data[col] = df_categorical[col].map(vocab).values
                categorical_cols_order.append(col) # Guarda a ordem

            encoded_categorical_array = np.stack(list(encoded_categorical_data.values()), axis=1)

        # Salvar os vocabulários
        vocabs_path = os.path.join(output_dir, 'person_categorical_vocabs.json')
        with open(vocabs_path, 'w') as f:
            json.dump(vocabularies, f, indent=4)
        print(f"Saved categorical vocabularies to {vocabs_path}")

        # Salvar features categóricas codificadas
        categorical_features_path = os.path.join(output_dir, 'person_features_categorical_indices.npy')
        np.save(categorical_features_path, encoded_categorical_array)
        print(f"Saved encoded categorical features (indices) to {categorical_features_path} (Shape: {encoded_categorical_array.shape})")
        # Salvar nomes das colunas categóricas na ordem correta
        categorical_cols_order_path = os.path.join(output_dir, 'person_categorical_cols_order.json')
        with open(categorical_cols_order_path, 'w') as f:
            json.dump(categorical_cols_order, f)
        print(f"Saved categorical column order to {categorical_cols_order_path}")


        print("--- Person Node Feature Preparation Completed Successfully ---")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        traceback.print_exc()
    except KeyError as e:
        print(f"Error: Column not found during processing: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

# --- Bloco para Execução Standalone ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Person node features for GNN.")
    parser.add_argument('--input', required=True, help='Path to the input shop_person_features_processed.csv file.')
    parser.add_argument('--output-dir', required=True, help='Directory to save the processed features, scaler, vocabs, and ID map.')

    args = parser.parse_args()

    prepare_person_features(args.input, args.output_dir)