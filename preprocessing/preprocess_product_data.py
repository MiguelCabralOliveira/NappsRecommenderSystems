# preprocessing/preprocess_product_data.py
import argparse
import os
import pandas as pd
import traceback
import re
import numpy as np
import json # Para salvar mapas e ordens de colunas
import joblib # Para salvar scalers
from sklearn.preprocessing import RobustScaler # Ou outro scaler

# --- Import all necessary processor functions ---
try:
    from .product_processor.availableForSale_processor import process_avaiable_for_sale
    from .product_processor.collectionHandle_processor import process_collections
    from .product_processor.createdAt_processor import process_created_at
    from .product_processor.handle_processor import process_handles_tfidf
    from .product_processor.isGiftCard_processor import process_gift_card
    from .product_processor.metafields_processor import process_metafields
    from .product_processor.product_type_processor import process_product_type
    from .product_processor.tags_processor import process_tags
    from .product_processor.title_processor import process_titles_tfidf
    from .product_processor.variants_processor import process_variants
    from .product_processor.vendor_processor import process_vendors
    from .product_processor.tfidf_processor import process_descriptions_tfidf
except ImportError as e:
    print(f"Error importing product processors: {e}. Check paths and __init__.py files.")
    raise

# --- Colunas para GNN Produto (Exemplo - AJUSTE CONFORME SEU CSV FINAL) ---
# Esta lista deve ser refinada com base nas colunas reais geradas
NUMERIC_COLS_TO_SCALE_GNN_PROD = [
    'creation_recency_norm', 'price_min', 'price_max', 'price_avg',
    'discount_avg_perc', 'has_discount', 'inventory_total', 'n_variants',
    # Colunas TF-IDF, Dummies e Metafields Numéricos serão adicionadas dinamicamente abaixo
]

CATEGORICAL_COLS_TO_EMBED_GNN_PROD = [
    # Adicione aqui colunas categóricas de produto que *não* são dummies, se houver
    # Ex: 'product_type_raw' se você mantiver a coluna original e não usar dummies
]

ID_COL_GNN_PROD = 'product_id'
# --- Fim Colunas GNN Produto ---


# --- Função Auxiliar para Preparação GNN Produto (Movida para dentro) ---
def _prepare_product_gnn_features(df_final: pd.DataFrame, output_dir_gnn: str):
    """
    Internal function to prepare and save GNN-ready features from the final product df.
    """
    print(f"\n--- Preparing Product Node Features for GNN ---")
    print(f"Output directory for GNN data: {output_dir_gnn}")
    os.makedirs(output_dir_gnn, exist_ok=True)

    if df_final.empty:
        print("Input DataFrame for GNN prep is empty. Skipping GNN feature generation.")
        return

    if ID_COL_GNN_PROD not in df_final.columns:
         print(f"CRITICAL Error: ID column '{ID_COL_GNN_PROD}' not found. Cannot prepare GNN features.")
         return

    # Identificar colunas TF-IDF e Dummies automaticamente
    tfidf_cols = [col for col in df_final.columns if 'tfidf' in col.lower()] # Mais robusto
    dummy_cols = [col for col in df_final.columns if col.startswith(('coll_', 'age_', 'quarter_', 'prodtype_', 'vendor_', 'tag_', 'opt_'))]
    meta_numeric_cols = [col for col in df_final.columns if col.startswith('meta_') and pd.api.types.is_numeric_dtype(df_final[col])]

    # Atualizar lista de colunas numéricas GNN
    # Começa com as definidas manualmente e adiciona as detectadas dinamicamente
    numeric_cols_gnn_base = [col for col in NUMERIC_COLS_TO_SCALE_GNN_PROD if col in df_final.columns]
    numeric_cols_gnn = list(set(numeric_cols_gnn_base + tfidf_cols + dummy_cols + meta_numeric_cols))
    # Garantir que a coluna ID não está na lista de features
    numeric_cols_gnn = [col for col in numeric_cols_gnn if col != ID_COL_GNN_PROD and col in df_final.columns]

    categorical_cols_gnn = [col for col in CATEGORICAL_COLS_TO_EMBED_GNN_PROD if col in df_final.columns]

    print(f"Identified {len(numeric_cols_gnn)} numeric features for GNN (incl. TF-IDF/Dummies/Meta).")
    print(f"Identified {len(categorical_cols_gnn)} categorical features for GNN embedding.")

    # Selecionar Features para GNN
    cols_to_keep_for_gnn = [ID_COL_GNN_PROD] + numeric_cols_gnn + categorical_cols_gnn
    df_gnn = df_final[cols_to_keep_for_gnn].copy()
    print(f"Selected {len(df_gnn.columns) - 1} features for GNN processing.")

    # Mapeamento ID -> Índice
    print("Creating Product ID to index mapping...")
    df_gnn = df_gnn.sort_values(by=ID_COL_GNN_PROD).reset_index(drop=True) # Ordenar por ID
    product_id_to_index = {str(prod_id): index for index, prod_id in enumerate(df_gnn[ID_COL_GNN_PROD])} # Garantir string
    id_map_path = os.path.join(output_dir_gnn, 'product_id_to_index_map.json')
    with open(id_map_path, 'w') as f:
        json.dump(product_id_to_index, f)
    print(f"Saved Product ID mapping to {id_map_path}")

    df_features_gnn = df_gnn.drop(columns=[ID_COL_GNN_PROD])

    # Processar Features Numéricas GNN
    print("Processing numeric features for GNN...")
    scaled_numeric_features = np.zeros((len(df_features_gnn), 0))
    if numeric_cols_gnn:
        df_numeric_gnn = df_features_gnn[numeric_cols_gnn].copy()
        initial_nans = df_numeric_gnn.isnull().sum().sum()
        if initial_nans > 0:
            print(f"Imputing {initial_nans} NaN values in numeric GNN features with 0...")
            df_numeric_gnn.fillna(0, inplace=True)
        # Usar StandardScaler pode ser melhor se TF-IDF/Dummies dominarem
        scaler = RobustScaler() # Ou StandardScaler()
        print(f"Applying {type(scaler).__name__}...")
        # Garantir que todas as colunas são numéricas antes de escalar
        for col in df_numeric_gnn.columns:
             df_numeric_gnn[col] = pd.to_numeric(df_numeric_gnn[col], errors='coerce')
        df_numeric_gnn.fillna(0, inplace=True) # Preencher NaNs introduzidos por to_numeric

        scaled_numeric_features = scaler.fit_transform(df_numeric_gnn)
        scaler_path = os.path.join(output_dir_gnn, 'product_numeric_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Saved numeric scaler to {scaler_path}")
    else:
        print("No numeric features selected for GNN.")

    numeric_features_path = os.path.join(output_dir_gnn, 'product_features_numeric_scaled.npy')
    np.save(numeric_features_path, scaled_numeric_features)
    print(f"Saved scaled numeric features to {numeric_features_path} (Shape: {scaled_numeric_features.shape})")
    numeric_cols_order_path = os.path.join(output_dir_gnn, 'product_numeric_cols_order.json')
    with open(numeric_cols_order_path, 'w') as f: json.dump(numeric_cols_gnn, f) # Salva a lista final usada
    print(f"Saved numeric column order to {numeric_cols_order_path}")


    # Processar Features Categóricas GNN (se houver)
    print("Processing categorical features for GNN...")
    encoded_categorical_array = np.zeros((len(df_features_gnn), 0))
    vocabularies = {}
    categorical_cols_order = []
    if categorical_cols_gnn:
        df_categorical_gnn = df_features_gnn[categorical_cols_gnn].copy()
        encoded_categorical_data = {}
        for col in categorical_cols_gnn:
            print(f"  Encoding column: {col}")
            df_categorical_gnn[col] = df_categorical_gnn[col].fillna('__UNKNOWN__').astype(str)
            unique_values = sorted([v for v in df_categorical_gnn[col].unique() if v != '__UNKNOWN__'])
            vocab = {val: i+1 for i, val in enumerate(unique_values)}
            vocab['__UNKNOWN__'] = 0
            vocabularies[col] = vocab
            print(f"    Vocabulary size for {col}: {len(vocab)}")
            encoded_categorical_data[col] = df_categorical_gnn[col].map(vocab).values
            categorical_cols_order.append(col)
        encoded_categorical_array = np.stack(list(encoded_categorical_data.values()), axis=1)
    else:
        print("No categorical features selected for GNN embedding.")

    vocabs_path = os.path.join(output_dir_gnn, 'product_categorical_vocabs.json')
    with open(vocabs_path, 'w') as f: json.dump(vocabularies, f, indent=4)
    print(f"Saved categorical vocabularies to {vocabs_path}")

    categorical_features_path = os.path.join(output_dir_gnn, 'product_features_categorical_indices.npy')
    np.save(categorical_features_path, encoded_categorical_array)
    print(f"Saved encoded categorical features (indices) to {categorical_features_path} (Shape: {encoded_categorical_array.shape})")
    categorical_cols_order_path = os.path.join(output_dir_gnn, 'product_categorical_cols_order.json')
    with open(categorical_cols_order_path, 'w') as f: json.dump(categorical_cols_order, f) # Salva a lista final usada
    print(f"Saved categorical column order to {categorical_cols_order_path}")

    print("--- Product Node Feature Preparation for GNN Finished ---")
# --- Fim Função Auxiliar GNN Produto ---


# --- Função Principal Atualizada ---
# --- ASSINATURA ATUALIZADA: Recebe product_gnn_output_dir ---
def run_product_preprocessing(
    input_path: str,
    output_path: str, # Caminho para o CSV intermediário
    shop_id: str,
    product_gnn_output_dir: str # Novo: Diretório para saídas GNN
):
    """
    Orchestrates the preprocessing of product data, saves intermediate CSV,
    AND prepares/saves GNN-ready node features.

    Args:
        input_path: Path to the raw product data CSV.
        output_path: Path to save the intermediate processed product features CSV.
        shop_id: The identifier for the shop being processed.
        product_gnn_output_dir: Directory to save GNN-ready product features, scaler, vocabs, ID map.
    """
    print(f"\n--- Starting Product Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Intermediate Output CSV: {output_path}")
    print(f"Shop ID provided: {shop_id}")
    print(f"GNN Features Output Dir: {product_gnn_output_dir}") # Log do novo dir

    # Base output directory para modelos TF-IDF, etc. (do CSV intermediário)
    output_dir_intermediate = os.path.dirname(output_path)
    models_output_dir_intermediate = os.path.join(output_dir_intermediate, "models")
    os.makedirs(models_output_dir_intermediate, exist_ok=True)
    print(f"Intermediate output directory: {output_dir_intermediate}")
    print(f"Intermediate models output directory: {models_output_dir_intermediate}")

    final_df = pd.DataFrame() # Inicializar

    # --- Load Raw Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {len(df)} raw product rows from {input_path}.")
        if df.empty:
            print("Input DataFrame is empty. Creating empty output file and skipping GNN prep.")
            pd.DataFrame(columns=['product_id']).to_csv(output_path, index=False)
            # Criar arquivos GNN vazios? Ou apenas retornar? Retornar é mais simples.
            return
        if 'product_id' not in df.columns:
            raise ValueError("Input product data must contain a 'product_id' column.")
        df['product_id'] = df['product_id'].astype(str)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error loading input product CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Apply Processors Sequentially ---
    processed_df = df.copy()
    try:
        # 1. Filtering
        processed_df = process_gift_card(processed_df)
        if processed_df.empty: raise ValueError("DataFrame empty after gift card filter.")
        processed_df = process_avaiable_for_sale(processed_df)
        if processed_df.empty: raise ValueError("DataFrame empty after available_for_sale filter.")

        # 2. Basic Column Transformations & Dummies
        processed_df = process_created_at(processed_df)
        processed_df = process_product_type(processed_df)
        processed_df = process_vendors(processed_df)

        # 3. List/Dict Column Processing & Dummies
        processed_df = process_tags(processed_df)
        processed_df = process_collections(processed_df)

        # 4. Variant Processing
        processed_df = process_variants(processed_df)

        # 5. Text Processing (TF-IDF)
        processed_df = process_titles_tfidf(processed_df, models_output_dir_intermediate, shop_id)
        processed_df = process_handles_tfidf(processed_df, models_output_dir_intermediate, shop_id)
        processed_df = process_descriptions_tfidf(processed_df, output_dir_intermediate, shop_id)

        # 6. Metafields Processing
        processed_df, _, _ = process_metafields(processed_df, output_dir_intermediate, shop_id)

        # --- Final Checks ---
        print("\nPerforming final checks on processed product data...")
        if 'product_id' not in processed_df.columns:
            raise ValueError("'product_id' column lost during processing!")
        processed_df['product_id'] = processed_df['product_id'].astype(str)

        # Preencher NaNs restantes em colunas numéricas com 0 ANTES de salvar CSV e preparar GNN
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if processed_df[col].isnull().any():
                # print(f"Filling remaining NaNs in numeric column '{col}' with 0.") # Reduzir verbosidade
                processed_df[col] = processed_df[col].fillna(0)

        # Garantir product_id como primeira coluna
        cols_order = ['product_id'] + [col for col in processed_df.columns if col != 'product_id']
        final_df = processed_df[cols_order].copy() # Usar cópia para evitar SettingWithCopyWarning

        print(f"\nFinal Intermediate Product DataFrame shape: {final_df.shape}")
        # print(f"Final columns ({len(final_df.columns)}): {final_df.columns.tolist()}") # Opcional

    except ValueError as ve:
         print(f"\n--- Processing stopped due to empty data after a step: {ve} ---")
         # Salvar CSV vazio e pular GNN prep
         pd.DataFrame(columns=['product_id']).to_csv(output_path, index=False)
         print(f"Empty intermediate product CSV saved to: {output_path}")
         print("Skipping GNN product feature preparation due to empty DataFrame.")
         return # Sair da função aqui
    except Exception as e:
        print(f"\n--- Error during product processing sequence: {e} ---")
        traceback.print_exc()
        raise # Propaga o erro, não salva CSV nem prepara GNN

    # --- Save Intermediate Final Processed Data ---
    try:
        os.makedirs(output_dir_intermediate, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved intermediate processed product data to: {output_path}")
    except Exception as e:
        print(f"Error saving intermediate processed product CSV to '{output_path}': {e}")
        traceback.print_exc()
        # Considerar se deve parar ou tentar GNN prep
        # raise

    # --- Preparar e Salvar Features para GNN ---
    # Usa o 'final_df' que foi salvo no CSV
    try:
        _prepare_product_gnn_features(final_df, product_gnn_output_dir)
    except Exception as e:
        print(f"Error during GNN product feature preparation step: {e}")
        traceback.print_exc()
        print("Warning: GNN product feature preparation failed.")

    print(f"--- Finished Product Data Preprocessing Orchestration (including GNN prep) ---")


# --- Command Line Execution Logic (Atualizado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess product data and prepare GNN node features.')
    parser.add_argument('--input', required=True, help='Path to the input products_raw.csv file.')
    parser.add_argument('--output-csv', required=True, help='Path to save the intermediate products_final_all_features.csv file.')
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop.')
    # --- NOVO ARGUMENTO ---
    parser.add_argument('--output-gnn-dir', required=True, help='Directory to save GNN-ready product features, scaler, vocabs, and ID map.')
    args = parser.parse_args()

    try:
        # --- CHAMADA ATUALIZADA ---
        run_product_preprocessing(
            args.input,
            args.output_csv,
            args.shop_id,
            args.output_gnn_dir # Passar novo argumento
        )
        print("\nProduct preprocessing & GNN prep finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error: # Captura empty DataFrame errors etc.
        print(f"\nError: {val_error}")
        exit(1)
    except Exception as e:
        print(f"\n--- Product preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)