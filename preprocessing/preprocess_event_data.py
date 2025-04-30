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
from tqdm import tqdm # Import tqdm if not already present in extract_interactions...

# --- Import the core processing function and utils ---
try:
    # Relative import from the sub-package
    from .event_processor.process_event_interactions import extract_interactions_from_events
    # Import the final columns definition as well
    from .event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS as FINAL_INTERACTION_COLUMNS_CSV
    # Import utils if needed by _prepare_gnn_edge_data (not strictly needed here but good practice)
    from .utils import safe_json_parse
except ImportError:
    # Fallback
    print("Warning: Could not use relative imports for event processor/utils. Attempting direct import.")
    try:
        from event_processor.process_event_interactions import extract_interactions_from_events
        from event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS as FINAL_INTERACTION_COLUMNS_CSV
        from utils import safe_json_parse # Assuming utils.py is in the parent directory
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import event_processor module or utils.")
         sys.exit(1)

# --- Configurações GNN ---
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


# --- Função Auxiliar para Preparação GNN (MODIFICADA com DEBUG e timestamp robusto) ---
def _prepare_gnn_edge_data(
    interactions_df: pd.DataFrame, # Recebe o DF já processado
    person_id_map_path: str,
    product_id_map_path: str,
    output_dir_gnn: str,
    include_edge_features: bool
):
    """
    Internal function to prepare and save GNN-ready edge data from the processed interactions df.
    Includes debugging step to save removed interactions and check saved npz files.
    Uses robust two-pass timestamp handling assuming CSV might be reloaded as strings.
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
        person_match = df_gnn[ID_COL_PERSON_GNN].isin(person_id_map)
        product_match = df_gnn[ID_COL_PRODUCT_GNN].isin(product_id_map)
        type_match = df_gnn[INTERACTION_COL_GNN].isin(INTERACTION_TYPE_MAP_GNN)
        keep_mask = person_match & product_match & type_match
        remove_mask = ~keep_mask
        removed_interactions_df = df_gnn[remove_mask].copy()
        num_removed = len(removed_interactions_df)
        print(f"Identified {num_removed} interactions that would be removed by filters.")
        if not removed_interactions_df.empty:
            removed_interactions_df['debug_person_match'] = person_match[remove_mask]
            removed_interactions_df['debug_product_match'] = product_match[remove_mask]
            removed_interactions_df['debug_type_match'] = type_match[remove_mask]
            sample_size = min(1000, num_removed)
            removed_sample_df = removed_interactions_df.head(sample_size)
            debug_output_path = os.path.join(output_dir_gnn, "debug_removed_interactions_sample.csv")
            try:
                removed_sample_df.to_csv(debug_output_path, index=False)
                print(f"Saved a sample of {len(removed_sample_df)} removed interactions to: {debug_output_path}")
            except Exception as e_save:
                 print(f"Error saving debug CSV for removed interactions: {e_save}")
        # --- FIM DEBUG ---

        # Filtrar e Mapear IDs
        print("Filtering interactions by ID maps and mapping IDs...")
        df_gnn = df_gnn[keep_mask].copy()
        filtered_count = len(df_gnn)
        removed_count_actual = initial_count - filtered_count
        print(f"Actually removed {removed_count_actual} interactions (should match identified count).")
        print(f"Remaining interactions after filtering: {filtered_count}")
        if df_gnn.empty:
            print("No valid interactions remaining after map filtering. Skipping GNN edge generation.")
            return

        print("Mapping IDs to indices...")
        df_gnn['person_idx'] = df_gnn[ID_COL_PERSON_GNN].map(person_id_map)
        df_gnn['product_idx'] = df_gnn[ID_COL_PRODUCT_GNN].map(product_id_map)
        df_gnn['relation'] = df_gnn[INTERACTION_COL_GNN].map(INTERACTION_TYPE_MAP_GNN)

        print("\nDEBUG: Value counts for 'relation' column after mapping:")
        print(df_gnn['relation'].value_counts())

        # --- DEBUG PRINT ANTES DA CONVERSÃO DE TIMESTAMP ---
        print("\nDEBUG (inside _prepare_gnn_edge_data): Checking timestamp dtype BEFORE conversion attempt:")
        if TIMESTAMP_COL_GNN in df_gnn.columns:
            print(df_gnn[TIMESTAMP_COL_GNN].dtype)
            print("Sample timestamps BEFORE conversion:")
            print(df_gnn[TIMESTAMP_COL_GNN].head(10).to_string())
            print(df_gnn[TIMESTAMP_COL_GNN].tail(10).to_string())
        else:
            print(f"'{TIMESTAMP_COL_GNN}' column not found before conversion attempt!")
        # --- FIM DEBUG PRINT ---

        # --- VERSÃO ROBUSTA Timestamp Handling (Two-Pass) ---
        print("Ensuring timestamp column is datetime and sorting...")
        if TIMESTAMP_COL_GNN not in df_gnn.columns:
             print(f"Error: Timestamp column '{TIMESTAMP_COL_GNN}' missing. Cannot proceed.")
             return # Ou raise error

        # Aplicar a lógica two-pass diretamente aqui
        if not pd.api.types.is_datetime64_any_dtype(df_gnn[TIMESTAMP_COL_GNN]):
            print(f"Warning: Column '{TIMESTAMP_COL_GNN}' is not datetime type. Attempting robust conversion...")
            original_timestamps_f4 = df_gnn[TIMESTAMP_COL_GNN].astype(str).str.strip()
            format_pass1_f4 = '%Y-%m-%d %H:%M:%S.%f%z'
            format_pass2_f4 = '%Y-%m-%d %H:%M:%S%z'

            print(f"  Attempting conversion with format 1: {format_pass1_f4}")
            converted_pass1_f4 = pd.to_datetime(original_timestamps_f4, format=format_pass1_f4, errors='coerce', utc=True)
            failures_pass1_mask_f4 = converted_pass1_f4.isna()
            num_failures_pass1_f4 = failures_pass1_mask_f4.sum()
            print(f"  Pass 1 failed for {num_failures_pass1_f4} rows.")

            converted_pass2_f4 = converted_pass1_f4.copy()
            if num_failures_pass1_f4 > 0:
                print(f"  Attempting conversion with format 2: {format_pass2_f4} on failed rows...")
                converted_pass2_f4[failures_pass1_mask_f4] = pd.to_datetime(
                    original_timestamps_f4[failures_pass1_mask_f4],
                    format=format_pass2_f4,
                    errors='coerce',
                    utc=True
                )
                num_failures_pass2_f4 = converted_pass2_f4.isna().sum()
                print(f"  Total NaNs after both passes: {num_failures_pass2_f4}")
                if num_failures_pass2_f4 < num_failures_pass1_f4:
                     print(f"  Pass 2 successfully converted {num_failures_pass1_f4 - num_failures_pass2_f4} timestamps.")
            else:
                print("  No failures in Pass 1, skipping Pass 2.")
                num_failures_pass2_f4 = 0

            df_gnn[TIMESTAMP_COL_GNN] = converted_pass2_f4 # Assign the result back
        else:
            print(f"Column '{TIMESTAMP_COL_GNN}' is already datetime type. Verifying timezone...")
            # Ensure it's UTC aware if it is datetime
            if df_gnn[TIMESTAMP_COL_GNN].dt.tz is None:
                print(f"Warning: Column '{TIMESTAMP_COL_GNN}' is datetime but timezone naive. Localizing to UTC...")
                try:
                    df_gnn[TIMESTAMP_COL_GNN] = df_gnn[TIMESTAMP_COL_GNN].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                except Exception as tz_err:
                    print(f"Error localizing timezone: {tz_err}. NaTs might be introduced.")
            elif str(df_gnn[TIMESTAMP_COL_GNN].dt.tz) != 'UTC':
                print(f"Warning: Column '{TIMESTAMP_COL_GNN}' has timezone {df_gnn[TIMESTAMP_COL_GNN].dt.tz}. Converting to UTC...")
                df_gnn[TIMESTAMP_COL_GNN] = df_gnn[TIMESTAMP_COL_GNN].dt.tz_convert('UTC')

        # Drop rows where timestamp became NaT AFTER the potential conversions above
        initial_rows_before_ts_dropna = len(df_gnn)
        df_gnn.dropna(subset=[TIMESTAMP_COL_GNN], inplace=True)
        rows_dropped_ts = initial_rows_before_ts_dropna - len(df_gnn)
        if rows_dropped_ts > 0:
             print(f"Dropped {rows_dropped_ts} interactions due to NaT timestamps during final check/conversion.")
        else:
             print("No rows dropped due to NaT timestamps after robust conversion.")

        if df_gnn.empty:
             print("No valid interactions remaining after final timestamp check. Skipping GNN edge generation.")
             return
        # --- FIM Timestamp Handling Robusto ---

        # --- DEBUG PRINT DEPOIS DA CONVERSÃO DE TIMESTAMP ---
        print("\nDEBUG (inside _prepare_gnn_edge_data): Checking timestamp dtype AFTER conversion attempt:")
        if TIMESTAMP_COL_GNN in df_gnn.columns:
            print(df_gnn[TIMESTAMP_COL_GNN].dtype)
            print("Sample timestamps AFTER conversion:")
            print(df_gnn[TIMESTAMP_COL_GNN].head(10).to_string())
            print(df_gnn[TIMESTAMP_COL_GNN].tail(10).to_string())
        else:
            print(f"'{TIMESTAMP_COL_GNN}' column not found AFTER conversion attempt!")
        # --- FIM DEBUG PRINT ---

        # Now sort by the verified timestamp column
        print(f"Sorting {len(df_gnn)} interactions by timestamp...")
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

            print(f"\nDEBUG: Processing groups for split '{split}':")
            for relation, group in grouped:
                print(f"  -> Processing relation group: {relation}, Number of edges: {len(group)}")
                edge_type = ('Person', relation, 'Product')
                src = group['person_idx'].values
                dst = group['product_idx'].values
                edge_index_dict[str(edge_type)] = np.stack([src, dst], axis=0)

                if include_edge_features and split in edge_features_data:
                    edge_features_relation = {}
                    group_indices = group.index
                    original_indices_in_split = df_split.index.get_indexer(group_indices)

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

            # Salvar edge_index (estrutura do grafo)
            edge_index_path = os.path.join(split_output_dir, 'edge_index.npz')
            print(f"    DEBUG: Dictionary keys intended for edge_index.npz in split '{split}': {list(edge_index_dict.keys())}")
            print(f"    DEBUG: Dictionary intended for edge_index.npz has {len(edge_index_dict)} items.")
            try:
                np.savez(edge_index_path, **edge_index_dict)
                print(f"    Successfully called np.savez for edge_index '{edge_index_path}'.")
                try:
                    reloaded_npz = np.load(edge_index_path)
                    keys_actually_saved = list(reloaded_npz.files)
                    reloaded_npz.close()
                    print(f"    DEBUG: Keys found immediately after reloading '{edge_index_path}': {keys_actually_saved}")
                    if len(keys_actually_saved) != len(edge_index_dict):
                         print(f"    ERROR DEBUG: Mismatch! Intended to save {len(edge_index_dict)} keys but file contains {len(keys_actually_saved)}.")
                         intended_keys = set(edge_index_dict.keys())
                         missing_keys = intended_keys - set(keys_actually_saved)
                         if missing_keys: print(f"     Missing keys: {missing_keys}")
                except Exception as e_reload:
                    print(f"    ERROR DEBUG: Failed to reload npz immediately after saving: {e_reload}")
            except Exception as e_savez:
                 print(f"    ERROR during np.savez call for edge_index: {e_savez}")
                 traceback.print_exc()
            print(f"    Saving edge indices finished for {edge_index_path}")

            # Salvar features de aresta (se preparadas)
            if include_edge_features and edge_features_split_dict:
                edge_features_path = os.path.join(split_output_dir, 'edge_features.npz')
                save_dict = {}
                for edge_type_str, features in edge_features_split_dict.items():
                    if 'numeric' in features: save_dict[f"{edge_type_str}_numeric"] = features['numeric']
                    if 'categorical' in features: save_dict[f"{edge_type_str}_categorical"] = features['categorical']
                if save_dict:
                    try:
                        np.savez(edge_features_path, **save_dict)
                        print(f"    Saved edge features to {edge_features_path}")
                    except Exception as e_savez_feat:
                         print(f"    ERROR during np.savez call for edge_features: {e_savez_feat}")
                         traceback.print_exc()

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


# --- Função Principal ---
def run_event_preprocessing(
    input_path: str,
    output_path: str, # Caminho para o CSV intermediário
    person_id_map_path: str,
    product_id_map_path: str,
    edge_gnn_output_dir: str,
    include_edge_features: bool
):
    """
    Orchestrates the preprocessing of event data to extract interactions,
    saves the intermediate CSV, AND prepares/saves GNN-ready edge data.
    """
    print(f"\n--- Starting Event Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Intermediate Output CSV: {output_path}")
    print(f"Person ID Map: {person_id_map_path}")
    print(f"Product ID Map: {product_id_map_path}")
    print(f"GNN Edges Output Dir: {edge_gnn_output_dir}")
    print(f"Include Edge Features: {include_edge_features}")

    interactions_df = pd.DataFrame() # Initialize

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
        # This function now handles the two-pass timestamp conversion internally
        interactions_df = extract_interactions_from_events(event_df)
    except Exception as e:
        print(f"Error during event interaction extraction: {e}")
        traceback.print_exc()
        raise # Stop if extraction fails

    # --- Save Intermediate Processed Interactions CSV ---
    try:
        if interactions_df.empty:
             print("\nInteraction DataFrame is empty after processing. Saving empty intermediate file.")
             output_dir_csv = os.path.dirname(output_path)
             if output_dir_csv: os.makedirs(output_dir_csv, exist_ok=True)
             pd.DataFrame(columns=FINAL_INTERACTION_COLUMNS_CSV).to_csv(output_path, index=False, na_rep='')
        else:
            output_dir_csv = os.path.dirname(output_path)
            os.makedirs(output_dir_csv, exist_ok=True)
            print(f"\nEnsured intermediate output directory exists: {output_dir_csv}")
            print("\nData types in final DataFrame before save:")
            print(interactions_df.info())
            # Save with na_rep='' to handle potential pandas NA types correctly in CSV
            interactions_df.to_csv(output_path, index=False, na_rep='')
            print(f"Successfully saved intermediate processed event interactions to: {output_path}")

    except Exception as e:
        print(f"Error saving intermediate event interactions CSV to '{output_path}': {e}")
        traceback.print_exc()
        # Consider stopping here
        # raise

    # --- Preparar e Salvar Dados de Aresta para GNN ---
    # This step is only called explicitly in Stage 4 of run_preprocessing.py
    # The call here might be considered redundant in the 'full_pipeline' flow,
    # but allows running this script standalone to generate CSV + attempt GNN prep (if maps exist).
    # NOTE: The _prepare_gnn_edge_data function now handles robust timestamp parsing internally.
    try:
        if not interactions_df.empty:
            # Check if maps exist before calling the GNN prep function
            if not os.path.exists(person_id_map_path):
                 print(f"Warning/Error: Cannot prepare GNN edges YET. Person ID map not found at: {person_id_map_path}")
            elif not os.path.exists(product_id_map_path):
                 print(f"Error: Cannot prepare GNN edges. Product ID map not found at: {product_id_map_path}")
            else:
                 # Call the GNN preparation function (with robust timestamp logic included)
                 _prepare_gnn_edge_data(
                     interactions_df, # Pass the extracted DF (timestamps should be correct)
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


# --- Command Line Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess event data, extract interactions, and prepare GNN edge data.')
    parser.add_argument('--input', required=True, help='Path to the input person_event_data_filtered.csv file.')
    parser.add_argument('--output-csv', required=True, help='Path to save the intermediate event_interactions_processed.csv file.')
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