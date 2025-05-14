# preprocessing/preprocess_person_data.py
import argparse
import os
import pandas as pd
import numpy as np
import traceback
import sys
import joblib
import json
from sklearn.preprocessing import RobustScaler

# Use relative imports for the processing functions within the person_processor sub-package
try:
    from .person_processor.process_direct_features import process_direct_person_columns
    from .person_processor.process_properties_features import process_person_properties
    from .person_processor.calculate_event_metrics import calculate_person_event_metrics
except ImportError:
    print("Warning: Could not use relative imports for person_processor. Attempting direct import.")
    try:
        from person_processor.process_direct_features import process_direct_person_columns
        from person_processor.process_properties_features import process_person_properties
        from person_processor.calculate_event_metrics import calculate_person_event_metrics
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import person_processor modules.")
         sys.exit(1)

# --- Colunas para GNN (Definidas na análise anterior) ---
NUMERIC_COLS_TO_SCALE_GNN = [
    'is_identified', 'days_since_creation', 'days_since_last_update',
    'shopify_account_age_days', 'in_eu', 'unique_products_interacted',
    'view_count', 'purchase_count', 'wishlist_count', 'share_count',
    'checkout_start_count', 'distinct_channels_count', 'variant_meta_added_count',
    'checkout_update_count', 'order_update_count', 'days_since_last_interaction',
    'days_since_first_interaction', 'avg_purchase_value'
]

CATEGORICAL_COLS_TO_EMBED_GNN = [
    'shopify_currency', 'country_iso', 'timezone', 'most_frequent_channel'
]

ID_COL_GNN = 'person_id_hex'
# --- Fim Colunas GNN ---


def get_final_person_columns():
    """Returns the list of expected final columns for the intermediate person CSV."""
    return [
        'person_id_hex', 'is_identified', 'days_since_creation', 'days_since_last_update',
        'shopify_currency', 'shopify_account_age_days', 'country_iso', 'in_eu', 'timezone',
        'total_interactions', 'unique_products_interacted', 'view_count', 'purchase_count',
        'wishlist_count', 'share_count', 'checkout_start_count', 'distinct_channels_count',
        'variant_meta_added_count', 'checkout_update_count', 'order_update_count', 'is_buyer',
        'days_since_last_interaction', 'days_since_first_interaction', 'most_frequent_channel',
        'avg_purchase_value'
        # Removidas colunas que não parecem ser geradas consistentemente ou são muito específicas
        # 'shopify_state', 'shopify_verified_email', 'city', 'days_since_last_session'
    ]

# --- Função Auxiliar para Preparação GNN ---
def _prepare_person_gnn_features(df_final: pd.DataFrame, output_dir_gnn: str):
    """
    Internal function to prepare and save GNN-ready features from the final aggregated person df.
    """
    print(f"\n--- Preparing Person Node Features for GNN ---")
    print(f"Output directory for GNN data: {output_dir_gnn}")
    os.makedirs(output_dir_gnn, exist_ok=True)

    if df_final.empty:
        print("Input DataFrame for GNN prep is empty. Skipping GNN feature generation.")
        return

    if ID_COL_GNN not in df_final.columns:
         print(f"CRITICAL Error: ID column '{ID_COL_GNN}' not found in final person DataFrame. Cannot prepare GNN features.")
         return

    available_cols = df_final.columns.tolist()
    numeric_final_gnn = [col for col in NUMERIC_COLS_TO_SCALE_GNN if col in available_cols]
    categorical_final_gnn = [col for col in CATEGORICAL_COLS_TO_EMBED_GNN if col in available_cols]
    missing_numeric = [col for col in NUMERIC_COLS_TO_SCALE_GNN if col not in available_cols]
    missing_categorical = [col for col in CATEGORICAL_COLS_TO_EMBED_GNN if col not in available_cols]
    if missing_numeric: print(f"Warning: Missing expected numeric GNN columns: {missing_numeric}")
    if missing_categorical: print(f"Warning: Missing expected categorical GNN columns: {missing_categorical}")

    cols_to_keep_for_gnn = [ID_COL_GNN] + numeric_final_gnn + categorical_final_gnn
    df_gnn = df_final[cols_to_keep_for_gnn].copy()
    print(f"Selected {len(df_gnn.columns) - 1} features for GNN processing.")

    print("Creating ID to index mapping...")
    df_gnn = df_gnn.sort_values(by=ID_COL_GNN).reset_index(drop=True)
    person_id_to_index = {person_id: index for index, person_id in enumerate(df_gnn[ID_COL_GNN])}
    id_map_path = os.path.join(output_dir_gnn, 'person_id_to_index_map.json')
    with open(id_map_path, 'w') as f:
        json.dump(person_id_to_index, f)
    print(f"Saved ID mapping to {id_map_path}")

    df_features_gnn = df_gnn.drop(columns=[ID_COL_GNN])

    print("Processing numeric features for GNN...")
    scaled_numeric_features = np.zeros((len(df_features_gnn), 0))
    if numeric_final_gnn:
        df_numeric_gnn = df_features_gnn[numeric_final_gnn].copy()
        initial_nans = df_numeric_gnn.isnull().sum().sum()
        if initial_nans > 0:
            print(f"Imputing {initial_nans} NaN values in numeric GNN features with 0...")
            df_numeric_gnn.fillna(0, inplace=True)
        scaler = RobustScaler()
        print("Applying RobustScaler...")
        scaled_numeric_features = scaler.fit_transform(df_numeric_gnn)
        scaler_path = os.path.join(output_dir_gnn, 'person_numeric_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Saved numeric scaler to {scaler_path}")
    else:
        print("No numeric features selected for GNN.")

    numeric_features_path = os.path.join(output_dir_gnn, 'person_features_numeric_scaled.npy')
    np.save(numeric_features_path, scaled_numeric_features)
    print(f"Saved scaled numeric features to {numeric_features_path} (Shape: {scaled_numeric_features.shape})")
    numeric_cols_order_path = os.path.join(output_dir_gnn, 'person_numeric_cols_order.json')
    with open(numeric_cols_order_path, 'w') as f: json.dump(numeric_final_gnn, f)
    print(f"Saved numeric column order to {numeric_cols_order_path}")

    print("Processing categorical features for GNN...")
    encoded_categorical_array = np.zeros((len(df_features_gnn), 0))
    vocabularies = {}
    categorical_cols_order = []
    if categorical_final_gnn:
        df_categorical_gnn = df_features_gnn[categorical_final_gnn].copy()
        encoded_categorical_data = {}
        for col in categorical_final_gnn:
            print(f"  Encoding column: {col}")
            df_categorical_gnn[col] = df_categorical_gnn[col].fillna('__UNKNOWN__').astype(str)
            unique_values = sorted([v for v in df_categorical_gnn[col].unique() if v != '__UNKNOWN__'])
            vocab = {val: i+1 for i, val in enumerate(unique_values)}
            vocab['__UNKNOWN__'] = 0
            vocabularies[col] = vocab
            print(f"    Vocabulary size for {col}: {len(vocab)}")
            encoded_categorical_data[col] = df_categorical_gnn[col].map(vocab).values
            categorical_cols_order.append(col)
        if encoded_categorical_data: # Check if dict is not empty
            encoded_categorical_array = np.stack(list(encoded_categorical_data.values()), axis=1)
    else:
        print("No categorical features selected for GNN.")

    vocabs_path = os.path.join(output_dir_gnn, 'person_categorical_vocabs.json')
    with open(vocabs_path, 'w') as f: json.dump(vocabularies, f, indent=4)
    print(f"Saved categorical vocabularies to {vocabs_path}")

    categorical_features_path = os.path.join(output_dir_gnn, 'person_features_categorical_indices.npy')
    np.save(categorical_features_path, encoded_categorical_array)
    print(f"Saved encoded categorical features (indices) to {categorical_features_path} (Shape: {encoded_categorical_array.shape})")
    categorical_cols_order_path = os.path.join(output_dir_gnn, 'person_categorical_cols_order.json')
    with open(categorical_cols_order_path, 'w') as f: json.dump(categorical_cols_order, f)
    print(f"Saved categorical column order to {categorical_cols_order_path}")

    print("--- Person Node Feature Preparation for GNN Finished ---")

# --- Função Principal Atualizada ---
def run_person_preprocessing(
    person_input_path: str,
    event_input_path: str,
    output_path: str,
    person_gnn_output_dir: str,
    # --- NOVO ARGUMENTO ---
    train_cutoff_timestamp: pd.Timestamp = None
):
    """
    Orchestrates the preprocessing of person data.
    Calculates event metrics using an optional train_cutoff_timestamp.
    """
    print(f"\n--- Starting Person Data Preprocessing Orchestration ---")
    print(f"Person Input file: {person_input_path}")
    print(f"Event Input file (for metrics): {event_input_path}")
    print(f"Intermediate Output CSV: {output_path}")
    print(f"GNN Features Output Dir: {person_gnn_output_dir}")
    if train_cutoff_timestamp:
        print(f"Using Train Cutoff Timestamp for event metrics: {train_cutoff_timestamp}")
    else:
        print("No Train Cutoff Timestamp provided for event metrics (will use all event data).")

    final_df = pd.DataFrame()

    try:
        if not os.path.exists(person_input_path):
             raise FileNotFoundError(f"Person input file not found: {person_input_path}")
        person_df_raw = pd.read_csv(person_input_path, low_memory=False)
        print(f"Loaded {len(person_df_raw)} raw person rows from {person_input_path}.")

        if person_df_raw.empty:
            print("Input person DataFrame is empty. Cannot proceed.")
            return
        if 'person_id_hex' not in person_df_raw.columns:
            raise ValueError("Input person data must contain 'person_id_hex' column.")
        # Garantir que 'is_identified' existe para process_person_properties
        if 'is_identified' not in person_df_raw.columns:
            print("Warning: 'is_identified' column missing in raw person data. Adding default False.")
            person_df_raw['is_identified'] = False
        if 'properties' not in person_df_raw.columns:
            print("Warning: 'properties' column missing in raw person data. Adding empty dict string.")
            person_df_raw['properties'] = "{}"


    except Exception as e:
        print(f"Error loading or validating input person CSV '{person_input_path}': {e}")
        traceback.print_exc()
        raise

    try:
        print("\nProcessing direct columns...")
        df_direct = process_direct_person_columns(person_df_raw.copy())

        print("\nProcessing properties columns...")
        df_props = process_person_properties(person_df_raw.copy())

    except Exception as e:
        print(f"Error during direct/properties feature processing step: {e}")
        traceback.print_exc()
        raise

    print("\n--- Merging Direct and Properties Features ---")
    try:
        persons_features_df = pd.merge(df_direct, df_props, on='person_id_hex', how='left')
        # Remover duplicatas se houver, mantendo a primeira ocorrência
        persons_features_df.drop_duplicates(subset=['person_id_hex'], keep='first', inplace=True)
        print(f"Shape after merging direct and properties features: {persons_features_df.shape}")
    except Exception as e:
        print(f"Error during merging direct/properties features: {e}")
        traceback.print_exc()
        raise

    print("\n--- Calculating and Merging Event Metrics ---")
    try:
        # --- USAR O CUTOFF AQUI ---
        person_event_metrics_df = calculate_person_event_metrics(event_input_path, cutoff_timestamp=train_cutoff_timestamp)
        if not person_event_metrics_df.empty:
            persons_features_df = persons_features_df.merge(
                person_event_metrics_df, left_on='person_id_hex', right_index=True, how='left'
            )
            # Preencher NaNs para métricas de evento (pessoas sem eventos no período de cutoff)
            metric_cols = person_event_metrics_df.columns
            for col in metric_cols:
                if col in persons_features_df.columns:
                    if pd.api.types.is_numeric_dtype(persons_features_df[col]):
                        persons_features_df[col] = persons_features_df[col].fillna(0)
                    else:
                        persons_features_df[col] = persons_features_df[col].fillna('Unknown')
            print(f"Shape after merging event metrics: {persons_features_df.shape}")
        else:
            print("No event metrics calculated/returned. Adding default metric columns.")
            default_metric_cols = [
                'purchase_count', 'total_interactions', 'unique_products_interacted',
                'view_count', 'wishlist_count', 'share_count', 'checkout_start_count',
                'distinct_channels_count', 'variant_meta_added_count', 'checkout_update_count',
                'order_update_count', 'is_buyer', 'days_since_last_interaction',
                'days_since_first_interaction', 'avg_purchase_value'
            ]
            for col in default_metric_cols: persons_features_df[col] = 0
            persons_features_df['most_frequent_channel'] = 'Unknown'
            if 'days_since_last_interaction' in persons_features_df: persons_features_df['days_since_last_interaction'] = 9999
            if 'days_since_first_interaction' in persons_features_df: persons_features_df['days_since_first_interaction'] = 9999


    except Exception as e:
        print(f"Error calculating or merging person event metrics: {e}")
        traceback.print_exc()
        # Adicionar colunas default em caso de erro também
        default_metric_cols = [
            'purchase_count', 'total_interactions', 'unique_products_interacted',
            'view_count', 'wishlist_count', 'share_count', 'checkout_start_count',
            'distinct_channels_count', 'variant_meta_added_count', 'checkout_update_count',
            'order_update_count', 'is_buyer', 'days_since_last_interaction',
            'days_since_first_interaction', 'avg_purchase_value'
        ]
        for col in default_metric_cols:
            if col not in persons_features_df.columns: persons_features_df[col] = 0
        if 'most_frequent_channel' not in persons_features_df.columns: persons_features_df['most_frequent_channel'] = 'Unknown'
        if 'days_since_last_interaction' in persons_features_df: persons_features_df['days_since_last_interaction'] = persons_features_df['days_since_last_interaction'].fillna(9999)
        if 'days_since_first_interaction' in persons_features_df: persons_features_df['days_since_first_interaction'] = persons_features_df['days_since_first_interaction'].fillna(9999)


    FINAL_COLUMNS_CSV = get_final_person_columns()
    print("\nFinalizing columns for intermediate CSV...")
    final_df = persons_features_df.copy()

    # Adicionar colunas faltantes com defaults antes de selecionar para o CSV
    for col in FINAL_COLUMNS_CSV:
        if col not in final_df.columns:
            print(f"Column '{col}' missing from processed DataFrame. Adding with default.")
            if pd.api.types.is_numeric_dtype(final_df.get(col, pd.Series(dtype=float))): # Check type of potential existing or default
                final_df[col] = 0
            else:
                final_df[col] = 'Unknown' # Default para strings
            if col in ['days_since_last_interaction', 'days_since_first_interaction']:
                final_df[col] = 9999


    try:
        final_df_csv = final_df[FINAL_COLUMNS_CSV].copy()
        print(f"Selected columns for intermediate CSV. Shape: {final_df_csv.shape}")
    except KeyError as e:
        missing_cols = [col for col in FINAL_COLUMNS_CSV if col not in final_df.columns]
        print(f"Error: Could not select all required intermediate CSV columns. Missing: {missing_cols}")
        print(f"Available columns: {final_df.columns.tolist()}")
        raise ValueError(f"Could not select all required intermediate CSV columns. Missing: {missing_cols}") from e

    try:
        output_dir_csv = os.path.dirname(output_path)
        os.makedirs(output_dir_csv, exist_ok=True)
        print(f"\nEnsured intermediate output directory exists: {output_dir_csv}")
        final_df_csv.to_csv(output_path, index=False, na_rep='')
        print(f"Successfully saved intermediate processed person data to: {output_path}")
    except Exception as e:
        print(f"Error saving intermediate output CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

    try:
        _prepare_person_gnn_features(final_df, person_gnn_output_dir) # Usa final_df que tem todas as colunas
    except Exception as e:
        print(f"Error during GNN feature preparation step: {e}")
        traceback.print_exc()
        print("Warning: GNN feature preparation failed, but intermediate CSV might be saved.")

    print(f"--- Finished Person Data Preprocessing Orchestration (including GNN prep) ---")

# --- Command Line Execution Logic (Atualizado) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess person data and prepare GNN node features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--person-input', required=True, help='Path to the raw person data CSV.')
    parser.add_argument('--event-input', required=True, help='Path to the processed event interactions CSV.')
    parser.add_argument('--output-csv', required=True, help='Path to save the intermediate processed person features CSV.')
    parser.add_argument('--output-gnn-dir', required=True, help='Directory to save GNN-ready features, scaler, vocabs, and ID map.')
    # --- NOVO ARGUMENTO PARA CUTOFF ---
    parser.add_argument('--cutoff-timestamp', type=str, default=None,
                        help='Optional UTC timestamp string (YYYY-MM-DD HH:MM:SS.ffffff[+/-HH:MM]) for event metrics cutoff.')

    args = parser.parse_args()

    cutoff_ts_obj = None
    if args.cutoff_timestamp:
        try:
            cutoff_ts_obj = pd.to_datetime(args.cutoff_timestamp, utc=True)
            print(f"Using provided cutoff timestamp: {cutoff_ts_obj}")
        except ValueError:
            print(f"Error: Invalid cutoff_timestamp format: {args.cutoff_timestamp}. Proceeding without cutoff.")

    try:
        run_person_preprocessing(
            args.person_input,
            args.event_input,
            args.output_csv,
            args.output_gnn_dir,
            train_cutoff_timestamp=cutoff_ts_obj # Passar o objeto Timestamp
        )
        print("\nPerson preprocessing & GNN prep finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error:
         print(f"\nError: {val_error}")
         exit(1)
    except Exception as e:
        print(f"\n--- Person preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)