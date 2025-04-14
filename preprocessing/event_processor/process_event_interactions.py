# preprocessing/event_processor/process_event_interactions.py
import pandas as pd
import argparse
import os
import traceback
from tqdm import tqdm

try:
    # Try relative import if part of a package
    from ..utils import safe_json_parse
except (ImportError, ValueError): # Catch ValueError too for potential relative import issues outside package
    try:
        # Try direct import if running as script or package structure is different
        from utils import safe_json_parse
    except ImportError:
        print("Error: Could not import safe_json_parse. Ensure utils.py is accessible.")
        print("If running as part of the 'preprocessing' package, ensure the package structure is correct.")
        print("If running as a standalone script, ensure utils.py is in the same directory or Python path.")
        exit(1) # Exit if the essential utility cannot be imported

# --- Define Output Columns at Module Level ---
OUTPUT_COLUMNS = [
    'person_id_hex', 'product_id', 'interaction_type', 'timestamp',
    'event_converted_price', 'event_converted_currency', 'event_quantity',
    'order_id', 'checkout_id', 'channel', 'session_id', 'event_id'
]
# --- End Define Output Columns ---


def extract_interactions_from_events(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o DataFrame de eventos e extrai uma lista estruturada de interações...
    """
    if not isinstance(event_df, pd.DataFrame):
        raise ValueError("Input 'event_df' must be a pandas DataFrame.")

    if event_df.empty:
        print("Input event DataFrame is empty. Returning empty interactions DataFrame.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"--- Iniciando Extração de Interações de {len(event_df)} eventos ---")
    interactions_list = []

    required_cols = ['person_id_hex', 'event', 'timestamp', 'event_id', 'properties', 'context']
    missing_cols = [col for col in required_cols if col not in event_df.columns]
    if missing_cols:
        raise ValueError(f"Input event DataFrame missing required columns: {missing_cols}")

    event_df_subset = event_df[required_cols].copy()

    # --- Loop to build interactions_list (Condensed for clarity - unchanged) ---
    for index, row in tqdm(event_df_subset.iterrows(), total=event_df_subset.shape[0], desc="Processando Eventos"):
        try:
            person_id = row.get('person_id_hex')
            event_type = row.get('event')
            timestamp = row.get('timestamp') # Get timestamp from row
            event_id = row.get('event_id')

            # Skip rows with clearly missing essential data early
            if not person_id or not event_type or pd.isna(timestamp) or timestamp == '':
                 continue

            properties_str = row.get('properties', '{}')
            context_str = row.get('context', '{}')
            properties = safe_json_parse(properties_str)
            context = safe_json_parse(context_str)
            if properties is None: properties = {}
            if context is None: context = {}
            channel = context.get('Channel')
            session_id = context.get('SessionID')

            interaction_data_base = {
                'person_id_hex': person_id, 'timestamp': timestamp, 'channel': channel,
                'session_id': session_id, 'event_id': event_id, 'product_id': None,
                'interaction_type': None, 'event_converted_price': None,
                'event_converted_currency': None, 'event_quantity': None,
                'order_id': None, 'checkout_id': None,
            }

            # Event Specific Logic (unchanged from previous correct versions)
            if event_type in ["Product Viewed", "Product Wishlisted", "Product Shared", "Variant Meta Added"]:
                product_id = properties.get('product_id')
                if product_id:
                    interaction_data = interaction_data_base.copy()
                    interaction_data['product_id'] = str(product_id)
                    interaction_data['interaction_type'] = event_type.lower().replace(" ", "_")
                    interaction_data['event_converted_price'] = properties.get('converted_price')
                    interaction_data['event_converted_currency'] = properties.get('converted_currency')
                    interaction_data['event_quantity'] = properties.get('quantity', 1 if event_type == "Variant Meta Added" else None)
                    interactions_list.append(interaction_data)

            elif event_type in ["Shopify Order Created", "Shopify Order Updated", "Shopify Checkout Created", "Shopify Checkout Updated"]:
                line_items = properties.get('line_items')
                order_id = properties.get('order_id')
                checkout_id = properties.get('checkout_id')
                if isinstance(line_items, list) and line_items:
                    base_interaction_type = 'ordered_item' if 'Order' in event_type else 'checkout_item'
                    for item_index, item in enumerate(line_items):
                         try:
                            if isinstance(item, dict):
                                product_id = item.get('product_id')
                                if product_id:
                                    item_interaction = interaction_data_base.copy()
                                    item_interaction['product_id'] = str(product_id)
                                    item_interaction['interaction_type'] = base_interaction_type
                                    item_interaction['event_converted_price'] = item.get('converted_price')
                                    item_interaction['event_converted_currency'] = item.get('converted_currency')
                                    item_interaction['event_quantity'] = item.get('quantity')
                                    item_interaction['order_id'] = order_id
                                    item_interaction['checkout_id'] = checkout_id
                                    interactions_list.append(item_interaction)
                         except Exception as item_e:
                             print(f"\nError processing item {item_index}: {item_e}")
                             continue
        except Exception as row_e:
             print(f"\nGeneral error processing row {index}: {row_e}")
             continue
    # --- End Loop ---

    print(f"\n--- Extraction Loop Completed: {len(interactions_list)} potential interactions found ---")
    if not interactions_list: return pd.DataFrame(columns=OUTPUT_COLUMNS)

    interactions_df = pd.DataFrame(interactions_list)
    print(f"DataFrame created from list with shape: {interactions_df.shape}")

    # --- Final Cleanup and Type Conversion ---
    print("\n--- Performing final cleanup and type conversions (including pd.to_datetime) ---")

    # Keep a copy of the original timestamps for the second pass
    if 'timestamp' in interactions_df.columns:
        original_timestamps = interactions_df['timestamp'].astype(str).str.strip()
    else:
        print("Warning: Timestamp column missing before conversion process.")
        original_timestamps = pd.Series(dtype=str) # Empty series

    # --- Two-Pass Timestamp Conversion ---
    print("\n--- Starting Two-Pass Timestamp Conversion ---")

    # Pass 1: Try format with microseconds
    print("Pass 1: Converting timestamp using format '%Y-%m-%d %H:%M:%S.%f%z'...")
    format_pass1 = '%Y-%m-%d %H:%M:%S.%f%z'
    if not original_timestamps.empty:
        interactions_df['timestamp'] = pd.to_datetime(
            original_timestamps, # Use the cleaned original strings
            format=format_pass1,
            errors='coerce',
            utc=True
        )
        pass1_failures = interactions_df['timestamp'].isnull()
        print(f"Pass 1 completed. Failures (became NaT): {pass1_failures.sum()}")
    else:
        pass1_failures = pd.Series(dtype=bool) # Empty series
        print("Pass 1 skipped (no timestamp data).")


    # Pass 2: Retry failures with format without microseconds
    if pass1_failures.any():
        print(f"Pass 2: Retrying {pass1_failures.sum()} failures using format '%Y-%m-%d %H:%M:%S%z'...")
        format_pass2 = '%Y-%m-%d %H:%M:%S%z'
        # Apply only to rows that failed in pass 1, using original timestamp strings
        interactions_df.loc[pass1_failures, 'timestamp'] = pd.to_datetime(
            original_timestamps[pass1_failures], # Use original strings for failed rows
            format=format_pass2,
            errors='coerce',
            utc=True
        )
        pass2_failures = interactions_df['timestamp'].isnull() & pass1_failures # Check only those that failed Pass 1
        print(f"Pass 2 completed. Remaining failures (still NaT): {pass2_failures.sum()}")
        if pass2_failures.sum() > 0:
            print("Warning: Some timestamps failed both formats.")
            print("Sample failed timestamps (original value):")
            print(original_timestamps[pass2_failures].head())
    else:
        print("Pass 2 skipped (no failures in Pass 1).")

    print("--- Two-Pass Timestamp Conversion Finished ---")


    # --- Other Type Conversions ---
    print("\nPerforming other type conversions...")
    # Convert numeric fields, coercing errors to NaN
    interactions_df['event_converted_price'] = pd.to_numeric(interactions_df['event_converted_price'], errors='coerce')
    interactions_df['event_quantity'] = pd.to_numeric(interactions_df['event_quantity'], errors='coerce')
    # Ensure IDs are strings.
    interactions_df['product_id'] = interactions_df['product_id'].astype(str)
    interactions_df['person_id_hex'] = interactions_df['person_id_hex'].astype(str)
    # Convert order/checkout IDs
    interactions_df['order_id'] = interactions_df['order_id'].apply(lambda x: str(int(x)) if pd.notna(x) and x != '' else None)
    interactions_df['checkout_id'] = interactions_df['checkout_id'].apply(lambda x: str(int(x)) if pd.notna(x) and x != '' else None)
    print("Type conversions completed.")


    # --- START DEBUGGING BLOCK (AFTER CONVERSION) ---
    # Keep this block to verify the final state before dropna
    print("\n--- Data BEFORE dropna (AFTER TWO-PASS pd.to_datetime) ---")
    print(f"Shape after conversion: {interactions_df.shape}")
    print("Value Counts (interaction_type) after conversion:")
    print(interactions_df['interaction_type'].value_counts(dropna=False))

    # Check for NaT timestamps specifically for ALL types now
    null_timestamps_after_conv = interactions_df['timestamp'].isnull()
    print(f"\nTotal Null timestamps AFTER conversion: {null_timestamps_after_conv.sum()}")
    if null_timestamps_after_conv.sum() > 0:
         print("Interaction types with Null timestamps AFTER conversion:")
         print(interactions_df.loc[null_timestamps_after_conv, 'interaction_type'].value_counts())

    rows_to_drop_mask = interactions_df['timestamp'].isnull() | interactions_df['product_id'].isin([None, 'None', 'nan'])
    rows_to_drop = interactions_df[rows_to_drop_mask]

    print(f"\nRows that will be dropped by dropna/filter: {len(rows_to_drop)}")
    if not rows_to_drop.empty:
        print("Value counts of interaction_type for rows TO BE DROPPED:")
        print(rows_to_drop['interaction_type'].value_counts())
        print("Reason for dropping (count):")
        print(f" - Null/NaT Timestamp: {rows_to_drop['timestamp'].isnull().sum()}")
        print(f" - Null/None/nan string Product ID: {rows_to_drop['product_id'].isin([None, 'None', 'nan']).sum()}")
    else:
        print("No rows identified for dropping based on timestamp or product_id.")

    print("\n--- Applying dropna(['product_id', 'timestamp']) and filtering 'None'/'nan' product_id ---")
    # --- END DEBUGGING BLOCK ---


    # Drop rows where essential identifiers are null
    interactions_df.dropna(subset=['product_id', 'timestamp'], inplace=True)
    interactions_df = interactions_df[~interactions_df['product_id'].isin(['None', 'nan'])] # Remove 'None'/'nan' strings if they exist


    print("\n--- Data AFTER dropna/filter ---")
    if interactions_df.empty:
        print("DataFrame is empty after dropping rows with null product_id or timestamp.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"DataFrame de Interações final tem {len(interactions_df)} linhas.")
    print("Tipos de interação encontrados (final):")
    print(interactions_df['interaction_type'].value_counts())

    try:
        interactions_df = interactions_df[OUTPUT_COLUMNS]
    except KeyError as e:
        print(f"Warning: Missing expected columns after processing: {e}. Returning available columns.")
        existing_cols = [col for col in OUTPUT_COLUMNS if col in interactions_df.columns]
        interactions_df = interactions_df[existing_cols]

    return interactions_df


# --- Command Line Execution Logic (unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processa eventos para extrair interações Person-Product.')
    parser.add_argument('--input', required=True, help='Caminho para o ficheiro de eventos filtrados (person_event_data_filtered.csv).')
    parser.add_argument('--output', required=True, help='Caminho para guardar o ficheiro de interações processadas (event_interactions_processed.csv).')
    args = parser.parse_args()

    print(f"A ler eventos de: {args.input}")
    try:
        # Load with low_memory=False if properties/context are large JSONs
        event_df_loaded = pd.read_csv(args.input, low_memory=False)
        print(f"Lidos {len(event_df_loaded)} eventos.")

        processed_interactions_df = extract_interactions_from_events(event_df_loaded)

        if not processed_interactions_df.empty:
            print(f"\nVerificando o DataFrame final antes de salvar ({len(processed_interactions_df)} linhas):")
            print(processed_interactions_df.info())
            print(processed_interactions_df.head())

            output_dir = os.path.dirname(args.output)
            if output_dir: # Ensure directory exists only if output path includes one
                os.makedirs(output_dir, exist_ok=True)
                print(f"Diretório de saída '{output_dir}' assegurado.")

            processed_interactions_df.to_csv(args.output, index=False)
            print(f"\nInterações processadas guardadas com sucesso em: {args.output}")
        else:
            # This block is executed if processed_interactions_df is empty AFTER processing
            print("\nNenhuma interação processada válida para guardar após a limpeza.")
            output_dir = os.path.dirname(args.output)
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
            # Use the module-level constant OUTPUT_COLUMNS here
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.output, index=False)
            print(f"Ficheiro de saída vazio criado em: {args.output}")


        print("\nProcessamento de interações concluído.")

    except FileNotFoundError:
        print(f"Erro: Ficheiro de input não encontrado em {args.input}")
        exit(1)
    except ValueError as ve:
        print(f"Erro de Valor durante o processamento: {ve}")
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n--- Ocorreu um erro inesperado durante o processamento: {e} ---")
        traceback.print_exc()
        exit(1)