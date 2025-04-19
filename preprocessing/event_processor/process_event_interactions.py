# preprocessing/event_processor/process_event_interactions.py
import pandas as pd
import argparse
import os
import traceback
from tqdm import tqdm
import json

try:
    from ..utils import safe_json_parse
except (ImportError, ValueError):
    try:
        from utils import safe_json_parse
    except ImportError:
        print("Error: Could not import safe_json_parse...")
        exit(1)

# FINAL Output Columns - Replaced checkout_sequence with person_interaction_sequence
FINAL_OUTPUT_COLUMNS = [
    'person_id_hex', 'product_id', 'interaction_type', 'timestamp',
    'event_converted_price', 'checkout_id', 'channel', 'person_interaction_sequence' # New sequence column
]

# Columns needed DURING processing (checkout_sequence removed implicitly)
PROCESSING_COLUMNS = [
    'person_id_hex', 'product_id', 'interaction_type', 'timestamp',
    'event_converted_price', 'event_converted_currency', 'event_quantity',
    'order_id', 'checkout_id', 'channel', 'session_id', 'event_id'
]

def get_refined_channel(event_type, properties, context):
    # (Function unchanged)
    webhook_event_types = ["Shopify Order Created", "Shopify Order Updated", "Shopify Checkout Created", "Shopify Checkout Updated"]
    if event_type in webhook_event_types:
        platform = properties.get('platform')
        if platform == '': return 'website'
        elif platform == 'ios': return 'ios'
        elif platform == 'android': return 'android'
        else: return 'unknown'
    else:
        os_info = context.get('Os')
        if isinstance(os_info, dict):
            os_name = os_info.get('Name', '').lower()
            if 'ios' in os_name: return 'ios'
            elif 'android' in os_name: return 'android'
            else: return 'unknown'
        else: return 'unknown'


def extract_interactions_from_events(event_df: pd.DataFrame) -> pd.DataFrame:
    # (Docstring unchanged)
    if not isinstance(event_df, pd.DataFrame):
        raise ValueError("Input 'event_df' must be a pandas DataFrame.")
    if event_df.empty:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    print(f"--- Iniciando Extração de Interações de {len(event_df)} eventos ---")
    interactions_list = []
    required_cols = ['person_id_hex', 'event', 'timestamp', 'event_id', 'properties', 'context']
    missing_cols = [col for col in required_cols if col not in event_df.columns]
    if missing_cols: raise ValueError(f"Input missing required columns: {missing_cols}")
    event_df_subset = event_df[required_cols].copy()

    # --- Loop to build interactions_list (Condensed - same logic) ---
    for index, row in tqdm(event_df_subset.iterrows(), total=event_df_subset.shape[0], desc="Processando Eventos"):
        try:
            person_id = row.get('person_id_hex')
            event_type = row.get('event')
            timestamp = row.get('timestamp')
            event_id = row.get('event_id')

            # Ensure essential fields are present
            if not person_id or pd.isna(person_id) or not event_type or pd.isna(timestamp) or timestamp == '':
                # print(f"Skipping row {index} due to missing essential data: person_id={person_id}, event_type={event_type}, timestamp={timestamp}")
                continue

            properties_str = row.get('properties', '{}')
            context_str = row.get('context', '{}')
            properties = safe_json_parse(properties_str)
            context = safe_json_parse(context_str)
            if properties is None: properties = {}
            if context is None: context = {}

            final_channel = get_refined_channel(event_type, properties, context)
            session_id = context.get('SessionID')

            interaction_data_base = {
                'person_id_hex': person_id, 'timestamp': timestamp, 'channel': final_channel,
                'session_id': session_id, 'event_id': event_id, 'product_id': None,
                'interaction_type': None, 'event_converted_price': None,
                'event_converted_currency': None, 'event_quantity': None,
                'order_id': None, 'checkout_id': None,
            }

            webhook_event_types = ["Shopify Order Created", "Shopify Order Updated", "Shopify Checkout Created", "Shopify Checkout Updated"]
            view_wishlist_types = ["Product Viewed", "Product Wishlisted", "Product Shared", "Variant Meta Added"]

            if event_type in view_wishlist_types:
                product_id = properties.get('product_id')
                if product_id: # Only add interaction if product_id exists for these types
                    interaction_data = interaction_data_base.copy()
                    interaction_data['product_id'] = str(product_id)
                    interaction_data['interaction_type'] = event_type
                    interaction_data['event_converted_price'] = properties.get('converted_price')
                    interaction_data['event_converted_currency'] = properties.get('converted_currency')
                    interaction_data['event_quantity'] = properties.get('quantity', 1 if event_type == "Variant Meta Added" else None)
                    interactions_list.append(interaction_data)
                # else: # Optional: Log if view/wishlist event lacks product_id
                #     print(f"Event {event_type} (ID: {event_id}) for person {person_id} lacks product_id. Skipping.")


            elif event_type in webhook_event_types:
                line_items = properties.get('line_items')
                order_id = properties.get('order_id')
                checkout_id = properties.get('checkout_id')
                if isinstance(line_items, list) and line_items:
                    items_added = 0
                    for item_index, item in enumerate(line_items):
                         try:
                            if isinstance(item, dict):
                                product_id = item.get('product_id')
                                if product_id: # Only add interaction if product_id exists in line item
                                    item_interaction = interaction_data_base.copy()
                                    item_interaction['product_id'] = str(product_id)
                                    item_interaction['interaction_type'] = event_type
                                    item_interaction['event_converted_price'] = item.get('converted_price')
                                    item_interaction['event_converted_currency'] = item.get('converted_currency')
                                    item_interaction['event_quantity'] = item.get('quantity')
                                    item_interaction['order_id'] = order_id
                                    item_interaction['checkout_id'] = checkout_id
                                    interactions_list.append(item_interaction)
                                    items_added += 1
                                # else: # Optional: Log if line item lacks product_id
                                #     print(f"Item {item_index} in event {event_type} (ID: {event_id}) for person {person_id} lacks product_id. Skipping item.")
                         except Exception as item_e:
                             print(f"\nError processing item {item_index} for event {event_id}: {item_e}")
                             continue
                    # if items_added == 0: # Optional: Log if webhook event had line_items but none had product_id
                    #     print(f"Webhook event {event_type} (ID: {event_id}) for person {person_id} had line_items but no valid product_ids found. Skipping event.")
                # else: # Optional: Log if webhook event lacks line_items
                #      print(f"Webhook event {event_type} (ID: {event_id}) for person {person_id} lacks line_items list. Skipping.")

            # else: # Optional: Log other event types being skipped
                # print(f"Skipping event type '{event_type}' (ID: {event_id}) for person {person_id}.")

        except Exception as row_e:
             print(f"\nGeneral error processing row {index} (EventID: {row.get('event_id', 'N/A')}): {row_e}")
             traceback.print_exc() # Print traceback for debugging
             continue
    # --- End Loop ---


    print(f"\n--- Extraction Loop Completed: {len(interactions_list)} potential interactions found ---")
    if not interactions_list: return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    interactions_df = pd.DataFrame(interactions_list)
    print(f"DataFrame created from list with shape: {interactions_df.shape}")

    # --- Final Cleanup and Type Conversion ---
    print("\n--- Performing cleanup and type conversions ---")
    # Two-Pass Timestamp Conversion
    if 'timestamp' in interactions_df.columns:
        original_timestamps = interactions_df['timestamp'].astype(str).str.strip()
        print("\n--- Starting Two-Pass Timestamp Conversion ---")
        format_pass1='%Y-%m-%d %H:%M:%S.%f%z'; format_pass2='%Y-%m-%d %H:%M:%S%z'
        pass1_failures=pd.Series(False, index=interactions_df.index)
        if not original_timestamps.empty:
            interactions_df['timestamp']=pd.to_datetime(original_timestamps, format=format_pass1, errors='coerce', utc=True)
            pass1_failures=interactions_df['timestamp'].isnull()
        if pass1_failures.any():
            interactions_df.loc[pass1_failures, 'timestamp']=pd.to_datetime(original_timestamps[pass1_failures], format=format_pass2, errors='coerce', utc=True)
            pass2_failures = interactions_df['timestamp'].isnull() & pass1_failures
            if pass2_failures.sum() > 0: print(f"Warning: {pass2_failures.sum()} timestamps failed both formats.")
        print("--- Two-Pass Timestamp Conversion Finished ---")
    else: print("Warning: Timestamp column missing before conversion process.")

    # Other Type Conversions
    print("\nPerforming other type conversions...")
    num_cols_to_convert = ['event_converted_price', 'event_quantity']
    numeric_id_cols = ['order_id', 'checkout_id']
    string_id_cols = ['person_id_hex', 'product_id', 'session_id', 'event_id']
    for col in num_cols_to_convert:
        if col in interactions_df.columns: interactions_df[col] = pd.to_numeric(interactions_df[col], errors='coerce')
    for col in numeric_id_cols:
         if col in interactions_df.columns:
             numeric_series = pd.to_numeric(interactions_df[col], errors='coerce')
             # Use Int64 for nullable integers
             interactions_df[col] = numeric_series.astype('float64').astype('Int64')
    for col in string_id_cols:
         if col in interactions_df.columns:
             # Use pandas StringDtype for nullable strings
             interactions_df[col] = interactions_df[col].fillna(pd.NA).astype('string')
    if 'channel' in interactions_df.columns: interactions_df['channel'] = interactions_df['channel'].fillna('unknown').astype(str)
    if 'interaction_type' in interactions_df.columns: interactions_df['interaction_type'] = interactions_df['interaction_type'].astype(str)
    if 'event_converted_currency' in interactions_df.columns: interactions_df['event_converted_currency'] = interactions_df['event_converted_currency'].astype(str)
    print("Type conversions completed.")

    # --- Drop rows with null essential identifiers AFTER type conversion ---
    # Essential for sequencing: person_id_hex, timestamp
    # Essential for interaction meaning: product_id (for most interactions)
    print("\n--- Applying dropna(['person_id_hex', 'timestamp', 'product_id']) ---")
    initial_rows_before_dropna = len(interactions_df)
    interactions_df.dropna(subset=['person_id_hex', 'timestamp', 'product_id'], inplace=True)
    # Explicitly check for NA/None again after potential type conversions
    interactions_df = interactions_df[~(interactions_df['person_id_hex'].isna())]
    interactions_df = interactions_df[~(interactions_df['timestamp'].isna())]
    interactions_df = interactions_df[~(interactions_df['product_id'].isna())]
    rows_dropped_initial = initial_rows_before_dropna - len(interactions_df)
    print(f"Dropped {rows_dropped_initial} rows due to null person_id_hex, timestamp, or product_id.")
    if interactions_df.empty:
        print("DataFrame is empty after essential dropna.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    # --- ADD PERSON INTERACTION SEQUENCE ---
    print("\n--- Adding Person Interaction Sequence Number ---")
    # Sort by person and then timestamp to establish the correct order for sequencing
    print("Sorting interactions by person_id_hex and timestamp...")
    interactions_df.sort_values(by=['person_id_hex', 'timestamp'],
                                ascending=[True, True],
                                inplace=True,
                                na_position='last') # NAs should have been dropped, but safe practice

    # Calculate the sequence number within each person group
    print("Calculating sequence numbers...")
    # Use sort=False in groupby as we've already sorted the DataFrame
    interactions_df['person_interaction_sequence'] = interactions_df.groupby('person_id_hex', sort=False).cumcount() + 1

    # Convert the sequence number to nullable integer
    interactions_df['person_interaction_sequence'] = interactions_df['person_interaction_sequence'].astype('Int64')

    print("Person interaction sequence added.")
    print("Sample sequence numbers:")
    print(interactions_df[['person_id_hex', 'timestamp', 'interaction_type', 'person_interaction_sequence']].head(15)) # Show more rows
    # --- End Add Person Interaction Sequence ---

    # --- Select ONLY the desired final columns ---
    print(f"\nSelecting final columns: {FINAL_OUTPUT_COLUMNS}")
    try:
        # Ensure dtypes before final selection (should be correct, but double-check)
        if 'checkout_id' in interactions_df.columns:
             interactions_df['checkout_id'] = interactions_df['checkout_id'].astype('Int64')
        if 'person_interaction_sequence' in interactions_df.columns:
             interactions_df['person_interaction_sequence'] = interactions_df['person_interaction_sequence'].astype('Int64')

        final_interactions_df = interactions_df[FINAL_OUTPUT_COLUMNS].copy()
    except KeyError as e:
        print(f"Error selecting final columns: {e}.")
        available_cols = interactions_df.columns.tolist()
        missing_final = [col for col in FINAL_OUTPUT_COLUMNS if col not in available_cols]
        print(f"  Desired final columns missing from DataFrame: {missing_final}")
        print(f"  Available columns in DataFrame: {available_cols}")
        print("  Attempting to return available columns that ARE in FINAL_OUTPUT_COLUMNS...")
        fallback_cols = [col for col in FINAL_OUTPUT_COLUMNS if col in available_cols]
        if not fallback_cols:
             print("Error: No desired columns remain. Returning empty DataFrame.")
             return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)
        final_interactions_df = interactions_df[fallback_cols].copy()

    print(f"Final DataFrame shape after column selection: {final_interactions_df.shape}")
    print("Final columns selected:")
    print(final_interactions_df.columns.tolist())

    # --- FINAL CHECK ---
    print("\nData types in final DataFrame before save:")
    print(final_interactions_df.info())

    # Final sort is optional but can be nice for reviewing the output file
    # final_interactions_df.sort_values(by=['person_id_hex', 'person_interaction_sequence'], inplace=True)
    # print("\nFinal DataFrame sorted by person_id_hex, person_interaction_sequence.")

    return final_interactions_df


# --- Command Line Execution Logic (Updated print sample) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processa eventos para extrair interações Person-Product.')
    parser.add_argument('--input', required=True, help='Caminho para o ficheiro de eventos filtrados (person_event_data_filtered.csv).')
    parser.add_argument('--output', required=True, help='Caminho para guardar o ficheiro de interações processadas (event_interactions_processed.csv).')
    args = parser.parse_args()

    print(f"A ler eventos de: {args.input}")
    try:
        event_df_loaded = pd.read_csv(args.input, low_memory=False)
        print(f"Lidos {len(event_df_loaded)} eventos.")
        processed_interactions_df = extract_interactions_from_events(event_df_loaded)
        if not processed_interactions_df.empty:
            print(f"\nVerificando o DataFrame final antes de salvar ({len(processed_interactions_df)} linhas):")
            # Display a sample sorted by person and sequence to verify
            print(processed_interactions_df.sort_values(by=['person_id_hex', 'person_interaction_sequence']).head(25))
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Diretório de saída '{output_dir}' assegurado.")
            # Save CSV - Use na_rep='' to ensure NAs are saved as empty strings in CSV
            processed_interactions_df.to_csv(args.output, index=False, na_rep='')
            print(f"\nInterações processadas guardadas com sucesso em: {args.output}")
        else:
            print("\nNenhuma interação processada válida para guardar após a limpeza.")
            output_dir = os.path.dirname(args.output)
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)
            # Create empty file with correct columns if no interactions found
            pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS).to_csv(args.output, index=False)
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