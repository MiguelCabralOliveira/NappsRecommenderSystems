import pandas as pd
import binascii
import traceback

def convert_binary_id(binary_id):
    """Converts a binary ID (bytes or memoryview) to a hex string representation."""
    # Adiciona um print para ver inputs problemáticos (descomenta se necessário)
    # if not isinstance(binary_id, (bytes, memoryview)) and not pd.isna(binary_id):
    #     print(f"DEBUG: Unexpected type for convert_binary_id: {type(binary_id)}, value: {binary_id}")
    try:
        if isinstance(binary_id, bytes):
            return binascii.hexlify(binary_id).decode('utf-8')
        elif isinstance(binary_id, memoryview):
            return binascii.hexlify(binary_id.tobytes()).decode('utf-8')
        elif pd.isna(binary_id):
            return None
        else:
            return str(binary_id)
    except Exception as e:
        # Loga qualquer erro durante a conversão individual
        print(f"ERROR in convert_binary_id for input type {type(binary_id)}: {e}")
        return None # Retorna None em caso de erro na conversão

def process_loaded_data(person_df: pd.DataFrame, person_event_df: pd.DataFrame):
    print("\n--- Starting Initial Data Processing ---")
    processed_person_df = None
    processed_event_df = None

    # Process Person DataFrame
    if person_df is not None and not person_df.empty:
        processed_person_df = person_df.copy()
        if 'id' in processed_person_df.columns:
            print("Processing 'id' column in person_df...")
            try:
                print("  Applying convert_binary_id to 'id'...")
                processed_person_df['person_id_hex'] = processed_person_df['id'].apply(convert_binary_id)
                print("  Conversion applied.")

                print(f"  Shape before dropna (person_df): {processed_person_df.shape}")
                print(f"  Nulls in person_id_hex before dropna: {processed_person_df['person_id_hex'].isnull().sum()}")
                processed_person_df = processed_person_df.dropna(subset=['person_id_hex'])
                print(f"  Shape after dropna (person_df): {processed_person_df.shape}")

                if not processed_person_df.empty:
                    print("  Attempting to drop original 'id' column...")
                    processed_person_df = processed_person_df.drop(columns=['id'])
                    print("  Original 'id' column dropped.")
                else:
                    print("  Skipping drop 'id' because DataFrame is empty after dropna.")

            except Exception as e:
                print(f"  Error processing 'id' column in person_df: {e}")
                traceback.print_exc()
        else:
            print("  Warning: 'id' column not found in person_df.")
    else:
        print("Person DataFrame is None or empty.")

    # Process Person Event DataFrame
    if person_event_df is not None and not person_event_df.empty:
        processed_event_df = person_event_df.copy()
        if 'person_id' in processed_event_df.columns:
            print("Processing 'person_id' column in person_event_df...")
            try:
                print("  Applying convert_binary_id to 'person_id'...")
                processed_event_df['person_id_hex'] = processed_event_df['person_id'].apply(convert_binary_id)
                print("  Conversion applied.")

                print(f"  Shape before dropna (event_df): {processed_event_df.shape}")
                print(f"  Nulls in person_id_hex before dropna: {processed_event_df['person_id_hex'].isnull().sum()}")
                processed_event_df = processed_event_df.dropna(subset=['person_id_hex'])
                print(f"  Shape after dropna (event_df): {processed_event_df.shape}")

                if not processed_event_df.empty:
                    print("  Attempting to drop original 'person_id' column...")
                    processed_event_df = processed_event_df.drop(columns=['person_id'])
                    print("  Original 'person_id' column dropped.")
                else:
                    print("  Skipping drop 'person_id' because DataFrame is empty after dropna.")

                # Verifica se colunas essenciais ainda existem
                if 'event' not in processed_event_df.columns:
                     print("  CRITICAL WARNING: 'event' column missing after processing!")
                if 'properties' not in processed_event_df.columns:
                     print("  CRITICAL WARNING: 'properties' column missing after processing!")


            except Exception as e:
                print(f"  Error processing 'person_id' column in person_event_df: {e}")
                traceback.print_exc()
                return processed_person_df, None
        else:
            print("  Warning: 'person_id' column not found in person_event_df.")
            return processed_person_df, None
    else:
        print("Person Event DataFrame is None or empty.")

    print("--- Initial Data Processing Finished ---")
    # Adiciona um print final para verificar os DFs antes de retornar
    print(f"Final processed_person_df shape: {processed_person_df.shape if processed_person_df is not None else 'None'}")
    print(f"Final processed_event_df shape: {processed_event_df.shape if processed_event_df is not None else 'None'}")
    return processed_person_df, processed_event_df