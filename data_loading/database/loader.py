import pandas as pd
import traceback
from sqlalchemy import Engine
import os


from ..utils import save_dataframe 
from .client import get_db_engine 
from .queries import fetch_person_data, fetch_distinct_event_types, fetch_filtered_event_data 


def load_all_db_data(shop_id: str, days_limit: int, db_config: dict, output_base_path: str) -> bool:
    """
    Loads person data, filtered event data, and saves them to local/S3 path.

    Args:
        shop_id: The shop identifier.
        days_limit: How many days of event data to fetch.
        db_config: Dictionary with DB connection details.
        output_base_path: The base directory (local) or S3 URI prefix (s3://bucket/prefix).
                          Filenames will be appended to this path.

    Returns:
        True if data fetching and saving seemed okay (adjust logic as needed), False otherwise.
    """
    engine: Engine | None = None
    overall_success = True # Assume success unless an error occurs
    distinct_event_types = []
    person_df = pd.DataFrame()
    person_event_df = pd.DataFrame()

    try:
        print("Attempting to establish database connection...")
        engine = get_db_engine(db_config)
        print("Database engine created.")

        # Fetch data
        print(f"Fetching person data for shop_id: {shop_id}...")
        person_df = fetch_person_data(engine, shop_id)

        print(f"Fetching distinct event types for shop_id: {shop_id} (last {days_limit} days)...")
        distinct_event_types = fetch_distinct_event_types(engine, shop_id, days_limit)

        print(f"Fetching filtered event data for shop_id: {shop_id} (last {days_limit} days)...")
        person_event_df = fetch_filtered_event_data(engine, shop_id, days_limit)

        # --- Validation Checks (Warnings) ---
        if person_event_df.empty:
             if not distinct_event_types:
                  print("\nWarning: No distinct event types found in the specified period.")
             else:
                  print("\nWarning: No events matching the desired types were found.")
        if person_df.empty:
             print("\nWarning: No person data was loaded.")
        # -----------------------------------------

        # --- Save Data ---
        person_filename = f"{shop_id}_person_data_loaded.csv"
        event_filename = f"{shop_id}_person_event_data_filtered.csv"

        # Construir caminhos completos usando output_base_path
        person_path = os.path.join(output_base_path, person_filename).replace("\\", "/")
        event_path = os.path.join(output_base_path, event_filename).replace("\\", "/")

        print(f"Attempting to save person data to: {person_path}")
        save_dataframe(person_df, person_path)
        # Nota: save_dataframe atualmente não retorna status, apenas imprime erros.
        # Se um save falhar com exceção, o bloco except abaixo tratará.

        print(f"Attempting to save event data to: {event_path}")
        save_dataframe(person_event_df, event_path)

        print("Data fetching and saving process completed.")
        # Retorna True se nenhuma exceção grave ocorreu.
        # Poderia ser mais robusto se save_dataframe retornasse status.
        return overall_success

    except Exception as e:
        print(f"\n--- Error in load_all_db_data: {e} ---")
        traceback.print_exc()
        return False # Indicate critical failure
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase engine disposed, connection closed.")