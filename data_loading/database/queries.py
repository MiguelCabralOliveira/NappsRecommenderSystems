# data_loading/database/queries.py
import pandas as pd
import traceback
from sqlalchemy import text, Engine # Use Engine type hint

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

# --- Define the specific events to fetch ---
# Keep this configurable or load from external config if needed
DESIRED_EVENT_TYPES = [
    "Collection List Viewed", "Product Searched", "Product Shared",
    "Product Viewed", "Product Wishlisted", "Shopify Checkout Created",
    "Shopify Checkout Updated", "Shopify Order Created", "Shopify Order Updated",
    "Variant Meta Added"
]
# -----------------------------------------

def fetch_person_data(engine: Engine, shop_id: str) -> pd.DataFrame:
    """Fetches person data for a specific shop_id (ID as hex)."""
    print(f"Fetching person data for shop_id: {shop_id}...")
    query = text("""
        SELECT
            encode(id, 'hex') AS person_id_hex,
            tenant_id,
            properties,
            is_identified,
            created_at,
            updated_at
        FROM person
        WHERE tenant_id = :shop_id;
    """)
    try:
        person_df = pd.read_sql(query, engine, params={'shop_id': shop_id})
        print(f"Successfully fetched {len(person_df)} person records.")
        if 'person_id_hex' in person_df.columns:
             null_ids = person_df['person_id_hex'].isnull().sum()
             if null_ids > 0:
                 print(f"Warning: Found {null_ids} null values in 'person_id_hex' (person data).")
        else:
             print("Warning: 'person_id_hex' column not found in person data.")
        if 'properties' in person_df.columns:
             # Ensure 'properties' exists before trying to convert type
             person_df['properties'] = person_df['properties'].astype(str)
        return person_df
    except Exception as e:
        print(f"Error fetching person data: {e}")
        traceback.print_exc()
        return pd.DataFrame() # Return empty DF on error

def fetch_distinct_event_types(engine: Engine, shop_id: str, days_limit: int) -> list:
    """Fetches ALL distinct event types within the time limit."""
    print(f"Fetching ALL distinct event types for shop_id: {shop_id} (last {days_limit} days)...")
    # Use bind parameters for days_limit for safety and potential DB optimization
    query_str = f"""
        SELECT DISTINCT event
        FROM person_event
        WHERE tenant_id = :shop_id
          AND timestamp >= NOW() - INTERVAL '1 day' * :days_limit;
    """
    query = text(query_str)
    try:
        params = {'shop_id': shop_id, 'days_limit': days_limit}
        result = pd.read_sql(query, engine, params=params)
        event_types = result['event'].tolist()
        print(f"Found {len(event_types)} distinct event types in total.")
        return event_types
    except Exception as e:
        print(f"Error fetching distinct event types: {e}")
        traceback.print_exc()
        return []

def fetch_filtered_event_data(engine: Engine, shop_id: str, days_limit: int) -> pd.DataFrame:
    """Fetches SPECIFIC person_event data (filtered by DESIRED_EVENT_TYPES)."""
    print(f"Fetching SPECIFIC person_event data for shop_id: {shop_id} (last {days_limit} days)...")
    print(f"Filtering for event types: {DESIRED_EVENT_TYPES}")

    # Use bind parameters for days_limit and the event list tuple
    query_str = f"""
        SELECT
            encode(person_id, 'hex') AS person_id_hex,
            tenant_id, event_id, event, type, user_id, anonymous_id,
            properties, context, person_properties, timestamp
        FROM person_event
        WHERE tenant_id = :shop_id
          AND timestamp >= NOW() - INTERVAL '1 day' * :days_limit
          AND event IN :event_list;
    """
    query = text(query_str)

    try:
        params = {
            'shop_id': shop_id,
            'days_limit': days_limit,
            'event_list': tuple(DESIRED_EVENT_TYPES) # Use tuple for IN clause
        }
        person_event_df = pd.read_sql(query, engine, params=params)
        print(f"Successfully fetched {len(person_event_df)} filtered person_event records.")

        if 'person_id_hex' in person_event_df.columns:
            null_ids = person_event_df['person_id_hex'].isnull().sum()
            if null_ids > 0:
                print(f"Warning: Found {null_ids} null values in 'person_id_hex' (event data).")
        else:
            print("Warning: 'person_id_hex' column not found in event data.")

        for col in ['properties', 'context', 'person_properties']:
            if col in person_event_df.columns:
                # Check if column exists before trying conversion
                person_event_df[col] = person_event_df[col].astype(str)

        return person_event_df
    except Exception as e:
        print(f"Error fetching filtered person_event data: {e}")
        traceback.print_exc()
        return pd.DataFrame() # Return empty DF on error