# data_loader.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import traceback

load_dotenv()

# --- Define the specific events to fetch ---
DESIRED_EVENT_TYPES = [
    "Collection List Viewed",
    "Product Searched",
    "Product Shared",
    "Product Viewed",
    "Product Wishlisted",
    "Shopify Checkout Created",
    "Shopify Checkout Updated",
    "Shopify Order Created",
    "Shopify Order Updated",
    "Variant Meta Added"
]
# -----------------------------------------

def get_db_engine():
    """Creates and returns a SQLAlchemy engine."""
    db_host = os.getenv('DB_HOST')
    db_password = os.getenv('DB_PASSWORD')
    db_user = os.getenv('DB_USER')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')

    if not all([db_host, db_password, db_user, db_name, db_port]):
        raise ValueError("Database credentials not found in .env file.")

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    try:
        engine = create_engine(connection_string)
        # Optional: Test connection immediately
        # with engine.connect() as connection:
        #     print("Database connection successful.")
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        traceback.print_exc()
        raise


def load_person_data(engine, shop_id: str) -> pd.DataFrame:
    """
    Fetches person data for a specific shop_id.
    The 'id' (person_id) is fetched directly as a hex string.
    """
    print(f"\nFetching person data for shop_id: {shop_id}...")
    # --- Modified Query: Fetch id as hex string ---
    query = text("""
        SELECT
            encode(id, 'hex') AS person_id_hex, -- Fetch as hex string
            tenant_id,
            properties,
            is_identified,
            created_at,
            updated_at
        FROM person
        WHERE tenant_id = :shop_id;
    """)
    # ---------------------------------------------
    try:
        person_df = pd.read_sql(query, engine, params={'shop_id': shop_id})
        print(f"Successfully fetched {len(person_df)} person records.")
        # Ensure 'person_id_hex' exists and check for nulls
        if 'person_id_hex' in person_df.columns:
             null_ids = person_df['person_id_hex'].isnull().sum()
             if null_ids > 0:
                 print(f"Warning: Found {null_ids} null values in 'person_id_hex' after loading.")
             # Optional: Drop rows with null person_id_hex if they are invalid
             # person_df = person_df.dropna(subset=['person_id_hex'])
             # print(f"Shape after dropping null person_id_hex: {person_df.shape}")
        else:
             print("Warning: 'person_id_hex' column not found after query execution.")

        if 'properties' in person_df.columns:
             person_df['properties'] = person_df['properties'].astype(str) # Keep this for consistency

        return person_df
    except Exception as e:
        print(f"Error fetching person data: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def fetch_distinct_event_types(engine, shop_id: str, days_limit: int) -> list:
    """
    Fetches distinct event types within the time limit for the shop_id.
    (Remains unchanged)
    """
    print(f"\nFetching ALL distinct event types for shop_id: {shop_id} (last {days_limit} days)...")
    query = text(f"""
        SELECT DISTINCT event
        FROM person_event
        WHERE tenant_id = :shop_id
          AND timestamp >= NOW() - INTERVAL '{days_limit} day';
    """)
    try:
        result = pd.read_sql(query, engine, params={'shop_id': shop_id})
        event_types = result['event'].tolist()
        print(f"Found {len(event_types)} distinct event types in total.")
        return event_types
    except Exception as e:
        print(f"Error fetching distinct event types: {e}")
        traceback.print_exc()
        return []


def load_person_event_data(engine, shop_id: str, days_limit: int = 90) -> pd.DataFrame:
    """
    Fetches SPECIFIC person_event data for a shop_id within a time limit,
    filtered by the DESIRED_EVENT_TYPES list.
    The 'person_id' is fetched directly as a hex string.
    """
    print(f"\nFetching SPECIFIC person_event data for shop_id: {shop_id} (last {days_limit} days)...")
    print(f"Filtering for event types: {DESIRED_EVENT_TYPES}")

    # --- Modified Query: Fetch person_id as hex string and filter events ---
    query = text(f"""
        SELECT
            encode(person_id, 'hex') AS person_id_hex, -- Fetch as hex string
            tenant_id,
            event_id,
            event,
            type,
            user_id,
            anonymous_id,
            properties,
            context,
            person_properties,
            timestamp
        FROM person_event
        WHERE tenant_id = :shop_id
          AND timestamp >= NOW() - INTERVAL '{days_limit} day'
          AND event IN :event_list;
    """)
    # --------------------------------------------------------------------

    try:
        params = {
            'shop_id': shop_id,
            'event_list': tuple(DESIRED_EVENT_TYPES) # Must be a tuple for IN clause
        }

        person_event_df = pd.read_sql(query, engine, params=params)
        print(f"Successfully fetched {len(person_event_df)} filtered person_event records.")

        # Ensure 'person_id_hex' exists and check for nulls
        if 'person_id_hex' in person_event_df.columns:
            null_ids = person_event_df['person_id_hex'].isnull().sum()
            if null_ids > 0:
                print(f"Warning: Found {null_ids} null values in 'person_id_hex' after loading event data.")
            # Optional: Drop rows with null person_id_hex if they are invalid
            # person_event_df = person_event_df.dropna(subset=['person_id_hex'])
            # print(f"Shape after dropping null person_id_hex from events: {person_event_df.shape}")
        else:
            print("Warning: 'person_id_hex' column not found in event data after query execution.")


        for col in ['properties', 'context', 'person_properties']:
            if col in person_event_df.columns:
                person_event_df[col] = person_event_df[col].astype(str)

        return person_event_df
    except Exception as e:
        print(f"Error fetching filtered person_event data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def load_all_data(shop_id: str, days_limit: int = 90):
    """
    Loads person data and SPECIFIC filtered person_event data (with hex IDs),
    and all distinct event types found in the period.
    """
    engine = None
    distinct_event_types = []
    person_df = pd.DataFrame()
    person_event_df = pd.DataFrame()
    try:
        print("Attempting to establish database connection...")
        engine = get_db_engine()
        print("Database engine created.")

        person_df = load_person_data(engine, shop_id)
        distinct_event_types = fetch_distinct_event_types(engine, shop_id, days_limit)
        person_event_df = load_person_event_data(engine, shop_id, days_limit)

        # Validation check after loading
        if not person_event_df.empty:
            found_events = person_event_df['event'].unique()
            print(f"\nFound {len(found_events)} of the desired event types in the loaded data:")
            print(found_events)
        elif not distinct_event_types:
             print("\nWarning: No distinct event types found at all in the specified period.")
        else:
             print("\nWarning: No events matching the desired types were found in the specified period.")


        return person_df, person_event_df, distinct_event_types
    except Exception as e:
        print(f"\n--- Error in load_all_data: {e} ---")
        traceback.print_exc()
        return person_df, person_event_df, distinct_event_types # Return potentially partial data
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")

# Standalone test block
if __name__ == "__main__":
    print("=== Running data_loader.py standalone test ===")
    test_shop_id = os.getenv("TEST_SHOP_ID", "sjstarcorp") # Example fallback
    test_days = 90

    print(f"Using Shop ID: {test_shop_id}")

    try:
        person_df, person_event_df, distinct_events = load_all_data(test_shop_id, test_days)

        print("\n--- Person Data Sample (with person_id_hex) ---")
        if person_df is not None and not person_df.empty:
            print(person_df.head())
            print(f"Shape: {person_df.shape}")
            if 'person_id_hex' in person_df.columns:
                print("Sample person_id_hex:", person_df['person_id_hex'].iloc[0])
                print("Type of person_id_hex:", type(person_df['person_id_hex'].iloc[0]))
        else:
            print("No person data loaded or an error occurred.")

        print("\n--- Person Event Data Sample (Filtered, with person_id_hex) ---")
        if person_event_df is not None and not person_event_df.empty:
            print(person_event_df.head())
            print(f"Shape: {person_event_df.shape}")
            if 'person_id_hex' in person_event_df.columns:
                 print("Sample person_id_hex:", person_event_df['person_id_hex'].iloc[0])
                 print("Type of person_id_hex:", type(person_event_df['person_id_hex'].iloc[0]))
            print("\n--- Value Counts of Loaded (Filtered) Events ---")
            print(person_event_df['event'].value_counts())
        else:
            print("No filtered event data loaded or an error occurred.")


        print("\n--- All Distinct Event Types Found in Period ---")
        print(f"Total distinct types found: {len(distinct_events)}")
        print(distinct_events[:20])
        if len(distinct_events) > 20: print("...")


        print("\nStandalone test completed.")
    except Exception as e:
        print(f"\nStandalone test failed: {e}")
        traceback.print_exc()