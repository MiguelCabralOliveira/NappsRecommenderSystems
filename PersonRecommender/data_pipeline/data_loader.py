# data_loader.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import traceback

load_dotenv()

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
        with engine.connect() as connection:
            print("Database connection successful.")
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        traceback.print_exc()
        raise


def load_person_data(engine, shop_id: str) -> pd.DataFrame:
    """Fetches person data for a specific shop_id."""
    
    print(f"\nFetching person data for shop_id: {shop_id}...")
    query = text("""
        SELECT id, tenant_id, properties, is_identified, created_at, updated_at
        FROM person
        WHERE tenant_id = :shop_id;
    """)
    try:
        person_df = pd.read_sql(query, engine, params={'shop_id': shop_id})
        print(f"Successfully fetched {len(person_df)} person records.")
        if 'properties' in person_df.columns:
             person_df['properties'] = person_df['properties'].astype(str)
        # Ensure 'id' (person_id from person table) is loaded, handle potential binary later if needed
        return person_df
    except Exception as e:
        print(f"Error fetching person data: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def fetch_distinct_event_types(engine, shop_id: str, days_limit: int) -> list:
    """Fetches distinct event types within the time limit for the shop_id."""
    
    print(f"\nFetching distinct event types for shop_id: {shop_id} (last {days_limit} days)...")
    query = text(f"""
        SELECT DISTINCT event
        FROM person_event
        WHERE tenant_id = :shop_id
          AND timestamp >= NOW() - INTERVAL '{days_limit} day';
    """)
    try:
        result = pd.read_sql(query, engine, params={'shop_id': shop_id})
        event_types = result['event'].tolist()
        print(f"Found {len(event_types)} distinct event types.")
        return event_types
    except Exception as e:
        print(f"Error fetching distinct event types: {e}")
        traceback.print_exc()
        return []


def load_person_event_data(engine, shop_id: str, days_limit: int = 90) -> pd.DataFrame:
    """
    Fetches ALL person_event data for a specific shop_id within a time limit.
    Crucially includes person_id.
    """
    print(f"\nFetching ALL person_event data for shop_id: {shop_id} (last {days_limit} days)...")

    # Ensure person_id is selected
    query = text(f"""
        SELECT
            person_id,  -- Make sure this column is selected
            tenant_id,
            event_id,
            event,
            type,
            user_id,    -- This is likely the Shopify Customer ID (numeric/string)
            anonymous_id,
            properties,
            context,
            person_properties,
            timestamp
        FROM person_event
        WHERE tenant_id = :shop_id
          AND timestamp >= NOW() - INTERVAL '{days_limit} day';
    """)

    try:
        params = {'shop_id': shop_id}
        person_event_df = pd.read_sql(query, engine, params=params)
        print(f"Successfully fetched {len(person_event_df)} total person_event records.")

        # Check if person_id was loaded (it might be bytes)
        if 'person_id' in person_event_df.columns:
            print(f"Column 'person_id' loaded. Sample type: {type(person_event_df['person_id'].iloc[0]) if not person_event_df.empty else 'N/A'}")
        else:
            print("Warning: Column 'person_id' not found in fetched data!")

        # Basic check/conversion for JSON-like columns
        for col in ['properties', 'context', 'person_properties']:
            if col in person_event_df.columns:
                person_event_df[col] = person_event_df[col].astype(str)

        return person_event_df
    except Exception as e:
        print(f"Error fetching person_event data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def load_all_data(shop_id: str, days_limit: int = 90):
    """Loads person data, all person_event data, and distinct event types."""
   
    engine = None
    distinct_event_types = []
    person_df = pd.DataFrame()
    person_event_df = pd.DataFrame()
    try:
        engine = get_db_engine()
        person_df = load_person_data(engine, shop_id)
        distinct_event_types = fetch_distinct_event_types(engine, shop_id, days_limit)
        person_event_df = load_person_event_data(engine, shop_id, days_limit)
        return person_df, person_event_df, distinct_event_types
    finally:
        if engine:
            engine.dispose()
            print("\nDatabase connection closed.")

# Standalone test block
if __name__ == "__main__":
    
    print("=== Running data_loader.py standalone test ===")
    test_shop_id = "sjstarcorp"
    test_days = 90

    try:
        person_df, person_event_df, distinct_events = load_all_data(test_shop_id, test_days)

        print("\n--- Person Data Sample ---")
        print(person_df.head())
        print(f"Shape: {person_df.shape}")

        print("\n--- Person Event Data Sample (All Events) ---")
        print(person_event_df.head())
        print(f"Shape: {person_event_df.shape}")
        if not person_event_df.empty and 'person_id' in person_event_df.columns:
             print("Sample person_id (raw):", person_event_df['person_id'].iloc[0])
             print("Type of person_id:", type(person_event_df['person_id'].iloc[0]))


        print("\n--- Distinct Event Types Found ---")
        print(f"Total distinct types: {len(distinct_events)}")
        print(distinct_events[:20])
        if len(distinct_events) > 20: print("...")

        print("\n--- Value Counts of Loaded Events ---")
        if not person_event_df.empty:
            print(person_event_df['event'].value_counts())
        else:
            print("No event data loaded.")

        print("\nStandalone test completed.")
    except Exception as e:
        print(f"\nStandalone test failed: {e}")
        traceback.print_exc()