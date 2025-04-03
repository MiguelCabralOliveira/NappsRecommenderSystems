# preprocessing/process_properties_features.py
import pandas as pd
import json
import traceback
import ast

# --- JSON Parsing Helper ---
def safe_json_parse(properties_str):
    """Safely parses a JSON-like string, returning None on failure."""
    if pd.isna(properties_str) or not isinstance(properties_str, str) or properties_str.strip() == '{}' or properties_str.strip() == '':
        return None
    try:
        cleaned_str = properties_str.replace("'", '"')
        cleaned_str = cleaned_str.replace(': None', ': null').replace(', None', ', null').replace('{None', '{null')
        cleaned_str = cleaned_str.replace(': True', ': true').replace(', True', ', true').replace('{True', '{true')
        cleaned_str = cleaned_str.replace(': False', ': false').replace(', False', ', false').replace('{False', '{false')
        return json.loads(cleaned_str)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed_data = ast.literal_eval(properties_str)
        return parsed_data if isinstance(parsed_data, dict) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None

def parse_properties_column(df: pd.DataFrame) -> pd.Series:
    """Parses the 'properties' column string into Python objects."""
    if 'properties' not in df.columns:
        raise ValueError("Input DataFrame must contain 'properties'")
    print("--- Parsing 'properties' column ---")
    parsed_series = df['properties'].apply(safe_json_parse)
    fail_count = parsed_series.isnull().sum()
    total_count = len(df)
    non_empty_count = df['properties'].fillna('').str.strip().ne('').sum()
    fail_rate = (fail_count / non_empty_count * 100) if non_empty_count > 0 else 0
    print(f"Properties Parsing Summary: {fail_count}/{non_empty_count} non-empty failed ({fail_rate:.2f}% failure rate).")
    if fail_rate > 80 and non_empty_count > 10:
         print("WARNING: High failure rate parsing 'properties'. Check format.")
    print("--- Finished parsing 'properties' column ---")
    return parsed_series

# --- Main Properties Processing Function ---
def process_person_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and processes all relevant features from the 'properties' column.
    """
    print("--- Processing Person Properties Features ---")
    required_cols = ['person_id_hex', 'properties', 'is_identified']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns for properties processing: {missing}")

    # Start with ID and parse properties
    processed_df = df[['person_id_hex']].copy()
    parsed_props = parse_properties_column(df)
    now = pd.Timestamp.now(tz='UTC')
    MISSING_SESSION_DAYS = 9999

    # Create a temporary DF for easier processing
    temp_df = pd.DataFrame({
        'is_identified': df['is_identified'],
        'parsed_props': parsed_props
    })

    # --- Extract Features ---

    # Shopify State (Identified Only)
    print("Processing: properties -> shopify_state")
    processed_df['shopify_state'] = temp_df.apply(
        lambda row: row['parsed_props'].get('shopify_state') if row['is_identified'] and isinstance(row['parsed_props'], dict) else None,
        axis=1
    ).fillna('Unknown')

    # Shopify Currency (Identified Only)
    print("Processing: properties -> shopify_currency")
    processed_df['shopify_currency'] = temp_df.apply(
        lambda row: row['parsed_props'].get('shopify_currency') if row['is_identified'] and isinstance(row['parsed_props'], dict) else None,
        axis=1
    ).fillna('Unknown')

    # Shopify Verified Email (Identified Only)
    print("Processing: properties -> shopify_verified_email")
    processed_df['shopify_verified_email'] = temp_df.apply(
        lambda row: int(row['parsed_props'].get('shopify_verified_email', False) or False) if row['is_identified'] and isinstance(row['parsed_props'], dict) else 0,
        axis=1
    )

    # Shopify Account Age (Identified Only)
    print("Processing: properties -> shopify_account_age_days")
    def get_account_age(row):
        days = 0
        if row['is_identified'] and isinstance(row['parsed_props'], dict):
            created_str = row['parsed_props'].get('shopify_created_at')
            if created_str:
                created_dt = pd.to_datetime(created_str, errors='coerce')
                if pd.notna(created_dt):
                    if created_dt.tzinfo is None: created_dt = created_dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                    else: created_dt = created_dt.tz_convert('UTC')
                    if pd.notna(created_dt):
                        delta = now - created_dt
                        days = delta.days if pd.notna(delta) else 0
        return days if pd.notna(days) else 0
    processed_df['shopify_account_age_days'] = temp_df.apply(get_account_age, axis=1).fillna(0).astype(int)

    # City (Mainly Anonymous, but check all)
    print("Processing: properties -> city")
    processed_df['city'] = temp_df.apply(
        lambda row: row['parsed_props'].get('city') if isinstance(row['parsed_props'], dict) else None,
        axis=1
    ).fillna('Unknown')

    # Country ISO (Mainly Anonymous)
    print("Processing: properties -> country_iso")
    processed_df['country_iso'] = temp_df.apply(
        lambda row: row['parsed_props'].get('country_iso') if isinstance(row['parsed_props'], dict) else None,
        axis=1
    ).fillna('Unknown')

    # In EU (Mainly Anonymous)
    print("Processing: properties -> in_eu")
    processed_df['in_eu'] = temp_df.apply(
        lambda row: int(row['parsed_props'].get('in_eu', False) or False) if isinstance(row['parsed_props'], dict) else 0,
        axis=1
    )

    # Timezone (Mainly Anonymous)
    print("Processing: properties -> timezone")
    processed_df['timezone'] = temp_df.apply(
        lambda row: row['parsed_props'].get('timezone') if isinstance(row['parsed_props'], dict) else None,
        axis=1
    ).fillna('Unknown')

    # Days Since Last Session (Mainly Anonymous)
    print("Processing: properties -> days_since_last_session")
    def get_days_since_session(row):
        days = MISSING_SESSION_DAYS
        if isinstance(row['parsed_props'], dict):
            session_ts_str = row['parsed_props'].get('last_session_timestamp')
            if session_ts_str:
                session_dt = pd.to_datetime(session_ts_str, errors='coerce')
                if pd.notna(session_dt):
                    if session_dt.tzinfo is None: session_dt = session_dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                    else: session_dt = session_dt.tz_convert('UTC')
                    if pd.notna(session_dt):
                        delta = now - session_dt
                        days = delta.days if pd.notna(delta) else MISSING_SESSION_DAYS
        return days if pd.notna(days) else MISSING_SESSION_DAYS
    processed_df['days_since_last_session'] = temp_df.apply(get_days_since_session, axis=1).fillna(MISSING_SESSION_DAYS).astype(int)

    print("--- Finished Processing Person Properties Features ---")
    return processed_df