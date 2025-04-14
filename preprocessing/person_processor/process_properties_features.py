# preprocessing/person_processor/process_properties_features.py
import pandas as pd
import json
import traceback
import ast
# Use relative import for the utility function
from ..utils import safe_json_parse # Adjusted relative import

# --- Main Properties Processing Function ---
def process_person_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and processes all relevant features from the 'properties' column.
    Relies on 'person_id_hex' and 'is_identified' being present in the input df.
    """
    print("--- Processing Person Properties Features ---")
    required_cols = ['person_id_hex', 'properties', 'is_identified']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns for properties processing: {missing}")

    # Start with ID and parse properties
    processed_df = df[['person_id_hex']].copy() # Keep only ID initially
    parsed_props = df['properties'].apply(safe_json_parse) # Parse the original properties column
    now = pd.Timestamp.now(tz='UTC')
    MISSING_SESSION_DAYS = 9999 # Define a constant for missing session days

    # Create a temporary Series/DF for easier processing based on parsed props
    temp_props_df = pd.DataFrame({
        'is_identified': df['is_identified'], # Use original is_identified
        'parsed_props': parsed_props
    })

    # --- Extract Features ---
    # Helper to safely get values from the parsed dictionary
    def safe_get(props_dict, key, default=None):
        return props_dict.get(key, default) if isinstance(props_dict, dict) else default

    # Shopify State (Identified Only)
    print("Processing: properties -> shopify_state")
    processed_df['shopify_state'] = temp_props_df.apply(
        lambda row: safe_get(row['parsed_props'], 'shopify_state') if row['is_identified'] else None,
        axis=1
    ).fillna('Unknown')

    # Shopify Currency (Identified Only)
    print("Processing: properties -> shopify_currency")
    processed_df['shopify_currency'] = temp_props_df.apply(
        lambda row: safe_get(row['parsed_props'], 'shopify_currency') if row['is_identified'] else None,
        axis=1
    ).fillna('Unknown')

    # Shopify Verified Email (Identified Only)
    print("Processing: properties -> shopify_verified_email")
    processed_df['shopify_verified_email'] = temp_props_df.apply(
        lambda row: int(safe_get(row['parsed_props'], 'shopify_verified_email', False) or False) if row['is_identified'] else 0,
        axis=1
    ).astype(int) # Ensure integer type

    # Shopify Account Age (Identified Only)
    print("Processing: properties -> shopify_account_age_days")
    def get_account_age(row):
        days = 0
        if row['is_identified']:
            created_str = safe_get(row['parsed_props'], 'shopify_created_at')
            if created_str:
                # Ensure conversion to datetime and handle timezone (assume UTC)
                created_dt = pd.to_datetime(created_str, errors='coerce', utc=True)
                if pd.notna(created_dt):
                    delta = now - created_dt
                    days = delta.days if pd.notna(delta) else 0
        # Ensure non-negative age and handle potential NaNs from date parsing
        return max(0, int(days)) if pd.notna(days) else 0
    processed_df['shopify_account_age_days'] = temp_props_df.apply(get_account_age, axis=1).fillna(0).astype(int)

    # City (Mainly Anonymous, but check all)
    print("Processing: properties -> city")
    processed_df['city'] = temp_props_df['parsed_props'].apply(lambda x: safe_get(x, 'city')).fillna('Unknown')

    # Country ISO (Mainly Anonymous)
    print("Processing: properties -> country_iso")
    processed_df['country_iso'] = temp_props_df['parsed_props'].apply(lambda x: safe_get(x, 'country_iso')).fillna('Unknown')

    # In EU (Mainly Anonymous)
    print("Processing: properties -> in_eu")
    processed_df['in_eu'] = temp_props_df['parsed_props'].apply(lambda x: int(safe_get(x, 'in_eu', False) or False)).astype(int) # Ensure integer

    # Timezone (Mainly Anonymous)
    print("Processing: properties -> timezone")
    timezone_series = temp_props_df['parsed_props'].apply(lambda x: safe_get(x, 'timezone'))
    timezone_series = timezone_series.replace('', None) # Treat empty strings as None
    processed_df['timezone'] = timezone_series.fillna('Unknown')

    # Days Since Last Session (Mainly Anonymous)
    print("Processing: properties -> days_since_last_session")
    def get_days_since_session(row):
        days = MISSING_SESSION_DAYS
        session_ts_str = safe_get(row['parsed_props'], 'last_session_timestamp')
        if session_ts_str:
            # Ensure conversion to datetime and handle timezone (assume UTC)
            session_dt = pd.to_datetime(session_ts_str, errors='coerce', utc=True)
            if pd.notna(session_dt):
                delta = now - session_dt
                days = delta.days if pd.notna(delta) else MISSING_SESSION_DAYS
        # Ensure non-negative days and handle potential NaNs
        return max(0, int(days)) if pd.notna(days) else MISSING_SESSION_DAYS
    processed_df['days_since_last_session'] = temp_props_df.apply(get_days_since_session, axis=1).fillna(MISSING_SESSION_DAYS).astype(int)

    print("--- Finished Processing Person Properties Features ---")
    print(f"Properties features processed: {processed_df.columns.tolist()}")
    # Ensure person_id_hex is the first column for clarity if needed, though merge handles it
    # cols = ['person_id_hex'] + [col for col in processed_df.columns if col != 'person_id_hex']
    # processed_df = processed_df[cols]
    return processed_df