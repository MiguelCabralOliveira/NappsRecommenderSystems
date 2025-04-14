# preprocessing/person_processor/process_direct_features.py
import pandas as pd
import traceback

def process_direct_person_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the direct columns (non-properties) of the person DataFrame.
    Keeps person_id_hex, converts is_identified, calculates date differences.
    """
    print("--- Processing Direct Person Columns ---")
    if 'person_id_hex' not in df.columns:
        raise ValueError("Input DataFrame must contain 'person_id_hex'")

    # Select necessary input columns + id
    required_cols = ['person_id_hex', 'is_identified', 'created_at', 'updated_at']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required direct columns: {missing}")

    # Start with the ID
    processed_df = df[['person_id_hex']].copy()
    now = pd.Timestamp.now(tz='UTC') # Use UTC for consistency

    # 1. Process is_identified
    try:
        print("Processing: is_identified (Convert to Int)")
        processed_df['is_identified'] = df['is_identified'].fillna(False).astype(int)
    except Exception as e:
        print(f"ERROR processing is_identified: {e}")
        traceback.print_exc()
        processed_df['is_identified'] = 0 # Default to 0 (False) on error

    # 2. Process created_at -> days_since_creation
    try:
        print("Processing: created_at (Calculate days_since_creation)")
        # Ensure conversion to datetime and handle timezone (assume UTC if naive)
        created_at_dt = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
        # Alternative handling if input might not be UTC:
        # created_at_dt = pd.to_datetime(df['created_at'], errors='coerce')
        # if pd.api.types.is_datetime64_any_dtype(created_at_dt):
        #     if created_at_dt.dt.tz is None:
        #         created_at_dt = created_at_dt.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        #     else:
        #         created_at_dt = created_at_dt.dt.tz_convert('UTC')

        delta = now - created_at_dt
        processed_df['days_since_creation'] = delta.dt.days.fillna(0).astype(int)
    except Exception as e:
        print(f"ERROR processing created_at: {e}")
        traceback.print_exc()
        processed_df['days_since_creation'] = 0 # Default on error

    # 3. Process updated_at -> days_since_last_update
    try:
        print("Processing: updated_at (Calculate days_since_last_update)")
        # Ensure conversion to datetime and handle timezone (assume UTC if naive)
        updated_at_dt = pd.to_datetime(df['updated_at'], errors='coerce', utc=True)
        # Alternative handling like created_at if needed

        delta = now - updated_at_dt
        processed_df['days_since_last_update'] = delta.dt.days.fillna(0).astype(int)
    except Exception as e:
        print(f"ERROR processing updated_at: {e}")
        traceback.print_exc()
        processed_df['days_since_last_update'] = 0 # Default on error

    print("--- Finished Processing Direct Person Columns ---")
    print(f"Direct features processed: {processed_df.columns.tolist()}")
    return processed_df