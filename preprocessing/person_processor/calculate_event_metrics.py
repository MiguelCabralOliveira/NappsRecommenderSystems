# preprocessing/person_processor/calculate_event_metrics.py
import pandas as pd
import numpy as np
import os
import traceback

def calculate_person_event_metrics(event_interactions_path: str) -> pd.DataFrame:
    """
    Loads processed event interactions and calculates person-level summary metrics.

    Args:
        event_interactions_path: Path to the 'event_interactions_processed.csv' file.

    Returns:
        DataFrame indexed by 'person_id_hex' with calculated metrics,
        or an empty DataFrame if the input file doesn't exist or is empty.
    """
    print(f"--- Calculating Person Metrics from Event Interactions ---")
    print(f"Loading event interactions from: {event_interactions_path}")

    if not os.path.exists(event_interactions_path):
        print(f"Warning: Event interactions file not found at {event_interactions_path}. Skipping event metrics calculation.")
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

    try:
        dtypes = {
            'person_id_hex': 'string', 'product_id': 'string', 'interaction_type': 'string',
            'event_converted_price': 'float64', 'checkout_id': 'Int64', 'channel': 'string',
            'person_interaction_sequence': 'Int64'
            # Note: Don't specify dtype for timestamp here, we'll convert it manually
        }
        events_df = pd.read_csv(
            event_interactions_path,
            dtype=dtypes,
            # parse_dates=['timestamp'], # Remove parse_dates from here
            low_memory=False
        )
        print(f"Loaded {len(events_df)} interactions.")

        if events_df.empty:
            print("Event interactions file is empty. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # --- Start Correction: Explicit Datetime Conversion ---
        print("Converting 'timestamp' column to datetime...")
        if 'timestamp' in events_df.columns:
            # Convert to datetime, coercing errors to NaT (Not a Time)
            # Force UTC timezone awareness during conversion
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], errors='coerce', utc=True)

            # Check how many values failed conversion (became NaT)
            nat_count = events_df['timestamp'].isna().sum()
            if nat_count > 0:
                print(f"Warning: {nat_count} timestamp values could not be parsed and were set to NaT.")

            # Drop rows with NaT timestamps as they cannot be used in calculations
            initial_rows = len(events_df)
            events_df.dropna(subset=['timestamp'], inplace=True)
            rows_dropped = initial_rows - len(events_df)
            if rows_dropped > 0:
                print(f"Dropped {rows_dropped} rows due to invalid timestamps (NaT).")

        else:
            print("Error: 'timestamp' column not found in the loaded events data. Cannot proceed with metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))
        # --- End Correction ---


        if events_df.empty:
            print("No valid timestamps remaining after conversion and dropping NaT. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # Now we can safely use the .dt accessor because the column is guaranteed to be datetime
        # The timezone check below might be redundant if utc=True worked, but it's safe to keep.
        print("Ensuring timestamp column is timezone-aware (UTC)...")
        if events_df['timestamp'].dt.tz is None:
            print("Warning: Timestamps were still naive after conversion attempt. Localizing to UTC.")
            # This case should be less likely now with utc=True in to_datetime
            events_df['timestamp'] = events_df['timestamp'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
            # Drop any NaTs potentially introduced by tz_localize
            events_df.dropna(subset=['timestamp'], inplace=True)
        else:
            # Ensure it's UTC, convert if it's some other timezone
            if str(events_df['timestamp'].dt.tz) != 'UTC':
                 print(f"Converting timestamps from {events_df['timestamp'].dt.tz} to UTC...")
                 events_df['timestamp'] = events_df['timestamp'].dt.tz_convert('UTC')

        if events_df.empty:
            print("No valid timestamps remaining after timezone handling. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))


        print("Aggregating event metrics per person...")
        now_utc = pd.Timestamp.now(tz='UTC')
        agg_dict = {
            'total_interactions': pd.NamedAgg(column='timestamp', aggfunc='count'),
            'unique_products_interacted': pd.NamedAgg(column='product_id', aggfunc='nunique'),
            'view_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Viewed').sum()),
            'purchase_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Order Created').sum()),
            'wishlist_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Wishlisted').sum()),
            'share_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Shared').sum()),
            'checkout_start_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Checkout Created').sum()),
            'last_interaction_timestamp': pd.NamedAgg(column='timestamp', aggfunc='max'),
            'first_interaction_timestamp': pd.NamedAgg(column='timestamp', aggfunc='min'),
            'distinct_channels_count': pd.NamedAgg(column='channel', aggfunc='nunique'),
            'variant_meta_added_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Variant Meta Added').sum()),
            'checkout_update_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Checkout Updated').sum()),
            'order_update_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Order Updated').sum()),
        }
        person_metrics = events_df.groupby('person_id_hex').agg(**agg_dict)

        print("Calculating derived metrics...")
        person_metrics['is_buyer'] = (person_metrics['purchase_count'] > 0).astype(int)
        person_metrics['days_since_last_interaction'] = (now_utc - person_metrics['last_interaction_timestamp']).dt.days.fillna(9999).astype(int)
        person_metrics['days_since_first_interaction'] = (now_utc - person_metrics['first_interaction_timestamp']).dt.days.fillna(9999).astype(int)

        most_freq_channel = events_df.groupby('person_id_hex')['channel'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').rename('most_frequent_channel')
        person_metrics = person_metrics.join(most_freq_channel, how='left')
        person_metrics['most_frequent_channel'] = person_metrics['most_frequent_channel'].fillna('Unknown')

        purchase_events = events_df[events_df['interaction_type'] == 'Shopify Order Created'].copy()
        if not purchase_events.empty and 'event_converted_price' in purchase_events.columns:
             purchase_events['event_converted_price'] = pd.to_numeric(purchase_events['event_converted_price'], errors='coerce')
             purchase_summary = purchase_events.groupby('person_id_hex')['event_converted_price'].agg(['sum', 'count'])
             purchase_summary['avg_purchase_value'] = (purchase_summary['sum'] / purchase_summary['count'].replace(0, 1)).fillna(0)
             person_metrics = person_metrics.join(purchase_summary[['avg_purchase_value']], how='left')
             person_metrics['avg_purchase_value'] = person_metrics['avg_purchase_value'].fillna(0)
        else:
             person_metrics['avg_purchase_value'] = 0.0

        person_metrics = person_metrics.drop(columns=['last_interaction_timestamp', 'first_interaction_timestamp'], errors='ignore')
        print(f"Calculated metrics for {len(person_metrics)} persons.")
        print("Sample metrics:")
        print(person_metrics.head()) # Print head to see if values are non-zero now
        print("\nMetrics columns:", person_metrics.columns.tolist())

        return person_metrics

    except Exception as e:
        print(f"Error calculating person event metrics: {e}")
        traceback.print_exc()
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))