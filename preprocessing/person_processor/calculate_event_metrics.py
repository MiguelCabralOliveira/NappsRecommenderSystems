# preprocessing/person_processor/calculate_event_metrics.py
import pandas as pd
import numpy as np
import os
import traceback

def calculate_person_event_metrics(event_interactions_path: str) -> pd.DataFrame:
    """
    Loads processed event interactions and calculates person-level summary metrics,
    identifying buyers directly from cleaned data before aggregation.

    Args:
        event_interactions_path: Path to the 'event_interactions_processed.csv' file.

    Returns:
        DataFrame indexed by 'person_id_hex' with calculated metrics,
        or an empty DataFrame if the input file doesn't exist or is empty.
    """
    print(f"--- Calculating Person Metrics from Event Interactions (Direct Buyer Check) ---")
    print(f"Loading event interactions from: {event_interactions_path}")

    if not os.path.exists(event_interactions_path):
        print(f"Warning: Event interactions file not found at {event_interactions_path}. Skipping event metrics calculation.")
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

    try:
        dtypes = {
            'person_id_hex': 'string', 'product_id': 'string', 'interaction_type': 'string',
            'event_converted_price': 'float64', 'checkout_id': 'Int64', 'channel': 'string',
            'person_interaction_sequence': 'Int64'
            # Timestamp dtype is handled manually below
        }
        events_df = pd.read_csv(
            event_interactions_path,
            dtype=dtypes,
            # No parse_dates here, read as object
            low_memory=False
        )
        print(f"Loaded {len(events_df)} interactions (timestamp as object initially).")

        if events_df.empty:
            print("Event interactions file is empty. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # --- Two-Pass Timestamp Conversion ---
        print("Converting 'timestamp' column using two-pass format check...")
        if 'timestamp' in events_df.columns:
            timestamps_obj = events_df['timestamp'].astype(str).str.strip() # Work on a copy as strings
            format_pass1 = '%Y-%m-%d %H:%M:%S.%f%z'
            format_pass2 = '%Y-%m-%d %H:%M:%S%z'

            print(f"Attempting conversion with format 1: {format_pass1}")
            # Pass 1: With microseconds
            converted_pass1 = pd.to_datetime(timestamps_obj, format=format_pass1, errors='coerce', utc=True)
            failures_pass1_mask = converted_pass1.isna()
            num_failures_pass1 = failures_pass1_mask.sum()
            print(f"Pass 1 failed for {num_failures_pass1} rows.")

            # Pass 2: Without microseconds (only on rows that failed pass 1)
            converted_pass2 = converted_pass1.copy() # Start with results from pass 1
            if num_failures_pass1 > 0:
                print(f"Attempting conversion with format 2: {format_pass2} on failed rows...")
                converted_pass2[failures_pass1_mask] = pd.to_datetime(
                    timestamps_obj[failures_pass1_mask], # Only try converting the failed ones
                    format=format_pass2,
                    errors='coerce',
                    utc=True
                )
                num_failures_pass2 = converted_pass2.isna().sum() # Total failures after both passes
                print(f"Pass 2 failed for {num_failures_pass2 - (len(events_df) - num_failures_pass1)} additional rows (if any). Total NaNs: {num_failures_pass2}")

                if num_failures_pass2 < num_failures_pass1:
                    print(f"Pass 2 successfully converted {num_failures_pass1 - num_failures_pass2} timestamps.")

            else:
                print("No failures in Pass 1, skipping Pass 2.")
                num_failures_pass2 = 0 # No failures overall

            # Assign the final converted series back to the DataFrame
            events_df['timestamp'] = converted_pass2

            # Drop rows that failed both passes
            if num_failures_pass2 > 0:
                initial_rows = len(events_df)
                events_df.dropna(subset=['timestamp'], inplace=True)
                rows_dropped = initial_rows - len(events_df)
                print(f"Dropped {rows_dropped} rows due to NaT timestamps after both conversion passes.")
            else:
                print("Successfully converted all timestamps using two-pass approach.")

        else:
             print("Error: 'timestamp' column not found in the loaded events data.")
             return pd.DataFrame(index=pd.Index([], name='person_id_hex'))
        # --- End Two-Pass Conversion ---

        if events_df.empty:
            print("No valid timestamps remaining after two-pass conversion and dropping NaT. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # Timezone should be handled by utc=True in to_datetime, verify just in case
        print("Verifying timestamp column timezone (should be UTC)...")
        if pd.api.types.is_datetime64_any_dtype(events_df['timestamp']):
            if events_df['timestamp'].dt.tz is None or str(events_df['timestamp'].dt.tz) != 'UTC':
                print(f"Warning: Timestamps are not UTC-aware after conversion (tz={events_df['timestamp'].dt.tz}). Forcing UTC.")
                try:
                    # Force interpretation as UTC if naive, or convert if different tz
                    if events_df['timestamp'].dt.tz is None:
                         events_df['timestamp'] = events_df['timestamp'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                    else:
                         events_df['timestamp'] = events_df['timestamp'].dt.tz_convert('UTC')
                    events_df.dropna(subset=['timestamp'], inplace=True) # Drop NaTs potentially introduced
                except Exception as tz_err:
                     print(f"Error during timezone enforcement: {tz_err}")
            else:
                print("Timestamps are correctly timezone-aware (UTC).")
        else:
            print("Error: Timestamp column is not datetime after conversion steps.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        if events_df.empty:
            print("No valid timestamps remaining after timezone enforcement. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # --- *** STEP 5: Identify Buyers Directly *** ---
        print("\nIdentifying buyers directly from cleaned event data...")
        buyer_ids = set(
            events_df.loc[events_df['interaction_type'] == 'Shopify Order Created', 'person_id_hex'].unique()
        )
        print(f"Found {len(buyer_ids)} unique person_id_hex with 'Shopify Order Created' events.")
        # --- *** End Identify Buyers *** ---

        # --- DEBUG PRINT 2: Check Unique Interaction Types (should be correct now) ---
        if 'interaction_type' in events_df.columns:
            print("\n--- DEBUG: Unique Interaction Types Found in cleaned data ---")
            unique_types = events_df['interaction_type'].unique()
            print(unique_types)
            target_type = 'Shopify Order Created'
            if target_type in unique_types:
                 print(f"'{target_type}' IS PRESENT in unique types.") # <-- Should be present now
            else:
                 print(f"WARNING: '{target_type}' IS NOT PRESENT in unique types found! (Check cleaning steps)")
            print("--- END DEBUG ---")
        else:
            # This shouldn't happen if loading worked
            print("\n--- DEBUG: 'interaction_type' column NOT FOUND ---")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))


        # --- Aggregate Metrics (Step 6) ---
        print("\nAggregating event metrics per person...")
        now_utc = pd.Timestamp.now(tz='UTC')

        agg_dict = {
            # Keep purchase_count for potential comparison/validation
            'purchase_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Order Created').sum()),
            'total_interactions': pd.NamedAgg(column='timestamp', aggfunc='count'),
            'unique_products_interacted': pd.NamedAgg(column='product_id', aggfunc='nunique'),
            'view_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Viewed').sum()),
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
        print(f"Aggregated metrics for {len(person_metrics)} persons.")

        # --- DEBUG PRINT 4: Check purchase_count from aggregation (should match direct check now) ---
        print("\n--- DEBUG: Metrics DataFrame AFTER aggregation (showing purchase_count) ---")
        print(person_metrics[['total_interactions', 'purchase_count']].head(20))
        print("\nDistribution of aggregated 'purchase_count':")
        print(person_metrics['purchase_count'].value_counts())
        print("--- END DEBUG ---")

        # --- Calculate Derived Metrics ---
        print("\nCalculating derived metrics...")

        # --- *** STEP 7: Create is_buyer using the identified set *** ---
        # Use the index of person_metrics (which is person_id_hex)
        person_metrics['is_buyer'] = person_metrics.index.isin(buyer_ids).astype(int)
        # --- *** End Create is_buyer *** ---

        # --- DEBUG PRINT 5: Check is_buyer AFTER calculation ---
        print("\n--- DEBUG: Metrics DataFrame AFTER calculating 'is_buyer' (using direct check) ---")
        # Show purchase_count (from agg) and is_buyer (from direct check) side-by-side
        print(person_metrics[['purchase_count', 'is_buyer']].head(20))
        print("\nDistribution of 'is_buyer':")
        is_buyer_counts = person_metrics['is_buyer'].value_counts()
        print(is_buyer_counts)
        if 1 in is_buyer_counts:
            print(f"SUCCESS: Found {is_buyer_counts[1]} buyers.")
        else:
            print("WARNING: Still found 0 buyers. Re-check buyer identification step and input data.")
        print("--- END DEBUG ---")

        # --- Calculate other derived metrics ---
        person_metrics['days_since_last_interaction'] = (now_utc - person_metrics['last_interaction_timestamp']).dt.days.fillna(9999).astype(int)
        person_metrics['days_since_first_interaction'] = (now_utc - person_metrics['first_interaction_timestamp']).dt.days.fillna(9999).astype(int)

        most_freq_channel = events_df.groupby('person_id_hex')['channel'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').rename('most_frequent_channel')
        person_metrics = person_metrics.join(most_freq_channel, how='left')
        person_metrics['most_frequent_channel'] = person_metrics['most_frequent_channel'].fillna('Unknown')

        # Calculate average purchase value (using the cleaned events_df)
        purchase_events = events_df[events_df['interaction_type'] == 'Shopify Order Created'].copy()
        if not purchase_events.empty and 'event_converted_price' in purchase_events.columns:
             purchase_events['event_converted_price'] = pd.to_numeric(purchase_events['event_converted_price'], errors='coerce')
             purchase_events.dropna(subset=['event_converted_price'], inplace=True)
             if not purchase_events.empty:
                 purchase_summary = purchase_events.groupby('person_id_hex')['event_converted_price'].agg(['sum', 'count'])
                 purchase_summary['avg_purchase_value'] = (purchase_summary['sum'] / purchase_summary['count'].replace(0, 1)).fillna(0)
                 person_metrics = person_metrics.join(purchase_summary[['avg_purchase_value']], how='left')
                 person_metrics['avg_purchase_value'] = person_metrics['avg_purchase_value'].fillna(0)
             else:
                person_metrics['avg_purchase_value'] = 0.0
        else:
             person_metrics['avg_purchase_value'] = 0.0

        # Drop intermediate timestamp columns
        person_metrics = person_metrics.drop(columns=['last_interaction_timestamp', 'first_interaction_timestamp'], errors='ignore')

        print(f"\nCalculated metrics for {len(person_metrics)} persons (final).")
        print("Sample metrics (FINAL):")
        print(person_metrics.head())
        print("\nMetrics columns:", person_metrics.columns.tolist())
        print("\nFinal Metrics DataFrame Info:")
        print(person_metrics.info())

        return person_metrics

    except FileNotFoundError:
        print(f"Error: File not found at {event_interactions_path}")
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))
    except Exception as e:
        print(f"Error calculating person event metrics: {e}")
        traceback.print_exc()
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))