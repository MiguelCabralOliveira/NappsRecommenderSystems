# preprocessing/preprocess_person_data.py
import argparse
import os
import pandas as pd
import traceback
import sys

# Use relative imports for the processing functions within the person_processor sub-package
try:
    from .person_processor.process_direct_features import process_direct_person_columns
    from .person_processor.process_properties_features import process_person_properties
    # Import the new function
    from .person_processor.calculate_event_metrics import calculate_person_event_metrics
except ImportError:
    print("Warning: Could not use relative imports. Attempting direct import.")
    try:
        from person_processor.process_direct_features import process_direct_person_columns
        from person_processor.process_properties_features import process_person_properties
        from person_processor.calculate_event_metrics import calculate_person_event_metrics
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import person_processor modules.")
         sys.exit(1)


def get_final_person_columns():
    """Returns the list of expected final columns for person data, including event metrics."""
    # Define the final column order and selection
    return [
        'person_id_hex',
        # Direct Features
        'is_identified',
        'days_since_creation',
        'days_since_last_update',
        # Properties Features (Selected)
        'shopify_currency',
        'shopify_account_age_days',
        'country_iso',
        'in_eu',
        'timezone',
        # Event Metrics (Added)
        'total_interactions',
        'unique_products_interacted',
        'view_count',
        'purchase_count',
        'wishlist_count',
        'share_count',
        'checkout_start_count',
        'distinct_channels_count',
        'variant_meta_added_count',
        'checkout_update_count',
        'order_update_count',
        'is_buyer',
        'days_since_last_interaction',
        'days_since_first_interaction',
        'most_frequent_channel',
        'avg_purchase_value'
    ]

# Modified function signature to accept event_input_path
def run_person_preprocessing(person_input_path: str, event_input_path: str, output_path: str):
    """
    Orchestrates the preprocessing of person data: loads raw data, processes features,
    calculates event metrics, merges them, and saves the final output.

    Args:
        person_input_path: Path to the raw person data CSV (e.g., persons_gql.csv).
        event_input_path: Path to the processed event interactions CSV
                           (e.g., event_interactions_processed.csv).
        output_path: Path to save the final processed person features CSV
                     (e.g., persons_processed.csv).
    """
    print(f"\n--- Starting Person Data Preprocessing Orchestration ---")
    print(f"Person Input file: {person_input_path}")
    print(f"Event Input file (for metrics): {event_input_path}")
    print(f"Output file: {output_path}")

    # --- Load Raw Person Data ---
    try:
        if not os.path.exists(person_input_path):
             raise FileNotFoundError(f"Person input file not found: {person_input_path}")
        person_df_raw = pd.read_csv(person_input_path, low_memory=False)
        print(f"Loaded {len(person_df_raw)} raw person rows from {person_input_path}.")

        if person_df_raw.empty:
            print("Input person DataFrame is empty. Creating empty output file.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(columns=get_final_person_columns()).to_csv(output_path, index=False)
            print(f"Empty output file created at: {output_path}")
            return
        if 'person_id_hex' not in person_df_raw.columns:
            raise ValueError("Input person data must contain 'person_id_hex' column.")
        required_processing_cols = ['is_identified', 'created_at', 'updated_at', 'properties']
        missing_essential = [col for col in required_processing_cols if col not in person_df_raw.columns]
        if missing_essential:
             print(f"Warning: Input data missing expected columns for processing: {missing_essential}. Adding defaults.")
             for col in missing_essential: person_df_raw[col] = None

    except FileNotFoundError:
        print(f"Error: Input file not found at {person_input_path}")
        raise
    except Exception as e:
        print(f"Error loading or validating input person CSV '{person_input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Process Direct and Properties Features ---
    try:
        print("\nProcessing direct columns...")
        direct_cols_req = ['person_id_hex', 'is_identified', 'created_at', 'updated_at']
        if all(col in person_df_raw.columns for col in direct_cols_req):
            df_direct = process_direct_person_columns(person_df_raw.copy())
            print(f"Direct features processed, shape: {df_direct.shape}")
        else:
            print("Skipping direct columns processing due to missing required input columns.")
            df_direct = person_df_raw[['person_id_hex']].copy()
            df_direct['is_identified'] = 0
            df_direct['days_since_creation'] = 0
            df_direct['days_since_last_update'] = 0

        print("\nProcessing properties columns...")
        props_cols_req = ['person_id_hex', 'properties', 'is_identified']
        if all(col in person_df_raw.columns for col in props_cols_req):
            df_props = process_person_properties(person_df_raw.copy())
            print(f"Properties features processed, shape: {df_props.shape}")
        else:
            print("Skipping properties columns processing due to missing required input columns.")
            df_props = person_df_raw[['person_id_hex']].copy()
            prop_defaults = { 'shopify_state': 'Unknown', 'shopify_currency': 'Unknown', 'shopify_verified_email': 0, 'shopify_account_age_days': 0, 'city': 'Unknown', 'country_iso': 'Unknown', 'in_eu': 0, 'timezone': 'Unknown', 'days_since_last_session': 9999 }
            for col, default in prop_defaults.items(): df_props[col] = default

    except Exception as e:
        print(f"Error during direct/properties feature processing step: {e}")
        traceback.print_exc()
        raise

    # --- Merge Direct and Properties Features ---
    print("\n--- Merging Direct and Properties Features ---")
    try:
        if 'person_id_hex' not in df_direct.columns or 'person_id_hex' not in df_props.columns:
             raise ValueError("Critical error: 'person_id_hex' missing before merge.")
        persons_features_df = pd.merge(df_direct, df_props, on='person_id_hex', how='left')
        print(f"Merged direct/properties DataFrame shape: {persons_features_df.shape}")
        if persons_features_df.columns.duplicated().any():
            dups = persons_features_df.columns[persons_features_df.columns.duplicated()].tolist()
            print(f"Warning: Duplicate columns found after merge: {dups}. Keeping first instance.")
            persons_features_df = persons_features_df.loc[:,~persons_features_df.columns.duplicated()]
    except Exception as e:
        print(f"Error during merging direct/properties features: {e}")
        traceback.print_exc()
        raise

    # --- Calculate and Merge Event Metrics --- ## CORRECTED MERGE LOGIC ##
    print("\n--- Calculating and Merging Event Metrics ---")
    try:
        person_event_metrics_df = calculate_person_event_metrics(event_input_path)

        if not person_event_metrics_df.empty:
            print(f"Merging {len(person_event_metrics_df)} rows of event metrics...")

            # Ensure the metrics df has person_id_hex as index (should be true from groupby)
            if person_event_metrics_df.index.name != 'person_id_hex':
                 print(f"Warning: Event metrics index name is '{person_event_metrics_df.index.name}', expected 'person_id_hex'. Attempting to reset/set index.")
                 if 'person_id_hex' in person_event_metrics_df.columns:
                     try:
                         person_event_metrics_df = person_event_metrics_df.set_index('person_id_hex')
                         print("Successfully set 'person_id_hex' as index for metrics.")
                     except Exception as idx_err:
                         print(f"Error setting index: {idx_err}. Cannot merge metrics.")
                         person_event_metrics_df = pd.DataFrame() # Prevent merge attempt
                 else:
                     print("Error: 'person_id_hex' not found as index or column in metrics. Cannot merge.")
                     person_event_metrics_df = pd.DataFrame() # Prevent merge attempt

            if not person_event_metrics_df.empty:
                 # Use left_on (column name) and right_index=True
                 persons_features_df = persons_features_df.merge(
                     person_event_metrics_df,
                     left_on='person_id_hex', # Merge based on this column in the left df
                     right_index=True,       # Merge based on the index in the right df
                     how='left'              # Keep all persons, add metrics where available
                 )

                 print(f"Merged event metrics. New shape: {persons_features_df.shape}")

                 # Fill NaN values for metrics for persons who had no events
                 fill_values = {
                     'total_interactions': 0, 'unique_products_interacted': 0, 'view_count': 0,
                     'purchase_count': 0, 'wishlist_count': 0, 'share_count': 0,
                     'checkout_start_count': 0, 'distinct_channels_count': 0,
                     'variant_meta_added_count': 0, 'checkout_update_count': 0, 'order_update_count': 0,
                     'is_buyer': 0, 'days_since_last_interaction': 9999,
                     'days_since_first_interaction': 9999, 'most_frequent_channel': 'Unknown',
                     'avg_purchase_value': 0.0
                 }
                 cols_to_fill = {col: val for col, val in fill_values.items() if col in persons_features_df.columns}
                 persons_features_df.fillna(value=cols_to_fill, inplace=True)
                 print("Filled NA values in event metrics columns with defaults.")

                 # Ensure correct integer/float types for metric columns after fillna
                 metric_cols = list(fill_values.keys())
                 for col in metric_cols:
                     if col in persons_features_df.columns:
                         if col == 'most_frequent_channel':
                             persons_features_df[col] = persons_features_df[col].astype(str)
                         elif col == 'avg_purchase_value':
                             persons_features_df[col] = pd.to_numeric(persons_features_df[col], errors='coerce').fillna(0.0).astype(float)
                         else: # Counts and days
                             persons_features_df[col] = pd.to_numeric(persons_features_df[col], errors='coerce').fillna(fill_values.get(col, 0)).astype('Int64')
        else:
            print("No event metrics calculated/returned. Adding default metric columns.")
            # Add default columns if merge didn't happen
            fill_values = { 'total_interactions': 0, 'unique_products_interacted': 0, 'view_count': 0, 'purchase_count': 0, 'wishlist_count': 0, 'share_count': 0, 'checkout_start_count': 0, 'distinct_channels_count': 0, 'variant_meta_added_count': 0, 'checkout_update_count': 0, 'order_update_count': 0, 'is_buyer': 0, 'days_since_last_interaction': 9999, 'days_since_first_interaction': 9999, 'most_frequent_channel': 'Unknown', 'avg_purchase_value': 0.0 }
            for col, default_val in fill_values.items():
                if col not in persons_features_df.columns:
                    persons_features_df[col] = default_val
                    if col == 'most_frequent_channel': persons_features_df[col] = persons_features_df[col].astype(str)
                    elif col == 'avg_purchase_value': persons_features_df[col] = persons_features_df[col].astype(float)
                    else: persons_features_df[col] = persons_features_df[col].astype('Int64')

    except Exception as e:
        print(f"Error calculating or merging person event metrics: {e}")
        traceback.print_exc()
        print("Continuing without event metrics due to error.")
        # Ensure default columns exist if error occurred
        fill_values = { 'total_interactions': 0, 'unique_products_interacted': 0, 'view_count': 0, 'purchase_count': 0, 'wishlist_count': 0, 'share_count': 0, 'checkout_start_count': 0, 'distinct_channels_count': 0, 'variant_meta_added_count': 0, 'checkout_update_count': 0, 'order_update_count': 0, 'is_buyer': 0, 'days_since_last_interaction': 9999, 'days_since_first_interaction': 9999, 'most_frequent_channel': 'Unknown', 'avg_purchase_value': 0.0 }
        for col, default_val in fill_values.items():
            if col not in persons_features_df.columns:
                 persons_features_df[col] = default_val
                 if col == 'most_frequent_channel': persons_features_df[col] = persons_features_df[col].astype(str)
                 elif col == 'avg_purchase_value': persons_features_df[col] = persons_features_df[col].astype(float)
                 else: persons_features_df[col] = persons_features_df[col].astype('Int64')


    # --- Finalize Columns and Save ---
    FINAL_COLUMNS = get_final_person_columns() # Get the updated list including metrics
    print("\nFinalizing columns...")
    final_df = persons_features_df # Use the df that now includes metrics

    # Check if all desired final columns are present, add defaults if missing (safeguard)
    available_cols_before_select = final_df.columns.tolist()
    missing_final_cols = []
    for col in FINAL_COLUMNS:
        if col not in available_cols_before_select:
            missing_final_cols.append(col)
            fill_values = { 'total_interactions': 0, 'unique_products_interacted': 0, 'view_count': 0, 'purchase_count': 0, 'wishlist_count': 0, 'share_count': 0, 'checkout_start_count': 0, 'distinct_channels_count': 0, 'variant_meta_added_count': 0, 'checkout_update_count': 0, 'order_update_count': 0, 'is_buyer': 0, 'days_since_last_interaction': 9999, 'days_since_first_interaction': 9999, 'most_frequent_channel': 'Unknown', 'avg_purchase_value': 0.0, 'is_identified': 0, 'days_since_creation': 0, 'days_since_last_update': 0, 'shopify_currency': 'Unknown', 'shopify_account_age_days': 0, 'country_iso': 'Unknown', 'in_eu': 0, 'timezone': 'Unknown'}
            default_val = fill_values.get(col, None)
            final_df[col] = default_val
            print(f"Warning: Expected final column '{col}' was missing. Added with default: {default_val}.")
            if isinstance(default_val, str): final_df[col] = final_df[col].astype(str)
            elif isinstance(default_val, float): final_df[col] = final_df[col].astype(float)
            elif isinstance(default_val, int): final_df[col] = final_df[col].astype('Int64')

    if missing_final_cols:
        print(f"Columns added with defaults: {missing_final_cols}")

    # Select and reorder columns *only* to the desired FINAL_COLUMNS
    try:
        final_df = final_df[FINAL_COLUMNS]
        print(f"Selected final columns defined by get_final_person_columns(). Shape: {final_df.shape}")
    except KeyError as e:
        available_cols = final_df.columns.tolist()
        required_but_missing = [c for c in FINAL_COLUMNS if c not in available_cols]
        print(f"CRITICAL Error selecting final columns: {e}. Required but missing: {required_but_missing}.")
        print(f"Available columns before selection attempt: {available_cols}")
        raise ValueError(f"Could not select all required final columns. Missing: {required_but_missing}") from e

    # Final type check (optional but good practice)
    print("\nFinal Data Types:")
    print(final_df.info())

    # --- Save Output ---
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nEnsured output directory exists: {output_dir}")
        final_df.to_csv(output_path, index=False, na_rep='') # Save NA as empty string
        print(f"Successfully saved processed person data to: {output_path}")
        print(f"--- Finished Person Data Preprocessing Orchestration ---")
    except Exception as e:
        print(f"Error saving output CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

# --- Command Line Execution Logic (for standalone testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess person data. Merges direct, properties, and event-based features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--person-input', required=True, help='Path to the raw person data CSV (e.g., shop_persons_gql.csv).')
    parser.add_argument('--event-input', required=True, help='Path to the processed event interactions CSV (e.g., shop_event_interactions_processed.csv).')
    parser.add_argument('--output', required=True, help='Path to save the final processed person features CSV (e.g., shop_persons_processed.csv).')
    args = parser.parse_args()
    try:
        run_person_preprocessing(args.person_input, args.event_input, args.output)
        print("\nPerson preprocessing finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error:
         print(f"\nError: {val_error}")
         exit(1)
    except Exception as e:
        print(f"\n--- Person preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)