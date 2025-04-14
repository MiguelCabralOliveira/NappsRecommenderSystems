# preprocessing/product_processor/vendor_processor.py
import pandas as pd
import traceback

def process_vendors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process vendor field using one-hot encoding (dummy variables).

    Args:
        df: pandas DataFrame containing product data with vendor column.

    Returns:
        DataFrame: processed DataFrame with vendor dummies.
    """
    print("Processing: vendor (Creating Dummies)")
    if 'vendor' not in df.columns:
        print("Warning: 'vendor' column not found. Skipping.")
        return df

    df_processed = df.copy()

    try:
        # Fill NaN values with 'Unknown' before creating dummies
        df_processed['vendor'] = df_processed['vendor'].fillna('Unknown').astype(str)
        # Optional: Treat empty strings as 'Unknown' too
        df_processed['vendor'] = df_processed['vendor'].replace('', 'Unknown')

        if df_processed['vendor'].nunique() == 1 and df_processed['vendor'].iloc[0] == 'Unknown':
             print("Only 'Unknown' vendor found. Skipping dummy creation.")
             df_processed = df_processed.drop('vendor', axis=1, errors='ignore')
             return df_processed

        # Create dummy variables
        vendor_dummies = pd.get_dummies(
            df_processed['vendor'],
            prefix='vendor',
            prefix_sep='_',
            dtype=int
        )

        # Drop the original vendor column
        df_processed = df_processed.drop('vendor', axis=1, errors='ignore')

        # Concatenate the dummy columns
        result_df = pd.concat([df_processed, vendor_dummies], axis=1)

        print(f"Created {len(vendor_dummies.columns)} vendor dummy features.")
        return result_df

    except Exception as e:
        print(f"Error processing vendors: {e}")
        traceback.print_exc()
        # Return original df without vendor column in case of error
        return df.drop('vendor', axis=1, errors='ignore')