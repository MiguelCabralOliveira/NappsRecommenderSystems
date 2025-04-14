# preprocessing/product_processor/product_type_processor.py
import pandas as pd
import traceback

def process_product_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process product_type field using dummy variables.

    Args:
        df: pandas DataFrame containing product data with product_type column.

    Returns:
        DataFrame: processed DataFrame with product_type dummies.
    """
    print("Processing: product_type (Creating Dummies)")
    if 'product_type' not in df.columns:
        print("Warning: 'product_type' column not found. Skipping.")
        return df

    df_processed = df.copy()

    try:
        # Fill NaN values with 'Unknown' before creating dummies
        df_processed['product_type'] = df_processed['product_type'].fillna('Unknown').astype(str)
        # Optional: Treat empty strings as 'Unknown' too
        df_processed['product_type'] = df_processed['product_type'].replace('', 'Unknown')


        if df_processed['product_type'].nunique() == 1 and df_processed['product_type'].iloc[0] == 'Unknown':
             print("Only 'Unknown' product type found. Skipping dummy creation.")
             df_processed = df_processed.drop('product_type', axis=1, errors='ignore')
             return df_processed

        # Create dummy variables
        product_type_dummies = pd.get_dummies(
            df_processed['product_type'],
            prefix='prodtype', # Shorter prefix
            prefix_sep='_',
            dtype=int
        )

        # Drop the original product_type column
        df_processed = df_processed.drop('product_type', axis=1, errors='ignore')

        # Concatenate the dummy columns
        result_df = pd.concat([df_processed, product_type_dummies], axis=1)

        print(f"Created {len(product_type_dummies.columns)} product type dummy features.")
        return result_df

    except Exception as e:
        print(f"Error processing product_type: {e}")
        traceback.print_exc()
        # Return original df without product_type column in case of error
        return df.drop('product_type', axis=1, errors='ignore')