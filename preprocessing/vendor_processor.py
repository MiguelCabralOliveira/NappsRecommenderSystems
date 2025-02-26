import pandas as pd


def process_vendors(df):
    """
    Process vendor field using one-hot encoding (dummy variables)
    Args:
        df: pandas DataFrame containing product data with vendor column
    Returns:
        DataFrame: processed DataFrame with vendor dummies
    """
    # Create dummy variables for vendors
    vendor_dummies = pd.get_dummies(
        df['vendor'],
        prefix='vendor',
        prefix_sep='_',
        dtype=int
    )
    
    # Create a new DataFrame with all original columns except vendor
    result_df = df.drop('vendor', axis=1)
    
    # Add the vendor dummy columns
    result_df = pd.concat([result_df, vendor_dummies], axis=1)
    
    return result_df


if __name__ == "__main__":
    # Debug/testing mode
    input_df = pd.read_csv("../products.csv")
    df_processed = process_vendors(input_df)
    
    vendor_columns = [col for col in df_processed.columns if col.startswith('vendor_')]
    print(f"Created {len(vendor_columns)} vendor features:")
    for col in vendor_columns:
        print(f"- {col}")