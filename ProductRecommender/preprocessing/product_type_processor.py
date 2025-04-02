import pandas as pd


def process_product_type(df):
    """
    Process product_type field using dummy variables
    Args:
        df: pandas DataFrame containing product data with product_type column
    Returns:
        DataFrame: processed DataFrame with product_type dummies
    """
    # Make sure we're working with a copy to avoid modifying the original
    df = df.copy()
    
    # If product_type column doesn't exist, return the DataFrame unchanged
    if 'product_type' not in df.columns:
        return df
    
    # Create dummy variables for product types
    product_type_dummies = pd.get_dummies(
        df['product_type'],
        prefix='product_type',
        prefix_sep='_',
        dtype=int
    )
    
    # Drop the product_type column from our working copy
    df = df.drop('product_type', axis=1)
    
    # Add the product_type dummy columns
    result_df = pd.concat([df, product_type_dummies], axis=1)
    
    return result_df


if __name__ == "__main__":
    # Debug/testing mode
    input_df = pd.read_csv("../products.csv")
    df_processed = process_product_type(input_df)
    
    product_type_columns = [col for col in df_processed.columns if col.startswith('product_type_')]
    print(f"Created {len(product_type_columns)} product type features:")
    for col in product_type_columns:
        print(f"- {col}")