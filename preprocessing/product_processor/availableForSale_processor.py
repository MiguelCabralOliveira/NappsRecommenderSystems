# preprocessing/product_processor/availableForSale_processor.py
import pandas as pd

def process_avaiable_for_sale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters products that are available for sale and drops the column.

    Args:
        df: Input DataFrame with 'available_for_sale' column.

    Returns:
        DataFrame containing only products available for sale, without the column.
    """
    print("Processing: available_for_sale (Filtering)")
    if 'available_for_sale' not in df.columns:
        print("Warning: 'available_for_sale' column not found. Skipping.")
        return df

    original_count = len(df)
    # Ensure boolean comparison works correctly, handle potential string values
    if pd.api.types.is_object_dtype(df['available_for_sale']) or pd.api.types.is_bool_dtype(df['available_for_sale']):
         # Convert potential string 'True'/'False' to boolean if needed, or handle actual booleans
         if pd.api.types.is_object_dtype(df['available_for_sale']):
             df['available_for_sale'] = df['available_for_sale'].astype(str).str.lower() == 'true'
         df_filtered = df[df['available_for_sale'] == True].copy()
    else:
         print("Warning: 'available_for_sale' column has unexpected type. Attempting conversion.")
         try:
             df_filtered = df[df['available_for_sale'].astype(bool) == True].copy()
         except Exception as e:
             print(f"Error converting 'available_for_sale' to boolean: {e}. Skipping filter.")
             return df

    df_filtered = df_filtered.drop(columns=['available_for_sale'])
    removed_count = original_count - len(df_filtered)
    print(f"Removed {removed_count} products not available for sale.")
    return df_filtered