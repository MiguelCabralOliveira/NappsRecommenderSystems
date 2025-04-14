# preprocessing/product_processor/isGiftCard_processor.py
import pandas as pd

def process_gift_card(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out products that are gift cards and drops the column.

    Args:
        df: Input DataFrame with 'is_gift_card' column.

    Returns:
        DataFrame containing only non-gift card products, without the column.
    """
    print("Processing: is_gift_card (Filtering)")
    if 'is_gift_card' not in df.columns:
        print("Warning: 'is_gift_card' column not found. Skipping.")
        return df

    original_count = len(df)
    # Ensure boolean comparison works correctly
    if pd.api.types.is_object_dtype(df['is_gift_card']) or pd.api.types.is_bool_dtype(df['is_gift_card']):
        if pd.api.types.is_object_dtype(df['is_gift_card']):
            df['is_gift_card'] = df['is_gift_card'].astype(str).str.lower() == 'true'
        df_filtered = df[df['is_gift_card'] == False].copy() # Keep non-gift cards
    else:
        print("Warning: 'is_gift_card' column has unexpected type. Attempting conversion.")
        try:
            df_filtered = df[df['is_gift_card'].astype(bool) == False].copy()
        except Exception as e:
            print(f"Error converting 'is_gift_card' to boolean: {e}. Skipping filter.")
            return df


    df_filtered = df_filtered.drop(columns=['is_gift_card'])
    removed_count = original_count - len(df_filtered)
    print(f"Removed {removed_count} gift card products.")
    return df_filtered