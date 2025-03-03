import pandas as pd


def process_collections(df):
    """
    Process collection titles using dummy variables
    Args:
        df: pandas DataFrame containing product data
    Returns:
        DataFrame: processed DataFrame with collection dummies
    """
    # Make sure we're working with a copy to avoid modifying the original
    df = df.copy()
    
    # If collections column doesn't exist, return the DataFrame unchanged
    if 'collections' not in df.columns:
        return df
    
    # Extract collection titles from dictionaries
    def get_collection_title(collection_dict):
        if isinstance(collection_dict, dict):
            return collection_dict.get('title', '')
        return collection_dict

    # First, explode the collections column and extract titles
    collections_series = df['collections'].explode().apply(get_collection_title)
    
    # Remove empty strings and None values
    collections_series = collections_series[collections_series.notna() & (collections_series != '')]
    
    # Create dummy variables for collections
    collection_dummies = pd.get_dummies(
        collections_series,
        prefix='collections',
        prefix_sep='_',
        dtype=int
    )
    
    # Aggregate back to product level using maximum value
    collection_dummies = collection_dummies.groupby(level=0).max()
    
    # Drop the collections column from our working copy
    df = df.drop('collections', axis=1)
    
    # Add the collection dummy columns
    # This preserves the filtering and column changes from previous processors
    # because we're using the filtered and modified df as our base
    result_df = pd.concat([df, collection_dummies], axis=1)
    
    return result_df