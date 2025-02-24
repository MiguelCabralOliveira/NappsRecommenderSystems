import pandas as pd


def process_collections(df):
    """
    Process collection titles using dummy variables
    Args:
        df: pandas DataFrame containing product data
    Returns:
        DataFrame: processed DataFrame with collection dummies
    """
    # Create dummy variables for collections
    collection_dummies = pd.get_dummies(
        df['collection_title'], 
        prefix='collection',
        prefix_sep='_',
        dtype=int
    )
    
    # Create a new DataFrame with all original columns except collection_title
    result_df = df.drop('collection_title', axis=1)
    
    # Add the collection dummy columns
    result_df = pd.concat([result_df, collection_dummies], axis=1)
    
    return result_df


if __name__ == "__main__":
    # Debug/testing mode
    input_df = pd.read_csv("c:/Users/Administrator/NappsRecommenderSystem/products.csv")
    df_processed = process_collections(input_df)
    df_processed.to_csv("c:/Users/Administrator/NappsRecommenderSystem/products_with_collections.csv", index=False)
    
    collection_columns = [col for col in df_processed.columns if col.startswith('collection_')]
    print(f"Created {len(collection_columns)} collection features:")
    for col in collection_columns:
        print(f"- {col}")