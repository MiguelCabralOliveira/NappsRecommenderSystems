import pandas as pd
import ast

def process_tags(df):
    """
    Process tags using dummy variables
    Args:
        df: pandas DataFrame containing product data
    Returns:
        DataFrame: processed DataFrame with tag dummies
    """
    # Tags are already in list format, just need to explode them
    tags_series = df['tags'].explode()
    
    # Create dummy variables for tags
    tag_dummies = pd.get_dummies(
        tags_series,
        prefix='tag',
        prefix_sep='_',
        dtype=int
    )
    
    # Aggregate back to product level
    tag_dummies = tag_dummies.groupby(level=0).max()
    
    # Add dummy columns to original dataframe
    df_with_tags = pd.concat([df, tag_dummies], axis=1)
    
    # Drop the original tags column
    df_with_tags = df_with_tags.drop('tags', axis=1)
    
    return df_with_tags


if __name__ == "__main__":
    # Debug/testing mode
    input_df = pd.read_csv("c:/Users/Administrator/NappsRecommenderSystem/products.csv")
    df_processed = process_tags(input_df)
    
    tag_columns = [col for col in df_processed.columns if col.startswith('tag_')]
    print(f"Created {len(tag_columns)} tag features:")
    for col in tag_columns:
        print(f"- {col}")