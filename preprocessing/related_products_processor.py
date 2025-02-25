import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler

def process_related_products(df):
    """
    Process related_products field into count-based features
    Args:
        df: pandas DataFrame containing product data with related_products column
    Returns:
        DataFrame: processed DataFrame with related product count features
    """
    # Extract count of related products
    def count_related(related_array):
        if pd.isna(related_array):
            return 0
        
        if isinstance(related_array, str):
            try:
                related_ids = ast.literal_eval(related_array)
                return len(related_ids)
            except:
                return 0
        else:
            return len(related_array) if related_array else 0
    
    # Add count feature
    df['related_products_count'] = df['related_products'].apply(count_related)
    
    # Create a map of how many times each product is recommended
    recommendation_counts = {}
    for _, row in df.iterrows():
        product_id = row['product_id']
        related_array = row['related_products']
        
        if pd.notna(related_array):
            if isinstance(related_array, str):
                try:
                    related_ids = ast.literal_eval(related_array)
                except:
                    related_ids = []
            else:
                related_ids = related_array if related_array else []
                
            for rel_id in related_ids:
                recommendation_counts[rel_id] = recommendation_counts.get(rel_id, 0) + 1
    
    # Add feature for how many times a product is recommended by others
    df['recommended_by_others_count'] = df['product_id'].map(
        lambda x: recommendation_counts.get(x, 0)
    )
    
    # Normalize the counts
    scaler = MinMaxScaler()
    count_features = ['related_products_count', 'recommended_by_others_count']
    
    # Handle NaN values
    df[count_features] = df[count_features].fillna(0)
    
    # Only normalize if we have non-zero data
    if df[count_features].sum().sum() > 0:
        df[count_features] = scaler.fit_transform(df[count_features])
    
    # Drop the original related_products column
    result_df = df.drop('related_products', axis=1)
    
    return result_df


if __name__ == "__main__":
    # Debug/testing mode
    input_df = pd.read_csv("../products.csv")
    df_processed = process_related_products(input_df)
    
    # Check the results
    print("\nProcessed related products data:")
    if 'related_products_count' in df_processed.columns:
        print(f"Related products count stats:\n{df_processed['related_products_count'].describe()}")
    if 'recommended_by_others_count' in df_processed.columns:
        print(f"Recommended by others count stats:\n{df_processed['recommended_by_others_count'].describe()}")
    
    print(f"\nVerifying that related_products column was dropped: {'related_products' not in df_processed.columns}")