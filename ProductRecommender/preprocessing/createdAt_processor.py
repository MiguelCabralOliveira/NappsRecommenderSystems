import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def process_created_at(df):
    """
    Process createdAt timestamps for products
    
    Args:
        df: pandas DataFrame containing product data with createdAt column
        
    Returns:
        DataFrame: processed DataFrame with normalized timestamp features
    """
    
    df = df.copy()
    
    # Check if createdAt column exists
    if 'createdAt' not in df.columns:
        print("Warning: 'createdAt' column not found, skipping timestamp processing")
        return df
    
    print("Processing createdAt timestamps...")
    
    # Parse timestamps to datetime objects
    df['createdAt_datetime'] = pd.to_datetime(df['createdAt'], utc=True)
    
    # Extract various time components
    df['created_year'] = df['createdAt_datetime'].dt.year
    df['created_month'] = df['createdAt_datetime'].dt.month
    df['created_day'] = df['createdAt_datetime'].dt.day
    df['created_hour'] = df['createdAt_datetime'].dt.hour
    
    # Calculate days since the earliest product
    min_date = df['createdAt_datetime'].min()
    df['days_since_first_product'] = (df['createdAt_datetime'] - min_date).dt.total_seconds() / (24 * 3600)
    
    # Create seasons (1=Spring, 2=Summer, 3=Fall, 4=Winter)
    df['created_season'] = df['created_month'].apply(
        lambda month: 1 if month in [3, 4, 5] else 
                      2 if month in [6, 7, 8] else 
                      3 if month in [9, 10, 11] else 4
    )
    
    # Convert to numeric columns for modeling
    numeric_date_features = [
        'created_year', 'created_month', 'created_day',
        'created_hour', 'days_since_first_product', 'created_season'
    ]
    
    # Normalize the numeric features
    scaler = MinMaxScaler()
    df[numeric_date_features] = scaler.fit_transform(df[numeric_date_features])
    
    # Create additional features
    
    # Is product recent (released in last 3 months from the latest product)
    latest_date = df['createdAt_datetime'].max()
    three_months_ago = latest_date - pd.Timedelta(days=90)
    df['is_recent_product'] = df['createdAt_datetime'].ge(three_months_ago).astype(int)
    
    # Group products by release periods (yearly quarters)
    df['release_quarter'] = df['createdAt_datetime'].dt.to_period('Q').astype(str)
    release_quarter_dummies = pd.get_dummies(
        df['release_quarter'], 
        prefix='release_quarter', 
        prefix_sep='_',
        dtype=int
    )
    df = pd.concat([df, release_quarter_dummies], axis=1)
    
    # Drop intermediate columns
    df = df.drop(['createdAt_datetime', 'release_quarter', 'createdAt'], axis=1)
    
    print(f"Created {len(numeric_date_features)} normalized time features")
    print(f"Created {len(release_quarter_dummies.columns)} release quarter dummy variables")
    
    return df

if __name__ == "__main__":
    # Test the processor with a sample dataframe
    sample_data = pd.DataFrame({
        'product_id': range(1, 6),
        'createdAt': [
            '2022-01-04T12:46:23Z',
            '2022-01-04T16:16:27Z',
            '2022-05-10T10:30:00Z',
            '2023-02-15T08:45:12Z',
            '2023-08-22T14:20:45Z'
        ]
    })
    
    processed_df = process_created_at(sample_data)
    print("\nSample of processed data:")
    print(processed_df.head())
    
    time_features = [col for col in processed_df.columns if 'created_' in col or 'release_' in col or 'days_since' in col or 'is_recent' in col]
    print(f"\nCreated time features: {time_features}")