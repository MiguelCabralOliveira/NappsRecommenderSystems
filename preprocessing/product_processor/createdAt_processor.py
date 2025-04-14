# preprocessing/product_processor/createdAt_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import traceback

def process_created_at(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process createdAt timestamps for products, creating normalized time features.

    Args:
        df: pandas DataFrame containing product data with createdAt column.

    Returns:
        DataFrame: processed DataFrame with normalized timestamp features.
    """
    print("Processing: createdAt (Creating Time Features)")
    if 'createdAt' not in df.columns:
        print("Warning: 'createdAt' column not found, skipping timestamp processing.")
        return df

    df_processed = df.copy()

    try:
        # Parse timestamps to datetime objects, forcing UTC
        df_processed['createdAt_datetime'] = pd.to_datetime(df_processed['createdAt'], errors='coerce', utc=True)

        # Drop rows where parsing failed
        original_count = len(df_processed)
        df_processed.dropna(subset=['createdAt_datetime'], inplace=True)
        if len(df_processed) < original_count:
            print(f"Warning: Dropped {original_count - len(df_processed)} rows due to invalid 'createdAt' format.")

        if df_processed.empty:
            print("No valid 'createdAt' dates found after parsing. Skipping.")
            # Drop original column and return if empty
            return df_processed.drop(columns=['createdAt'], errors='ignore')

        # Calculate 'now' in UTC for recency calculations
        now = pd.Timestamp.now(tz='UTC')

        # Feature: Days since creation
        df_processed['days_since_creation'] = (now - df_processed['createdAt_datetime']).dt.days

        # Normalize days_since_creation (0 = newest, 1 = oldest)
        # Handle case where all products were created on the same day (max_days = min_days)
        min_days = df_processed['days_since_creation'].min()
        max_days = df_processed['days_since_creation'].max()
        if max_days > min_days:
             df_processed['creation_recency_norm'] = 1 - ((df_processed['days_since_creation'] - min_days) / (max_days - min_days))
        else:
             df_processed['creation_recency_norm'] = 1.0 # All have the same recency

        # Feature: Product Age Category (e.g., New (<90d), Mid (90-365d), Old (>365d))
        df_processed['product_age_category'] = pd.cut(
            df_processed['days_since_creation'],
            bins=[-1, 90, 365, np.inf],
            labels=['New', 'Mid', 'Old'],
            right=True # Intervals are (lower, upper]
        )
        age_dummies = pd.get_dummies(df_processed['product_age_category'], prefix='age', prefix_sep='_', dtype=int)
        df_processed = pd.concat([df_processed, age_dummies], axis=1)


        # Feature: Release Quarter (e.g., 2023-Q1) - Categorical
        df_processed['release_quarter'] = df_processed['createdAt_datetime'].dt.to_period('Q').astype(str)
        quarter_dummies = pd.get_dummies(df_processed['release_quarter'], prefix='quarter', prefix_sep='_', dtype=int)
        df_processed = pd.concat([df_processed, quarter_dummies], axis=1)


        # Feature: Year - could be useful for trends
        # df_processed['created_year'] = df_processed['createdAt_datetime'].dt.year # Keep as categorical or normalize?
        # year_dummies = pd.get_dummies(df_processed['created_year'], prefix='year', prefix_sep='_', dtype=int)
        # df_processed = pd.concat([df_processed, year_dummies], axis=1)


        # Drop intermediate and original columns
        df_processed = df_processed.drop(columns=[
            'createdAt', 'createdAt_datetime', 'days_since_creation',
            'product_age_category', 'release_quarter', #'created_year'
        ], errors='ignore')

        print(f"Created time features: creation_recency_norm, age dummies ({age_dummies.columns.tolist()}), quarter dummies ({quarter_dummies.columns.tolist()})")

    except Exception as e:
        print(f"Error processing 'createdAt': {e}")
        traceback.print_exc()
        # Return original df without createdAt if error occurs
        return df.drop(columns=['createdAt'], errors='ignore')

    return df_processed