# preprocessing/product_processor/handle_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import traceback
import os
import joblib # Using joblib for saving the vectorizer model

def clean_handle(handle):
    """Clean handle and prepare for TF-IDF processing"""
    if pd.isna(handle):
        return ""
    text = str(handle).lower()
    text = text.replace('-', ' ').replace('_', ' ') # Treat parts separated by hyphen/underscore as distinct words
    text = re.sub(r'[^\w\s]', '', text) # Remove non-alphanumeric (keep spaces)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def process_handles_tfidf(df: pd.DataFrame, output_dir: str, shop_id: str, max_features=30) -> pd.DataFrame:
    """
    Process product handles with TF-IDF and save the vectorizer.

    Args:
        df: DataFrame containing product data with 'handle' column.
        output_dir: Base directory to save the TF-IDF model.
        shop_id: The shop ID, used for naming the saved model file.
        max_features: Maximum number of TF-IDF features to generate.

    Returns:
        DataFrame: processed DataFrame with handle TF-IDF features added.
                 Returns the original DataFrame (minus 'handle') if processing fails or no handles exist.
    """
    print(f"Processing: handle (TF-IDF with max_features={max_features})")
    if 'handle' not in df.columns:
        print("Warning: 'handle' column not found. Skipping.")
        return df

    df_processed = df.copy()
    model_filename = f"{shop_id}_handle_tfidf_vectorizer.joblib"
    model_path = os.path.join(output_dir, model_filename)

    try:
        # Clean handles
        df_processed['clean_handle'] = df_processed['handle'].apply(clean_handle)

        # Filter out rows where clean_handle might be empty after cleaning
        mask_valid_handles = df_processed['clean_handle'].str.len() > 0
        if not mask_valid_handles.any():
             print("No valid handles found after cleaning. Skipping TF-IDF.")
             return df_processed.drop(columns=['handle', 'clean_handle'], errors='ignore')

        valid_handles_series = df_processed.loc[mask_valid_handles, 'clean_handle']

        # Create and fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2) # Consider single words and bigrams
        )
        tfidf_matrix = tfidf.fit_transform(valid_handles_series)

        # Save the fitted vectorizer
        os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
        joblib.dump(tfidf, model_path)
        print(f"Saved handle TF-IDF vectorizer to: {model_path}")

        # Get feature names with handle_tfidf_ prefix
        feature_names = [f"handle_tfidf_{name}" for name in tfidf.get_feature_names_out()]

        # Create DataFrame with TF-IDF features, using the index from the valid handles
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=valid_handles_series.index # IMPORTANT: Align with original DF
        )

        # Join the TF-IDF features back to the main DataFrame
        # Using join ensures alignment based on index
        df_processed = df_processed.join(tfidf_df)

        # Fill NaNs introduced by the join (for rows that had no valid handle) with 0
        df_processed[feature_names] = df_processed[feature_names].fillna(0)

        print(f"Created {len(feature_names)} handle TF-IDF features.")

    except Exception as e:
        print(f"Error processing handles with TF-IDF: {e}")
        traceback.print_exc()
        # Return the DataFrame without handle features in case of error
        return df_processed.drop(columns=['handle', 'clean_handle'], errors='ignore')

    # Drop original handle and intermediate columns
    df_processed = df_processed.drop(columns=['handle', 'clean_handle'], errors='ignore')

    return df_processed