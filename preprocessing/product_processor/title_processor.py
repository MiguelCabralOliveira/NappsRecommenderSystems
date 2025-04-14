# preprocessing/product_processor/title_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import traceback
import joblib # Using joblib for saving the vectorizer model
import os

def clean_title(title):
    """Clean title text and normalize"""
    if pd.isna(title):
        return ""
    text = str(title).lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation, keep spaces
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def process_titles_tfidf(df: pd.DataFrame, output_dir: str, shop_id: str, max_features=50) -> pd.DataFrame:
    """
    Process product titles with TF-IDF and save the vectorizer.

    Args:
        df: DataFrame containing product data with 'product_title' column.
        output_dir: Base directory to save the TF-IDF model.
        shop_id: The shop ID, used for naming the saved model file.
        max_features: Maximum number of TF-IDF features to generate.

    Returns:
        DataFrame: processed DataFrame with title TF-IDF features added.
                 Returns the original DataFrame (minus 'product_title') if processing fails.
    """
    print(f"Processing: product_title (TF-IDF with max_features={max_features})")
    title_col = 'product_title' # Or just 'title' depending on input
    if title_col not in df.columns:
        print(f"Warning: '{title_col}' column not found. Skipping.")
        return df

    df_processed = df.copy()
    model_filename = f"{shop_id}_title_tfidf_vectorizer.joblib"
    model_path = os.path.join(output_dir, model_filename)


    try:
        # Clean product titles
        df_processed['clean_title'] = df_processed[title_col].apply(clean_title)

        # Filter out rows where clean_title might be empty
        mask_valid_titles = df_processed['clean_title'].str.len() > 0
        if not mask_valid_titles.any():
            print("No valid titles found after cleaning. Skipping TF-IDF.")
            return df_processed.drop(columns=[title_col, 'clean_title'], errors='ignore')

        valid_titles_series = df_processed.loc[mask_valid_titles, 'clean_title']

        # Create and fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2) # Single words and bigrams
        )
        tfidf_matrix = tfidf.fit_transform(valid_titles_series)

        # Save the fitted vectorizer
        os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
        joblib.dump(tfidf, model_path)
        print(f"Saved title TF-IDF vectorizer to: {model_path}")


        # Get feature names with title_tfidf_ prefix
        feature_names = [f"title_tfidf_{name}" for name in tfidf.get_feature_names_out()]

        # Create DataFrame with TF-IDF features, using the index from valid titles
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=valid_titles_series.index # Align with original DF index
        )

        # Join the TF-IDF features back to the main DataFrame
        df_processed = df_processed.join(tfidf_df)

        # Fill NaNs introduced by the join (for rows that had no valid title) with 0
        df_processed[feature_names] = df_processed[feature_names].fillna(0)

        print(f"Created {len(feature_names)} title TF-IDF features.")

    except Exception as e:
        print(f"Error processing titles with TF-IDF: {e}")
        traceback.print_exc()
        # Return the DataFrame without title features in case of error
        return df_processed.drop(columns=[title_col, 'clean_title'], errors='ignore')

    # Drop original title and intermediate columns
    df_processed = df_processed.drop(columns=[title_col, 'clean_title'], errors='ignore')

    return df_processed