# handle_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

def clean_handle(handle):
    """Clean handle and prepare for TF-IDF processing"""
    if pd.isna(handle):
        return ""
    
    # Convert to lowercase
    text = str(handle).lower()
    
    # Replace hyphens and underscores with spaces to treat words separately
    text = text.replace('-', ' ').replace('_', ' ')
    
    # Remove any remaining special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_handles_tfidf(df, output_prefix="products"):
    """
    Process product handles with TF-IDF and save results
    
    Returns:
        tuple: (final_df, tfidf) where:
            - final_df: DataFrame with TF-IDF features
            - tfidf: The fitted TF-IDF vectorizer
    """
    # Store all non-handle columns to preserve them later
    preserve_cols = [col for col in df.columns if col != 'handle']
    
    # Clean handles
    df['clean_handle'] = df['handle'].apply(clean_handle)
    
    # Filter out empty handles
    mask_valid_handles = df['clean_handle'].str.len() > 0
    valid_handles = df.loc[mask_valid_handles, 'clean_handle']
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=30,  # Fewer features than titles, as handles are typically shorter
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(valid_handles)
    
    # Get feature names with handle_ prefix
    feature_names = [f"handle_{name}" for name in tfidf.get_feature_names_out()]
    
    # Save artifacts
    np.save(f"{output_prefix}_handle_tfidf_matrix.npy", tfidf_matrix.toarray())
    pd.DataFrame(feature_names).to_csv(
        f"{output_prefix}_handle_tfidf_features.csv", index=False, header=False)
    
    # Add TF-IDF vectors to original DataFrame with prefixed column names
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=valid_handles.index)
    
    final_df = df.copy()
    final_df = final_df.join(tfidf_df)
    
    # Drop intermediary processing columns
    final_df = final_df.drop(['clean_handle'], axis=1)
    
    # Save the processed DataFrame
    if output_prefix:
        final_df.to_csv(f"{output_prefix}_with_handle_tfidf.csv", index=False)
    
    return final_df, tfidf

if __name__ == "__main__":
    df, vectorizer = process_handles_tfidf()
    print("Handle TF-IDF processing complete!")