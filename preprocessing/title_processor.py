# title_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

def clean_title(title):
    """Clean title text and normalize"""
    if pd.isna(title):
        return ""
    
    # Convert to lowercase
    text = str(title).lower()
    
    
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    
    return text

def process_titles_tfidf(df, output_prefix="products"):
    """
    Process product titles with TF-IDF and save results
    
    Returns:
        tuple: (final_df, tfidf, similar_products) where:
            - final_df: DataFrame with TF-IDF features
            - tfidf: The fitted TF-IDF vectorizer
            - similar_products: DataFrame with pairs of similar products
    """
    # Store all non-title columns to preserve them later
    preserve_cols = [col for col in df.columns if col != 'product_title']
    
    # Clean product titles
    df['clean_title'] = df['product_title'].apply(clean_title)
    
    # Filter out empty titles
    mask_valid_titles = df['clean_title'].str.len() > 0
    valid_titles = df.loc[mask_valid_titles, 'clean_title']
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=50,  # Using fewer features than description since titles are shorter
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(valid_titles)
    
    # Get feature names with title_ prefix
    feature_names = [f"title_{name}" for name in tfidf.get_feature_names_out()]
    
    # Save artifacts
    np.save(f"{output_prefix}_title_tfidf_matrix.npy", tfidf_matrix.toarray())
    pd.DataFrame(feature_names).to_csv(
        f"{output_prefix}_title_tfidf_features.csv", index=False, header=False)
    
    # Add TF-IDF vectors to original DataFrame with prefixed column names
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=valid_titles.index)
    
    final_df = df.copy()
    final_df = final_df.join(tfidf_df)
    
    # Drop intermediary processing columns
    final_df = final_df.drop(['clean_title'], axis=1)
    
    # Create a DataFrame for downstream processing if needed
    recommendation_df = final_df.copy()
    
    # Save the recommendation DataFrame
    if output_prefix:
        recommendation_df.to_csv(f"{output_prefix}_title_recommendation.csv", index=False)
        final_df.to_csv(f"{output_prefix}_with_title_tfidf.csv", index=False)
    
    return final_df, tfidf

if __name__ == "__main__":
    df, vectorizer = process_titles_tfidf()
    print("Title TF-IDF processing complete!")