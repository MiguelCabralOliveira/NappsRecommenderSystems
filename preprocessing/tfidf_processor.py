# tfidf_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
import numpy as np

def clean_html(html):
    """Clean HTML content and extract meaningful text"""
    if pd.isna(html):
        return ""
    
    # Remove HTML tags
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    
    # Clean special characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'&[a-z]+;', '', text)  # Remove HTML entities
    
    return text.lower()

def analyze_product_similarity(tfidf_matrix, df, similarity_threshold=0.85):
    """
    Analyze and identify products that are too similar (likely same product, different color) 
    Returns DataFrame with similarity information
    """
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    similar_products = []
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            similarity = cosine_sim[i][j]
            if similarity > similarity_threshold:
                similar_products.append({
                    'product1': df.iloc[i]['product_title'],
                    'product2': df.iloc[j]['product_title'],
                    'similarity': similarity
                })
    
    return pd.DataFrame(similar_products)

def process_descriptions_tfidf(df, output_prefix="products"):
    
    """
    Process descriptions with TF-IDF and save results
    TF-IDF (Term Frequency-Inverse Documentation Frequency),
    is a technique to analyze the importance of a word relative to the document 

    Term Frequency (TF): Measures how often a word appears in a document.
    Inverse Document Frequency (IDF): Measures how important a word is across all documents.

    The TF-IDF score is calculated as:
    TF-IDF = TF * IDF

    TF = (word_count_in_doc / total_words_in_doc)
    IDF = log(N / df_i)

    I want to identify the keywords that are most important for the product description.
    """
    # Clean HTML descriptions
    df['clean_text'] = df['description'].apply(clean_html)
    
    # Filter out empty descriptions
    valid_descriptions = df[df['clean_text'].str.len() > 0]['clean_text']
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(valid_descriptions)
    
    # Analyze product similarities
    similar_products = analyze_product_similarity(tfidf_matrix, df)
    if not similar_products.empty:
        print("\nPotentially duplicate products (different colors/variants):")
        print(similar_products)
        similar_products.to_csv(f"{output_prefix}_similar_products.csv", index=False)
    
    # Save artifacts
    np.save(f"{output_prefix}_tfidf_matrix.npy", tfidf_matrix.toarray())
    pd.DataFrame(tfidf.get_feature_names_out()).to_csv(
        f"{output_prefix}_tfidf_features.csv", index=False, header=False)
    
    # Add TF-IDF vectors to original DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf.get_feature_names_out(),
        index=valid_descriptions.index)
    
    # Merge with original data
    final_df = df.copy()
    final_df = final_df.join(tfidf_df)

    recommendation_df = final_df.drop(['product_id', 'description', 'handle'], axis=1)
    
    # Save the recommendation DataFrame
    if output_prefix:
        recommendation_df.to_csv(f"{output_prefix}_recommendation.csv", index=False)
        final_df.to_csv(f"{output_prefix}_with_tfidf.csv", index=False)
    
    return final_df, recommendation_df, tfidf, similar_products

if __name__ == "__main__":
    df, vectorizer, similar_products = process_descriptions_tfidf()
    print("TF-IDF processing complete!")
    if not similar_products.empty:
        print(f"\nFound {len(similar_products)} pairs of similar products")