# word2vec_processor.py
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessário para tokenização
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

def tokenize_text(text):
    """Tokenize text into words and remove stopwords"""
    if pd.isna(text) or not text:
        return []
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords, non-alphanumeric tokens and short tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words 
              and token.isalnum() and len(token) > 2]
    
    return tokens

def analyze_product_similarity(product_vectors, df, similarity_threshold=0.85):
    """
    Analyze and identify products that are too similar (likely same product, different color) 
    Returns DataFrame with similarity information including product IDs in both directions
    """
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(product_vectors)
    
    similar_products = []
    for i in range(len(cosine_sim)):
        for j in range(len(cosine_sim)):
            # Skip self-comparisons
            if i == j:
                continue
                
            similarity = cosine_sim[i][j]
            # Fix floating point imprecisions - round values very close to 1.0
            if similarity > 0.9999:
                similarity = 1.0
            
            # Only add if above threshold
            if similarity > similarity_threshold:
                similar_products.append({
                    'product1_id': df.iloc[i]['product_id'],
                    'product1': df.iloc[i]['product_title'],
                    'product2_id': df.iloc[j]['product_id'],
                    'product2': df.iloc[j]['product_title'],
                    'similarity': similarity
                })
    
    return pd.DataFrame(similar_products)

def process_descriptions_word2vec(df, output_prefix="products", vector_size=100, window=5, min_count=1):
    """
    Process descriptions with Word2Vec and save results
    
    Args:
        df: DataFrame with product data including 'description' column
        output_prefix: Prefix for output files
        vector_size: Dimensionality of the Word2Vec vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum frequency of words to include in the model
        
    Returns:
        tuple: (final_df, recommendation_df, w2v_model, similar_products) where:
            - final_df: DataFrame with Word2Vec features
            - recommendation_df: DataFrame ready for recommendation model
            - w2v_model: The trained Word2Vec model
            - similar_products: DataFrame with pairs of similar products
    """
    # Store all non-description columns to preserve them later
    preserve_cols = [col for col in df.columns if col != 'description']
    
    # Clean HTML descriptions
    df['clean_text'] = df['description'].apply(clean_html)
    
    # Filter out empty descriptions
    mask_valid_descriptions = df['clean_text'].str.len() > 0
    valid_df = df.loc[mask_valid_descriptions].copy()
    
    # Tokenize text
    valid_df['tokens'] = valid_df['clean_text'].apply(tokenize_text)
    
    # Filter out products with no tokens after processing
    valid_df = valid_df[valid_df['tokens'].map(len) > 0]
    
    print(f"Training Word2Vec model on {len(valid_df)} product descriptions...")
    
    # Train Word2Vec model
    w2v_model = Word2Vec(
        sentences=valid_df['tokens'].tolist(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )
    
    print(f"Word2Vec model trained. Vocabulary size: {len(w2v_model.wv.key_to_index)}")
    
    # Create product vectors by averaging word vectors
    def product_to_vec(tokens):
        word_vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if not word_vecs:
            return np.zeros(vector_size)
        return np.mean(word_vecs, axis=0)
    
    valid_df['product_vector'] = valid_df['tokens'].apply(product_to_vec)
    
    # Convert list of vectors to array for similarity calculation
    product_vectors = np.vstack(valid_df['product_vector'].values)
    
    # Analyze product similarities
    similar_products = analyze_product_similarity(product_vectors, valid_df)
    if not similar_products.empty:
        print("\nPotentially duplicate products (different colors/variants):")
        print(similar_products)
        similar_products.to_csv(f"{output_prefix}_similar_products.csv", index=False)
    
    # Create feature names with description_w2v_ prefix
    feature_names = [f"description_w2v_{i}" for i in range(vector_size)]
    
    # Create DataFrame with Word2Vec features
    w2v_df = pd.DataFrame(
        product_vectors,
        columns=feature_names,
        index=valid_df.index
    )
    
    # Save the Word2Vec model for later use
    w2v_model.save(f"{output_prefix}_word2vec_model.model")
    
    # Save product vectors
    np.save(f"{output_prefix}_product_vectors.npy", product_vectors)
    
    # Add Word2Vec vectors to original DataFrame
    final_df = df.copy()
    final_df = final_df.join(w2v_df)
    
    # Drop intermediate processing columns
    final_df = final_df.drop(['description', 'clean_text'], axis=1, errors='ignore')
    if 'tokens' in final_df.columns:
        final_df = final_df.drop(['tokens'], axis=1)
    if 'product_vector' in final_df.columns:
        final_df = final_df.drop(['product_vector'], axis=1)
    
    # Create recommendation DataFrame
    recommendation_cols = [col for col in final_df.columns if col != 'handle']
    recommendation_df = final_df[recommendation_cols].copy()
    
    # Save the recommendation DataFrame
    if output_prefix:
        recommendation_df.to_csv(f"{output_prefix}_recommendation.csv", index=False)
        final_df.to_csv(f"{output_prefix}_with_word2vec.csv", index=False)
    
    return final_df, recommendation_df, w2v_model, similar_products

if __name__ == "__main__":
    df, recommendation_df, w2v_model, similar_products = process_descriptions_word2vec()
    print("Word2Vec processing complete!")
    if not similar_products.empty:
        print(f"\nFound {len(similar_products)} pairs of similar products")