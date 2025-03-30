# description_processor.py
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize

# Download necessário para tokenização
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
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

def enhance_similarity_scores(similarity_matrix, method='squared', factor=1.5):
    """
    Enhance similarity scores to get higher values for similar items
    
    Args:
        similarity_matrix: Original cosine similarity matrix
        method: Enhancement method ('squared', 'power', 'exponential')
        factor: Enhancement factor (only used for 'power' method)
        
    Returns:
        Enhanced similarity matrix
    """
    if method == 'squared':
        # Square the values but keep the sign (increases high values more than low values)
        enhanced = np.square(similarity_matrix) * np.sign(similarity_matrix)
    elif method == 'power':
        # Raise to a power (factor) - increases contrast between high and low values
        enhanced = np.power(np.abs(similarity_matrix), factor) * np.sign(similarity_matrix)
    elif method == 'exponential':
        # Applies exponential scaling - dramatically increases high values
        enhanced = (np.exp(np.abs(similarity_matrix)) - 1) * np.sign(similarity_matrix)
    else:
        return similarity_matrix
    
    # Ensure values stay in [-1, 1] range
    max_val = np.max(np.abs(enhanced))
    if max_val > 1:
        enhanced = enhanced / max_val
    
    return enhanced

def analyze_product_similarity(product_vectors, df, similarity_threshold=0.4, 
                              enhance_method='squared'):
    """
    Analyze and identify products that are too similar (likely same product, different color)
    Returns DataFrame with similarity information including product IDs in both directions
    
    Args:
        product_vectors: Matrix of product vector representations
        df: DataFrame with product metadata
        similarity_threshold: Minimum similarity to consider products as similar
        enhance_method: Method to enhance similarity scores ('squared', 'power', 'exponential', None)
    """
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(product_vectors)
    
    # Enhance similarity scores if requested
    if enhance_method:
        cosine_sim = enhance_similarity_scores(cosine_sim, method=enhance_method)
    
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
    
    # Create DataFrame and sort by similarity (descending)
    result_df = pd.DataFrame(similar_products)
    if not result_df.empty:
        result_df = result_df.sort_values('similarity', ascending=False)
    
    return result_df

def weight_token_importance(tokens, idf_weights=None):
    """
    Weight tokens by their importance (similar to IDF weighting in TF-IDF)
    
    Args:
        tokens: List of tokens
        idf_weights: Dictionary of token IDF weights (if None, all tokens are weighted equally)
        
    Returns:
        Dictionary with token weights
    """
    if idf_weights is None:
        # If no IDF weights provided, weight all tokens equally
        return {token: 1.0 for token in tokens}
    else:
        # Weight tokens by their IDF value (more unique tokens get higher weights)
        return {token: idf_weights.get(token, 1.0) for token in tokens}

def calculate_idf_weights(token_lists, smooth=True):
    """
    Calculate IDF weights for tokens across all documents
    
    Args:
        token_lists: List of lists of tokens from all documents
        smooth: Whether to use smooth IDF formula (1+log(N/df)) instead of log(N/df)
        
    Returns:
        Dictionary mapping tokens to their IDF weights
    """
    # Count document frequency for each token
    doc_count = {}
    total_docs = len(token_lists)
    
    for doc_tokens in token_lists:
        # Count each token only once per document
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_count[token] = doc_count.get(token, 0) + 1
    
    # Calculate IDF weights
    idf_weights = {}
    for token, df in doc_count.items():
        if smooth:
            idf_weights[token] = 1.0 + np.log(total_docs / df)
        else:
            idf_weights[token] = np.log(total_docs / df)
    
    return idf_weights

def process_descriptions_word2vec(df, output_dir="results", output_prefix="products", 
                                vector_size=100, window=5, min_count=1, 
                                similarity_threshold=0.4, enhance_method='squared',
                                normalize_vectors=True, use_idf_weighting=True):
    """
    Process descriptions with Word2Vec and save results
    
    Args:
        df: DataFrame with product data including 'description' column
        output_dir: Directory to save output files
        output_prefix: Prefix for output files
        vector_size: Dimensionality of the Word2Vec vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum frequency of words to include in the model
        similarity_threshold: Minimum similarity to consider products as similar
        enhance_method: Method to enhance similarity scores (None, 'squared', 'power', 'exponential')
        normalize_vectors: Whether to normalize vectors before similarity calculation
        use_idf_weighting: Whether to use IDF weighting for token importance
        
    Returns:
        tuple: (final_df, recommendation_df, w2v_model, similar_products) where:
            - final_df: DataFrame with Word2Vec features
            - recommendation_df: DataFrame ready for recommendation model
            - w2v_model: The trained Word2Vec model
            - similar_products: DataFrame with pairs of similar products
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Calculate IDF weights if requested
    idf_weights = None
    if use_idf_weighting:
        print("Calculating IDF weights for tokens...")
        idf_weights = calculate_idf_weights(valid_df['tokens'].tolist())
    
    # Train Word2Vec model
    w2v_model = Word2Vec(
        sentences=valid_df['tokens'].tolist(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )
    
    print(f"Word2Vec model trained. Vocabulary size: {len(w2v_model.wv.key_to_index)}")
    
    # Create product vectors by averaging word vectors with token weights
    def product_to_vec(tokens):
        # Get token weights (IDF or uniform)
        token_weights = weight_token_importance(tokens, idf_weights)
        
        # Get weighted word vectors
        word_vecs = []
        weights = []
        
        for word in tokens:
            if word in w2v_model.wv:
                weight = token_weights.get(word, 1.0)
                word_vecs.append(w2v_model.wv[word] * weight)
                weights.append(weight)
        
        if not word_vecs:
            return np.zeros(vector_size)
        
        # Weighted average
        if sum(weights) > 0:
            return np.sum(word_vecs, axis=0) / sum(weights)
        else:
            return np.mean(word_vecs, axis=0)
    
    valid_df['product_vector'] = valid_df['tokens'].apply(product_to_vec)
    
    # Convert list of vectors to array for similarity calculation
    product_vectors = np.vstack(valid_df['product_vector'].values)
    
    # Normalize vectors if requested (makes cosine similarity more decisive)
    if normalize_vectors:
        print("Normalizing product vectors...")
        product_vectors = normalize(product_vectors, norm='l2', axis=1)
    
    # Analyze product similarities with enhanced scoring
    similar_products = analyze_product_similarity(
        product_vectors, 
        valid_df, 
        similarity_threshold=similarity_threshold,
        enhance_method=enhance_method
    )
    
    if not similar_products.empty:
        print("\nPotentially duplicate products (different colors/variants):")
        print(similar_products.head(10))  # Show just the top 10 most similar pairs
        
        # Calculate similarity statistics
        similarity_stats = {
            'mean': similar_products['similarity'].mean(),
            'median': similar_products['similarity'].median(),
            'min': similar_products['similarity'].min(),
            'max': similar_products['similarity'].max(),
            'std': similar_products['similarity'].std(),
            'count': len(similar_products)
        }
        
        print("\nSimilarity Statistics:")
        for stat, value in similarity_stats.items():
            print(f"{stat}: {value:.4f}" if isinstance(value, float) else f"{stat}: {value}")
        
        similar_products.to_csv(f"{output_dir}/{output_prefix}_similar_products.csv", index=False)
    
    # Create feature names with description_w2v_ prefix
    feature_names = [f"description_w2v_{i}" for i in range(vector_size)]
    
    # Create DataFrame with Word2Vec features
    w2v_df = pd.DataFrame(
        product_vectors,
        columns=feature_names,
        index=valid_df.index
    )
    
    # Save the Word2Vec model for later use
    w2v_model.save(f"{output_dir}/{output_prefix}_word2vec_model.model")
    
    # Save product vectors
    np.save(f"{output_dir}/{output_prefix}_product_vectors.npy", product_vectors)
    
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
        recommendation_df.to_csv(f"{output_dir}/{output_prefix}_recommendation.csv", index=False)
        final_df.to_csv(f"{output_dir}/{output_prefix}_with_word2vec.csv", index=False)
        
        # Save the similarity statistics
        if not similar_products.empty:
            pd.DataFrame([similarity_stats]).to_csv(
                f"{output_dir}/{output_prefix}_similarity_stats.csv", index=False
            )
    
    return final_df, recommendation_df, w2v_model, similar_products

if __name__ == "__main__":
    df, recommendation_df, w2v_model, similar_products = process_descriptions_word2vec()
    print("Word2Vec processing complete!")
    if not similar_products.empty:
        print(f"\nFound {len(similar_products)} pairs of similar products")