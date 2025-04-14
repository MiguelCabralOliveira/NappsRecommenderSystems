# preprocessing/product_processor/tfidf_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
import numpy as np
import os
import traceback
import joblib # Para salvar o modelo

def clean_html(html):
    """Clean HTML content and extract meaningful text"""
    if pd.isna(html):
        return ""
    try:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ")
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'&[a-z]+;', '', text)  # Remove HTML entities
        return text.lower()
    except Exception:
        return str(html) # Fallback

def analyze_product_similarity(tfidf_matrix, df, product_id_col='product_id', title_col='product_title', similarity_threshold=0.85):
    """
    Analyze and identify products that are too similar based on TF-IDF matrix.
    Returns DataFrame with similarity information.
    """
    if tfidf_matrix.shape[0] != len(df):
         print(f"Warning: Mismatch between TF-IDF matrix rows ({tfidf_matrix.shape[0]}) and DataFrame rows ({len(df)}). Similarity analysis may be incorrect.")
         # Attempt to proceed if possible, otherwise return empty
         if tfidf_matrix.shape[0] < len(df):
             print("Using TF-IDF matrix subset for similarity.")
         else:
             print("Cannot perform similarity analysis due to shape mismatch.")
             return pd.DataFrame()

    # Calculate cosine similarity only on the valid matrix part
    cosine_sim = cosine_similarity(tfidf_matrix)

    similar_products = []
    df_subset = df.iloc[:tfidf_matrix.shape[0]] # Ensure we only index valid part of df

    for i in range(cosine_sim.shape[0]):
        for j in range(i + 1, cosine_sim.shape[0]): # Avoid duplicates (j > i) and self-comparison
            similarity = cosine_sim[i, j]
            if similarity > 0.9999: similarity = 1.0

            if similarity >= similarity_threshold: # Use >= for threshold inclusivity
                try:
                    # Use .iloc for safe positional indexing on the potentially filtered df_subset
                    p1_id = df_subset.iloc[i][product_id_col]
                    p1_title = df_subset.iloc[i][title_col] if title_col in df_subset.columns else 'N/A'
                    p2_id = df_subset.iloc[j][product_id_col]
                    p2_title = df_subset.iloc[j][title_col] if title_col in df_subset.columns else 'N/A'

                    similar_products.append({
                        'product1_id': p1_id,
                        'product1_title': p1_title,
                        'product2_id': p2_id,
                        'product2_title': p2_title,
                        'description_similarity': similarity
                    })
                except IndexError:
                     print(f"IndexError during similarity analysis at indices {i}, {j}. Skipping pair.")
                     continue
                except KeyError as ke:
                     print(f"KeyError during similarity analysis (likely missing column '{ke}'). Skipping pair.")
                     continue


    return pd.DataFrame(similar_products)

def process_descriptions_tfidf(df: pd.DataFrame, output_dir: str, shop_id: str, max_features=200, similarity_threshold=0.85) -> pd.DataFrame:
    """
    Process product descriptions with TF-IDF, save the vectorizer model and similarity report.

    Args:
        df: DataFrame containing product data with 'description' column.
        output_dir: Base directory for saving outputs (e.g., 'results/{shop_id}/preprocessed').
                    Models go into 'output_dir/models/'. Similarity report goes into 'output_dir'.
        shop_id: The shop ID for naming saved files.
        max_features: Maximum number of TF-IDF features.
        similarity_threshold: Threshold for reporting similar products based on description.

    Returns:
        DataFrame: Processed DataFrame with description TF-IDF features added and raw description removed.
                   Returns the original DataFrame (minus 'description') if processing fails.
    """
    print(f"Processing: description (TF-IDF with max_features={max_features})")
    if 'description' not in df.columns:
        print("Warning: 'description' column not found. Skipping TF-IDF processing.")
        return df

    df_processed = df.copy()
    # Define output paths
    model_filename = f"{shop_id}_description_tfidf_vectorizer.joblib"
    similarity_filename = f"{shop_id}_description_similarity_report.csv"
    models_dir = os.path.join(output_dir, "models")
    model_path = os.path.join(models_dir, model_filename)
    similarity_path = os.path.join(output_dir, similarity_filename)

    os.makedirs(models_dir, exist_ok=True) # Ensure models subdirectory exists

    try:
        # Clean HTML descriptions
        df_processed['clean_text'] = df_processed['description'].apply(clean_html)

        # Filter out rows where clean_text might be empty after cleaning
        # Keep track of original indices for joining back
        mask_valid_descriptions = df_processed['clean_text'].str.len() > 0
        if not mask_valid_descriptions.any():
            print("No valid description text found after cleaning. Skipping TF-IDF.")
            return df_processed.drop(columns=['description', 'clean_text'], errors='ignore')

        valid_descriptions_series = df_processed.loc[mask_valid_descriptions, 'clean_text']
        df_valid_subset = df_processed.loc[mask_valid_descriptions] # DF subset for similarity analysis

        # Create and fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2) # Consider single words and bigrams
        )
        tfidf_matrix = tfidf.fit_transform(valid_descriptions_series)

        # Save the fitted vectorizer model
        joblib.dump(tfidf, model_path)
        print(f"Saved description TF-IDF vectorizer to: {model_path}")

        # Analyze product similarities based on description TF-IDF
        if 'product_id' in df_valid_subset.columns:
            similar_products = analyze_product_similarity(
                tfidf_matrix,
                df_valid_subset, # Pass the subset corresponding to the matrix
                product_id_col='product_id',
                title_col='product_title', # Make sure 'product_title' exists if using this
                similarity_threshold=similarity_threshold
            )
            if not similar_products.empty:
                print(f"\nFound {len(similar_products)} pairs of products with description similarity >= {similarity_threshold}.")
                try:
                    similar_products.sort_values(by='description_similarity', ascending=False, inplace=True)
                    similar_products.to_csv(similarity_path, index=False)
                    print(f"Saved description similarity report to: {similarity_path}")
                except Exception as e_sim:
                    print(f"Error saving similarity report: {e_sim}")
            else:
                print(f"No product pairs found with description similarity >= {similarity_threshold}.")
        else:
             print("Skipping similarity analysis: 'product_id' column missing in relevant data subset.")

        # Get feature names with description_tfidf_ prefix
        feature_names = [f"desc_tfidf_{name}" for name in tfidf.get_feature_names_out()] # Shorter prefix

        # Create DataFrame with TF-IDF features, using the index from the valid descriptions
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=valid_descriptions_series.index 
        )

        # Join the TF-IDF features back to the main DataFrame (which includes all rows)
        df_processed = df_processed.join(tfidf_df)

        # Fill NaNs introduced by the join (for rows that had no valid description) with 0
        df_processed[feature_names] = df_processed[feature_names].fillna(0)

        print(f"Created {len(feature_names)} description TF-IDF features.")

    except Exception as e:
        print(f"Error processing descriptions with TF-IDF: {e}")
        traceback.print_exc()
        # Return the DataFrame without TF-IDF features in case of error, but drop raw cols
        return df_processed.drop(columns=['description', 'clean_text'], errors='ignore')

    # Drop original description and intermediate columns
    df_processed = df_processed.drop(columns=['description', 'clean_text'], errors='ignore')

    return df_processed