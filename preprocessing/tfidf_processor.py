# tfidf_processor.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

def process_descriptions_tfidf(input_path="products.csv", output_prefix="products"):

    """"
    Process descriptions with TF-IDF and save results
    TF-IDF (Term Frequency-Inverse Documentation Frequency),
    is a tecnique to analyze the importance of a word relative to the document 

    Term Frequency (TF): Measures how often a word appears in a document.
    Inverse Document Frequency (IDF): Measures how important a word is across all documents.

    The TF-IDF score is calculated as:
    TF-IDF = TF * IDF

    TF = (word_count_in_doc / total_words_in_doc)

    IDF = log(N / df_i)

    I want to identify the keywords that are most important for the product description.\
    """

    # Load data
    df = pd.read_csv(input_path)
    
    # Clean HTML descriptions
    df['clean_text'] = df['description_html'].apply(clean_html)
    
    # Filter out empty descriptions
    valid_descriptions = df[df['clean_text'].str.len() > 0]['clean_text']
    
    # Create TF-IDF vectorizer (FIXED HERE)
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )  # Added missing closing parenthesis
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(valid_descriptions)
    
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
    final_df = df.join(tfidf_df)
    final_df.to_csv(f"{output_prefix}_with_tfidf.csv", index=False)
    
    return final_df, tfidf

if __name__ == "__main__":
    df, vectorizer = process_descriptions_tfidf()
    print("TF-IDF processing complete!")