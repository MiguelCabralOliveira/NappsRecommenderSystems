import pandas as pd

def process_gif_card(df):
    """
    Process gift card using dummy variables
    Args:
        df: pandas DataFrame containing product data
    Returns:
        DataFrame: processed DataFrame with gift card dummies
    """
    # Extract gift card from dictionaries
    boolean_columns = ["is_gift_card"]