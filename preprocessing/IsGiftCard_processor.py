import pandas as pd

def process_gif_card(df):

    # Transform the True and False to binary format, 1 and 0

    df['is_gift_card'] = df
    ['is_gift_card'].map({'True': 1, 'False': 0})
    return df
