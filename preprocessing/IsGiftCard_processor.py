import pandas as pd

def process_gif_card(df):

    df = df[df['is_gift_card'] == False]

    df = df.drop(columns=['is_gift_card'])

    return df
    