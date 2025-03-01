import pandas as pd

def process_avaiable_for_sale(df):

    df = df[df['available_for_sale'] == True]

    df = df.drop(columns=['available_for_sale'])

    return df
    