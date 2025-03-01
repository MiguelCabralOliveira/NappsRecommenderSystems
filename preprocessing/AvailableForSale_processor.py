import pandas as pd

def process_avaiable_for_sale(df):
    # The products that have the fields avaiable for sale false should be removed from the dataset

    df = df[df['avaiable_for_sale'] == True]
    return df