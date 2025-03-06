import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_variants(df):
    """
    Process variants column to extract meaningful features
    Args:
        df: pandas DataFrame containing product data with variants column
    Returns:
        DataFrame: processed DataFrame with variant features
    """
    def extract_variant_features(variants):
        if not variants:
            return pd.Series({
                'min_price': np.nan,
                'max_price': np.nan,
                'has_discount': 0,
                'size_options': [],
                'color_options': []
            })

        # Price related features
        prices = [float(v['price']['amount']) for v in variants]
        compare_prices = [float(v['compareAtPrice']['amount']) if v['compareAtPrice'] else float(v['price']['amount']) 
                         for v in variants]
        
        # Calculate discount - explicit integer conversion for has_discount
        has_discount = 1 if any(cp > p for p, cp in zip(prices, compare_prices)) else 0
        avg_discount = np.mean([(cp - p) / cp * 100 
                              for p, cp in zip(prices, compare_prices) 
                              if cp > p]) if has_discount else 0

        # Handle options
        size_options = []
        color_options = []
        other_options = []

        for variant in variants:
            for option in variant['selectedOptions']:
                option_name = option['name'].lower()
                option_value = variant['title']
                
                if 'size' in option_name:
                    size_options.append(option_value)
                elif 'color' in option_name or 'colour' in option_name:
                    color_options.append(option_value)
                else:
                    other_options.append(f"{option_name}:{option_value}")

        # Remove duplicates
        size_options = list(set(size_options))
        color_options = list(set(color_options))
        other_options = list(set(other_options))

        return pd.Series({
            'min_price': min(prices),
            'max_price': max(prices),
            'has_discount': has_discount,
            'size_options': size_options,
            'color_options': color_options,
            'other_options': other_options,
        })

    # Apply the function to variants column
    variant_features = df['variants'].apply(extract_variant_features)
    
    # Drop variants column
    result_df = df.drop(['variants'], axis=1)
    
    # Concatenate the new features
    result_df = pd.concat([result_df, variant_features], axis=1)
    
    # Create MinMaxScaler for numerical features
    scaler = MinMaxScaler()
    numerical_features = ['min_price', 'max_price']
    result_df[numerical_features] = scaler.fit_transform(result_df[numerical_features])
    
    option_dummies = []
    for option_type in ['size_options', 'color_options', 'other_options']:
        all_options = []
        for options in result_df[option_type]:
            if options:
                all_options.extend(options)
        
        if all_options:
            unique_options = sorted(list(set(all_options)))
            # Add a 'variant_' prefix to make the source clear
            if option_type == 'size_options':
                prefix = 'variant_size'
            elif option_type == 'color_options':
                prefix = 'variant_color'
            else:
                prefix = 'variant_other'
            
            # Create all dummy columns at once with appropriate prefix
            dummy_data = {
                f'{prefix}_{opt}': [1 if opt in x else 0 for x in result_df[option_type]]
                for opt in unique_options
            }
            option_dummies.append(pd.DataFrame(dummy_data, index=result_df.index))
    
    # Drop original options columns
    result_df = result_df.drop(['size_options', 'color_options', 'other_options'], axis=1)
    
    # Concatenate all dummy variables at once
    if option_dummies:
        result_df = pd.concat([result_df] + option_dummies, axis=1)
    
    return result_df


if __name__ == "__main__":
    # Debug/testing mode
    input_df = pd.read_csv("../products.csv")
    df_processed = process_variants(input_df)
    
    # Test has_discount is explicitly integer
    if 'has_discount' in df_processed.columns:
        print(f"Has discount dtype: {df_processed['has_discount'].dtype}")
        print(f"Has discount values: {df_processed['has_discount'].unique()}")
    
    print(f"\nTotal features after processing: {len(df_processed.columns)}")