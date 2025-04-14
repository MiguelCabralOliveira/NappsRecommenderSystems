# preprocessing/product_processor/variants_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import json
import re
import traceback
import ast

def safe_load_variants(variants_data):
    """Safely load variants data, which might be stringified JSON or list."""
    if pd.isna(variants_data):
        return []
    if isinstance(variants_data, list):
        return variants_data 
    if isinstance(variants_data, str):
        try:
            # Handle potential NaN string
            if variants_data.lower() == 'nan': return []
            cleaned_str = variants_data.replace("': '", '": "').replace("', '", '", "') \
                                   .replace("{'", '{"').replace("'}", '"}') \
                                   .replace("': ", '": ').replace(", '", ', "') \
                                   .replace(": True", ": true").replace(": False", ": false") \
                                   .replace(": None", ": null")
            loaded = json.loads(cleaned_str)
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            try:
                eval_loaded = ast.literal_eval(variants_data)
                return eval_loaded if isinstance(eval_loaded, list) else []
            except (ValueError, SyntaxError, TypeError, MemoryError):

                return []
        except Exception as e:

             return [] 
    return []


def extract_variant_features(variants_list):
    """Extracts features from a list of variant dictionaries for a single product."""
    if not variants_list or not isinstance(variants_list, list):
        # Return default values if no variants
        return pd.Series({
            'price_min': np.nan,
            'price_max': np.nan,
            'price_avg': np.nan,
            'discount_avg_perc': 0.0, 
            'has_discount': 0,
            'inventory_total': 0,
            'n_variants': 0,
            'option_values': {} 
        }, dtype='object')

    prices = []
    compare_prices = []
    inventory_quantities = []
    option_values = {} 

    for variant in variants_list:
        if not isinstance(variant, dict): continue 

        # --- Price ---
        price_info = variant.get('price')
        compare_price_info = variant.get('compareAtPrice')
        price = np.nan
        compare_price = np.nan

        if isinstance(price_info, dict):
            price = pd.to_numeric(price_info.get('amount'), errors='coerce')
        if isinstance(compare_price_info, dict):
            compare_price = pd.to_numeric(compare_price_info.get('amount'), errors='coerce')

        if pd.isna(price) and pd.notna(compare_price):
            price = compare_price
        if pd.isna(compare_price) and pd.notna(price):
             compare_price = price

        if pd.notna(price): prices.append(price)
        if pd.notna(compare_price): compare_prices.append(compare_price)


        # --- Inventory ---
        inv = variant.get('inventoryQuantity')
        if pd.notna(inv) and isinstance(inv, (int, float)):
            inventory_quantities.append(int(inv))


        # --- Options ---
        options = variant.get('selectedOptions')
        if isinstance(options, list):
            for option in options:
                 if isinstance(option, dict):
                    name = option.get('name')
                    value = option.get('value')
                    if name and value and isinstance(name, str) and isinstance(value, str):
                        name_clean = name.lower().strip().replace(" ", "_") # Clean option name
                        value_clean = value.strip()
                        if name_clean not in option_values:
                             option_values[name_clean] = set() # Use set for uniqueness
                        option_values[name_clean].add(value_clean)

    # --- Calculate Aggregated Features ---
    n_variants = len(variants_list)
    price_min = min(prices) if prices else np.nan
    price_max = max(prices) if prices else np.nan
    price_avg = np.mean(prices) if prices else np.nan
    inventory_total = sum(inventory_quantities) if inventory_quantities else 0

    # Discount calculation
    discount_percentages = []
    has_discount = 0
    if len(prices) == len(compare_prices): # Ensure lists match
        for p, cp in zip(prices, compare_prices):
            if pd.notna(p) and pd.notna(cp) and cp > p and cp > 0: # Valid comparison and actual discount
                 discount_percentages.append(((cp - p) / cp) * 100)
                 has_discount = 1

    discount_avg_perc = np.mean(discount_percentages) if discount_percentages else 0.0


    # Convert option sets back to lists for the series
    final_options = {name: list(values) for name, values in option_values.items()}

    return pd.Series({
        'price_min': price_min,
        'price_max': price_max,
        'price_avg': price_avg,
        'discount_avg_perc': discount_avg_perc,
        'has_discount': has_discount,
        'inventory_total': inventory_total,
        'n_variants': n_variants,
        'option_values': final_options # Keep structured options temporarily
    })

def process_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process variants column to extract meaningful features like price stats,
    inventory, options, and creates dummy variables for common options.

    Args:
        df: pandas DataFrame containing product data with variants column.

    Returns:
        DataFrame: processed DataFrame with variant features added.
    """
    print("Processing: variants (Extracting Features & Options)")
    if 'variants' not in df.columns:
        print("Warning: 'variants' column not found. Skipping.")
        return df

    df_processed = df.copy()

    try:
        # 1. Safely load variants data
        parsed_variants = df_processed['variants'].apply(safe_load_variants)

        # 2. Apply feature extraction row by row
        variant_features = parsed_variants.apply(extract_variant_features)

        # 3. Concatenate the new aggregated features
        df_processed = pd.concat([df_processed, variant_features.drop(columns=['option_values'])], axis=1)

        # --- Normalize Numerical Variant Features ---
        numerical_features = ['price_min', 'price_max', 'price_avg', 'discount_avg_perc', 'inventory_total', 'n_variants']
        cols_to_normalize = [col for col in numerical_features if col in df_processed.columns]

        # Fill NaNs in price columns before scaling (e.g., with 0 or median)
        # Let's use 0 for prices/inventory if missing, assuming unavailable/not applicable
        df_processed['price_min'] = df_processed['price_min'].fillna(0)
        df_processed['price_max'] = df_processed['price_max'].fillna(0)
        df_processed['price_avg'] = df_processed['price_avg'].fillna(0)
        df_processed['inventory_total'] = df_processed['inventory_total'].fillna(0)
        df_processed['n_variants'] = df_processed['n_variants'].fillna(0)
        # discount_avg_perc and has_discount have defaults


        if cols_to_normalize:
            print(f"Normalizing numerical variant features: {cols_to_normalize}")
            # Using RobustScaler as it's less sensitive to outliers often seen in prices/inventory
            scaler = RobustScaler()
            try:
                 df_processed[cols_to_normalize] = scaler.fit_transform(df_processed[cols_to_normalize])
                 print("Applied RobustScaler to numerical variant features.")
            except Exception as scale_e:
                 print(f"Error applying RobustScaler to variant features: {scale_e}. Skipping normalization.")


        # --- Process Option Dummies ---
        # Extract all unique option name-value pairs across all products
        all_option_pairs = {} # key: option_name, value: set of values
        temp_options = variant_features['option_values'] # Get the Series of dicts

        for options_dict in temp_options:
             if isinstance(options_dict, dict):
                for name, values_list in options_dict.items():
                     if name not in all_option_pairs: all_option_pairs[name] = set()
                     all_option_pairs[name].update(values_list)

        # Create dummy variables for common/important options (e.g., size, color)
        option_dummies_list = []
        MAX_DUMMIES_PER_OPTION = 20 # Limit to avoid excessive columns

        for name, unique_values in all_option_pairs.items():
            if not unique_values: continue

            # Limit the number of dummies per option type
            value_counts = pd.Series(dtype=int)
            for idx, options_dict in temp_options.items():
                 if isinstance(options_dict, dict) and name in options_dict:
                     for val in options_dict[name]:
                         value_counts.loc[val] = value_counts.get(val, 0) + 1

            top_values = value_counts.nlargest(MAX_DUMMIES_PER_OPTION).index.tolist()
            print(f"Creating dummies for option '{name}' (Top {len(top_values)} values): {top_values}")


            dummy_prefix = f"opt_{name}" # e.g., opt_color, opt_size
            # Create dummies only for top values
            for value in top_values:
                col_name = f"{dummy_prefix}_{re.sub(r'[^a-z0-9_]+', '', value.lower().replace(' ','_'))[:20]}" # Clean value for col name
                # Check if product has this option value
                has_value = temp_options.apply(lambda d: isinstance(d,dict) and name in d and value in d[name])
                option_dummies_list.append(pd.Series(has_value.astype(int), name=col_name))

        # Concatenate all dummy Series/DataFrames
        if option_dummies_list:
            option_dummies_df = pd.concat(option_dummies_list, axis=1)
            # Join dummies back to the main dataframe
            df_processed = pd.concat([df_processed, option_dummies_df], axis=1)
            # Fill potential NaNs in dummy columns with 0
            df_processed[option_dummies_df.columns] = df_processed[option_dummies_df.columns].fillna(0).astype(int)
            print(f"Added {len(option_dummies_df.columns)} variant option dummy features.")


        # Drop the original variants column and intermediate option storage
        df_processed = df_processed.drop(columns=['variants', 'option_values'], errors='ignore')


    except Exception as e:
        print(f"Error processing variants: {e}")
        traceback.print_exc()
        # Return original df without variants column in case of error
        return df.drop('variants', axis=1, errors='ignore')

    return df_processed