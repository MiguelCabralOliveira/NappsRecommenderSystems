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
        # Ensure all items are dicts if possible, handle non-dicts gracefully
        return [v for v in variants_data if isinstance(v, dict)]
    if isinstance(variants_data, str):
        try:
            # Handle potential NaN string or empty representation
            if variants_data.lower() == 'nan' or variants_data.strip() in ('', '[]', '{}'):
                 return []
            # Try direct JSON loading first after basic cleaning
            cleaned_str = variants_data.replace("': '", '": "').replace("', '", '", "') \
                                   .replace("{'", '{"').replace("'}", '"}') \
                                   .replace("': ", '": ').replace(", '", ', "') \
                                   .replace(": True", ": true").replace(": False", ": false") \
                                   .replace(": None", ": null")
            loaded = json.loads(cleaned_str)
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            # Fallback to literal_eval for Python list/dict strings
            try:
                # Ensure it's not overly complex/large to avoid issues
                if len(variants_data) > 10000: # Heuristic limit
                    print("Warning: Variant string too long for literal_eval, skipping.")
                    return []
                eval_loaded = ast.literal_eval(variants_data)
                # Ensure result is a list of dicts
                if isinstance(eval_loaded, list):
                     return [v for v in eval_loaded if isinstance(v, dict)]
                else:
                     return []
            except (ValueError, SyntaxError, TypeError, MemoryError):
                print(f"Warning: Could not parse variants string: {variants_data[:100]}...")
                return []
        except Exception as e:
             print(f"Warning: Unexpected error parsing variants string: {e}")
             return []
    # Handle other unexpected types
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
        }, dtype='object') # Specify dtype='object' for mixed types

    prices = []
    compare_prices = []
    inventory_quantities = []
    option_values = {}

    for variant in variants_list:
        # Ensure variant is a dictionary before proceeding
        if not isinstance(variant, dict):
            # print(f"Warning: Skipping non-dict item in variants list: {variant}")
            continue

        # --- Price ---
        price_info = variant.get('price')
        compare_price_info = variant.get('compareAtPrice')
        price = np.nan
        compare_price = np.nan

        # Safely extract price amount
        if isinstance(price_info, dict):
            price = pd.to_numeric(price_info.get('amount'), errors='coerce')
        elif isinstance(price_info, (int, float, str)): # Handle direct price values
             price = pd.to_numeric(price_info, errors='coerce')

        # Safely extract compareAtPrice amount
        if isinstance(compare_price_info, dict):
            compare_price = pd.to_numeric(compare_price_info.get('amount'), errors='coerce')
        elif isinstance(compare_price_info, (int, float, str)): # Handle direct compare price values
             compare_price = pd.to_numeric(compare_price_info, errors='coerce')


        # Handle cases where one price is missing but the other exists
        if pd.isna(price) and pd.notna(compare_price):
            price = compare_price # Use compare price if price is missing
        # Set compare_price to price if compare_price is missing but price exists (no discount)
        if pd.isna(compare_price) and pd.notna(price):
             compare_price = price

        if pd.notna(price): prices.append(price)
        if pd.notna(compare_price): compare_prices.append(compare_price)


        # --- Inventory ---
        inv = variant.get('inventoryQuantity')
        if pd.notna(inv) and isinstance(inv, (int, float)):
            inventory_quantities.append(int(inv))
        elif isinstance(inv, str) and inv.isdigit(): # Handle numeric strings
             inventory_quantities.append(int(inv))


        # --- Options ---
        options = variant.get('selectedOptions')
        if isinstance(options, list):
            for option in options:
                 if isinstance(option, dict):
                    name = option.get('name')
                    value = option.get('value')
                    # Ensure name and value are valid strings before processing
                    if name and value and isinstance(name, str) and isinstance(value, str):
                        name_clean = name.lower().strip().replace(" ", "_") # Clean option name
                        value_clean = value.strip()
                        if name_clean and value_clean: # Ensure they are not empty after cleaning
                            if name_clean not in option_values:
                                 option_values[name_clean] = set() # Use set for uniqueness
                            option_values[name_clean].add(value_clean)

    # --- Calculate Aggregated Features ---
    n_variants = len(variants_list) # Use original list length for count
    price_min = min(prices) if prices else np.nan
    price_max = max(prices) if prices else np.nan
    price_avg = np.mean(prices) if prices else np.nan
    inventory_total = sum(inventory_quantities) if inventory_quantities else 0

    # Discount calculation
    discount_percentages = []
    has_discount = 0
    # Use original prices/compare_prices lists which might have different lengths if parsing failed for some
    num_comparisons = min(len(prices), len(compare_prices))
    for i in range(num_comparisons):
        p = prices[i]
        cp = compare_prices[i]
        # Ensure both are valid numbers, compare_price > price, and compare_price > 0
        if pd.notna(p) and pd.notna(cp) and cp > p and cp > 0:
             discount_percentages.append(((cp - p) / cp) * 100)
             has_discount = 1 # Mark if at least one variant has a discount

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
        'option_values': final_options
    }, dtype='object') # Specify dtype='object'

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
    original_index = df_processed.index # Guardar índice original

    try:
        # 1. Safely load variants data
        parsed_variants = df_processed['variants'].apply(safe_load_variants)

        # 2. Apply feature extraction row by row
        variant_features = parsed_variants.apply(extract_variant_features)

        # 3. Concatenate the new aggregated features (exceto options)
        # Ensure the index aligns if extract_variant_features returned a Series with a different index
        variant_features.index = original_index
        df_processed = pd.concat([df_processed, variant_features.drop(columns=['option_values'])], axis=1)

        # --- Normalize Numerical Variant Features ---
        numerical_features = ['price_min', 'price_max', 'price_avg', 'discount_avg_perc', 'inventory_total', 'n_variants']
        cols_to_normalize = [col for col in numerical_features if col in df_processed.columns]
        if cols_to_normalize:
             print(f"Normalizing numerical variant features: {cols_to_normalize}")
             # Fill NaNs before scaling
             df_processed['price_min'] = df_processed['price_min'].fillna(df_processed['price_min'].median() if not df_processed['price_min'].isnull().all() else 0)
             df_processed['price_max'] = df_processed['price_max'].fillna(df_processed['price_max'].median() if not df_processed['price_max'].isnull().all() else 0)
             df_processed['price_avg'] = df_processed['price_avg'].fillna(df_processed['price_avg'].median() if not df_processed['price_avg'].isnull().all() else 0)
             df_processed['inventory_total'] = df_processed['inventory_total'].fillna(0)
             df_processed['n_variants'] = df_processed['n_variants'].fillna(0)
             # discount_avg_perc and has_discount have defaults (0)

             scaler = RobustScaler()
             try:
                  # Ensure columns are numeric before scaling
                  df_processed[cols_to_normalize] = df_processed[cols_to_normalize].apply(pd.to_numeric, errors='coerce').fillna(0)
                  df_processed[cols_to_normalize] = scaler.fit_transform(df_processed[cols_to_normalize])
                  print("Applied RobustScaler to numerical variant features.")
             except ValueError as ve:
                  print(f"ValueError during scaling (check for non-numeric data or constant columns): {ve}. Skipping scaling for these columns.")
             except Exception as scale_e:
                  print(f"Error applying RobustScaler to variant features: {scale_e}. Skipping normalization.")


        # --- Process Option Dummies ---
        all_option_pairs = {}
        temp_options = variant_features['option_values'] # Get the Series of dicts

        for options_dict in temp_options:
             if isinstance(options_dict, dict):
                for name, values_list in options_dict.items():
                     if name not in all_option_pairs: all_option_pairs[name] = set()
                     # Add only non-empty string values
                     valid_values = {v for v in values_list if isinstance(v, str) and v.strip()}
                     if valid_values:
                         all_option_pairs[name].update(valid_values)

        option_dummies_list = []
        MAX_DUMMIES_PER_OPTION = 20

        for name, unique_values in all_option_pairs.items():
            if not unique_values: continue

            # Calculate value counts based on the original Series of option dicts
            value_counts = pd.Series(dtype=int)
            for idx in temp_options.index: # Iterate using the index
                 options_dict = temp_options.loc[idx]
                 if isinstance(options_dict, dict) and name in options_dict:
                     # Count occurrences of each value for this option name
                     for val in options_dict[name]:
                         if isinstance(val, str) and val.strip(): # Ensure it's a valid string
                             value_counts.loc[val] = value_counts.get(val, 0) + 1

            # Get top values based on counts
            top_values = value_counts.nlargest(MAX_DUMMIES_PER_OPTION).index.tolist()
            if not top_values: continue # Skip if no values remain after filtering/counting

            print(f"Creating dummies for option '{name}' (Top {len(top_values)} values): {top_values}")

            dummy_prefix = f"opt_{name}"
            for value in top_values:
                # Clean column name more robustly
                clean_val_suffix = re.sub(r'[^\w\-_]', '', value.lower()).replace(' ', '_').replace('-', '_')
                clean_val_suffix = re.sub(r'_+', '_', clean_val_suffix).strip('_')
                if not clean_val_suffix: # Handle cases where value becomes empty after cleaning
                    clean_val_suffix = f"value_{hash(value) % 10000}" # Use a hash part as fallback
                col_name = f"{dummy_prefix}_{clean_val_suffix[:30]}"

                # Create boolean Series indicating presence of the value for the option name
                has_value = temp_options.apply(lambda d: isinstance(d,dict) and name in d and value in d.get(name, []))
                # Convert to int, name the Series, and ensure index matches original
                option_dummies_list.append(pd.Series(has_value.astype(int), name=col_name, index=original_index))

        # Concatenate all dummy Series/DataFrames
        if option_dummies_list:
            option_dummies_df = pd.concat(option_dummies_list, axis=1)

            # --- CORREÇÃO: Usar join para adicionar as colunas ---
            # Join on the index, which should be consistent
            df_processed = df_processed.join(option_dummies_df)

            # Fill NaNs introduced by the join (for rows that didn't have certain options)
            # Also handles potential NaNs if a dummy Series was all False initially
            df_processed[option_dummies_df.columns] = df_processed[option_dummies_df.columns].fillna(0).astype(int)
            # --- FIM CORREÇÃO ---

            print(f"Added {len(option_dummies_df.columns)} variant option dummy features.")


        # Drop the original variants column and intermediate option storage
        df_processed = df_processed.drop(columns=['variants', 'option_values'], errors='ignore')


    except Exception as e:
        print(f"Error processing variants: {e}")
        traceback.print_exc()
        # Return original df without variants column in case of error
        # Ensure original index is preserved if returning early
        return df.drop('variants', axis=1, errors='ignore')

    # Ensure the final DataFrame has the original index if it was modified
    if not df_processed.index.equals(original_index):
         print("Warning: Index mismatch after processing variants. Attempting to restore original index.")
         df_processed = df_processed.reindex(original_index)

    return df_processed