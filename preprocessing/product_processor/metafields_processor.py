# preprocessing/product_processor/metafields_processor.py

import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler # Using RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import traceback
from bs4 import BeautifulSoup
import re
import joblib # For saving TF-IDF models
import ast # Import ast for literal_eval

# Attempt to import colormath, provide warning if unavailable
try:
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000
    COLORMATH_AVAILABLE = True
except ImportError:
    print("Warning: 'colormath' library not found. Color similarity calculations will be skipped.")
    COLORMATH_AVAILABLE = False
    # Define dummy objects/functions if needed elsewhere, or just guard calls
    class LabColor: pass # Dummy
    def convert_color(c, target): return None # Dummy
    def delta_e_cie2000(c1, c2): return 1000.0 # Dummy, large difference

# --- Constants ---
WEIGHT_CONVERSIONS = {
    'GRAMS': 1.0, 'G': 1.0, 'KILOGRAMS': 1000.0, 'KG': 1000.0,
    'OUNCES': 28.3495, 'OZ': 28.3495, 'POUNDS': 453.592, 'LB': 453.592,
    'LBS': 453.592, 'MILLIGRAMS': 0.001, 'MG': 0.001,
}
LENGTH_CONVERSIONS = {
    'MILLIMETERS': 1.0, 'MM': 1.0, 'CENTIMETERS': 10.0, 'CM': 10.0,
    'METERS': 1000.0, 'M': 1000.0, 'INCHES': 25.4, 'IN': 25.4,
    'FEET': 304.8, 'FT': 304.8, 'YARDS': 914.4, 'YD': 914.4,
}
# --- End Constants ---

# --- Helper Functions ---
# (Todas as funções auxiliares (safe_json_parse_meta, clean_html, hex_to_lab, etc.)
#  permanecem as mesmas da versão completa anterior que te dei.
#  Certifica-te que estão todas aqui.)

def safe_json_parse_meta(json_like_str):
    if pd.isna(json_like_str) or not isinstance(json_like_str, str): return None
    try:
        cleaned = json_like_str.replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false')
        return json.loads(cleaned)
    except json.JSONDecodeError: return None

def clean_html(html_content):
    if not html_content or not isinstance(html_content, str): return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception: return str(html_content)

def hex_to_lab(hex_color):
    if not COLORMATH_AVAILABLE: return None
    if not isinstance(hex_color, str): return None
    hex_color = hex_color.lstrip('#')
    if not re.match(r'^[0-9a-fA-F]{6}$', hex_color):
        if re.match(r'^[0-9a-fA-F]{3}$', hex_color): hex_color = "".join([c*2 for c in hex_color])
        else: return None
    try:
        r,g,b = int(hex_color[0:2],16)/255.0, int(hex_color[2:4],16)/255.0, int(hex_color[4:6],16)/255.0
        rgb = sRGBColor(r, g, b, is_upscaled=False); lab = convert_color(rgb, LabColor)
        return lab
    except Exception: return None

def color_similarity(hex_color1, hex_color2):
    if not COLORMATH_AVAILABLE: return 0.0
    lab1, lab2 = hex_to_lab(hex_color1), hex_to_lab(hex_color2)
    if lab1 is None or lab2 is None: return 0.0
    try:
        delta_e = delta_e_cie2000(lab1, lab2)
        similarity = max(0.0, 1.0 - (delta_e / 100.0)); return similarity
    except Exception: return 0.0

def calculate_product_color_similarity(product1_colors, product2_colors):
    if not COLORMATH_AVAILABLE or not product1_colors or not product2_colors: return 0.0
    c1_set = set(c.lstrip('#') for c in product1_colors if isinstance(c, str))
    c2_set = set(c.lstrip('#') for c in product2_colors if isinstance(c, str))
    valid_c1 = [f"#{c}" for c in c1_set if re.match(r'^[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', c)]
    valid_c2 = [f"#{c}" for c in c2_set if re.match(r'^[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', c)]
    if not valid_c1 or not valid_c2: return 0.0
    max_sim, total_sim, pair_count = 0.0, 0.0, 0
    for c1 in valid_c1:
        for c2 in valid_c2:
            sim = color_similarity(c1, c2); max_sim = max(max_sim, sim); total_sim += sim; pair_count += 1
    avg_sim = total_sim / pair_count if pair_count > 0 else 0.0
    overall_similarity = 0.7 * max_sim + 0.3 * avg_sim; return overall_similarity

def extract_text_from_rich_text(rich_text_value):
    text_parts = []; data = safe_json_parse_meta(rich_text_value)
    if not isinstance(data, dict) or data.get('type') != 'root': return str(rich_text_value)
    def extract_node_text(node):
        if isinstance(node, dict):
            if node.get('type') == 'text' and 'value' in node: text_parts.append(node['value'])
            if 'children' in node and isinstance(node['children'], list):
                for child in node['children']: extract_node_text(child)
    if 'children' in data and isinstance(data['children'], list):
        for child in data['children']: extract_node_text(child)
    full_text = ' '.join(text_parts); return re.sub(r'\s+', ' ', full_text).strip()

def extract_product_references(reference_value):
    product_ids = []; data = safe_json_parse_meta(reference_value)
    if isinstance(data, list): items = data
    elif isinstance(reference_value, str): items = [reference_value]
    else: items = []
    for item in items:
        if isinstance(item, str) and 'gid://shopify/Product/' in item:
            try: product_id = item.split('/')[-1]; product_ids.append(product_id) if product_id.isdigit() else None
            except: pass
    return product_ids

def convert_to_standard_unit(value, unit, is_weight=False):
    if pd.isna(value) or pd.isna(unit): return None
    try:
        numeric_value = float(value); unit_upper = str(unit).upper()
        conversions = WEIGHT_CONVERSIONS if is_weight else LENGTH_CONVERSIONS
        factor = conversions.get(unit_upper)
        if factor is None: print(f"Warning: Unknown unit '{unit}'. Returning raw value."); return numeric_value
        return numeric_value * factor
    except (ValueError, TypeError): return None

def process_dimension_weight(field_value, is_weight=False):
    data = safe_json_parse_meta(field_value)
    if not isinstance(data, dict) or 'value' not in data or 'unit' not in data: return None
    return convert_to_standard_unit(data['value'], data['unit'], is_weight)

def extract_hex_color(value):
    if not isinstance(value, str): return None
    val_clean = value.strip().lstrip('#')
    if re.match(r'^[0-9a-fA-F]{6}$', val_clean): return f"#{val_clean}"
    if re.match(r'^[0-9a-fA-F]{3}$', val_clean): return f"#{val_clean[0]*2}{val_clean[1]*2}{val_clean[2]*2}"
    return None

def process_single_metafield(metafield, product_id):
    if not isinstance(metafield, dict) or 'key' not in metafield or 'value' not in metafield: return None, None, None
    namespace, key, value, m_type = metafield.get('namespace',''), metafield.get('key',''), metafield['value'], metafield.get('type','').lower()
    if pd.isna(value) or value == '': return None, None, None
    unique_key = f"meta_{namespace}_{key}"; processed_value = None; type_category = 'other'
    try:
        if 'product_reference' in m_type: refs = extract_product_references(value); processed_value = refs; type_category = 'reference'
        elif 'dimension' in m_type: processed_value = process_dimension_weight(value, is_weight=False); type_category = 'numeric' if processed_value is not None else 'other'
        elif 'weight' in m_type: processed_value = process_dimension_weight(value, is_weight=True); type_category = 'numeric' if processed_value is not None else 'other'
        elif 'color' in m_type: hex_color = extract_hex_color(value); processed_value, type_category = (hex_color, 'color') if hex_color else (str(value), 'text')
        elif 'rich_text' in m_type: processed_value = extract_text_from_rich_text(value); type_category = 'text'
        elif 'multi_line_text_field' in m_type or 'single_line_text_field' in m_type: processed_value = clean_html(str(value)); type_category = 'text'
        elif 'number_' in m_type or 'rating' in m_type or 'float' in m_type: processed_value = pd.to_numeric(value, errors='coerce'); type_category = 'numeric' if pd.notna(processed_value) else 'other'; processed_value = None if type_category=='other' else processed_value
        elif 'boolean' in m_type: processed_value = 1 if str(value).lower() in ('true', '1', 'yes', 't') else 0; type_category = 'numeric'
        elif 'json' in m_type: processed_value = str(value); type_category = 'text'
        elif 'url' in m_type: processed_value = str(value); type_category = 'text'
        elif 'date' in m_type:
             try: dt_val = pd.to_datetime(value, errors='coerce', utc=True); processed_value = dt_val.isoformat() if pd.notna(dt_val) else str(value)
             except: processed_value = str(value)
             type_category = 'text'
        elif 'metaobject_reference' in m_type: processed_value = str(value); type_category = 'text'
        else: processed_value = str(value); type_category = 'text'
        if type_category == 'text' and isinstance(processed_value, str): processed_value = processed_value.strip(); processed_value = None if not processed_value else processed_value
    except Exception as e: processed_value = None; type_category = 'other'
    return unique_key, processed_value, type_category

# --- Vectorization, Normalization, Saving Functions ---
# (Funções vectorize_metafield_text_features, normalize_metafield_numeric_features,
# calculate_and_save_color_similarity, save_meta_product_references permanecem
# EXATAMENTE as mesmas da versão completa anterior que te dei. Inclui-as aqui.)

def vectorize_metafield_text_features(df: pd.DataFrame, text_cols: set, output_dir: str, shop_id: str, max_features=50) -> pd.DataFrame:
    valid_text_cols = [col for col in text_cols if col in df.columns]
    if not valid_text_cols: print("No metafield text columns found in DataFrame to vectorize."); return df
    print(f"Vectorizing {len(valid_text_cols)} metafield text columns...")
    df_processed = df.copy();
    def combine_row_text(row): texts = [str(row[col]) for col in valid_text_cols if pd.notna(row[col])]; return " ".join(texts)
    combined_text = df_processed.apply(combine_row_text, axis=1)
    mask_has_text = combined_text.str.len() > 0
    if not mask_has_text.any(): print("No text content found after combining. Skipping TF-IDF."); df_processed = df_processed.drop(columns=valid_text_cols, errors='ignore'); return df_processed
    valid_text_series = combined_text[mask_has_text]; tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 1))
    try:
        tfidf_matrix = tfidf.fit_transform(valid_text_series)
        model_filename = f"{shop_id}_metafield_text_tfidf_vectorizer.joblib"; model_path = os.path.join(output_dir, "models", model_filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True); joblib.dump(tfidf, model_path); print(f"Saved metafield text TF-IDF vectorizer to: {model_path}")
        feature_names = [f"meta_tfidf_{name}" for name in tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=valid_text_series.index)
        df_processed = df_processed.join(tfidf_df); df_processed[feature_names] = df_processed[feature_names].fillna(0); print(f"Created {len(feature_names)} metafield TF-IDF features.")
    except Exception as e: print(f"Error during metafield text TF-IDF: {e}"); traceback.print_exc()
    df_processed = df_processed.drop(columns=valid_text_cols, errors='ignore'); print(f"Removed original processed metafield text columns: {valid_text_cols}"); return df_processed

def normalize_metafield_numeric_features(df: pd.DataFrame, numeric_cols: set) -> pd.DataFrame:
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns]
    if not valid_numeric_cols: print("No metafield numeric columns found in DataFrame to normalize."); return df
    print(f"Normalizing {len(valid_numeric_cols)} numeric metafield columns...")
    df_processed = df.copy()
    for col in valid_numeric_cols:
        if df_processed[col].isnull().any(): median_val = df_processed[col].median(); df_processed[col] = df_processed[col].fillna(median_val if pd.notna(median_val) else 0)
    scaler = RobustScaler()
    try:
        data_to_scale = df_processed[valid_numeric_cols];
        if data_to_scale.shape[0] > 1 and data_to_scale.nunique(dropna=False).all() > 0:
             data_to_scale = data_to_scale.apply(pd.to_numeric, errors='coerce').fillna(0); scaled_data = scaler.fit_transform(data_to_scale); df_processed[valid_numeric_cols] = scaled_data; print("Applied RobustScaler to numeric metafield columns.")
        elif data_to_scale.shape[0] > 0: print("Warning: Insufficient variance/data for RobustScaler. Setting scaled values to 0."); df_processed[valid_numeric_cols] = 0.0
        else: print("No data to scale.")
    except ValueError as ve: print(f"ValueError during scaling (likely constant values): {ve}. Setting columns to 0."); df_processed[valid_numeric_cols] = 0.0
    except Exception as e: print(f"Error applying RobustScaler: {e}. Skipping normalization."); traceback.print_exc()
    return df_processed

def calculate_and_save_color_similarity(product_colors: dict, output_path: str, top_n=10):
    if not COLORMATH_AVAILABLE: print("Color similarity skipped: colormath library not available."); return pd.DataFrame()
    print(f"Calculating color similarities for {len(product_colors)} products.");
    if not product_colors: print("No product color data found. Skipping color similarity."); return pd.DataFrame()
    all_pairs = []; product_ids = list(product_colors.keys())
    for i in range(len(product_ids)):
        source_id = product_ids[i]; source_hex_colors = list(set(product_colors[source_id]))
        similarities = []
        for j in range(len(product_ids)):
            if i == j: continue
            target_id = product_ids[j]; target_hex_colors = list(set(product_colors.get(target_id, [])))
            sim = calculate_product_color_similarity(source_hex_colors, target_hex_colors)
            if sim > 0.1: similarities.append({'target_product_id': target_id, 'color_similarity': sim})
        similarities.sort(key=lambda x: x['color_similarity'], reverse=True)
        for sim_data in similarities[:top_n]: all_pairs.append({'source_product_id': source_id, 'target_product_id': sim_data['target_product_id'], 'color_similarity': sim_data['color_similarity']})
    similarity_df = pd.DataFrame(all_pairs)
    if not similarity_df.empty:
        try: os.makedirs(os.path.dirname(output_path), exist_ok=True); similarity_df.to_csv(output_path, index=False); print(f"Saved color similarity data ({len(similarity_df)} pairs) to: {output_path}")
        except Exception as e: print(f"Error saving color similarity CSV: {e}"); traceback.print_exc()
    else: print("No significant color similarities found or calculated.")
    return similarity_df

def save_meta_product_references(product_references: dict, output_path: str):
    print(f"Processing {len(product_references)} products with potential references.")
    if not product_references: print("No product references found to save."); return pd.DataFrame()
    flat_references = []
    for source_id, refs_by_key in product_references.items():
        if isinstance(refs_by_key, dict):
             for unique_meta_key, targets in refs_by_key.items():
                 rel_type = '_'.join(unique_meta_key.split('_')[2:])
                 if isinstance(targets, list):
                     for target_id in targets: flat_references.append({'source_product_id': source_id, 'target_product_id': target_id, 'relationship_type': rel_type})
    references_df = pd.DataFrame(flat_references)
    if not references_df.empty:
        try: os.makedirs(os.path.dirname(output_path), exist_ok=True); references_df.to_csv(output_path, index=False); print(f"Saved product reference data ({len(references_df)} links) to: {output_path}")
        except Exception as e: print(f"Error saving product references CSV: {e}"); traceback.print_exc()
    else: print("No valid product references formatted for saving.")
    return references_df


# --- Main Processing Function ---
def process_metafields(df: pd.DataFrame, output_dir: str, shop_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to process metafields column in a product DataFrame.
    Extracts features, normalizes numerics, vectorizes text, calculates color similarity,
    and saves references and color similarity data.
    """
    print("\n--- Processing Metafields ---")
    if 'metafields' not in df.columns:
        print("Warning: 'metafields' column not found. Skipping.")
        return df, pd.DataFrame(), pd.DataFrame()
    if 'product_id' not in df.columns:
        raise ValueError("DataFrame must contain 'product_id' column for metafield processing.")

    df_processed = df.copy()
    df_processed['product_id'] = df_processed['product_id'].astype(str)

    all_meta_data = []
    collected_product_colors = {}
    collected_product_references = {}
    all_text_cols = set()
    all_numeric_cols = set()
    all_color_cols = set()

    print("Step 1: Extracting raw data from metafields column...")
    skipped_rows = 0
    for index, row in df_processed.iterrows():
        product_id = row['product_id']
        metafields_raw = row['metafields']
        product_meta_values = {'product_id': product_id}

        # --- CORRECTED/IMPROVED PARSING BLOCK ---
        metafields_list = [] # Default to empty list
        if isinstance(metafields_raw, list):
            metafields_list = metafields_raw # Already a list
        elif isinstance(metafields_raw, str) and metafields_raw.strip() and metafields_raw.strip() != '[]':
            try:
                # Try ast.literal_eval first for Python dict/list string representations
                parsed = ast.literal_eval(metafields_raw)
                if isinstance(parsed, list):
                    metafields_list = parsed
                else:
                    # print(f"Warning: ast.literal_eval did not return a list for product {product_id}. Type: {type(parsed)}")
                    pass # Keep metafields_list empty
            except (ValueError, SyntaxError, TypeError, MemoryError):
                # If ast fails, try json.loads with cleaning
                try:
                    cleaned_str = metafields_raw.replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false')
                    parsed_json = json.loads(cleaned_str)
                    if isinstance(parsed_json, list):
                        metafields_list = parsed_json
                    else:
                         # print(f"Warning: json.loads did not return a list for product {product_id}. Type: {type(parsed_json)}")
                         pass # Keep metafields_list empty
                except json.JSONDecodeError:
                    # print(f"Warning: Failed to parse metafields string for product {product_id} with both ast and json.")
                    skipped_rows += 1
                    pass # Keep metafields_list empty if both fail
        # --- END CORRECTED PARSING BLOCK ---

        # Process only if metafields_list is not empty
        if metafields_list:
            product_refs_for_row = {}
            product_colors_for_row = []
            has_valid_metafield = False # Flag to check if any valid metafield was processed

            for metafield in metafields_list:
                 # Ensure metafield is a dictionary before processing
                 if not isinstance(metafield, dict):
                     continue

                 unique_key, value, type_cat = process_single_metafield(metafield, product_id)

                 if unique_key and value is not None:
                     product_meta_values[unique_key] = value
                     has_valid_metafield = True # Mark that we found at least one
                     if type_cat == 'text': all_text_cols.add(unique_key)
                     elif type_cat == 'numeric': all_numeric_cols.add(unique_key)
                     elif type_cat == 'color': all_color_cols.add(unique_key); product_colors_for_row.append(value)
                     elif type_cat == 'reference' and isinstance(value, list) and value: product_refs_for_row[unique_key] = value

            if has_valid_metafield: # Only append if useful data was extracted
                 all_meta_data.append(product_meta_values)
                 if product_colors_for_row: collected_product_colors[product_id] = list(set(product_colors_for_row))
                 if product_refs_for_row: collected_product_references[product_id] = product_refs_for_row
        # else: # Optional: Log if a row had non-empty raw data but failed parsing or yielded no values
            # if isinstance(metafields_raw, str) and metafields_raw.strip() and metafields_raw.strip() != '[]':
            #     print(f"Row {index} (Product {product_id}): No valid metafields extracted after parsing.")

    if skipped_rows > 0:
         print(f"Warning: Skipped parsing metafields string for {skipped_rows} rows due to parsing errors.")

    if not all_meta_data:
         print("No valid metafield data extracted from any product.")
         if 'metafields' in df_processed.columns: df_processed = df_processed.drop(columns=['metafields'], errors='ignore')
         return df_processed, pd.DataFrame(), pd.DataFrame()

    meta_features_df = pd.DataFrame(all_meta_data)
    # Now check the number of columns *excluding* product_id
    num_new_features = len(meta_features_df.columns) - 1
    print(f"Created DataFrame with {num_new_features} raw features from metafields for {len(meta_features_df)} products.")

    if num_new_features == 0:
        print("Warning: Although metafields were processed, no valid key-value pairs were extracted into features.")
        if 'metafields' in df_processed.columns: df_processed = df_processed.drop(columns=['metafields'], errors='ignore')
        return df_processed, pd.DataFrame(), pd.DataFrame() # Return early if no features created


    # Merge extracted features back into the main DataFrame
    if 'metafields' in df_processed.columns: df_processed = df_processed.drop(columns=['metafields'], errors='ignore')
    df_processed = pd.merge(df_processed, meta_features_df, on='product_id', how='left')
    print(f"DataFrame shape after merging metafield features: {df_processed.shape}")

    # --- Post-Processing Steps ---
    print("\nStep 2: Post-processing extracted metafield features...")
    df_processed = normalize_metafield_numeric_features(df_processed, all_numeric_cols)
    df_processed = vectorize_metafield_text_features(df_processed, all_text_cols, output_dir, shop_id)
    refs_output_path = os.path.join(output_dir, f"{shop_id}_product_references.csv")
    references_df = save_meta_product_references(collected_product_references, refs_output_path)
    color_sim_output_path = os.path.join(output_dir, f"{shop_id}_color_similarity.csv")
    color_similarity_df = calculate_and_save_color_similarity(collected_product_colors, color_sim_output_path)
    cols_to_drop_color = [col for col in all_color_cols if col in df_processed.columns]
    if cols_to_drop_color:
        print(f"Removing original metafield color columns: {cols_to_drop_color}")
        df_processed = df_processed.drop(columns=cols_to_drop_color, errors='ignore')

    print("--- Finished Processing Metafields ---")
    return df_processed, references_df, color_similarity_df