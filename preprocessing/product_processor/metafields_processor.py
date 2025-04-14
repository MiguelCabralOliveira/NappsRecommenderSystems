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
# Weight conversions to grams
WEIGHT_CONVERSIONS = {
    'GRAMS': 1.0, 'G': 1.0, # Added short unit
    'KILOGRAMS': 1000.0, 'KG': 1000.0,
    'OUNCES': 28.3495, 'OZ': 28.3495,
    'POUNDS': 453.592, 'LB': 453.592, 'LBS': 453.592,
    'MILLIGRAMS': 0.001, 'MG': 0.001,
}

# Length conversions to millimeters
LENGTH_CONVERSIONS = {
    'MILLIMETERS': 1.0, 'MM': 1.0,
    'CENTIMETERS': 10.0, 'CM': 10.0,
    'METERS': 1000.0, 'M': 1000.0,
    'INCHES': 25.4, 'IN': 25.4,
    'FEET': 304.8, 'FT': 304.8,
    'YARDS': 914.4, 'YD': 914.4,
}

# --- Helper Functions ---

def safe_json_parse_meta(json_like_str):
    """Safely parses JSON potentially found in metafield values."""
    if pd.isna(json_like_str) or not isinstance(json_like_str, str):
        return None
    try:
        # Basic cleaning for common non-standard JSON issues
        cleaned = json_like_str.replace("'", '"').replace('None', 'null').replace('True', 'true').replace('False', 'false')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None # Return None if parsing fails

def clean_html(html_content):
    """Clean HTML from text content"""
    if not html_content or not isinstance(html_content, str):
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # Additional cleanup for common HTML entities/artifacts
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        # Fallback to return original content if parsing fails
        return str(html_content)

def hex_to_lab(hex_color):
    """Convert hex color string to LabColor object."""
    if not COLORMATH_AVAILABLE: return None
    if not isinstance(hex_color, str): return None

    hex_color = hex_color.lstrip('#')
    if not re.match(r'^[0-9a-fA-F]{6}$', hex_color): # Basic validation
        if re.match(r'^[0-9a-fA-F]{3}$', hex_color): # Handle 3-digit hex
            hex_color = "".join([c*2 for c in hex_color])
        else:
            return None

    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        rgb = sRGBColor(r, g, b, is_upscaled=False) # Use is_upscaled=False for 0-1 range
        lab = convert_color(rgb, LabColor)
        return lab
    except Exception as e:
        # print(f"Error converting hex {hex_color} to Lab: {e}")
        return None

def color_similarity(hex_color1, hex_color2):
    """Calculate similarity between two hex colors (0-1 scale)."""
    if not COLORMATH_AVAILABLE: return 0.0

    lab1 = hex_to_lab(hex_color1)
    lab2 = hex_to_lab(hex_color2)

    if lab1 is None or lab2 is None:
        # If conversion failed or colors invalid, consider them dissimilar
        return 0.0

    try:
        # CIEDE2000 is perceptually more accurate
        delta_e = delta_e_cie2000(lab1, lab2)
        # Convert distance (0=same, >0=different) to similarity (1=same, <1=different)
        # Normalize based on a max perceptual distance (e.g., 100)
        similarity = max(0.0, 1.0 - (delta_e / 100.0))
        return similarity
    except Exception as e:
        # print(f"Error calculating delta_e between {hex_color1} and {hex_color2}: {e}")
        return 0.0 # Return dissimilar on error

def calculate_product_color_similarity(product1_colors, product2_colors):
    """Calculate overall color similarity between two lists of hex colors."""
    if not COLORMATH_AVAILABLE or not product1_colors or not product2_colors:
        return 0.0

    # Ensure unique colors in hex format
    c1_set = set(c.lstrip('#') for c in product1_colors if isinstance(c, str))
    c2_set = set(c.lstrip('#') for c in product2_colors if isinstance(c, str))

    valid_c1 = [f"#{c}" for c in c1_set if re.match(r'^[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', c)]
    valid_c2 = [f"#{c}" for c in c2_set if re.match(r'^[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', c)]

    if not valid_c1 or not valid_c2:
        return 0.0

    # Calculate pairwise similarities
    max_sim = 0.0
    total_sim = 0.0
    pair_count = 0

    for c1 in valid_c1:
        for c2 in valid_c2:
            sim = color_similarity(c1, c2)
            max_sim = max(max_sim, sim)
            total_sim += sim
            pair_count += 1

    avg_sim = total_sim / pair_count if pair_count > 0 else 0.0

    # Weighted average: emphasize the best match more
    overall_similarity = 0.7 * max_sim + 0.3 * avg_sim
    return overall_similarity


def extract_text_from_rich_text(rich_text_value):
    """Extract plain text from Shopify's rich text metafield structure."""
    text_parts = []
    data = safe_json_parse_meta(rich_text_value) # Parse if it's a string

    if not isinstance(data, dict) or data.get('type') != 'root':
        # If not the expected structure, or parsing failed, return original as string
        return str(rich_text_value)

    def extract_node_text(node):
        if isinstance(node, dict):
            if node.get('type') == 'text' and 'value' in node:
                text_parts.append(node['value'])
            # Recursively process children
            if 'children' in node and isinstance(node['children'], list):
                for child in node['children']:
                    extract_node_text(child)

    if 'children' in data and isinstance(data['children'], list):
        for child in data['children']:
            extract_node_text(child)

    full_text = ' '.join(text_parts)
    return re.sub(r'\s+', ' ', full_text).strip() # Normalize whitespace


def extract_product_references(reference_value):
    """Extract list of product GIDs or IDs from product reference metafields."""
    product_ids = []
    data = safe_json_parse_meta(reference_value) # Try parsing JSON list first

    if isinstance(data, list): # Handles list.product_reference
        items = data
    elif isinstance(reference_value, str): # Handles single product_reference string
        items = [reference_value]
    else:
        items = [] # Invalid format

    for item in items:
        if isinstance(item, str) and 'gid://shopify/Product/' in item:
            try:
                # Extract the numeric ID part
                product_id = item.split('/')[-1]
                if product_id.isdigit():
                    product_ids.append(product_id)
            except:
                pass # Ignore if splitting/parsing fails
    return product_ids


def convert_to_standard_unit(value, unit, is_weight=False):
    """Convert measurement to standard unit (grams for weight, mm for length)."""
    if pd.isna(value) or pd.isna(unit): return None
    try:
        numeric_value = float(value)
        unit_upper = str(unit).upper()
        conversions = WEIGHT_CONVERSIONS if is_weight else LENGTH_CONVERSIONS
        factor = conversions.get(unit_upper)
        if factor is None:
             print(f"Warning: Unknown unit '{unit}' for {'weight' if is_weight else 'length'}. Returning raw value.")
             return numeric_value # Return raw value if unit is unknown
        return numeric_value * factor
    except (ValueError, TypeError):
        return None # Failed conversion


def process_dimension_weight(field_value, is_weight=False):
    """Process dimension or weight type metafields, converting to standard unit."""
    data = safe_json_parse_meta(field_value) # Expects JSON like {"value": 10, "unit": "CM"}
    if not isinstance(data, dict) or 'value' not in data or 'unit' not in data:
        return None # Invalid format
    return convert_to_standard_unit(data['value'], data['unit'], is_weight)


def extract_hex_color(value):
    """Extracts a valid hex color (#RRGGBB or #RGB) from a value."""
    if not isinstance(value, str): return None
    val_clean = value.strip().lstrip('#')
    if re.match(r'^[0-9a-fA-F]{6}$', val_clean):
        return f"#{val_clean}"
    if re.match(r'^[0-9a-fA-F]{3}$', val_clean):
        return f"#{val_clean[0]*2}{val_clean[1]*2}{val_clean[2]*2}" # Expand shorthand
    return None


def process_single_metafield(metafield, product_id):
    """
    Processes a single metafield dictionary.
    Returns: tuple (unique_key, processed_value, type_category)
             type_category can be 'text', 'numeric', 'color', 'reference', 'other'
    """
    if not isinstance(metafield, dict) or 'key' not in metafield or 'value' not in metafield:
        return None, None, None

    namespace = metafield.get('namespace', '')
    key = metafield.get('key', '')
    value = metafield['value']
    m_type = metafield.get('type', '').lower()

    # Handle potential null/empty values early
    if pd.isna(value) or value == '':
        return None, None, None

    unique_key = f"meta_{namespace}_{key}"
    processed_value = None
    type_category = 'other' # Default category

    try:
        # --- Type-Specific Processing ---
        if 'product_reference' in m_type:
            refs = extract_product_references(value)
            processed_value = refs # Store the list of referenced IDs
            type_category = 'reference'
        elif 'dimension' in m_type:
            processed_value = process_dimension_weight(value, is_weight=False)
            if processed_value is not None: type_category = 'numeric'
        elif 'weight' in m_type:
            processed_value = process_dimension_weight(value, is_weight=True)
            if processed_value is not None: type_category = 'numeric'
        elif 'color' in m_type:
            hex_color = extract_hex_color(value)
            if hex_color:
                processed_value = hex_color # Store the hex code
                type_category = 'color'
            else: # If not a direct hex color, treat as text
                processed_value = str(value)
                type_category = 'text'
        elif 'rich_text' in m_type:
            processed_value = extract_text_from_rich_text(value)
            type_category = 'text'
        elif 'multi_line_text_field' in m_type or 'single_line_text_field' in m_type:
            processed_value = clean_html(str(value)) # Clean just in case
            type_category = 'text'
        elif 'number_integer' in m_type or 'number_decimal' in m_type or 'rating' in m_type or 'float' in m_type:
            processed_value = pd.to_numeric(value, errors='coerce')
            if pd.notna(processed_value): type_category = 'numeric'
            else: processed_value = None # Ensure invalid numerics are None
        elif 'boolean' in m_type:
            processed_value = 1 if str(value).lower() in ('true', '1', 'yes', 't') else 0
            type_category = 'numeric' # Treat boolean as numeric 0/1
        elif 'json' in m_type:
            # Decide how to handle JSON: stringify, extract specific keys, or ignore?
            # For now, treat as text
            processed_value = str(value) # Keep simple
            type_category = 'text'
        elif 'url' in m_type:
             # Treat URL as text, could potentially extract domain later
             processed_value = str(value)
             type_category = 'text'
        elif 'date_time' in m_type or 'date' in m_type:
             # Convert to days since epoch or similar numeric? For now, treat as text
             try:
                 dt_val = pd.to_datetime(value, errors='coerce', utc=True)
                 processed_value = dt_val.isoformat() if pd.notna(dt_val) else str(value)
             except:
                 processed_value = str(value)
             type_category = 'text' # Or potentially 'datetime' if handled separately
        # Add handling for 'file_reference', 'page_reference', 'variant_reference', 'metaobject_reference' if needed
        elif 'metaobject_reference' in m_type:
            # Basic handling: just use the GID string as text for now
            # More complex parsing might involve fetching metaobject data if available
            processed_value = str(value)
            type_category = 'text' # Or 'reference' if treated differently
        else:
            # Default for unknown types: treat as text
            processed_value = str(value)
            type_category = 'text'

        # Final cleanup for text fields
        if type_category == 'text' and isinstance(processed_value, str):
            processed_value = processed_value.strip()
            if not processed_value: # Turn empty strings into None
                 processed_value = None

    except Exception as e:
        # print(f"Error processing metafield {unique_key} for product {product_id}: {e}")
        processed_value = None # Nullify on error
        type_category = 'other'

    return unique_key, processed_value, type_category

# --- Vectorization, Normalization, Saving Functions ---

def vectorize_metafield_text_features(df: pd.DataFrame, text_cols: set, output_dir: str, shop_id: str, max_features=50) -> pd.DataFrame:
    """Applies TF-IDF to specified text columns derived from metafields."""
    valid_text_cols = [col for col in text_cols if col in df.columns]
    if not valid_text_cols:
        print("No metafield text columns found in DataFrame to vectorize.")
        return df

    print(f"Vectorizing {len(valid_text_cols)} metafield text columns...")
    df_processed = df.copy()

    # Combine text, handling NaNs
    def combine_row_text(row):
        texts = [str(row[col]) for col in valid_text_cols if pd.notna(row[col])]
        return " ".join(texts)

    combined_text = df_processed.apply(combine_row_text, axis=1)

    # Filter out rows with no combined text
    mask_has_text = combined_text.str.len() > 0
    if not mask_has_text.any():
        print("No text content found after combining. Skipping TF-IDF.")
        df_processed = df_processed.drop(columns=valid_text_cols, errors='ignore')
        return df_processed

    valid_text_series = combined_text[mask_has_text]

    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 1) # Keep simple
    )

    try:
        tfidf_matrix = tfidf.fit_transform(valid_text_series)
        model_filename = f"{shop_id}_metafield_text_tfidf_vectorizer.joblib"
        model_path = os.path.join(output_dir, "models", model_filename) # Save in models subdir
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(tfidf, model_path)
        print(f"Saved metafield text TF-IDF vectorizer to: {model_path}")

        feature_names = [f"meta_tfidf_{name}" for name in tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=valid_text_series.index)

        df_processed = df_processed.join(tfidf_df)
        df_processed[feature_names] = df_processed[feature_names].fillna(0) # Fill NaNs for rows without text
        print(f"Created {len(feature_names)} metafield TF-IDF features.")

    except Exception as e:
        print(f"Error during metafield text TF-IDF: {e}")
        traceback.print_exc()

    # Drop the original text columns used for TF-IDF
    df_processed = df_processed.drop(columns=valid_text_cols, errors='ignore')
    print(f"Removed original processed metafield text columns: {valid_text_cols}")
    return df_processed


def normalize_metafield_numeric_features(df: pd.DataFrame, numeric_cols: set) -> pd.DataFrame:
    """Normalizes numeric columns derived from metafields using RobustScaler."""
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns]
    if not valid_numeric_cols:
        print("No metafield numeric columns found in DataFrame to normalize.")
        return df

    print(f"Normalizing {len(valid_numeric_cols)} numeric metafield columns...")
    df_processed = df.copy()

    # Fill NaNs before scaling (use median as RobustScaler is less sensitive)
    for col in valid_numeric_cols:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val if pd.notna(median_val) else 0) # Fallback to 0 if median is NaN
            # print(f"Filled NaNs in {col} with median/0 value.")

    # Apply RobustScaler
    scaler = RobustScaler()
    try:
        # Scale only if there's variance, otherwise set to 0
        data_to_scale = df_processed[valid_numeric_cols]
        if data_to_scale.shape[0] > 1 and data_to_scale.nunique().all() > 0: # Check for actual data and variance
             scaled_data = scaler.fit_transform(data_to_scale)
             df_processed[valid_numeric_cols] = scaled_data
             print("Applied RobustScaler to numeric metafield columns.")
        elif data_to_scale.shape[0] > 0: # Data exists but no variance or single row
             print("Warning: Insufficient variance/data for RobustScaler. Setting scaled values to 0.")
             df_processed[valid_numeric_cols] = 0.0
        else: # Empty data
             print("No data to scale.")


    except ValueError as ve: # Handle cases like all values being the same
         print(f"ValueError during scaling (likely constant values): {ve}. Setting columns to 0.")
         df_processed[valid_numeric_cols] = 0.0 # Set to 0 if scaling fails
    except Exception as e:
        print(f"Error applying RobustScaler: {e}. Skipping normalization.")
        traceback.print_exc()

    return df_processed


def calculate_and_save_color_similarity(product_colors: dict, output_path: str, top_n=10):
    """Calculates color similarity and saves to CSV."""
    if not COLORMATH_AVAILABLE:
        print("Color similarity skipped: colormath library not available.")
        return pd.DataFrame()

    print(f"Calculating color similarities for {len(product_colors)} products.")
    if not product_colors:
        print("No product color data found. Skipping color similarity.")
        return pd.DataFrame() # Return empty DF

    all_pairs = []
    product_ids = list(product_colors.keys())

    for i in range(len(product_ids)):
        source_id = product_ids[i]
        source_hex_colors = list(set(product_colors[source_id])) # Unique colors for source

        similarities = []
        for j in range(len(product_ids)):
            if i == j: continue # Skip self-comparison

            target_id = product_ids[j]
            target_hex_colors = list(set(product_colors.get(target_id, []))) # Unique colors for target

            sim = calculate_product_color_similarity(source_hex_colors, target_hex_colors)

            if sim > 0.1: # Threshold to reduce file size and noise
                similarities.append({'target_product_id': target_id, 'color_similarity': sim})

        # Sort and get top N for the source product
        similarities.sort(key=lambda x: x['color_similarity'], reverse=True)
        for sim_data in similarities[:top_n]:
            all_pairs.append({
                'source_product_id': source_id,
                'target_product_id': sim_data['target_product_id'],
                'color_similarity': sim_data['color_similarity']
            })

    similarity_df = pd.DataFrame(all_pairs)

    if not similarity_df.empty:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            similarity_df.to_csv(output_path, index=False)
            print(f"Saved color similarity data ({len(similarity_df)} pairs) to: {output_path}")
        except Exception as e:
            print(f"Error saving color similarity CSV: {e}")
            traceback.print_exc()
    else:
        print("No significant color similarities found or calculated.")

    return similarity_df


def save_meta_product_references(product_references: dict, output_path: str):
    """Saves the extracted product references to CSV."""
    print(f"Saving {len(product_references)} product reference relationships.")
    if not product_references:
        print("No product references found to save.")
        return pd.DataFrame() # Return empty DF

    flat_references = []
    # product_references dict format: { product_id: { unique_meta_key: [target_ids], ... }, ... }
    for source_id, refs_by_key in product_references.items():
        if isinstance(refs_by_key, dict):
             for unique_meta_key, targets in refs_by_key.items():
                 # Extract original relationship type from unique_meta_key (meta_namespace_key -> key)
                 rel_type = '_'.join(unique_meta_key.split('_')[2:]) # Heuristic to get original key
                 if isinstance(targets, list):
                     for target_id in targets:
                         flat_references.append({
                             'source_product_id': source_id,
                             'target_product_id': target_id,
                             'relationship_type': rel_type
                         })

    references_df = pd.DataFrame(flat_references)

    if not references_df.empty:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            references_df.to_csv(output_path, index=False)
            print(f"Saved product reference data ({len(references_df)} links) to: {output_path}")
        except Exception as e:
            print(f"Error saving product references CSV: {e}")
            traceback.print_exc()
    else:
        print("No valid product references formatted for saving.")

    return references_df

# --- Main Processing Function ---

def process_metafields(df: pd.DataFrame, output_dir: str, shop_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to process metafields column in a product DataFrame.
    Extracts features, normalizes numerics, vectorizes text, calculates color similarity,
    and saves references and color similarity data.

    Args:
        df: DataFrame with product data, including 'product_id' and 'metafields'.
        output_dir: Base directory for saving auxiliary files (similarity, refs). Models go in output_dir/models.
        shop_id: Shop identifier for naming auxiliary files.

    Returns:
        tuple:
            - pd.DataFrame: The main DataFrame with metafield features added.
            - pd.DataFrame: DataFrame containing product reference relationships.
            - pd.DataFrame: DataFrame containing color similarity scores.
    """
    print("\n--- Processing Metafields ---")
    if 'metafields' not in df.columns:
        print("Warning: 'metafields' column not found. Skipping.")
        return df, pd.DataFrame(), pd.DataFrame()
    if 'product_id' not in df.columns:
        raise ValueError("DataFrame must contain 'product_id' column for metafield processing.")

    df_processed = df.copy()
    # Ensure product_id is string
    df_processed['product_id'] = df_processed['product_id'].astype(str)

    # Data collection during iteration
    all_meta_data = [] # List of dictionaries, one per product
    collected_product_colors = {} # {product_id: [colors]}
    collected_product_references = {} # {product_id: {unique_key: [refs]}}
    all_text_cols = set()
    all_numeric_cols = set()
    all_color_cols = set()

    print("Step 1: Extracting raw data from metafields column...")
    for index, row in df_processed.iterrows():
        product_id = row['product_id']
        metafields_raw = row['metafields']
        product_meta_values = {'product_id': product_id} # Start with product_id

        # Safely load metafields if they are stringified JSON
        if isinstance(metafields_raw, str):
            try:
                metafields_list = json.loads(metafields_raw)
                if not isinstance(metafields_list, list): metafields_list = []
            except:
                metafields_list = []
        elif isinstance(metafields_raw, list):
            metafields_list = metafields_raw
        else:
            metafields_list = []

        # Process each metafield for the current product
        product_refs_for_row = {}
        product_colors_for_row = []

        for metafield in metafields_list:
            unique_key, value, type_cat = process_single_metafield(metafield, product_id)

            if unique_key and value is not None:
                product_meta_values[unique_key] = value # Add feature to product's dict

                # Track column types and collect specific data
                if type_cat == 'text':
                    all_text_cols.add(unique_key)
                elif type_cat == 'numeric':
                    all_numeric_cols.add(unique_key)
                elif type_cat == 'color':
                    all_color_cols.add(unique_key)
                    product_colors_for_row.append(value)
                elif type_cat == 'reference':
                     # Value should be the list of target IDs
                     if isinstance(value, list) and value:
                         product_refs_for_row[unique_key] = value

        all_meta_data.append(product_meta_values)
        if product_colors_for_row:
            collected_product_colors[product_id] = list(set(product_colors_for_row)) # Store unique colors
        if product_refs_for_row:
            collected_product_references[product_id] = product_refs_for_row

    # Create DataFrame from collected data
    if not all_meta_data:
         print("No metafield data extracted.")
         df_processed = df_processed.drop(columns=['metafields'], errors='ignore')
         return df_processed, pd.DataFrame(), pd.DataFrame()

    meta_features_df = pd.DataFrame(all_meta_data)
    print(f"Created DataFrame with {len(meta_features_df.columns)-1} raw features from metafields for {len(meta_features_df)} products.")

    # Merge extracted features back into the main DataFrame
    # Drop original metafields column before merge to avoid potential conflicts if keys matched column name
    df_processed = df_processed.drop(columns=['metafields'], errors='ignore')
    # Use left merge to ensure all original products are kept
    df_processed = pd.merge(df_processed, meta_features_df, on='product_id', how='left')
    print(f"DataFrame shape after merging metafield features: {df_processed.shape}")

    # --- Post-Processing Steps ---
    print("\nStep 2: Post-processing extracted metafield features...")

    # Normalize numeric columns
    df_processed = normalize_metafield_numeric_features(df_processed, all_numeric_cols)

    # Vectorize text columns
    df_processed = vectorize_metafield_text_features(df_processed, all_text_cols, output_dir, shop_id)

    # Save product references
    refs_output_path = os.path.join(output_dir, f"{shop_id}_product_references.csv")
    references_df = save_meta_product_references(collected_product_references, refs_output_path)

    # Calculate and save color similarities
    color_sim_output_path = os.path.join(output_dir, f"{shop_id}_color_similarity.csv")
    color_similarity_df = calculate_and_save_color_similarity(collected_product_colors, color_sim_output_path)

    # Optionally remove original color columns (similarity captured separately)
    cols_to_drop_color = [col for col in all_color_cols if col in df_processed.columns]
    if cols_to_drop_color:
        print(f"Removing original metafield color columns: {cols_to_drop_color}")
        df_processed = df_processed.drop(columns=cols_to_drop_color, errors='ignore')

    print("--- Finished Processing Metafields ---")
    return df_processed, references_df, color_similarity_df