import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import re

# Unit conversion constants
# Weight conversions to grams
WEIGHT_CONVERSIONS = {
    'GRAMS': 1.0,
    'KILOGRAMS': 1000.0,
    'OUNCES': 28.3495,
    'POUNDS': 453.592,
    'MILLIGRAMS': 0.001
}

# Length conversions to millimeters
LENGTH_CONVERSIONS = {
    'MILLIMETERS': 1.0,
    'CENTIMETERS': 10.0,
    'METERS': 1000.0,
    'INCHES': 25.4,
    'FEET': 304.8,
    'YARDS': 914.4
}

# Global dictionary to store product colors
ALL_PRODUCT_COLORS = {}

def hex_to_lab(hex_color):
    """Convert hex color to LAB color space"""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB (0-1 scale)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    # Convert RGB to LAB
    rgb = sRGBColor(r, g, b)
    lab = convert_color(rgb, LabColor)
    
    return lab

def color_similarity(color1, color2):
    """Calculate similarity between two colors (0-1 scale, higher means more similar)"""
    try:
        # Convert hex to Lab
        lab1 = hex_to_lab(color1)
        lab2 = hex_to_lab(color2)
        
        try:
            # Calculate color difference using CIEDE2000 (perceptually uniform)
            delta_e = delta_e_cie2000(lab1, lab2)
            
            # Convert distance to similarity score (smaller distance = higher similarity)
            # Max meaningful distance is around 100 for completely different colors
            similarity = max(0, 1 - (delta_e / 100))
            
            return similarity
        except AttributeError as e:
            # Handle the asscalar error
            if "asscalar" in str(e):
                # Fallback to a simpler distance calculation
                l1, a1, b1 = lab1.get_value_tuple()
                l2, a2, b2 = lab2.get_value_tuple()
                
                # Calculate Euclidean distance
                distance = ((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)**0.5
                
                # Convert to similarity (0-1 scale)
                # Max meaningful distance in Lab space is around 100-150
                similarity = max(0, 1 - (distance / 150))
                
                return similarity
            else:
                raise
    except Exception as e:
        print(f"Error calculating color similarity: {e}")
        return 0

        
def calculate_product_color_similarity(product1_colors, product2_colors):
    """
    Calculate color similarity between two products with multiple colors
    Returns a similarity score between 0 and 1
    """
    # Handle empty color arrays
    if not product1_colors or not product2_colors:
        return 0
    
    # Compute all pairwise similarities
    similarity_matrix = np.zeros((len(product1_colors), len(product2_colors)))
    
    for i, color1 in enumerate(product1_colors):
        for j, color2 in enumerate(product2_colors):
            similarity_matrix[i, j] = color_similarity(color1, color2)
    
    # Calculate overall similarity using a weighted approach:
    # 1. Dominant color match (max similarity between any color pair)
    dominant_similarity = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0
    
    # 2. Average similarity across all combinations
    average_similarity = np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0
    
    # 3. Combine with more weight on dominant similarity (70/30 split)
    overall_similarity = 0.7 * dominant_similarity + 0.3 * average_similarity
    
    return overall_similarity

def extract_text_from_rich_text(rich_text_value):
    """Extract text from rich text field structure"""
    try:
        # Load the JSON if it's a string
        if isinstance(rich_text_value, str):
            try:
                data = json.loads(rich_text_value)
            except json.JSONDecodeError:
                return rich_text_value
        else:
            data = rich_text_value
            
        # Process rich text content
        if isinstance(data, dict) and 'type' in data and data['type'] == 'root':
            extracted_text = []
            
            def extract_text_from_nodes(nodes):
                for node in nodes:
                    if node.get('type') == 'text' and 'value' in node:
                        extracted_text.append(node['value'])
                    elif 'children' in node and isinstance(node['children'], list):
                        extract_text_from_nodes(node['children'])
            
            if 'children' in data and isinstance(data['children'], list):
                extract_text_from_nodes(data['children'])
            
            return ' '.join(extracted_text)
        
        return str(data)
    except Exception as e:
        print(f"Error extracting text from rich text: {e}")
        return str(rich_text_value)

def extract_product_references(reference_value):
    """
    Extract product IDs from product reference metafields
    Returns a list of product IDs
    """
    try:
        # Parse the reference value if it's a string
        if isinstance(reference_value, str):
            try:
                data = json.loads(reference_value)
            except json.JSONDecodeError:
                if 'gid://shopify/Product/' in reference_value:
                    # Handle single product reference
                    return [reference_value.split('/')[-1]]
                return []
        else:
            data = reference_value
            
        # For list.product_reference, extract product IDs
        if isinstance(data, list):
            product_ids = []
            for item in data:
                if isinstance(item, str) and 'gid://shopify/Product/' in item:
                    # Extract just the numeric ID part
                    product_id = item.split('/')[-1]
                    product_ids.append(product_id)
            
            return product_ids
            
        return []
            
    except Exception as e:
        print(f"Error extracting product references: {e}")
        return []

def extract_hex_colors_from_metaobject(references):
    """
    Extract hex color values from metaobject references
    Returns an array of hex color values
    """
    try:
        if not references or not isinstance(references, dict) or 'nodes' not in references:
            return []
            
        nodes = references['nodes']
        colors = []
        
        for node in nodes:
            # Check all fields for color values
            if 'fields' in node:
                for field in node['fields']:
                    # Look for fields with key 'color' or value that looks like a hex color
                    if (field.get('key', '').lower() == 'color' or 
                        field.get('key', '').lower() == 'hex' or 
                        field.get('key', '').lower() == 'code') and field.get('value'):
                        
                        value = field.get('value', '')
                        # Ensure it's a valid hex color
                        if isinstance(value, str) and (value.startswith('#') and len(value) in [4, 7]):
                            colors.append(value)
                            
                    # Also look for color values in field names that contain 'color'
                    elif 'color' in field.get('key', '').lower() and field.get('value'):
                        value = field.get('value', '')
                        # Check if it looks like a hex code
                        if isinstance(value, str) and re.match(r'^#?[0-9A-Fa-f]{3,6}$', value):
                            # Add # if missing
                            if not value.startswith('#'):
                                value = f"#{value}"
                            colors.append(value)
        
        return colors
        
    except Exception as e:
        print(f"Error extracting color values: {e}")
        return []

def extract_colors_from_variant_options(variant_options):
    """
    Extract color information from variant options
    Returns a list of color values
    """
    if not variant_options or not isinstance(variant_options, list):
        return []
    
    colors = []
    for option in variant_options:
        if ('color' in option.lower() or 'colour' in option.lower()) and option:
            # Extract just the color part, assuming format like "Color: Blue" or similar
            parts = option.split(':')
            if len(parts) > 1:
                color_value = parts[1].strip()
                if color_value:
                    colors.append(color_value)
            else:
                colors.append(option)
    
    return colors

def extract_metaobject_reference_data(reference_value, references=None, metaobject_type=None):
    """
    Extract data from metaobject references
    Args:
        reference_value: The metaobject reference value
        references: The references data structure if available
        metaobject_type: The type of the metaobject if known
    """
    try:
        # Check if this is a color-related metaobject
        is_color_type = metaobject_type and 'color' in metaobject_type.lower()
        
        # If it's a color type and we have references, extract hex colors
        if is_color_type and references:
            color_values = extract_hex_colors_from_metaobject(references)
            # Return color values as-is (as array) instead of joining them
            if color_values:
                return color_values
        
        # If we have direct references data with nodes, process it
        if references and isinstance(references, dict) and 'nodes' in references:
            nodes = references['nodes']
            
            # For color-related metaobjects, prioritize color extraction
            if 'color' in str(references).lower():
                color_values = extract_hex_colors_from_metaobject(references)
                if color_values:
                    return color_values
            
            # Extract values from the fields of each node
            results = []
            for node in nodes:
                if 'fields' in node and node['fields']:
                    # Get the first field's value (prioritize)
                    if len(node['fields']) > 0:
                        results.append(node['fields'][0].get('value', ''))
            
            if results:
                return " ".join(results)
        
        # Parse the reference value if it's a string
        if isinstance(reference_value, str):
            try:
                data = json.loads(reference_value)
            except json.JSONDecodeError:
                # Return empty string for single metaobject references
                if 'gid://shopify/Metaobject/' in reference_value:
                    return ""
                return reference_value
        else:
            data = reference_value
            
        # If it's a list of metaobject IDs without resolved data, return empty string
        if isinstance(data, list) and all(isinstance(item, str) and 'gid://shopify/Metaobject/' in item for item in data):
            return ""
            
        return ""
    except Exception as e:
        print(f"Error extracting data from metaobject reference: {e}")
        return ""

def clean_html(html_content):
    """Clean HTML from text content"""
    if not html_content or not isinstance(html_content, str):
        return ""
        
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html_content

def convert_to_standard_unit(value, unit, is_weight=False):
    """Convert measurement to standard unit (grams for weight, millimeters for dimensions)"""
    if is_weight:
        # Convert to grams
        conversion_factor = WEIGHT_CONVERSIONS.get(unit.upper(), 1.0)
    else:
        # Convert to millimeters
        conversion_factor = LENGTH_CONVERSIONS.get(unit.upper(), 1.0)
    
    return value * conversion_factor

def process_dimension(dimension_value, include_unit=False):
    """
    Process dimension type metafields, converting to standard unit (millimeters)
    
    Args:
        dimension_value: The dimension value to process
        include_unit: Whether to include the unit in the output or just return the numerical value
        
    Returns:
        If include_unit=True: Formatted string with converted values and units
        If include_unit=False: Numerical value converted to standard unit (millimeters)
    """
    try:
        if isinstance(dimension_value, str):
            data = json.loads(dimension_value)
        else:
            data = dimension_value
            
        # Handle list of dimensions
        if isinstance(data, list):
            if include_unit:
                return " ".join([
                    f"{convert_to_standard_unit(item.get('value', 0), item.get('unit', 'MILLIMETERS'))} mm" 
                    for item in data
                ])
            else:
                # Return the average of all dimensions in standard units
                if not data:
                    return 0.0
                    
                total = sum([
                    convert_to_standard_unit(item.get('value', 0), item.get('unit', 'MILLIMETERS')) 
                    for item in data
                ])
                return total / len(data)
            
        # Handle single dimension
        if isinstance(data, dict):
            converted_value = convert_to_standard_unit(
                data.get('value', 0), 
                data.get('unit', 'MILLIMETERS')
            )
            
            if include_unit:
                return f"{converted_value} mm"
            else:
                return converted_value
                
        return 0.0
    except Exception as e:
        print(f"Error processing dimension: {e}")
        return 0.0 if not include_unit else ""

def process_weight(weight_value, include_unit=False):
    """
    Process weight type metafields, converting to standard unit (grams)
    
    Args:
        weight_value: The weight value to process
        include_unit: Whether to include the unit in the output or just return the numerical value
        
    Returns:
        If include_unit=True: Formatted string with converted value and unit
        If include_unit=False: Numerical value converted to standard unit (grams)
    """
    try:
        if isinstance(weight_value, str):
            data = json.loads(weight_value)
        else:
            data = weight_value
            
        if isinstance(data, dict):
            converted_value = convert_to_standard_unit(
                data.get('value', 0), 
                data.get('unit', 'GRAMS'),
                is_weight=True
            )
            
            if include_unit:
                return f"{converted_value} g"
            else:
                return converted_value
                
        return 0.0
    except Exception as e:
        print(f"Error processing weight: {e}")
        return 0.0 if not include_unit else ""

def process_metafield_value(metafield):
    """Process a single metafield based on its type"""
    if not metafield or 'value' not in metafield:
        return None
        
    metafield_type = metafield.get('type', '')
    metafield_key = metafield.get('key', '')
    value = metafield['value']
    
    # Check if this is a dimension or weight field by type or key
    is_dimension = 'dimension' in metafield_type.lower() or 'dimension' in metafield_key.lower()
    is_weight = 'weight' in metafield_type.lower() or 'weight' in metafield_key.lower()
    is_color = 'color' in metafield_type.lower() or 'color' in metafield_key.lower()
    
    # Handle different metafield types
    if is_dimension:
        # For dimension, return the numerical value in standard unit (mm)
        return process_dimension(value, include_unit=False)
    elif is_weight:
        # For weight, return the numerical value in standard unit (g)
        return process_weight(value, include_unit=False)
    elif is_color and isinstance(value, str) and (value.startswith('#') or re.match(r'^[0-9A-Fa-f]{6}$', value)):
        # Ensure hex color has # prefix
        if not value.startswith('#'):
            value = f"#{value}"
        return [value]  # Return as a list for consistency with other color processing
    elif 'rich_text_field' in metafield_type:
        return extract_text_from_rich_text(value)
    elif 'product_reference' in metafield_type:
        # For product references, return empty string in main processing
        # They'll be handled separately
        return ""
    elif 'metaobject_reference' in metafield_type:
        # Pass references data and metaobject type if available
        references = metafield.get('references')
        metaobject_type = None
        
        # Try to determine if this is a color-related metaobject
        if 'color' in metafield_key.lower():
            metaobject_type = 'color'
        elif references and 'nodes' in references:
            for node in references['nodes']:
                if 'type' in node and 'color' in node['type'].lower():
                    metaobject_type = node['type']
                    break
                    
        return extract_metaobject_reference_data(value, references, metaobject_type)
    elif 'multi_line_text_field' in metafield_type:
        return clean_html(value)
    elif 'single_line_text_field' in metafield_type or 'url' in metafield_type:
        return str(value)
    elif 'number_integer' in metafield_type or 'number_decimal' in metafield_type or 'rating' in metafield_type:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    elif 'boolean' in metafield_type:
        if isinstance(value, bool):
            return 1 if value else 0
        try:
            return 1 if value.lower() in ('true', 't', 'yes', 'y', '1') else 0
        except (ValueError, AttributeError):
            return 0
    elif 'json' in metafield_type:
        try:
            # For JSON, we'll convert to a string representation
            if isinstance(value, str):
                data = json.loads(value)
            else:
                data = value
            return json.dumps(data)
        except:
            return str(value)
    else:
        # Default handling for other types
        return str(value)

def vectorize_text_features(metafields_df):
    """
    Apply TF-IDF vectorization to text features
    Returns dict mapping feature name to vectorized data
    """
    text_features = {}
    text_columns = metafields_df.select_dtypes(include=['object']).columns
    
    if len(text_columns) == 0:
        return {}
        
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        max_df=0.8,
        min_df=0.01  
    )
    
    for col in text_columns:
        # Convert column name to string to avoid errors with integer columns
        col_str = str(col)
        
        # Skip color, dimension, weight, and product reference columns
        if any(keyword in col_str.lower() for keyword in ['color', 'colour', 'dimension', 'weight', 'product_reference']):
            continue
            
        # Skip columns with array values (like our color arrays)
        if metafields_df[col].apply(lambda x: isinstance(x, list)).any():
            continue
            
        # Fill NA values with empty string
        text_data = metafields_df[col].fillna('').astype(str)
        
        # Skip if all values are empty after conversion
        if (text_data == '').all():
            continue
            
        try:
            # Apply TF-IDF
            feature_matrix = vectorizer.fit_transform(text_data)
            feature_names = [f"{col}_{name}" for name in vectorizer.get_feature_names_out()]
            text_features[col] = pd.DataFrame(
                feature_matrix.toarray(),
                columns=feature_names,
                index=metafields_df.index
            )
        except Exception as e:
            print(f"Error vectorizing column {col}: {e}")
    
    return text_features

def normalize_numeric_features(metafields_df):
    """Normalize numeric features using MinMaxScaler"""
    numeric_columns = metafields_df.select_dtypes(include=np.number).columns
    
    if len(numeric_columns) == 0:
        return metafields_df
        
    scaler = MinMaxScaler()
    metafields_df[numeric_columns] = scaler.fit_transform(
        metafields_df[numeric_columns].fillna(0)
    )
    
    return metafields_df

def process_metafields(product, metafields_config=None):
    """
    Process all metafields for a product
    Args:
        product: Dictionary containing product data with 'metafields' key
        metafields_config: List of metafield configurations with keys and types
    Returns:
        Tuple of (processed_data, product_references)
        - processed_data: Dictionary with processed metafield data
        - product_references: Dictionary with product reference relationships
    """
    global ALL_PRODUCT_COLORS
    
    processed_data = {}
    product_references = {}
    product_colors = []
    
    if 'metafields' not in product or not product['metafields']:
        return processed_data, product_references
    
    # Get product ID
    product_id = product.get('product_id', None)
    if not product_id:
        return processed_data, product_references
    
    # Extract just the numeric part if it's a GID
    if isinstance(product_id, str) and 'gid://shopify/Product/' in product_id:
        product_id = product_id.split('/')[-1]
    
    # Create a mapping of metafield keys to their types from config
    metafield_types = {}
    if metafields_config:
        for field in metafields_config:
            key = f"{field.get('namespace', '')}_{field.get('key', '')}"
            metafield_types[key] = field.get('type', '')
    
    # Process each metafield
    for metafield in product['metafields']:
        if not metafield or 'key' not in metafield:
            continue
            
        # Create a unique key using namespace and key
        namespace = metafield.get('namespace', '')
        key = metafield.get('key', '')
        unique_key = f"{namespace}_{key}"
        
        # Add type information from config if available
        if unique_key in metafield_types:
            metafield['type'] = metafield_types[unique_key]
        
        # Get field type and check if it's a special field type
        field_type = metafield.get('type', '').lower()
        is_color_related = 'color' in key.lower() or 'colour' in key.lower() or 'color' in field_type
        is_dimension_related = 'dimension' in key.lower() or 'dimension' in field_type
        is_weight_related = 'weight' in key.lower() or 'weight' in field_type
        is_product_reference = 'product_reference' in field_type
        
        # Special handling for product reference metafields
        if is_product_reference and 'value' in metafield:
            # Extract product references to a separate structure
            referenced_products = extract_product_references(metafield['value'])
            
            if referenced_products:
                # Store the relationship type and referenced products
                product_references[unique_key] = {
                    'source_product_id': product_id,
                    'relationship_type': key,
                    'referenced_product_ids': referenced_products
                }
            
            # Skip further processing of this field
            continue
        
        # Special handling for color-related metafields
        if is_color_related and 'references' in metafield and metafield['references']:
            colors = extract_hex_colors_from_metaobject(metafield['references'])
            if colors:
                # Store the colors as an array directly, not as a joined string
                processed_data[unique_key] = colors
                
                # Also store in global product colors dictionary for later similarity calculation
                ALL_PRODUCT_COLORS[product_id] = colors
                
                continue
            
            # Also try direct color value processing for non-reference color fields
            if 'value' in metafield:
                value = metafield['value']
                if isinstance(value, str):
                    # Check if it's a direct hex color value
                    if value.startswith('#') and len(value) in [4, 7]:
                        processed_data[unique_key] = [value]
                        product_colors.append(value)
                        continue
                    
                    # Try to parse if it's JSON
                    try:
                        color_data = json.loads(value)
                        if isinstance(color_data, dict) and 'hex' in color_data:
                            hex_value = color_data['hex']
                            if not hex_value.startswith('#'):
                                hex_value = f"#{hex_value}"
                            processed_data[unique_key] = [hex_value]
                            product_colors.append(hex_value)
                            continue
                    except json.JSONDecodeError:
                        pass
        
        # Special handling for dimension metafields
        elif is_dimension_related and 'value' in metafield:
            processed_data[unique_key] = process_dimension(metafield['value'], include_unit=False)
            continue
            
        # Special handling for weight metafields
        elif is_weight_related and 'value' in metafield:
            processed_data[unique_key] = process_weight(metafield['value'], include_unit=False)
            continue
        
        # For metaobject references with available references data
        if metafield.get('type') in ['list.metaobject_reference', 'metaobject_reference'] and 'references' in metafield and metafield['references']:
            if metafield['references'] and 'nodes' in metafield['references'] and metafield['references']['nodes']:
                values = []
                for node in metafield['references']['nodes']:
                    if 'fields' in node and node['fields']:
                        # Get the first field's value
                        first_field = node['fields'][0]
                        if 'value' in first_field:
                            values.append(first_field['value'])
                
                if values:
                    processed_data[unique_key] = " ".join(values)
                    continue
        
        # For all other types, use the metafield value processor
        if 'value' in metafield:
            processed_value = process_metafield_value(metafield)
            processed_data[unique_key] = processed_value
            
            # If this returns a list of colors, add them to product_colors
            if is_color_related and isinstance(processed_value, list):
                product_colors.extend(processed_value)
    
    # Check variant colors if available
    if 'variants' in product and product['variants']:
        for variant in product['variants']:
            if 'selectedOptions' in variant:
                for option in variant['selectedOptions']:
                    option_name = option.get('name', '').lower()
                    option_value = option.get('value', '')
                    
                    if ('color' in option_name or 'colour' in option_name) and option_value:
                        # If it's a hex color, add it directly
                        if option_value.startswith('#'):
                            product_colors.append(option_value)
                        # Otherwise add it as a text color for name matching
                        else:
                            color_key = f"variant_color_{option_value.lower().replace(' ', '_')}"
                            processed_data[color_key] = 1
    
    # If we have product colors, store in global dictionary
    if product_colors:
        # Remove duplicates
        unique_colors = list(set(product_colors))
        ALL_PRODUCT_COLORS[product_id] = unique_colors
        
        # Also store in processed data as a consolidated field
        processed_data['product_colors'] = unique_colors
    
    return processed_data, product_references

def calculate_color_similarities(top_n=10, output_file="color_similarity.csv"):
    """
    Calculate color similarities between all products and save to CSV
    
    Args:
        top_n: Number of similar products to keep for each product
        output_file: CSV filename to save the results
    
    Returns:
        DataFrame: DataFrame with color similarity data
    """
    global ALL_PRODUCT_COLORS
    
    if not ALL_PRODUCT_COLORS:
        print("No color data found. Skipping color similarity calculation.")
        return pd.DataFrame()
    
    print(f"Calculating color similarities for {len(ALL_PRODUCT_COLORS)} products...")
    
    # Prepare results storage
    similarity_data = []
    
    # Generate all product pairs
    product_ids = list(ALL_PRODUCT_COLORS.keys())
    
    for i, source_id in enumerate(product_ids):
        source_colors = ALL_PRODUCT_COLORS[source_id]
        product_similarities = []
        
        for j, target_id in enumerate(product_ids):
            # Skip comparing a product to itself
            if source_id == target_id:
                continue
                
            target_colors = ALL_PRODUCT_COLORS[target_id]
            sim = calculate_product_color_similarity(source_colors, target_colors)
            
            # Only include if similarity is above a threshold
            if sim > 0.2:  # Skip very dissimilar products
                product_similarities.append({
                    'source_product_id': source_id,
                    'target_product_id': target_id,
                    'color_similarity': sim
                })
        
        # Sort by similarity (descending) and keep top_n
        product_similarities.sort(key=lambda x: x['color_similarity'], reverse=True)
        top_similarities = product_similarities[:top_n]
        
        # Add to overall results
        similarity_data.extend(top_similarities)
    
    # Create DataFrame from similarity data
    similarity_df = pd.DataFrame(similarity_data)
    
    # Save to CSV
    if not similarity_df.empty:
        similarity_df.to_csv(output_file, index=False)
        print(f"Color similarity data saved to {output_file}")
        print(f"Generated {len(similarity_df)} color similarity pairs")
    else:
        print("No color similarity data generated.")
    
    return similarity_df

def apply_tfidf_processing(df):
    """
    Apply TF-IDF processing to text columns and normalize numeric columns
    Args:
        df: DataFrame with processed metafields
    Returns:
        DataFrame with TF-IDF vectorized text and normalized numeric features
    """
    # Skip processing if DataFrame is empty
    if df.empty:
        return df
        
    # Process color arrays first - separate out color columns
    color_columns = []
    for col in df.columns:
        if isinstance(col, str) and ('color' in col.lower() or 'colour' in col.lower()):
            if df[col].apply(lambda x: isinstance(x, list)).any():
                color_columns.append(col)
    
    # Keep all non-list columns for further processing
    non_color_df = df.drop(columns=color_columns, errors='ignore')
    
    # Normalize numeric features (including dimension and weight values)
    non_color_df = normalize_numeric_features(non_color_df)
    
    # Vectorize text features (except color, dimension, weight, product_reference)
    text_features = vectorize_text_features(non_color_df)
    
    # Combine all features
    result_dfs = [non_color_df]
    
    # Add back the color columns
    for col in color_columns:
        # Convert list columns to indicator variables
        # - If column has color arrays, keep as is
        result_dfs.append(pd.DataFrame({col: df[col]}))
    
    # Add text feature dataframes
    result_dfs.extend(text_features.values())
    
    if len(result_dfs) > 1:
        return pd.concat(result_dfs, axis=1)
    return df

def save_product_references(product_references, output_file="product_references.csv"):
    """
    Save product references to a separate CSV file
    """
    # Flatten the product references
    flat_references = []
    
    for rel_type, rel_data in product_references.items():
        source_id = rel_data['source_product_id']
        relationship = rel_data['relationship_type']
        
        for target_id in rel_data['referenced_product_ids']:
            flat_references.append({
                'source_product_id': source_id,
                'target_product_id': target_id,
                'relationship_type': relationship
            })
    
    # Create DataFrame
    if flat_references:
        df = pd.DataFrame(flat_references)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Product references saved to {output_file} ({len(flat_references)} relationships)")
        
        return df
    else:
        print(f"No product references found, CSV not created")
        return pd.DataFrame()

def save_color_similarity_data():
    """
    Calculate and save color similarity data from global product colors
    
    Returns:
        DataFrame with color similarity data
    """
    global ALL_PRODUCT_COLORS
    
    # Check if we have color data
    if not ALL_PRODUCT_COLORS:
        print("No product color data found for similarity calculation")
        return pd.DataFrame()
    
    # Calculate color similarities
    similarity_df = calculate_color_similarities(top_n=10, output_file="products_color_similarity.csv")
    
    # Log the results
    if not similarity_df.empty:
        print(f"Generated color similarity data for {similarity_df['source_product_id'].nunique()} products")
        print(f"Total similarity pairs: {len(similarity_df)}")
    else:
        print("No color similarity data generated")
    
    return similarity_df

def process_all_products(products_data, metafields_config=None, output_prefix=None):
    """
    Process all products in a dataset
    
    Args:
        products_data: List of product dictionaries or DataFrame
        metafields_config: Optional metafields configuration
        output_prefix: Prefix for output files
        
    Returns:
        Tuple of (processed_df, references_df, color_similarity_df)
    """
    global ALL_PRODUCT_COLORS
    # Clear the global colors dict before starting
    ALL_PRODUCT_COLORS.clear()
    
    all_processed_data = []
    all_product_references = {}
    
    # Convert DataFrame to list of dictionaries if needed
    if isinstance(products_data, pd.DataFrame):
        products_list = products_data.to_dict('records')
    else:
        products_list = products_data
    
    # Process each product
    for product in products_list:
        processed_data, product_references = process_metafields(product, metafields_config)
        
        # Add product ID to processed data
        product_id = product.get('product_id', None)
        if product_id:
            # Extract just the numeric part if it's a GID
            if isinstance(product_id, str) and 'gid://shopify/Product/' in product_id:
                product_id = product_id.split('/')[-1]
                
            processed_data['product_id'] = product_id
            
        # Add other essential product data
        for key in ['product_title', 'handle', 'vendor']:
            if key in product:
                processed_data[key] = product[key]
        
        all_processed_data.append(processed_data)
        
        # Collect all product references
        for key, value in product_references.items():
            all_product_references[f"{product_id}_{key}"] = value
    
    # Create DataFrame from processed data
    processed_df = pd.DataFrame(all_processed_data)
    
    # Calculate color similarities
    color_similarity_df = None
    if ALL_PRODUCT_COLORS:
        color_similarity_file = f"{output_prefix}_color_similarity.csv" if output_prefix else "products_color_similarity.csv"
        color_similarity_df = calculate_color_similarities(output_file=color_similarity_file)
    
    # Apply TF-IDF processing
    processed_df = apply_tfidf_processing(processed_df)
    
    # Save product references to separate CSV
    references_df = None
    if output_prefix and all_product_references:
        references_file = f"{output_prefix}_product_references.csv"
        references_df = save_product_references(all_product_references, references_file)
    
    # Save processed data
    if output_prefix:
        processed_df.to_csv(f"{output_prefix}_processed.csv", index=False)
        print(f"Processed data saved to {output_prefix}_processed.csv")
    
    return processed_df, references_df, color_similarity_df