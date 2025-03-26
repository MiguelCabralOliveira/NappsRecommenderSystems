import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os
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

# Track which columns are text fields that need TF-IDF processing and removal
TEXT_FIELD_COLUMNS = []

# Track which columns contain color data and should be removed after similarity processing
COLOR_COLUMNS = []

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

def is_text_field_in_metaobject(metaobject):
    """
    Check if the metaobject has text fields that should be processed with TF-IDF
    Returns True if the first field has a text type
    """
    if not metaobject or not isinstance(metaobject, dict) or 'fields' not in metaobject:
        return False
    
    # Check if fields is a list with at least one item
    if not metaobject['fields'] or not isinstance(metaobject['fields'], list):
        return False
    
    # Get the first field
    first_field = metaobject['fields'][0]
    
    # Check if it's a text field
    field_type = first_field.get('type', '').lower()
    return 'single_line_text_field' in field_type or 'multi_line_text_field' in field_type or 'rich_text_field' in field_type

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

def extract_metaobject_reference_data(reference_value, references=None, metaobject_type=None, unique_key=None):
    """
    Extract data from metaobject references
    Args:
        reference_value: The metaobject reference value
        references: The references data structure if available
        metaobject_type: The type of the metaobject if known
        unique_key: The unique key for this metafield
    """
    global TEXT_FIELD_COLUMNS, COLOR_COLUMNS
    
    try:
        # Check if this is a color-related metaobject
        is_color_type = metaobject_type and 'color' in metaobject_type.lower()
        
        # If it's a color type and we have references, extract hex colors
        if is_color_type and references:
            color_values = extract_hex_colors_from_metaobject(references)
            # Return color values as-is (as array) instead of joining them
            if color_values:
                # Track this column as one containing color data
                if unique_key:
                    COLOR_COLUMNS.append(unique_key)
                return color_values
        
        # If we have direct references data with nodes, process it
        if references and isinstance(references, dict) and 'nodes' in references:
            nodes = references['nodes']
            
            # Check if any of the nodes have text fields that need TF-IDF processing
            has_text_fields = any(is_text_field_in_metaobject(node) for node in nodes)
            
            # Check if this might be a color metaobject
            might_be_color = any('color' in node.get('type', '').lower() for node in nodes if 'type' in node)
            
            # Extract values from the fields of each node
            field_values = []
            has_color_values = False
            
            for node in nodes:
                if 'fields' in node and node['fields']:
                    # Check for color fields in this node
                    for field in node['fields']:
                        if field.get('key', '').lower() == 'color' and field.get('value', '').startswith('#'):
                            has_color_values = True
                            break
                            
                    # Get the label field as priority if it exists
                    label_field = next((f for f in node['fields'] if f.get('key') == 'label'), None)
                    
                    if label_field and 'value' in label_field:
                        field_values.append(label_field['value'])
                    # If no label field, get the first field with a value
                    elif len(node['fields']) > 0 and 'value' in node['fields'][0]:
                        field_values.append(node['fields'][0]['value'])
            
            if field_values:
                # If this appears to be a color metaobject, track the column
                if might_be_color or has_color_values:
                    if unique_key:
                        COLOR_COLUMNS.append(unique_key)
                
                # If this metaobject reference has text fields, mark it for TF-IDF processing
                if has_text_fields and unique_key:
                    TEXT_FIELD_COLUMNS.append(unique_key)
                
                # Return a list for single values (this preserves categorical nature)
                # or join with commas for multiple values
                if len(field_values) == 1:
                    return field_values[0]
                return ", ".join(field_values)
        
        # Parse the reference value if it's a string
        if isinstance(reference_value, str):
            try:
                data = json.loads(reference_value)
                
                # If it's a list of IDs, return empty string - we can't process without references
                if isinstance(data, list) and all(isinstance(item, str) and 'gid://shopify/Metaobject/' in item for item in data):
                    return ""
                
                return str(data)
            except json.JSONDecodeError:
                # Return empty string for single metaobject references without data
                if 'gid://shopify/Metaobject/' in reference_value:
                    return ""
                return reference_value
        else:
            data = reference_value
            
        # If it's a list of metaobject IDs without resolved data, return empty string
        if isinstance(data, list) and all(isinstance(item, str) and 'gid://shopify/Metaobject/' in item for item in data):
            return ""
            
        return str(data) if data else ""
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
    global TEXT_FIELD_COLUMNS, COLOR_COLUMNS
    
    if not metafield or 'value' not in metafield:
        return None
        
    metafield_type = metafield.get('type', '').lower()
    metafield_key = metafield.get('key', '')
    namespace = metafield.get('namespace', '')
    unique_key = f"{namespace}_{metafield_key}"
    value = metafield['value']
    
    # Check if this is a dimension or weight field by type or key
    is_dimension = 'dimension' in metafield_type or 'dimension' in metafield_key.lower()
    is_weight = 'weight' in metafield_type or 'weight' in metafield_key.lower()
    is_color = 'color' in metafield_type or 'color' in metafield_key.lower() or 'color' in namespace.lower()
    
    # Track text fields for later TF-IDF processing and removal
    is_text_field = 'multi_line_text_field' in metafield_type or 'single_line_text_field' in metafield_type or 'rich_text_field' in metafield_type
    if is_text_field:
        TEXT_FIELD_COLUMNS.append(unique_key)
    
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
        # Track this color column
        COLOR_COLUMNS.append(unique_key)
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
        if 'color' in metafield_key.lower() or 'colour' in metafield_key.lower():
            metaobject_type = 'color'
            # Track this column as potentially containing color data
            COLOR_COLUMNS.append(unique_key)
        elif references and 'nodes' in references:
            for node in references['nodes']:
                if 'type' in node and 'color' in node['type'].lower():
                    metaobject_type = node['type']
                    # Track this column as potentially containing color data
                    COLOR_COLUMNS.append(unique_key)
                    break
                    
        return extract_metaobject_reference_data(value, references, metaobject_type, unique_key)
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
    global TEXT_FIELD_COLUMNS
    
    text_features = {}
    text_columns = [col for col in metafields_df.columns if col in TEXT_FIELD_COLUMNS]
    
    if len(text_columns) == 0:
        return {}
    
    print(f"Applying TF-IDF to {len(text_columns)} text columns: {text_columns}")
    
    # Group text columns based on content characteristics
    short_text_columns = []
    long_text_columns = []
    
    for col in text_columns:
        # Skip if column doesn't exist (might have been removed)
        if col not in metafields_df.columns:
            continue
            
        # Fill NA values with empty string
        text_data = metafields_df[col].fillna('').astype(str)
        
        # Skip if all values are empty after conversion
        if (text_data == '').all():
            continue
        
        # Check if this is likely a short text field (like from metaobjects)
        # by checking average text length and number of unique values
        avg_length = text_data.str.len().mean()
        unique_ratio = text_data.nunique() / len(text_data)
        
        if avg_length < 100 or unique_ratio < 0.5:
            short_text_columns.append(col)
        else:
            long_text_columns.append(col)
    
    # Process short text columns (like fabric types, labels, etc.) using word counts instead of TF-IDF
    if short_text_columns:
        for col in short_text_columns:
            text_data = metafields_df[col].fillna('').astype(str)
            
            # For short text fields, just create indicator variables
            # This is more reliable than TF-IDF for categorical text data from metaobjects
            unique_values = text_data.unique()
            
            # Remove empty values
            unique_values = [v for v in unique_values if v]
            
            # Create indicator variables with metafield_ prefix
            for value in unique_values:
                value_safe = re.sub(r'\W+', '_', value.lower())[:30]  # create safe column name
                feature_name = f"metafield_{col}_{value_safe}"  # Add metafield_ prefix
                text_features[feature_name] = (text_data == value).astype(int)
    
    # Process long text columns using TF-IDF as before
    if long_text_columns:
        # Combine all long text fields for better corpus statistics
        combined_text = metafields_df[long_text_columns].fillna('').apply(
            lambda row: ' '.join(row.astype(str)), axis=1
        )
        
        # Initialize vectorizer with more flexible parameters for small datasets
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            max_df=0.95,  # More permissive max_df
            min_df=1      # Allow terms that appear in just one document
        )
        
        try:
            # Apply TF-IDF to combined text
            feature_matrix = vectorizer.fit_transform(combined_text)
            # Add metafield_ prefix to feature names
            feature_names = [f"metafield_{name}" for name in vectorizer.get_feature_names_out()]
            
            text_features['combined_text'] = pd.DataFrame(
                feature_matrix.toarray(),
                columns=feature_names,
                index=metafields_df.index
            )
        except Exception as e:
            print(f"Error vectorizing combined text fields: {e}")
    
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
    global ALL_PRODUCT_COLORS, TEXT_FIELD_COLUMNS, COLOR_COLUMNS
    
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
        is_color_related = ('color' in key.lower() or 'colour' in key.lower() or 
                           'color' in namespace.lower() or 'colour' in namespace.lower() or
                           'color' in field_type.lower())
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
                
                # Track this column as containing color data
                if unique_key not in COLOR_COLUMNS:
                    COLOR_COLUMNS.append(unique_key)
                
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
                        
                        # Track this column as containing color data
                        if unique_key not in COLOR_COLUMNS:
                            COLOR_COLUMNS.append(unique_key)
                            
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
                            
                            # Track this column as containing color data
                            if unique_key not in COLOR_COLUMNS:
                                COLOR_COLUMNS.append(unique_key)
                                
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
                nodes = metafield['references']['nodes']
                
                # Check if any node has a text field that should be processed with TF-IDF
                has_text_fields = any(is_text_field_in_metaobject(node) for node in nodes)
                if has_text_fields:
                    TEXT_FIELD_COLUMNS.append(unique_key)
                    
                # Check if any node appears to be related to colors
                is_color_metaobject = False
                for node in nodes:
                    if 'type' in node and ('color' in node['type'].lower() or 'colour' in node['type'].lower()):
                        is_color_metaobject = True
                        break
                    
                    # Check fields for color values
                    if 'fields' in node:
                        for field in node['fields']:
                            if field.get('key', '').lower() in ['color', 'colour', 'hex'] and isinstance(field.get('value', ''), str):
                                if field['value'].startswith('#') or re.match(r'^[0-9A-Fa-f]{6}$', field['value']):
                                    is_color_metaobject = True
                                    break
                
                if is_color_metaobject and unique_key not in COLOR_COLUMNS:
                    COLOR_COLUMNS.append(unique_key)
                    
                values = []
                for node in nodes:
                    if 'fields' in node and node['fields']:
                        # Get the label field as priority if it exists
                        label_field = next((f for f in node['fields'] if f.get('key') == 'label'), None)
                        
                        if label_field and 'value' in label_field:
                            values.append(label_field['value'])
                        # If no label field, get the first field with a value
                        elif len(node['fields']) > 0 and 'value' in node['fields'][0]:
                            values.append(node['fields'][0]['value'])
                
                if values:
                    processed_data[unique_key] = ", ".join(values)
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
                            
                            # Track this column as potentially color-related
                            if color_key not in COLOR_COLUMNS:
                                COLOR_COLUMNS.append(color_key)
    
    # If we have product colors, store in global dictionary
    if product_colors:
        # Remove duplicates
        unique_colors = list(set(product_colors))
        ALL_PRODUCT_COLORS[product_id] = unique_colors
        
        # Also store in processed data as a consolidated field
        processed_data['product_colors'] = unique_colors
        
        # Track the consolidated product_colors column
        if 'product_colors' not in COLOR_COLUMNS:
            COLOR_COLUMNS.append('product_colors')
    
    return processed_data, product_references

def apply_tfidf_processing(df):
    """
    Apply TF-IDF processing to text columns and normalize numeric columns
    Args:
        df: DataFrame with processed metafields
    Returns:
        DataFrame with TF-IDF vectorized text and normalized numeric features
    """
    global TEXT_FIELD_COLUMNS
    
    # Skip processing if DataFrame is empty
    if df.empty:
        return df
    
    # First, make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Normalize numeric features (including dimension and weight values)
    numeric_columns = df.select_dtypes(include=np.number).columns
    
    if len(numeric_columns) > 0:
        # Handle missing values before scaling
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Use robust scaling for better handling of outliers
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Vectorize text features for all text columns in TEXT_FIELD_COLUMNS
    text_features = vectorize_text_features(df)
    
    # Handle the case where text features are returned as Series instead of DataFrames
    for key, feature_data in list(text_features.items()):
        if isinstance(feature_data, pd.Series):
            # Convert Series to DataFrame
            text_features[key] = pd.DataFrame(feature_data, columns=[key])
    
    # Combine all features
    result_dfs = [df]
    
    for feature_name, feature_df in text_features.items():
        if isinstance(feature_df, pd.DataFrame):
            result_dfs.append(feature_df)
    
    # Create result DataFrame with all features combined
    if len(result_dfs) > 1:
        result_df = pd.concat(result_dfs, axis=1)
    else:
        result_df = df
    
    # Remove the original text columns that were processed
    columns_to_drop = [col for col in result_df.columns if col in TEXT_FIELD_COLUMNS]
    if columns_to_drop:
        print(f"Removing {len(columns_to_drop)} text columns that were processed with TF-IDF: {columns_to_drop}")
        result_df = result_df.drop(columns=columns_to_drop)
    
    return result_df

def calculate_color_similarities(top_n=10, output_dir="results", output_file="color_similarity.csv"):
    """
    Calculate color similarities between all products and save to CSV
    
    Args:
        top_n: Number of similar products to keep for each product
        output_dir: Output directory
        output_file: CSV filename to save the results
    
    Returns:
        DataFrame: DataFrame with color similarity data
    """
    global ALL_PRODUCT_COLORS
    
    if not ALL_PRODUCT_COLORS:
        print("No color data found. Skipping color similarity calculation.")
        return pd.DataFrame()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    full_path = f"{output_dir}/{output_file}"
    
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
        similarity_df.to_csv(full_path, index=False)
        print(f"Color similarity data saved to {full_path}")
        print(f"Generated {len(similarity_df)} color similarity pairs")
    else:
        print("No color similarity data generated.")
    
    return similarity_df

def save_color_similarity_data(output_dir="results"):
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate color similarities
    similarity_df = calculate_color_similarities(top_n=10, output_file=f"{output_dir}/products_color_similarity.csv")
    
    # Log the results
    if not similarity_df.empty:
        print(f"Generated color similarity data for {similarity_df['source_product_id'].nunique()} products")
        print(f"Total similarity pairs: {len(similarity_df)}")
    else:
        print("No color similarity data generated")
    
    return similarity_df

def remove_color_columns(df):
    """
    Remove all color-related columns from the dataframe
    
    Args:
        df: DataFrame with processed metafields
        
    Returns:
        DataFrame with color columns removed
    """
    global COLOR_COLUMNS
    
    if not COLOR_COLUMNS:
        print("No color columns detected for removal.")
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Get columns that exist in the dataframe
    cols_to_remove = [col for col in COLOR_COLUMNS if col in df.columns]
    
    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} color columns after similarity processing: {cols_to_remove}")
        df = df.drop(columns=cols_to_remove)
    else:
        print("No color columns found in dataframe to remove.")
    
    return df

def save_product_references(product_references, output_dir="results", output_file="product_references.csv"):
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        full_path = f"{output_dir}/{output_file}"
        df.to_csv(full_path, index=False)
        print(f"Product references saved to {full_path} ({len(flat_references)} relationships)")
        
        return df
    else:
        print(f"No product references found, CSV not created")
        return pd.DataFrame()
    
def process_all_products(products_data, metafields_config=None, output_dir="results", output_prefix=None):
    """
    Process all products in a dataset
    
    Args:
        products_data: List of product dictionaries or DataFrame
        metafields_config: Optional metafields configuration
        output_dir: Directory for output files
        output_prefix: Prefix for output files
        
    Returns:
        Tuple of (processed_df, references_df, color_similarity_df)
    """
    global ALL_PRODUCT_COLORS, TEXT_FIELD_COLUMNS, COLOR_COLUMNS
    # Clear the global colors dict and text columns list before starting
    ALL_PRODUCT_COLORS.clear()
    TEXT_FIELD_COLUMNS.clear()
    COLOR_COLUMNS.clear()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        color_similarity_df = calculate_color_similarities(output_file=color_similarity_file, output_dir=output_dir)
    
    # After calculating color similarities, remove color columns
    processed_df = remove_color_columns(processed_df)
    
    # Apply TF-IDF processing - this will remove text columns after processing them
    processed_df = apply_tfidf_processing(processed_df)
    
    # Save product references to separate CSV
    references_df = None
    if output_prefix and all_product_references:
        references_file = f"{output_prefix}_product_references.csv"
        references_df = save_product_references(all_product_references, output_dir, references_file)
    
    # Save processed data
    if output_prefix:
        processed_df.to_csv(f"{output_dir}/{output_prefix}_processed.csv", index=False)
        print(f"Processed data saved to {output_dir}/{output_prefix}_processed.csv")
    
    return processed_df, references_df, color_similarity_df