import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

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
    """
    try:
        if not references or not isinstance(references, dict) or 'nodes' not in references:
            return ""
            
        nodes = references['nodes']
        colors = []
        
        for node in nodes:
            if 'fields' in node:
                # Check if there's a 'color' field directly
                for field in node['fields']:
                    if field.get('key') == 'color' and field.get('value', '').startswith('#'):
                        colors.append(field['value'])
                        break
        
        if colors:
            return " ".join(colors)
        return ""
        
    except Exception as e:
        print(f"Error extracting color values: {e}")
        return ""

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
            if color_values:
                return color_values
        
        # If we have direct references data with nodes, process it
        if references and isinstance(references, dict) and 'nodes' in references:
            nodes = references['nodes']
            
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
    
    # Handle different metafield types
    if is_dimension:
        # For dimension, return the numerical value in standard unit (mm)
        return process_dimension(value, include_unit=False)
    elif is_weight:
        # For weight, return the numerical value in standard unit (g)
        return process_weight(value, include_unit=False)
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
        max_features=100,  # Limit features to prevent dimensionality explosion
        stop_words='english',
        max_df=0.8,        # Ignore terms that appear in >80% of documents
        min_df=0.01        # Ignore terms that appear in <1% of documents  
    )
    
    for col in text_columns:
        # Convert column name to string to avoid errors with integer columns
        col_str = str(col)
        
        # Skip color, dimension, weight, and product reference columns
        if any(keyword in col_str.lower() for keyword in ['color', 'dimension', 'weight', 'product_reference']):
            continue
            
        # Skip columns with all empty/NA values
        if metafields_df[col].isna().all() or (metafields_df[col] == '').all():
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
    processed_data = {}
    product_references = {}
    
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
        is_color_related = 'color' in key.lower() or 'color' in field_type
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
                processed_data[unique_key] = colors
                continue
        
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
    
    return processed_data, product_references

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
        
    # Normalize numeric features (including dimension and weight values)
    df = normalize_numeric_features(df)
    
    # Vectorize text features (except color, dimension, weight, product_reference)
    text_features = vectorize_text_features(df)
    
    # Combine all features
    result_dfs = [df]
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

def process_all_products(products_data, metafields_config=None, output_prefix=None):
    """
    Process all products in a dataset
    
    Args:
        products_data: List of product dictionaries or DataFrame
        metafields_config: Optional metafields configuration
        output_prefix: Prefix for output files
        
    Returns:
        Tuple of (processed_df, references_df)
    """
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
    
    return processed_df, references_df