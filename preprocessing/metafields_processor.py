import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

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

def extract_metaobject_reference_data(reference_value):
    """
    Extract color hex values from metaobject references
    Specifically looks for values starting with # in hex format
    """
    try:
        # Handle string values that need to be parsed
        if isinstance(reference_value, str):
            try:
                data = json.loads(reference_value)
            except json.JSONDecodeError:
                return reference_value
        else:
            data = reference_value
            
        collected_values = []
        
        # Handle array of references (list.metaobject_reference)
        if isinstance(data, list):
            # Return empty if list is empty
            if not data:
                return ""
                
            # If list contains simple strings (likely IDs), just return joined
            if all(isinstance(item, str) for item in data):
                return " ".join(data)
        
        # Extract from references structure
        if isinstance(data, dict) or isinstance(data, list):
            # Access references data structure
            references_data = None
            if isinstance(data, dict) and 'references' in data and 'nodes' in data['references']:
                references_data = data['references']['nodes']
            elif isinstance(data, list) and 'references' in data[0] and 'nodes' in data[0]['references']:
                references_data = data[0]['references']['nodes']
                
            # Process the nodes if we found them
            if references_data:
                for node in references_data:
                    if 'fields' in node:
                        for field in node['fields']:
                            # Look specifically for values that look like hex colors
                            value = field.get('value', '')
                            if isinstance(value, str) and value.startswith('#'):
                                collected_values.append(value)
                            # For nested objects like "color" field
                            elif field.get('key') == 'color' and isinstance(value, str):
                                collected_values.append(value)
        
        # Return the collected color values as a space-separated string
        if collected_values:
            return " ".join(collected_values)
        else:
            # Fallback to original string if no color values found
            return str(data)
            
    except Exception as e:
        print(f"Error extracting data from metaobject reference: {e}")
        return str(reference_value)
def clean_html(html_content):
    """Clean HTML from text content"""
    if not html_content or not isinstance(html_content, str):
        return ""
        
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html_content

def process_metafield_value(metafield):
    """Process a single metafield based on its type"""
    if not metafield or 'value' not in metafield:
        return None
        
    metafield_type = metafield.get('type', '')
    value = metafield['value']
    
    # Handle different metafield types
    if metafield_type == 'rich_text_field':
        return extract_text_from_rich_text(value)
    elif metafield_type in ('list.metaobject_reference', 'metaobject_reference'):
        return extract_metaobject_reference_data(value)
    elif metafield_type == 'multi_line_text_field':
        return clean_html(value)
    elif metafield_type in ('single_line_text_field', 'url'):
        return str(value)
    elif metafield_type in ('number_integer', 'number_decimal', 'rating'):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    elif metafield_type == 'boolean':
        if isinstance(value, bool):
            return 1 if value else 0
        try:
            return 1 if value.lower() in ('true', 't', 'yes', 'y', '1') else 0
        except (ValueError, AttributeError):
            return 0
    elif metafield_type == 'json':
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
        Dictionary with processed metafield data

    Basicly what is happening here is that we are taking the metafields from the product and separating them into different columns.

    This method basicly handles all the diferent metafield types and returns a dictionary with the processed metafield data.
    """
    processed_data = {}
    
    if 'metafields' not in product or not product['metafields']:
        return processed_data
    
    # Create a mapping of metafield keys to their types from config
    metafield_types = {}
    if metafields_config:
        for field in metafields_config:
            key = f"{field.get('namespace', '')}_{field.get('key', '')}"
            metafield_types[key] = field.get('type', '')

    
    
    # Process each metafield
    for metafield in product['metafields']:
        if not metafield or 'key' not in metafield or 'value' not in metafield:
            continue
            
        # Create a unique key using namespace and key
        namespace = metafield.get('namespace', '')
        key = metafield.get('key', '')
        unique_key = f"{namespace}_{key}"
        
        # Add type information from config if available
        if unique_key in metafield_types:
            metafield['type'] = metafield_types[unique_key]
            
        # Process the metafield value
        processed_value = process_metafield_value(metafield)
        processed_data[unique_key] = processed_value
    
    return processed_data

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
        
    # Normalize numeric features
    df = normalize_numeric_features(df)
    
    # Vectorize text features
    text_features = vectorize_text_features(df)
    
    # Combine all features
    result_dfs = [df]
    result_dfs.extend(text_features.values())
    
    if len(result_dfs) > 1:
        return pd.concat(result_dfs, axis=1)
    return df