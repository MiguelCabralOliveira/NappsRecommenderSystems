import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_metafield_keys():
    """Load metafield keys from JSON configuration"""
    with open('shopifyInfo/metafiled_keys.json', 'r') as f:
        return json.load(f)

def process_metafield_value(value, field_type):
    """Process metafield value based on its type"""
    if not value:
        return None
    
    try:
        if field_type == "product_reference":
            return value
        else:
            try:
                json_value = json.loads(value)
                if isinstance(json_value, dict) and 'value' in json_value:
                    value = json_value['value']
                else:
                    value = json_value
                
                # Process numerical values
                if isinstance(value, (int, float)):
                    scaler = MinMaxScaler()
                    normalized_value = scaler.fit_transform([[float(value)]])[0][0]
                    return normalized_value
                # Process string values
                elif isinstance(value, str):
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([value])
                    return tfidf_matrix.toarray()[0]
                return value
            except json.JSONDecodeError:
                if isinstance(value, str):
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([value])
                    return tfidf_matrix.toarray()[0]
                return value
    except:
        return value

def process_metafields(product, metafields):
    """Process all metafields for a product"""
    metafields_data = {}
    if 'metafields' in product and product['metafields']:
        for i, metafield in enumerate(product['metafields']):
            if metafield and 'value' in metafield:
                field_type = metafields[i]['type']
                processed_value = process_metafield_value(metafield['value'], field_type)
                metafields_data[metafields[i]['key']] = processed_value
            else:
                metafields_data[metafields[i]['key']] = None
    return metafields_data