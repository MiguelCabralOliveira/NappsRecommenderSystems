# preprocessing/product_processor/collectionHandle_processor.py
import pandas as pd
import json
import traceback
import ast # Importar ast para literal_eval

def safe_load_collections(collections_data):
    """
    Safely load collections data, which might be a stringified list of dicts,
    an actual list, or other formats. Prioritizes ast.literal_eval for string parsing.
    """
    if pd.isna(collections_data):
        return []
    if isinstance(collections_data, list):
        # If already a list, ensure it contains dicts or convert strings if simple
        processed_list = []
        for item in collections_data:
            if isinstance(item, dict):
                processed_list.append(item)
            elif isinstance(item, str): # Handle list of strings case
                 processed_list.append({'title': item.strip()}) # Assume string is title
        return processed_list
    if isinstance(collections_data, str):
        try:
            # Prioritize ast.literal_eval for strings like '[{...}, {...}]'
            loaded = ast.literal_eval(collections_data)
            if isinstance(loaded, list):
                 # Ensure list contains dicts
                 return [item for item in loaded if isinstance(item, dict)]
            else:
                 # If eval resulted in something else, maybe treat original string as title? Unlikely for collections.
                 return []
        except (ValueError, SyntaxError, TypeError, MemoryError):
             # If literal_eval fails, fallback to json.loads with cleaning
            try:
                # Basic cleaning for JSON compatibility
                cleaned_str = collections_data.replace("'", '"')
                # Handle None, True, False specifically for JSON
                cleaned_str = cleaned_str.replace(': None', ': null').replace(': True', ': true').replace(': False', ': false')
                loaded = json.loads(cleaned_str)
                if isinstance(loaded, list):
                    # Ensure list contains dicts
                    return [item for item in loaded if isinstance(item, dict)]
                else:
                    return [] # Parsed but not a list
            except json.JSONDecodeError:
                # If all parsing fails, assume it's a single collection title string? Less likely.
                # print(f"Could not parse collection string: {collections_data[:100]}...")
                # title = collections_data.strip()
                # return [{'title': title}] if title else []
                return [] # Safer to return empty if cannot parse structure
            except Exception as e_json:
                # Catch other JSON errors
                # print(f"Other JSON error parsing collection string: {e_json}")
                return []
        except Exception as e_ast:
            # Catch other literal_eval errors
            # print(f"Other AST error parsing collection string: {e_ast}")
            return []
    # If it's not a list or string (or parsing failed), return empty
    return []


def get_collection_titles(collection_list):
    """Extracts 'title' strings from a list of collection dictionaries."""
    titles = []
    if isinstance(collection_list, list):
        for collection_dict in collection_list:
            # Ensure it's a dictionary and has a 'title' key with a non-empty string value
            if isinstance(collection_dict, dict):
                title = collection_dict.get('title')
                # Check if title is a non-empty string
                if title and isinstance(title, str) and title.strip():
                    titles.append(title.strip())
    return titles

def process_collections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process collection titles using dummy variables. Handles collections stored
    as stringified lists of dicts or actual lists. Correctly parses titles.

    Args:
        df: pandas DataFrame containing product data with 'collections' column.

    Returns:
        DataFrame: processed DataFrame with collection dummies based on titles.
    """
    print("Processing: collections (Creating Dummies - Corrected Parsing)")
    if 'collections' not in df.columns:
        print("Warning: 'collections' column not found. Skipping.")
        return df

    df_processed = df.copy()

    try:
        # 1. Safely load and parse the collections data using the improved function
        parsed_collections = df_processed['collections'].apply(safe_load_collections)

        # 2. Extract titles using the corrected function
        df_processed['collection_titles'] = parsed_collections.apply(get_collection_titles)

        # Check if any titles were actually extracted before exploding
        if df_processed['collection_titles'].apply(len).sum() == 0:
             print("No collection titles were extracted after parsing. Skipping dummy creation.")
             df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')
             return df_processed

        # 3. Explode the lists of titles
        exploded_collections = df_processed[['collection_titles']].explode('collection_titles')

        # Filter out rows where collection_titles might be NaN or empty after explode
        exploded_collections = exploded_collections.dropna(subset=['collection_titles'])
        exploded_collections = exploded_collections[exploded_collections['collection_titles'] != '']

        if exploded_collections.empty:
            print("No valid collection titles remaining after exploding and cleaning.")
            df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')
            return df_processed

        # 4. Create dummy variables using the CORRECT titles
        # Let pandas handle potential special characters in titles safely during get_dummies
        collection_dummies = pd.get_dummies(
            exploded_collections['collection_titles'],
            prefix='coll', # Shorter prefix
            prefix_sep='_',
            dtype=int
        )

        print(f"Unique collection titles found: {collection_dummies.shape[1]}")

        # 5. Aggregate back to product level
        collection_dummies = collection_dummies.groupby(level=0).max()

        # 6. Drop the original 'collections' and intermediate 'collection_titles' column
        df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')

        # 7. Concatenate the dummy columns
        result_df = pd.concat([df_processed, collection_dummies], axis=1)

        # Fill NaN values in newly added dummy columns with 0
        dummy_cols = collection_dummies.columns
        result_df[dummy_cols] = result_df[dummy_cols].fillna(0).astype(int)

        print(f"Created {len(dummy_cols)} CORRECTED collection dummy features.")
        return result_df

    except Exception as e:
        print(f"Error processing collections: {e}")
        traceback.print_exc()
        # Return the DataFrame without the collection dummies in case of error
        df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')
        return df_processed