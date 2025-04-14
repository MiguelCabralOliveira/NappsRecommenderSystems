# preprocessing/product_processor/collectionHandle_processor.py
import pandas as pd
import json
import traceback

def safe_load_collections(collections_data):
    """Safely load collections data, which might be stringified JSON or list."""
    if pd.isna(collections_data):
        return []
    if isinstance(collections_data, list):
        return collections_data # Already a list
    if isinstance(collections_data, str):
        try:
            # Try parsing as JSON
            loaded = json.loads(collections_data.replace("'", '"'))
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            # Try splitting if it's a simple comma-separated string (less likely for collections)
            # if ',' in collections_data:
            #     return [c.strip() for c in collections_data.split(',')]
            # If single value string? Assume it's a single collection title
            return [{'title': collections_data}] # Wrap in expected structure
        except Exception: # Catch other potential errors during loading
             traceback.print_exc()
             return []
    return [] # Return empty list for other types or errors


def process_collections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process collection titles using dummy variables.
    Handles collections stored as stringified lists of dicts or actual lists.

    Args:
        df: pandas DataFrame containing product data.

    Returns:
        DataFrame: processed DataFrame with collection dummies.
    """
    print("Processing: collections (Creating Dummies)")
    if 'collections' not in df.columns:
        print("Warning: 'collections' column not found. Skipping.")
        return df

    # Make sure we're working with a copy to avoid modifying the original df directly
    df_processed = df.copy()

    try:
        # 1. Safely load and parse the collections data
        # This Series will contain lists of dictionaries or empty lists
        parsed_collections = df_processed['collections'].apply(safe_load_collections)

        # 2. Extract titles - handle potential missing 'title' key
        def get_collection_titles(collection_list):
            titles = []
            if isinstance(collection_list, list):
                for collection_dict in collection_list:
                    if isinstance(collection_dict, dict):
                        title = collection_dict.get('title')
                        if title and isinstance(title, str) and title.strip():
                            titles.append(title.strip())
                    elif isinstance(collection_dict, str) and collection_dict.strip(): # Handle if it's just a list of strings
                        titles.append(collection_dict.strip())
            return titles

        # Apply title extraction; result is a Series of lists of titles
        df_processed['collection_titles'] = parsed_collections.apply(get_collection_titles)

        # 3. Explode the lists of titles to create one row per product-collection pair
        exploded_collections = df_processed[['collection_titles']].explode('collection_titles')

        # Filter out rows where collection_titles might be NaN or empty after explode
        exploded_collections = exploded_collections.dropna(subset=['collection_titles'])
        exploded_collections = exploded_collections[exploded_collections['collection_titles'] != '']


        if exploded_collections.empty:
            print("No valid collection titles found after parsing and exploding.")
            df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')
            return df_processed

        # 4. Create dummy variables
        collection_dummies = pd.get_dummies(
            exploded_collections['collection_titles'],
            prefix='collections',
            prefix_sep='_',
            dtype=int
        )

        # 5. Aggregate back to product level using maximum value (presence indicator)
        # We need to group by the original DataFrame index
        collection_dummies = collection_dummies.groupby(level=0).max()

        # 6. Drop the original 'collections' and intermediate 'collection_titles' column
        df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')

        # 7. Concatenate the dummy columns to the processed DataFrame
        # Use pd.concat which aligns based on index
        result_df = pd.concat([df_processed, collection_dummies], axis=1)

        # Fill NaN values in newly added dummy columns with 0
        dummy_cols = collection_dummies.columns
        result_df[dummy_cols] = result_df[dummy_cols].fillna(0).astype(int)

        print(f"Created {len(dummy_cols)} collection dummy features.")
        return result_df

    except Exception as e:
        print(f"Error processing collections: {e}")
        traceback.print_exc()
        # Return the DataFrame without the collection dummies in case of error
        df_processed = df_processed.drop(columns=['collections', 'collection_titles'], errors='ignore')
        return df_processed