# preprocessing/product_processor/tags_processor.py
import pandas as pd
import json
import traceback
import ast # For literal_eval as a fallback

def safe_load_tags(tags_data):
    """Safely load tags data, which might be stringified list or actual list."""
    if pd.isna(tags_data):
        return []
    if isinstance(tags_data, list):
        # Ensure all elements are strings
        return [str(tag).strip() for tag in tags_data if isinstance(tag, (str, int, float)) and str(tag).strip()]
    if isinstance(tags_data, str):
        # Common case: comma-separated string
        if ',' in tags_data and '[' not in tags_data: # Avoid interpreting JSON array string as CSV
            return [tag.strip() for tag in tags_data.split(',') if tag.strip()]
        # Try parsing as JSON list of strings
        try:
            loaded = json.loads(tags_data.replace("'", '"'))
            if isinstance(loaded, list):
                 # Ensure all elements are strings
                return [str(tag).strip() for tag in loaded if isinstance(tag, (str, int, float)) and str(tag).strip()]
            else:
                 # If it parsed but wasn't a list, treat original string as single tag
                 return [tags_data.strip()] if tags_data.strip() else []
        except json.JSONDecodeError:
            # Try ast.literal_eval as fallback for Python list string representation
            try:
                eval_loaded = ast.literal_eval(tags_data)
                if isinstance(eval_loaded, list):
                    return [str(tag).strip() for tag in eval_loaded if isinstance(tag, (str, int, float)) and str(tag).strip()]
                else:
                    # If eval worked but wasn't list, treat original string as single tag
                    return [tags_data.strip()] if tags_data.strip() else []
            except (ValueError, SyntaxError, TypeError):
                 # If all parsing fails, treat the whole string as a single tag
                 return [tags_data.strip()] if tags_data.strip() else []
        except Exception: # Catch other potential errors during loading
             traceback.print_exc()
             return [tags_data.strip()] if tags_data.strip() else [] # Fallback to single tag
    # Handle other types (e.g., numbers) by converting to string tag
    if isinstance(tags_data, (int, float)):
        return [str(tags_data)]

    return [] # Return empty list for other types or errors

def process_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process tags using dummy variables. Handles tags stored as strings or lists.

    Args:
        df: pandas DataFrame containing product data.

    Returns:
        DataFrame: processed DataFrame with tag dummies.
    """
    print("Processing: tags (Creating Dummies)")
    if 'tags' not in df.columns:
        print("Warning: 'tags' column not found. Skipping.")
        return df

    df_processed = df.copy()

    try:
        # 1. Safely load tags into lists of strings
        df_processed['parsed_tags'] = df_processed['tags'].apply(safe_load_tags)

        # 2. Explode the lists to get one row per product-tag pair
        exploded_tags = df_processed[['parsed_tags']].explode('parsed_tags')

        # Filter out rows where parsed_tags might be NaN or empty string after explode
        exploded_tags = exploded_tags.dropna(subset=['parsed_tags'])
        exploded_tags = exploded_tags[exploded_tags['parsed_tags'] != '']

        if exploded_tags.empty:
            print("No valid tags found after parsing and exploding.")
            df_processed = df_processed.drop(columns=['tags', 'parsed_tags'], errors='ignore')
            return df_processed

        # 3. Create dummy variables for tags
        tag_dummies = pd.get_dummies(
            exploded_tags['parsed_tags'],
            prefix='tag',
            prefix_sep='_',
            dtype=int
        )

        # 4. Aggregate back to product level using maximum value (presence indicator)
        # Group by the original DataFrame index
        tag_dummies = tag_dummies.groupby(level=0).max()

        # 5. Drop the original 'tags' and intermediate 'parsed_tags' column
        df_processed = df_processed.drop(columns=['tags', 'parsed_tags'], errors='ignore')

        # 6. Concatenate the dummy columns to the processed DataFrame
        result_df = pd.concat([df_processed, tag_dummies], axis=1)

        # Fill NaN values in newly added dummy columns with 0
        dummy_cols = tag_dummies.columns
        result_df[dummy_cols] = result_df[dummy_cols].fillna(0).astype(int)

        print(f"Created {len(dummy_cols)} tag dummy features.")
        return result_df

    except Exception as e:
        print(f"Error processing tags: {e}")
        traceback.print_exc()
        # Return the DataFrame without tag dummies in case of error
        df_processed = df_processed.drop(columns=['tags', 'parsed_tags'], errors='ignore')
        return df_processed