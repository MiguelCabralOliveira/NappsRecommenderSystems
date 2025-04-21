# training/similarity/utils.py

import pandas as pd
import json
import os
import traceback
import joblib # Using joblib as it was used in the original WeightedKNN

def load_dataframe(file_path: str, required_cols: list = None) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file with error handling and validation.

    Args:
        file_path: Path to the CSV file.
        required_cols: Optional list of column names that must be present.

    Returns:
        Loaded pandas DataFrame, or an empty DataFrame if loading fails or
        required columns are missing.
    """
    if not file_path or not isinstance(file_path, str):
        print("Error: Invalid file path provided for loading DataFrame.")
        return pd.DataFrame()

    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"Loading DataFrame from: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded {len(df)} rows.")

        if df.empty:
            print("Warning: Loaded DataFrame is empty.")
            return df

        # Validate required columns if specified
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Error: Loaded DataFrame from {file_path} is missing required columns: {missing_cols}")
                return pd.DataFrame() # Return empty if validation fails

        return df

    except pd.errors.EmptyDataError:
        print(f"Warning: File at {file_path} is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading DataFrame from {file_path}: {type(e).__name__} - {e}")
        traceback.print_exc()
        return pd.DataFrame()

def load_json_config(file_path: str) -> dict | None:
    """
    Loads a JSON configuration file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dictionary with the configuration, or None if loading fails.
    """
    if not file_path or not isinstance(file_path, str):
        print("Error: Invalid file path provided for loading JSON config.")
        return None

    if not os.path.exists(file_path):
        print(f"Warning: JSON config file not found at {file_path}. Returning None.")
        return None

    print(f"Loading JSON config from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("Successfully loaded JSON config.")
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading JSON config from {file_path}: {type(e).__name__} - {e}")
        traceback.print_exc()
        return None

def ensure_dir(directory_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if directory_path:
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")
            # Decide if this should raise an error or just warn

def save_model_component(component, file_path: str, compress: int = 3):
    """Saves a model component using joblib."""
    if component is not None:
        try:
            ensure_dir(os.path.dirname(file_path))
            joblib.dump(component, file_path, compress=compress)
            # print(f"Saved component to {file_path}") # Optional: reduce verbosity
            return True
        except Exception as e:
            print(f"Error saving component to {file_path}: {e}")
            return False
    return False # Nothing to save

def load_model_component(file_path: str):
    """Loads a model component using joblib."""
    if os.path.exists(file_path):
        try:
            component = joblib.load(file_path)
            # print(f"Loaded component from {file_path}") # Optional: reduce verbosity
            return component
        except Exception as e:
            print(f"Error loading component from {file_path}: {e}")
            return None
    else:
        # print(f"Component file not found at {file_path}") # Optional: reduce verbosity
        return None

def save_dataframe(df: pd.DataFrame, file_path: str):
    """Saves a DataFrame to a CSV file."""
    if df is not None: # Allow saving empty DataFrames if needed
        try:
            ensure_dir(os.path.dirname(file_path))
            df.to_csv(file_path, index=False)
            print(f"DataFrame with {len(df)} rows saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving DataFrame to {file_path}: {e}")
            return False
    else:
        print("Warning: Attempted to save a None DataFrame.")
        return False

def save_json_report(data: dict, file_path: str):
    """Saves a dictionary (like evaluation results) to a JSON file."""
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"JSON report saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON report to {file_path}: {e}")
        return False

def save_text_report(content: str, file_path: str):
    """Saves text content (like a summary) to a file."""
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Text report saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving text report to {file_path}: {e}")
        return False