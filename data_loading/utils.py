import os
import pandas as pd
import traceback
# Try importing optional S3 dependencies, handle if missing
try:
    import s3fs
    import fsspec
    _S3FS_AVAILABLE = True
except ImportError:
    _S3FS_AVAILABLE = False
    print("Info: 's3fs'/'fsspec' not installed. S3 saving support will be unavailable.")


def save_dataframe(df: pd.DataFrame, full_path: str):
    """
    Helper function to save dataframes to local path or S3 URI.
    Requires s3fs and fsspec installed for S3 support.
    """
    if df is None:
        print(f"Skipping save for {os.path.basename(full_path)}: DataFrame is None (error likely occurred before save).")
        return # Nothing to save

    if df.empty:
        print(f"Warning: DataFrame for {os.path.basename(full_path)} is empty. Saving empty file to {full_path}")
        # Decide if saving empty files is desired. If not, uncomment the 'return' below.
        # return

    is_s3_path = full_path.startswith("s3://")

    try:
        # Ensure directory exists for local paths before saving
        if not is_s3_path:
            output_dir = os.path.dirname(full_path)
            if output_dir: # Only create if path includes a directory
                 os.makedirs(output_dir, exist_ok=True)
        elif not _S3FS_AVAILABLE:
             # If it's an S3 path but library is missing, raise error before trying pandas
             raise ImportError("Attempting to save to S3, but 's3fs'/'fsspec' libraries are not installed.")

        # Pandas uses s3fs automatically for s3:// paths if s3fs is installed
        print(f"Attempting to save DataFrame ({len(df)} rows) to {full_path}...")
        df.to_csv(full_path, index=False, encoding='utf-8') # Specify encoding
        print(f"Successfully saved {os.path.basename(full_path)} to {full_path}")

    except ImportError as ie:
         # This specifically catches the explicit check above for S3
         print(f"ERROR: Missing dependency for S3: {ie}")
         print("Ensure 's3fs' and 'fsspec' are included in requirements.txt and installed.")
         # Re-raise or handle as appropriate for your workflow
         # raise ie
    except Exception as e:
        print(f"ERROR saving {os.path.basename(full_path)} to {full_path}: {e}")
        print("Check path validity, permissions (local or S3 bucket policy/credentials), and available disk space.")
        traceback.print_exc()
        # Re-raise or handle as appropriate for your workflow
        # raise e