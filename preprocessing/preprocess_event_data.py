# preprocessing/preprocess_event_data.py
import argparse
import os
import pandas as pd
# import numpy as np # Não parece ser usado diretamente aqui
import traceback
# import re # Não parece ser usado diretamente aqui
import sys
# import json # Não parece ser usado diretamente aqui
# from sklearn.preprocessing import RobustScaler # Não usado aqui
# import joblib # Não usado aqui
# from tqdm import tqdm # Já está em process_event_interactions

# --- Import the core processing function and utils ---
try:
    from .event_processor.process_event_interactions import extract_interactions_from_events
    from .event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS as FINAL_INTERACTION_COLUMNS_CSV
    # from .utils import safe_json_parse # Não usado diretamente aqui
except ImportError:
    print("Warning: Could not use relative imports for event processor/utils. Attempting direct import.")
    try:
        from event_processor.process_event_interactions import extract_interactions_from_events
        from event_processor.process_event_interactions import FINAL_OUTPUT_COLUMNS as FINAL_INTERACTION_COLUMNS_CSV
        # from utils import safe_json_parse
    except ModuleNotFoundError as e:
         print(f"Import Error: {e}. Failed to import event_processor module or utils.")
         sys.exit(1)

# A função _prepare_gnn_edge_data foi removida daqui, pois prepare_edge_data.py será usado.

# --- Função Principal ---
def run_event_preprocessing(
    input_path: str,
    output_path: str # Caminho para o CSV intermediário
    # Os argumentos relacionados à preparação GNN foram removidos daqui
):
    """
    Orchestrates the preprocessing of event data to extract interactions and
    saves the intermediate CSV. GNN edge preparation is now handled by a separate script.
    """
    print(f"\n--- Starting Event Data Preprocessing Orchestration (CSV Extraction Only) ---")
    print(f"Input file: {input_path}")
    print(f"Intermediate Output CSV: {output_path}")

    interactions_df = pd.DataFrame()

    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        event_df = pd.read_csv(input_path, low_memory=False)
        if event_df.empty:
            print("Input event DataFrame is empty. Cannot proceed.")
            # Salvar CSV vazio se o input estiver vazio
            output_dir_csv = os.path.dirname(output_path)
            if output_dir_csv: os.makedirs(output_dir_csv, exist_ok=True)
            pd.DataFrame(columns=FINAL_INTERACTION_COLUMNS_CSV).to_csv(output_path, index=False, na_rep='')
            print(f"Empty intermediate event CSV saved to: {output_path}")
            return # Retorna se o input estiver vazio
    except Exception as e:
        print(f"Error loading input event CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    try:
        print("\nExtracting and processing interactions...")
        interactions_df = extract_interactions_from_events(event_df)
    except Exception as e:
        print(f"Error during event interaction extraction: {e}")
        traceback.print_exc()
        raise

    try:
        output_dir_csv = os.path.dirname(output_path)
        os.makedirs(output_dir_csv, exist_ok=True)
        print(f"\nEnsured intermediate output directory exists: {output_dir_csv}")

        if interactions_df.empty:
             print("\nInteraction DataFrame is empty after processing. Saving empty intermediate file.")
             pd.DataFrame(columns=FINAL_INTERACTION_COLUMNS_CSV).to_csv(output_path, index=False, na_rep='')
        else:
            print("\nData types in final DataFrame before save:")
            print(interactions_df.info())
            interactions_df.to_csv(output_path, index=False, na_rep='')
            print(f"Successfully saved intermediate processed event interactions to: {output_path}")

    except Exception as e:
        print(f"Error saving intermediate event interactions CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

    print(f"--- Finished Event Data Preprocessing Orchestration (CSV Extraction Only) ---")


# --- Command Line Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess event data and extract interactions to CSV.')
    parser.add_argument('--input', required=True, help='Path to the input person_event_data_filtered.csv file.')
    parser.add_argument('--output-csv', required=True, help='Path to save the intermediate event_interactions_processed.csv file.')
    # Argumentos GNN removidos do CLI deste script

    args = parser.parse_args()

    try:
        run_event_preprocessing(
            args.input,
            args.output_csv
        )
        print("\nEvent preprocessing (CSV extraction) finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error:
         print(f"\nError: {val_error}")
         exit(1)
    except Exception as e:
        print(f"\n--- Event preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)