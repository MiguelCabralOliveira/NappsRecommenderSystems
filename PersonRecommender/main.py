# main.py
import argparse
import os
import pandas as pd
import json
import traceback

# --- Local Imports ---
from data_pipeline.data_loader import load_all_data
# Import the specific person data preprocessing orchestrator
from preprocessing.preprocess_person_data import run_person_preprocessing

# Placeholders for future steps
# from graph_builder import build_graph_data
# from model_trainer import train_model
# from embedding_generator import generate_embeddings
# from similarity_calculator import calculate_similarities

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Load, preprocess, and prepare user interaction data.')
    parser.add_argument('--shop-id', required=True, help='The tenant_id for the shop.')
    parser.add_argument('--output-dir', default='results', help='Base directory to save results.')
    parser.add_argument('--days', type=int, default=90, help='Number of past days of event data to load.')
    args = parser.parse_args()

    run_output_dir = os.path.join(args.output_dir, args.shop_id)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output directory: {run_output_dir}")

    print(f"\nStarting pipeline for shop_id: {args.shop_id}")

    # Define file paths consistently
    person_loaded_path = os.path.join(run_output_dir, "person_data_loaded.csv")
    event_loaded_path = os.path.join(run_output_dir, "person_event_data_filtered.csv")
    events_discovered_path = os.path.join(run_output_dir, "discovered_event_types.json")
    person_processed_path = os.path.join(run_output_dir, "person_features_processed.csv")
    event_processed_path = os.path.join(run_output_dir, "event_data_processed.csv") # Processed event data (still placeholder)

    # === Step 1: Load Data ===
    print("\n=== Step 1: Loading Data ===")
    person_df_loaded = None # Use distinct name
    event_df_loaded = None  # Use distinct name
    distinct_event_types = []
    try:
        person_df_loaded, event_df_loaded, distinct_event_types = load_all_data(args.shop_id, args.days)

        print("\n--- Data Loading Summary ---")
        print(f"Person records loaded: {len(person_df_loaded) if person_df_loaded is not None else 'None'}")
        print(f"Filtered Person event records loaded: {len(event_df_loaded) if event_df_loaded is not None else 'None'}")
        print(f"Distinct event types found (all): {len(distinct_event_types) if distinct_event_types else 'None'}")

        # Save distinct event types
        if distinct_event_types:
            with open(events_discovered_path, 'w') as f: json.dump(distinct_event_types, f, indent=4)
            print(f"Discovered event types saved to {events_discovered_path}")

        # --- Validation after loading ---
        loading_successful = True
        if person_df_loaded is None or person_df_loaded.empty:
            print("Warning: Person data loading resulted in None or empty DataFrame.")
            # loading_successful = False # Decide if critical

        if event_df_loaded is None or event_df_loaded.empty:
            print("Warning: Filtered Event data loading resulted in None or empty DataFrame.")
            loading_successful = False # Event data is likely essential

        if not loading_successful:
            print("Stopping pipeline due to missing essential data after loading.")
            return

        # --- Save Loaded Data ---
        print("\n--- Saving Loaded Data ---")
        person_df_loaded.to_csv(person_loaded_path, index=False)
        print(f"Loaded person data saved to {person_loaded_path}")
        event_df_loaded.to_csv(event_loaded_path, index=False)
        print(f"Filtered person event data saved to {event_loaded_path}")

    except Exception as e:
        print(f"\n--- Critical Error during Step 1 (Data Loading): {e} ---")
        traceback.print_exc()
        return

    # === Step 2: Preprocessing Person Data ===
    print("\n=== Step 2: Preprocessing Person Data ===")
    try:
        # Ensure input file exists before calling preprocessing
        if not os.path.exists(person_loaded_path):
             raise FileNotFoundError(f"Person loaded file not found for preprocessing: {person_loaded_path}")

        # Call the person data preprocessing orchestrator
        run_person_preprocessing(
            input_path=person_loaded_path,
            output_path=person_processed_path,
        )
        print("Person data preprocessing step completed.")

        # Optional: Check if output file was created
        if not os.path.exists(person_processed_path):
            print(f"Warning: Processed person file was not created at {person_processed_path}")
            raise RuntimeError("Processed person file missing after preprocessing step.")

    except Exception as e:
        print(f"\n--- Critical Error during Step 2 (Person Preprocessing): {e} ---")
        traceback.print_exc()
        return # Stop pipeline if preprocessing fails

    # === Step 2b: Preprocessing Event Data (Placeholder) ===
    print("\n=== Step 2b: Preprocessing Event Data (Placeholder) ===")
    try:
        if not os.path.exists(event_loaded_path):
             print(f"Warning: Event loaded file not found at {event_loaded_path}, skipping placeholder processing.")
        else:
            # Placeholder: Copy event data for now
            # Replace this with a call to an event data preprocessor when ready
            print(f"Placeholder: Copying event data from {event_loaded_path} to {event_processed_path}")
            shutil.copyfile(event_loaded_path, event_processed_path)
            print("Event data placeholder processing (copy) complete.")
            if not os.path.exists(event_processed_path):
                 print(f"Warning: Processed event file was not created at {event_processed_path}")

    except Exception as e:
        print(f"\n--- Error during Step 2b (Event Preprocessing Placeholder): {e} ---")
        traceback.print_exc()
        # Decide if this is critical - likely not if it's just a placeholder

    # === Step 3: Graph Data Preparation (Placeholder) ===
    print("\n=== Step 3: Graph Data Preparation (Placeholder) ===")
    # This step would now conceptually load/use:
    # - person_processed_path
    # - event_processed_path (currently a copy of filtered data)
    print(f"Input for this step would be: {person_processed_path}, {event_processed_path}")
    # node_maps, edge_data, node_counts, interaction_details_df = build_graph_data(person_processed_path, event_processed_path, ...)
    # ...
    print("Skipping Graph Data Preparation.")


    # === Step 4: Model Training (Placeholder) ===
    print("\n=== Step 4: Model Training (Placeholder) ===")
    # Uses artifacts from Step 3
    print("Skipping Model Training.")


    # === Step 5: Embedding Generation (Placeholder) ===
    print("\n=== Step 5: Embedding Generation (Placeholder) ===")
    # Uses artifacts from Step 3 & 4
    print("Skipping Embedding Generation.")


    # === Step 6: Similarity Calculation (Placeholder) ===
    print("\n=== Step 6: Similarity Calculation (Placeholder) ===")
    # Uses embeddings from Step 5
    print("Skipping Similarity Calculation.")


    print("\n=== Pipeline Finished ===")


if __name__ == "__main__":
    # Need to import shutil for the placeholder copy in main
    import shutil
    main()