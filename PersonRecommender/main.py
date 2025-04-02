# main.py
import argparse
import os
import pandas as pd
import json
import traceback

# --- Local Imports ---
from data_pipeline.data_loader import load_all_data
from data_pipeline.data_processor import process_loaded_data 
# Placeholders for future steps
# from graph_builder import build_graph_data
# from model_trainer import train_model
# from embedding_generator import generate_embeddings
# from similarity_calculator import calculate_similarities

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Load, process, and prepare user interaction data.')
    parser.add_argument('--shop-id', required=True, help='The tenant_id for the shop.')
    parser.add_argument('--output-dir', default='results', help='Base directory to save results.')
    parser.add_argument('--days', type=int, default=90, help='Number of past days of event data to load.')
    args = parser.parse_args()

    run_output_dir = os.path.join(args.output_dir, args.shop_id)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output directory: {run_output_dir}")

    print(f"\nStarting pipeline for shop_id: {args.shop_id}")

    # === Step 1: Load Data ===
    print("\n=== Step 1: Loading Data ===")
    try:
        raw_person_df, raw_event_df, distinct_event_types = load_all_data(args.shop_id, args.days)
        print("\n--- Raw Data Loading Summary ---")
        print(f"Raw Person records loaded: {len(raw_person_df) if raw_person_df is not None else 'None'}")
        print(f"Raw Person event records loaded: {len(raw_event_df) if raw_event_df is not None else 'None'}")
        print(f"Distinct event types found: {len(distinct_event_types) if distinct_event_types else 'None'}")

        # Save distinct event types (doesn't need processing)
        if distinct_event_types:
            events_path = os.path.join(run_output_dir, "discovered_event_types.json")
            with open(events_path, 'w') as f: json.dump(distinct_event_types, f, indent=4)
            print(f"Discovered event types saved to {events_path}")

    except Exception as e:
        print(f"\n--- Critical Error during Step 1 (Data Loading): {e} ---")
        traceback.print_exc()
        return # Stop pipeline if loading fails

    # === Step 2: Initial Data Processing ===
    print("\n=== Step 2: Initial Data Processing ===")
    try:
        person_df, event_df = process_loaded_data(raw_person_df, raw_event_df)

        # Check if processing was successful
        processing_successful = True
        if person_df is None or person_df.empty:
            print("Warning: Person data processing resulted in None or empty DataFrame.")
            # Decide if this is critical - maybe allow proceeding if only event data is needed?
            # processing_successful = False # Uncomment if person data is essential

        if event_df is None or event_df.empty:
            print("Warning: Event data processing resulted in None or empty DataFrame. Cannot proceed.")
            processing_successful = False # Event data is essential for the graph

        if not processing_successful:
            print("Stopping pipeline due to errors or missing data after initial processing.")
            return

        # Save the processed dataframes
        print("\n--- Processed Data Summary ---")
        if person_df is not None:
            print(f"Processed Person records: {len(person_df)}")
            person_processed_path = os.path.join(run_output_dir, "person_data_processed.csv")
            person_df.to_csv(person_processed_path, index=False)
            print(f"Processed person data saved to {person_processed_path}")
            # print("\nProcessed Person DataFrame Head:")
            # print(person_df.head()) # Optional: reduce verbosity

        if event_df is not None:
            print(f"Processed Person event records: {len(event_df)}")
            event_processed_path = os.path.join(run_output_dir, "person_event_data_processed.csv")
            event_df.to_csv(event_processed_path, index=False)
            print(f"Processed person event data saved to {event_processed_path}")
            # print("\nProcessed Person Event DataFrame Head:")
            # print(event_df.head()) # Optional: reduce verbosity

    except Exception as e:
        print(f"\n--- Critical Error during Step 2 (Initial Processing): {e} ---")
        traceback.print_exc()
        return # Stop pipeline if processing fails

    # === Step 3: Graph Data Preparation (Placeholder) ===
    print("\n=== Step 3: Graph Data Preparation (Placeholder) ===")
    # node_maps, edge_data, node_counts, interaction_details_df = build_graph_data(person_df, event_df, distinct_event_types)
    # if node_maps is None:
    #     print("Graph building failed. Exiting.")
    #     return
    # # Save graph artifacts (mappings, edge summary, interactions)
    # print("Graph data prepared (Placeholder).")
    print("Skipping Graph Data Preparation.")


    # === Step 4: Model Training (Placeholder) ===
    print("\n=== Step 4: Model Training (Placeholder) ===")
    # trained_model_path = train_model(node_maps, edge_data, node_counts, run_output_dir)
    # if trained_model_path is None:
    #     print("Model training failed. Exiting.")
    #     return
    # print(f"Model trained (Placeholder): {trained_model_path}")
    print("Skipping Model Training.")


    # === Step 5: Embedding Generation (Placeholder) ===
    print("\n=== Step 5: Embedding Generation (Placeholder) ===")
    # user_embeddings_df = generate_embeddings(trained_model_path, node_maps, edge_data, node_counts, run_output_dir)
    # if user_embeddings_df is None:
    #     print("Embedding generation failed. Exiting.")
    #     return
    # print(f"Embeddings generated (Placeholder): {len(user_embeddings_df)} users.")
    print("Skipping Embedding Generation.")


    # === Step 6: Similarity Calculation (Placeholder) ===
    print("\n=== Step 6: Similarity Calculation (Placeholder) ===")
    # calculate_similarities(user_embeddings_df, run_output_dir)
    # print("Similarities calculated (Placeholder).")
    print("Skipping Similarity Calculation.")


    print("\n=== Pipeline Finished (Initial Steps Complete) ===")


if __name__ == "__main__":
    main()