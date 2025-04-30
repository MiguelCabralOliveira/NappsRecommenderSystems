# /training/graph/dataset.py

import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import joblib # To load scalers

# --- Helper Functions to Load Artifacts ---

def load_json(path):
    """Loads a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def load_numpy(path):
    """Loads a NumPy file (.npy)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NumPy file not found: {path}")
    return np.load(path)

def load_npz(path):
    """Loads a NumPy archive file (.npz)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NumPy archive not found: {path}")
    return np.load(path)

def load_joblib(path):
    """Loads a Joblib file (e.g., scaler)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Joblib file not found: {path}")
    return joblib.load(path)

# --- Main Data Loading Function ---

def load_hetero_graph_data(gnn_data_dir: str, device: torch.device = torch.device('cpu')):
    """
    Loads all preprocessed GNN data artifacts for nodes and edges (train/valid/test splits)
    and assembles them into PyTorch Geometric HeteroData objects.

    Args:
        gnn_data_dir (str): The root directory where the GNN data was saved
                             (e.g., 'results/{shop_id}/gnn_data').
        device (torch.device): The device to load the tensors onto ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
            - data_dict (dict[str, HeteroData]): Dictionary mapping 'train', 'valid', 'test'
                                                 to their respective HeteroData objects.
                                                 Contains node features and edge indices/features
                                                 relevant to each split based on temporal split.
            - metadata (tuple): Metadata tuple (node_types, edge_types_found) derived from the data.
                                Use this for initializing HeteroGNN models.
            - node_feature_info (dict): Info about node features (e.g., num features, vocabs).
            - edge_feature_info (dict): Info about edge features (e.g., vocabs, scalers).
    """
    print(f"--- Loading Heterogeneous Graph Data from: {gnn_data_dir} ---")

    # Define paths based on gnn_data_dir and expected subdirectories
    person_node_dir = os.path.join(gnn_data_dir, 'person_nodes')
    product_node_dir = os.path.join(gnn_data_dir, 'product_nodes')
    edge_data_dir = os.path.join(gnn_data_dir, 'edges')

    # --- 1. Load Node Data ---
    print("Loading node data...")
    node_data = {}
    node_feature_info = {'Person': {}, 'Product': {}}

    # Person Nodes
    try:
        node_feature_info['Person']['id_map'] = load_json(os.path.join(person_node_dir, 'person_id_to_index_map.json'))
        node_feature_info['Person']['num_nodes'] = len(node_feature_info['Person']['id_map'])
        print(f"  Loaded Person ID map ({node_feature_info['Person']['num_nodes']} nodes).")

        # Load numeric features if they exist
        numeric_feat_path = os.path.join(person_node_dir, 'person_features_numeric_scaled.npy')
        if os.path.exists(numeric_feat_path):
            node_data[('Person', 'x_numeric')] = torch.from_numpy(load_numpy(numeric_feat_path)).float().to(device)
            numeric_cols = load_json(os.path.join(person_node_dir, 'person_numeric_cols_order.json'))
            node_feature_info['Person']['num_numeric_features'] = node_data[('Person', 'x_numeric')].shape[1]
            node_feature_info['Person']['numeric_cols'] = numeric_cols
            print(f"    Loaded Person numeric features ({node_feature_info['Person']['num_numeric_features']} features).")
        else:
             node_feature_info['Person']['num_numeric_features'] = 0
             node_data[('Person', 'x_numeric')] = torch.empty((node_feature_info['Person']['num_nodes'], 0), dtype=torch.float, device=device) # Empty tensor
             print("    No Person numeric features found.")

        # Load categorical features if they exist
        categorical_feat_path = os.path.join(person_node_dir, 'person_features_categorical_indices.npy')
        if os.path.exists(categorical_feat_path):
            node_data[('Person', 'x_categorical')] = torch.from_numpy(load_numpy(categorical_feat_path)).long().to(device)
            node_feature_info['Person']['categorical_vocabs'] = load_json(os.path.join(person_node_dir, 'person_categorical_vocabs.json'))
            categorical_cols = load_json(os.path.join(person_node_dir, 'person_categorical_cols_order.json'))
            node_feature_info['Person']['categorical_cols'] = categorical_cols
            node_feature_info['Person']['num_categorical_features'] = len(categorical_cols)
            print(f"    Loaded Person categorical features ({node_feature_info['Person']['num_categorical_features']} features).")
            # Validate shape
            if node_data[('Person', 'x_categorical')].shape[0] != node_feature_info['Person']['num_nodes']:
                 raise ValueError("Person categorical feature rows mismatch number of nodes.")
            if node_data[('Person', 'x_categorical')].shape[1] != len(categorical_cols):
                 raise ValueError("Person categorical feature columns mismatch number of columns in order.")
        else:
             node_feature_info['Person']['num_categorical_features'] = 0
             node_feature_info['Person']['categorical_vocabs'] = {}
             node_feature_info['Person']['categorical_cols'] = []
             node_data[('Person', 'x_categorical')] = torch.empty((node_feature_info['Person']['num_nodes'], 0), dtype=torch.long, device=device) # Empty tensor
             print("    No Person categorical features found.")

    except FileNotFoundError as e:
        print(f"Error loading Person node data: {e}. Cannot proceed.")
        raise
    except Exception as e:
        print(f"Unexpected error loading Person node data: {e}")
        raise

    # Product Nodes
    try:
        node_feature_info['Product']['id_map'] = load_json(os.path.join(product_node_dir, 'product_id_to_index_map.json'))
        node_feature_info['Product']['num_nodes'] = len(node_feature_info['Product']['id_map'])
        print(f"  Loaded Product ID map ({node_feature_info['Product']['num_nodes']} nodes).")

        # Load numeric features if they exist
        numeric_feat_path = os.path.join(product_node_dir, 'product_features_numeric_scaled.npy')
        if os.path.exists(numeric_feat_path):
            node_data[('Product', 'x_numeric')] = torch.from_numpy(load_numpy(numeric_feat_path)).float().to(device)
            numeric_cols = load_json(os.path.join(product_node_dir, 'product_numeric_cols_order.json'))
            node_feature_info['Product']['num_numeric_features'] = node_data[('Product', 'x_numeric')].shape[1]
            node_feature_info['Product']['numeric_cols'] = numeric_cols
            print(f"    Loaded Product numeric features ({node_feature_info['Product']['num_numeric_features']} features).")
        else:
             node_feature_info['Product']['num_numeric_features'] = 0
             node_data[('Product', 'x_numeric')] = torch.empty((node_feature_info['Product']['num_nodes'], 0), dtype=torch.float, device=device) # Empty tensor
             print("    No Product numeric features found.")

        # Load categorical features if they exist
        categorical_feat_path = os.path.join(product_node_dir, 'product_features_categorical_indices.npy')
        if os.path.exists(categorical_feat_path) and os.path.getsize(categorical_feat_path) > 0:
            cat_indices_np = load_numpy(categorical_feat_path)
            if cat_indices_np.size > 0: # Check if array is not empty
                 node_data[('Product', 'x_categorical')] = torch.from_numpy(cat_indices_np).long().to(device)
                 node_feature_info['Product']['categorical_vocabs'] = load_json(os.path.join(product_node_dir, 'product_categorical_vocabs.json'))
                 categorical_cols = load_json(os.path.join(product_node_dir, 'product_categorical_cols_order.json'))
                 node_feature_info['Product']['categorical_cols'] = categorical_cols
                 node_feature_info['Product']['num_categorical_features'] = len(categorical_cols)
                 print(f"    Loaded Product categorical features ({node_feature_info['Product']['num_categorical_features']} features).")
                 # Validate shape
                 if node_data[('Product', 'x_categorical')].shape[0] != node_feature_info['Product']['num_nodes']:
                      raise ValueError("Product categorical feature rows mismatch number of nodes.")
                 if node_data[('Product', 'x_categorical')].shape[1] != len(categorical_cols):
                     raise ValueError("Product categorical feature columns mismatch number of columns in order.")
            else:
                 # Handle case where file exists but array is empty (e.g., shape (N, 0))
                 node_feature_info['Product']['num_categorical_features'] = 0
                 node_feature_info['Product']['categorical_vocabs'] = {}
                 node_feature_info['Product']['categorical_cols'] = []
                 node_data[('Product', 'x_categorical')] = torch.empty((node_feature_info['Product']['num_nodes'], 0), dtype=torch.long, device=device)
                 print("    Product categorical feature file was empty.")
        else:
             node_feature_info['Product']['num_categorical_features'] = 0
             node_feature_info['Product']['categorical_vocabs'] = {}
             node_feature_info['Product']['categorical_cols'] = []
             node_data[('Product', 'x_categorical')] = torch.empty((node_feature_info['Product']['num_nodes'], 0), dtype=torch.long, device=device)
             print("    No Product categorical features found.")

    except FileNotFoundError as e:
        print(f"Error loading Product node data: {e}. Cannot proceed.")
        raise
    except Exception as e:
        print(f"Unexpected error loading Product node data: {e}")
        raise

    # --- 2. Load Edge Data (Indices and Features per Split) ---
    print("\nLoading edge data (train/valid/test)...")
    data_dict = {'train': HeteroData(), 'valid': HeteroData(), 'test': HeteroData()}
    edge_feature_info = {} # Store edge feature vocabs/scalers once
    edge_types_found = set() # Keep track of all unique edge types encountered

    # Load edge feature metadata if it exists (only needs to be loaded once)
    edge_numeric_scaler = None
    edge_numeric_cols = []
    edge_cat_vocabs = {}
    edge_cat_cols = []
    edge_feat_dir = edge_data_dir # Root edge directory

    numeric_scaler_path = os.path.join(edge_feat_dir, 'edge_numeric_scaler.joblib')
    if os.path.exists(numeric_scaler_path):
        edge_numeric_scaler = load_joblib(numeric_scaler_path)
        edge_numeric_cols = load_json(os.path.join(edge_feat_dir, 'edge_numeric_cols_order.json'))
        edge_feature_info['numeric_scaler'] = edge_numeric_scaler
        edge_feature_info['numeric_cols'] = edge_numeric_cols
        print("  Loaded edge numeric feature scaler and column order.")

    cat_vocabs_path = os.path.join(edge_feat_dir, 'edge_categorical_vocabs.json')
    if os.path.exists(cat_vocabs_path):
        edge_cat_vocabs = load_json(cat_vocabs_path)
        edge_cat_cols = load_json(os.path.join(edge_feat_dir, 'edge_categorical_cols_order.json'))
        edge_feature_info['categorical_vocabs'] = edge_cat_vocabs
        edge_feature_info['categorical_cols'] = edge_cat_cols
        print("  Loaded edge categorical feature vocabs and column order.")

    # Load data for each split
    for split in ['train', 'valid', 'test']:
        print(f"  Loading {split} split...")
        split_dir = os.path.join(edge_data_dir, split)
        data = data_dict[split]

        # Add node features (same features for all splits, point to the same tensors)
        for (node_type, feature_key), tensor in node_data.items():
             # Assign features directly using node_type as the key
             if feature_key == 'x_numeric':
                 data[node_type].x_numeric = tensor
             elif feature_key == 'x_categorical':
                 data[node_type].x_categorical = tensor
             # Add num_nodes information
             data[node_type].num_nodes = node_feature_info[node_type]['num_nodes']

        # Load edge indices
        edge_index_path = os.path.join(split_dir, 'edge_index.npz')
        if not os.path.exists(edge_index_path):
            print(f"    Warning: edge_index.npz not found for {split} split. Split will have no edges.")
            continue # Skip to next split if no edges file

        edge_indices_npz = load_npz(edge_index_path)

        # --- DEBUG POINT 5 ---
        print(f"\nDEBUG: Keys found in {split} edge_index.npz: {list(edge_indices_npz.files)}")
        # --- FIM DEBUG ---

        for edge_key_str in edge_indices_npz.files:
            # Reconstruct the edge type tuple from the string key
            try:
                # Use eval carefully or a safer parsing method if keys are complex
                # Assuming keys are simple tuples like "('Person', 'viewed', 'Product')"
                edge_type = eval(edge_key_str)
                if not (isinstance(edge_type, tuple) and len(edge_type) == 3):
                     raise ValueError(f"Invalid edge type key format: {edge_key_str}")
                edge_types_found.add(edge_type) # Track unique edge types
            except:
                print(f"    Warning: Could not parse edge type key '{edge_key_str}'. Skipping.")
                continue

            edge_index = torch.from_numpy(edge_indices_npz[edge_key_str]).long().to(device)
            data[edge_type].edge_index = edge_index
            print(f"      Loaded edge index for {edge_type} ({edge_index.shape[1]} edges).")

        # Load edge features (if they exist for this split)
        edge_features_path = os.path.join(split_dir, 'edge_features.npz')
        if os.path.exists(edge_features_path):
            edge_features_npz = load_npz(edge_features_path)
            print(f"    Loading edge features for {split} split...")
            for feat_key_str in edge_features_npz.files:
                # Extract edge type and feature type (numeric/categorical) from key
                try:
                    parts = feat_key_str.split(')_') # Split between tuple string and type
                    edge_type_str = parts[0] + ')'
                    feature_type = parts[1]
                    edge_type = eval(edge_type_str)
                    if not (isinstance(edge_type, tuple) and len(edge_type) == 3):
                         raise ValueError(f"Invalid edge type key format in features: {edge_type_str}")
                except:
                    print(f"    Warning: Could not parse edge feature key '{feat_key_str}'. Skipping.")
                    continue

                # Assign features based on type
                if feature_type == 'numeric':
                    feat_tensor = torch.from_numpy(edge_features_npz[feat_key_str]).float().to(device)
                    data[edge_type].edge_attr_numeric = feat_tensor
                    print(f"      Loaded numeric edge features for {edge_type} ({feat_tensor.shape}).")
                elif feature_type == 'categorical':
                    feat_tensor = torch.from_numpy(edge_features_npz[feat_key_str]).long().to(device)
                    data[edge_type].edge_attr_categorical = feat_tensor
                    print(f"      Loaded categorical edge features for {edge_type} ({feat_tensor.shape}).")

    # --- 3. Finalize Metadata ---
    node_types_found = list(node_feature_info.keys()) # ['Person', 'Product']
    # Convert set of edge types to a list for metadata
    edge_types_list = sorted(list(edge_types_found)) # Sort for consistent order
    metadata = (node_types_found, edge_types_list)
    print(f"\nFinal Graph Metadata:")
    print(f"  Node Types: {metadata[0]}")
    print(f"  Edge Types: {metadata[1]}")


    print(f"--- Heterogeneous Graph Data Loading Finished ---")
    return data_dict, metadata, node_feature_info, edge_feature_info

# --- Example Usage (if run directly) ---
# (O bloco __main__ permanece o mesmo)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Load preprocessed GNN data.")
    parser.add_argument('--data-dir', required=True, help="Path to the root GNN data directory (e.g., 'results/shop_id/gnn_data')")
    parser.add_argument('--device', default='cpu', help="Device to load data onto ('cpu' or 'cuda')")
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        data_dict, metadata, node_info, edge_info = load_hetero_graph_data(args.data_dir, device)

        # Print some basic info about the loaded data
        print("\n--- Loaded Data Summary ---")
        for split, data in data_dict.items():
            print(f"\nSplit: {split}")
            print(data)
            # Check if specific keys exist and print shapes
            if 'Person' in data and hasattr(data['Person'], 'x_numeric'):
                print(f"  Person numeric features shape: {data['Person'].x_numeric.shape}")
            if 'Product' in data and hasattr(data['Product'], 'x_numeric'):
                print(f"  Product numeric features shape: {data['Product'].x_numeric.shape}")
            if 'Person' in data and hasattr(data['Person'], 'x_categorical'):
                print(f"  Person categorical features shape: {data['Person'].x_categorical.shape}")
            if 'Product' in data and hasattr(data['Product'], 'x_categorical'):
                 print(f"  Product categorical features shape: {data['Product'].x_categorical.shape}")

            if metadata[1]: # Check if edge_types_list is not empty
                 for edge_type in metadata[1]: # Iterate through found edge types
                      if edge_type in data:
                           print(f"  Edge type {edge_type}:")
                           print(f"    edge_index shape: {data[edge_type].edge_index.shape}")
                           if hasattr(data[edge_type], 'edge_attr_numeric'):
                               print(f"    edge_attr_numeric shape: {data[edge_type].edge_attr_numeric.shape}")
                           if hasattr(data[edge_type], 'edge_attr_categorical'):
                                print(f"    edge_attr_categorical shape: {data[edge_type].edge_attr_categorical.shape}")
            else:
                 print("  No edge types found in metadata for detailed breakdown.")


        print("\nNode Feature Info:")
        print(json.dumps(node_info, indent=2, default=str)) # Use default=str for non-serializable items if any
        print("\nEdge Feature Info:")
        print(json.dumps(edge_info, indent=2, default=str))

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()