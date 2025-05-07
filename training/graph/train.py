# /training/graph/train.py

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import logging

# Assume .model, .dataset, .evaluate, .utils are in the same directory or accessible
from .model import HeteroGNNLinkPredictor
from .dataset import load_hetero_graph_data
# Importar ambas as funções de evaluate
from .evaluate import calculate_auc, calculate_ranking_metrics
# Importar utils
from .utils import set_seed, setup_logging, save_config, load_config, plot_tsne_embeddings


# --- Argument Parsing ---
# (parse_args function - Nenhuma alteração necessária aqui)
def parse_args():
    parser = argparse.ArgumentParser(description="Train Heterogeneous GNN for Link Prediction")

    # Data Paths
    parser.add_argument('--data-dir', type=str, required=True,
                        help="Path to the root GNN data directory (e.g., 'results/shop_id/gnn_data')")
    parser.add_argument('--output-dir', type=str, default='./training_output',
                        help="Directory to save models, logs, and results")

    # Model Hyperparameters
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help="Number of hidden channels in GNN layers and embeddings")
    parser.add_argument('--num-gnn-layers', type=int, default=2,
                        help="Number of GNN layers")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument('--gat-heads', type=int, default=1,
                        help="Number of attention heads in GATConv layers (if using GAT)")

    # Feature Processing Hyperparameters
    parser.add_argument('--node-emb-dim', type=int, default=128,
                         help="Target embedding dimension after initial node feature projection")


    # Training Hyperparameters
    parser.add_argument('--link-pred-edge', type=str, default='purchased',
                        help="The relation type (e.g., 'purchased', 'viewed') to focus link prediction on.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help="Weight decay for Adam optimizer")
    parser.add_argument('--epochs', type=int, default=50, # Adjust as needed
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=512, # Adjust based on GPU memory
                        help="Batch size for LinkNeighborLoader")
    parser.add_argument('--num-neighbors', type=str, default="15,10",
                        help="Number of neighbors to sample per layer (comma-separated string, e.g., '15,10' for 2 layers)")
    parser.add_argument('--neg-sampling-ratio', type=float, default=1.0,
                        help="Ratio of negative samples to positive samples per batch")

    # System Configuration
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('cpu', 'cuda', or 'auto')")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Post-process num_neighbors
    try:
        args.num_neighbors = [int(n) for n in args.num_neighbors.split(',')]
        if len(args.num_neighbors) != args.num_gnn_layers:
             parser.error(f"Number of entries in --num-neighbors ({len(args.num_neighbors)}) must match --num-gnn-layers ({args.num_gnn_layers})")
    except ValueError:
        parser.error("--num-neighbors must be a comma-separated list of integers.")

    # Determine device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)

    # Set seed - Call utility function
    set_seed(args.seed)

    # Setup logging - Call utility function BEFORE printing args
    # Ensure output directory exists before setting up logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir) # Assumes setup_logging is imported from .utils

    logging.info(f"Using device: {args.device}")
    logging.info(f"Set random seed: {args.seed}")

    return args


# --- Node Feature Preprocessing Helper ---
# (NodeFeatureProcessor class remains unchanged)
class NodeFeatureProcessor(nn.Module):
    def __init__(self, node_feature_info, target_dim):
        super().__init__()
        self.target_dim = target_dim
        self.embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.node_types = list(node_feature_info.keys())

        total_input_dims = {}

        logging.info("Initializing Node Feature Processor...")
        for node_type, info in node_feature_info.items():
            current_dim = 0
            # Create embeddings for categorical features
            if info.get('num_categorical_features', 0) > 0:
                self.embeddings[node_type] = nn.ModuleDict()
                for i, col_name in enumerate(info['categorical_cols']):
                    # Check if vocab exists and is not empty
                    if col_name not in info.get('categorical_vocabs', {}) or not info['categorical_vocabs'][col_name]:
                         logging.warning(f"Skipping embedding for {node_type}.{col_name}: Vocabulary missing or empty.")
                         continue
                    vocab_size = len(info['categorical_vocabs'][col_name])
                    if vocab_size == 0:
                         logging.warning(f"Skipping embedding for {node_type}.{col_name}: Vocabulary size is 0.")
                         continue
                    # Simple heuristic for embedding size (can be tuned)
                    emb_size = min(50, (vocab_size + 1) // 2)
                    emb_size = max(emb_size, 4) # Minimum embedding size
                    self.embeddings[node_type][f'emb_{i}'] = nn.Embedding(vocab_size, emb_size)
                    current_dim += emb_size
                    logging.info(f"  Created embedding for {node_type}.{col_name} (Vocab: {vocab_size}, Dim: {emb_size})")

            # Account for numeric features
            num_numeric = info.get('num_numeric_features', 0)
            current_dim += num_numeric
            logging.info(f"  Accounting for {num_numeric} numeric features for {node_type}")

            total_input_dims[node_type] = current_dim

            # Create projection layer if needed
            if current_dim != target_dim and current_dim > 0: # Only project if there are features
                self.projections[node_type] = nn.Linear(current_dim, target_dim)
                logging.info(f"  Created projection layer for {node_type}: {current_dim} -> {target_dim}")
            elif current_dim == 0:
                 # Handle node types with no features - create a learnable embedding per node
                 logging.warning(f"Node type {node_type} has no input features. Creating a learnable embedding.")
                 # Need the number of nodes of this type
                 num_nodes = info.get('num_nodes', None)
                 if num_nodes is None: raise ValueError(f"Cannot create learnable embedding for {node_type} without num_nodes info.")
                 self.embeddings[node_type] = nn.Embedding(num_nodes, target_dim) # One embedding vector per node
                 self.projections[node_type] = nn.Identity() # No projection needed after learnable embedding
            else: # current_dim == target_dim
                self.projections[node_type] = nn.Identity()
                logging.info(f"  No projection needed for {node_type} (Input dim {current_dim} == Target dim {target_dim})")

    def forward(self, data):
        """ Processes raw numeric/categorical features into initial embeddings. """
        out_x_dict = {}
        # Determina o device a partir de um parâmetro existente (ex: da primeira projeção)
        # Ou define um default se nenhuma camada existir ainda
        default_device = 'cpu'
        if self.projections:
            first_proj_key = next(iter(self.projections))
            if hasattr(self.projections[first_proj_key], 'weight'):
                 default_device = self.projections[first_proj_key].weight.device
            elif hasattr(self.projections[first_proj_key], 'bias') and self.projections[first_proj_key].bias is not None:
                 default_device = self.projections[first_proj_key].bias.device
        elif self.embeddings:
             # Tenta obter de uma camada de embedding se não houver projeções
             first_emb_node = next(iter(self.embeddings))
             if isinstance(self.embeddings[first_emb_node], nn.ModuleDict): # Categorical embeddings
                 if self.embeddings[first_emb_node]: # Check if ModuleDict is not empty
                     first_emb_layer_key = next(iter(self.embeddings[first_emb_node]))
                     default_device = self.embeddings[first_emb_node][first_emb_layer_key].weight.device
             elif isinstance(self.embeddings[first_emb_node], nn.Embedding): # Node embedding
                 default_device = self.embeddings[first_emb_node].weight.device


        for node_type in self.node_types:
            # Determina o número de nós para este tipo NO BATCH ATUAL
            num_nodes_in_batch = 0
            if hasattr(data[node_type], 'num_nodes'):
                num_nodes_in_batch = data[node_type].num_nodes
            elif hasattr(data[node_type], 'n_id'): # n_id usually present in batches from LinkNeighborLoader
                num_nodes_in_batch = data[node_type].n_id.shape[0]
            elif hasattr(data[node_type], 'x_numeric') and data[node_type].x_numeric.shape[0] > 0:
                 num_nodes_in_batch = data[node_type].x_numeric.shape[0]
            elif hasattr(data[node_type], 'x_categorical') and data[node_type].x_categorical.shape[0] > 0:
                 num_nodes_in_batch = data[node_type].x_categorical.shape[0]
            elif hasattr(data[node_type], 'x') and data[node_type].x.shape[0] > 0: # Fallback genérico
                 num_nodes_in_batch = data[node_type].x.shape[0]
            # Handle case where data might be the full graph (not a batch)
            elif not hasattr(data, 'batch_size'): # Heuristic check for full graph data
                 if node_type in data.metadata[0]:
                      num_nodes_in_batch = data[node_type].num_nodes # Use total nodes if processing full graph

            # Se não conseguirmos determinar nós, não podemos processar este tipo no batch
            if num_nodes_in_batch == 0 and not (node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.Embedding)):
                 logging.warning(f"Could not determine number of nodes for type '{node_type}' in current data object. Skipping feature processing for this type.")
                 out_x_dict[node_type] = torch.zeros((0, self.target_dim), device=default_device)
                 continue # Pula para o próximo node_type

            # Handle node types that start with a learnable embedding (no raw features)
            if node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.Embedding):
                 # If processing a batch, use n_id. If processing full graph, use all indices.
                 if hasattr(data[node_type], 'n_id'):
                     node_indices = data[node_type].n_id
                 else: # Assume processing full graph
                     node_indices = torch.arange(num_nodes_in_batch, device=default_device)

                 if node_indices.numel() == 0:
                     x_processed = torch.zeros((0, self.target_dim), device=default_device)
                 elif node_indices.max() >= self.embeddings[node_type].num_embeddings:
                      raise IndexError(f"Index out of bounds for learnable embedding of type {node_type}. Max index: {node_indices.max()}, Num embeddings: {self.embeddings[node_type].num_embeddings}")
                 else:
                      x_processed = self.embeddings[node_type](node_indices)
                 out_x_dict[node_type] = x_processed
                 continue

            # Process raw features if they exist
            features_to_concat = []
            # Process categorical features
            if node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.ModuleDict):
                if hasattr(data[node_type], 'x_categorical') and data[node_type].x_categorical.numel() > 0: # Check if tensor is not empty
                    cat_indices = data[node_type].x_categorical
                    if cat_indices.shape[0] != num_nodes_in_batch:
                         logging.warning(f"Row mismatch for {node_type} categorical features. Expected {num_nodes_in_batch}, got {cat_indices.shape[0]}. Skipping categorical.")
                    elif cat_indices.shape[1] == 0:
                         logging.debug(f"Node type {node_type} has x_categorical tensor but 0 columns.")
                    else:
                        num_expected_cat_features = len(self.embeddings[node_type])
                        if cat_indices.shape[1] != num_expected_cat_features:
                             logging.warning(f"Mismatch in number of categorical features for {node_type}. Expected {num_expected_cat_features}, got {cat_indices.shape[1]}. Trying to use available.")
                             num_cat_to_process = min(cat_indices.shape[1], num_expected_cat_features)
                        else:
                             num_cat_to_process = cat_indices.shape[1]

                        emb_keys = list(self.embeddings[node_type].keys()) # Get keys in order
                        for i in range(num_cat_to_process):
                             emb_key = emb_keys[i]
                             emb_layer = self.embeddings[node_type][emb_key]
                             if cat_indices[:, i].numel() > 0:
                                 max_idx = cat_indices[:, i].max()
                                 if max_idx >= emb_layer.num_embeddings:
                                      raise IndexError(f"Index {max_idx} out of bounds for embedding {emb_key} (size {emb_layer.num_embeddings}) in node type {node_type}.")
                                 features_to_concat.append(emb_layer(cat_indices[:, i]))
                elif self.embeddings[node_type]:
                     logging.debug(f"Node type {node_type} has categorical embeddings defined but no x_categorical data.")

            # Add numeric features
            if hasattr(data[node_type], 'x_numeric') and data[node_type].x_numeric.numel() > 0:
                 if data[node_type].x_numeric.shape[0] != num_nodes_in_batch:
                      logging.warning(f"Row mismatch for {node_type} numeric features. Expected {num_nodes_in_batch}, got {data[node_type].x_numeric.shape[0]}. Skipping numeric.")
                 elif data[node_type].x_numeric.shape[1] > 0:
                      features_to_concat.append(data[node_type].x_numeric)

            # Combina e Projeta
            if not features_to_concat:
                 logging.warning(f"Node type {node_type} has no features to process (numeric or categorical). Creating zeros.")
                 x_combined = torch.zeros((num_nodes_in_batch, 0), device=default_device)
            else:
                 x_combined = torch.cat(features_to_concat, dim=-1)

            # Aplica a projeção (Linear ou Identity)
            if node_type in self.projections:
                 if x_combined.shape[1] == 0 and self.projections[node_type].in_features != 0:
                      logging.warning(f"Projection layer for {node_type} expects input dim {self.projections[node_type].in_features} but got 0. Creating zeros.")
                      x_processed = torch.zeros((num_nodes_in_batch, self.target_dim), device=default_device)
                 elif x_combined.shape[1] != self.projections[node_type].in_features and not isinstance(self.projections[node_type], nn.Identity):
                      logging.error(f"Input dimension mismatch for {node_type} projection. Expected {self.projections[node_type].in_features}, got {x_combined.shape[1]}. Creating zeros.")
                      x_processed = torch.zeros((num_nodes_in_batch, self.target_dim), device=default_device)
                 else:
                      x_processed = self.projections[node_type](x_combined)
            else:
                 logging.error(f"No projection layer found for {node_type} during forward pass.")
                 x_processed = torch.zeros((num_nodes_in_batch, self.target_dim), device=default_device)


            out_x_dict[node_type] = x_processed

        return out_x_dict


# (train_epoch function remains the same as the previous corrected version)
def train_epoch(model, node_feature_processor, loader, optimizer, device, edge_type_tuple):
    model.train()
    node_feature_processor.train()
    total_loss = 0.0
    total_examples = 0
    batch_num = 0
    processed_batches = 0
    skipped_batches_no_labels = 0

    for batch in loader:
        batch = batch.to(device)
        batch_num += 1
        optimizer.zero_grad()

        supervision_edge_label_index = None
        supervision_edge_label = None
        labels_found = False

        try:
            edge_store = batch[edge_type_tuple]
            supervision_edge_label_index = edge_store.edge_label_index
            supervision_edge_label = edge_store.edge_label
            labels_found = True
        except (KeyError, AttributeError):
            if hasattr(batch, 'edge_label_index') and hasattr(batch, 'edge_label'):
                supervision_edge_label_index = batch.edge_label_index
                supervision_edge_label = batch.edge_label
                labels_found = True

        if not labels_found or supervision_edge_label_index is None or supervision_edge_label is None:
            skipped_batches_no_labels += 1
            if batch_num % 50 == 1 or batch_num == len(loader):
                logging.warning(f"Skipping training batch {batch_num}/{len(loader)} - Failed to access 'edge_label_index' or 'edge_label' for type {edge_type_tuple} (tried nested and top-level). Batch keys: {list(batch.keys())}")
            else:
                logging.debug(f"Skipping training batch {batch_num} - Failed to access 'edge_label_index' or 'edge_label' for type {edge_type_tuple}. Batch keys: {list(batch.keys())}")
            continue

        if supervision_edge_label_index.shape[1] == 0:
            skipped_batches_no_labels += 1
            logging.warning(f"Skipping training batch {batch_num} - 'edge_label_index' for type {edge_type_tuple} is empty (shape: {supervision_edge_label_index.shape}).")
            continue

        try:
            initial_x_dict = node_feature_processor(batch)
            h_dict = model(initial_x_dict, batch.edge_index_dict)
            pred_scores = model.predict_link_score(h_dict, supervision_edge_label_index, edge_type_tuple)
            loss = F.binary_cross_entropy_with_logits(pred_scores, supervision_edge_label.float())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_examples += pred_scores.numel()
            processed_batches += 1
        except KeyError as e:
             logging.warning(f"Skipping training batch {batch_num} due to KeyError during feature/embedding access: {e}. Check h_dict/edge_index_dict keys. Available batch keys: {list(batch.keys())}", exc_info=False)
             continue
        except IndexError as e:
             logging.error(f"IndexError during training batch {batch_num}: {e}. Check node indices and embedding access.", exc_info=True)
             continue
        except Exception as e:
             logging.error(f"Error during training batch {batch_num}: {e}", exc_info=True)
             continue

    if processed_batches == 0:
        logging.warning(f"No examples were processed in the training epoch (all {batch_num} batches skipped or failed). Skipped {skipped_batches_no_labels} due to missing/empty/inaccessible labels.")
        return 0.0
    elif skipped_batches_no_labels > 0:
         logging.warning(f"Processed {processed_batches}/{batch_num} batches in epoch. Skipped {skipped_batches_no_labels} due to missing/empty/inaccessible labels for {edge_type_tuple}.")

    return total_loss / processed_batches if processed_batches > 0 else 0.0


# --- Evaluation Function Wrapper ---
@torch.no_grad()
def evaluate(model, node_feature_processor, loader, device, edge_type_tuple, metric='AUC', output_dir=None, epoch=None):
    """ Wrapper function to call specific evaluation metrics. """
    if metric == 'AUC':
        # Passar output_dir e epoch para calculate_auc
        return calculate_auc(model, node_feature_processor, loader, device, edge_type_tuple, output_dir, epoch)
    elif metric == 'Ranking':
         # Definir os K's aqui ou passar como argumento se necessário
         k_values = [5, 10, 20]
         return calculate_ranking_metrics(model, node_feature_processor, loader, device, edge_type_tuple, k_list=k_values)
    else:
        logging.error(f"Unsupported evaluation metric: {metric}")
        raise ValueError(f"Unsupported evaluation metric: {metric}")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    # Configurar logging do matplotlib/pillow para WARNING para reduzir verbosidade
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f"Starting training script with arguments: {args}")

    # Save the configuration used for this run
    save_config(args, args.output_dir)

    # --- Load Data ---
    logging.info("--- Loading Data ---")
    try:
        # Carregar dados para a CPU inicialmente
        data_dict, metadata, node_feature_info, edge_feature_info = load_hetero_graph_data(args.data_dir, device='cpu')
        train_data, valid_data, test_data = data_dict['train'], data_dict['valid'], data_dict['test']
        logging.info("Data loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Failed to load data: {e}. Check --data-dir path.")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}", exc_info=True)
        exit(1)

    # --- Determine Edge Type for Prediction ---
    edge_type_to_predict = ('Person', args.link_pred_edge, 'Product')
    logging.info(f"Target edge type for link prediction: {edge_type_to_predict}")
    if edge_type_to_predict not in metadata[1]:
         logging.error(f"Specified --link-pred-edge '{args.link_pred_edge}' "
                       f"does not correspond to a valid edge type found in the data: {metadata[1]}")
         exit(1)

    # --- Prepare DataLoaders ---
    logging.info("--- Preparing DataLoaders ---")
    try:
        # Train Loader Check
        type_exists_in_list = edge_type_to_predict in train_data.edge_types
        edge_store = None
        has_edges = False
        if type_exists_in_list:
            try:
                edge_store = train_data[edge_type_to_predict]
                if 'edge_index' in edge_store and isinstance(edge_store['edge_index'], torch.Tensor):
                    has_edges = edge_store['edge_index'].shape[1] > 0
            except KeyError: type_exists_in_list = False
        if not type_exists_in_list or not has_edges:
            logging.error(f"Cannot create train_loader: No edges found for target type {edge_type_to_predict} in training data.")
            exit(1)
        train_loader = LinkNeighborLoader(
            train_data, num_neighbors=args.num_neighbors, batch_size=args.batch_size,
            edge_label_index=(edge_type_to_predict, edge_store['edge_index']),
            edge_label=torch.ones(edge_store['edge_index'].size(1)),
            neg_sampling_ratio=args.neg_sampling_ratio, shuffle=True, num_workers=0,
        )
        logging.info("Train loader created successfully.")

        # Validation Loader Check
        val_loader = None
        valid_type_exists_in_list = edge_type_to_predict in valid_data.edge_types
        valid_has_edges = False
        valid_edge_store = None
        if valid_type_exists_in_list:
            try:
                valid_edge_store = valid_data[edge_type_to_predict]
                if 'edge_index' in valid_edge_store and isinstance(valid_edge_store['edge_index'], torch.Tensor):
                    valid_has_edges = valid_edge_store['edge_index'].shape[1] > 0
            except KeyError: valid_type_exists_in_list = False
        if valid_type_exists_in_list and valid_has_edges:
            eval_batch_size = args.batch_size * 2
            val_loader = LinkNeighborLoader(
                valid_data, num_neighbors=args.num_neighbors, batch_size=eval_batch_size,
                edge_label_index=(edge_type_to_predict, valid_edge_store['edge_index']),
                edge_label=torch.ones(valid_edge_store['edge_index'].size(1)),
                neg_sampling_ratio=args.neg_sampling_ratio, shuffle=False, num_workers=0,
            )
            logging.info(f"Validation loader created (eval batch size: {eval_batch_size}).")
        else:
             logging.warning(f"No edges found for target type {edge_type_to_predict} in validation data. Skipping validation loader creation.")

        # Test Loader Check
        test_loader = None
        test_type_exists_in_list = edge_type_to_predict in test_data.edge_types
        test_has_edges = False
        test_edge_store = None
        if test_type_exists_in_list:
            try:
                test_edge_store = test_data[edge_type_to_predict]
                if 'edge_index' in test_edge_store and isinstance(test_edge_store['edge_index'], torch.Tensor):
                    test_has_edges = test_edge_store['edge_index'].shape[1] > 0
            except KeyError: test_type_exists_in_list = False
        if test_type_exists_in_list and test_has_edges:
            eval_batch_size = args.batch_size * 2
            test_loader = LinkNeighborLoader(
                test_data, num_neighbors=args.num_neighbors, batch_size=eval_batch_size,
                edge_label_index=(edge_type_to_predict, test_edge_store['edge_index']),
                edge_label=torch.ones(test_edge_store['edge_index'].size(1)),
                neg_sampling_ratio=args.neg_sampling_ratio, shuffle=False, num_workers=0,
            )
            logging.info(f"Test loader created (eval batch size: {eval_batch_size}).")
        else:
             logging.warning(f"No edges found for target type {edge_type_to_predict} in test data. Skipping test loader creation.")

        logging.info("DataLoaders preparation finished.")

    except Exception as e:
        logging.error(f"Failed to create DataLoaders: {e}", exc_info=True)
        exit(1)

    # --- Initialize Feature Processor and Model ---
    logging.info("--- Initializing Model ---")
    try:
        node_processor = NodeFeatureProcessor(node_feature_info, args.node_emb_dim).to(args.device)
        model = HeteroGNNLinkPredictor(
            node_types=metadata[0],
            edge_types=metadata[1],
            metadata=metadata,
            hidden_channels=args.hidden_channels, # Usar hidden_channels para GNN
            num_gnn_layers=args.num_gnn_layers,
            dropout=args.dropout,
            gat_heads=args.gat_heads
        ).to(args.device)
        logging.info(f"Node Feature Processor:\n{node_processor}")
        logging.info(f"GNN Model:\n{model}")

        optimizer = optim.Adam(
            list(node_processor.parameters()) + list(model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    except Exception as e:
        logging.error(f"Failed to initialize model/optimizer: {e}", exc_info=True)
        exit(1)

    # --- Training & Evaluation Loop ---
    logging.info("--- Starting Training ---")
    best_val_auc = 0
    best_epoch = 0
    epochs_no_improve = 0
    patience = 10 # Número de épocas sem melhoria antes de parar
    min_delta = 0.0001 # Melhoria mínima na Val AUC para resetar a paciência
    history = {'epoch': [], 'train_loss': [], 'val_auc': []}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss = train_epoch(model, node_processor, train_loader, optimizer, args.device, edge_type_to_predict)

        val_auc = 0.0
        if val_loader:
             # NÃO passar output_dir/epoch durante validação
             val_auc = evaluate(model, node_processor, val_loader, args.device, edge_type_to_predict, metric='AUC')
        else:
             if epoch == 1: logging.info("Skipping validation (no validation data/loader for target edge type).")

        end_time = time.time()
        epoch_time = end_time - start_time

        history['epoch'].append(epoch)
        history['train_loss'].append(loss)
        history['val_auc'].append(val_auc)

        logging.info(f'Epoch: {epoch:03d}, Avg Batch Loss: {loss:.6f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.2f}s')

        # Early Stopping Check
        if val_loader:
            if val_auc > best_val_auc + min_delta:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_no_improve = 0 # Resetar contador
                # Salvar o melhor modelo
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                try:
                    torch.save({
                        'epoch': epoch,
                        'node_processor_state_dict': node_processor.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_auc': best_val_auc,
                        'args': vars(args)
                    }, checkpoint_path)
                    logging.info(f"  -> New best model saved to {checkpoint_path} (Val AUC: {best_val_auc:.4f})")
                except Exception as e:
                     logging.error(f"Error saving checkpoint: {e}")
            else:
                epochs_no_improve += 1
                logging.info(f"  Val AUC did not improve significantly for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}. Best Val AUC {best_val_auc:.4f} at epoch {best_epoch}.")
                break # Sai do loop de treino

    logging.info("--- Training Finished ---")
    final_epoch_num = epoch # Guardar a última época executada

    if val_loader:
        logging.info(f"Best Validation AUC: {best_val_auc:.4f} at Epoch {best_epoch}")
    else:
         # Salvar modelo final se não houver validação (usar o estado atual)
         logging.info("Training finished (no validation performed). Saving final model.")
         checkpoint_path = os.path.join(args.output_dir, 'final_model.pt')
         try:
            torch.save({
                    'epoch': final_epoch_num,
                    'node_processor_state_dict': node_processor.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args)
                }, checkpoint_path)
            logging.info(f"Final model saved to {checkpoint_path}")
         except Exception as e:
            logging.error(f"Error saving final model: {e}")


    # --- Final Testing ---
    logging.info("--- Testing on Test Set ---")
    model_to_load_path = ""
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')

    # Priorizar sempre o best_model se existir
    if os.path.exists(best_model_path):
        model_to_load_path = best_model_path
        logging.info(f"Loading best model from validation: {model_to_load_path}")
    else:
         final_model_path = os.path.join(args.output_dir, 'final_model.pt')
         if os.path.exists(final_model_path):
              model_to_load_path = final_model_path
              logging.warning(f"Best model checkpoint not found or no validation done, loading final model from: {model_to_load_path}")
         else:
              logging.warning("No suitable model checkpoint found ('best_model.pt' or 'final_model.pt'). Cannot run final test.")
              model_to_load_path = None

    test_auc = 0.0
    test_ranking_metrics = {} # Inicializar

    if model_to_load_path and test_loader:
        try:
            logging.info(f"Attempting to load checkpoint with torch.load(..., weights_only=False)")
            checkpoint = torch.load(model_to_load_path, map_location=args.device, weights_only=False)

            # Recarregar args e recriar modelos
            loaded_args_dict = checkpoint.get('args', {})
            current_args_dict = vars(args).copy()
            current_args_dict.update(loaded_args_dict)
            if isinstance(current_args_dict.get('device'), str):
                 current_args_dict['device'] = torch.device(current_args_dict['device'])
            if isinstance(current_args_dict.get('num_neighbors'), str):
                try:
                    current_args_dict['num_neighbors'] = [int(n) for n in current_args_dict['num_neighbors'].split(',')]
                except ValueError:
                    logging.error("Invalid num_neighbors format in loaded args, using current args value.")
                    current_args_dict['num_neighbors'] = args.num_neighbors # Fallback
            if 'gat_heads' not in current_args_dict:
                 current_args_dict['gat_heads'] = args.gat_heads

            loaded_args = argparse.Namespace(**current_args_dict)

            node_processor = NodeFeatureProcessor(node_feature_info, loaded_args.node_emb_dim).to(args.device)
            model = HeteroGNNLinkPredictor(
                node_types=metadata[0], edge_types=metadata[1], metadata=metadata,
                hidden_channels=loaded_args.hidden_channels, # Usar hidden_channels dos args carregados
                num_gnn_layers=loaded_args.num_gnn_layers,
                dropout=loaded_args.dropout,
                gat_heads=loaded_args.gat_heads
            ).to(args.device)

            node_processor.load_state_dict(checkpoint['node_processor_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_epoch = checkpoint.get('epoch', 'N/A')
            logging.info(f"Loaded model state from epoch {loaded_epoch}")

            # Calcular AUC (com plots finais)
            test_auc = evaluate(model, node_processor, test_loader, args.device, edge_type_to_predict, metric='AUC', output_dir=args.output_dir, epoch=None)
            logging.info(f'Final Test AUC: {test_auc:.4f}')

            # Calcular métricas de ranking
            logging.info("--- Calculating Ranking Metrics on Test Set ---")
            test_ranking_metrics = evaluate(model, node_processor, test_loader, args.device, edge_type_to_predict, metric='Ranking')

            # Gerar Plot t-SNE
            logging.info("--- Generating Final Embeddings for t-SNE Plot ---")
            model.eval()
            node_processor.eval()
            with torch.no_grad():
                # Usar dados completos (ex: treino) para obter todos os embeddings
                full_data_for_emb = data_dict['train'].clone().to(args.device)
                initial_features = node_processor(full_data_for_emb)
                final_embeddings = model(initial_features, full_data_for_emb.edge_index_dict)

                # Carregar info extra (is_buyer)
                extra_info = None
                # --- CORREÇÃO: Obter shop_id do data_dir ---
                try:
                    # Extrair shop_id do path data_dir (ex: ./results/shop_id/gnn_data -> shop_id)
                    shop_id_from_path = os.path.basename(os.path.dirname(args.data_dir))
                    person_features_path = os.path.join(os.path.dirname(args.data_dir), "..", "preprocessed", f"{shop_id_from_path}_person_features_processed.csv")
                    person_id_map_path = os.path.join(args.data_dir, 'person_nodes', 'person_id_to_index_map.json')
                    logging.info(f"Attempting to load extra person info from: {person_features_path}")
                    if os.path.exists(person_features_path) and os.path.exists(person_id_map_path):
                        person_df = pd.read_csv(person_features_path, usecols=['person_id_hex', 'is_buyer'], low_memory=False)
                        with open(person_id_map_path, 'r') as f:
                             person_id_map = json.load(f)
                        person_df['person_idx'] = person_df['person_id_hex'].map(person_id_map)
                        person_df = person_df.dropna(subset=['person_idx'])
                        person_df['person_idx'] = person_df['person_idx'].astype(int)
                        num_person_nodes = final_embeddings['Person'].shape[0]
                        person_df = person_df.set_index('person_idx').reindex(range(num_person_nodes))
                        person_df['is_buyer'] = person_df['is_buyer'].fillna(0).astype(int)
                        extra_info = person_df
                        logging.info("Loaded 'is_buyer' information for Person nodes.")
                    else:
                         logging.warning(f"Could not find person features file ({person_features_path}) or ID map ({person_id_map_path}) for t-SNE labels.")
                except Exception as e:
                    logging.warning(f"Could not load or process extra person info: {e}", exc_info=True)
                # --- FIM CORREÇÃO ---

                # Plotar embeddings (todos os tipos juntos, coloridos por tipo)
                plot_tsne_embeddings(final_embeddings, args.output_dir, filename_prefix="final_embeddings")

                # Plotar embeddings de Person, coloridos por 'is_buyer'
                if extra_info is not None and 'is_buyer' in extra_info.columns:
                     plot_tsne_embeddings(final_embeddings, args.output_dir,
                                          filename_prefix="final_person_embeddings",
                                          node_type_to_plot='Person',
                                          extra_info_df=extra_info,
                                          id_col='person_id_hex', # Coluna original para referência
                                          label_col='is_buyer') # Coluna para colorir

        except FileNotFoundError:
            logging.error(f"Model checkpoint not found at {model_to_load_path}")
        except Exception as e:
             logging.error(f"Error during testing or plotting: {e}", exc_info=True)
    elif not test_loader:
        logging.warning("Skipping testing phase as no test loader was created.")
    elif not model_to_load_path:
         logging.warning("Skipping testing phase as no model checkpoint was found.")


    # Save final results summary
    results = {
        'best_val_auc': best_val_auc if val_loader else None,
        'test_auc': test_auc if test_loader and model_to_load_path else None,
        'test_ranking_metrics': test_ranking_metrics if test_ranking_metrics else None,
        'best_epoch': best_epoch if val_loader else None,
        'final_epoch': final_epoch_num, # Usar a última época executada
        'config': vars(args).copy()
        }
    try:
        # Serialização da config
        results['config']['device'] = str(results['config']['device'])
        if isinstance(results['config'].get('num_neighbors'), list):
            results['config']['num_neighbors'] = ",".join(map(str, results['config']['num_neighbors']))

        with open(os.path.join(args.output_dir, 'final_summary.json'), 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Final summary saved to {os.path.join(args.output_dir, 'final_summary.json')}")
    except Exception as e:
        logging.error(f"Failed to save final summary: {e}")

    # Save training history
    try:
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
        logging.info(f"Training history saved to {os.path.join(args.output_dir, 'training_history.csv')}")
    except Exception as e:
        logging.error(f"Failed to save training history: {e}")


    logging.info("--- Script Finished ---")