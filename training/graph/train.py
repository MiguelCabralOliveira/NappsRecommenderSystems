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
import logging # Import logging

# Assume .model, .dataset, .evaluate, .utils are in the same directory or accessible
from .model import HeteroGNNLinkPredictor
from .dataset import load_hetero_graph_data
from .evaluate import calculate_auc
from .utils import set_seed, setup_logging, save_config, load_config


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Heterogeneous GraphSAGE for Link Prediction")

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
    # Feature Processing Hyperparameters
    parser.add_argument('--node-emb-dim', type=int, default=128,
                         help="Target embedding dimension after initial node feature projection")


    # Training Hyperparameters
    parser.add_argument('--link-pred-edge', type=str, default='purchased',
                        help="The relation type (e.g., 'purchased', 'viewed') to focus link prediction on.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
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
# This class handles the initial embedding lookup and projection
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
                    vocab_size = len(info['categorical_vocabs'][col_name])
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
                 first_emb_layer_key = next(iter(self.embeddings[first_emb_node]))
                 default_device = self.embeddings[first_emb_node][first_emb_layer_key].weight.device
             elif isinstance(self.embeddings[first_emb_node], nn.Embedding): # Node embedding
                 default_device = self.embeddings[first_emb_node].weight.device


        for node_type in self.node_types:
            # Determina o número de nós para este tipo NO BATCH ATUAL
            num_nodes_in_batch = 0
            if hasattr(data[node_type], 'num_nodes'):
                num_nodes_in_batch = data[node_type].num_nodes
            elif hasattr(data[node_type], 'n_id'):
                num_nodes_in_batch = data[node_type].n_id.shape[0]
            elif hasattr(data[node_type], 'x_numeric') and data[node_type].x_numeric.shape[0] > 0:
                 num_nodes_in_batch = data[node_type].x_numeric.shape[0]
            elif hasattr(data[node_type], 'x_categorical') and data[node_type].x_categorical.shape[0] > 0:
                 num_nodes_in_batch = data[node_type].x_categorical.shape[0]
            elif hasattr(data[node_type], 'x') and data[node_type].x.shape[0] > 0: # Fallback genérico
                 num_nodes_in_batch = data[node_type].x.shape[0]

            # Se não conseguirmos determinar nós, não podemos processar este tipo no batch
            if num_nodes_in_batch == 0 and not (node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.Embedding)):
                 # A menos que seja um embedding aprendível (que usa n_id), não ter nós é um problema
                 logging.warning(f"Could not determine number of nodes for type '{node_type}' in current batch. Skipping feature processing for this type.")
                 # Atribuir um tensor vazio pode causar problemas, talvez seja melhor pular?
                 # Ou criar um tensor (0, target_dim) se as camadas GNN conseguirem lidar?
                 # Por segurança, vamos criar um tensor (0, target_dim) mas logar o aviso.
                 out_x_dict[node_type] = torch.zeros((0, self.target_dim), device=default_device)
                 continue # Pula para o próximo node_type

            # Handle node types that start with a learnable embedding (no raw features)
            if node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.Embedding):
                 if not hasattr(data[node_type], 'n_id'):
                      raise ValueError(f"Batch missing node indices ('n_id') for node type {node_type} which uses learnable embeddings.")
                 node_indices = data[node_type].n_id
                 # Garante que os índices não excedem o tamanho do embedding
                 if node_indices.max() >= self.embeddings[node_type].num_embeddings:
                      raise IndexError(f"Index out of bounds for learnable embedding of type {node_type}. Max index: {node_indices.max()}, Num embeddings: {self.embeddings[node_type].num_embeddings}")
                 x_processed = self.embeddings[node_type](node_indices) # Dimensão já é target_dim
                 # Não precisa de projeção adicional neste caso
                 out_x_dict[node_type] = x_processed
                 continue # Processamento deste tipo de nó está completo

            # Process raw features if they exist
            features_to_concat = []
            # Process categorical features
            if node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.ModuleDict):
                if hasattr(data[node_type], 'x_categorical') and data[node_type].x_categorical.shape[1] > 0:
                    cat_indices = data[node_type].x_categorical
                    # Verifica se o número de colunas categóricas corresponde ao número de embeddings
                    if cat_indices.shape[1] != len(self.embeddings[node_type]):
                         logging.warning(f"Mismatch in number of categorical features for {node_type}. Expected {len(self.embeddings[node_type])}, got {cat_indices.shape[1]}. Adjusting.")
                         # Tenta usar apenas as colunas disponíveis
                         num_cat_to_process = min(cat_indices.shape[1], len(self.embeddings[node_type]))
                    else:
                         num_cat_to_process = cat_indices.shape[1]

                    for i, (emb_key, emb_layer) in enumerate(self.embeddings[node_type].items()):
                        if i < num_cat_to_process:
                             # Verifica se os índices estão dentro dos limites do vocabulário
                             max_idx = cat_indices[:, i].max()
                             if max_idx >= emb_layer.num_embeddings:
                                  raise IndexError(f"Index {max_idx} out of bounds for embedding {emb_key} (size {emb_layer.num_embeddings}) in node type {node_type}.")
                             features_to_concat.append(emb_layer(cat_indices[:, i]))
                        else:
                             break # Sai se não houver mais colunas de dados
                elif self.embeddings[node_type]: # Se embeddings existem mas não há dados x_categorical
                     logging.debug(f"Node type {node_type} has categorical embeddings defined but no x_categorical data in batch.")

            # Add numeric features
            if hasattr(data[node_type], 'x_numeric') and data[node_type].x_numeric.shape[1] > 0:
                 features_to_concat.append(data[node_type].x_numeric)

            # Combina e Projeta
            if not features_to_concat:
                 # Se não há features raw E não é um embedding aprendível, cria zeros
                 logging.warning(f"Node type {node_type} has no raw input features in this batch. Creating zeros.")
                 x_combined = torch.zeros((num_nodes_in_batch, 0), device=default_device) # Começa com dimensão 0
            else:
                 x_combined = torch.cat(features_to_concat, dim=-1)

            # Aplica a projeção (Linear ou Identity)
            if node_type in self.projections:
                 x_processed = self.projections[node_type](x_combined)
            else:
                 # Se não há camada de projeção definida (deveria ser Identity se dim==target_dim ou learnable emb)
                 # Mas como fallback, verifica a dimensão
                 if x_combined.shape[1] != self.target_dim:
                      logging.error(f"Node type {node_type} has no projection layer but combined dimension ({x_combined.shape[1]}) != target_dim ({self.target_dim}). This shouldn't happen.")
                      # Tenta criar zeros com a dimensão correta como último recurso
                      x_processed = torch.zeros((num_nodes_in_batch, self.target_dim), device=x_combined.device)
                 else:
                      x_processed = x_combined # Dimensão já está correta

            out_x_dict[node_type] = x_processed

        return out_x_dict


# --- Training and Evaluation Functions ---

def train_epoch(model, node_feature_processor, loader, optimizer, device, edge_type_tuple):
    model.train()
    node_feature_processor.train() # Set processor to train mode too
    total_loss = total_examples = 0
    batch_num = 0
    for batch in loader:
        batch = batch.to(device)
        batch_num += 1
        optimizer.zero_grad()

        try:
            # 1. Process raw node features
            initial_x_dict = node_feature_processor(batch)

            # 2. Run GNN
            h_dict = model(initial_x_dict, batch.edge_index_dict)

            # 3. Predict scores
            # Get edge store for the target type within the batch
            batch_edge_store = batch.get(edge_type_tuple, None)
            if batch_edge_store is None or 'edge_label_index' not in batch_edge_store or 'edge_label' not in batch_edge_store:
                 logging.warning(f"Skipping training batch {batch_num} - missing required edge data (label/index) for {edge_type_tuple}.")
                 continue
            # Additional check: ensure edge_label_index is not empty
            if batch_edge_store['edge_label_index'].shape[1] == 0:
                 logging.warning(f"Skipping training batch {batch_num} - edge_label_index for {edge_type_tuple} is empty.")
                 continue

            pred_scores = model.predict_link_score(h_dict, batch_edge_store['edge_label_index'], edge_type_tuple)

            # 4. Calculate Loss
            loss = F.binary_cross_entropy_with_logits(pred_scores, batch_edge_store['edge_label'].float())

            # 5. Backpropagate
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred_scores.numel()
            total_examples += pred_scores.numel()

        except KeyError as e:
             logging.warning(f"Skipping training batch {batch_num} due to KeyError: {e}. Check data/batch structure.")
             continue
        except IndexError as e:
             logging.error(f"IndexError during training batch {batch_num}: {e}. Check node indices and embedding access.", exc_info=True)
             continue
        except Exception as e:
             logging.error(f"Error during training batch {batch_num}: {e}", exc_info=True) # Log traceback
             continue # Skip batch on error

    if total_examples == 0:
        logging.warning("No examples were processed in the training epoch.")
        return 0.0
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, node_feature_processor, loader, device, edge_type_tuple, metric='AUC'):
    # Use the calculate_auc function from evaluate.py
    if metric == 'AUC':
        return calculate_auc(model, node_feature_processor, loader, device, edge_type_tuple)
    # Add calls to other evaluation functions (e.g., calculate_hit_rate_at_k) if implemented
    # elif metric == 'HitRate@10':
    #     return calculate_hit_rate_at_k(model, node_feature_processor, loader, device, edge_type_tuple, k=10)
    else:
        raise ValueError(f"Unsupported evaluation metric: {metric}")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Starting training script with arguments: {args}")

    # Save the configuration used for this run
    save_config(args, args.output_dir)

    # --- Load Data ---
    logging.info("--- Loading Data ---")
    # Load data onto CPU first, move batches to GPU later if available
    try:
        # Ensure dataset.py uses the corrected assignment: data[edge_type]['edge_index'] = ...
        data_dict, metadata, node_feature_info, edge_feature_info = load_hetero_graph_data(args.data_dir, device='cpu') # Load to CPU
        train_data, valid_data, test_data = data_dict['train'], data_dict['valid'], data_dict['test']
        logging.info("Data loaded successfully.")
        logging.info(f"Train data object:\n{train_data}")
        logging.info(f"Validation data object:\n{valid_data}")
        logging.info(f"Test data object:\n{test_data}")
    except FileNotFoundError as e:
        logging.error(f"Failed to load data: {e}. Check --data-dir path.")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}", exc_info=True)
        exit(1)

    # --- Determine Edge Type for Prediction ---
    # Construct the tuple dynamically based on the argument
    edge_type_to_predict = ('Person', args.link_pred_edge, 'Product')
    logging.info(f"Target edge type for link prediction: {edge_type_to_predict}")

    # Validate edge type against metadata derived from loaded data
    if edge_type_to_predict not in metadata[1]:
         logging.error(f"Specified --link-pred-edge '{args.link_pred_edge}' "
                       f"does not correspond to a valid edge type found in the data: {metadata[1]}")
         exit(1)

    # --- Prepare DataLoaders ---
    logging.info("--- Preparing DataLoaders ---")
    try:
        # --- REVISED CHECK LOGIC using .edge_types ---
        logging.debug(f"Target edge type: {edge_type_to_predict}")
        logging.debug(f"Available edge types in train_data.edge_types: {train_data.edge_types}")

        # Check 1: Does the edge type exist in the official list?
        type_exists_in_list = edge_type_to_predict in train_data.edge_types

        # Check 2: If it exists, access the store and check for edge_index and shape
        edge_store = None
        has_edges = False
        if type_exists_in_list:
            try:
                # Acessa diretamente usando a chave tuple, pois sabemos que deve existir
                edge_store = train_data[edge_type_to_predict]
                if 'edge_index' in edge_store and isinstance(edge_store['edge_index'], torch.Tensor):
                    has_edges = edge_store['edge_index'].shape[1] > 0
                else:
                    logging.warning(f"Edge type {edge_type_to_predict} found in edge_types, but store lacks 'edge_index' or it's not a Tensor.")
            except KeyError:
                 # Isto não deveria acontecer se type_exists_in_list for True, mas é uma salvaguarda
                 logging.error(f"KeyError accessing train_data[{edge_type_to_predict}] even though it's in edge_types list!")
                 type_exists_in_list = False # Marca como não existente para o if abaixo
        # --- END REVISED CHECK LOGIC ---

        logging.debug(f"FINAL CHECK before IF: type_exists_in_list? {type_exists_in_list}")
        logging.debug(f"FINAL CHECK before IF: has_edges? {has_edges}")

        # If the type doesn't exist in the list OR it exists but has no edges
        if not type_exists_in_list or not has_edges:
            logging.error(f"Cannot create train_loader: No edges found for target type {edge_type_to_predict} in training data (Revised check failed).")
            exit(1)

        # If the check passes, we can safely create the loader
        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            edge_label_index=(edge_type_to_predict, edge_store['edge_index']), # Use edge_store from check
            edge_label=torch.ones(edge_store['edge_index'].size(1)),
            neg_sampling_ratio=args.neg_sampling_ratio,
            shuffle=True,
            num_workers=0
        )
        logging.info("Train loader created successfully.")

        # --- Apply similar revised checks for valid_data and test_data ---
        val_loader = None
        valid_type_exists_in_list = edge_type_to_predict in valid_data.edge_types
        valid_has_edges = False
        valid_edge_store = None
        if valid_type_exists_in_list:
            try:
                valid_edge_store = valid_data[edge_type_to_predict]
                if 'edge_index' in valid_edge_store and isinstance(valid_edge_store['edge_index'], torch.Tensor):
                    valid_has_edges = valid_edge_store['edge_index'].shape[1] > 0
            except KeyError:
                 valid_type_exists_in_list = False

        if valid_type_exists_in_list and valid_has_edges:
            val_loader = LinkNeighborLoader(
                valid_data,
                num_neighbors=args.num_neighbors,
                batch_size=args.batch_size,
                edge_label_index=(edge_type_to_predict, valid_edge_store['edge_index']),
                edge_label=torch.ones(valid_edge_store['edge_index'].size(1)),
                neg_sampling_ratio=args.neg_sampling_ratio,
                shuffle=False,
                num_workers=0
            )
            logging.info("Validation loader created.")
        else:
             logging.warning(f"No edges found for target type {edge_type_to_predict} in validation data. Skipping validation loader. (exists_in_list={valid_type_exists_in_list}, has_edges={valid_has_edges})")


        test_loader = None
        test_type_exists_in_list = edge_type_to_predict in test_data.edge_types
        test_has_edges = False
        test_edge_store = None
        if test_type_exists_in_list:
            try:
                test_edge_store = test_data[edge_type_to_predict]
                if 'edge_index' in test_edge_store and isinstance(test_edge_store['edge_index'], torch.Tensor):
                    test_has_edges = test_edge_store['edge_index'].shape[1] > 0
            except KeyError:
                 test_type_exists_in_list = False

        if test_type_exists_in_list and test_has_edges:
            test_loader = LinkNeighborLoader(
                test_data,
                num_neighbors=args.num_neighbors,
                batch_size=args.batch_size,
                edge_label_index=(edge_type_to_predict, test_edge_store['edge_index']),
                edge_label=torch.ones(test_edge_store['edge_index'].size(1)),
                neg_sampling_ratio=args.neg_sampling_ratio,
                shuffle=False,
                num_workers=0
            )
            logging.info("Test loader created.")
        else:
             logging.warning(f"No edges found for target type {edge_type_to_predict} in test data. Skipping test loader. (exists_in_list={test_type_exists_in_list}, has_edges={test_has_edges})")


        logging.info("DataLoaders preparation finished.")
        # Example: Check first batch from train_loader
        try:
            first_batch = next(iter(train_loader))
            logging.info("Sample batch structure (from train_loader):")
            logging.info(first_batch)
            # Check the specific edge type in the batch
            if edge_type_to_predict in first_batch:
                 logging.info(f" Edge label index shape for {edge_type_to_predict}: {first_batch[edge_type_to_predict].edge_label_index.shape}")
                 logging.info(f" Edge label shape for {edge_type_to_predict}: {first_batch[edge_type_to_predict].edge_label.shape}")
            else:
                 logging.warning(f"Target edge type {edge_type_to_predict} not found in the first training batch edge stores.")
        except StopIteration:
            logging.warning("Train loader is empty!")
        except Exception as batch_err:
             logging.error(f"Error inspecting first batch: {batch_err}", exc_info=True)


    except Exception as e:
        logging.error(f"Failed to create DataLoaders: {e}", exc_info=True)
        exit(1)

    # --- Initialize Feature Processor and Model ---
    logging.info("--- Initializing Model ---")
    try:
        node_processor = NodeFeatureProcessor(node_feature_info, args.node_emb_dim).to(args.device)
        model = HeteroGNNLinkPredictor(
            node_types=metadata[0],
            edge_types=metadata[1], # Use all edge types found in data for model structure
            metadata=metadata,
            hidden_channels=args.node_emb_dim, # GNN layers use the projected dimension
            num_gnn_layers=args.num_gnn_layers,
            dropout=args.dropout
        ).to(args.device)
        logging.info(f"Model:\n{model}")

        # Combine parameters from both modules for the optimizer
        optimizer = optim.Adam(list(node_processor.parameters()) + list(model.parameters()), lr=args.lr)
        # Loss is calculated inside train_epoch now
        # criterion = nn.BCEWithLogitsLoss()
    except Exception as e:
        logging.error(f"Failed to initialize model/optimizer: {e}", exc_info=True)
        exit(1)

    # --- Training & Evaluation Loop ---
    logging.info("--- Starting Training ---")
    best_val_auc = 0
    best_epoch = 0
    history = {'epoch': [], 'train_loss': [], 'val_auc': []}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        loss = train_epoch(model, node_processor, train_loader, optimizer, args.device, edge_type_to_predict)

        # Validate (only if val_loader exists)
        val_auc = 0.0 # Default if no validation
        if val_loader:
             val_auc = evaluate(model, node_processor, val_loader, args.device, edge_type_to_predict, metric='AUC')
        else:
             # Log only once or periodically if skipping validation
             if epoch == 1: logging.info("Skipping validation (no validation data/loader for target edge type).")


        end_time = time.time()
        epoch_time = end_time - start_time

        history['epoch'].append(epoch)
        history['train_loss'].append(loss)
        history['val_auc'].append(val_auc) # Store 0.0 if no validation

        logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.2f}s')

        # Save best model checkpoint based on validation AUC (only if validation happens)
        if val_loader and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            try:
                torch.save({
                    'epoch': epoch,
                    'node_processor_state_dict': node_processor.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_auc': best_val_auc,
                    'args': vars(args) # Save args dict
                }, checkpoint_path)
                logging.info(f"  -> New best model saved to {checkpoint_path}")
            except Exception as e:
                 logging.error(f"Error saving checkpoint: {e}")

    logging.info("--- Training Finished ---")
    if val_loader:
        logging.info(f"Best Validation AUC: {best_val_auc:.4f} at Epoch {best_epoch}")
    else:
         logging.info("Training finished (no validation performed). Saving final model.")
         # Save the final model if no validation was done
         checkpoint_path = os.path.join(args.output_dir, 'final_model.pt')
         try:
            torch.save({
                    'epoch': args.epochs,
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
    # Determine which model checkpoint to load
    model_to_load_path = ""
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')

    if val_loader and os.path.exists(best_model_path):
        model_to_load_path = best_model_path
        logging.info(f"Loading best model based on validation from: {model_to_load_path}")
    elif not val_loader and os.path.exists(final_model_path):
         model_to_load_path = final_model_path
         logging.info(f"Loading final model (no validation performed) from: {model_to_load_path}")
    elif os.path.exists(final_model_path): # Fallback to final model if best doesn't exist
         model_to_load_path = final_model_path
         logging.info(f"Best model not found, loading final model from: {model_to_load_path}")
    else:
         logging.warning("No suitable model checkpoint found (.pt file) to run final test.")
         model_to_load_path = None

    test_auc = 0.0 # Default test result
    if model_to_load_path and test_loader: # Only test if model and loader exist
        try:
            checkpoint = torch.load(model_to_load_path, map_location=args.device)
            # Re-initialize models using saved args if possible, otherwise current args
            # Use Namespace to handle potential missing keys gracefully
            loaded_args_dict = checkpoint.get('args', vars(args))
            # Convert device string back to torch.device if necessary
            if isinstance(loaded_args_dict.get('device'), str):
                 loaded_args_dict['device'] = torch.device(loaded_args_dict['device'])
            loaded_args = argparse.Namespace(**loaded_args_dict)


            # Ensure num_neighbors is a list of ints if loaded from checkpoint args
            if isinstance(loaded_args.num_neighbors, str):
                loaded_args.num_neighbors = [int(n) for n in loaded_args.num_neighbors.split(',')]

            # Recreate node processor and model with potentially loaded hyperparameters
            node_processor = NodeFeatureProcessor(node_feature_info, loaded_args.node_emb_dim).to(args.device)
            model = HeteroGNNLinkPredictor(
                node_types=metadata[0],
                edge_types=metadata[1],
                metadata=metadata,
                hidden_channels=loaded_args.node_emb_dim, # Use node_emb_dim for consistency
                num_gnn_layers=loaded_args.num_gnn_layers,
                dropout=loaded_args.dropout # Use loaded dropout
            ).to(args.device)

            node_processor.load_state_dict(checkpoint['node_processor_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model state from epoch {checkpoint.get('epoch', 'N/A')}")

            test_auc = evaluate(model, node_processor, test_loader, args.device, edge_type_to_predict, metric='AUC')
            logging.info(f'Final Test AUC: {test_auc:.4f}')
        except FileNotFoundError:
            logging.error(f"Model checkpoint not found at {model_to_load_path}")
        except Exception as e:
             logging.error(f"Error during testing: {e}", exc_info=True)
    elif not test_loader:
        logging.warning("Skipping testing phase as no test loader was created.")
    elif not model_to_load_path:
         logging.warning("Skipping testing phase as no model checkpoint was found.")


    # Save final results summary
    results = {
        'best_val_auc': best_val_auc if val_loader else None,
        'test_auc': test_auc if test_loader and model_to_load_path else None,
        'best_epoch': best_epoch if val_loader else None,
        'final_epoch': args.epochs,
        'config': vars(args) # Include final config used for this run
        }
    try:
        # Ensure device object is converted to string for JSON serialization
        results['config']['device'] = str(results['config']['device'])
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