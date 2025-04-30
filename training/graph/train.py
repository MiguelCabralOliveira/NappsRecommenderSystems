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
    setup_logging(args.output_dir)

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
                    emb_size = 32 # Example embedding size per categorical feature
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
        for node_type in self.node_types:
            # Handle node types that start with a learnable embedding (no raw features)
            if node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.Embedding):
                 # Assume the batch provides node indices for this type if it has no raw features
                 if not hasattr(data[node_type], 'n_id'):
                      raise ValueError(f"Batch missing node indices ('n_id') for node type {node_type} which uses learnable embeddings.")
                 node_indices = data[node_type].n_id
                 x_combined = self.embeddings[node_type](node_indices)
            else:
                 # Process raw features if they exist
                 features_to_concat = []
                 # Process categorical features
                 if node_type in self.embeddings and isinstance(self.embeddings[node_type], nn.ModuleDict):
                     cat_indices = data[node_type].x_categorical
                     for i, (emb_key, emb_layer) in enumerate(self.embeddings[node_type].items()):
                         features_to_concat.append(emb_layer(cat_indices[:, i]))

                 # Add numeric features
                 if hasattr(data[node_type], 'x_numeric') and data[node_type].x_numeric.shape[1] > 0:
                      features_to_concat.append(data[node_type].x_numeric)

                 if not features_to_concat:
                      # Should ideally be handled by learnable embedding case above, but as fallback:
                      logging.warning(f"Node type {node_type} has no input features and no learnable embedding defined. Creating zeros.")
                      num_nodes = data[node_type].num_nodes if hasattr(data[node_type], 'num_nodes') else data[node_type].n_id.shape[0] # Get num nodes from batch
                      x_combined = torch.zeros((num_nodes, self.target_dim), device=self.projections[node_type].weight.device if hasattr(self.projections[node_type], 'weight') else 'cpu' )
                 else:
                      x_combined = torch.cat(features_to_concat, dim=-1)

            # Project to target dimension (Identity if no projection needed)
            out_x_dict[node_type] = self.projections[node_type](x_combined)

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
            if edge_type_tuple not in batch or 'edge_label_index' not in batch[edge_type_tuple] or 'edge_label' not in batch[edge_type_tuple]:
                 logging.warning(f"Skipping training batch {batch_num} - missing required edge data for {edge_type_tuple}.")
                 continue

            pred_scores = model.predict_link_score(h_dict, batch[edge_type_tuple].edge_label_index, edge_type_tuple)

            # 4. Calculate Loss
            loss = F.binary_cross_entropy_with_logits(pred_scores, batch[edge_type_tuple].edge_label.float())

            # 5. Backpropagate
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred_scores.numel()
            total_examples += pred_scores.numel()

        except KeyError as e:
             logging.warning(f"Skipping training batch {batch_num} due to KeyError: {e}. Check data/batch structure.")
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
        data_dict, metadata, node_feature_info, edge_feature_info = load_hetero_graph_data(args.data_dir, device='cpu') # Load to CPU
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

    # Validate edge type
    if edge_type_to_predict not in metadata[1]:
         logging.error(f"Specified --link-pred-edge '{args.link_pred_edge}' "
                       f"does not correspond to a valid edge type found in the data: {metadata[1]}")
         exit(1)
    if edge_type_to_predict not in train_data.edge_types:
         logging.warning(f"Target edge type {edge_type_to_predict} not found in the training split edges. "
                         "Training might not be effective for this edge type.")


    # --- Prepare DataLoaders ---
    logging.info("--- Preparing DataLoaders ---")
    try:
        # Check if edge type exists and has edges before creating loader
        if edge_type_to_predict not in train_data or train_data[edge_type_to_predict].num_edges == 0:
            logging.error(f"Cannot create train_loader: No edges found for target type {edge_type_to_predict} in training data.")
            exit(1)

        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            edge_label_index=(edge_type_to_predict, train_data[edge_type_to_predict].edge_index),
            edge_label=torch.ones(train_data[edge_type_to_predict].edge_index.size(1)), # Labels for positive edges
            neg_sampling_ratio=args.neg_sampling_ratio,
            shuffle=True,
            num_workers=0 # Adjust based on your system
        )

        # Only create val/test loaders if the edge type exists in the corresponding data splits
        val_loader = None
        if edge_type_to_predict in valid_data and valid_data[edge_type_to_predict].num_edges > 0:
            val_loader = LinkNeighborLoader(
                valid_data, # Use validation graph structure
                num_neighbors=args.num_neighbors,
                batch_size=args.batch_size,
                edge_label_index=(edge_type_to_predict, valid_data[edge_type_to_predict].edge_index),
                edge_label=torch.ones(valid_data[edge_type_to_predict].edge_index.size(1)),
                neg_sampling_ratio=args.neg_sampling_ratio,
                shuffle=False,
                num_workers=0
            )
            logging.info("Validation loader created.")
        else:
             logging.warning(f"No edges found for target type {edge_type_to_predict} in validation data. Skipping validation loader.")


        test_loader = None
        if edge_type_to_predict in test_data and test_data[edge_type_to_predict].num_edges > 0:
            test_loader = LinkNeighborLoader(
                test_data,
                num_neighbors=args.num_neighbors,
                batch_size=args.batch_size,
                edge_label_index=(edge_type_to_predict, test_data[edge_type_to_predict].edge_index),
                edge_label=torch.ones(test_data[edge_type_to_predict].edge_index.size(1)),
                neg_sampling_ratio=args.neg_sampling_ratio,
                shuffle=False,
                num_workers=0
            )
            logging.info("Test loader created.")
        else:
             logging.warning(f"No edges found for target type {edge_type_to_predict} in test data. Skipping test loader.")


        logging.info("DataLoaders preparation finished.")
        # Example: Check first batch from train_loader
        try:
            first_batch = next(iter(train_loader))
            logging.info("Sample batch structure (from train_loader):")
            logging.info(first_batch)
            logging.info(f" Edge label index shape for {edge_type_to_predict}: {first_batch[edge_type_to_predict].edge_label_index.shape}")
            logging.info(f" Edge label shape for {edge_type_to_predict}: {first_batch[edge_type_to_predict].edge_label.shape}")
        except StopIteration:
            logging.warning("Train loader is empty!")
        except KeyError:
             logging.warning(f"Edge type {edge_type_to_predict} not found in the first training batch edges. Check data loading.")

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
            hidden_channels=args.node_emb_dim,
            num_gnn_layers=args.num_gnn_layers,
            dropout=args.dropout
        ).to(args.device)
        logging.info(f"Model:\n{model}")

        optimizer = optim.Adam(list(node_processor.parameters()) + list(model.parameters()), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss() # Still needed if loss calc happens outside train_epoch
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
             logging.info(f"Epoch {epoch}: Skipping validation (no validation data for target edge type).")


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
    if val_loader and os.path.exists(os.path.join(args.output_dir, 'best_model.pt')):
        model_to_load_path = os.path.join(args.output_dir, 'best_model.pt')
        logging.info("Loading best model based on validation.")
    elif not val_loader and os.path.exists(os.path.join(args.output_dir, 'final_model.pt')):
         model_to_load_path = os.path.join(args.output_dir, 'final_model.pt')
         logging.info("Loading final model (no validation was performed).")
    else:
         logging.warning("No suitable model checkpoint found (.pt file) to run final test.")
         model_to_load_path = None

    test_auc = 0.0 # Default test result
    if model_to_load_path and test_loader: # Only test if model and loader exist
        try:
            checkpoint = torch.load(model_to_load_path, map_location=args.device)
            # Re-initialize models before loading state dict - crucial!
            loaded_args = checkpoint.get('args', vars(args)) # Use saved args if possible
            node_processor = NodeFeatureProcessor(node_feature_info, loaded_args['node_emb_dim']).to(args.device)
            model = HeteroGNNLinkPredictor(
                node_types=metadata[0],
                edge_types=metadata[1],
                metadata=metadata,
                hidden_channels=loaded_args['node_emb_dim'],
                num_gnn_layers=loaded_args['num_gnn_layers'],
                dropout=loaded_args['dropout']
            ).to(args.device)

            node_processor.load_state_dict(checkpoint['node_processor_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")

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
        'config': vars(args) # Include final config
        }
    try:
        with open(os.path.join(args.output_dir, 'final_summary.json'), 'w') as f:
            json.dump(results, f, indent=4, default=str) # use default=str for device obj
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