# /training/graph/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- ALTERAÇÃO: Importar GATConv e remover SAGEConv se não for mais usado ---
from torch_geometric.nn import HeteroConv, GATConv, Linear
import logging
# from torch_geometric.nn import HeteroConv, SAGEConv, Linear # Linha antiga

class HeteroGNNLinkPredictor(nn.Module):
    """
    A Heterogeneous GAT-based model for link prediction between
    'Person' and 'Product' nodes.

    Assumes node features are processed into initial embeddings before GNN layers.
    Uses dot product for link prediction score.
    """
    def __init__(self, node_types, edge_types, metadata,
                 hidden_channels, num_gnn_layers=2, dropout=0.2, gat_heads=1): # Adicionado gat_heads
        """
        Args:
            node_types (list[str]): List of node type names (e.g., ['Person', 'Product']).
            edge_types (list[tuple]): List of edge type tuples (e.g., [('Person', 'viewed', 'Product'), ...]).
            metadata (tuple): Metadata tuple (node_types, edge_types) for HeteroConv.
                               Should be generated based on the actual edge types found in the data.
            hidden_channels (int): Dimension of the hidden embeddings after GNN layers.
            num_gnn_layers (int): Number of GNN layers.
            dropout (float): Dropout probability used within GATConv and potentially between layers.
            gat_heads (int): Number of attention heads in GATConv layers. Output dim will be hidden_channels * gat_heads.

        Note: If gat_heads > 1, the output dimension of HeteroConv will be hidden_channels * gat_heads.
              Consider adding a Linear layer after the GNNs if you need the final embedding size
              to be exactly hidden_channels. For simplicity with heads=1, output dim remains hidden_channels.
        """
        super().__init__()

        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if not isinstance(num_gnn_layers, int) or num_gnn_layers <= 0:
            raise ValueError("num_gnn_layers must be a positive integer.")
        if not isinstance(gat_heads, int) or gat_heads <= 0:
            raise ValueError("gat_heads must be a positive integer.")


        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.metadata = metadata # Store metadata (node_types, edge_types)
        self.gat_heads = gat_heads

        # --- GNN Layers ---
        self.convs = nn.ModuleList()
        input_channels = hidden_channels # Assuming input from NodeFeatureProcessor is hidden_channels

        for i in range(num_gnn_layers):
            # Determine output channels for this layer
            # If it's the last layer and heads > 1, output is hidden * heads
            # Otherwise, intermediate layers also output hidden * heads (GAT standard)
            # For simplicity, if heads=1, output is just hidden_channels
            # If heads > 1, we might need a final projection layer later.
            # Let's assume for now the output of GAT is hidden_channels (by setting heads=1 or adjusting GAT output)
            # To keep it simple: Use heads=1 OR ensure GATConv output dim is hidden_channels
            # Example using heads=1:
            out_channels = hidden_channels # Output dim matches hidden_channels if heads=1

            # --- ALTERAÇÃO: Usar GATConv em vez de SAGEConv ---
            conv_dict = {
                edge_type: GATConv(in_channels=(-1, -1), # Infer input sizes
                                   out_channels=out_channels,
                                   heads=self.gat_heads,
                                   dropout=self.dropout, # Dropout within GAT attention mechanism
                                   add_self_loops=False, # HeteroConv might handle self-loops depending on edge_types
                                   concat=False if self.gat_heads == 1 else True # Concatenate heads only if > 1
                                  )
                for edge_type in edge_types
            }
            # --- FIM ALTERAÇÃO ---

            # Wrap the GATConvs in HeteroConv
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)

            # Update input channels for the next layer if heads > 1 and concat=True
            # If heads=1 or concat=False, input_channels remains hidden_channels
            if self.gat_heads > 1 and i < num_gnn_layers - 1: # Check concat implicitly via heads > 1
                 input_channels = out_channels * self.gat_heads # Input for next layer is concatenated heads
            # else: # If heads=1, input for next layer is just out_channels (which is hidden_channels)
            #      input_channels = out_channels # This line is implicitly covered

        # --- ALTERAÇÃO: Final projection if using multiple heads ---
        # If multiple heads were used, the output dim is hidden_channels * gat_heads.
        # We might need a final linear layer per node type to project back to hidden_channels.
        self.final_projection = nn.ModuleDict()
        if self.gat_heads > 1:
            for node_type in node_types:
                 # Input dim is hidden_channels * heads, output is hidden_channels
                 self.final_projection[node_type] = Linear(hidden_channels * self.gat_heads, hidden_channels)
        else:
             # If only 1 head, no projection needed unless desired for other reasons
             for node_type in node_types:
                  self.final_projection[node_type] = nn.Identity()
        # --- FIM ALTERAÇÃO ---


    def forward(self, x_dict, edge_index_dict):
        """
        Performs message passing using GAT layers to compute final node embeddings.

        Args:
            x_dict (dict[str, torch.Tensor]): Dictionary mapping node types to initial feature tensors.
                                               Shape: [num_nodes, input_feat_dim (should match hidden_channels)].
            edge_index_dict (dict[tuple, torch.Tensor]): Dictionary mapping edge types to edge_index tensors.
                                                          Shape: [2, num_edges].

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping node types to final embedding tensors.
                                      Shape: [num_nodes, hidden_channels].
        """
        h_dict = x_dict # Start with initial features/embeddings

        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)

            # Apply activation and dropout (after each layer except the last one IF no final projection)
            # If we have a final projection, apply activation/dropout before it.
            # Let's apply after each conv layer for now, similar to SAGE implementation.
            # Note: GATConv already has internal dropout. This adds dropout *between* layers.
            if i < self.num_gnn_layers - 1:
                 h_dict = {key: F.elu(h) for key, h in h_dict.items()} # ELU often works well with GAT
                 h_dict = {key: F.dropout(h, p=self.dropout, training=self.training)
                           for key, h in h_dict.items()}

        # Apply final projection if defined (i.e., if gat_heads > 1)
        # Apply activation after the final GNN layer / before final projection? Or after projection? Let's try after projection.
        h_dict = {node_type: self.final_projection[node_type](h) for node_type, h in h_dict.items()}
        # Optional: Final activation after projection
        h_dict = {key: F.elu(h) for key, h in h_dict.items()}


        return h_dict # Return final embeddings


    def predict_link_score(self, h_dict, edge_label_index, edge_type):
        """
        Computes the link prediction score (dot product) for given edges of a specific type.
        (Function remains the same as before)
        """
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
             raise ValueError(f"edge_type must be a tuple of length 3, got {edge_type}")

        src_node_type, _, dst_node_type = edge_type

        if src_node_type not in h_dict or dst_node_type not in h_dict:
             raise KeyError(f"Embeddings for source '{src_node_type}' or destination '{dst_node_type}' "
                           f"not found in h_dict. Available keys: {list(h_dict.keys())}")

        if edge_label_index.dim() != 2 or edge_label_index.shape[0] != 2:
             raise ValueError(f"edge_label_index must have shape [2, num_edges], got {edge_label_index.shape}")

        src_indices, dst_indices = edge_label_index

        try:
            src_emb = h_dict[src_node_type][src_indices]
            dst_emb = h_dict[dst_node_type][dst_indices]
        except IndexError as e:
            max_src_idx = h_dict[src_node_type].shape[0] - 1
            max_dst_idx = h_dict[dst_node_type].shape[0] - 1
            logging.error(f"IndexError accessing embeddings for link prediction:")
            logging.error(f"  Edge type: {edge_type}")
            logging.error(f"  Source node type: {src_node_type}, Max index allowed: {max_src_idx}, Indices accessed max: {src_indices.max().item() if src_indices.numel() > 0 else 'N/A'}")
            logging.error(f"  Dest node type: {dst_node_type}, Max index allowed: {max_dst_idx}, Indices accessed max: {dst_indices.max().item() if dst_indices.numel() > 0 else 'N/A'}")
            logging.error(f"  Original error: {e}", exc_info=True)
            # Tentativa de dar mais contexto sobre os índices problemáticos
            if src_indices.numel() > 0 and src_indices.max() > max_src_idx:
                 logging.error(f"  Problematic source indices: {src_indices[src_indices > max_src_idx]}")
            if dst_indices.numel() > 0 and dst_indices.max() > max_dst_idx:
                 logging.error(f"  Problematic destination indices: {dst_indices[dst_indices > max_dst_idx]}")
            raise e # Re-raise the exception after logging

        # Compute score using dot product
        score = (src_emb * dst_emb).sum(dim=-1)

        return score