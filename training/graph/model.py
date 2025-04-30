# /training/graph/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroGNNLinkPredictor(nn.Module):
    """
    A Heterogeneous GraphSAGE-based model for link prediction between
    'Person' and 'Product' nodes.

    Assumes node features are processed into initial embeddings before GNN layers.
    Uses dot product for link prediction score.
    """
    def __init__(self, node_types, edge_types, metadata,
                 hidden_channels, num_gnn_layers=2, dropout=0.2):
        """
        Args:
            node_types (list[str]): List of node type names (e.g., ['Person', 'Product']).
            edge_types (list[tuple]): List of edge type tuples (e.g., [('Person', 'viewed', 'Product'), ...]).
            metadata (tuple): Metadata tuple (node_types, edge_types) for HeteroConv.
                               Should be generated based on the actual edge types found in the data.
            hidden_channels (int): Dimension of the hidden embeddings after GNN layers.
            num_gnn_layers (int): Number of GNN layers.
            dropout (float): Dropout probability.

        Note: This model expects the initial node features (x_dict passed to forward)
              to already be projected/embedded to a suitable dimension (e.g., hidden_channels
              or another dimension handled by the first SAGEConv).
              The SAGEConv layers defined here expect input dimension == output dimension (hidden_channels)
              for simplicity in this example, except potentially the very first layer if input projection
              is handled externally. If input features `x_dict` have different dims,
              the first SAGEConv layer or an initial projection layer needs adjustment.
        """
        super().__init__()

        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if not isinstance(num_gnn_layers, int) or num_gnn_layers <= 0:
            raise ValueError("num_gnn_layers must be a positive integer.")

        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.metadata = metadata # Store metadata (node_types, edge_types)

        # --- GNN Layers ---
        self.convs = nn.ModuleList()
        for i in range(num_gnn_layers):
            # Create SAGEConv layers for each edge type
            # For HeteroConv, specify input feature sizes as (-1, -1) to infer from data
            # Output size is hidden_channels
            conv_dict = {
            edge_type: SAGEConv((-1, -1), hidden_channels, aggr='mean')
            for edge_type in edge_types
        }

            # Wrap the SAGEConvs in HeteroConv
            # aggr='sum' aggregates results for node types receiving from multiple edge types
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)

        # Optional: A final linear layer per node type if needed for projection after GNNs
        # self.final_lins = nn.ModuleDict()
        # for node_type in node_types:
        #     self.final_lins[node_type] = Linear(hidden_channels, hidden_channels)


    def forward(self, x_dict, edge_index_dict):
        """
        Performs message passing to compute final node embeddings.

        Args:
            x_dict (dict[str, torch.Tensor]): Dictionary mapping node types to initial feature tensors.
                                               Shape for tensor node_type: [num_nodes_in_batch, input_feat_dim]
                                               Assumes input_feat_dim matches hidden_channels or is handled
                                               by an external projection layer before calling this.
            edge_index_dict (dict[tuple, torch.Tensor]): Dictionary mapping edge types to edge_index tensors.
                                                          Shape for tensor edge_type: [2, num_edges_in_batch]

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping node types to final embedding tensors.
                                      Shape for tensor node_type: [num_nodes_in_batch, hidden_channels].
        """
        h_dict = x_dict # Start with initial features/embeddings

        for i, conv in enumerate(self.convs):
            # Apply HeteroConv layer
            # Input h_dict from previous layer (or initial x_dict), edge indices for this batch
            h_dict = conv(h_dict, edge_index_dict)

            # Apply activation and dropout (after each layer except the last)
            if i < self.num_gnn_layers - 1:
                 # Apply ReLU activation to the output of the convolution for each node type
                 h_dict = {key: F.relu(h) for key, h in h_dict.items()}
                 # Apply dropout
                 h_dict = {key: F.dropout(h, p=self.dropout, training=self.training)
                           for key, h in h_dict.items()}

        # Optional: Apply final linear layers if defined
        # h_dict = {node_type: self.final_lins[node_type](h) for node_type, h in h_dict.items()}

        return h_dict # Return final embeddings after all GNN layers


    def predict_link_score(self, h_dict, edge_label_index, edge_type):
        """
        Computes the link prediction score (dot product) for given edges of a specific type.

        Args:
            h_dict (dict[str, torch.Tensor]): Dictionary of final node embeddings from forward pass.
            edge_label_index (torch.Tensor): Edge indices [2, num_edges_to_predict]
                                             for which to compute scores. Rows contain
                                             indices for source and destination nodes *within the batch*.
            edge_type (tuple): The specific edge type (e.g., ('Person', 'purchased', 'Product'))
                               to predict scores for.

        Returns:
            torch.Tensor: A tensor of prediction scores (logits before sigmoid). Shape: [num_edges_to_predict].
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

        # Get embeddings for the specific source and destination node indices involved in the links
        # These indices usually correspond to nodes within the current batch/subgraph
        try:
            src_emb = h_dict[src_node_type][src_indices]
            dst_emb = h_dict[dst_node_type][dst_indices]
        except IndexError as e:
            print(f"IndexError accessing embeddings: {e}")
            print(f"Source node type: {src_node_type}, Max index allowed: {h_dict[src_node_type].shape[0]-1}, Indices accessed: {src_indices.max().item()}")
            print(f"Dest node type: {dst_node_type}, Max index allowed: {h_dict[dst_node_type].shape[0]-1}, Indices accessed: {dst_indices.max().item()}")
            raise e

        # Compute score using dot product
        # Ensure embeddings have the same dimension (hidden_channels)
        score = (src_emb * dst_emb).sum(dim=-1)

        return score

# Note: The initial feature processing (creating x_dict with embeddings/projections
# from raw numeric/categorical features) is expected to happen outside this model definition,
# typically in the Dataset loading or the training script before calling model.forward().