# /training/graph/evaluate.py

import torch
from sklearn.metrics import roc_auc_score
import numpy as np

@torch.no_grad()
def calculate_auc(model, node_feature_processor, loader, device, edge_type_tuple):
    """
    Calculates the Area Under the ROC Curve (AUC) for link prediction.

    Args:
        model: The trained HeteroGNNLinkPredictor model.
        node_feature_processor: The NodeFeatureProcessor module.
        loader: DataLoader (e.g., LinkNeighborLoader) for the evaluation split.
        device: The device ('cpu' or 'cuda').
        edge_type_tuple: The specific edge type (e.g., ('Person', 'purchased', 'Product'))
                         for which to evaluate links.

    Returns:
        float: The calculated AUC score.
    """
    model.eval()
    node_feature_processor.eval() # Set processor to eval mode too
    all_preds = []
    all_labels = []

    print(f"  Evaluating AUC for edge type: {edge_type_tuple}...")
    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        num_batches += 1
        try:
            # 1. Process features
            initial_x_dict = node_feature_processor(batch)

            # 2. Get final embeddings
            h_dict = model(initial_x_dict, batch.edge_index_dict)

            # 3. Predict scores for the target edge type in the batch
            # Make sure the edge type exists in the batch edge_label_index
            if edge_type_tuple not in batch or 'edge_label_index' not in batch[edge_type_tuple]:
                 print(f"    Warning: Skipping batch {num_batches} during AUC calculation - missing edge_label_index for {edge_type_tuple}.")
                 continue

            pred_scores = model.predict_link_score(h_dict, batch[edge_type_tuple].edge_label_index, edge_type_tuple)

            # Ensure labels exist
            if 'edge_label' not in batch[edge_type_tuple]:
                print(f"    Warning: Skipping batch {num_batches} during AUC calculation - missing edge_label for {edge_type_tuple}.")
                continue

            all_preds.append(pred_scores.cpu())
            all_labels.append(batch[edge_type_tuple].edge_label.cpu())

        except KeyError as e:
            print(f"    Warning: Skipping batch {num_batches} due to KeyError: {e}. Likely missing node/edge type in batch.")
            continue
        except Exception as e:
             print(f"    Error during evaluation batch {num_batches}: {e}")
             # Decide if you want to skip or raise
             continue # Skip batch on error

    print(f"  Finished processing {num_batches} batches for AUC.")

    if not all_preds or not all_labels:
        print("  Warning: No predictions or labels collected during AUC evaluation. Returning 0.0.")
        return 0.0

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    if len(np.unique(labels)) < 2:
        print(f"  Warning: Only one class present in labels ({np.unique(labels)}). AUC is not defined. Returning 0.5.")
        return 0.5 # Or handle as appropriate (e.g., return NaN or raise error)

    try:
        auc = roc_auc_score(labels, preds)
        print(f"  AUC calculated: {auc:.4f}")
        return auc
    except ValueError as e:
         print(f"  Error calculating AUC score: {e}. Check labels and predictions.")
         print(f"  Labels unique values: {np.unique(labels)}")
         print(f"  Predictions sample: {preds[:10]}")
         return 0.0 # Or NaN


def calculate_hit_rate_at_k(model, node_feature_processor, loader, device, edge_type_tuple, k=10):
    """
    Calculates Hit Rate @ K.
    NOTE: LinkNeighborLoader might not be ideal for direct HitRate@K calculation
          if it doesn't guarantee the positive edge and its corresponding negative
          samples are evaluated together for the *same source node*.
          This implementation assumes the loader provides batches suitable for ranking,
          which might require a custom loader or post-processing.
          This is a simplified example assuming direct ranking is possible from loader output.

    Args:
        model: The trained model.
        node_feature_processor: The node feature processor.
        loader: DataLoader. **Crucially, this loader needs to yield batches where for each positive edge (src, pos_dst), there are multiple negative edges (src, neg_dst_1), (src, neg_dst_2), ... for the *same source node* src.**
        device: Device ('cpu' or 'cuda').
        edge_type_tuple: Edge type to evaluate.
        k (int): The 'K' in HitRate@K.

    Returns:
        float: Hit Rate @ K score.
    """
    model.eval()
    node_feature_processor.eval()
    hits = 0
    total_samples = 0 # Total number of positive samples evaluated

    print(f"  Evaluating HitRate@{k} for edge type: {edge_type_tuple}...")
    print(f"  WARNING: This evaluation assumes the loader provides positive edge scores alongside corresponding negative scores for ranking.")

    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        num_batches += 1
        try:
            # 1. Process features & Get embeddings
            initial_x_dict = node_feature_processor(batch)
            h_dict = model(initial_x_dict, batch.edge_index_dict)

            # 2. Predict scores
            if edge_type_tuple not in batch or 'edge_label_index' not in batch[edge_type_tuple] or 'edge_label' not in batch[edge_type_tuple]:
                print(f"    Warning: Skipping batch {num_batches} during HitRate calculation - missing required data for {edge_type_tuple}.")
                continue

            edge_label_index = batch[edge_type_tuple].edge_label_index
            edge_label = batch[edge_type_tuple].edge_label
            pred_scores = model.predict_link_score(h_dict, edge_label_index, edge_type_tuple)

            # --- Logic to calculate HitRate@K ---
            # This requires grouping scores by source node and checking rank.
            # It's complex with standard loaders. Assuming a simplified structure for demo:
            # Pair positive edge scores with their corresponding negative scores based on the loader's structure.
            # This heavily depends on how LinkNeighborLoader samples negatives and structures the batch.

            # **Simplified Example (Likely Needs Adjustment based on Loader Output):**
            # Assume edge_label_index contains blocks like [pos_src, neg_src1, neg_src2,...], [pos_dst, neg_dst1, neg_dst2,...]
            # And edge_label is [1, 0, 0, ...]

            # Placeholder logic - requires knowledge of how loader pairs positive/negative samples
            # You might need to iterate through unique source nodes in the batch label index,
            # gather scores for positive/negative edges associated with that source, rank them, and check hits.
            positive_mask = edge_label == 1
            negative_mask = edge_label == 0

            # This is a very basic check, not a proper ranked evaluation
            if positive_mask.sum() > 0 and negative_mask.sum() > 0 :
                 # Example: Check if max score among positives is higher than max score among negatives (not HitRate@K)
                 # A real HitRate@K needs ranking within groups associated with the same source node.
                 # This requires careful handling of indices and potentially a different loader setup.
                 pass # Implement proper HitRate@K logic here if needed

            # Since the loader structure for ranking isn't guaranteed, we'll skip the actual calculation
            # print("    HitRate@K calculation skipped due to loader structure uncertainty.")


        except KeyError as e:
            print(f"    Warning: Skipping batch {num_batches} due to KeyError: {e}.")
            continue
        except Exception as e:
             print(f"    Error during evaluation batch {num_batches}: {e}")
             continue

    print(f"  Finished processing {num_batches} batches for HitRate.")

    if total_samples == 0:
        print("  Warning: No positive samples found or processed for HitRate evaluation. Returning 0.0.")
        return 0.0

    hit_rate = hits / total_samples
    print(f"  HitRate@{k} calculated: {hit_rate:.4f} (Based on {hits} hits out of {total_samples} positive samples - REQUIRES CORRECT IMPLEMENTATION)")
    return hit_rate # Return the calculated rate (needs implementation)