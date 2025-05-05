# /training/graph/evaluate.py

import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import logging

@torch.no_grad()
def calculate_auc(model, node_feature_processor, loader, device, edge_type_tuple):
    model.eval()
    node_feature_processor.eval()
    all_preds = []
    all_labels = []

    logging.info(f"  Evaluating AUC for edge type signature: {edge_type_tuple}...")
    num_batches = 0
    skipped_batches_no_labels = 0
    processed_samples = 0

    for batch in loader:
        batch = batch.to(device)
        num_batches += 1

        try:
            # --- CORREÇÃO v11 (Direct access with try-except) ---
            try:
                # Attempt direct access
                supervision_edge_label_index = batch[edge_type_tuple].edge_label_index
                supervision_edge_label = batch[edge_type_tuple].edge_label

                # Check if the labels are not empty (important!)
                if supervision_edge_label_index.shape[1] == 0:
                    skipped_batches_no_labels += 1 # Count as skipped due to no labels
                    logging.warning(f"    Skipping evaluation batch {num_batches} - Nested 'edge_label_index' for type {edge_type_tuple} is empty.")
                    continue

            except (KeyError, AttributeError):
                # This block executes if the key edge_type_tuple doesn't exist OR
                # if the object at that key doesn't have .edge_label_index or .edge_label
                skipped_batches_no_labels += 1
                # Log only periodically or at DEBUG level
                if num_batches % 50 == 1 or num_batches == len(loader):
                    logging.warning(f"    Skipping evaluation batch {num_batches}/{len(loader)} - Failed to access nested 'edge_label_index' or 'edge_label' for type {edge_type_tuple}. Batch keys: {list(batch.keys())}")
                else:
                     logging.debug(f"    Skipping evaluation batch {num_batches} - Failed to access nested 'edge_label_index' or 'edge_label' for type {edge_type_tuple}. Batch keys: {list(batch.keys())}")
                continue
            # --- FIM CORREÇÃO v11 ---


            # 1. Process features
            initial_x_dict = node_feature_processor(batch)

            # 2. Get embeddings
            h_dict = model(initial_x_dict, batch.edge_index_dict)

            # 3. Predict scores
            pred_scores = model.predict_link_score(h_dict, supervision_edge_label_index, edge_type_tuple)

            # Append results
            all_preds.append(pred_scores.cpu())
            all_labels.append(supervision_edge_label.cpu())
            processed_samples += supervision_edge_label.numel()

        except KeyError as e:
            # This KeyError would likely be for accessing node features/embeddings
            logging.warning(f"    Warning: Skipping evaluation batch {num_batches} due to KeyError during feature/embedding access: {e}. Check h_dict/edge_index_dict keys. Available batch keys: {list(batch.keys())}", exc_info=False)
            continue
        except IndexError as e:
             logging.error(f"    IndexError during evaluation batch {num_batches}: {e}. Check node indices and embedding access.", exc_info=True)
             continue
        except Exception as e:
             logging.error(f"    Error during evaluation batch {num_batches}: {e}", exc_info=True)
             continue

    logging.info(f"  Finished processing {num_batches} batches for AUC.")
    if skipped_batches_no_labels > 0:
        logging.warning(f"    Skipped {skipped_batches_no_labels}/{num_batches} batches because nested labels for {edge_type_tuple} were missing/empty/inaccessible.")

    if not all_preds or not all_labels:
        logging.warning(f"  Warning: No predictions or labels collected during AUC evaluation ({processed_samples} samples processed across {num_batches-skipped_batches_no_labels} batches). Returning 0.0.")
        return 0.0

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    if len(np.unique(labels)) < 2:
        if len(labels) > 0:
            logging.warning(f"  Warning: Only one class present in labels ({np.unique(labels)}). AUC is not defined. Returning 0.5.")
            return 0.5
        else:
            logging.warning(f"  Warning: No labels collected. AUC is not defined. Returning 0.0.")
            return 0.0

    try:
        auc = roc_auc_score(labels, preds)
        logging.info(f"  AUC calculated: {auc:.4f} (on {len(labels)} samples from {num_batches-skipped_batches_no_labels} batches)")
        return auc
    except ValueError as e:
         logging.error(f"  Error calculating AUC score: {e}. Check labels and predictions.")
         logging.error(f"  Labels unique values: {np.unique(labels)}")
         logging.error(f"  Predictions sample (first 10): {preds[:10]}")
         return 0.0


# --- calculate_hit_rate_at_k (update access logic similarly if used) ---
def calculate_hit_rate_at_k(model, node_feature_processor, loader, device, edge_type_tuple, k=10):
    model.eval()
    node_feature_processor.eval()
    hits = 0
    total_samples = 0 # Total number of positive samples evaluated

    logging.info(f"  Evaluating HitRate@{k} for edge type: {edge_type_tuple}...")
    logging.warning(f"  WARNING: This evaluation assumes the loader provides positive edge scores alongside corresponding negative scores for ranking.")

    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        num_batches += 1
        try:
            # --- Refined Check Logic v10 (Access top-level) ---
            if not (hasattr(batch, 'edge_label_index') and hasattr(batch, 'edge_label')):
                logging.warning(f"    Warning: Skipping batch {num_batches} during HitRate calculation - Missing top-level labels.")
                continue

            edge_label_index = batch.edge_label_index
            edge_label = batch.edge_label

            if edge_label_index.shape[1] == 0:
                logging.warning(f"    Warning: Skipping batch {num_batches} during HitRate calculation - Top-level labels are empty.")
                continue
            # --- End Refined Check v10 ---

            # 1. Process features & Get embeddings
            initial_x_dict = node_feature_processor(batch)
            h_dict = model(initial_x_dict, batch.edge_index_dict)

            # 2. Predict scores
            pred_scores = model.predict_link_score(h_dict, edge_label_index, edge_type_tuple)

            # --- Logic to calculate HitRate@K ---
            # (Placeholder logic remains the same - needs proper implementation based on batch structure)
            positive_mask = edge_label == 1
            negative_mask = edge_label == 0

            if positive_mask.sum() > 0 and negative_mask.sum() > 0 :
                 # Implement proper HitRate@K logic here if needed
                 # This part is still complex and depends heavily on how negatives are sampled and grouped
                 pass

            # logging.info("    HitRate@K calculation skipped due to loader structure uncertainty.")


        except KeyError as e:
            logging.warning(f"    Warning: Skipping batch {num_batches} due to KeyError during prediction/embedding access: {e}.")
            continue
        except Exception as e:
             logging.error(f"    Error during evaluation batch {num_batches}: {e}")
             continue

    logging.info(f"  Finished processing {num_batches} batches for HitRate.")

    if total_samples == 0:
        logging.warning("  Warning: No positive samples found or processed for HitRate evaluation. Returning 0.0.")
        return 0.0

    hit_rate = hits / total_samples
    logging.info(f"  HitRate@{k} calculated: {hit_rate:.4f} (Based on {hits} hits out of {total_samples} positive samples - REQUIRES CORRECT IMPLEMENTATION)")
    return hit_rate # Return the calculated rate (needs implementation)