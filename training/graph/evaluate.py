# /training/graph/evaluate.py

import torch
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import math # Para NDCG

# --- Função helper para salvar plots ---
def save_plot(fig, output_dir, filename):
    """Salva a figura matplotlib no diretório especificado."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight')
        logging.info(f"  Plot saved to: {filepath}")
        plt.close(fig) # Fecha a figura para libertar memória
    except Exception as e:
        logging.error(f"  Error saving plot {filename}: {e}")


@torch.no_grad()
def calculate_auc(model, node_feature_processor, loader, device, edge_type_tuple, output_dir=None, epoch=None):
    """
    Calculates AUC and optionally saves plots.

    Args:
        model: The trained GNN model.
        node_feature_processor: The node feature processor module.
        loader: DataLoader for evaluation data.
        device: The device to run evaluation on.
        edge_type_tuple: The specific edge type to evaluate.
        output_dir (str, optional): Directory to save plots. Defaults to None (no plots saved).
        epoch (int, optional): Current epoch number (used for plot filenames). Defaults to None.
    """
    model.eval()
    node_feature_processor.eval()
    all_preds = []
    all_labels = []

    logging.info(f"  Evaluating AUC for edge type signature: {edge_type_tuple}...")
    num_batches = 0
    skipped_batches_no_labels = 0
    processed_samples = 0
    processed_batches_count = 0

    for batch in loader:
        batch = batch.to(device)
        num_batches += 1

        supervision_edge_label_index = None
        supervision_edge_label = None
        labels_found = False

        # --- Acesso robusto aos labels (v12) ---
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
            if num_batches % 50 == 1 or num_batches == len(loader):
                logging.warning(f"    Skipping evaluation batch {num_batches}/{len(loader)} - Failed to access 'edge_label_index' or 'edge_label' for type {edge_type_tuple} (tried nested and top-level). Batch keys: {list(batch.keys())}")
            else:
                 logging.debug(f"    Skipping evaluation batch {num_batches} - Failed to access 'edge_label_index' or 'edge_label' for type {edge_type_tuple}. Batch keys: {list(batch.keys())}")
            continue

        if supervision_edge_label_index.shape[1] == 0:
            skipped_batches_no_labels += 1
            logging.warning(f"    Skipping evaluation batch {num_batches} - 'edge_label_index' for type {edge_type_tuple} is empty (shape: {supervision_edge_label_index.shape}).")
            continue
        # --- Fim Acesso ---

        try:
            initial_x_dict = node_feature_processor(batch)
            h_dict = model(initial_x_dict, batch.edge_index_dict)
            pred_scores = model.predict_link_score(h_dict, supervision_edge_label_index, edge_type_tuple)
            pred_probs = torch.sigmoid(pred_scores).cpu()

            all_preds.append(pred_probs)
            all_labels.append(supervision_edge_label.cpu())
            processed_samples += supervision_edge_label.numel()
            processed_batches_count +=1

        except KeyError as e:
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
        logging.warning(f"    Skipped {skipped_batches_no_labels}/{num_batches} batches because labels for {edge_type_tuple} were missing/empty/inaccessible.")

    if not all_preds or not all_labels:
        logging.warning(f"  Warning: No predictions or labels collected during AUC evaluation ({processed_samples} samples processed across {processed_batches_count} batches). Returning 0.0.")
        return 0.0

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    if len(np.unique(labels)) < 2:
        if len(labels) > 0:
            logging.warning(f"  Warning: Only one class present in labels ({np.unique(labels)}). AUC is not defined. Returning 0.5.")
            if output_dir is not None: # Plotar histograma mesmo com uma classe, se pedido
                try:
                    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 5))
                    ax_hist.hist(preds, bins=30, alpha=0.7, label=f'Class {np.unique(labels)[0]} Preds')
                    ax_hist.set_title(f'Prediction Distribution (Only Class {np.unique(labels)[0]} Present)')
                    ax_hist.set_xlabel('Predicted Probability')
                    ax_hist.set_ylabel('Frequency')
                    ax_hist.legend()
                    plot_filename = f"prediction_histogram_epoch{epoch if epoch else 'final'}_single_class.png"
                    save_plot(fig_hist, output_dir, plot_filename)
                except Exception as plot_err:
                    logging.error(f"  Error generating histogram plot for single class: {plot_err}")
            return 0.5
        else:
            logging.warning(f"  Warning: No labels collected. AUC is not defined. Returning 0.0.")
            return 0.0

    try:
        auc = roc_auc_score(labels, preds)
        logging.info(f"  AUC calculated: {auc:.4f} (on {len(labels)} samples from {processed_batches_count} batches)")

        # Gerar e salvar plots APENAS se output_dir for fornecido (normalmente só no teste final)
        if output_dir is not None:
            plot_suffix = f"_epoch{epoch}" if epoch is not None else "_final"

            # 1. Histograma das Predições
            try:
                fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 5))
                preds_pos = preds[labels == 1]
                preds_neg = preds[labels == 0]
                ax_hist.hist(preds_neg, bins=30, alpha=0.7, label='Negative Class (0)', color='red')
                ax_hist.hist(preds_pos, bins=30, alpha=0.7, label='Positive Class (1)', color='green')
                ax_hist.set_title(f'Prediction Distribution - {edge_type_tuple}{plot_suffix}')
                ax_hist.set_xlabel('Predicted Probability')
                ax_hist.set_ylabel('Frequency')
                ax_hist.legend()
                save_plot(fig_hist, output_dir, f"prediction_histogram{plot_suffix}.png")
            except Exception as plot_err:
                logging.error(f"  Error generating histogram plot: {plot_err}")

            # 2. Curva ROC
            try:
                fpr, tpr, thresholds = roc_curve(labels, preds)
                fig_roc, ax_roc = plt.subplots(1, 1, figsize=(6, 6))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'Receiver Operating Characteristic - {edge_type_tuple}{plot_suffix}')
                ax_roc.legend(loc="lower right")
                ax_roc.grid(True)
                save_plot(fig_roc, output_dir, f"roc_curve{plot_suffix}.png")
            except Exception as plot_err:
                logging.error(f"  Error generating ROC curve plot: {plot_err}")

        return auc
    except ValueError as e:
         logging.error(f"  Error calculating AUC score: {e}. Check labels and predictions.")
         logging.error(f"  Labels unique values: {np.unique(labels)}")
         logging.error(f"  Predictions sample (first 10): {preds[:10]}")
         return 0.0


# --- Função para calcular métricas de ranking (Com correção NDCG) ---
@torch.no_grad()
def calculate_ranking_metrics(model, node_feature_processor, loader, device, edge_type_tuple, k_list=[10, 20]):
    """
    Calculates Precision@K, Recall@K, and NDCG@K based on loader output.
    Note: This provides an *approximation* based on the positive/negative samples
          present in the batches provided by LinkNeighborLoader. It does not rank
          against all possible items.
    """
    model.eval()
    node_feature_processor.eval()

    all_user_metrics = defaultdict(lambda: defaultdict(list))
    max_k = max(k_list)

    logging.info(f"  Evaluating Ranking Metrics @{k_list} for edge type: {edge_type_tuple}...")
    num_batches = 0
    skipped_batches_no_labels = 0
    processed_users = 0 # Contar utilizadores únicos com itens relevantes encontrados nos batches

    for batch in loader:
        batch = batch.to(device)
        num_batches += 1

        supervision_edge_label_index = None
        supervision_edge_label = None
        labels_found = False

        # --- Acesso robusto aos labels (v12) ---
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
            continue
        if supervision_edge_label_index.shape[1] == 0:
            skipped_batches_no_labels += 1
            continue
        # --- Fim Acesso ---

        try:
            initial_x_dict = node_feature_processor(batch)
            h_dict = model(initial_x_dict, batch.edge_index_dict)
            pred_scores = model.predict_link_score(h_dict, supervision_edge_label_index, edge_type_tuple)

            src_nodes = supervision_edge_label_index[0].cpu().numpy()
            dst_nodes = supervision_edge_label_index[1].cpu().numpy()
            scores = pred_scores.cpu().numpy()
            labels = supervision_edge_label.cpu().numpy()

            user_predictions = defaultdict(list)
            for i in range(len(src_nodes)):
                user_predictions[src_nodes[i]].append((scores[i], labels[i], dst_nodes[i]))

            users_processed_in_batch = set()
            for user_id, predictions in user_predictions.items():
                predictions.sort(key=lambda x: x[0], reverse=True)
                relevant_items = {item_id for score, label, item_id in predictions if label == 1}
                num_relevant = len(relevant_items)

                if num_relevant == 0:
                    continue

                users_processed_in_batch.add(user_id)

                for k in k_list:
                    top_k_items_info = predictions[:k]
                    top_k_item_ids = [item_id for _, _, item_id in top_k_items_info]
                    hits_at_k = len(set(top_k_item_ids) & relevant_items)

                    precision = hits_at_k / k if k > 0 else 0.0
                    all_user_metrics[k]['precision'].append(precision)

                    recall = hits_at_k / num_relevant
                    all_user_metrics[k]['recall'].append(recall)

                    # --- CORREÇÃO CÁLCULO NDCG v2 ---
                    dcg = 0.0
                    for i in range(len(top_k_item_ids)): # Iterar sobre os top K itens
                        item_id = top_k_item_ids[i]
                        if item_id in relevant_items:
                            gain = 1.0 # Ganho binário
                            dcg += gain / math.log2(i + 2) # log2(rank + 1)

                    # Calcular IDCG (Ideal DCG)
                    idcg = 0.0
                    # O número de itens relevantes que *podem* contribuir para IDCG@k é min(k, num_relevant)
                    num_ideal_items = min(k, num_relevant)
                    for i in range(num_ideal_items):
                        # No ranking ideal, os primeiros 'num_ideal_items' têm ganho 1
                        idcg += 1.0 / math.log2(i + 2) # log2(rank + 1)

                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    # --- FIM CORREÇÃO CÁLCULO NDCG v2 ---
                    all_user_metrics[k]['ndcg'].append(ndcg)

            processed_users += len(users_processed_in_batch)

        except KeyError as e:
            logging.warning(f"    Warning: Skipping ranking batch {num_batches} due to KeyError: {e}.")
            continue
        except IndexError as e:
             logging.error(f"    IndexError during ranking batch {num_batches}: {e}.", exc_info=True)
             continue
        except Exception as e:
             logging.error(f"    Error during ranking batch {num_batches}: {e}", exc_info=True)
             continue

    logging.info(f"  Finished processing {num_batches} batches for Ranking Metrics.")
    if skipped_batches_no_labels > 0:
        logging.warning(f"    Skipped {skipped_batches_no_labels}/{num_batches} ranking batches due to missing/empty labels.")

    if processed_users == 0:
         logging.warning("  Warning: No users with relevant items found across batches to calculate ranking metrics.")
         return {k: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0} for k in k_list}

    final_metrics = {}
    logging.info(f"  Calculating average ranking metrics across {processed_users} unique users with relevant items in batches...")
    for k in k_list:
        precisions = all_user_metrics[k]['precision']
        recalls = all_user_metrics[k]['recall']
        ndcgs = all_user_metrics[k]['ndcg']
        final_metrics[k] = {
            'precision': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'ndcg': np.mean(ndcgs) if ndcgs else 0.0,
        }
        logging.info(f"  Avg Metrics @{k}: Precision={final_metrics[k]['precision']:.4f}, Recall={final_metrics[k]['recall']:.4f}, NDCG={final_metrics[k]['ndcg']:.4f}")

    return final_metrics


# (calculate_hit_rate_at_k permanece como placeholder)
def calculate_hit_rate_at_k(model, node_feature_processor, loader, device, edge_type_tuple, k=10):
    model.eval()
    node_feature_processor.eval()
    hits = 0
    total_samples = 0 # Total number of positive samples evaluated

    logging.info(f"  Evaluating HitRate@{k} for edge type: {edge_type_tuple}...")
    logging.warning(f"  WARNING: This evaluation assumes the loader provides positive edge scores alongside corresponding negative scores for ranking.")
    logging.warning(f"  WARNING: HitRate@K logic is currently a placeholder and not fully implemented.")


    num_batches = 0
    skipped_batches_no_labels = 0

    for batch in loader:
        batch = batch.to(device)
        num_batches += 1

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
            logging.warning(f"    Warning: Skipping batch {num_batches} during HitRate calculation - Missing labels for {edge_type_tuple}.")
            continue

        if supervision_edge_label_index.shape[1] == 0:
            skipped_batches_no_labels += 1
            logging.warning(f"    Warning: Skipping batch {num_batches} during HitRate calculation - Labels for {edge_type_tuple} are empty.")
            continue

        edge_label_index = supervision_edge_label_index
        edge_label = supervision_edge_label

        try:
            initial_x_dict = node_feature_processor(batch)
            h_dict = model(initial_x_dict, batch.edge_index_dict)
            pred_scores = model.predict_link_score(h_dict, edge_label_index, edge_type_tuple)

            positive_mask = edge_label == 1

            if positive_mask.sum() > 0 :
                 logging.debug(f"    Batch {num_batches}: Found {positive_mask.sum().item()} positive samples for HitRate (placeholder).")
                 pass

        except KeyError as e:
            logging.warning(f"    Warning: Skipping batch {num_batches} due to KeyError during prediction/embedding access: {e}.")
            continue
        except IndexError as e:
             logging.error(f"    IndexError during HitRate evaluation batch {num_batches}: {e}.", exc_info=True)
             continue
        except Exception as e:
             logging.error(f"    Error during HitRate evaluation batch {num_batches}: {e}", exc_info=True)
             continue

    logging.info(f"  Finished processing {num_batches} batches for HitRate.")
    if skipped_batches_no_labels > 0:
        logging.warning(f"    Skipped {skipped_batches_no_labels}/{num_batches} HitRate batches due to missing/empty labels for {edge_type_tuple}.")


    if total_samples == 0:
        logging.warning("  Warning: No positive samples found or processed for HitRate evaluation (HitRate logic not fully implemented). Returning 0.0.")
        return 0.0

    hit_rate = hits / total_samples
    logging.info(f"  HitRate@{k} calculated: {hit_rate:.4f} (Based on {hits} hits out of {total_samples} positive samples - REQUIRES CORRECT IMPLEMENTATION)")
    return hit_rate