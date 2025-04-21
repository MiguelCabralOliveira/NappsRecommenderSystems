# training/similarity/knn/evaluation.py

import pandas as pd
import numpy as np
import os
import traceback
from typing import Dict, Optional, List, Set
from collections import defaultdict

# Import utility functions from the same module
try:
    from ..utils import load_dataframe, save_json_report
except ImportError:
    from utils import load_dataframe, save_json_report # Fallback

def calculate_catalog_coverage(recommendations_df: pd.DataFrame, catalog_ids: Set[str], k: int) -> float:
    """
    Calculates the catalog coverage@k.
    """
    if recommendations_df.empty or not catalog_ids:
        if not catalog_ids: print("Warning (Coverage): Catalog ID set is empty.")
        return 0.0
    try:
        top_k_recs = recommendations_df[recommendations_df['rank'] <= k]
        recommended_items_in_top_k = set(top_k_recs['recommended_product_id'].astype(str).unique())
        coverage = len(recommended_items_in_top_k.intersection(catalog_ids)) / len(catalog_ids)
        return coverage * 100
    except KeyError as e: print(f"Error (Coverage): Missing expected column: {e}"); return 0.0
    except Exception as e: print(f"Error calculating Coverage: {type(e).__name__} - {e}"); return 0.0


def calculate_personalization(recommendations_df: pd.DataFrame, k: int) -> float:
    """
    Calculates personalization based on average Jaccard distance.
    """
    if recommendations_df.empty or 'source_product_id' not in recommendations_df.columns or 'recommended_product_id' not in recommendations_df.columns:
        return 0.0
    try:
        top_k_recs = recommendations_df[recommendations_df['rank'] <= k]
        top_k_recs['source_product_id'] = top_k_recs['source_product_id'].astype(str)
        top_k_recs['recommended_product_id'] = top_k_recs['recommended_product_id'].astype(str)
        recs_per_source = top_k_recs.groupby('source_product_id')['recommended_product_id'].apply(set).to_dict()
        source_ids = list(recs_per_source.keys())
        if len(source_ids) < 2: return 0.0

        total_distance = 0.0; pair_count = 0
        max_pairs_to_sample = 50000
        total_possible_pairs = len(source_ids) * (len(source_ids) - 1) // 2
        if total_possible_pairs > max_pairs_to_sample:
            print(f"Sampling {max_pairs_to_sample} pairs for personalization calculation...")
            idx_pairs = np.random.choice(len(source_ids), size=(max_pairs_to_sample, 2), replace=True)
        else:
            indices = np.triu_indices(len(source_ids), k=1); idx_pairs = np.array(indices).T

        for i, j in idx_pairs:
            if i == j : continue
            list1 = recs_per_source.get(source_ids[i], set()); list2 = recs_per_source.get(source_ids[j], set())
            intersection = len(list1.intersection(list2)); union = len(list1.union(list2))
            if union > 0: total_distance += (1.0 - (intersection / union)); pair_count += 1
        return (total_distance / pair_count) * 100 if pair_count > 0 else 0.0
    except KeyError as e: print(f"Error (Personalization): Missing expected column: {e}"); return 0.0
    except Exception as e: print(f"Error calculating Personalization: {type(e).__name__} - {e}"); return 0.0


def calculate_intra_list_similarity(recommendations_df: pd.DataFrame,
                                    similarity_matrix_df: pd.DataFrame, k: int) -> float:
    """
    Calculates the average Intra-List Similarity (ILS)@k. Requires unfiltered similarities.
    """
    if recommendations_df.empty or similarity_matrix_df is None or similarity_matrix_df.empty or k <= 1:
        if similarity_matrix_df is None or similarity_matrix_df.empty: print("Warning (ILS): Similarity matrix data not provided or empty.")
        return 0.0
    required_sim_cols = ['source_product_id', 'recommended_product_id', 'similarity_score']
    if not all(col in similarity_matrix_df.columns for col in required_sim_cols):
        print(f"Warning (ILS): Similarity matrix missing required columns: {required_sim_cols}"); return 0.0

    print("Calculating Intra-List Similarity (ILS)...")
    try:
        similarity_matrix_df['source_product_id'] = similarity_matrix_df['source_product_id'].astype(str)
        similarity_matrix_df['recommended_product_id'] = similarity_matrix_df['recommended_product_id'].astype(str)
        similarity_lookup = defaultdict(dict)
        for row in similarity_matrix_df.itertuples(index=False):
            src_id, rec_id, sim_score = row.source_product_id, row.recommended_product_id, row.similarity_score
            if pd.notna(sim_score): similarity_lookup[src_id][rec_id] = sim_score; similarity_lookup[rec_id][src_id] = sim_score
        # print(f"Built similarity lookup for {len(similarity_lookup)} items.") # Reduce verbosity

        top_k_recs = recommendations_df[recommendations_df['rank'] <= k].copy()
        if top_k_recs.empty: print("Warning (ILS): No recommendations found at k={k}."); return 0.0
        top_k_recs['source_product_id'] = top_k_recs['source_product_id'].astype(str)
        top_k_recs['recommended_product_id'] = top_k_recs['recommended_product_id'].astype(str)
        recs_per_source = top_k_recs.groupby('source_product_id')['recommended_product_id'].apply(list).to_dict()

        total_ils = 0.0; valid_lists_count = 0
        for source_id, rec_list in recs_per_source.items():
            if len(rec_list) < 2: continue
            list_similarity_sum = 0.0; pair_count_in_list = 0
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item_i = rec_list[i]; item_j = rec_list[j]
                    sim = similarity_lookup.get(item_i, {}).get(item_j, 0.0)
                    list_similarity_sum += sim; pair_count_in_list += 1
            if pair_count_in_list > 0: avg_list_similarity = list_similarity_sum / pair_count_in_list; total_ils += avg_list_similarity; valid_lists_count += 1

        average_ils = (total_ils / valid_lists_count) if valid_lists_count > 0 else 0.0
        # print(f"Average ILS calculated: {average_ils:.4f} across {valid_lists_count} lists.") # Reduce verbosity
        return average_ils * 100
    except KeyError as e: print(f"Error (ILS): Missing expected column: {e}"); return 0.0
    except Exception as e: print(f"Error calculating ILS: {type(e).__name__} - {e}"); traceback.print_exc(); return 0.0

def calculate_popularity_bias(recommendations_df: pd.DataFrame, popularity_df: Optional[pd.DataFrame], k: int) -> Optional[float]:
    """
    Calculates the average popularity score of items in the top-k recommendations.
    """
    if recommendations_df.empty or popularity_df is None or popularity_df.empty \
       or 'product_id' not in popularity_df.columns or 'popularity_score' not in popularity_df.columns:
        return None
    try:
        popularity_df['product_id'] = popularity_df['product_id'].astype(str)
        pop_map = popularity_df.set_index('product_id')['popularity_score'].to_dict()
        top_k_recs = recommendations_df[recommendations_df['rank'] <= k].copy()
        if top_k_recs.empty: return 0.0
        top_k_recs['recommended_product_id'] = top_k_recs['recommended_product_id'].astype(str)
        top_k_recs['popularity'] = top_k_recs['recommended_product_id'].map(pop_map).fillna(0)
        return top_k_recs['popularity'].mean()
    except KeyError as e: print(f"Error (Popularity): Missing expected column: {e}"); return None
    except Exception as e: print(f"Error calculating Popularity Bias: {type(e).__name__} - {e}"); return None

# *** NOVA FUNÇÃO ***
def calculate_similarity_stats(recommendations_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculates basic statistics for the similarity scores in the recommendations.
    """
    stats = {
        'avg_similarity': None,
        'std_dev_similarity': None,
        'p25_similarity': None,
        'p50_similarity (median)': None,
        'p75_similarity': None,
        'min_similarity': None,
        'max_similarity': None
    }
    if recommendations_df.empty or 'similarity_score' not in recommendations_df.columns:
        print("Warning (Sim Stats): Recommendations empty or missing 'similarity_score'.")
        return stats

    try:
        similarities = recommendations_df['similarity_score'].dropna()
        if similarities.empty:
             print("Warning (Sim Stats): No valid similarity scores found.")
             return stats

        stats['avg_similarity'] = similarities.mean()
        stats['std_dev_similarity'] = similarities.std()
        stats['p25_similarity'] = similarities.quantile(0.25)
        stats['p50_similarity (median)'] = similarities.median()
        stats['p75_similarity'] = similarities.quantile(0.75)
        stats['min_similarity'] = similarities.min()
        stats['max_similarity'] = similarities.max()

    except Exception as e:
        print(f"Error calculating Similarity Stats: {type(e).__name__} - {e}")
        # Reset stats to None if error occurs during calculation
        for key in stats: stats[key] = None

    return stats

# *** Função run_evaluation ATUALIZADA ***
def run_evaluation(recs_path: str, k: int, output_dir: str,
                   catalog_path: Optional[str] = None,
                   pop_path: Optional[str] = None,
                   knn_sims_path: Optional[str] = None,
                   results_filename: Optional[str] = None) -> Optional[Dict]:
    """
    Runs a suite of evaluation metrics on a set of recommendations.
    """
    print(f"\n--- Starting Evaluation for k={k} ---")
    print(f"Recommendations file: {recs_path}")
    print(f"Catalog file: {catalog_path if catalog_path else 'Not provided'}")
    print(f"Popularity file: {pop_path if pop_path else 'Not provided'}")
    print(f"KNN Similarities file (for ILS): {knn_sims_path if knn_sims_path else 'Not provided'}")

    # Load Data
    recs_df = load_dataframe(recs_path, required_cols=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']) # Add similarity_score requirement
    catalog_df = load_dataframe(catalog_path, required_cols=['product_id']) if catalog_path else None
    popularity_df = load_dataframe(pop_path) if pop_path else None
    # Load unfiltered sims ONLY if knn_sims_path is provided (needed for ILS)
    knn_sims_df = load_dataframe(knn_sims_path, required_cols=['source_product_id', 'recommended_product_id', 'similarity_score']) if knn_sims_path else None


    if recs_df.empty:
        print("Error: Could not load recommendations data (or missing required columns). Aborting evaluation.")
        return None

    # Prepare IDs
    catalog_ids: Set[str] = set()
    try:
        recs_df['source_product_id'] = recs_df['source_product_id'].astype(str)
        recs_df['recommended_product_id'] = recs_df['recommended_product_id'].astype(str)
        if catalog_df is not None and not catalog_df.empty:
            catalog_ids = set(catalog_df['product_id'].astype(str).unique())
            if not catalog_ids: print("Warning: Catalog loaded but contains no unique product IDs.")
        else:
             print("Info: Catalog data not available, Coverage metric will be skipped.")
    except Exception as e:
        print(f"Error preparing IDs for evaluation: {type(e).__name__} - {e}"); traceback.print_exc(); return None

    # Calculate Metrics
    metrics = {'k': k, 'recommendations_file': os.path.basename(recs_path)}
    print(f"\nCalculating metrics @{k}...")

    # Coverage
    metrics['catalog_coverage_pct'] = calculate_catalog_coverage(recs_df, catalog_ids, k) if catalog_ids else None

    # Personalization
    metrics['personalization_distance_pct'] = calculate_personalization(recs_df, k)

    # ILS (requires unfiltered sims)
    metrics['intra_list_similarity_pct'] = calculate_intra_list_similarity(recs_df, knn_sims_df, k) if knn_sims_df is not None else None
    if metrics['intra_list_similarity_pct'] is None and knn_sims_path: print("Warning (ILS): Failed to calculate, possibly due to missing similarities file or content.")

    # Popularity Bias
    metrics['avg_recommendation_popularity'] = calculate_popularity_bias(recs_df, popularity_df, k)
    if metrics['avg_recommendation_popularity'] is None: print("Skipping Popularity Bias: Popularity file not provided or invalid.")

    # *** Adicionar Métricas de Similaridade ***
    sim_stats = calculate_similarity_stats(recs_df) # Calcula as estatísticas das similaridades *neste ficheiro*
    metrics.update(sim_stats) # Adiciona as estatísticas ao dicionário principal

    # --- Impressão e Gravação ---
    print("\nEvaluation Metrics Calculated:")
    # Sort keys for consistent printing (optional)
    sorted_metrics_keys = ['k', 'recommendations_file'] + sorted([k for k in metrics if k not in ['k', 'recommendations_file']])
    for metric in sorted_metrics_keys:
        value = metrics.get(metric)
        if metric in ['k', 'recommendations_file']: print(f"- {metric}: {value}")
        elif value is not None: print(f"- {metric}: {value:.4f}")
        else: print(f"- {metric}: Skipped or Failed")

    if results_filename is None:
        base_rec_name = os.path.basename(recs_path).replace('.csv', '')
        results_filename = f"evaluation_k{k}_{base_rec_name}.json"
    output_path = os.path.join(output_dir, results_filename)

    if save_json_report(metrics, output_path):
        print(f"Evaluation results saved to {output_path}")
    else:
        print("Failed to save evaluation results.")

    print("--- Evaluation Finished ---")
    return metrics