# training/similarity/evaluation.py

import pandas as pd
import numpy as np
import os
import traceback
from typing import Dict, Optional, List
from collections import defaultdict

# Import utility functions from the same module
from ..utils import load_dataframe, save_json_report

def calculate_catalog_coverage(recommendations_df: pd.DataFrame, catalog_ids: set, k: int) -> float:
    """
    Calculates the catalog coverage@k.
    Percentage of items in the catalog that appear in the top-k recommendations for at least one item.
    """
    if recommendations_df.empty or not catalog_ids:
        return 0.0

    # Get recommendations within top-k
    top_k_recs = recommendations_df[recommendations_df['rank'] <= k]

    # Find unique recommended items within the top-k lists
    recommended_items_in_top_k = set(top_k_recs['recommended_product_id'].unique())

    # Calculate coverage: intersection of recommended items and catalog items / total catalog items
    coverage = len(recommended_items_in_top_k.intersection(catalog_ids)) / len(catalog_ids)
    return coverage * 100 # Return as percentage

def calculate_personalization(recommendations_df: pd.DataFrame, k: int) -> float:
    """
    Calculates personalization based on the average Jaccard distance between
    top-k recommendation lists of different source items.
    Higher value means lists are more dissimilar (more personalized).
    Lower value means lists are more similar (less personalized).
    """
    if recommendations_df.empty or 'source_product_id' not in recommendations_df.columns:
        return 0.0

    # Get top-k recommendations for each source item
    top_k_recs = recommendations_df[recommendations_df['rank'] <= k]
    recs_per_source = top_k_recs.groupby('source_product_id')['recommended_product_id'].apply(set).to_dict()

    source_ids = list(recs_per_source.keys())
    if len(source_ids) < 2:
        return 0.0 # Need at least two lists to compare

    total_distance = 0.0
    pair_count = 0

    # Compare lists for pairs of source items (can be computationally intensive for large N)
    # Optimization: Sample pairs if the number of source items is very large
    max_pairs_to_sample = 50000 # Limit comparisons for performance
    indices = np.arange(len(source_ids))

    if len(source_ids) * (len(source_ids) - 1) // 2 > max_pairs_to_sample:
        print(f"Sampling {max_pairs_to_sample} pairs for personalization calculation (out of {len(source_ids)} items)...")
        sampled_pairs = np.random.choice(indices, (max_pairs_to_sample, 2), replace=True) # Sample with replacement
    else:
        sampled_pairs = np.array(np.triu_indices(len(source_ids), k=1)).T # Get all unique upper-triangle pairs

    for i, j in sampled_pairs:
        if i == j: continue # Ensure no self-comparison if sampling with replacement

        list1 = recs_per_source.get(source_ids[i], set())
        list2 = recs_per_source.get(source_ids[j], set())

        intersection = len(list1.intersection(list2))
        union = len(list1.union(list2))

        if union > 0:
            jaccard_similarity = intersection / union
            jaccard_distance = 1.0 - jaccard_similarity
            total_distance += jaccard_distance
            pair_count += 1

    return (total_distance / pair_count) * 100 if pair_count > 0 else 0.0 # Return avg distance as percentage


def calculate_intra_list_similarity(recommendations_df: pd.DataFrame,
                                    similarity_matrix_df: pd.DataFrame, k: int) -> float:
    """
    Calculates the average Intra-List Similarity (ILS)@k.
    Measures the average similarity between all pairs of items within each top-k list.
    Lower ILS is often desirable (more diversity within a list).

    Args:
        recommendations_df: DataFrame with columns 'source_product_id', 'recommended_product_id', 'rank'.
        similarity_matrix_df: DataFrame containing pre-calculated item-item similarities.
                              Expected columns: 'source_product_id', 'recommended_product_id', 'similarity_score'.
                              This should represent the underlying similarity used by the KNN model BEFORE filtering.
        k: The cutoff for the recommendation lists.

    Returns:
        Average ILS score (0 to 100), or 0.0 if calculation fails.
    """
    if recommendations_df.empty or similarity_matrix_df.empty or k <= 1:
        return 0.0

    print("Calculating Intra-List Similarity (ILS)...")
    # Create a lookup dictionary for similarities for faster access
    # Ensure IDs are strings for consistent matching
    similarity_matrix_df['source_product_id'] = similarity_matrix_df['source_product_id'].astype(str)
    similarity_matrix_df['recommended_product_id'] = similarity_matrix_df['recommended_product_id'].astype(str)
    similarity_lookup = defaultdict(dict)
    for _, row in similarity_matrix_df.iterrows():
        # Store similarity in both directions for easy lookup
        similarity_lookup[row['source_product_id']][row['recommended_product_id']] = row['similarity_score']
        similarity_lookup[row['recommended_product_id']][row['source_product_id']] = row['similarity_score']
    print(f"Built similarity lookup for {len(similarity_lookup)} items.")


    # Get top-k recommendations for each source item
    top_k_recs = recommendations_df[recommendations_df['rank'] <= k].copy()
    # Ensure IDs are strings here too
    top_k_recs['source_product_id'] = top_k_recs['source_product_id'].astype(str)
    top_k_recs['recommended_product_id'] = top_k_recs['recommended_product_id'].astype(str)
    recs_per_source = top_k_recs.groupby('source_product_id')['recommended_product_id'].apply(list).to_dict()

    total_ils = 0.0
    valid_lists_count = 0

    # Iterate through each source item's recommendation list
    for source_id, rec_list in recs_per_source.items():
        if len(rec_list) < 2:
            continue # Need at least two items to compare

        list_similarity_sum = 0.0
        pair_count_in_list = 0

        # Iterate through all unique pairs within the list
        for i in range(len(rec_list)):
            for j in range(i + 1, len(rec_list)):
                item_i = rec_list[i]
                item_j = rec_list[j]

                # Look up similarity - handle cases where similarity might be missing
                sim = similarity_lookup.get(item_i, {}).get(item_j, 0.0) # Default to 0 if pair not found

                # Add similarity to the sum for this list
                list_similarity_sum += sim
                pair_count_in_list += 1

        # Calculate average similarity for this list
        if pair_count_in_list > 0:
            avg_list_similarity = list_similarity_sum / pair_count_in_list
            total_ils += avg_list_similarity
            valid_lists_count += 1
        # else: # Should not happen if len(rec_list) >= 2
            # print(f"Warning: No pairs found for source {source_id} with list {rec_list}")


    # Calculate overall average ILS
    average_ils = (total_ils / valid_lists_count) if valid_lists_count > 0 else 0.0
    print(f"Average ILS calculated: {average_ils:.4f} across {valid_lists_count} lists.")
    return average_ils * 100 # Return as percentage

def calculate_popularity_bias(recommendations_df: pd.DataFrame, popularity_df: pd.DataFrame, k: int) -> float:
    """
    Calculates the average popularity score of items in the top-k recommendations.
    Requires a popularity DataFrame mapping 'product_id' to a 'popularity_score' (e.g., order count).
    Higher value indicates recommendations tend towards more popular items.
    """
    if recommendations_df.empty or popularity_df is None or popularity_df.empty or 'product_id' not in popularity_df.columns or 'popularity_score' not in popularity_df.columns:
        print("Warning: Cannot calculate popularity bias. Missing recommendations or valid popularity data.")
        return 0.0

    # Create popularity lookup
    # Ensure IDs are strings for consistent matching
    popularity_df['product_id'] = popularity_df['product_id'].astype(str)
    pop_map = popularity_df.set_index('product_id')['popularity_score'].to_dict()

    # Get top-k recommendations
    top_k_recs = recommendations_df[recommendations_df['rank'] <= k].copy()
    top_k_recs['recommended_product_id'] = top_k_recs['recommended_product_id'].astype(str)


    # Map popularity scores
    top_k_recs['popularity'] = top_k_recs['recommended_product_id'].map(pop_map).fillna(0) # Assign 0 if product not in pop map

    # Calculate average popularity across all top-k recommendations
    if top_k_recs.empty:
        return 0.0

    avg_popularity = top_k_recs['popularity'].mean()
    return avg_popularity


def run_evaluation(recs_path: str, catalog_path: str, k: int, output_dir: str,
                   pop_path: Optional[str] = None, knn_sims_path: Optional[str] = None,
                   results_filename: Optional[str] = None) -> Optional[Dict]:
    """
    Runs a suite of evaluation metrics on a set of recommendations.

    Args:
        recs_path: Path to the recommendations CSV file. Expected columns:
                   'source_product_id', 'recommended_product_id', 'rank'.
        catalog_path: Path to the product catalog CSV file. Expected column: 'product_id'.
        k: The number of top recommendations to consider (e.g., @k).
        output_dir: Directory to save the evaluation results JSON.
        pop_path: Optional path to the popularity CSV file. Expected columns:
                  'product_id', 'popularity_score'.
        knn_sims_path: Optional path to the *unfiltered* KNN similarity file (required for ILS).
                       Expected columns: 'source_product_id', 'recommended_product_id', 'similarity_score'.
        results_filename: Optional name for the output JSON file. If None, a default name is generated.

    Returns:
        Dictionary containing the calculated metrics, or None if evaluation fails.
    """
    print(f"\n--- Starting Evaluation for k={k} ---")
    print(f"Recommendations file: {recs_path}")
    print(f"Catalog file: {catalog_path}")
    print(f"Popularity file: {pop_path}")
    print(f"KNN Similarities file (for ILS): {knn_sims_path}")

    # --- Load Data ---
    recs_df = load_dataframe(recs_path, required_cols=['source_product_id', 'recommended_product_id', 'rank'])
    catalog_df = load_dataframe(catalog_path, required_cols=['product_id'])
    popularity_df = load_dataframe(pop_path) if pop_path else None
    knn_sims_df = load_dataframe(knn_sims_path) if knn_sims_path else None # Load sims for ILS

    if recs_df.empty or catalog_df.empty:
        print("Error: Could not load recommendations or catalog data. Aborting evaluation.")
        return None

    # Ensure consistent string IDs for sets/lookups
    try:
        recs_df['source_product_id'] = recs_df['source_product_id'].astype(str)
        recs_df['recommended_product_id'] = recs_df['recommended_product_id'].astype(str)
        catalog_ids = set(catalog_df['product_id'].astype(str).unique())
        if not catalog_ids: raise ValueError("Catalog IDs are empty.")
    except Exception as e:
        print(f"Error preparing IDs for evaluation: {e}")
        return None

    # --- Calculate Metrics ---
    metrics = {'k': k}
    print(f"\nCalculating metrics @{k}...")

    try: metrics['catalog_coverage_pct'] = calculate_catalog_coverage(recs_df, catalog_ids, k)
    except Exception as e: print(f"Error calculating Coverage: {e}"); metrics['catalog_coverage_pct'] = None

    try: metrics['personalization_distance_pct'] = calculate_personalization(recs_df, k)
    except Exception as e: print(f"Error calculating Personalization: {e}"); metrics['personalization_distance_pct'] = None

    # ILS requires the unfiltered similarity scores
    if knn_sims_df is not None and not knn_sims_df.empty:
        try: metrics['intra_list_similarity_pct'] = calculate_intra_list_similarity(recs_df, knn_sims_df, k)
        except Exception as e: print(f"Error calculating ILS: {e}"); metrics['intra_list_similarity_pct'] = None
    else:
        print("Skipping Intra-List Similarity (ILS): Unfiltered KNN similarities file not provided or empty.")
        metrics['intra_list_similarity_pct'] = None

    if popularity_df is not None and not popularity_df.empty:
        try: metrics['avg_recommendation_popularity'] = calculate_popularity_bias(recs_df, popularity_df, k)
        except Exception as e: print(f"Error calculating Popularity Bias: {e}"); metrics['avg_recommendation_popularity'] = None
    else:
        print("Skipping Popularity Bias: Popularity file not provided or empty.")
        metrics['avg_recommendation_popularity'] = None

    print("\nEvaluation Metrics Calculated:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"- {metric}: {value:.4f}")
        else:
            print(f"- {metric}: Calculation Failed")

    # --- Save Results ---
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