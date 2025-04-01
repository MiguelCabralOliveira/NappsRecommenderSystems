import pandas as pd
import numpy as np
import math
import argparse
import os
import json
from collections import defaultdict

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions (Standardize IDs, Load Data, Build Maps) ---

def standardize_gid(product_id):
    """Ensure product ID is in the format 'gid://shopify/Product/NUMBER'."""
    if pd.isna(product_id): return None
    id_str = str(product_id)
    if id_str.startswith('gid://shopify/Product/'): return id_str
    elif id_str.isdigit(): return f"gid://shopify/Product/{id_str}"
    else:
        parts = id_str.split('/')
        if parts[-1].isdigit(): return f"gid://shopify/Product/{parts[-1]}"
        return None

def load_recommendations(filepath, required_cols):
    """Loads recommendations, ensuring required columns exist."""
    print(f"  Loading data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"  Error: File not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File {filepath} missing required columns: {required_cols}")

        id_cols_to_standardize = [col for col in ['source_product_id', 'recommended_product_id'] if col in df.columns]
        for col in id_cols_to_standardize:
             df[col] = df[col].apply(standardize_gid)

        id_cols_to_check = [col for col in ['source_product_id', 'recommended_product_id'] if col in df.columns]
        df = df.dropna(subset=id_cols_to_check)

        print(f"  Loaded {len(df)} rows from {filepath}.")
        if 'rank' in df.columns and 'source_product_id' in df.columns:
            return df.sort_values(by=['source_product_id', 'rank'])
        return df
    except Exception as e:
        print(f"  Error loading file {filepath}: {e}")
        return None

def build_popularity_map(popularity_filepath):
    """Builds a map of {product_id: popularity_score}."""
    if not popularity_filepath or not os.path.exists(popularity_filepath):
        print("  Popularity file not provided or not found. Popularity metrics disabled.")
        return None
    print(f"  Building popularity map from: {popularity_filepath}")
    pop_map = {}
    try:
        req_cols = ['product_id']
        potential_pop_cols = ['checkout_count', 'order_count', 'frequency', 'popularity', 'count']
        pop_df = pd.read_csv(popularity_filepath)
        if not all(col in pop_df.columns for col in req_cols):
            raise ValueError(f"Popularity file missing required columns: {req_cols}")

        pop_col = None
        for col in potential_pop_cols:
            if col in pop_df.columns: pop_col = col; break
        if not pop_col: raise ValueError(f"Popularity file missing a recognized popularity column ({potential_pop_cols})")

        pop_df['product_id'] = pop_df['product_id'].apply(standardize_gid)
        pop_df = pop_df.dropna(subset=['product_id', pop_col])
        pop_map = pop_df.set_index('product_id')[pop_col].to_dict()
        print(f"  Built popularity map for {len(pop_map)} products.")
        return pop_map
    except Exception as e:
        print(f"  Error building popularity map: {e}")
        return None

def build_knn_similarity_map(knn_recs_filepath):
    """Builds a map {source_id: {rec_id: similarity}} from KNN results."""
    if not knn_recs_filepath or not os.path.exists(knn_recs_filepath):
        print("  KNN similarity file not provided or not found. ILS metric disabled.")
        return None
    print(f"  Building KNN similarity map from: {knn_recs_filepath}")
    sim_map = defaultdict(dict)
    try:
        knn_cols_req = ['source_product_id', 'recommended_product_id']
        potential_sim_cols = ['similarity_score', 'similarity', 'score']
        knn_df = pd.read_csv(knn_recs_filepath)

        if not all(col in knn_df.columns for col in knn_cols_req):
             raise ValueError(f"KNN recommendations file missing required columns: {knn_cols_req}")

        sim_col = None
        for col in potential_sim_cols:
             if col in knn_df.columns: sim_col = col; break
        if not sim_col: raise ValueError(f"KNN recommendations file missing similarity column ({potential_sim_cols})")

        knn_df['source_product_id'] = knn_df['source_product_id'].apply(standardize_gid)
        knn_df['recommended_product_id'] = knn_df['recommended_product_id'].apply(standardize_gid)
        knn_df = knn_df.dropna(subset=['source_product_id', 'recommended_product_id', sim_col])

        for _, row in knn_df.iterrows():
            source_id = row['source_product_id']
            rec_id = row['recommended_product_id']
            similarity = row[sim_col]
            sim_map[source_id][rec_id] = similarity

        print(f"  Built KNN similarity map with similarities for {len(sim_map)} source products.")
        return sim_map
    except Exception as e:
        print(f"  Error building KNN similarity map: {e}")
        return None

def get_catalog_ids(catalog_filepath):
    """Gets a set of all unique product IDs from the catalog file."""
    if not catalog_filepath or not os.path.exists(catalog_filepath):
        print("  Error: Catalog file not provided or not found.")
        return None
    print(f"  Loading catalog product IDs from: {catalog_filepath}")
    try:
        cat_df = pd.read_csv(catalog_filepath, usecols=['product_id'])
        cat_df['product_id'] = cat_df['product_id'].apply(standardize_gid)
        catalog_ids = set(cat_df['product_id'].dropna())
        print(f"  Loaded {len(catalog_ids)} unique product IDs from catalog.")
        return catalog_ids
    except KeyError:
         print(f"  Error: 'product_id' column not found in catalog file: {catalog_filepath}")
         return None
    except Exception as e:
        print(f"  Error loading catalog IDs: {e}")
        return None

# --- Metric Calculation Functions ---

def calculate_ils_at_k(recommendations, k, similarity_map):
    """Calculates Intra-List Similarity@k."""
    top_k_recs = recommendations[:k]
    if len(top_k_recs) < 2: return 0.0
    total_similarity = 0.0; pair_count = 0
    for i in range(len(top_k_recs)):
        for j in range(i + 1, len(top_k_recs)):
            item1, item2 = top_k_recs[i], top_k_recs[j]
            sim = 0.0
            if item1 in similarity_map and item2 in similarity_map.get(item1, {}):
                 sim = similarity_map[item1].get(item2, 0.0)
            elif item2 in similarity_map and item1 in similarity_map.get(item2, {}):
                 sim = similarity_map[item2].get(item1, 0.0)
            total_similarity += sim; pair_count += 1
    return total_similarity / pair_count if pair_count > 0 else 0.0

def calculate_pop_bias_at_k(recommendations, k, popularity_map):
    """Calculates average popularity of items in the recommendation list."""
    top_k_recs = recommendations[:k]
    if not top_k_recs: return 0.0
    total_popularity = 0.0; count = 0
    for rec_id in top_k_recs:
        pop = popularity_map.get(rec_id)
        if pop is not None and pd.notna(pop):
             total_popularity += pop; count += 1
    # Return mean popularity, or 0 if no items had popularity scores
    return total_popularity / count if count > 0 else 0.0

# --- Plotting Functions ---

def plot_distribution(data, title, xlabel, filename, output_dir):
    """Generates and saves a histogram/KDE plot for a distribution."""
    if not data: # Check if data list is empty
        print(f"  Skipping plot '{title}': No data available.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30)
    mean_val = np.nanmean(data)
    median_val = np.nanmedian(data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.3f}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=120)
        print(f"  Plot saved to: {filepath}")
    except Exception as e:
        print(f"  Error saving plot {filepath}: {e}")
    plt.close()

# --- Core Evaluation Logic ---

def evaluate_model(recs_df, catalog_ids, k, pop_map=None, knn_sim_map=None):
    """Runs the evaluation loop and calculates metrics."""
    all_ils = []
    all_pop_bias = []
    unique_recommended_items = set()

    grouped_recs = recs_df.groupby('source_product_id')
    total_sources = len(grouped_recs)
    if total_sources == 0:
        print("  Warning: No source products found in recommendations for evaluation.")
        return {}, [], [] # Return empty dict and lists

    print(f"  Evaluating recommendations for {total_sources} source products at k={k}...")
    processed_count = 0
    # Simplified progress indication
    for source_id, user_recs_df in grouped_recs:
        processed_count += 1
        if processed_count % 100 == 0 or processed_count == total_sources:
             print(f"   Processed {processed_count}/{total_sources} source products...")

        recommendations = user_recs_df['recommended_product_id'].tolist()
        unique_recommended_items.update(recommendations[:k])

        if knn_sim_map:
            ils = calculate_ils_at_k(recommendations, k, knn_sim_map)
            all_ils.append(ils)
        if pop_map:
            pop_bias = calculate_pop_bias_at_k(recommendations, k, pop_map)
            all_pop_bias.append(pop_bias)

    # Calculate Overall Metrics
    catalog_coverage = 0.0
    if catalog_ids:
        catalog_size = len(catalog_ids)
        if catalog_size > 0:
            catalog_coverage = len(unique_recommended_items) / catalog_size
    else: print("  Warning: Catalog IDs not available, cannot calculate Coverage.")

    # Use nanmean which ignores NaNs. If all were NaN, result is NaN.
    mean_ils = np.nanmean(all_ils) if all_ils else None
    mean_pop_bias = np.nanmean(all_pop_bias) if all_pop_bias else None

    results = {
        f"CatalogCoverage@{k}": catalog_coverage,
        f"IntraListSimilarity@{k}": mean_ils if pd.notna(mean_ils) else None,
        f"PopularityBias@{k}": mean_pop_bias if pd.notna(mean_pop_bias) else None,
        "TotalSourceProductsEvaluated": total_sources,
        "K": k
    }
    # Return raw lists for plotting
    return results, all_ils, all_pop_bias

# --- Main Callable Function ---

def run_evaluation(recs_path, catalog_path, k, output_dir, pop_path=None, knn_sims_path=None, results_filename="evaluation_results.json"):
    """
    Loads data, runs evaluation, saves results (JSON and plots).

    Args:
        recs_path (str): Path to the recommendation file.
        catalog_path (str): Path to the product catalog file.
        k (int): Value of K for top-K metrics.
        output_dir (str): Directory to save results and plots.
        pop_path (str, optional): Path to the popular products file.
        knn_sims_path (str, optional): Path to the raw KNN similarities file.
        results_filename (str): Filename for the JSON results.

    Returns:
        dict: Dictionary containing the calculated evaluation metrics, or None on failure.
    """
    print("\n--- Starting Model Evaluation ---")
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    # Load Data
    recs_df = load_recommendations(recs_path, required_cols=['source_product_id', 'recommended_product_id', 'rank'])
    catalog_ids = get_catalog_ids(catalog_path)
    pop_map = build_popularity_map(pop_path)
    knn_sim_map = build_knn_similarity_map(knn_sims_path)

    if recs_df is None or catalog_ids is None:
        print("  Error: Could not load essential input files (--recs, --catalog). Evaluation aborted.")
        return None

    # Perform Evaluation
    evaluation_results, all_ils_scores, all_pop_bias_scores = evaluate_model(
        recs_df=recs_df,
        catalog_ids=catalog_ids,
        k=k,
        pop_map=pop_map,
        knn_sim_map=knn_sim_map
    )

    if not evaluation_results: # Check if evaluation returned empty results
         print("  Evaluation did not produce results (maybe no recommendations?). Skipping saving.")
         return None

    # --- Print Results ---
    print("\n--- Evaluation Metrics ---")
    for metric, value in evaluation_results.items():
        if isinstance(value, float): print(f"  {metric:<25}: {value:.4f}")
        elif value is not None: print(f"  {metric:<25}: {value}")
        else: print(f"  {metric:<25}: Not Calculated")

    # --- Generate Plots ---
    print("\n--- Generating Plots ---")
    if all_ils_scores:
        plot_distribution(
            all_ils_scores,
            f'Distribution of Intra-List Similarity (ILS) @{k}',
            f'ILS @{k}',
            f'ils_distribution_k{k}.png',
            output_dir
        )
    if all_pop_bias_scores:
         # Filter out potential NaNs before plotting popularity
        valid_pop_scores = [p for p in all_pop_bias_scores if pd.notna(p)]
        plot_distribution(
            valid_pop_scores,
            f'Distribution of Average Recommendation Popularity @{k}',
            f'Avg Popularity @{k}',
            f'popularity_bias_distribution_k{k}.png',
            output_dir
        )

    # --- Save Results JSON ---
    json_output_path = os.path.join(output_dir, results_filename)
    try:
        serializable_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, (np.number)): # Handles numpy int/float types
                serializable_results[key] = float(value) if not pd.isna(value) else None
            elif value is not None:
                 serializable_results[key] = value

        with open(json_output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nEvaluation results saved to: {json_output_path}")
    except Exception as e:
        print(f"\nError saving evaluation results to JSON: {e}")

    print("--- Evaluation Complete ---")
    return evaluation_results

# --- Standalone Execution Block (for testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate recommendation model performance (Beyond Accuracy Metrics - Standalone).')
    parser.add_argument('--recs', type=str, required=True, help='Path to the recommendation file.')
    parser.add_argument('--catalog', type=str, required=True, help='Path to the product catalog file.')
    parser.add_argument('--pop', type=str, default=None, help='(Optional) Path to popular products file.')
    parser.add_argument('--knn-sims', type=str, default=None, help='(Optional) Path to raw KNN similarities file.')
    parser.add_argument('-k', type=int, default=10, help='Value of K for metrics (default: 10)')
    parser.add_argument('--output-dir', type=str, default='evaluation_output', help='Directory to save results/plots.')
    parser.add_argument('--results-file', type=str, default='evaluation_results.json', help='Filename for JSON results.')

    args = parser.parse_args()

    # Call the main function
    run_evaluation(
        recs_path=args.recs,
        catalog_path=args.catalog,
        k=args.k,
        output_dir=args.output_dir,
        pop_path=args.pop,
        knn_sims_path=args.knn_sims,
        results_filename=args.results_file
    )