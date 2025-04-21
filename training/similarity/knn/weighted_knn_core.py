# training/similarity/weighted_knn_core.py

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import traceback
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import os
import json
import joblib
import warnings
import time
from itertools import product
from typing import Optional, Dict, List, Tuple, Any

# Import utility functions from the same module
from ..utils import save_model_component, load_model_component, save_json_report , save_dataframe

class _WeightedKNN:
    """
    Private implementation of the weighted KNN algorithm for product recommendations.
    Feature weights can be configured externally. Includes optional variant filtering.
    This class should typically be instantiated and managed by the knn_pipeline functions.
    """
    def __init__(self, n_neighbors: int = 12, algorithm: str = 'auto', metric: str = 'euclidean', random_state: int = 52):
        """
        Initialize the WeightedKNN model.

        Args:
            n_neighbors: Base number of neighbors to find (k+1 will be used internally).
            algorithm: Algorithm for NearestNeighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
            metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.).
            random_state: Random seed for reproducibility.
        """
        if n_neighbors <= 0:
             raise ValueError("n_neighbors must be positive.")

        self.n_neighbors = n_neighbors # The number of recommendations requested
        self._knn_k = n_neighbors + 1 # The k used in NearestNeighbors (includes self)
        self.algorithm = algorithm
        self.metric = metric
        self.random_state = random_state

        # Model components - initialized during fitting/loading
        self.knn_model: Optional[NearestNeighbors] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_weights: Optional[Dict[str, float]] = None # Effective weights used
        self.feature_groups: Optional[Dict[str, List[str]]] = None # Identified feature groups
        self.feature_names_in_order: Optional[List[str]] = None # Crucial for applying weights/scaler

        # Data/Results - populated during fitting
        self.product_ids: Optional[pd.Series] = None # Product IDs corresponding to rows
        # self.product_titles: Optional[pd.Series] = None # Titles are not strictly needed for core model logic
        self.X_weighted: Optional[np.ndarray] = None # Weighted & scaled feature matrix used for fitting KNN
        self.recommendation_matrix: Optional[Dict[Any, List[Tuple[Any, float]]]] = None # {source_id: [(rec_id, sim), ...]}

        # Default weights - used if no external weights are provided during set_feature_weights
        # These should reflect reasonable starting points or known good values.
        self._default_base_weights = {
            'description_tfidf': 1.0,
            'title_tfidf': 1.2,
            'handle_tfidf': 0.8,
            'metafield_data': 1.0, # Might need splitting if some metafields are numeric vs text TFIDF
            'collections': 1.4,
            'tags': 1.8,
            'product_type': 1.8,
            'vendor': 1.0,
            'variant_size': 1.0,
            'variant_color': 1.0,
            'variant_other': 1.0,
            'price': 0.8,
            'time_features': 0.5,
            'release_quarter': 0.5,
            'availability': 0.6,
            'gift_card': 0.1,
            'other_numeric': 0.5
        }

    def _identify_feature_groups(self, df_columns: List[str]) -> Dict[str, List[str]]:
        """
        Automatically identify feature groups based on column names.

        Args:
            df_columns: List of column names from the numeric feature DataFrame.

        Returns:
            Dictionary where keys are group names and values are lists of feature names.
        """
        # Initialize feature groups
        feature_groups = {
            'description_tfidf': [], 'title_tfidf': [], 'handle_tfidf': [],
            'metafield_data': [], 'collections': [], 'tags': [], 'product_type': [],
            'vendor': [], 'variant_size': [], 'variant_color': [], 'variant_other': [],
            'price': [], 'time_features': [], 'release_quarter': [],
            'availability': [], 'gift_card': [], 'other_numeric': []
        }

        print("Identifying feature groups from column names...")
        for col in df_columns:
            assigned = False
            # Use startswith for prefixes - order matters (more specific first)
            if col.startswith('desc_tfidf_'): feature_groups['description_tfidf'].append(col); assigned = True
            elif col.startswith('title_tfidf_'): feature_groups['title_tfidf'].append(col); assigned = True
            elif col.startswith('handle_tfidf_'): feature_groups['handle_tfidf'].append(col); assigned = True
            elif col.startswith('meta_tfidf_') or col.startswith('meta_'): feature_groups['metafield_data'].append(col); assigned = True # Broader meta check
            elif col.startswith('coll_'): feature_groups['collections'].append(col); assigned = True
            elif col.startswith('tag_'): feature_groups['tags'].append(col); assigned = True
            elif col.startswith('prodtype_'): feature_groups['product_type'].append(col); assigned = True
            elif col.startswith('vendor_'): feature_groups['vendor'].append(col); assigned = True
            elif col.startswith('opt_size_'): feature_groups['variant_size'].append(col); assigned = True # Check prefixes from variant processor
            elif col.startswith('opt_color_'): feature_groups['variant_color'].append(col); assigned = True
            elif col.startswith('opt_'): feature_groups['variant_other'].append(col); assigned = True # Catch remaining 'opt_'
            elif col.startswith('age_') or col.startswith('creation_recency_'): feature_groups['time_features'].append(col); assigned = True
            elif col.startswith('quarter_'): feature_groups['release_quarter'].append(col); assigned = True
            elif col in ['price_min', 'price_max', 'price_avg', 'discount_avg_perc', 'has_discount']: feature_groups['price'].append(col); assigned = True
            # Explicit flags (assuming they are numeric 0/1)
            elif col == 'available_for_sale_numeric': feature_groups['availability'].append(col); assigned = True # Assuming name change if processed
            elif col == 'is_gift_card_numeric': feature_groups['gift_card'].append(col); assigned = True # Assuming name change if processed

            # Catch-all for remaining numeric features (should be numeric by this point)
            if not assigned:
                # We assume the input df_columns only contains numeric features already
                feature_groups['other_numeric'].append(col)

        # Remove empty groups and store
        self.feature_groups = {k: v for k, v in feature_groups.items() if v}

        # Print summary
        print("\nIdentified Feature Groups:")
        total_features_grouped = 0
        for group, cols in self.feature_groups.items():
            print(f"- {group}: {len(cols)} features")
            total_features_grouped += len(cols)
        print(f"Total features assigned to groups: {total_features_grouped}")
        if total_features_grouped != len(df_columns):
             print(f"Warning: {len(df_columns) - total_features_grouped} features not assigned (check 'other_numeric').")

        return self.feature_groups

    def set_feature_weights(self, base_weights: Optional[Dict[str, float]] = None, normalize_by_count: bool = True):
        """
        Set effective weights for each feature group. Uses defaults if base_weights is None.

        Args:
            base_weights: Dictionary mapping feature group names to base weights.
                          If None, internal default weights will be used.
            normalize_by_count: If True, adjusts base weights based on the number
                                of features in each group relative to the average.
        """
        if self.feature_groups is None:
            raise ValueError("Feature groups must be identified before setting weights.")

        final_base_weights = {}
        if base_weights is None:
            print("Using internal default base weights.")
            final_base_weights = self._default_base_weights.copy()
        else:
            print("Using provided base weights.")
            final_base_weights = base_weights.copy()

        # Ensure all identified groups have a base weight (use default 1.0 if missing)
        for group in self.feature_groups:
            if group not in final_base_weights:
                default_val = self._default_base_weights.get(group, 1.0) # Try default dict first
                final_base_weights[group] = default_val
                print(f"Assigning default base weight of {default_val:.2f} to group '{group}' (not found in provided weights).")

        feature_counts = {group: len(features) for group, features in self.feature_groups.items() if features}
        self.feature_weights = {} # This will store the *effective* weights

        if normalize_by_count:
            valid_counts = [c for c in feature_counts.values() if c > 0]
            if not valid_counts:
                target_count = 1.0
                print("Warning: No features found in any group for normalization.")
            else:
                target_count = max(1, np.mean(valid_counts))
            print(f"Normalizing weights using target feature count: {target_count:.2f}")

            for group, features in self.feature_groups.items():
                count = feature_counts.get(group, 0)
                base_w = final_base_weights.get(group, 1.0) # Get the base weight for this group
                if count > 0:
                    norm_factor = np.sqrt(target_count / count)
                    self.feature_weights[group] = base_w * norm_factor
                else:
                    self.feature_weights[group] = base_w # Apply base weight directly if count is 0
        else:
            print("Not normalizing weights by feature count.")
            for group in self.feature_groups:
                self.feature_weights[group] = final_base_weights.get(group, 1.0)

        # Print effective weights applied
        print("\nEffective Feature Weights Applied:")
        for group, weight in self.feature_weights.items():
            base = final_base_weights.get(group, "N/A")
            count = feature_counts.get(group, 0)
            norm_status = "(Normalized)" if normalize_by_count and count > 0 else ""
            print(f"- {group:<20}: {weight:.4f} {norm_status} (Base: {base}, Count: {count})")

    def _apply_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the calculated effective weights to the feature matrix columns.
        Assumes X columns align with self.feature_names_in_order.

        Args:
            X: Scaled feature matrix (numpy array).

        Returns:
            Weighted feature matrix (numpy array).
        """
        if self.feature_weights is None or self.feature_groups is None:
            raise ValueError("Feature weights/groups not set.")
        if self.feature_names_in_order is None:
             raise ValueError("Feature order (feature_names_in_order) not available.")
        if X.shape[1] != len(self.feature_names_in_order):
            raise ValueError(f"Shape mismatch: X has {X.shape[1]} cols, expected {len(self.feature_names_in_order)}.")

        X_weighted = X.copy()
        feature_to_index = {name: idx for idx, name in enumerate(self.feature_names_in_order)}
        applied_weights_vector = np.ones(X.shape[1]) # Default weight 1

        print("Applying feature weights to scaled data...")
        for group, features in self.feature_groups.items():
            weight = self.feature_weights.get(group, 1.0)
            indices_for_group = [feature_to_index[feature_name] for feature_name in features if feature_name in feature_to_index]
            if indices_for_group:
                applied_weights_vector[indices_for_group] = weight
            # else: # Optional warning
            #     if features: print(f"Warning: No features found in matrix for group '{group}'.")

        X_weighted = X_weighted * applied_weights_vector
        return X_weighted

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles NaN values in the numeric feature DataFrame (fillna with 0)."""
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            print(f"Found {total_nans} missing values (NaNs). Filling with 0...")
            # print(nan_counts[nan_counts > 0]) # Optional detail
            df = df.fillna(0)
            if df.isnull().sum().sum() > 0:
                raise ValueError("Failed to remove all NaN values after fillna(0). Check data types.")
        return df

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to prepare features: ensure numeric, handle NaNs, store order.

        Args:
            features_df: DataFrame containing only the feature columns.

        Returns:
            Processed DataFrame ready for scaling.
        """
        # 1. Ensure numeric (should mostly be done upstream, but double-check)
        numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) != features_df.shape[1]:
             non_numeric = features_df.columns.difference(numeric_cols).tolist()
             print(f"Warning: Non-numeric columns found in feature DataFrame: {non_numeric}. Attempting conversion/dropping.")
             for col in non_numeric:
                  try:
                       features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                  except Exception:
                       print(f"Could not convert '{col}' to numeric.")
             features_df = features_df.select_dtypes(include=np.number) # Keep only numeric after attempt

        if features_df.empty:
             raise ValueError("No numeric features remaining after preparation.")

        # 2. Handle NaNs
        features_df = self._handle_missing_values(features_df)

        # 3. Handle Infs (replace with large finite numbers)
        if np.isinf(features_df.values).any():
            inf_count = np.isinf(features_df.values).sum()
            print(f"Warning: Found {inf_count} infinite values. Replacing with large finite numbers.")
            features_df = features_df.replace([np.inf, -np.inf], [1e9, -1e9])

        # 4. Store final feature order
        self.feature_names_in_order = features_df.columns.tolist()

        return features_df

    def fit(self, features_df: pd.DataFrame, product_ids: pd.Series):
        """
        Fit the weighted KNN model to the provided feature data.

        Args:
            features_df: DataFrame containing ONLY the numeric feature columns.
            product_ids: Series containing the product IDs, aligned with features_df rows.

        Returns:
            self: The fitted model instance.
        """
        fit_start_time = time.time()
        print("\n--- Fitting Weighted KNN Model ---")
        if features_df.empty: raise ValueError("Input features_df is empty.")
        if product_ids.empty: raise ValueError("Input product_ids is empty.")
        if len(features_df) != len(product_ids):
            raise ValueError(f"Row count mismatch: features_df has {len(features_df)}, product_ids has {len(product_ids)}.")

        self.product_ids = product_ids.copy() # Store aligned product IDs

        try:
            # --- 1. Prepare Features ---
            print("Preparing features (checking numeric, handling NaN/Inf)...")
            features_prepared_df = self._prepare_features(features_df) # Stores feature order

            # --- 2. Identify Groups & Set Weights (if not already set externally) ---
            if self.feature_groups is None:
                self._identify_feature_groups(self.feature_names_in_order)
            if self.feature_weights is None:
                self.set_feature_weights(base_weights=None, normalize_by_count=True) # Use defaults

            # --- 3. Standardize Data ---
            print("Standardizing data using StandardScaler...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features_prepared_df)
            # Check for NaNs after scaling (can happen with zero variance columns)
            if np.isnan(X_scaled).any():
                print("Warning: NaNs found after scaling (likely zero variance columns). Filling with 0.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            # --- 4. Apply Weights ---
            self.X_weighted = self._apply_weights(X_scaled)
            # Final check for NaN/Inf in weighted data
            if np.isnan(self.X_weighted).any() or np.isinf(self.X_weighted).any():
                 print("ERROR: NaN or Inf values remain in weighted matrix! Attempting cleanup...")
                 self.X_weighted = np.nan_to_num(self.X_weighted, nan=0.0, posinf=1e9, neginf=-1e9)
                 if np.isnan(self.X_weighted).any() or np.isinf(self.X_weighted).any():
                      raise ValueError("Failed to clean NaN/Inf from weighted matrix.")


            # --- 5. Fit KNN Model ---
            n_samples = self.X_weighted.shape[0]
            # Adjust k for NearestNeighbors model (must be <= n_samples)
            effective_knn_k = min(self._knn_k, n_samples)
            if effective_knn_k < self._knn_k:
                 print(f"Warning: n_neighbors+1 ({self._knn_k}) is > n_samples ({n_samples}). Setting internal k to {effective_knn_k}.")
            if effective_knn_k <= 0:
                 raise ValueError(f"Cannot fit KNN: Not enough samples ({n_samples}) for internal k={effective_knn_k}.")

            print(f"Fitting KNN model (internal k={effective_knn_k})...")
            self.knn_model = NearestNeighbors(
                n_neighbors=effective_knn_k, # Use adjusted k
                algorithm=self.algorithm, metric=self.metric, n_jobs=-1
            )
            self.knn_model.fit(self.X_weighted)

            # --- 6. Pre-compute Recommendations (Distances/Indices/Similarity Matrix) ---
            self._compute_recommendation_matrix()

            fit_end_time = time.time()
            print(f"--- KNN Fitting Complete (Duration: {fit_end_time - fit_start_time:.2f}s) ---")
            return self

        except Exception as e:
            print(f"\n--- ERROR during KNN fitting: {type(e).__name__} - {str(e)} ---")
            traceback.print_exc()
            # Reset state to indicate failure
            self.knn_model = None
            self.recommendation_matrix = None
            self.X_weighted = None
            raise e # Re-raise the exception


    def _compute_recommendation_matrix(self):
        """Computes and stores the full recommendation matrix."""
        if self.knn_model is None or self.X_weighted is None or self.product_ids is None:
             raise ValueError("Cannot compute recommendations: Model not fitted or data missing.")

        print("Finding nearest neighbors and computing similarities...")
        distances, indices = self.knn_model.kneighbors(self.X_weighted)

        self.recommendation_matrix = {}
        source_product_ids = self.product_ids.tolist() # Faster access

        for i in range(len(indices)):
            source_product_id = source_product_ids[i]
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            product_recs = []

            # Iterate through neighbors (skip index 0 - self)
            for j in range(1, len(neighbor_indices)):
                neighbor_idx = neighbor_indices[j]
                dist = neighbor_distances[j]
                try:
                    recommended_product_id = source_product_ids[neighbor_idx]
                    # Similarity calculation (ensure non-negative distance)
                    similarity = 1.0 / (1.0 + max(0, dist) + 1e-9)
                    similarity = max(0.0, min(1.0, similarity)) # Clamp to [0, 1]
                    product_recs.append((recommended_product_id, similarity))
                except IndexError:
                     print(f"Warning: Index {neighbor_idx} out of bounds for product IDs. Skipping neighbor for source {source_product_id}.")
                     continue
                # Stop if we have enough neighbors (based on the original request)
                # Note: The internal _knn_k includes self, so we need n_neighbors actual recommendations
                if len(product_recs) >= self.n_neighbors:
                     break

            self.recommendation_matrix[source_product_id] = product_recs
        print(f"Recommendation matrix computed for {len(self.recommendation_matrix)} products.")


    def get_recommendations(self, product_id: Any, n_recommendations: Optional[int] = None) -> List[Tuple[Any, float]]:
        """
        Get pre-computed recommendations for a specific product ID.

        Args:
            product_id: ID of the product to get recommendations for.
            n_recommendations: Number of recommendations to return (default: self.n_neighbors).

        Returns:
            List of (recommended_product_id, similarity_score) tuples, sorted by similarity.
        """
        if self.recommendation_matrix is None:
            print(f"Error: Recommendation matrix not available for product ID {product_id}. Model might not be fitted.")
            return []

        product_id_str = str(product_id) # Ensure string comparison if needed

        if product_id_str not in self.recommendation_matrix:
            # Simple check, could be enhanced with fuzzy matching if IDs vary slightly
            print(f"Warning: Product ID '{product_id_str}' not found in the recommendation matrix.")
            return []

        n_to_return = self.n_neighbors if n_recommendations is None else n_recommendations
        if n_to_return <= 0: return []

        # Recommendations are already sorted by similarity from _compute_recommendation_matrix
        recommendations = self.recommendation_matrix[product_id_str]

        # Filter self-recommendations (should be redundant if computation is correct, but safe)
        filtered_recommendations = [(rec_id, sim) for rec_id, sim in recommendations if str(rec_id) != product_id_str]

        return filtered_recommendations[:n_to_return]


    def export_recommendations(self, output_file: str, n_recommendations: Optional[int] = None,
                             apply_variant_filter: bool = False, similar_products_df: Optional[pd.DataFrame] = None,
                             products_df_for_titles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Export product recommendations to a CSV file.

        Args:
            output_file: Path to save the recommendations CSV.
            n_recommendations: Max number of recommendations per product (default: self.n_neighbors).
            apply_variant_filter: If True, filter out recommendations listed as similar variants.
            similar_products_df: DataFrame with pairs of similar products (needed if apply_variant_filter=True).
                                 Expected columns like 'product1_id', 'product2_id'.
            products_df_for_titles: Optional DataFrame with original product data including
                                    'product_id' and 'product_title' to add titles to the export.

        Returns:
            DataFrame containing the exported recommendations.
        """
        export_start_time = time.time()
        if self.recommendation_matrix is None:
            print("Error: Cannot export recommendations, matrix not built. Run fit() first.")
            return pd.DataFrame()

        n_to_export = self.n_neighbors if n_recommendations is None else n_recommendations
        if n_to_export <= 0:
             print("Warning: n_recommendations requested is <= 0. Exporting empty file.")
             pd.DataFrame(columns=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']).to_csv(output_file, index=False)
             return pd.DataFrame()

        rows = []
        total_variant_filtered_count = 0
        source_ids = list(self.recommendation_matrix.keys())
        print(f"Exporting top {n_to_export} recommendations for {len(source_ids)} products...")

        # --- Prepare Variant Filtering Lookup (Efficiently) ---
        variant_exclude_map = {}
        use_variant_filter = apply_variant_filter and similar_products_df is not None and not similar_products_df.empty
        if use_variant_filter:
            col1, col2 = None, None
            if 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns: col1, col2 = 'product1_id', 'product2_id'
            # Add fallbacks if needed: elif 'product1' in similar_products_df...
            if col1 and col2:
                print("Building variant exclusion map...")
                # Ensure IDs are strings for consistent lookup
                similar_products_df[col1] = similar_products_df[col1].astype(str)
                similar_products_df[col2] = similar_products_df[col2].astype(str)
                for _, row in similar_products_df.iterrows():
                    p1, p2 = row[col1], row[col2]
                    variant_exclude_map.setdefault(p1, set()).add(p2)
                    variant_exclude_map.setdefault(p2, set()).add(p1)
                print(f"Variant map built for {len(variant_exclude_map)} products.")
            else:
                print("Warning: Could not find variant ID columns in similar_products_df. Variant filtering disabled.")
                use_variant_filter = False

        # --- Iterate and Build Export Rows ---
        for source_product_id in source_ids:
            source_product_id_str = str(source_product_id) # Use string version
            recommendations = self.recommendation_matrix.get(source_product_id_str, [])

            # Initial filter (self-recommendations - should be handled in compute, but safe)
            current_recs = [(str(rec_id), sim) for rec_id, sim in recommendations if str(rec_id) != source_product_id_str]

            # Apply variant filter if enabled
            variants_filtered_this_product = 0
            if use_variant_filter:
                exclude_ids = variant_exclude_map.get(source_product_id_str, set())
                if exclude_ids:
                    count_before = len(current_recs)
                    current_recs = [(rec_id, sim) for rec_id, sim in current_recs if rec_id not in exclude_ids]
                    variants_filtered_this_product = count_before - len(current_recs)
                    total_variant_filtered_count += variants_filtered_this_product

            # Take Top N *after* filtering
            final_recs_for_product = current_recs[:n_to_export]

            # Add to export rows
            for rank, (rec_id, similarity) in enumerate(final_recs_for_product):
                rows.append({
                    'source_product_id': source_product_id_str, # Export consistent string IDs
                    'recommended_product_id': rec_id,
                    'rank': rank + 1,
                    'similarity_score': similarity
                })

        # --- Create DataFrame and Save ---
        recommendations_df = pd.DataFrame(rows)

        if not recommendations_df.empty:
             # Add titles if requested
            if products_df_for_titles is not None and 'product_id' in products_df_for_titles.columns and 'product_title' in products_df_for_titles.columns:
                print("Adding product titles...")
                try:
                     # Create map using string IDs
                    products_df_for_titles['product_id_str'] = products_df_for_titles['product_id'].astype(str)
                    title_map = products_df_for_titles.set_index('product_id_str')['product_title'].to_dict()
                    recommendations_df['source_product_title'] = recommendations_df['source_product_id'].map(title_map).fillna('Title NF')
                    recommendations_df['recommended_product_title'] = recommendations_df['recommended_product_id'].map(title_map).fillna('Title NF')
                    # Reorder columns
                    cols_order = ['source_product_id', 'source_product_title', 'recommended_product_id', 'recommended_product_title', 'rank', 'similarity_score']
                    recommendations_df = recommendations_df[[col for col in cols_order if col in recommendations_df.columns]]
                except Exception as title_err:
                    print(f"Warning: Error adding titles: {title_err}.")
                    # Ensure columns don't exist if mapping failed
                    recommendations_df = recommendations_df.drop(columns=['source_product_title', 'recommended_product_title'], errors='ignore')

            # Save the DataFrame
            if save_dataframe(recommendations_df, output_file):
                 export_end_time = time.time()
                 print(f"Recommendations exported to {output_file} (Duration: {export_end_time - export_start_time:.2f}s)")
                 if use_variant_filter: print(f"Total recommendations filtered as variants: {total_variant_filtered_count}")
            # Else: save_dataframe prints error
        else:
            print(f"WARNING: No recommendations generated to export to {output_file}.")
            # Optionally save empty file with headers
            pd.DataFrame(columns=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']).to_csv(output_file, index=False)

        return recommendations_df


    def calculate_average_similarity(self) -> float:
        """Calculate the average similarity score across all generated recommendations."""
        if self.recommendation_matrix is None: return 0.0
        all_similarities = [sim for recs in self.recommendation_matrix.values() for _, sim in recs]
        return np.mean(all_similarities) if all_similarities else 0.0

    def visualize_feature_importance(self, output_file: str):
        """Visualize assigned effective feature weights."""
        # (Implementation can be kept similar to the previous version)
        if self.feature_weights is None or self.feature_groups is None or self.feature_names_in_order is None:
            print("Cannot visualize feature importance: Weights, groups, or feature order not set.")
            return
        try:
            feature_importance_map = {feature_name: self.feature_weights.get(group, 1.0)
                                      for group, features in self.feature_groups.items()
                                      for feature_name in features}
            if not feature_importance_map: print("No feature importance data."); return
            importance_series = pd.Series(feature_importance_map).sort_values(ascending=False)
            num_features_to_show = min(50, len(importance_series))
            top_features = importance_series.head(num_features_to_show)

            plt.figure(figsize=(14, max(8, num_features_to_show * 0.3)))
            # ... (rest of plotting logic - shortening names, barh plot, savefig, close) ...
            y_pos = np.arange(len(top_features))
            plt.barh(y_pos, top_features.values, align='center', color=sns.color_palette("viridis", len(top_features)))
            plt.yticks(y_pos, [name[:45] + '...' if len(name) > 45 else name for name in top_features.index], fontsize=9) # Simple shortening
            plt.gca().invert_yaxis()
            plt.xlabel('Assigned Effective Feature Weight')
            plt.title(f'Top {num_features_to_show} Features by Assigned Group Weight')
            plt.tight_layout(pad=1.5)
            plt.savefig(output_file, dpi=150); plt.close()
            print(f"Feature importance visualization saved to {output_file}")
        except Exception as e: print(f"Error visualizing feature importance: {e}")

    def visualize_tsne(self, output_file: str, sample_size: int = 1000, perplexity: int = 30):
        """Visualize weighted features using t-SNE."""
        # (Implementation can be kept similar to the previous version, using self.X_weighted)
        if self.X_weighted is None: print("Cannot visualize t-SNE: Weighted features not available."); return
        try:
            X_to_plot = np.nan_to_num(self.X_weighted, nan=0.0, posinf=1e9, neginf=-1e9) # Clean data
            num_samples = X_to_plot.shape[0]
            indices = np.arange(num_samples)
            if num_samples > sample_size:
                 indices = np.random.choice(num_samples, sample_size, replace=False)
            X_sample = X_to_plot[indices]
            if X_sample.shape[0] <= 1: print("Skipping t-SNE: Not enough samples."); return

            effective_perplexity = min(max(1, perplexity), X_sample.shape[0] - 1)
            if effective_perplexity != perplexity: print(f"Adjusted t-SNE perplexity to {effective_perplexity}.")

            print(f"Applying t-SNE (perplexity={effective_perplexity}, samples={X_sample.shape[0]})...")
            tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=effective_perplexity,
                        init='pca', learning_rate='auto', n_jobs=-1)
            X_tsne = tsne.fit_transform(X_sample)

            plt.figure(figsize=(12, 10))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=15) # Simple plot for now
            plt.title(f't-SNE Visualization of {len(X_sample)} Products (Weighted Features)')
            plt.xlabel('t-SNE Dimension 1'); plt.ylabel('t-SNE Dimension 2')
            plt.xticks([]); plt.yticks([]); plt.tight_layout()
            plt.savefig(output_file, dpi=150); plt.close()
            print(f"t-SNE visualization saved to {output_file}")
        except Exception as e: print(f"Error visualizing t-SNE: {e}")


    def visualize_similarity_distribution(self, output_file="similarity_distribution.png"):
        """Visualize the distribution of similarity scores."""
        # (Implementation can be kept similar to the previous version)
        if self.recommendation_matrix is None: print("Cannot visualize similarity: Matrix not built."); return
        try:
            all_similarities = np.array([sim for recs in self.recommendation_matrix.values() for _, sim in recs])
            all_similarities = all_similarities[np.isfinite(all_similarities)]
            if len(all_similarities) == 0: print("No finite similarity scores found."); return

            plt.figure(figsize=(10, 6))
            plt.hist(all_similarities, bins=50, alpha=0.75, color='skyblue', edgecolor='black') # Use more bins maybe
            plt.xlabel('Similarity Score'); plt.ylabel('Frequency')
            plt.title('Distribution of Product Similarity Scores'); plt.grid(axis='y', linestyle='--', alpha=0.6)
            mean_sim = np.mean(all_similarities); median_sim = np.median(all_similarities)
            plt.axvline(mean_sim, color='red', linestyle='dashed', label=f'Mean: {mean_sim:.3f}')
            plt.axvline(median_sim, color='green', linestyle='dashed', label=f'Median: {median_sim:.3f}')
            plt.legend(); plt.xlim(0, 1); plt.tight_layout()
            plt.savefig(output_file, dpi=120); plt.close()
            print(f"Similarity distribution visualization saved to {output_file}")
            # Print stats
            print("\nSimilarity Score Stats:"); print(f"- Mean: {mean_sim:.4f}, Median: {median_sim:.4f}, Std Dev: {np.std(all_similarities):.4f}, Min: {np.min(all_similarities):.4f}, Max: {np.max(all_similarities):.4f}")

        except Exception as e: print(f"Error visualizing similarity distribution: {e}")


    def save_model(self, output_dir: str):
        """Saves the fitted model components."""
        print(f"Saving model components to directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        saved_any = False
        saved_any |= save_model_component(self.knn_model, os.path.join(output_dir, "knn_model.pkl"))
        saved_any |= save_model_component(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        # Save parameters (including feature order!)
        model_params = {
            'feature_weights': self.feature_weights, 'feature_groups': self.feature_groups,
            'feature_names_in_order': self.feature_names_in_order, # CRUCIAL
            'n_neighbors': self.n_neighbors, 'algorithm': self.algorithm,
            'metric': self.metric, 'random_state': self.random_state,
        }
        saved_any |= save_model_component(model_params, os.path.join(output_dir, "model_params.pkl"))
        # Save product IDs (essential for using the loaded model)
        saved_any |= save_model_component(self.product_ids, os.path.join(output_dir, "product_ids.pkl.z"), compress=3)
        # Optionally save recommendation matrix (can be large, can be recomputed)
        # save_model_component(self.recommendation_matrix, os.path.join(output_dir, "recommendation_matrix.pkl.z"), compress=3)

        if saved_any: print("Model components saved.")
        else: print("Warning: No model components were saved (model might be unfit).")

    @classmethod
    def load_model(cls, input_dir: str):
        """Loads a previously saved model."""
        print(f"Loading model components from directory: {input_dir}")
        if not os.path.isdir(input_dir): raise FileNotFoundError(f"Model directory not found: {input_dir}")

        params = load_model_component(os.path.join(input_dir, "model_params.pkl"))
        if params is None: raise FileNotFoundError("model_params.pkl not found or failed to load.")

        # Create instance with loaded params
        instance = cls(
            n_neighbors=params.get('n_neighbors', 12), # Provide defaults
            algorithm=params.get('algorithm', 'auto'),
            metric=params.get('metric', 'euclidean'),
            random_state=params.get('random_state', 42)
        )

        # Load essential components
        instance.knn_model = load_model_component(os.path.join(input_dir, "knn_model.pkl"))
        instance.scaler = load_model_component(os.path.join(input_dir, "scaler.pkl"))
        instance.product_ids = load_model_component(os.path.join(input_dir, "product_ids.pkl.z"))
        # Load parameters into instance attributes
        instance.feature_weights = params.get('feature_weights')
        instance.feature_groups = params.get('feature_groups')
        instance.feature_names_in_order = params.get('feature_names_in_order')

        # Optionally load recommendation matrix if needed immediately
        rec_matrix_path = os.path.join(input_dir, "recommendation_matrix.pkl.z")
        if os.path.exists(rec_matrix_path):
             instance.recommendation_matrix = load_model_component(rec_matrix_path)
             print("Loaded pre-computed recommendation matrix.")
        else:
             print("Recommendation matrix not found. It needs to be recomputed if needed.")


        # --- Critical Validations ---
        if instance.knn_model is None: print("CRITICAL WARNING: KNN model component missing.")
        if instance.scaler is None: print("CRITICAL WARNING: Scaler component missing.")
        if instance.product_ids is None: print("CRITICAL WARNING: Product IDs missing. Model cannot generate recommendations by ID.")
        if instance.feature_names_in_order is None: print("CRITICAL WARNING: Feature order missing. Model cannot be safely applied to new data.")

        print("Model loading complete.")
        return instance


# --- WeightOptimizer Class ---
class WeightOptimizer:
    """
    Finds optimal feature group weights for the _WeightedKNN model.
    """
    def __init__(self, features_df: pd.DataFrame, product_ids: pd.Series,
                 n_neighbors: int = 12, output_dir: str = "weight_optimization", random_state: int = 42):
        """
        Initialize the weight optimizer.

        Args:
            features_df: DataFrame with ONLY numeric features.
            product_ids: Series of product IDs aligned with features_df.
            n_neighbors: Number of neighbors for KNN evaluation.
            output_dir: Directory to save optimization results.
            random_state: Random seed.
        """
        if features_df.empty or product_ids.empty or len(features_df) != len(product_ids):
            raise ValueError("Invalid input features_df or product_ids for WeightOptimizer.")

        self.features_df = features_df.copy() # Store prepared features
        self.product_ids = product_ids.copy() # Store aligned IDs
        self.n_neighbors = n_neighbors
        self.output_dir = output_dir
        self.random_state = random_state
        os.makedirs(output_dir, exist_ok=True)

        self.optimization_results: List[Dict] = []
        self.best_weights: Optional[Dict[str, float]] = None
        self.best_score: float = -1.0
        self.best_model: Optional[_WeightedKNN] = None # Stores the best fitted model instance

        # Initialize base model attributes needed for optimization runs
        self._base_knn_config = {'n_neighbors': n_neighbors, 'random_state': random_state}
        self._feature_groups = self._identify_groups_for_opt(features_df.columns.tolist())
        self._default_weights = self._get_defaults_for_opt()

        # Evaluate baseline (default weights)
        self._evaluate_baseline()

    def _identify_groups_for_opt(self, df_columns: List[str]) -> Dict[str, List[str]]:
        """Internal helper to identify groups just for the optimizer."""
        temp_model = _WeightedKNN(**self._base_knn_config)
        return temp_model._identify_feature_groups(df_columns)

    def _get_defaults_for_opt(self) -> Dict[str, float]:
        """Internal helper to get default weights relevant to identified groups."""
        temp_model = _WeightedKNN(**self._base_knn_config)
        defaults = temp_model._default_base_weights.copy()
        # Ensure all identified groups have a default
        for group in self._feature_groups:
            if group not in defaults:
                defaults[group] = 1.0
        return defaults

    def _evaluate_baseline(self):
        """Evaluates the default weights as a starting point."""
        print("\nEvaluating default weights as baseline...")
        baseline_score = self.evaluate_weights(self._default_weights, "default_baseline", save_model_if_best=True)
        if baseline_score > self.best_score: # Use > only
            print(f"Default weights established baseline score: {baseline_score:.4f}")
        else:
            print(f"Default weights score ({baseline_score:.4f}) didn't improve initial best ({self.best_score:.4f}).")


    def evaluate_weights(self, base_weights: Dict[str, float], weight_name: str = "custom", save_model_if_best: bool = False) -> float:
        """
        Evaluate a specific set of base weights.

        Args:
            base_weights: Dictionary of base weights for feature groups.
            weight_name: Name for logging and results.
            save_model_if_best: If True, store the model instance if it's the best so far.

        Returns:
            Average similarity score (0.0 on error).
        """
        eval_start_time = time.time()
        print(f"\n--- Evaluating weights: '{weight_name}' ---")
        model = _WeightedKNN(**self._base_knn_config) # Create new instance
        score = 0.0
        error_msg = None

        try:
            # Pass identified groups, set weights, fit
            model.feature_groups = self._feature_groups # Use groups identified by optimizer
            model.set_feature_weights(base_weights, normalize_by_count=True) # Normalize for fair comparison usually
            # Fit using the *stored* prepared features and IDs
            model.fit(self.features_df, self.product_ids)

            # Check if fit was successful (recommendation_matrix should exist)
            if model.recommendation_matrix is not None:
                score = model.calculate_average_similarity()
                if not pd.notna(score): score = 0.0 # Handle potential NaN score
            else:
                 print(f"ERROR: Model fitting failed for weights '{weight_name}'. Recommendation matrix is None.")
                 error_msg = "Fit failed (matrix is None)"
                 score = 0.0

        except Exception as e:
            print(f"\n--- ERROR evaluating weights '{weight_name}': {type(e).__name__} - {str(e)} ---")
            traceback.print_exc()
            score = 0.0
            error_msg = str(e)

        eval_time = time.time() - eval_start_time
        # Record result
        result = {
            'weight_name': weight_name, 'weights': base_weights.copy(), 'avg_similarity': score,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'eval_time_seconds': eval_time, 'error': error_msg
        }
        self.optimization_results.append(result)

        print(f"--- Result for '{weight_name}': Avg Similarity = {score:.4f} (Eval time: {eval_time:.2f}s) {'[ERROR]' if error_msg else ''} ---")

        # Update best score and potentially the best model instance
        # Use > comparison, as >= could lead to replacing with equally good but later models
        if error_msg is None and score > self.best_score:
             print(f"*** New best weights found! Score: {score:.4f} (Prev best: {self.best_score:.4f}) ***")
             self.best_score = score
             self.best_weights = base_weights.copy()
             if save_model_if_best:
                  print("Storing this model instance as the best.")
                  self.best_model = model # Store the fitted model object
             # else: # Optional: Clear previous best model if not saving this one
                 # if self.best_model is not None and model is not self.best_model: self.best_model = None


        return score

    # --- Grid Search ---
    def grid_search(self, weight_options: Optional[Dict[str, list]] = None, fixed_weights: Optional[Dict[str, float]] = None, save_results: bool = True):
        # (Implementation remains the same as before, uses self.evaluate_weights)
        if weight_options is None: weight_options = {}
        if fixed_weights is None: fixed_weights = {}
        all_groups = list(self._feature_groups.keys())
        search_space = {}
        default_range = [0.5, 1.0, 2.0, 5.0]
        print("Building grid search space...")
        for group in all_groups:
            if group in fixed_weights: search_space[group] = [fixed_weights[group]]
            elif group in weight_options: search_space[group] = weight_options[group]
            else: search_space[group] = default_range
            print(f"- Group '{group}': {'Fixed at '+str(search_space[group][0]) if group in fixed_weights else 'Options = '+str(search_space[group])}")

        search_keys = list(search_space.keys()); search_values = list(search_space.values())
        all_combinations = list(product(*search_values))
        total_combinations = len(all_combinations)
        if total_combinations == 0: print("Error: No combinations for grid search."); return self.get_optimization_results()
        if total_combinations > 5000: print(f"Warning: Grid search has {total_combinations} combinations, this may take time.")

        print(f"\nStarting grid search ({total_combinations} combinations)...")
        grid_start_time = time.time()
        for i, weight_combination in enumerate(all_combinations):
            current_weights = dict(zip(search_keys, weight_combination))
            self.evaluate_weights(current_weights, f"grid_comb_{i+1}", save_model_if_best=True)
        print(f"\nGrid search completed in {time.time() - grid_start_time:.2f} seconds.")
        if save_results: self.save_optimization_results()
        return self.get_optimization_results()


    # --- Random Search ---
    def random_search(self, n_iterations: int = 50, weight_range: Tuple[float, float] = (0.1, 5.0), save_results: bool = True):
        # (Implementation remains the same as before, uses self.evaluate_weights)
        all_groups = list(self._feature_groups.keys())
        if not all_groups: print("Error: No feature groups for random search."); return self.get_optimization_results()
        print(f"\nStarting random search ({n_iterations} iterations, range: {weight_range})...")
        random_start_time = time.time()
        for i in range(n_iterations):
            current_weights = {group: np.random.uniform(weight_range[0], weight_range[1]) for group in all_groups}
            self.evaluate_weights(current_weights, f"random_iter_{i+1}", save_model_if_best=True)
        print(f"\nRandom search completed in {time.time() - random_start_time:.2f} seconds.")
        if save_results: self.save_optimization_results()
        return self.get_optimization_results()


    # --- Evolutionary Search Helpers ---
    def _tournament_selection(self, population, tournament_size):
        if not population: raise ValueError("Empty population for tournament.")
        actual_size = min(tournament_size, len(population))
        if actual_size <= 0: return population[0] # Should not happen if called correctly
        indices = np.random.choice(len(population), size=actual_size, replace=False)
        participants = [population[i] for i in indices]
        return max(participants, key=lambda x: x['fitness'])

    def _crossover(self, weights1, weights2):
        child = {}
        for group in weights1: # Assume both have same keys
            child[group] = weights1[group] if np.random.random() < 0.5 else weights2[group]
        return child

    def _mutate(self, weights, mutation_rate, weight_range):
        for group in weights:
            if np.random.random() < mutation_rate:
                weights[group] = np.random.uniform(weight_range[0], weight_range[1])
                weights[group] = max(0.01, weights[group]) # Ensure minimum weight


    # --- Evolutionary Search ---
    def evolutionary_search(self, population_size: int = 20, generations: int = 10, mutation_rate: float = 0.2,
                            crossover_rate: float = 0.8, tournament_size: int = 3,
                            weight_range: Tuple[float, float] = (0.1, 5.0), save_results: bool = True):
        # (Implementation remains the same as before, uses self.evaluate_weights and helpers)
        all_groups = list(self._feature_groups.keys())
        if not all_groups: print("Error: No feature groups for evolutionary search."); return self.get_optimization_results()
        if population_size <= 1 or generations <= 0 or tournament_size >= population_size:
            print("Error: Invalid evolutionary parameters."); return self.get_optimization_results()

        print(f"\nStarting evolutionary search (Pop: {population_size}, Gens: {generations})...")
        evo_start_time = time.time()

        # Initialize Population
        population = []
        print("Initializing Generation 0...")
        for i in range(population_size):
            weights = {group: np.random.uniform(weight_range[0], weight_range[1]) for group in all_groups}
            fitness = self.evaluate_weights(weights, f"evo_gen0_ind{i+1}", save_model_if_best=True)
            population.append({'weights': weights, 'fitness': fitness, 'name': f"evo_gen0_ind{i+1}"})
        population.sort(key=lambda x: x['fitness'], reverse=True)
        print(f"Generation 0 Initialized. Best fitness: {population[0]['fitness']:.4f}")

        # Evolution Loop
        for gen in range(1, generations + 1):
            gen_start_time = time.time()
            print(f"\n--- Starting Generation {gen}/{generations} ---")
            new_population = []
            n_elite = max(1, population_size // 10) # Keep top 10% or 1
            if n_elite > 0: new_population.extend(population[:n_elite]) # Elitism

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, tournament_size)
                parent2 = self._tournament_selection(population, tournament_size)
                child_weights = self._crossover(parent1['weights'], parent2['weights']) if np.random.random() < crossover_rate else (parent1['weights'].copy() if parent1['fitness'] >= parent2['fitness'] else parent2['weights'].copy())
                self._mutate(child_weights, mutation_rate, weight_range)
                child_fitness = self.evaluate_weights(child_weights, f"evo_gen{gen}_ind{len(new_population)+1}", save_model_if_best=True)
                new_population.append({'weights': child_weights, 'fitness': child_fitness, 'name': f"evo_gen{gen}_ind{len(new_population)+1}"})

            population = new_population
            population.sort(key=lambda x: x['fitness'], reverse=True)
            print(f"--- Generation {gen} Completed (Time: {time.time() - gen_start_time:.2f}s). Best fitness: {population[0]['fitness']:.4f} ---")

        print(f"\nEvolutionary search completed in {time.time() - evo_start_time:.2f} seconds.")
        if save_results: self.save_optimization_results()
        return self.get_optimization_results()


    # --- Result Handling ---
    def get_optimization_results(self) -> pd.DataFrame:
        """Returns all optimization results as a DataFrame."""
        # (Implementation remains the same as before)
        if not self.optimization_results: return pd.DataFrame()
        results_list_for_df = []
        all_groups = set().union(*(res.get('weights', {}).keys() for res in self.optimization_results))
        sorted_groups = sorted(list(all_groups))
        for result in self.optimization_results:
             row = { 'weight_name': result.get('weight_name'), 'avg_similarity': result.get('avg_similarity'),
                     'timestamp': result.get('timestamp'), 'eval_time_seconds': result.get('eval_time_seconds'),
                     'error': result.get('error')}
             base_weights = result.get('weights', {})
             for group in sorted_groups: row[f"weight_{group}"] = base_weights.get(group)
             results_list_for_df.append(row)
        results_df = pd.DataFrame(results_list_for_df)
        if 'avg_similarity' in results_df.columns: results_df = results_df.sort_values('avg_similarity', ascending=False, na_position='last')
        return results_df

    def save_optimization_results(self, filename: str = "optimization_results.csv"):
        """Saves optimization results to CSV and best weights to JSON."""
        # (Implementation remains the same as before)
        results_df = self.get_optimization_results()
        if results_df.empty: print("No optimization results to save."); return
        full_csv_path = os.path.join(self.output_dir, filename)
        if save_dataframe(results_df, full_csv_path): print(f"Optimization results saved to {full_csv_path}")

        if self.best_weights:
             best_weights_file = os.path.join(self.output_dir, "best_weights.json")
             if save_json_report(self.best_weights, best_weights_file):
                  print(f"Best weights (Score: {self.best_score:.4f}) saved to {best_weights_file}")
        else: print("No best weights recorded.")


    def build_optimized_model(self, save_model_files: bool = True) -> Optional[_WeightedKNN]:
        """
        Builds and fits a new model using the best weights found.

        Returns:
            The fitted _WeightedKNN model instance, or None if optimization failed or
            no best weights were found.
        """
        print("\n--- Building Final Model with Optimized Weights ---")
        weights_to_use = None
        model_to_return = None

        if self.best_weights:
            print(f"Using best weights found (Avg Similarity: {self.best_score:.4f})")
            weights_to_use = self.best_weights
            # Check if the best model instance was stored and seems valid
            if self.best_model and self.best_model.recommendation_matrix is not None:
                 # Optional: More rigorous check if weights match?
                 print("Using the pre-fitted best model instance.")
                 model_to_return = self.best_model
            # else: # If best model instance wasn't stored or seems invalid, refit
                 # print("Best model instance not available or invalid, refitting...")
                 # weights_to_use = self.best_weights # Already set
                 # model_to_return = None # Force refit below
        else:
            print("Warning: No best weights found during optimization. Falling back to default weights.")
            weights_to_use = self._default_weights
            if not weights_to_use:
                 print("Error: Cannot build model, default weights are also missing.")
                 return None
            # Force refit if using defaults after optimization attempt
            model_to_return = None


        # If we don't have a valid pre-fitted model, create and fit a new one
        if model_to_return is None:
             print("Building and fitting a new model instance...")
             model_to_return = _WeightedKNN(**self._base_knn_config)
             model_to_return.feature_groups = self._feature_groups # Use optimizer's groups
             model_to_return.set_feature_weights(weights_to_use, normalize_by_count=True)
             try:
                 # Fit using the stored features/IDs from the optimizer's init
                 model_to_return.fit(self.features_df, self.product_ids)
                 if model_to_return.recommendation_matrix is None:
                      print("Error: Fitting the final model failed (no recommendation matrix).")
                      return None
             except Exception as e:
                 print(f"Error fitting the final model: {e}")
                 return None

        # Save the final model (either the stored best or the newly fitted one)
        if save_model_files:
            final_model_dir = os.path.join(self.output_dir, "optimized_model_final")
            print(f"\nSaving the final model components to: {final_model_dir}")
            model_to_return.save_model(final_model_dir)

        print("--- Final Model Building Complete ---")
        return model_to_return