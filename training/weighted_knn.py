# weighted_knn.py

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



class _WeightedKNN:
    """
    Private implementation of weighted KNN algorithm for product recommendations
    where different feature groups can be assigned different weights.
    This class should NOT be instantiated directly outside of the run_knn function.
    """
    def __init__(self, n_neighbors=12, algorithm='auto', metric='euclidean', random_state=42):
        """
        Initialize the WeightedKNN model.

        Args:
            n_neighbors: Number of neighbors to consider for each recommendation
            algorithm: Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            metric: Distance metric to use ('euclidean', 'manhattan', 'cosine', etc.)
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.random_state = random_state
        self.knn_model = None
        self.feature_weights = None
        self.feature_groups = None
        self.scaler = StandardScaler()
        self.product_ids = None
        self.product_titles = None
        self.recommendation_matrix = None 
        self.distances = None 
        self.indices = None 
        self.X_weighted = None 
        self.feature_names_in_order = None 

    def load_data(self, data_file):
        """Loads data and extracts features, handling potential issues."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print(f"Loading data from {data_file}")
        df = pd.read_csv(data_file, low_memory=False)

        # Keep product ID and title for reference (if they exist)
        self.product_ids = df['product_id'].copy() if 'product_id' in df.columns else None
        self.product_titles = df['product_title'].copy() if 'product_title' in df.columns else None

        # Basic check for product_id
        if self.product_ids is None:
             print("Warning: 'product_id' column not found. Using index as fallback ID.")
             self.product_ids = pd.Series(df.index.astype(str)) # Use index if no ID

        if self.product_ids.isnull().any():
             nan_count = self.product_ids.isnull().sum()
             print(f"Warning: Found {nan_count} missing product IDs. Filling with index (prefixed).")
             # Fill missing IDs using index, prefixing to avoid collisions
             # Create a boolean mask for NaNs
             nan_mask = self.product_ids.isnull()
             # Apply the fill only where needed using the mask and df.index aligned with the mask
             self.product_ids.loc[nan_mask] = df.index[nan_mask].map(lambda x: f"missing_id_{x}")


        # Remove non-feature columns (be explicit)
        # Expanded list based on known columns from pipeline
        non_feature_cols = ['product_id', 'product_title', 'handle', 'description',
                            'tags', 'collections', 'variants', 'metafields', 'metafields_config', # Original complex types
                            'created_at' # Original timestamp if not processed
                           ]
        potential_feature_cols = [col for col in df.columns if col not in non_feature_cols]

        # Robust numeric filtering
        numeric_cols = []
        potential_feature_df = df[potential_feature_cols].copy() # Work on a copy
        for col in potential_feature_df.columns:
            # Optimization: Check dtype first
            if pd.api.types.is_numeric_dtype(potential_feature_df[col]):
                # Check for infinities which can cause issues later
                if not np.isinf(potential_feature_df[col]).any():
                    numeric_cols.append(col)
                else:
                     print(f"Skipping column '{col}': Contains infinite values.")
            else:
                # Attempt conversion only if not already numeric
                try:
                    # Attempt conversion, coerce errors to NaN
                    converted_col = pd.to_numeric(potential_feature_df[col], errors='coerce')
                    # Only keep if *some* values could be converted and it's not all NaN
                    if converted_col.notnull().any():
                        # Check for infinities after conversion
                        if not np.isinf(converted_col).any():
                             numeric_cols.append(col)
                        else:
                             print(f"Skipping column '{col}' post-conversion: Contains infinite values.")

                    # else: # Optional: report columns that were fully non-numeric or all NaN post-conversion
                    #     if converted_col.isnull().all():
                    #          print(f"Skipping column '{col}': All values are NaN after attempted numeric conversion.")
                    #     else: # Should be caught by is_numeric_dtype, but safety check
                    #          print(f"Skipping column '{col}': Cannot convert to numeric.")
                except Exception as e: # Catch broader errors during conversion attempt
                    print(f"Skipping column '{col}' due to error during numeric check: {e}")

        # Keep only confirmed numeric columns without Infs
        if len(numeric_cols) < len(potential_feature_cols):
            dropped_cols_count = len(potential_feature_cols) - len(numeric_cols)
            print(f"Removed {dropped_cols_count} non-numeric, problematic (e.g., Inf), or empty columns from feature set.")
            feature_df = df[numeric_cols].copy() # Use the original df to select final cols
        else:
            feature_df = df[numeric_cols].copy()

        if feature_df.empty:
             raise ValueError("No usable numeric features found in the data.")

        print(f"Loaded {len(df)} products with {len(feature_df.columns)} usable numeric features.")

        # Store feature names in the order they will be processed
        self.feature_names_in_order = feature_df.columns.tolist()

        # Handle missing values *after* identifying numeric columns
        feature_df = self.handle_missing_values(feature_df)

        return feature_df

    def identify_feature_groups(self, df):
        """
        Automatically identify and group features based on their names.
        Uses the stored feature order (self.feature_names_in_order).

        Args:
            df: DataFrame with features (used mainly to get column names if not stored)

        Returns:
            dict: Dictionary of feature groups
        """
        # Initialize feature groups, adding new ones for title and handle
        feature_groups = {
            'description_tfidf': [],
            'title_tfidf': [],          
            'handle_tfidf': [],         
            'metafield_data': [],
            'collections': [],
            'tags': [],
            'product_type': [],
            'vendor': [],
            'variant_size': [],
            'variant_color': [],
            'variant_other': [],        # For non-size/color variant options
            'price': [],
            'time_features': [],        # Combined time features
            'release_quarter': [],      # Dummies for release quarter
            # 'seasons': [], # Removed 'seasons' as separate, included in time_features if named appropriately
            'availability': [],         # availability flag
            'gift_card': [],            # gift card flag
            'other_numeric': []         # Catch-all for remaining numeric features
        }

        # Use the stored feature names if available, otherwise use df.columns
        columns_to_process = self.feature_names_in_order if self.feature_names_in_order else df.columns

        # Pattern matching for feature groups based on column names
        for col in columns_to_process:
            assigned = False
            # Order checks from specific to general
            if col.startswith('description_'):
                feature_groups['description_tfidf'].append(col); assigned = True
            elif col.startswith('title_'):          # <<< NEW CHECK
                feature_groups['title_tfidf'].append(col); assigned = True
            elif col.startswith('handle_'):         # <<< NEW CHECK
                feature_groups['handle_tfidf'].append(col); assigned = True
            elif col.startswith('metafield_'):
                feature_groups['metafield_data'].append(col); assigned = True
            elif col.startswith('collections_'):
                feature_groups['collections'].append(col); assigned = True
            elif col.startswith('tag_'):
                feature_groups['tags'].append(col); assigned = True
            elif col.startswith('product_type_'):
                feature_groups['product_type'].append(col); assigned = True
            elif col.startswith('vendor_'):
                feature_groups['vendor'].append(col); assigned = True
            elif col.startswith('variant_size_'):
                feature_groups['variant_size'].append(col); assigned = True
            elif col.startswith('variant_color_'):
                feature_groups['variant_color'].append(col); assigned = True
            elif col.startswith('variant_') and not col.startswith('variant_size_') and not col.startswith('variant_color_'): # Catch other variants
                 feature_groups['variant_other'].append(col); assigned = True
            # Time features (more specific)
            elif col.startswith('created_') or col in ['days_since_first_product', 'is_recent_product']:
                feature_groups['time_features'].append(col); assigned = True
            # Release quarter dummies
            elif col.startswith('release_quarter_'):
                feature_groups['release_quarter'].append(col); assigned = True
            # Price features
            elif col in ['min_price', 'max_price', 'has_discount'] or col.startswith('price_'):
                 feature_groups['price'].append(col); assigned = True
            # Availability
            elif col == 'available_for_sale':
                 feature_groups['availability'].append(col); assigned = True
            # Gift Card
            elif col == 'is_gift_card':
                 feature_groups['gift_card'].append(col); assigned = True

            # If still not assigned, and it's confirmed numeric, put in 'other_numeric'
            elif not assigned and pd.api.types.is_numeric_dtype(df[col]):
                 feature_groups['other_numeric'].append(col); assigned = True

            # Optional: Log unassigned columns (should be rare if load_data worked)
            # if not assigned:
            #     print(f"Warning: Column '{col}' (dtype: {df[col].dtype}) was not assigned to any feature group.")


        # Remove empty groups
        self.feature_groups = {k: v for k, v in feature_groups.items() if v}

        # Print feature group summary
        print("\nIdentified Feature Groups:")
        total_features_grouped = 0
        for group, cols in self.feature_groups.items():
            print(f"- {group}: {len(cols)} features")
            total_features_grouped += len(cols)
        print(f"Total features assigned to groups: {total_features_grouped}")
        if total_features_grouped != len(columns_to_process):
             ungrouped_count = len(columns_to_process) - total_features_grouped
             print(f"Warning: {ungrouped_count} features were not assigned to any group. Check 'other_numeric' or potential unassigned features.")


        # Verify all features are in exactly one group (more robust check)
        all_grouped_features = set()
        feature_counts_in_groups = {}
        for group, features in self.feature_groups.items():
            for feature in features:
                 feature_counts_in_groups[feature] = feature_counts_in_groups.get(feature, 0) + 1
                 all_grouped_features.add(feature)

        duplicates = {f: count for f, count in feature_counts_in_groups.items() if count > 1}
        if duplicates:
             print(f"Warning: Some features are listed in multiple groups! Duplicates: {duplicates}")

        if set(columns_to_process) != all_grouped_features:
            ungrouped = set(columns_to_process) - all_grouped_features
            print(f"Warning: The following features were not assigned to any group: {ungrouped}")


        return self.feature_groups

    def set_feature_weights(self, base_weights=None, normalize_by_count=True):
        """
        Set weights for each feature group, with optional normalization by feature count.

        Args:
            base_weights: Dictionary mapping feature group names to base weights.
                          If None, default weights will be used.
            normalize_by_count: Whether to normalize weights by feature count.

        Returns:
            dict: The effective feature weights applied to the model.
        """
        if self.feature_groups is None:
            raise ValueError("Feature groups must be identified before setting weights. Call identify_feature_groups() first.")

        if base_weights is None:
            # Define comprehensive default weights INCLUDING new groups
            base_weights = {
                'description_tfidf': 1.0,
                'title_tfidf': 1.2,          # <<< NEW DEFAULT (Slightly higher than description?)
                'handle_tfidf': 0.8,         # <<< NEW DEFAULT (Lower weight for handle?)
                'metafield_data': 1.0,
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
                'other_numeric': 0.5         # Default for uncategorized numeric
            }
            print("Using default base weights.")
        else:
            print("Using provided base weights.")

        # Ensure all identified groups have a base weight (default to 1.0 if missing)
        final_base_weights = base_weights.copy()
        for group in self.feature_groups:
            if group not in final_base_weights:
                final_base_weights[group] = 1.0 # Assign default 1.0
                print(f"Assigning default base weight of 1.0 to group '{group}' (not found in provided weights).")

        # Count features in each group
        feature_counts = {group: len(features) for group, features in self.feature_groups.items()}

        print("\nFeature Count per Group:")
        for group, count in feature_counts.items():
            print(f"- {group}: {count} features")

        # Calculate effective weights
        self.feature_weights = {}
        if normalize_by_count:
            # Target count can be adjusted, represents a 'typical' group size for normalization
            valid_counts = [c for c in feature_counts.values() if c > 0]
            if not valid_counts: # Handle case where all groups are empty (edge case)
                target_count = 1.0
                print("Warning: No features found in any group for normalization.")
            else:
                target_count = max(1, np.mean(valid_counts)) # Use mean non-zero count
            # target_count = 30 # Or a fixed value like 30? Experiment.
            print(f"Normalizing weights using target count: {target_count:.2f}")

            for group, features in self.feature_groups.items():
                count = feature_counts.get(group, 0) # Use count, default to 0
                if count == 0:
                    # If a group somehow has 0 features but exists, assign base weight directly
                    self.feature_weights[group] = final_base_weights[group]
                else:
                    # Formula: effective_weight = base_weight * sqrt(target_count / feature_count)
                    # Using sqrt helps moderate the effect of very large/small groups
                    norm_factor = np.sqrt(target_count / count)
                    self.feature_weights[group] = final_base_weights[group] * norm_factor
        else:
            print("Not normalizing weights by feature count.")
            # Use base weights directly if not normalizing, ensuring all groups are covered
            for group in self.feature_groups:
                 self.feature_weights[group] = final_base_weights[group]


        # Print effective weights applied
        print("\nEffective Feature Weights Applied:")
        for group, weight in self.feature_weights.items():
            base = final_base_weights.get(group, "N/A")
            count = feature_counts.get(group, 0)
            norm_status = "(Normalized)" if normalize_by_count and count > 0 else ""
            print(f"- {group:<20}: {weight:.4f} {norm_status} (Base: {base}, Count: {count})")

        return self.feature_weights

    def apply_weights(self, X):
        """
        Apply weights to the feature matrix based on identified groups.

        Args:
            X: Feature matrix (numpy array, assumes columns match self.feature_names_in_order).

        Returns:
            numpy array: Weighted feature matrix.
        """
        if self.feature_weights is None or self.feature_groups is None:
            raise ValueError("Feature weights or groups not set. Call identify_feature_groups() and set_feature_weights() first.")
        if self.feature_names_in_order is None:
             raise ValueError("Feature order not stored. Ensure fit() or load_data() was called.")
        if X.shape[1] != len(self.feature_names_in_order):
            raise ValueError(f"Input matrix X has {X.shape[1]} columns, but expected {len(self.feature_names_in_order)} based on stored feature names.")


        # Create a copy of the data to avoid modifying the original
        X_weighted = X.copy()

        # Create a mapping from feature index to weight
        # This relies on self.feature_names_in_order matching the columns of X
        feature_index_to_weight = np.ones(X.shape[1]) # Default weight is 1

        # --- Optimized Weight Application ---
        # Build index lookup map once
        feature_to_index = {name: idx for idx, name in enumerate(self.feature_names_in_order)}

        for group, features in self.feature_groups.items():
            weight = self.feature_weights.get(group, 1.0) # Get weight for the group
            # Find indices for all features in this group efficiently
            indices_for_group = [feature_to_index[feature_name] for feature_name in features if feature_name in feature_to_index]

            if len(indices_for_group) != len(features):
                 print(f"Warning: Mismatch finding indices for group '{group}'. Some features might not exist in the matrix.")

            # Apply weight to the identified indices
            if indices_for_group: # Check if list is not empty
                feature_index_to_weight[indices_for_group] = weight


        # Apply weights column-wise using broadcasting (efficient)
        X_weighted = X_weighted * feature_index_to_weight

        # print("Applied weights to feature matrix.") # Optional: confirmation message
        return X_weighted

    def handle_missing_values(self, df):
        """
        Handle missing (NaN) values in the feature DataFrame.

        Args:
            df: DataFrame with numeric features.

        Returns:
            DataFrame: Processed DataFrame with missing values handled (e.g., filled).
        """
        # Check specifically for NaN values
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            print(f"Found {total_nans} missing values (NaNs) in the feature data.")
            # Optional: Show columns with NaNs
            # print("Columns with missing values:")
            # print(nan_counts[nan_counts > 0])

            # Strategy: Fill NaN values with 0
            # This is a simple strategy; others like mean/median imputation could be considered
            # but might be complex with weighted features. Zero is often reasonable for
            # sparse features like TF-IDF or dummy variables.
            print("Filling missing values with 0...")
            df_filled = df.fillna(0)

            # Verify no NaNs remain
            if df_filled.isnull().sum().sum() > 0:
                # This shouldn't happen with fillna(0) unless there are strange data types
                print("ERROR: Still have NaN values after filling! Check data types.")
                # Attempt to force conversion again? Might hide underlying issues.
                # df_filled = df_filled.astype(float).fillna(0)
                raise ValueError("Failed to remove all NaN values from data")
            return df_filled
        else:
            # print("No missing values (NaNs) found in the feature data.")
            return df # Return original if no NaNs

    def fit(self, data):
        """
        Fit the weighted KNN model to the data.

        Args:
            data: DataFrame with numeric features or path to CSV file.

        Returns:
            self: The fitted model instance.
        """
        fit_start_time = time.time()
        try:
            # --- 1. Load and Prepare Data ---
            if isinstance(data, str):
                feature_df = self.load_data(data) # load_data handles numeric extraction, NaNs etc.
            elif isinstance(data, pd.DataFrame):
                # Assume data might be the full DataFrame, need to extract features
                print("Input is DataFrame, preparing features...")
                # Re-extract IDs and titles if needed and not already set
                if self.product_ids is None and 'product_id' in data.columns:
                     self.product_ids = data['product_id'].copy()
                     if self.product_ids.isnull().any(): # Handle NaNs in IDs if loaded from DataFrame
                          nan_count = self.product_ids.isnull().sum()
                          print(f"Warning: Found {nan_count} missing product IDs in input DataFrame. Filling with index.")
                          nan_mask = self.product_ids.isnull()
                          self.product_ids.loc[nan_mask] = data.index[nan_mask].map(lambda x: f"missing_id_{x}")

                if self.product_titles is None and 'product_title' in data.columns:
                     self.product_titles = data['product_title'].copy()

                # Mimic numeric feature extraction from load_data
                non_feature_cols = ['product_id', 'product_title', 'handle', 'description',
                                    'tags', 'collections', 'variants', 'metafields', 'metafields_config',
                                    'created_at']
                potential_feature_cols = [col for col in data.columns if col not in non_feature_cols]
                numeric_cols = []
                for col in potential_feature_cols:
                     if pd.api.types.is_numeric_dtype(data[col]):
                          if not np.isinf(data[col]).any(): numeric_cols.append(col)
                     else:
                          try: # Attempt conversion
                               converted = pd.to_numeric(data[col], errors='coerce')
                               if converted.notnull().any() and not np.isinf(converted).any(): numeric_cols.append(col)
                          except: pass # Ignore errors here

                if not numeric_cols: raise ValueError("No usable numeric features found in the provided DataFrame.")
                feature_df = data[numeric_cols].copy()
                self.feature_names_in_order = feature_df.columns.tolist() # Store feature order
                feature_df = self.handle_missing_values(feature_df) # Handle NaNs in the extracted features
            else:
                raise TypeError("Input data must be a pandas DataFrame or a path to a CSV file.")

            if feature_df.empty:
                 raise ValueError("Feature DataFrame is empty after loading and preparation.")
            if self.product_ids is None: # Final check after potential loading from DataFrame
                 print("Warning: Product IDs are still missing after data loading. Using index.")
                 self.product_ids = pd.Series(feature_df.index).astype(str)


            # --- 2. Identify Groups & Set Weights (if not already done externally) ---
            if self.feature_groups is None:
                print("Identifying feature groups...")
                self.identify_feature_groups(feature_df)

            if self.feature_weights is None:
                print("Setting feature weights (using defaults)...")
                self.set_feature_weights() # Uses defaults if no base_weights provided


            # --- 3. Check Data Quality (Inf values - should be caught earlier, but safe check) ---
            if np.isinf(feature_df.values).any():
                 inf_counts = np.isinf(feature_df.values).sum()
                 print(f"Warning: Found {inf_counts} infinite values before scaling. Replacing with large finite values (1e9 / -1e9).")
                 feature_df.replace([np.inf, -np.inf], [1e9, -1e9], inplace=True)


            # --- 4. Standardize Data ---
            print("Standardizing data using StandardScaler...")
            # Fit the scaler and transform the data
            X_scaled = self.scaler.fit_transform(feature_df)
            # Check for NaNs introduced by scaler (can happen with zero variance columns)
            if np.isnan(X_scaled).any():
                nan_count_scaled = np.isnan(X_scaled).sum()
                print(f"Warning: {nan_count_scaled} NaNs found after scaling! This might be due to columns with zero variance.")
                nan_cols_scaled = feature_df.columns[np.isnan(X_scaled).any(axis=0)].tolist()
                print(f"Columns potentially causing issues: {nan_cols_scaled}")
                print("Attempting to fill NaNs post-scaling with 0...")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0) # Fill NaNs


            # --- 5. Apply Weights ---
            print("Applying feature weights...")
            self.X_weighted = self.apply_weights(X_scaled)

            # Verify no NaNs/Infs in weighted matrix AFTER weighting
            if np.isnan(self.X_weighted).any():
                nan_count_weighted = np.isnan(self.X_weighted).sum()
                print(f"ERROR: {nan_count_weighted} NaN values found in weighted feature matrix!")
                print("Attempting to fill NaNs in weighted matrix with 0...")
                self.X_weighted = np.nan_to_num(self.X_weighted, nan=0.0)
                # Consider raising error if this happens, indicates deeper issue
            if np.isinf(self.X_weighted).any():
                 inf_count_weighted = np.isinf(self.X_weighted).sum()
                 print(f"ERROR: {inf_count_weighted} Infinite values found in weighted feature matrix!")
                 print("Attempting to replace Infs in weighted matrix with large values...")
                 self.X_weighted = np.nan_to_num(self.X_weighted, posinf=1e9, neginf=-1e9)


            # --- 6. Fit KNN Model ---
            # Ensure n_neighbors is not larger than the number of samples
            n_samples = self.X_weighted.shape[0]
            # k must be > 0 and <= n_samples
            effective_n_neighbors = min(max(1, self.n_neighbors), n_samples -1) # Ensure k >= 1 and k < n_samples

            if effective_n_neighbors <= 0: # Should be caught by max(1,..) but safety
                 raise ValueError(f"Cannot fit KNN: Number of samples ({n_samples}) is too small for any neighbors.")

            if effective_n_neighbors < self.n_neighbors:
                 print(f"Warning: n_neighbors requested ({self.n_neighbors}) is >= number of samples ({n_samples}). Setting n_neighbors to {effective_n_neighbors}.")

            # The NearestNeighbors model requires n_neighbors to be k+1 (including self)
            knn_k = effective_n_neighbors + 1
            if knn_k > n_samples:
                 print(f"Adjusting KNN internal k from {knn_k} to {n_samples} (max possible).")
                 knn_k = n_samples


            print(f"Fitting KNN model with internal k={knn_k} (requesting {effective_n_neighbors} neighbors)...")
            self.knn_model = NearestNeighbors(
                n_neighbors=knn_k, # Use the adjusted k
                algorithm=self.algorithm,
                metric=self.metric,
                n_jobs=-1 # Use all available cores
            )
            self.knn_model.fit(self.X_weighted)


            # --- 7. Compute Neighbors and Similarities ---
            print("Finding nearest neighbors for all products...")
            # .kneighbors returns distances and indices of neighbors for each point
            self.distances, self.indices = self.knn_model.kneighbors(self.X_weighted)
            print("Nearest neighbors computation complete.")

            # --- 8. Build Recommendation Matrix ---
            print("Building recommendation matrix (product_id -> [(rec_id, similarity), ...])...")
            self.recommendation_matrix = {}
            products_with_issues = 0

            # Use the stored product IDs which should align with the rows of X_weighted
            if len(self.product_ids) != self.X_weighted.shape[0]:
                raise ValueError(f"Mismatch between number of product IDs ({len(self.product_ids)}) and data rows ({self.X_weighted.shape[0]}).")
            source_product_ids = self.product_ids.tolist()


            for i in range(len(self.indices)):
                source_product_id = source_product_ids[i]

                # Indices of neighbors (row i in self.indices)
                # Distances of neighbors (row i in self.distances)
                neighbor_indices = self.indices[i]
                neighbor_distances = self.distances[i]

                product_recs = []
                # Iterate through neighbors, starting from index 1 to skip self
                # The number of actual neighbors returned might be less than knn_k if k > n_samples
                num_neighbors_found = len(neighbor_indices)
                for j in range(1, num_neighbors_found):
                    neighbor_idx = neighbor_indices[j]
                    dist = neighbor_distances[j]

                    # Get the recommended product ID using the neighbor's index
                    try:
                        recommended_product_id = source_product_ids[neighbor_idx]
                    except IndexError:
                         print(f"Error: Neighbor index {neighbor_idx} out of bounds for product IDs list (size {len(source_product_ids)}). Skipping this neighbor for product {source_product_id}.")
                         continue


                    # Important: Ensure we are not recommending the product itself
                    # This can happen if duplicates exist or features are identical
                    if recommended_product_id == source_product_id:
                        # print(f"Warning: Self-recommendation detected for {source_product_id} at rank {j}. Skipping.")
                        continue # Skip this neighbor

                    # Convert distance to similarity score
                    # Using 1 / (1 + distance) for non-negative distances like Euclidean
                    # Add a small epsilon to avoid division by zero if distance is exactly 0
                    similarity = 1.0 / (1.0 + dist + 1e-9)

                    # Ensure similarity is within [0, 1] range (can be slightly > 1 if dist is negative? Should not happen with std metrics)
                    similarity = max(0.0, min(1.0, similarity))


                    product_recs.append((recommended_product_id, similarity))

                    # Stop if we have enough recommendations (based on the adjusted effective_n_neighbors)
                    if len(product_recs) >= effective_n_neighbors:
                         break

                # Store the recommendations (already sorted by distance/similarity by kneighbors)
                self.recommendation_matrix[source_product_id] = product_recs

                # Check if enough recommendations were generated
                if len(self.recommendation_matrix[source_product_id]) < effective_n_neighbors:
                     # This might happen due to self-filtering or reaching end of neighbors
                     # print(f"Note: Generated {len(self.recommendation_matrix[source_product_id])} recommendations for {source_product_id} (less than requested {effective_n_neighbors}).")
                     pass # Not necessarily an error


            fit_end_time = time.time()
            print(f"\nSuccessfully fitted KNN model to {len(feature_df)} products in {fit_end_time - fit_start_time:.2f} seconds.")
            print(f"Each product has up to {effective_n_neighbors} recommendations stored.")

            return self

        except Exception as e:
            print(f"\n--- ERROR during KNN fitting ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("\nTraceback:")
            traceback.print_exc()
            print("----------------------------")
            # Optionally re-raise or return a specific failure indicator
            # raise e # Re-raise the exception to stop execution
            print("KNN fitting failed. Returning the model instance in its current state (likely unusable).")
            return self # Return self, but it might be in an inconsistent state

    def get_recommendations(self, product_id, n_recommendations=None):
        """
        Get pre-computed recommendations for a specific product ID.

        Args:
            product_id: ID of the product to get recommendations for.
            n_recommendations: Number of recommendations to return (default: self.n_neighbors).

        Returns:
            list: List of (recommended_product_id, similarity_score) tuples,
                  or an empty list if the product_id is not found or has no recommendations.
        """
        if self.recommendation_matrix is None:
            # Check if the model wasn't fitted properly
            if self.knn_model is None:
                 print(f"Error: Model not fitted yet for product ID {product_id}. Call fit() first.")
                 return []
            else:
                 # Model fitted, but matrix not built (shouldn't happen with current fit logic)
                 print(f"Error: Recommendation matrix not available for product ID {product_id}, although model seems fitted.")
                 return []


        if product_id not in self.recommendation_matrix:
            # Find potential matches if ID format differs (e.g., with/without prefix)
            potential_matches = [pid for pid in self.recommendation_matrix.keys() if str(product_id) in str(pid)]
            if potential_matches:
                 print(f"Warning: Product ID '{product_id}' not found exactly. Potential matches: {potential_matches}. Using first match: {potential_matches[0]}")
                 product_id = potential_matches[0] # Try the first potential match
            else:
                 print(f"Warning: Product ID '{product_id}' not found in the recommendation matrix.")
                 return []


        # Use the originally requested n_neighbors if n_recommendations is not specified
        n_to_return = self.n_neighbors if n_recommendations is None else n_recommendations

        if n_to_return <= 0:
             return [] # Requesting zero or negative recommendations

        # Get precomputed recommendations for the product
        # The list should already be sorted by similarity (desc) from the fit step
        recommendations = self.recommendation_matrix[product_id]

        # Although self-recs *should* be filtered during fit, double-check here just in case
        filtered_recommendations = [(rec_id, sim) for rec_id, sim in recommendations if rec_id != product_id]

        # Return the top N requested recommendations
        return filtered_recommendations[:n_to_return]

    def calculate_average_similarity(self):
        """
        Calculate the average similarity score across all generated recommendations.

        Returns:
            float: Average similarity score, or 0.0 if no recommendations exist.
        """
        if self.recommendation_matrix is None:
            print("Warning: Cannot calculate average similarity, model not fitted or matrix not built.")
            return 0.0

        all_similarities = []
        for product_id, recommendations in self.recommendation_matrix.items():
            # Extract similarity scores (the second element of each tuple)
            similarities = [sim for _, sim in recommendations]
            all_similarities.extend(similarities)

        if not all_similarities:
            print("Warning: No similarity scores found in the recommendation matrix.")
            return 0.0

        # Calculate and return the average
        avg_similarity = np.mean(all_similarities)
        # print(f"Calculated average similarity: {avg_similarity:.4f}") # Optional print
        return avg_similarity

    def visualize_feature_importance(self, output_file="feature_importance.png"):
        """
        Visualize feature importance based on the *effective* feature weights assigned per group.
        Note: This shows group-level importance as assigned, not model-derived importance.

        Args:
            output_file: Path to save the visualization.

        Returns:
            None
        """
        if self.feature_weights is None or self.feature_groups is None:
            print("Cannot visualize feature importance: Weights or groups not set.")
            return
        if not self.feature_names_in_order:
             print("Cannot visualize feature importance: Feature names not available.")
             return


        try:
            # Create importance scores based on the *effective* weight of the group each feature belongs to
            feature_importance_map = {}
            for group, features in self.feature_groups.items():
                weight = self.feature_weights.get(group, 1.0)
                for feature_name in features:
                    feature_importance_map[feature_name] = weight

            # Convert map to a Series for easier sorting and plotting
            if not feature_importance_map:
                 print("No feature importance data to visualize.")
                 return
            importance_series = pd.Series(feature_importance_map)

            # Ensure the Series is sorted by the original feature order for consistency? Not needed for bar plot.
            # Sort by importance value for plotting
            importance_series = importance_series.sort_values(ascending=False)


            # Get top N features to display (e.g., top 30 or all if fewer)
            num_features_to_show = min(50, len(importance_series)) # Show more features?
            top_features = importance_series.head(num_features_to_show)

            # Create a horizontal bar chart
            plt.figure(figsize=(14, max(8, num_features_to_show * 0.3))) # Adjust size
            y_pos = np.arange(len(top_features))

            # Use the feature names directly from the Series index
            display_names = top_features.index.tolist()

            # Shorten long names for display (Improved)
            short_display_names = []
            prefix_map = {
                'description_': 'Desc: ', 'title_': 'Title: ', 'handle_': 'Handle: ', # Added title/handle
                'metafield_': 'Meta: ', 'tag_': 'Tag: ', 'product_type_': 'Type: ',
                'collections_': 'Coll: ', 'vendor_': 'Vend: ', 'variant_size_': 'Size: ',
                'variant_color_': 'Color: ','variant_other_': 'VarOther: ',
                'created_': 'Time: ', 'release_quarter_': 'Quarter: ', 'price_': 'Price: '
            }
            max_len = 45 # Max length for labels

            for name in display_names:
                 display_name = name
                 for prefix, short_prefix in prefix_map.items():
                      if name.startswith(prefix):
                           display_name = short_prefix + name[len(prefix):]
                           break # Stop after first match

                 if len(display_name) > max_len: # Apply max length
                      display_name = display_name[:max_len-3] + "..."
                 short_display_names.append(display_name)


            # Plotting
            plt.barh(y_pos, top_features.values, align='center', color=sns.color_palette("viridis", len(top_features)))
            plt.yticks(y_pos, short_display_names, fontsize=9) # Adjust fontsize if needed
            plt.gca().invert_yaxis() # Display highest importance at the top
            plt.xlabel('Assigned Effective Feature Weight')
            plt.title(f'Top {num_features_to_show} Features by Assigned Group Weight')
            plt.tight_layout(pad=1.5) # Adjust layout padding
            plt.savefig(output_file, dpi=150) # Increase resolution?
            plt.close() # Close the plot to free memory

            print(f"Feature importance visualization saved to {output_file}")

        except Exception as e:
            print(f"Error during feature importance visualization: {type(e).__name__} - {str(e)}")
            print("Skipping feature importance visualization.")
            # Optionally print traceback for debugging
            # import traceback
            # traceback.print_exc()

    def visualize_tsne(self, output_file="tsne_visualization.png", sample_size=1000, perplexity=30):
        """
        Visualize data using t-SNE dimensionality reduction on the *weighted* features.

        Args:
            output_file: Path to save the visualization.
            sample_size: Max number of points to use for t-SNE (it's slow).
            perplexity: t-SNE perplexity parameter.
        Returns:
            None
        """
        if self.X_weighted is None:
            print("Cannot visualize t-SNE: Weighted feature matrix (X_weighted) not available. Run fit() first.")
            return

        try:
            X_to_plot = self.X_weighted # Use the already weighted matrix

            # --- Data Quality Check (already done in fit, but double-check) ---
            if np.isnan(X_to_plot).any() or np.isinf(X_to_plot).any():
                print("Warning: NaN or Inf values detected in weighted matrix for t-SNE. Attempting to clean...")
                X_to_plot = np.nan_to_num(X_to_plot, nan=0.0, posinf=1e9, neginf=-1e9)
                if np.isnan(X_to_plot).any() or np.isinf(X_to_plot).any():
                     print("Error: Could not clean NaN/Inf values for t-SNE. Skipping visualization.")
                     return


            # --- Sampling ---
            num_samples_total = X_to_plot.shape[0]
            indices_original = np.arange(num_samples_total) # Keep track of original indices

            if num_samples_total > sample_size:
                print(f"Sampling {sample_size} points out of {num_samples_total} for t-SNE visualization...")
                # Use random choice for sampling
                indices_sampled = np.random.choice(num_samples_total, sample_size, replace=False)
                X_sample = X_to_plot[indices_sampled]
            else:
                print(f"Using all {num_samples_total} points for t-SNE visualization.")
                indices_sampled = indices_original # Use all original indices
                X_sample = X_to_plot


            if X_sample.shape[0] <= 1:
                 print("Skipping t-SNE: Not enough samples after potential sampling (<= 1).")
                 return


            # --- Adjust t-SNE Perplexity ---
            # Perplexity should be less than the number of samples
            effective_perplexity = min(perplexity, X_sample.shape[0] - 1)
            if effective_perplexity <= 0:
                print(f"Warning: Sample size ({X_sample.shape[0]}) too small for perplexity. Setting perplexity to min(5, n_samples-1).")
                effective_perplexity = min(5, max(1, X_sample.shape[0] - 1)) # Ensure perplexity is at least 1

            if effective_perplexity != perplexity:
                 print(f"Adjusted t-SNE perplexity from {perplexity} to {effective_perplexity}.")


            # --- Apply t-SNE ---
            print(f"Applying t-SNE (perplexity={effective_perplexity}, samples={X_sample.shape[0]})... (this may take a moment)")
            tsne_start_time = time.time()
            tsne = TSNE(n_components=2, random_state=self.random_state,
                        perplexity=effective_perplexity, n_iter=300, # Lower iterations for speed? 300 is often okay.
                        init='pca', # PCA initialization is often faster and stabler
                        learning_rate='auto', # Recommended in recent sklearn
                        n_jobs=-1) # Use all cores
            X_tsne = tsne.fit_transform(X_sample)
            tsne_end_time = time.time()
            print(f"t-SNE computation finished in {tsne_end_time - tsne_start_time:.2f} seconds.")


            # --- Determine Point Colors (Optional - Example using primary feature group weight) ---
            # This strategy colors points based on the weight of their dominant feature group.
            point_colors = np.zeros(len(X_sample)) # Default color
            group_to_color_map = {} # Map group names to numeric values for coloring
            color_idx = 0
            if self.feature_groups and self.feature_weights and self.feature_names_in_order:
                try:
                    # Find the group with the highest weight for each feature
                    feature_to_group = {}
                    for group, features in self.feature_groups.items():
                         for feature in features: feature_to_group[feature] = group

                    # Determine the dominant group for each *sampled* point based on feature weights
                    sampled_feature_indices = [self.feature_names_in_order.index(f) for f in self.feature_names_in_order] # Get indices once
                    dominant_group_indices = np.argmax(X_sample[:, sampled_feature_indices], axis=1) # Find index of max value per row (approx dominant feature)

                    for i, sample_idx in enumerate(indices_sampled): # Iterate through sampled points
                         dominant_feature_idx_in_sample = dominant_group_indices[i]
                         dominant_feature_name = self.feature_names_in_order[dominant_feature_idx_in_sample]
                         dominant_group = feature_to_group.get(dominant_feature_name, 'other_numeric') # Fallback group

                         if dominant_group not in group_to_color_map:
                              group_to_color_map[dominant_group] = color_idx
                              color_idx += 1
                         point_colors[i] = group_to_color_map[dominant_group]

                except Exception as e:
                     print(f"Warning: Error determining point colors based on feature groups ({e}). Using default color.")


            # --- Plotting ---
            plt.figure(figsize=(12, 10))
            n_colors = len(np.unique(point_colors))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors,
                                  cmap=plt.get_cmap('tab20', max(1, n_colors)), # Use tab20 or similar, ensure at least 1 color
                                  alpha=0.6, s=15) # Adjust alpha/size
            # Optional: Add legend for colors if using group-based coloring
            if group_to_color_map:
                 handles = [plt.Line2D([0], [0], marker='o', color='w', label=group,
                                     markerfacecolor=plt.get_cmap('tab20', max(1, n_colors))(val / max(1, n_colors-1) if n_colors > 1 else 0.0)) # Normalize color index
                            for group, val in group_to_color_map.items()]
                 plt.legend(handles=handles, title="Dominant Feature Group", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')


            plt.title(f't-SNE Visualization of {len(X_sample)} Products (Weighted Features)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.xticks([]) # Hide axis ticks for cleaner look
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(output_file, dpi=150) # Save with higher DPI
            plt.close() # Close the plot

            print(f"t-SNE visualization saved to {output_file}")

        except Exception as e:
            print(f"Error during t-SNE visualization: {type(e).__name__} - {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping t-SNE visualization.")

    def visualize_similarity_distribution(self, output_file="similarity_distribution.png"):
        """
        Visualize the distribution of the calculated similarity scores between products
        and their nearest neighbors (excluding self).

        Args:
            output_file: Path to save the visualization.

        Returns:
            None
        """
        if self.recommendation_matrix is None:
            print("Cannot visualize similarity distribution: Recommendation matrix not built. Run fit() first.")
            return

        try:
            # Extract all similarity scores from the recommendation matrix
            all_similarities = []
            for recs in self.recommendation_matrix.values():
                all_similarities.extend([sim for _, sim in recs])

            if not all_similarities:
                print("No similarity scores found to visualize.")
                return

            flat_similarities = np.array(all_similarities)
            # Filter out potential non-finite values just in case
            flat_similarities = flat_similarities[np.isfinite(flat_similarities)]

            if len(flat_similarities) == 0:
                 print("No finite similarity scores found after filtering.")
                 return

            # --- Plotting Histogram ---
            plt.figure(figsize=(10, 6))
            # Use auto bins or specify a reasonable number
            plt.hist(flat_similarities, bins='auto', alpha=0.75, color='skyblue', edgecolor='black')
            plt.xlabel('Similarity Score (1 / (1 + distance))')
            plt.ylabel('Frequency')
            plt.title('Distribution of Product Similarity Scores (Neighbor vs Source)')
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Add mean/median lines for context
            mean_sim = np.mean(flat_similarities)
            median_sim = np.median(flat_similarities)
            plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_sim:.3f}')
            plt.axvline(median_sim, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_sim:.3f}')
            plt.legend()
            plt.xlim(0, 1) # Ensure x-axis is within [0, 1] range for similarity

            plt.tight_layout()
            plt.savefig(output_file, dpi=120)
            plt.close() # Close the plot

            print(f"Similarity distribution visualization saved to {output_file}")

            # --- Print Statistics ---
            print("\nSimilarity Score Statistics:")
            print(f"- Count:  {len(flat_similarities)}")
            print(f"- Mean:   {mean_sim:.4f}")
            print(f"- Median: {median_sim:.4f}")
            print(f"- Std Dev:{np.std(flat_similarities):.4f}")
            print(f"- Min:    {np.min(flat_similarities):.4f}")
            print(f"- Max:    {np.max(flat_similarities):.4f}")
            # Percentiles can also be informative
            print(f"- 25th Pctl: {np.percentile(flat_similarities, 25):.4f}")
            print(f"- 75th Pctl: {np.percentile(flat_similarities, 75):.4f}")


        except Exception as e:
            print(f"Error during similarity distribution visualization: {type(e).__name__} - {str(e)}")
            print("Skipping similarity distribution visualization.")

    def save_model(self, output_dir="models"):
        """
        Save the essential parts of the fitted model to disk.

        Args:
            output_dir: Directory to save the model files.

        Returns:
            None
        """
        if self.knn_model is None or self.scaler is None:
             print("Warning: Model is not fully fitted or essential components (KNN, scaler) are missing. Saving incomplete model.")
             # Decide whether to proceed or raise an error

        try:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving model components to directory: {output_dir}")

            # --- Save Components using joblib (good for scikit-learn objects) ---
            if self.knn_model:
                joblib.dump(self.knn_model, os.path.join(output_dir, "knn_model.pkl"))
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
            if self.recommendation_matrix:
                # Recommendation matrix can be large, consider compression
                joblib.dump(self.recommendation_matrix, os.path.join(output_dir, "recommendation_matrix.pkl.z"), compress=3) # Add compression
            # Save distances/indices? Usually large and derivable from model+data, maybe optional.
            # if self.distances is not None: joblib.dump(self.distances, os.path.join(output_dir, "distances.pkl.z"), compress=3)
            # if self.indices is not None: joblib.dump(self.indices, os.path.join(output_dir, "indices.pkl.z"), compress=3)


            # --- Save Metadata and Parameters (use JSON or pickle) ---
            model_params = {
                'feature_weights': self.feature_weights,
                'feature_groups': self.feature_groups,
                'feature_names_in_order': self.feature_names_in_order, # CRUCIAL: Save feature order!
                'n_neighbors': self.n_neighbors,
                'algorithm': self.algorithm,
                'metric': self.metric,
                'random_state': self.random_state,
                # Store product IDs and titles if they were loaded (can be large)
                'has_product_ids': self.product_ids is not None,
                # 'has_product_titles': self.product_titles is not None, # Titles less critical for model function
            }
            # Save parameters using pickle (easier for complex structures like dicts of lists)
            joblib.dump(model_params, os.path.join(output_dir, "model_params.pkl"))

            # Optionally save IDs if they exist (can be large) - Required for using the loaded model
            if self.product_ids is not None:
                 joblib.dump(self.product_ids, os.path.join(output_dir, "product_ids.pkl.z"), compress=3)
            # Save titles separately if needed, less critical for model reload functionality
            # if self.product_titles is not None:
            #      joblib.dump(self.product_titles, os.path.join(output_dir, "product_titles.pkl.z"), compress=3)


            print(f"Model components saved successfully.")

        except Exception as e:
            print(f"Error saving model to {output_dir}: {type(e).__name__} - {str(e)}")
            import traceback
            traceback.print_exc()

    def load_model(self, input_dir="models"):
        """
        Load a previously saved model from disk.

        Args:
            input_dir: Directory containing the saved model files.

        Returns:
            self: The loaded model instance.
        """
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Model directory not found or is not a directory: {input_dir}")

        print(f"Loading model components from directory: {input_dir}")
        load_successful = True

        try:
            # --- Load Components ---
            knn_path = os.path.join(input_dir, "knn_model.pkl")
            if os.path.exists(knn_path): self.knn_model = joblib.load(knn_path)
            else: print(f"Warning: KNN model file not found at {knn_path}"); load_successful = False

            scaler_path = os.path.join(input_dir, "scaler.pkl")
            if os.path.exists(scaler_path): self.scaler = joblib.load(scaler_path)
            else: print(f"Warning: Scaler file not found at {scaler_path}"); load_successful = False

            # Load recommendation matrix (check compressed name first)
            rec_path_z = os.path.join(input_dir, "recommendation_matrix.pkl.z")
            rec_path = os.path.join(input_dir, "recommendation_matrix.pkl")
            if os.path.exists(rec_path_z): self.recommendation_matrix = joblib.load(rec_path_z)
            elif os.path.exists(rec_path): self.recommendation_matrix = joblib.load(rec_path)
            else:
                print(f"Warning: Recommendation matrix file not found. Recommendations need recalculation if required.")
                self.recommendation_matrix = None # Explicitly set to None

            # --- Load Parameters ---
            params_path = os.path.join(input_dir, "model_params.pkl")
            if os.path.exists(params_path):
                params = joblib.load(params_path)
                self.feature_weights = params.get('feature_weights')
                self.feature_groups = params.get('feature_groups')
                self.feature_names_in_order = params.get('feature_names_in_order') # Load feature order!
                self.n_neighbors = params.get('n_neighbors', self.n_neighbors)
                self.algorithm = params.get('algorithm', self.algorithm)
                self.metric = params.get('metric', self.metric)
                self.random_state = params.get('random_state', self.random_state)

                # --- Crucial Check: Feature Order ---
                if self.feature_names_in_order is None:
                     print("CRITICAL WARNING: Loaded model parameters do not contain 'feature_names_in_order'. Model may be unusable if feature structure has changed.")
                     load_successful = False # Consider this a critical failure

                # Load product IDs if indicated (Required)
                if params.get('has_product_ids'):
                     ids_path_z = os.path.join(input_dir, "product_ids.pkl.z")
                     ids_path = os.path.join(input_dir, "product_ids.pkl")
                     if os.path.exists(ids_path_z): self.product_ids = joblib.load(ids_path_z)
                     elif os.path.exists(ids_path): self.product_ids = joblib.load(ids_path)
                     else:
                          print("CRITICAL WARNING: Model params indicate product IDs should exist, but file not found. Loaded model cannot generate recommendations by ID.")
                          self.product_ids = None
                          load_successful = False # Cannot function without IDs
                else:
                     # If params say no IDs, but we need them...
                     print("Warning: Model parameters indicate no product IDs were saved. Model cannot generate recommendations by ID.")
                     self.product_ids = None
                     # Don't necessarily fail load, but warn user

            else:
                print(f"CRITICAL WARNING: Model parameters file not found at {params_path}. Model metadata is missing, loaded model is likely unusable.")
                load_successful = False # Metadata is essential

            if load_successful and self.knn_model and self.scaler and self.product_ids is not None and self.feature_names_in_order is not None:
                print("Model components loaded successfully.")
            else:
                print("Model loading finished with warnings or missing critical components. Model might be incomplete or unusable.")

            return self

        except Exception as e:
            print(f"Error loading model from {input_dir}: {type(e).__name__} - {str(e)}")
            import traceback
            traceback.print_exc()
            print("Returning model instance in potentially inconsistent state due to loading error.")
            return self

    def export_recommendations(self, output_file="all_recommendations.csv", n_recommendations=None,
                             products_df=None, similar_products_df=None):
        """
        Export all product recommendations to a CSV file, optionally filtering
        out too-similar products (variants).

        Args:
            output_file: Path to save the recommendations CSV.
            n_recommendations: Max number of recommendations per product (default: self.n_neighbors).
            products_df: DataFrame with original product data (needed for titles, optional).
            similar_products_df: DataFrame with pairs of too-similar products to exclude (optional).
                                 Expected columns like 'product1_id', 'product2_id'.

        Returns:
            pd.DataFrame: DataFrame containing the exported recommendations, or empty DataFrame on failure.
        """
        export_start_time = time.time()
        if self.recommendation_matrix is None:
            print("Error: Cannot export recommendations, matrix not built. Run fit() first.")
            return pd.DataFrame() # Return empty DataFrame

        n_to_export = self.n_neighbors if n_recommendations is None else n_recommendations
        if n_to_export <= 0:
             print("Warning: n_recommendations requested is <= 0. Exporting empty file.")
             pd.DataFrame(columns=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']).to_csv(output_file, index=False)
             return pd.DataFrame()

        # Prepare data for export
        rows = []
        total_variant_filtered_count = 0 # Track how many were filtered as variants
        total_self_recs_filtered = 0 # Track self-recs removed

        # --- Prepare Variant Filtering Lookup (More Efficient) ---
        variant_exclude_map = {} # {product_id: set_of_variant_ids_to_exclude}
        use_variant_filter = False
        if similar_products_df is not None and not similar_products_df.empty:
            col1_id, col2_id = None, None
            # Determine columns defensively
            if 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns:
                col1_id, col2_id = 'product1_id', 'product2_id'
            elif 'product1' in similar_products_df.columns and 'product2' in similar_products_df.columns: # Fallback
                col1_id, col2_id = 'product1', 'product2'

            if col1_id and col2_id:
                use_variant_filter = True
                print("Building variant exclusion map for filtering...")
                for _, row in similar_products_df.iterrows():
                    p1 = row[col1_id]
                    p2 = row[col2_id]
                    if pd.notna(p1) and pd.notna(p2):
                        # Add exclusion in both directions
                        if p1 not in variant_exclude_map: variant_exclude_map[p1] = set()
                        if p2 not in variant_exclude_map: variant_exclude_map[p2] = set()
                        variant_exclude_map[p1].add(p2)
                        variant_exclude_map[p2].add(p1)
                print(f"Variant map built for {len(variant_exclude_map)} products.")
            else:
                print("Warning: Could not find required columns in similar_products_df. Variant filtering disabled.")


        # --- Iterate through each product and its recommendations ---
        print(f"Exporting top {n_to_export} recommendations for {len(self.recommendation_matrix)} products...")
        processed_count = 0
        log_interval = max(1, len(self.recommendation_matrix) // 10) # Log progress ~10 times

        for source_product_id, recommendations in self.recommendation_matrix.items():
            processed_count += 1
            # Progress indicator
            if processed_count % log_interval == 0 or processed_count == 1 or processed_count == len(self.recommendation_matrix):
                 print(f"Processing product {processed_count}/{len(self.recommendation_matrix)}: {source_product_id}")

            # 1. Initial Filter: Remove self-recommendations (should be redundant)
            current_recs = [(rec_id, sim) for rec_id, sim in recommendations if rec_id != source_product_id]
            self_filtered = len(recommendations) - len(current_recs)
            if self_filtered > 0: total_self_recs_filtered += self_filtered

            # 2. Filter Variants (if enabled and applicable)
            variants_filtered_this_product = 0
            if use_variant_filter and source_product_id in variant_exclude_map:
                exclude_ids = variant_exclude_map[source_product_id]
                count_before = len(current_recs)
                current_recs = [(rec_id, sim) for rec_id, sim in current_recs if rec_id not in exclude_ids]
                variants_filtered_this_product = count_before - len(current_recs)
                total_variant_filtered_count += variants_filtered_this_product


            # 3. Take Top N recommendations *after* filtering
            final_recs_for_product = current_recs[:n_to_export]

            # 4. Add to export rows with rank
            for rank, (rec_id, similarity) in enumerate(final_recs_for_product):
                # Final check (should be unnecessary but safe)
                if rec_id != source_product_id:
                    rows.append({
                        'source_product_id': source_product_id,
                        'recommended_product_id': rec_id,
                        'rank': rank + 1,
                        'similarity_score': similarity
                    })


        # --- Create DataFrame and Save ---
        recommendations_df = pd.DataFrame(rows)

        if not recommendations_df.empty:
            # Add product titles if possible (using a potentially large products_df)
            if products_df is not None and 'product_id' in products_df.columns and 'product_title' in products_df.columns:
                 print("Adding product titles to exported recommendations...")
                 try:
                      # Ensure product_id is string type in both frames for merging/mapping
                      products_df['product_id_str'] = products_df['product_id'].astype(str)
                      title_map = products_df.set_index('product_id_str')['product_title'].to_dict()

                      recommendations_df['source_product_id_str'] = recommendations_df['source_product_id'].astype(str)
                      recommendations_df['recommended_product_id_str'] = recommendations_df['recommended_product_id'].astype(str)

                      recommendations_df['source_product_title'] = recommendations_df['source_product_id_str'].map(title_map).fillna('Title Not Found')
                      recommendations_df['recommended_product_title'] = recommendations_df['recommended_product_id_str'].map(title_map).fillna('Title Not Found')

                      # Drop temporary string columns
                      recommendations_df = recommendations_df.drop(columns=['product_id_str', 'source_product_id_str', 'recommended_product_id_str'], errors='ignore')

                      # Reorder columns
                      cols_order = ['source_product_id', 'source_product_title', 'recommended_product_id', 'recommended_product_title', 'rank', 'similarity_score']
                      recommendations_df = recommendations_df[[col for col in cols_order if col in recommendations_df.columns]]
                 except Exception as title_err:
                      print(f"Warning: Error adding titles: {title_err}. Proceeding without titles.")
                      # Ensure columns don't exist if mapping failed partially
                      recommendations_df = recommendations_df.drop(columns=['source_product_title', 'recommended_product_title'], errors='ignore')

            # Final check for self-recommendations (should never happen)
            self_recs_final = recommendations_df[recommendations_df['source_product_id'] == recommendations_df['recommended_product_id']]
            if not self_recs_final.empty:
                print(f"\nCRITICAL ERROR: Found {len(self_recs_final)} self-recommendations in the final export DataFrame! Removing...")
                recommendations_df = recommendations_df[recommendations_df['source_product_id'] != recommendations_df['recommended_product_id']]

            try:
                recommendations_df.to_csv(output_file, index=False)
                export_end_time = time.time()
                print(f"\nRecommendations exported successfully to {output_file} in {export_end_time - export_start_time:.2f} seconds.")
                print(f"Exported {len(recommendations_df)} recommendation pairs.")
                if use_variant_filter and total_variant_filtered_count > 0:
                    print(f"Total recommendations filtered out as similar variants: {total_variant_filtered_count}")
                if total_self_recs_filtered > 0:
                     print(f"Warning: Total self-recommendations filtered during export: {total_self_recs_filtered} (Might indicate duplicate products/features).")


            except Exception as e:
                print(f"\nError saving recommendations DataFrame to {output_file}: {e}")
                return recommendations_df # Return the DataFrame even if saving failed

        else:
            print("\nWARNING: No valid recommendations generated after filtering. Export file will be empty or not created.")
            try: # Save empty file with headers
                 pd.DataFrame(columns=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']).to_csv(output_file, index=False)
            except Exception as e: print(f"Error saving empty recommendations file: {e}")


        return recommendations_df


# --- WeightOptimizer Class (Updated Defaults) ---
class WeightOptimizer:
    """
    Weight optimizer for KNN recommendations to find optimal feature group weights
    """
    def __init__(self, df, n_neighbors=12, output_dir="weight_optimization_results", random_state=42):
        """
        Initialize the weight optimizer

        Args:
            df: DataFrame with product features (including IDs, titles etc.)
            n_neighbors: Number of neighbors for KNN
            output_dir: Directory to save optimization results
            random_state: Random seed for reproducibility
        """
        self.df = df.copy() # Keep original df with all columns
        self.n_neighbors = n_neighbors
        self.output_dir = output_dir
        self.random_state = random_state

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Keep track of optimization runs
        self.optimization_results = []
        self.best_weights = None
        self.best_score = -1.0 # Initialize with a low score
        self.best_model = None # Stores the best fitted _WeightedKNN model instance

        # Initialize a base model to get feature groups and prepare feature DataFrame
        self._init_base_model()

    def _init_base_model(self):
        """Initialize a base KNN model, identify feature groups, and prepare feature df."""
        print("Initializing Weight Optimizer Base Model...")
        # Use a temporary model instance just for setup
        temp_model = _WeightedKNN(n_neighbors=self.n_neighbors, random_state=self.random_state)

        # Extract features using the same logic as temp_model.fit would if given the DataFrame
        # (This avoids calling fit just for setup)
        non_feature_cols = ['product_id', 'product_title', 'handle', 'description',
                            'tags', 'collections', 'variants', 'metafields', 'metafields_config',
                            'created_at']
        potential_feature_cols = [col for col in self.df.columns if col not in non_feature_cols]
        numeric_cols = []
        for col in potential_feature_cols:
             if pd.api.types.is_numeric_dtype(self.df[col]):
                 if not np.isinf(self.df[col]).any(): numeric_cols.append(col)
             else:
                 try:
                      converted = pd.to_numeric(self.df[col], errors='coerce')
                      if converted.notnull().any() and not np.isinf(converted).any(): numeric_cols.append(col)
                 except: pass
        if not numeric_cols: raise ValueError("WeightOptimizer: No numeric features found.")

        self.feature_df = self.df[numeric_cols].copy()
        temp_model.feature_names_in_order = self.feature_df.columns.tolist() # Store order
        self.feature_df = temp_model.handle_missing_values(self.feature_df) # Handle NaNs

        # Store product IDs and titles from the original DataFrame
        self.product_ids = self.df['product_id'].copy() if 'product_id' in self.df.columns else pd.Series(self.df.index).astype(str)
        if self.product_ids.isnull().any(): # Handle NaNs in IDs again here
             nan_count = self.product_ids.isnull().sum()
             print(f"Optimizer Warning: Found {nan_count} missing product IDs. Filling with index.")
             nan_mask = self.product_ids.isnull()
             self.product_ids.loc[nan_mask] = self.df.index[nan_mask].map(lambda x: f"missing_id_{x}")
        self.product_titles = self.df['product_title'].copy() if 'product_title' in self.df.columns else None

        # Identify feature groups using the prepared feature DataFrame
        self.feature_groups = temp_model.identify_feature_groups(self.feature_df)

        # Define default weights (INCLUDING NEW GROUPS) used by the optimizer
        self.default_weights = {
            'description_tfidf': 1.0,
            'title_tfidf': 1.2,          # <<< UPDATED OPTIMIZER DEFAULT
            'handle_tfidf': 0.8,         # <<< UPDATED OPTIMIZER DEFAULT
            'metafield_data': 1.0,
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
        # Ensure all identified groups have a default weight
        for group in self.feature_groups:
             if group not in self.default_weights:
                  self.default_weights[group] = 1.0 # Assign default if missing

        # Evaluate default weights as baseline
        print("\nEvaluating default weights as baseline...")
        baseline_score = self.evaluate_weights(self.default_weights, "default_baseline", save_model_if_best=True)
        if self.best_score < 0 or baseline_score >= self.best_score: # Adjusted check
             print(f"Default weights established baseline score: {baseline_score:.4f}")
        else:
             print(f"Default weights score ({baseline_score:.4f}) is not better than initial best ({self.best_score:.4f}).")


    def evaluate_weights(self, weights, weight_name="custom", save_model_if_best=False):
        """
        Evaluate a specific set of weights by fitting a KNN model and calculating avg similarity.

        Args:
            weights: Dictionary of base weights for each feature group.
            weight_name: Name to identify this weight configuration in results.
            save_model_if_best: Whether to store the fitted model if it achieves the best score so far.

        Returns:
            float: Average similarity score achieved with these weights (0.0 on error).
        """
        eval_start_time = time.time()
        print(f"\n--- Evaluating weights: '{weight_name}' ---")

        # Create a new model instance for this evaluation
        model = _WeightedKNN(n_neighbors=self.n_neighbors, random_state=self.random_state)

        # Set essential attributes from the base setup
        model.product_ids = self.product_ids # Pass IDs
        # model.product_titles = self.product_titles # Titles not needed for fitting
        model.feature_groups = self.feature_groups # Use the identified groups
        model.feature_names_in_order = self.feature_df.columns.tolist() # Use the defined feature order

        try:
            # Set the weights for this evaluation (normalize by count is often good for comparison)
            model.set_feature_weights(weights, normalize_by_count=True)

            # Fit the model using the prepared feature DataFrame
            # Pass the DataFrame directly, not a path
            model.fit(self.feature_df) # Pass only the numeric feature df

            # Ensure the model fitted successfully before calculating similarity
            if model.recommendation_matrix is None:
                 print(f"ERROR: Model fitting failed for weights '{weight_name}'. Cannot calculate similarity.")
                 # Record failure and return 0
                 result = {
                     'weight_name': weight_name, 'weights': weights.copy(), 'avg_similarity': 0.0,
                     'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'eval_time_seconds': time.time() - eval_start_time,
                     'error': 'Fit failed, recommendation matrix is None'
                 }
                 self.optimization_results.append(result)
                 return 0.0

            # Calculate average similarity score
            avg_similarity = model.calculate_average_similarity()

            # Record the result
            result = {
                'weight_name': weight_name,
                'weights': weights.copy(), # Store the base weights used
                'avg_similarity': avg_similarity,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'eval_time_seconds': time.time() - eval_start_time,
                 'error': None # Indicate success
            }
            self.optimization_results.append(result)

            print(f"--- Result for '{weight_name}': Avg Similarity = {avg_similarity:.4f} (Eval time: {result['eval_time_seconds']:.2f}s) ---")

            # Update best if this is better
            # Use a small tolerance for comparison? e.g., >= self.best_score + 1e-6
            # Ensure avg_similarity is not NaN before comparison
            if pd.notna(avg_similarity) and avg_similarity > self.best_score:
                print(f"*** New best weights found! Avg similarity: {avg_similarity:.4f} (Previous best: {self.best_score:.4f}) ***")
                self.best_score = avg_similarity
                self.best_weights = weights.copy() # Store the base weights that led to this score
                if save_model_if_best:
                    print("Saving this model instance as the best found so far.")
                    self.best_model = model # Store the entire fitted model object
                # else: # Clear previous best model if not saving this one and it exists
                #      if self.best_model is not None and model is not self.best_model:
                #           self.best_model = None


            return avg_similarity if pd.notna(avg_similarity) else 0.0 # Return 0 if similarity was NaN

        except Exception as e:
            print(f"\n--- ERROR evaluating weights '{weight_name}' ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            print("-------------------------------------------------")
            # Record failure in results
            result = {
                 'weight_name': weight_name, 'weights': weights.copy(), 'avg_similarity': 0.0,
                 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'eval_time_seconds': time.time() - eval_start_time,
                 'error': str(e)
            }
            self.optimization_results.append(result)
            return 0.0 # Return 0.0 similarity score on error

    # --- Grid Search, Random Search, Evolutionary Search (No changes needed in logic, defaults updated) ---
    # Keep the methods grid_search, random_search, evolutionary_search, _tournament_selection,
    # _crossover, _mutate as they were. The core change was updating the default_weights
    # and ensuring identify_feature_groups works correctly.

    def grid_search(self, weight_options=None, fixed_weights=None, save_results=True):
        """
        Perform grid search over specified weight combinations for feature groups.
        (Method body remains the same as before)
        """
        if weight_options is None: weight_options = {}
        if fixed_weights is None: fixed_weights = {}

        all_groups = list(self.feature_groups.keys())
        search_space = {}
        default_range = [0.5, 1.0, 2.0, 5.0] # Example default range

        # Build the grid search space
        print("Building grid search space...")
        for group in all_groups:
            if group in fixed_weights:
                search_space[group] = [fixed_weights[group]] # Use fixed weight as a single-item list
                print(f"- Group '{group}': Fixed weight = {fixed_weights[group]}")
            elif group in weight_options:
                search_space[group] = weight_options[group]
                print(f"- Group '{group}': Options = {weight_options[group]}")
            else:
                search_space[group] = default_range # Use default range if not specified
                print(f"- Group '{group}': Using default options = {default_range}")

        # Generate all combinations
        search_keys = list(search_space.keys())
        search_values = list(search_space.values())
        all_combinations = list(product(*search_values)) # Use itertools.product

        total_combinations = len(all_combinations)
        if total_combinations == 0:
             print("Error: No combinations generated for grid search.")
             return self.get_optimization_results() # Return current results (likely just baseline)
        if total_combinations > 10000: # Add a warning for very large grids
             print(f"Warning: Grid search involves {total_combinations} combinations, which may take a very long time.")


        print(f"\nStarting grid search with {total_combinations} weight combinations...")
        grid_start_time = time.time()

        # Evaluate each combination
        for i, weight_combination in enumerate(all_combinations):
            combination_count = i + 1
            # Create weight dictionary for this combination
            current_weights = dict(zip(search_keys, weight_combination))

            # Set a unique name
            combination_name = f"grid_comb_{combination_count}"

            # Print progress
            log_interval_grid = max(1, total_combinations // 20) # Log ~20 times
            if combination_count % log_interval_grid == 0 or combination_count == 1 or combination_count == total_combinations:
                 elapsed_time = time.time() - grid_start_time
                 time_per_comb = elapsed_time / combination_count if combination_count > 0 else 0
                 estimated_total_time = time_per_comb * total_combinations
                 remaining_time = estimated_total_time - elapsed_time
                 print(f"\n[Grid Search Progress: {combination_count}/{total_combinations}] "
                       f"Avg time/comb: {time_per_comb:.2f}s | Elapsed: {elapsed_time:.1f}s | Est. Remaining: {remaining_time:.1f}s")

            # Evaluate this weight combination, save the model if it's the best
            _ = self.evaluate_weights(current_weights, combination_name, save_model_if_best=True)

        grid_end_time = time.time()
        print(f"\nGrid search completed in {grid_end_time - grid_start_time:.2f} seconds.")

        # Save results to CSV if requested
        if save_results:
            self.save_optimization_results()

        # Return DataFrame with all results
        return self.get_optimization_results()

    def random_search(self, n_iterations=50, weight_range=(0.1, 10.0), save_results=True):
        """
        Perform random search over weight combinations.
        (Method body remains the same as before)
        """
        all_groups = list(self.feature_groups.keys())
        if not all_groups:
             print("Error: No feature groups identified. Cannot perform random search.")
             return self.get_optimization_results()

        print(f"\nStarting random search with {n_iterations} iterations...")
        print(f"Weight range for random generation: {weight_range}")
        random_start_time = time.time()

        # Generate and evaluate random combinations
        for i in range(n_iterations):
            iteration_count = i + 1
            # Generate random weights for each feature group
            current_weights = {
                group: np.random.uniform(weight_range[0], weight_range[1])
                for group in all_groups
            }

            # Set a unique name
            combination_name = f"random_iter_{iteration_count}"

            # Print progress
            log_interval_random = max(1, n_iterations // 20) # Log ~20 times
            if iteration_count % log_interval_random == 0 or iteration_count == 1 or iteration_count == n_iterations:
                elapsed_time = time.time() - random_start_time
                time_per_iter = elapsed_time / iteration_count if iteration_count > 0 else 0
                estimated_total_time = time_per_iter * n_iterations
                remaining_time = estimated_total_time - elapsed_time
                print(f"\n[Random Search Progress: {iteration_count}/{n_iterations}] "
                       f"Avg time/iter: {time_per_iter:.2f}s | Elapsed: {elapsed_time:.1f}s | Est. Remaining: {remaining_time:.1f}s")


            # Evaluate this weight combination, save the model if it's the best
            _ = self.evaluate_weights(current_weights, combination_name, save_model_if_best=True)

        random_end_time = time.time()
        print(f"\nRandom search completed in {random_end_time - random_start_time:.2f} seconds.")

        # Save results to CSV if requested
        if save_results:
            self.save_optimization_results()

        # Return DataFrame with all results
        return self.get_optimization_results()

    def evolutionary_search(self, population_size=10, generations=5, mutation_rate=0.2,
                            crossover_rate=0.8, tournament_size=3,
                            weight_range=(0.1, 10.0), save_results=True):
        """
        Perform evolutionary search to find optimal weights using a genetic algorithm approach.
        (Method body remains the same as before)
        """
        all_groups = list(self.feature_groups.keys())
        if not all_groups:
             print("Error: No feature groups identified. Cannot perform evolutionary search.")
             return self.get_optimization_results()
        if population_size <= 1 or generations <= 0 or tournament_size >= population_size: # Adjusted checks
             print(f"Error: Invalid evolutionary parameters (pop_size={population_size} > 1, gens={generations} > 0, tourn_size={tournament_size} < pop_size).")
             return self.get_optimization_results()


        print(f"\nStarting evolutionary search:")
        print(f"- Population Size: {population_size}")
        print(f"- Generations: {generations}")
        print(f"- Mutation Rate: {mutation_rate}")
        print(f"- Crossover Rate: {crossover_rate}")
        print(f"- Tournament Size: {tournament_size}")
        print(f"- Weight Range: {weight_range}")
        evo_start_time = time.time()

        # --- Initialize Population ---
        print("\nInitializing Generation 0...")
        population = []
        init_eval_times = []
        for i in range(population_size):
            weights = {group: np.random.uniform(weight_range[0], weight_range[1]) for group in all_groups}
            individual_name = f"evo_gen0_ind{i+1}"
            eval_start = time.time()
            fitness = self.evaluate_weights(weights, individual_name, save_model_if_best=True)
            init_eval_times.append(time.time() - eval_start)
            population.append({'weights': weights, 'fitness': fitness, 'name': individual_name})

        # Sort initial population by fitness (descending)
        population.sort(key=lambda x: x['fitness'], reverse=True)
        print(f"Generation 0 Initialized. Best fitness: {population[0]['fitness']:.4f}. Avg eval time: {np.mean(init_eval_times):.2f}s")

        # --- Evolution Loop ---
        total_evals = population_size # Track total evaluations
        for gen in range(1, generations + 1):
            gen_start_time = time.time()
            print(f"\n--- Starting Generation {gen}/{generations} ---")

            new_population = []
            gen_eval_times = []

            # Elitism: Keep the best 'n_elite' individuals (e.g., 1 or 2)
            n_elite = max(1, population_size // 10) # Keep top 10% or at least 1
            if n_elite > 0 and len(population) > 0:
                 # print(f"Applying elitism: Keeping best {n_elite} individual(s).")
                 for i in range(min(n_elite, len(population))):
                      # Add elite individuals without re-evaluating
                      new_population.append(population[i])


            # Generate the rest of the population through selection, crossover, mutation
            while len(new_population) < population_size:
                # Selection: Choose two parents using tournament selection
                parent1 = self._tournament_selection(population, tournament_size)
                parent2 = self._tournament_selection(population, tournament_size)

                # Crossover: Create child weights
                if np.random.random() < crossover_rate:
                    child_weights = self._crossover(parent1['weights'], parent2['weights'])
                else:
                    # If no crossover, clone the better parent found in selection
                    child_weights = parent1['weights'].copy() if parent1['fitness'] >= parent2['fitness'] else parent2['weights'].copy()


                # Mutation: Apply mutation to the child's weights
                self._mutate(child_weights, mutation_rate, weight_range)

                # Evaluate fitness of the new child
                total_evals += 1
                child_name = f"evo_gen{gen}_ind{len(new_population)+1}"
                eval_start = time.time()
                child_fitness = self.evaluate_weights(child_weights, child_name, save_model_if_best=True)
                gen_eval_times.append(time.time() - eval_start)

                # Add the new child to the next generation's population
                new_population.append({'weights': child_weights, 'fitness': child_fitness, 'name': child_name})

            # Replace old population with the new one
            population = new_population

            # Sort the new generation by fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)

            gen_end_time = time.time()
            avg_gen_eval_time = np.mean(gen_eval_times) if gen_eval_times else 0
            print(f"--- Generation {gen} Completed (Time: {gen_end_time - gen_start_time:.2f}s | Avg Eval: {avg_gen_eval_time:.2f}s) ---")
            print(f"Best fitness in Generation {gen}: {population[0]['fitness']:.4f} (Individual: {population[0]['name']})")


        evo_end_time = time.time()
        print(f"\nEvolutionary search completed in {evo_end_time - evo_start_time:.2f} seconds.")
        print(f"Total evaluations performed: {total_evals}")

        # Save results to CSV if requested
        if save_results:
            self.save_optimization_results()

        # Return DataFrame with all results
        return self.get_optimization_results()


    def _tournament_selection(self, population, tournament_size):
        """Helper for evolutionary search: Selects the best individual from a random tournament."""
        # (Method body remains the same as before)
        if not population: raise ValueError("Cannot perform tournament selection on an empty population.")
        if tournament_size <= 0: tournament_size = 1
        actual_tournament_size = min(tournament_size, len(population))
        tournament_indices = np.random.choice(len(population), size=actual_tournament_size, replace=False)
        tournament_participants = [population[i] for i in tournament_indices]
        return max(tournament_participants, key=lambda x: x['fitness'])

    def _crossover(self, weights1, weights2):
        """Helper for evolutionary search: Performs uniform crossover between two weight sets."""
        # (Method body remains the same as before)
        child_weights = {}
        all_groups = list(weights1.keys())
        for group in all_groups:
            if np.random.random() < 0.5: child_weights[group] = weights1[group]
            else: child_weights[group] = weights2[group]
        return child_weights

    def _mutate(self, weights, mutation_rate, weight_range):
        """Helper for evolutionary search: Mutates weights in place with a given probability."""
        # (Method body remains the same as before)
        for group in weights:
            if np.random.random() < mutation_rate:
                weights[group] = np.random.uniform(weight_range[0], weight_range[1])
                # Ensure weight doesn't go below a minimum threshold (e.g., 0.01)
                weights[group] = max(0.01, weights[group])


    # --- Result Handling and Saving (No changes needed) ---
    def get_optimization_results(self):
        """
        Get all optimization run results compiled into a DataFrame.
        (Method body remains the same as before)
        """
        if not self.optimization_results:
            print("No optimization results recorded yet.")
            return pd.DataFrame()
        # Prepare data for DataFrame conversion
        results_list_for_df = []
        all_groups = set()
        for result in self.optimization_results:
             if 'weights' in result and isinstance(result['weights'], dict):
                  all_groups.update(result['weights'].keys())
        sorted_groups = sorted(list(all_groups))
        for result in self.optimization_results:
             row_data = {
                 'weight_name': result.get('weight_name', 'N/A'),
                 'avg_similarity': result.get('avg_similarity', 0.0),
                 'timestamp': result.get('timestamp', 'N/A'),
                 'eval_time_seconds': result.get('eval_time_seconds', np.nan),
                 'error': result.get('error', None)
             }
             base_weights = result.get('weights', {})
             for group in sorted_groups:
                  row_data[f"weight_{group}"] = base_weights.get(group, np.nan)
             results_list_for_df.append(row_data)
        results_df = pd.DataFrame(results_list_for_df)
        if 'avg_similarity' in results_df.columns:
            results_df = results_df.sort_values('avg_similarity', ascending=False, na_position='last')
        return results_df

    def save_optimization_results(self, filename=None):
        """
        Save optimization results (including weights) to a CSV file and best weights to JSON.
        (Method body remains the same as before)
        """
        results_df = self.get_optimization_results()
        if results_df.empty: print("No optimization results to save."); return
        if filename is None: filename = os.path.join(self.output_dir, "optimization_results.csv")
        try:
            results_df.to_csv(filename, index=False)
            print(f"Optimization results saved to {filename}")
        except Exception as e: print(f"Error saving optimization results CSV to {filename}: {e}")
        if self.best_weights:
            best_weights_file = os.path.join(self.output_dir, "best_weights.json")
            try:
                with open(best_weights_file, 'w') as f: json.dump(self.best_weights, f, indent=2)
                print(f"Best weights (achieving score {self.best_score:.4f}) saved to {best_weights_file}")
            except Exception as e: print(f"Error saving best weights JSON to {best_weights_file}: {e}")
        else: print("No best weights recorded during optimization.")

    def visualize_optimization_results(self, top_n=20, filename=None):
        """
        Create a bar plot showing the performance of the top N weight configurations.
        (Method body remains the same as before)
        """
        results_df = self.get_optimization_results()
        if results_df.empty: print("No optimization results to visualize."); return
        if 'error' in results_df.columns: plot_df = results_df[results_df['error'].isnull()].head(top_n)
        else: plot_df = results_df.head(top_n)
        if plot_df.empty: print("No valid (non-error) optimization results to visualize."); return
        plt.figure(figsize=(12, max(6, len(plot_df) * 0.4)))
        sns.barplot(x='avg_similarity', y='weight_name', data=plot_df, palette='viridis')
        plt.title(f'Top {len(plot_df)} Weight Configurations by Average Similarity')
        plt.xlabel('Average Similarity Score')
        plt.ylabel('Weight Configuration Name')
        plt.tight_layout()
        if filename is None: filename = os.path.join(self.output_dir, "optimization_results.png")
        try:
            plt.savefig(filename, dpi=120)
            plt.close()
            print(f"Optimization results visualization saved to {filename}")
        except Exception as e: print(f"Error saving optimization results plot to {filename}: {e}")


    def build_optimized_model(self, save_model_files=True):
        """
        Build and fit a new _WeightedKNN model using the best weights found during optimization.
        (Method body remains the same as before)
        """
        print("\n--- Building Final Model with Optimized Weights ---")
        if self.best_weights is None:
            print("Warning: No best weights found. Falling back to default weights.")
            weights_to_use = self.default_weights
            if not weights_to_use: print("Error: Cannot build model, default weights also missing."); return None
        else:
            print(f"Using best weights found (Avg Similarity: {self.best_score:.4f})")
            weights_to_use = self.best_weights

        # Check if we stored the best model instance directly
        if self.best_model and self.best_model.recommendation_matrix is not None and self.best_model.feature_weights == self.best_weights: # Add check weights match
             print("Using the pre-fitted best model instance directly.")
             model = self.best_model
        else:
             print("Building and fitting a new model instance with the best weights...")
             model = _WeightedKNN(n_neighbors=self.n_neighbors, random_state=self.random_state)
             model.product_ids = self.product_ids
             model.feature_groups = self.feature_groups
             model.feature_names_in_order = self.feature_df.columns.tolist()
             model.set_feature_weights(weights_to_use, normalize_by_count=True)
             try:
                 model.fit(self.feature_df)
                 if model.recommendation_matrix is None: print("Error: Fitting the final optimized model failed."); return None
             except Exception as e: print(f"Error fitting the final optimized model: {e}"); return None

        if save_model_files:
            save_dir = os.path.join(self.output_dir, "optimized_model_final")
            print(f"\nSaving the final optimized model components to: {save_dir}")
            model.save_model(save_dir)

        print("--- Optimized Model Building Complete ---")
        return model


# --- Standalone Function: find_similar_products (No changes needed) ---
def find_similar_products(product_id, knn_model, products_df, top_n=12,
                        exclude_similar_variants=True, similar_products_df=None):
    """
    Find similar products for a given product ID using a fitted _WeightedKNN model.
    (Method body remains the same as before)
    """
    if not isinstance(knn_model, _WeightedKNN) or knn_model.recommendation_matrix is None:
        print(f"Error: Invalid or unfitted KNN model provided for finding similar products to {product_id}.")
        return pd.DataFrame()
    if products_df is None or 'product_id' not in products_df.columns:
        print("Warning: products_df missing or lacks 'product_id'. Cannot retrieve titles.")
        # Decide how to handle title missing - proceed without or fail?
        # return pd.DataFrame()

    # 1. Get Raw Recommendations
    raw_recs = knn_model.get_recommendations(product_id, n_recommendations=top_n * 2 + 5) # Get more for filtering
    if not raw_recs: return pd.DataFrame()

    similar_df = pd.DataFrame(raw_recs, columns=['recommended_product_id', 'similarity'])

    # 2. Filter Self
    similar_df = similar_df[similar_df['recommended_product_id'] != product_id].copy()

    # 3. Filter Variants
    variants_filtered_count = 0
    if exclude_similar_variants and similar_products_df is not None and not similar_products_df.empty:
        exclude_ids = set()
        col1_id, col2_id = None, None
        if 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns: col1_id, col2_id = 'product1_id', 'product2_id'
        elif 'product1' in similar_products_df.columns and 'product2' in similar_products_df.columns: col1_id, col2_id = 'product1', 'product2'
        if col1_id and col2_id:
            try:
                matches1 = similar_products_df[similar_products_df[col1_id] == product_id]
                if not matches1.empty: exclude_ids.update(matches1[col2_id].tolist())
                matches2 = similar_products_df[similar_products_df[col2_id] == product_id]
                if not matches2.empty: exclude_ids.update(matches2[col1_id].tolist())
            except Exception as e: print(f"Warning: Error during variant filtering for {product_id}: {e}")
            if exclude_ids:
                count_before = len(similar_df)
                similar_df = similar_df[~similar_df['recommended_product_id'].isin(exclude_ids)]
                variants_filtered_count = count_before - len(similar_df)

    # 4. Select Top N
    final_similar_df = similar_df.head(top_n).copy()

    # 5. Add Titles (if possible)
    if products_df is not None and 'product_title' in products_df.columns and 'product_id' in products_df.columns:
        try:
            # Ensure consistent types for mapping
            products_df['product_id_str'] = products_df['product_id'].astype(str)
            title_map = products_df.set_index('product_id_str')['product_title'].to_dict()
            final_similar_df['recommended_product_id_str'] = final_similar_df['recommended_product_id'].astype(str)
            final_similar_df['product_title'] = final_similar_df['recommended_product_id_str'].map(title_map).fillna('Title Not Found')
            final_similar_df = final_similar_df.drop(columns=['recommended_product_id_str'], errors='ignore')
            products_df = products_df.drop(columns=['product_id_str'], errors='ignore') 
        except Exception as title_err:
             print(f"Warning adding title for {product_id}: {title_err}")
             final_similar_df['product_title'] = 'Title Error'
    else:
        final_similar_df['product_title'] = 'Title N/A' # Fallback

    # 6. Final Formatting
    final_similar_df = final_similar_df.rename(columns={'recommended_product_id': 'product_id'})
    final_columns = ['product_id', 'similarity', 'product_title']
    final_similar_df = final_similar_df[[col for col in final_columns if col in final_similar_df.columns]]

    if variants_filtered_count > 0:
         # print(f"Filtered {variants_filtered_count} similar variants while finding recommendations for {product_id}.")
         pass # Reduce verbosity


    return final_similar_df


def run_knn(df, output_dir="results", n_neighbors=12, save_model=True,
          similar_products_df=None,
          optimize_weights=False, optimization_method='grid', optimization_params=None):
    """
    Main public interface for running weighted KNN for product recommendations.
    Handles model initialization, fitting, optional optimization, visualization, and export.
    """
    print("\n" + "="*30 + " Running Weighted KNN Recommendation Pipeline " + "="*30)
    pipeline_start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if df is None or df.empty: print("Error: Input DataFrame 'df' is empty or None."); return pd.DataFrame(), None # Return model as None
    # Product ID check handled within _WeightedKNN

    if similar_products_df is None or similar_products_df.empty:
        print("Note: No 'similar_products_df' provided. Variant filtering disabled during export/find_similar.")
    else: # Basic check for variant df columns
        has_cols1 = 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns
        has_cols2 = 'product1' in similar_products_df.columns and 'product2' in similar_products_df.columns
        if not (has_cols1 or has_cols2): print("Warning: 'similar_products_df' lacks expected columns. Variant filtering may fail.")
        else: print(f"Using provided 'similar_products_df' ({len(similar_products_df)} pairs) for filtering.")

    final_model = None
    optimizer = None # Initialize optimizer variable
    if optimize_weights:
        print(f"\n=== Starting Feature Weight Optimization (Method: {optimization_method}) ===")
        opt_start_time = time.time()
        opt_dir = os.path.join(output_dir, "weight_optimization")
        try:
            optimizer = WeightOptimizer(df=df.copy(), n_neighbors=n_neighbors, output_dir=opt_dir, random_state=42)
            if optimization_params is None: optimization_params = {}
            print("Running optimization...")
            if optimization_method == 'grid': optimizer.grid_search(**optimization_params, save_results=True)
            elif optimization_method == 'random': optimizer.random_search(**optimization_params, save_results=True)
            elif optimization_method == 'evolutionary': optimizer.evolutionary_search(**optimization_params, save_results=True)
            else: print(f"Warning: Unknown optimization method '{optimization_method}'. Defaulting to grid."); optimizer.grid_search(save_results=True)
            optimizer.visualize_optimization_results()
            print("\nBuilding final KNN model using optimized weights...")
            final_model = optimizer.build_optimized_model(save_model_files=save_model)
            opt_end_time = time.time()
            print(f"=== Weight Optimization Finished (Duration: {opt_end_time - opt_start_time:.2f}s) ===")
        except Exception as e:
            print(f"\n--- ERROR during Weight Optimization: {e} ---")
            traceback.print_exc()
            print("Optimization failed. Attempting to proceed with default weights...")
            final_model = None

    if final_model is None:
        print("\n=== Building KNN Model with Default Weights ===")
        model_build_start_time = time.time()
        try:
            final_model = _WeightedKNN(n_neighbors=n_neighbors, random_state=42)
            final_model.fit(df.copy()) # Pass copy to fit
            model_build_end_time = time.time()
            if final_model.recommendation_matrix is not None:
                 print(f"Model fitting with default weights completed in {model_build_end_time - model_build_start_time:.2f}s.")
                 if save_model:
                      print("\nSaving the model fitted with default weights...")
                      save_dir_default = os.path.join(output_dir, "model_default_weights")
                      final_model.save_model(save_dir_default)
            else: print("Error: Failed to fit model even with default weights."); final_model = None
        except Exception as e:
            print(f"\n--- ERROR during Model Fitting with Default Weights: {e} ---")
            traceback.print_exc(); final_model = None

    all_recommendations_df = pd.DataFrame()
    # Define pre-filter output path
    knn_pre_filter_output_file = os.path.join(output_dir, "knn_recommendations_pre_variant_filter.csv")

    if final_model is None or final_model.recommendation_matrix is None:
         print("\nError: No valid KNN model was fitted. Cannot proceed.")
         # Ensure we return model as None if it failed
         final_model_return = None # Explicitly set return value
    else:
        final_model_return = final_model # Store model to return later
        print("\n=== Generating Visualizations and Exporting Results ===")

        # --- START: Save Pre-Variant Filter Recommendations ---
        print(f"\nSaving raw KNN recommendations BEFORE variant filtering to: {knn_pre_filter_output_file}")
        pre_filter_rows = []
        if final_model.recommendation_matrix:
            for source_id, recommendations in final_model.recommendation_matrix.items():
                rank = 0
                # Ensure recommendations is a list of tuples
                if not isinstance(recommendations, list):
                    print(f"Warning: Unexpected format for recommendations of {source_id}. Skipping pre-filter save for this product.")
                    continue

                for rec_tuple in recommendations:
                     # Basic check for tuple structure
                     if not isinstance(rec_tuple, tuple) or len(rec_tuple) != 2:
                          print(f"Warning: Invalid recommendation item {rec_tuple} for {source_id}. Skipping.")
                          continue

                     rec_id, similarity = rec_tuple
                     if rec_id != source_id: # Filter self-recommendations
                         rank += 1
                         pre_filter_rows.append({
                             'source_product_id': source_id,
                             'recommended_product_id': rec_id,
                             'rank': rank,
                             'similarity_score': similarity
                         })
                         if rank >= n_neighbors: # Stop after reaching desired number
                             break
            try:
                 pre_filter_df = pd.DataFrame(pre_filter_rows)
                 # --- Add Titles (Optional but helpful) ---
                 if not pre_filter_df.empty and df is not None and 'product_id' in df.columns and 'product_title' in df.columns:
                      try:
                           df['product_id_str'] = df['product_id'].astype(str)
                           title_map = df.set_index('product_id_str')['product_title'].to_dict()
                           pre_filter_df['source_product_id_str'] = pre_filter_df['source_product_id'].astype(str)
                           pre_filter_df['recommended_product_id_str'] = pre_filter_df['recommended_product_id'].astype(str)
                           pre_filter_df['source_product_title'] = pre_filter_df['source_product_id_str'].map(title_map).fillna('Title NF')
                           pre_filter_df['recommended_product_title'] = pre_filter_df['recommended_product_id_str'].map(title_map).fillna('Title NF')
                           pre_filter_df = pre_filter_df.drop(columns=['source_product_id_str', 'recommended_product_id_str'], errors='ignore')
                           df = df.drop(columns=['product_id_str'], errors='ignore')
                           # Reorder cols
                           cols_order_pre = ['source_product_id', 'source_product_title', 'recommended_product_id', 'recommended_product_title', 'rank', 'similarity_score']
                           pre_filter_df = pre_filter_df[[col for col in cols_order_pre if col in pre_filter_df.columns]]
                      except Exception as title_err_pre: print(f"Warning: Error adding titles to pre-filter KNN: {title_err_pre}")

                 pre_filter_df.to_csv(knn_pre_filter_output_file, index=False)
                 print(f"Successfully saved {len(pre_filter_df)} pre-filter recommendation pairs.")
            except Exception as e:
                 print(f"\n--- Error Saving Pre-Filter KNN Recommendations: {e} ---")
                 traceback.print_exc()
        else:
             print("Warning: Recommendation matrix is empty. Cannot save pre-filter recommendations.")
        # --- END: Save Pre-Variant Filter Recommendations ---


        vis_success = {'importance': False, 'tsne': False, 'distribution': False}
        try: final_model.visualize_feature_importance(output_file=os.path.join(output_dir, "feature_importance.png")); vis_success['importance'] = True
        except Exception as e: print(f"Error generating feature importance plot: {e}")
        try: final_model.visualize_tsne(output_file=os.path.join(output_dir, "tsne_visualization.png")); vis_success['tsne'] = True
        except Exception as e: print(f"Error generating t-SNE plot: {e}")
        try: final_model.visualize_similarity_distribution(output_file=os.path.join(output_dir, "similarity_distribution.png")); vis_success['distribution'] = True
        except Exception as e: print(f"Error generating similarity distribution plot: {e}")

        try:
            print("\nExporting all recommendations (with variant filtering)...")
            all_recommendations_df = final_model.export_recommendations(
                output_file=os.path.join(output_dir, "all_recommendations.csv"),
                n_recommendations=n_neighbors,
                products_df=df, # Pass original df for titles
                similar_products_df=similar_products_df # Pass variant df for filtering
            )
            if not all_recommendations_df.empty: print(f"Successfully exported {len(all_recommendations_df)} filtered recommendation pairs.")
            else: print("Export completed, but no recommendations were generated or exported after filtering.")
        except Exception as e:
            print(f"\n--- Error Exporting Filtered Recommendations: {e} ---")
            traceback.print_exc(); all_recommendations_df = pd.DataFrame()

        # --- Example Similar Products ---
        # (Keep this section as is)
        if not df.empty and 'product_id' in df.columns:
             try:
                 # Use a robust way to get a sample ID (e.g., first non-null)
                 sample_product_id = df['product_id'].dropna().iloc[0] if not df['product_id'].dropna().empty else None
                 if sample_product_id:
                     print(f"\nGenerating example recommendations for sample product: {sample_product_id}")
                     # Get title safely
                     sample_title = df.loc[df['product_id'] == sample_product_id, 'product_title'].iloc[0] if 'product_title' in df.columns else 'N/A'
                     if sample_title != 'N/A': print(f"Sample Product Title: {sample_title}")

                     similar_products_example_df = find_similar_products(
                         product_id=sample_product_id, knn_model=final_model, products_df=df,
                         top_n=n_neighbors, exclude_similar_variants=True, similar_products_df=similar_products_df
                     )
                     if not similar_products_example_df.empty:
                         print("\nExample Similar Products Found:"); print(similar_products_example_df.to_string(index=False))
                         example_file = os.path.join(output_dir, "similar_products_example.csv")
                         similar_products_example_df.to_csv(example_file, index=False)
                         print(f"Example similar products saved to {example_file}")
                     else: print(f"No similar products found for the sample product {sample_product_id} after filtering.")
                 else: print("Could not find a valid sample product ID to generate example.")
             except Exception as e: print(f"\nError generating similar products example: {e}")


        # --- Summary Report ---
        # (Keep this section as is - maybe add mention of the pre-filter file?)
        try:
            summary_path = os.path.join(output_dir, "knn_summary_report.txt")
            with open(summary_path, "w", encoding='utf-8') as f:
                f.write("=== Weighted KNN Recommendation Summary ===\n\n")
                # ... (rest of summary info) ...
                f.write("\nRecommendations Exported:\n")
                if os.path.exists(knn_pre_filter_output_file): f.write(f"- knn_recommendations_pre_variant_filter.csv: Yes\n") # Added line
                export_status = 'Yes' if not all_recommendations_df.empty else 'No'
                f.write(f"- all_recommendations.csv (Filtered): {export_status} ({len(all_recommendations_df)} pairs)\n")
                if os.path.exists(os.path.join(output_dir, 'similar_products_example.csv')): f.write("- similar_products_example.csv: Yes\n")
                # ... (rest of summary info) ...
            print(f"\nKNN summary report saved to {summary_path}")
        except Exception as e: print(f"\nError generating summary report: {e}")


    pipeline_end_time = time.time()
    print("\n" + "="*30 + f" KNN Pipeline Finished (Total Time: {pipeline_end_time - pipeline_start_time:.2f}s) " + "="*30)
    # Return both the filtered df and the final model object
    return all_recommendations_df, final_model_return