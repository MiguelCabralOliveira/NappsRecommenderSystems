# weighted_knn.py

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# Removido PCA não utilizado aqui
from sklearn.manifold import TSNE
import seaborn as sns
import os
import json
import joblib
import warnings
# Removido defaultdict não utilizado aqui
# Removido re não utilizado aqui
import time
from itertools import product
# Removido KFold não utilizado aqui


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
        self.recommendation_matrix = None # Stores {product_id: [(rec_id, similarity), ...]}
        self.distances = None # Distances from kneighbors
        self.indices = None # Indices from kneighbors
        self.X_weighted = None # Store the weighted matrix for potential reuse
        self.feature_names_in_order = None # Store the order of features used

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
             print(f"Warning: Found {self.product_ids.isnull().sum()} missing product IDs. Filling with index.")
             # Fill missing IDs using index, prefixing to avoid collisions
             self.product_ids = self.product_ids.fillna(pd.Series(df.index).apply(lambda x: f"missing_id_{x}"))

        # Remove non-feature columns (be explicit)
        non_feature_cols = ['product_id', 'product_title', 'handle', 'description'] # Add others if known
        feature_cols = [col for col in df.columns if col not in non_feature_cols and df[col].dtype != 'object'] # Attempt initial numeric filter

        # More robust numeric filtering
        numeric_cols = []
        potential_feature_df = df[feature_cols].copy() # Work on a copy
        for col in potential_feature_df.columns:
            try:
                # Attempt conversion, coerce errors to NaN
                converted_col = pd.to_numeric(potential_feature_df[col], errors='coerce')
                # Only keep if *some* values could be converted and it's not all NaN
                if not converted_col.isnull().all() and converted_col.notnull().any():
                    numeric_cols.append(col)
                # else: # Optional: report columns that were fully non-numeric
                #     print(f"Skipping column '{col}': Cannot convert to numeric.")
            except Exception as e: # Catch broader errors during conversion attempt
                print(f"Skipping column '{col}' due to error during numeric check: {e}")

        # Keep only confirmed numeric columns
        if len(numeric_cols) < len(feature_cols):
            dropped_cols = [col for col in feature_cols if col not in numeric_cols]
            print(f"Removed {len(dropped_cols)} non-numeric or problematic columns from feature set.") # Example: {dropped_cols}") # Be careful printing potentially long lists
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
        # Initialize feature groups
        feature_groups = {
            'description_tfidf': [], 'metafield_data': [], 'collections': [],
            'tags': [], 'product_type': [], 'vendor': [],
            'variant_size': [], 'variant_color': [], 'variant_other': [],
            'price': [], 'numerical': [], 'time_features': [],
            'release_quarter': [], 'seasons': [], 'availability': [],
            'gift_card': [], 'other': [] # Catch-all group
        }

        # Use the stored feature names if available, otherwise use df.columns
        columns_to_process = self.feature_names_in_order if self.feature_names_in_order else df.columns

        # Pattern matching for feature groups based on column names
        for col in columns_to_process:
            assigned = False
            if col.startswith('description_'):
                feature_groups['description_tfidf'].append(col); assigned = True
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
            elif col.startswith('variant_other_'):
                 feature_groups['variant_other'].append(col); assigned = True
            # Time features
            elif (col.startswith('created_') and col != 'created_season') or col in ['days_since_first_product', 'is_recent_product']:
                feature_groups['time_features'].append(col); assigned = True
            # Release quarter dummies
            elif col.startswith('release_quarter_'):
                feature_groups['release_quarter'].append(col); assigned = True
             # Seasonal feature
            elif col == 'created_season':
                feature_groups['seasons'].append(col); assigned = True
            # Price features
            elif col in ['min_price', 'max_price', 'has_discount'] or col.startswith('price_'):
                 feature_groups['price'].append(col); assigned = True
             # Availability
            elif col == 'available_for_sale':
                 feature_groups['availability'].append(col); assigned = True
            # Gift Card
            elif col == 'is_gift_card':
                 feature_groups['gift_card'].append(col); assigned = True
            # Default to numerical for any other explicitly numeric types (though most should be caught above)
            elif df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                # Check if it wasn't assigned already to avoid duplication if logic overlaps
                if not assigned:
                    feature_groups['numerical'].append(col); assigned = True

            # If still not assigned, put in 'other'
            if not assigned:
                 # Optional: Check if it's numeric before adding to 'other' or create a specific group?
                 if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                     feature_groups['other'].append(col)
                 # else: # Handle potential non-numeric columns that slipped through load_data?
                 #     print(f"Warning: Column '{col}' was not assigned to a feature group and is not numeric.")


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
             print(f"Warning: Number of grouped features ({total_features_grouped}) does not match total input features ({len(columns_to_process)}). Check 'Other' group or unassigned features.")


        # Verify all features are in exactly one group
        all_grouped_features = set(f for features in self.feature_groups.values() for f in features)
        if len(all_grouped_features) != total_features_grouped:
             print("Warning: Some features might be listed in multiple groups!")
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
            # Define more comprehensive default weights
            base_weights = {
                'description_tfidf': 1.0,
                'metafield_data': 1.0,
                'collections': 1.4,
                'tags': 1.8,
                'product_type': 1.8,
                'vendor': 1.0,
                'variant_size': 1.0,
                'variant_color': 1.0,
                'variant_other': 1.0,
                'price': 0.8, # Added price group
                'numerical': 0.7, # Renamed from 'numerical' in old code, consistency needed
                'time_features': 0.5,
                'release_quarter': 0.5,
                'seasons': 0.8,
                'availability': 0.6, # Added availability
                'gift_card': 0.1, # Low weight for gift cards typically
                'other': 0.5 # Default for uncategorized
            }
            print("Using default base weights.")
        else:
            print("Using provided base weights.")

        # Ensure all identified groups have a base weight (default to 1.0 if missing)
        final_base_weights = base_weights.copy()
        for group in self.feature_groups:
            if group not in final_base_weights:
                final_base_weights[group] = 1.0
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
            target_count = max(1, np.mean([c for c in feature_counts.values() if c > 0])) # Use mean non-zero count
            # target_count = 30 # Or a fixed value
            print(f"Normalizing weights using target count: {target_count:.2f}")

            for group, features in self.feature_groups.items():
                count = feature_counts.get(group, 1) # Use count, default to 1 to avoid division by zero
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

        current_index = 0
        for group, features in self.feature_groups.items():
            weight = self.feature_weights.get(group, 1.0) # Get weight for the group
            for feature_name in features:
                try:
                    # Find the index of this feature in the ordered list
                    idx = self.feature_names_in_order.index(feature_name)
                    feature_index_to_weight[idx] = weight
                except ValueError:
                    print(f"Warning: Feature '{feature_name}' from group '{group}' not found in the stored feature order. Skipping its weight application.")
                except IndexError:
                     print(f"Error: Index out of bounds when trying to assign weight for feature '{feature_name}'. Matrix shape might be inconsistent.")


        # Apply weights column-wise using broadcasting (more efficient)
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
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            print(f"Found {total_nans} missing values (NaNs) in the feature data.")
            # print("Columns with missing values:")
            # print(nan_counts[nan_counts > 0]) # Show counts per column with NaNs

            # Strategy: Fill NaN values with 0
            # This is a simple strategy; others like mean/median imputation could be considered
            # but might be complex with weighted features. Zero is often reasonable for
            # sparse features like TF-IDF or dummy variables.
            print("Filling missing values with 0...")
            df_filled = df.fillna(0)

            # Verify no NaNs remain
            if df_filled.isnull().sum().sum() > 0:
                # This shouldn't happen with fillna(0) unless there are strange data types
                print("ERROR: Still have NaN values after filling!")
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
                feature_df = self.load_data(data)
            elif isinstance(data, pd.DataFrame):
                # Assume data is already the feature DataFrame
                # Re-extract IDs and titles if needed and not already set
                if self.product_ids is None and 'product_id' in data.columns:
                     self.product_ids = data['product_id'].copy()
                if self.product_titles is None and 'product_title' in data.columns:
                     self.product_titles = data['product_title'].copy()
                # Select only numeric features, handling potential non-numeric cols passed
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) < len(data.columns):
                     print(f"Warning: Input DataFrame contains non-numeric columns. Using only {len(numeric_cols)} numeric columns.")
                feature_df = data[numeric_cols].copy()
                self.feature_names_in_order = feature_df.columns.tolist() # Store feature order
                feature_df = self.handle_missing_values(feature_df) # Handle NaNs
            else:
                raise TypeError("Input data must be a pandas DataFrame or a path to a CSV file.")

            if feature_df.empty:
                 raise ValueError("Feature DataFrame is empty after loading and preparation.")
            if self.product_ids is None: # Check again after potential loading from DataFrame
                 print("Warning: Product IDs are still missing after data loading. Using index.")
                 self.product_ids = pd.Series(feature_df.index).astype(str)


            # --- 2. Identify Groups & Set Weights (if not already done) ---
            if self.feature_groups is None:
                print("Identifying feature groups...")
                self.identify_feature_groups(feature_df)

            if self.feature_weights is None:
                print("Setting feature weights (using defaults)...")
                self.set_feature_weights() # Uses defaults if no base_weights provided


            # --- 3. Check Data Quality (Inf values) ---
            numeric_cols = feature_df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                inf_counts = np.isinf(feature_df[numeric_cols].values).sum()
                if inf_counts > 0:
                    print(f"Warning: Found {inf_counts} infinite values. Replacing with large finite values (1e9 / -1e9).")
                    # Replace directly in the DataFrame before scaling
                    feature_df.replace([np.inf, -np.inf], [1e9, -1e9], inplace=True)


            # --- 4. Standardize Data ---
            print("Standardizing data using StandardScaler...")
            # Fit the scaler and transform the data
            X_scaled = self.scaler.fit_transform(feature_df)
            # Check for NaNs introduced by scaler (shouldn't happen with prior checks)
            if np.isnan(X_scaled).any():
                print("ERROR: NaNs found after scaling! Check data for columns with zero variance.")
                # Option: Try to fix or raise error
                nan_cols_scaled = feature_df.columns[np.isnan(X_scaled).any(axis=0)].tolist()
                print(f"Columns with NaNs after scaling: {nan_cols_scaled}")
                print("Attempting to fill NaNs post-scaling with 0...")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0) # Fill NaNs potentially from zero-variance cols
                # raise ValueError("NaN values found after scaling data.")


            # --- 5. Apply Weights ---
            print("Applying feature weights...")
            self.X_weighted = self.apply_weights(X_scaled)

            # Verify no NaNs/Infs in weighted matrix
            if np.isnan(self.X_weighted).any():
                print(f"ERROR: NaN values found in weighted feature matrix! Count: {np.isnan(self.X_weighted).sum()}")
                print("Attempting to fill NaNs in weighted matrix with 0...")
                self.X_weighted = np.nan_to_num(self.X_weighted, nan=0.0)
                # raise ValueError("NaN values in weighted feature matrix.")
            if np.isinf(self.X_weighted).any():
                 print(f"ERROR: Infinite values found in weighted feature matrix! Count: {np.isinf(self.X_weighted).sum()}")
                 print("Attempting to replace Infs in weighted matrix with large values...")
                 self.X_weighted = np.nan_to_num(self.X_weighted, posinf=1e9, neginf=-1e9)
                 # raise ValueError("Infinite values in weighted feature matrix.")


            # --- 6. Fit KNN Model ---
            # Ensure n_neighbors is not larger than the number of samples
            effective_n_neighbors = min(self.n_neighbors, self.X_weighted.shape[0] - 1)
            if effective_n_neighbors <= 0:
                 raise ValueError(f"Cannot fit KNN: Number of samples ({self.X_weighted.shape[0]}) is too small for n_neighbors={self.n_neighbors}.")
            if effective_n_neighbors < self.n_neighbors:
                 print(f"Warning: n_neighbors ({self.n_neighbors}) is >= number of samples. Setting n_neighbors to {effective_n_neighbors}.")

            print(f"Fitting KNN model with k={effective_n_neighbors + 1} (to account for self)...")
            self.knn_model = NearestNeighbors(
                n_neighbors=effective_n_neighbors + 1, # +1 because the item itself is its nearest neighbor
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
            source_product_ids = self.product_ids.tolist()

            for i in range(len(self.indices)):
                source_product_id = source_product_ids[i]

                # Indices of neighbors (row i in self.indices)
                # Distances of neighbors (row i in self.distances)
                neighbor_indices = self.indices[i]
                neighbor_distances = self.distances[i]

                product_recs = []
                for j in range(1, len(neighbor_indices)): # Start from 1 to skip self (index 0)
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

                    product_recs.append((recommended_product_id, similarity))

                # Store the recommendations, ensuring we have up to n_neighbors
                self.recommendation_matrix[source_product_id] = product_recs[:self.n_neighbors]

                # Check if enough recommendations were generated
                if len(self.recommendation_matrix[source_product_id]) < self.n_neighbors:
                     # This might happen due to self-filtering or reaching end of neighbors
                     # print(f"Note: Generated {len(self.recommendation_matrix[source_product_id])} recommendations for {source_product_id} (less than {self.n_neighbors}).")
                     pass # Not necessarily an error


            fit_end_time = time.time()
            print(f"\nSuccessfully fitted KNN model to {len(feature_df)} products in {fit_end_time - fit_start_time:.2f} seconds.")
            print(f"Each product has up to {self.n_neighbors} recommendations stored.")

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
            print(f"Warning: Product ID '{product_id}' not found in the recommendation matrix.")
            return []

        # Use the stored n_neighbors if n_recommendations is not specified
        if n_recommendations is None:
            n_recommendations = self.n_neighbors
        elif n_recommendations <= 0:
             return [] # Requesting zero or negative recommendations

        # Get precomputed recommendations for the product
        # The list should already be sorted by similarity (desc) from the fit step
        recommendations = self.recommendation_matrix[product_id]

        # Although self-recs *should* be filtered during fit, double-check here just in case
        filtered_recommendations = [(rec_id, sim) for rec_id, sim in recommendations if rec_id != product_id]

        # Return the top N requested recommendations
        return filtered_recommendations[:n_recommendations]

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
            importance_series = pd.Series(feature_importance_map)

            # Ensure the Series is sorted by the original feature order for consistency? Not needed for bar plot.
            # Sort by importance value for plotting
            importance_series = importance_series.sort_values(ascending=False)


            # Get top N features to display (e.g., top 30 or all if fewer)
            num_features_to_show = min(30, len(importance_series))
            top_features = importance_series.head(num_features_to_show)

            # Create a horizontal bar chart
            plt.figure(figsize=(12, max(8, num_features_to_show * 0.4))) # Adjust height based on N
            y_pos = np.arange(len(top_features))

            # Use the feature names directly from the Series index
            display_names = top_features.index.tolist()

            # Shorten long names for display
            short_display_names = []
            for name in display_names:
                 # Simple shortening logic (can be customized)
                 prefix_map = {
                      'description_': 'Desc: ', 'metafield_': 'Meta: ', 'tag_': 'Tag: ',
                      'product_type_': 'Type: ', 'collections_': 'Coll: ', 'vendor_': 'Vend: ',
                      'variant_size_': 'Size: ', 'variant_color_': 'Color: ','variant_other_': 'Var: ',
                      'created_': 'Time: ', 'release_quarter_': 'Quarter: '
                 }
                 display_name = name
                 for prefix, short_prefix in prefix_map.items():
                      if name.startswith(prefix):
                           display_name = short_prefix + name[len(prefix):]
                           break # Stop after first match

                 if len(display_name) > 40: # Max length
                      display_name = display_name[:37] + "..."
                 short_display_names.append(display_name)


            # Plotting
            plt.barh(y_pos, top_features.values, align='center')
            plt.yticks(y_pos, short_display_names)
            plt.gca().invert_yaxis() # Display highest importance at the top
            plt.xlabel('Assigned Effective Feature Weight')
            plt.title(f'Top {num_features_to_show} Features by Assigned Group Weight')
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            plt.savefig(output_file)
            plt.close() # Close the plot to free memory

            print(f"Feature importance visualization saved to {output_file}")

        except Exception as e:
            print(f"Error during feature importance visualization: {type(e).__name__} - {str(e)}")
            print("Skipping feature importance visualization.")

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
            num_samples = X_to_plot.shape[0]
            indices = np.arange(num_samples) # Keep track of original indices

            if num_samples > sample_size:
                print(f"Sampling {sample_size} points out of {num_samples} for t-SNE visualization...")
                # Use random choice for sampling
                indices = np.random.choice(num_samples, sample_size, replace=False)
                X_sample = X_to_plot[indices]
            else:
                print(f"Using all {num_samples} points for t-SNE visualization.")
                X_sample = X_to_plot
                # indices remain 0 to num_samples-1

            if X_sample.shape[0] <= 1:
                 print("Skipping t-SNE: Not enough samples after potential sampling.")
                 return


            # --- Adjust t-SNE Perplexity ---
            # Perplexity should be less than the number of samples
            effective_perplexity = min(perplexity, X_sample.shape[0] - 1)
            if effective_perplexity <= 0:
                print(f"Warning: Sample size ({X_sample.shape[0]}) too small for perplexity. Setting perplexity to 5 (or sample_size-1 if smaller).")
                effective_perplexity = min(5, max(1, X_sample.shape[0] - 1)) # Ensure perplexity is at least 1

            if effective_perplexity != perplexity:
                 print(f"Adjusted t-SNE perplexity to {effective_perplexity}.")


            # --- Apply t-SNE ---
            print(f"Applying t-SNE (perplexity={effective_perplexity})... (this may take a moment)")
            tsne_start_time = time.time()
            tsne = TSNE(n_components=2, random_state=self.random_state,
                        perplexity=effective_perplexity, n_iter=300, # Lower iterations for speed?
                        init='pca', # PCA initialization is often faster and stabler
                        learning_rate='auto', # Recommended in recent sklearn
                        n_jobs=-1) # Use all cores
            X_tsne = tsne.fit_transform(X_sample)
            tsne_end_time = time.time()
            print(f"t-SNE computation finished in {tsne_end_time - tsne_start_time:.2f} seconds.")


            # --- Determine Point Colors (Optional - Example using first neighbor) ---
            # This coloring strategy aims to group products that recommend similar items.
            # It requires the recommendation matrix or indices to be available.
            point_colors = np.zeros(len(X_sample)) # Default color if indices are missing
            if self.indices is not None and len(self.indices) == num_samples:
                try:
                    # Get the indices of the *sampled* points
                    sampled_neighbor_indices = self.indices[indices]

                    # Use the index of the *first* valid neighbor (excluding self) to assign a color group
                    # Taking modulo a number limits the distinct colors used by the colormap
                    first_neighbor_indices = sampled_neighbor_indices[:, 1] # Index 1 is the first neighbor after self
                    point_colors = first_neighbor_indices % 20 # Use 20 distinct colors from cmap

                except IndexError:
                    print("Warning: Could not determine point colors based on neighbors due to index mismatch. Using default color.")
                except Exception as e:
                     print(f"Warning: Error determining point colors ({e}). Using default color.")
            else:
                 print("Warning: Neighbor indices not available or size mismatch. Using default color for t-SNE plot.")


            # --- Plotting ---
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors, cmap='tab20', alpha=0.7, s=20) # Smaller points?
            # Optional: Add color bar if colors represent something meaningful
            # plt.colorbar(scatter, label='Nearest Neighbor Group (Modulo 20)')
            plt.title(f't-SNE Visualization of {len(X_sample)} Products (Weighted Features)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.xticks([]) # Hide axis ticks for cleaner look
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(output_file)
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
        # We can use the pre-calculated recommendation_matrix which stores similarities
        if self.recommendation_matrix is None:
            print("Cannot visualize similarity distribution: Recommendation matrix not built. Run fit() first.")
            # Alternative: Could try using self.distances if available
            # if self.distances is None:
            #      print("Cannot visualize similarity distribution: Distances not available. Run fit() first.")
            #      return
            # # Calculate similarities from distances if matrix is missing
            # print("Warning: Recommendation matrix missing, calculating similarities from distances...")
            # # Exclude self-distance (column 0) and handle potential inf distances
            # valid_distances = self.distances[:, 1:]
            # valid_distances = np.nan_to_num(valid_distances, nan=np.inf, posinf=np.inf, neginf=0.0) # Treat NaN/neg as far
            # flat_similarities = 1.0 / (1.0 + valid_distances.flatten() + 1e-9)
            # flat_similarities = flat_similarities[np.isfinite(flat_similarities)] # Remove any remaining Infs
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

            # --- Plotting Histogram ---
            plt.figure(figsize=(10, 6))
            plt.hist(flat_similarities, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
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

            plt.tight_layout()
            plt.savefig(output_file)
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
        if self.knn_model is None or self.scaler is None or self.recommendation_matrix is None:
             print("Warning: Model is not fully fitted or essential components are missing. Saving incomplete model.")
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
                # Recommendation matrix can be large, consider compression?
                joblib.dump(self.recommendation_matrix, os.path.join(output_dir, "recommendation_matrix.pkl"))

            # --- Save Metadata and Parameters (use JSON or pickle) ---
            model_params = {
                'feature_weights': self.feature_weights,
                'feature_groups': self.feature_groups,
                'feature_names_in_order': self.feature_names_in_order, # Save feature order!
                'n_neighbors': self.n_neighbors,
                'algorithm': self.algorithm,
                'metric': self.metric,
                'random_state': self.random_state,
                # Store product IDs and titles if they were loaded
                'has_product_ids': self.product_ids is not None,
                'has_product_titles': self.product_titles is not None,
            }
            # Save parameters using pickle (easier for complex structures like dicts of lists)
            joblib.dump(model_params, os.path.join(output_dir, "model_params.pkl"))

            # Optionally save IDs/titles if they exist (can be large)
            if self.product_ids is not None:
                 joblib.dump(self.product_ids, os.path.join(output_dir, "product_ids.pkl"))
            # if self.product_titles is not None:
            #      joblib.dump(self.product_titles, os.path.join(output_dir, "product_titles.pkl"))


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
            if os.path.exists(knn_path):
                self.knn_model = joblib.load(knn_path)
            else:
                print(f"Warning: KNN model file not found at {knn_path}")
                load_successful = False

            scaler_path = os.path.join(input_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print(f"Warning: Scaler file not found at {scaler_path}")
                load_successful = False

            rec_path = os.path.join(input_dir, "recommendation_matrix.pkl")
            if os.path.exists(rec_path):
                self.recommendation_matrix = joblib.load(rec_path)
            else:
                # Recommendation matrix might be optional if you intend to re-calculate neighbors
                print(f"Warning: Recommendation matrix file not found at {rec_path}. Recommendations need recalculation if required.")
                # self.recommendation_matrix = None # Explicitly set to None

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

                # Load product IDs/titles if indicated
                if params.get('has_product_ids'):
                     ids_path = os.path.join(input_dir, "product_ids.pkl")
                     if os.path.exists(ids_path):
                          self.product_ids = joblib.load(ids_path)
                     else:
                          print("Warning: Model params indicate product IDs should exist, but product_ids.pkl not found.")
                          self.product_ids = None
                # if params.get('has_product_titles'):
                #      titles_path = os.path.join(input_dir, "product_titles.pkl")
                #      if os.path.exists(titles_path):
                #           self.product_titles = joblib.load(titles_path)
                #      else:
                #           print("Warning: Model params indicate product titles should exist, but product_titles.pkl not found.")
                #           self.product_titles = None

                # --- Crucial Check: Feature Order ---
                if self.feature_names_in_order is None:
                     print("CRITICAL WARNING: Loaded model parameters do not contain 'feature_names_in_order'. Applying weights or using the model might yield incorrect results if feature order has changed.")
                     load_successful = False # Consider this a critical failure for reliable use


            else:
                print(f"Warning: Model parameters file not found at {params_path}. Model metadata is missing.")
                load_successful = False # Metadata is essential

            if load_successful:
                print("Model components loaded successfully.")
            else:
                print("Model loading finished with warnings or missing components. Model might be incomplete or unusable.")

            return self

        except Exception as e:
            print(f"Error loading model from {input_dir}: {type(e).__name__} - {str(e)}")
            import traceback
            traceback.print_exc()
            # raise e # Optionally re-raise
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

        if n_recommendations is None:
            n_recommendations = self.n_neighbors
        elif n_recommendations <= 0:
             print("Warning: n_recommendations requested is <= 0. Exporting empty file.")
             # Create and save empty DataFrame with correct headers
             pd.DataFrame(columns=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']).to_csv(output_file, index=False)
             return pd.DataFrame()

        # Prepare data for export
        rows = []
        total_variant_filtered_count = 0 # Track how many were filtered as variants
        total_self_recs_filtered = 0 # Track self-recs removed (should be 0 if fit() is correct)

        # Determine which columns to use in similar_products_df
        use_variant_filter = False
        col1_id, col2_id = None, None
        if similar_products_df is not None and not similar_products_df.empty:
            # Try standard names first
            if 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns:
                col1_id, col2_id = 'product1_id', 'product2_id'
                use_variant_filter = True
            # Fallback to names without '_id'
            elif 'product1' in similar_products_df.columns and 'product2' in similar_products_df.columns:
                col1_id, col2_id = 'product1', 'product2'
                use_variant_filter = True
            else:
                print("Warning: Could not find required columns ('product1_id'/'product1', 'product2_id'/'product2') in similar_products_df. Variant filtering disabled.")
                use_variant_filter = False


        # --- Iterate through each product and its recommendations ---
        print(f"Exporting top {n_recommendations} recommendations for {len(self.recommendation_matrix)} products...")
        for i, (source_product_id, recommendations) in enumerate(self.recommendation_matrix.items()):

            # Progress indicator
            if (i + 1) % 500 == 0 or i == 0 or i == len(self.recommendation_matrix) - 1:
                 print(f"Processing product {i+1}/{len(self.recommendation_matrix)}: {source_product_id}")

            # 1. Initial Filter: Remove self-recommendations (should be redundant)
            current_recs = [(rec_id, sim) for rec_id, sim in recommendations if rec_id != source_product_id]
            self_filtered = len(recommendations) - len(current_recs)
            if self_filtered > 0:
                 total_self_recs_filtered += self_filtered
                 # print(f"Note: Removed {self_filtered} self-recommendation(s) for {source_product_id} during export.")


            # 2. Filter Variants (if enabled and applicable)
            variants_filtered_this_product = 0
            if use_variant_filter:
                exclude_ids = set()
                try:
                    # Find IDs similar to the source product
                    matches1 = similar_products_df[similar_products_df[col1_id] == source_product_id]
                    if not matches1.empty:
                        exclude_ids.update(matches1[col2_id].tolist())

                    matches2 = similar_products_df[similar_products_df[col2_id] == source_product_id]
                    if not matches2.empty:
                        exclude_ids.update(matches2[col1_id].tolist())

                except KeyError as e:
                     print(f"Error accessing columns '{col1_id}' or '{col2_id}' while filtering variants for {source_product_id}. Filtering skipped for this product. Error: {e}")
                     exclude_ids = set() # Reset exclude_ids on error
                except Exception as e: # Catch other potential errors during filtering
                     print(f"Unexpected error filtering variants for {source_product_id}: {e}")
                     exclude_ids = set()


                if exclude_ids:
                    # Apply the filter
                    count_before_variant_filter = len(current_recs)
                    current_recs = [(rec_id, sim) for rec_id, sim in current_recs if rec_id not in exclude_ids]
                    variants_filtered_this_product = count_before_variant_filter - len(current_recs)
                    total_variant_filtered_count += variants_filtered_this_product
                    # if variants_filtered_this_product > 0: # Optional: log per-product filtering
                    #     print(f"  - Filtered {variants_filtered_this_product} variant(s) for {source_product_id}")


            # 3. Take Top N recommendations *after* filtering
            final_recs_for_product = current_recs[:n_recommendations]

            # 4. Add to export rows with rank
            for rank, (rec_id, similarity) in enumerate(final_recs_for_product):
                 # Final check (should be unnecessary now but safe)
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
            # Add product titles if possible
            if products_df is not None and 'product_id' in products_df.columns and 'product_title' in products_df.columns:
                 print("Adding product titles to exported recommendations...")
                 title_map = products_df.set_index('product_id')['product_title'].to_dict()
                 recommendations_df['source_product_title'] = recommendations_df['source_product_id'].map(title_map).fillna('Title Not Found')
                 recommendations_df['recommended_product_title'] = recommendations_df['recommended_product_id'].map(title_map).fillna('Title Not Found')
                 # Reorder columns
                 cols_order = ['source_product_id', 'source_product_title', 'recommended_product_id', 'recommended_product_title', 'rank', 'similarity_score']
                 # Ensure all expected columns exist before reordering
                 recommendations_df = recommendations_df[[col for col in cols_order if col in recommendations_df.columns]]


            # Final check for self-recommendations (should never happen)
            self_recs_final = recommendations_df[recommendations_df['source_product_id'] == recommendations_df['recommended_product_id']]
            if not self_recs_final.empty:
                print(f"\nCRITICAL ERROR: Found {len(self_recs_final)} self-recommendations in the final export DataFrame! This indicates a bug.")
                print("Attempting to remove them before saving...")
                recommendations_df = recommendations_df[recommendations_df['source_product_id'] != recommendations_df['recommended_product_id']]

            try:
                recommendations_df.to_csv(output_file, index=False)
                export_end_time = time.time()
                print(f"\nRecommendations exported successfully to {output_file} in {export_end_time - export_start_time:.2f} seconds.")
                print(f"Exported {len(recommendations_df)} recommendation pairs.")
                if total_variant_filtered_count > 0:
                    print(f"Total recommendations filtered out as similar variants: {total_variant_filtered_count}")
                if total_self_recs_filtered > 0:
                     print(f"Warning: Total self-recommendations filtered during export: {total_self_recs_filtered} (This might indicate duplicate products in input data).")


            except Exception as e:
                print(f"\nError saving recommendations DataFrame to {output_file}: {e}")
                # Return the DataFrame even if saving failed
                return recommendations_df

        else:
            print("\nWARNING: No valid recommendations generated after filtering. Export file will be empty or not created.")
            # Optionally save an empty file with headers
            try:
                 pd.DataFrame(columns=['source_product_id', 'recommended_product_id', 'rank', 'similarity_score']).to_csv(output_file, index=False)
            except Exception as e:
                 print(f"Error saving empty recommendations file: {e}")


        return recommendations_df


# --- WeightOptimizer Class (No changes needed based on the request, keeping as is) ---
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

        # Keep non-feature columns (adjust if needed)
        self.non_feature_cols = ['product_id', 'product_title', 'handle', 'description']
        # Features are identified dynamically by _init_base_model

        # Initialize a base model to get feature groups and prepare feature DataFrame
        self._init_base_model()

    def _init_base_model(self):
        """Initialize a base KNN model, identify feature groups, and prepare feature df."""
        print("Initializing Weight Optimizer Base Model...")
        # Use a temporary model instance just for setup
        temp_model = _WeightedKNN(n_neighbors=self.n_neighbors, random_state=self.random_state)

        # Extract potential features (numeric only) - mimicking load_data logic
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        # Remove known non-features explicitly if they are numeric
        numeric_cols = [col for col in numeric_cols if col not in self.non_feature_cols]

        if not numeric_cols:
             raise ValueError("WeightOptimizer: No numeric features found in the input DataFrame.")

        # Create the feature DataFrame used for all optimization runs
        self.feature_df = self.df[numeric_cols].copy()
        temp_model.feature_names_in_order = self.feature_df.columns.tolist() # Store feature order
        self.feature_df = temp_model.handle_missing_values(self.feature_df) # Handle NaNs

        # Store product IDs and titles from the original DataFrame
        self.product_ids = self.df['product_id'].copy() if 'product_id' in self.df.columns else pd.Series(self.df.index).astype(str)
        self.product_titles = self.df['product_title'].copy() if 'product_title' in self.df.columns else None

        # Identify feature groups using the prepared feature DataFrame
        self.feature_groups = temp_model.identify_feature_groups(self.feature_df)

        # Define default weights (can be customized) - ensure all groups are covered
        self.default_weights = {
            'description_tfidf': 1.0, 'metafield_data': 1.0, 'collections': 1.4,
            'tags': 1.8, 'product_type': 1.8, 'vendor': 1.0,
            'variant_size': 1.0, 'variant_color': 1.0, 'variant_other': 1.0,
            'price': 0.8, 'numerical': 0.7, 'time_features': 0.5,
            'release_quarter': 0.5, 'seasons': 0.8, 'availability': 0.6,
            'gift_card': 0.1, 'other': 0.5
        }
        # Ensure all identified groups have a default weight
        for group in self.feature_groups:
             if group not in self.default_weights:
                  self.default_weights[group] = 1.0 # Assign default if missing

        # Evaluate default weights as baseline
        print("\nEvaluating default weights as baseline...")
        baseline_score = self.evaluate_weights(self.default_weights, "default_baseline", save_model_if_best=True)
        if baseline_score > self.best_score:
             print(f"Default weights established baseline score: {baseline_score:.4f}")
        else:
             # This case should ideally not happen if best_score starts low
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
        model.product_ids = self.product_ids
        model.product_titles = self.product_titles
        model.feature_groups = self.feature_groups # Use the identified groups
        model.feature_names_in_order = self.feature_df.columns.tolist() # Use the defined feature order

        try:
            # Set the weights for this evaluation (normalize by count is often good for comparison)
            model.set_feature_weights(weights, normalize_by_count=True)

            # Fit the model using the prepared feature DataFrame
            # Pass the DataFrame directly, not a path
            model.fit(self.feature_df)

            # Ensure the model fitted successfully before calculating similarity
            if model.recommendation_matrix is None:
                 print(f"ERROR: Model fitting failed for weights '{weight_name}'. Cannot calculate similarity.")
                 return 0.0

            # Calculate average similarity score
            avg_similarity = model.calculate_average_similarity()

            # Record the result
            result = {
                'weight_name': weight_name,
                'weights': weights.copy(), # Store the base weights used
                'avg_similarity': avg_similarity,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'eval_time_seconds': time.time() - eval_start_time
            }
            self.optimization_results.append(result)

            print(f"--- Result for '{weight_name}': Avg Similarity = {avg_similarity:.4f} (Eval time: {result['eval_time_seconds']:.2f}s) ---")

            # Update best if this is better
            # Use a small tolerance for comparison? e.g., >= self.best_score + 1e-6
            if avg_similarity > self.best_score:
                print(f"*** New best weights found! Avg similarity: {avg_similarity:.4f} (Previous best: {self.best_score:.4f}) ***")
                self.best_score = avg_similarity
                self.best_weights = weights.copy() # Store the base weights that led to this score
                if save_model_if_best:
                    print("Saving this model instance as the best found so far.")
                    self.best_model = model # Store the entire fitted model object
                # else: # Ensure best_model is None if not saving this one
                #     self.best_model = None # Or keep the previous best if it existed? Decide strategy.


            return avg_similarity

        except Exception as e:
            print(f"\n--- ERROR evaluating weights '{weight_name}' ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            print("-------------------------------------------------")
            # Record failure in results?
            result = {
                 'weight_name': weight_name, 'weights': weights.copy(), 'avg_similarity': 0.0,
                 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'eval_time_seconds': time.time() - eval_start_time,
                 'error': str(e)
            }
            self.optimization_results.append(result)
            return 0.0 # Return 0.0 similarity score on error

    def grid_search(self, weight_options=None, fixed_weights=None, save_results=True):
        """
        Perform grid search over specified weight combinations for feature groups.

        Args:
            weight_options (dict): {group_name: [list_of_weights_to_try], ...}.
                                   If None or empty, uses a default range for all groups.
            fixed_weights (dict): {group_name: fixed_weight_value, ...}. These groups
                                  will not be varied.
            save_results (bool): Whether to save detailed results to CSV upon completion.

        Returns:
            pd.DataFrame: DataFrame containing the results of the grid search.
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
            if combination_count % 10 == 0 or combination_count == 1 or combination_count == total_combinations:
                 elapsed_time = time.time() - grid_start_time
                 estimated_total_time = (elapsed_time / combination_count) * total_combinations if combination_count > 0 else 0
                 remaining_time = estimated_total_time - elapsed_time
                 print(f"\n[Grid Search Progress: {combination_count}/{total_combinations}] "
                       f"Elapsed: {elapsed_time:.1f}s | Est. Remaining: {remaining_time:.1f}s")

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

        Args:
            n_iterations: Number of random weight combinations to try.
            weight_range: Tuple (min_weight, max_weight) for random generation.
            save_results: Whether to save detailed results to CSV upon completion.

        Returns:
            pd.DataFrame: DataFrame containing the results of the random search.
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
            if iteration_count % 10 == 0 or iteration_count == 1 or iteration_count == n_iterations:
                elapsed_time = time.time() - random_start_time
                estimated_total_time = (elapsed_time / iteration_count) * n_iterations if iteration_count > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                print(f"\n[Random Search Progress: {iteration_count}/{n_iterations}] "
                      f"Elapsed: {elapsed_time:.1f}s | Est. Remaining: {remaining_time:.1f}s")


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

        Args:
            population_size: Number of individuals (weight sets) in each generation.
            generations: Number of generations to evolve.
            mutation_rate: Probability of a single weight mutating.
            crossover_rate: Probability of two parents performing crossover.
            tournament_size: Number of individuals competing in selection tournaments.
            weight_range: Tuple (min_weight, max_weight) for weights.
            save_results: Whether to save detailed results to CSV upon completion.

        Returns:
            pd.DataFrame: DataFrame containing the results of the evolutionary search.
        """
        all_groups = list(self.feature_groups.keys())
        if not all_groups:
             print("Error: No feature groups identified. Cannot perform evolutionary search.")
             return self.get_optimization_results()
        if population_size <= 0 or generations <= 0:
             print("Error: Population size and generations must be positive.")
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
        for i in range(population_size):
            # Generate random weights
            weights = {group: np.random.uniform(weight_range[0], weight_range[1]) for group in all_groups}
            individual_name = f"evo_gen0_ind{i+1}"
            # Evaluate fitness (similarity score) - save if best
            fitness = self.evaluate_weights(weights, individual_name, save_model_if_best=True)
            population.append({'weights': weights, 'fitness': fitness, 'name': individual_name})

        # Sort initial population by fitness (descending)
        population.sort(key=lambda x: x['fitness'], reverse=True)
        print(f"Generation 0 Initialized. Best fitness: {population[0]['fitness']:.4f}")

        # --- Evolution Loop ---
        total_evals = population_size # Track total evaluations
        for gen in range(1, generations + 1):
            gen_start_time = time.time()
            print(f"\n--- Starting Generation {gen}/{generations} ---")

            new_population = []

            # Elitism: Keep the best 'n_elite' individuals (e.g., 1 or 2)
            n_elite = 1
            if n_elite > 0 and len(population) > 0:
                 print(f"Applying elitism: Keeping best {n_elite} individual(s).")
                 for i in range(min(n_elite, len(population))):
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
                    # If no crossover, randomly pick one parent's weights to pass on (or average?)
                    child_weights = parent1['weights'].copy() # Simple approach: clone parent 1

                # Mutation: Apply mutation to the child's weights
                self._mutate(child_weights, mutation_rate, weight_range)

                # Evaluate fitness of the new child
                total_evals += 1
                child_name = f"evo_gen{gen}_ind{len(new_population)+1}"
                child_fitness = self.evaluate_weights(child_weights, child_name, save_model_if_best=True)

                # Add the new child to the next generation's population
                new_population.append({'weights': child_weights, 'fitness': child_fitness, 'name': child_name})

            # Replace old population with the new one
            population = new_population

            # Sort the new generation by fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)

            gen_end_time = time.time()
            print(f"--- Generation {gen} Completed (Time: {gen_end_time - gen_start_time:.2f}s) ---")
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
        if not population:
            raise ValueError("Cannot perform tournament selection on an empty population.")
        if tournament_size <= 0:
             tournament_size = 1 # Ensure size is at least 1
        actual_tournament_size = min(tournament_size, len(population)) # Cannot pick more than available

        # Randomly select individuals for the tournament without replacement
        tournament_indices = np.random.choice(len(population), size=actual_tournament_size, replace=False)
        tournament_participants = [population[i] for i in tournament_indices]

        # Return the individual with the highest fitness from the tournament
        return max(tournament_participants, key=lambda x: x['fitness'])

    def _crossover(self, weights1, weights2):
        """Helper for evolutionary search: Performs uniform crossover between two weight sets."""
        child_weights = {}
        all_groups = list(weights1.keys()) # Assume both parents have the same groups

        for group in all_groups:
            # Randomly choose the weight for this group from either parent
            if np.random.random() < 0.5:
                child_weights[group] = weights1[group]
            else:
                child_weights[group] = weights2[group]

        return child_weights

    def _mutate(self, weights, mutation_rate, weight_range):
        """Helper for evolutionary search: Mutates weights in place with a given probability."""
        for group in weights:
            if np.random.random() < mutation_rate:
                # Apply mutation: replace with a new random value in the allowed range
                weights[group] = np.random.uniform(weight_range[0], weight_range[1])
        # No return needed as it modifies the dictionary in place


    def get_optimization_results(self):
        """
        Get all optimization run results compiled into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns for weight_name, avg_similarity, timestamp,
                          and individual weight columns (e.g., weight_tags, weight_vendor).
                          Returns empty DataFrame if no results exist.
        """
        if not self.optimization_results:
            print("No optimization results recorded yet.")
            return pd.DataFrame()

        # Prepare data for DataFrame conversion
        results_list_for_df = []
        # Get all unique group names encountered across all results (robustness)
        all_groups = set()
        for result in self.optimization_results:
             if 'weights' in result and isinstance(result['weights'], dict):
                  all_groups.update(result['weights'].keys())
        sorted_groups = sorted(list(all_groups)) # Consistent column order


        for result in self.optimization_results:
             row_data = {
                 'weight_name': result.get('weight_name', 'N/A'),
                 'avg_similarity': result.get('avg_similarity', 0.0),
                 'timestamp': result.get('timestamp', 'N/A'),
                 'eval_time_seconds': result.get('eval_time_seconds', np.nan),
                 'error': result.get('error', None) # Include error message if present
             }
             # Add weight columns
             base_weights = result.get('weights', {}) # Get the base weights used
             for group in sorted_groups:
                  row_data[f"weight_{group}"] = base_weights.get(group, np.nan) # Use NaN if group wasn't in this specific result

             results_list_for_df.append(row_data)


        # Create DataFrame
        results_df = pd.DataFrame(results_list_for_df)

        # Sort by average similarity (descending), handle potential NaNs in sorting
        if 'avg_similarity' in results_df.columns:
            results_df = results_df.sort_values('avg_similarity', ascending=False, na_position='last')

        return results_df

    def save_optimization_results(self, filename=None):
        """
        Save optimization results (including weights) to a CSV file and best weights to JSON.

        Args:
            filename: Custom filename for the CSV (default: optimization_results.csv in output_dir).
        """
        results_df = self.get_optimization_results()

        if results_df.empty:
            print("No optimization results to save.")
            return

        if filename is None:
            filename = os.path.join(self.output_dir, "optimization_results.csv")

        try:
            results_df.to_csv(filename, index=False)
            print(f"Optimization results saved to {filename}")
        except Exception as e:
             print(f"Error saving optimization results CSV to {filename}: {e}")


        # Also save the best weights found to a separate JSON file for easy loading
        if self.best_weights:
            best_weights_file = os.path.join(self.output_dir, "best_weights.json")
            try:
                with open(best_weights_file, 'w') as f:
                    json.dump(self.best_weights, f, indent=2)
                print(f"Best weights (achieving score {self.best_score:.4f}) saved to {best_weights_file}")
            except Exception as e:
                 print(f"Error saving best weights JSON to {best_weights_file}: {e}")
        else:
             print("No best weights recorded during optimization.")

    def visualize_optimization_results(self, top_n=20, filename=None):
        """
        Create a bar plot showing the performance of the top N weight configurations.

        Args:
            top_n: Number of top results to display.
            filename: Custom filename for the plot (default: optimization_results.png in output_dir).
        """
        results_df = self.get_optimization_results()

        if results_df.empty:
            print("No optimization results to visualize.")
            return

        # Get top N results (already sorted)
        # Exclude results with errors if an 'error' column exists and is not null
        if 'error' in results_df.columns:
             plot_df = results_df[results_df['error'].isnull()].head(top_n)
        else:
             plot_df = results_df.head(top_n)


        if plot_df.empty:
             print("No valid (non-error) optimization results to visualize.")
             return

        # Plot
        plt.figure(figsize=(12, max(6, len(plot_df) * 0.4))) # Adjust height
        sns.barplot(x='avg_similarity', y='weight_name', data=plot_df, palette='viridis')
        plt.title(f'Top {len(plot_df)} Weight Configurations by Average Similarity')
        plt.xlabel('Average Similarity Score')
        plt.ylabel('Weight Configuration Name')
        plt.tight_layout()

        # Save plot
        if filename is None:
            filename = os.path.join(self.output_dir, "optimization_results.png")

        try:
            plt.savefig(filename)
            plt.close() # Close plot
            print(f"Optimization results visualization saved to {filename}")
        except Exception as e:
             print(f"Error saving optimization results plot to {filename}: {e}")


    def build_optimized_model(self, save_model_files=True):
        """
        Build and fit a new _WeightedKNN model using the best weights found during optimization.

        Args:
            save_model_files: Whether to save the components of the final optimized model to disk.

        Returns:
            _WeightedKNN: The fitted model instance using the best weights, or None if optimization failed.
        """
        print("\n--- Building Final Model with Optimized Weights ---")
        if self.best_weights is None:
            print("Warning: No best weights found during optimization.")
            # Option 1: Fallback to default weights
            print("Falling back to default weights.")
            weights_to_use = self.default_weights
            if not weights_to_use: # Ensure defaults were set
                 print("Error: Cannot build model, default weights also missing.")
                 return None
            # Option 2: Return None or raise error
            # print("Error: Cannot build optimized model without best weights.")
            # return None
        else:
            print(f"Using best weights found (Avg Similarity: {self.best_score:.4f})")
            # print(json.dumps(self.best_weights, indent=2)) # Print best weights
            weights_to_use = self.best_weights

        # Check if we stored the best model instance directly during optimization
        # This avoids re-fitting if `save_model_if_best=True` was used effectively
        if self.best_model and self.best_model.recommendation_matrix is not None:
             print("Using the pre-fitted best model instance directly.")
             model = self.best_model
        else:
             # If no pre-fitted model stored, or it seems invalid, build and fit a new one
             print("Building and fitting a new model instance with the best weights...")
             model = _WeightedKNN(n_neighbors=self.n_neighbors, random_state=self.random_state)

             # Set essential attributes
             model.product_ids = self.product_ids
             model.product_titles = self.product_titles
             model.feature_groups = self.feature_groups
             model.feature_names_in_order = self.feature_df.columns.tolist()

             # Set the best weights found
             model.set_feature_weights(weights_to_use, normalize_by_count=True) # Normalize usually intended

             # Fit the model
             try:
                 model.fit(self.feature_df)
                 if model.recommendation_matrix is None:
                      print("Error: Fitting the final optimized model failed.")
                      return None # Return None if fit fails
             except Exception as e:
                  print(f"Error fitting the final optimized model: {e}")
                  return None


        # Save the final optimized model components if requested
        if save_model_files:
            save_dir = os.path.join(self.output_dir, "optimized_model_final")
            print(f"\nSaving the final optimized model components to: {save_dir}")
            model.save_model(save_dir)

        print("--- Optimized Model Building Complete ---")
        return model


# --- Standalone Function: find_similar_products (Uses a fitted model) ---

def find_similar_products(product_id, knn_model, products_df, top_n=12,
                        exclude_similar_variants=True, similar_products_df=None):
    """
    Find similar products for a given product ID using a fitted _WeightedKNN model.
    Includes logic to filter out known similar variants.

    Args:
        product_id: The ID of the product to find similar items for.
        knn_model: A fitted _WeightedKNN model instance.
        products_df: DataFrame containing original product data (must have 'product_id',
                     optionally 'product_title').
        top_n: The maximum number of similar products to return.
        exclude_similar_variants: If True, use similar_products_df to filter variants.
        similar_products_df: DataFrame with pairs of too-similar products (variants).
                               Expected columns like 'product1_id', 'product2_id'.

    Returns:
        pd.DataFrame: DataFrame of similar products with columns like
                      'product_id', 'similarity', 'product_title', or empty DataFrame on error/no results.
    """
    if not isinstance(knn_model, _WeightedKNN) or knn_model.recommendation_matrix is None:
        print(f"Error: Invalid or unfitted KNN model provided for finding similar products to {product_id}.")
        return pd.DataFrame()

    if products_df is None or 'product_id' not in products_df.columns:
        print("Error: products_df is missing or does not contain 'product_id' column. Cannot retrieve titles.")
        # Proceed without titles, but titles are highly recommended
        # return pd.DataFrame() # Or decide to proceed without titles

    # 1. Get Raw Recommendations from the KNN model
    # Request slightly more recommendations initially to allow for filtering
    raw_recs = knn_model.get_recommendations(product_id, n_recommendations=top_n * 2 + 5) # Get more

    if not raw_recs:
        # get_recommendations already prints warnings if product_id not found
        # print(f"No initial recommendations found for product ID {product_id}.")
        return pd.DataFrame()

    # Create DataFrame from raw recommendations
    similar_df = pd.DataFrame(raw_recs, columns=['recommended_product_id', 'similarity'])

    # 2. Filter Self-Recommendations (should be redundant if model is correct)
    initial_count = len(similar_df)
    similar_df = similar_df[similar_df['recommended_product_id'] != product_id].copy()
    if len(similar_df) < initial_count:
        print(f"Note: Removed self-recommendation found for {product_id} in find_similar_products.")


    # 3. Filter Similar Variants (if requested)
    variants_filtered_count = 0
    if exclude_similar_variants and similar_products_df is not None and not similar_products_df.empty:
        exclude_ids = set()
        col1_id, col2_id = None, None
        # Determine columns defensively
        if 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns:
            col1_id, col2_id = 'product1_id', 'product2_id'
        elif 'product1' in similar_products_df.columns and 'product2' in similar_products_df.columns:
            col1_id, col2_id = 'product1', 'product2'

        if col1_id and col2_id:
            try:
                matches1 = similar_products_df[similar_products_df[col1_id] == product_id]
                if not matches1.empty: exclude_ids.update(matches1[col2_id].tolist())
                matches2 = similar_products_df[similar_products_df[col2_id] == product_id]
                if not matches2.empty: exclude_ids.update(matches2[col1_id].tolist())
            except Exception as e:
                 print(f"Warning: Error during variant filtering for {product_id}: {e}. Skipping.")
                 exclude_ids = set() # Clear exclusions on error

            if exclude_ids:
                count_before_variant_filter = len(similar_df)
                similar_df = similar_df[~similar_df['recommended_product_id'].isin(exclude_ids)]
                variants_filtered_count = count_before_variant_filter - len(similar_df)
                if variants_filtered_count > 0:
                    # print(f"  - Filtered {variants_filtered_count} similar variant(s) for product {product_id}.")
                    pass # Keep output less verbose
        # else: # Warning about missing columns already printed by export_recommendations if called first
        #     pass


    # 4. Select Top N results *after* all filtering
    # Recommendations should already be sorted by similarity by get_recommendations
    final_similar_df = similar_df.head(top_n).copy()

    # 5. Add Product Titles (if possible)
    if 'product_title' in products_df.columns:
        title_map = products_df.set_index('product_id')['product_title'].to_dict()
        final_similar_df['product_title'] = final_similar_df['recommended_product_id'].map(title_map).fillna('Title Not Found')
    else:
        final_similar_df['product_title'] = 'Title N/A' # Fallback if titles unavailable

    # 6. Final Formatting
    # Rename column for clarity
    final_similar_df = final_similar_df.rename(columns={'recommended_product_id': 'product_id'})
    # Select and order final columns
    final_columns = ['product_id', 'similarity', 'product_title']
    final_similar_df = final_similar_df[[col for col in final_columns if col in final_similar_df.columns]]

    if variants_filtered_count > 0:
         print(f"Filtered {variants_filtered_count} similar variants while finding recommendations for {product_id}.")


    return final_similar_df


# --- Main Public Interface Function: run_knn ---

def run_knn(df, output_dir="results", n_neighbors=12, save_model=True,
          similar_products_df=None, # <<< Added argument
          optimize_weights=False, optimization_method='grid', optimization_params=None):
    """
    Main public interface for running weighted KNN for product recommendations.
    Handles model initialization, fitting, optional optimization, visualization, and export.

    Args:
        df (pd.DataFrame): DataFrame with processed features AND original identifiers
                           like 'product_id', 'product_title' (if available).
        output_dir (str): Directory to save all results and model files.
        n_neighbors (int): Number of neighbors (recommendations) per product.
        save_model (bool): Whether to save the fitted model components.
        similar_products_df (pd.DataFrame, optional):
                           DataFrame with pairs of too-similar products (variants)
                           to be excluded from recommendations. Expected columns like
                           'product1_id', 'product2_id'. Defaults to None.
        optimize_weights (bool): If True, run weight optimization before final fitting.
        optimization_method (str): Method ('grid', 'random', 'evolutionary') if optimizing.
        optimization_params (dict, optional): Parameters for the chosen optimization method.

    Returns:
        pd.DataFrame: DataFrame containing all generated KNN recommendations
                      (after filtering variants if applicable), or an empty DataFrame on failure.
    """
    print("\n" + "="*30 + " Running Weighted KNN Recommendation Pipeline " + "="*30)
    pipeline_start_time = time.time()

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    # Configure warnings (optional)
    # warnings.filterwarnings('always', category=UserWarning)

    # --- Input Validation ---
    if df is None or df.empty:
        print("Error: Input DataFrame 'df' is empty or None. Cannot run KNN.")
        return pd.DataFrame()
    if 'product_id' not in df.columns:
         print("Warning: 'product_id' column missing in input DataFrame. Using index as fallback.")
         # Add index as product_id if missing - do this on a copy if needed later
         # df['product_id'] = df.index.astype(str) # Modify df directly? Risky. Handled in _WeightedKNN.

    if similar_products_df is None or similar_products_df.empty:
        print("Note: No 'similar_products_df' provided. Variant filtering will be disabled.")
        # Create an empty DataFrame with expected columns? Not strictly necessary, handled internally.
        # similar_products_df = pd.DataFrame(columns=['product1_id', 'product2_id'])
    else:
        # Basic check for expected columns
        has_cols1 = 'product1_id' in similar_products_df.columns and 'product2_id' in similar_products_df.columns
        has_cols2 = 'product1' in similar_products_df.columns and 'product2' in similar_products_df.columns
        if not (has_cols1 or has_cols2):
             print("Warning: 'similar_products_df' provided, but lacks expected columns ('product1_id'/'product2_id' or 'product1'/'product2'). Variant filtering might fail.")
        else:
             print(f"Using provided 'similar_products_df' ({len(similar_products_df)} pairs) for variant filtering.")


    # --- Model Initialization and Fitting ---
    final_model = None # Variable to hold the model instance to be used for export

    # If weight optimization is requested
    if optimize_weights:
        print(f"\n=== Starting Feature Weight Optimization (Method: {optimization_method}) ===")
        opt_start_time = time.time()
        opt_dir = os.path.join(output_dir, "weight_optimization") # Dedicated subdir

        try:
            # Create and run the optimizer
            optimizer = WeightOptimizer(
                df=df.copy(), # Pass a copy of the full df to the optimizer
                n_neighbors=n_neighbors,
                output_dir=opt_dir,
                random_state=42 # Use a fixed seed for optimizer reproducibility
            )

            # Set default optimization parameters if none provided
            if optimization_params is None:
                optimization_params = {}
                print("No optimization parameters provided, using defaults.")

            # Run the selected optimization method
            if optimization_method == 'grid':
                optimizer.grid_search(
                    weight_options=optimization_params.get('weight_options'),
                    fixed_weights=optimization_params.get('fixed_weights'),
                    save_results=True
                )
            elif optimization_method == 'random':
                 optimizer.random_search(
                    n_iterations=optimization_params.get('n_iterations', 50),
                    weight_range=optimization_params.get('weight_range', (0.1, 10.0)),
                    save_results=True
                )
            elif optimization_method == 'evolutionary':
                 optimizer.evolutionary_search(
                    population_size=optimization_params.get('population_size', 10),
                    generations=optimization_params.get('generations', 5),
                    mutation_rate=optimization_params.get('mutation_rate', 0.2),
                    crossover_rate=optimization_params.get('crossover_rate', 0.8),
                    tournament_size=optimization_params.get('tournament_size', 3),
                    weight_range=optimization_params.get('weight_range', (0.1, 10.0)),
                    save_results=True
                )
            else:
                print(f"Warning: Unknown optimization method '{optimization_method}'. Defaulting to grid search.")
                optimizer.grid_search(save_results=True) # Fallback to grid

            # Visualize optimization results
            optimizer.visualize_optimization_results()

            # Build the final model using the best weights found
            print("\nBuilding final KNN model using optimized weights...")
            final_model = optimizer.build_optimized_model(save_model_files=save_model) # Pass save_model flag

            opt_end_time = time.time()
            print(f"=== Weight Optimization Finished (Duration: {opt_end_time - opt_start_time:.2f}s) ===")

        except Exception as e:
            print(f"\n--- ERROR during Weight Optimization ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Optimization failed. Attempting to proceed with default weights...")
            final_model = None # Ensure model is None if optimization failed
            # Do not set optimize_weights = False here, just let it fall through


    # If not optimizing OR if optimization failed, use default weights
    if final_model is None:
        print("\n=== Building KNN Model with Default Weights ===")
        model_build_start_time = time.time()
        try:
            # Initialize standard KNN model
            final_model = _WeightedKNN(n_neighbors=n_neighbors, random_state=42) # Use fixed seed

            # Fit the model using the input DataFrame 'df'
            # The 'fit' method handles extracting features, IDs, titles, scaling, weighting, etc.
            final_model.fit(df.copy()) # Pass a copy to avoid modifying original df

            model_build_end_time = time.time()
            if final_model.recommendation_matrix is not None: # Check if fit was successful
                 print(f"Model fitting with default weights completed in {model_build_end_time - model_build_start_time:.2f}s.")
                 # Save the model if requested and optimization wasn't attempted or failed
                 if save_model:
                      print("\nSaving the model fitted with default weights...")
                      save_dir_default = os.path.join(output_dir, "model_default_weights")
                      final_model.save_model(save_dir_default)
            else:
                 print("Error: Failed to fit model even with default weights.")
                 final_model = None # Explicitly set to None on failure


        except Exception as e:
            print(f"\n--- ERROR during Model Fitting with Default Weights ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            final_model = None # Ensure model is None on error


    # --- Post-Fitting Steps (Visualizations, Export) ---
    all_recommendations_df = pd.DataFrame() # Initialize as empty df

    if final_model is None or final_model.recommendation_matrix is None:
         print("\nError: No valid KNN model was fitted. Cannot proceed with visualizations or export.")
    else:
        print("\n=== Generating Visualizations and Exporting Results ===")
        # Create visualizations using the fitted final_model
        # Wrap each in try/except to ensure robustness
        vis_success = {'importance': False, 'tsne': False, 'distribution': False}
        try:
            print("Generating feature importance visualization...")
            final_model.visualize_feature_importance(output_file=os.path.join(output_dir, "feature_importance.png"))
            vis_success['importance'] = True
        except Exception as e: print(f"Error generating feature importance plot: {e}")

        try:
            print("Generating t-SNE visualization...")
            # t-SNE uses model.X_weighted internally
            final_model.visualize_tsne(output_file=os.path.join(output_dir, "tsne_visualization.png"))
            vis_success['tsne'] = True
        except Exception as e: print(f"Error generating t-SNE plot: {e}")

        try:
            print("Generating similarity distribution visualization...")
            final_model.visualize_similarity_distribution(output_file=os.path.join(output_dir, "similarity_distribution.png"))
            vis_success['distribution'] = True
        except Exception as e: print(f"Error generating similarity distribution plot: {e}")


        # --- Export Recommendations ---
        try:
            print("\nExporting all recommendations...")
            all_recommendations_df = final_model.export_recommendations(
                output_file=os.path.join(output_dir, "all_recommendations.csv"),
                n_recommendations=n_neighbors, # Use the requested number
                products_df=df, # Pass original df for titles
                similar_products_df=similar_products_df # Pass the variant df for filtering
            )
            if not all_recommendations_df.empty:
                 print(f"Successfully exported {len(all_recommendations_df)} recommendation pairs.")
            else:
                 print("Export completed, but no recommendations were generated or exported.")

        except Exception as e:
            print(f"\n--- Error Exporting Recommendations ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            all_recommendations_df = pd.DataFrame() # Ensure empty on error


        # --- Example: Find Similar Products for Sample ---
        if not df.empty and 'product_id' in df.columns:
             try:
                 # Select a sample product ID (e.g., the first one)
                 sample_product_id = df['product_id'].iloc[0]
                 print(f"\nGenerating example recommendations for sample product: {sample_product_id}")
                 if 'product_title' in df.columns:
                      sample_title = df.loc[df['product_id'] == sample_product_id, 'product_title'].iloc[0]
                      print(f"Sample Product Title: {sample_title}")


                 similar_products_example_df = find_similar_products(
                     product_id=sample_product_id,
                     knn_model=final_model,
                     products_df=df, # Pass original df for titles
                     top_n=n_neighbors,
                     exclude_similar_variants=True, # Usually want to exclude in examples too
                     similar_products_df=similar_products_df # Pass variant info
                 )

                 if not similar_products_example_df.empty:
                     print("\nExample Similar Products Found:")
                     print(similar_products_example_df.to_string(index=False))
                     # Save example to CSV
                     example_file = os.path.join(output_dir, "similar_products_example.csv")
                     similar_products_example_df.to_csv(example_file, index=False)
                     print(f"Example similar products saved to {example_file}")
                 else:
                     print(f"No similar products found for the sample product {sample_product_id} after filtering.")

             except Exception as e:
                 print(f"\nError generating similar products example: {e}")


        # --- Summary Report ---
        try:
            summary_path = os.path.join(output_dir, "knn_summary_report.txt")
            with open(summary_path, "w", encoding='utf-8') as f:
                f.write("=== Weighted KNN Recommendation Summary ===\n\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input Products: {len(df)}\n")
                f.write(f"Features Used: {len(final_model.feature_names_in_order) if final_model.feature_names_in_order else 'N/A'}\n")
                f.write(f"KNN Neighbors (k): {final_model.n_neighbors}\n")
                f.write(f"Distance Metric: {final_model.metric}\n")
                f.write(f"Weight Optimization Used: {'Yes' if optimize_weights else 'No'}\n")
                if optimize_weights:
                    f.write(f"Optimization Method: {optimization_method}\n")
                    if final_model.feature_weights: # Check if weights are available
                        f.write(f"Best Avg Similarity Achieved: {optimizer.best_score:.4f}\n" if optimizer and hasattr(optimizer,'best_score') else "N/A\n")


                f.write("\nFeature Group Weights (Effective):\n")
                if final_model.feature_weights:
                    for group, weight in sorted(final_model.feature_weights.items()):
                        f.write(f"- {group}: {weight:.4f}\n")
                else:
                    f.write("  Weights not available.\n")

                f.write("\nVisualizations Generated:\n")
                f.write(f"- Feature Importance: {'Yes' if vis_success['importance'] else 'No'}\n")
                f.write(f"- t-SNE Plot: {'Yes' if vis_success['tsne'] else 'No'}\n")
                f.write(f"- Similarity Distribution: {'Yes' if vis_success['distribution'] else 'No'}\n")

                f.write("\nRecommendations Exported:\n")
                export_status = 'Yes' if not all_recommendations_df.empty else 'No'
                export_count = len(all_recommendations_df) if not all_recommendations_df.empty else 0
                f.write(f"- all_recommendations.csv: {export_status} ({export_count} pairs)\n")
                if os.path.exists(os.path.join(output_dir, 'similar_products_example.csv')):
                     f.write("- similar_products_example.csv: Yes\n")


                f.write("\nModel Saved:\n")
                model_saved = False
                if save_model:
                     model_dir = os.path.join(output_dir, "optimized_model_final") if optimize_weights and final_model else os.path.join(output_dir, "model_default_weights")
                     if os.path.exists(os.path.join(model_dir, "knn_model.pkl")): # Check for a key file
                          model_saved = True
                          f.write(f"- Model components saved to: {model_dir}\n")

                if not model_saved:
                     f.write("- Model not saved or save location unclear.\n")


            print(f"\nKNN summary report saved to {summary_path}")
        except Exception as e:
            print(f"\nError generating summary report: {e}")


    # --- Finalization ---
    pipeline_end_time = time.time()
    print("\n" + "="*30 + f" KNN Pipeline Finished (Total Time: {pipeline_end_time - pipeline_start_time:.2f}s) " + "="*30)

    # Return the DataFrame with all recommendations (could be empty)
    return all_recommendations_df