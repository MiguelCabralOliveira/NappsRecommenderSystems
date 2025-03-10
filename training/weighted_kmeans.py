def run_optimal_k_analysis(input_file, output_dir="results", min_k=2, max_k=30):
    """
    Run only the optimal k analysis without performing full clustering
    
    Args:
        input_file: Path to the input CSV file with processed features
        output_dir: Directory to save results
        min_k: Minimum number of clusters to try
        max_k: Maximum number of clusters to try
        
    Returns:
        int: Optimal number of clusters
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    # Initialize model
    model = WeightedKMeans()
    
    # Remove non-feature columns
    non_feature_cols = ['product_id', 'product_title', 'handle']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    feature_df = df[feature_cols]
    
    # Handle missing values
    feature_df = model.handle_missing_values(feature_df)
    
    # Identify feature groups
    try:
        model.identify_feature_groups(feature_df)
    except Exception as e:
        print(f"Error identifying feature groups: {str(e)}")
        return None
    
    # Set weights
    try:
        weights = {
            'product_type': 1.8,
            'tags': 1.8,
            'description_tfidf': 1.0,
            'metafield_data': 1.0,
            'collections': 1.0,
            'vendor': 1.0,
            'variant_size': 1.0,
            'variant_color': 1.0,
            'variant_other': 1.0,
            'numerical': 1.0
        }
        model.set_feature_weights(weights)
    except Exception as e:
        print(f"Error setting feature weights: {str(e)}")
    
    # Standardize data
    print("Standardizing data...")
    X = model.scaler.fit_transform(feature_df)
    
    # Apply weights
    print("Applying feature weights...")
    X_weighted = model.apply_weights(X)
    
    # Find optimal k
    k_range = range(min_k, max_k + 1)
    optimal_k = model.find_optimal_k(X_weighted, k_range, output_dir)
    
    print(f"\nOptimal number of clusters analysis complete!")
    print(f"To use this optimal value ({optimal_k}), run clustering with:")
    print(f"python main.py --clusters {optimal_k}")
    
    return optimal_k# weighted_kmeans.py
# Modify the weighted_kmeans.py file to be imported as a module

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os
import joblib
from collections import defaultdict
import re


class WeightedKMeans:
    """
    Implements a weighted KMeans clustering algorithm for product data
    where different feature groups can be assigned different weights.
    """
    def __init__(self, n_clusters=5, random_state=42, max_iter=300):
        """
        Initialize the WeightedKMeans model.
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations for KMeans
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.kmeans = None
        self.feature_weights = None
        self.feature_groups = None
        self.scaler = StandardScaler()
        self.silhouette_avg = None
        self.product_ids = None
        self.product_titles = None
        self.cluster_counts = None
        
    def load_data(self, data_file):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        # Keep product ID and title for reference
        self.product_ids = df['product_id'].copy() if 'product_id' in df.columns else None
        self.product_titles = df['product_title'].copy() if 'product_title' in df.columns else None
        
        # Remove non-feature columns
        non_feature_cols = ['product_id', 'product_title', 'handle']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        feature_df = df[feature_cols]
        
        print(f"Loaded {len(df)} products with {len(feature_cols)} features")
        
        # Handle missing values
        feature_df = self.handle_missing_values(feature_df)
        
        return feature_df
    
    def identify_feature_groups(self, df):
        """
        Automatically identify and group features based on their names and types
        
        Args:
            df: DataFrame with features
            
        Returns:
            dict: Dictionary of feature groups
        """
        # Initialize feature groups
        feature_groups = {
            'description_tfidf': [],
            'metafield_data': [],
            'collections': [],
            'tags': [],
            'product_type': [],
            'vendor': [],
            'variant_size': [],
            'variant_color': [],
            'variant_other': [],
            'numerical': []
        }
        
        # Pattern matching for feature groups based on column names
        for col in df.columns:
            if col.startswith('description_'):
                feature_groups['description_tfidf'].append(col)
            elif col.startswith('metafield_'):
                feature_groups['metafield_data'].append(col)
            elif col.startswith('collections_'):
                feature_groups['collections'].append(col)
            elif col.startswith('tag_'):
                feature_groups['tags'].append(col)
            elif col.startswith('product_type_'):
                feature_groups['product_type'].append(col)
            elif col.startswith('vendor_'):
                feature_groups['vendor'].append(col)
            elif col.startswith('variant_size_'):
                feature_groups['variant_size'].append(col)
            elif col.startswith('variant_color_'):
                feature_groups['variant_color'].append(col)
            elif col.startswith('variant_other_'):
                feature_groups['variant_other'].append(col)
            elif col in ['min_price', 'max_price', 'has_discount']:
                feature_groups['numerical'].append(col)
            else:
                # Default to numerical for any other columns
                if col in df.select_dtypes(include=np.number).columns:
                    feature_groups['numerical'].append(col)
        
        # Remove empty groups
        self.feature_groups = {k: v for k, v in feature_groups.items() if v}
        
        # Print feature group summary
        print("\nFeature Group Summary:")
        for group, cols in self.feature_groups.items():
            print(f"{group}: {len(cols)} features")
            
        return self.feature_groups
    
    def set_feature_weights(self, weights=None):
        """
        Set weights for each feature group
        
        Args:
            weights: Dictionary mapping feature group names to weights
                    If None, default weights will be used
        """
        if weights is None:
            # Default weights with higher weights for tags and product_type as requested
            weights = {
                'tags': 1.8,                # Higher weight for tags
                'product_type': 1.8,        # Higher weight for product types
                'description_tfidf': 1.0,   # Standard weight for descriptive text
                'metafield_data': 1.0,      # Standard weight for metafield data
                'collections': 1.0,         # Standard weight for collections
                'vendor': 1.0,              # Standard weight for vendor
                'variant_size': 1.0,        # Standard weight for size variants
                'variant_color': 1.0,       # Standard weight for color variants
                'variant_other': 1.0,       # Standard weight for other variants
                'numerical': 1.0            # Standard weight for numerical data
            }
        
        # Validate that weights exist for all feature groups
        for group in self.feature_groups:
            if group not in weights:
                weights[group] = 1.0  # Default weight if not specified
        
        self.feature_weights = weights
        print("\nFeature Weights:")
        for group, weight in self.feature_weights.items():
            if group in self.feature_groups:
                print(f"{group}: {weight}")
                
        return self.feature_weights
    
    def apply_weights(self, X):
        """
        Apply weights to the feature matrix
        
        Args:
            X: Feature matrix (numpy array)
            
        Returns:
            numpy array: Weighted feature matrix
        """
        if self.feature_weights is None or self.feature_groups is None:
            raise ValueError("Feature weights or groups not set. Call set_feature_weights() first.")
        
        # Create a copy of the data to avoid modifying the original
        X_weighted = X.copy()
        
        # Map each feature to its corresponding group and weight
        feature_to_weight = {}
        start_idx = 0
        
        for group, features in self.feature_groups.items():
            weight = self.feature_weights.get(group, 1.0)
            for _ in range(len(features)):
                feature_to_weight[start_idx] = weight
                start_idx += 1
        
        # Apply weights to each feature
        for i in range(X_weighted.shape[1]):
            weight = feature_to_weight.get(i, 1.0)
            X_weighted[:, i] *= weight
            
        return X_weighted
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame
        
        Args:
            df: DataFrame to check and process
            
        Returns:
            DataFrame: Processed DataFrame with no NaN values
        """
        # Check for NaN values
        nan_counts = df.isnull().sum().sum()
        if nan_counts > 0:
            print(f"Warning: Found {nan_counts} missing values in the dataset")
            print("Columns with missing values:")
            missing_cols = df.columns[df.isnull().any()].tolist()
            for col in missing_cols:
                count = df[col].isnull().sum()
                print(f"  - {col}: {count} missing values")
            
            # Fill NaN values with 0
            print("Filling missing values with 0...")
            df = df.fillna(0)
            
            # Verify no NaNs remain
            if df.isnull().sum().sum() > 0:
                print("ERROR: Still have NaN values after filling!")
                raise ValueError("Failed to remove all NaN values from data")
        
        return df
        
    def fit(self, data, find_optimal_k=False, k_range=None):
        """
        Fit the weighted KMeans model to the data
        
        Args:
            data: DataFrame or path to CSV file
            find_optimal_k: Whether to find the optimal number of clusters
            k_range: Range of k values to try if finding optimal k
            
        Returns:
            self: The fitted model
        """
        try:
            # Load data if string is provided
            if isinstance(data, str):
                df = self.load_data(data)
            else:
                df = data.copy()
                
            # Always handle missing values
            df = self.handle_missing_values(df)
                
            # Identify feature groups if not already done
            if self.feature_groups is None:
                self.identify_feature_groups(df)
                
            # Set default weights if not already done
            if self.feature_weights is None:
                self.set_feature_weights()
            
            # Check for infinite values
            inf_counts = np.isinf(df.values).sum()
            if inf_counts > 0:
                print(f"Warning: Found {inf_counts} infinite values in the dataset. Replacing with large values.")
                df = df.replace([np.inf, -np.inf], [1e9, -1e9])
            
            # Standardize the data
            print("Standardizing data...")
            X = self.scaler.fit_transform(df)
            
            # Apply weights to features
            print("Applying feature weights...")
            X_weighted = self.apply_weights(X)
            
            # Verify no NaNs in weighted matrix
            if np.isnan(X_weighted).any():
                print("ERROR: NaN values found in weighted feature matrix!")
                print(f"NaN count: {np.isnan(X_weighted).sum()}")
                raise ValueError("NaN values in weighted feature matrix")
            
            # Optionally find optimal k
            if find_optimal_k:
                self.find_optimal_k(X_weighted, k_range)
            
            # Fit KMeans
            print(f"\nFitting KMeans with {self.n_clusters} clusters...")
            self.kmeans = KMeans(
                n_clusters=self.n_clusters, 
                random_state=self.random_state,
                max_iter=self.max_iter
            ).fit(X_weighted)
            
            # Calculate silhouette score
            self.silhouette_avg = silhouette_score(X_weighted, self.kmeans.labels_)
            print(f"Silhouette Score: {self.silhouette_avg:.4f}")
            
            # Store number of products per cluster
            self.cluster_counts = np.bincount(self.kmeans.labels_)
            print("\nNumber of products per cluster:")
            for i, count in enumerate(self.cluster_counts):
                print(f"Cluster {i}: {count} products")
                
            return self
            
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            print("Clustering failed but continuing with the rest of the pipeline.")
            # Return self to allow the pipeline to continue
            return self
    
    def find_optimal_k(self, X, k_range=None, output_dir="results"):
        """
        Find optimal number of clusters using Davies-Bouldin Index
        Lower Davies-Bouldin values indicate better clustering.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to try
            output_dir: Directory to save visualization
            
        Returns:
            int: Optimal number of clusters
        """
        try:
            if k_range is None:
                k_range = range(2, 31)  # Start from 2 clusters and go up to 30
                
            print("\n" + "="*50)
            print("FINDING OPTIMAL NUMBER OF CLUSTERS")
            print("="*50)
            print("\nAnalyzing cluster quality for k = %d to %d..." % (k_range[0], k_range[-1]))
            
            davies_bouldin_scores = []
            silhouette_scores = []
            calinski_harabasz_scores = []
            inertia_scores = []
            
            previous_db_score = float('inf')
            consecutive_increases = 0
            optimal_k = k_range[0]  # Default to first k
            
            # Create a results table
            results_table = []
            
            for k in k_range:
                print(f"Evaluating k={k}...", end=" ")
                
                # Check for NaN/inf values and fix them
                if np.isnan(X).any() or np.isinf(X).any():
                    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
                
                # Fit KMeans with current k
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, max_iter=self.max_iter).fit(X)
                
                # Calculate Davies-Bouldin Index (lower is better)
                db_score = davies_bouldin_score(X, kmeans.labels_)
                davies_bouldin_scores.append(db_score)
                
                # Calculate Silhouette score (higher is better)
                sil_score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(sil_score)
                
                # Calculate Calinski-Harabasz Index (higher is better)
                ch_score = calinski_harabasz_score(X, kmeans.labels_)
                calinski_harabasz_scores.append(ch_score)
                
                # Calculate inertia (sum of squared distances, lower is better)
                inertia = kmeans.inertia_
                inertia_scores.append(inertia)
                
                # Add to results table
                results_table.append({
                    'k': k,
                    'davies_bouldin': db_score,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'inertia': inertia
                })
                
                print(f"Davies-Bouldin: {db_score:.4f}, Silhouette: {sil_score:.4f}")
                
                # Check if Davies-Bouldin score increased (worse clustering)
                if db_score < previous_db_score:
                    # Better score, reset counter and update optimal k
                    consecutive_increases = 0
                    optimal_k = k
                else:
                    # Worse score, increment counter
                    consecutive_increases += 1
                    # Stop if we've seen 3 consecutive increases in DB index
                    if consecutive_increases >= 3:
                        print(f"Davies-Bouldin Index has increased for 3 consecutive k values.")
                        print(f"This indicates diminishing returns for adding more clusters.")
                        print(f"Stopping search at k={k}.")
                        break
                
                previous_db_score = db_score
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results to CSV
            results_df = pd.DataFrame(results_table)
            results_df.to_csv(f"{output_dir}/optimal_k_metrics.csv", index=False)
            
            # Find best k for each metric
            best_k_db = k_range[np.argmin(davies_bouldin_scores)]
            best_k_silhouette = k_range[np.argmax(silhouette_scores)]
            best_k_ch = k_range[np.argmax(calinski_harabasz_scores)]
            
            # Find elbow point for inertia
            elbow_k = self._find_elbow_point(k_range[:len(inertia_scores)], inertia_scores)
            
            print("\n" + "="*50)
            print("OPTIMAL NUMBER OF CLUSTERS RESULTS")
            print("="*50)
            print(f"Davies-Bouldin Index (lower is better): k = {best_k_db}")
            print(f"Silhouette Score (higher is better): k = {best_k_silhouette}")
            print(f"Calinski-Harabasz Index (higher is better): k = {best_k_ch}")
            print(f"Inertia Elbow Point: k = {elbow_k}")
            print("\nRecommended number of clusters: k = {optimal_k}")
            print("(Based on Davies-Bouldin with early stopping)")
            print("="*50)
            
            # Update n_clusters
            self.n_clusters = optimal_k
            
            # Plot results for all metrics
            plt.figure(figsize=(15, 15))
            
            # First subplot for Davies-Bouldin Index
            plt.subplot(2, 2, 1)
            plt.plot(k_range[:len(davies_bouldin_scores)], davies_bouldin_scores, 'o-', color='red')
            plt.axvline(x=optimal_k, color='green', linestyle='--')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Davies-Bouldin Index')
            plt.title('Davies-Bouldin Index (Lower is Better)')
            plt.grid(True)
            
            # Second subplot for Silhouette Score
            plt.subplot(2, 2, 2)
            plt.plot(k_range[:len(silhouette_scores)], silhouette_scores, 'o-', color='blue')
            plt.axvline(x=best_k_silhouette, color='green', linestyle='--')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score (Higher is Better)')
            plt.grid(True)
            
            # Third subplot for Calinski-Harabasz Index
            plt.subplot(2, 2, 3)
            plt.plot(k_range[:len(calinski_harabasz_scores)], calinski_harabasz_scores, 'o-', color='purple')
            plt.axvline(x=best_k_ch, color='green', linestyle='--')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Calinski-Harabasz Index')
            plt.title('Calinski-Harabasz Index (Higher is Better)')
            plt.grid(True)
            
            # Fourth subplot for Inertia (Elbow Method)
            plt.subplot(2, 2, 4)
            plt.plot(k_range[:len(inertia_scores)], inertia_scores, 'o-', color='orange')
            plt.axvline(x=elbow_k, color='green', linestyle='--')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Inertia (Elbow Method)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/optimal_k_analysis.png")
            plt.close()
            
            print(f"Optimal k analysis visualization saved to {output_dir}/optimal_k_analysis.png")
            print(f"Detailed metrics saved to {output_dir}/optimal_k_metrics.csv")
            
            return optimal_k
            
        except Exception as e:
            print(f"Error during optimal k analysis: {str(e)}")
            print("Using default number of clusters instead.")
            return self.n_clusters
            
    def _find_elbow_point(self, k_values, inertia_values):
        """
        Find the elbow point in the inertia curve (Elbow method)
        
        Args:
            k_values: List of k values
            inertia_values: List of inertia values
            
        Returns:
            int: k value at the elbow point
        """
        try:
            # Convert to numpy arrays
            k = np.array(k_values)
            inertia = np.array(inertia_values)
            
            # Calculate the angle between consecutive points
            angles = []
            for i in range(1, len(k)-1):
                # Create vectors
                v1 = np.array([k[i]-k[i-1], inertia[i]-inertia[i-1]])
                v2 = np.array([k[i+1]-k[i], inertia[i+1]-inertia[i]])
                
                # Normalize vectors
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Calculate angle
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                angles.append(angle)
            
            # Find the k value with the maximum angle (sharpest turn)
            if angles:
                elbow_idx = np.argmax(angles) + 1  # +1 because we started at index 1
                return k[elbow_idx]
            else:
                return k[0]  # Default to first k if we can't calculate
        except Exception as e:
            print(f"Error finding elbow point: {str(e)}")
            if len(k_values) > 0:
                return k_values[0]  # Default to first k
            else:
                return 5  # Absolute default

    def get_cluster_labels(self):
        """
        Get cluster labels for the data
        
        Returns:
            numpy array: Cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.kmeans.labels_
    
    def visualize_clusters(self, n_components=2, output_file='cluster_visualization.png'):
        """
        Visualize clusters using PCA for dimensionality reduction
        
        Args:
            n_components: Number of PCA components for visualization
            output_file: Path to save the visualization
            
        Returns:
            None
        """
        try:
            if self.kmeans is None:
                print("Model not fitted yet. Skipping cluster visualization.")
                return
                
            # Get data and apply weights
            df = self.kmeans.cluster_centers_
            
            # Check for NaN or inf values
            if np.isnan(df).any() or np.isinf(df).any():
                print("Warning: Cluster centers contain NaN or inf values. Fixing them...")
                df = np.nan_to_num(df, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(df)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # First plot: Cluster centers
            plt.subplot(1, 2, 1)
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', marker='o', s=100)
            for i, (x, y) in enumerate(reduced_data):
                plt.annotate(f'Cluster {i}', (x, y), textcoords="offset points", 
                            xytext=(0, 10), ha='center')
            plt.title('Cluster Centers (PCA)')
            plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True)
            
            # Second plot: Distribution of samples by cluster
            plt.subplot(1, 2, 2)
            plt.bar(range(self.n_clusters), self.cluster_counts)
            plt.xlabel('Cluster')
            plt.ylabel('Number of Products')
            plt.title('Products per Cluster')
            plt.xticks(range(self.n_clusters))
            plt.grid(True, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Cluster visualization saved to {output_file}")
        except Exception as e:
            print(f"Error during cluster visualization: {str(e)}")
            print("Skipping cluster visualization and continuing with the analysis.")
    
    def visualize_feature_importance(self, feature_names, output_file="feature_importance.png"):
        """
        Visualize feature importance across all clusters
        
        Args:
            feature_names: List of feature names
            output_file: Path to save the visualization
            
        Returns:
            None
        """
        try:
            if self.kmeans is None:
                print("Model not fitted yet. Skipping feature importance visualization.")
                return
            
            if isinstance(feature_names, pd.Index):
                feature_names = feature_names.tolist()
                
            # Get weighted cluster centers
            centers = self.kmeans.cluster_centers_
            
            # Check for NaN or inf values
            if np.isnan(centers).any() or np.isinf(centers).any():
                print("Warning: Cluster centers contain NaN or inf values. Fixing them...")
                centers = np.nan_to_num(centers, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Calculate overall feature importance as the mean absolute value across clusters
            feature_importance = np.mean(np.abs(centers), axis=0)
            
            # Ensure feature_names and feature_importance match in length
            if len(feature_names) != len(feature_importance):
                print(f"Warning: Feature names count ({len(feature_names)}) doesn't match "
                      f"feature importance count ({len(feature_importance)})")
                # Create default feature names if needed
                if len(feature_names) < len(feature_importance):
                    feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
                # Truncate feature_importance if needed
                feature_importance = feature_importance[:len(feature_names)]
            
            # Get top features
            num_features = min(20, len(feature_importance))
            top_indices = np.argsort(feature_importance)[-num_features:]
            
            # Create a horizontal bar chart
            plt.figure(figsize=(12, 10))
            y_pos = np.arange(len(top_indices))
            
            # Create readable feature names
            display_names = []
            for idx in top_indices:
                if idx >= len(feature_names):
                    name = f"Feature_{idx}"
                else:
                    name = feature_names[idx]
                    
                # Format for readability
                if name.startswith('product_type_'):
                    name = f"Type: {name.replace('product_type_', '')}"
                elif name.startswith('vendor_'):
                    name = f"Vendor: {name.replace('vendor_', '')}"
                elif name.startswith('collections_'):
                    name = f"Collection: {name.replace('collections_', '')}"
                elif name.startswith('tag_'):
                    name = f"Tag: {name.replace('tag_', '')}"
                elif name.startswith('description_'):
                    name = f"Desc: {name.replace('description_', '')}"
                elif name.startswith('metafield_'):
                    name = f"Meta: {name.replace('metafield_', '')}"
                elif name.startswith('variant_'):
                    if name.startswith('variant_size_'):
                        name = f"Size: {name.replace('variant_size_', '')}"
                    elif name.startswith('variant_color_'):
                        name = f"Color: {name.replace('variant_color_', '')}"
                    elif name.startswith('variant_other_'):
                        name = f"Var: {name.replace('variant_other_', '')}"
                
                # Truncate long names
                if len(name) > 30:
                    name = name[:27] + "..."
                    
                display_names.append(name)
            
            # Plot
            plt.barh(y_pos, [feature_importance[i] for i in top_indices])
            plt.yticks(y_pos, display_names)
            plt.xlabel('Feature Importance')
            plt.title('Top Features by Importance')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Feature importance visualization saved to {output_file}")
        except Exception as e:
            print(f"Error during feature importance visualization: {str(e)}")
            print("Skipping feature importance visualization and continuing with the analysis.")
    
    def visualize_tsne(self, X, output_file="tsne_visualization.png"):
        """
        Visualize data using t-SNE dimensionality reduction
        
        Args:
            X: Feature matrix
            output_file: Path to save the visualization
            
        Returns:
            None
        """
        if self.kmeans is None:
            print("Model not fitted yet. Skipping t-SNE visualization.")
            return
        
        try:
            # Check for NaN values in the input
            if np.isnan(X).any():
                print(f"Warning: Input contains {np.isnan(X).sum()} NaN values. Fixing them...")
                X = np.nan_to_num(X, nan=0.0)
            
            # Check for infinite values
            if np.isinf(X).any():
                print(f"Warning: Input contains {np.isinf(X).sum()} infinite values. Fixing them...")
                X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Apply weights
            X_weighted = self.apply_weights(X)
            
            # Verify data quality after weighting
            if np.isnan(X_weighted).any() or np.isinf(X_weighted).any():
                print("Warning: Weighted matrix contains bad values. Fixing them...")
                X_weighted = np.nan_to_num(X_weighted, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Sample data if there are too many points (t-SNE is computationally expensive)
            max_samples = 1000
            if X_weighted.shape[0] > max_samples:
                print(f"Sampling {max_samples} points for t-SNE visualization")
                indices = np.random.choice(X_weighted.shape[0], max_samples, replace=False)
                X_sample = X_weighted[indices]
                labels_sample = self.kmeans.labels_[indices]
            else:
                X_sample = X_weighted
                labels_sample = self.kmeans.labels_
            
            # Apply t-SNE
            print("Applying t-SNE dimensionality reduction (this may take a while)...")
            perplexity = min(30, X_sample.shape[0] - 1)  # Adjust perplexity for small datasets
            tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity)
            X_tsne = tsne.fit_transform(X_sample)
            
            # Plot
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title('t-SNE Visualization of Clusters')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"t-SNE visualization saved to {output_file}")
        except Exception as e:
            print(f"Error during t-SNE visualization: {str(e)}")
            print("Skipping t-SNE visualization and continuing with the rest of the analysis.")
    
    def get_top_features_per_cluster(self, feature_names, top_n=10):
        """
        Get top features for each cluster
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return per cluster
            
        Returns:
            dict: Dictionary mapping clusters to their top features
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Ensure feature_names is a list
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
            
        # Get cluster centers
        centers = self.kmeans.cluster_centers_
        
        top_features = {}
        
        for cluster_idx in range(self.n_clusters):
            # Get importance of each feature for this cluster
            feature_importance = centers[cluster_idx]
            
            # Get index of top features
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            # Map indices to feature names
            top_features[cluster_idx] = [(feature_names[i], feature_importance[i]) for i in top_indices]
            
        return top_features
    
    def evaluate_clusters(self):
        """
        Evaluate the quality of clusters using multiple metrics
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            if self.kmeans is None:
                print("Model not fitted yet. Skipping cluster evaluation.")
                return {}
            
            # Get original features and labels
            X_weighted = self.apply_weights(self.scaler.transform(self.scaler.inverse_transform(self.kmeans.cluster_centers_)))
            labels = self.kmeans.labels_
            
            # Calculate silhouette score - we already have this from fitting
            silhouette = self.silhouette_avg
            
            # Calculate inertia (sum of squared distances)
            inertia = self.kmeans.inertia_
            
            # For Davies-Bouldin and Calinski-Harabasz, we need the actual data points
            # not just the centers. So we'll use the original data that's already been
            # scaled and weighted during model.fit()
            # 
            # Note: We can't compute these metrics on the cluster centers themselves,
            # so we just report inertia and silhouette scores
            
            # Store results
            results = {
                'silhouette': silhouette,
                'inertia': inertia
            }
            
            # Try to compute other metrics if possible
            try:
                # This requires the original data (all points)
                X_orig = self.apply_weights(self.scaler.transform(self.scaler.inverse_transform(self.kmeans.cluster_centers_)))
                
                # Check dimensions are compatible
                if len(self.kmeans.labels_) == X_orig.shape[0]:
                    # Calculate Davies-Bouldin index
                    db_score = davies_bouldin_score(X_orig, self.kmeans.labels_)
                    results['davies_bouldin'] = db_score
                    
                    # Calculate Calinski-Harabasz index
                    ch_score = calinski_harabasz_score(X_orig, self.kmeans.labels_)
                    results['calinski_harabasz'] = ch_score
            except Exception as e:
                print(f"Could not compute advanced metrics: {str(e)}")
                print("This is normal and won't affect clustering quality")
            
            print("\nCluster Evaluation Metrics:")
            print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
            if 'davies_bouldin' in results:
                print(f"Davies-Bouldin Index: {results['davies_bouldin']:.4f} (lower is better)")
            if 'calinski_harabasz' in results:
                print(f"Calinski-Harabasz Index: {results['calinski_harabasz']:.4f} (higher is better)")
            print(f"Inertia: {inertia:.4f} (lower is better)")
            
            return results
            
        except Exception as e:
            print(f"Error evaluating clusters: {str(e)}")
            return {}
    
    def interpret_clusters(self, df, top_n=5):
        """
        Interpret clusters by summarizing key characteristics
        
        Args:
            df: Original DataFrame with features
            top_n: Number of top features to show per cluster
            
        Returns:
            dict: Dictionary of cluster interpretations
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Get feature names
        feature_names = df.columns.tolist()
        
        # Get top features per cluster
        top_features = self.get_top_features_per_cluster(feature_names, top_n=top_n)
        
        # Create interpretations
        interpretations = {}
        
        print("\nCluster Interpretations:")
        for cluster_idx in range(self.n_clusters):
            features = top_features[cluster_idx]
            feature_names_list = [f[0] for f in features]
            
            # Create a human-readable interpretation
            interpretation = f"Cluster {cluster_idx} ({self.cluster_counts[cluster_idx]} products):\n"
            interpretation += "Key characteristics:\n"
            
            for name, value in features:
                # Format feature name for readability
                display_name = name
                if name.startswith('product_type_'):
                    display_name = f"Product Type: {name.replace('product_type_', '')}"
                elif name.startswith('vendor_'):
                    display_name = f"Vendor: {name.replace('vendor_', '')}"
                elif name.startswith('collections_'):
                    display_name = f"Collection: {name.replace('collections_', '')}"
                elif name.startswith('tag_'):
                    display_name = f"Tag: {name.replace('tag_', '')}"
                elif name.startswith('description_'):
                    display_name = f"Description Term: {name.replace('description_', '')}"
                elif name.startswith('metafield_'):
                    display_name = f"Metafield: {name.replace('metafield_', '')}"
                elif name.startswith('variant_size_'):
                    display_name = f"Size Variant: {name.replace('variant_size_', '')}"
                elif name.startswith('variant_color_'):
                    display_name = f"Color Variant: {name.replace('variant_color_', '')}"
                elif name.startswith('variant_other_'):
                    display_name = f"Other Variant: {name.replace('variant_other_', '')}"
                    
                interpretation += f"  - {display_name}: {value:.4f}\n"
                
            interpretations[cluster_idx] = interpretation
            print(interpretation)
            
        return interpretations
    
    def export_centroids(self, output_file="cluster_centroids.csv"):
        """
        Export cluster centroids for external visualization
        
        Args:
            output_file: Path to save the centroids
            
        Returns:
            DataFrame: Centroids dataframe
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Get feature names
        feature_names = []
        for group, features in self.feature_groups.items():
            feature_names.extend(features)
            
        # Create DataFrame with centroids
        centroids_df = pd.DataFrame(
            self.kmeans.cluster_centers_,
            columns=feature_names
        )
        
        # Add cluster ID
        centroids_df['cluster_id'] = range(self.n_clusters)
        
        # Add cluster size
        centroids_df['cluster_size'] = self.cluster_counts
        
        # Save to CSV
        centroids_df.to_csv(output_file, index=False)
        print(f"Cluster centroids exported to {output_file}")
        
        return centroids_df
    
    def export_cluster_assignments(self, output_file="cluster_assignments.csv"):
        """
        Export cluster assignments for all products
        
        Args:
            output_file: Path to save the assignments
            
        Returns:
            DataFrame: Assignments dataframe
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Create DataFrame with cluster assignments
        assignments_df = pd.DataFrame({
            'product_id': self.product_ids,
            'product_title': self.product_titles,
            'cluster': self.kmeans.labels_
        })
        
        # Save to CSV
        assignments_df.to_csv(output_file, index=False)
        print(f"Cluster assignments exported to {output_file}")
        
        return assignments_df
    
    def predict(self, X):
        """
        Predict clusters for new data
        
        Args:
            X: Feature matrix or DataFrame
            
        Returns:
            numpy array: Cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Standardize
        X_scaled = self.scaler.transform(X)
        
        # Apply weights
        X_weighted = self.apply_weights(X_scaled)
        
        # Predict
        return self.kmeans.predict(X_weighted)
    
    def save_model(self, output_dir="models"):
        """
        Save the model to disk
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            None
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save KMeans model
        joblib.dump(self.kmeans, f"{output_dir}/kmeans_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        
        # Save weights and groups
        joblib.dump({
            'feature_weights': self.feature_weights,
            'feature_groups': self.feature_groups,
            'silhouette_avg': self.silhouette_avg,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'cluster_counts': self.cluster_counts.tolist() if self.cluster_counts is not None else None
        }, f"{output_dir}/model_params.pkl")
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir="models"):
        """
        Load a saved model
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            WeightedKMeans: Loaded model
        """
        # Check if model exists
        kmeans_path = f"{model_dir}/kmeans_model.pkl"
        scaler_path = f"{model_dir}/scaler.pkl"
        params_path = f"{model_dir}/model_params.pkl"
        
        if not (os.path.exists(kmeans_path) and os.path.exists(scaler_path) and os.path.exists(params_path)):
            raise FileNotFoundError(f"Model files not found in {model_dir}")
            
        # Load files
        kmeans = joblib.load(kmeans_path)
        scaler = joblib.load(scaler_path)
        params = joblib.load(params_path)
        
        # Create instance
        instance = cls(
            n_clusters=params['n_clusters'],
            random_state=params['random_state'],
            max_iter=params['max_iter']
        )
        
        # Set attributes
        instance.kmeans = kmeans
        instance.scaler = scaler
        instance.feature_weights = params['feature_weights']
        instance.feature_groups = params['feature_groups']
        instance.silhouette_avg = params['silhouette_avg']
        instance.cluster_counts = np.array(params['cluster_counts']) if params['cluster_counts'] is not None else None
        
        print(f"Model loaded from {model_dir}")
        return instance


def map_products_to_clusters(products_df, clusters_df):
    """
    Map products to their clusters and analyze characteristics
    
    Args:
        products_df: DataFrame with original product data
        clusters_df: DataFrame with cluster assignments
        
    Returns:
        DataFrame: Product data with cluster assignments
    """
    # Merge dataframes
    merged_df = products_df.merge(
        clusters_df[['product_id', 'cluster']], 
        on='product_id', 
        how='left'
    )
    
    # Generate cluster stats
    cluster_stats = merged_df.groupby('cluster').agg({
        'product_title': 'count',
        'min_price': 'mean',
        'max_price': 'mean',
        'has_discount': 'mean'
    }).rename(columns={'product_title': 'product_count'})
    
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    return merged_df


def find_similar_products(product_id, products_df, clusters_df, top_n=5):
    """
    Find similar products based on cluster assignment
    
    Args:
        product_id: Product ID to find similar products for
        products_df: DataFrame with original product data
        clusters_df: DataFrame with cluster assignments
        top_n: Number of similar products to return
        
    Returns:
        DataFrame: Similar products
    """
    # Find the cluster for the given product
    product_cluster = clusters_df[clusters_df['product_id'] == product_id]['cluster'].values
    
    if len(product_cluster) == 0:
        print(f"Product ID {product_id} not found")
        return pd.DataFrame()
        
    product_cluster = product_cluster[0]
    
    # Get products in the same cluster
    cluster_products = clusters_df[clusters_df['cluster'] == product_cluster]
    
    # Exclude the query product
    similar_products = cluster_products[cluster_products['product_id'] != product_id]
    
    # Return top N similar products
    return similar_products.head(top_n)


# Create a function to run the clustering as part of the main script
def run_clustering(input_file, output_dir="results", n_clusters=10, find_optimal_k=False,
                  min_k=10, max_k=30, save_model=True):
    """
    Run weighted KMeans clustering on processed product data
    
    Args:
        input_file: Path to the input CSV file with processed features
        output_dir: Directory to save results
        n_clusters: Number of clusters to create
        find_optimal_k: Whether to find the optimal number of clusters
        min_k: Minimum number of clusters to try
        max_k: Maximum number of clusters to try
        save_model: Whether to save the model
        
    Returns:
        DataFrame: Product data with cluster assignments
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Keep original data for reference
    orig_df = df.copy()
    
    # Initialize model
    model = WeightedKMeans(n_clusters=n_clusters)
    
    # Identify feature groups
    feature_df = df.copy()
    
    # Save reference columns before removing them
    product_ids = df['product_id'].copy() if 'product_id' in df.columns else None
    product_titles = df['product_title'].copy() if 'product_title' in df.columns else None
    
    # Remove non-feature columns
    non_feature_cols = ['product_id', 'product_title', 'handle']
    feature_cols = [col for col in feature_df.columns if col not in non_feature_cols]
    feature_df = feature_df[feature_cols]
    
    # Store reference columns in model for later use
    model.product_ids = product_ids
    model.product_titles = product_titles
    
    # Identify feature groups
    try:
        feature_groups = model.identify_feature_groups(feature_df)
    except Exception as e:
        print(f"Error identifying feature groups: {str(e)}")
        return pd.DataFrame()
    
    # Set custom weights
    try:
        weights = {
            'product_type': 1.8,        # Higher weight for product types  
            'tags': 1.8,                # Higher weight for tags
            'description_tfidf': 1.0,   # Standard weight
            'metafield_data': 1.0,      # Standard weight
            'collections': 1.0,         # Standard weight
            'vendor': 1.0,              # Standard weight
            'variant_size': 1.0,        # Standard weight
            'variant_color': 1.0,       # Standard weight
            'variant_other': 1.0,       # Standard weight
            'numerical': 1.0            # Standard weight
        }
        model.set_feature_weights(weights)
    except Exception as e:
        print(f"Error setting feature weights: {str(e)}")
        # Continue with default weights
    
    # Find optimal k if requested
    k_range = None
    if find_optimal_k:
        k_range = range(min_k, max_k + 1)
    
    # Fit model
    try:
        model.fit(feature_df, find_optimal_k=find_optimal_k, k_range=k_range)
    except Exception as e:
        print(f"Error during model fitting: {str(e)}")
        return pd.DataFrame()
    
    # Only proceed with visualizations and exports if model was fitted successfully
    if model.kmeans is None:
        print("Model was not fitted successfully. Skipping visualizations and exports.")
        return pd.DataFrame()
    
    # Create visualizations - wrap each in try/except to prevent one failure from stopping everything
    try:
        model.visualize_clusters(output_file=f"{output_dir}/clusters.png")
    except Exception as e:
        print(f"Error creating cluster visualization: {str(e)}")
    
    try:
        model.visualize_feature_importance(feature_df.columns, output_file=f"{output_dir}/feature_importance.png")
    except Exception as e:
        print(f"Error creating feature importance visualization: {str(e)}")
    
    # t-SNE visualization with extra safety
    try:
        # Create a clean copy of the data for t-SNE
        tsne_data = model.scaler.transform(feature_df)
        # Replace any NaN or inf values
        tsne_data = np.nan_to_num(tsne_data, nan=0.0, posinf=1e9, neginf=-1e9)
        model.visualize_tsne(tsne_data, output_file=f"{output_dir}/tsne.png")
    except Exception as e:
        print(f"Error creating t-SNE visualization: {str(e)}")
    
    # Export results - again with error handling
    centroids_df = None
    assignments_df = None
    
    try:
        centroids_df = model.export_centroids(output_file=f"{output_dir}/centroids.csv")
    except Exception as e:
        print(f"Error exporting centroids: {str(e)}")
    
    try:
        assignments_df = model.export_cluster_assignments(output_file=f"{output_dir}/assignments.csv")
    except Exception as e:
        print(f"Error exporting cluster assignments: {str(e)}")
        return pd.DataFrame()  # Can't continue without assignments
    
    # Interpret clusters
    try:
        model.interpret_clusters(feature_df)
    except Exception as e:
        print(f"Error interpreting clusters: {str(e)}")
    
    # Evaluate clusters
    try:
        model.evaluate_clusters()
    except Exception as e:
        print(f"Error evaluating clusters: {str(e)}")
    
    # Map products to clusters
    mapped_df = pd.DataFrame()
    try:
        if assignments_df is not None:
            mapped_df = map_products_to_clusters(orig_df, assignments_df)
            mapped_df.to_csv(f"{output_dir}/products_with_clusters.csv", index=False)
    except Exception as e:
        print(f"Error mapping products to clusters: {str(e)}")
    
    # Example: Find similar products
    try:
        if len(df) > 0 and assignments_df is not None:
            sample_product_id = df['product_id'].iloc[0]
            similar_products = find_similar_products(sample_product_id, orig_df, assignments_df)
            
            print("\nExample: Similar Products")
            print(f"For product: {df[df['product_id'] == sample_product_id]['product_title'].iloc[0]}")
            if not similar_products.empty:
                for _, row in similar_products.iterrows():
                    print(f"- {row['product_title']}")
    except Exception as e:
        print(f"Error finding similar products: {str(e)}")
    
    # Save model if requested
    if save_model:
        try:
            model.save_model(f"{output_dir}/model")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    print(f"\nClustering complete! Results saved to {output_dir}")
    
    return mapped_df


# Now modify the main script to call the clustering
from shopifyInfo.shopify_queries import get_all_products
from preprocessing.tfidf_processor import process_descriptions_tfidf
from preprocessing.collectionHandle_processor import process_collections
from preprocessing.tags_processor import process_tags
from preprocessing.variants_processor import process_variants
from preprocessing.vendor_processor import process_vendors
from preprocessing.metafields_processor import process_metafields, apply_tfidf_processing, save_color_similarity_data, remove_color_columns
from preprocessing.isGiftCard_processor import process_gif_card
from preprocessing.product_type_processor import process_product_type
from preprocessing.availableForSale_processor import process_avaiable_for_sale
import pandas as pd
import argparse


def export_sample(df, step_name):
    """Export the first row of the DataFrame to a CSV for debugging"""
    if len(df) > 0:
        sample_df = df.head(1)
        filename = f"sample_after_{step_name}.csv"
        sample_df.to_csv(filename, index=False)
        print(f"Exported sample to {filename}")


def document_feature_sources(df, output_file="feature_sources.csv"):
    """
    Document the sources of all features in the processed DataFrame
    """
    feature_sources = []
    
    # Categorize columns by their prefixes
    for column in df.columns:
        if column.startswith('description_'):
            source = 'TF-IDF from product descriptions'
        elif column.startswith('metafield_'):
            source = 'TF-IDF from product metafields'
        elif column.startswith('tag_'):
            source = 'Tag dummy variable'
        elif column.startswith('product_type_'):
            source = 'Product type dummy variable'
        elif column.startswith('collections_'):
            source = 'Collection dummy variable'
        elif column.startswith('vendor_'):
            source = 'Vendor dummy variable'
        elif column.startswith('variant_'):
            source = 'Variant option dummy variable'
        elif column.startswith('price_'):
            source = 'Price-related feature from variants'
        elif column == 'product_id' or column == 'product_title' or column == 'handle':
            source = 'Original product information'
        else:
            source = 'Other/Unknown'
        
        feature_sources.append({
            'feature_name': column,
            'source': source
        })
    
    # Create DataFrame and save to CSV
    sources_df = pd.DataFrame(feature_sources)
    sources_df.to_csv(output_file, index=False)
    print(f"Feature sources documented in {output_file}")
    
    return sources_df


def main():
    """Main function to process product data from Shopify store"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process product data and perform clustering')
    parser.add_argument('--clusters', type=int, default=10,
                      help='Number of clusters for weighted KMeans (default: 10)')
    parser.add_argument('--find-optimal-k', action='store_true',
                      help='Find optimal number of clusters')
    parser.add_argument('--optimal-k-only', action='store_true',
                      help='Only run the optimal k analysis without performing clustering')
    parser.add_argument('--min-k', type=int, default=2,
                      help='Minimum number of clusters to try when finding optimal k (default: 2)')
    parser.add_argument('--max-k', type=int, default=30,
                      help='Maximum number of clusters to try when finding optimal k (default: 30)')
    parser.add_argument('--skip-clustering', action='store_true',
                      help='Skip the clustering step')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save clustering results (default: results)')
    args = parser.parse_args()
    
    # If optimal-k-only is specified, just run that analysis and exit
    if args.optimal_k_only:
        # Check if the tfidf file exists already
        if os.path.exists("products_with_tfidf.csv"):
            print("Found existing processed data file. Using it for optimal k analysis.")
            run_optimal_k_analysis(
                input_file="products_with_tfidf.csv",
                output_dir=args.output_dir,
                min_k=args.min_k,
                max_k=args.max_k
            )
            return
        print("No existing processed data file found. Running full data processing first.")
    
    # Get shop ID from user input
    shop_id = input("Shop ID: ")
    
    print("Fetching data from Shopify...")
    raw_data = get_all_products(shop_id)
    
    if raw_data is None:
        print("Failed to retrieve data from Shopify.")
        return
    
    # Save raw data immediately after importing
    raw_data.to_csv("products.csv", index=False)
    print("Raw data saved to products.csv")
    
    # Export raw data sample
    export_sample(raw_data, "raw_data")
    
    # Start with the raw data for processing
    df = raw_data.copy()
    
    # Process the data through the pipeline
    print("\nStarting data processing pipeline...")
    
    # 1. Process metafields
    print("Processing metafields...")
    all_processed_data = []
    all_product_references = {}

    for _, row in df.iterrows():
        processed_data, product_refs = process_metafields(row, row.get('metafields_config', []))
        all_processed_data.append(processed_data)
        
        # Collect product references
        product_id = row.get('product_id', '')
        for key, value in product_refs.items():
            all_product_references[f"{product_id}_{key}"] = value

    # Create DataFrame from processed metafields
    metafields_processed = pd.DataFrame(all_processed_data, index=df.index)

    # Save product references if any were found
    if all_product_references:
        from preprocessing.metafields_processor import save_product_references
        save_product_references(all_product_references, "product_references.csv")
        print("Product references saved to product_references.csv")
    
    # Calculate and save color similarity data before removing color columns
    print("Calculating color similarities...")
    color_similarity_df = save_color_similarity_data()
    if not color_similarity_df.empty:
        print(f"Color similarity data saved for {color_similarity_df['source_product_id'].nunique()} products")
    
    # Remove color columns after similarity calculation
    print("Removing color columns after similarity calculation...")
    metafields_processed = remove_color_columns(metafields_processed)

    # Apply TF-IDF processing to the metafields data - this will also remove original text columns
    metafields_processed = apply_tfidf_processing(metafields_processed)

    # Add processed metafields to the dataframe
    if 'metafields' in df.columns:
        df = df.drop('metafields', axis=1)
    if 'metafields_config' in df.columns:
        df = df.drop('metafields_config', axis=1)

    # Combine the processed metafields with the original dataframe
    df = pd.concat([df, metafields_processed], axis=1)
    export_sample(df, "metafields")
    
    # 2. Process is_gift_card
    print("Processing is_gift_card...")
    df = process_gif_card(df)
    export_sample(df, "gift_card")

    # 3. Process available_for_sale
    print("Processing available_for_sale...")
    df = process_avaiable_for_sale(df)
    export_sample(df, "available_for_sale")

    print("Processing product types...")
    df = process_product_type(df)
    export_sample(df, "product_type")
    
    # 4. Process tags
    print("Processing tags...")
    df = process_tags(df)
    export_sample(df, "tags")
    
    # 5. Process collections
    print("Processing collections...")
    df = process_collections(df)
    export_sample(df, "collections")

    # 6. Process vendors
    print("Processing vendors...")
    df = process_vendors(df)
    export_sample(df, "vendors")
    
    # 7. Process variants
    print("Processing variants...")
    df = process_variants(df)
    export_sample(df, "variants")
    
    # 8. Process text with TF-IDF
    print("Processing product descriptions with TF-IDF...")
    tfidf_df, recommendation_df, _, similar_products = process_descriptions_tfidf(df)
    export_sample(tfidf_df, "tfidf")
    export_sample(recommendation_df, "recommendation")
    
    # Save the processed data
    df.to_csv("products_with_variants.csv", index=False)
    tfidf_df.to_csv("products_with_tfidf.csv", index=False)
    recommendation_df.to_csv("products_recommendation.csv", index=False)
    
    # Document feature sources
    document_feature_sources(tfidf_df, "feature_sources.csv")
    
    # Output completion message for data processing
    print("\nData processing complete!")
    print("\nProcessed files saved:")
    print("- products.csv (raw data)")
    print("- products_with_variants.csv (features before TF-IDF)")
    print("- products_with_tfidf.csv (complete features)")
    print("- products_recommendation.csv (ready for recommendation model)")
    print("- products_tfidf_matrix.npy (TF-IDF vector representations)")
    print("- products_tfidf_features.csv (TF-IDF feature names)")
    print("- feature_sources.csv (documentation of feature sources)")
    
    if not color_similarity_df.empty:
        print("- products_color_similarity.csv (similar products by color)")
    
    # If optimal-k-only is specified, run just that analysis and exit
    if args.optimal_k_only:
        run_optimal_k_analysis(
            input_file="products_with_tfidf.csv",
            output_dir=args.output_dir,
            min_k=args.min_k,
            max_k=args.max_k
        )
        return
    
    # Run clustering analysis if not skipped
    if not args.skip_clustering:
        print("\n" + "="*50)
        print("Starting clustering analysis...")
        print("="*50)
        
        try:
            # Run the clustering function
            clustered_df = run_clustering(
                input_file="products_with_tfidf.csv",
                output_dir=args.output_dir,
                n_clusters=args.clusters,
                find_optimal_k=args.find_optimal_k,
                min_k=args.min_k,
                max_k=args.max_k,
                save_model=True
            )
            
            if not clustered_df.empty:
                print("\nClustering analysis complete!")
                print(f"Cluster assignments and visualizations saved to {args.output_dir}")
                print("- products_with_clusters.csv (products with cluster assignments)")
            else:
                print("\nClustering analysis did not produce results.")
                print("Data processing completed successfully, but clustering was not successful.")
        except Exception as e:
            print(f"\nError during clustering analysis: {str(e)}")
            print("Data processing completed successfully, but clustering encountered an error.")
    else:
        print("\nClustering analysis skipped. Use --skip-clustering=False to run clustering.")


if __name__ == "__main__":
    main()