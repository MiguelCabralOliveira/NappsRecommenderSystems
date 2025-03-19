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
import warnings
from collections import defaultdict
import re


class _WeightedKMeans:
    """
    Private implementation of weighted KMeans clustering algorithm for product data
    where different feature groups can be assigned different weights.
    This class should NOT be instantiated directly outside of the run_clustering function.
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
            
        print(f"Loading data from {data_file}")
        df = pd.read_csv(data_file, low_memory=False)
        
        # Keep product ID and title for reference
        self.product_ids = df['product_id'].copy() if 'product_id' in df.columns else None
        self.product_titles = df['product_title'].copy() if 'product_title' in df.columns else None
        
        # Remove non-feature columns
        non_feature_cols = ['product_id', 'product_title', 'handle']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        feature_df = df[feature_cols]
        
        # Filter out columns with non-numeric data
        numeric_cols = []
        for col in feature_df.columns:
            try:
                # Try to convert a sample to float
                pd.to_numeric(feature_df[col].dropna().head(1))
                numeric_cols.append(col)
            except (ValueError, TypeError):
                print(f"Skipping non-numeric column: {col}")
        
        # Keep only numeric columns for clustering
        if len(numeric_cols) < len(feature_df.columns):
            dropped_cols = [col for col in feature_df.columns if col not in numeric_cols]
            print(f"Removed {len(dropped_cols)} non-numeric columns from clustering")
            feature_df = feature_df[numeric_cols]
        
        print(f"Loaded {len(df)} products with {len(feature_df.columns)} usable numeric features")
        
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
            'numerical': [],
            'time_features': [],      
            'release_quarter': [],    
            'seasons': []              
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
            # Seasonal feature
            elif col == 'created_season':
                feature_groups['seasons'].append(col)
            # Time features from createdAt processor
            elif (col.startswith('created_') and col != 'created_season') or col == 'days_since_first_product' or col == 'is_recent_product':
                feature_groups['time_features'].append(col)
            # Release quarter dummies
            elif col.startswith('release_quarter_'):
                feature_groups['release_quarter'].append(col)
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
    
    def set_feature_weights(self, base_weights=None, normalize_by_count=True):
        """
        Set weights for each feature group, with optional normalization by feature count
        
        Args:
            base_weights: Dictionary mapping feature group names to base weights
                        If None, default weights will be used
            normalize_by_count: Whether to normalize weights by feature count
        
        Returns:
            dict: The effective feature weights
        """
        if base_weights is None:
            
            base_weights = {
                'tags': 1.8,          
                'product_type': 1.8,        
                'description_tfidf': 1.0,   
                'metafield_data': 1.0,      
                'collections': 1.4,         
                'vendor': 1.0,              
                'variant_size': 1.0,        
                'variant_color': 1.0,       
                'variant_other': 1.0,       
                'numerical': 1.0,            
                'time_features' : 1.0,
                'release_quarter': 1.5,
                'seasons' : 1.4
            }
        
        for group in self.feature_groups:
            if group not in base_weights:
                base_weights[group] = 1.0  
        
        # Count features in each group
        feature_counts = {group: len(features) for group, features in self.feature_groups.items()}
        
        # Print feature counts
        print("\nFeature Count per Group:")
        for group, count in feature_counts.items():
            print(f"{group}: {count} features")
        
        # Calculate effective weights based on feature counts
        if normalize_by_count:
        
            target_count = 30  
            
            
            # Formula: effective_weight = base_weight * (target_count / feature_count)
            self.feature_weights = {}
            for group in self.feature_groups:
                count = feature_counts.get(group, 1)  # Avoid division by zero
                norm_factor = target_count / count
                self.feature_weights[group] = base_weights[group] * norm_factor
        else:
            # Use base weights directly if not normalizing
            self.feature_weights = base_weights
        
        # Print effective weights
        print("\nEffective Feature Weights:")
        for group, weight in self.feature_weights.items():
            if group in self.feature_groups:
                base = base_weights[group]
                count = feature_counts.get(group, 0)
                print(f"{group}: {weight:.4f} (base: {base}, count: {count})")
        
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
            
            # Check for non-numeric values and convert columns to numeric if possible
            for col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(0)  # Replace NaNs from failed conversions with 0
                    except:
                        print(f"Warning: Could not convert column {col} to numeric type")
            
            # Check for infinite values in numeric columns
            try:
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    inf_counts = np.isinf(df[numeric_cols].values).sum()
                    if inf_counts > 0:
                        print(f"Warning: Found {inf_counts} infinite values in the dataset. Replacing with large values.")
                        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], [1e9, -1e9])
            except Exception as e:
                print(f"Warning when checking for infinities: {e}")
            
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
                
            print("\n" + "="*200)
            print("FINDING OPTIMAL NUMBER OF CLUSTERS")
            print("="*200)
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
            
            print("\n" + "="*200)
            print("OPTIMAL NUMBER OF CLUSTERS RESULTS")
            print("="*200)   
            print(f"Davies-Bouldin Index (lower is better): k = {best_k_db}")
            print(f"Silhouette Score (higher is better): k = {best_k_silhouette}")
            print(f"Calinski-Harabasz Index (higher is better): k = {best_k_ch}")
            print(f"Inertia Elbow Point: k = {elbow_k}")
            print("\nRecommended number of clusters: k = {optimal_k}")
            print("(Based on Davies-Bouldin with early stopping)")
            print("="*200)
            
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
                return 12 # Absolute default

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
            print("Skipping t-SNE visualization and continuing with the analysis.")
    
    def get_top_features_per_cluster(self, feature_names, top_n=10):
        """
        Get top features for each cluster, accounting for feature weights
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Ensure feature_names is a list
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
            
        # Get cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Create a mapping from feature index to its group
        feature_to_group = {}
        idx = 0
        for group, features in self.feature_groups.items():
            for _ in features:
                if idx < len(feature_names):
                    feature_to_group[idx] = group
                    idx += 1
        
        top_features = {}
        
        for cluster_idx in range(self.n_clusters):
            # Get raw importance of each feature for this cluster
            raw_importance = centers[cluster_idx]
            
            # Adjust importance by feature weights for ranking purposes
            weighted_importance = np.zeros_like(raw_importance)
            for i, imp in enumerate(raw_importance):
                if i < len(feature_names):  # Safety check
                    group = feature_to_group.get(i, None)
                    if group and group in self.feature_weights:
                        # Apply weights to adjust importance
                        weighted_importance[i] = imp * self.feature_weights[group]
                    else:
                        weighted_importance[i] = imp
            
            # Get index of top features based on weighted importance
            top_indices = np.argsort(weighted_importance)[-top_n:][::-1]
            
            # Map indices to feature names with their raw values
            # We display raw values but sort by weighted importance
            top_features[cluster_idx] = [(feature_names[i], raw_importance[i]) 
                                        for i in top_indices 
                                        if i < len(feature_names)]
        
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
            
            # Calculate silhouette score - we already have this from fitting
            silhouette = self.silhouette_avg
            
            # Calculate inertia (sum of squared distances)
            inertia = self.kmeans.inertia_
            
            # Store results
            results = {
                'silhouette': silhouette,
                'inertia': inertia
            }
            
            print("\nCluster Evaluation Metrics:")
            print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
            print(f"Inertia: {inertia:.4f} (lower is better)")
            
            return results
            
        except Exception as e:
            print(f"Error evaluating clusters: {str(e)}")
            return {}
    
    def interpret_clusters(self, df, top_n=12):
        """
        Interpret clusters with enforced diversity of feature types
        """
        feature_names = df.columns.tolist()
        
        # Categorize all features
        feature_categories = {
            'description_tfidf': [],
            'product_type': [],
            'tags': [],
            'collections': [],
            'metafield_data': [],
            'vendor': [],
            'variant_size': [],
            'variant_color': [],
            'variant_other': [],
            'numerical': []
        }
        
        for i, name in enumerate(feature_names):
            if name.startswith('description_'):
                feature_categories['description_tfidf'].append(i)
            elif name.startswith('product_type_'):
                feature_categories['product_type'].append(i)
            elif name.startswith('tag_'):
                feature_categories['tags'].append(i)
            elif name.startswith('collections_'):
                feature_categories['collections'].append(i)
            elif name.startswith('metafield_'):
                feature_categories['metafield_data'].append(i)
            elif name.startswith('vendor_'):
                feature_categories['vendor'].append(i)
            elif name.startswith('variant_size_'):
                feature_categories['variant_size'].append(i)
            elif name.startswith('variant_color_'):
                feature_categories['variant_color'].append(i)
            elif name.startswith('variant_other_'):
                feature_categories['variant_other'].append(i)
            elif name in ['min_price', 'max_price', 'has_discount']:
                feature_categories['numerical'].append(i)
        
        interpretations = {}
        print("\nCluster Interpretations:")
        
        for cluster_idx in range(self.n_clusters):
            # Get raw centroid values
            centroid = self.kmeans.cluster_centers_[cluster_idx]
            
            # Create a balanced selection from different feature types
            balanced_features = []
            
            # Limit description features to max 2
            desc_indices = feature_categories['description_tfidf']
            if desc_indices:
                top_desc = sorted([(i, centroid[i]) for i in desc_indices], 
                                key=lambda x: x[1], reverse=True)[:2]
                balanced_features.extend(top_desc)
            
            # For each category, add the top feature if available
            for category, indices in feature_categories.items():
                if category == 'description_tfidf':
                    continue  # Already handled
                
                if indices:
                    top_feature = max([(i, centroid[i]) for i in indices], 
                                    key=lambda x: x[1])
                    if top_feature[1] > 0:  # Only add if value is positive
                        balanced_features.append(top_feature)
            
            # Sort by value for display
            balanced_features.sort(key=lambda x: x[1], reverse=True)
            
            # Format interpretation
            interpretation = f"Cluster {cluster_idx} ({self.cluster_counts[cluster_idx]} products):\n"
            interpretation += "Key characteristics:\n"
            
            for idx, value in balanced_features[:top_n]:
                name = feature_names[idx]
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
            
        # Make sure we don't have more centroids columns than feature names
        centers = self.kmeans.cluster_centers_
        if centers.shape[1] > len(feature_names):
            print(f"Warning: Centroid dimensions ({centers.shape[1]}) exceed feature names count ({len(feature_names)})")
            # Extend feature names with generic names
            additional_names = [f"feature_{i}" for i in range(len(feature_names), centers.shape[1])]
            feature_names.extend(additional_names)
        
        # Create DataFrame with centroids
        centroids_df = pd.DataFrame(
            self.kmeans.cluster_centers_[:, :len(feature_names)],
            columns=feature_names[:centers.shape[1]]
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
            
        # Check if we have product IDs
        if self.product_ids is None:
            print("Warning: No product IDs available. Using index as product identifier.")
            product_ids = [f"product_{i}" for i in range(len(self.kmeans.labels_))]
        else:
            product_ids = self.product_ids
            
        # Check if we have product titles
        if self.product_titles is None:
            product_titles = [""] * len(self.kmeans.labels_)
        else:
            product_titles = self.product_titles
        
        # Ensure lengths match
        min_length = min(len(product_ids), len(product_titles), len(self.kmeans.labels_))
        
        # Create DataFrame with cluster assignments
        assignments_df = pd.DataFrame({
            'product_id': product_ids[:min_length],
            'product_title': product_titles[:min_length],
            'cluster': self.kmeans.labels_[:min_length]
        })
        
        # Save to CSV
        assignments_df.to_csv(output_file, index=False)
        print(f"Cluster assignments exported to {output_file}")
        
        return assignments_df
    
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
    
    # Check for products that weren't assigned to clusters
    unassigned = merged_df['cluster'].isna().sum()
    if unassigned > 0:
        print(f"Warning: {unassigned} products were not assigned to any cluster")
    
    # Generate cluster stats
    try:
        numeric_cols = ['min_price', 'max_price', 'has_discount']
        stats_cols = [col for col in numeric_cols if col in merged_df.columns]
        
        if stats_cols:
            cluster_stats = merged_df.groupby('cluster').agg({
                **{'product_id': 'count'},
                **{col: 'mean' for col in stats_cols}
            }).rename(columns={'product_id': 'product_count'})
            
            print("\nCluster Statistics:")
            print(cluster_stats)
    except Exception as e:
        print(f"Error generating cluster statistics: {str(e)}")
    
    return merged_df


def find_similar_products(product_id, products_df, clusters_df, similar_products_df=None, top_n=12):
    """
    Find similar products based on cluster assignment, excluding too-similar variants
    
    Args:
        product_id: Product ID to find similar products for
        products_df: DataFrame with original product data
        clusters_df: DataFrame with cluster assignments
        similar_products_df: DataFrame with pairs of too-similar products to exclude
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
    
    # Get the product title for filtering
    product_title = products_df[products_df['product_id'] == product_id]['product_title'].iloc[0]
    
    # Get products in the same cluster
    cluster_products = clusters_df[clusters_df['cluster'] == product_cluster]
    
    # Exclude the query product
    similar_products = cluster_products[cluster_products['product_id'] != product_id]
    
    # Filter out too-similar products if we have similarity data
    if similar_products_df is not None and not similar_products_df.empty:
        # Get titles of all recommended products
        recommended_titles = products_df.loc[products_df['product_id'].isin(similar_products['product_id']), 'product_title']
        
        # Find titles to exclude (too similar to the query product)
        exclude_titles = []
        
        # Check if the product is in product1 column
        matches1 = similar_products_df[similar_products_df['product1'] == product_title]
        if not matches1.empty:
            exclude_titles.extend(matches1['product2'].tolist())
            
        # Check if the product is in product2 column
        matches2 = similar_products_df[similar_products_df['product2'] == product_title]
        if not matches2.empty:
            exclude_titles.extend(matches2['product1'].tolist())
            
        # Filter the recommendations to exclude too-similar products
        if exclude_titles:
            filtered_product_ids = []
            for pid in similar_products['product_id']:
                title = products_df[products_df['product_id'] == pid]['product_title'].iloc[0]
                if title not in exclude_titles:
                    filtered_product_ids.append(pid)
            
            similar_products = similar_products[similar_products['product_id'].isin(filtered_product_ids)]
    
    # Return top N similar products
    return similar_products.head(top_n)



def run_clustering(input_file, output_dir="results", n_clusters=10, find_optimal_k=True,
                   min_k=10, max_k=50, save_model=True, similar_products_df=None):
    """
    Main public interface for running weighted KMeans clustering on processed product data.
    This is the ONLY function that should be called directly from outside this module.
    
    Args:
        input_file: Path to the input CSV file with processed features
        output_dir: Directory to save results
        n_clusters: Number of clusters to create
        find_optimal_k: Whether to find the optimal number of clusters
        min_k: Minimum number of clusters to try
        max_k: Maximum number of clusters to try
        save_model: Whether to save the model
        similar_products_df: DataFrame with pairs of similar products (directly passed)
        
    Returns:
        DataFrame: Product data with cluster assignments
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure warnings to be more visible
    warnings.filterwarnings('always', category=UserWarning)
    
    # Load data
    print(f"Loading data from {input_file}")
    try:
        # Add low_memory=False to handle mixed types
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Keep original data for reference
    orig_df = df.copy()
    
    # Handle the similar products dataframe
    if similar_products_df is None or similar_products_df.empty:
        # Create empty DataFrame if none provided
        similar_products_df = pd.DataFrame(columns=['product1', 'product2', 'similarity'])
        print("No similarity data provided, recommendations won't filter variants")
    else:
        print(f"Using provided similarity data for {len(similar_products_df)} product pairs")
    
    # Initialize model
    model = _WeightedKMeans(n_clusters=n_clusters)
    
    # The load_data method will now automatically filter non-numeric columns
    feature_df = model.load_data(input_file)
    
    # Identify feature groups
    try:
        feature_groups = model.identify_feature_groups(feature_df)
    except Exception as e:
        print(f"Error identifying feature groups: {str(e)}")
        return pd.DataFrame()
    
    # Set custom weights
    try:
        base_weights = {
            'product_type': 1.8,        # Higher weight for product types  
            'tags': 1.8,                # Higher weight for tags
            'description_tfidf': 1.0,   # Standard weight
            'metafield_data': 1.0,      # Standard weight
            'collections': 1.0,         # Standard weight
            'vendor': 1.0,              # Standard weight
            'variant_size': 1.0,        # Standard weight
            'variant_color': 1.0,       # Standard weight
            'variant_other': 1.0,       # Standard weight
            'numerical': 1.0,           # Standard weight
            'time_features': 1.0,       # Standard weight
            'release_quarter': 1.5,     # Higher weight for release quarters
            'seasons': 1.4              # Higher weight for seasons
        }
        model.set_feature_weights(base_weights, normalize_by_count=True)
    except Exception as e:
        print(f"Error setting feature weights: {str(e)}")
    
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
    
    # Set custom weights
    try:
        base_weights = {
            'product_type': 1.8,        # Higher weight for product types  
            'tags': 1.8,                # Higher weight for tags
            'description_tfidf': 1.0,   # Standard weight
            'metafield_data': 1.0,      # Standard weight
            'collections': 1.0,         # Standard weight
            'vendor': 1.0,              # Standard weight
            'variant_size': 1.0,        # Standard weight
            'variant_color': 1.0,       # Standard weight
            'variant_other': 1.0,       # Standard weight
            'numerical': 1.0,           # Standard weight
            'time_features': 1.0,       # Standard weight
            'release_quarter': 1.5,     # Higher weight for release quarters
            'seasons': 1.4              # Higher weight for seasons
        }
        model.set_feature_weights(base_weights, normalize_by_count=True)
    except Exception as e:
        print(f"Error setting feature weights: {str(e)}")
    
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
    
    # Track visualization success/failure
    visualizations_successful = {
        'clusters': False,
        'feature_importance': False,
        'tsne': False,
        'centroids': False,
        'assignments': False,
        'interpretations': False,
        'evaluation': False
    }
    
    # Create visualizations - wrap each in try/except but always attempt all visualizations
    print("\nGenerating visualizations and analysis...")
    
    # 1. Cluster visualization
    try:
        model.visualize_clusters(output_file=f"{output_dir}/clusters.png")
        visualizations_successful['clusters'] = True
    except Exception as e:
        print(f"Error creating cluster visualization: {str(e)}")
    
    # 2. Feature importance visualization
    try:
        model.visualize_feature_importance(feature_df.columns, output_file=f"{output_dir}/feature_importance.png")
        visualizations_successful['feature_importance'] = True
    except Exception as e:
        print(f"Error creating feature importance visualization: {str(e)}")
    
    # 3. t-SNE visualization
    try:
        # Create a clean copy of the data for t-SNE
        tsne_data = model.scaler.transform(feature_df)
        # Replace any NaN or inf values
        tsne_data = np.nan_to_num(tsne_data, nan=0.0, posinf=1e9, neginf=-1e9)
        model.visualize_tsne(tsne_data, output_file=f"{output_dir}/tsne.png")
        visualizations_successful['tsne'] = True
    except Exception as e:
        print(f"Error creating t-SNE visualization: {str(e)}")
    
    # 4. Export centroids
    centroids_df = None
    try:
        centroids_df = model.export_centroids(output_file=f"{output_dir}/centroids.csv")
        visualizations_successful['centroids'] = True
    except Exception as e:
        print(f"Error exporting centroids: {str(e)}")
    
    # 5. Export cluster assignments
    assignments_df = None
    try:
        assignments_df = model.export_cluster_assignments(output_file=f"{output_dir}/assignments.csv")
        visualizations_successful['assignments'] = True
    except Exception as e:
        print(f"Error exporting cluster assignments: {str(e)}")
        # Don't return early, try to generate other visualizations
    
    # 6. Interpret clusters
    try:
        model.interpret_clusters(feature_df)
        visualizations_successful['interpretations'] = True
    except Exception as e:
        print(f"Error interpreting clusters: {str(e)}")
    
    # 7. Evaluate clusters
    try:
        model.evaluate_clusters()
        visualizations_successful['evaluation'] = True
    except Exception as e:
        print(f"Error evaluating clusters: {str(e)}")
    
    # Check if we have any missing visualizations
    missing_visualizations = [name for name, success in visualizations_successful.items() if not success]
    if missing_visualizations:
        print(f"\nWarning: The following visualizations could not be generated: {', '.join(missing_visualizations)}")
    
    # Map products to clusters
    mapped_df = pd.DataFrame()
    if assignments_df is not None:
        try:
            mapped_df = map_products_to_clusters(orig_df, assignments_df)
            mapped_df.to_csv(f"{output_dir}/products_with_clusters.csv", index=False)
            print(f"Products with cluster assignments saved to {output_dir}/products_with_clusters.csv")
        except Exception as e:
            print(f"Error mapping products to clusters: {str(e)}")
    
    # Example: Find similar products for a sample product
    if len(df) > 0 and assignments_df is not None:
        try:
            sample_product_id = df['product_id'].iloc[0]
            similar_products = find_similar_products(
                sample_product_id, 
                orig_df, 
                assignments_df, 
                similar_products_df,
                top_n=12
            )
            
            print("\nExample: Similar Products")
            print(f"For product: {df[df['product_id'] == sample_product_id]['product_title'].iloc[0]}")
            if not similar_products.empty:
                for _, row in similar_products.iterrows():
                    print(f"- {row['product_title']}")
                    
                # Save similar products to CSV for the sample
                similar_products.to_csv(f"{output_dir}/similar_products_example.csv", index=False)
                print(f"Example similar products saved to {output_dir}/similar_products_example.csv")
        except Exception as e:
            print(f"Error generating similar products example: {str(e)}")
    
    # Save model if requested
    if save_model:
        try:
            model.save_model(f"{output_dir}/model")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    # Generate summary report
    try:
        with open(f"{output_dir}/clustering_summary.txt", "w", encoding='utf-8') as f:
            f.write("=== Clustering Analysis Summary ===\n\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Number of products: {len(df)}\n")
            f.write(f"Number of features: {len(feature_df.columns)}\n")
            f.write(f"Number of clusters: {model.n_clusters}\n\n")
            
            f.write("Cluster distribution:\n")
            for i, count in enumerate(model.cluster_counts):
                f.write(f"Cluster {i}: {count} products ({count/len(df)*100:.1f}%)\n")
            
            f.write("\nSilhouette Score: {:.4f}\n".format(model.silhouette_avg))
            
            f.write("\nFiles generated:\n")
            for key, success in visualizations_successful.items():
                status = "✓" if success else "✗"
                f.write(f"{status} {key}\n")
        
        print(f"Clustering summary saved to {output_dir}/clustering_summary.txt")
    except Exception as e:
        print(f"Error generating summary report: {str(e)}")
    
    print(f"\nClustering complete! Results saved to {output_dir}")
    
    return mapped_df