# weighted_kmeans.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from collections import defaultdict
import re
import argparse

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
        """
        Load data from a processed CSV file
        
        Args:
            data_file: Path to the processed CSV file (e.g., products_with_tfidf.csv)
            
        Returns:
            DataFrame: Loaded data
        """
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
        
        print(f"Loaded {len(df)} products with {len(feature_cols)} features")
        return df[feature_cols]
    
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
        # Load data if string is provided
        if isinstance(data, str):
            df = self.load_data(data)
        else:
            df = data.copy()
            
        # Identify feature groups if not already done
        if self.feature_groups is None:
            self.identify_feature_groups(df)
            
        # Set default weights if not already done
        if self.feature_weights is None:
            self.set_feature_weights()
        
        # Standardize the data
        X = self.scaler.fit_transform(df)
        
        # Apply weights to features
        X_weighted = self.apply_weights(X)
        
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
    
    def find_optimal_k(self, X, k_range=None):
        """
        Find optimal number of clusters using Davies-Bouldin Index
        Lower Davies-Bouldin values indicate better clustering.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to try
            
        Returns:
            int: Optimal number of clusters
        """
        if k_range is None:
            k_range = range(10, 31)  # Start from 10 clusters and go up to 30
            
        print("\nFinding optimal number of clusters using Davies-Bouldin Index...")
        davies_bouldin_scores = []
        silhouette_scores = []
        
        previous_db_score = float('inf')
        consecutive_increases = 0
        optimal_k = k_range[0]  # Default to first k
        
        for k in k_range:
            print(f"Trying k={k}...", end=" ")
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, max_iter=self.max_iter).fit(X)
            
            # Calculate Davies-Bouldin Index (lower is better)
            db_score = davies_bouldin_score(X, kmeans.labels_)
            davies_bouldin_scores.append(db_score)
            
            # Also calculate silhouette score for reference
            sil_score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(sil_score)
            
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
                    print(f"Davies-Bouldin Index has increased for 3 consecutive k values. Stopping search.")
                    break
            
            previous_db_score = db_score
        
        print(f"Optimal number of clusters (lowest Davies-Bouldin Index): {optimal_k}")
        
        # Update n_clusters
        self.n_clusters = optimal_k
        
        # Plot Davies-Bouldin scores
        plt.figure(figsize=(12, 8))
        
        # First subplot for Davies-Bouldin Index
        plt.subplot(2, 1, 1)
        plt.plot(k_range[:len(davies_bouldin_scores)], davies_bouldin_scores, 'o-', color='red')
        plt.axvline(x=optimal_k, color='green', linestyle='--')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('Davies-Bouldin Index For Optimal k (Lower is Better)')
        plt.grid(True)
        
        # Second subplot for Silhouette Score
        plt.subplot(2, 1, 2)
        plt.plot(k_range[:len(silhouette_scores)], silhouette_scores, 'o-', color='blue')
        plt.axvline(x=optimal_k, color='green', linestyle='--')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score For Reference (Higher is Better)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_k_davies_bouldin.png')
        plt.close()
        
        return optimal_k

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
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Get data and apply weights
        df = self.kmeans.cluster_centers_
        
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
    
    def visualize_feature_importance(self, feature_names, output_file="feature_importance.png"):
        """
        Visualize feature importance across all clusters
        
        Args:
            feature_names: List of feature names
            output_file: Path to save the visualization
            
        Returns:
            None
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
            
        # Get weighted cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Calculate overall feature importance as the mean absolute value across clusters
        feature_importance = np.mean(np.abs(centers), axis=0)
        
        # Get top features
        num_features = min(20, len(feature_importance))
        top_indices = np.argsort(feature_importance)[-num_features:]
        
        # Create a horizontal bar chart
        plt.figure(figsize=(12, 10))
        y_pos = np.arange(len(top_indices))
        
        # Create readable feature names
        display_names = []
        for idx in top_indices:
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
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Apply weights
        X_weighted = self.apply_weights(X)
        
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
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get the data and labels
        X = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        X_weighted = self.apply_weights(self.scaler.transform(X))
        labels = self.kmeans.labels_
        
        # Calculate silhouette score
        silhouette = self.silhouette_avg
        
        # Calculate Davies-Bouldin index
        db_score = davies_bouldin_score(X_weighted, labels)
        
        # Calculate Calinski-Harabasz index
        ch_score = calinski_harabasz_score(X_weighted, labels)
        
        # Calculate inertia (sum of squared distances)
        inertia = self.kmeans.inertia_
        
        # Store results
        results = {
            'silhouette': silhouette,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score,
            'inertia': inertia
        }
        
        print("\nCluster Evaluation Metrics:")
        print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
        print(f"Calinski-Harabasz Index: {ch_score:.4f} (higher is better)")
        print(f"Inertia: {inertia:.4f} (lower is better)")
        
        return results
    
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


def main():
    """Main function to run clustering"""
    parser = argparse.ArgumentParser(description='Run Weighted KMeans clustering on product data')
    parser.add_argument('--input', type=str, default='products_with_tfidf.csv',
                      help='Path to input CSV file with processed product data')
    parser.add_argument('--clusters', type=int, default=10,
                      help='Number of clusters to create (default is 10)')
    parser.add_argument('--find-optimal-k', action='store_true',
                      help='Find optimal number of clusters')
    parser.add_argument('--min-k', type=int, default=10,
                      help='Minimum number of clusters to try when finding optimal k (default is 10)')
    parser.add_argument('--max-k', type=int, default=30,
                      help='Maximum number of clusters to try when finding optimal k (default is 30)')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--save-model', action='store_true',
                      help='Save the model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Keep original data for reference
    orig_df = df.copy()
    
    # Initialize model with default 10 clusters
    model = WeightedKMeans(n_clusters=args.clusters)
    
    # Identify feature groups
    feature_df = df.copy()
    
    # Remove non-feature columns
    non_feature_cols = ['product_id', 'product_title', 'handle']
    feature_cols = [col for col in feature_df.columns if col not in non_feature_cols]
    feature_df = feature_df[feature_cols]
    
    # Identify feature groups
    feature_groups = model.identify_feature_groups(feature_df)
    
    # Set custom weights - giving higher importance to product_type and tags
    # These are easily adjustable for future fine-tuning
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
    
    # Find optimal k if requested
    k_range = None
    if args.find_optimal_k:
        k_range = range(args.min_k, args.max_k + 1)
    
    # Fit model
    model.fit(feature_df, find_optimal_k=args.find_optimal_k, k_range=k_range)
    
    # Create visualizations
    model.visualize_clusters(output_file=f"{args.output_dir}/clusters.png")
    model.visualize_feature_importance(feature_df.columns, output_file=f"{args.output_dir}/feature_importance.png")
    
    # t-SNE visualization on standardized data
    model.visualize_tsne(model.scaler.transform(feature_df), output_file=f"{args.output_dir}/tsne.png")
    
    # Export results
    model.export_centroids(output_file=f"{args.output_dir}/centroids.csv")
    assignments_df = model.export_cluster_assignments(output_file=f"{args.output_dir}/assignments.csv")
    
    # Interpret clusters
    model.interpret_clusters(feature_df)
    
    # Evaluate clusters with enhanced metrics
    model.evaluate_clusters()
    
    # Map products to clusters
    mapped_df = map_products_to_clusters(orig_df, assignments_df)
    mapped_df.to_csv(f"{args.output_dir}/products_with_clusters.csv", index=False)
    
    # Example: Find similar products
    if len(df) > 0:
        sample_product_id = df['product_id'].iloc[0]
        similar_products = find_similar_products(sample_product_id, orig_df, assignments_df)
        
        print("\nExample: Similar Products")
        print(f"For product: {df[df['product_id'] == sample_product_id]['product_title'].iloc[0]}")
        if not similar_products.empty:
            for _, row in similar_products.iterrows():
                print(f"- {row['product_title']}")
    
    # Save model if requested
    if args.save_model:
        model.save_model(f"{args.output_dir}/model")
    
    print(f"\nClustering complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()