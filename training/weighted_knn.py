import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os
import joblib
import warnings
from collections import defaultdict
import re

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
        self.similarity_matrix = None
        
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
        
        # Keep only numeric columns for similarity computation
        if len(numeric_cols) < len(feature_df.columns):
            dropped_cols = [col for col in feature_df.columns if col not in numeric_cols]
            print(f"Removed {len(dropped_cols)} non-numeric columns from feature set")
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
                'time_features': 1.0,
                'release_quarter': 1.5,
                'seasons': 1.4
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
        
    def fit(self, data):
        """
        Fit the weighted KNN model to the data
        
        Args:
            data: DataFrame or path to CSV file
            
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
            
            # Fit KNN model
            print(f"\nFitting KNN with {self.n_neighbors} neighbors...")
            self.knn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors + 1,  # +1 because the product itself will be included
                algorithm=self.algorithm,
                metric=self.metric
            )
            self.knn_model.fit(X_weighted)
            
            # Store the weighted feature matrix for later use
            self.X_weighted = X_weighted
            
            # Compute similarity matrix (all product pairs)
            print("Computing similarity matrix (this may take a while for large datasets)...")
            self.distances, self.indices = self.knn_model.kneighbors(X_weighted)
            
            # Create recommendation matrix - each row is a product, each column is a recommended product
            self.recommendation_matrix = {}
            for i in range(len(self.indices)):
                product_id = self.product_ids.iloc[i] if self.product_ids is not None else f"product_{i}"
                # Skip the first neighbor (itself) and take the next n_neighbors
                neighbor_indices = self.indices[i, 1:]
                neighbor_ids = [self.product_ids.iloc[idx] if self.product_ids is not None else f"product_{idx}" 
                              for idx in neighbor_indices]
                neighbor_distances = self.distances[i, 1:]
                # Convert distances to similarities (1 / (1 + distance))
                similarities = 1 / (1 + neighbor_distances)
                
                # Store as a list of (product_id, similarity) tuples
                self.recommendation_matrix[product_id] = list(zip(neighbor_ids, similarities))
            
            print(f"Successfully fitted KNN model to {len(df)} products")
            print(f"Each product has {self.n_neighbors} recommended similar products")
                
            return self
            
        except Exception as e:
            print(f"Error during KNN fitting: {str(e)}")
            import traceback
            traceback.print_exc()
            print("KNN fitting failed but continuing with the rest of the pipeline.")
            return self
    
    def get_recommendations(self, product_id, n_recommendations=None):
        """
        Get recommendations for a specific product
        
        Args:
            product_id: ID of the product to get recommendations for
            n_recommendations: Number of recommendations to return (default: self.n_neighbors)
            
        Returns:
            list: List of (product_id, similarity_score) tuples
        """
        if self.recommendation_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if product_id not in self.recommendation_matrix:
            print(f"Product ID {product_id} not found in the training data")
            return []
        
        if n_recommendations is None:
            n_recommendations = self.n_neighbors
            
        # Return the precomputed recommendations
        return self.recommendation_matrix[product_id][:n_recommendations]
    
    def get_recommendations_for_all(self, n_recommendations=None):
        """
        Get recommendations for all products
        
        Args:
            n_recommendations: Number of recommendations to return per product
            
        Returns:
            dict: Dictionary mapping product IDs to their recommendations
        """
        if self.recommendation_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        if n_recommendations is None:
            n_recommendations = self.n_neighbors
            
        # Return recommendations for all products, limited to n_recommendations
        return {product_id: recs[:n_recommendations] for product_id, recs in self.recommendation_matrix.items()}
    
    def visualize_feature_importance(self, feature_names, output_file="feature_importance.png"):
        """
        Visualize feature importance based on feature weights
        
        Args:
            feature_names: List of feature names
            output_file: Path to save the visualization
            
        Returns:
            None
        """
        try:
            if isinstance(feature_names, pd.Index):
                feature_names = feature_names.tolist()
            
            # Create feature importance based on weights
            feature_importance = np.zeros(len(feature_names))
            start_idx = 0
            
            for group, features in self.feature_groups.items():
                weight = self.feature_weights.get(group, 1.0)
                for i in range(len(features)):
                    if start_idx < len(feature_importance):
                        feature_importance[start_idx] = weight
                        start_idx += 1
            
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
            plt.xlabel('Feature Weight')
            plt.title('Top Features by Weight')
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
            else:
                X_sample = X_weighted
                indices = np.arange(X_weighted.shape[0])
            
            # Apply t-SNE
            print("Applying t-SNE dimensionality reduction (this may take a while)...")
            perplexity = min(30, X_sample.shape[0] - 1)  # Adjust perplexity for small datasets
            tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity)
            X_tsne = tsne.fit_transform(X_sample)
            
            # Determine point colors based on nearest neighbor relationships
            # Products with similar recommendations will have similar colors
            if self.indices is not None:
                # Create a smaller subset of indices corresponding to our t-SNE sample
                sample_indices = self.indices[indices]
                
                # Use the first neighbor (excluding self) to determine color
                # This way products that recommend similar items will have similar colors
                colors = sample_indices[:, 1] % 20  # Modulo to limit the number of colors
            else:
                colors = np.zeros(len(X_sample))
            
            # Plot
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='tab20', alpha=0.7)
            plt.colorbar(scatter, label='Product Group')
            plt.title('t-SNE Visualization of Products')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"t-SNE visualization saved to {output_file}")
        except Exception as e:
            print(f"Error during t-SNE visualization: {str(e)}")
            print("Skipping t-SNE visualization and continuing with the analysis.")
    
    def visualize_similarity_distribution(self, output_file="similarity_distribution.png"):
        """
        Visualize the distribution of similarity scores
        
        Args:
            output_file: Path to save the visualization
            
        Returns:
            None
        """
        try:
            if self.distances is None:
                print("Model not fitted yet. Skipping similarity distribution visualization.")
                return
                
            # Convert distances to similarities (1 / (1 + distance))
            similarities = 1 / (1 + self.distances[:, 1:])  # Exclude self-similarity
            
            # Flatten the similarity matrix
            flat_similarities = similarities.flatten()
            
            # Create a histogram
            plt.figure(figsize=(10, 6))
            plt.hist(flat_similarities, bins=50, alpha=0.7)
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Product Similarity Scores')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Similarity distribution visualization saved to {output_file}")
            
            # Also calculate some statistics
            print("\nSimilarity Statistics:")
            print(f"Mean similarity: {np.mean(flat_similarities):.4f}")
            print(f"Median similarity: {np.median(flat_similarities):.4f}")
            print(f"Min similarity: {np.min(flat_similarities):.4f}")
            print(f"Max similarity: {np.max(flat_similarities):.4f}")
            
        except Exception as e:
            print(f"Error during similarity distribution visualization: {str(e)}")
            print("Skipping similarity distribution visualization and continuing with the analysis.")
    
    def save_model(self, output_dir="models"):
        """
        Save the model to disk
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            None
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save KNN model
        joblib.dump(self.knn_model, f"{output_dir}/knn_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        
        # Save recommendations matrix (excluding actual model as it can be large)
        joblib.dump(self.recommendation_matrix, f"{output_dir}/recommendation_matrix.pkl")
        
        # Save weights and groups
        joblib.dump({
            'feature_weights': self.feature_weights,
            'feature_groups': self.feature_groups,
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'random_state': self.random_state
        }, f"{output_dir}/model_params.pkl")
        
        print(f"Model saved to {output_dir}")
    
    def load_model(self, input_dir="models"):
        """
        Load the model from disk
        
        Args:
            input_dir: Directory to load the model from
            
        Returns:
            self: The loaded model
        """
        # Check if directory exists
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Model directory not found: {input_dir}")
        
        # Load KNN model
        knn_path = f"{input_dir}/knn_model.pkl"
        if os.path.exists(knn_path):
            self.knn_model = joblib.load(knn_path)
        else:
            print(f"Warning: KNN model file not found at {knn_path}")
        
        # Load scaler
        scaler_path = f"{input_dir}/scaler.pkl"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: Scaler file not found at {scaler_path}")
        
        # Load recommendation matrix
        rec_path = f"{input_dir}/recommendation_matrix.pkl"
        if os.path.exists(rec_path):
            self.recommendation_matrix = joblib.load(rec_path)
        else:
            print(f"Warning: Recommendation matrix file not found at {rec_path}")
        
        # Load model parameters
        params_path = f"{input_dir}/model_params.pkl"
        if os.path.exists(params_path):
            params = joblib.load(params_path)
            self.feature_weights = params.get('feature_weights')
            self.feature_groups = params.get('feature_groups')
            self.n_neighbors = params.get('n_neighbors', self.n_neighbors)
            self.algorithm = params.get('algorithm', self.algorithm)
            self.metric = params.get('metric', self.metric)
            self.random_state = params.get('random_state', self.random_state)
        else:
            print(f"Warning: Model parameters file not found at {params_path}")
        
        print(f"Model loaded from {input_dir}")
        return self
    
    def export_recommendations(self, output_file="recommendations.csv", n_recommendations=None):
        """
        Export all product recommendations to a CSV file
        
        Args:
            output_file: Path to save the recommendations
            n_recommendations: Number of recommendations to export per product
            
        Returns:
            DataFrame: DataFrame with recommendations
        """
        if self.recommendation_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if n_recommendations is None:
            n_recommendations = self.n_neighbors
        
        # Prepare data for export
        rows = []
        for product_id, recommendations in self.recommendation_matrix.items():
            product_recommendations = recommendations[:n_recommendations]
            for rank, (rec_id, similarity) in enumerate(product_recommendations):
                rows.append({
                    'source_product_id': product_id,
                    'recommended_product_id': rec_id,
                    'rank': rank + 1,
                    'similarity_score': similarity
                })
        
        # Create DataFrame
        recommendations_df = pd.DataFrame(rows)
        
        # Save to CSV
        recommendations_df.to_csv(output_file, index=False)
        print(f"Recommendations exported to {output_file}")
        
        return recommendations_df


def find_similar_products(product_id, knn_model, products_df, top_n=12, exclude_similar_variants=True, similar_products_df=None):
    """
    Find similar products using the KNN model
    
    Args:
        product_id: Product ID to find similar products for
        knn_model: Fitted _WeightedKNN model
        products_df: DataFrame with original product data
        top_n: Number of similar products to return
        exclude_similar_variants: Whether to exclude products that are too similar (likely variants)
        similar_products_df: DataFrame with pairs of too-similar products to exclude
        
    Returns:
        DataFrame: Similar products
    """
    # Get recommendations from the KNN model
    recommendations = knn_model.get_recommendations(product_id, n_recommendations=top_n * 2)  # Get more than needed in case we filter some out
    
    if not recommendations:
        print(f"No recommendations found for product ID {product_id}")
        return pd.DataFrame()
    
    # Get the product title for filtering
    product_title = products_df[products_df['product_id'] == product_id]['product_title'].iloc[0] if 'product_title' in products_df.columns else None
    
    # Extract product IDs and similarities
    similar_product_ids = [rec_id for rec_id, _ in recommendations]
    similarities = [sim for _, sim in recommendations]
    
    # Create initial DataFrame with recommendations
    similar_products = pd.DataFrame({
        'product_id': similar_product_ids,
        'similarity': similarities
    })
    
    # Add product titles if available
    if 'product_title' in products_df.columns:
        product_titles = {}
        for pid in similar_product_ids:
            try:
                title = products_df[products_df['product_id'] == pid]['product_title'].iloc[0]
                product_titles[pid] = title
            except (IndexError, KeyError):
                product_titles[pid] = f"Unknown Product {pid}"
        
        similar_products['product_title'] = similar_products['product_id'].map(product_titles)
    
    # Filter out too-similar products if requested and we have similarity data
    if exclude_similar_variants and similar_products_df is not None and not similar_products_df.empty and product_title is not None:
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
        if exclude_titles and 'product_title' in similar_products.columns:
            similar_products = similar_products[~similar_products['product_title'].isin(exclude_titles)]
    
    # Return top N similar products
    return similar_products.head(top_n)

def run_knn(df, output_dir="results", n_neighbors=12, save_model=True, similar_products_df=None):
    """
    Main public interface for running weighted KNN for product recommendations.
    This function should be called directly from outside this module.
    
    Args:
        df: DataFrame with processed features
        output_dir: Directory to save results
        n_neighbors: Number of neighbors to find for each product
        save_model: Whether to save the model
        similar_products_df: DataFrame with pairs of similar products (directly passed)
        
    Returns:
        DataFrame: DataFrame with all product recommendations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure warnings to be more visible
    warnings.filterwarnings('always', category=UserWarning)
    
    # Check if DataFrame is valid
    if df is None or len(df) == 0:
        print("Error: Empty or invalid DataFrame provided")
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
    model = _WeightedKNN(n_neighbors=n_neighbors)
    
    # Extract features for KNN (this replaces the load_data method call)
    feature_df = model._prepare_features(df)
    
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
    
    # Fit the KNN model
    try:
        model.fit(feature_df)
    except Exception as e:
        print(f"Error during model fitting: {str(e)}")
        return pd.DataFrame()
    
    # Track visualization success/failure
    visualizations_successful = {
        'feature_importance': False,
        'tsne': False,
        'similarity_distribution': False,
        'recommendations': False
    }
    
    # Create visualizations - wrap each in try/except but always attempt all visualizations
    print("\nGenerating visualizations and analysis...")
    
    # 1. Feature importance visualization
    try:
        model.visualize_feature_importance(feature_df.columns, output_file=f"{output_dir}/feature_importance.png")
        visualizations_successful['feature_importance'] = True
    except Exception as e:
        print(f"Error creating feature importance visualization: {str(e)}")
    
    # 2. t-SNE visualization
    try:
        # Create a clean copy of the data for t-SNE
        tsne_data = model.scaler.transform(feature_df)
        # Replace any NaN or inf values
        tsne_data = np.nan_to_num(tsne_data, nan=0.0, posinf=1e9, neginf=-1e9)
        model.visualize_tsne(tsne_data, output_file=f"{output_dir}/tsne.png")
        visualizations_successful['tsne'] = True
    except Exception as e:
        print(f"Error creating t-SNE visualization: {str(e)}")
    
    # 3. Similarity distribution visualization
    try:
        model.visualize_similarity_distribution(output_file=f"{output_dir}/similarity_distribution.png")
        visualizations_successful['similarity_distribution'] = True
    except Exception as e:
        print(f"Error creating similarity distribution visualization: {str(e)}")
    
    # 4. Export recommendations
    all_recommendations_df = None
    try:
        all_recommendations_df = model.export_recommendations(output_file=f"{output_dir}/all_recommendations.csv")
        visualizations_successful['recommendations'] = True
    except Exception as e:
        print(f"Error exporting recommendations: {str(e)}")
    
    # Check if we have any missing visualizations
    missing_visualizations = [name for name, success in visualizations_successful.items() if not success]
    if missing_visualizations:
        print(f"\nWarning: The following visualizations could not be generated: {', '.join(missing_visualizations)}")
    
    # Example: Find similar products for a sample product
    if len(df) > 0 and 'product_id' in df.columns:
        try:
            sample_product_id = df['product_id'].iloc[0]
            similar_products = find_similar_products(
                sample_product_id, 
                model, 
                orig_df,
                top_n=12,
                similar_products_df=similar_products_df
            )
            
            print("\nExample: Similar Products")
            if 'product_title' in df.columns:
                print(f"For product: {df[df['product_id'] == sample_product_id]['product_title'].iloc[0]}")
            
            if not similar_products.empty:
                if 'product_title' in similar_products.columns:
                    for _, row in similar_products.iterrows():
                        print(f"- {row['product_title']} (similarity: {row['similarity']:.4f})")
                else:
                    for _, row in similar_products.iterrows():
                        print(f"- Product ID: {row['product_id']} (similarity: {row['similarity']:.4f})")
                    
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
        with open(f"{output_dir}/knn_summary.txt", "w", encoding='utf-8') as f:
            f.write("=== KNN Recommendation Summary ===\n\n")
            f.write(f"Number of products: {len(df)}\n")
            f.write(f"Number of features: {len(feature_df.columns)}\n")
            f.write(f"Number of neighbors per product: {model.n_neighbors}\n\n")
            
            f.write("Feature Group Weights:\n")
            for group, weight in model.feature_weights.items():
                f.write(f"{group}: {weight:.4f}\n")
            
            f.write("\nFiles generated:\n")
            for key, success in visualizations_successful.items():
                status = "✓" if success else "✗"
                f.write(f"{status} {key}\n")
        
        print(f"KNN summary saved to {output_dir}/knn_summary.txt")
    except Exception as e:
        print(f"Error generating summary report: {str(e)}")
    
    print(f"\nKNN recommendation model complete! Results saved to {output_dir}")
    
    return all_recommendations_df

def run_hybrid_recommendations(input_file, output_dir="results", method="knn", 
                        n_neighbors=12, save_model=True, similar_products_df=None):
    """
    Main public interface for running KNN-based product recommendations.
    This unified function exists for backward compatibility.
    
    Args:
        input_file: Path to the input CSV file with processed features
        output_dir: Directory to save results
        method: Recommendation method (only "knn" is supported in this version)
        n_neighbors: Number of neighbors to find for each product
        save_model: Whether to save the model
        similar_products_df: DataFrame with pairs of similar products (directly passed)
        
    Returns:
        DataFrame: Recommendations data
    """
    method = method.lower()
    
    if method == "knn" or method != "kmeans":  # Default to KNN if not kmeans
        print(f"Running K-NN based recommendations with {n_neighbors} neighbors per product...")
        return run_knn(
            input_file=input_file, 
            output_dir=output_dir, 
            n_neighbors=n_neighbors, 
            save_model=save_model, 
            similar_products_df=similar_products_df
        )
    else:
        raise ValueError("This version only supports KNN. The K-Means implementation has been removed.")

# Extension to find_similar_products to work with product_id
def get_product_recommendations(product_id, model, products_df, top_n=12, 
                               exclude_similar_variants=True, similar_products_df=None):
    """
    Unified function to get product recommendations
    
    Args:
        product_id: Product ID to find similar products for
        model: Fitted _WeightedKNN model
        products_df: DataFrame with original product data
        top_n: Number of similar products to return
        exclude_similar_variants: Whether to exclude products that are too similar
        similar_products_df: DataFrame with pairs of too-similar products to exclude
        
    Returns:
        DataFrame: Similar products with similarity scores
    """
    if hasattr(model, 'get_recommendations'):
        return find_similar_products(
            product_id=product_id,
            knn_model=model,
            products_df=products_df,
            top_n=top_n,
            exclude_similar_variants=exclude_similar_variants,
            similar_products_df=similar_products_df
        )
    else:
        raise ValueError("Model is not a valid KNN model")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run product recommendations')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with processed features')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--neighbors', type=int, default=12, help='Number of neighbors to find per product')
    parser.add_argument('--skip-model-save', action='store_true', help='Skip saving the model')
    
    args = parser.parse_args()
    
    # Run the KNN model
    recommendations_df = run_knn(
        input_file=args.input,
        output_dir=args.output_dir,
        n_neighbors=args.neighbors,
        save_model=not args.skip_model_save
    )
    
    print(f"Finished generating recommendations for {len(recommendations_df['source_product_id'].unique())} products")