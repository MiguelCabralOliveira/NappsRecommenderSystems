import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ProductRecommender:
    def __init__(self, n_clusters=10, random_state=42):
        """
        Initialize the product recommender with K-means clustering
        
        Args:
            n_clusters: Number of clusters for K-means (default: 10)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.product_data = None
        self.similar_products_dict = {} 
        
    def _prepare_data(self, df):
        """
        Prepare data for clustering by selecting appropriate features
        
        Args:
            df: DataFrame with product data including TF-IDF features
            
        Returns:
            DataFrame with selected features for clustering
        """
        # Exclude these columns from feature set
        exclude_columns = [
            'product_title', 'product_id', 'vendor', 'description', 
            'handle', 'product_type', 'clean_text', 'related_products',
            'cluster'  # Also exclude the cluster column if it exists
        ]
        
        # Get all feature columns (everything not in exclude_columns)
        self.feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Create a copy of the DataFrame with selected features
        features_df = df[self.feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # First, convert all object/string columns with numbers to numeric if possible
        for col in features_df.columns:
            # Skip columns that are already numeric
            if pd.api.types.is_numeric_dtype(features_df[col]):
                continue
                
            # Try to convert to numeric, but set errors='coerce' to handle non-convertible values
            try:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                # Replace NaN values from failed conversions with 0
                features_df[col] = features_df[col].fillna(0)
            except:
                # For columns that can't be converted to numeric at all, 
                # we'll handle them separately
                pass
        
        # For any columns that still aren't numeric, use dummy variables or drop them
        non_numeric_cols = features_df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"Warning: Dropping non-numeric columns for clustering: {list(non_numeric_cols)}")
            features_df = features_df.drop(columns=non_numeric_cols)
        
        # Ensure all values are finite (replace inf with large values)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df

    def load_similarity_data(self, similarity_file):
        """
        Load similarity data from CSV file
        
        Args:
            similarity_file: Path to the similarity CSV file
        """
        try:
            similarity_df = pd.read_csv(similarity_file)
            # Filter pairs with similarity above the threshold
            high_similarity_df = similarity_df[similarity_df['similarity'] > 0.9]  # Using 0.9 as default threshold
            
            # Build a dictionary mapping each product to its highly similar products
            self.similar_products_dict = {}
            for _, row in high_similarity_df.iterrows():
                product1 = row['product1']
                product2 = row['product2']
                
                if product1 not in self.similar_products_dict:
                    self.similar_products_dict[product1] = []
                if product2 not in self.similar_products_dict:
                    self.similar_products_dict[product2] = []
                    
                self.similar_products_dict[product1].append(product2)
                self.similar_products_dict[product2].append(product1)
                
            print(f"Loaded {len(high_similarity_df)} high-similarity product pairs")
        except Exception as e:
            print(f"Error loading similarity data: {e}")
            self.similar_products_dict = {}
    
    def find_optimal_clusters(self, df, max_clusters=30):
        """
        Find optimal number of clusters using silhouette score
        
        Args:
            df: DataFrame with product data
            max_clusters: Maximum number of clusters to try
            
        Returns:
            The optimal number of clusters
        """
        print("Finding optimal number of clusters...")
        
        try:
            features_df = self._prepare_data(df)
            
            # Check if we have enough data for clustering
            if len(features_df) < 3:
                print("Not enough data for finding optimal clusters, using default value.")
                return min(self.n_clusters, len(features_df) - 1) or 1
            
            # Make sure we have valid numeric data
            if features_df.empty or features_df.isnull().all().all():
                print("No valid numeric data for clustering, using default value.")
                return min(self.n_clusters, len(df) - 1) or 1
            
            # Verify all data is numeric
            if not all(pd.api.types.is_numeric_dtype(features_df[col]) for col in features_df.columns):
                print("Non-numeric data detected, using default value.")
                return min(self.n_clusters, len(df) - 1) or 1
            
            # Scale the features
            features_scaled = self.scaler.fit_transform(features_df)
            
            # Try different numbers of clusters
            silhouette_scores = []
            min_clusters = 2  # Silhouette score requires at least 2 clusters
            
            # Limit max_clusters based on data size
            effective_max_clusters = min(max_clusters + 1, len(features_df) // 2)
            
            # Ensure we have a valid range of clusters to try
            if min_clusters >= effective_max_clusters:
                print(f"Not enough data for silhouette analysis, using default value.")
                return min(self.n_clusters, len(features_df) - 1) or 1
            
            for n_clusters in range(min_clusters, effective_max_clusters):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(features_scaled)
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
                except Exception as e:
                    print(f"Error with {n_clusters} clusters: {e}")
                    # Don't add invalid scores
            
            # Find the best number of clusters, with a fallback
            if silhouette_scores:
                optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + min_clusters
                
                # Plot silhouette scores
                plt.figure(figsize=(10, 6))
                plt.plot(range(min_clusters, min_clusters + len(silhouette_scores)), silhouette_scores, marker='o')
                plt.xlabel('Number of Clusters')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score for Different Cluster Numbers')
                plt.axvline(x=optimal_clusters, color='r', linestyle='--')
                plt.savefig('optimal_clusters.png')
                plt.close()
                
                print(f"Optimal number of clusters: {optimal_clusters}")
                return optimal_clusters
            else:
                # Fallback if no valid silhouette scores
                print("Could not determine optimal clusters, using default value.")
                return min(self.n_clusters, len(features_df) - 1) or 1
                
        except Exception as e:
            print(f"Error in find_optimal_clusters: {e}")
            return min(self.n_clusters, len(df) - 1) or 1
            
    def fit(self, df, find_optimal=True, max_clusters=30):
        """
        Fit the K-means model on product data
        
        Args:
            df: DataFrame with product data including TF-IDF features
            find_optimal: Whether to find optimal number of clusters
            max_clusters: Maximum number of clusters to try if find_optimal is True
        """
        self.product_data = df.copy()
        
        # Find optimal number of clusters if requested
        if find_optimal and len(df) > 2:
            self.n_clusters = self.find_optimal_clusters(df, max_clusters)
        
        # Prepare features
        features_df = self._prepare_data(df)
        
        # Scale the features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Apply K-means clustering
        print(f"Fitting K-means with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans.fit(features_scaled)
        
        # Add cluster labels to the product data
        self.product_data['cluster'] = self.kmeans.predict(features_scaled)
        
        # Create cluster statistics
        cluster_stats = self.product_data.groupby('cluster').agg(
            count=('product_title', 'count'),
            products=('product_title', lambda x: list(x)[:5]),  # Sample 5 products from each cluster
        )
        
        print("\nCluster Statistics:")
        print(cluster_stats[['count']])
        
        # Visualize clusters (if data allows for 2D visualization)
        self._visualize_clusters()
        
        return self
    
    def _visualize_clusters(self):
        """
        Visualize clusters using dimensionality reduction
        """
        from sklearn.decomposition import PCA
        
        if len(self.product_data) < 3:
            print("Not enough data for visualization")
            return
            
        # Important change: Use the same feature set that was used for clustering
        # Don't include the 'cluster' column that was added after fitting
        features_df = self._prepare_data(self.product_data)
        features_scaled = self.scaler.transform(features_df)
        
        # Use PCA to reduce dimensions to 2D for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create a DataFrame for plotting
        viz_df = pd.DataFrame({
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'cluster': self.product_data['cluster'],
            'product': self.product_data['product_title']
        })
        
        # Plot the clusters
        plt.figure(figsize=(12, 8))
        
        # Create a scatter plot with different colors for each cluster
        scatter = sns.scatterplot(data=viz_df, x='x', y='y', hue='cluster', palette='viridis', alpha=0.7)
        
        # Add labels to the data points
        for i in range(min(len(viz_df), 20)):  # Limit labels to prevent clutter
            plt.text(
                viz_df.iloc[i]['x'] + 0.02, viz_df.iloc[i]['y'] + 0.02,
                viz_df.iloc[i]['product'][:20] + '...' if len(viz_df.iloc[i]['product']) > 20 else viz_df.iloc[i]['product'],
                fontsize=8
            )
        
        plt.title('Product Clusters Visualization (PCA 2D projection)')
        plt.tight_layout()
        plt.savefig('product_clusters.png')
        plt.close()
        
    def get_product_cluster(self, product_id):
        """
        Get the cluster for a specific product
        
        Args:
            product_id: The product ID to look up
            
        Returns:
            The cluster number or None if product not found
        """
        product_row = self.product_data[self.product_data['product_id'] == product_id]
        if len(product_row) > 0:
            return product_row.iloc[0]['cluster']
        return None
    
    def get_same_cluster_products(self, product_id, n_recommendations=5):
        """
        Get products from the same cluster as the specified product
        
        Args:
            product_id: The product ID to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended product IDs
        """
        product_cluster = self.get_product_cluster(product_id)
        if product_cluster is None:
            return []
        
        # Get other products in the same cluster
        cluster_products = self.product_data[
            (self.product_data['cluster'] == product_cluster) & 
            (self.product_data['product_id'] != product_id)
        ]
        
        # Return top N recommendations
        recommendations = cluster_products.sort_values(
            by='rating', ascending=False
        ).head(n_recommendations)
        
        return recommendations[['product_id', 'product_title']].to_dict('records')
    
    def _filter_similar_products(self, recommendations, product_id, product_title):
        """
        Filter out products that are too similar to the query product
        
        Args:
            recommendations: List of recommendation dictionaries
            product_id: The ID of the product being recommended for
            product_title: The title of the product being recommended for
            
        Returns:
            Filtered list of recommendations
        """
        # If similarity dictionary is not initialized, return all recommendations
        if not hasattr(self, 'similar_products_dict') or not self.similar_products_dict:
            return recommendations
        
        filtered_recommendations = []
        
        # Get list of products too similar to the query product
        similar_products = self.similar_products_dict.get(product_title, [])
        
        for rec in recommendations:
            # Only include the recommendation if it's not in the similar products list
            if rec['product_title'] not in similar_products:
                filtered_recommendations.append(rec)
            else:
                print(f"Filtering out {rec['product_title']} - too similar to {product_title}")
        
        return filtered_recommendations
    
    def get_recommendations(self, product_id, n_recommendations=5, strategy='hybrid'):
        """
        Get product recommendations based on clustering
        
        Args:
            product_id: The product ID to get recommendations for
            n_recommendations: Number of recommendations to return
            strategy: Recommendation strategy
                - 'cluster': Only recommend from same cluster
                - 'collection': Prefer products from same collection
                - 'hybrid': Balanced recommendation (default)
                
        Returns:
            List of recommended product dictionaries
        """
        if self.kmeans is None:
            print("Model not fitted yet!")
            return []
            
        product_row = self.product_data[self.product_data['product_id'] == product_id]
        
        if len(product_row) == 0:
            print(f"Product {product_id} not found!")
            return []
            
        product = product_row.iloc[0]
        product_title = product['product_title']
        
        # Check if we have cluster information
        if 'cluster' not in product or pd.isna(product['cluster']):
            print("Cluster information not available for this product.")
            product_cluster = 0  # Default fallback
        else:
            product_cluster = product['cluster']
        
        # Check if product has existing product recommendations
        if 'related_products' in product and not pd.isna(product['related_products']) and product['related_products']:
            if isinstance(product['related_products'], str):
                try:
                    # Try to parse as JSON if it's a string
                    import json
                    related_products = json.loads(product['related_products'])
                    if isinstance(related_products, list) and len(related_products) > 0:
                        print(f"Using {min(len(related_products), n_recommendations)} existing recommendations")
                        existing_recommendations = []
                        for related_id in related_products[:n_recommendations]:
                            related_product = self.product_data[self.product_data['product_id'] == related_id]
                            if len(related_product) > 0:
                                existing_recommendations.append({
                                    'product_id': related_id,
                                    'product_title': related_product.iloc[0]['product_title'],
                                    'source': 'existing'
                                })
                        
                        # Filter out similar products from existing recommendations
                        filtered_recommendations = self._filter_similar_products(
                            existing_recommendations, product_id, product_title
                        )
                        return filtered_recommendations
                except:
                    pass  # If parsing fails, continue with algorithmic recommendations
        
        # Find products in the same cluster
        cluster_products = self.product_data[
            (self.product_data['cluster'] == product_cluster) & 
            (self.product_data['product_id'] != product_id)
        ]
        
        # If no products in the same cluster, use all products
        if len(cluster_products) == 0:
            print("No products found in the same cluster, using all products.")
            cluster_products = self.product_data[self.product_data['product_id'] != product_id]
        
        # Extract collection information
        product_collections = [col for col in self.product_data.columns if col.startswith('collection_') and product.get(col, 0) == 1]
        
        # Prepare recommendations based on strategy
        if strategy == 'cluster':
            # Simply recommend from the same cluster
            if 'rating' in cluster_products.columns:
                # Ensure rating is numeric
                cluster_products['rating_numeric'] = pd.to_numeric(cluster_products['rating'], errors='coerce').fillna(0)
                recommendations = cluster_products.sort_values(
                    by='rating_numeric', ascending=False
                ).head(n_recommendations * 2)  # Get more than needed to account for filtering
            else:
                recommendations = cluster_products.head(n_recommendations * 2)
                
        elif strategy == 'collection':
            # Prioritize same collection products
            if len(product_collections) > 0:
                # Find products in the same collections
                collection_filter = self.product_data[product_collections].sum(axis=1) > 0
                collection_products = self.product_data[
                    collection_filter & 
                    (self.product_data['product_id'] != product_id)
                ]
                
                # If enough collection products, use them, otherwise fall back to cluster
                if len(collection_products) >= n_recommendations:
                    if 'rating' in collection_products.columns:
                        # Ensure rating is numeric
                        collection_products['rating_numeric'] = pd.to_numeric(collection_products['rating'], errors='coerce').fillna(0)
                        recommendations = collection_products.sort_values(
                            by='rating_numeric', ascending=False
                        ).head(n_recommendations * 2)
                    else:
                        recommendations = collection_products.head(n_recommendations * 2)
                else:
                    # Fill remaining slots with cluster recommendations
                    if 'rating' in collection_products.columns:
                        # Ensure rating is numeric
                        collection_products['rating_numeric'] = pd.to_numeric(collection_products['rating'], errors='coerce').fillna(0)
                        collection_recs = collection_products.sort_values(by='rating_numeric', ascending=False)
                    else:
                        collection_recs = collection_products
                    
                    remaining_slots = n_recommendations * 2 - len(collection_recs)
                    
                    # Exclude already selected products
                    already_selected = set(collection_recs['product_id'])
                    remaining_cluster_products = cluster_products[
                        ~cluster_products['product_id'].isin(already_selected)
                    ]
                    
                    if 'rating' in remaining_cluster_products.columns:
                        # Ensure rating is numeric
                        remaining_cluster_products['rating_numeric'] = pd.to_numeric(remaining_cluster_products['rating'], errors='coerce').fillna(0)
                        cluster_recs = remaining_cluster_products.sort_values(
                            by='rating_numeric', ascending=False
                        ).head(remaining_slots)
                    else:
                        cluster_recs = remaining_cluster_products.head(remaining_slots)
                    
                    # Combine recommendations
                    recommendations = pd.concat([collection_recs, cluster_recs])
            else:
                # No collection info, fall back to cluster
                if 'rating' in cluster_products.columns:
                    # Ensure rating is numeric
                    cluster_products['rating_numeric'] = pd.to_numeric(cluster_products['rating'], errors='coerce').fillna(0)
                    recommendations = cluster_products.sort_values(
                        by='rating_numeric', ascending=False
                    ).head(n_recommendations * 2)
                else:
                    recommendations = cluster_products.head(n_recommendations * 2)
                    
        else:  # hybrid strategy
            # Calculate a weighted score that considers both cluster and collection
            all_candidate_products = self.product_data[
                self.product_data['product_id'] != product_id
            ].copy()
            
            # Initialize score
            all_candidate_products['rec_score'] = 0
            
            # Add score for cluster match (2 points)
            all_candidate_products.loc[
                all_candidate_products['cluster'] == product_cluster, 'rec_score'
            ] += 2
            
            # Add score for collection match (3 points)
            if len(product_collections) > 0:
                collection_score = all_candidate_products[product_collections].sum(axis=1)
                all_candidate_products['rec_score'] += collection_score * 3
            
            # Add rating score (up to 1 point)
            if 'rating' in all_candidate_products.columns:
                # Ensure rating is numeric
                all_candidate_products['rating_numeric'] = pd.to_numeric(all_candidate_products['rating'], errors='coerce').fillna(0)
                max_rating = all_candidate_products['rating_numeric'].max()
                if max_rating > 0:
                    all_candidate_products['rec_score'] += all_candidate_products['rating_numeric'] / max_rating
            
            # Get top recommendations (get more than needed to account for filtering)
            recommendations = all_candidate_products.sort_values(
                by='rec_score', ascending=False
            ).head(n_recommendations * 2)
        
        # If we don't have enough recommendations, just return what we have
        if len(recommendations) == 0:
            print("No recommendations found for this product.")
            return []
        
        # Convert to list of dictionaries
        recommendation_list = recommendations[['product_id', 'product_title']].to_dict('records')
        
        # Add the source of recommendation
        for rec in recommendation_list:
            rec['source'] = strategy
        
        # Filter out similar products
        filtered_recommendations = self._filter_similar_products(
            recommendation_list, product_id, product_title
        )
        
        # Return only the requested number of recommendations
        return filtered_recommendations[:n_recommendations]

    def generate_all_recommendations(self, product_df, n_recommendations=5, strategy='hybrid'):
        """
        Generate recommendations for all products in the dataframe
        
        Args:
            product_df: DataFrame containing product data
            n_recommendations: Number of recommendations to generate per product
            strategy: Recommendation strategy ('hybrid', 'cluster', or 'collection')
            
        Returns:
            DataFrame with recommendations for each product
        """
        import pandas as pd
        from tqdm import tqdm  # For progress bar
        
        # Create an empty list to store recommendation data
        all_recommendations = []
        
        # Process each product
        print(f"Generating {n_recommendations} {strategy} recommendations for {len(product_df)} products...")
        
        # Use tqdm for progress tracking
        for _, row in tqdm(product_df.iterrows(), total=len(product_df)):
            product_id = row['product_id']
            product_title = row['product_title']
            
            # Get recommendations for this product
            recommendations = self.get_recommendations(
                product_id, 
                n_recommendations=n_recommendations, 
                strategy=strategy
            )
            
            # If recommendations were found, add them to our list
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    all_recommendations.append({
                        'source_product_id': product_id,
                        'source_product_title': product_title,
                        'recommendation_rank': i,
                        'recommended_product_id': rec['product_id'],
                        'recommended_product_title': rec['product_title'],
                        'recommendation_source': rec['source']
                    })
        
        # Convert list to DataFrame
        recommendations_df = pd.DataFrame(all_recommendations)
        
        return recommendations_df

    def save_model(self, filename='recommender_model.pkl'):
        """
        Save the trained model to a file
        
        Args:
            filename: Output filename
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'n_clusters': self.n_clusters,
                'feature_columns': self.feature_columns,
                'random_state': self.random_state
            }, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, product_data, filename='recommender_model.pkl'):
        """
        Load a trained model from a file
        
        Args:
            product_data: DataFrame with product data
            filename: Input filename
            
        Returns:
            Loaded ProductRecommender instance
        """
        import pickle
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        recommender = cls(n_clusters=model_data['n_clusters'], random_state=model_data['random_state'])
        recommender.kmeans = model_data['kmeans']
        recommender.scaler = model_data['scaler']
        recommender.feature_columns = model_data['feature_columns']
        recommender.product_data = product_data
        
        # Predict clusters for the provided product data
        features_df = recommender._prepare_data(product_data)
        features_scaled = recommender.scaler.transform(features_df)
        recommender.product_data['cluster'] = recommender.kmeans.predict(features_scaled)
        
        return recommender