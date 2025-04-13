# Shopify Product Recommendation System

## Overview

This project implements a hybrid product recommendation system specifically designed for Shopify stores. It leverages content-based filtering using a weighted K-Nearest Neighbors (KNN) approach on product features extracted from the Shopify API and combines these recommendations with collaborative filtering signals derived from recent order data (popularity, co-purchase patterns).

The system performs the following key steps:
1.  **Data Fetching:** Retrieves product data (including metadata, variants, collections, etc.) from the Shopify Storefront GraphQL API and recent order data from a PostgreSQL database.
2.  **Data Preprocessing:** Cleans, transforms, and engineers features from the raw data. This includes handling HTML, text vectorization (TF-IDF), categorical encoding (dummy variables), numerical scaling, timestamp processing, and specific handling for Shopify metafields (including colors, dimensions, weights, and references).
3.  **Weighted KNN Modeling:** Fits a KNN model where different groups of features (e.g., text, price, tags) can be assigned different weights, allowing for fine-tuning of similarity calculations. Includes optional weight optimization using Grid Search, Random Search, or Evolutionary Algorithms.
4.  **Hybrid Recommendation Generation:** Combines the content-based KNN similarity scores with product popularity and co-purchase frequency data to produce a final, ranked list of hybrid recommendations.
5.  **Evaluation:** Assesses the quality of the generated recommendations using metrics beyond simple accuracy, such as Catalog Coverage, Intra-List Similarity (ILS), and Popularity Bias.
6.  **Output Generation:** Saves processed data, feature information, recommendation lists (KNN raw, KNN filtered, Hybrid), model files, evaluation results, and visualizations to a specified output directory.

## Key Features

*   **Hybrid Approach:** Combines content-based KNN with collaborative signals (popularity, co-purchase groups) for potentially more relevant and diverse recommendations.
*   **Rich Feature Engineering:** Processes a wide range of Shopify product attributes including titles, descriptions, handles, tags, collections, product types, vendors, variants (price, options), metafields (text, numeric, color, dimension, weight, references), and creation timestamps.
*   **Weighted KNN:** Allows assigning different importance levels to various feature groups during similarity calculation.
*   **Weight Optimization:** Includes methods (Grid, Random, Evolutionary) to automatically find potentially better feature weights based on maximizing average KNN similarity.
*   **Metafield Processing:** Sophisticated handling of Shopify metafields, including automatic type detection, unit conversion (weight, dimension), color similarity calculation (CIEDE2000), reference extraction, and text vectorization.
*   **Configurable Pipeline:** Uses command-line arguments to control various aspects like the number of neighbors (K), output directory, skipping steps, enabling optimization, and tuning hybrid parameters.
*   **Comprehensive Evaluation:** Measures recommendation quality using Catalog Coverage, Intra-List Similarity (ILS), and Popularity Bias @ K.
*   **Detailed Output:** Generates multiple CSV files for raw data, processed features, intermediate steps, final recommendations, feature sources, and evaluation metrics, along with plots for feature importance, data distribution (t-SNE), and similarity scores.
*   **Robust Error Handling:** Includes `try-except` blocks and detailed logging/tracebacks to handle potential issues during data fetching, processing, or model training.

## Workflow / Pipeline (`main.py`)

1.  **Initialization:** Parse command-line arguments, set up the output directory.
2.  **Fetch Order Data:** Query the database (`database_query.py`) for recent order items to derive `popular_products_df` and `product_groups_df`. Handle potential errors gracefully.
3.  **Fetch Shopify Data:**
    *   Authenticate and get metafield definitions using `metafields_fetcher.py`.
    *   Get store URL and token using `shop_settings.py` or the fetcher.
    *   Fetch all products recursively using pagination via Shopify GraphQL API (`shopify_queries.py`). Includes metafields, variants, collections, etc.
4.  **Data Preprocessing Pipeline:** Apply a sequence of transformations using modules in the `preprocessing/` directory:
    *   `metafields_processor.py`: Parses complex metafield data, extracts colors, dimensions, weights, references. Calculates color similarity, applies TF-IDF to text metafields.
    *   `isGiftCard_processor.py`: Filters out gift card products.
    *   `availableForSale_processor.py`: Filters out products not available for sale.
    *   `product_type_processor.py`: Converts product types to dummy variables.
    *   `tags_processor.py`: Converts tags to dummy variables.
    *   `collectionHandle_processor.py`: Converts collections to dummy variables.
    *   `vendor_processor.py`: Converts vendors to dummy variables.
    *   `variants_processor.py`: Extracts price features (min/max, discount) and variant options (size, color, other) as dummy variables.
    *   `createdAt_processor.py`: Extracts and normalizes time-based features from the product creation date.
    *   `title_processor.py`: Applies TF-IDF to product titles.
    *   `handle_processor.py`: Applies TF-IDF to product handles.
    *   `tfidf_processor.py`: Applies TF-IDF to product descriptions. Also identifies potentially very similar products (e.g., variants) based on description similarity.
5.  **Feature Documentation:** Generate `feature_sources.csv` detailing the origin of each column in the final feature set.
6.  **Weighted KNN:** (`weighted_knn.py`)
    *   **Weight Optimization (Optional):** If enabled via `--optimize-weights`, use `WeightOptimizer` to find best feature group weights using the specified method (`--optimization-method`). Saves optimization results.
    *   **Model Fitting:** Fit the `_WeightedKNN` model using either optimized or default weights. This involves scaling features, applying weights, and finding nearest neighbors.
    *   **Export Raw KNN:** Save raw KNN recommendations *before* variant filtering (`knn_recommendations_pre_variant_filter.csv`). This is crucial input for evaluation (ILS) and the hybrid combiner.
    *   **Export Filtered KNN:** Export KNN recommendations *after* filtering out potentially similar variants identified earlier (`all_recommendations.csv`).
    *   **Visualizations:** Generate plots for feature importance (assigned weights), t-SNE (data distribution), and similarity score distribution.
    *   **Save Model:** Optionally save the fitted KNN model, scaler, and parameters using `joblib`.
7.  **Hybrid Recommendation Generation:** (`recommendation_combiner.py`)
    *   If KNN was successful, load the raw KNN recommendations (`knn_recommendations_pre_variant_filter.csv`).
    *   Load popularity and group data.
    *   Combine KNN similarity with popularity and group frequency boosts using an adaptive strategy.
    *   Re-rank recommendations based on the hybrid score.
    *   Save the final hybrid recommendations (`hybrid_recommendations.csv`).
8.  **Evaluation:** (`evaluation.py`)
    *   Evaluate the *raw* KNN recommendations (`knn_recommendations_pre_variant_filter.csv`).
    *   Evaluate the *final* recommendations (either hybrid or filtered KNN, whichever was generated last).
    *   Optionally evaluate the *filtered* KNN recommendations if different from the final ones.
    *   Calculate Catalog Coverage, Intra-List Similarity (ILS), and Popularity Bias @ K.
    *   Save evaluation metrics to JSON files and generate distribution plots for ILS and Popularity Bias.
9.  **Completion:** Print summary and indicate pipeline finish.

## Directory Structure

├── shopifyInfo/
│ ├── init.py
│ ├── database_query.py # Fetches order data from PostgreSQL DB
│ ├── metafields_fetcher.py # Handles Napps auth & fetches metafield definitions
│ ├── shop_settings.py # Fetches Shopify store URL & storefront token
│ └── shopify_queries.py # Fetches product data via Shopify GraphQL API
├── preprocessing/
│ ├── init.py
│ ├── availableForSale_processor.py # Filters unavailable products
│ ├── collectionHandle_processor.py # Processes collections -> dummies
│ ├── createdAt_processor.py # Processes creation timestamp features
│ ├── description_processor.py # Processes descriptions -> Word2Vec (UNUSED?)
│ ├── handle_processor.py # Processes handles -> TF-IDF
│ ├── isGiftCard_processor.py # Filters gift cards
│ ├── metafields_processor.py # Processes complex metafields (key module)
│ ├── product_type_processor.py # Processes product types -> dummies
│ ├── tags_processor.py # Processes tags -> dummies
│ ├── tfidf_processor.py # Processes descriptions -> TF-IDF
│ ├── title_processor.py # Processes titles -> TF-IDF
│ ├── variants_processor.py # Extracts features from variants
│ └── vendor_processor.py # Processes vendors -> dummies
├── training/
│ ├── init.py
│ ├── evaluation.py # Calculates recommendation evaluation metrics
│ ├── recommendation_combiner.py# Combines KNN with popularity/groups
│ └── weighted_knn.py # Implements Weighted KNN algorithm & optimization
├── main.py # Main script to run the pipeline
├── requirements.txt # (Assumed) Python package dependencies
└── README.md # This file

## Modules Description

*   **`main.py`**: The main executable script. Orchestrates the entire data fetching, preprocessing, training, combination, and evaluation pipeline. Handles command-line arguments.
*   **`shopifyInfo/database_query.py`**: Connects to a PostgreSQL database and fetches recent order items, deriving popularity counts and co-purchase groups.
*   **`shopifyInfo/metafields_fetcher.py`**: Handles a multi-step authentication process with a specific backend (`master.napps-solutions.com`) to retrieve metafield definitions (key, namespace, type) for a given shop.
*   **`shopifyInfo/shop_settings.py`**: Retrieves basic shop settings (Store URL, Storefront Access Token) from the Napps backend.
*   **`shopifyInfo/shopify_queries.py`**: Uses the Shopify Storefront GraphQL API to fetch product data, including variants, collections, and specified metafields, handling pagination.
*   **`preprocessing/*.py`**: Each file in this directory is responsible for processing a specific attribute of the product data (e.g., `tags_processor.py` converts tags list into dummy variables). `metafields_processor.py` is particularly complex, handling various metafield types, units, colors, and references.
*   **`training/weighted_knn.py`**: Implements the core Weighted KNN logic (`_WeightedKNN` class). Includes feature scaling, weight application, neighbor finding, model saving/loading, and visualization (importance, t-SNE, similarity distribution). Also contains the `WeightOptimizer` class for optimizing feature weights.
*   **`training/recommendation_combiner.py`**: Implements the logic to combine KNN similarity scores with popularity and co-purchase group data to generate a final hybrid recommendation list using boosting factors.
*   **`training/evaluation.py`**: Calculates non-accuracy based evaluation metrics: Catalog Coverage@K, Intra-List Similarity (ILS)@K, and Popularity Bias@K. Saves results to JSON and generates distribution plots.

## Setup / Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or
    venv\Scripts\activate    # Windows
    ```
3.  **Install Dependencies:** Make sure you have a `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries include:* `pandas`, `numpy`, `scikit-learn`, `requests`, `psycopg2-binary` (or `psycopg2`), `python-dotenv`, `joblib`, `matplotlib`, `seaborn`, `beautifulsoup4`, `lxml`, `SQLAlchemy`, `colormath`, `nltk`.
4.  **Download NLTK Data:** The text processing steps require NLTK data. Run this in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```
5.  **Configure Environment Variables:** Create a `.env` file in the project root directory.

## Configuration (`.env` file)

Create a file named `.env` in the root directory with the following variables:

```dotenv
# Credentials for Napps backend (used by metafields_fetcher.py)
EMAIL="your_admin_email@example.com"
PASSWORD="your_admin_password"

# Database Credentials (used by database_query.py)
DB_HOST="your_database_host"
DB_USER="your_database_user"
DB_PASSWORD="your_database_password"
DB_NAME="your_database_name"
DB_PORT="5432" # Or your database port

```

The script will prompt for the Shop ID.

## Command-line Arguments:


```
--neighbors K (int, default: 20): Number of neighbors for KNN and K for evaluation.

--output-dir DIR (str, default: 'results'): Directory to save all output files.

--skip-recommendations: Flag to skip KNN fitting, hybrid combination, and evaluation. Runs only data fetching and preprocessing.

--skip-evaluation: Flag to skip the evaluation step after generating recommendations.

--optimize-weights: Flag to enable feature weight optimization for KNN.

--optimization-method METHOD (str, choices: ['grid', 'random', 'evolutionary'], default: 'evolutionary'): Method for weight optimization.

--opt-iterations N (int, default: 50): Number of iterations for random search.

--population-size N (int, default: 30): Population size for evolutionary search.

--generations N (int, default: 15): Number of generations for evolutionary search.

--strong-sim S (float, default: 0.3): KNN similarity threshold for 'strong' signal in hybrid combiner.

--high-sim S (float, default: 0.95): KNN similarity threshold to filter out very similar items (e.g., exact variants) in hybrid combiner.

--min-sim S (float, default: 0.01): Minimum KNN similarity to include a recommendation in hybrid combiner.

--pop-boost F (float, default: 0.05): Boost factor for popularity in hybrid score calculation.

--group-boost F (float, default: 0.03): Boost factor for co-purchase group frequency in hybrid score calculation.´
```

Example:

python main.py --neighbors 15 --output-dir my_shop_results --optimize-weights --optimization-method evolutionary --generations 20

## Output Files (in --output-dir)

```
products_raw.csv: Raw product data fetched from Shopify.

recent_order_items.csv: Raw recent order items from the database.

popular_products.csv: Products ranked by popularity (checkout count).

product_groups.csv: Products grouped by shared checkouts.

sample_after_*.csv: CSV containing the first row of the DataFrame after specific preprocessing steps (for debugging).

products_features_before_desc_tfidf.csv: Intermediate state before the final TF-IDF step.

products_final_all_features.csv: The final DataFrame with all engineered features used for KNN.

feature_sources.csv: Documents the origin/source of each feature column.

similar_products.csv: Pairs of products deemed highly similar by the description TF-IDF processor (potential variants).

product_references.csv: Extracted relationships between products from metafields.

products_color_similarity.csv: Pre-calculated color similarity scores between products.

weight_optimization/: (If --optimize-weights) Contains optimization results (optimization_results.csv, best_weights.json, plots).

optimized_model_final/: Saved components of the KNN model fitted with optimized weights.

model_default_weights/: (If optimization is skipped or fails) Saved components of the KNN model fitted with default weights.

knn_recommendations_pre_variant_filter.csv: Raw KNN recommendations (source_id, rec_id, rank, similarity) before filtering similar variants. Crucial for ILS evaluation and hybrid input.

all_recommendations.csv: KNN recommendations after filtering similar variants based on similar_products.csv.

hybrid_recommendations.csv: Final recommendations after combining KNN, popularity, and groups.

similar_products_example.csv: Example similar products for one sample product.

knn_summary_report.txt: Text summary of the KNN run, weights, and outputs.

evaluation_metrics/: Contains JSON files with evaluation results (Coverage, ILS, PopBias) for different recommendation sets (e.g., evaluation_k20_knn_pre_filter.json, evaluation_k20_final_hybrid_recommendations.json).

*.png: Plots generated during the process (feature importance, t-SNE, similarity distribution, evaluation metric distributions).

```

## Core Concepts

Weighted KNN: The core content-based algorithm. It calculates similarity based on a distance metric (e.g., Euclidean) in the feature space. The "weighted" aspect means features belonging to different groups (e.g., TF-IDF description, price, tags) contribute differently to the distance calculation based on assigned weights.

TF-IDF (Term Frequency-Inverse Document Frequency): A text vectorization technique used to convert product titles, descriptions, and handles into numerical feature vectors. It emphasizes words that are frequent in a specific product's text but rare across all products.

Hybrid Recommendation: The system combines the strengths of content-based filtering (KNN similarity based on product features) and collaborative filtering (using signals from user behavior like popularity and co-purchases). This helps mitigate the cold-start problem and can improve relevance and serendipity.

Metafield Processing: Shopify metafields allow storing custom data. This system specifically processes common types like text, numbers, booleans, colors (calculating perceptual similarity), dimensions/weights (standardizing units), and references (extracting relationships).

Evaluation Metrics:

Catalog Coverage: Measures the percentage of unique items in the catalog that appear in any recommendation list. Indicates how much of the inventory the recommender promotes.

Intra-List Similarity (ILS): Calculates the average similarity between items within the same recommendation list. Lower ILS generally indicates more diverse recommendations. Uses the raw KNN similarity scores for calculation.

Popularity Bias: Measures the average popularity of the recommended items. High bias means the system tends to recommend already popular items.

## Important Notes & Considerations

Authentication: The metafields_fetcher.py relies on specific authentication endpoints (master.napps-solutions.com) and admin credentials. This might need adjustment depending on the deployment environment or if the auth method changes.

Database Schema: database_query.py assumes a specific table (order_line_item) and columns (checkout_id, product_id, product_name, timestamp, tenant_id) in the PostgreSQL database.

Shopify API Version: The GraphQL query uses 2025-01. Ensure this is compatible or update as needed. The @inContext directive is used, defaulting to US/EN - this might need adjustment for other locales.

Computational Cost: Running the full pipeline, especially with weight optimization (particularly Grid Search or large Evolutionary parameters), can be computationally intensive and time-consuming.

Feature Engineering: The quality of recommendations heavily depends on the quality and relevance of the features engineered during preprocessing. The weighting in KNN also significantly impacts results.

NLTK Data: Requires downloading NLTK 'punkt' and 'stopwords' datasets upon first run if not already present.

Color Similarity: Uses colormath for CIEDE2000 calculation. A fallback Euclidean distance in Lab space is implemented for potential asscalar errors in older library versions.

Word2Vec Processor: description_processor.py exists and implements Word2Vec, but the main.py script currently calls tfidf_processor.py for description processing. The Word2Vec code might be legacy or an alternative path not fully integrated into the main workflow shown.

Error Handling: While error handling is present, failures in critical steps like data fetching or initial model fitting might prevent the pipeline from completing successfully. Check logs/output for details.


# Dependencies (Major)
```
pandas

numpy

scikit-learn

requests

psycopg2 / psycopg2-binary

SQLAlchemy

python-dotenv

joblib

matplotlib

seaborn

beautifulsoup4

lxml

colormath

nltk

```