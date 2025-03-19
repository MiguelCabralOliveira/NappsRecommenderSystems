import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def fetch_recent_order_items(shop_id):
    """ 
    Fetch recent order items and return three dataframes:
    1. All recent order items
    2. Popular products (products with most checkout_ids)
    3. Product groups (products that share checkout_ids)
    """
    db_host = os.getenv('DB_HOST')
    db_password = os.getenv('DB_PASSWORD')
    db_user = os.getenv('DB_USER')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')

    # Create SQLAlchemy engine
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    
    try:
        # Query for all recent order items
        base_query = """
        WITH last_orders AS (
            SELECT checkout_id 
            FROM order_line_item 
            WHERE timestamp::date >= NOW() - INTERVAL '60 day'
        ),
        recent_order_items AS (
            SELECT * 
            FROM order_line_item 
            WHERE checkout_id IN (SELECT checkout_id FROM last_orders)
        )
        SELECT * 
        FROM recent_order_items 
        WHERE tenant_id = %s;
        """
        all_items_df = pd.read_sql(base_query, engine, params=(shop_id,))
        print(f"Successfully fetched {len(all_items_df)} recent order items")
        
        # Query for popular products
        popular_query = """
        WITH last_orders AS (
            SELECT checkout_id 
            FROM order_line_item 
            WHERE timestamp::date >= NOW() - INTERVAL '60 day'
        ),
        recent_order_items AS (
            SELECT * 
            FROM order_line_item 
            WHERE checkout_id IN (SELECT checkout_id FROM last_orders)
            AND tenant_id = %s
        ),
        product_checkout_counts AS (
            SELECT 
                product_id,
                product_name,
                COUNT(DISTINCT checkout_id) as checkout_count
            FROM recent_order_items
            GROUP BY product_id, product_name
            ORDER BY checkout_count DESC
        )
        SELECT * FROM product_checkout_counts;
        """
        popular_products_df = pd.read_sql(popular_query, engine, params=(shop_id,))
        print(f"Successfully fetched {len(popular_products_df)} products ordered by popularity")
        
        # Query for product groups
        groups_query = """
        WITH last_orders AS (
            SELECT checkout_id 
            FROM order_line_item 
            WHERE timestamp::date >= NOW() - INTERVAL '60 day'
        ),
        recent_order_items AS (
            SELECT * 
            FROM order_line_item 
            WHERE checkout_id IN (SELECT checkout_id FROM last_orders)
            AND tenant_id = %s
        )
        SELECT 
            checkout_id,
            array_agg(product_id) as product_ids,
            array_agg(product_name) as product_names,
            count(*) as item_count
        FROM recent_order_items
        GROUP BY checkout_id
        ORDER BY item_count DESC;
        """
        product_groups_df = pd.read_sql(groups_query, engine, params=(shop_id,))
        print(f"Successfully fetched {len(product_groups_df)} checkout groups")
        
        return all_items_df, popular_products_df, product_groups_df

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    # Standalone execution with test shop ID
    print("=== Running database_query.py standalone test ===")
    test_shop_id = "letseatit"
    
    try:
        # Get all three dataframes
        result_df, popular_products_df, product_groups_df = fetch_recent_order_items(test_shop_id)
        
        print("\nSample data from regular order items:")
        print(result_df.head())
        print(f"\nData shape: {result_df.shape}")
        
        print("\nMost popular products (by number of checkout_ids):")
        print(popular_products_df.head(10))
        print(f"\nPopular products data shape: {popular_products_df.shape}")
        
        print("\nProduct groups by checkout_id:")
        print(product_groups_df.head(10))
        print(f"\nProduct groups data shape: {product_groups_df.shape}")
        
        print("Standalone test completed successfully")
    except Exception as e:
        print(f"Standalone test failed: {e}")