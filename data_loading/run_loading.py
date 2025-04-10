# data_loading/run_loading.py

import argparse
import os
import traceback
import pandas as pd
from dotenv import load_dotenv
import sys # Needed only if manipulating path, which we avoid now

# Imports RELATIVOS dentro do pacote data_loading
from .database.loader import load_all_db_data
from .shopify.loader import load_all_shopify_products
# Importa save_dataframe de utils (agora no mesmo nível)
from .utils import save_dataframe

# Define o caminho para o .env um nível ACIMA do diretório atual (data_loading)
# __file__ é data_loading/run_loading.py
# os.path.dirname(__file__) é data_loading/
# os.path.dirname(os.path.dirname(__file__)) é NappsRecommender/
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
loaded_env = load_dotenv(dotenv_path=dotenv_path, verbose=True)

if loaded_env:
    print(f".env file loaded successfully from: {dotenv_path}")
else:
    # Se não encontrou acima, tenta carregar a partir do diretório de trabalho atual
    # Isso pode ajudar se o script for executado de NappsRecommender/
    print(f"Warning: .env file not found at {dotenv_path}. Trying current working directory.")
    dotenv_path_cwd = os.path.join(os.getcwd(), '.env')
    loaded_env_cwd = load_dotenv(dotenv_path=dotenv_path_cwd, verbose=True)
    if loaded_env_cwd:
        print(f".env file loaded successfully from current working directory: {dotenv_path_cwd}")
    else:
        print(f"Warning: .env file not found in current working directory ({os.getcwd()}) either.")


# save_dataframe agora é importado de .utils

def main():
    parser = argparse.ArgumentParser(description='Load data locally from Shopify/Database.')
    parser.add_argument('--source', required=True, choices=['shopify', 'database', 'all'],
                        help='Specify the data source to load.')
    parser.add_argument('--shop-id', required=True,
                        help='The Shop ID (tenant_id) for which to load data.')
    # Ajuste: O diretório de saída será relativo ao diretório PAI (NappsRecommender)
    parser.add_argument('--output-dir', required=True,
                        help='Directory relative to project root (e.g., ./results/shop_id) to save the output CSV files.')
    parser.add_argument('--days-limit', type=int, default=90,
                        help='Number of past days to fetch event data for (database source only).')

    args = parser.parse_args()

    print(f"--- Local Data Loading Initiated ---")
    # Calcula o caminho absoluto do diretório de saída baseado na raiz do projeto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir_abs = os.path.abspath(os.path.join(project_root, args.output_dir))
    os.makedirs(output_dir_abs, exist_ok=True) # Garante que o diretório exista
    print(f"Output Target: Local Directory '{output_dir_abs}'")

    db_success = True
    shopify_success = True

    # --- Database Loading ---
    if args.source in ['database', 'all']:
        print(f"\n--- Starting Database Loading (Shop: {args.shop_id}) ---")
        try:
            db_config = {
                'host': os.getenv('DB_HOST'), 'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'), 'name': os.getenv('DB_NAME'),
                'port': os.getenv('DB_PORT')
            }
            missing = [k for k, v in db_config.items() if v is None and k != 'password']
            if missing:
                 raise ValueError(f"Database credentials not fully set in .env file or environment. Missing: {missing}")

            db_success = load_all_db_data(
                shop_id=args.shop_id,
                days_limit=args.days_limit,
                db_config=db_config,
                output_base_path=output_dir_abs # Passa o caminho absoluto calculado
            )
            print(f"--- Database Loading Finished (Success: {db_success}) ---")

        except Exception as e:
            print(f"--- Database Loading Failed Critically ---")
            print(f"Error: {e}")
            traceback.print_exc()
            db_success = False

    # --- Shopify Loading ---
    if args.source in ['shopify', 'all']:
        print(f"\n--- Starting Shopify Product Loading (Shop: {args.shop_id}) ---")
        try:
            email = os.getenv('EMAIL')
            password = os.getenv('PASSWORD')
            if not email or not password:
                raise ValueError("Shopify admin EMAIL and PASSWORD not set in .env file or environment.")

            shopify_success = load_all_shopify_products(
                shop_id=args.shop_id,
                email=email,
                password=password,
                output_base_path=output_dir_abs # Passa o caminho absoluto calculado
            )
            print(f"--- Shopify Product Loading Finished (Success: {shopify_success}) ---")

        except Exception as e:
            print(f"--- Shopify Product Loading Failed Critically ---")
            print(f"Error: {e}")
            traceback.print_exc()
            shopify_success = False

    print("\n--- Local Data Loading Script Complete ---")
    if args.source == 'all' and (not db_success or not shopify_success):
        print("--- WARNING: One or more loading steps reported errors when running 'all'. ---")
    elif args.source == 'database' and not db_success:
        print("--- WARNING: Database loading step reported errors. ---")
    elif args.source == 'shopify' and not shopify_success:
         print("--- WARNING: Shopify loading step reported errors. ---")

    # Exit with error code if needed for automation
    # if (args.source == 'all' and (not db_success or not shopify_success)) or \
    #    (args.source == 'database' and not db_success) or \
    #    (args.source == 'shopify' and not shopify_success):
    #     sys.exit(1)

if __name__ == "__main__":
    main()
    # Example Usage from NappsRecommender/ directory:
    # python -m data_loading.run_loading --source all --shop-id your_shop_id --output-dir ./results/your_shop_id