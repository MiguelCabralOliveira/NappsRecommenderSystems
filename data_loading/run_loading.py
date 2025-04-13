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

# --- Carregamento do .env ---
# Define o caminho para o .env um nível ACIMA do diretório atual (data_loading)
# __file__ é data_loading/run_loading.py
# os.path.dirname(__file__) é data_loading/
# os.path.dirname(os.path.dirname(__file__)) é a raiz do projeto (onde .env deve estar)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
loaded_env = load_dotenv(dotenv_path=dotenv_path, verbose=True)

if loaded_env:
    print(f".env file loaded successfully from: {dotenv_path}")
else:
    # Tenta carregar a partir do diretório de trabalho atual como fallback
    print(f"Warning: .env file not found at {dotenv_path}. Trying current working directory.")
    dotenv_path_cwd = os.path.join(os.getcwd(), '.env')
    loaded_env_cwd = load_dotenv(dotenv_path=dotenv_path_cwd, verbose=True)
    if loaded_env_cwd:
        print(f".env file loaded successfully from current working directory: {dotenv_path_cwd}")
    else:
        print(f"Warning: .env file not found in standard project locations.")
        print(f"Ensure your .env file with credentials (DB_HOST, DB_USER, etc., EMAIL, PASSWORD) exists in the project root: {project_root}")

# --- Fim do Carregamento do .env ---


def main():
    parser = argparse.ArgumentParser(description='Load data locally from Shopify/Database into results/raw/{shop_id}/.')
    parser.add_argument('--source', required=True, choices=['shopify', 'database', 'all'],
                        help='Specify the data source to load.')
    parser.add_argument('--shop-id', required=True,
                        help='The Shop ID (tenant_id). This will also determine the output subfolder (results/raw/{shop_id}).')
    # REMOVIDO: --output-dir - agora é determinado internamente
    parser.add_argument('--days-limit', type=int, default=90,
                        help='Number of past days to fetch event data for (database source only).')

    args = parser.parse_args()

    print(f"--- Local Data Loading Initiated for Shop ID: {args.shop_id} ---")


    output_dir_abs = os.path.join(project_root, 'results', 'raw', args.shop_id)
    try:
        os.makedirs(output_dir_abs, exist_ok=True) # Garante que o diretório exista
        print(f"Output Target: Local Directory '{output_dir_abs}'")
    except OSError as e:
        print(f"FATAL ERROR: Could not create output directory: {output_dir_abs}")
        print(f"Error details: {e}")
        sys.exit(1) # Termina o script se não conseguir criar a pasta

    db_success = True
    shopify_success = True

    # --- Database Loading ---
    if args.source in ['database', 'all']:
        print(f"\n--- Starting Database Loading ---")
        try:
            db_config = {
                'host': os.getenv('DB_HOST'), 'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'), 'name': os.getenv('DB_NAME'),
                'port': os.getenv('DB_PORT')
            }
            missing_db = [k for k, v in db_config.items() if v is None and k != 'password'] # Permite password vazia/None
            if missing_db:
                 raise ValueError(f"Database credentials not fully set in .env file or environment. Missing: {missing_db}")

            # Passa o caminho absoluto calculado para a função de loading
            db_success = load_all_db_data(
                shop_id=args.shop_id,
                days_limit=args.days_limit,
                db_config=db_config,
                output_base_path=output_dir_abs
            )
            print(f"--- Database Loading Finished (Success: {db_success}) ---")

        except Exception as e:
            print(f"--- Database Loading Failed Critically ---")
            print(f"Error: {e}")
            traceback.print_exc()
            db_success = False

    # --- Shopify Loading ---
    if args.source in ['shopify', 'all']:
        print(f"\n--- Starting Shopify Product Loading ---")
        try:
            email = os.getenv('EMAIL')
            password = os.getenv('PASSWORD')
            if not email or not password:
                raise ValueError("Shopify admin EMAIL and PASSWORD not set in .env file or environment.")

            # Passa o caminho absoluto calculado para a função de loading
            shopify_success = load_all_shopify_products(
                shop_id=args.shop_id,
                email=email,
                password=password,
                output_base_path=output_dir_abs
            )
            print(f"--- Shopify Product Loading Finished (Success: {shopify_success}) ---")

        except Exception as e:
            print(f"--- Shopify Product Loading Failed Critically ---")
            print(f"Error: {e}")
            traceback.print_exc()
            shopify_success = False

    # --- Conclusão ---
    print("\n--- Local Data Loading Script Complete ---")
    if args.source == 'all' and (not db_success or not shopify_success):
        print("--- WARNING: One or more loading steps reported errors when running 'all'. ---")
    elif args.source == 'database' and not db_success:
        print("--- WARNING: Database loading step reported errors. ---")
    elif args.source == 'shopify' and not shopify_success:
         print("--- WARNING: Shopify loading step reported errors. ---")

    # Opcional: Sair com código de erro se houver falha, útil para automação
    if (args.source == 'all' and (not db_success or not shopify_success)) or \
       (args.source == 'database' and not db_success) or \
       (args.source == 'shopify' and not shopify_success):
        print("Exiting with error code due to failures.")
        # sys.exit(1) # Descomentar se quiser que o script falhe explicitamente

if __name__ == "__main__":
    main()
    # Example Usage from the project root directory (e.g., NappsRecommender/):
    # python -m data_loading.run_loading --source all --shop-id your_shop_id
    # Output will be saved in ./results/raw/your_shop_id/