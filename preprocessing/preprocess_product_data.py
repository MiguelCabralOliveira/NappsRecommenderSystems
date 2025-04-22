# preprocessing/preprocess_product_data.py
import argparse
import os
import pandas as pd
import traceback
import re
import numpy as np # Import numpy for numeric checks

# --- Import all necessary processor functions ---
try:
    from .product_processor.availableForSale_processor import process_avaiable_for_sale
    from .product_processor.collectionHandle_processor import process_collections
    from .product_processor.createdAt_processor import process_created_at
    from .product_processor.handle_processor import process_handles_tfidf
    from .product_processor.isGiftCard_processor import process_gift_card
    from .product_processor.metafields_processor import process_metafields
    from .product_processor.product_type_processor import process_product_type
    from .product_processor.tags_processor import process_tags
    from .product_processor.title_processor import process_titles_tfidf
    from .product_processor.variants_processor import process_variants
    from .product_processor.vendor_processor import process_vendors
    # --- ADDED IMPORT for description processor ---
    from .product_processor.tfidf_processor import process_descriptions_tfidf
except ImportError as e:
    print(f"Error importing product processors: {e}. Check paths and __init__.py files.")
    raise # Stop execution if imports fail critically

# --- Main Orchestration Function ---
# --- ASSINATURA ATUALIZADA: Recebe shop_id ---
def run_product_preprocessing(input_path: str, output_path: str, shop_id: str):
    """
    Orchestrates the preprocessing of product data by calling individual processors.

    Args:
        input_path: Path to the raw product data CSV (e.g., .../raw/shop_products_raw.csv).
        output_path: Path to save the processed product features CSV (e.g., .../preprocessed/shop_products_final.csv).
        shop_id: The identifier for the shop being processed.
    """
    print(f"\n--- Starting Product Data Preprocessing Orchestration ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Shop ID provided: {shop_id}") # Log do shop_id recebido

    # Verifica se um shop_id válido foi passado
    if not shop_id:
         raise ValueError("shop_id must be provided to run_product_preprocessing.")

    # Base output directory é o diretório do ficheiro de output
    output_dir = os.path.dirname(output_path)
    # Models vão para subdiretório (a criação é garantida em run_preprocessing.py)
    models_output_dir = os.path.join(output_dir, "models")
    # Garante que existem (embora run_preprocessing.py já o faça)
    os.makedirs(models_output_dir, exist_ok=True)
    print(f"Main output directory: {output_dir}")
    print(f"Models output directory: {models_output_dir}")


    # --- Load Raw Data ---
    try:
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded {len(df)} raw product rows from {input_path}.")
        if df.empty:
            print("Input DataFrame is empty. Creating empty output file.")
            # Cria ficheiro vazio com apenas a coluna product_id
            pd.DataFrame(columns=['product_id']).to_csv(output_path, index=False)
            print(f"Empty output file created at: {output_path}")
            return
        if 'product_id' not in df.columns:
            raise ValueError("Input product data must contain a 'product_id' column.")
        df['product_id'] = df['product_id'].astype(str) # Garante tipo string

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error loading input product CSV '{input_path}': {e}")
        traceback.print_exc()
        raise

    # --- Apply Processors Sequentially ---
    processed_df = df.copy() # Começa com uma cópia

    try:
        # 1. Filtering (remove produtos indesejados primeiro)
        processed_df = process_gift_card(processed_df)
        if processed_df.empty: raise ValueError("DataFrame empty after gift card filter.")
        processed_df = process_avaiable_for_sale(processed_df)
        if processed_df.empty: raise ValueError("DataFrame empty after available_for_sale filter.")

        # 2. Basic Column Transformations & Dummies
        processed_df = process_created_at(processed_df)
        processed_df = process_product_type(processed_df)
        processed_df = process_vendors(processed_df)

        # 3. List/Dict Column Processing & Dummies
        processed_df = process_tags(processed_df)
        processed_df = process_collections(processed_df)

        # 4. Variant Processing (Extrai features, options, normaliza preços)
        processed_df = process_variants(processed_df)

        # 5. Text Processing (TF-IDF) - Usa shop_id fornecido
        # Passa models_output_dir para guardar vectorizers e o shop_id
        processed_df = process_titles_tfidf(processed_df, models_output_dir, shop_id)
        processed_df = process_handles_tfidf(processed_df, models_output_dir, shop_id)
        # Passa output_dir para similarity report e shop_id
        processed_df = process_descriptions_tfidf(processed_df, output_dir, shop_id)


        # 6. Metafields Processing (Complexo - pode adicionar features, guarda aux files)
        # Passa output_dir e shop_id
        processed_df, references_df, color_similarity_df = process_metafields(processed_df, output_dir, shop_id)
        # references_df e color_similarity_df são guardados pela função, não usados aqui

        # --- Final Checks ---
        print("\nPerforming final checks on processed product data...")
        if 'product_id' not in processed_df.columns:
            raise ValueError("'product_id' column lost during processing!")
        processed_df['product_id'] = processed_df['product_id'].astype(str)

        # Opcional: Preenche NaNs restantes em colunas numéricas com 0
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if processed_df[col].isnull().any():
                print(f"Filling remaining NaNs in numeric column '{col}' with 0.")
                processed_df[col] = processed_df[col].fillna(0)

        # Garante product_id como primeira coluna
        cols = ['product_id'] + [col for col in processed_df.columns if col != 'product_id']
        final_df = processed_df[cols]

        print(f"\nFinal Processed Product DataFrame shape: {final_df.shape}")
        print(f"Final columns ({len(final_df.columns)}): {final_df.columns.tolist()}")

    except ValueError as ve: # Captura erros de DFs vazios durante processamento
         print(f"\n--- Processing stopped due to empty data after a step: {ve} ---")
         raise
    except Exception as e:
        print(f"\n--- Error during product processing sequence: {e} ---")
        traceback.print_exc()
        raise # Propaga o erro

    # --- Save Final Processed Data ---
    try:
        # Garante que diretório de output existe (já deve existir)
        os.makedirs(output_dir, exist_ok=True)

        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved processed product data to: {output_path}")
        print(f"--- Finished Product Data Preprocessing Orchestration ---")

    except Exception as e:
        print(f"Error saving final processed product CSV to '{output_path}': {e}")
        traceback.print_exc()
        raise

# --- Command Line Execution Logic (Atualizado para aceitar shop-id) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess product data (Orchestrator).')
    parser.add_argument('--input', required=True, help='Path to the input products_raw.csv file.')
    parser.add_argument('--output', required=True, help='Path to save the final products_final_all_features.csv file.')
    # --- ADICIONADO: shop-id é necessário para execução standalone ---
    parser.add_argument('--shop-id', required=True, help='Identifier for the shop.')
    args = parser.parse_args()

    try:
        # --- ALTERADO: Passa args.shop_id ---
        run_product_preprocessing(args.input, args.output, args.shop_id)
        print("\nProduct preprocessing finished successfully via command line.")
    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
        exit(1)
    except ValueError as val_error: # Captura empty DataFrame errors etc.
        print(f"\nError: {val_error}")
        exit(1)
    except Exception as e:
        print(f"\n--- Product preprocessing pipeline failed: {e} ---")
        traceback.print_exc()
        exit(1)

    # Example Usage (from project root 'NappsRecommenderSystems/'):
    # python -m preprocessing.preprocess_product_data --input results/missusstore/raw/missusstore_shopify_products_raw.csv --output results/missusstore/preprocessed/missusstore_products_final_all_features.csv --shop-id missusstore