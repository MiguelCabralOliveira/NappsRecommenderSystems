# preprocessing/create_product_features.py
import pandas as pd
import numpy as np
import argparse
import os
import traceback
from tqdm import tqdm

def calculate_product_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features agregadas para cada produto com base nas interações.
    """
    if not isinstance(interactions_df, pd.DataFrame):
        raise ValueError("Input 'interactions_df' must be a pandas DataFrame.")
    if interactions_df.empty:
        print("Input interactions DataFrame is empty. Returning empty product features DataFrame.")
        return pd.DataFrame(columns=['product_id']) # Retorna DF vazio com coluna mínima

    print(f"--- Iniciando Cálculo de Features Agregadas para Produtos ---")
    print(f"Input: {len(interactions_df)} interações.")

    # Garantir que timestamp é datetime
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
    # Remover interações onde o timestamp falhou (se houver)
    interactions_df.dropna(subset=['timestamp'], inplace=True)

    if interactions_df.empty:
        print("No valid interactions left after timestamp conversion. Returning empty DataFrame.")
        return pd.DataFrame(columns=['product_id'])

    # Calcular 'agora' para features de recência
    now = pd.Timestamp.now(tz='UTC')

    # --- Agregação Principal ---
    print("Agregando features básicas por produto...")
    product_features = interactions_df.groupby('product_id').agg(
        n_interactions=('timestamp', 'size'),
        n_unique_persons=('person_id_hex', 'nunique'),
        first_interaction_date=('timestamp', 'min'),
        last_interaction_date=('timestamp', 'max')
    ).reset_index() # Transforma product_id de índice para coluna

    # --- Contagem por Tipo de Interação ---
    print("Calculando contagens por tipo de interação...")
    interaction_counts = interactions_df.groupby(['product_id', 'interaction_type']).size().unstack(fill_value=0)
    # Renomear colunas para clareza (ex: 'product_viewed' -> 'n_views')
    interaction_counts.columns = [f'n_{col}' for col in interaction_counts.columns]
    # Juntar com as features principais
    product_features = pd.merge(product_features, interaction_counts, on='product_id', how='left')
    # Preencher NaNs com 0 para contagens (caso um produto não tenha tido um certo tipo de interação)
    count_cols = [col for col in product_features.columns if col.startswith('n_')]
    product_features[count_cols] = product_features[count_cols].fillna(0).astype(int)


    # --- Agregações Condicionais (Preço, Quantidade) ---

    # Preço médio quando visto
    print("Calculando preço médio visto...")
    avg_price_viewed = interactions_df[interactions_df['interaction_type'] == 'product_viewed'].groupby('product_id')['event_converted_price'].mean().reset_index()
    avg_price_viewed.rename(columns={'event_converted_price': 'avg_price_viewed'}, inplace=True)
    product_features = pd.merge(product_features, avg_price_viewed, on='product_id', how='left')

    # Preço médio quando em checkout
    print("Calculando preço médio em checkout...")
    avg_price_checkout = interactions_df[interactions_df['interaction_type'] == 'checkout_item'].groupby('product_id')['event_converted_price'].mean().reset_index()
    avg_price_checkout.rename(columns={'event_converted_price': 'avg_price_checkout'}, inplace=True)
    product_features = pd.merge(product_features, avg_price_checkout, on='product_id', how='left')

    # Preço médio quando comprado & Quantidade total comprada
    print("Calculando preço médio e quantidade total comprada...")
    ordered_items_df = interactions_df[interactions_df['interaction_type'] == 'ordered_item'].copy()
    if not ordered_items_df.empty:
        # Calcular quantidade * preço para obter valor total por item de linha
        ordered_items_df['item_total_value'] = ordered_items_df['event_converted_price'] * ordered_items_df['event_quantity']

        ordered_agg = ordered_items_df.groupby('product_id').agg(
            total_quantity_ordered=('event_quantity', 'sum'),
            # Calcular preço médio ponderado pela quantidade ou valor total?
            # Preço médio simples dos itens de linha:
            # avg_price_ordered_simple=('event_converted_price', 'mean'),
            # Valor total vendido:
            total_value_ordered=('item_total_value', 'sum')
        ).reset_index()

        # Calcular preço médio ponderado (Valor Total / Quantidade Total)
        # Evitar divisão por zero se quantidade for 0 (não deve acontecer, mas por segurança)
        ordered_agg['avg_price_ordered_weighted'] = np.where(
            ordered_agg['total_quantity_ordered'] > 0,
            ordered_agg['total_value_ordered'] / ordered_agg['total_quantity_ordered'],
            0 # Ou np.nan se preferir
        )

        # Selecionar e renomear colunas para juntar
        ordered_agg = ordered_agg[['product_id', 'total_quantity_ordered', 'total_value_ordered', 'avg_price_ordered_weighted']]
        product_features = pd.merge(product_features, ordered_agg, on='product_id', how='left')
    else:
        # Adicionar colunas com NaN se não houver itens comprados
        product_features['total_quantity_ordered'] = np.nan
        product_features['total_value_ordered'] = np.nan
        product_features['avg_price_ordered_weighted'] = np.nan


    # --- Features Temporais Derivadas ---
    print("Calculando features temporais...")
    product_features['days_since_last_interaction'] = (now - product_features['last_interaction_date']).dt.days
    product_features['days_since_first_interaction'] = (now - product_features['first_interaction_date']).dt.days
    # Podemos remover as datas originais se não forem mais necessárias
    # product_features.drop(columns=['first_interaction_date', 'last_interaction_date'], inplace=True)


    # --- Limpeza Final ---
    print("Limpando NaNs e ajustando tipos...")
    # Preencher NaNs restantes (ex: preços médios para produtos nunca vistos/comprados)
    # Para contagens, já preenchemos com 0. Para preços/quantidades, 0 pode ser apropriado.
    numeric_cols = product_features.select_dtypes(include=np.number).columns.tolist()
    # Excluir colunas de contagem que já foram tratadas e IDs/dias
    cols_to_fill_zero = [
        col for col in numeric_cols
        if col not in count_cols and 'product_id' not in col and 'days_since' not in col
    ]
    product_features[cols_to_fill_zero] = product_features[cols_to_fill_zero].fillna(0)

    # Garantir que product_id é string
    product_features['product_id'] = product_features['product_id'].astype(str)

    print(f"--- Cálculo de Features de Produto Concluído ---")
    print(f"Total de produtos únicos com features: {len(product_features)}")
    print(f"Colunas finais: {product_features.columns.tolist()}")

    return product_features

# --- Lógica para Execução via Linha de Comando ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcula features agregadas para produtos a partir das interações.')
    parser.add_argument('--input', required=True, help='Caminho para o ficheiro de interações processadas (event_interactions_processed.csv).')
    parser.add_argument('--output', required=True, help='Caminho para guardar o ficheiro de features de produto (product_features_aggregated.csv).')
    args = parser.parse_args()

    print(f"A ler interações de: {args.input}")
    try:
        interactions_df_loaded = pd.read_csv(args.input, low_memory=False) # low_memory=False pode ajudar com tipos mistos
        print(f"Lidas {len(interactions_df_loaded)} interações.")

        product_features_df = calculate_product_features(interactions_df_loaded)

        if not product_features_df.empty:
            # Criar diretório de output se não existir
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            # Guardar o resultado
            product_features_df.to_csv(args.output, index=False)
            print(f"\nFeatures de produto guardadas com sucesso em: {args.output}")
        else:
            print("\nNenhuma feature de produto calculada para guardar.")

        print("\nProcessamento de features de produto concluído.")

    except FileNotFoundError:
        print(f"Erro: Ficheiro de input não encontrado em {args.input}")
        exit(1)
    except Exception as e:
        print(f"\n--- Ocorreu um erro durante o processamento: {e} ---")
        traceback.print_exc()
        exit(1)

    # Exemplo de como correr:
    # python -m preprocessing.create_product_features --input results/sjstarcorp/event_interactions_processed.csv --output results/sjstarcorp/product_features_aggregated.csv