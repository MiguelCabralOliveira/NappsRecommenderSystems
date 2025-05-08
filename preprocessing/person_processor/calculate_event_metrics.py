# preprocessing/person_processor/calculate_event_metrics.py
import pandas as pd
import numpy as np
import os
import traceback
import argparse # Adicionado para execução standalone com cutoff

def calculate_person_event_metrics(event_interactions_path: str, cutoff_timestamp: pd.Timestamp = None) -> pd.DataFrame:
    """
    Loads processed event interactions up to an optional cutoff timestamp and
    calculates person-level summary metrics based on that period. Identifies
    buyers based on events within that period.

    Args:
        event_interactions_path: Path to the 'event_interactions_processed.csv' file.
        cutoff_timestamp (pd.Timestamp, optional): If provided, only interactions
            at or before this timestamp (UTC) will be considered for metric
            calculation. Defaults to None (use all data).

    Returns:
        DataFrame indexed by 'person_id_hex' with calculated metrics based on the
        specified period, or an empty DataFrame if the input file doesn't exist
        or is empty after filtering.
    """
    if cutoff_timestamp:
        print(f"--- Calculating Person Metrics from Event Interactions (Cutoff: {cutoff_timestamp}) ---")
    else:
        print(f"--- Calculating Person Metrics from Event Interactions (No Cutoff - Using All Data) ---")
        print("WARNING: No cutoff timestamp provided. Metrics will be calculated using all interactions, which may lead to data leakage if used for temporal GNN training.")

    print(f"Loading event interactions from: {event_interactions_path}")

    if not os.path.exists(event_interactions_path):
        print(f"Warning: Event interactions file not found at {event_interactions_path}. Skipping event metrics calculation.")
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

    try:
        dtypes = {
            'person_id_hex': 'string', 'product_id': 'string', 'interaction_type': 'string',
            'event_converted_price': 'float64', 'checkout_id': 'Int64', 'channel': 'string',
            'person_interaction_sequence': 'Int64'
        }
        # Ler timestamp como string primeiro para conversão robusta
        events_df = pd.read_csv(
            event_interactions_path,
            dtype=dtypes,
            low_memory=False
        )
        print(f"Loaded {len(events_df)} total interactions.")

        if events_df.empty:
            print("Event interactions file is empty. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # --- Two-Pass Timestamp Conversion ---
        print("Converting 'timestamp' column...")
        if 'timestamp' in events_df.columns:
            timestamps_obj = events_df['timestamp'].astype(str).str.strip()
            format_pass1 = '%Y-%m-%d %H:%M:%S.%f%z'
            format_pass2 = '%Y-%m-%d %H:%M:%S%z'

            converted_pass1 = pd.to_datetime(timestamps_obj, format=format_pass1, errors='coerce', utc=True)
            failures_pass1_mask = converted_pass1.isna()
            num_failures_pass1 = failures_pass1_mask.sum()

            converted_pass2 = converted_pass1.copy()
            if num_failures_pass1 > 0:
                converted_pass2[failures_pass1_mask] = pd.to_datetime(
                    timestamps_obj[failures_pass1_mask],
                    format=format_pass2, errors='coerce', utc=True
                )
                num_failures_pass2 = converted_pass2.isna().sum()
                if num_failures_pass2 > 0:
                     print(f"Warning: Dropping {num_failures_pass2} rows due to invalid timestamp format after two passes.")
                     events_df['timestamp'] = converted_pass2
                     events_df.dropna(subset=['timestamp'], inplace=True)
                else:
                     events_df['timestamp'] = converted_pass2
                     print("Successfully converted all timestamps using two-pass approach.")
            else:
                 events_df['timestamp'] = converted_pass1
                 print("Successfully converted all timestamps using format 1.")

        else:
             print("Error: 'timestamp' column not found.")
             return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        if events_df.empty:
            print("No valid timestamps remaining after conversion/dropping. Skipping metrics calculation.")
            return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

        # --- Filtragem Temporal (NOVO) ---
        if cutoff_timestamp:
            if not isinstance(cutoff_timestamp, pd.Timestamp):
                 try:
                      # Tentar converter se for string
                      cutoff_timestamp = pd.to_datetime(cutoff_timestamp, utc=True)
                 except Exception as e:
                      print(f"Error converting cutoff_timestamp: {e}. Ignoring cutoff.")
                      cutoff_timestamp = None

            if cutoff_timestamp and cutoff_timestamp.tzinfo is None:
                 print("Warning: cutoff_timestamp is timezone-naive. Assuming UTC.")
                 cutoff_timestamp = cutoff_timestamp.tz_localize('UTC')

            if cutoff_timestamp:
                initial_rows = len(events_df)
                print(f"Filtering interactions before or at {cutoff_timestamp}...")
                events_df = events_df[events_df['timestamp'] <= cutoff_timestamp].copy()
                rows_filtered = initial_rows - len(events_df)
                print(f"Removed {rows_filtered} interactions occurring after the cutoff timestamp.")
                if events_df.empty:
                    print("No interactions remaining after applying the cutoff timestamp. Skipping metrics calculation.")
                    return pd.DataFrame(index=pd.Index([], name='person_id_hex'))
        # --- Fim Filtragem Temporal ---

        # --- Identificar Compradores (usando dados filtrados) ---
        print("\nIdentifying buyers based on filtered event data...")
        buyer_ids = set(
            events_df.loc[events_df['interaction_type'] == 'Shopify Order Created', 'person_id_hex'].unique()
        )
        print(f"Found {len(buyer_ids)} unique buyers within the considered period.")
        # --- Fim Identificar Compradores ---

        # --- Aggregate Metrics (usando dados filtrados) ---
        print("\nAggregating event metrics per person (using filtered data)...")
        now_utc = cutoff_timestamp if cutoff_timestamp else pd.Timestamp.now(tz='UTC') # Usar cutoff como 'agora' se fornecido

        agg_dict = {
            'purchase_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Order Created').sum()),
            'total_interactions': pd.NamedAgg(column='timestamp', aggfunc='count'),
            'unique_products_interacted': pd.NamedAgg(column='product_id', aggfunc='nunique'),
            'view_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Viewed').sum()),
            'wishlist_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Wishlisted').sum()),
            'share_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Product Shared').sum()),
            'checkout_start_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Checkout Created').sum()),
            'last_interaction_timestamp': pd.NamedAgg(column='timestamp', aggfunc='max'),
            'first_interaction_timestamp': pd.NamedAgg(column='timestamp', aggfunc='min'),
            'distinct_channels_count': pd.NamedAgg(column='channel', aggfunc='nunique'),
            'variant_meta_added_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Variant Meta Added').sum()),
            'checkout_update_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Checkout Updated').sum()),
            'order_update_count': pd.NamedAgg(column='interaction_type', aggfunc=lambda x: (x == 'Shopify Order Updated').sum()),
        }
        # Agrupar pelos IDs presentes no events_df filtrado
        person_metrics = events_df.groupby('person_id_hex').agg(**agg_dict)
        print(f"Aggregated metrics for {len(person_metrics)} persons based on filtered data.")

        # --- Calcular Métricas Derivadas (usando dados filtrados) ---
        print("\nCalculating derived metrics...")

        # Criar is_buyer usando o conjunto buyer_ids (já filtrado temporalmente)
        person_metrics['is_buyer'] = person_metrics.index.isin(buyer_ids).astype(int)

        # Calcular tempos desde a última/primeira interação RELATIVO AO CUTOFF (ou 'agora' se não houver cutoff)
        person_metrics['days_since_last_interaction'] = (now_utc - person_metrics['last_interaction_timestamp']).dt.days.fillna(9999).astype(int)
        person_metrics['days_since_first_interaction'] = (now_utc - person_metrics['first_interaction_timestamp']).dt.days.fillna(9999).astype(int)

        # Calcular canal mais frequente (usando dados filtrados)
        most_freq_channel = events_df.groupby('person_id_hex')['channel'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').rename('most_frequent_channel')
        person_metrics = person_metrics.join(most_freq_channel, how='left')
        person_metrics['most_frequent_channel'] = person_metrics['most_frequent_channel'].fillna('Unknown')

        # Calcular valor médio de compra (usando dados filtrados)
        purchase_events = events_df[events_df['interaction_type'] == 'Shopify Order Created'].copy()
        if not purchase_events.empty and 'event_converted_price' in purchase_events.columns:
             purchase_events['event_converted_price'] = pd.to_numeric(purchase_events['event_converted_price'], errors='coerce')
             purchase_events.dropna(subset=['event_converted_price'], inplace=True)
             if not purchase_events.empty:
                 purchase_summary = purchase_events.groupby('person_id_hex')['event_converted_price'].agg(['sum', 'count'])
                 purchase_summary['avg_purchase_value'] = (purchase_summary['sum'] / purchase_summary['count'].replace(0, 1)).fillna(0)
                 person_metrics = person_metrics.join(purchase_summary[['avg_purchase_value']], how='left')
                 person_metrics['avg_purchase_value'] = person_metrics['avg_purchase_value'].fillna(0)
             else:
                person_metrics['avg_purchase_value'] = 0.0
        else:
             person_metrics['avg_purchase_value'] = 0.0

        # Drop intermediate timestamp columns
        person_metrics = person_metrics.drop(columns=['last_interaction_timestamp', 'first_interaction_timestamp'], errors='ignore')

        print(f"\nCalculated metrics for {len(person_metrics)} persons (final).")
        print("Sample metrics (FINAL):")
        print(person_metrics.head())
        print("\nMetrics columns:", person_metrics.columns.tolist())
        print("\nFinal Metrics DataFrame Info:")
        person_metrics.info() # Usar info() em vez de print(info())

        return person_metrics

    except FileNotFoundError:
        print(f"Error: File not found at {event_interactions_path}")
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))
    except Exception as e:
        print(f"Error calculating person event metrics: {e}")
        traceback.print_exc()
        return pd.DataFrame(index=pd.Index([], name='person_id_hex'))

# --- Bloco para Execução Standalone (com argumento de cutoff) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcula métricas agregadas por pessoa a partir de interações de eventos, com cutoff temporal opcional.')
    parser.add_argument('--input', required=True, help='Caminho para o ficheiro de interações processadas (event_interactions_processed.csv).')
    parser.add_argument('--output', required=True, help='Caminho para guardar o ficheiro de métricas por pessoa (person_event_metrics.csv).')
    parser.add_argument('--cutoff', required=False, default=None, help='Timestamp de corte opcional (formato YYYY-MM-DD HH:MM:SS). Apenas eventos até esta data/hora (UTC) serão considerados.')

    args = parser.parse_args()

    cutoff_ts = None
    if args.cutoff:
        try:
            cutoff_ts = pd.to_datetime(args.cutoff, utc=True)
            print(f"Using cutoff timestamp: {cutoff_ts}")
        except ValueError:
            print(f"Error: Invalid cutoff timestamp format: {args.cutoff}. Please use YYYY-MM-DD HH:MM:SS. Calculating metrics without cutoff.")

    person_metrics_df = calculate_person_event_metrics(args.input, cutoff_timestamp=cutoff_ts)

    if not person_metrics_df.empty:
        print(f"\nMétricas calculadas para {len(person_metrics_df)} pessoas.")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Diretório de saída '{output_dir}' assegurado.")
        person_metrics_df.to_csv(args.output, index=True) # Index=True para manter person_id_hex como índice
        print(f"Métricas por pessoa guardadas com sucesso em: {args.output}")
    else:
        print("\nNenhuma métrica por pessoa foi calculada ou DataFrame resultante está vazio.")
        # Opcional: Criar ficheiro vazio se necessário
        # pd.DataFrame().to_csv(args.output, index=True)
        # print(f"Ficheiro de saída vazio (ou não criado) em: {args.output}")

    print("\nCálculo de métricas concluído.")