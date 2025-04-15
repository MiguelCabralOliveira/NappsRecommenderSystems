# preprocessing/process_event_interactions.py
import pandas as pd
import argparse
import os
import traceback
from tqdm import tqdm

try:
    from .utils import safe_json_parse
except ImportError:
    from utils import safe_json_parse

def extract_interactions_from_events(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o DataFrame de eventos e extrai uma lista estruturada de interações,
    principalmente focada em Person-Product.
    Utiliza 'converted_price' e 'converted_currency'.
    Ignora as colunas 'person_properties' e 'context' do input, exceto para
    extrair 'Channel' e 'SessionID' do 'context'.
    Corrige o processamento de múltiplos line_items por evento.
    """
    if not isinstance(event_df, pd.DataFrame):
        raise ValueError("Input 'event_df' must be a pandas DataFrame.")

    output_columns = [
        'person_id_hex', 'product_id', 'interaction_type', 'timestamp',
        'event_converted_price', 'event_converted_currency', 'event_quantity',
        'order_id', 'checkout_id', 'channel', 'session_id', 'event_id'
    ]

    if event_df.empty:
        print("Input event DataFrame is empty. Returning empty interactions DataFrame.")
        return pd.DataFrame(columns=output_columns)

    print(f"--- Iniciando Extração de Interações de {len(event_df)} eventos ---")
    interactions_list = []

    required_cols = ['person_id_hex', 'event', 'timestamp', 'event_id', 'properties', 'context']
    missing_cols = [col for col in required_cols if col not in event_df.columns]
    if missing_cols:
        raise ValueError(f"Input event DataFrame missing required columns: {missing_cols}")

    event_df_subset = event_df[required_cols].copy()

    for index, row in tqdm(event_df_subset.iterrows(), total=event_df_subset.shape[0], desc="Processando Eventos"):
        try: # Try geral para parsing inicial e dados básicos da linha
            person_id = row.get('person_id_hex')
            event_type = row.get('event')
            timestamp = row.get('timestamp')
            event_id = row.get('event_id')

            if not person_id or not event_type or not timestamp:
                continue

            properties_str = row.get('properties', '{}')
            context_str = row.get('context', '{}')
            properties = safe_json_parse(properties_str)
            context = safe_json_parse(context_str)

            if properties is None: properties = {}
            if context is None: context = {}

            channel = context.get('Channel')
            session_id = context.get('SessionID')

            interaction_data_base = {
                'person_id_hex': person_id,
                'timestamp': timestamp,
                'channel': channel,
                'session_id': session_id,
                'event_id': event_id,
                'product_id': None,
                'interaction_type': None,
                'event_converted_price': None,
                'event_converted_currency': None,
                'event_quantity': None,
                'order_id': None,
                'checkout_id': None,
            }

            # --- Lógica Específica por Tipo de Evento ---

            # 1. Eventos com product_id direto
            if event_type in ["Product Viewed", "Product Wishlisted", "Product Shared", "Variant Meta Added"]:
                product_id = properties.get('product_id')
                if product_id:
                    interaction_data = interaction_data_base.copy()
                    interaction_data['product_id'] = str(product_id)
                    interaction_data['interaction_type'] = event_type.lower().replace(" ", "_")
                    interaction_data['event_converted_price'] = properties.get('converted_price')
                    interaction_data['event_converted_currency'] = properties.get('converted_currency')
                    interaction_data['event_quantity'] = properties.get('quantity', 1 if event_type == "Variant Meta Added" else None)
                    interactions_list.append(interaction_data)

            # 2. Eventos com line_items
            elif event_type in ["Shopify Order Created", "Shopify Checkout Created", "Shopify Checkout Updated"]:
                line_items = properties.get('line_items')
                order_id = properties.get('order_id')
                checkout_id = properties.get('checkout_id')

                if isinstance(line_items, list) and line_items:
                    base_interaction_type = 'ordered_item' if 'Order' in event_type else 'checkout_item'

                    for item_index, item in enumerate(line_items):
                        
                        try:
                            if isinstance(item, dict):
                                product_id = item.get('product_id')
                                if product_id:
                                    item_interaction = interaction_data_base.copy()
                                    item_interaction['product_id'] = str(product_id)
                                    item_interaction['interaction_type'] = base_interaction_type
                                    item_interaction['event_converted_price'] = item.get('converted_price')
                                    item_interaction['event_converted_currency'] = item.get('converted_currency')
                                    item_interaction['event_quantity'] = item.get('quantity')
                                    item_interaction['order_id'] = order_id
                                    item_interaction['checkout_id'] = checkout_id
                                    interactions_list.append(item_interaction)
                                # else: # Opcional: Logar se um item não tiver product_id
                                #    print(f"Aviso: Item {item_index} na linha {index} (Evento: {event_type}) sem product_id.")
                            # else: # Opcional: Logar se um item não for um dicionário
                            #    print(f"Aviso: Item {item_index} na linha {index} (Evento: {event_type}) não é um dicionário.")

                        except Exception as item_e:
                            # Se der erro a processar UM item, regista e continua para o PRÓXIMO ITEM
                            print(f"\nErro ao processar item {item_index} na linha {index} (Evento: {event_type}, Person: {person_id}): {item_e}")
                            # traceback.print_exc() # Descomentar para debug detalhado do erro do item
                            continue # Continua para o próximo item na lista line_items
                        # *** Fim do try...except do item ***
                    # *** Fim do loop for item in line_items ***

            # 3. Outros eventos ignorados para interações Person-Product

        except Exception as row_e:
            # Se der erro ANTES do loop de items (ex: parsing inicial), regista e continua para a PRÓXIMA LINHA
            print(f"\nErro geral ao processar linha {index} (Evento: {event_type}, Person: {person_id}): {row_e}")
            # traceback.print_exc() # Descomentar para debug detalhado do erro da linha
            continue # Continua para a próxima linha no event_df_subset

    print(f"\n--- Extração Concluída: {len(interactions_list)} interações Person-Product encontradas ---")

    if not interactions_list:
         print("Nenhuma interação válida foi extraída.")
         return pd.DataFrame(columns=output_columns)

    interactions_df = pd.DataFrame(interactions_list)

    # Limpeza final e conversão de tipos (igual à versão anterior)
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
    interactions_df['event_converted_price'] = pd.to_numeric(interactions_df['event_converted_price'], errors='coerce')
    interactions_df['event_quantity'] = pd.to_numeric(interactions_df['event_quantity'], errors='coerce')
    interactions_df['product_id'] = interactions_df['product_id'].astype(str)
    interactions_df['person_id_hex'] = interactions_df['person_id_hex'].astype(str)
    interactions_df['order_id'] = interactions_df['order_id'].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    interactions_df['checkout_id'] = interactions_df['checkout_id'].apply(lambda x: str(int(x)) if pd.notna(x) else None)

    interactions_df.dropna(subset=['product_id', 'timestamp'], inplace=True)

    print(f"DataFrame de Interações final tem {len(interactions_df)} linhas.")
    print("Tipos de interação encontrados:")
    print(interactions_df['interaction_type'].value_counts())

    interactions_df = interactions_df[output_columns]

    return interactions_df

# --- Lógica para Execução via Linha de Comando (sem alterações) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processa eventos para extrair interações Person-Product.')
    parser.add_argument('--input', required=True, help='Caminho para o ficheiro de eventos filtrados (person_event_data_filtered.csv).')
    parser.add_argument('--output', required=True, help='Caminho para guardar o ficheiro de interações processadas (event_interactions_processed.csv).')
    args = parser.parse_args()

    print(f"A ler eventos de: {args.input}")
    try:
        event_df_loaded = pd.read_csv(args.input)
        print(f"Lidos {len(event_df_loaded)} eventos.")

        processed_interactions_df = extract_interactions_from_events(event_df_loaded)

        if not processed_interactions_df.empty:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            processed_interactions_df.to_csv(args.output, index=False)
            print(f"\nInterações processadas guardadas com sucesso em: {args.output}")
        else:
            print("\nNenhuma interação processada para guardar.")

        print("\nProcessamento de interações concluído.")

    except FileNotFoundError:
        print(f"Erro: Ficheiro de input não encontrado em {args.input}")
        exit(1)
    except ValueError as ve:
        print(f"Erro de Valor: {ve}")
        exit(1)
    except Exception as e:
        print(f"\n--- Ocorreu um erro durante o processamento: {e} ---")
        traceback.print_exc()
        exit(1)