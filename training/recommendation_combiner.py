# recommendation_combiner.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import argparse

# --- Helper Function for Scaling ---
def robust_min_max_scale(series: pd.Series, min_val: float = 0.0, max_val: float = 1.0) -> pd.Series:
    """
    Aplica min-max scaling a uma Series pandas de forma robusta.
    Trata casos onde todos os valores são iguais para evitar divisão por zero.

    Args:
        series: A Series pandas para escalar.
        min_val: O valor mínimo da escala de saída.
        max_val: O valor máximo da escala de saída.

    Returns:
        A Series pandas com os valores escalados.
    """
    if series.empty or series.isnull().all():
        return pd.Series([min_val] * len(series), index=series.index) # Retorna valor mínimo se vazia/NaN

    min_s = series.min()
    max_s = series.max()
    range_s = max_s - min_s

    if range_s == 0:
        # Se todos os valores válidos são iguais, retorna o valor médio da escala
        scaled_value = min_val + (max_val - min_val) / 2.0
        return pd.Series([scaled_value] * len(series), index=series.index)
    else:
        # Aplica a escala padrão
        scaled_series = min_val + (series - min_s) / range_s * (max_val - min_val)
        # Garantir que NaNs permaneçam NaNs (fillna pode ser necessário depois se não desejado)
        scaled_series[series.isnull()] = np.nan
        return scaled_series

# --- Nova Função Combinadora com Boosting ---
def combine_recommendations_boosted(
    product_id: str,
    knn_recommendations: pd.DataFrame, # DataFrame com source_product_id, recommended_product_id, similarity_score
    popular_products_df: Optional[pd.DataFrame], # DataFrame com product_id, product_name, checkout_count
    product_groups_df: Optional[pd.DataFrame], # DataFrame com checkout_id, product_ids, item_count
    total_recommendations: int = 12,
    strong_similarity_threshold: float = 0.3, # Limiar KNN forte para alocação adaptativa
    high_similarity_threshold: float = 0.9, # Limiar superior para filtrar KNN inicial (evitar variantes exatas)
    min_similarity_threshold: float = 0.05, # Limiar inferior para filtrar KNN inicial (remover muito irrelevantes)
    pop_boost_factor: float = 0.05, # Quanto a popularidade normalizada influencia o score final
    group_boost_factor: float = 0.03 # Quanto a frequência de grupo normalizada influencia o score final
) -> pd.DataFrame:
    """
    Combina recomendações de KNN, Popularidade e Grupos usando uma abordagem adaptativa
    baseada na força do KNN e re-ranking com boosting de popularidade/grupos.

    Args:
        product_id: O ID do produto para o qual gerar recomendações.
        knn_recommendations: DataFrame de recomendações KNN pré-calculadas.
                             Deve conter 'source_product_id', 'recommended_product_id', 'similarity_score'.
        popular_products_df: DataFrame de produtos populares, ordenado por popularidade (e.g., checkout_count).
                             Deve conter 'product_id', 'checkout_count', opcionalmente 'product_name'.
        product_groups_df: DataFrame de grupos de produtos (e.g., comprados juntos).
                           Deve conter 'product_ids' (lista/string de IDs no grupo).
        total_recommendations: O número final desejado de recomendações.
        strong_similarity_threshold: Limiar de similaridade KNN para considerar o sinal KNN "forte".
        high_similarity_threshold: Limiar de similaridade KNN para excluir recomendações (demasiado similares).
        min_similarity_threshold: Limiar de similaridade KNN para incluir recomendações.
        pop_boost_factor: Fator multiplicativo para o boost de popularidade normalizada.
        group_boost_factor: Fator multiplicativo para o boost de frequência de grupo normalizada.

    Returns:
        DataFrame com as recomendações combinadas e re-rankeadas, ou DataFrame vazio se ocorrer erro.
    """

    # --- 1. Validação e Padronização de IDs ---
    if knn_recommendations is None or knn_recommendations.empty:
         #print(f"Aviso: DataFrame de recomendações KNN vazio fornecido para {product_id}. Não é possível gerar recomendações.")
         # Retornar vazio silenciosamente se não houver KNNs, pois outras fontes podem existir
         return pd.DataFrame()

    if 'similarity_score' not in knn_recommendations.columns:
        if 'similarity' in knn_recommendations.columns:
             knn_recommendations = knn_recommendations.rename(columns={'similarity': 'similarity_score'})
        else:
             print(f"Erro Crítico: Coluna 'similarity_score' (ou 'similarity') não encontrada nas recomendações KNN para {product_id}")
             return pd.DataFrame()

    if 'source_product_id' not in knn_recommendations.columns:
        print(f"Erro Crítico: Coluna 'source_product_id' não encontrada nas recomendações KNN para {product_id}")
        return pd.DataFrame()
    if 'recommended_product_id' not in knn_recommendations.columns:
         print(f"Erro Crítico: Coluna 'recommended_product_id' não encontrada nas recomendações KNN para {product_id}")
         return pd.DataFrame()

    # Padronizar ID do produto fonte
    if pd.isna(product_id):
         print(f"Erro: product_id fornecido é NaN.")
         return pd.DataFrame()
    if not str(product_id).startswith('gid://shopify/Product/'):
        standardized_product_id = f"gid://shopify/Product/{product_id}"
    else:
        standardized_product_id = product_id

    # Lidar com DataFrames opcionais vazios
    if popular_products_df is None: popular_products_df = pd.DataFrame()
    if product_groups_df is None: product_groups_df = pd.DataFrame()

    # --- 2. Filtrar KNN Recs Inicial ---
    filtered_knn = knn_recommendations[
        (knn_recommendations['source_product_id'] == standardized_product_id) &
        (knn_recommendations['similarity_score'] >= min_similarity_threshold) &
        (knn_recommendations['similarity_score'] < high_similarity_threshold) &
        (knn_recommendations['recommended_product_id'] != standardized_product_id)
    ].copy()

    #if filtered_knn.empty: # Comentado para reduzir verbosidade
        #print(f"Aviso: Nenhuma recomendação KNN válida encontrada para {product_id} após filtragem inicial.")
        # Continuar mesmo assim

    # --- 3. Alocação Adaptativa de Slots ---
    strong_knn_count = len(filtered_knn[filtered_knn['similarity_score'] >= strong_similarity_threshold])
    if not filtered_knn.empty and strong_knn_count >= total_recommendations // 2:
        num_knn = min(len(filtered_knn), int(total_recommendations * 0.75))
        num_popular = min(max(0, total_recommendations - num_knn), 2)
        num_groups = max(0, total_recommendations - num_knn - num_popular)
    else:
        num_knn = min(len(filtered_knn), total_recommendations // 2)
        remaining = total_recommendations - num_knn
        num_popular = remaining // 2
        num_groups = remaining - num_popular

    # --- 4. Recolher Candidatos Iniciais ---
    knn_recs_initial = filtered_knn.head(num_knn).copy()
    if not knn_recs_initial.empty:
        knn_recs_initial['source'] = 'similarity'

    already_recommended_ids = set(knn_recs_initial['recommended_product_id']) if not knn_recs_initial.empty else set()
    already_recommended_ids.add(standardized_product_id)

    popular_candidates_initial = pd.DataFrame()
    pop_counts_initial = {}
    if not popular_products_df.empty and 'product_id' in popular_products_df.columns and 'checkout_count' in popular_products_df.columns and num_popular > 0:
        if 'product_id_standardized' not in popular_products_df.columns: # Padronizar apenas se necessário
             popular_products_df['product_id_standardized'] = popular_products_df['product_id'].apply(
                 lambda x: f"gid://shopify/Product/{x}" if pd.notna(x) and not str(x).startswith('gid://shopify/Product/') else str(x) if pd.notna(x) else None
             )
             popular_products_df = popular_products_df.dropna(subset=['product_id_standardized'])

        popular_candidates_initial = popular_products_df[
            ~popular_products_df['product_id_standardized'].isin(already_recommended_ids)
        ].head(num_popular).copy()

        if not popular_candidates_initial.empty:
            popular_candidates_initial = popular_candidates_initial.rename(columns={'product_id_standardized': 'recommended_product_id'})
            already_recommended_ids.update(popular_candidates_initial['recommended_product_id'])
            pop_counts_initial = popular_candidates_initial.set_index('recommended_product_id')['checkout_count'].to_dict()

    group_candidates_initial = pd.DataFrame()
    group_frequencies = {}
    if not product_groups_df.empty and 'product_ids' in product_groups_df.columns and num_groups > 0:
        product_in_groups_ids = []
        for _, row in product_groups_df.iterrows():
            product_ids_in_group = row.get('product_ids', [])
            if isinstance(product_ids_in_group, str):
                try:
                    product_ids_in_group = [id.strip() for id in product_ids_in_group.strip('[]').split(',') if id.strip()]
                except Exception: product_ids_in_group = []
            if not isinstance(product_ids_in_group, list): product_ids_in_group = []

            standardized_ids_in_group = set()
            is_source_in_group = False
            for pid in product_ids_in_group:
                if pd.notna(pid) and str(pid).strip():
                    pid_str = str(pid).strip()
                    std_pid = f"gid://shopify/Product/{pid_str}" if not pid_str.startswith('gid://shopify/Product/') else pid_str
                    standardized_ids_in_group.add(std_pid)
                    if std_pid == standardized_product_id: is_source_in_group = True

            if is_source_in_group:
                for std_pid in standardized_ids_in_group:
                    if std_pid != standardized_product_id: product_in_groups_ids.append(std_pid)

        if product_in_groups_ids:
            group_counts = pd.Series(product_in_groups_ids).value_counts().reset_index()
            group_counts.columns = ['recommended_product_id', 'frequency']
            group_candidates_initial = group_counts[~group_counts['recommended_product_id'].isin(already_recommended_ids)].head(num_groups).copy()
            if not group_candidates_initial.empty:
                 already_recommended_ids.update(group_candidates_initial['recommended_product_id'])
                 group_frequencies = group_candidates_initial.set_index('recommended_product_id')['frequency'].to_dict()

    # --- 5. Recolher Candidatos de Fallback (Popularidade) ---
    fallback_candidates = pd.DataFrame()
    pop_counts_fallback = {}
    current_candidate_count = (0 if knn_recs_initial.empty else len(knn_recs_initial)) + \
                              (0 if popular_candidates_initial.empty else len(popular_candidates_initial)) + \
                              (0 if group_candidates_initial.empty else len(group_candidates_initial))
    needed = total_recommendations - current_candidate_count

    if needed > 0 and not popular_products_df.empty and 'product_id_standardized' in popular_products_df.columns and 'checkout_count' in popular_products_df.columns:
        fallback_candidates = popular_products_df[
            ~popular_products_df['product_id_standardized'].isin(already_recommended_ids)
        ].head(needed).copy()
        if not fallback_candidates.empty:
             fallback_candidates = fallback_candidates.rename(columns={'product_id_standardized': 'recommended_product_id'})
             pop_counts_fallback = fallback_candidates.set_index('recommended_product_id')['checkout_count'].to_dict()

    # --- 6. Criar Pool Unificado de Candidatos ---
    candidate_pool = {}
    if not knn_recs_initial.empty:
        for _, row in knn_recs_initial.iterrows():
            rec_id = row['recommended_product_id']
            if rec_id not in candidate_pool:
                candidate_pool[rec_id] = {'knn_sim': row['similarity_score'], 'source': 'similarity'}
    if not popular_candidates_initial.empty:
         for rec_id in popular_candidates_initial['recommended_product_id']:
            if rec_id not in candidate_pool:
                 candidate_pool[rec_id] = {'knn_sim': 0.0, 'source': 'popularity'}
            candidate_pool[rec_id]['pop_count'] = pop_counts_initial.get(rec_id, 0)
            candidate_pool[rec_id].setdefault('source', 'popularity')
    if not group_candidates_initial.empty:
         for rec_id in group_candidates_initial['recommended_product_id']:
             if rec_id not in candidate_pool:
                 candidate_pool[rec_id] = {'knn_sim': 0.0, 'source': 'frequently_bought_together'}
             candidate_pool[rec_id]['group_freq'] = group_frequencies.get(rec_id, 0)
             candidate_pool[rec_id].setdefault('source', 'frequently_bought_together')
    if not fallback_candidates.empty:
        for rec_id in fallback_candidates['recommended_product_id']:
             if rec_id not in candidate_pool:
                 candidate_pool[rec_id] = {'knn_sim': 0.0, 'source': 'fallback_popularity'}
             candidate_pool[rec_id]['pop_count'] = pop_counts_fallback.get(rec_id, 0)
             candidate_pool[rec_id].setdefault('source', 'fallback_popularity')

    if not candidate_pool:
        #print(f"Aviso: Nenhum candidato de recomendação encontrado para o produto {product_id} de nenhuma fonte.")
        return pd.DataFrame()

    # Converter pool para DataFrame
    pool_df = pd.DataFrame.from_dict(candidate_pool, orient='index')
    pool_df = pool_df.reset_index().rename(columns={'index': 'recommended_product_id'})

    # --- GARANTIR EXISTÊNCIA E TIPOS DAS COLUNAS ANTES DE USAR ---
    if 'knn_sim' not in pool_df.columns: pool_df['knn_sim'] = 0.0
    if 'pop_count' not in pool_df.columns: pool_df['pop_count'] = 0
    if 'group_freq' not in pool_df.columns: pool_df['group_freq'] = 0 # <<< CORREÇÃO AQUI
    if 'source' not in pool_df.columns: pool_df['source'] = 'unknown'

    pool_df['knn_sim'] = pool_df['knn_sim'].fillna(0.0)
    pool_df['pop_count'] = pool_df['pop_count'].fillna(0).astype(float)
    pool_df['group_freq'] = pool_df['group_freq'].fillna(0).astype(float) # <<< CORREÇÃO AQUI
    pool_df['source'] = pool_df['source'].fillna('unknown')
    # --- FIM DA GARANTIA ---


    # --- 7. Calcular Score Híbrido (com Boosting) ---
    if pool_df['pop_count'].max() > 0:
        log_pop = np.log1p(pool_df['pop_count'])
        pool_df['pop_norm'] = robust_min_max_scale(log_pop, 0.0, 1.0)
    else:
        pool_df['pop_norm'] = 0.0

    if pool_df['group_freq'].max() > 0: # <<< ACESSO SEGURO AGORA
        pool_df['group_norm'] = robust_min_max_scale(pool_df['group_freq'], 0.0, 1.0)
    else:
        pool_df['group_norm'] = 0.0

    pool_df['final_score'] = (
        pool_df['knn_sim'] +
        pop_boost_factor * pool_df['pop_norm'] +
        group_boost_factor * pool_df['group_norm']
    )

    # --- 8. Ordenar e Selecionar Top N ---
    final_recs = pool_df.sort_values('final_score', ascending=False).head(total_recommendations)

    # --- 9. Formatação Final do DataFrame de Saída ---
    final_recs['rank'] = range(1, len(final_recs) + 1)
    final_recs['source_product_id'] = standardized_product_id

    title_map = {}
    if not popular_products_df.empty and 'product_name' in popular_products_df.columns and 'product_id_standardized' in popular_products_df.columns:
         title_map = popular_products_df.drop_duplicates(subset=['product_id_standardized']).set_index('product_id_standardized')['product_name'].to_dict()

    final_recs['product_title'] = final_recs['recommended_product_id'].map(title_map).fillna('Title Not Found')

    columns_to_keep = [
        'source_product_id', 'recommended_product_id', 'product_title', 'rank',
        'final_score', 'source', 'knn_sim', 'pop_count', 'group_freq'
    ]
    final_recs_output = final_recs[[col for col in columns_to_keep if col in final_recs.columns]].copy()
    final_recs_output = final_recs_output.rename(columns={'final_score': 'hybrid_score', 'knn_sim': 'base_knn_similarity'})

    # --- Verificações Finais de Sanidade ---
    if len(final_recs_output) != final_recs_output['recommended_product_id'].nunique():
        print(f"AVISO FINAL: Duplicatas detectadas nas recomendações finais para {product_id}! Removendo...")
        final_recs_output = final_recs_output.drop_duplicates(subset=['recommended_product_id'], keep='first').copy()
        final_recs_output['rank'] = range(1, len(final_recs_output) + 1)

    if standardized_product_id in final_recs_output['recommended_product_id'].values:
        print(f"ERRO FINAL: Auto-recomendação encontrada para {product_id}! Removendo...")
        final_recs_output = final_recs_output[final_recs_output['recommended_product_id'] != standardized_product_id].copy()
        final_recs_output['rank'] = range(1, len(final_recs_output) + 1)

    return final_recs_output


# --- Função Geradora Principal (Chama a função combinadora) ---
def generate_hybrid_recommendations_adaptative(
    all_knn_recommendations: pd.DataFrame,
    products_df: pd.DataFrame, # Necessário para obter a lista de product_ids únicos
    popular_products_df: Optional[pd.DataFrame] = None,
    product_groups_df: Optional[pd.DataFrame] = None,
    total_recommendations: int = 12,
    strong_similarity_threshold: float = 0.3,
    high_similarity_threshold: float = 0.9,
    min_similarity_threshold: float = 0.05,
    pop_boost_factor: float = 0.05, # Passar os fatores para a função combinadora
    group_boost_factor: float = 0.03
) -> pd.DataFrame:
    """
    Gera recomendações híbridas combinadas para todos os produtos únicos em products_df.
    (Docstring omitido para brevidade, é o mesmo da versão anterior)
    """
    # Verificar inputs essenciais
    if all_knn_recommendations is None or all_knn_recommendations.empty:
        print("Erro: DataFrame de recomendações KNN (all_knn_recommendations) está vazio ou é None.")
        return pd.DataFrame()
    if products_df is None or products_df.empty or 'product_id' not in products_df.columns:
        print("Erro: DataFrame de produtos (products_df) está vazio, é None ou não contém 'product_id'.")
        return pd.DataFrame()

    # Inicializar DataFrames opcionais se forem None
    if popular_products_df is None: popular_products_df = pd.DataFrame()
    if product_groups_df is None: product_groups_df = pd.DataFrame()

    # Padronizar IDs em popular_products_df uma vez fora do loop para eficiência
    if not popular_products_df.empty and 'product_id' in popular_products_df.columns:
        popular_products_df['product_id_standardized'] = popular_products_df['product_id'].apply(
             lambda x: f"gid://shopify/Product/{x}" if pd.notna(x) and not str(x).startswith('gid://shopify/Product/') else str(x) if pd.notna(x) else None
         )
        popular_products_df = popular_products_df.dropna(subset=['product_id_standardized'])


    all_recommendations = [] # Lista para guardar os DataFrames de recomendação por produto
    unique_product_ids = products_df['product_id'].unique()

    print(f"Gerando recomendações híbridas para {len(unique_product_ids)} produtos únicos...")

    # Loop por cada produto para gerar suas recomendações
    for i, product_id in enumerate(unique_product_ids):
        if pd.isna(product_id):
            #print(f"Aviso: Pulando product_id NaN encontrado no índice {i}.")
            continue

        if (i + 1) % 200 == 0 or i == 0 or i == len(unique_product_ids) - 1: # Log a cada 200
             print(f"Processando produto {i+1}/{len(unique_product_ids)}: {product_id}")

        try:
            # Chamar a NOVA função combinadora com boosting
            hybrid_recs = combine_recommendations_boosted(
                product_id=str(product_id), # Garantir que é string
                knn_recommendations=all_knn_recommendations,
                popular_products_df=popular_products_df, # Passar df já com ID padronizado
                product_groups_df=product_groups_df,
                total_recommendations=total_recommendations,
                strong_similarity_threshold=strong_similarity_threshold,
                high_similarity_threshold=high_similarity_threshold,
                min_similarity_threshold=min_similarity_threshold,
                pop_boost_factor=pop_boost_factor,
                group_boost_factor=group_boost_factor
            )

            if not hybrid_recs.empty:
                all_recommendations.append(hybrid_recs)

        except Exception as e:
            # Logar o erro mas continuar o processo para outros produtos
            print(f"Erro ao gerar recomendações para produto {product_id}: {type(e).__name__} - {str(e)}")
            # import traceback # Descomentar para ver traceback completo se necessário
            # traceback.print_exc()


    # Concatenar todas as recomendações geradas num único DataFrame
    if all_recommendations:
        print("\nConcatenando todas as recomendações geradas...")
        try:
            recommendations_df = pd.concat(all_recommendations, ignore_index=True)
        except Exception as e:
            print(f"Erro ao concatenar recomendações: {e}")
            return pd.DataFrame() # Retornar vazio se concatenação falhar

        # Verificação final de sanidade: remover auto-recomendações
        if 'source_product_id' in recommendations_df.columns and 'recommended_product_id' in recommendations_df.columns:
            self_recommendations = recommendations_df[recommendations_df['source_product_id'] == recommendations_df['recommended_product_id']]
            if not self_recommendations.empty:
                print(f"ERRO FINAL: Encontrados {len(self_recommendations)} casos de produtos recomendando a si mesmos no DataFrame final! Removendo...")
                recommendations_df = recommendations_df[recommendations_df['source_product_id'] != recommendations_df['recommended_product_id']].copy()

        print(f"Geração de recomendações híbridas concluída. Total de pares recomendados: {len(recommendations_df)}")
        return recommendations_df
    else:
        print("Nenhuma recomendação híbrida foi gerada para nenhum produto.")
        return pd.DataFrame() # Retornar DataFrame vazio


# --- Função para Salvar e Analisar Recomendações ---
def save_hybrid_recommendations(
    hybrid_recommendations: pd.DataFrame,
    output_file: str = "hybrid_recommendations.csv"
) -> None:
    """
    Salva o DataFrame de recomendações híbridas em CSV e imprime estatísticas sobre as fontes.
    (Docstring omitido para brevidade)
    """
    if hybrid_recommendations is None or hybrid_recommendations.empty:
        print("Aviso: DataFrame de recomendações híbridas está vazio ou é None. Nada para salvar.")
        return

    try:
        hybrid_recommendations.to_csv(output_file, index=False)
        print(f"\nRecomendações híbridas salvas com sucesso em: {output_file}")

        if 'source' in hybrid_recommendations.columns:
            source_counts = hybrid_recommendations['source'].value_counts()
            total_recs = len(hybrid_recommendations)
            print("\nDistribuição das fontes principais de recomendação (no pool inicial):")
            if total_recs > 0:
                for source, count in source_counts.items():
                    percentage = 100.0 * count / total_recs
                    print(f"  - {source:<25}: {count:>7} ({percentage:>5.1f}%)")
            else:
                 print("  (Nenhuma recomendação para analisar fontes)")

        if 'source_product_id' in hybrid_recommendations.columns and 'recommended_product_id' in hybrid_recommendations.columns:
             self_recs = hybrid_recommendations[hybrid_recommendations['source_product_id'] == hybrid_recommendations['recommended_product_id']]
             if not self_recs.empty:
                  print(f"\nAVISO CRÍTICO: {len(self_recs)} casos de produtos recomendando a si mesmos foram encontrados no ficheiro salvo!")

    except Exception as e:
        print(f"\nErro ao salvar as recomendações híbridas em {output_file}: {type(e).__name__} - {str(e)}")


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    print("Executando recommendation_combiner.py como script principal...")

    parser = argparse.ArgumentParser(description='Gerar recomendações híbridas adaptativas combinando KNN, Popularidade e Grupos.')
    parser.add_argument('--knn-recs', type=str, required=True, help='Caminho para o ficheiro CSV com todas as recomendações KNN pré-calculadas.')
    parser.add_argument('--products', type=str, required=True, help='Caminho para o ficheiro CSV com a lista de produtos (necessário para IDs).')
    parser.add_argument('--popular', type=str, help='(Opcional) Caminho para o ficheiro CSV com produtos populares (ordenados).')
    parser.add_argument('--groups', type=str, help='(Opcional) Caminho para o ficheiro CSV com grupos de produtos comprados juntos.')
    parser.add_argument('--output', type=str, default='hybrid_recommendations.csv', help='Caminho para o ficheiro CSV de saída das recomendações híbridas.')
    parser.add_argument('--total-recs', type=int, default=12, help='Número total de recomendações a gerar por produto.')
    parser.add_argument('--strong-sim', type=float, default=0.3, help='Limiar de similaridade KNN para considerar sinal forte.')
    parser.add_argument('--high-sim', type=float, default=0.9, help='Limiar de similaridade KNN para filtrar recomendações muito altas (prováveis variantes).')
    parser.add_argument('--min-sim', type=float, default=0.05, help='Limiar de similaridade KNN mínimo para considerar uma recomendação KNN.')
    parser.add_argument('--pop-boost', type=float, default=0.05, help='Fator de boost para popularidade no score híbrido.')
    parser.add_argument('--group-boost', type=float, default=0.03, help='Fator de boost para frequência de grupo no score híbrido.')

    args = parser.parse_args()

    # Carregar DataFrames
    print(f"Carregando recomendações KNN de: {args.knn_recs}")
    try: all_knn_recommendations = pd.read_csv(args.knn_recs)
    except Exception as e: print(f"Erro ao carregar {args.knn_recs}: {e}"); exit(1)

    print(f"Carregando lista de produtos de: {args.products}")
    try:
        products_df = pd.read_csv(args.products)
        if 'product_id' not in products_df.columns: print(f"Erro: Coluna 'product_id' não encontrada em {args.products}"); exit(1)
    except Exception as e: print(f"Erro ao carregar {args.products}: {e}"); exit(1)

    popular_products_df = None
    if args.popular:
        print(f"Carregando produtos populares de: {args.popular}")
        try:
            popular_products_df = pd.read_csv(args.popular)
            if 'product_id' not in popular_products_df.columns or 'checkout_count' not in popular_products_df.columns:
                 print(f"Aviso: Ficheiro popular '{args.popular}' não contém 'product_id' e/ou 'checkout_count'. Não será usado."); popular_products_df = None
        except Exception as e: print(f"Aviso: Erro ao carregar {args.popular}: {e}. Continuando sem ele.")

    product_groups_df = None
    if args.groups:
        print(f"Carregando grupos de produtos de: {args.groups}")
        try:
            product_groups_df = pd.read_csv(args.groups)
            if 'product_ids' not in product_groups_df.columns:
                 print(f"Aviso: Ficheiro de grupos '{args.groups}' não contém 'product_ids'. Não será usado."); product_groups_df = None
        except Exception as e: print(f"Aviso: Erro ao carregar {args.groups}: {e}. Continuando sem ele.")

    # Gerar recomendações
    print("\nIniciando a geração das recomendações híbridas...")
    hybrid_recommendations = generate_hybrid_recommendations_adaptative(
        all_knn_recommendations=all_knn_recommendations,
        products_df=products_df,
        popular_products_df=popular_products_df,
        product_groups_df=product_groups_df,
        total_recommendations=args.total_recs,
        strong_similarity_threshold=args.strong_sim,
        high_similarity_threshold=args.high_sim,
        min_similarity_threshold=args.min_sim,
        pop_boost_factor=args.pop_boost,
        group_boost_factor=args.group_boost
    )

    # Salvar
    if hybrid_recommendations is not None and not hybrid_recommendations.empty:
        save_hybrid_recommendations(hybrid_recommendations, args.output)
    else:
        print("\nNenhuma recomendação híbrida foi gerada. O ficheiro de saída não será criado ou estará vazio.")

    print("\nExecução do script recommendation_combiner.py concluída.")