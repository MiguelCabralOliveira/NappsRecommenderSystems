import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def combine_recommendations_adaptative(
    product_id: str,
    knn_recommendations: pd.DataFrame,
    popular_products_df: pd.DataFrame,
    product_groups_df: pd.DataFrame,
    total_recommendations: int = 12,
    strong_similarity_threshold: float = 0.3,
    high_similarity_threshold: float = 0.9,
    min_similarity_threshold: float = 0.05
) -> pd.DataFrame:
    # Verificar se as colunas necessárias existem
    if 'similarity' not in knn_recommendations.columns:
        if 'similarity_score' in knn_recommendations.columns:
            # Renomear a coluna para o nome esperado
            knn_recommendations = knn_recommendations.rename(columns={'similarity_score': 'similarity'})
        else:
            # Se não existir, criar uma coluna vazia
            print(f"Aviso: Coluna 'similarity' não encontrada nas recomendações KNN para o produto {product_id}")
            return pd.DataFrame()  # Retornar DataFrame vazio
    
    if 'source_product_id' not in knn_recommendations.columns:
        if 'source_product' in knn_recommendations.columns:
            knn_recommendations = knn_recommendations.rename(columns={'source_product': 'source_product_id'})
        else:
            print(f"Aviso: Coluna 'source_product_id' não encontrada nas recomendações KNN para o produto {product_id}")
            return pd.DataFrame()
    
    # Resto do código permanece o mesmo...
    filtered_knn = knn_recommendations[
        (knn_recommendations['source_product_id'] == product_id) & 
        (knn_recommendations['similarity'] >= min_similarity_threshold) &
        (knn_recommendations['similarity'] < high_similarity_threshold)
    ].copy()
    
    filtered_knn['source'] = 'similarity'
    
    strong_knn_count = len(filtered_knn[filtered_knn['similarity'] >= strong_similarity_threshold])
    
    if strong_knn_count >= total_recommendations // 2:
        num_knn = min(len(filtered_knn), int(total_recommendations * 0.75))
        num_popular = min(total_recommendations - num_knn, 2)
        num_groups = total_recommendations - num_knn - num_popular
    else:
        num_knn = min(len(filtered_knn), total_recommendations // 2)
        remaining = total_recommendations - num_knn
        num_popular = remaining // 2
        num_groups = remaining - num_popular
    
    knn_recs = filtered_knn.head(num_knn)
    
    popular_recs = pd.DataFrame()
    if not popular_products_df.empty and num_popular > 0:
        popular = popular_products_df[
            (popular_products_df['product_id'] != product_id) & 
            ~popular_products_df['product_id'].isin(knn_recs['recommended_product_id'])
        ].head(num_popular).copy()
        
        if not popular.empty:
            max_checkout = popular_products_df['checkout_count'].max()
            min_checkout = popular_products_df['checkout_count'].min()
            range_checkout = max(1, max_checkout - min_checkout)
            
            popular['similarity'] = 0.2 + (0.3 * (popular['checkout_count'] - min_checkout) / range_checkout)
            popular['rank'] = popular.index + 1
            popular['source'] = 'popularity'
            popular = popular.rename(columns={'product_id': 'recommended_product_id', 'product_name': 'product_title'})
            popular['source_product_id'] = product_id
            
            columns_to_keep = ['source_product_id', 'recommended_product_id', 'similarity', 'rank', 'source']
            if 'product_title' in popular.columns:
                columns_to_keep.append('product_title')
            
            popular_recs = popular[columns_to_keep]
    
    group_recs = pd.DataFrame()
    if not product_groups_df.empty and num_groups > 0:
        product_in_groups = []
        
        for _, row in product_groups_df.iterrows():
            product_ids = row.get('product_ids', [])
            
            if isinstance(product_ids, str):
                try:
                    product_ids = product_ids.strip('[]').split(',')
                    product_ids = [id.strip() for id in product_ids]
                except:
                    product_ids = []
            
            if product_id in product_ids:
                for other_id in product_ids:
                    if other_id != product_id:
                        product_in_groups.append({
                            'group_id': row.get('checkout_id', ''),
                            'recommended_product_id': other_id,
                            'item_count': row.get('item_count', 1)
                        })
        
        if product_in_groups:
            groups_df = pd.DataFrame(product_in_groups)
            
            product_frequency = groups_df['recommended_product_id'].value_counts().reset_index()
            product_frequency.columns = ['recommended_product_id', 'frequency']
            
            already_recommended = pd.concat([knn_recs, popular_recs])['recommended_product_id'].unique()
            product_frequency = product_frequency[
                ~product_frequency['recommended_product_id'].isin(already_recommended)
            ].head(num_groups)
            
            if not product_frequency.empty:
                max_freq = product_frequency['frequency'].max()
                min_freq = product_frequency['frequency'].min()
                range_freq = max(1, max_freq - min_freq)
                
                product_frequency['similarity'] = 0.15 + (0.3 * (product_frequency['frequency'] - min_freq) / range_freq)
                product_frequency['rank'] = range(1, len(product_frequency) + 1)
                product_frequency['source'] = 'frequently_bought_together'
                product_frequency['source_product_id'] = product_id
                
                product_frequency['product_title'] = product_frequency['recommended_product_id']
                
                columns_to_keep = ['source_product_id', 'recommended_product_id', 'similarity', 'rank', 'source']
                if 'product_title' in product_frequency.columns:
                    columns_to_keep.append('product_title')
                
                group_recs = product_frequency[columns_to_keep]
    
    current_count = len(knn_recs) + len(popular_recs) + len(group_recs)
    
    if current_count < total_recommendations and not popular_products_df.empty:
        already_recommended = pd.concat(
            [knn_recs, popular_recs, group_recs], ignore_index=True
        )['recommended_product_id'].unique()
        
        more_popular = popular_products_df[
            (popular_products_df['product_id'] != product_id) & 
            ~popular_products_df['product_id'].isin(already_recommended)
        ].head(total_recommendations - current_count).copy()
        
        if not more_popular.empty:
            max_checkout = popular_products_df['checkout_count'].max()
            min_checkout = popular_products_df['checkout_count'].min()
            range_checkout = max(1, max_checkout - min_checkout)
            
            more_popular['similarity'] = 0.1 + (0.1 * (more_popular['checkout_count'] - min_checkout) / range_checkout)
            more_popular['rank'] = more_popular.index + 1
            more_popular['source'] = 'fallback_popularity'
            more_popular = more_popular.rename(columns={'product_id': 'recommended_product_id', 'product_name': 'product_title'})
            more_popular['source_product_id'] = product_id
            
            columns_to_keep = ['source_product_id', 'recommended_product_id', 'similarity', 'rank', 'source']
            if 'product_title' in more_popular.columns:
                columns_to_keep.append('product_title')
            
            popular_recs = pd.concat([popular_recs, more_popular[columns_to_keep]], ignore_index=True)
    
    combined_recs = pd.concat([knn_recs, popular_recs, group_recs], ignore_index=True)
    
    combined_recs = combined_recs.sort_values('similarity', ascending=False)
    combined_recs['rank'] = range(1, len(combined_recs) + 1)
    
    return combined_recs.head(total_recommendations)


def generate_hybrid_recommendations_adaptative(
    all_knn_recommendations: pd.DataFrame,
    products_df: pd.DataFrame,
    popular_products_df: Optional[pd.DataFrame] = None,
    product_groups_df: Optional[pd.DataFrame] = None,
    total_recommendations: int = 12,
    strong_similarity_threshold: float = 0.3,
    high_similarity_threshold: float = 0.9,
    min_similarity_threshold: float = 0.05
) -> pd.DataFrame:
    if popular_products_df is None:
        popular_products_df = pd.DataFrame()
    
    if product_groups_df is None:
        product_groups_df = pd.DataFrame()
    
    all_recommendations = []
    
    unique_product_ids = products_df['product_id'].unique()
    
    for product_id in unique_product_ids:
        try:
            hybrid_recs = combine_recommendations_adaptative(
                product_id=product_id,
                knn_recommendations=all_knn_recommendations,
                popular_products_df=popular_products_df,
                product_groups_df=product_groups_df,
                total_recommendations=total_recommendations,
                strong_similarity_threshold=strong_similarity_threshold,
                high_similarity_threshold=high_similarity_threshold,
                min_similarity_threshold=min_similarity_threshold
            )
            all_recommendations.append(hybrid_recs)
        except Exception as e:
            print(f"Erro ao gerar recomendações para produto {product_id}: {str(e)}")
    
    if all_recommendations:
        return pd.concat(all_recommendations, ignore_index=True)
    else:
        return pd.DataFrame()


def save_hybrid_recommendations(
    hybrid_recommendations: pd.DataFrame,
    output_file: str = "hybrid_recommendations.csv"
) -> None:
    hybrid_recommendations.to_csv(output_file, index=False)
    print(f"Recomendações híbridas salvas em {output_file}")
    
    if not hybrid_recommendations.empty and 'source' in hybrid_recommendations.columns:
        source_counts = hybrid_recommendations['source'].value_counts()
        print("\nDistribuição das fontes de recomendação:")
        for source, count in source_counts.items():
            percentage = 100 * count / len(hybrid_recommendations)
            print(f"  - {source}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerar recomendações híbridas adaptativas')
    parser.add_argument('--knn-recs', type=str, required=True, help='Caminho para CSV com recomendações KNN')
    parser.add_argument('--products', type=str, required=True, help='Caminho para CSV com produtos')
    parser.add_argument('--popular', type=str, help='Caminho para CSV com produtos populares')
    parser.add_argument('--groups', type=str, help='Caminho para CSV com grupos de produtos')
    parser.add_argument('--output', type=str, default='hybrid_recommendations.csv', help='Caminho para arquivo de saída')
    parser.add_argument('--total-recs', type=int, default=12, help='Total de recomendações por produto')
    parser.add_argument('--strong-sim', type=float, default=0.3, help='Limiar para similaridade forte')
    parser.add_argument('--high-sim', type=float, default=0.9, help='Limiar para produtos muito similares')
    parser.add_argument('--min-sim', type=float, default=0.05, help='Similaridade mínima para KNN')
    
    args = parser.parse_args()
    
    all_knn_recommendations = pd.read_csv(args.knn_recs)
    products_df = pd.read_csv(args.products)
    
    popular_products_df = None
    if args.popular:
        popular_products_df = pd.read_csv(args.popular)
    
    product_groups_df = None
    if args.groups:
        product_groups_df = pd.read_csv(args.groups)
    
    hybrid_recommendations = generate_hybrid_recommendations_adaptative(
        all_knn_recommendations=all_knn_recommendations,
        products_df=products_df,
        popular_products_df=popular_products_df,
        product_groups_df=product_groups_df,
        total_recommendations=args.total_recs,
        strong_similarity_threshold=args.strong_sim,
        high_similarity_threshold=args.high_sim,
        min_similarity_threshold=args.min_sim
    )
    
    save_hybrid_recommendations(hybrid_recommendations, args.output)