# training/similarity/knn/knn_pipeline.py

import pandas as pd
import numpy as np
import os
import time
import traceback
import json
from typing import Optional, Dict, Tuple, Any

# Imports (Ajustados para funcionar dentro deste ficheiro)
try:
    from .weighted_knn_core import _WeightedKNN, WeightOptimizer
    from .evaluation import run_evaluation
    # Import relativo para utils DENTRO do pacote similarity
    from ..utils import ensure_dir, save_text_report, load_dataframe, load_json_config, save_dataframe # Adicionado save_dataframe
except ImportError:
     print("Warning: KNN Pipeline using fallback imports.")
     from weighted_knn_core import _WeightedKNN, WeightOptimizer
     from evaluation import run_evaluation
     # Tenta importar utils do diretório acima (training) - Ajustar se necessário
     # É melhor garantir que os imports relativos funcionam
     try:
         # Assuming utils.py is at training/similarity/utils.py
         from ..utils import ensure_dir, save_text_report, load_dataframe, load_json_config, save_dataframe
     except ImportError:
         print("Error: Could not import utility functions.")
         raise

def run_knn_training(
    features_df: pd.DataFrame,
    product_ids: pd.Series,
    output_dir: str,
    n_neighbors: int,
    products_df_for_titles: Optional[pd.DataFrame] = None, # Passar DF com IDs e Títulos
    base_weights: Optional[Dict[str, float]] = None,
    optimize_weights: bool = False,
    optimization_method: str = 'evolutionary',
    optimization_params: Optional[Dict[str, Any]] = None,
    filter_variants: bool = False,
    similar_products_df: Optional[pd.DataFrame] = None,
    catalog_csv_path: Optional[str] = None, # Argumento para path do catálogo
    skip_evaluation: bool = False,
    save_model: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[_WeightedKNN]]:
    """
    Main pipeline function for training/optimizing the Weighted KNN model.
    Includes optional evaluation step that requires catalog_csv_path for coverage.
    Accepts an optional DataFrame for adding product titles to the export.
    """
    print("\n" + "="*30 + " Running Weighted KNN Training Pipeline " + "="*30)
    pipeline_start_time = time.time()
    ensure_dir(output_dir)

    # Basic validation
    if features_df is None or features_df.empty or product_ids is None or product_ids.empty:
        print("Error: Input features_df or product_ids is empty or None.")
        return pd.DataFrame(), None
    if len(features_df) != len(product_ids):
         print(f"Error: Mismatch in length between features ({len(features_df)}) and product IDs ({len(product_ids)}).")
         return pd.DataFrame(), None
    if filter_variants and (similar_products_df is None or similar_products_df.empty):
        print("Warning: Variant filtering requested but similar_products_df is missing or empty. Disabling filter.")
        filter_variants = False
    # Validação do DF de títulos
    if products_df_for_titles is not None and ('product_id' not in products_df_for_titles.columns or 'product_title' not in products_df_for_titles.columns):
        print("Warning: products_df_for_titles provided but missing 'product_id' or 'product_title'. Titles will not be added.")
        products_df_for_titles = None

    # Initialize
    final_model: Optional[_WeightedKNN] = None
    optimizer: Optional[WeightOptimizer] = None
    best_weights_used = base_weights # Começa com os pesos base (podem ser None)

    # 1. Weight Optimization (Optional)
    if optimize_weights:
        print(f"\n=== Starting Feature Weight Optimization (Method: {optimization_method}) ===")
        opt_start_time = time.time()
        opt_dir = os.path.join(output_dir, "weight_optimization")
        try:
            optimizer = WeightOptimizer(features_df=features_df.copy(), product_ids=product_ids.copy(), n_neighbors=n_neighbors, output_dir=opt_dir, random_state=42)
            if optimization_params is None: optimization_params = {}
            print(f"Running {optimization_method} optimization...")
            if optimization_method == 'grid': optimizer.grid_search(**optimization_params, save_results=True)
            elif optimization_method == 'random': optimizer.random_search(**optimization_params, save_results=True)
            elif optimization_method == 'evolutionary': optimizer.evolutionary_search(**optimization_params, save_results=True)
            else: optimizer.evolutionary_search(**optimization_params, save_results=True); print("Defaulted to evolutionary.")
            optimizer.save_optimization_results(filename="optimization_results_details.csv")
            print("\nBuilding final KNN model using optimized weights...")
            final_model = optimizer.build_optimized_model(save_model_files=save_model) # Otimizador tenta construir
            if final_model:
                 best_weights_used = optimizer.best_weights # Atualiza os pesos a usar com os otimizados
                 print("Successfully built model using optimized weights.")
            else:
                 print("Error: Failed to build model after optimization. Will attempt fallback.")
                 # Mantém final_model=None para forçar fallback
            opt_end_time = time.time()
            print(f"=== Weight Optimization Finished (Duration: {opt_end_time - opt_start_time:.2f}s) ===")
        except Exception as e:
            print(f"\n--- ERROR during Weight Optimization: {type(e).__name__} - {str(e)} ---"); traceback.print_exc()
            final_model = None # Garante fallback
            # Mantém best_weights_used como os base originais

    # 2. Fit Model with Base/Default or Optimized Weights (if optimization failed or wasn't run)
    if final_model is None:
        # Define o modo e os pesos a usar (base originais ou os default da classe se base=None)
        fit_mode = "Base/Default" if not optimize_weights else "Fallback (Base/Default)"
        print(f"\n=== Building KNN Model with {fit_mode} Weights ===")
        weights_for_fitting = best_weights_used # Usa os pesos base/otimizados se disponíveis
        model_build_start_time = time.time()
        try:
            final_model = _WeightedKNN(n_neighbors=n_neighbors, random_state=42)
            # Identifica grupos ANTES de definir pesos
            final_model._identify_feature_groups(features_df.columns.tolist())
            # Define pesos (weights_for_fitting pode ser None, a função trata disso)
            final_model.set_feature_weights(base_weights=weights_for_fitting, normalize_by_count=True)
            # Fit
            final_model.fit(features_df.copy(), product_ids.copy())
            model_build_end_time = time.time()
            if final_model.recommendation_matrix is not None:
                 print(f"Model fitting with {fit_mode} weights completed in {model_build_end_time - model_build_start_time:.2f}s.")
                 # Atualiza best_weights_used com os pesos efetivamente aplicados (após defaults/normalização)
                 best_weights_used = final_model.feature_weights
                 if save_model:
                      # O nome da pasta reflete se a otimização foi *tentada* e se os pesos *atuais* são diferentes dos default
                      ran_opt = optimize_weights
                      # Compara os pesos *atuais* do modelo com os default internos
                      is_different_from_default = final_model.feature_weights != final_model._default_base_weights
                      model_save_dir_name = "model_optimized_weights" if ran_opt and is_different_from_default else "model_default_weights"
                      model_save_dir = os.path.join(output_dir, model_save_dir_name)
                      print(f"\nSaving the model fitted with {fit_mode} weights to: {model_save_dir}")
                      final_model.save_model(model_save_dir)
            else:
                 print(f"Error: Failed to fit model with {fit_mode} weights.");
                 final_model = None # Garante que o modelo é None se o fit falhar
        except Exception as e:
             print(f"ERROR during Model Fitting: {e}");
             traceback.print_exc();
             final_model = None # Garante que o modelo é None em caso de erro

    # 3. Generate Visualizations, Export Recs, Evaluate
    all_recommendations_df: pd.DataFrame = pd.DataFrame()
    final_recommendations_path: Optional[str] = None
    unfiltered_recs_path = os.path.join(output_dir, "knn_recommendations_unfiltered.csv")
    final_recs_path_filtered = os.path.join(output_dir, "knn_recommendations_final.csv")
    vis_success = {'importance': False, 'tsne': False, 'distribution': False}

    if final_model is None or final_model.recommendation_matrix is None:
         print("\nError: No valid KNN model was fitted. Cannot generate recommendations or evaluate.")
    else:
        print("\n=== Generating Visualizations and Exporting Results ===")
        vis_dir = os.path.join(output_dir, "visualizations"); ensure_dir(vis_dir)
        # Gerar visualizações
        try: final_model.visualize_feature_importance(output_file=os.path.join(vis_dir, "feature_importance.png")); vis_success['importance'] = True
        except Exception as e: print(f"Error generating feature importance plot: {e}")
        try: final_model.visualize_tsne(output_file=os.path.join(vis_dir, "tsne_visualization.png")); vis_success['tsne'] = True
        except Exception as e: print(f"Error generating t-SNE plot: {e}")
        try: final_model.visualize_similarity_distribution(output_file=os.path.join(vis_dir, "similarity_distribution.png")); vis_success['distribution'] = True
        except Exception as e: print(f"Error generating similarity distribution plot: {e}")

        # --- DEBUG: Verifica o DF de títulos ---
        print("\n--- DEBUG: Checking products_df_for_titles before export ---")
        if products_df_for_titles is not None:
            print(products_df_for_titles.info())
            print(products_df_for_titles.head())
        else:
            print("products_df_for_titles is None.")
        print("--- END DEBUG ---")
        # --- FIM DEBUG ---

        # --- Exportar Recomendações ---
        # Export Unfiltered
        print(f"\nExporting UNFILTERED KNN recommendations to: {unfiltered_recs_path}")
        unfiltered_recs_df = pd.DataFrame() # Inicializa como vazio
        try:
             unfiltered_recs_df = final_model.export_recommendations(
                 unfiltered_recs_path, n_neighbors, False, None, products_df_for_titles
             )
             if unfiltered_recs_df.empty: print("Warning: Unfiltered export resulted in empty DataFrame.")
        except Exception as e: print(f"Error Exporting Unfiltered Recs: {e}"); traceback.print_exc()

        # Export Final (Potentially Filtered)
        print(f"\nExporting FINAL recommendations to: {final_recs_path_filtered}")
        print(f"(Variant filtering {'ENABLED' if filter_variants else 'DISABLED'})")
        all_recommendations_df = pd.DataFrame() # Inicializa como vazio
        try:
             all_recommendations_df = final_model.export_recommendations(
                 final_recs_path_filtered, n_neighbors, filter_variants, similar_products_df, products_df_for_titles
             )
             final_recommendations_path = final_recs_path_filtered if not all_recommendations_df.empty else None
             if all_recommendations_df.empty: print("Warning: Final export resulted in empty DataFrame.")
        except Exception as e: print(f"Error Exporting Final Recs: {e}"); traceback.print_exc(); final_recommendations_path = None
        # --- Fim Exportar Recomendações ---

        # --- Avaliação ---
        if not skip_evaluation:
            print("\n" + "="*30 + " Starting Evaluation " + "="*30)
            eval_output_dir = os.path.join(output_dir, "evaluation_metrics"); ensure_dir(eval_output_dir)
            print(f"Evaluation Catalog Path: {catalog_csv_path if catalog_csv_path else 'Not Provided (Coverage Skipped)'}")

            # Usa o ficheiro UNFILTERED como base para calcular ILS (se existir)
            knn_sims_for_ils_path = unfiltered_recs_path if os.path.exists(unfiltered_recs_path) else None
            if not knn_sims_for_ils_path: print("Warning: Unfiltered recommendations file not found, ILS metric cannot be calculated.")

            # Evaluate UNFILTERED
            if not unfiltered_recs_df.empty: # Avalia só se o DF não for vazio
                 print(f"\n--- Evaluating UNFILTERED Recs ({os.path.basename(unfiltered_recs_path)}) ---")
                 try: run_evaluation(unfiltered_recs_path, n_neighbors, eval_output_dir, catalog_csv_path, None, knn_sims_for_ils_path, f"evaluation_k{n_neighbors}_knn_unfiltered.json")
                 except Exception as e: print(f"Error during UNFILTERED Eval: {e}"); traceback.print_exc()
            else: print("Skipping UNFILTERED Evaluation: No data generated or file not found.")

            # Evaluate FINAL
            if final_recommendations_path and not all_recommendations_df.empty: # Avalia só se o DF não for vazio e o path existir
                if final_recommendations_path != unfiltered_recs_path: # Só avalia se for diferente
                    print(f"\n--- Evaluating FINAL Recs ({os.path.basename(final_recommendations_path)}) ---")
                    try: run_evaluation(final_recommendations_path, n_neighbors, eval_output_dir, catalog_csv_path, None, knn_sims_for_ils_path, f"evaluation_k{n_neighbors}_knn_final.json")
                    except Exception as e: print(f"Error during FINAL Eval: {e}"); traceback.print_exc()
                else: print("\nSkipping FINAL evaluation (same as unfiltered).")
            else: print("\nSkipping FINAL Evaluation: No data generated or file not found.")
        else: print("\nSkipping Evaluation step.")
        # --- Fim Avaliação ---

        # --- Summary Report ---
        print("\nGenerating Summary Report...")
        summary_path = os.path.join(output_dir, "knn_summary_report.txt")
        try:
            # Tenta obter dados do modelo final, mesmo que algumas partes tenham falhado
            avg_similarity = final_model.calculate_average_similarity() if final_model else 0.0
            num_products = len(final_model.product_ids) if final_model and final_model.product_ids is not None else 'N/A'
            num_features = len(final_model.feature_names_in_order) if final_model and final_model.feature_names_in_order is not None else 'N/A'
            model_algorithm = final_model.algorithm if final_model else 'N/A'
            model_metric = final_model.metric if final_model else 'N/A'

            report_content = ["=== Weighted KNN Training Summary ===", ""]
            report_content.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"Total Pipeline Duration: {time.time() - pipeline_start_time:.2f} seconds\n")
            report_content.append(f"Input Products: {num_products}")
            report_content.append(f"Input Features: {num_features}\n")
            report_content.append(f"KNN Parameters:")
            report_content.append(f"- n_neighbors (k): {n_neighbors}")
            report_content.append(f"- algorithm: {model_algorithm}")
            report_content.append(f"- metric: {model_metric}\n")
            report_content.append(f"Weight Optimization: {'Enabled (' + optimization_method + ')' if optimize_weights else 'Disabled'}")
            # Mostra os pesos efetivamente usados no modelo final (best_weights_used pode ter sido atualizado)
            if best_weights_used:
                 report_content.append("Effective Weights Used:")
                 try: weights_str = json.dumps(best_weights_used, indent=2)
                 except TypeError: weights_str = str(best_weights_used) # Fallback para string
                 report_content.append(weights_str + "\n")
            report_content.append(f"Variant Filtering: {'Enabled' if filter_variants else 'Disabled'}")
            if filter_variants and similar_products_df is not None: report_content.append(f"- Using {len(similar_products_df)} pairs from similar products file.\n")
            report_content.append("Model Performance:")
            report_content.append(f"- Average Recommendation Similarity: {avg_similarity:.4f}\n")
            report_content.append("Outputs Generated:")
            report_content.append(f"- Unfiltered Recommendations: {'Yes' if os.path.exists(unfiltered_recs_path) and not unfiltered_recs_df.empty else 'No/Empty'} ({os.path.basename(unfiltered_recs_path)})")
            final_recs_status = 'Yes' if final_recommendations_path and not all_recommendations_df.empty else 'No/Empty'
            report_content.append(f"- Final Recommendations: {final_recs_status} ({os.path.basename(final_recs_path_filtered)})")
            model_save_status = 'Yes' if save_model and final_model and os.path.exists(os.path.join(output_dir, "model_default_weights")) or os.path.exists(os.path.join(output_dir, "model_optimized_weights")) else 'No (Skipped/Failed)'
            report_content.append(f"- Saved Model Components: {model_save_status}")
            eval_dir = os.path.join(output_dir, "evaluation_metrics")
            eval_files_exist = os.path.exists(eval_dir) and any(f.endswith('.json') for f in os.listdir(eval_dir))
            eval_status = 'Skipped' if skip_evaluation else ('Yes' if eval_files_exist else 'No (Or Failed)')
            report_content.append(f"- Evaluation Metrics: {eval_status}")
            vis_list = [name for name, success in vis_success.items() if success]
            report_content.append(f"- Visualizations Generated: {', '.join(vis_list) if vis_list else 'None'}")
            save_text_report("\n".join(report_content), summary_path)
        except Exception as e: print(f"\nError generating summary report: {e}"); traceback.print_exc()
        # --- Fim Summary Report ---

    pipeline_end_time = time.time()
    print("\n" + "="*30 + f" KNN Pipeline Finished (Total Time: {pipeline_end_time - pipeline_start_time:.2f}s) " + "="*30)

    return all_recommendations_df, final_model

# --- FIM DE run_knn_training ---

# O BLOCO if __name__ == "__main__" foi REMOVIDO daqui.
# A execução deve ser feita através do script train_knn_model.py